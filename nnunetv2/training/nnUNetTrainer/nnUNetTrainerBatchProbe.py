# -*- coding: utf-8 -*-
"""自动 batch 探测 + 梯度累积训练器变体（v5，二分逼近 + 真实梯度累积）

设计（2026-08-01）:
- 目标: 每个模型自动探测性能最优 batch，再用梯度累积把有效 batch 拉齐
  到规定值（nominal，默认 16），保证多模型训练效果可比。
- 三个 batch 概念解耦:
    nominal_batch_size（规定值）: plans.json 名义 batch，决定每 epoch 图像
      总数 = 250 × nominal = 4000 恒定（训练量可比）
    actual_batch_size（并行 batch）: 探测值，dataloader 每步实际处理的图像
      数（受显存限制）
    grad_accum_steps（累积步数）: actual < nominal 时，累积 N 步的梯度后
      才更新一次，有效梯度 batch = actual × accum ≈ nominal
  每 epoch 物理迭代数 = ceil(4000 / actual)，其中每 accum 步做一次
  optimizer.step（有效更新次数 = ceil(4000 / (actual × accum)) ≈ 250）。
- 插入点（关键）: override `_warmup_kernels()` —— 此时
    network / optimizer / loss 已全部就绪（基类 initialize L240-291），
    dataloader 尚未创建（on_train_start 才建）。探测修改 self.batch_size
    会被 get_dataloaders(L1021) 正确消费；warmup Phase 2 随后用探测后的
    batch 跑 2 步真实训练，顺带验证探测结果。
- 探测流程（单点外推 + 二分逼近）:
    1. batch=1&2 差分测 per_sample 显存开销（消固定开销误差）
    2. 线性外推 candidate = avail_mb / per_sample（向下取偶）
    3. candidate 验证失败 → 二分逼近 [1, candidate] 真实边界（任意偶数
       粒度，非减半跳变）
    4. 判据: 物理显存安全阀(92%) / OOM / 功耗骤降(<前档50%) / TP骤降
       (<前档70%) → 不可用；功耗≥85%TDP → 满载直接接受
- 梯度累积: override train_step，accum 步 forward+backward 累积梯度后
  统一 clip+step。loss 除以 accum 保持尺度一致。CUDAGraph 组合时若
  accum>1 自动降级 eager（累积本身已减少 step 次数，graph 收益有限）。
- 运行期: 复用基类 NVML GPU 监控（每 epoch 打印 util/power/mem 均值峰值）。
- 用法: -tr nnUNetTrainerBatchProbe -p nnUNetPlans_bs16 [--disable_checkpointing]
"""
import gc
import os
import time

import torch
import torch.distributed as dist
from torch.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import (
    load_json, save_json, join, isfile, maybe_mkdir_p)
from nnunetv2.utilities.helpers import dummy_context

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerBatchProbe(nnUNetTrainer):
    """自动探测性能最优 batch + 梯度累积拉齐有效 batch。

    三个 batch 概念解耦: nominal（规定值，定每 epoch 数据量）/ actual
    （并行 batch，dataloader 实际值）/ accum（梯度累积把有效 batch 拉回
    nominal）。train_step 实现真实累积（accum 步 forward+backward 后统一
    clip+step），CUDAGraph 组合时 accum>1 自动降级 eager。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # 必须关键字传递 device: MRO 中 SegResNet 等子类签名是
        # (..., unpack_dataset=True, device=...)，位置传参会把 device 错位塞进
        # unpack_dataset，导致 SegResNet 的 device 落回默认(无 index) → DDP 下
        # 兜底成 cuda:local_rank → 绑错卡（2026-08-11 DGX 实测 GPU 1,2,3）。
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # 规定 batch（plans.json 名义值，仅记录/报告）vs 实际运行 batch
        # （self.batch_size，探测结果，dataloader/tqdm 真正使用的值）
        self.nominal_batch_size = self.configuration_manager.batch_size
        # 目标有效 batch（所有模型统一拉齐到该值）
        self.target_batch_size = 16
        # 探测上限（防止小卡上无限二分）
        self.max_probe_batch = 64
        # 物理显存安全阀: 整卡(含其他进程) used/total 超过该比例即停止探测
        # （Windows WDDM 下继续分配会溢出到共享显存，速度断崖且不报 OOM）
        self.vram_safe_ratio = 0.92
        # 吞吐测量: 每档探测的 forward+backward 次数（取最快）
        self.probe_iters = 2
        # 每 epoch 优化步数基准（nnUNet 经验值，有效 batch 下每 epoch 处理
        # iterations_per_epoch_effective × effective_batch 张图）
        self.iterations_per_epoch_effective = 250
        # 探测结果
        self.actual_batch_size = None
        self.grad_accum_steps = 1
        self.probe_log = []          # [{batch, peak_mb, imgs_per_sec, oom?}]
        self.do_batch_probe = True   # False 则跳过探测，直接用 plans batch
        # probe_only: True 时探测完直接退出（不跑训练 epoch）。默认 False
        # （正常训练），仅 --probe only 命令行置 True。
        self.probe_only = False
        # 探测缓存模式（命令行 --probe 透传）:
        #   auto  (默认): 环境指纹匹配 → 用缓存；不匹配 → 重新探测
        #   force: 忽略缓存，强制重新探测（结果覆盖缓存）
        #   only : 只探测不训练（等价 probe_only=True）
        self.probe_cache_mode = 'auto'
        self.probe_cache_file = None   # 探测结果缓存路径（output_folder 下）

    # ------------------------------------------------------------------
    # NVML 物理显存查询（独立于基类 GPU monitor，探测阶段专用）
    # ------------------------------------------------------------------
    def _init_nvml_for_probe(self):
        """初始化 NVML，返回 {'nvml':..., 'handle':..., 'total_mb':..., 'used_mb':...,
        'power_limit_w':...}。

        失败时返回 None（探测退化为仅靠 OOM 兜底，Linux 上可用）。
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            # 对齐基类 GPU monitor: NVML index 是物理 GPU 序号，local_rank 是逻辑索引
            # （-gpu 0,5,6,7 时 rank1 绑定 GPU5，但 local_rank=1 会让 NVML 读到物理
            #  GPU1=llama 的卡，探测显存预算全错）。用基类的物理索引映射。
            try:
                phys_idx = self._get_physical_gpu_index()
            except Exception:
                phys_idx = self.local_rank
            handle = pynvml.nvmlDeviceGetHandleByIndex(phys_idx)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # 功耗上限（TDP），部分卡不支持查询 → 默认 None（退化处理）
            power_limit_w = None
            try:
                power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:
                pass
            return {
                'nvml': pynvml,
                'handle': handle,
                'total_mb': info.total / 1024 ** 2,
                'used_mb': info.used / 1024 ** 2,
                'power_limit_w': power_limit_w,
            }
        except Exception as e:
            self.print_to_log_file(
                f"[BatchProbe] NVML unavailable ({e}); probe relies on OOM only")
            return None

    @staticmethod
    def _nvml_used_mb(nvml_info):
        """读取当前整卡物理显存已用 MB；失败返回 None。"""
        try:
            info = nvml_info['nvml'].nvmlDeviceGetMemoryInfo(nvml_info['handle'])
            return info.used / 1024 ** 2
        except Exception:
            return None

    @staticmethod
    def _nvml_power_w(nvml_info):
        """读取当前整卡功耗 W；失败返回 None。"""
        try:
            return nvml_info['nvml'].nvmlDeviceGetPowerUsage(nvml_info['handle']) / 1000.0
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 探测核心
    # ------------------------------------------------------------------
    def _probe_batch_size_impl(self):
        """power 逼近探测最大可承受 batch（不超 target）。返回实际 batch。

        判定标准（关键）:
        - 不用 OOM 异常作主判据（Windows WDDM 下分配可"看似成功"但溢出到
          共享显存/系统 RAM，速度断崖且不报错）
        - 主判据 = NVML 物理显存使用率安全阀: 每档探测后整卡 used/total
          超过阈值（默认 0.92）即停止，即使没有 OOM
        - OOM 异常仅作兜底（Linux 或驱动真的报错时）
        """
        if self.device.type != 'cuda' or not self.do_batch_probe:
            return self.batch_size

        # 初始化 NVML 物理显存查询（探测发生在 _init_gpu_monitor 之前，
        # 基类 monitor 在 on_train_start 才初始化，这里独立初始化）
        nvml = self._init_nvml_for_probe()

        plans_bs = self.batch_size
        target = self.target_batch_size
        cap = self.max_probe_batch
        patch = self.configuration_manager.patch_size
        total_mb = nvml['total_mb'] if nvml else None
        baseline_used_mb = nvml['used_mb'] if nvml else None
        if total_mb is None:
            # NVML 不可用（pynvml 未安装等）fallback: torch 侧物理显存查询
            # 不依赖 pynvml，同样能拿到整卡物理容量（无 used_mb/功耗信息）
            try:
                total_mb = torch.cuda.get_device_properties(
                    self.device).total_memory / 1024 ** 2
                self.print_to_log_file(
                    f"[BatchProbe] NVML absent — torch physical VRAM fallback: "
                    f"total={total_mb:.0f}MB")
            except Exception:
                total_mb = None
        self.print_to_log_file(
            f"[BatchProbe] plans batch={plans_bs}, target={target}, cap={cap}, "
            f"GPU physical: total={total_mb}MB, used_now={baseline_used_mb}MB "
            f"(safe_ratio={self.vram_safe_ratio})")

        # 初始余量检查: 当前物理已用已超安全线 → 探测无意义，直接用 plans
        # batch（宁保守不探测，避免探测本身的分配溢出到共享显存）
        if total_mb and baseline_used_mb and baseline_used_mb / total_mb > self.vram_safe_ratio:
            self.print_to_log_file(
                f"[BatchProbe] WARNING: physical VRAM already {baseline_used_mb:.0f}MB/"
                f"{total_mb:.0f}MB = {baseline_used_mb / total_mb:.1%} > "
                f"{self.vram_safe_ratio:.0%} — skip probing, keep plans batch={plans_bs}")
            self.actual_batch_size = plans_bs
            self.grad_accum_steps = max(1, -(-target // plans_bs)) if plans_bs < target else 1
            return plans_bs

        # 物理显存完全不可知（NVML + torch 查询都失败）→ 仅靠 OOM 判定，
        # 无法做单点外推/二分（都依赖 total_mb），退化为倍增探测
        if total_mb is None:
            self.print_to_log_file(
                "[BatchProbe] no physical VRAM info at all — fall back to "
                "OOM-only doubling probe")
            return self._probe_batch_size_oom_only(plans_bs, target, cap, patch)

        # ==================================================================
        # 单点外推法（用户要求: 只测 batch=1 一次，线性外推最大安全 batch）
        # 原理: 显存 ≈ 固定开销(权重/optimizer/CUDA context) + batch×每样本开销
        #   每样本开销 = batch=2 峰值 - batch=1 峰值（差分消掉固定开销误差）
        #   max_batch  = floor(物理可用 × 安全余量 / 每样本开销)
        # 之后单次验证: 外推值跑一次，NVML 超限/OOM 则减半重试
        # ==================================================================
        # 先预热 cuDNN benchmark（探测发生在基类 warmup 之前，若不预热则
        # 第一次 forward 触发 autotune 慢 + 分配大 workspace，污染峰值测量）
        # 用探测网络（DDP 下已替换 SyncBN，forward 无 NCCL 同步）
        gc.collect()
        torch.cuda.empty_cache()
        warm = torch.randn((1, self.num_input_channels, *patch), device=self.device)
        with torch.no_grad():
            for _ in range(2):
                _ = self._probe_net(warm)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        used_before_mb = self._nvml_used_mb(nvml) or 0.0

        # --- 差分测量: batch=1 与 batch=2 的峰值差 = 每样本开销 ---
        self.print_to_log_file(
            f"[BatchProbe] probe batch=1 & 2 (VRAM used_before={used_before_mb:.0f}MB)...")
        ok1, peak1_mb, tp1, pw1, note1 = self._probe_measure(1, patch, nvml, total_mb)
        if not ok1:
            # batch=1 都溢出/超限 → 连最小 batch 都不安全，退回 plans batch
            self.print_to_log_file(
                f"[BatchProbe] batch=1 FAILED ({note1 or 'unknown'}) — keep plans batch={plans_bs}")
            self.actual_batch_size = plans_bs
            self.grad_accum_steps = max(1, -(-target // plans_bs)) if plans_bs < target else 1
            self.num_iterations_per_epoch = self.iterations_per_epoch_effective * self.grad_accum_steps
            return plans_bs
        ok2, peak2_mb, tp2, pw2, note2 = self._probe_measure(2, patch, nvml, total_mb)
        if not ok2:
            peak2_mb = peak1_mb  # batch=2 失败则退化为 batch=1 单点（保守）

        per_sample_mb = peak2_mb - peak1_mb     # 每样本显存增量（差分）
        if per_sample_mb <= 0:
            per_sample_mb = peak1_mb             # 兜底: 差分异常时保守假设
        self.print_to_log_file(
            f"[BatchProbe] batch=1 peak={peak1_mb:.0f}MB, batch=2 peak={peak2_mb:.0f}MB, "
            f"per_sample≈{per_sample_mb:.0f}MB, tp1={tp1:.1f} img/s, "
            f"power1={pw1:.0f}W")

        # --- 线性外推: 物理可用预算内最大 batch（取偶，允许任意 2 的倍数
        # 而非仅 16/32/64 幂次，避免浪费可用显存或跨过吞吐峰值） ---
        avail_mb = total_mb * self.vram_safe_ratio - used_before_mb
        candidate = int(avail_mb / per_sample_mb)
        candidate = max(2, (candidate // 2) * 2)   # 向下取到最近偶数
        candidate = min(candidate, cap)
        self.print_to_log_file(
            f"[BatchProbe] extrapolation: avail={avail_mb:.0f}MB → candidate batch={candidate}")

        # --- 验证循环: 外推值可能因激活非线性/分配器碎片偏高 → 二分逼近
        # 真实边界（任意偶数粒度，而非减半跳变 64→32→16 跳过中间档）。
        # 二分区间 [lo, hi]: lo=已知安全下界, hi=外推 candidate。
        # 每档判定: 显存安全阀超限/OOM/功耗骤降/TP骤降 → 该档不可用（二分
        # 向下）；满载（功耗≥85%TDP）→ 已饱和，直接接受。
        power_limit = nvml['power_limit_w'] if nvml else None
        # 下界: batch=1 已实测安全（前面差分测量成功），上界: 外推 candidate
        lo = 1
        hi = candidate
        best, best_tp, best_pw = 1, tp1, pw1

        # 先验证候选上界 hi（外推值）——若直接可用则无需二分
        ok, peak_mb, tp, pw, note = self._probe_measure(hi, patch, nvml, total_mb)
        if ok:
            best, best_tp, best_pw = hi, tp, pw
            saturated = (power_limit is not None and pw >= power_limit * 0.85)
            self.print_to_log_file(
                f"[BatchProbe]   candidate={hi} VERIFIED: peak={peak_mb:.0f}MB, "
                f"tp={tp:.1f} img/s, power={pw:.0f}W"
                + (" SATURATED" if saturated else ""))
            if saturated:
                self.print_to_log_file(
                    f"[BatchProbe]     GPU saturated ({pw:.0f}W ≥ "
                    f"{power_limit * 0.85:.0f}W = 85% TDP) — accept {hi}")
            else:
                # 未满载但安全 → 该档已是最优（吞吐峰值由外推保证接近）
                pass
        else:
            self.print_to_log_file(
                f"[BatchProbe]   candidate={hi} {note or 'failed'} — binary search "
                f"[{lo}, {hi}]")
            # 二分逼近边界: 显存安全/功耗/TP 均单调，找最大可用档
            saturated_found = False
            while hi - lo > 2:
                mid = lo + ((hi - lo) // 4) * 2   # 中点向下取偶，保证偶数且非恒等
                if mid <= lo or mid >= hi:
                    mid = lo + 2 if lo + 2 < hi else lo
                ok, peak_mb, tp, pw, note = self._probe_measure(mid, patch, nvml, total_mb)
                if ok:
                    best, best_tp, best_pw = mid, tp, pw
                    saturated = (power_limit is not None and pw >= power_limit * 0.85)
                    self.print_to_log_file(
                        f"[BatchProbe]     mid={mid} OK: peak={peak_mb:.0f}MB, "
                        f"tp={tp:.1f} img/s, power={pw:.0f}W"
                        + (" SATURATED" if saturated else ""))
                    if saturated:
                        saturated_found = True
                        break   # 满载即吞吐峰值，直接接受，不再二分
                    lo = mid   # 向上移动下界
                else:
                    self.print_to_log_file(
                        f"[BatchProbe]     mid={mid} {note or 'failed'} — lower hi")
                    hi = mid   # 向下移动上界
            # 注: 不做 "lo 复核"——二分收敛时 L304/L313 同步更新 best=mid 与
            # lo=mid，循环退出时 lo==best 恒成立，复核分支不可达，故不保留。
            self.print_to_log_file(
                f"[BatchProbe]   binary search converged: best={best} "
                f"(tp={best_tp:.1f} img/s, power={best_pw:.0f}W)")

        # 最终选择: 验证成功的 batch（若吞吐峰值在更小档，取候选即可；
        # 单点外推已接近峰值，不再精细搜索）
        self.actual_batch_size = best
        # 梯度累积: 实际 batch < 规定 batch 时，用累积补齐到规定值
        # （effective_batch = actual × accum ≈ nominal）
        self.grad_accum_steps = max(1, -(-self.nominal_batch_size // best)) \
            if best < self.nominal_batch_size else 1
        # 每 epoch 图像总数恒定（跟随规定 batch）:
        #   effective_imgs_per_epoch = 规定迭代数 × 规定 batch = 250 × 16 = 4000
        # 实际 batch 只改变物理迭代数（数据量不变，速度由吞吐决定）:
        #   num_iterations_per_epoch = ceil(4000 / actual_batch)
        #   actual=52 → 77 步   actual=16 → 250 步   actual=2 → 2000 步(accum=8)
        effective_imgs = self.iterations_per_epoch_effective * self.nominal_batch_size
        self.num_iterations_per_epoch = max(
            1, -(-effective_imgs // best))   # ceil 除法
        self.print_to_log_file(
            f"[BatchProbe] RESULT: nominal_batch={self.nominal_batch_size} "
            f"(plans 规定值), actual_batch={best} (探测值, 驱动训练), "
            f"grad_accum_steps={self.grad_accum_steps}, "
            f"effective_batch={best * self.grad_accum_steps} (≈nominal), "
            f"num_iterations_per_epoch={self.num_iterations_per_epoch} "
            f"(imgs/epoch={self.num_iterations_per_epoch * best} ≈ "
            f"effective_imgs={effective_imgs}, "
            f"tp={best_tp:.1f} img/s, power={best_pw:.0f}W/"
            f"{(power_limit or 0):.0f}W TDP if available)")
        return best

    # ------------------------------------------------------------------
    # DDP 安全包装: 探测期无 NCCL + 结果跨 rank 同步
    # ------------------------------------------------------------------
    def _probe_batch_size(self):
        """DDP 安全包装: 探测换裸网络（消除 NCCL）+ 前后 barrier + 结果全局 min。

        背景（2026-08-11 DGX 2 卡事故）: initialize() 在 _warmup_kernels 之前
        已把 network 包成 DDP + SyncBatchNorm。原探测直接用 self.network 做
        forward+backward → SyncBN forward 的 all-reduce 与 DDP backward 的
        梯度 all-reduce 在各 rank 探测序列/时机不同步时 collective 错位，
        NCCL 死锁 600s 超时（且 OOM 中断 backward 会污染 NCCL work 队列）。
        修复: 探测阶段换用裸网络（network.module）并临时替换 SyncBatchNorm
        → BatchNorm，探测期彻底无 NCCL；探测结束 barrier 对齐后 all_reduce
        取全局最小 batch（V100-32GB 与 16GB 卡探测值不同，DDP 训练要求
        所有 rank batch 一致，取 min 保证小卡不 OOM）。返回值即同步后的
        batch（_warmup_kernels 的 _apply_probe_result 消费它）。
        """
        if self.device.type != 'cuda' or not self.do_batch_probe:
            return self.batch_size

        restore = self._enter_probe_network()
        try:
            if self.is_ddp:
                # 显式 device_ids 避免 torch 提示"用当前设备"的 UserWarning
                # （run_ddp 已 set_device(cuda, rank)，这里只是消除噪音）
                dist.barrier(device_ids=[self.device.index])
            best = self._probe_batch_size_impl()
        finally:
            restore()   # 还原 SyncBatchNorm（训练继续用 DDP 网络）

        if self.is_ddp:
            # 所有 rank 探测完才进入 warmup/training（避免一方先进入 warmup
            # Phase 2 的真实 DDP 同步而另一方还在探测 → collective 错位）
            dist.barrier(device_ids=[self.device.index])
            # 探测结果同步: 各 rank 用自己的卡探测出不同 batch，DDP 训练
            # batch 必须全局一致 → 取所有 rank 最小值（小卡不 OOM）
            best_t = torch.tensor([self.actual_batch_size],
                                  device=self.device, dtype=torch.int32)
            dist.all_reduce(best_t, op=dist.ReduceOp.MIN)
            best = int(best_t.item())
            self.actual_batch_size = best
            # 基于同步后的 batch 重算派生量（探测期各 rank 用本地值算过，
            # 同步取 min 后可能变小，累积步数/迭代数随之变化）
            self.grad_accum_steps = max(
                1, -(-self.nominal_batch_size // best)) \
                if best < self.nominal_batch_size else 1
            effective_imgs = (self.iterations_per_epoch_effective
                              * self.nominal_batch_size)
            self.num_iterations_per_epoch = max(1, -(-effective_imgs // best))
            self.print_to_log_file(
                f"[BatchProbe] DDP sync: global_min_batch={best}, "
                f"accum={self.grad_accum_steps}, "
                f"iters/epoch={self.num_iterations_per_epoch}")
        return best

    def _enter_probe_network(self):
        """探测阶段用裸网络（DDP 下绕过 wrapper 的梯度同步），并临时把
        SyncBatchNorm 替换为普通 BatchNorm（其 forward 不再触发 NCCL
        all-reduce）。返回恢复函数。

        注意: DDP 的 no_sync() 只能抑制 backward 的梯度 all-reduce，
        SyncBatchNorm 的 forward 统计量同步无开关可关，必须替换模块。
        替换机制: 新 BN 模块的 Parameter 是全新拷贝（values .data.copy_()，
        非引用 SyncBN 参数对象），optimizer 仍持原 SyncBN 的陈旧 Parameter 引用；
        但探测期仅 forward+backward+zero_grad、不调 optimizer.step()，故陈旧
        引用无副作用；restore() 把原 SyncBN 挂回后 optimizer 自动指向恢复的
        Parameter，训练参数正常更新。
        """
        self._probe_net = self.network.module if self.is_ddp else self.network
        # 仅 DDP 下才存在 SyncBN（convert_sync_batchnorm 在 DDP wrap 前调用）
        if not self.is_ddp:
            return lambda: None
        import torch.nn as nn
        is_3d = len(self.configuration_manager.patch_size) == 3
        bn_cls = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
        replaced = []   # [(parent, attr, original_module)]
        for name, mod in self._probe_net.named_modules():
            if isinstance(mod, nn.SyncBatchNorm):
                bn = bn_cls(mod.num_features, eps=mod.eps,
                            momentum=mod.momentum, affine=mod.affine,
                            track_running_stats=mod.track_running_stats)
                if mod.affine:
                    bn.weight.data.copy_(mod.weight.data)
                    bn.bias.data.copy_(mod.bias.data)
                # track_running_stats=False 时 running_* 为 None，需判空
                if mod.running_mean is not None:
                    bn.running_mean.copy_(mod.running_mean)
                    bn.running_var.copy_(mod.running_var)
                    bn.num_batches_tracked.copy_(mod.num_batches_tracked)
                parent_name, _, attr = name.rpartition('.')
                parent = (self._probe_net.get_submodule(parent_name)
                          if parent_name else self._probe_net)
                setattr(parent, attr, bn)
                replaced.append((parent, attr, mod))
        if replaced:
            self.print_to_log_file(
                f"[BatchProbe] probe network: replaced {len(replaced)} "
                f"SyncBatchNorm → {bn_cls.__name__} (no NCCL during probe)")

        # 探测阶段必须同时关闭 loss 的 DDP all-gather: nnUNet 的 DiceLoss
        # 在 ddp=True & batch_dice=True 时用 AllGatherGrad —— 其 forward 即
        # torch.distributed.all_gather、backward 即 all_reduce（见
        # nnunetv2/utilities/ddp_allgather.py）。即使探测网络是裸网络，
        # loss 内部仍会触发 NCCL → 两个 rank 探测节奏不同即 collective
        # 错位死锁（2026-08-11 DGX 实测卡死在 probe batch=1&2）。
        # ddp=False 后 loss 退化为纯本地计算（batch_dice 聚合不涉及 dist）。
        self._probe_loss_ddp_backup = []
        for mod in self.loss.modules():
            if hasattr(mod, 'ddp') and getattr(mod, 'ddp'):
                self._probe_loss_ddp_backup.append(mod)
                mod.ddp = False
        if self._probe_loss_ddp_backup:
            self.print_to_log_file(
                f"[BatchProbe] probe loss: disabled ddp all-gather on "
                f"{len(self._probe_loss_ddp_backup)} loss module(s)")

        def restore():
            for parent, attr, orig in replaced:
                setattr(parent, attr, orig)
            for mod in self._probe_loss_ddp_backup:
                mod.ddp = True
        return restore

    def _probe_batch_size_oom_only(self, plans_bs, target, cap, patch):
        """物理显存完全不可知（NVML + torch 查询均失败）时的退化路径:
        仅靠 OOM 判定，无法做单点外推/二分（都依赖 total_mb）。

        策略: batch=4 起倍增（4,8,16,...cap，batch=1/2 已由差分测量成功），
        首个 OOM 即停，取最后成功档。_probe_measure 在 nvml=None 时:
        - 峰值用 torch.cuda.max_memory_allocated（无 NVML 依赖）
        - 安全阀 ratio=None → 跳过
        - 功耗 0 且 power_limit=None → 满载判定恒 False
        唯一判据就是 OOM 异常兜底。
        """
        gc.collect()
        torch.cuda.empty_cache()
        best, best_tp, best_peak = 1, 0.0, 0.0
        b = 4
        while b <= cap:
            ok, peak_mb, tp, pw, note = self._probe_measure(b, patch, None, None)
            if not ok:
                self.print_to_log_file(
                    f"[BatchProbe]   batch={b} {note or 'failed'} — stop "
                    f"(OOM-only path)")
                break
            best, best_tp, best_peak = b, tp, peak_mb
            self.print_to_log_file(
                f"[BatchProbe]   batch={b} OK: peak={peak_mb:.0f}MB, "
                f"tp={tp:.1f} img/s")
            b *= 2
        self.actual_batch_size = best
        self.grad_accum_steps = max(1, -(-self.nominal_batch_size // best)) \
            if best < self.nominal_batch_size else 1
        effective_imgs = self.iterations_per_epoch_effective * self.nominal_batch_size
        self.num_iterations_per_epoch = max(1, -(-effective_imgs // best))
        self.print_to_log_file(
            f"[BatchProbe] RESULT (OOM-only): actual_batch={best}, "
            f"grad_accum_steps={self.grad_accum_steps}, "
            f"num_iterations_per_epoch={self.num_iterations_per_epoch}, "
            f"tp={best_tp:.1f} img/s")
        return best

    def _probe_measure(self, batch: int, patch, nvml, total_mb):
        """单档探测: 返回 (ok, peak_mb, imgs_per_sec, note)。

        峰值测量（关键）:
        - 预热 2 次 trial 让 cuDNN autotune 完成（batch 尺寸改变会触发
          autotune，其一次性 workspace 峰值可达数 GB，不代表稳态训练内存）
        - empty_cache + reset_peak 后测第三次 trial 的稳态峰值
        - 吞吐用 probe_iters 次独立计时取最快（缓存已热，计时稳定）
        - 物理显存安全阀: 整卡 used/total > vram_safe_ratio → ok=False（不报
          OOM 但继续分配会溢出共享显存，必须主动停止）
        - OOM 异常兜底（Linux / 驱动真报错）
        """
        gc.collect()
        torch.cuda.empty_cache()
        try:
            # 预热: 2 次 trial 完成 cuDNN autotune（workspace 峰值不纳入测量）
            self._probe_trial(batch, patch)
            self._probe_trial(batch, patch)
            self.optimizer.zero_grad(set_to_none=True)
            # 稳态峰值: 清缓存 + 重置统计后测一次
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            self._probe_trial(batch, patch)
            peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            # 物理显存安全阀
            cur_used_mb = self._nvml_used_mb(nvml)
            ratio = (cur_used_mb / total_mb) if (cur_used_mb and total_mb) else None
            if ratio is not None and ratio > self.vram_safe_ratio:
                return False, peak_mb, 0.0, 0.0, \
                    f"VRAM {cur_used_mb:.0f}/{total_mb:.0f}MB ({ratio:.0%} > {self.vram_safe_ratio:.0%})"
            # 吞吐: 多次计时取最快（trial 内不重复 reset 峰值，避免干扰）
            # 同时轮询 NVML 功耗取峰值（GPU 满载时功耗高）
            self.optimizer.zero_grad(set_to_none=True)
            times = []
            power_peak = 0.0
            for _ in range(max(1, self.probe_iters)):
                torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()
                # 采样功耗要在 GPU 忙时进行——trial 是异步提交的，
                # _probe_trial 返回后 GPU 仍在执行；此时轮询 NVML 才能
                # 采到峰值功耗（synchronize 之后 GPU 已空闲，功耗回落，
                # 只能采到余温）
                self._probe_trial(batch, patch)
                for _ in range(3):   # 忙时轮询 3 次取峰值
                    p = self._nvml_power_w(nvml)
                    if p:
                        power_peak = max(power_peak, p)
                torch.cuda.synchronize(self.device)
                times.append(time.perf_counter() - t0)
            tp = batch / min(times) if times else 0.0
            self.probe_log.append({'batch': batch, 'peak_mb': round(peak_mb, 1),
                                   'imgs_per_sec': round(tp, 1),
                                   'power_w': round(power_peak, 1)})
            return True, peak_mb, tp, power_peak, None
        except RuntimeError as e:
            if self._is_oom(e):
                # OOM 后释放缓存块 + 回收 Python 对象，避免碎片/残留导致
                # 后续更小 batch 档误判 OOM（实测 32GB 卡 batch=7 误报）
                gc.collect()
                torch.cuda.empty_cache()
                self.probe_log.append({'batch': batch, 'peak_mb': None,
                                       'oom': str(e)[:120]})
                return False, None, 0.0, 0.0, 'OOM'
            raise

    def _probe_trial(self, batch: int, patch):
        """单档探测: 一次完整 forward+loss+backward（train 模式）。

        与 _warmup_kernels Phase 2 一致，但不做 optimizer.step（探测纯为
        显存测量，step 会让权重漂移影响后续训练）。DS 模式下 target 需
        按每个输出分辨率生成 list（DeepSupervisionWrapper 断言要求）。
        """
        # 探测用裸网络（DDP 下 _probe_net = network.module + BN 替换），
        # 非 DDP 时 _probe_net = self.network，与基类行为一致
        net = getattr(self, '_probe_net', self.network)
        num_output_channels = self.label_manager.num_segmentation_heads
        dummy_batch = torch.randn(
            (batch, self.num_input_channels, *patch), device=self.device)
        with torch.no_grad():
            dummy_out = net(dummy_batch)
        ds_output_shapes = [o.shape for o in dummy_out] if isinstance(dummy_out, (list, tuple)) else None
        if ds_output_shapes is not None:
            dummy_target = [
                torch.randint(0, max(1, num_output_channels), (batch, *s[1:]),
                              device=self.device, dtype=torch.long)
                for s in ds_output_shapes
            ]
        else:
            dummy_target = torch.randint(
                0, max(1, num_output_channels), (batch, 1, *patch),
                device=self.device, dtype=torch.long)
        with torch.autocast(self.device.type, dtype=self.autocast_dtype,
                            enabled=self.device.type == 'cuda'):
            output = net(dummy_batch)
            l = self.loss(output, dummy_target)
        l.backward()
        # 不 step，仅测量; zero_grad 释放梯度供下一档复用
        self.optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def _halve_even(x: int) -> int:
        """减半并取最近偶数（最低 2）。

        注意 (x // 2) * 2 对偶数 x 是恒等变换（64→64），必须用
        (x // 4) * 2：64→32, 58→28, 30→14。确保每轮验证真降档。
        """
        return max(2, (x // 4) * 2)

    @staticmethod
    def _is_oom(e: RuntimeError) -> bool:
        """OOM 判定（兼容 cuDNN / CPU allocator 错误消息）"""
        msg = str(e)
        return any(s in msg for s in (
            "out of memory", "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED",
            "DefaultCPUAllocator: can't allocate memory",
            "CUDA out of memory"))

    # ------------------------------------------------------------------
    # 梯度累积 train_step（真实累积，accum 步 forward+backward 后统一 step）
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        """梯度累积版训练步。

        每 accum 步: 前 accum-1 步只 forward+backward（梯度累积到 param.grad，
        DDP 下用 self.network.no_sync() 抑制非边界步的梯度 all-reduce），第
        accum 步做 clip+step+update。loss 除以 grad_accum_steps 保持梯度尺度
        与 batch=nominal 一致。accum=1 时行为与基类完全一致。

        DDP no_sync 收益: accum=N 时 NCCL all-reduce 次数从 N 降到 1（边界步
        统一 all-reduce），DGX V100 PCIe 实测训练吞吐提升约 5-15%。DDP 包装
        在最外层（torch.compile 在 DDP 内层），no_sync 上下文可用。
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        accum = getattr(self, 'grad_accum_steps', 1)
        self._accum_step_counter = getattr(self, '_accum_step_counter', 0) + 1
        is_accum_boundary = (self._accum_step_counter % accum == 0)

        with autocast(self.device.type, dtype=self.autocast_dtype, enabled=True) \
                if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)
            if accum > 1:
                l = l / accum   # 累积平均，保持梯度尺度一致

        # DDP + accum>1: 非边界步用 no_sync 抑制梯度 all-reduce（每步只累积
        # 本地梯度，边界步统一 all-reduce + step）。单卡 / accum=1 走
        # dummy_context，与基类行为一致。
        ddp_no_sync = (self.is_ddp and accum > 1 and not is_accum_boundary)
        no_sync_ctx = (self.network.no_sync() if ddp_no_sync
                       else dummy_context())
        with no_sync_ctx:
            if self.grad_scaler is not None:
                self.grad_scaler.scale(l).backward()
            else:
                l.backward()
        # 边界步: unscale + clip + step + update + 清梯度
        if is_accum_boundary:
            if self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_step_counter = 0
        return {'loss': l.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    # 探测结果缓存（环境指纹匹配，训练环境未更改时直接复用）
    # ------------------------------------------------------------------
    def _probe_fingerprint(self) -> dict:
        """环境指纹: GPU + 配置 + 数据。任一变化 → 缓存失效。"""
        try:
            gpu_name = torch.cuda.get_device_name(self.local_rank)
        except Exception:
            gpu_name = 'unknown'
        vram_total_mb = None
        nvml = self._init_nvml_for_probe()
        if nvml:
            vram_total_mb = nvml['total_mb']
        # 网络架构类: 优先从 configuration_manager 的 architecture 取
        try:
            network_class = self.configuration_manager.architecture['network_class_name']
        except Exception:
            try:
                network_class = self.plans_manager.plans['configurations'][
                    self.configuration_name]['architecture'].get(
                        'network_class', '')
            except Exception:
                network_class = ''
        return {
            'gpu_name': gpu_name,
            'vram_total_mb': vram_total_mb,
            'trainer_class': self.__class__.__name__,
            'dataset': self.dataset_json.get('dataset_name', ''),
            'fold': self.fold,
            'plans_identifier': self.plans_manager.plans_name,
            'configuration': self.configuration_name,
            'patch_size': list(self.configuration_manager.patch_size),
            'network_class': network_class,
            'batch_nominal': self.nominal_batch_size,
        }

    def _load_probe_cache(self) -> dict:
        """读取缓存文件并校验指纹。命中返回缓存 dict，否则 None。"""
        cache_file = getattr(self, 'probe_cache_file', None)
        if not cache_file or not isfile(cache_file):
            return None
        try:
            cache = load_json(cache_file)
        except Exception:
            return None
        fp = self._probe_fingerprint()
        cached_fp = cache.get('fingerprint', {})
        # 指纹逐字段比较（None 值跳过——NVML 不可用时不做显存校验）
        for k, v in fp.items():
            if v is None:
                continue
            if k not in cached_fp or cached_fp[k] != v:
                self.print_to_log_file(
                    f"[BatchProbe] cache MISS: fingerprint field '{k}' "
                    f"changed ({cached_fp.get(k)} != {v})")
                return None
        return cache

    def _save_probe_cache(self, results: dict):
        """探测结果写入缓存文件（含指纹 + 时间戳）。"""
        cache_file = getattr(self, 'probe_cache_file', None)
        if not cache_file:
            return
        try:
            cache = {
                'fingerprint': self._probe_fingerprint(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                **results,
            }
            maybe_mkdir_p(os.path.dirname(cache_file))
            save_json(cache, cache_file)
            self.print_to_log_file(
                f"[BatchProbe] cache saved: {cache_file}")
        except Exception as e:
            self.print_to_log_file(
                f"[BatchProbe] cache save FAILED ({e}) — training continues")

    def _apply_probe_result(self, new_bs: int):
        """应用探测结果到训练器（batch_size + 派生量），并同步到缓存。"""
        if new_bs != self.batch_size:
            self.print_to_log_file(
                f"[BatchProbe] applying probed batch_size {new_bs} "
                f"(nominal was {self.nominal_batch_size})")
            self.batch_size = new_bs

    # ------------------------------------------------------------------
    # 插入点: warmup 前探测（此时 network/optimizer/loss 就绪，
    # dataloader 未建；探测结果被 get_dataloaders 消费）
    # ------------------------------------------------------------------
    def _warmup_kernels(self):
        if self.do_batch_probe:
            # 缓存文件路径: output_folder 在 initialize 中已设置
            if not self.probe_cache_file and self.output_folder:
                self.probe_cache_file = join(
                    self.output_folder, 'batch_probe_cache.json')
            mode = getattr(self, 'probe_cache_mode', 'auto')
            # 'only' 模式: 只探测不训练
            if mode == 'only':
                self.probe_only = True

            new_bs = None
            cache = None
            if mode != 'force':
                cache = self._load_probe_cache()
                if cache is not None:
                    new_bs = cache.get('actual_batch_size')
                    self.actual_batch_size = new_bs   # 与 batch_size 同步（__init__ 留了 None）
                    self.grad_accum_steps = cache.get('grad_accum_steps', 1)
                    self.num_iterations_per_epoch = cache.get(
                        'num_iterations_per_epoch',
                        self.iterations_per_epoch_effective)
                    self.print_to_log_file(
                        f"[BatchProbe] cache HIT: actual_batch={new_bs}, "
                        f"accum={self.grad_accum_steps}, "
                        f"iters/epoch={self.num_iterations_per_epoch} "
                        f"(saved {cache.get('timestamp', '?')})")

            # DDP 下同步探测决策: 任何 rank 缓存失效 → 所有 rank 一起重新探测。
            # 否则 MISS 的 rank 进入 _probe_batch_size 的 barrier，而 HIT 的 rank
            # 直接跳过 → barrier 等不到 → 死锁（2026-08-11 4 卡混合 cache 实锤）。
            need_probe = new_bs is None
            if self.is_ddp:
                _need_t = torch.tensor([1 if need_probe else 0], device=self.device)
                dist.all_reduce(_need_t, op=dist.ReduceOp.MAX)
                need_probe = bool(_need_t.item())
                if need_probe and new_bs is not None:
                    self.print_to_log_file(
                        "[BatchProbe] peer rank cache MISS — forcing re-probe on all ranks (DDP sync)")
            if need_probe:
                # 无缓存/缓存失效/force/peer MISS → 全 rank 统一真实探测
                new_bs = self._probe_batch_size()
                # 探测结果写缓存（含指纹）
                self._save_probe_cache({
                    'actual_batch_size': self.actual_batch_size,
                    'grad_accum_steps': self.grad_accum_steps,
                    'num_iterations_per_epoch': self.num_iterations_per_epoch,
                })

            self._apply_probe_result(new_bs)
            if self.probe_only:
                # 纯测试模式: 探测完直接退出，不跑训练 epoch
                # （initialize 尚未完全收尾，但结果已打印到日志）
                import sys
                self.print_to_log_file(
                    f"[BatchProbe] probe_only=True — exiting after probe. "
                    f"Suggested: nominal={self.nominal_batch_size}, "
                    f"actual_batch={new_bs}, accum_steps={self.grad_accum_steps}, "
                    f"iterations/epoch={self.num_iterations_per_epoch}")
                sys.exit(0)
        super()._warmup_kernels()
