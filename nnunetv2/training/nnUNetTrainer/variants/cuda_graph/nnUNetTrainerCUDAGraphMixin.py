"""
nnUNetTrainerCUDAGraphMixin - CUDA Graphs training acceleration mixin
======================================================================

Why
---
With small batch sizes (e.g. batch=4) and fixed patch sizes, GPU utilization is
dominated by CPU-side kernel launch overhead: each step launches thousands of
small kernels, each costing 1-5 us of CPU dispatch. CUDA Graphs capture the whole
forward+loss+backward kernel sequence into a single graph and replay it with one
launch, eliminating the per-kernel CPU overhead. Measured speedup at batch=4 is
~1.3-1.65x (Llama-3.1-8B benchmark), and it is fully compatible with mamba_ssm's
triton kernels because capture records the actual launched kernels (no dynamo
tracing, unlike torch.compile).

Constraints
-----------
- Single GPU only. Falls back to eager training under DDP.
- Networks must have dropout_prob=0 (a captured dropout mask would be frozen).
- When grad_scaler is present (fp16): backward is captured, but
  unscale_/step/update stay OUTSIDE the graph (scaler state is Python control
  flow; capturing it would freeze the loss scale).
- When grad_scaler is None (bf16, e.g. LightMamba2Net): full step benefits most.

Usage
-----
    from nnunetv2.training.nnUNetTrainer.variants.cuda_graph.nnUNetTrainerCUDAGraphMixin import (
        nnUNetTrainerCUDAGraphMixin)

    class MyTrainer(nnUNetTrainerCUDAGraphMixin, BaseTrainer):
        pass

Capture strategy: lazy capture on the FIRST train_step call, using the real
batch's shapes to build static buffers (guarantees shapes match the dataloader
exactly, including deep-supervision target lists). Subsequent steps replay.
"""
import gc
import os

import torch
from torch.amp import autocast


class nnUNetTrainerCUDAGraphMixin:
    # Set to False in a subclass to disable CUDA Graphs (for eager baselines).
    use_cuda_graphs = True
    # 验证独立小 batch（None = 用训练 batch）。CUDA Graph 私有池锁定大 batch
    # 训练激活后，验证若仍用大 batch 会与锁定激活叠加溢出（WDDM 静默 swap
    # 断崖降速）。缩小验证 batch 让验证 forward 激活装进剩余显存，零重捕获
    # 开销。这是默认方案（方案 D）。
    val_batch_size = 8
    # 重型网络训练器集中登记表（batch=4 训练显存实测 ≥ ~4G 者）:
    # M2Net ~9G / SSND2Net ~14G / LightMamba2Net ~6G / LM2Net ~5.5G /
    # SwT2Net ~5G / UNETR2Net ~4G。新增重型训练器只需把类名加入此集合，
    # 无需在每个训练器文件里重复标记（get_val_batch_size 按类名自动检测）。
    HEAVY_MODEL_TRAINERS = frozenset({
        'nnUNetTrainerM2NetBatchProbeCUDAGraph',
        'nnUNetTrainerSSND2NetBatchProbeCUDAGraph',
        'nnUNetTrainerLightMamba2NetBatchProbeCUDAGraph',
        'nnUNetTrainerLightMamba2NetCUDAGraph',
        'nnUNetTrainerLM2NetBatchProbeCUDAGraph',
        'nnUNetTrainerSwT2NetBatchProbeCUDAGraph',
        'nnUNetTrainerUNETR2NetBatchProbeCUDAGraph',
    })
    # 显式标记: True/False 覆盖自动检测；None（默认）→ 按 HEAVY_MODEL_TRAINERS
    # 类名自动判定。重型时验证 batch 自动减半（默认 8 → 4），避免验证 eager
    # forward 叠加在锁定的 CUDA Graph 私有池上把峰值推过物理显存。
    heavy_model = None
    # 验证前是否释放 graph 以腾显存（方案 E，备选）。默认关闭——释放后下一
    # 个 train_step 需重新捕获，warmup 会重跑 cuDNN autotune（SegResNet 实测
    # 每 epoch ~78s，比训练本身还慢），故默认走 val_batch_size 小 batch 方案。
    # 若需大验证 batch 且接受重捕获开销，子类可置 True 并设 val_batch_size=None。
    release_graph_for_validation = False
    # 验证 forward 是否也捕获 CUDA Graph(默认开)。验证是 36 次/图
    # (9 tile × 4 mirror TTA)的 launch-bound 前向，捕获收益远大于训练侧。
    # 置 False 或设环境变量 NNUNET_VAL_NO_CUDAGRAPH=1 可回退 eager(A/B 基线)。
    use_cuda_graph_for_validation = True
    # 捕获后私有池外置检测阈值：若整卡空闲显存占比低于该值，判定 graph 私有池
    # 已被 WDDM 驱动静默溢出到共享内存（replay 走 PCIe，慢约 8x），弃用 graph。
    # 配合 BatchProbeCUDAGraph 的 0.80 探测安全阀，正常情况下捕获后仍有 >5%
    # 空闲，不会误判；仅当整卡被压满（异常）时触发。
    _GRAPH_SPILL_FREE_RATIO = 0.02
    # 降 batch 重捕获的最大次数。每次降档都要重建 dataloader（augmenter worker
    # 进程重启，约 1 分钟）并重新捕获（1-2 分钟），若不设上限，最坏情况会长时间
    # 反复尝试。实测本项目显存超订幅度下，降 1-2 个 batch 即可显著缓解。
    _MAX_BATCH_REDUCTIONS = 4


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_graph: torch.cuda.CUDAGraph | None = None
        self.static_input: torch.Tensor | None = None
        self.static_target = None          # Tensor or list of Tensors (DS)
        self.static_loss: torch.Tensor | None = None
        self._capture_attempted = False
        # 梯度累积支持: 从宿主类继承 grad_accum_steps（BatchProbe 设置），
        # 无累积时默认为 1（行为与原始 mixin 完全一致）
        self._accum_step_counter = 0
        # 本次 iteration 是否累积边界（由 train_step 计算，内层 BatchProbe 只读）
        self._accum_boundary = True

    # ------------------------------------------------------------------ #
    #  Training step override: replay path + lazy capture on first call   #
    # ------------------------------------------------------------------ #
    def train_step(self, batch: dict) -> dict:
        # Move data to device up-front so both eager and graph paths share it
        data = batch['data'].to(self.device, non_blocking=True)
        if isinstance(batch['target'], list):
            target = [t.to(self.device, non_blocking=True) for t in batch['target']]
        else:
            target = batch['target'].to(self.device, non_blocking=True)

        # accum 下限保护：accum<=0 会让边界判定退化成"每步都是边界"，独立
        # BatchProbe 的 `counter % accum` 更会直接除零。
        accum = max(1, getattr(self, 'grad_accum_steps', 1))
        # 累积边界判定只在此处递增一次计数器，结果经 _accum_boundary 传给内层
        # train_step（BatchProbe）。历史问题：mixin 与 BatchProbe 各自递增同一个
        # _accum_step_counter，计数器被双递增 → BatchProbe 每步都看到偶数 →
        # is_accum_boundary 恒 True → 梯度累积完全失效（每步都 all-reduce 全部
        # 梯度 + optimizer.step，no_sync 一次不用）。内层改为只读 token 后
        # accum>1 才真正生效。
        # 规则（本项目唯一实现；BatchProbe 独立使用处逐字对齐，勿只改一边）：
        # 计数到 accum 归零并标记边界。
        self._accum_step_counter += 1
        if self._accum_step_counter >= accum:
            self._accum_step_counter = 0
            self._accum_boundary = True
        else:
            self._accum_boundary = False
        is_boundary = self._accum_boundary

        # --- Replay path (steady state) ---
        if self.cuda_graph is not None:
            # 每次 replay 的完整训练步。⚠️ zero_grad 必须用 set_to_none=False：
            # CUDA Graph 捕获的 backward 重放时写入的是捕获时记录的固定 grad 内存
            # 地址；set_to_none=True 会释放该内存并置 param.grad=None，导致图内
            # backward 的梯度不可见，GradScaler.step 报 "No inf checks were
            # recorded"。就地置零(zero_())保持地址稳定。
            #
            # 梯度累积（accum>1，与 BatchProbe.train_step / eager 版同语义）:
            # 非边界步只 replay（backward 在已有 grad 上**累加**，autograd 语义，
            # 写入同一地址）；边界步 replay 完成第 N 次累积后 step，再 zero_grad
            # 开启下一周期。顺序必须是 replay → step → zero（eager 版亦为 step
            # 后清零）；若在 replay 前清零会丢弃前 N-1 步已累积的梯度，每 step
            # 只剩 1 个 micro-batch 起作用（2026-09-15 实测 grad-norm 0.30 vs
            # 正确 0.92）。
            self._copy_into_static(data, target)
            self.cuda_graph.replay()
            if is_boundary:
                self._graph_optimizer_step()
                self.optimizer.zero_grad(set_to_none=False)
            return {'loss': self.static_loss.detach().cpu().numpy()}

        # --- Lazy capture on first step (single GPU only) ---
        # 捕获失败/私有池外置时不再永久回退 eager，而是降 1 个 batch 后重捕获
        # （eager 回退会让整个 run 失去 1.3-6x 的 graph 收益，详见该方法注释）。
        if (self.use_cuda_graphs and self.device.type == 'cuda' and not self.is_ddp
                and not self._capture_attempted):
            return self._attempt_capture_with_batch_fallback(data, target)

        # --- Eager path (first step, DDP, CPU, 或 batch 已降至下限仍无法捕获) ---
        return super().train_step({'data': data, 'target': target})

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #
    def _attempt_capture_with_batch_fallback(self, data, target):
        """捕获 CUDA Graph；VRAM 不足时逐次降 1 个 batch 重试，而非回退 eager。

        历史行为：捕获失败或私有池被 WDDM 外置 → 永久回退 eager。但 graph 相对
        eager 有 1.3-6x 收益（本项目实测 SwinTransformerUnet 11.4x、SwinUMamba
        4.4x、U-Net 5.9x），一旦回退整个 run 都无法恢复。现改为降低训练 batch
        （每次 -1，见 _reduce_batch_for_recapture）后重建 dataloader 并重捕获，
        以牺牲少量 batch 换取保持 graph 加速。仅当 batch 已降到 1 仍失败（或降档
        次数达 _MAX_BATCH_REDUCTIONS）才回退 eager 并明确告警。

        返回：{'loss': ...}，与 train_step 契约一致。
        """
        while True:
            self._capture_attempted = True   # 标记本次尝试；降档时会复位以便重试
            try:
                captured = self._capture_cuda_graph(data, target)
            except Exception as e:
                # OOM 或不受支持的算子：不立即回退，交给下面的降档逻辑
                captured = False
                self.print_to_log_file(
                    f"CUDA Graphs: capture FAILED ({type(e).__name__}: {e}).")

            if captured:
                # capture 自身已完成一次完整 step（_graph_optimizer_step），等同于
                # 一个累积边界 → 计数器归零，避免后续 replay 的累积相位错位。
                self._accum_step_counter = 0
                self._accum_boundary = True
                # capture 已用真实数据完成一次完整训练步（forward+backward+step），
                # 直接返回该 loss，不再重复 replay 同一 batch
                return {'loss': self.static_loss.detach().cpu().numpy()}

            # 捕获失败：彻底释放 graph 私有池与静态 buffer，再决定是否降档
            self.cuda_graph = None
            self.static_input = self.static_target = self.static_loss = None
            gc.collect()
            torch.cuda.empty_cache()

            if not self._reduce_batch_for_recapture():
                self.print_to_log_file(
                    "CUDA Graphs: 无法继续降 batch，回退 eager 训练。")
                return super().train_step({'data': data, 'target': target})

            # 用降档后的 dataloader 取真实 batch（捕获要求 shape 与 dataloader 一致）
            batch = next(self.dataloader_train)
            data = batch['data'].to(self.device, non_blocking=True)
            if isinstance(batch['target'], list):
                target = [t.to(self.device, non_blocking=True) for t in batch['target']]
            else:
                target = batch['target'].to(self.device, non_blocking=True)

    def _reduce_batch_for_recapture(self) -> bool:
        """训练 batch 减 1 并重建全部派生状态，供下一次图捕获使用。

        返回 True 表示降档成功、可以重试捕获；False 表示不能继续降档。

        为什么每一步都必须做（漏改任一项都会让训练循环静默错乱）
        ------------------------------------------------------
        (1) batch_size：nnUNetDataLoader 在**构造时**就固化了 batch_size（用于采样
            数量与批次张量预分配），直接改训练器属性不会改变已存在 loader 的输出，
            因此必须重建 dataloader。
        (2) grad_accum_steps：CUDA Graph 内的 `loss / accum` 缩放是在捕获时**固化进
            计算图**的（见 _graph_forward_backward），accum 与新 batch 不匹配会导致
            梯度尺度错误；replay 的累积相位计数也依赖它。
        (3) num_iterations_per_epoch：它是 run_training 的显式循环上界（不由 batch
            推导），须按「每 epoch 样本数恒定」重算，否则每 epoch 数据量随 batch
            缩小而减少。注意本 epoch 上界已固定，新值自下个 epoch 起生效。
        (4) val 迭代数：get_dataloaders 会按 batch 比例放大验证迭代数以保持验证图数
            恒定，该换算必须幂等（见 get_dataloaders 的基线记录），否则重建时会二次
            放大。
        """
        if self.batch_size <= 1:
            self.print_to_log_file(
                "[CUDAGraph] batch 已为 1，无法继续降档。")
            return False
        done = getattr(self, '_batch_reduction_count', 0)
        if done >= self._MAX_BATCH_REDUCTIONS:
            self.print_to_log_file(
                f"[CUDAGraph] 已降档 {done} 次（上限 "
                f"{self._MAX_BATCH_REDUCTIONS}），停止降档。")
            return False
        self._batch_reduction_count = done + 1

        old_bs = self.batch_size
        self.batch_size = old_bs - 1
        # BatchProbe 语义：actual_batch_size 才是驱动训练的实际值
        self.actual_batch_size = self.batch_size

        nominal = getattr(self, 'nominal_batch_size', None)
        if nominal:
            # 保持有效 batch ≈ nominal（与 BatchProbe 同款 ceil 公式）
            self.grad_accum_steps = max(1, -(-nominal // self.batch_size))
            iters_base = getattr(self, 'iterations_per_epoch_effective', None)
            if iters_base:
                effective_imgs = iters_base * nominal
                self.num_iterations_per_epoch = max(
                    1, -(-effective_imgs // self.batch_size))

        # 关闭旧 dataloader（结束其 worker 进程/线程），再按新 batch 重建
        for dl in (getattr(self, 'dataloader_train', None),
                   getattr(self, 'dataloader_val', None)):
            if dl is None:
                continue
            try:
                dl._finish()
            except Exception as e:  # noqa: BLE001 — 关闭失败不应阻断降档
                self.print_to_log_file(
                    f"[CUDAGraph] 关闭旧 dataloader 失败（继续降档）: "
                    f"{type(e).__name__}: {e}")
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        # 复位捕获/累积状态。_capture_attempted 不能靠 _release_cuda_graph 复位
        # （它在 cuda_graph is None 时早退，而捕获失败时恰好是 None）。
        self._capture_attempted = False
        self._accum_step_counter = 0
        self._accum_boundary = True
        # 丢弃失败捕获残留的梯度，避免与新 batch 的梯度叠加
        self.optimizer.zero_grad(set_to_none=True)

        self.print_to_log_file(
            f"[CUDAGraph] VRAM 不足，训练 batch {old_bs} → {self.batch_size}"
            f"（第 {self._batch_reduction_count}/{self._MAX_BATCH_REDUCTIONS} 次），"
            f"grad_accum_steps={getattr(self, 'grad_accum_steps', 1)}，"
            f"num_iterations_per_epoch="
            f"{getattr(self, 'num_iterations_per_epoch', '?')}，"
            "重建 dataloader 后重试捕获。")
        # 落盘：让降档结果对后续运行持续有效（见 _persist_batch_to_probe_cache）
        self._persist_batch_to_probe_cache()
        return True

    def _persist_batch_to_probe_cache(self) -> None:
        """把降档后的实际 batch 写回 probe 缓存，使其对后续运行持续生效。

        为什么必须写回
        --------------
        probe 缓存的指纹只含 batch_nominal（plans 规定值，降档后不变），**不含**
        actual_batch_size。若只改内存不落盘：进程重启后缓存照旧判定 HIT，读回
        降档前的过大 batch → 训练中途又得重走一遍降档（每次降档需重建 dataloader
        + 重捕获，约 3-5 分钟）。

        只对实现了 probe 缓存的宿主（nnUNetTrainerBatchProbe 系）生效；
        nnUNetTrainerLightMamba2NetCUDAGraph 这类纯 graph 组合没有缓存，直接跳过。

        字段须与 _warmup_kernels 的 cache HIT 读取严格对齐；_save_probe_cache
        自带异常捕获与日志，故此处不再重复 try/except。
        """
        save_fn = getattr(self, '_save_probe_cache', None)
        if save_fn is None:
            return
        save_fn({
            'actual_batch_size': self.batch_size,
            'grad_accum_steps': getattr(self, 'grad_accum_steps', 1),
            'num_iterations_per_epoch': getattr(
                self, 'num_iterations_per_epoch', None),
            # 标注来源，便于事后区分「探测所得」与「显存不足降档所得」
            'probe_source': 'cudagraph_batch_reduction',
        })

    def _do_i_compile(self):
        """CUDA Graphs 与 torch.compile 互斥：两者都消除 kernel launch 开销，
        叠加会冲突（compile 把网络包装成 OptimizedModule，破坏 graph capture 的
        纯净性，且部分 Mamba/SSM 网络在 dynamo 下崩溃）。CUDAGraph 训练器统一禁用。
        """
        return False

    def _use_pin_memory(self) -> bool:
        """CUDAGraph 训练器关闭 pinned memory。

        batchgenerators 的 pin_memory 在独立线程里调用 cudaHostRegister，
        CUDA Graph 捕获期间该操作属于被禁止的 stream 操作，会报
        "operation not permitted when stream is capturing" 并杀死 dataloader
        worker 线程（Blackwell / torch 2.10 实测）。关闭后 worker 只做 CPU
        增广与 H2D 拷贝，不再与捕获冲突。
        """
        return False

    def get_val_batch_size(self):
        """验证 batch 用独立小值（val_batch_size），默认 8。

        验证 forward 激活 ≈ 训练 forward 激活 × (val_batch / train_batch)，
        batch=8 时缩小 8 倍，装进 graph 锁定后剩余显存，避免叠加溢出。
        返回 None 配置时回退到训练 batch（配合 release_graph_for_validation）。
        重型网络（heavy_model=True）时再减半（8 → 4），进一步压低验证峰值。
        """
        vb = getattr(self, 'val_batch_size', None)
        if vb is None:
            return self.batch_size
        heavy = getattr(self, 'heavy_model', None)
        if heavy is None:
            # 未显式标记 → 用集中登记表按类名自动检测（减少每文件重复标记）
            heavy = type(self).__name__ in type(self).HEAVY_MODEL_TRAINERS
        if heavy:
            vb = max(1, vb // 2)
            self.print_to_log_file(
                f"[CUDAGraph] heavy_model=True — val_batch_size halved to {vb}")
        return vb

    def get_dataloaders(self):
        """验证 batch 缩小时按比例放大验证迭代数，保持每 epoch 验证图数恒定
        （nnUNet 默认 50 步 × 训练 batch），保证 fake dice 的统计口径跨训练器可比。
        """
        vb = self.get_val_batch_size()
        if vb != self.batch_size:
            # 幂等：必须基于「原始基线」换算，而不是上一次的结果。降 batch 重捕获
            # 会再次调用本方法重建 dataloader，按现值放大就会二次放大、验证迭代数
            # 逐次膨胀。故首次调用时把基类原始值记为基线。
            base = getattr(self, '_val_iterations_base', None)
            if base is None:
                base = self.num_val_iterations_per_epoch
                self._val_iterations_base = base
            self.num_val_iterations_per_epoch = max(
                1, -(-base * self.batch_size // vb))
            self.print_to_log_file(
                f"[CUDAGraph] val_batch_size={vb} (train batch={self.batch_size}) — "
                f"num_val_iterations_per_epoch={self.num_val_iterations_per_epoch}")
        return super().get_dataloaders()

    def _cuda_free_ratio(self):
        """当前设备空闲显存占比（0-1）；查询失败返回 None。

        用于捕获后判定 graph 私有池是否被 WDDM 驱动外置到共享内存。
        """
        try:
            free_b, total_b = torch.cuda.mem_get_info(self.device)
            return (free_b / total_b) if total_b else None
        except Exception:
            return None

    def _copy_into_static(self, data: torch.Tensor, target):
        """Copy real batch into static buffers (non_blocking, no sync)."""
        self.static_input.copy_(data)
        if isinstance(self.static_target, list):
            for buf, t in zip(self.static_target, target):
                buf.copy_(t)
        else:
            self.static_target.copy_(target)

    def _graph_forward_backward(self):
        """forward + loss + backward. Must run inside autocast context.

        梯度累积（accum>1）时 loss 除以 accum——graph 捕获的静态 forward/
        backward 是"每次 replay 重算"，loss 缩放需在捕获时固定到计算图中，
        与 eager 累积版（BatchProbe.train_step 中 l/accum）尺度一致。
        """
        accum = getattr(self, 'grad_accum_steps', 1)
        with autocast(self.device.type, dtype=self.autocast_dtype,
                      enabled=True, cache_enabled=False):
            output = self.network(self.static_input)
            self.static_loss = self.loss(output, self.static_target)
            if accum > 1:
                self.static_loss = self.static_loss / accum
        if self.grad_scaler is not None:
            self.grad_scaler.scale(self.static_loss).backward()
        else:
            self.static_loss.backward()

    def _graph_optimizer_step(self):
        """Optimizer step OUTSIDE the graph (scaler logic stays dynamic)."""
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

    def _capture_cuda_graph(self, data: torch.Tensor, target):
        """Allocate static buffers from real batch shapes, warm up on a side
        stream, then capture forward+loss+backward into a CUDA Graph.

        Warmup does forward+backward WITHOUT optimizer.step() so model weights
        are NOT touched by warmup noise (random/real data both fine). Capture
        runs on the real first batch; the single optimizer.step() after capture
        consumes the real gradient - equivalent to one normal training step.
        """
        if self.device.type != 'cuda' or self.is_ddp:
            return False  # graph disabled on CPU / DDP

        self.print_to_log_file(
            "CUDA Graphs: capturing training step "
            f"(batch={data.shape[0]}, patch={tuple(data.shape[2:])}, "
            f"AMP={self.autocast_dtype}) ...")

        # 0. 捕获前回收 allocator 缓存/碎片并同步，尽量给 graph 私有池留连续显存
        #    （探针可能留下大块缓存；见 BatchProbe._reclaim_vram_after_probe）。
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        # 1. Static buffers with shapes taken from the real batch (exact match
        #    with the dataloader, including deep-supervision target list).
        #    Fill with the REAL first batch so the captured forward/backward
        #    computes meaningful gradients (no random-data weight pollution).
        self.static_input = torch.empty_like(data)
        if isinstance(target, list):
            self.static_target = [torch.empty_like(t) for t in target]
        else:
            self.static_target = torch.empty_like(target)
        self._copy_into_static(data, target)

        # 2. Warm up on a side stream (3 forward+backward passes, NO step):
        #    allocates the graph's private memory pool, finishes cudnn autotune
        #    and warms up backward kernels WITHOUT updating model weights.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                self._graph_forward_backward()
        torch.cuda.current_stream().wait_stream(s)

        # 释放 warmup 产生的 autograd 图：self.static_loss 是实例属性，会一直
        # 引用最后一次 warmup 的 loss，导致 AccumulateGrad 节点停留在 side
        # stream（capture 在主 stream 时触发 stream mismatch 警告，可能破坏
        # capture）。置 None + 同步后 autograd 图可被回收。
        self.static_loss = None
        # ⚠️ 必须 set_to_none=False：让 p.grad 保持为「已存在的零张量」。
        #
        # AccumulateGrad 对 leaf 的 grad 有 Python 级分支：
        #     grad is None  → 赋值（拷贝）
        #     grad 非 None  → 就地 +=
        # 该分支在 capture 时被固化进图。若 capture 前用 set_to_none=True，
        # grad 变 None → 图中记录的是「赋值」→ replay 每次都**覆盖**而非累加，
        # 「replay 复用 N 次累积梯度」的设计前提彻底失效（accum>1 时每步实际
        # 只用到最后 1 个 micro-batch，有效 batch 退化为 actual_batch）。
        # 2026-09-15 最小探针实证（leaf 单次 backward 贡献 2.0）：
        #     capture 时 grad=None     → replay1=2.0, replay2=2.0  （覆盖）
        #     capture 时 grad=零张量   → replay1=2.0, replay2=4.0  （累加）
        # warmup 最后一轮 backward 已物化 grads，此处就地清零即可保留张量。
        self.optimizer.zero_grad(set_to_none=False)
        torch.cuda.synchronize()

        # 3. Capture: forward + loss + backward inside the graph
        #    cache_enabled=False -> casts are explicit in-graph kernels, so
        #    replay always casts from the CURRENT weights (no stale fp16 cache)
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self._graph_forward_backward()

        # 4. 私有池外置检测：WDDM 显存超订时驱动不报 OOM，而是把 graph 私有池
        #    静默溢出到共享内存，此后每次 replay 的激活都走 PCIe（实测慢 ~8x、
        #    功耗腰斩）。捕获后若整卡几乎无空闲显存，判为外置 → 弃用 graph 走
        #    eager（捕获产生的梯度丢弃，由下一次 eager train_step 重算）。
        free_ratio = self._cuda_free_ratio()
        if free_ratio is not None and free_ratio < self._GRAPH_SPILL_FREE_RATIO:
            self.print_to_log_file(
                f"CUDA Graphs: capture completed but free VRAM only "
                f"{free_ratio:.1%} (< {self._GRAPH_SPILL_FREE_RATIO:.0%}) — "
                "graph 私有池可能已被驱动外置到共享内存（replay 走 PCIe）。"
                "将降低 1 个训练 batch 后重试捕获（不再回退 eager）。")
            self.cuda_graph = None
            self.static_input = None
            self.static_target = None
            self.static_loss = None
            gc.collect()
            torch.cuda.empty_cache()
            # 丢弃捕获期间产生的梯度，避免回退 eager 时与 eager backward 叠加
            self.optimizer.zero_grad(set_to_none=True)
            return False

        # 5. Consume the gradient produced inside the capture (outside graph),
        #    then zero (set_to_none=False 保持图锁定的 grad 地址稳定），让随后
        #    的 replay 周期从干净的累加器开始；否则 capture 残留会被计入第一
        #    个累积周期（多算一步）。
        self._graph_optimizer_step()
        self.optimizer.zero_grad(set_to_none=False)

        self.print_to_log_file("CUDA Graphs: capture complete, replay mode on.")
        return True

    # ------------------------------------------------------------------ #
    #  Validation: release graph to free VRAM, re-capture on next train   #
    # ------------------------------------------------------------------ #
    def on_validation_epoch_start(self):
        """验证前释放训练 CUDA Graph 以腾出显存。

        CUDA Graph 的私有内存池锁定了大 batch 的 forward+backward 全部中间
        激活（graph 不销毁不释放）。验证阶段是 eager forward，若不清空 graph，
        验证激活会与锁定激活叠加，在 16GB 卡上溢出到共享显存导致断崖降速。
        验证只需权重 + forward 激活，释放 graph 后普通池有足够空间。
        下一个 train_step 因 _capture_attempted=False 会自动重新捕获。
        """
        if getattr(self, 'release_graph_for_validation', True):
            self._release_cuda_graph()
        super().on_validation_epoch_start()

    def _release_cuda_graph(self):
        """销毁训练 graph 及其静态 buffer，释放私有池 + 普通池缓存块。

        顺序：先同步（确保最后一次 replay 完成，避免 pending kernel 引用
        graph），再置 None 触发 CUDAGraph.__del__ 释放私有池，gc.collect 兜底
        （确保 __del__ 被执行），empty_cache 归还普通池 free 块，最后同步
        确保释放完成。
        """
        if self.cuda_graph is None:
            return
        self.print_to_log_file(
            "CUDA Graphs: releasing graph before validation "
            "(free VRAM for val forward; re-capture on next train step)")
        torch.cuda.synchronize(self.device)
        self.cuda_graph = None
        self.static_input = None
        self.static_target = None
        self.static_loss = None
        self._capture_attempted = False
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ #
    #  Validation forward CUDA Graph                                       #
    # ------------------------------------------------------------------ #
    def configure_validation_predictor(self, predictor):
        """把验证推理的 network forward 也捕获成 CUDA Graph（默认开）。

        perform_actual_validation 每张图要做 9 tile × 4 mirror TTA = 36 次前向，
        且该前向是 CPU launch-bound（实测耗时与 batch 无关），因此图的收益比
        训练侧更大。捕获与三重自检都委托给 predictor 侧的 CudaGraphForward
        （捕获失败 / 数值不符 / 无速度收益 → 自动永久回退 eager）。

        回退开关：use_cuda_graph_for_validation=False 或环境变量
        NNUNET_VAL_NO_CUDAGRAPH=1。DDP / CPU 下不启用（与训练侧一致）。
        """
        super().configure_validation_predictor(predictor)
        if not (getattr(self, 'use_cuda_graphs', True)
                and getattr(self, 'use_cuda_graph_for_validation', True)):
            return
        if os.environ.get('NNUNET_VAL_NO_CUDAGRAPH', '').lower() in (
                '1', 'true', 't', 'yes', 'on'):
            self.print_to_log_file(
                "[CUDAGraph] NNUNET_VAL_NO_CUDAGRAPH set — 验证 forward 保持 eager")
            return
        if self.is_ddp or self.device.type != 'cuda':
            return
        armed = predictor.enable_cuda_graph_forward(
            autocast_dtype=self.autocast_dtype,
            spill_free_ratio=self._GRAPH_SPILL_FREE_RATIO,
            logger=self.print_to_log_file)
        self.print_to_log_file(
            f"[CUDAGraph] 验证 forward CUDA Graph: "
            f"{'已武装（首次前向惰性捕获）' if armed else '不可用，走 eager'}")
