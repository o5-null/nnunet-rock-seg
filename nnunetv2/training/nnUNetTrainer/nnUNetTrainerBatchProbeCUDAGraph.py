# -*- coding: utf-8 -*-
"""自动 batch 探测 + CUDA Graphs 加速训练器（组合变体）

组合 nnUNetTrainerCUDAGraphMixin（kernel launch 加速，含梯度累积 replay）
与 nnUNetTrainerBatchProbe（自动 batch 探测 + 梯度累积拉齐）:
- 探测阶段: _warmup_kernels 前自动探测最大安全 batch（batch=1/2 差分外推
  + 二分逼近验证），修改 self.batch_size 供 dataloader 消费
- 训练阶段: CUDA Graph 支持梯度累积（mixin 增强）——
    accum==1: 每步 replay + step（原始行为）
    accum>1 : replay 复用 N 次（非边界步只 replay 不 step，梯度自然累积
      到同一地址；边界步 step）。kernel launch 开销除以 N。
  graph 捕获失败（OOM/不支持/DDP）→ mixin 自动回退 eager（BatchProbe
  的累积 train_step）。
- compile: 与 graph 互斥。graph 路径禁用；`use_cuda_graphs=False`（强制
  eager）时复用基类 _do_i_compile 决定（env 优先 + Windows triton 检测）。
- 探测 + 决策顺序:
    1. _warmup_kernels → 探测 → 得 actual_batch / grad_accum_steps
    2. 探测完成后: graph 路径保持启用（支持累积 replay），compile 仅当
       强制 eager 时考虑
    3. on_train_start 后第一个 train_step: 惰性捕获 graph，之后 replay

用法:
  # 纯探测（探测完写缓存退出）
  .venv\\Scripts\\python.exe -m nnunetv2.run.run_training 2 2d 0 ^
      -tr nnUNetTrainerBatchProbeCUDAGraph -p nnUNetPlans_bs16 --disable_checkpointing --probe only

  # 完整训练（探测 + graph/累积加速）
  .venv\\Scripts\\python.exe -m nnunetv2.run.run_training 2 2d 0 ^
      -tr nnUNetTrainerBatchProbeCUDAGraph -p nnUNetPlans_bs16
"""
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBatchProbe import nnUNetTrainerBatchProbe
from nnunetv2.training.nnUNetTrainer.variants.cuda_graph.nnUNetTrainerCUDAGraphMixin import (
    nnUNetTrainerCUDAGraphMixin)


class nnUNetTrainerBatchProbeCUDAGraph(nnUNetTrainerCUDAGraphMixin, nnUNetTrainerBatchProbe):
    """自动 batch 探测 + CUDA Graph（支持梯度累积 replay）。

    accum==1 → 每步 replay+step；accum>1 → replay 累积 N 步后 step。
    探测逻辑（BatchProbe._warmup_kernels）与 graph 捕获（mixin）天然衔接:
    探测在 warmup 阶段完成并设置 batch_size / grad_accum_steps，graph 在
    第一个 train_step 用探测后的 batch shape 惰性捕获。
    """

    def _do_i_compile(self):
        """graph 路径统一禁用 compile（互斥）。注意此方法在探测前调用
        （initialize L277），此处恒返回 False 避免预 compile；若用户强制
        use_cuda_graphs=False 走 eager，_maybe_compile_after_probe 在探测
        后复用基类 _do_i_compile 决策（env 优先 + triton 检测）。
        """
        return False

    def _maybe_compile_after_probe(self):
        """探测完成后调用: 仅当 **强制 eager**（use_cuda_graphs=False，如
        DDP 或用户显式禁用）时，复用基类 _do_i_compile 决定是否 compile。

        graph 路径（默认）不 compile——graph 累积已消除 launch 开销，且
        compile 与 graph 互斥（dynamo 包装破坏 capture 纯净性）。
        """
        if self.use_cuda_graphs:
            return  # graph 路径，compile 互斥，跳过
        if self.is_ddp:
            return  # DDP 下 compile 包装 DDP 外层，语义不符
        # super(CUDAGraphMixin, self) → 从 BatchProbe 开始找 _do_i_compile
        # → nnUNetTrainer._do_i_compile（env 优先 + triton 检测）
        if super(nnUNetTrainerCUDAGraphMixin, self)._do_i_compile():
            self.print_to_log_file(
                "[CUDAGraph] CUDA Graphs off (eager) — enabling torch.compile")
            self.network = torch.compile(self.network)
        else:
            self.print_to_log_file(
                "[CUDAGraph] CUDA Graphs off (eager); torch.compile "
                "skipped (基类 _do_i_compile 返回 False)")

    def _warmup_kernels(self):
        # 探测（BatchProbe 版）: 设置 actual_batch / grad_accum_steps
        super()._warmup_kernels()
        # 探测完成后: graph 支持累积 replay，无需禁 graph；仅当强制 eager
        # 时尝试 compile 兜底
        if getattr(self, 'grad_accum_steps', 1) > 1:
            self.print_to_log_file(
                f"[CUDAGraph] grad_accum_steps={self.grad_accum_steps} > 1 — "
                f"CUDA Graph replay 累积模式（非边界步只 replay 不 step）")
        self._maybe_compile_after_probe()
