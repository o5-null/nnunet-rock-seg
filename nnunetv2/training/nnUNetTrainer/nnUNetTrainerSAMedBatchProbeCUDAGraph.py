# -*- coding: utf-8 -*-
"""nnUNetTrainerSAMedBatchProbeCUDAGraph — nnUNetTrainerSAMed + 自动 batch 探测 + CUDA Graph 累积加速

组合 nnUNetTrainerBatchProbeCUDAGraph（探测 + graph 累积 replay）与 nnUNetTrainerSAMed。

加速路径（探测后按 accum 自动选择）:
- 所有 accum → CUDA Graph replay 累积（mixin 增强版）:
    accum==1: 每步 replay + step
    accum>1 : replay 复用 N 次（非边界步只 replay 不 step，梯度累积到
      同一地址），kernel launch 开销除以 N
- graph 捕获失败（OOM/不支持/DDP）→ 回退 eager 累积（BatchProbe.train_step）
- compile: graph 路径互斥禁用；强制 eager 时复用基类 _do_i_compile 决策

MRO: [T, BatchProbeCUDAGraph, CUDAGraphMixin, BatchProbe, nnUNetTrainerSAMed, ...]
（2026-08-01 脚本验证菱形继承合法线性化）

用法:
  # 纯探测（写缓存后退出）
  python -m nnunetv2.run.run_training 2 2d 0 ^
      -tr nnUNetTrainerSAMedBatchProbeCUDAGraph -p nnUNetPlans_bs16 --probe only
  # 完整训练（缓存命中秒用 / 未命中先探测）
  python -m nnunetv2.run.run_training 2 2d 0 ^
      -tr nnUNetTrainerSAMedBatchProbeCUDAGraph -p nnUNetPlans_bs16
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBatchProbeCUDAGraph import nnUNetTrainerBatchProbeCUDAGraph
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSAMed import nnUNetTrainerSAMed


class nnUNetTrainerSAMedBatchProbeCUDAGraph(nnUNetTrainerBatchProbeCUDAGraph, nnUNetTrainerSAMed):
    """nnUNetTrainerSAMed + 自动 batch 探测 + CUDA Graph 累积加速。"""
    pass
