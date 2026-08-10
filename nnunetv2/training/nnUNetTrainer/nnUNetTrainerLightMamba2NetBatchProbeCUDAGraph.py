# -*- coding: utf-8 -*-
"""LightMamba2Net + 自动 batch 探测 + CUDA Graph 训练器（组合变体）

组合 BatchProbeCUDAGraph（探测 + graph 累积）+ LightMamba2Net（Mamba-2
网络 + NaN 检测）。

加速路径（探测后按 accum 自动选择）:
- 所有 accum → CUDA Graph replay 累积（mixin 增强版）:
    accum==1: 每步 replay + step
    accum>1 : replay 复用 N 次（非边界步只 replay 不 step，梯度累积到
      同一地址），kernel launch 开销除以 N
- graph 捕获失败（OOM/不支持/DDP）→ 回退 eager 累积（BatchProbe.train_step）
- compile: graph 路径互斥禁用；且 LightMamba2Net._do_i_compile 恒 False
  （Mamba2 SSM 在 triton JIT 下崩溃），故 compile 永不启用

MRO: [T, BatchProbeCUDAGraph, CUDAGraphMixin, BatchProbe, LightMamba2Net,
      MedNeXtBase, nnUNetTrainer]
- BatchProbeCUDAGraph._warmup_kernels: 探测 + 决策（优先）
- CUDAGraphMixin.train_step: graph replay 累积路径
- BatchProbe.train_step: eager 累积（graph 回退时）
- LightMamba2Net.on_epoch_end: NaN 检测回退（保留）
- LightMamba2Net.build_network_architecture: Mamba-2 网络构建

用法:
  # 纯探测（写缓存后退出）
  python -m nnunetv2.run.run_training 2 2d 0 ^
      -tr nnUNetTrainerLightMamba2NetBatchProbeCUDAGraph -p nnUNetPlans_bs16 --probe only
  # 完整训练（缓存命中秒用 / 未命中先探测）
  python -m nnunetv2.run.run_training 2 2d 0 ^
      -tr nnUNetTrainerLightMamba2NetBatchProbeCUDAGraph -p nnUNetPlans_bs16
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBatchProbeCUDAGraph import nnUNetTrainerBatchProbeCUDAGraph
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMamba2Net import nnUNetTrainerLightMamba2Net


class nnUNetTrainerLightMamba2NetBatchProbeCUDAGraph(
        nnUNetTrainerBatchProbeCUDAGraph,
        nnUNetTrainerLightMamba2Net):
    """LightMamba2Net + 自动 batch 探测 + CUDA Graph 累积加速。

    MRO 验证通过（菱形继承合法线性化）:
    BatchProbeCUDAGraph → CUDAGraphMixin → BatchProbe → LightMamba2Net
    → MedNeXtBase → nnUNetTrainer
    """
    pass
