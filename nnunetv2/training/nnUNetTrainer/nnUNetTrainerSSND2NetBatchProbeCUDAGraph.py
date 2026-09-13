# -*- coding: utf-8 -*-
"""nnUNetTrainerSSND2NetBatchProbeCUDAGraph — nnUNetTrainerSSND2Net + 自动 batch 探测 + CUDA Graph 累积加速

组合 nnUNetTrainerBatchProbeCUDAGraph（探测 + graph 累积 replay）与 nnUNetTrainerSSND2Net。

加速路径（探测后按 accum 自动选择）:
- 所有 accum → CUDA Graph replay 累积（mixin 增强版）:
    accum==1: 每步 replay + step
    accum>1 : replay 复用 N 次（非边界步只 replay 不 step，梯度累积到
      同一地址），kernel launch 开销除以 N
- graph 捕获失败（OOM/不支持/DDP）→ 回退 eager 累积（BatchProbe.train_step）
- compile: graph 路径互斥禁用；强制 eager 时复用基类 _do_i_compile 决策

MRO: [T, BatchProbeCUDAGraph, CUDAGraphMixin, BatchProbe, nnUNetTrainerSSND2Net, ...]
（2026-08-01 脚本验证菱形继承合法线性化）

用法:
  # 纯探测（写缓存后退出）
  python -m nnunetv2.run.run_training 2 2d 0 ^
      -tr nnUNetTrainerSSND2NetBatchProbeCUDAGraph -p nnUNetPlans_bs16 --probe only
  # 完整训练（缓存命中秒用 / 未命中先探测）
  python -m nnunetv2.run.run_training 2 2d 0 ^
      -tr nnUNetTrainerSSND2NetBatchProbeCUDAGraph -p nnUNetPlans_bs16
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBatchProbeCUDAGraph import nnUNetTrainerBatchProbeCUDAGraph
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSSND2Net import nnUNetTrainerSSND2Net


class nnUNetTrainerSSND2NetBatchProbeCUDAGraph(nnUNetTrainerBatchProbeCUDAGraph, nnUNetTrainerSSND2Net):
    """nnUNetTrainerSSND2Net + 自动 batch 探测 + CUDA Graph 累积加速。

    重型（batch=4 ~14G，全项目最重）——验证 batch 减半由基类自动处理。

    CUDA Graph 私有池显存预留覆盖（2026-09-13）:
    基类为 graph 私有池保守预留 20% 物理显存（vram_safe_ratio → 0.80）。但对
    SSND2Net 在 10GB RTX 3080 上**过于保守**：实测 batch=2 稳态物理占用
    9015/10240MB = 88%，被 0.80 阀拒 → 探测塌到 actual_batch=1、accum=16、
    功耗仅 74W/320W（GPU 被饿着跑）。而 2026-08-23/24 在安全阀 0.92 下
    batch=2 已稳定训练（97% util / 230W / 754–896s/epoch），捕获后仍有 >12%
    空闲显存，从未溢出。
    故此处把预留收到 0.10（vram_safe_ratio → 0.90）：既恢复 batch=2（88% ≤
    90%，留 2% 余量），又不动基类默认值以免影响其他训练器的安全性。真正的
    溢出兜底仍由 mixin 的 _GRAPH_SPILL_FREE_RATIO 捕获后检测负责。
    """
    GRAPH_POOL_VRAM_RESERVE_RATIO = 0.10

