"""
nnUNetTrainerLightMamba2NetCUDAGraph
====================================
LightMamba2Net (Mamba-2 / U2Net-style) trainer with CUDA Graphs acceleration.

Purpose
-------
LightMamba2Net at batch=4 / patch=256 runs at ~40% GPU utilization on an RTX
3080, dominated by CPU kernel-launch overhead (~178 MambaLayer calls per step).
CUDA Graphs capture the step and replay it with a single launch, eliminating
that overhead (~1.3-1.65x expected at batch=4).

This trainer is the EXPERIMENT group; the eager baseline remains
nnUNetTrainerLightMamba2Net, so both can be compared on identical configs.

Key facts inherited from the base trainer:
  - bf16 autocast + grad_scaler=None (NaN fix, 2026-07-31)
  - torch.compile disabled (triton JIT crashes on Windows)
  - No dropout in LightMamba2Net (dropout_prob=None) -> safe for graph capture

Usage
-----
python -m nnunetv2.run.run_training 2 2d 0 -tr nnUNetTrainerLightMamba2NetCUDAGraph
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMamba2Net import \
    nnUNetTrainerLightMamba2Net
from nnunetv2.training.nnUNetTrainer.variants.cuda_graph.nnUNetTrainerCUDAGraphMixin import \
    nnUNetTrainerCUDAGraphMixin


class nnUNetTrainerLightMamba2NetCUDAGraph(nnUNetTrainerCUDAGraphMixin,
                                           nnUNetTrainerLightMamba2Net):
    """LightMamba2Net + CUDA Graphs: capture forward+loss+backward into a graph
    and replay per step, removing CPU kernel-launch overhead at fixed batch."""
    pass
