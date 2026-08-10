"""
nnUNetTrainerLightSS2DMambaUNetCUDAGraph
========================================
LightSS2DMambaUNet (SS2D Mamba) trainer with CUDA Graphs acceleration.

Speed baseline (Dataset002, batch=4, patch=256): ~59 s/epoch eager.
SS2D cross-scan does many small kernels (.float() copies + transpose/flip)
- CUDA Graphs removes the launch overhead.

Network built WITHOUT dropout (default dropout=0., not passed) -> graph-safe.

Usage
-----
python -m nnunetv2.run.run_training 2 2d 0 -tr nnUNetTrainerLightSS2DMambaUNetCUDAGraph
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightSS2DMambaUNet import \
    nnUNetTrainerLightSS2DMambaUNet
from nnunetv2.training.nnUNetTrainer.variants.cuda_graph.nnUNetTrainerCUDAGraphMixin import \
    nnUNetTrainerCUDAGraphMixin


class nnUNetTrainerLightSS2DMambaUNetCUDAGraph(nnUNetTrainerCUDAGraphMixin,
                                               nnUNetTrainerLightSS2DMambaUNet):
    """LightSS2DMambaUNet + CUDA Graphs."""
    pass
