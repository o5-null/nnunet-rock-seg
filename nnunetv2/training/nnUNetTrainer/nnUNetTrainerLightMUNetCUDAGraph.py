"""
nnUNetTrainerLightMUNetCUDAGraph
================================
LightMUNet (Mamba) trainer with CUDA Graphs acceleration.

Speed baseline (Dataset002, batch=4, patch=256): ~114 s/epoch eager.
Mamba1 sequential scan + many layers -> heavy kernel-launch overhead.

Network built without dropout (dropout_prob not passed, default None)
-> graph-safe.

Usage
-----
python -m nnunetv2.run.run_training 2 2d 0 -tr nnUNetTrainerLightMUNetCUDAGraph
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMUNet import \
    nnUNetTrainerLightMUNet
from nnunetv2.training.nnUNetTrainer.variants.cuda_graph.nnUNetTrainerCUDAGraphMixin import \
    nnUNetTrainerCUDAGraphMixin


class nnUNetTrainerLightMUNetCUDAGraph(nnUNetTrainerCUDAGraphMixin,
                                       nnUNetTrainerLightMUNet):
    """LightMUNet + CUDA Graphs."""
    pass
