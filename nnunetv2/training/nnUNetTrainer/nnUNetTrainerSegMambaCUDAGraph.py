"""
nnUNetTrainerSegMambaCUDAGraph
==============================
SegMamba (Mamba1) trainer with CUDA Graphs acceleration.

Speed baseline (Dataset002, batch=4, patch=256): ~119 s/epoch eager.
Mamba1 selective-scan + deep U-Net -> many small kernels; CUDA Graphs
removes the per-kernel CPU launch overhead.

SegMamba has no dropout layers -> graph-safe.

Usage
-----
python -m nnunetv2.run.run_training 2 2d 0 -tr nnUNetTrainerSegMambaCUDAGraph
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegMamba import \
    nnUNetTrainerSegMamba
from nnunetv2.training.nnUNetTrainer.variants.cuda_graph.nnUNetTrainerCUDAGraphMixin import \
    nnUNetTrainerCUDAGraphMixin


class nnUNetTrainerSegMambaCUDAGraph(nnUNetTrainerCUDAGraphMixin,
                                     nnUNetTrainerSegMamba):
    """SegMamba + CUDA Graphs."""
    pass
