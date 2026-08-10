"""
nnUNetTrainerSwinUNETRCUDAGraph
===============================
SwinUNETR (Transformer) trainer with CUDA Graphs acceleration.

Speed baseline (Dataset002, batch=4, patch=256): ~33.5 s/epoch eager.
CUDA Graphs removes CPU kernel-launch overhead (attention has many small
kernels) - expected ~2-3x at this batch size.

SwinUNETR uses drop_rate=0.0 / attn_drop_rate=0.0 -> safe for graph capture.

Usage
-----
python -m nnunetv2.run.run_training 2 2d 0 -tr nnUNetTrainerSwinUNETRCUDAGraph
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUNETR import \
    nnUNetTrainerSwinUNETR
from nnunetv2.training.nnUNetTrainer.variants.cuda_graph.nnUNetTrainerCUDAGraphMixin import \
    nnUNetTrainerCUDAGraphMixin


class nnUNetTrainerSwinUNETRCUDAGraph(nnUNetTrainerCUDAGraphMixin,
                                      nnUNetTrainerSwinUNETR):
    """SwinUNETR + CUDA Graphs (fp16 + GradScaler: backward captured, step outside)."""
    pass
