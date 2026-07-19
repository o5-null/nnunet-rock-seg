"""
SwinTransformerUnet 训练器 — 纯 Swin Transformer UNet (Transformer 基线)

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标
"""
from functools import partial
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.swt import SwinTransformerUnet
import torch
from torch import nn
from torch.nn import LayerNorm


class nnUNetTrainerSwinTransformerUnet(nnUNetTrainer_MedNeXtBase):
    """
    SwinTransformerUnet: Pure Swin Transformer UNet (Transformer baseline)
    无 Mamba/SSM 依赖。
    """
    # SwinTransformerUnet (swt.py) 没有 deep supervision 侧输出，禁用 DS
    _ds_enabled = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        模型没有 .decoder 属性，deep supervision 已通过模型内部的 self.deep_supervision 管理。
        只需同步该标志即可。
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        mod.deep_supervision = enabled

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = SwinTransformerUnet(
            patch_size=4,
            in_ch=num_input_channels,
            out_ch=num_output_channels,
            depths=(2, 2, 9, 2),
            embed_dim=96,
            num_heads=(3, 6, 12, 24),
            window_size=7,
            qkv_bias=True,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0,
            attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        print(f"SwinTransformerUnet: {model}")
        return model
