"""
SwinUMamba 训练器 — Swin Transformer + Mamba (VMamba) 混合架构

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标

v2 迁移说明:
  - 使用标准 v2 签名 build_network_architecture
  - 工厂函数 get_swin_umamba_from_plans 仅需 num_output_channels/num_input_channels
  - 由于 use_pretrain=False，移除原版 freeze_encoder 逻辑（随机初始化无意义）
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

import torch
from nnunetv2.nets.SwinUMamba import get_swin_umamba_from_plans


class nnUNetTrainerSwinUMamba(nnUNetTrainer_MedNeXtBase):
    """
    SwinUMamba: Swin Transformer encoder + VMamba (VSSM) encoder + UNETR-style decoder
    支持 2D 分割任务，内部使用固定 feat_size=[48, 96, 192, 384, 768]。
    """
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

        model = get_swin_umamba_from_plans(
            num_segmentation_heads=num_output_channels,
            num_input_channels=num_input_channels,
            deep_supervision=enable_deep_supervision,
            use_pretrain=False,
        )

        print(f"SwinUMamba: {model}")
        return model
