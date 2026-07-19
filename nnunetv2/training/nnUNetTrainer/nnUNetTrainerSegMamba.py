"""
SegMamba 训练器 — SegMamba 架构

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.nets.seg_mamba.segmamba import SegMamba
import torch
from torch import nn


class nnUNetTrainerSegMamba(nnUNetTrainer_MedNeXtBase):
    """
    SegMamba: Mamba-based segmentation architecture
    支持 2D/3D。
    """
    # SegMamba 架构没有 deep supervision 侧输出，禁用 DS 以避免
    # DeepSupervisionWrapper 因 model 返回单个 tensor 而断言失败
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

        model = SegMamba(
            spatial_dims=len(configuration_manager.patch_size),
            in_ch=num_input_channels,
            out_ch=num_output_channels,
        )
        model.apply(InitWeights_He(1e-2))
        model.apply(init_last_bn_before_add_to_0)

        print(f"SegMamba: {model}")
        return model
