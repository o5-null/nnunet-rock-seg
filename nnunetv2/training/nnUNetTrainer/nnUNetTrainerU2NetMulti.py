"""
U2NetMulti 训练器 — 多尺度 U2Net (CNN 基线)

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.nets.u2net_multi import U2NET
import torch
from torch import nn


class nnUNetTrainerU2NetMulti(nnUNetTrainer_MedNeXtBase):
    """
    U2Net Multi: Multi-scale U2Net variant (CNN baseline)
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            ndim = len(self.configuration_manager.patch_size)
            # U2Net forward 将所有侧输出上采样到全分辨率 (d1 的大小)，
            # 所有 7 个输出 (d0-d6) 空间尺寸相同，因此 DS target 也必须全是 1.0 倍
            return [[1.0] * ndim] * 7
        else:
            return None

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

        model = U2NET(
            spatial_dims=len(configuration_manager.patch_size),
            in_ch=num_input_channels,
            out_ch=num_output_channels,
            deep_supervision=enable_deep_supervision,
        )
        model.apply(InitWeights_He(1e-2))
        model.apply(init_last_bn_before_add_to_0)

        print(f"U2NetMulti: {model}")
        return model
