"""
U2Net 训练器 — 两级嵌套 U-Net (CNN 基线)

参考 nnUZoo 官方实现:
  https://github.com/MIC-DKFZ/nnUZoo

论文: From Claims to Evidence (arXiv:2503.01306)

U2Net forward 将所有侧输出上采样到全分辨率，因此所有 7 个输出 (d0-d6)
空间尺寸相同 → DS target 全部设为 1.0 倍，不做下采样。
"""
from typing import Union, List, Tuple
import torch
from torch import nn

from nnunetv2.nets.u2net import U2NET
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0


class nnUNetTrainerU2Net(nnUNetTrainer):
    """
    U2Net: 两级嵌套 U-Net (纯 CNN, 无 Mamba/Transformer)
    所有 DS 输出同分辨率 → _get_deep_supervision_scales 返回全 1.0
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # nnUZoo 中显式开启 deep supervision（确保覆盖任何类变量覆盖）
        self.enable_deep_supervision = True

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            # U2Net 的 7 个输出 (d0-d6) 全部上采样到全分辨率
            # → DS target 保持原图尺寸, 不需要下采样
            return [[1.0, 1.0]] * 7
        else:
            return None

    def set_deep_supervision_enabled(self, enabled: bool):
        """U2Net 没有 .decoder 属性, deep supervision 通过 self.deep_supervision 管理"""
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        model = U2NET(
            in_ch=num_input_channels,
            out_ch=num_output_channels,
            deep_supervision=enable_deep_supervision,
        )
        model.apply(InitWeights_He(1e-2))
        model.apply(init_last_bn_before_add_to_0)
        print(f"U2Net built. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model
