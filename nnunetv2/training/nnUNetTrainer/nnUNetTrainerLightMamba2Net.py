"""
LightMamba2Net 训练器 — 轻量级 Mamba-2 架构

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标

v2 迁移说明:
  - 使用标准 v2 签名 build_network_architecture
  - 内联工厂逻辑，直接构造 LightMamba2Net
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.nets.light_mamba2net import LightMamba2Net
import torch
from torch import nn


class nnUNetTrainerLightMamba2Net(nnUNetTrainer_MedNeXtBase):
    """
    LightMamba2Net: Lightweight Mamba-2 based U2Net-style architecture
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # 保持 AMP 为 fp16（基类默认）。Mamba2 SSM 的 A_log/dt 指数运算在 fp16 下
        # 动态范围不足容易 NaN，MambaLayer 内部单独用 bf16 包裹 SSM 调用（仅降此一级）。
        # conv 层和 loss 计算走 fp16（速度优势），互不影响。

    def _get_deep_supervision_scales(self):
        """
        LightMamba2Net 的侧输出 (d0-d6) 处于固定分辨率: d0(1×), d1(1×), d2(1/2), d3(1/4), d4(1/8), d5(1/16), d6(1/32)。
        使用 spatial_dims 确保 2D/3D 通用。参考 nnUZoo nnUNetTrainerSSND2Net._get_deep_supervision_scales。
        """
        if self.enable_deep_supervision:
            ndim = len(self.configuration_manager.patch_size)
            return [[1.0] * ndim, [1.0] * ndim,
                    [0.5] * ndim, [0.25] * ndim,
                    [0.125] * ndim, [0.0625] * ndim,
                    [0.03125] * ndim]
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

        model = LightMamba2Net(
            spatial_dims=len(configuration_manager.patch_size),
            input_patch_size=configuration_manager.patch_size,
            in_ch=num_input_channels,
            out_ch=num_output_channels,
            deep_supervision=enable_deep_supervision,
        )
        model.apply(InitWeights_He(1e-2))
        model.apply(init_last_bn_before_add_to_0)

        print(f"LightMamba2Net: {model}")
        return model
