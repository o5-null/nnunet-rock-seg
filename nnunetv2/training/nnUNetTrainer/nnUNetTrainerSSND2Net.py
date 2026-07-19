"""
SSND2Net (SS2D2Net) 训练器 — X2Net 系列中性能最优的 Mamba-SS2D 混合架构

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标 (Precision/Recall/Std Dice)

v2 迁移说明:
  - 使用 build_network_architecture 标准 v2 签名
    (plans_manager, configuration_manager, num_input_channels, num_output_channels, ...)
  - 直接使用 num_output_channels 取代 dataset_json → get_label_manager 调用链
  - 移除 nnUZoo 原版中基类已覆盖的 configure_optimizers / on_epoch_end 等冗余方法
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.nets.ssnd2net import SSND2Net
import torch
from torch import nn


class nnUNetTrainerSSND2Net(nnUNetTrainer_MedNeXtBase):
    """
    SSND2Net: Selective Scan 2D Network
    U2Net-style nested encoder-decoder with SS2D (Selective Scan 2D) Mamba blocks.
    支持 2D/3D 通过 spatial_dims 自动适配。
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _get_deep_supervision_scales(self):
        """
        参考 nnUZoo nnUNetTrainerSSND2Net._get_deep_supervision_scales。
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
        """
        构建 SSND2Net 模型。

        v2 标准签名 — 不依赖 dataset_json，直接使用 plans_manager 和 configuration_manager
        解析模型配置，num_output_channels 由调用方传入（num_segmentation_heads）。
        """
        model = SSND2Net(
            spatial_dims=len(configuration_manager.patch_size),
            factorization_type="cross-scan",
            in_ch=num_input_channels,
            out_ch=num_output_channels,
            deep_supervision=enable_deep_supervision,
            input_patch_size=configuration_manager.patch_size
        )
        model.apply(InitWeights_He(1e-2))
        model.apply(init_last_bn_before_add_to_0)

        print(f"SSND2Net: {model}")
        return model
