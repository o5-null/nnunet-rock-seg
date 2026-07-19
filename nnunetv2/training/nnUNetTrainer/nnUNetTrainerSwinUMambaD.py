"""
SwinUMambaD 训练器 — Swin-UMamba 深度变体（Mamba 编解码器）

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标

v2 迁移说明:
  - 使用标准 v2 签名 build_network_architecture
  - 内联原 get_swin_umamba_d_from_plans 工厂逻辑，直接用 num_output_channels 替代
    plans_manager.get_label_manager(dataset_json) 调用链
  - 仅支持 2D（原版断言）
  - 移除 freeze_encoder 逻辑（use_pretrain=False，随机初始化无意义）
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.nets.SwinUMambaD import SwinUMambaD
import torch
from torch import nn


class nnUNetTrainerSwinUMambaD(nnUNetTrainer_MedNeXtBase):
    """
    SwinUMambaD: Swin-UMamba Deep — VSSM (VMamba) encoder + Mamba-based decoder
    仅支持 2D 分割任务。
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _get_deep_supervision_scales(self):
        """
        SwinUMambaD 的 UNetResDecoder 产生 4 个侧输出（seg_outputs[::-1] 后）:
          seg[0]: 1× (全分辨率), seg[1]: 1/4, seg[2]: 1/8, seg[3]: 1/16
        与 PlainConvUNet 的 7 层 DS 不同，必须重写 scale 以匹配实际空间分辨率。
        """
        if self.enable_deep_supervision:
            ndim = len(self.configuration_manager.patch_size)
            return [[1.0] * ndim,
                    [0.25] * ndim,
                    [0.125] * ndim,
                    [0.0625] * ndim]
        else:
            return None

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        dim = len(configuration_manager.patch_size)
        assert dim == 2, "SwinUMambaD only supports 2D"

        vss_args = dict(
            in_chans=num_input_channels,
            patch_size=4,
            dims=96,
            drop_path_rate=0.2,
        )

        decoder_args = dict(
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            drop_path_rate=0.2,
            d_state=16,
        )

        model = SwinUMambaD(vss_args, decoder_args)
        model.apply(InitWeights_He(1e-2))
        model.apply(init_last_bn_before_add_to_0)

        print(f"SwinUMambaD: {model}")
        return model
