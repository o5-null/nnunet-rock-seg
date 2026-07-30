"""
LightSS2DMambaUNet 训练器 — 轻量级 SS2D Mamba UNet

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.nets.LightSS2DMambaUNet import LightSS2DMambaUNet
import torch
from torch import nn


class nnUNetTrainerLightSS2DMambaUNet(nnUNetTrainer_MedNeXtBase):
    """
    LightSS2DMambaUNet: Lightweight SS2D Mamba UNet
    """
    # LightSS2DMambaUNet 没有 deep supervision 侧输出，禁用 DS
    _ds_enabled = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _do_i_compile(self) -> bool:
        """
        SS2D-Mamba 网络大量小 forward 嵌套，Dynamo 因 AMP
        (fp16/fp32) 和 train/val (grad_mode) 交替触发反复重编译。
        Mamba/SS2D CUDA kernel 已是手写优化，compile 收益极低，
        直接关闭以避免重编译开销和日志噪音。
        """
        return False

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        浅层设置，实际网络 forward() 不检查 deep_supervision 标志。
        仅为了兼容基类调用而保留。
        """
        pass

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = LightSS2DMambaUNet(
            spatial_dims=len(configuration_manager.patch_size),
            in_channels=num_input_channels,
            out_channels=num_output_channels,
        )
        model.apply(InitWeights_He(1e-2))
        model.apply(init_last_bn_before_add_to_0)

        print(f"LightSS2DMambaUNet: {model}")
        return model
