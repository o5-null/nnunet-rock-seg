"""
U-Mamba Bot Trainer — 继承 nnUNetTrainer_MedNeXtBase 基类
Residual Encoder + UMamba Bottleneck + Residual Decoder + Skip Connections
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBot import get_umamba_bot_from_plans
import torch
import numpy as np


class nnUNetTrainerUMambaBot(nnUNetTrainer_MedNeXtBase):
    """
    Residual Encoder + UMamba Bottleneck + Residual Decoder + Skip Connections
    继承 MedNeXtBase 以启用 TF32 加速和增强指标
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # 调整 patch_size，确保能被所有 stride 整除
        original_patch_size = self.configuration_manager.patch_size
        pool_op_kernel_sizes = self.configuration_manager.pool_op_kernel_sizes

        # 计算各维度的总下采样倍数
        dim = len(original_patch_size)
        total_stride = [1] * dim
        for stride in pool_op_kernel_sizes:
            for i in range(dim):
                total_stride[i] *= stride[i] if i < len(stride) else 1

        # 调整 patch_size 使其能被 total_stride 整除
        new_patch_size = list(original_patch_size)
        for i in range(dim):
            remainder = original_patch_size[i] % total_stride[i]
            if remainder != 0:
                new_patch_size[i] = original_patch_size[i] + (total_stride[i] - remainder)

        if new_patch_size != list(original_patch_size):
            self.configuration_manager.configuration['patch_size'] = new_patch_size
            self.plans_manager.plans['configurations'][self.configuration_name]['patch_size'] = new_patch_size
            self.print_to_log_file(f"Patch size changed from {original_patch_size} to {new_patch_size} "
                                   f"to be divisible by total stride {total_stride}")

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_umamba_bot_from_plans(plans_manager, configuration_manager,
                                          num_input_channels, num_output_channels,
                                          deep_supervision=enable_deep_supervision)

        print("UMambaBot: {}".format(model))

        return model


class nnUNetTrainerUMambaBot_100epochs(nnUNetTrainerUMambaBot):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.num_iterations_per_epoch = 250
