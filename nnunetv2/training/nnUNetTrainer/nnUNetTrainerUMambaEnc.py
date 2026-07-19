"""
U-Mamba Enc Trainer — 继承 nnUNetTrainer_MedNeXtBase 基类
UMamba Encoder + Residual Decoder + Skip Connections
"""
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

from nnunetv2.nets.UMambaEnc import get_umamba_enc_from_plans


class nnUNetTrainerUMambaEnc(nnUNetTrainer_MedNeXtBase):
    """
    UMamba Encoder + Residual Decoder + Skip Connections
    继承 MedNeXtBase 以启用 TF32 加速和增强指标
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_umamba_enc_from_plans(plans_manager, configuration_manager,
                                          num_input_channels, num_output_channels,
                                          deep_supervision=enable_deep_supervision)

        print("UMambaEnc: {}".format(model))

        return model
