"""
U-Mamba Enc Trainer — 继承 nnUNetTrainer_MedNeXtBase 基类
UMamba Encoder + Residual Decoder + Skip Connections
"""
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.nets.UMambaEnc import get_umamba_enc_from_plans


class nnUNetTrainerUMambaEnc(nnUNetTrainer_MedNeXtBase):
    """
    UMamba Encoder + Residual Decoder + Skip Connections
    继承 MedNeXtBase 以启用 TF32 加速和增强指标
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # 项目统一训练协议（对齐其余 nnUZoo 移植模型的 AdamW + Cosine）。
        # 事实说明：U-Mamba 官方 nnUNetTrainerUMambaEnc 并未覆盖 configure_optimizers，
        # 走 nnU-Net 默认 SGD(lr=1e-2, momentum=0.99, nesterov) + PolyLR；且 nnUZoo
        # 参考实现中不存在 UMambaBot/UMambaEnc。故此处**非"复原原版"**，而是为满足
        # Dataset002 跨模型横向可比性（experiments_analysis/REPORT.md P2）统一协议。
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2

    def configure_optimizers(self):
        """AdamW(lr=1e-4, wd=5e-2, eps=1e-5) + CosineAnnealingLR，对齐项目其余 Mamba 模型。"""
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")
        return optimizer, scheduler

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_umamba_enc_from_plans(plans_manager, configuration_manager,
                                          num_input_channels, num_output_channels,
                                          deep_supervision=enable_deep_supervision)

        print(f"UMambaEnc built. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        return model
