"""
SwinUMamba 训练器 — Swin Transformer + Mamba (VMamba) 混合架构

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标

v2 迁移说明:
  - 使用标准 v2 签名 build_network_architecture
  - 工厂函数 get_swin_umamba_from_plans 仅需 num_output_channels/num_input_channels
  - 由于 use_pretrain=False，移除原版 freeze_encoder 逻辑（随机初始化无意义）
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
from nnunetv2.nets.SwinUMamba import get_swin_umamba_from_plans


class nnUNetTrainerSwinUMamba(nnUNetTrainer_MedNeXtBase):
    """
    SwinUMamba: Swin Transformer encoder + VMamba (VSSM) encoder + UNETR-style decoder
    支持 2D 分割任务，内部使用固定 feat_size=[48, 96, 192, 384, 768]。
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # 必须关键字传 device：MRO 下一跳 nnUNetTrainer_MedNeXtBase.__init__ 的签名是
        # (plans, configuration, fold, dataset_json, unpack_dataset=True, device=...)，
        # 位置传参会把 device 错位塞进 unpack_dataset，device 落回默认（无 index）→
        # DDP 下兜底成 cuda:local_rank 而绑错卡（同 2026-08-11 事故）。
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # nnUZoo 原版超参（对应 nnUZoo nnUNetTrainerSwinUMamba.__init__）
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2

    def configure_optimizers(self):
        """按 nnUZoo 原版恢复 AdamW + CosineAnnealingLR（勿再当冗余方法删除）。

        基类 nnUNetTrainer 提供的是 nnUNet 官方
        SGD(lr=1e-2, momentum=0.99, nesterov=True, wd=3e-5) + PolyLRScheduler，
        与原版 AdamW(lr=1e-4, wd=5e-2) + CosineAnnealingLR 训练协议不等价
        （学习率相差 100 倍），误删会使训练发散至 NaN（2026-09-15 SSND2Net 实证）。
        """
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

    def _do_i_compile(self) -> bool:
        """
        SwinUMamba 网络规模大，torch.compile 编译阶段耗时且占用显存，
        收益不明显，直接禁用编译。
        """
        return False

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

        model = get_swin_umamba_from_plans(
            num_segmentation_heads=num_output_channels,
            num_input_channels=num_input_channels,
            deep_supervision=enable_deep_supervision,
            use_pretrain=False,
        )

        print(f"SwinUMamba built. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model
