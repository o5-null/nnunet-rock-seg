"""
SwinTransformerUnet 训练器 — 纯 Swin Transformer UNet (Transformer 基线)

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标
"""
from functools import partial
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.swt import SwinTransformerUnet
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import LayerNorm


class nnUNetTrainerSwinTransformerUnet(nnUNetTrainer_MedNeXtBase):
    """
    SwinTransformerUnet: Pure Swin Transformer UNet (Transformer baseline)
    无 Mamba/SSM 依赖。
    """
    # SwinTransformerUnet (swt.py) 没有 deep supervision 侧输出，禁用 DS
    _ds_enabled = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # 必须关键字传 device：MRO 下一跳 nnUNetTrainer_MedNeXtBase.__init__ 的签名是
        # (plans, configuration, fold, dataset_json, unpack_dataset=True, device=...)，
        # 位置传参会把 device 错位塞进 unpack_dataset，device 落回默认（无 index）→
        # DDP 下兜底成 cuda:local_rank 而绑错卡（同 2026-08-11 事故）。
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # nnUZoo 原版超参（对应 nnUZoo nnUNetTrainerSwinTransformerUnet.__init__）
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

        model = SwinTransformerUnet(
            patch_size=4,
            in_ch=num_input_channels,
            out_ch=num_output_channels,
            depths=(2, 2, 9, 2),
            embed_dim=96,
            num_heads=(3, 6, 12, 24),
            window_size=7,
            qkv_bias=True,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0,
            attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        print(f"SwinTransformerUnet built. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model
