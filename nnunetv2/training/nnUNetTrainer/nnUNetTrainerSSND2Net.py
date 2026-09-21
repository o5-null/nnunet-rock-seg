"""
SSND2Net (SS2D2Net) 训练器 — X2Net 系列中性能最优的 Mamba-SS2D 混合架构

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标 (Precision/Recall/Std Dice)

v2 迁移说明:
  - 使用 build_network_architecture 标准 v2 签名
    (plans_manager, configuration_manager, num_input_channels, num_output_channels, ...)
  - 直接使用 num_output_channels 取代 dataset_json → get_label_manager 调用链
  - 移除 nnUZoo 原版中基类已覆盖的 on_epoch_end 等冗余方法
  - ⚠️ configure_optimizers 不属于"冗余方法"，不可移除：nnUZoo 原版是
    AdamW(lr=1e-4, wd=5e-2) + CosineAnnealingLR，而 nnUNet 基类提供的是
    SGD(lr=1e-2, wd=3e-5) + PolyLRScheduler，两者不等价（lr 差 100 倍）。
    早期误删导致 SSND2Net 以 SGD lr=1e-2 训练 → loss 逐 epoch 攀升 → NaN
    （2026-09-15 定位根因）。现按 nnUZoo 原版恢复。
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.nets.ssnd2net import SSND2Net
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class nnUNetTrainerSSND2Net(nnUNetTrainer_MedNeXtBase):
    """
    SSND2Net: Selective Scan 2D Network
    U2Net-style nested encoder-decoder with SS2D (Selective Scan 2D) Mamba blocks.
    支持 2D/3D 通过 spatial_dims 自动适配。
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # 必须关键字传 device：MRO 下一跳 nnUNetTrainer_MedNeXtBase.__init__ 的签名是
        # (plans, configuration, fold, dataset_json, unpack_dataset=True, device=...)，
        # 位置传参会把 device 错位塞进 unpack_dataset，device 落回默认（无 index）→
        # DDP 下兜底成 cuda:local_rank 而绑错卡（同 2026-08-11 SegResNet/MedNeXtBase 事故）。
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # nnUZoo 原版超参（对应 nnUZoo nnUNetTrainerSSND2Net.__init__）：
        # initial_lr=1e-4 / weight_decay=5e-2，配合下面的 AdamW + CosineAnnealingLR。
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        # ========== 数值稳定性修复: fp16 → bf16 (2026-09-16) ==========
        # 根因: fp16 只有 5 位指数（动态范围约 6e-5 .. 65504），SSM 选择性扫描的 exp/log 与深监督累加 的
        # 数值范围易越界 → 溢出成 NaN。bf16 指数位与 fp32 相同（8 位），动态范围一致，
        # 从源头消除溢出（同 LightMamba2Net 2026-07-31 的修复思路）。
        # 实证: 同网络族的 LightMamba2Net fp16 版 nan=101 崩于 ep77，
        # bf16 版跑满 100 epoch（nan=0, dice 0.6266）。
        self.autocast_dtype = torch.bfloat16
        # bf16 无需梯度缩放。显式禁用 GradScaler —— 它在 NaN 步只会静默跳过
        # optimizer.step，让权重在「看似训练」中悄悄退化（LightMUNet 空转 88 epoch 的机制之一）。
        self.grad_scaler = None

    def configure_optimizers(self):
        """按 nnUZoo 原版恢复 AdamW + CosineAnnealingLR（勿再当"冗余方法"删除）。

        基类 nnUNetTrainer 提供的是 nnUNet 官方默认 SGD(lr=1e-2, momentum=0.99,
        nesterov=True, wd=3e-5) + PolyLRScheduler，与本模型（SS2D/selective-scan
        混合，407 LayerNorm + 471 InstanceNorm2d）原版训练协议不同：原版以
        AdamW 的自适应步长配 lr=1e-4，而 SGD 用 1e-2 是 100 倍步长，训练在数个
        epoch 内发散至 NaN（2026-09-15 实测 loss 2.08→9.91→nan）。
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

        print(f"SSND2Net built. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model
