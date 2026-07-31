"""
LightMamba2Net 训练器 — 轻量级 Mamba-2 架构

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标

v2 迁移说明:
  - 使用标准 v2 签名 build_network_architecture
  - 内联工厂逻辑，直接构造 LightMamba2Net

数值稳定性修复 (2026-07-31):
  - AMP 从 fp16 改为 bf16: Mamba2 SSM 的 A_log/dt 指数运算在 fp16 下动态范围不足，
    Epoch 41 左右出现 train/val loss NaN 且无法恢复（详见 docs/lightmamba2net-nan-crash-2026-07-31.md）
  - bf16 拥有与 fp32 相同的 8 位指数，从根本上消除 SSM 离散化指数溢出
  - grad_scaler 置 None: bf16 无需梯度缩放，且避免 GradScaler 静默跳过 NaN step 掩盖问题
  - on_epoch_end 添加 NaN 检测: 一旦 train_loss 为 NaN 自动回退到 checkpoint_best
"""
import numpy as np
import shutil
from os.path import join, isfile

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
        # ========== 数值稳定性修复: fp16 → bf16 (2026-07-31) ==========
        # 根因: Mamba2 SSM 的 A_log/dt 指数运算在 fp16 下动态范围不足（8 位指数缺失），
        # 深层 U²-Net 残差累积后触发溢出，Epoch 41 起 loss 变 NaN 且无法自愈。
        # bf16 指数位与 fp32 相同（8 位），动态范围一致，从源头消除 SSM 溢出。
        self.autocast_dtype = torch.bfloat16
        # bf16 无需梯度缩放；显式禁用 GradScaler，避免 NaN step 被静默跳过导致
        # 权重悄悄退化而训练空转（本次事故 34+ epoch 白白浪费的元凶之一）。
        self.grad_scaler = None

    def _do_i_compile(self):
        """LightMamba2Net 的 Mamba2 SSM 计算图在 triton JIT 编译时崩溃（Windows + triton 3.7.1）"""
        return False

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

    def on_epoch_end(self):
        """
        在父类 epoch 结束处理之前检测 NaN loss。

        2026-07-31 事故: Epoch 41 train_loss 变 NaN 后训练继续空转 34+ epoch
        （ema_fg_dice 从 0.5130 崩至 0.0185），GPU 白白浪费 ~4.5 小时。
        此处一旦检测到 train_loss 非有限值：
          - 若存在 checkpoint_best（NaN 前的干净权重）→ 回退并继续
          - 否则抛 RuntimeError 终止训练，避免空转
        """
        # train_loss 在 on_train_epoch_end 已写入 logger；NaN 意味着权重已不可信
        try:
            tr_loss = float(self.logger.get_value('train_losses', step=-1))
        except (IndexError, TypeError, ValueError):
            tr_loss = float('nan')

        if not np.isfinite(tr_loss):
            self.print_to_log_file(f'[NaN DETECTED] train_loss = {tr_loss}. Rolling back to checkpoint_best ...')
            best_ckpt = join(self.output_folder, 'checkpoint_best.pth')
            if isfile(best_ckpt):
                self.print_to_log_file(f'Found {best_ckpt}, restoring clean weights ...')
                # 直接用干净的 best checkpoint 覆盖 checkpoint_latest（--c 续训加载的就是它），
                # 其中包含完整的网络权重 / optimizer / logger / current_epoch(=42)，
                # 用户下次 --c 重启即从 Epoch 41 的干净状态继续，无任何错位。
                shutil.copyfile(best_ckpt, join(self.output_folder, 'checkpoint_latest.pth'))
                self.print_to_log_file('Overwrote checkpoint_latest.pth with clean weights.')
                self.print_to_log_file('Training stopped. Restart with --c to continue from clean weights.')
                raise SystemExit(f'[NaN DETECTED] Rolled back to checkpoint_best. Restart with --c to continue.')
            else:
                self.print_to_log_file('[NaN DETECTED] No checkpoint_best found. Aborting training.')
                raise RuntimeError('NaN loss detected and no checkpoint_best available for rollback.')
        # 正常路径: 交给父类（MedNeXtBase → nnUNetTrainer）处理日志/checkpoint
        super().on_epoch_end()
