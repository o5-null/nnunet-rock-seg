"""
TransferLearningBase mixin — 迁移学习基类 (batchgeneratorsv2 版)

将 MedNeXt-Volcanic-Rocks 项目中原始的 batchgenerators v1 风格空间增强
替换为官方 nnUNet 使用的 batchgeneratorsv2 API。

包含:
  - TverskyLoss: 可调节 alpha/beta 的自定义损失函数
  - TransferLearningBase: 提供渐进解冻、自适应损失调整、early stopping 等功能的 mixin

所有空间/强度数据增强均通过 batchgeneratorsv2 的 ComposeTransforms 流水线完成。
"""

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch import nn, autocast
import torch
import numpy as np
import time
from collections import deque
import os
import shutil

# batchgeneratorsv2 导入 — 替代原 v1 手动增强实现
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform


class TverskyLoss(nn.Module):
    """Tversky 损失函数 — 可独立调节 FP/FN 惩罚权重

    loss = 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    alpha 控制 FP 惩罚, beta 控制 FN 惩罚。
    """

    def __init__(self, apply_nonlin=None, batch_dice=False,
                 do_bg=True, smooth=1e-5, alpha=0.3, beta=0.7):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-7

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)

        shp_x = x.shape
        axes = [0] + list(range(2, len(shp_x))) if self.batch_dice else list(range(2, len(shp_x)))

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        tp = torch.clamp(tp, min=0)
        fp = torch.clamp(fp, min=0)
        fn = torch.clamp(fn, min=0)

        nominator = tp
        denominator = tp + self.alpha * fp + self.beta * fn

        tversky = (nominator + self.smooth) / (denominator + self.smooth + self.eps)
        tversky = torch.clamp(tversky, self.eps, 1.0 - self.eps)

        if not self.do_bg:
            tversky = tversky[1:] if self.batch_dice else tversky[:, 1:]

        if torch.isnan(tversky).any():
            tversky = torch.where(torch.isnan(tversky),
                                 torch.ones_like(tversky) * self.eps,
                                 tversky)

        return -tversky.mean()


class TransferLearningBase:
    """迁移学习 mixin — 与 nnUNetTrainer 通过多继承组合使用

    提供:
      - 渐进解冻 (progressive unfreezing)
      - Tversky 损失 + 自适应参数调整
      - batchgeneratorsv2 空间/强度数据增强
      - 基于 dice_ignore_fn 的 early stopping
      - 最佳模型跟踪保存
    """

    def _init_transfer_learning_params(self):
        """初始化迁移学习超参数"""
        self.num_epochs = 100
        self.num_iterations_per_epoch = 100
        self.initial_lr = 1e-4
        self.weight_decay = 1e-4
        self.patience = 20
        self.validate_every = 2

        self.freeze_epochs = 10
        self.encoder_lr_factor = 0.01
        self.unfrozen_layers = set()

        self.use_warm_restarts = False
        self.warm_restart_T0 = 40
        self.warm_restart_T_mult = 2
        self.warm_restart_eta_min = 5e-7

        self.early_stop_patience = 25
        self.early_stop_min_delta = 0.001

        self.progressive_unfreeze_schedule = {
            15: ["encoder.stages.3"],
            30: ["encoder.stages.2", "encoder.stages.3"],
            50: ["encoder.stages.1", "encoder.stages.2", "encoder.stages.3"],
            70: "all"
        }

        self.unfreeze_recall_threshold = 0.35
        self.dice_plateau_patience = 5
        self.max_wait_epochs = 40
        self.epochs_without_dice_improvement = 0
        self.best_dice_for_unfreeze = 0.0

        self.tversky_alpha = 0.3
        self.tversky_beta = 0.7

        self.enable_adaptive_loss = True
        self.adaptive_adjust_interval = 20
        self.dice_ignore_fn_target = 0.85
        self.dice_ignore_fn_ma_window = 5
        self.smooth_adjust_k = 0.01
        self.alpha_min = 0.1
        self.alpha_max = 0.5

        self.track_dice_ignore_fn = True
        self.best_dice_ignore_fn = 0.0
        self.best_dice_ignore_fn_epoch = 0
        self.best_dice_ignore_fn_model_path = None
        self.dice_ignore_fn_history = []

        self.train_losses = []
        self.val_losses = []
        self.precision_history = []
        self.recall_history = []
        self.alpha_history = []

        self.enable_spatial_augmentation = True

        # --- batchgeneratorsv2 增强参数 ---
        self.p_rotation = 0.3
        self.rotation_for_DA = {
            "x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            "y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            "z": (-5. / 360 * 2. * np.pi, 5. / 360 * 2. * np.pi),
        }

        self.p_elastic_deform = 0.25
        self.elastic_deform_sigma_range = (5.0, 15.0)
        self.elastic_deform_magnitude_range = (50.0, 150.0)

        self.p_scaling = 0.2
        self.scale_range = (0.85, 1.15)
        self.p_synchronize_scaling_across_axes = 0.8

        self.p_gamma = 0.5
        self.gamma_range = (0.7, 1.5)

        self.p_gaussian_noise = 0.3
        self.gaussian_noise_variance = (0.0, 0.05)

        self.p_gaussian_blur = 0.15
        self.gaussian_blur_sigma = (0.3, 0.8)

        self.p_brightness = 0.2
        self.brightness_multiplier_range = (0.85, 1.15)

        self.p_contrast = 0.2
        self.contrast_range = (0.85, 1.15)

        self.p_low_res = 0.2
        self.low_res_scale_range = (0.7, 1.0)

        self.do_mirroring = True

        self.preserve_layer_structure = True
        self.layer_axis = 2

    # ------------------------------------------------------------------
    #  batchgeneratorsv2 增强流水线
    # ------------------------------------------------------------------

    def _build_augmentation_pipeline(self, patch_size):
        """构建 batchgeneratorsv2 数据增强 ComposeTransforms 流水线

        替代原 v1 手动 torch 增强实现 (gamma/noise/blur/brightness/contrast),
        并新增原代码中定义但未使用的 SpatialTransform (旋转/缩放/弹性变形) 和 MirrorTransform。

        Args:
            patch_size: 输入数据的空间维度 (d, h, w)

        Returns:
            ComposeTransforms 实例
        """
        transforms = []

        # batchgeneratorsv2 空间变换 — 替代原 v1 SpatialTransform
        rotation_range = self.rotation_for_DA["x"]
        transforms.append(
            SpatialTransform(
                patch_size=patch_size,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=self.p_elastic_deform,
                elastic_deform_scale=self.elastic_deform_sigma_range,
                elastic_deform_magnitude=self.elastic_deform_magnitude_range,
                p_rotation=self.p_rotation,
                rotation=rotation_range,
                p_scaling=self.p_scaling,
                scaling=self.scale_range,
                p_synchronize_scaling_across_axes=self.p_synchronize_scaling_across_axes,
                bg_style_seg_sampling=False,
                border_mode_seg="constant",
            )
        )

        # batchgeneratorsv2 镜像变换 — 替代原 v1 MirrorTransform
        if self.do_mirroring:
            transforms.append(MirrorTransform(allowed_axes=(0, 1, 2)))

        # Gamma 变换 (v2 RandomTransform + GammaTransform)
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast(self.gamma_range),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=self.p_gamma
        ))

        # 高斯噪声 (v2 GaussianNoiseTransform)
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=self.gaussian_noise_variance,
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=self.p_gaussian_noise
        ))

        # 高斯模糊 (v2 GaussianBlurTransform)
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=self.gaussian_blur_sigma,
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5,
                benchmark=True
            ), apply_probability=self.p_gaussian_blur
        ))

        # 亮度倍增 (v2 MultiplicativeBrightnessTransform)
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast(self.brightness_multiplier_range),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=self.p_brightness
        ))

        # 对比度调整 (v2 ContrastTransform)
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast(self.contrast_range),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=self.p_contrast
        ))

        # 低分辨率模拟 (v2 SimulateLowResolutionTransform)
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=self.low_res_scale_range,
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=self.p_low_res
        ))

        return ComposeTransforms(transforms)

    def _apply_spatial_augmentation(self, data, target):
        """应用 batchgeneratorsv2 数据增强流水线

        将 data/target 转为 numpy -> ComposeTransforms -> 转回 torch。
        保留原方法的形状校验、NaN/Inf 保护逻辑。
        """
        if not self.enable_spatial_augmentation:
            return data, target

        if isinstance(target, list):
            return data, target

        original_shape = data.shape
        original_target_shape = target.shape if isinstance(target, torch.Tensor) else None

        # 构建 batchgeneratorsv2 增强流水线
        patch_size = data.shape[2:]  # (d, h, w)
        transforms = self._build_augmentation_pipeline(patch_size)

        # batchgeneratorsv2 变换操作 numpy 数组，需转换
        data_np = data.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        data_dict = {"data": data_np, "seg": target_np}
        result = transforms(**data_dict)

        # 转回 torch 并放回原设备
        data = torch.from_numpy(result["data"]).to(data.device)
        target = torch.from_numpy(result["seg"]).to(target.device)

        data = torch.clamp(data, -1.0, 2.0)

        # 形状一致性校验
        if data.shape != original_shape:
            self.print_to_log_file(f"[WARNING] Augmentation changed shape: {original_shape} -> {data.shape}")
            data = torch.clamp(data, 0.0, 1.0)

        if isinstance(target, torch.Tensor) and original_target_shape is not None:
            if target.shape != original_target_shape:
                self.print_to_log_file(f"[WARNING] Target shape changed: {original_target_shape} -> {target.shape}")
                target = None

        if torch.isnan(data).any() or torch.isinf(data).any():
            self.print_to_log_file(f"[WARNING] NaN/Inf after augmentation")
            return torch.clamp(data, 0.0, 1.0), target

        return data, target

    # ------------------------------------------------------------------
    #  配置打印
    # ------------------------------------------------------------------

    def _print_transfer_config(self, trainer_name="Transfer Learning"):
        self.print_to_log_file("=" * 60)
        self.print_to_log_file(f"{trainer_name} - Transfer Learning Config")
        self.print_to_log_file("=" * 60)
        self.print_to_log_file(f"Epochs: {self.num_epochs}, Freeze: {self.freeze_epochs}")
        self.print_to_log_file(f"LR: {self.initial_lr}, Encoder LR factor: {self.encoder_lr_factor}")
        self.print_to_log_file(f"Tversky Alpha: {self.tversky_alpha}, Beta: {self.tversky_beta}")
        self.print_to_log_file(f"Adaptive adjust interval: {self.adaptive_adjust_interval}")
        self.print_to_log_file(f"Spatial augmentation: {self.enable_spatial_augmentation}")
        self.print_to_log_file("=" * 60)

    # ------------------------------------------------------------------
    #  损失 / 优化器 / 调度器
    # ------------------------------------------------------------------

    def configure_loss_function_transfer(self):
        from nnunetv2.utilities.helpers import softmax_helper_dim1

        self.print_to_log_file(f"Using Tversky Loss (alpha={self.tversky_alpha}, beta={self.tversky_beta})")

        return TverskyLoss(
            apply_nonlin=softmax_helper_dim1,
            batch_dice=True,
            do_bg=False,
            smooth=1e-5,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta
        )

    def configure_optimizers_transfer(self):
        encoder_params = []
        decoder_params = []

        for name, param in self.network.named_parameters():
            if param.requires_grad:
                if "encoder" in name.lower():
                    encoder_params.append(param)
                else:
                    decoder_params.append(param)

        param_groups = [
            {"params": encoder_params, "lr": self.initial_lr * self.encoder_lr_factor},
            {"params": decoder_params, "lr": self.initial_lr}
        ]

        optimizer = AdamW(param_groups, weight_decay=self.weight_decay, eps=1e-5)

        if getattr(self, "use_warm_restarts", False):
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=getattr(self, "warm_restart_T0", 40),
                T_mult=getattr(self, "warm_restart_T_mult", 2),
                eta_min=getattr(self, "warm_restart_eta_min", 5e-7)
            )
            self.print_to_log_file(f"Using CosineAnnealingWarmRestarts: T0={self.warm_restart_T0}, T_mult={self.warm_restart_T_mult}")
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        total = sum(p.numel() for p in self.network.parameters())
        trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.print_to_log_file(f"Params: {trainable}/{total} trainable ({trainable/total*100:.1f}%)")

        return optimizer, scheduler

    # ------------------------------------------------------------------
    #  渐进解冻
    # ------------------------------------------------------------------

    def _check_dice_plateau(self):
        if self.current_epoch >= self.max_wait_epochs:
            return True
        if self.epochs_without_dice_improvement >= self.dice_plateau_patience:
            return True
        return False

    def _progressive_unfreeze(self):
        if self.current_epoch not in self.progressive_unfreeze_schedule:
            return False

        current_recall = self.recall_history[-1] if self.recall_history else 0
        if current_recall < self.unfreeze_recall_threshold:
            return False

        if not self._check_dice_plateau():
            return False

        layers = self.progressive_unfreeze_schedule[self.current_epoch]

        self.print_to_log_file(f"Unfreezing at epoch {self.current_epoch}: {layers}")

        if layers == "all":
            for name, param in self.network.named_parameters():
                if "encoder" in name.lower():
                    param.requires_grad = True
                    self.unfrozen_layers.add(name)
        else:
            for pattern in layers:
                for name, param in self.network.named_parameters():
                    if pattern in name and not param.requires_grad:
                        param.requires_grad = True
                        self.unfrozen_layers.add(name)

        self.optimizer, self.lr_scheduler = self.configure_optimizers_transfer()
        return True

    def on_train_epoch_start_transfer(self):
        if self.current_epoch >= self.freeze_epochs:
            self._progressive_unfreeze()

        encoder_lr = self.optimizer.param_groups[0]["lr"]
        decoder_lr = self.optimizer.param_groups[1]["lr"]

        status = "frozen" if self.current_epoch < self.freeze_epochs else \
                 "unfreezing" if len(self.unfrozen_layers) == 0 else \
                 "partial" if len(self.unfrozen_layers) < 10 else "full"

        self.print_to_log_file(f"Epoch {self.current_epoch}: encoder_lr={encoder_lr:.2e}, decoder_lr={decoder_lr:.2e}, status={status}")

    # ------------------------------------------------------------------
    #  训练步骤 (含 batchgeneratorsv2 增强)
    # ------------------------------------------------------------------

    def train_step_transfer(self, data, target):
        if self.enable_spatial_augmentation:
            data, target = self._apply_spatial_augmentation(data, target)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True):
            output = self.network(data)
            l = self.loss(output, target)

        if torch.isnan(l) or torch.isinf(l):
            self.print_to_log_file(f"[WARNING] NaN/Inf loss at epoch {self.current_epoch}")
            return {"loss": np.array([1.0])}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
            self.optimizer.step()

        loss_val = l.detach().cpu().numpy()
        self.train_losses.append(float(loss_val))

        return {"loss": loss_val}

    # ------------------------------------------------------------------
    #  自适应损失调整
    # ------------------------------------------------------------------

    def _adapt_loss_parameters(self):
        if not self.enable_adaptive_loss:
            return False

        if len(self.dice_ignore_fn_history) < self.dice_ignore_fn_ma_window:
            return False

        current = self.dice_ignore_fn_history[-1]
        ma = np.mean(self.dice_ignore_fn_history[-self.dice_ignore_fn_ma_window:])

        if self.train_losses and self.train_losses[-1] > 0.3:
            self.print_to_log_file(f"[ADAPT] Skipping: unstable loss {self.train_losses[-1]:.4f}")
            return False

        old_alpha = self.tversky_alpha
        deviation = self.dice_ignore_fn_target - ma

        if deviation > 0.15 and old_alpha < 0.4:
            new_alpha = min(old_alpha + self.smooth_adjust_k, self.alpha_max)
        elif deviation < -0.05 and old_alpha > 0.2:
            new_alpha = max(old_alpha - self.smooth_adjust_k, self.alpha_min)
        else:
            return False

        if abs(new_alpha - old_alpha) > 0.005:
            self.tversky_alpha = new_alpha
            self.tversky_beta = 1.0 - new_alpha
            self.alpha_history.append(self.tversky_alpha)

            self.print_to_log_file(f"[ADAPT] alpha: {old_alpha:.3f} -> {self.tversky_alpha:.3f}, beta: {self.tversky_beta:.3f}")

            if hasattr(self, "enable_deep_supervision") and self.enable_deep_supervision:
                from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
                if isinstance(self.loss, DeepSupervisionWrapper):
                    self.loss.loss = self.configure_loss_function_transfer()
                    self.print_to_log_file(f"Using Tversky Loss (alpha={self.tversky_alpha}, beta={self.tversky_beta})")
                else:
                    self.loss = self.configure_loss_function_transfer()
            else:
                self.loss = self.configure_loss_function_transfer()
            return True

        return False

    # ------------------------------------------------------------------
    #  验证阶段
    # ------------------------------------------------------------------

    def on_validation_epoch_end_transfer(self, val_outputs):
        if val_outputs and len(val_outputs) > 0:
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import collate_outputs
            outputs = collate_outputs(val_outputs)
            tp = np.sum(outputs["tp_hard"], 0)
            fp = np.sum(outputs["fp_hard"], 0)
            fn = np.sum(outputs["fn_hard"], 0)

            std_dice = [2*i/(2*i+j+k) if (2*i+j+k) > 0 else 0 for i,j,k in zip(tp,fp,fn)]
            dice_ignore_fn = [2*i/(2*i+j) if (2*i+j) > 0 else 0 for i,j in zip(tp,fp)]
            precision = [i/(i+j) if (i+j) > 0 else 0 for i,j in zip(tp,fp)]
            recall = [i/(i+k) if (i+k) > 0 else 0 for i,k in zip(tp,fn)]

            self.current_val_metrics = {
                "std_dice": np.nanmean(std_dice),
                "dice_ignore_fn": np.nanmean(dice_ignore_fn),
                "precision": np.nanmean(precision),
                "recall": np.nanmean(recall)
            }

            self.precision_history.append(self.current_val_metrics["precision"])
            self.recall_history.append(self.current_val_metrics["recall"])
            self.dice_ignore_fn_history.append(self.current_val_metrics["dice_ignore_fn"])

    # ------------------------------------------------------------------
    #  最佳模型保存
    # ------------------------------------------------------------------

    def _save_best_dice_ignore_fn_model(self):
        try:
            if hasattr(self, "output_folder"):
                path = os.path.join(self.output_folder, f"best_dice_ignore_fn_epoch_{self.current_epoch}.pth")
                checkpoint = {
                    "epoch": self.current_epoch,
                    "model_state_dict": self.network.state_dict(),
                    "best_dice_ignore_fn": self.best_dice_ignore_fn,
                }

                if self.best_dice_ignore_fn_model_path and os.path.exists(self.best_dice_ignore_fn_model_path):
                    os.remove(self.best_dice_ignore_fn_model_path)

                torch.save(checkpoint, path)
                self.best_dice_ignore_fn_model_path = path
                self.print_to_log_file(f"Saved best ignore-FN dice model: {self.best_dice_ignore_fn:.4f}")

                latest = os.path.join(self.output_folder, "best_dice_ignore_fn_latest.pth")
                shutil.copy2(path, latest)
        except Exception as e:
            self.print_to_log_file(f"[ERROR] Failed to save model: {e}")

    # ------------------------------------------------------------------
    #  数据增强配置打印
    # ------------------------------------------------------------------

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size_transfer(self):
        self.print_to_log_file("\nData Augmentation Config:")

        layer_protection = "enabled" if getattr(self, "preserve_layer_structure", False) else "disabled"
        self.print_to_log_file(f"Layer-Aware Rotation {layer_protection}:")
        self.print_to_log_file(f"  X-axis: +/-{np.degrees(self.rotation_for_DA['x'][1]):.0f} deg")
        self.print_to_log_file(f"  Y-axis: +/-{np.degrees(self.rotation_for_DA['y'][1]):.0f} deg")
        self.print_to_log_file(f"  Z-axis: +/-{np.degrees(self.rotation_for_DA['z'][1]):.0f} deg")

        self.print_to_log_file(f"Elastic Deformation:")
        self.print_to_log_file(f"  Probability: {self.p_elastic_deform}")
        self.print_to_log_file(f"  Sigma: {self.elastic_deform_sigma_range}")
        self.print_to_log_file(f"  Magnitude: {self.elastic_deform_magnitude_range}")

        self.print_to_log_file(f"Scaling:")
        self.print_to_log_file(f"  Probability: {self.p_scaling}")
        self.print_to_log_file(f"  Range: {self.scale_range}")

        self.print_to_log_file(f"Intensity Transforms:")
        self.print_to_log_file(f"  Gamma: p={self.p_gamma}, range={self.gamma_range}")
        self.print_to_log_file(f"  Gaussian Noise: p={self.p_gaussian_noise}, var={self.gaussian_noise_variance}")
        self.print_to_log_file(f"  Gaussian Blur: p={self.p_gaussian_blur}, sigma={self.gaussian_blur_sigma}")
        self.print_to_log_file(f"  Brightness: p={self.p_brightness}, range={self.brightness_multiplier_range}")
        self.print_to_log_file(f"  Contrast: p={self.p_contrast}, range={self.contrast_range}")
        self.print_to_log_file(f"  Low Resolution: p={self.p_low_res}, scale={self.low_res_scale_range}")

        self.print_to_log_file(f"Mirroring: {self.do_mirroring}")

        initial_patch_size = self.configuration_manager.patch_size
        return self.rotation_for_DA, self.do_mirroring, initial_patch_size, self.scale_range

    # ------------------------------------------------------------------
    #  Epoch 结束 (含 early stopping)
    # ------------------------------------------------------------------

    def on_epoch_end_transfer(self):
        if hasattr(self, "best_dice_ignore_fn_epoch") and self.current_epoch > self.freeze_epochs + 10:
            no_improve = self.current_epoch - self.best_dice_ignore_fn_epoch
            if no_improve > self.early_stop_patience:
                self.print_to_log_file(f"[EARLY STOP] No improvement for {no_improve} epochs at epoch {self.current_epoch}")
                return False
        return True

    # ------------------------------------------------------------------
    #  训练总结
    # ------------------------------------------------------------------

    def _print_training_summary_transfer(self):
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("Training Summary (Transfer Learning)")
        self.print_to_log_file("=" * 60)
        if self.train_losses:
            self.print_to_log_file(f"Train loss: {self.train_losses[0]:.4f} -> {self.train_losses[-1]:.4f}")
        self.print_to_log_file(f"Best ignore-FN dice: {self.best_dice_ignore_fn:.4f} (epoch {self.best_dice_ignore_fn_epoch})")
        if self.alpha_history:
            self.print_to_log_file(f"Alpha adjustments: {len(self.alpha_history)} times")
            self.print_to_log_file(f"Final alpha/beta: {self.tversky_alpha:.3f}/{self.tversky_beta:.3f}")
        self.print_to_log_file("=" * 60)
