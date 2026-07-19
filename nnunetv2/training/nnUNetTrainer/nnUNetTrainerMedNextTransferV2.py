from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMedNext import (
    nnUNetTrainerMedNext,
    nnUNetTrainerV2_MedNeXt_B_kernel5,
    nnUNetTrainerV2_MedNeXt_L_kernel5,
)
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn, autocast
import numpy as np
import time
from collections import deque
import os
import shutil


class TverskyLoss(nn.Module):
    """Tversky Loss - 处理类别不平衡的分割任务"""
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


class nnUNetTrainerV2_MedNeXt_B_TransferLearningV2(nnUNetTrainerV2_MedNeXt_B_kernel5):
    """
    MedNeXt迁移学习训练器V2 - 修复版
    修复问题：
    1. 禁用频繁的自适应参数调整（每3轮改为每20轮）
    2. 使用标准Tversky Loss替代Focal Tversky Loss
    3. 更保守的Alpha/Beta初始值（0.3/0.7）
    4. 更稳定的学习率调度（CosineAnnealingLR替代WarmRestarts）
    5. 增加梯度裁剪阈值
    """
    
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        # 注意：新基类链不接收 unpack_dataset 参数
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # 训练参数
        self.num_epochs = 100
        self.num_iterations_per_epoch = 100
        self.initial_lr = 1e-4
        self.weight_decay = 1e-4
        self.patience = 20
        self.validate_every = 2
        
        # 冻结与解冻
        self.freeze_epochs = 10
        self.encoder_lr_factor = 0.01
        self.unfrozen_layers = set()
        
        # 渐进式解冻计划
        self.progressive_unfreeze_schedule = {
            15: ['encoder.stages.3'],
            30: ['encoder.stages.2', 'encoder.stages.3'],
            50: ['encoder.stages.1', 'encoder.stages.2', 'encoder.stages.3'],
            70: 'all'
        }
        
        # 解冻触发条件
        self.unfreeze_recall_threshold = 0.35
        self.dice_plateau_patience = 5
        self.max_wait_epochs = 40
        self.epochs_without_dice_improvement = 0
        self.best_dice_for_unfreeze = 0.0
        
        # Tversky Loss参数 - 更保守的初始值
        self.tversky_alpha = 0.3  # 从0.05改为0.3，更平衡
        self.tversky_beta = 0.7   # 从0.95改为0.7
        
        # 自适应调整 - 大幅降低频率
        self.enable_adaptive_loss = True
        self.adaptive_adjust_interval = 20  # 从3改为20，避免频繁调整
        self.dice_ignore_fn_target = 0.85
        self.dice_ignore_fn_ma_window = 5
        self.smooth_adjust_k = 0.01
        self.alpha_min = 0.1
        self.alpha_max = 0.5
        
        # 指标追踪
        self.track_dice_ignore_fn = True
        self.best_dice_ignore_fn = 0.0
        self.best_dice_ignore_fn_epoch = 0
        self.best_dice_ignore_fn_model_path = None
        self.dice_ignore_fn_history = []
        
        # 统计
        self.train_losses = []
        self.val_losses = []
        self.precision_history = []
        self.recall_history = []
        self.alpha_history = []
        
        self._print_config()
    
    def _print_config(self):
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("MedNeXt Transfer Learning Trainer V2")
        self.print_to_log_file("=" * 60)
        self.print_to_log_file(f"Epochs: {self.num_epochs}, Freeze: {self.freeze_epochs}")
        self.print_to_log_file(f"LR: {self.initial_lr}, Encoder LR factor: {self.encoder_lr_factor}")
        self.print_to_log_file(f"Tversky Alpha: {self.tversky_alpha}, Beta: {self.tversky_beta}")
        self.print_to_log_file(f"Adaptive adjust interval: {self.adaptive_adjust_interval}")
        self.print_to_log_file("=" * 60)
    
    def configure_loss_function(self):
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
    
    def configure_optimizers(self):
        encoder_params = []
        decoder_params = []
        
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                if 'encoder' in name.lower():
                    encoder_params.append(param)
                else:
                    decoder_params.append(param)
        
        param_groups = [
            {'params': encoder_params, 'lr': self.initial_lr * self.encoder_lr_factor},
            {'params': decoder_params, 'lr': self.initial_lr}
        ]
        
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay, eps=1e-5)
        
        # 使用更稳定的CosineAnnealingLR替代WarmRestarts
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
        
        total = sum(p.numel() for p in self.network.parameters())
        trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.print_to_log_file(f"Params: {trainable}/{total} trainable ({trainable/total*100:.1f}%)")
        
        return optimizer, scheduler
    
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
        
        if layers == 'all':
            for name, param in self.network.named_parameters():
                if 'encoder' in name.lower():
                    param.requires_grad = True
                    self.unfrozen_layers.add(name)
        else:
            for pattern in layers:
                for name, param in self.network.named_parameters():
                    if pattern in name and not param.requires_grad:
                        param.requires_grad = True
                        self.unfrozen_layers.add(name)
        
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        return True
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        
        if self.current_epoch >= self.freeze_epochs:
            self._progressive_unfreeze()
        
        encoder_lr = self.optimizer.param_groups[0]['lr']
        decoder_lr = self.optimizer.param_groups[1]['lr']
        
        status = "frozen" if self.current_epoch < self.freeze_epochs else \
                 "unfreezing" if len(self.unfrozen_layers) == 0 else \
                 "partial" if len(self.unfrozen_layers) < 10 else "full"
        
        self.print_to_log_file(f"Epoch {self.current_epoch}: encoder_lr={encoder_lr:.2e}, decoder_lr={decoder_lr:.2e}, status={status}")
    
    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(self.device.type, enabled=True):
            output = self.network(data)
            l = self.loss(output, target)
        
        if torch.isnan(l) or torch.isinf(l):
            self.print_to_log_file(f"[WARNING] NaN/Inf loss at epoch {self.current_epoch}")
            return {'loss': np.array([1.0])}
        
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
        
        return {'loss': loss_val}
    
    def _adapt_loss_parameters(self):
        if not self.enable_adaptive_loss:
            return False
        
        if len(self.dice_ignore_fn_history) < self.dice_ignore_fn_ma_window:
            return False
        
        current = self.dice_ignore_fn_history[-1]
        ma = np.mean(self.dice_ignore_fn_history[-self.dice_ignore_fn_ma_window:])
        
        # 检查训练是否稳定
        if self.train_losses and self.train_losses[-1] > 0.3:
            self.print_to_log_file(f"[ADAPT] Skipping: unstable loss {self.train_losses[-1]:.4f}")
            return False
        
        old_alpha = self.tversky_alpha
        deviation = self.dice_ignore_fn_target - ma
        
        # 保守调整
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
            self.loss = self.configure_loss_function()
            return True
        
        return False
    
    def on_validation_epoch_end(self, val_outputs=None):
        if val_outputs and len(val_outputs) > 0:
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import collate_outputs
            outputs = collate_outputs(val_outputs)
            tp = np.sum(outputs['tp_hard'], 0)
            fp = np.sum(outputs['fp_hard'], 0)
            fn = np.sum(outputs['fn_hard'], 0)
            
            std_dice = [2*i/(2*i+j+k) if (2*i+j+k) > 0 else 0 for i,j,k in zip(tp,fp,fn)]
            dice_ignore_fn = [2*i/(2*i+j) if (2*i+j) > 0 else 0 for i,j in zip(tp,fp)]
            precision = [i/(i+j) if (i+j) > 0 else 0 for i,j in zip(tp,fp)]
            recall = [i/(i+k) if (i+k) > 0 else 0 for i,k in zip(tp,fn)]
            
            self.current_val_metrics = {
                'std_dice': np.nanmean(std_dice),
                'dice_ignore_fn': np.nanmean(dice_ignore_fn),
                'precision': np.nanmean(precision),
                'recall': np.nanmean(recall)
            }
            
            self.precision_history.append(self.current_val_metrics['precision'])
            self.recall_history.append(self.current_val_metrics['recall'])
            self.dice_ignore_fn_history.append(self.current_val_metrics['dice_ignore_fn'])
        
        super().on_validation_epoch_end(val_outputs)
        
        if hasattr(self, 'current_val_metrics'):
            m = self.current_val_metrics
            self.print_to_log_file(f"Val - StdDice: {m['std_dice']:.4f}, IgnoreFNDice: {m['dice_ignore_fn']:.4f}, "
                                   f"Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")
            
            # 保存最佳ignore FN dice模型
            if self.track_dice_ignore_fn and m['dice_ignore_fn'] > self.best_dice_ignore_fn:
                self.best_dice_ignore_fn = m['dice_ignore_fn']
                self.best_dice_ignore_fn_epoch = self.current_epoch
                self._save_best_dice_ignore_fn_model()
            
            # 更新plateau计数器
            if m['dice_ignore_fn'] > self.best_dice_for_unfreeze + 0.001:
                self.best_dice_for_unfreeze = m['dice_ignore_fn']
                self.epochs_without_dice_improvement = 0
            else:
                self.epochs_without_dice_improvement += 1
            
            # 自适应调整
            if self.enable_adaptive_loss and self.current_epoch > 0 and \
               self.current_epoch % self.adaptive_adjust_interval == 0:
                self._adapt_loss_parameters()
    
    def _save_best_dice_ignore_fn_model(self):
        try:
            if hasattr(self, 'output_folder'):
                path = os.path.join(self.output_folder, f"best_dice_ignore_fn_epoch_{self.current_epoch}.pth")
                checkpoint = {
                    'epoch': self.current_epoch,
                    'model_state_dict': self.network.state_dict(),
                    'best_dice_ignore_fn': self.best_dice_ignore_fn,
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
    
    def on_epoch_end(self):
        continue_training = super().on_epoch_end()
        
        if hasattr(self, 'best_val_eval_criterion_MA') and self.current_epoch > self.freeze_epochs + 5:
            no_improve = self.current_epoch - self.best_val_eval_criterion_MA_epoch
            if no_improve > self.patience:
                self.print_to_log_file(f"Early stopping at epoch {self.current_epoch}")
                return False
        
        return continue_training
    
    def run_training(self):
        self.print_to_log_file("Starting training with V2 trainer")
        start = time.time()
        
        try:
            result = super().run_training()
            elapsed = time.time() - start
            self.print_to_log_file(f"Training completed in {elapsed/60:.1f} minutes")
            self._print_summary()
            return result
        except Exception as e:
            self.print_to_log_file(f"[ERROR] Training failed: {e}")
            raise
    
    def _print_summary(self):
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("Training Summary")
        self.print_to_log_file("=" * 60)
        if self.train_losses:
            self.print_to_log_file(f"Train loss: {self.train_losses[0]:.4f} -> {self.train_losses[-1]:.4f}")
        self.print_to_log_file(f"Best ignore-FN dice: {self.best_dice_ignore_fn:.4f} (epoch {self.best_dice_ignore_fn_epoch})")
        if self.alpha_history:
            self.print_to_log_file(f"Alpha adjustments: {len(self.alpha_history)} times")
            self.print_to_log_file(f"Final alpha/beta: {self.tversky_alpha:.3f}/{self.tversky_beta:.3f}")
        self.print_to_log_file("=" * 60)


class nnUNetTrainerV2_MedNeXt_L_TransferLearningV2(nnUNetTrainerV2_MedNeXt_L_kernel5):
    """
    MedNeXt Large模型迁移学习训练器V2
    基于B模型V2版本的相同配置，适配L模型更大的参数量
    """

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        # 注意：新基类链不接收 unpack_dataset 参数
        super().__init__(plans, configuration, fold, dataset_json, device)

        # 训练参数 - L模型使用更保守的设置
        self.num_epochs = 100
        self.num_iterations_per_epoch = 100
        self.initial_lr = 5e-5  # L模型使用更低的学习率
        self.weight_decay = 1e-4
        self.patience = 20
        self.validate_every = 2

        # 冻结与解冻 - L模型需要更长的冻结期
        self.freeze_epochs = 15  # L模型冻结更久
        self.encoder_lr_factor = 0.005  # L模型编码器学习率更低
        self.unfrozen_layers = set()

        # 渐进式解冻计划 - 针对L模型更深的结构
        self.progressive_unfreeze_schedule = {
            20: ['encoder.stages.3'],
            35: ['encoder.stages.2', 'encoder.stages.3'],
            55: ['encoder.stages.1', 'encoder.stages.2', 'encoder.stages.3'],
            75: 'all'
        }

        # 解冻触发条件
        self.unfreeze_recall_threshold = 0.35
        self.dice_plateau_patience = 5
        self.max_wait_epochs = 45  # L模型等待更久
        self.epochs_without_dice_improvement = 0
        self.best_dice_for_unfreeze = 0.0

        # Tversky Loss参数
        self.tversky_alpha = 0.3
        self.tversky_beta = 0.7

        # 自适应调整
        self.enable_adaptive_loss = True
        self.adaptive_adjust_interval = 20
        self.dice_ignore_fn_target = 0.85
        self.dice_ignore_fn_ma_window = 5
        self.smooth_adjust_k = 0.01
        self.alpha_min = 0.1
        self.alpha_max = 0.5

        # 指标追踪
        self.track_dice_ignore_fn = True
        self.best_dice_ignore_fn = 0.0
        self.best_dice_ignore_fn_epoch = 0
        self.best_dice_ignore_fn_model_path = None
        self.dice_ignore_fn_history = []

        # 统计
        self.train_losses = []
        self.val_losses = []
        self.precision_history = []
        self.recall_history = []
        self.alpha_history = []

        self._print_config()

    def _print_config(self):
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("MedNeXt Large Transfer Learning Trainer V2")
        self.print_to_log_file("=" * 60)
        self.print_to_log_file(f"Epochs: {self.num_epochs}, Freeze: {self.freeze_epochs}")
        self.print_to_log_file(f"LR: {self.initial_lr}, Encoder LR factor: {self.encoder_lr_factor}")
        self.print_to_log_file(f"Tversky Alpha: {self.tversky_alpha}, Beta: {self.tversky_beta}")
        self.print_to_log_file(f"Adaptive adjust interval: {self.adaptive_adjust_interval}")
        self.print_to_log_file("=" * 60)

    def configure_loss_function(self):
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

    def configure_optimizers(self):
        encoder_params = []
        decoder_params = []

        for name, param in self.network.named_parameters():
            if param.requires_grad:
                if 'encoder' in name.lower():
                    encoder_params.append(param)
                else:
                    decoder_params.append(param)

        param_groups = [
            {'params': encoder_params, 'lr': self.initial_lr * self.encoder_lr_factor},
            {'params': decoder_params, 'lr': self.initial_lr}
        ]

        optimizer = AdamW(param_groups, weight_decay=self.weight_decay, eps=1e-5)

        # 使用更稳定的CosineAnnealingLR替代WarmRestarts
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        total = sum(p.numel() for p in self.network.parameters())
        trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.print_to_log_file(f"Params: {trainable}/{total} trainable ({trainable/total*100:.1f}%)")

        return optimizer, scheduler

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

        if layers == 'all':
            for name, param in self.network.named_parameters():
                if 'encoder' in name.lower():
                    param.requires_grad = True
                    self.unfrozen_layers.add(name)
        else:
            for pattern in layers:
                for name, param in self.network.named_parameters():
                    if pattern in name and not param.requires_grad:
                        param.requires_grad = True
                        self.unfrozen_layers.add(name)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        return True

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        if self.current_epoch >= self.freeze_epochs:
            self._progressive_unfreeze()

        encoder_lr = self.optimizer.param_groups[0]['lr']
        decoder_lr = self.optimizer.param_groups[1]['lr']

        status = "frozen" if self.current_epoch < self.freeze_epochs else \
                 "unfreezing" if len(self.unfrozen_layers) == 0 else \
                 "partial" if len(self.unfrozen_layers) < 15 else "full"

        self.print_to_log_file(f"Epoch {self.current_epoch}: encoder_lr={encoder_lr:.2e}, decoder_lr={decoder_lr:.2e}, status={status}")

    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True):
            output = self.network(data)
            l = self.loss(output, target)

        if torch.isnan(l) or torch.isinf(l):
            self.print_to_log_file(f"[WARNING] NaN/Inf loss at epoch {self.current_epoch}")
            return {'loss': np.array([1.0])}

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

        return {'loss': loss_val}

    def _adapt_loss_parameters(self):
        if not self.enable_adaptive_loss:
            return False

        if len(self.dice_ignore_fn_history) < self.dice_ignore_fn_ma_window:
            return False

        current = self.dice_ignore_fn_history[-1]
        ma = np.mean(self.dice_ignore_fn_history[-self.dice_ignore_fn_ma_window:])

        # 检查训练是否稳定
        if self.train_losses and self.train_losses[-1] > 0.3:
            self.print_to_log_file(f"[ADAPT] Skipping: unstable loss {self.train_losses[-1]:.4f}")
            return False

        old_alpha = self.tversky_alpha
        deviation = self.dice_ignore_fn_target - ma

        # 保守调整
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
            self.loss = self.configure_loss_function()
            return True

        return False

    def on_validation_epoch_end(self, val_outputs=None):
        if val_outputs and len(val_outputs) > 0:
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import collate_outputs
            outputs = collate_outputs(val_outputs)
            tp = np.sum(outputs['tp_hard'], 0)
            fp = np.sum(outputs['fp_hard'], 0)
            fn = np.sum(outputs['fn_hard'], 0)

            std_dice = [2*i/(2*i+j+k) if (2*i+j+k) > 0 else 0 for i,j,k in zip(tp,fp,fn)]
            dice_ignore_fn = [2*i/(2*i+j) if (2*i+j) > 0 else 0 for i,j in zip(tp,fp)]
            precision = [i/(i+j) if (i+j) > 0 else 0 for i,j in zip(tp,fp)]
            recall = [i/(i+k) if (i+k) > 0 else 0 for i,k in zip(tp,fn)]

            self.current_val_metrics = {
                'std_dice': np.nanmean(std_dice),
                'dice_ignore_fn': np.nanmean(dice_ignore_fn),
                'precision': np.nanmean(precision),
                'recall': np.nanmean(recall)
            }

            self.precision_history.append(self.current_val_metrics['precision'])
            self.recall_history.append(self.current_val_metrics['recall'])
            self.dice_ignore_fn_history.append(self.current_val_metrics['dice_ignore_fn'])

        super().on_validation_epoch_end(val_outputs)

        if hasattr(self, 'current_val_metrics'):
            m = self.current_val_metrics
            self.print_to_log_file(f"Val - StdDice: {m['std_dice']:.4f}, IgnoreFNDice: {m['dice_ignore_fn']:.4f}, "
                                   f"Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")

            # 保存最佳ignore FN dice模型
            if self.track_dice_ignore_fn and m['dice_ignore_fn'] > self.best_dice_ignore_fn:
                self.best_dice_ignore_fn = m['dice_ignore_fn']
                self.best_dice_ignore_fn_epoch = self.current_epoch
                self._save_best_dice_ignore_fn_model()

            # 更新plateau计数器
            if m['dice_ignore_fn'] > self.best_dice_for_unfreeze + 0.001:
                self.best_dice_for_unfreeze = m['dice_ignore_fn']
                self.epochs_without_dice_improvement = 0
            else:
                self.epochs_without_dice_improvement += 1

            # 自适应调整
            if self.enable_adaptive_loss and self.current_epoch > 0 and \
               self.current_epoch % self.adaptive_adjust_interval == 0:
                self._adapt_loss_parameters()

    def _save_best_dice_ignore_fn_model(self):
        try:
            if hasattr(self, 'output_folder'):
                path = os.path.join(self.output_folder, f"best_dice_ignore_fn_epoch_{self.current_epoch}.pth")
                checkpoint = {
                    'epoch': self.current_epoch,
                    'model_state_dict': self.network.state_dict(),
                    'best_dice_ignore_fn': self.best_dice_ignore_fn,
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

    def on_epoch_end(self):
        continue_training = super().on_epoch_end()

        if hasattr(self, 'best_val_eval_criterion_MA') and self.current_epoch > self.freeze_epochs + 5:
            no_improve = self.current_epoch - self.best_val_eval_criterion_MA_epoch
            if no_improve > self.patience:
                self.print_to_log_file(f"Early stopping at epoch {self.current_epoch}")
                return False

        return continue_training

    def run_training(self):
        self.print_to_log_file("Starting training with V2 trainer (Large model)")
        start = time.time()

        try:
            result = super().run_training()
            elapsed = time.time() - start
            self.print_to_log_file(f"Training completed in {elapsed/60:.1f} minutes")
            self._print_summary()
            return result
        except Exception as e:
            self.print_to_log_file(f"[ERROR] Training failed: {e}")
            raise

    def _print_summary(self):
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("Training Summary (Large Model)")
        self.print_to_log_file("=" * 60)
        if self.train_losses:
            self.print_to_log_file(f"Train loss: {self.train_losses[0]:.4f} -> {self.train_losses[-1]:.4f}")
        self.print_to_log_file(f"Best ignore-FN dice: {self.best_dice_ignore_fn:.4f} (epoch {self.best_dice_ignore_fn_epoch})")
        if self.alpha_history:
            self.print_to_log_file(f"Alpha adjustments: {len(self.alpha_history)} times")
            self.print_to_log_file(f"Final alpha/beta: {self.tversky_alpha:.3f}/{self.tversky_beta:.3f}")
        self.print_to_log_file("=" * 60)
