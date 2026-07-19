from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMedNext import (
    nnUNetTrainerMedNext,
    nnUNetTrainerV2_MedNeXt_B_kernel5,
    create_mednextv1_base
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import nn, autocast
from torch.cuda.amp import GradScaler
import numpy as np
import time
from datetime import datetime, timedelta
from collections import deque
import os
import shutil


class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin: callable = None, batch_dice: bool = False, 
                 do_bg: bool = True, smooth: float = 1e-5, alpha: float = 0.3, beta: float = 0.7):
        super(TverskyLoss, self).__init__()
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
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        
        tp = torch.clamp(tp, min=0)
        fp = torch.clamp(fp, min=0)
        fn = torch.clamp(fn, min=0)
        
        nominator = tp
        denominator = tp + self.alpha * fp + self.beta * fn
        
        tversky = (nominator + self.smooth) / (denominator + self.smooth + self.eps)
        tversky = torch.clamp(tversky, self.eps, 1.0 - self.eps)
        
        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        
        if torch.isnan(tversky).any():
            tversky = torch.where(torch.isnan(tversky), 
                                 torch.ones_like(tversky) * self.eps, 
                                 tversky)
        
        return -tversky.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, apply_nonlin: callable = None, batch_dice: bool = False,
                 do_bg: bool = True, smooth: float = 1e-5, alpha: float = 0.3, 
                 beta: float = 0.7, gamma: float = 1.33):
        super(FocalTverskyLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = 1e-7
    
    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        
        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        
        tp = torch.clamp(tp, min=0)
        fp = torch.clamp(fp, min=0)
        fn = torch.clamp(fn, min=0)
        
        nominator = tp
        denominator = tp + self.alpha * fp + self.beta * fn
        
        tversky = (nominator + self.smooth) / (denominator + self.smooth + self.eps)
        tversky = torch.clamp(tversky, self.eps, 1.0 - self.eps)
        
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)
        
        if not self.do_bg:
            if self.batch_dice:
                focal_tversky = focal_tversky[1:]
            else:
                focal_tversky = focal_tversky[:, 1:]
        
        if torch.isnan(focal_tversky).any():
            focal_tversky = torch.where(torch.isnan(focal_tversky), 
                                       torch.ones_like(focal_tversky), 
                                       focal_tversky)
        
        return focal_tversky.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.4, tversky_weight: float = 0.6,
                 apply_nonlin: callable = None, batch_dice: bool = False,
                 do_bg: bool = True, smooth: float = 1., alpha: float = 0.3, beta: float = 0.7):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        
        dice_nominator = 2 * tp
        dice_denominator = 2 * tp + fp + fn
        dice = (dice_nominator + self.smooth) / (dice_denominator + self.smooth)
        
        tversky_nominator = tp
        tversky_denominator = tp + self.alpha * fp + self.beta * fn
        tversky = (tversky_nominator + self.smooth) / (tversky_denominator + self.smooth)
        
        if not self.do_bg:
            if self.batch_dice:
                dice = dice[1:]
                tversky = tversky[1:]
            else:
                dice = dice[:, 1:]
                tversky = tversky[:, 1:]
        
        dice_loss = -dice.mean()
        tversky_loss = -tversky.mean()
        
        return self.dice_weight * dice_loss + self.tversky_weight * tversky_loss


class nnUNetTrainerV2_MedNeXt_B_TransferLearning(nnUNetTrainerV2_MedNeXt_B_kernel5):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        self.num_epochs = 100
        self.num_iterations_per_epoch = 100
        
        self.initial_lr = 1e-4
        self.weight_decay = 1e-4
        
        self.patience = 20
        self.validate_every = 2
        
        self.freeze_epochs = 10
        self.encoder_lr_factor = 0.01
        
        self.enable_progressive_unfreezing = True
        self.progressive_unfreeze_schedule = {
            10: ['encoder.stages.3'],
            20: ['encoder.stages.2', 'encoder.stages.3'],
            35: ['encoder.stages.1', 'encoder.stages.2', 'encoder.stages.3'],
            50: 'all'
        }
        self.unfrozen_layers = set()
        
        self.unfreeze_recall_threshold = 0.35
        self.unfreeze_loss_stable_epochs = 3
        self.loss_plateau_history = deque(maxlen=10)
        
        self.enable_patience_trigger = True
        self.dice_plateau_patience = 5
        self.max_wait_epochs = 30
        self.dice_improvement_history = []
        self.best_dice_for_unfreeze = 0.0
        self.epochs_without_dice_improvement = 0
        
        self.track_dice_ignore_fn = True
        self.best_dice_ignore_fn = 0.0
        self.best_dice_ignore_fn_epoch = 0
        self.dice_ignore_fn_history = []
        self.save_best_dice_ignore_fn = True
        self.best_dice_ignore_fn_model_path = None
        
        self.tversky_alpha = 0.05
        self.tversky_beta = 0.95
        self.alpha_beta_sum = 1.0
        
        self.enable_adaptive_loss = True
        self.adaptive_adjust_interval = 3
        
        self.dice_ignore_fn_target = 0.90
        self.dice_ignore_fn_ma_window = 3
        
        self.smooth_adjust_k = 0.015
        
        self.alpha_min = 0.02
        self.alpha_max = 0.15
        
        self.dice_ignore_fn_ma_history = deque(maxlen=10)
        
        self.enable_gradient_monitoring = True
        self.gradient_monitor_interval = 10
        self.fg_gradient_threshold = 0.1
        self.fg_gradient_boost = 2.0
        self.gradient_norm_history = deque(maxlen=100)
        self.fg_gradient_ratio_history = deque(maxlen=50)
        
        self.enable_multi_objective = True
        self.optimize_dice_ignore_fn_only = True
        self.multi_objective_weights = {
            'dice_ignore_fn': 1.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        self.mo_adjustment_strategy = 'dice_ignore_fn_focused'
        
        self.enable_spatial_augmentation = True
        self.elastic_deformation = True
        self.elastic_sigma_range = (5, 15)
        self.elastic_alpha_range = (50, 150)
        self.random_gamma = True
        self.gamma_range = (0.7, 1.5)
        self.additive_noise = True
        self.noise_variance = (0.0, 0.05)
        self.simulate_layer_structure = True
        self.layer_direction_range = (-15, 15)
        
        self.epoch_start_time = None
        self.total_train_time = 0
        self.best_dice = 0.0
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.precision_history = []
        self.recall_history = []
        self.precision_ma_history = []
        self.recall_ma_history = []
        self.alpha_history = []
        self.beta_history = []
        self.adjustment_reasons = []
        
        self._print_initialization_log()

    def _print_initialization_log(self):
        self.print_to_log_file("\n" + "=" * 70)
        self.print_to_log_file("Transfer Learning Trainer Initialization")
        self.print_to_log_file("=" * 70)
        
        self.print_to_log_file("\nDataset Config:")
        self.print_to_log_file(f"  Dataset name: {self.dataset_json.get('name', 'Unknown')}")
        self.print_to_log_file(f"  Training samples: {len(self.dataset_json.get('training', []))}")
        self.print_to_log_file(f"  Test samples: {len(self.dataset_json.get('test', []))}")
        self.print_to_log_file(f"  Num classes: {self.label_manager.num_segmentation_heads}")
        
        self.print_to_log_file("\nTraining Config:")
        self.print_to_log_file(f"  Total epochs: {self.num_epochs}")
        self.print_to_log_file(f"  Iterations per epoch: {self.num_iterations_per_epoch}")
        self.print_to_log_file(f"  Validate every: {self.validate_every} epochs")
        
        self.print_to_log_file("\nOptimizer Config:")
        self.print_to_log_file(f"  Initial LR: {self.initial_lr:.2e}")
        self.print_to_log_file(f"  Weight decay: {self.weight_decay:.2e}")
        self.print_to_log_file(f"  Optimizer: AdamW")
        self.print_to_log_file(f"  Scheduler: CosineAnnealingWarmRestarts")
        
        self.print_to_log_file("\nTransfer Learning Config:")
        self.print_to_log_file(f"  Freeze epochs: {self.freeze_epochs}")
        self.print_to_log_file(f"  Encoder LR factor: {self.encoder_lr_factor}")
        self.print_to_log_file(f"  Early stop patience: {self.patience}")
        
        self.print_to_log_file("\nProgressive Unfreeze Schedule:")
        for epoch, layers in self.progressive_unfreeze_schedule.items():
            if layers == 'all':
                self.print_to_log_file(f"    Epoch {epoch}: Unfreeze all")
            else:
                self.print_to_log_file(f"    Epoch {epoch}: {', '.join(layers)}")
        
        self.print_to_log_file("\nTversky Loss Config:")
        self.print_to_log_file(f"  Alpha (FP weight): {self.tversky_alpha}")
        self.print_to_log_file(f"  Beta (FN weight): {self.tversky_beta}")
        
        self.print_to_log_file("\nAdaptive Config:")
        self.print_to_log_file(f"  Enable: {self.enable_adaptive_loss}")
        self.print_to_log_file(f"  Adjust interval: {self.adaptive_adjust_interval}")
        self.print_to_log_file(f"  Target dice_ignore_fn: {self.dice_ignore_fn_target:.2f}")
        
        self.print_to_log_file("\nSpatial Augmentation Config:")
        self.print_to_log_file(f"  Enable: {self.enable_spatial_augmentation}")
        self.print_to_log_file(f"  Elastic deformation: {self.elastic_deformation}")
        self.print_to_log_file(f"  Random gamma: {self.random_gamma}")
        self.print_to_log_file(f"  Additive noise: {self.additive_noise}")
        
        self.print_to_log_file("\nHardware Config:")
        self.print_to_log_file(f"  Device: {self.device}")
        if self.device.type == 'cuda':
            self.print_to_log_file(f"  GPU: {torch.cuda.get_device_name(0)}")
            self.print_to_log_file(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        self.print_to_log_file("\n" + "=" * 70)

    def configure_loss_function(self):
        from nnunetv2.utilities.helpers import softmax_helper_dim1
        
        self.print_to_log_file(f"\nUsing Tversky Loss (Alpha={self.tversky_alpha}, Beta={self.tversky_beta})")
        
        loss = TverskyLoss(
            apply_nonlin=softmax_helper_dim1,
            batch_dice=True,
            do_bg=False,
            smooth=1e-5,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta
        )
        
        return loss

    def configure_optimizers(self):
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        self.print_to_log_file(f"\nParam stats:")
        self.print_to_log_file(f"  Total: {total_params:,}")
        self.print_to_log_file(f"  Trainable: {trainable_params:,}")
        self.print_to_log_file(f"  Frozen: {total_params - trainable_params:,}")
        
        encoder_params = []
        decoder_params = []
        
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                if 'encoder' in name.lower() or 'down' in name.lower():
                    encoder_params.append(param)
                else:
                    decoder_params.append(param)
        
        param_groups = [
            {'params': encoder_params, 'lr': self.initial_lr * self.encoder_lr_factor, 'name': 'encoder'},
            {'params': decoder_params, 'lr': self.initial_lr, 'name': 'decoder'}
        ]
        
        optimizer = AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999)
        )
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.print_to_log_file(f"\nOptimizer config done:")
        self.print_to_log_file(f"  Encoder LR: {self.initial_lr * self.encoder_lr_factor:.2e}")
        self.print_to_log_file(f"  Decoder LR: {self.initial_lr:.2e}")
        
        return optimizer, scheduler

    def _check_loss_plateau(self):
        if len(self.loss_plateau_history) < self.unfreeze_loss_stable_epochs + 1:
            return False
        
        recent_losses = list(self.loss_plateau_history)[-self.unfreeze_loss_stable_epochs:]
        loss_changes = [abs(recent_losses[i] - recent_losses[i-1]) / (abs(recent_losses[i-1]) + 1e-8) 
                        for i in range(1, len(recent_losses))]
        return all(change < 0.10 for change in loss_changes)
    
    def _check_dice_plateau(self):
        if not self.enable_patience_trigger:
            return False
        
        if self.current_epoch >= self.max_wait_epochs:
            self.print_to_log_file(f"   Max wait epochs reached: {self.max_wait_epochs}, forcing unfreeze")
            return True
        
        if self.epochs_without_dice_improvement >= self.dice_plateau_patience:
            self.print_to_log_file(f"   Dice plateau for {self.epochs_without_dice_improvement} epochs, triggering unfreeze")
            return True
        
        return False
    
    def _update_dice_improvement_status(self, current_dice):
        if current_dice > self.best_dice_for_unfreeze + 0.001:
            self.best_dice_for_unfreeze = current_dice
            self.epochs_without_dice_improvement = 0
            return True
        else:
            self.epochs_without_dice_improvement += 1
            return False
    
    def _progressive_unfreeze(self):
        if not self.enable_progressive_unfreezing:
            return False
        
        current_recall = self.recall_history[-1] if self.recall_history else 0
        if current_recall < self.unfreeze_recall_threshold:
            self.print_to_log_file(f"   Recall {current_recall:.3f} < {self.unfreeze_recall_threshold:.3f}, skipping unfreeze")
            return False
        
        if self.current_epoch >= self.max_wait_epochs and self.current_epoch not in self.progressive_unfreeze_schedule:
            if self.current_epoch >= 50:
                self.progressive_unfreeze_schedule[self.current_epoch] = 'all'
            elif self.current_epoch >= 35:
                self.progressive_unfreeze_schedule[self.current_epoch] = ['encoder.stages.1', 'encoder.stages.2', 'encoder.stages.3']
            elif self.current_epoch >= 20:
                self.progressive_unfreeze_schedule[self.current_epoch] = ['encoder.stages.2', 'encoder.stages.3']
            else:
                self.progressive_unfreeze_schedule[self.current_epoch] = ['encoder.stages.3']
            self.print_to_log_file(f"   Force adding unfreeze schedule at epoch {self.current_epoch}")
        
        if self.current_epoch not in self.progressive_unfreeze_schedule:
            return False
        
        loss_stable = self._check_loss_plateau()
        dice_plateau = self._check_dice_plateau()
        
        if not (loss_stable or dice_plateau):
            self.print_to_log_file(f"   Conditions not met: Loss stable={loss_stable}, Dice plateau={dice_plateau}")
            return False
        
        layers_to_unfreeze = self.progressive_unfreeze_schedule[self.current_epoch]
        
        trigger_reason = []
        if loss_stable:
            trigger_reason.append("Loss stable")
        if dice_plateau:
            trigger_reason.append("Dice plateau")
        if self.current_epoch >= self.max_wait_epochs:
            trigger_reason.append("Max wait epochs")
        
        self.print_to_log_file(f"\nEpoch {self.current_epoch}: Progressive Unfreeze")
        self.print_to_log_file(f"   Recall: {current_recall:.3f} >= {self.unfreeze_recall_threshold:.3f}")
        self.print_to_log_file(f"   Trigger: {' + '.join(trigger_reason)}")
        
        if layers_to_unfreeze == 'all':
            for name, param in self.network.named_parameters():
                if 'encoder' in name.lower():
                    param.requires_grad = True
                    self.unfrozen_layers.add(name)
            self.print_to_log_file(f"   Unfrozen all encoder layers")
        else:
            for layer_pattern in layers_to_unfreeze:
                for name, param in self.network.named_parameters():
                    if layer_pattern in name and not param.requires_grad:
                        param.requires_grad = True
                        self.unfrozen_layers.add(name)
                        self.print_to_log_file(f"   Unfrozen: {name}")
        
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        
        return True
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        
        self.epoch_start_time = time.time()
        
        if not hasattr(self, 'network'):
            return
        
        if self.current_epoch >= self.freeze_epochs:
            self._progressive_unfreeze()
        
        current_lr = self.optimizer.param_groups[0]['lr']
        encoder_lr = self.optimizer.param_groups[0]['lr'] if len(self.optimizer.param_groups) > 0 else current_lr
        decoder_lr = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else current_lr
        
        self.learning_rates.append(current_lr)
        
        self.print_to_log_file(f"\nEpoch {self.current_epoch}/{self.num_epochs} start")
        self.print_to_log_file(f"   Encoder LR: {encoder_lr:.2e}")
        self.print_to_log_file(f"   Decoder LR: {decoder_lr:.2e}")
        
        if self.current_epoch < self.freeze_epochs:
            self.print_to_log_file(f"   Status: Encoder frozen, training decoder only")
        elif len(self.unfrozen_layers) == 0:
            self.print_to_log_file(f"   Status: Waiting for unfreeze conditions")
        elif len(self.unfrozen_layers) < 10:
            self.print_to_log_file(f"   Status: Progressive unfreezing ({len(self.unfrozen_layers)} layers)")
        else:
            self.print_to_log_file(f"   Status: Full network fine-tuning")

    def _apply_spatial_augmentation(self, data, target):
        if not self.enable_spatial_augmentation:
            return data, target
        
        data = torch.clamp(data, 0.0, 1.0)
        
        if self.random_gamma and np.random.random() < 0.5:
            gamma = np.random.uniform(*self.gamma_range)
            gamma = np.clip(gamma, 0.5, 2.0)
            data = torch.pow(data + 1e-6, gamma)
            data = torch.clamp(data, 0.0, 10.0)
        
        if self.additive_noise and np.random.random() < 0.3:
            noise_variance = np.random.uniform(*self.noise_variance)
            noise_variance = min(noise_variance, 0.1)
            noise = torch.randn_like(data) * np.sqrt(noise_variance)
            data = data + noise
            data = torch.clamp(data, -1.0, 2.0)
        
        if torch.isnan(data).any() or torch.isinf(data).any():
            self.print_to_log_file(f"[WARNING] NaN/Inf after augmentation, skipping")
            return batch['data'].to(self.device, non_blocking=True), target
        
        return data, target
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        data, target = self._apply_spatial_augmentation(data, target)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)
        
        if torch.isnan(l) or torch.isinf(l):
            with torch.no_grad():
                output_min = output.min().item() if not torch.isnan(output).any() else 'nan'
                output_max = output.max().item() if not torch.isnan(output).any() else 'nan'
                output_mean = output.mean().item() if not torch.isnan(output).any() else 'nan'
                target_sum = target.sum().item() if not torch.isnan(target).any() else 'nan'
                
                self.print_to_log_file(f"[WARNING] Epoch {self.current_epoch}: NaN/Inf loss detected")
                self.print_to_log_file(f"  Output range: [{output_min}, {output_max}], mean: {output_mean}, target sum: {target_sum}")
                
                has_nan_param = any(torch.isnan(p).any() for p in self.network.parameters() if p.requires_grad)
                if has_nan_param:
                    self.print_to_log_file(f"  [CRITICAL] NaN in network parameters!")
            
            return {'loss': np.array([1.0])}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            
            if self.enable_gradient_monitoring and self.current_epoch < 30:
                self._monitor_and_boost_gradient(target)
            
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            
            if self.enable_gradient_monitoring and self.current_epoch < 30:
                self._monitor_and_boost_gradient(target)
            
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        loss_value = l.detach().cpu().numpy()
        
        if not hasattr(self, 'nan_loss_count'):
            self.nan_loss_count = 0
        if np.isnan(loss_value) or np.isinf(loss_value):
            self.nan_loss_count += 1
            self.print_to_log_file(f"[WARNING] Epoch {self.current_epoch}: loss={loss_value}, nan count={self.nan_loss_count}")
            if self.nan_loss_count >= 5:
                self.print_to_log_file(f"[ALERT] Consecutive NaN losses, reducing LR by 50%")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                self.nan_loss_count = 0
            return {'loss': np.array([1.0])}
        else:
            self.nan_loss_count = 0
        
        if not hasattr(self, 'epoch_losses'):
            self.epoch_losses = []
        self.epoch_losses.append(loss_value)
        
        self.loss_plateau_history.append(float(loss_value))

        return {'loss': loss_value}
    
    def _monitor_and_boost_gradient(self, target):
        try:
            total_fg_grad_norm = 0.0
            total_bg_grad_norm = 0.0
            fg_param_count = 0
            
            for name, param in self.network.named_parameters():
                if param.grad is not None and 'decoder' in name.lower():
                    grad_norm = param.grad.norm().item()
                    self.gradient_norm_history.append(grad_norm)
                    
                    if 'seg' in name.lower() or 'out' in name.lower():
                        total_fg_grad_norm += grad_norm
                        fg_param_count += 1
                    else:
                        total_bg_grad_norm += grad_norm
            
            if fg_param_count > 0 and total_bg_grad_norm > 0:
                avg_fg_grad = total_fg_grad_norm / fg_param_count
                avg_bg_grad = total_bg_grad_norm / (len(list(self.network.parameters())) - fg_param_count + 1e-8)
                
                fg_ratio = avg_fg_grad / (avg_fg_grad + avg_bg_grad + 1e-8)
                self.fg_gradient_ratio_history.append(fg_ratio)
                
                if fg_ratio < self.fg_gradient_threshold:
                    boost_factor = min(self.fg_gradient_boost, self.fg_gradient_threshold / (fg_ratio + 1e-8))
                    
                    for name, param in self.network.named_parameters():
                        if param.grad is not None and ('seg' in name.lower() or 'out' in name.lower()):
                            param.grad *= boost_factor
                    
                    if len(self.fg_gradient_ratio_history) % 20 == 0:
                        self.print_to_log_file(f"   Gradient boost: fg ratio {fg_ratio:.1%} < {self.fg_gradient_threshold:.1%}, boost {boost_factor:.1f}x")
        
        except Exception as e:
            pass

    def on_train_epoch_end(self, train_outputs=None):
        super().on_train_epoch_end(train_outputs)
        
        if hasattr(self, 'epoch_losses') and len(self.epoch_losses) > 0:
            avg_loss = np.mean(self.epoch_losses)
            std_loss = np.std(self.epoch_losses)
            min_loss = np.min(self.epoch_losses)
            max_loss = np.max(self.epoch_losses)
            
            self.train_losses.append(avg_loss)
            
            epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
            self.total_train_time += epoch_time
            
            remaining_epochs = self.num_epochs - self.current_epoch - 1
            estimated_remaining_time = remaining_epochs * (self.total_train_time / (self.current_epoch + 1))
            
            self.print_to_log_file(f"\nEpoch {self.current_epoch} training stats:")
            self.print_to_log_file(f"   Avg loss: {avg_loss:.6f} +/- {std_loss:.6f}")
            self.print_to_log_file(f"   Loss range: [{min_loss:.6f}, {max_loss:.6f}]")
            self.print_to_log_file(f"   Epoch time: {timedelta(seconds=int(epoch_time))}")
            self.print_to_log_file(f"   Total time: {timedelta(seconds=int(self.total_train_time))}")
            self.print_to_log_file(f"   ETA: {timedelta(seconds=int(estimated_remaining_time))}")
            
            self.epoch_losses = []

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.print_to_log_file(f"\nValidation Epoch {self.current_epoch} start")
        self.val_start_time = time.time()

    def _get_current_phase_targets(self, epoch):
        for phase, config in self.phase_thresholds.items():
            if epoch <= config['epoch_end']:
                return config['precision_target'], config['recall_target'], phase
        last_phase = list(self.phase_thresholds.keys())[-1]
        config = self.phase_thresholds[last_phase]
        return config['precision_target'], config['recall_target'], last_phase
    
    def _calculate_moving_average(self, values, window):
        if len(values) < window:
            return np.mean(values) if values else 0.0
        return np.mean(values[-window:])
    
    def _adapt_loss_parameters(self, precision, recall, dice=None):
        if not self.enable_adaptive_loss:
            return False
        
        if len(self.dice_ignore_fn_history) == 0:
            return False
        
        current_dice_ignore_fn = self.dice_ignore_fn_history[-1]
        
        dice_ignore_fn_ma = self._calculate_moving_average(
            list(self.dice_ignore_fn_history), 
            self.dice_ignore_fn_ma_window
        )
        self.dice_ignore_fn_ma_history.append(dice_ignore_fn_ma)
        
        old_alpha = self.tversky_alpha
        old_beta = self.tversky_beta
        
        dice_ignore_fn_deviation = self.dice_ignore_fn_target - dice_ignore_fn_ma
        
        self.print_to_log_file(f"   Dice ignore FN optimization")
        self.print_to_log_file(f"   Current: {current_dice_ignore_fn:.4f}, MA: {dice_ignore_fn_ma:.4f}, Target: {self.dice_ignore_fn_target:.4f}")
        
        if hasattr(self, 'train_losses') and len(self.train_losses) > 0:
            recent_train_loss = self.train_losses[-1]
            if recent_train_loss > 0.5:
                self.print_to_log_file(f"   Unstable loss ({recent_train_loss:.4f}), skipping adaptive adjust")
                return False
        
        adjusted = False
        adjustment_reason = ""
        net_alpha_change = 0.0
        
        if len(self.dice_ignore_fn_history) >= 2:
            recent_trend = self.dice_ignore_fn_history[-1] - self.dice_ignore_fn_history[-2]
            
            if recent_trend < -0.02:
                adjust_factor = 1.0
                
                if old_alpha < 0.08:
                    net_alpha_change = self.smooth_adjust_k * adjust_factor
                    adjustment_reason = f"Dice ignore FN dropping ({recent_trend:+.4f}), increasing alpha"
                elif old_alpha > 0.12:
                    net_alpha_change = -self.smooth_adjust_k * adjust_factor
                    adjustment_reason = f"Dice ignore FN dropping ({recent_trend:+.4f}), decreasing alpha"
                else:
                    net_alpha_change = self.smooth_adjust_k * 0.5 if recent_trend < -0.05 else -self.smooth_adjust_k * 0.5
                    adjustment_reason = f"Dice ignore FN dropping ({recent_trend:+.4f}), fine-tuning alpha"
                    
            elif dice_ignore_fn_deviation > 0.1:
                if old_alpha < 0.08:
                    net_alpha_change = self.smooth_adjust_k * 0.5
                    adjustment_reason = f"Dice ignore FN below target (gap {dice_ignore_fn_deviation:.4f}), increasing alpha"
                elif old_alpha > 0.12:
                    net_alpha_change = -self.smooth_adjust_k * 0.5
                    adjustment_reason = f"Dice ignore FN below target (gap {dice_ignore_fn_deviation:.4f}), decreasing alpha"
                else:
                    self.print_to_log_file(f"   Alpha in good range, keeping current params (alpha={self.tversky_alpha:.3f})")
                    return False
            else:
                if abs(recent_trend) < 0.01:
                    self.print_to_log_file(f"   Dice ignore FN stable, no adjustment needed (alpha={self.tversky_alpha:.3f}, beta={self.tversky_beta:.3f})")
                    return False
                else:
                    net_alpha_change = self.smooth_adjust_k * 0.3 if recent_trend > 0 else -self.smooth_adjust_k * 0.3
                    adjustment_reason = "Dice ignore FN near target, conservative adjustment"
        else:
            if dice_ignore_fn_deviation > 0.15:
                net_alpha_change = self.smooth_adjust_k * 0.5
                adjustment_reason = "Dice ignore FN initial adjustment (conservative)"
            else:
                self.print_to_log_file(f"   Insufficient data, keeping current params (alpha={self.tversky_alpha:.3f}, beta={self.tversky_beta:.3f})")
                return False
        
        new_alpha = self.tversky_alpha + net_alpha_change
        new_alpha = np.clip(new_alpha, self.alpha_min, self.alpha_max)
        new_beta = self.alpha_beta_sum - new_alpha
        
        if abs(new_alpha - old_alpha) > 0.005:
            self.tversky_alpha = new_alpha
            self.tversky_beta = new_beta
            adjusted = True
            
            self.print_to_log_file(f"   {adjustment_reason}")
            self.print_to_log_file(f"   Alpha: {old_alpha:.3f} -> {self.tversky_alpha:.3f}")
            self.print_to_log_file(f"   Beta: {old_beta:.3f} -> {self.tversky_beta:.3f}")
            
            self.loss = self.configure_loss_function()
            
            self.alpha_history.append(self.tversky_alpha)
            self.beta_history.append(self.tversky_beta)
            self.adjustment_reasons.append(adjustment_reason)
            
            return True
        else:
            self.print_to_log_file(f"   Change too small, keeping current params (alpha={self.tversky_alpha:.3f}, beta={self.tversky_beta:.3f})")
        
        return False

    def on_validation_epoch_end(self, val_outputs=None):
        if val_outputs is not None and len(val_outputs) > 0:
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import collate_outputs
            outputs_collated = collate_outputs(val_outputs)
            tp = np.sum(outputs_collated['tp_hard'], 0)
            fp = np.sum(outputs_collated['fp_hard'], 0)
            fn = np.sum(outputs_collated['fn_hard'], 0)
            
            global_dc_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0 
                                   for i, j, k in zip(tp, fp, fn)]
            
            dice_ignore_fn_per_class = [2 * i / (2 * i + j) if (2 * i + j) > 0 else 0 
                                        for i, j in zip(tp, fp)]
            
            precision_per_class = [i / (i + j) if (i + j) > 0 else 0 
                                   for i, j in zip(tp, fp)]
            
            recall_per_class = [i / (i + k) if (i + k) > 0 else 0 
                                for i, k in zip(tp, fn)]
            
            iou_per_class = [i / (i + j + k) if (i + j + k) > 0 else 0 
                             for i, j, k in zip(tp, fp, fn)]
            
            self.current_val_metrics = {
                'standard_dice': global_dc_per_class,
                'dice_ignore_fn': dice_ignore_fn_per_class,
                'precision': precision_per_class,
                'recall': recall_per_class,
                'iou': iou_per_class,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            precision_mean = np.nanmean(precision_per_class)
            recall_mean = np.nanmean(recall_per_class)
            self.precision_history.append(precision_mean)
            self.recall_history.append(recall_mean)
        
        super().on_validation_epoch_end(val_outputs)
        
        if hasattr(self, 'current_val_metrics') and 'iou' in self.current_val_metrics:
            iou_mean = np.nanmean(self.current_val_metrics['iou'])
            self.logger.log('iou', iou_mean, self.current_epoch)
        
        val_time = time.time() - self.val_start_time if hasattr(self, 'val_start_time') else 0
        
        if hasattr(self, 'val_loss'):
            self.val_losses.append(self.val_loss)
            self.print_to_log_file(f"\nValidation complete Epoch {self.current_epoch}:")
            self.print_to_log_file(f"   Val loss: {self.val_loss:.6f}")
            self.print_to_log_file(f"   Val time: {timedelta(seconds=int(val_time))}")
        
        if hasattr(self, 'current_val_metrics'):
            metrics = self.current_val_metrics
            
            self.print_to_log_file(f"\nDice metrics:")
            
            std_dice_mean = np.nanmean(metrics['standard_dice'])
            self.print_to_log_file(f"   Standard Dice: {std_dice_mean:.4f}")
            
            dice_ignore_fn_mean = np.nanmean(metrics['dice_ignore_fn'])
            self.dice_ignore_fn_history.append(dice_ignore_fn_mean)
            self.print_to_log_file(f"   Dice ignore FN: {dice_ignore_fn_mean:.4f}")
            
            if self.track_dice_ignore_fn and dice_ignore_fn_mean > self.best_dice_ignore_fn:
                self.best_dice_ignore_fn = dice_ignore_fn_mean
                self.best_dice_ignore_fn_epoch = self.current_epoch
                self.print_to_log_file(f"   New best dice ignore FN: {self.best_dice_ignore_fn:.4f}")
                
                if self.save_best_dice_ignore_fn:
                    self._save_best_dice_ignore_fn_model()
            else:
                self.print_to_log_file(f"   Best dice ignore FN: {self.best_dice_ignore_fn:.4f} (Epoch {self.best_dice_ignore_fn_epoch})")
            
            precision_mean = np.nanmean(metrics['precision'])
            recall_mean = np.nanmean(metrics['recall'])
            self.print_to_log_file(f"   Precision: {precision_mean:.4f}")
            self.print_to_log_file(f"   Recall: {recall_mean:.4f}")
            
            iou_mean = np.nanmean(metrics['iou'])
            self.print_to_log_file(f"   IoU: {iou_mean:.4f}")
            
            self.print_to_log_file(f"\nPixel stats (foreground):")
            self.print_to_log_file(f"   TP: {np.sum(metrics['tp']):,}")
            self.print_to_log_file(f"   FP: {np.sum(metrics['fp']):,}")
            self.print_to_log_file(f"   FN: {np.sum(metrics['fn']):,}")
            
            if self.enable_adaptive_loss and self.current_epoch > 0 and self.current_epoch % self.adaptive_adjust_interval == 0:
                self.print_to_log_file(f"\nAdaptive parameter adjustment (Epoch {self.current_epoch}):")
                was_adjusted = self._adapt_loss_parameters(precision_mean, recall_mean, std_dice_mean)
                if not was_adjusted:
                    self.print_to_log_file(f"   Current params good, no adjustment needed")
                    self.print_to_log_file(f"   Current alpha={self.tversky_alpha:.3f}, beta={self.tversky_beta:.3f}")
        
        if hasattr(self, 'best_val_eval_criterion_MA'):
            current_dice = self.best_val_eval_criterion_MA
            
            if self.enable_patience_trigger:
                has_improved = self._update_dice_improvement_status(current_dice)
                if has_improved:
                    self.print_to_log_file(f"\n   New best Dice: {current_dice:.4f} (Patience counter reset)")
                else:
                    self.print_to_log_file(f"\n   Current Dice: {current_dice:.4f} ({self.epochs_without_dice_improvement} epochs without improvement)")
            else:
                if current_dice > self.best_dice:
                    self.print_to_log_file(f"\n   New best Dice: {current_dice:.4f}")
                else:
                    self.print_to_log_file(f"\n   Current Dice: {current_dice:.4f} (best: {self.best_dice:.4f})")
            
            if current_dice > self.best_dice:
                self.best_dice = current_dice
            
            self.print_to_log_file(f"   Best model epoch: {self.best_val_eval_criterion_MA_epoch}")

    def on_epoch_end(self) -> bool:
        continue_training = super().on_epoch_end()
        
        if hasattr(self, 'best_val_eval_criterion_MA') and self.current_epoch > self.freeze_epochs + 5:
            epochs_without_improvement = self.current_epoch - self.best_val_eval_criterion_MA_epoch
            if epochs_without_improvement > self.patience:
                self.print_to_log_file(f"\nEarly stop triggered!")
                self.print_to_log_file(f"   {self.patience} epochs without improvement")
                self.print_to_log_file(f"   Best epoch: {self.best_val_eval_criterion_MA_epoch}")
                self.print_to_log_file(f"   Current epoch: {self.current_epoch}")
                
                self._print_training_summary()
                return False
        
        return continue_training

    def _save_best_dice_ignore_fn_model(self):
        try:
            if hasattr(self, 'output_folder'):
                checkpoint_name = f"checkpoint_best_dice_ignore_fn_epoch_{self.current_epoch}.pth"
                model_path = os.path.join(self.output_folder, checkpoint_name)
                
                checkpoint = {
                    'epoch': self.current_epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice_ignore_fn': self.best_dice_ignore_fn,
                    'standard_dice': self.best_dice if hasattr(self, 'best_dice') else 0.0,
                    'precision': np.nanmean(self.current_val_metrics['precision']) if hasattr(self, 'current_val_metrics') else 0.0,
                    'recall': np.nanmean(self.current_val_metrics['recall']) if hasattr(self, 'current_val_metrics') else 0.0,
                }
                
                if self.best_dice_ignore_fn_model_path is not None and os.path.exists(self.best_dice_ignore_fn_model_path):
                    try:
                        os.remove(self.best_dice_ignore_fn_model_path)
                        self.print_to_log_file(f"   Deleted old best model: {os.path.basename(self.best_dice_ignore_fn_model_path)}")
                    except Exception as e:
                        self.print_to_log_file(f"   [WARNING] Failed to delete old model: {e}")
                
                torch.save(checkpoint, model_path)
                self.best_dice_ignore_fn_model_path = model_path
                
                self.print_to_log_file(f"   Saved best dice ignore FN model: {checkpoint_name}")
                self.print_to_log_file(f"      Dice ignore FN: {self.best_dice_ignore_fn:.4f}")
                self.print_to_log_file(f"      Standard Dice: {checkpoint['standard_dice']:.4f}")
                self.print_to_log_file(f"      Precision: {checkpoint['precision']:.4f}")
                
                latest_path = os.path.join(self.output_folder, "checkpoint_best_dice_ignore_fn_latest.pth")
                shutil.copy2(model_path, latest_path)
                
        except Exception as e:
            self.print_to_log_file(f"   [ERROR] Failed to save best dice ignore FN model: {e}")

    def _print_training_summary(self):
        self.print_to_log_file("\n" + "=" * 70)
        self.print_to_log_file("Training Summary")
        self.print_to_log_file("=" * 70)
        
        self.print_to_log_file(f"\nTime stats:")
        self.print_to_log_file(f"   Total epochs: {self.current_epoch + 1}/{self.num_epochs}")
        self.print_to_log_file(f"   Total time: {timedelta(seconds=int(self.total_train_time))}")
        self.print_to_log_file(f"   Avg epoch time: {timedelta(seconds=int(self.total_train_time / (self.current_epoch + 1)))}")
        
        if self.train_losses:
            self.print_to_log_file(f"\nLoss stats:")
            self.print_to_log_file(f"   Initial train loss: {self.train_losses[0]:.6f}")
            self.print_to_log_file(f"   Final train loss: {self.train_losses[-1]:.6f}")
            self.print_to_log_file(f"   Loss reduction: {(self.train_losses[0] - self.train_losses[-1]):.6f}")
        
        if self.val_losses:
            self.print_to_log_file(f"\nValidation stats:")
            self.print_to_log_file(f"   Best val Dice (standard): {self.best_dice:.4f}")
            self.print_to_log_file(f"   Best model epoch: {self.best_val_eval_criterion_MA_epoch}")
            
            if self.track_dice_ignore_fn:
                self.print_to_log_file(f"\nDice ignore FN stats:")
                self.print_to_log_file(f"   Best dice ignore FN: {self.best_dice_ignore_fn:.4f}")
                self.print_to_log_file(f"   Best epoch: {self.best_dice_ignore_fn_epoch}")
                if self.best_dice_ignore_fn_model_path and os.path.exists(self.best_dice_ignore_fn_model_path):
                    self.print_to_log_file(f"   Model file: {os.path.basename(self.best_dice_ignore_fn_model_path)}")
                if len(self.dice_ignore_fn_history) > 0:
                    self.print_to_log_file(f"   Avg dice ignore FN: {np.mean(self.dice_ignore_fn_history):.4f}")
                    self.print_to_log_file(f"   Final dice ignore FN: {self.dice_ignore_fn_history[-1]:.4f}")
        
        if self.enable_progressive_unfreezing:
            self.print_to_log_file(f"\nProgressive unfreeze stats:")
            self.print_to_log_file(f"   Final unfrozen layers: {len(self.unfrozen_layers)}")
            if len(self.unfrozen_layers) > 0:
                total_encoder_layers = sum(1 for name, _ in self.network.named_parameters() if 'encoder' in name.lower())
                unfreeze_progress = len(self.unfrozen_layers) / total_encoder_layers * 100 if total_encoder_layers > 0 else 0
                self.print_to_log_file(f"   Unfreeze progress: {unfreeze_progress:.1f}%")
        
        if hasattr(self, 'current_val_metrics'):
            metrics = self.current_val_metrics
            self.print_to_log_file(f"\nFinal validation metrics:")
            
            std_dice_mean = np.nanmean(metrics['standard_dice'])
            dice_ignore_fn_mean = np.nanmean(metrics['dice_ignore_fn'])
            precision_mean = np.nanmean(metrics['precision'])
            recall_mean = np.nanmean(metrics['recall'])
            
            self.print_to_log_file(f"   Standard Dice: {std_dice_mean:.4f}")
            self.print_to_log_file(f"   Dice ignore FN: {dice_ignore_fn_mean:.4f}")
            self.print_to_log_file(f"   Precision: {precision_mean:.4f}")
            self.print_to_log_file(f"   Recall: {recall_mean:.4f}")
            
            if precision_mean + recall_mean > 0:
                f1 = 2 * precision_mean * recall_mean / (precision_mean + recall_mean)
                self.print_to_log_file(f"   F1 Score: {f1:.4f}")
        
        if self.enable_gradient_monitoring and len(self.fg_gradient_ratio_history) > 0:
            self.print_to_log_file(f"\nGradient monitoring stats:")
            self.print_to_log_file(f"   FG gradient ratio: avg={np.mean(self.fg_gradient_ratio_history):.1%}")
            self.print_to_log_file(f"   FG gradient range: [{np.min(self.fg_gradient_ratio_history):.1%}, {np.max(self.fg_gradient_ratio_history):.1%}]")
        
        if self.enable_adaptive_loss and len(self.alpha_history) > 0:
            self.print_to_log_file(f"\nAdaptive parameter adjustment history:")
            self.print_to_log_file(f"   Initial: alpha=0.20, beta=0.80")
            self.print_to_log_file(f"   Final: alpha={self.tversky_alpha:.3f}, beta={self.tversky_beta:.3f}")
            self.print_to_log_file(f"   Adjustments: {len(self.alpha_history)}")
        
        if len(self.precision_history) > 0:
            self.print_to_log_file(f"\nPrecision/Recall trend:")
            self.print_to_log_file(f"   Precision: initial={self.precision_history[0]:.4f}, final={self.precision_history[-1]:.4f}")
            self.print_to_log_file(f"   Recall: initial={self.recall_history[0]:.4f}, final={self.recall_history[-1]:.4f}")
        
        self.print_to_log_file("\n" + "=" * 70)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        if self.simulate_layer_structure:
            base_rotation = np.random.uniform(-10, 10)
            rotation_for_DA = {
                'x': ((base_rotation - 15) / 360 * 2. * np.pi, (base_rotation + 15) / 360 * 2. * np.pi),
                'y': ((-10) / 360 * 2. * np.pi, (10) / 360 * 2. * np.pi),
                'z': ((-5) / 360 * 2. * np.pi, (5) / 360 * 2. * np.pi),
            }
        else:
            rotation_for_DA = {
                'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                'y': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                'z': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            }
        
        scale_range = (0.85, 1.15)
        do_mirroring = True
        initial_patch_size = self.configuration_manager.patch_size
        
        self.print_to_log_file("\nData augmentation config:")
        self.print_to_log_file(f"   Rotation X: +/-15 deg")
        self.print_to_log_file(f"   Rotation Y: +/-10 deg")
        self.print_to_log_file(f"   Rotation Z: +/-5 deg")
        self.print_to_log_file(f"   Scale range: [{scale_range[0]:.2f}, {scale_range[1]:.2f}]")
        self.print_to_log_file(f"   Mirroring: {do_mirroring}")
        
        if self.enable_spatial_augmentation:
            self.print_to_log_file(f"\n   Spatial augmentation:")
            
            if self.elastic_deformation:
                self.print_to_log_file(f"      Elastic deformation: sigma={self.elastic_sigma_range}, alpha={self.elastic_alpha_range}")
            
            if self.random_gamma:
                self.print_to_log_file(f"      Random gamma: range={self.gamma_range}")
            
            if self.additive_noise:
                self.print_to_log_file(f"      Additive noise: var={self.noise_variance}")
        
        self.print_to_log_file(f"   Initial patch size: {initial_patch_size}")
        
        return rotation_for_DA, do_mirroring, initial_patch_size, scale_range

    def run_training(self):
        start_time = time.time()
        
        self.print_to_log_file("\n" + "=" * 70)
        self.print_to_log_file("Starting transfer learning training")
        self.print_to_log_file(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.print_to_log_file("=" * 70 + "\n")
        
        try:
            result = super().run_training()
            
            total_time = time.time() - start_time
            self.print_to_log_file("\n" + "=" * 70)
            self.print_to_log_file("Training completed successfully!")
            self.print_to_log_file(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.print_to_log_file(f"Total time: {timedelta(seconds=int(total_time))}")
            self.print_to_log_file("=" * 70)
            
            self._print_training_summary()
            
            return result
            
        except Exception as e:
            self.print_to_log_file("\n" + "=" * 70)
            self.print_to_log_file("Training terminated with error!")
            self.print_to_log_file(f"Error type: {type(e).__name__}")
            self.print_to_log_file(f"Error message: {str(e)}")
            self.print_to_log_file("=" * 70)
            raise


class nnUNetTrainerV2_MedNeXt_B_TransferLearning_100epochs(nnUNetTrainerV2_MedNeXt_B_TransferLearning):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
        self.freeze_epochs = 3
        self.patience = 25


class nnUNetTrainerV2_MedNeXt_B_TransferLearning_AggressiveDA(nnUNetTrainerV2_MedNeXt_B_TransferLearning):
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA = {
            'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        }
        
        scale_range = (0.7, 1.3)
        do_mirroring = True
        initial_patch_size = self.configuration_manager.patch_size
        
        self.print_to_log_file("\nAggressive data augmentation config:")
        self.print_to_log_file(f"   Rotation range: +/-30 deg")
        self.print_to_log_file(f"   Scale range: [{scale_range[0]:.2f}, {scale_range[1]:.2f}]")
        
        return rotation_for_DA, do_mirroring, initial_patch_size, scale_range


class nnUNetTrainerV2_MedNeXt_B_TransferLearning_LowLR(nnUNetTrainerV2_MedNeXt_B_TransferLearning):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 5e-5
        self.freeze_epochs = 2
        self.num_epochs = 80
        self.patience = 15
        
        self.print_to_log_file("\nLow LR config:")
        self.print_to_log_file(f"   LR: {self.initial_lr:.2e}")
        self.print_to_log_file(f"   Freeze epochs: {self.freeze_epochs}")
        self.print_to_log_file(f"   Total epochs: {self.num_epochs}")


class nnUNetTrainerV2_MedNeXt_B_TransferLearning_FocalTversky(nnUNetTrainerV2_MedNeXt_B_TransferLearning):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.focal_gamma = 1.33
        
    def configure_loss_function(self):
        from nnunetv2.utilities.helpers import softmax_helper_dim1
        
        self.print_to_log_file(f"\nUsing Focal Tversky Loss:")
        self.print_to_log_file(f"  Alpha={self.tversky_alpha}, Beta={self.tversky_beta}, Gamma={self.focal_gamma}")
        
        loss = FocalTverskyLoss(
            apply_nonlin=softmax_helper_dim1,
            batch_dice=True,
            do_bg=False,
            smooth=1e-5,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
            gamma=self.focal_gamma
        )
        
        return loss
