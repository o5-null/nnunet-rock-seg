from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMedNext import (
    nnUNetTrainerV2_MedNeXt_B_kernel5,
    nnUNetTrainerV2_MedNeXt_L_kernel5,
    nnUNetTrainerV2_MedNeXt_M_kernel5,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTransferBase import TransferLearningBase
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
import torch
from torch import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import time


class MedNeXtTransferV4Base(TransferLearningBase):

    def _init_transfer_learning_params_v4(self):
        self.num_epochs = 100
        self.num_iterations_per_epoch = 50
        self.initial_lr = 3e-5
        self.weight_decay = 5e-5
        self.patience = 30
        self.validate_every = 1

        self.freeze_epochs = 10
        self.encoder_lr_factor = 0.008
        self.unfrozen_layers = set()

        self.progressive_unfreeze_schedule = {
            15: ['encoder.stages.3'],
            30: ['encoder.stages.2', 'encoder.stages.3'],
            50: ['encoder.stages.1', 'encoder.stages.2', 'encoder.stages.3'],
            75: 'all'
        }

        self.unfreeze_recall_threshold = 0.30
        self.dice_plateau_patience = 8
        self.max_wait_epochs = 50
        self.epochs_without_dice_improvement = 0
        self.best_dice_for_unfreeze = 0.0

        self.tversky_alpha = 0.25
        self.tversky_beta = 0.75

        self.enable_adaptive_loss = True
        self.adaptive_adjust_interval = 15
        self.dice_ignore_fn_target = 0.88
        self.dice_ignore_fn_ma_window = 7
        self.smooth_adjust_k = 0.015
        self.alpha_min = 0.15
        self.alpha_max = 0.45

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
        self.p_rotation = 0.25
        self.rotation_for_DA = {
            'x': (-12. / 360 * 2. * np.pi, 12. / 360 * 2. * np.pi),
            'y': (-12. / 360 * 2. * np.pi, 12. / 360 * 2. * np.pi),
            'z': (-3. / 360 * 2. * np.pi, 3. / 360 * 2. * np.pi),
        }

        self.p_elastic_deform = 0.20
        self.elastic_deform_sigma_range = (4.0, 12.0)
        self.elastic_deform_magnitude_range = (40.0, 120.0)

        self.p_scaling = 0.15
        self.scale_range = (0.90, 1.10)
        self.p_synchronize_scaling_across_axes = 0.85

        self.p_gamma = 0.45
        self.gamma_range = (0.75, 1.4)

        self.p_gaussian_noise = 0.25
        self.gaussian_noise_variance = (0.0, 0.04)

        self.p_gaussian_blur = 0.12
        self.gaussian_blur_sigma = (0.25, 0.7)

        self.p_brightness = 0.18
        self.brightness_multiplier_range = (0.88, 1.12)

        self.p_contrast = 0.18
        self.contrast_range = (0.88, 1.12)

        self.p_low_res = 0.15
        self.low_res_scale_range = (0.75, 1.0)

        self.do_mirroring = True
        self.preserve_layer_structure = True
        self.layer_axis = 2

        self.use_warm_restarts = True
        self.warm_restart_T0 = 40
        self.warm_restart_T_mult = 2
        self.warm_restart_eta_min = 5e-7

        self.early_stop_patience = 25
        self.early_stop_min_delta = 0.001

    def configure_optimizers_transfer(self):
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

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay, eps=1e-5)

        if self.use_warm_restarts:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.warm_restart_T0,
                T_mult=self.warm_restart_T_mult,
                eta_min=self.warm_restart_eta_min
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6
            )

        total = sum(p.numel() for p in self.network.parameters())
        trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.print_to_log_file(f"Params: {trainable}/{total} trainable ({trainable/total*100:.1f}%)")

        return optimizer, scheduler

    def on_epoch_end_transfer(self):
        if self.current_epoch > self.freeze_epochs + 10:
            no_improve = self.current_epoch - self.best_dice_ignore_fn_epoch
            if no_improve > self.early_stop_patience:
                self.print_to_log_file(f"[EARLY STOP] No improvement for {no_improve} epochs at epoch {self.current_epoch}")
                return False
        return True

    def _print_transfer_config(self, trainer_name="Transfer Learning V4"):
        self.print_to_log_file("=" * 60)
        self.print_to_log_file(f"{trainer_name} - Optimized Transfer Learning Config")
        self.print_to_log_file("=" * 60)
        self.print_to_log_file(f"Epochs: {self.num_epochs}, Freeze: {self.freeze_epochs}")
        self.print_to_log_file(f"LR: {self.initial_lr}, Encoder LR factor: {self.encoder_lr_factor}")
        self.print_to_log_file(f"Tversky Alpha: {self.tversky_alpha}, Beta: {self.tversky_beta}")
        self.print_to_log_file(f"Warm Restarts: {self.use_warm_restarts}, T0: {self.warm_restart_T0}")
        self.print_to_log_file(f"Early Stop Patience: {self.early_stop_patience}")
        self.print_to_log_file(f"Spatial augmentation: {self.enable_spatial_augmentation}")
        self.print_to_log_file("=" * 60)


class nnUNetTrainerV2_MedNeXt_L_TransferLearningV4(
    nnUNetTrainerV2_MedNeXt_L_kernel5, 
    MedNeXtTransferV4Base
):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        # 注意：新基类链不接收 unpack_dataset 参数
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._init_transfer_learning_params_v4()
        self.initial_lr = 3e-5
        self.encoder_lr_factor = 0.008
        self.freeze_epochs = 10
        self._print_transfer_config("MedNeXt Large Transfer Learning V4")

    def configure_loss_function(self):
        return self.configure_loss_function_transfer()

    def configure_optimizers(self):
        return self.configure_optimizers_transfer()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.on_train_epoch_start_transfer()

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        return self.train_step_transfer(data, target)

    def on_validation_epoch_end(self, val_outputs=None):
        self.on_validation_epoch_end_transfer(val_outputs)
        super().on_validation_epoch_end(val_outputs)

        if hasattr(self, 'current_val_metrics'):
            m = self.current_val_metrics
            self.print_to_log_file(f"Val - StdDice: {m['std_dice']:.4f}, IgnoreFNDice: {m['dice_ignore_fn']:.4f}, "
                                   f"Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")

            if self.track_dice_ignore_fn and m['dice_ignore_fn'] > self.best_dice_ignore_fn:
                self.best_dice_ignore_fn = m['dice_ignore_fn']
                self.best_dice_ignore_fn_epoch = self.current_epoch
                self._save_best_dice_ignore_fn_model()

            if m['dice_ignore_fn'] > self.best_dice_for_unfreeze + 0.001:
                self.best_dice_for_unfreeze = m['dice_ignore_fn']
                self.epochs_without_dice_improvement = 0
            else:
                self.epochs_without_dice_improvement += 1

            if self.enable_adaptive_loss and self.current_epoch > 0 and \
               self.current_epoch % self.adaptive_adjust_interval == 0:
                self._adapt_loss_parameters()

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        return self.configure_rotation_dummyDA_mirroring_and_inital_patch_size_transfer()

    def on_epoch_end(self):
        continue_training = super().on_epoch_end()
        if not self.on_epoch_end_transfer():
            return False
        return continue_training

    def run_training(self):
        self.print_to_log_file("Starting MedNeXt Large Transfer Learning V4 training")
        start = time.time()

        try:
            result = super().run_training()
            elapsed = time.time() - start
            self.print_to_log_file(f"Training completed in {elapsed/60:.1f} minutes")
            self._print_training_summary_transfer()
            return result
        except Exception as e:
            self.print_to_log_file(f"[ERROR] Training failed: {e}")
            raise


class nnUNetTrainerV2_MedNeXt_B_TransferLearningV4(
    nnUNetTrainerV2_MedNeXt_B_kernel5, 
    MedNeXtTransferV4Base
):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        # 注意：新基类链不接收 unpack_dataset 参数
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._init_transfer_learning_params_v4()
        self.initial_lr = 4e-5
        self.encoder_lr_factor = 0.01
        self.freeze_epochs = 8
        self._print_transfer_config("MedNeXt Base Transfer Learning V4")

    def configure_loss_function(self):
        return self.configure_loss_function_transfer()

    def configure_optimizers(self):
        return self.configure_optimizers_transfer()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.on_train_epoch_start_transfer()

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        return self.train_step_transfer(data, target)

    def on_validation_epoch_end(self, val_outputs=None):
        self.on_validation_epoch_end_transfer(val_outputs)
        super().on_validation_epoch_end(val_outputs)

        if hasattr(self, 'current_val_metrics'):
            m = self.current_val_metrics
            self.print_to_log_file(f"Val - StdDice: {m['std_dice']:.4f}, IgnoreFNDice: {m['dice_ignore_fn']:.4f}, "
                                   f"Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")

            if self.track_dice_ignore_fn and m['dice_ignore_fn'] > self.best_dice_ignore_fn:
                self.best_dice_ignore_fn = m['dice_ignore_fn']
                self.best_dice_ignore_fn_epoch = self.current_epoch
                self._save_best_dice_ignore_fn_model()

            if m['dice_ignore_fn'] > self.best_dice_for_unfreeze + 0.001:
                self.best_dice_for_unfreeze = m['dice_ignore_fn']
                self.epochs_without_dice_improvement = 0
            else:
                self.epochs_without_dice_improvement += 1

            if self.enable_adaptive_loss and self.current_epoch > 0 and \
               self.current_epoch % self.adaptive_adjust_interval == 0:
                self._adapt_loss_parameters()

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        return self.configure_rotation_dummyDA_mirroring_and_inital_patch_size_transfer()

    def on_epoch_end(self):
        continue_training = super().on_epoch_end()
        if not self.on_epoch_end_transfer():
            return False
        return continue_training

    def run_training(self):
        self.print_to_log_file("Starting MedNeXt Base Transfer Learning V4 training")
        start = time.time()

        try:
            result = super().run_training()
            elapsed = time.time() - start
            self.print_to_log_file(f"Training completed in {elapsed/60:.1f} minutes")
            self._print_training_summary_transfer()
            return result
        except Exception as e:
            self.print_to_log_file(f"[ERROR] Training failed: {e}")
            raise
