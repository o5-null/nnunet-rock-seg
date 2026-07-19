from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMedNext import (
    nnUNetTrainerMedNext,
    nnUNetTrainerV2_MedNeXt_B_kernel5,
    nnUNetTrainerV2_MedNeXt_L_kernel5,
    nnUNetTrainerV2_MedNeXt_M_kernel5,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTransferBase import TransferLearningBase
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
import torch
from torch import autocast
import numpy as np
import time


class nnUNetTrainerV2_MedNeXt_B_TransferLearningV3(nnUNetTrainerV2_MedNeXt_B_kernel5, TransferLearningBase):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._init_transfer_learning_params()
        self._print_transfer_config("MedNeXt Base Transfer Learning V3")

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
        self.print_to_log_file("Starting MedNeXt Base Transfer Learning V3 training")
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


class nnUNetTrainerV2_MedNeXt_L_TransferLearningV3(nnUNetTrainerV2_MedNeXt_L_kernel5, TransferLearningBase):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._init_transfer_learning_params()
        self.initial_lr = 5e-5
        self.encoder_lr_factor = 0.005
        self.freeze_epochs = 15
        self._print_transfer_config("MedNeXt Large Transfer Learning V3")

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
        self.print_to_log_file("Starting MedNeXt Large Transfer Learning V3 training")
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


class nnUNetTrainerV2_MedNeXt_M_TransferLearningV3(nnUNetTrainerV2_MedNeXt_M_kernel5, TransferLearningBase):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._init_transfer_learning_params()
        self._print_transfer_config("MedNeXt Medium Transfer Learning V3")

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
        self.print_to_log_file("Starting MedNeXt Medium Transfer Learning V3 training")
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


class nnUNetTrainerV2_MedNeXt_B_TransferLearningV3_100epochs(nnUNetTrainerV2_MedNeXt_B_TransferLearningV3):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100


class nnUNetTrainerV2_MedNeXt_L_TransferLearningV3_100epochs(nnUNetTrainerV2_MedNeXt_L_TransferLearningV3):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        self.num_iterations_per_epoch = 250


class nnUNetTrainerV2_MedNeXt_M_TransferLearningV3_100epochs(nnUNetTrainerV2_MedNeXt_M_TransferLearningV3):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
