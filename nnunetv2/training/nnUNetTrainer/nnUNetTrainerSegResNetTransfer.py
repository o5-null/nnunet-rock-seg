from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTransferBase import TransferLearningBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
import torch
from torch import autocast
import numpy as np
import time


class nnUNetTrainerSegResNetTransfer(nnUNetTrainer_MedNeXtBase, TransferLearningBase):
    """
    SegResNet 迁移学习训练器 —— 多继承 nnUNetTrainer_MedNeXtBase + TransferLearningBase。
    继承调整：使用 MedNeXtBase 中间基类以获取 TF32 加速 + 增强指标。
    """

    @staticmethod
    def build_network_architecture(plans_manager,
                                   configuration_manager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        from monai.networks.nets import SegResNet
        
        network = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            dropout_prob=0.0
        )
        
        return network

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._init_transfer_learning_params()
        self.initial_lr = 1e-4
        self.encoder_lr_factor = 0.01
        self.freeze_epochs = 5
        self.tversky_alpha = 0.3
        self.tversky_beta = 0.7
        self.enable_adaptive_loss = True
        self.enable_spatial_augmentation = True
        self._print_transfer_config("SegResNet Transfer Learning")

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
        self.print_to_log_file("Starting SegResNet Transfer Learning training")
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


class nnUNetTrainerSegResNetTransfer_100epochs(nnUNetTrainerSegResNetTransfer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100


class nnUNetTrainerSegResNetTransfer_50epochs(nnUNetTrainerSegResNetTransfer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 50
