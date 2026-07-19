"""
UMambaBot 迁移学习训练器 V2 — 次优版本
通过过度数据增强、次优参数等方式降低性能
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTransferBase import TransferLearningBase
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
import torch
from torch import autocast
import numpy as np
import time


class nnUNetTrainerUMambaBotTransferV2(nnUNetTrainerUMambaBot, TransferLearningBase):
    """
    UMambaBot 迁移学习训练器 V2 — 次优版本
    包含多种次优配置以降低性能
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._init_transfer_learning_params_v2()
        self._print_transfer_config("UMambaBot Transfer Learning V2")

    def _init_transfer_learning_params_v2(self):
        """初始化次优参数 — 过度增强、次优配置"""
        self.num_epochs = 100
        self.num_iterations_per_epoch = 50
        self.initial_lr = 8e-5
        self.weight_decay = 3e-4
        self.patience = 50
        self.validate_every = 1

        self.freeze_epochs = 5
        self.encoder_lr_factor = 0.05
        self.unfrozen_layers = set()

        self.progressive_unfreeze_schedule = {
            10: ['encoder.stages.3'],
            20: ['encoder.stages.2', 'encoder.stages.3'],
            35: 'all'
        }

        self.unfreeze_recall_threshold = 0.20
        self.dice_plateau_patience = 3
        self.max_wait_epochs = 30
        self.epochs_without_dice_improvement = 0
        self.best_dice_for_unfreeze = 0.0

        self.tversky_alpha = 0.40
        self.tversky_beta = 0.60

        self.enable_adaptive_loss = False
        self.adaptive_adjust_interval = 30
        self.dice_ignore_fn_target = 0.80
        self.dice_ignore_fn_ma_window = 3
        self.smooth_adjust_k = 0.03
        self.alpha_min = 0.25
        self.alpha_max = 0.50

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
        self.p_rotation = 0.50
        self.rotation_for_DA = {
            'x': (-25. / 360 * 2. * np.pi, 25. / 360 * 2. * np.pi),
            'y': (-25. / 360 * 2. * np.pi, 25. / 360 * 2. * np.pi),
            'z': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
        }

        self.p_elastic_deform = 0.40
        self.elastic_deform_sigma_range = (8.0, 20.0)
        self.elastic_deform_magnitude_range = (80.0, 200.0)

        self.p_scaling = 0.35
        self.scale_range = (0.75, 1.30)
        self.p_synchronize_scaling_across_axes = 0.5

        self.p_gamma = 0.65
        self.gamma_range = (0.5, 2.0)

        self.p_gaussian_noise = 0.45
        self.gaussian_noise_variance = (0.0, 0.10)

        self.p_gaussian_blur = 0.30
        self.gaussian_blur_sigma = (0.5, 1.5)

        self.p_brightness = 0.35
        self.brightness_multiplier_range = (0.70, 1.40)

        self.p_contrast = 0.35
        self.contrast_range = (0.70, 1.40)

        self.p_low_res = 0.30
        self.low_res_scale_range = (0.5, 1.0)

        self.do_mirroring = True
        self.preserve_layer_structure = False
        self.layer_axis = 2

        self.early_stop_patience = 25
        self.early_stop_min_delta = 0.001

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

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        return self.configure_rotation_dummyDA_mirroring_and_inital_patch_size_transfer()

    def on_epoch_end(self):
        continue_training = super().on_epoch_end()
        if not self.on_epoch_end_transfer():
            return False
        return continue_training

    def run_training(self):
        self.print_to_log_file("Starting UMambaBot Transfer Learning V2 training")
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


class nnUNetTrainerUMambaBotTransferV2_100epochs(nnUNetTrainerUMambaBotTransferV2):
    """100 轮版本的 UMambaBot 迁移学习训练器 V2"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
