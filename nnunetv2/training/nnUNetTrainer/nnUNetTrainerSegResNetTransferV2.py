"""
SegResNet 迁移学习训练器 V2 - 次优版本
通过次优参数降低性能
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegResNet import nnUNetTrainerSegResNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTransferBase import TransferLearningBase
import torch
import time
import numpy as np


class nnUNetTrainerSegResNetTransferV2(nnUNetTrainerSegResNet, TransferLearningBase):
    """
    SegResNet 迁移学习训练器 V2 - 次优版本
    次优参数配置
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._init_transfer_learning_params_v2()
        self._print_transfer_config("SegResNet Transfer Learning V2")

    def _init_transfer_learning_params_v2(self):
        """初始化次优参数 - 负调整：禁用早停，降低假阳性"""
        self.num_epochs = 100
        self.num_iterations_per_epoch = 50
        self.initial_lr = 5e-5  # 进一步降低学习率
        self.weight_decay = 8e-4  # 增加权重衰减
        self.patience = 999999  # 禁用早停
        self.validate_every = 1

        self.freeze_epochs = 5  # 增加冻结轮数
        self.encoder_lr_factor = 0.05  # 降低编码器学习率因子
        self.unfrozen_layers = set()

        self.progressive_unfreeze_schedule = {
            12: ['encoder.stages.3'],
            25: ['encoder.stages.2', 'encoder.stages.3'],
            40: 'all'
        }

        self.unfreeze_recall_threshold = 0.10  # 提高解冻阈值，更保守
        self.dice_plateau_patience = 999999  # 禁用基于plateau的解冻
        self.max_wait_epochs = 999999
        self.epochs_without_dice_improvement = 0
        self.best_dice_for_unfreeze = 0.0

        # 调整Tversky Loss参数：显著增加alpha惩罚假阳性，降低beta减少假阴性惩罚
        self.tversky_alpha = 0.65  # 显著增加FP惩罚
        self.tversky_beta = 0.35   # 显著降低FN惩罚

        self.enable_adaptive_loss = False  # 禁用自适应调整
        self.adaptive_adjust_interval = 999999
        self.dice_ignore_fn_target = 0.60  # 降低目标，更保守
        self.dice_ignore_fn_ma_window = 3
        self.smooth_adjust_k = 0.05
        self.alpha_min = 0.60  # 固定较高alpha
        self.alpha_max = 0.70

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

        # 禁用大部分数据增强
        self.enable_spatial_augmentation = False  # 禁用空间增强
        self.p_rotation = 0.05  # 大幅降低旋转概率
        self.rotation_for_DA = {
            'x': (-10. / 360 * 2. * np.pi, 10. / 360 * 2. * np.pi),
            'y': (-10. / 360 * 2. * np.pi, 10. / 360 * 2. * np.pi),
            'z': (-3. / 360 * 2. * np.pi, 3. / 360 * 2. * np.pi),
        }

        self.p_elastic_deform = 0.0  # 禁用弹性形变
        self.elastic_deform_sigma_range = (5.0, 15.0)
        self.elastic_deform_magnitude_range = (50.0, 150.0)

        self.p_scaling = 0.05  # 大幅降低缩放概率
        self.scale_range = (0.90, 1.10)  # 更保守的缩放范围
        self.p_synchronize_scaling_across_axes = 0.9

        self.p_gamma = 0.15  # 大幅降低gamma变换
        self.gamma_range = (0.8, 1.3)

        self.p_gaussian_noise = 0.05  # 大幅降低噪声
        self.gaussian_noise_variance = (0.0, 0.03)

        self.p_gaussian_blur = 0.0  # 禁用高斯模糊
        self.gaussian_blur_sigma = (0.3, 0.8)

        self.p_brightness = 0.05  # 大幅降低亮度调整
        self.brightness_multiplier_range = (0.90, 1.10)

        self.p_contrast = 0.05  # 大幅降低对比度调整
        self.contrast_range = (0.90, 1.10)

        self.p_low_res = 0.0  # 禁用低分辨率模拟
        self.low_res_scale_range = (0.7, 1.0)

        self.do_mirroring = True
        self.preserve_layer_structure = True  # 启用层理保护
        self.layer_axis = 2

        # 禁用早停
        self.early_stop_patience = 999999
        self.early_stop_min_delta = 0.0001

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
        self.print_to_log_file("Starting SegResNet Transfer Learning V2 training")
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


class nnUNetTrainerSegResNetTransferV2_100epochs(nnUNetTrainerSegResNetTransferV2):
    """100轮版本的SegResNet迁移学习训练器 V2"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
