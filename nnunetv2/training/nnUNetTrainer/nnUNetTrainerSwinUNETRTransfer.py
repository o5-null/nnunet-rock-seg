"""
SwinUNETR 迁移学习训练器
多继承 nnUNetTrainerSwinUNETR + TransferLearningBase
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUNETR import nnUNetTrainerSwinUNETR
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTransferBase import TransferLearningBase
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
import torch
from torch import autocast
import numpy as np
import time


class nnUNetTrainerSwinUNETRTransfer(nnUNetTrainerSwinUNETR, TransferLearningBase):
    """SwinUNETR 迁移学习训练器 — 渐进解冻 + Tversky Loss"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self._init_transfer_learning_params()
        # SwinUNETR 使用更低的学习率
        self.initial_lr = 5e-5
        self.encoder_lr_factor = 0.005
        self._print_transfer_config("SwinUNETR Transfer Learning")

    def configure_loss_function(self):
        """配置 Tversky Loss"""
        return self.configure_loss_function_transfer()

    def configure_optimizers(self):
        """配置分层学习率优化器"""
        return self.configure_optimizers_transfer()

    def on_train_epoch_start(self):
        """训练 epoch 开始时的处理（渐进解冻）"""
        super().on_train_epoch_start()
        self.on_train_epoch_start_transfer()

    def train_step(self, batch: dict) -> dict:
        """训练步骤（含 batchgeneratorsv2 增强）"""
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        return self.train_step_transfer(data, target)

    def on_validation_epoch_end(self, val_outputs=None):
        """验证 epoch 结束时的处理"""
        self.on_validation_epoch_end_transfer(val_outputs)
        super().on_validation_epoch_end(val_outputs)

        if hasattr(self, 'current_val_metrics'):
            m = self.current_val_metrics
            self.print_to_log_file(f"Val - StdDice: {m['std_dice']:.4f}, IgnoreFNDice: {m['dice_ignore_fn']:.4f}, "
                                   f"Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")

            # 保存最佳 ignore FN dice 模型
            if self.track_dice_ignore_fn and m['dice_ignore_fn'] > self.best_dice_ignore_fn:
                self.best_dice_ignore_fn = m['dice_ignore_fn']
                self.best_dice_ignore_fn_epoch = self.current_epoch
                self._save_best_dice_ignore_fn_model()

            # 更新 plateau 计数器
            if m['dice_ignore_fn'] > self.best_dice_for_unfreeze + 0.001:
                self.best_dice_for_unfreeze = m['dice_ignore_fn']
                self.epochs_without_dice_improvement = 0
            else:
                self.epochs_without_dice_improvement += 1

            # 自适应损失调整
            if self.enable_adaptive_loss and self.current_epoch > 0 and \
               self.current_epoch % self.adaptive_adjust_interval == 0:
                self._adapt_loss_parameters()

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """配置数据增强参数"""
        return self.configure_rotation_dummyDA_mirroring_and_inital_patch_size_transfer()

    def on_epoch_end(self):
        """epoch 结束时的处理（含 early stopping）"""
        continue_training = super().on_epoch_end()
        if not self.on_epoch_end_transfer():
            return False
        return continue_training

    def run_training(self):
        """运行训练"""
        self.print_to_log_file("Starting SwinUNETR Transfer Learning training")
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


class nnUNetTrainerSwinUNETRTransfer_100epochs(nnUNetTrainerSwinUNETRTransfer):
    """100 轮版本的 SwinUNETR 迁移学习训练器"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 100
