from typing import List

import numpy as np
import torch
from torch import distributed as dist

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs


class nnUNetTrainer_MedNeXtBase(nnUNetTrainer):
    """
    MedNeXt 基础 Trainer —— 在官方 nnUNetTrainer 之上叠加 TF32 加速和增强指标。

    被所有 MedNeXt 系列模型 Trainer 继承，以非侵入方式叠加以下能力：
      - TF32 矩阵乘法加速（Ampere+ GPU）
      - Precision / Recall / Dice_Ignore_FN / Std_Dice 增强指标
      - DDP 同步下的指标聚合

    子类可通过 _ds_enabled 类变量控制 deep supervision（默认开启）。
    """

    # 类级别 deep supervision 开关 —— 子类可覆写
    _ds_enabled = True

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # 先调用父类初始化，确保 self.device / self.is_ddp 等属性已就绪
        # nnUNetTrainer.__init__ 会设置 self.enable_deep_supervision = True
        # unpack_dataset 由 nnUNetTrainer 内部默认处理，不传递到父类（父类签名不包含此参数）
        super().__init__(plans, configuration, fold, dataset_json, device)

        # ========== TF32 加速配置 —— 来自 MedNeXt 项目 ==========
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            import os
            if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            torch.set_float32_matmul_precision('high')

        # ========== TF32 状态由 nnUNetTrainer.initialize() 统一打印 ==========

        # ========== 注册 MedNeXt 增强指标到 MetaLogger ==========
        # LocalLogger.log() 会 assert key 必须存在于 my_fantastic_logging 中，
        # 因此需要在 __init__ 中预先注册自定义指标 key。
        for key in ['precision', 'recall', 'dice_ignore_fn', 'std_dice']:
            if key not in self.logger.local_logger.my_fantastic_logging:
                self.logger.local_logger.my_fantastic_logging[key] = []

        # 用类变量覆盖父类设置的 deep supervision（子类可声明 _ds_enabled = False 来禁用）
        self.enable_deep_supervision = self.__class__._ds_enabled

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        在父类默认 logging 之后，叠加 MedNeXt 增强指标（precision / recall / dice_ignore_fn / std_dice）。
        """
        # 先调用父类处理官方默认 logging（mean_fg_dice, dice_per_class, val_losses）
        super().on_validation_epoch_end(val_outputs)

        # 再次聚合 outputs 计算增强指标
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        # DDP 同步 —— 仿 MedNeXt 第 1030-1047 行
        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

        # 去掉背景类（channel 0）
        tp_hard = tp[1:]
        fp_hard = fp[1:]
        fn_hard = fn[1:]

        total_tp = np.sum(tp_hard)
        total_fp = np.sum(fp_hard)
        total_fn = np.sum(fn_hard)

        # 计算增强指标 —— 仿 MedNeXt 第 1058-1062 行
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        dice_ignore_fn = 2 * total_tp / (2 * total_tp + total_fp + 1e-8)
        std_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)

        # 通过 MetaLogger 记录增强指标
        self.logger.log('precision', precision, self.current_epoch)
        self.logger.log('recall', recall, self.current_epoch)
        self.logger.log('dice_ignore_fn', dice_ignore_fn, self.current_epoch)
        self.logger.log('std_dice', std_dice, self.current_epoch)

    def on_epoch_end(self):
        """
        在父类默认 epoch 结束处理之后，打印 MedNeXt 增强指标。
        全零时跳过（模型尚未学到有效前景预测），非全零时附带 Δ 变化量和箭头指示。
        """
        # 先调用父类处理官方默认行为
        super().on_epoch_end()

        # 通过 MetaLogger 的 get_value API 取出增强指标并打印
        precision_list = self.logger.get_value('precision', step=None)
        if len(precision_list) > 0:
            precision = precision_list[-1]
            recall = self.logger.get_value('recall', step=-1)
            dice_ignore_fn = self.logger.get_value('dice_ignore_fn', step=-1)
            std_dice = self.logger.get_value('std_dice', step=-1)

            # 全零则跳过：模型尚未学到有效前景预测，输出无意义
            if precision == 0.0 and recall == 0.0 and dice_ignore_fn == 0.0 and std_dice == 0.0:
                return

            # 计算 Δ 变化量（higher is better）
            if len(precision_list) >= 2:
                p_delta = precision_list[-1] - precision_list[-2]
                if abs(p_delta) < 1e-8:
                    p_arrow = '→'
                else:
                    p_arrow = '↑' if p_delta > 0 else '↓'
                p_str = f'(Δ {p_delta:+.4f}) {p_arrow}'
            else:
                p_str = ''

            self.print_to_log_file(
                f'  Precision: {np.round(precision, decimals=4)} {p_str}, '
                f'Recall: {np.round(recall, decimals=4)}, '
                f'Dice_Ignore_FN: {np.round(dice_ignore_fn, decimals=4)}, '
                f'Std_Dice: {np.round(std_dice, decimals=4)}'
            )
