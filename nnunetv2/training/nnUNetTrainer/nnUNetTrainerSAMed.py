# SAMed Trainer — 基于 SAM + LoRA 的 2D-only 分割模型
#
# SAMed 为 2D-only 模型，不使用 deep supervision，使用 2d_p256 / 2d_p512 配置。
#
# 移植自 MedNeXt-Volcanic-Rocks 项目。

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import \
    nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import torch
from torch.optim import AdamW
from torch import nn
from nnunetv2.nets.sam_lora_image_encoder import LoRA_Sam
from nnunetv2.nets.segment_anything.modeling.mask_decoder import MLP, MaskDecoder
from nnunetv2.nets.segment_anything import sam_model_registry
from nnunetv2.training.lr_scheduler.samedlr import CustomWarmupDecayLR
from monai.transforms import Resize
from torch._dynamo import OptimizedModule

from typing import Union
import os


class nnUNetTrainerSAMed(nnUNetTrainer_MedNeXtBase):
    """
    SAMed 基础 Trainer — 2D-only, SAM + LoRA, 无 deep supervision。
    """
    _ds_enabled = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # 调用 MedNeXtBase 基类初始化
        super().__init__(plans, configuration, fold, dataset_json, device)
        # SAMed 需要将 patch_size 对齐到 2^5=32 的倍数（SAM encoder 下采样倍数）
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [-1] * len(original_patch_size)
        for i in range(len(original_patch_size)):
            if (original_patch_size[i] / 2 ** 5) < 1 or ((original_patch_size[i] / 2 ** 5) % 1) != 0:
                new_patch_size[i] = round(original_patch_size[i] / 2 ** 5 + 0.5) * 2 ** 5
            else:
                new_patch_size[i] = original_patch_size[i]
        self.configuration_manager.configuration['patch_size'] = new_patch_size
        self.print_to_log_file("Patch size changed from {} to {}".format(original_patch_size, new_patch_size))
        self.plans_manager.plans['configurations'][self.configuration_name]['patch_size'] = new_patch_size
        self.initial_lr = 1e-3
        self.weight_decay = 0.01
        self.lr_decay = 0.9
        # 默认 resize/patch_size 占位（子类会覆盖）
        self.resize = Resize(spatial_size=(64, 64), mode='nearest')
        self.patch_size = 256

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        """
        构建 SAMed 网络（vit_b/vit_h + LoRA）。
        强制要求 SAM 预训练 checkpoint 存在，不可随机初始化。
        """
        # 从配置中获取 image_size，默认 256
        patch_size = configuration_manager.patch_size
        image_size = patch_size[0] if len(patch_size) == 2 else 256
        # 对齐到 32 的倍数
        if image_size % 32 != 0:
            image_size = round(image_size / 32) * 32

        # 确定模型类型和对应的 checkpoint 文件名
        if image_size >= 512:
            model_type = 'vit_h'
            checkpoint_name = 'sam_vit_h_4b8939.pth'
        else:
            model_type = 'vit_b'
            checkpoint_name = 'sam_vit_b_01ec64.pth'

        # 在多个候选路径中搜索 checkpoint
        search_paths = [
            checkpoint_name,
            f'checkpoints/{checkpoint_name}',
        ]
        checkpoint = None
        for p in search_paths:
            if os.path.exists(p):
                checkpoint = p
                break

        if checkpoint is None:
            raise FileNotFoundError(
                f"SAMed requires pretrained SAM weights ({checkpoint_name}) but the file was not found.\n"
                f"Searched in: {search_paths}\n\n"
                f"Please download {checkpoint_name} and place it in the checkpoints/ directory.\n"
                f"Download link: https://github.com/facebookresearch/segment-anything#model-checkpoints")

        sam, img_embedding_size = sam_model_registry[model_type](
            image_size=image_size, num_classes=8,
            checkpoint=checkpoint,
            pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])

        model = LoRA_Sam(sam, 4)
        model.sam.mask_decoder = MaskDecoder(
            transformer=model.sam.mask_decoder.transformer,
            transformer_dim=model.sam.mask_decoder.transformer_dim,
            num_multimask_outputs=num_output_channels - 1)
        return model

    def train_step(self, batch: dict) -> dict:
        """SAMed 训练步骤：将 target 降采样到低分辨率后计算 loss"""
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            low_res_label_batch = [self.resize(i.to(self.device, non_blocking=True).squeeze()) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
            low_res_label_batch = self.resize(target.squeeze())

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            outputs = self.network(data, True, self.patch_size)
            l = self.loss(outputs['low_res_logits'], low_res_label_batch.unsqueeze(1))

        self.grad_scaler.scale(l).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """SAMed 验证步骤：含在线 Dice 评估"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            low_res_label_batch = [self.resize(i.to(self.device, non_blocking=True).squeeze()) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
            low_res_label_batch = self.resize(target.squeeze())

        # 验证在 torch.no_grad() 下运行，无需 zero_grad；原 zero_grad(set_to_none=True)
        # 会把 param.grad 置 None，破坏 CUDAGraphMixin replay 锁定的梯度地址，
        # 导致下一轮训练 GradScaler 报 "No inf checks were recorded"。
        output = self.network(data, True, self.patch_size)
        del data

        l = self.loss(output['low_res_logits'], low_res_label_batch.unsqueeze(1))
        output_masks = output['masks']

        # 在线评估：计算 fake Dice
        axes = [0] + list(range(2, output_masks.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_masks) > 0.5).long()
        else:
            output_seg = output_masks.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output_masks.shape, device=output_masks.device,
                                                        dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, tn = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        tn_hard = tn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
            tn_hard = tn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard, 'tn_hard': tn_hard}

    def configure_optimizers(self):
        """SAMed 使用 AdamW + CustomWarmupDecayLR 调度器"""
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.initial_lr,
                          betas=(0.9, 0.999),
                          weight_decay=0.1)
        scheduler = CustomWarmupDecayLR(optimizer, warmup_period=10, max_iterations=self.num_epochs,
                                        base_lr=self.initial_lr, weight_decay=self.lr_decay)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        """SAMed 始终禁用 deep supervision"""
        pass

    def save_checkpoint(self, filename: str) -> None:
        """SAMed 特殊 checkpoint：仅保存 LoRA 参数"""
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.get_lora_parameters(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """SAMed 特殊 checkpoint 加载：使用 load_lora_parameters"""
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        else:
            checkpoint = filename_or_checkpoint
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_lora_parameters(new_state_dict)
            else:
                self.network.module.load_lora_parameters(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_lora_parameters(new_state_dict)
            else:
                self.network.load_lora_parameters(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])


class nnUNetTrainerV2_SAMed_h_r_4(nnUNetTrainerSAMed):
    """
    SAMed-H (vit_h) — 2D 512 patch 配置, rank=4 LoRA。
    使用 sam_vit_h_4b8939.pth 预训练权重, 适用于 2d_p512 配置。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.patch_size = 512
        self.resize = Resize(spatial_size=(128, 128), mode='nearest')
        self.lr_decay = 7

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        # num_output_channels = label_manager.num_segmentation_heads（由基类 initialize 传入）
        sam, img_embedding_size = sam_model_registry['vit_h'](image_size=512,
                                                               num_classes=8,
                                                               checkpoint='checkpoints/sam_vit_h_4b8939.pth',
                                                               pixel_mean=[0, 0, 0],
                                                               pixel_std=[1, 1, 1])
        model = LoRA_Sam(sam, 4)
        model.sam.mask_decoder = MaskDecoder(transformer=model.sam.mask_decoder.transformer,
                                             transformer_dim=model.sam.mask_decoder.transformer_dim,
                                             num_multimask_outputs=num_output_channels - 1
                                             )
        return model


class nnUNetTrainerV2_SAMed_h_r_4_100epochs(nnUNetTrainerV2_SAMed_h_r_4):
    """SAMed-H (vit_h) — 100 epochs 版本"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerV2_SAMed_b_r_4(nnUNetTrainerSAMed):
    """
    SAMed-B (vit_b) — 2D 256 patch 配置, rank=4 LoRA。
    使用 sam_vit_b_01ec64.pth 预训练权重, 适用于 2d_p256 配置。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.patch_size = 256
        self.resize = Resize(spatial_size=(64, 64), mode='nearest')

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        # num_output_channels = label_manager.num_segmentation_heads（由基类 initialize 传入）
        sam, img_embedding_size = sam_model_registry['vit_b'](image_size=256,
                                                               num_classes=8,
                                                               checkpoint='checkpoints/sam_vit_b_01ec64.pth',
                                                               pixel_mean=[0, 0, 0],
                                                               pixel_std=[1, 1, 1])
        model = LoRA_Sam(sam, 4)
        model.sam.mask_decoder = MaskDecoder(transformer=model.sam.mask_decoder.transformer,
                                             transformer_dim=model.sam.mask_decoder.transformer_dim,
                                             num_multimask_outputs=num_output_channels - 1
                                             )
        return model


class nnUNetTrainerV2_SAMed_b_r_4_100epochs(nnUNetTrainerV2_SAMed_b_r_4):
    """SAMed-B (vit_b) — 100 epochs 版本"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
