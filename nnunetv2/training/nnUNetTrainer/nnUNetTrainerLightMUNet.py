"""
LightM-UNet 训练器
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import \
    nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.helpers import dummy_context
from torch import nn, autocast
import torch

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from torch.cuda.amp import GradScaler

from nnunetv2.nets.LightMUNet import LightMUNet
from torch.optim import Adam


class nnUNetTrainerLightMUNet(nnUNetTrainer_MedNeXtBase):
    """LightM-UNet 训练器 — 无深度监督，使用 Adam 优化器与 PolyLR 调度"""
    _ds_enabled = False

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        # 必须关键字传 device：MRO 下一跳 nnUNetTrainer_MedNeXtBase.__init__ 的签名是
        # (plans, configuration, fold, dataset_json, unpack_dataset=True, device=...)，
        # 位置传参会把 device 错位塞进 unpack_dataset，device 落回默认（无 index）→
        # DDP 下兜底成 cuda:local_rank 而绑错卡（同 2026-08-11 事故）。
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # ========== 数值稳定性修复: fp16 → bf16 (2026-09-16) ==========
        # 根因: fp16 只有 5 位指数（动态范围约 6e-5 .. 65504），本网络大量使用
        # InstanceNorm（涉及统计量的倒数）与深监督多分辨率累加，数值范围易越界 →
        # 溢出成 NaN。bf16 指数位与 fp32 相同（8 位），动态范围一致，从源头消除溢出
        # （同 LightMamba2Net 2026-07-31 的修复思路）。
        # 实证: LightMUNet 的 fp16 版 2026-09-14 跑满 100 epoch 但 Pseudo dice 恒 0.0000，
        # 其 checkpoint_final 权重 335/335 张量全 NaN（权重彻底报废）；同网络族的
        # LightMamba2Net 切 bf16 后跑满 100 epoch（nan=0, dice=0.6266）。
        self.autocast_dtype = torch.bfloat16
        # bf16 无需梯度缩放。显式禁用 GradScaler —— 它在 NaN 步只会静默跳过
        # optimizer.step，让权重在「看似训练」中悄悄退化（LightMUNet 空转 88 epoch 的机制之一）。
        self.grad_scaler = None
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = LightMUNet(
            spatial_dims=len(configuration_manager.patch_size),
            init_filters=16,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
        )

        return model

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # 必须显式传 dtype=self.autocast_dtype：不传时 torch 默认 fp16，会让 __init__
        # 里的 bf16 设置在本方法内完全失效（本类自带 train_step，不走基类那条已传 dtype 的路径）。
        with autocast(self.device.type, dtype=self.autocast_dtype, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # 验证在 torch.no_grad() 下运行，无需 zero_grad；原 zero_grad(set_to_none=True)
        # 会把 param.grad 置 None，破坏 CUDAGraphMixin replay 锁定的梯度地址，
        # 导致下一轮训练 GradScaler 报 "No inf checks were recorded"。
        output = self.network(data)
        del data
        l = self.loss(output, target)

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
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
        optimizer = Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.9)
        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        pass


class nnUNetTrainerLightMUNet_100epochs(nnUNetTrainerLightMUNet):
    """100 轮版本的 LightM-UNet 训练器"""

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 100
        self.num_iterations_per_epoch = 250
