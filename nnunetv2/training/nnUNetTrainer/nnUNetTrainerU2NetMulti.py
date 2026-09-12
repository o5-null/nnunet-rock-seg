"""
U2NetMulti 训练器 — 多尺度 U2Net (CNN 基线)

论文: From Claims to Evidence (arXiv:2503.01306)
继承 nnUNetTrainer_MedNeXtBase 以复用 TF32 加速与增强指标
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MedNeXtBase import nnUNetTrainer_MedNeXtBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.nets.u2net_multi import U2NET
import torch
from torch import nn


class nnUNetTrainerU2NetMulti(nnUNetTrainer_MedNeXtBase):
    """
    U2Net Multi: Multi-scale U2Net variant (CNN baseline)
    """
    # 验证迭代数基值。**保持基类默认 50，与其它训练器口径一致。**
    #
    # 曾设为 25 让验证图像数减半以缩短 epoch（CUDAGraphMixin.get_dataloaders 会按
    # val_batch 等比放大该值：batch=19/val_batch=8 时 50→119 次、25→60 次）。实测
    # epoch 由 72.5s 降到 64.5s，但**验证集从 952 张缩到 480 张**，导致：
    #   1. pseudo dice 与其它训练器 / 历史 run **不可直接对比**；
    #   2. ema_fg_dice（驱动 checkpoint_best 选择）噪声变大。
    # 因此回退为 50。要再调低，需自行承担上述代价。
    NUM_VAL_ITERATIONS_PER_EPOCH = 50

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # device 必须以关键字传递：MRO 下一跳 MedNeXtBase.__init__ 的签名是
        # (..., unpack_dataset=True, device=...)，位置传参会把 device 错位塞进
        # unpack_dataset，使 device 落回无 index 的默认值（指定卡 / DDP 时绑错卡。
        # 同一修复见 nnUNetTrainerBatchProbe.__init__）。
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # 显式写入（= 基类默认 50），保证不被 MRO 链上的其它类改写，口径与其它训练器一致。
        self.num_val_iterations_per_epoch = self.NUM_VAL_ITERATIONS_PER_EPOCH

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            ndim = len(self.configuration_manager.patch_size)
            # U2NetMulti forward 让侧输出保持原生分辨率（d0/d1 全分辨率，
            # d2..d6 依次 1/2, 1/4, 1/8, 1/16, 1/32），因此 DS target 用对应的
            # 多分辨率尺度（标准 nnUNet DS）。相比旧的「7 个全分辨率」，DS 损失
            # 计算量降至约 1/2.6，且 dataloader 不再产生 7 份全分辨率 target 副本。
            return [[1.0] * ndim, [1.0] * ndim,
                    [0.5] * ndim, [0.25] * ndim, [0.125] * ndim,
                    [0.0625] * ndim, [0.03125] * ndim]
        else:
            return None

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        模型没有 .decoder 属性，deep supervision 已通过模型内部的 self.deep_supervision 管理。
        只需同步该标志即可。
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        mod.deep_supervision = enabled

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = U2NET(
            spatial_dims=len(configuration_manager.patch_size),
            in_ch=num_input_channels,
            out_ch=num_output_channels,
            deep_supervision=enable_deep_supervision,
        )
        model.apply(InitWeights_He(1e-2))
        model.apply(init_last_bn_before_add_to_0)

        print(f"U2NetMulti built. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model
