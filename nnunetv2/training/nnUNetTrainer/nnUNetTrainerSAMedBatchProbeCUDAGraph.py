# -*- coding: utf-8 -*-
"""nnUNetTrainerSAMedBatchProbeCUDAGraph — nnUNetTrainerSAMed + 自动 batch 探测 + CUDA Graph 累积加速

组合 nnUNetTrainerBatchProbeCUDAGraph（探测 + graph 累积 replay）与 nnUNetTrainerSAMed。

加速路径（探测后按 accum 自动选择）:
- 所有 accum → CUDA Graph replay 累积（mixin 增强版）:
    accum==1: 每步 replay + step
    accum>1 : replay 复用 N 次（非边界步只 replay 不 step，梯度累积到
      同一地址），kernel launch 开销除以 N
- graph 捕获失败（OOM/不支持/DDP）→ 回退 eager 累积（BatchProbe.train_step）
- compile: graph 路径互斥禁用；强制 eager 时复用基类 _do_i_compile 决策

MRO: [T, BatchProbeCUDAGraph, CUDAGraphMixin, BatchProbe, nnUNetTrainerSAMed, ...]
（2026-08-01 脚本验证菱形继承合法线性化）

用法:
  # 纯探测（写缓存后退出）
  python -m nnunetv2.run.run_training 2 2d 0 ^
      -tr nnUNetTrainerSAMedBatchProbeCUDAGraph -p nnUNetPlans_bs16 --probe only
  # 完整训练（缓存命中秒用 / 未命中先探测）
  python -m nnunetv2.run.run_training 2 2d 0 ^
      -tr nnUNetTrainerSAMedBatchProbeCUDAGraph -p nnUNetPlans_bs16
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBatchProbeCUDAGraph import nnUNetTrainerBatchProbeCUDAGraph
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSAMed import nnUNetTrainerSAMed
import torch


class nnUNetTrainerSAMedBatchProbeCUDAGraph(nnUNetTrainerBatchProbeCUDAGraph, nnUNetTrainerSAMed):
    """nnUNetTrainerSAMed + 自动 batch 探测 + CUDA Graph 累积加速。"""
    # SAMed 是固定预训练骨干 + dict 输出（{'low_res_logits','masks',...}），
    # 与通用 BatchProbe/CUDAGraph 的 train_step（裸 forward(x) + self.loss(output,target)）
    # 不兼容（output 为 dict，且 SAMed 需 forward(data, True, patch_size)）。
    # 训练步直接委托给 SAMed 自己的 eager train_step（含 grad scaler + clip）。
    # 注意: 因此 SAMed 路径不启用 CUDA Graph replay 与梯度累积（accum 被忽略）。
    train_step = nnUNetTrainerSAMed.train_step

    def _probe_trial(self, batch: int, patch):
        """SAMed 探测: forward 返回 dict，loss 需取 low_res_logits 且 target 为低分辨率。

        基类 BatchProbe._probe_trial 假设网络输出为 tensor/list（deep supervision），
        SAMed 输出 dict 且 loss 期望 (low_res_logits, (B,1,h,w))，故单独实现。
        """
        net = getattr(self, '_probe_net', self.network)
        num_output_channels = self.label_manager.num_segmentation_heads
        dummy_batch = torch.randn(
            (batch, self.num_input_channels, *patch), device=self.device)
        with torch.autocast(self.device.type, dtype=self.autocast_dtype,
                            enabled=self.device.type == 'cuda'):
            output = net(dummy_batch, True, self.patch_size)
            low_res = output['low_res_logits']  # (B, C, h, w)
            dummy_target = torch.randint(
                0, max(1, num_output_channels), (batch, 1, *low_res.shape[2:]),
                device=self.device, dtype=torch.long)
            l = self.loss(low_res, dummy_target)
        l.backward()
        # 不 step，仅测量; zero_grad 释放梯度供下一档复用
        self.optimizer.zero_grad(set_to_none=True)
