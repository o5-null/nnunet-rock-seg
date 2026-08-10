# -*- coding: utf-8 -*-
"""Mamba2 共享层封装（2026-08-01）

背景: 将原使用 mamba_ssm.Mamba（Mamba1）的网络层切换到 Mamba2
（mamba_ssm.modules.mamba2.Mamba2）。Mamba2 接口差异:
- headdim: 必填，d_model×expand 需整除 headdim（内部多头划分）
- chunk_size: 默认 256 导致 SSD kernel smem 37KB+ → occupancy 8-16%；
  128 提速 ~16% 且显存 -15%（实测 2026-08-01, RTX 3080）。必须为 2 的幂。
- bf16 保护: Mamba2 SSM 离散化（A_log/dt 指数）在 fp16 下动态范围不足易
  NaN；bf16（8 位指数 = fp32 级）可避免，且不影响外部 fp16 AMP 上下文。

用法（替换原 Mamba1 层）:
    from nnunetv2.nets.mamba2_wrapper import MambaLayer
    self.mamba = MambaLayer(input_dim=C, output_dim=C)   # 接口与 Mamba1 兼容

get_nheaddim: 从 light_mamba2net.MambaLayer 提取（headdim 计算逻辑）。
"""
import torch
from torch import nn
from mamba_ssm.modules.mamba2 import Mamba2 as Mamba


def get_nheaddim(d_model: int, expand: int) -> int:
    """计算 Mamba2 headdim: d_model×expand 能整除的最大 i（headdim 约束）。"""
    nheaddim = 1
    for i in range(1, int((d_model * expand / 8))):
        if d_model * expand / i % 8 == 0:
            nheaddim = i
    return nheaddim


class MambaLayer(nn.Module):
    """Mamba2 层，接口与 Mamba1（mamba_ssm.Mamba）兼容。

    参数与 Mamba1 相同（d_state/d_conv/expand），内部自动计算 headdim，
    chunk_size 固定 128（性能最优），forward 用 bf16 保护 SSM 稳定性。
    """

    def __init__(self, input_dim: int, output_dim: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 chunk_size: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size,   # 128: smem 降低 → occupancy 提升（实测更快）
            headdim=get_nheaddim(input_dim, expand),
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        # bf16 保护: Mamba2 SSM 离散化（A_log/dt 指数）fp16 下易 NaN
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
            x_mamba = self.mamba(x_norm)
        x_mamba = x_mamba.to(x.dtype) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out
