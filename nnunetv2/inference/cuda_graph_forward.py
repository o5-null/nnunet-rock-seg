# -*- coding: utf-8 -*-
"""CudaGraphForward —— 把固定 shape 的网络 forward 捕获成 CUDA Graph 并 replay。

为什么需要
----------
验证推理的瓶颈不是算力而是 kernel launch。2026-09-13 实测（RTX3080 /
torch 2.10.0+cu128 / SwT2Net 180.6M）：单次 (4,1,256,256) 前向 ~0.376 s，
且耗时与 batch 无关（bs1 0.343 / bs2 0.396 / bs4 0.376）——典型的 CPU 端
launch-bound（网络发出数千个细碎 kernel：roll/softmax/elementwise）。
训练侧用 CUDAGraphMixin 把 fwd+bwd 捕获成图（eager 375ms → 图 127ms），
验证侧此前没有图，于是每张图 9 tile × 4 mirror TTA = 36 次前向 × ~0.1 s
≈ 3.4 s/图，2265 张要 ~2.4 h。

本类只捕获 forward（推理没有 backward/optimizer/GradScaler），复用训练侧
已验证的 capture 约定：side-stream warmup → torch.cuda.graph 捕获 → 私有池
外置检测。

正确性防线（任一失败即永久回退 eager，绝不让验证算错或变慢）
------------------------------------------------------------
1. 捕获前 side-stream warmup 3 次：完成 cudnn autotune / triton JIT / 惰性
   初始化，避免 autotune 落进捕获区（会报 "operation not permitted when
   stream is capturing"）。
2. 捕获后立即 replay，与同输入下的 eager 结果比对（allclose），超差即弃用
   ——把"静默算错"变成"当场发现"。
3. 速度收益检测：replay 必须比 eager 明显更快。私有池被 WDDM 静默外置到共享
   内存时 replay 走 PCIe（实测慢 ~8x），此时必须弃用而不是继续用。这是外置
   检测的主力判据。
4. 私有池外置检测（整卡空闲显存 < spill_free_ratio）：与训练侧
   CUDAGraphMixin._GRAPH_SPILL_FREE_RATIO 同口径的补充判据。注意外置时物理
   显存被释放、free 反而升高，故它只覆盖"整卡被压满导致驱动外置"这一情形。

用法
----
    g = CudaGraphForward(net, torch.device('cuda:0'), autocast_dtype=torch.float16,
                         logger=print)
    y = g(x)          # 首次调用惰性捕获；之后同 shape 走 replay
"""
import gc
import time
from typing import Callable, Optional

import torch


class CudaGraphForward:
    """固定 shape 前向的 CUDA Graph 包装器（推理专用）。

    线程不安全；capture 要求独占当前 stream 的分配状态，故不要在多个线程里
    并发调用首次 forward。
    """

    def __init__(self,
                 net: torch.nn.Module,
                 device: torch.device,
                 autocast_dtype: torch.dtype = torch.float16,
                 spill_free_ratio: float = 0.02,
                 min_speedup: float = 1.1,
                 max_shapes: int = 3,
                 atol: float = 1e-2,
                 rtol: float = 1e-2,
                 logger: Optional[Callable[[str], None]] = None):
        self.net = net
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.spill_free_ratio = spill_free_ratio
        # replay 至少要快这么多倍才认为图真的生效（正常应有数倍收益，1.1x 只
        # 用来排除"图没生效/被外置"这类病态情形，不会因计时噪声误杀）
        self.min_speedup = min_speedup
        self.max_shapes = max_shapes
        self.atol = atol
        self.rtol = rtol
        self._log_fn = logger
        self.enabled = True
        self._graphs: dict = {}

    # ------------------------------------------------------------------ #
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or x.device.type != 'cuda':
            return self.net(x)
        # ⚠️ 必须在 self.device 的设备上下文里捕获/重放：torch.cuda.graph 依据
        # torch.cuda.current_device() 选择捕获流，而调用方可能只把张量放到
        # cuda:N 却没 set_device（nnUNet 全仓没有 torch.cuda.set_device）——
        # 此时捕获流落在 cuda:0 而 op 跑在 cuda:N，结果是**空图**（trivial op
        # 静默无效）或 cudaErrorStreamCaptureInvalidated（复杂网络直接失败）
        # （2026-09-13 在 cuda:1 上实测复现）。torch.cuda.device() 只在该块内
        # 切换当前设备，退出即还原，不污染调用方状态。
        with torch.cuda.device(self.device):
            return self._call_on_device(x)

    def _call_on_device(self, x: torch.Tensor) -> torch.Tensor:
        key = (tuple(x.shape), x.dtype)
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= self.max_shapes:
                self._log(f"[CudaGraphForward] shape {tuple(x.shape)} 超出 "
                          f"max_shapes={self.max_shapes}，该批走 eager")
                return self.net(x)
            entry = self._try_capture(key, x)
            if entry is None:
                # 捕获/自检失败 → 永久回退 eager，绝不打断验证
                self.enabled = False
                return self.net(x)
            self._graphs[key] = entry

        # 同一 stream 上 copy_ → replay → 读 output 天然有序（无需显式同步）。
        # 必须返回 clone：静态输出缓冲会被下一次 replay 覆写，而调用方
        # (_internal_maybe_mirror_and_predict) 会对返回值做累加。
        entry['input'].copy_(x, non_blocking=True)
        entry['graph'].replay()
        return entry['output'].clone()

    # ------------------------------------------------------------------ #
    def _autocast_forward(self, x: torch.Tensor) -> torch.Tensor:
        """图内/图外共用的前向。

        autocast 用 cache_enabled=False：权重的 fp16 cast 变成显式 kernel 落进
        图里，replay 时按**当前**权重重新 cast。若用默认 cache_enabled=True，
        捕获时缓存的 fp16 权重副本会被固化进图，而权重随后可能被 EMA /
        load_state_dict 原地改写 → 静默算错。
        """
        with torch.autocast(self.device.type, dtype=self.autocast_dtype,
                            enabled=True, cache_enabled=False):
            return self.net(x)

    def _try_capture(self, key, x: torch.Tensor) -> Optional[dict]:
        do = self.device
        shape, dtype = key
        self._log(f"[CudaGraphForward] 捕获验证 forward 图: shape={shape}, "
                  f"dtype={dtype}, AMP={self.autocast_dtype} ...")
        try:
            # 0) 回收 allocator 缓存/碎片并同步，尽量给 graph 私有池留连续显存
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(do)

            static_in = torch.empty(shape, dtype=dtype, device=do)
            static_in.copy_(x)

            # 1) side-stream warmup：完成 autotune / JIT / 惰性初始化
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._autocast_forward(static_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize(do)

            # 1b) eager 参考输出 + 参考耗时（供自检 2/3 使用）
            with torch.inference_mode():
                for _ in range(2):
                    self._autocast_forward(static_in)
                torch.cuda.synchronize(do)
                t0 = time.perf_counter()
                for _ in range(5):
                    ref = self._autocast_forward(static_in)
                torch.cuda.synchronize(do)
                eager_ms = (time.perf_counter() - t0) / 5 * 1000.0
                ref = ref.clone()

            # 2) 捕获（显存顺序：捕获会自动同步，前向的中间激活全部落在图的私有池）
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    static_out = self._autocast_forward(static_in)

            # 自检 2: 数值一致性
            graph.replay()
            torch.cuda.synchronize(do)
            if not torch.allclose(static_out, ref, atol=self.atol, rtol=self.rtol):
                diff = (static_out - ref).abs().max().item()
                raise RuntimeError(
                    f"replay 与 eager 结果不一致 (max|diff|={diff:.3e}) — 拒绝使用该图")

            # 自检 3: 速度收益
            for _ in range(2):
                graph.replay()
            torch.cuda.synchronize(do)
            t0 = time.perf_counter()
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize(do)
            replay_ms = (time.perf_counter() - t0) / 5 * 1000.0
            speedup = (eager_ms / replay_ms) if replay_ms > 0 else 0.0
            if speedup < self.min_speedup:
                raise RuntimeError(
                    f"replay 无收益 (eager {eager_ms:.1f}ms vs replay "
                    f"{replay_ms:.1f}ms = {speedup:.2f}x < {self.min_speedup}x) — "
                    "图可能已被驱动外置到共享内存")

            # 自检 4: 私有池外置（与训练侧同口径的补充判据）
            free = self._free_ratio()
            if free is not None and free < self.spill_free_ratio:
                raise RuntimeError(
                    f"捕获后整卡空闲仅 {free:.1%} (< {self.spill_free_ratio:.0%})，"
                    "私有池可能已被驱动外置")

            free_txt = f"整卡空闲 {free:.1%}" if free is not None else "整卡空闲 n/a"
            self._log(f"[CudaGraphForward] 捕获成功: shape={shape}，"
                      f"eager {eager_ms:.1f}ms → replay {replay_ms:.1f}ms "
                      f"({speedup:.2f}x)，{free_txt}")

            # 图内已用 cache_enabled=False、不依赖 autocast 缓存；清掉捕获前
            # 可能残留的 fp16 权重副本，避免其他 eager 路径复用陈旧副本
            if hasattr(torch, 'clear_autocast_cache'):
                torch.clear_autocast_cache()
            return {'graph': graph, 'input': static_in, 'output': static_out}
        except Exception as e:  # noqa: BLE001 — 捕获失败必须降级，不能中断验证
            self._log(f"[CudaGraphForward] 弃用 CUDA Graph，回退 eager 前向 "
                      f"({type(e).__name__}: {e})")
            gc.collect()
            torch.cuda.empty_cache()
            return None

    def _free_ratio(self) -> Optional[float]:
        """当前设备空闲显存占比（0-1）；查询失败返回 None。"""
        try:
            free_b, total_b = torch.cuda.mem_get_info(self.device)
            return (free_b / total_b) if total_b else None
        except Exception:  # noqa: BLE001
            return None

    def _log(self, msg: str):
        if self._log_fn is not None:
            self._log_fn(msg)
