"""
nnUNetTrainerCUDAGraphMixin - CUDA Graphs training acceleration mixin
======================================================================

Why
---
With small batch sizes (e.g. batch=4) and fixed patch sizes, GPU utilization is
dominated by CPU-side kernel launch overhead: each step launches thousands of
small kernels, each costing 1-5 us of CPU dispatch. CUDA Graphs capture the whole
forward+loss+backward kernel sequence into a single graph and replay it with one
launch, eliminating the per-kernel CPU overhead. Measured speedup at batch=4 is
~1.3-1.65x (Llama-3.1-8B benchmark), and it is fully compatible with mamba_ssm's
triton kernels because capture records the actual launched kernels (no dynamo
tracing, unlike torch.compile).

Constraints
-----------
- Single GPU only. Falls back to eager training under DDP.
- Networks must have dropout_prob=0 (a captured dropout mask would be frozen).
- When grad_scaler is present (fp16): backward is captured, but
  unscale_/step/update stay OUTSIDE the graph (scaler state is Python control
  flow; capturing it would freeze the loss scale).
- When grad_scaler is None (bf16, e.g. LightMamba2Net): full step benefits most.

Usage
-----
    from nnunetv2.training.nnUNetTrainer.variants.cuda_graph.nnUNetTrainerCUDAGraphMixin import (
        nnUNetTrainerCUDAGraphMixin)

    class MyTrainer(nnUNetTrainerCUDAGraphMixin, BaseTrainer):
        pass

Capture strategy: lazy capture on the FIRST train_step call, using the real
batch's shapes to build static buffers (guarantees shapes match the dataloader
exactly, including deep-supervision target lists). Subsequent steps replay.
"""
import gc

import torch
from torch.amp import autocast


class nnUNetTrainerCUDAGraphMixin:
    # Set to False in a subclass to disable CUDA Graphs (for eager baselines).
    use_cuda_graphs = True
    # 验证独立小 batch（None = 用训练 batch）。CUDA Graph 私有池锁定大 batch
    # 训练激活后，验证若仍用大 batch 会与锁定激活叠加溢出（WDDM 静默 swap
    # 断崖降速）。缩小验证 batch 让验证 forward 激活装进剩余显存，零重捕获
    # 开销。这是默认方案（方案 D）。
    val_batch_size = 8
    # 重型网络训练器集中登记表（batch=4 训练显存实测 ≥ ~4G 者）:
    # M2Net ~9G / SSND2Net ~14G / LightMamba2Net ~6G / LM2Net ~5.5G /
    # SwT2Net ~5G / UNETR2Net ~4G。新增重型训练器只需把类名加入此集合，
    # 无需在每个训练器文件里重复标记（get_val_batch_size 按类名自动检测）。
    HEAVY_MODEL_TRAINERS = frozenset({
        'nnUNetTrainerM2NetBatchProbeCUDAGraph',
        'nnUNetTrainerSSND2NetBatchProbeCUDAGraph',
        'nnUNetTrainerLightMamba2NetBatchProbeCUDAGraph',
        'nnUNetTrainerLightMamba2NetCUDAGraph',
        'nnUNetTrainerLM2NetBatchProbeCUDAGraph',
        'nnUNetTrainerSwT2NetBatchProbeCUDAGraph',
        'nnUNetTrainerUNETR2NetBatchProbeCUDAGraph',
    })
    # 显式标记: True/False 覆盖自动检测；None（默认）→ 按 HEAVY_MODEL_TRAINERS
    # 类名自动判定。重型时验证 batch 自动减半（默认 8 → 4），避免验证 eager
    # forward 叠加在锁定的 CUDA Graph 私有池上把峰值推过物理显存。
    heavy_model = None
    # 验证前是否释放 graph 以腾显存（方案 E，备选）。默认关闭——释放后下一
    # 个 train_step 需重新捕获，warmup 会重跑 cuDNN autotune（SegResNet 实测
    # 每 epoch ~78s，比训练本身还慢），故默认走 val_batch_size 小 batch 方案。
    # 若需大验证 batch 且接受重捕获开销，子类可置 True 并设 val_batch_size=None。
    release_graph_for_validation = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_graph: torch.cuda.CUDAGraph | None = None
        self.static_input: torch.Tensor | None = None
        self.static_target = None          # Tensor or list of Tensors (DS)
        self.static_loss: torch.Tensor | None = None
        self._capture_attempted = False
        # 梯度累积支持: 从宿主类继承 grad_accum_steps（BatchProbe 设置），
        # 无累积时默认为 1（行为与原始 mixin 完全一致）
        self._accum_step_counter = 0

    # ------------------------------------------------------------------ #
    #  Training step override: replay path + lazy capture on first call   #
    # ------------------------------------------------------------------ #
    def train_step(self, batch: dict) -> dict:
        # Move data to device up-front so both eager and graph paths share it
        data = batch['data'].to(self.device, non_blocking=True)
        if isinstance(batch['target'], list):
            target = [t.to(self.device, non_blocking=True) for t in batch['target']]
        else:
            target = batch['target'].to(self.device, non_blocking=True)

        accum = getattr(self, 'grad_accum_steps', 1)
        self._accum_step_counter += 1
        is_boundary = (self._accum_step_counter % accum == 0) if accum > 1 else True

        # --- Replay path (steady state) ---
        if self.cuda_graph is not None:
            # 每次 replay 的完整训练步。⚠️ zero_grad 必须用 set_to_none=False：
            # CUDA Graph 捕获的 backward 重放时写入的是捕获时记录的固定 grad 内存
            # 地址；set_to_none=True 会释放该内存并置 param.grad=None，导致图内
            # backward 的梯度不可见，GradScaler.step 报 "No inf checks were
            # recorded"。就地置零(zero_())保持地址稳定。
            #
            # 梯度累积（accum>1）: 非边界步**跳过** zero_grad → 保留已累积
            # 梯度；replay 的 backward 在已有 grad 上**累加**（autograd 语义，
            # 写入同一地址）。边界步才 zero_grad + step。这样 graph 捕获的
            # forward+loss+backward 被复用 N 次，kernel launch 开销再除以 N。
            if is_boundary:
                self.optimizer.zero_grad(set_to_none=False)
            self._copy_into_static(data, target)
            self.cuda_graph.replay()
            if is_boundary:
                self._graph_optimizer_step()
            return {'loss': self.static_loss.detach().cpu().numpy()}

        # --- Lazy capture on first step (single GPU only) ---
        if (self.use_cuda_graphs and self.device.type == 'cuda' and not self.is_ddp
                and not self._capture_attempted):
            self._capture_attempted = True
            try:
                self._capture_cuda_graph(data, target)
                # capture 已用真实数据完成一次完整训练步（forward+backward+step），
                # 直接返回该 loss，不再重复 replay 同一 batch
                return {'loss': self.static_loss.detach().cpu().numpy()}
            except Exception as e:
                # OOM or unsupported op: drop graph, keep training eager
                self.cuda_graph = None
                self.static_input = self.static_target = self.static_loss = None
                self.print_to_log_file(
                    f"CUDA Graphs: capture FAILED ({type(e).__name__}: {e}). "
                    "Falling back to eager training.")

        # --- Eager path (first step, DDP, CPU, or capture failed) ---
        return super().train_step({'data': data, 'target': target})

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #
    def _do_i_compile(self):
        """CUDA Graphs 与 torch.compile 互斥：两者都消除 kernel launch 开销，
        叠加会冲突（compile 把网络包装成 OptimizedModule，破坏 graph capture 的
        纯净性，且部分 Mamba/SSM 网络在 dynamo 下崩溃）。CUDAGraph 训练器统一禁用。
        """
        return False

    def _use_pin_memory(self) -> bool:
        """CUDAGraph 训练器关闭 pinned memory。

        batchgenerators 的 pin_memory 在独立线程里调用 cudaHostRegister，
        CUDA Graph 捕获期间该操作属于被禁止的 stream 操作，会报
        "operation not permitted when stream is capturing" 并杀死 dataloader
        worker 线程（Blackwell / torch 2.10 实测）。关闭后 worker 只做 CPU
        增广与 H2D 拷贝，不再与捕获冲突。
        """
        return False

    def get_val_batch_size(self):
        """验证 batch 用独立小值（val_batch_size），默认 8。

        验证 forward 激活 ≈ 训练 forward 激活 × (val_batch / train_batch)，
        batch=8 时缩小 8 倍，装进 graph 锁定后剩余显存，避免叠加溢出。
        返回 None 配置时回退到训练 batch（配合 release_graph_for_validation）。
        重型网络（heavy_model=True）时再减半（8 → 4），进一步压低验证峰值。
        """
        vb = getattr(self, 'val_batch_size', None)
        if vb is None:
            return self.batch_size
        heavy = getattr(self, 'heavy_model', None)
        if heavy is None:
            # 未显式标记 → 用集中登记表按类名自动检测（减少每文件重复标记）
            heavy = type(self).__name__ in type(self).HEAVY_MODEL_TRAINERS
        if heavy:
            vb = max(1, vb // 2)
            self.print_to_log_file(
                f"[CUDAGraph] heavy_model=True — val_batch_size halved to {vb}")
        return vb

    def get_dataloaders(self):
        """验证 batch 缩小时按比例放大验证迭代数，保持每 epoch 验证图数恒定
        （nnUNet 默认 50 步 × 训练 batch），保证 fake dice 的统计口径跨训练器可比。
        """
        vb = self.get_val_batch_size()
        if vb != self.batch_size:
            self.num_val_iterations_per_epoch = max(
                1, -(-self.num_val_iterations_per_epoch * self.batch_size // vb))
            self.print_to_log_file(
                f"[CUDAGraph] val_batch_size={vb} (train batch={self.batch_size}) — "
                f"num_val_iterations_per_epoch={self.num_val_iterations_per_epoch}")
        return super().get_dataloaders()

    def _copy_into_static(self, data: torch.Tensor, target):
        """Copy real batch into static buffers (non_blocking, no sync)."""
        self.static_input.copy_(data)
        if isinstance(self.static_target, list):
            for buf, t in zip(self.static_target, target):
                buf.copy_(t)
        else:
            self.static_target.copy_(target)

    def _graph_forward_backward(self):
        """forward + loss + backward. Must run inside autocast context.

        梯度累积（accum>1）时 loss 除以 accum——graph 捕获的静态 forward/
        backward 是"每次 replay 重算"，loss 缩放需在捕获时固定到计算图中，
        与 eager 累积版（BatchProbe.train_step 中 l/accum）尺度一致。
        """
        accum = getattr(self, 'grad_accum_steps', 1)
        with autocast(self.device.type, dtype=self.autocast_dtype,
                      enabled=True, cache_enabled=False):
            output = self.network(self.static_input)
            self.static_loss = self.loss(output, self.static_target)
            if accum > 1:
                self.static_loss = self.static_loss / accum
        if self.grad_scaler is not None:
            self.grad_scaler.scale(self.static_loss).backward()
        else:
            self.static_loss.backward()

    def _graph_optimizer_step(self):
        """Optimizer step OUTSIDE the graph (scaler logic stays dynamic)."""
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

    def _capture_cuda_graph(self, data: torch.Tensor, target):
        """Allocate static buffers from real batch shapes, warm up on a side
        stream, then capture forward+loss+backward into a CUDA Graph.

        Warmup does forward+backward WITHOUT optimizer.step() so model weights
        are NOT touched by warmup noise (random/real data both fine). Capture
        runs on the real first batch; the single optimizer.step() after capture
        consumes the real gradient - equivalent to one normal training step.
        """
        if self.device.type != 'cuda' or self.is_ddp:
            return  # graph disabled on CPU / DDP

        self.print_to_log_file(
            "CUDA Graphs: capturing training step "
            f"(batch={data.shape[0]}, patch={tuple(data.shape[2:])}, "
            f"AMP={self.autocast_dtype}) ...")

        # 1. Static buffers with shapes taken from the real batch (exact match
        #    with the dataloader, including deep-supervision target list).
        #    Fill with the REAL first batch so the captured forward/backward
        #    computes meaningful gradients (no random-data weight pollution).
        self.static_input = torch.empty_like(data)
        if isinstance(target, list):
            self.static_target = [torch.empty_like(t) for t in target]
        else:
            self.static_target = torch.empty_like(target)
        self._copy_into_static(data, target)

        # 2. Warm up on a side stream (3 forward+backward passes, NO step):
        #    allocates the graph's private memory pool, finishes cudnn autotune
        #    and warms up backward kernels WITHOUT updating model weights.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                self._graph_forward_backward()
        torch.cuda.current_stream().wait_stream(s)

        # 释放 warmup 产生的 autograd 图：self.static_loss 是实例属性，会一直
        # 引用最后一次 warmup 的 loss，导致 AccumulateGrad 节点停留在 side
        # stream（capture 在主 stream 时触发 stream mismatch 警告，可能破坏
        # capture）。置 None + 同步后 autograd 图可被回收。
        self.static_loss = None
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # 3. Capture: forward + loss + backward inside the graph
        #    cache_enabled=False -> casts are explicit in-graph kernels, so
        #    replay always casts from the CURRENT weights (no stale fp16 cache)
        self.cuda_graph = torch.cuda.CUDAGraph()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.cuda_graph):
            self._graph_forward_backward()

        # 4. Consume the gradient produced inside the capture (outside graph)
        self._graph_optimizer_step()

        self.print_to_log_file("CUDA Graphs: capture complete, replay mode on.")

    # ------------------------------------------------------------------ #
    #  Validation: release graph to free VRAM, re-capture on next train   #
    # ------------------------------------------------------------------ #
    def on_validation_epoch_start(self):
        """验证前释放训练 CUDA Graph 以腾出显存。

        CUDA Graph 的私有内存池锁定了大 batch 的 forward+backward 全部中间
        激活（graph 不销毁不释放）。验证阶段是 eager forward，若不清空 graph，
        验证激活会与锁定激活叠加，在 16GB 卡上溢出到共享显存导致断崖降速。
        验证只需权重 + forward 激活，释放 graph 后普通池有足够空间。
        下一个 train_step 因 _capture_attempted=False 会自动重新捕获。
        """
        if getattr(self, 'release_graph_for_validation', True):
            self._release_cuda_graph()
        super().on_validation_epoch_start()

    def _release_cuda_graph(self):
        """销毁训练 graph 及其静态 buffer，释放私有池 + 普通池缓存块。

        顺序：先同步（确保最后一次 replay 完成，避免 pending kernel 引用
        graph），再置 None 触发 CUDAGraph.__del__ 释放私有池，gc.collect 兜底
        （确保 __del__ 被执行），empty_cache 归还普通池 free 块，最后同步
        确保释放完成。
        """
        if self.cuda_graph is None:
            return
        self.print_to_log_file(
            "CUDA Graphs: releasing graph before validation "
            "(free VRAM for val forward; re-capture on next train step)")
        torch.cuda.synchronize(self.device)
        self.cuda_graph = None
        self.static_input = None
        self.static_target = None
        self.static_loss = None
        self._capture_attempted = False
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
