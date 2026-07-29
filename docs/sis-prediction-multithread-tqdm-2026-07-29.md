# 预测多线程 + tqdm 进度条改造

## 修改内容

**文件**: `nnunet-mednext-fork/nnunetv2/inference/predict_from_raw_data.py`

### 1. `predict_from_data_iterator` — 核心改造

- 新增 `num_processes_prediction: int = 1` 参数
- **单线程路径**（默认 `num_processes_prediction=1`）：
  - 惰性迭代原 data_iterator（内存友好）
  - 添加 tqdm 进度条，显示处理图像计数
  - 移除了原循环中每张图的冗余 print（Progress bar 已提供可视反馈）
  - `perform_everything_on_device` 信息移至循环外打印一次
- **多线程路径**（`num_processes_prediction > 1`）：
  - 收集所有预处理的 items 到列表（多线程分发需要全量数据）
  - 使用 `ThreadPoolExecutor` 并行提交预测任务
  - 使用 `threading.Lock` 保护 `predict_logits_from_preprocessed_data`（`load_state_dict` 非线程安全）
  - 通过 `as_completed` 收集结果，实时提交到 export 进程池
  - tqdm 显示总进度和 ETA
- 提取了 `_submit_export()` 和 `_wait_for_export_backpressure()` 辅助函数，消除重复代码

### 2. `predict_from_files` — 参数透传

- 新增 `num_processes_prediction: int = 1` 参数
- 透传到 `predict_from_data_iterator`

### 3. `predict_from_list_of_npy_arrays` — 参数透传

- 新增 `num_processes_prediction: int = 1` 参数
- 透传到 `predict_from_data_iterator`

### 4. CLI 入口 — 新增 `-nppred` 参数

- `predict_entry_point_modelfolder`：新增 `-nppred` argparse 参数
- `predict_entry_point`：新增 `-nppred` argparse 参数
- 支持环境变量 `nnUNet_nppred` 默认值覆盖
- 默认 `1`（完全向后兼容）

## 使用方式

```powershell
# 单线程（默认，行为与原版一致）
.venv\Scripts\python.exe -m nnunetv2.inference.predict_from_raw_data -i input -o output -d 1 -c 2d -f 0

# 多线程预测（例如 4 线程）
.venv\Scripts\python.exe -m nnunetv2.inference.predict_from_raw_data -i input -o output -d 1 -c 2d -f 0 -nppred 4

# 通过环境变量设置默认值
$env:nnUNet_nppred = "4"
```

## 设计说明

- **多线程 vs 多进程**: 使用 threading.Thread 而非 multiprocessing.Process，因为：
  - 无需 pickle 模型权重/配置到子进程
  - 共享 GPU 上下文，避免每个进程独立初始化 CUDA
  - lock 保护 `load_state_dict`（非线程安全），GPU 推理串行化
- **并行收益来源**:
  1. 数据加载（npy 文件 → torch tensor）在线程间并行
  2. 与 export 进程池的 pipelines 更紧密（异步提交流程）
  3. 多图像场景下 GPU 推理串行化开销被数据加载并行抵消
- **向后兼容**: `num_processes_prediction` 默认 1，所有原有 API 签名不变

## 文件变更统计

`predict_from_raw_data.py`: 新增 ~80 行，修改 ~20 行，删除 ~15 行（冗余 print）
