# Cross-Dataset Inference（跨数据集推理）

> 本页描述 **本 fork 特有** 的入口，上游 nnU-Net v2 没有等价功能。

## 用途

用 **A 数据集训练好的 checkpoint**，对 **B 数据集（目标数据集）的全量图像**跑滑窗推理，
并（默认）对目标数据集的标注做评估。全程零训练：不构造 trainer、不做 batch 探测、不走 DDP、
不装载 CUDA Graph。

与现有入口的差别：

| 入口 | 局限 |
|---|---|
| `nnUNetv2_predict -i <folder>` | 只认输入文件夹，不读目标数据集的 `dataset.json` / `splits_final.json`，不产出指标；`-d`（选模型）与 `-i`（给图像）语义分离，易错 |
| `nnUNetv2_train --val` | 无法跨数据集；硬性要求 `checkpoint_final.pth`；且会走完整 `trainer.initialize()`（网络重建 + BatchProbe 探测 + DDP + CUDA Graph 装载） |
| 本入口 | 一条命令完成「定位模型 → 全量推理 → 评估」，支持任意 checkpoint |

## 快速开始

```bash
python -m nnunetv2.run.run_cross_dataset_predict \
    -m 2 -t 5 -c 2d -f 0 \
    -tr nnUNetTrainerBatchProbeCUDAGraph -p nnUNetPlans_bs16 \
    -chk checkpoint_best.pth -gpu 0
```

Windows 下用 venv 内的解释器：

```powershell
.venv\Scripts\python.exe -m nnunetv2.run.run_cross_dataset_predict `
    -m 2 -t 5 -c 2d -f 0 -tr nnUNetTrainerBatchProbeCUDAGraph -p nnUNetPlans_bs16 `
    -chk checkpoint_best.pth -gpu 0
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-m` / `--model-dataset` | 必填 | 模型来源：数据集 id（如 `2`）、数据集名，或训练结果目录的绝对路径 |
| `-t` / `--target-dataset` | 必填 | 推理目标数据集（id 或名），其 `imagesTr` 下全部图像都会被推理 |
| `-c` / `--configuration` | 必填 | 配置名，如 `2d` |
| `-f` / `--fold` | 必填 | 折号，对应 `fold_<f>` 目录 |
| `-tr` / `--trainer` | `nnUNetTrainer` | 训练器类名 |
| `-p` / `--plans` | 自动发现 | plans 标识（结果目录名的组成部分）。省略时按 `<tr>__*__<c>` glob，唯一才采用，多个候选会报错并列出 |
| `-chk` / `--checkpoint` | `checkpoint_final.pth` | 文件名（在 `fold_<f>/` 下查找），**或一个已存在的 `.pth` 路径**（直接加载权重覆盖） |
| `--split` | `all` | `all` = 目标集 `imagesTr` 全部；`val` = 目标集 `splits_final.json` 中 `fold_<f>` 的 val 子集 |
| `--no_eval` | 关（即默认评估） | 跳过评估 |
| `--npz` | 关 | 同时保存预测概率 |
| `--disable_tta` | 关 | 关闭镜像 TTA |
| `-step_size` | `0.5` | 滑窗步长（占 patch 比例） |
| `-o` | `<模型目录>/fold_<f>/xpred_<目标集>_<split>/` | 输出目录 |
| `--limit` | `0`（不限） | **调试用**：只推理前 N 例 |
| `-device` | `cuda` | `cuda` / `cpu` / `mps` / `cuda:N` |
| `-gpu` | 无 | 物理 GPU 索引，如 `0`。本入口是**单卡推理**，只接受单个索引 |
| `-npp` / `-nps` / `-nppred` | `3` / `3` / `1` | 预处理进程数 / 分割导出进程数 / 并行预测线程数 |

## 设计约定

### 推理配置完全取自模型侧

`nnUNetPredictor.initialize_from_trained_model_folder` 读取的是**模型目录内**的 `plans.json` 与
`dataset.json`（训练时随 checkpoint 保存的副本）。因此归一化方案、patch 尺寸、网络结构、
reader 与 `file_ending` **全部来自模型侧**，目标数据集只提供图像与（可选的）标注。

这是跨数据集评估有效的前提：**归一化是训练协议的一部分**，权重与训练时的输入分布耦合，
不能被目标侧的 plans 覆盖，否则测到的是「归一化错配」而不是「跨域泛化」。

### 目标数据集的最小要求

目标数据集**不需要** `plans.json`、**不需要**跑 `plan_and_preprocess`、**不需要** `splits_final.json`
（`--split all` 时）。只需：

```
nnunet_raw/<目标数据集>/
    imagesTr/<case>_0000<file_ending>     # 图像
    labelsTr/<case><file_ending>          # 标注（--eval 时用）
    dataset.json                          # file_ending / labels / numTraining
```

图像尺寸在不同 case 之间**允许不同**（nnUNet 2D 会按需 pad / 滑窗）。

## 输出

```
<模型目录>/fold_<f>/xpred_<目标集>_<split>/
    <case>.png          # 预测 mask，与原图同名同尺寸
    summary.json        # 评估结果（含 metric_per_case / mean / foreground_mean）
    <case>.npz          # 仅当 --npz
```

终端会打印 `foreground_mean` 的 Dice / IoU / Precision / Recall / TP / FP / FN / TN。

## 注意事项

### `--limit` 只能用于调试

`--limit N` 取的是 `splits[0]['val'][:N]` 或 `imagesTr` 排序后的前 N 例，**不代表整体**。
实测同一 checkpoint：前 20 例 Dice 0.2910，而全量 2265 例为 0.5881（相差 2 倍）。

**绝不可以用 `--limit` 的小样本 Dice 判断模型好坏或对比模型。**

### 评估的 `chill=True`

`compute_metrics_on_folder2` 的 `folder_ref` 是目标集全量标注，而 `folder_pred` 在
`--split val` / `--limit` 下只是子集。`chill=False` 会断言「每个 ref 文件都在 pred 中存在」而崩溃。
入口已固定 `chill=True`；`--split all` 全量推理时两者数量自然一致。

### 本 fork 的 reader 实际是 `NaturalImage2DIO`

`dataset.json` 里的 `overwrite_image_reader_writer: PILImageReaderWriter` 在本 fork **无效**
（`nnunetv2/imageio` 下没有该类），会静默回退按 `file_ending` 自动选择，实际生效的是
**`NaturalImage2DIO`**：

```
Warning: Unable to find ioclass specified in dataset.json: PILImageReaderWriter
Using NaturalImage2DIO as reader/writer
```

`NaturalImage2DIO` 用 `skimage.io.imread`，支持 `.png` / `.bmp` / `.tif`，**保留原 dtype**
（uint16 不会被量化），`write_seg` 自适应 uint8 / uint16。因此 16bit 图像可以直接处理，
不需要降位深。

### 等价性

已验证：同一 checkpoint、同一 fold，本入口与官方 `perform_actual_validation` 的逐例 Dice
完全一致（20 例差异 ≤ 0.0003）。

## 相关

- [Run inference](run-inference.md)
- [Dataset and input format reference](../reference/dataset-format.md)
