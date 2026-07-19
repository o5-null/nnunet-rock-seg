# 移植 LightM-UNet + UNETR + SwinUNETR Trainer 记录

## 日期
2026-07-09

## 目标
将 MedNeXt-Volcanic-Rocks 项目中 6 个 Trainer 文件移植到 nnunet-mednext-fork，使其与新中间基类 `nnUNetTrainer_MedNeXtBase` / `nnUNetTrainer_MedNeXtNoDeepSupervision` 协同工作。

## 创建的文件

| 文件 | 大小 | 继承关系 |
|------|------|----------|
| `nnUNetTrainerLightMUNet.py` | 5921 B | `nnUNetTrainer_MedNeXtNoDeepSupervision` |
| `nnUNetTrainerLightMUNetTransfer.py` | 4726 B | `nnUNetTrainerLightMUNet` + `TransferLearningBase` |
| `nnUNetTrainerUNETR.py` | 6909 B | `nnUNetTrainer_MedNeXtNoDeepSupervision` |
| `nnUNetTrainerUNETRTransfer.py` | 4781 B | `nnUNetTrainerUNETR` + `TransferLearningBase` |
| `nnUNetTrainerSwinUNETR.py` | 7013 B | `nnUNetTrainer_MedNeXtNoDeepSupervision` |
| `nnUNetTrainerSwinUNETRTransfer.py` | 4833 B | `nnUNetTrainerSwinUNETR` + `TransferLearningBase` |

## 各家族继承结构

### LightM-UNet 家族
- `nnUNetTrainerLightMUNet` → `nnUNetTrainer_MedNeXtNoDeepSupervision` → `nnUNetTrainer_MedNeXtBase` → `nnUNetTrainer`
- 模型: `nnunetv2.nets.LightMUNet.LightMUNet`
- 优化器: Adam, PolyLR
- 无深度监督 (`set_deep_supervision_enabled` 空实现)

### UNETR 家族
- `nnUNetTrainerUNETR` → `nnUNetTrainer_MedNeXtNoDeepSupervision` → `nnUNetTrainer_MedNeXtBase` → `nnUNetTrainer`
- 模型: `monai.networks.nets.UNETR`
- 优化器: AdamW, PolyLR
- 自动对齐 patch_size 为 16 的倍数

### SwinUNETR 家族
- `nnUNetTrainerSwinUNETR` → `nnUNetTrainer_MedNeXtNoDeepSupervision` → `nnUNetTrainer_MedNeXtBase` → `nnUNetTrainer`
- 模型: `monai.networks.nets.SwinUNETR`
- 优化器: AdamW, CosineAnnealingLR
- 自动对齐 patch_size 为 32 的倍数

### Transfer 家族（全部三家的统一模式）
- `*Transfer` → `*BaseTrainer` + `TransferLearningBase`
- TransferLearningBase 提供: Tversky Loss, 渐进解冻, batchgeneratorsv2 增强, early stopping

## 关键改造点
1. 基类替换: `nnUNetTrainerNoDeepSupervision` → `nnUNetTrainer_MedNeXtNoDeepSupervision`
2. `__init__` 参数: 移除 `unpack_dataset` 参数传递（新基类不接受），使用 `device=device` 关键字参数
3. 导入路径: 保留原始 nets 导入路径不变
4. 保留所有 `_100epochs` 子类

## LSP 诊断结果
所有 6 个文件无 `reportCallIssue` 错误。剩余错误均为可接受类型：
- `reportMissingImports`: monai / 跨文件引用（环境未安装）
- `reportIncompatibleMethodOverride`: `build_network_architecture` / `configure_optimizers` 签名差异（继承自原始代码模式）
- `reportOptionalMemberAccess`: optimizer/grad_scaler 类型推断（运行时非 None）
