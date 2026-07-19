# nnUNetTrainerTransferBase.py - batchgeneratorsv2 移植报告

## 文件路径
`E:\code\nnunet\nnunet-mednext-fork\nnunetv2\training\nnUNetTrainer\nnUNetTrainerTransferBase.py`

## 识别到的 v1 使用位置清单

原始 `MedNeXt-Volcanic-Rocks` 项目的 `TransferLearningBase` 中 **没有直接使用 batchgenerators v1 import**。但存在以下需要替换的"隐式 v1 风格"实现：

| 位置 | 方法 | 原实现方式 | v2 替换方式 |
|------|------|-----------|------------|
| 295-345 | `_apply_spatial_augmentation` | 手动 torch 运算 (gamma/noise/blur/brightness/contrast) | `ComposeTransforms` + `RandomTransform` 包装的 v2 变换 |
| 347-383 | `_gaussian_blur_3d` | 手动 separable conv3d 实现 | `GaussianBlurTransform` (v2) |
| 385-390 | `_adjust_contrast` | 手动均值偏移 | `ContrastTransform` (v2) |
| 116-153 | 增强参数 | 定义了 `rotation_for_DA`, `p_elastic_deform` 等但从未使用 | 现在被 `_build_augmentation_pipeline` 使用 |

## 每个位置的 v2 替换说明

### 1. `_apply_spatial_augmentation` (核心替换)
- **原**: 逐项手动检查概率并应用 torch 运算
- **新**: 构建 `ComposeTransforms` 流水线，通过 `RandomTransform(transform, apply_probability=p)` 控制概率
- 数据流: `torch.Tensor` → `numpy` → `ComposeTransforms(data_dict)` → `torch.Tensor`
- 保留了形状校验、NaN/Inf 保护逻辑

### 2. `_gaussian_blur_3d` → 删除
- 被 `GaussianBlurTransform` (v2) 替代，使用 separable gaussian 实现

### 3. `_adjust_contrast` → 删除
- 被 `ContrastTransform` (v2) 替代

### 4. 新增 `_build_augmentation_pipeline`
- 将原定义但未使用的 `SpatialTransform` 参数 (旋转/缩放/弹性变形) 真正接入流水线
- 使用 v2 `SpatialTransform` 替代 v1 同名类
- 使用 v2 `MirrorTransform` 替代 v1 同名类

### 5. 新增 v2 变换
- `SpatialTransform` - 旋转/缩放/弹性变形
- `MirrorTransform` - 镜像
- `GammaTransform` - Gamma 校正
- `GaussianNoiseTransform` - 高斯噪声
- `GaussianBlurTransform` - 高斯模糊
- `MultiplicativeBrightnessTransform` - 亮度倍增
- `ContrastTransform` - 对比度调整
- `SimulateLowResolutionTransform` - 低分辨率模拟

## MetaLogger API 调研结论

`MetaLogger` 位于 `nnunetv2.training.logging.nnunet_logger`:

| API | 签名 | 说明 |
|-----|------|------|
| `log(key, value, step)` | `(str, Any, int)` | 写入值，step 通常为 epoch |
| `get_value(key, step)` | `(str, step=None)` | 读取值；step=None 返回全部历史列表 |
| `update_config(config)` | `(dict)` | 更新配置 |
| `get_checkpoint()` | `() -> dict` | 返回 `my_fantastic_logging` 字典 |

**关键发现**:
- `self.logger.my_fantastic_logging` 是 `LocalLogger` 的内部属性，`MetaLogger` 不暴露它
- `TransferLearningBase` 原代码未使用 `self.logger`，而是使用 `self.dice_ignore_fn_history` 等实例列表
- 原代码中的 `self.dice_ignore_fn_history` 列表已保留，用于 `_adapt_loss_parameters` 中的 MA 计算
- 若需要与 Phase 2 的 `MetaLogger` 交互，可通过 `self.logger.log('dice_ignore_fn', value, self.current_epoch)` 写入，通过 `self.logger.get_value('dice_ignore_fn', step=None)` 读取

## LSP 诊断结果摘要

**所有 error 均为可接受的 mixin 假阳性**:

1. **`reportAttributeAccessIssue`** (~60 个): `print_to_log_file`, `network`, `device`, `current_epoch`, `grad_scaler`, `output_folder`, `configuration_manager` - 这些属性来自 `nnUNetTrainer` 父类，mixin 在运行时通过多继承获得，pyright 无法静态推断

2. **`reportArgumentType`** (3 个): `BGContrast` 传给 `RandomScalar` 参数 - 这是官方 nnUNetTrainer 中同样使用的模式，运行时正常

3. **`reportCallIssue`** (已修复): `padding_value_seg` 参数在 v2 SpatialTransform 中不存在，已删除

**无真实 error**。所有诊断结果与官方 nnUNetTrainer.py 中的同类用法一致。
