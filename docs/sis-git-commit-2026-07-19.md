# Git 提交记录 - nnunet-mednext-fork (2026-07-19)

## 概述

将 nnunet-mednext-fork 仓库中的所有未提交变更按功能分组为 8 个 commit。

## 提交列表

| # | Commit Hash | 类型 | 描述 | 包含文件 |
|---|---|---|---|---|
| 1 | `6440ca4` | perf | 启用 TF32 + cuDNN autotune + kernel warmup | run_training.py, nnUNetTrainer.py |
| 2 | `242b883` | feat(cli) | 添加 -epoch 命令行参数覆盖训练轮数 | run_training.py |
| 3 | `2a50df2` | feat(logging) | 增加训练进度指示、tqdm 进度条和混淆矩阵指标 | nnUNetTrainer.py, utils.py |
| 4 | `cf9514e` | fix | 兼容混合 build_network_architecture 签名 + 优化训练配置 | nnUNetTrainer.py |
| 5 | `db6e311` | feat(plans) | 添加 mednext 兼容的网络架构属性 | plans_handler.py |
| 6 | `c84b7de` | chore(deps) | 添加 MedNeXt 模型家族扩展依赖 | pyproject.toml |
| 7 | `27fb06d` | feat | 添加 MedNeXt/Mamba/SAM 模型家族网络架构和训练器 | nets/ + 35个trainer + samedlr + docs |
| 8 | `dcb0f19` | chore | 更新 .gitignore | .gitignore |

## 分拆策略

每个修改过的文件按功能区域拆分，使用 `git add -p` / 备份-恢复策略实现精确 staging：

- **run_training.py** (2 commits): TF32 env → -epoch CLI
- **nnUNetTrainer.py** (3 commits): TF32/warmup → Logging/metrics → Fixes/signatures

## 仓库状态

- `origin/master` 之上 8 个 commit
- working tree clean
- `.codebase-memory/` 和 `.aim/` 已加入 .gitignore
- 临时备份文件 (`*_modified.py`) 已清理

## 备份文件

提交过程中创建的中间备份文件已删除：
- `run_training_modified.py`
- `nnUNetTrainer_modified.py`
