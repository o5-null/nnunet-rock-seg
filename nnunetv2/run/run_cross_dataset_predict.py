"""nnU-Net 跨数据集推理入口。

用途
----
用 **A 数据集训练好的 checkpoint**，对 **B 数据集（目标数据集）的全部图像**跑滑窗推理，
并（默认）对目标数据集的标注做评估。全程零训练：不构造 trainer、不做 batch 探测、不走 DDP、
不装载 CUDA Graph。

与既有入口的区别
----------------
- ``nnUNetv2_predict`` 只认 ``-i <文件夹>``，不感知"目标数据集"概念（不读其 dataset.json /
  splits_final.json），也不产出任何指标；
- ``nnUNetv2_train --val`` 无法跨数据集，且硬性要求 ``checkpoint_final.pth``，并会走完整的
  ``trainer.initialize()``（网络重建 + batch 探测 + DDP + CUDA Graph 装载）。

核心约定：推理配置完全取自模型侧
--------------------------------
``nnUNetPredictor.initialize_from_trained_model_folder`` 读取的是**模型目录内**的 ``plans.json``
与 ``dataset.json``（训练时随 checkpoint 一起保存的副本），因此归一化方案、patch 尺寸、网络结构、
reader 与 file_ending 全部来自模型侧。目标数据集只提供图像与（可选的）标注。

这是跨数据集评估有效的前提——归一化是训练协议的一部分，权重与训练时的输入分布耦合，
不能被目标侧的 plans 覆盖，否则测到的是"归一化错配"而不是"跨域泛化"。

用法示例
--------
.. code-block:: bash

    python -m nnunetv2.run.run_cross_dataset_predict \\
        -m 2 -t 5 -c 2d -f 0 \\
        -tr nnUNetTrainerBatchProbeCUDAGraph -p nnUNetPlans_bs16 \\
        -chk checkpoint_best.pth -gpu 0

``-p`` 可省略：给定 ``-m/-tr/-c`` 后会按 ``<tr>__*__<c>`` 自动发现唯一的模型目录；
存在多个候选时会报错并列出，供显式指定。
"""

import multiprocessing
import os
from typing import List, Optional, Tuple

import torch

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    isdir,
    isfile,
    load_json,
    maybe_mkdir_p,
    subfiles,
)
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.run.run_training import _parse_device_arg, _parse_gpu_list
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

# 单通道数据集里通道 0 的文件名标记，如 shale_00752_0000.png 里的 "_0000"
_CHANNEL_MARKER = '_0000'


# ======================================================================================
# 模型目录 / checkpoint 定位
# ======================================================================================

def resolve_model_folder(model_arg: str, trainer: str, plans: Optional[str],
                         configuration: str) -> Tuple[str, bool]:
    """定位模型目录（含 plans.json + dataset.json + fold_<f>/）。

    参数
    ----
    model_arg : 数据集 id/名，或一个已存在的模型目录绝对路径
    trainer / plans / configuration : 用于拼 ``<trainer>__<plans>__<configuration>``

    返回
    ----
    (model_folder, plans_auto_discovered)

    说明
    ----
    nnU-Net 把 plans 名编码进了结果目录名（``<trainer>__<plans>__<configuration>``），所以定位
    目录必须知道 plans 名。``-p`` 只参与路径拼接，不影响推理内容（真正生效的是该目录内的
    plans.json）。故 ``-p`` 省略时按 ``<tr>__*__<c>`` 自动发现，唯一才采用。
    """
    if isdir(model_arg):
        print(f'[模型目录] 直接使用给定路径: {model_arg}')
        return model_arg, False

    dataset_name = maybe_convert_to_dataset_name(model_arg)
    results_root = join(nnUNet_results, dataset_name)

    if plans is not None:
        folder = join(results_root, f'{trainer}__{plans}__{configuration}')
        if not isdir(folder):
            raise RuntimeError(
                f'模型目录不存在: {folder}\n'
                f'请检查 -tr/-p/-c 是否正确；若要自动发现，请省略 -p。'
            )
        print(f'[模型目录] {folder}')
        return folder, False

    # --- 自动发现 ---
    candidates: List[str] = []
    if isdir(results_root):
        for name in sorted(os.listdir(results_root)):
            if name.startswith(f'{trainer}__') and name.endswith(f'__{configuration}') \
                    and isdir(join(results_root, name)):
                candidates.append(name)

    if len(candidates) == 1:
        print(f'[模型目录] -p 未指定，自动发现唯一匹配: {candidates[0]}')
        return join(results_root, candidates[0]), True

    existing = sorted(d for d in os.listdir(results_root)) if isdir(results_root) else []

    if not candidates:
        raise RuntimeError(
            f'在 {results_root} 下找不到匹配 {trainer}__*__{configuration} 的模型目录。\n'
            f'该数据集下现有目录:\n  ' + '\n  '.join(existing) + '\n'
            f'请检查 -tr/-c，或用 -p 显式指定 plans 标识。'
        )

    raise RuntimeError(
        f'匹配 {trainer}__*__{configuration} 的目录有 {len(candidates)} 个，无法自动确定：\n  '
        + '\n  '.join(candidates) + '\n请用 -p 显式指定 plans 标识。'
    )


def resolve_checkpoint(fold_dir: str, checkpoint: str) -> Tuple[str, Optional[str]]:
    """解析 -chk 参数，返回 (用于初始化的 checkpoint 名, 外部权重路径或 None)。

    两种模式：
    1. ``checkpoint`` 指向一个**已存在的文件** → 直接加载该 ``.pth`` 覆盖网络权重。
       此时仍需要一个目录内的 checkpoint 来完成 ``initialize_from_trained_model_folder``
       （网络结构/训练器信息从它读取）。
    2. 否则视为**文件名**，在 ``fold_dir/`` 下查找。
    """
    local_ckpts = sorted(f for f in (os.listdir(fold_dir) if isdir(fold_dir) else []) if f.endswith('.pth'))

    if isfile(checkpoint):
        if not local_ckpts:
            raise RuntimeError(
                f'模型目录 {fold_dir} 内没有任何 .pth，无法初始化网络结构'
                f'（初始化需要 checkpoint 里的 trainer_name / init_args）。'
            )
        print(f'[checkpoint] 直接加载外部权重: {checkpoint}'
              f'（用 {local_ckpts[0]} 初始化网络结构）')
        return local_ckpts[0], checkpoint

    if not isfile(join(fold_dir, checkpoint)):
        raise RuntimeError(
            f'checkpoint 不存在: {join(fold_dir, checkpoint)}\n'
            f'该 fold 下现有 checkpoint: {local_ckpts if local_ckpts else "（无）"}'
        )
    print(f'[checkpoint] {join(fold_dir, checkpoint)}')
    return checkpoint, None


# ======================================================================================
# 目标数据集：file_ending 与 case 列表
# ======================================================================================

def get_target_file_ending(target_dataset_name: str) -> str:
    """取目标数据集的 file_ending（preprocessed 优先，回退 raw）。"""
    for root in (nnUNet_preprocessed, nnUNet_raw):
        dj = join(root, target_dataset_name, 'dataset.json')
        if isfile(dj):
            return load_json(dj)['file_ending']
    raise RuntimeError(
        f'找不到目标数据集 {target_dataset_name} 的 dataset.json'
        f'（nnUNet_preprocessed 与 nnUNet_raw 下都没有），无法确定 file_ending。'
    )


def collect_cases(target_dataset_name: str, file_ending: str, split: str, fold: int) -> List[str]:
    """收集待推理的 case id 列表。

    split='all'：目标数据集 imagesTr 下的全部单通道图像
    split='val'：目标数据集 splits_final.json 中 fold_<fold> 的 val 键
    """
    images_tr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    if not isdir(images_tr):
        raise RuntimeError(f'目标数据集图像目录不存在: {images_tr}')

    if split == 'val':
        splits_file = join(nnUNet_preprocessed, target_dataset_name, 'splits_final.json')
        if not isfile(splits_file):
            raise RuntimeError(f'--split val 需要 {splits_file}，但该文件不存在。')
        splits = load_json(splits_file)
        if fold >= len(splits):
            raise RuntimeError(f'fold {fold} 超出 {splits_file} 的范围（共 {len(splits)} 折）。')
        cases = list(splits[fold]['val'])
    else:
        marker = _CHANNEL_MARKER + file_ending
        cases = sorted(
            f[:-len(marker)] for f in subfiles(images_tr, suffix=file_ending, join=False)
            if f.endswith(marker)
        )

    if not cases:
        raise RuntimeError(f'目标数据集 {target_dataset_name} 在 split={split} 下没有任何 case。')

    # 校验图像确实存在（val 折的 case id 可能不在 imagesTr 里）
    missing = [c for c in cases if not isfile(join(images_tr, c + _CHANNEL_MARKER + file_ending))]
    if missing:
        raise RuntimeError(
            f'以下 {len(missing)} 个 case 在 {images_tr} 中找不到图像（前 5 个）: {missing[:5]}'
        )
    return cases


def find_target_plans(target_dataset_name: str, model_folder: str) -> str:
    """为**评估**寻找 plans.json：目标集 preprocessed 优先，最后回退模型侧。"""
    pre = join(nnUNet_preprocessed, target_dataset_name)
    if isdir(pre):
        exact = join(pre, 'nnUNetPlans.json')
        if isfile(exact):
            return exact
        cands = sorted(f for f in os.listdir(pre)
                       if f.startswith('nnUNetPlans') and f.endswith('.json'))
        if cands:
            return join(pre, cands[0])
    print('[评估] 目标集没有可用的 plans.json，改用模型侧 plans '
          '（label 定义与 file_ending 由 dataset.json 决定，评估结果不受影响）。')
    return join(model_folder, 'plans.json')


# ======================================================================================
# 主流程
# ======================================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='nnU-Net 跨数据集推理：用 A 数据集的 checkpoint 在 B 数据集上全量推理并评估（零训练）。',
        epilog='示例: python -m nnunetv2.run.run_cross_dataset_predict '
               '-m 2 -t 5 -c 2d -f 0 -tr nnUNetTrainerBatchProbeCUDAGraph '
               '-p nnUNetPlans_bs16 -chk checkpoint_best.pth -gpu 0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-m', '--model-dataset', type=str, required=True,
                        help='模型来源：数据集 id（如 2）、数据集名，或直接指向训练结果目录的绝对路径')
    parser.add_argument('-t', '--target-dataset', type=str, required=True,
                        help='推理目标数据集（id 或名），其 imagesTr 下全部图像都会被推理')
    parser.add_argument('-c', '--configuration', type=str, required=True,
                        help='配置名，如 2d')
    parser.add_argument('-f', '--fold', type=int, required=True,
                        help='折号（对应 fold_<f> 目录，也用于 --split val 取 splits_final.json 的折）')
    parser.add_argument('-tr', '--trainer', type=str, default='nnUNetTrainer',
                        help='训练器类名，默认 nnUNetTrainer')
    parser.add_argument('-p', '--plans', type=str, default=None,
                        help='plans 标识（结果目录名的组成部分）。省略则按 <tr>__*__<c> 自动发现，'
                             '有歧义时报错并列出候选')
    parser.add_argument('-chk', '--checkpoint', type=str, default='checkpoint_final.pth',
                        help='checkpoint：文件名（在 fold_<f>/ 下查找）或已存在的 .pth 路径（直接加载权重）')
    parser.add_argument('--split', type=str, default='all', choices=['all', 'val'],
                        help='推理范围：all=目标数据集 imagesTr 全部（默认）；'
                             'val=目标数据集 splits_final.json 中 fold_<f> 的 val 子集')
    parser.add_argument('--no_eval', action='store_true',
                        help='跳过评估（默认开启评估，对比目标集 labelsTr 写 summary.json）')
    parser.add_argument('--npz', action='store_true',
                        help='同时保存预测概率（.npz）')
    parser.add_argument('--disable_tta', action='store_true',
                        help='关闭镜像 TTA')
    parser.add_argument('-step_size', type=float, default=0.5,
                        help='滑窗步长（占 patch 比例），默认 0.5')
    parser.add_argument('-o', type=str, default=None,
                        help='输出目录。默认 <模型目录>/fold_<f>/xpred_<目标数据集>_<split>/')
    parser.add_argument('--limit', type=int, default=0,
                        help='调试用：只推理前 N 例（0=不限制）')
    parser.add_argument('-device', type=str, default='cuda',
                        help="设备：'cuda'（默认）、'cpu'、'mps' 或 'cuda:N'（N 为物理 GPU 索引）")
    parser.add_argument('-gpu', type=str, default=None,
                        help='[可选] 物理 GPU 索引，如 0。本入口为单卡推理，只接受单个索引')
    parser.add_argument('-npp', type=int, default=3,
                        help='预处理进程数，默认 3')
    parser.add_argument('-nps', type=int, default=3,
                        help='分割导出进程数，默认 3')
    parser.add_argument('-nppred', type=int, default=1,
                        help='并行预测线程数，默认 1')
    parser.add_argument('--disable_progress_bar', action='store_true',
                        help='关闭进度条')
    parser.add_argument('--not_on_device', action='store_true',
                        help='不使用 perform_everything_on_device（显存不足时可开）')
    args = parser.parse_args()

    # --- 环境变量校验 ---
    if nnUNet_raw is None or nnUNet_preprocessed is None or nnUNet_results is None:
        raise RuntimeError('请先设置环境变量 nnUNet_raw / nnUNet_preprocessed / nnUNet_results。')

    # --- 设备解析（复用 run_training.py 的模块级函数） ---
    device = _parse_device_arg(args.device)
    gpu_list = _parse_gpu_list(args.gpu) if args.gpu is not None else None

    if gpu_list is not None:
        if device.type != 'cuda':
            raise ValueError(f'-gpu {args.gpu} 只能与 cuda 设备一起使用（当前 -device {args.device}）。')
        if len(gpu_list) > 1:
            raise ValueError(f'本入口是单卡推理，-gpu 只接受单个物理索引，收到 {args.gpu}。')
        if device.index is not None and device.index != gpu_list[0]:
            raise ValueError(f'-device {args.device} 与 -gpu {args.gpu} 冲突，只能用其一。')
        device = torch.device('cuda', gpu_list[0])

    explicit_indices = gpu_list if gpu_list is not None else (
        [device.index] if (device.type == 'cuda' and device.index is not None) else None)
    if explicit_indices is not None and 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"WARNING: 指定了物理 GPU {explicit_indices}，清除 "
              f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r} 以避免重映射。")
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    if device.type == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())
    elif device.type == 'cuda':
        # 多线程对 GPU 无益，反而引入开销
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # --- 定位模型目录与 checkpoint ---
    if args.plans is not None and isdir(args.model_dataset):
        print('[注意] -m 是目录路径，-p 被忽略。')

    model_folder, plans_auto = resolve_model_folder(
        args.model_dataset, args.trainer, args.plans, args.configuration)

    for required in ('plans.json', 'dataset.json'):
        if not isfile(join(model_folder, required)):
            raise RuntimeError(f'模型目录缺少 {required}: {model_folder}')

    fold_dir = join(model_folder, f'fold_{args.fold}')
    if not isdir(fold_dir):
        raise RuntimeError(f'fold 目录不存在: {fold_dir}')

    chk_name, external_ckpt = resolve_checkpoint(fold_dir, args.checkpoint)

    # --- 目标数据集 ---
    target_name = maybe_convert_to_dataset_name(args.target_dataset)
    file_ending = get_target_file_ending(target_name)
    cases = collect_cases(target_name, file_ending, args.split, args.fold)

    if args.limit and args.limit > 0:
        print(f'[调试] --limit {args.limit}：仅推理前 {args.limit} 例（共 {len(cases)} 例）')
        cases = cases[:args.limit]

    print(f'[目标集] {target_name}  split={args.split}  fold={args.fold}  '
          f'file_ending={file_ending}  待推理 {len(cases)} 例')

    # --- 输出目录 ---
    out_dir = args.o if args.o is not None else \
        join(model_folder, f'fold_{args.fold}', f'xpred_{target_name}_{args.split}')
    maybe_mkdir_p(out_dir)
    print(f'[输出目录] {out_dir}')

    # --- 构造 predictor（配置完全来自模型目录） ---
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=not args.not_on_device,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=not args.disable_progress_bar,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=(args.fold,), checkpoint_name=chk_name)

    # 模式 1：外部 .pth 直接覆盖权重
    if external_ckpt is not None:
        sd = torch.load(external_ckpt, map_location='cpu', weights_only=False)
        weights = sd['network_weights'] if isinstance(sd, dict) and 'network_weights' in sd else sd
        net = predictor.network
        # initialize_from_trained_model_folder 已把网络挂到 predictor.network，但其类型标注为
        # Optional（__init__ 里先置 None），这里显式收窄类型：既能通过静态检查，
        # 也能在初始化异常时给出清晰错误而非 AttributeError。
        assert isinstance(net, torch.nn.Module), '网络未正确初始化，无法加载外部权重'
        net.load_state_dict(weights)
        net.eval()
        print(f'[checkpoint] 权重已覆盖为外部文件: {external_ckpt}')

    # --- 推理 ---
    images_tr = join(nnUNet_raw, target_name, 'imagesTr')
    list_of_lists = [[join(images_tr, c + _CHANNEL_MARKER + file_ending)] for c in cases]

    predictor.predict_from_files(
        list_of_lists,
        out_dir,
        save_probabilities=args.npz,
        overwrite=True,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        # 单阶段推理：不传 folder_with_segs_from_prev_stage（保持其默认 None）
        num_parts=1,
        part_id=0,
        num_processes_prediction=args.nppred,
    )

    # --- 评估 ---
    if not args.no_eval:
        labels_tr = join(nnUNet_raw, target_name, 'labelsTr')
        if not isdir(labels_tr):
            print(f'[评估] 目标集没有 labelsTr（{labels_tr}），跳过评估。')
        else:
            from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2

            dj_file = join(nnUNet_preprocessed, target_name, 'dataset.json')
            if not isfile(dj_file):
                dj_file = join(nnUNet_raw, target_name, 'dataset.json')
            plans_file = find_target_plans(target_name, model_folder)

            # chill=True 是必须的：folder_ref 是目标集全量标注，而 folder_pred 可能只是子集
            # （--split val / --limit）；chill=False 会断言 ref 里每个文件都在 pred 中存在而崩溃。
            compute_metrics_on_folder2(
                folder_ref=labels_tr,
                folder_pred=out_dir,
                dataset_json_file=dj_file,
                plans_file=plans_file,
                output_file=join(out_dir, 'summary.json'),
                num_processes=args.nps,
                chill=True,
            )

    # --- 汇总 ---
    print('=' * 72)
    print('跨数据集推理完成')
    print(f'  模型目录  : {model_folder}' + ('  (plans 自动发现)' if plans_auto else ''))
    print(f'  checkpoint: {args.checkpoint}' + ('  [外部 .pth 直接加载]' if external_ckpt else ''))
    print(f'  目标数据集: {target_name}   split={args.split}')
    print(f'  推理例数  : {len(cases)}')
    print(f'  输出目录  : {out_dir}')

    summary_path = join(out_dir, 'summary.json')
    if isfile(summary_path):
        fg = load_json(summary_path).get('foreground_mean', {})
        print('  评估指标  (foreground_mean):')
        for key in ('Dice', 'IoU', 'Precision', 'Recall', 'TP', 'FP', 'FN', 'TN'):
            if key in fg:
                val = fg[key]
                print(f'    {key:10s}: {val:.4f}' if isinstance(val, float) else f'    {key:10s}: {val}')
        print(f'  summary   : {summary_path}')
    elif not args.no_eval:
        print('  [警告] 未生成 summary.json，请检查评估阶段的输出。')
    print('=' * 72)


if __name__ == '__main__':
    main()
