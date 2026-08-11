import multiprocessing
import os
import socket
import sys
import datetime
from typing import Union, Optional, List

# === Default environment optimizations for RTX 3080 (Ampere) ===
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from torch.backends import cudnn


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          continue_training: bool = False,
                          device: torch.device = torch.device('cuda')):
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_trainer_class_by_name(trainer_name)

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    plans["continue_training"] = continue_training
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, device=device)
    return nnunet_trainer


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print("WARNING: Cannot continue training because there seems to be no checkpoint available to "
                               "continue from. Starting a new training...")
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError("Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    # 初始化进程组并设置 NCCL 超时: 探测/DDP 一旦死锁（如 collective 错位），
    # NCCL watchdog 超时后自动 abort 进程，避免 rank 无限阻塞导致 Ctrl+C 也无法
    # 终止（SIGINT 的 KeyboardInterrupt 要等 CUDA 调用返回，而 NCCL 死等 peer）。
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(minutes=3))


def cleanup_ddp():
    dist.destroy_process_group()


def shutdown_trainer_resources(nnunet_trainer):
    """主动关闭 trainer 的后台资源（数据加载器进程）。

    在 KeyboardInterrupt 中断路径调用，防止解释器关闭阶段 batchgenerators
    的 __del__ 访问已被系统关闭的 multiprocessing 句柄，打印
    "Exception ignored ... OSError: [WinError 6] 句柄无效" 等丑陋 traceback。
    与 on_train_end() 中的关闭逻辑一致，静默执行。
    """
    if nnunet_trainer is None:
        return
    try:
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
        from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            try:
                for dl in (getattr(nnunet_trainer, 'dataloader_train', None),
                           getattr(nnunet_trainer, 'dataloader_val', None)):
                    if dl is not None and isinstance(dl, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                        # 非 force 模式：先暂停生产者再排空队列，workers 可干净退出
                        dl._finish(timeout=5)
            finally:
                sys.stdout = old_stdout
    except Exception:
        # 清理是尽力而为，任何失败都不能阻塞退出
        pass


def exit_on_interrupt(msg: str, code: int = 130):
    """打印中断提示后立即终止进程（跳过解释器关闭阶段）。

    sys.exit() 会触发解释器 shutdown，此时所有对象的 __del__ 仍会被调用，
    batchgenerators 的析构函数访问已失效的 multiprocessing 句柄会再次打印
    OSError traceback。os._exit() 直接终止进程，彻底杜绝该类噪音。
    code 默认 130 = 128 + SIGINT(2)，符合"被 Ctrl+C 中断"的惯例。
    """
    print(msg)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, disable_checkpointing, c, val,
            pretrained_weights, npz, val_with_best, world_size, num_epochs=None, probe_mode='auto',
            profile_config=None, gpu_indices=None):
    setup_ddp(rank, world_size)
    # gpu_indices 为 -gpu 列表指定的各进程物理卡（未指定时按 rank 连续编号）。
    # 双保险: 优先读环境变量 NNUNET_DDP_GPUS（主进程 spawn 前写入，子进程继承，
    # 比 mp.spawn args 位置传递更可靠），再回退 args 里的 gpu_indices。
    env_gpus = os.environ.get('NNUNET_DDP_GPUS')
    if env_gpus:
        try:
            gpu_idx = [int(x) for x in env_gpus.split(',') if x.strip()][rank]
        except (ValueError, IndexError):
            gpu_idx = gpu_indices[rank] if gpu_indices else rank
    else:
        gpu_idx = gpu_indices[rank] if gpu_indices else rank
    torch.cuda.set_device(torch.device('cuda', gpu_idx))
    if gpu_indices is not None or env_gpus:
        print(f"[run_ddp] rank {rank}: physical GPU {gpu_idx} (from -gpu {gpu_indices or env_gpus})")
    else:
        print(f"[run_ddp] rank {rank}: physical GPU {gpu_idx} "
              f"(default sequential 0..{world_size - 1}; pass -gpu to pick specific GPUs)")

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p, c,
                                           device=torch.device('cuda', gpu_idx))

    # 命令行 -epoch 覆盖 trainer 默认值（优先级高于 trainer 类自带设置）
    # 必须在 run_training() 调用前设置，确保 LR scheduler 与训练循环都使用新值
    if num_epochs is not None:
        nnunet_trainer.num_epochs = num_epochs

    # 命令行 -profile 覆盖环境变量 nnUNet_profile（None 时保留 trainer 内读取的环境变量值）
    if profile_config is not None:
        nnunet_trainer.profile_config = profile_config

    # 命令行 --probe 控制探测缓存模式（auto/force/only）
    if hasattr(nnunet_trainer, 'probe_cache_mode'):
        nnunet_trainer.probe_cache_mode = probe_mode
        if probe_mode == 'only':
            nnunet_trainer.probe_only = True

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), 'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    try:
        if not val:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
        nnunet_trainer.perform_actual_validation(npz)
    except KeyboardInterrupt:
        shutdown_trainer_resources(nnunet_trainer)
        exit_on_interrupt(f"[DDP rank {rank}] 训练被用户中断 (Ctrl+C)，后台数据加载器已关闭，进程已干净退出。")
    finally:
        cleanup_ddp()


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 num_epochs: Optional[int] = None,
                 device: torch.device = torch.device('cuda'),
                 gpu_list: Optional[List[int]] = None,
                 probe_mode: str = 'auto',
                 profile_config: Optional[str] = None):
    # Enable TF32 on Ampere+ GPUs for ~1.5x matmul speedup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if plans_identifier == 'nnUNetPlans':
        print("\n############################\n"
              "INFO: You are using the old nnU-Net default plans. We have updated our recommendations. "
              "Please consider using those instead! "
              "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if val_with_best:
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    if num_gpus > 1:
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        # DDP 各进程绑定的物理 GPU:
        #  -gpu 0,5,6,7 显式列表 -> 按列表一一映射到 rank
        #  -device cuda:N / 默认 -> 从 N(默认 0) 起连续 num_gpus 张卡
        if gpu_list is not None:
            assert len(gpu_list) == num_gpus, \
                f'-gpu {gpu_list} has {len(gpu_list)} entries but num_gpus={num_gpus}. They must match.'
            gpu_indices = list(gpu_list)
        else:
            base_gpu = device.index if device.index is not None else 0
            gpu_indices = list(range(base_gpu, base_gpu + num_gpus))
        print(f"DDP: {num_gpus} processes -> physical GPUs {gpu_indices}")

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        # 环境变量传递 GPU 列表（spawn 子进程继承 env，比 args 位置传递更可靠）
        os.environ['NNUNET_DDP_GPUS'] = ','.join(str(i) for i in gpu_indices)
        if gpu_list is None:
            print(f"WARNING: -gpu not specified — binding ranks to sequential GPUs {gpu_indices}. "
                  f"On shared DGX (GPU1-4 may host llama-server) use '-gpu 0,5,6,7' etc.")

        procs = mp.spawn(run_ddp,
                         args=(
                             dataset_name_or_id,
                             configuration,
                             fold,
                             trainer_class_name,
                             plans_identifier,
                             disable_checkpointing,
                             continue_training,
                             only_run_validation,
                             pretrained_weights,
                             export_validation_probabilities,
                             val_with_best,
                             num_gpus,
                             num_epochs,
                             probe_mode,
                             profile_config,
                             gpu_indices),
                         nprocs=num_gpus,
                         join=False)
        try:
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            # 主进程 Ctrl+C: 主动终止所有 rank 子进程，防止孤儿进程滞留。
            # spawn 子进程是独立进程，主进程 os._exit 不会带走它们；rank 若
            # 阻塞在 NCCL 同步点也无法自行响应 SIGINT → 必须由主进程强制 kill。
            alive = [p for p in procs if p.is_alive()]
            for p in alive:
                p.terminate()
            for p in alive:
                p.join(timeout=15)
            exit_on_interrupt(f"训练被用户中断 (Ctrl+C)，{len(alive)} 个 rank 进程已强制终止。")
    else:
        # 单进程训练: gpu_list 若给出必须恰好 1 张（-gpu 1 即 cuda:gpu_list[0]）
        assert gpu_list is None or len(gpu_list) == 1, \
            f'-gpu {gpu_list} implies DDP (multi-GPU) but num_gpus={num_gpus}. Use -num_gpus {len(gpu_list)} or a single -gpu index.'
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, continue_training, device=device)

        # 命令行 -epoch 覆盖 trainer 默认值（优先级高于 trainer 类自带设置）
        # 必须在 run_training() 调用前设置，确保 LR scheduler(PolyLRScheduler 使用 self.num_epochs)
        # 与训练主循环 for epoch in range(..., self.num_epochs) 都使用新值
        if num_epochs is not None:
            nnunet_trainer.num_epochs = num_epochs

        # 命令行 -profile 覆盖环境变量 nnUNet_profile（None 时保留 trainer 内读取的环境变量值）
        if profile_config is not None:
            nnunet_trainer.profile_config = profile_config

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        # 命令行 --probe 控制探测缓存模式（auto/force/only）
        if hasattr(nnunet_trainer, 'probe_cache_mode'):
            nnunet_trainer.probe_cache_mode = probe_mode
            if probe_mode == 'only':
                nnunet_trainer.probe_only = True

        assert not (continue_training and only_run_validation), 'Cannot set --c and --val flag at the same time. Dummy.'

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        try:
            if not only_run_validation:
                nnunet_trainer.run_training()

            if val_with_best:
                nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
            nnunet_trainer.perform_actual_validation(export_validation_probabilities)
        except KeyboardInterrupt:
            shutdown_trainer_resources(nnunet_trainer)
            exit_on_interrupt("训练被用户中断 (Ctrl+C)，后台数据加载器已关闭，进程已干净退出。")


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Set the device for training: 'cuda' (GPU, default), 'cpu', 'mps', "
                             "or 'cuda:N' to pick a specific physical GPU by index (e.g. 'cuda:1'). "
                             "Index selects the physical GPU directly, no CUDA_VISIBLE_DEVICES needed. "
                             "Alternative: use -gpu N.")
    parser.add_argument('-gpu', type=str, default=None, required=False,
                        help='[OPTIONAL] Select physical GPU(s) by index directly, comma-separated. '
                             'Single: -gpu 1 (= -device cuda:1). Multi-GPU DDP: -gpu 0,5,6,7 uses '
                             'exactly those 4 physical GPUs (-num_gpus is derived from the list). '
                             'Overrides CUDA_VISIBLE_DEVICES remapping.')
    parser.add_argument('-epoch', type=int, default=None, required=False,
                        help='[OPTIONAL] 覆盖 trainer 的训练 epoch 数。不传则使用 trainer 类自带默认值 '
                             '(nnUNetTrainer 默认 1000，变体如 nnUNetTrainer_100epochs 带 100)。'
                             '显式传入会覆盖变体类设置，例如 -tr nnUNetTrainer_100epochs -epoch 50 实际跑 50 epoch。')
    parser.add_argument('--probe', type=str, default='auto', required=False,
                        choices=['auto', 'force', 'only'],
                        help='[OPTIONAL] batch 探测缓存模式 (nnUNetTrainerBatchProbe 系列): '
                             'auto=环境未变则用缓存(默认), force=忽略缓存强制重新探测, '
                             'only=只探测不训练(结果写缓存后退出)')
    parser.add_argument('-profile', type=str, default=None, required=False,
                        help='[OPTIONAL] 启用 torch.profiler 性能分析。格式: '
                             '"auto"=默认采样窗口(wait=5, warmup=2, active=3), '
                             '或 "wait,warmup,active" 自定义如 "5,2,3"。'
                             'trace 导出到 output_folder/profile/。'
                             '优先级高于环境变量 nnUNet_profile。')
    args = parser.parse_args()

    # --- 解析 -device 与 -gpu，支持直接按物理索引选卡（无需 CUDA_VISIBLE_DEVICES 重映射） ---
    def _parse_device_arg(device_str: str) -> torch.device:
        """解析 -device 字符串: cpu / mps / cuda / cuda:N（N 为物理 GPU 索引）。"""
        if device_str == 'cpu':
            return torch.device('cpu')
        if device_str == 'mps':
            return torch.device('mps')
        if device_str == 'cuda':
            return torch.device('cuda')
        if device_str.startswith('cuda:'):
            try:
                idx = int(device_str.split(':', 1)[1])
                return torch.device('cuda', idx)
            except ValueError as e:
                raise ValueError(f'Invalid -device value: {device_str!r}. Expected cuda:N with N an integer.') from e
        raise ValueError(f'-device must be one of cpu / cuda / cuda:N / mps. Got: {device_str!r}')

    def _parse_gpu_list(gpu_str: str) -> List[int]:
        """解析 -gpu 参数: 逗号分隔的物理 GPU 索引列表, 如 '0,5,6,7' -> [0,5,6,7]。"""
        parts = [p.strip() for p in gpu_str.split(',') if p.strip()]
        if not parts:
            raise ValueError(f'Invalid -gpu value: {gpu_str!r}. Expected comma-separated GPU indices.')
        try:
            indices = [int(p) for p in parts]
        except ValueError as e:
            raise ValueError(f'Invalid -gpu value: {gpu_str!r}. Expected comma-separated integer GPU indices.') from e
        if len(set(indices)) != len(indices):
            raise ValueError(f'Duplicate GPU index in -gpu {gpu_str!r}.')
        return indices

    device = _parse_device_arg(args.device)
    gpu_list = _parse_gpu_list(args.gpu) if args.gpu is not None else None

    # -gpu 与 -device cuda:M 冲突检测；-gpu 仅对 cuda 有效
    if gpu_list is not None:
        if device.type != 'cuda':
            raise ValueError(f'-gpu {args.gpu} is only valid together with a cuda device. Got -device {args.device}.')
        if device.index is not None:
            # -device cuda:M 只能与单元素且等值的 -gpu 列表共存
            if len(gpu_list) != 1 or gpu_list[0] != device.index:
                raise ValueError(f'Conflicting GPU selection: -device {args.device} vs -gpu {args.gpu}. Use only one of them.')
        device = torch.device('cuda', gpu_list[0])

    # -gpu 列表隐含 DDP 进程数: 与显式 -num_gpus 对齐检查（默认 1 时自动派生）
    if gpu_list is not None and len(gpu_list) > 1:
        if args.num_gpus != 1 and args.num_gpus != len(gpu_list):
            raise ValueError(f'-gpu {args.gpu} selects {len(gpu_list)} GPUs, but -num_gpus {args.num_gpus} was given. '
                             f'Remove -num_gpus or align it with the -gpu list length.')
        args.num_gpus = len(gpu_list)

    # 显式指定物理索引（-gpu 列表 或 -device cuda:N）时，清除 CUDA_VISIBLE_DEVICES 重映射，
    # 保证索引即物理 GPU 序号（torch 与 NVML 索引对齐）
    explicit_indices = gpu_list if gpu_list is not None else (
        [device.index] if (device.type == 'cuda' and device.index is not None) else None)
    if explicit_indices is not None:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"WARNING: -gpu/-device cuda:N selects physical GPU(s) {explicit_indices}; "
                  f"unsetting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r} to avoid remapping.")
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    if args.device in ['cpu', 'mps']:
        # cpu: 让 torch 用满线程；mps 无特殊设置
        if device.type == 'cpu':
            torch.set_num_threads(multiprocessing.cpu_count())
    else:
        # cuda 训练: 多线程对 GPU 无益，反而引入开销
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    try:
        run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                     args.num_gpus, args.npz, args.c, args.val, args.disable_checkpointing, args.val_best,
                     num_epochs=args.epoch, device=device, gpu_list=gpu_list,
                     probe_mode=args.probe, profile_config=args.profile)
    except KeyboardInterrupt:
        # 兜底：正常情况下 KeyboardInterrupt 已在 run_training() 内部处理，
        # 此处覆盖 trainer 尚未创建（如 argparse/环境准备阶段）就中断的场景。
        exit_on_interrupt("训练被用户中断 (Ctrl+C)，进程已干净退出。")


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # reduces the number of threads used for compiling. More threads don't help and can cause problems
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    # multiprocessing.set_start_method("spawn")
    run_training_entry()
