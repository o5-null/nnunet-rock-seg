import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from threading import Thread
from time import time, sleep
from typing import Tuple, Union, List

from tqdm import tqdm

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
try:
   from torch import GradScaler           # torch >= 2.3
   TORCH_HAS_OLD_GRADSCALER = False
except ImportError:
   from torch.cuda.amp import GradScaler  # torch < 2.3
   TORCH_HAS_OLD_GRADSCALER = True
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class nnUNetTrainer(object):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)

        # apex predator of grug is complexity
        # complexity bad
        # say again:
        # complexity very bad
        # you say now:
        # complexity very, very bad
        # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
        # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
        # one day code base understandable and grug can get work done, everything good!
        # next day impossible: complexity demon spirit has entered code and very dangerous situation!

        # OK OK I am guilty. But I tried.
        # https://www.osnews.com/images/comics/wtfm.jpg
        # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            if k in locals():
                self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        continue_training = plans.pop("continue_training")
        self.continue_training = continue_training
        logger_config = {"plans": plans, "configuration": configuration, "fold": fold, "dataset": dataset_json}
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed.is_set() else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results.is_set() else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}') if self.output_folder_base is not None else None

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier) \
            if self.preprocessed_dataset_folder_base is not None else None
        self.dataset_class = None  # -> initialize
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnUNet_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                if self.is_cascaded else None

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = True
        self.autocast_dtype = torch.float16  # 默认 fp16 AMP，子类可改为 bf16

        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = (GradScaler("cuda") if not TORCH_HAS_OLD_GRADSCALER else GradScaler()) if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = MetaLogger(self.output_folder, continue_training)
        self.logger.update_config(logger_config)

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self.print_to_log_file("Initializing trainer...")
            # Enable TF32 + cuDNN autotune on CUDA for ~15-25% speedup
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision('high')
                self.print_to_log_file("Enabled TF32 matmul, TF32 cuDNN, cuDNN benchmark, "
                                       "and float32 matmul precision 'high'")
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                    self.dataset_json)

            self.print_to_log_file("Building network architecture...")
            sig = inspect.signature(self.build_network_architecture)
            if 'plans_manager' in sig.parameters:
                params = list(sig.parameters.keys())
                if len(params) >= 2 and params[1] != 'configuration_manager':
                    # Hybrid signature: (plans_manager, dataset_json, configuration_manager, ...)
                    # Pass dataset_json in the second slot instead of configuration_manager
                    self.network = self.build_network_architecture(
                        self.plans_manager,
                        self.dataset_json,
                        self.configuration_manager,
                        self.num_input_channels,
                        self.enable_deep_supervision
                    ).to(self.device)
                else:
                    self.network = self.build_network_architecture(
                        self.plans_manager,
                        self.configuration_manager,
                        self.num_input_channels,
                        self.label_manager.num_segmentation_heads,
                        self.enable_deep_supervision
                    ).to(self.device)
            else:
                warnings.warn(
                    f"Trainer {self.__class__.__name__} uses the old build_network_architecture signature. "
                    "Please update to the new signature: "
                    "build_network_architecture(plans_manager, configuration_manager, "
                    "num_input_channels, num_output_channels, enable_deep_supervision). "
                    "The old signature will be removed in a future version.",
                    DeprecationWarning, stacklevel=2,
                )
                self.network = self.build_network_architecture(
                    self.configuration_manager.network_arch_class_name,
                    self.configuration_manager.network_arch_init_kwargs,
                    self.configuration_manager.network_arch_init_kwargs_req_import,
                    self.num_input_channels,
                    self.label_manager.num_segmentation_heads,
                    self.enable_deep_supervision
                ).to(self.device)
            # Mamba 的 causal_conv1d_cuda 是 C 扩展，Dynamo 无法追踪
            warnings.filterwarnings("ignore", category=UserWarning,
                                    module="torch._dynamo",
                                    message=".*causal_conv1d_cuda.*causal_conv1d_fwd.*")
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            n_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            self.print_to_log_file(f"Network built. Trainable parameters: {n_params:,}")

            self.print_to_log_file("Configuring optimizer and loss...")
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.print_to_log_file("Detecting dataset format...")
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.print_to_log_file(f"Using dataset class: {self.dataset_class.__name__}")

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            # register additional metric keys in the logger for per-epoch tracking
            extra_metrics = ['iou_per_class', 'precision_per_class', 'recall_per_class', 'specificity_per_class']
            for key in extra_metrics:
                if key not in self.logger.local_logger.my_fantastic_logging:
                    self.logger.local_logger.my_fantastic_logging[key] = list()

            self._warmup_kernels()
            self.was_initialized = True
            self.print_to_log_file("Initialization complete.")

            logger_config_hparas = {
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "oversample_foreground_percent": self.oversample_foreground_percent,
                "probabilistic_oversampling": self.probabilistic_oversampling,
                "num_iterations_per_epoch": self.num_iterations_per_epoch,
                "num_val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "num_epochs": self.num_epochs,
                "enable_deep_supervision": self.enable_deep_supervision,
                "batch_size": self.configuration_manager.batch_size
                }
            self.logger.update_config({"hparas": logger_config_hparas})
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _warmup_kernels(self):
        """Warm up cuDNN autotune + train-mode kernels to avoid stalls on epoch 0 step 1.

        Phase 1 — eval forward ×2 (triggers ``cudnn.benchmark`` for conv/norm).
        Phase 2 — full train-mode step ×2 (forward + loss + backward + optimizer
        with AMP autocast), which warms up backward kernels, GradScaler, AMP
        autocast dispatch, and optimizer kernel autotuning. Non-cuDNN architectures
        (transformers, Mamba) benefit primarily from phase 2.
        """
        if self.device.type != 'cuda':
            return

        # DDP: all ranks must forward to pass sync barriers, but only rank 0 logs
        show_log = not (self.is_ddp and self.local_rank != 0)
        pbar_kw = dict(disable=not show_log, leave=False)

        # ---- Phase 1: cuDNN benchmark (eval forward) ----
        if show_log:
            self.print_to_log_file("Phase 1/2 — cuDNN benchmark warmup (2 eval forward passes) ...")

        dummy = torch.randn(
            (1, self.num_input_channels, *self.configuration_manager.patch_size),
            device=self.device,
        )
        with torch.no_grad():
            for _ in tqdm(range(2), desc="cuDNN warmup", unit="fw", **pbar_kw):
                _ = self.network(dummy)

        # ---- Phase 2: full train-mode step (forward + loss + backward + optimizer) ----
        # 继续训练时跳过：kernel shape 跟之前一样，benchmark 虽然未持久化但前向缓存已由
        # Phase 1 重建，第一 step 的少量 backward 编译开销可接受。
        if getattr(self, 'continue_training', False):
            if show_log:
                self.print_to_log_file("Phase 2/2 — Skipped (continue training).")
        else:
            with torch.no_grad():
                dummy_out = self.network(dummy)
            # DS 模式下部分网络 (SwinUMambaD/M2Net 等) 返回 list[seg...]，取主输出(全分辨率)即可
            if isinstance(dummy_out, (list, tuple)):
                dummy_out = dummy_out[0]
            num_output_channels = dummy_out.shape[1]

            batch_size = getattr(self, 'batch_size', 2)
            dummy_batch = torch.randn(
                (batch_size, self.num_input_channels, *self.configuration_manager.patch_size),
                device=self.device,
            )
            dummy_target = torch.randint(
                0, max(1, num_output_channels),
                (batch_size, 1, *self.configuration_manager.patch_size),
                device=self.device,
                dtype=torch.long,
            )

            if show_log:
                self.print_to_log_file(
                    f"Phase 2/2 — Train-mode warmup (2 steps, batch={batch_size}, "
                    f"AMP={self.autocast_dtype}) ..."
                )

            for _ in tqdm(range(2), desc="Train warmup", unit="step", **pbar_kw):
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == 'cuda'):
                    output = self.network(dummy_batch)
                    l = self.loss(output, dummy_target)
                if self.grad_scaler is not None:
                    self.grad_scaler.scale(l).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    l.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.optimizer.step()

        if show_log:
            self.print_to_log_file("Kernel warmup complete.")

    def _do_i_compile(self):
        # new default: compile is enabled!

        # compile does not work on mps
        if self.device == torch.device('mps'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because of unsupported mps device")
            return False

        # CPU compile crashes for 2D models. Not sure if we even want to support CPU compile!? Better disable
        if self.device == torch.device('cpu'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because device is CPU")
            return False

        # Windows + torch.compile: 自动检测 triton，可用则默认启用。
        # 设 nnUNet_compile=0 可强制禁用，=1 可强制启用。
        if os.name == 'nt':
            # 环境变量显式指定时优先
            if 'nnUNet_compile' in os.environ.keys():
                val = os.environ['nnUNet_compile'].lower()
                if val in ('true', '1', 't'):
                    self.print_to_log_file("INFO: torch.compile enabled on Windows via nnUNet_compile=1")
                    return True
                elif val in ('false', '0', 'f'):
                    self.print_to_log_file("INFO: torch.compile disabled on Windows via nnUNet_compile=0")
                    return False
            # 未显式设置时自动检测 triton
            try:
                import triton
                self.print_to_log_file(f"INFO: triton {triton.__version__} detected, torch.compile enabled")
                return True
            except ImportError:
                self.print_to_log_file(
                    "INFO: triton not found, torch.compile disabled on Windows. "
                    "Install via: uv pip install triton-windows-xxx"
                )
                return False

        if 'nnUNet_compile' not in os.environ.keys():
            return True
        else:
            return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')

    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network', ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        # print(k)
                        pass
                if k in ['dataloader_train', 'dataloader_val']:
                    dl = getattr(self, k)
                    if hasattr(dl, 'generator'):
                        dct[k + '.generator'] = str(dl.generator)
                        if hasattr(dl.generator, 'transforms'):
                            try:
                                dct[k + '.generator.transforms'] = str(dl.generator.transforms)
                            except Exception as e:
                                dct[k + '.generator.transforms'] = f"Could not stringify generator.transforms: {type(e).__name__}: {e}"
                    if hasattr(dl, 'num_processes'):
                        dct[k + '.num_processes'] = str(dl.num_processes)
                    if hasattr(dl, 'transform'):
                        dct[k + '.transform'] = str(dl.transform)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        return get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
                self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            self.batch_size = self.configuration_manager.batch_size
        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = [global_batch_size // world_size] * world_size
            batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                                  if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                                  else batch_size_per_GPU[i]
                                  for i in range(len(batch_size_per_GPU))]
            assert sum(batch_size_per_GPU) == global_batch_size

            sample_id_low = 0 if my_rank == 0 else np.sum(batch_size_per_GPU[:my_rank])
            sample_id_high = np.sum(batch_size_per_GPU[:my_rank + 1])

            # This is how oversampling is determined in DataLoader
            # round(self.batch_size * (1 - self.oversample_foreground_percent))
            # We need to use the same scheme here because an oversample of 0.33 with a batch size of 2 will be rounded
            # to an oversample of 0.5 (1 sample random, one oversampled). This may get lost if we just numerically
            # compute oversample
            oversample = [True if not i < round(global_batch_size * (1 - self.oversample_foreground_percent)) else False
                          for i in range(global_batch_size)]

            if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                oversample_percent = 0.0
            elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                oversample_percent = 1.0
            else:
                oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / batch_size_per_GPU[my_rank]

            print("worker", my_rank, "oversample", oversample_percent)
            print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])

            self.batch_size = batch_size_per_GPU[my_rank]
            self.oversample_foreground_percent = oversample_percent

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # 固定 ±90° 旋转 + 镜像翻转（适用于岩石 CT/SEM 分割）
            rotation_for_DA = (-90. / 360 * 2. * np.pi, 90. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            # 实际使用的配置：batch_size 以训练器实际值为准（DDP 下为每 GPU 值），
            # architecture 替换为实际构建的网络结构。plan.json 中的 architecture 是
            # nnU-Net 规划器写入的默认模板（PlainConvUNet 等），本项目各模型
            # （LightMamba2Net / UMamba / UNETR ...）均在训练器内自行构建网络，
            # 超参数由模型定义，与 plan 模板无关，故直接打印模型真实结构。
            config = deepcopy(self.configuration_manager.configuration)
            config['batch_size'] = self.batch_size
            config['architecture'] = self._actual_architecture_summary()
            # 附加训练上下文: 数据集 / 损失 / 优化器 / AMP 精度
            config['dataset'] = self._dataset_summary()
            config['training'] = {
                'loss': self.loss.__class__.__name__ if self.loss is not None else None,
                'optimizer': self.optimizer.__class__.__name__ if self.optimizer is not None else None,
                'lr_scheduler': self.lr_scheduler.__class__.__name__ if self.lr_scheduler is not None else None,
                'initial_lr': getattr(self, 'initial_lr', None),
                'autocast_dtype': str(getattr(self, 'autocast_dtype', 'n/a')),
            }
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\n"
                                   f"Trainer: {self.__class__.__name__}\n"
                                   f"NumEpochs: {self.num_epochs}\n"
                                   f"Configuration name: {self.configuration_name}\n",
                                   config, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)

    def _dataset_summary(self) -> dict:
        """提取当前训练数据集信息（名称 / 标签 / 样本数 / 通道 / 文件后缀）。

        dataset_name 位于 plans.json 全局设置中；标签等其余字段来自 dataset.json。
        """
        dj = self.dataset_json or {}
        labels = dj.get('labels')
        # 反向映射 {name: id} -> {id: name}，便于阅读
        if isinstance(labels, dict) and labels and all(isinstance(v, int) for v in labels.values()):
            labels = {str(v): k for k, v in labels.items()}
        channels = dj.get('channel_names')
        return {
            'name': self.plans_manager.plans.get('dataset_name'),
            'num_training': dj.get('numTraining'),
            'labels': labels,
            'channels': list(channels.values()) if isinstance(channels, dict) else channels,
            'file_ending': dj.get('file_ending'),
        }

    def _actual_architecture_summary(self) -> dict:
        """从已构建的网络实例提取真实结构摘要（类名 + 构造参数 + 可训练参数量）。

        通过 inspect 反射网络 __init__ 签名，并用实例属性 / 训练器属性回填实际值，
        对任意自定义网络（LightMamba2Net / UMamba / UNETR / PlainConvUNet ...）通用。
        在 print_plans 被调用时（on_train_start）self.network 已构建完成。
        """
        if self.network is None:
            return {'network_class': 'not_built_yet', 'params': {}}
        net = self.network
        # 解包 DDP / torch.compile 包装层，拿到原始模型
        if isinstance(net, DDP):
            net = net.module
        if isinstance(net, OptimizedModule):
            net = net._orig_mod

        summary = {
            'network_class': f"{net.__class__.__module__}.{net.__class__.__name__}",
            'params': {},
            'trainable_params': int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
        }
        # 统计网络中的激活函数与归一化层类型及数量。
        # 自定义网络（LightMamba2Net 等）内部结构各异，按标准层类型遍历统计即可。
        act_stats, norm_stats = {}, {}
        for mod in net.modules():
            if isinstance(mod, (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.PReLU,
                                nn.Sigmoid, nn.Tanh, nn.Softmax, nn.Mish, nn.ELU, nn.Hardswish)):
                name = mod.__class__.__name__
                act_stats[name] = act_stats.get(name, 0) + 1
            elif isinstance(mod, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                  nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                  nn.LayerNorm, nn.GroupNorm)):
                name = mod.__class__.__name__
                norm_stats[name] = norm_stats.get(name, 0) + 1
        summary['activations'] = act_stats
        summary['normalizations'] = norm_stats
        try:
            sig = inspect.signature(net.__class__.__init__)
            for name, param in sig.parameters.items():
                if name in ('self', 'args', 'kwargs'):
                    continue
                attr = getattr(net, name, None)
                # 输入/输出通道数常未存为实例属性，用训练器实际值回填
                if attr is None and name in ('in_ch', 'in_channels', 'num_input_channels'):
                    attr = getattr(self, 'num_input_channels', None)
                if attr is None and name in ('out_ch', 'out_channels', 'num_output_channels'):
                    attr = getattr(self, 'num_output_channels', None)
                if attr is not None and not isinstance(attr, (nn.Module, nn.Parameter)) \
                        and isinstance(attr, (int, float, str, bool, list, tuple, dict, type(None))):
                    summary['params'][name] = attr
                    continue
                # 实例无该属性时回退到签名默认值
                if param.default is not inspect.Parameter.empty:
                    summary['params'][name] = param.default
                else:
                    summary['params'][name] = '<required>'
        except (TypeError, ValueError):
            pass
        return summary

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def plot_network_architecture(self):
        # NOTE: torchview (v0.2.7) globally monkey-patches __torch_function__ on all tensors
        # on import, which breaks train_step (0-d tensor iteration error). Disabled until
        # a compatible version is available or the bug is fixed upstream.
        if self.local_rank == 0:
            self.print_to_log_file("Network architecture plot disabled (torchview compatibility issue).")

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                         identifiers=None,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = self.dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        self.print_to_log_file("Setting up data augmentation pipeline...")
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        self.print_to_log_file("Creating training and validation datasets...")
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        self.print_to_log_file(f"Starting data loading workers ({get_allowed_n_proc_DA()} processes)...")
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=allowed_num_processes, seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 2), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # Train augmenter must be ready before training starts → start synchronously.
        # Val augmenter is only needed at epoch end → start in background thread to
        # overlap its process spawning with the first training epoch. This saves
        # ~1 min of startup time (12 fewer processes on the critical path).
        if isinstance(mt_gen_train, NonDetMultiThreadedAugmenter):
            mt_gen_train._start()
        if isinstance(mt_gen_val, NonDetMultiThreadedAugmenter):
            self._val_start_thread = Thread(target=mt_gen_val._start)
            self._val_start_thread.start()
        self.print_to_log_file("Data pipeline initialized (train ready, val workers starting in background).")
        return mt_gen_train, mt_gen_val

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                border_mode_seg='constant',
                padding_value_seg=-1,
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=0.5
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=0.5
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        mod.decoder.deep_supervision = enabled

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        # dataloaders must be instantiated here (instead of __init__) because they need access to the training data
        # which may not be present  when doing inference
        self.print_to_log_file("Preparing data loaders...")
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.local_rank == 0:
            self.print_to_log_file("Unpacking dataset (converting .npz to .npy)...")
            self.dataset_class.unpack_dataset(
                self.preprocessed_dataset_folder,
                overwrite_existing=False,
                num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
                verify=True)

        if self.is_ddp:
            dist.barrier()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copyfile(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                        join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_val, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        self.network.train()
        # PolyLRScheduler 是闭式形式，lr 直接由 current_epoch 计算，不依赖 optimizer.step 计数。
        # 但 PyTorch 会对 "step() 先于 optimizer.step()" 与显式 epoch 参数发出无害 UserWarning，
        # 每 epoch 刷屏，此处静默之。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, dtype=self.autocast_dtype, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, dtype=self.autocast_dtype, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, tn = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        tn_hard = tn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
            tn_hard = tn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard,
                'fn_hard': fn_hard, 'tn_hard': tn_hard}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)
        tn = np.sum(outputs_collated['tn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            tns = [None for _ in range(world_size)]
            dist.all_gather_object(tns, tn)
            tn = np.vstack([i[None] for i in tns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

        # compute additional confusion-matrix metrics
        with np.errstate(invalid='ignore', divide='ignore'):
            iou_pc = np.array([i / (i + j + k) for i, j, k in zip(tp, fp, fn)])
            prec_pc = np.array([i / (i + j) for i, j in zip(tp, fp)])
            rec_pc = np.array([i / (i + k) for i, k in zip(tp, fn)])
            spec_pc = np.array([k / (k + j) for k, j in zip(tn, fp)])
        self.logger.log('iou_per_class', list(iou_pc), self.current_epoch)
        self.logger.log('precision_per_class', list(prec_pc), self.current_epoch)
        self.logger.log('recall_per_class', list(rec_pc), self.current_epoch)
        self.logger.log('specificity_per_class', list(spec_pc), self.current_epoch)

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # helper: compute delta from previous epoch (None if first epoch)
        def _delta(key):
            try:
                v_now = self.logger.get_value(key, step=-1)
                v_prev = self.logger.get_value(key, step=-2)
                # handle list/array per-class metrics (element-wise subtraction)
                if isinstance(v_now, (list, tuple, np.ndarray)):
                    if isinstance(v_now, np.ndarray):
                        return list(v_now - np.asarray(v_prev))
                    return [a - b for a, b in zip(v_now, v_prev)]
                return v_now - v_prev
            except (IndexError, TypeError):
                return None

        # arrow: ↑ = improvement, ↓ = degradation, → = no change
        def _arrow(delta, higher_is_better=True):
            if delta is None:
                return ''
            if isinstance(delta, (list, tuple, type(None))):
                return ''
            if abs(delta) < 1e-8:
                return '→'
            if higher_is_better:
                return '↑' if delta > 0 else '↓'
            else:
                return '↓' if delta > 0 else '↑'

        # train loss (lower is better)
        tr_loss = float(self.logger.get_value('train_losses', step=-1))
        tr_delta = _delta('train_losses')
        if tr_delta is not None:
            self.print_to_log_file(f'train_loss    {tr_loss:.4f}   (Δ {tr_delta:+.4f}) {_arrow(tr_delta, False)}')
        else:
            self.print_to_log_file(f'train_loss    {tr_loss:.4f}')

        # val loss (lower is better)
        val_loss = float(self.logger.get_value('val_losses', step=-1))
        val_delta = _delta('val_losses')
        if val_delta is not None:
            self.print_to_log_file(f'val_loss      {val_loss:.4f}   (Δ {val_delta:+.4f}) {_arrow(val_delta, False)}')
        else:
            self.print_to_log_file(f'val_loss      {val_loss:.4f}')

        # helper: format a per-class metric list with optional delta + arrows
        def _fmt_per_class(key, label, width=13):
            raw = self.logger.get_value(key, step=-1)
            val_str = ', '.join(f'{v:.4f}' for v in raw)
            delta = _delta(key)
            if delta is not None:
                if len(delta) == 1:
                    # single class: compact format without brackets
                    d = delta[0]
                    self.print_to_log_file(
                        f'{label:<{width}} [{val_str}]   (Δ {d:+.4f}) {_arrow(d, True)}')
                else:
                    # multiple classes: list format with brackets
                    delta_str = ', '.join(f'{d:+.4f}' for d in delta)
                    arrow_str = ', '.join(_arrow(d, True) for d in delta)
                    self.print_to_log_file(
                        f'{label:<{width}} [{val_str}]   (Δ [{delta_str}]) [{arrow_str}]')
            else:
                self.print_to_log_file(f'{label:<{width}} [{val_str}]')

        # per-class metrics (higher is better)
        _fmt_per_class('dice_per_class_or_region', 'Pseudo dice')
        _fmt_per_class('iou_per_class', 'IoU')
        _fmt_per_class('precision_per_class', 'Precision')
        _fmt_per_class('recall_per_class', 'Recall')
        _fmt_per_class('specificity_per_class', 'Specificity')

        # mean foreground dice (higher is better)
        fg_dice = float(self.logger.get_value('mean_fg_dice', step=-1))
        fg_delta = _delta('mean_fg_dice')
        if fg_delta is not None:
            self.print_to_log_file(f'mean_fg_dice  {fg_dice:.4f}   (Δ {fg_delta:+.4f}) {_arrow(fg_delta, True)}')
        else:
            self.print_to_log_file(f'mean_fg_dice  {fg_dice:.4f}')

        # ema foreground dice (higher is better)
        ema = float(self.logger.get_value('ema_fg_dice', step=-1))
        ema_delta = _delta('ema_fg_dice')
        if ema_delta is not None:
            self.print_to_log_file(f'ema_fg_dice   {ema:.4f}   (Δ {ema_delta:+.4f}) {_arrow(ema_delta, True)}')
        else:
            self.print_to_log_file(f'ema_fg_dice   {ema:.4f}')

        # learning rate
        lr = float(self.logger.get_value('lrs', step=-1))
        self.print_to_log_file(f'lr            {lr:.6f}')

        # epoch time
        epoch_time = self.logger.get_value('epoch_end_timestamps', step=-1) - \
                     self.logger.get_value('epoch_start_timestamps', step=-1)
        self.print_to_log_file(f'Epoch time    {epoch_time:.2f} s')

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.get_value('ema_fg_dice', step=-1) > self._best_ema:
            self._best_ema = self.logger.get_value('ema_fg_dice', step=-1)
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def perform_actual_validation(self, save_probabilities: bool = False):
        if self.disable_checkpointing:
            self.print_to_log_file('Validation skipped, checkpointing is disabled')
            return
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            # 非 DDP 时该值不会被使用（下方 is_ddp 短路），提前初始化仅为类型检查器保证绑定
            last_barrier_at_idx = 0
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []
            disable_tqdm = self.local_rank != 0
            _val_keys = list(dataset_val.identifiers)

            # 数据预取线程 + 主线程 GPU 推理，解耦 I/O 与计算
            # 预取线程负责数据加载（numpy I/O 在 C 层释放 GIL）
            # 主线程专做 GPU 推理（CUDA kernel 释放 GIL），两者可重叠
            import queue as _queue
            import threading

            _prefetch_queue = _queue.Queue(maxsize=4)
            _prefetch_done = threading.Event()

            def _prefetch_worker():
                """后台线程：专责数据加载，预取到队列"""
                for k in _val_keys:
                    if _prefetch_done.is_set():
                        return
                    data, _, seg_prev, properties = dataset_val.load_case(k)
                    data = data[:]
                    if self.is_cascaded:
                        seg_prev = seg_prev[:]
                        data = np.vstack((data, convert_labelmap_to_one_hot(
                            seg_prev, self.label_manager.foreground_labels, output_dtype=data.dtype)))
                    _prefetch_queue.put((k, data, properties))
                _prefetch_queue.put(None)  # 哨兵

            prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
            prefetch_thread.start()

            # GPU 批量推理：积累 BATCH_SIZE 张图后一次 forward，提升 GPU 利用率
            val_batch_size = int(os.environ.get('nnUNet_val_batch_size', '4'))
            from acvl_utils.cropping_and_padding.padding import pad_nd_image as _pad_batch

            with tqdm(total=len(_val_keys), desc="Val", unit="img",
                      disable=disable_tqdm) as pbar:
                i = 0
                batch_items = []  # [(k, data, properties), ...]

                def _submit_batch(items_with_preds):
                    """为一批已完成推理的结果提交 export + 处理 next_stage + DDP barrier"""
                    nonlocal i
                    for (k, _, properties), prediction in items_with_preds:
                        pbar.set_postfix_str(k)

                        # export backpressure
                        proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list,
                                                                    results, allowed_num_queued=2)
                        while not proceed:
                            sleep(0.1)
                            proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list,
                                                                        results, allowed_num_queued=2)

                        output_filename_truncated = join(validation_output_folder, k)

                        # submit export
                        results.append(
                            segmentation_export_pool.starmap_async(
                                export_prediction_from_logits, (
                                    (prediction, properties, self.configuration_manager, self.plans_manager,
                                     self.dataset_json, output_filename_truncated, save_probabilities),
                                )
                            )
                        )

                        # if needed, export the softmax prediction for the next stage
                        if next_stages is not None:
                            for n in next_stages:
                                next_stage_config_manager = self.plans_manager.get_configuration(n)
                                expected_preprocessed_folder = join(nnUNet_preprocessed,
                                                                     self.plans_manager.dataset_name,
                                                                     next_stage_config_manager.data_identifier)
                                dataset_class = infer_dataset_class(expected_preprocessed_folder)

                                try:
                                    tmp = dataset_class(expected_preprocessed_folder, [k])
                                    d, _, _, _ = tmp.load_case(k)
                                except FileNotFoundError:
                                    self.print_to_log_file(
                                        f"Predicting next stage {n} failed for case {k} because "
                                        f"the preprocessed file is missing! "
                                        f"Run the preprocessing for this configuration first!")
                                    continue

                                target_shape = d.shape[1:]
                                output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                                output_file_truncated = join(output_folder, k)

                                results.append(segmentation_export_pool.starmap_async(
                                    resample_and_save, (
                                        (prediction, target_shape, output_file_truncated, self.plans_manager,
                                         self.configuration_manager,
                                         properties,
                                         self.dataset_json,
                                         default_num_processes,
                                         dataset_class),
                                    )
                                ))

                        # DDP barrier every 20 items
                        if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                            dist.barrier()

                        i += 1
                        pbar.update()

                while True:
                    item = _prefetch_queue.get()
                    if item is None:
                        break
                    k, data, properties = item
                    batch_items.append((k, data, properties))

                    if len(batch_items) >= val_batch_size:
                        # 批量 GPU 推理：先各自 pad 到 patch_size 保证同 shape，再 stack
                        _ps = self.configuration_manager.patch_size
                        batch_padded = [_pad_batch(d, _ps, 'constant', {'constant_values': 0}, True, None)[0]
                                        for _, d, _ in batch_items]
                        batch_tensor = torch.from_numpy(np.stack(batch_padded, axis=0))
                        predictions = predictor.predict_batch_return_logits(batch_tensor)

                        _submit_batch(zip(batch_items, [predictions[j] for j in range(len(batch_items))]))
                        batch_items = []

                # 处理剩余不足一批的图
                if batch_items:
                    _ps = self.configuration_manager.patch_size
                    batch_padded = [_pad_batch(d, _ps, 'constant', {'constant_values': 0}, True, None)[0]
                                    for _, d, _ in batch_items]
                    batch_tensor = torch.from_numpy(np.stack(batch_padded, axis=0))
                    predictions = predictor.predict_batch_return_logits(batch_tensor)
                    _submit_batch(zip(batch_items, [predictions[j] for j in range(len(batch_items))]))
                    batch_items = []

            _prefetch_done.set()  # 确保预取线程退出
            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            for label in metrics["mean"]:
                self.logger.log_summary(f"final_val/class_{label}_dice", metrics["mean"][label]["Dice"])
            self.logger.log_summary("final_val/foreground_dice", metrics['foreground_mean']["Dice"])
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            total_imgs = self.num_iterations_per_epoch * self.batch_size
            with tqdm(total=total_imgs, desc=f"Epoch {epoch} Train",
                      unit="img", leave=False, disable=self.local_rank != 0) as pbar:
                for batch_id in range(self.num_iterations_per_epoch):
                    output = self.train_step(next(self.dataloader_train))
                    train_outputs.append(output)
                    pbar.update(self.batch_size)
                    pbar.set_postfix(loss=float(output['loss']))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                total_val_imgs = self.num_val_iterations_per_epoch * self.batch_size
                with tqdm(total=total_val_imgs, desc=f"Epoch {epoch} Val",
                          unit="img", leave=False, disable=self.local_rank != 0) as pbar:
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        output = self.validation_step(next(self.dataloader_val))
                        val_outputs.append(output)
                        pbar.update(self.batch_size)
                        pbar.set_postfix(loss=float(output['loss']))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
