from __future__ import annotations

import random
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from alc_video.data.loader import DataBundle, build_train_loader, close_loaders
from alc_video.models.checkpoint import TRAINABLE_MODULES, CheckpointError
from alc_video.runtime.components import load_runtime_components
from alc_video.runtime.resources import release_resource
from alc_video.training.augmentation import (
    AugmentationExecutor,
    load_augmentation_config,
    load_unconditional_context,
)
from alc_video.training.config import RunConfig
from alc_video.training.execution.checkpoint import (
    load_training_checkpoint,
    resolve_resume_path,
    validate_training_output,
)
from alc_video.training.execution.distributed import (
    run_main_process_stage,
    scale_run_max_steps,
    seed_everything,
)
from alc_video.training.execution.optimizer import (
    ConstantLRScheduler,
    build_lr_scheduler,
    build_optimizer,
)
from alc_video.training.objective.module import TrainingModule, configure_pretrain
from alc_video.utils.errors import input_errors
from alc_video.utils.logging import (
    LOG,
    configure_logging,
    stage,
)

if TYPE_CHECKING:
    from accelerate import Accelerator


def _prepare_augmentation(
    cfg: RunConfig,
) -> tuple[AugmentationExecutor | None, torch.Tensor | None]:
    augmentation = (
        None
        if cfg.augmentation is None
        else AugmentationExecutor(load_augmentation_config(cfg.augmentation))
    )
    unconditional = None
    if (
        augmentation is not None
        and augmentation.config.condition.prompt_drop.probability > 0.0
    ):
        unconditional = load_unconditional_context(
            augmentation.config.condition.prompt_drop.unconditional_context
        )
    return augmentation, unconditional


def build_training_module(
    cfg: RunConfig,
    *,
    device: torch.device,
    resources: ExitStack,
    augmentation: AugmentationExecutor | None,
    unconditional: torch.Tensor | None,
) -> TrainingModule:
    components = load_runtime_components(
        cfg.model,
        cfg.encoder_catalog,
        cfg.representations,
        wan_model_root=cfg.wan_model_root,
        worldmirror_checkpoint=cfg.worldmirror_checkpoint,
        device=device,
        resources=resources,
        weights=cfg.checkpoint.initial_weights,
        trainable_modules=TRAINABLE_MODULES,
        prepare_network=configure_pretrain,
    )
    network, codec, text, worldmirror = (
        components.network,
        components.codec,
        components.text,
        components.mirror,
    )
    LOG.info("training policy=pretrain trainable_modules=%s", TRAINABLE_MODULES)
    return TrainingModule(
        conditioned_model=network,
        cache_config=cfg.cache,
        encoders=components.bindings,
        worldmirror=worldmirror,
        video_codec=codec,
        text_encoder=text,
        augmentation=augmentation,
        unconditional_context=unconditional,
        propagate_rgb=cfg.propagate_rgb,
    )


@dataclass
class TrainingSession:
    cfg: RunConfig
    accelerator: Accelerator
    data: DataBundle
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: ConstantLRScheduler
    resume: Path | None
    initial_step: int
    resume_rng: tuple | None


def prepare_session(
    cfg: RunConfig,
    accelerator: Accelerator,
    resources: ExitStack,
    *,
    log_level: str,
    config_path: Path | None,
) -> TrainingSession:
    with input_errors():
        resume = resolve_resume_path(cfg)
        run_main_process_stage(
            accelerator, lambda: validate_training_output(cfg), name="training output"
        )
    LOG.info(
        "train start config=%s output=%s device=%s precision=bf16 optimizer_offload=%s",
        config_path,
        cfg.checkpoint.output_path,
        accelerator.device,
        cfg.training.zero_optimizer_offload,
    )
    LOG.debug("effective training config=%r", cfg)
    with input_errors(), stage("data.scan", source=cfg.data.source, root=cfg.data.root):
        data = build_train_loader(cfg, accelerator=accelerator)
    resources.enter_context(release_resource(close_loaders, [data.loader]))
    with input_errors():
        augmentation, unconditional = _prepare_augmentation(cfg)
    log_path = None
    if cfg.logging.train_log_enabled:
        log_path = cfg.checkpoint.output_path / (
            cfg.logging.train_log_path or "train.log"
        )
    configure_logging(log_level, path=log_path, rank=int(accelerator.process_index))
    LOG.info(
        "data selection %s; sample_fps=16 target_frames=81 context_K=11..21 image=560x336",
        data.selection_counts,
    )
    cfg = scale_run_max_steps(
        cfg,
        sample_count=data.sample_count,
        world_size=int(accelerator.num_processes),
    )
    seed_everything(cfg.runtime.seed + int(accelerator.process_index))
    model = build_training_module(
        cfg,
        device=accelerator.device,
        resources=resources,
        augmentation=augmentation,
        unconditional=unconditional,
    )
    optimizer = build_optimizer(model, cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg)
    accelerator.register_for_checkpointing(lr_scheduler)
    LOG.info(
        "training plan updates=%s world_size=%s accumulation=%s seed=%s trainable_parameters=%s",
        cfg.training.max_steps,
        accelerator.num_processes,
        cfg.training.gradient_accumulation_steps,
        cfg.runtime.seed,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    with stage("training.prepare"):
        model, optimizer = accelerator.prepare(model, optimizer)
    lr_scheduler.optimizer = optimizer
    step, resume_rng = 0, None
    if resume is not None:
        with stage(
            "checkpoint.restore",
            path=resume,
            target_update=cfg.training.max_steps,
        ):
            step = load_training_checkpoint(
                accelerator, cfg, resume, loader=data.loader
            )
        LOG.info(
            "resume ready update=%s target_update=%s state=model,optimizer,lr_scheduler,RNG,reader",
            step,
            cfg.training.max_steps,
        )
        if lr_scheduler.step_count != step:
            raise CheckpointError(
                f"lr_scheduler step_count differs from checkpoint step: {lr_scheduler.step_count} != {step}"
            )
        resume_rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all(),
        )
    return TrainingSession(
        cfg, accelerator, data, model, optimizer, lr_scheduler, resume, step, resume_rng
    )
