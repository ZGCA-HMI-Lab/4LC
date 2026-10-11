from __future__ import annotations

import random
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from alc_video.runtime.resources import release_resource
from alc_video.training.config import RunConfig, load_run_config
from alc_video.training.execution.checkpoint import (
    resume_training_iterator,
    save_training_checkpoint,
)
from alc_video.training.execution.distributed import create_accelerator
from alc_video.training.execution.metrics import (
    MetricAccumulator,
    TrainingMetrics,
    format_summary,
    summarize_update,
)
from alc_video.training.execution.optimizer import resolve_gradient_clip
from alc_video.training.execution.session import prepare_session
from alc_video.training.execution.timing import (
    TimedLoader,
    TrainingTimingRecorder,
)
from alc_video.training.execution.update import train_batch
from alc_video.utils.errors import input_errors
from alc_video.utils.logging import (
    LOG,
    configure_logging,
)


def run_training(
    cfg: RunConfig, *, log_level: str = "INFO", config_path: Path | None = None
) -> Path:
    configure_logging(log_level)
    total_started = time.perf_counter()
    accelerator = create_accelerator(cfg)
    with ExitStack() as resources:
        resources.enter_context(release_resource(accelerator.end_training))
        session = prepare_session(
            cfg, accelerator, resources, log_level=log_level, config_path=config_path
        )
        cfg, data, model = session.cfg, session.data, session.model
        optimizer, resume, step = (
            session.optimizer,
            session.resume,
            session.initial_step,
        )
        resume_rng = session.resume_rng
        logger = (
            TrainingMetrics(cfg.logging, cfg.checkpoint.output_path)
            if accelerator.is_main_process
            else None
        )
        if logger is not None:
            resources.enter_context(release_resource(logger.finish))
        parameters = [p for p in model.parameters() if p.requires_grad]
        clip_norm, clip_type = resolve_gradient_clip(cfg)
        recorder = (
            None
            if cfg.training.profile_freq == 0
            else TrainingTimingRecorder(cfg.training.profile_freq, accelerator.device)
        )
        active_loader = data.loader if recorder is None else TimedLoader(data.loader)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        accumulated = MetricAccumulator()
        last_saved, last_metrics = step if resume is not None else -1, {}
        last_checkpoint = resume
        first_step = step
        skipped = 0
        progress_started = time.perf_counter()
        while step < cfg.training.max_steps:
            if resume_rng is None:
                iterator = iter(active_loader)
            else:
                python_rng, numpy_rng, torch_rng, cuda_rng = resume_rng
                iterator = resume_training_iterator(active_loader, torch_rng)
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                resume_rng = None
            saw_batch = False
            for batch in iterator:
                saw_batch = True
                started = time.perf_counter()
                if recorder is not None:
                    recorder.begin(
                        next_step=step + 1,
                        iteration_started=active_loader.last_started,
                        data_load_ms=active_loader.last_data_load_ms,
                    )
                context = dict(
                    update=step + 1,
                    sample=batch["sample_id"],
                    input=batch["source_path"],
                )
                if step == first_step and accumulated.count == 0:
                    LOG.info(
                        "first batch sample=%s frames=%s..%s K=%s input=%s",
                        batch["sample_id"],
                        int(batch["frame_indices"][0, 0]),
                        int(batch["frame_indices"][0, -1]),
                        batch["source_positions"].shape[1],
                        batch["source_path"],
                    )
                LOG.debug(
                    "batch %s video_shape=%s",
                    context,
                    tuple(batch["source_video"].shape),
                )
                update = train_batch(
                    session,
                    batch,
                    accumulated,
                    step=step,
                    context=context,
                    parameters=parameters,
                    clip_norm=clip_norm,
                    clip_type=clip_type,
                    recorder=recorder,
                )
                if not accelerator.sync_gradients:
                    continue
                if not update.successful:
                    skipped += 1
                    LOG.warning(
                        "optimizer update skipped %s skipped_total=%s", context, skipped
                    )
                    accumulated.reset_update()
                    continue
                if recorder is not None:
                    if accumulated.profile_statistics is None:
                        raise RuntimeError("training profile step is empty")
                    accumulated.profile_window = (
                        accumulated.profile_statistics
                        if accumulated.profile_window is None
                        else accumulated.profile_window + accumulated.profile_statistics
                    )
                    accumulated.profile_statistics = None
                step += 1
                last_metrics, train_metrics = summarize_update(
                    session, accumulated, update, started=started
                )
                if logger is not None:
                    logger.log(last_metrics, step=step, sample_id=batch["sample_id"])
                    if cfg.training.print_freq > 0 and (
                        step == first_step + 1
                        or step == cfg.training.max_steps
                        or step % cfg.training.print_freq == 0
                    ):
                        LOG.info(
                            format_summary(
                                step,
                                cfg.training.max_steps,
                                train_metrics,
                                elapsed=time.perf_counter() - progress_started,
                                completed=step - first_step,
                            )
                        )
                if cfg.training.save_freq > 0 and step % cfg.training.save_freq == 0:
                    last_checkpoint = save_training_checkpoint(
                        accelerator,
                        cfg,
                        step=step,
                        loader=data.loader,
                        metrics=last_metrics,
                    )
                    last_saved = step
                if step >= cfg.training.max_steps:
                    break
            if not saw_batch:
                raise RuntimeError("training loader yielded no batches")
        if step != last_saved:
            last_checkpoint = save_training_checkpoint(
                accelerator, cfg, step=step, loader=data.loader, metrics=last_metrics
            )
        if last_checkpoint is None:
            raise RuntimeError("training completed without a checkpoint")
        LOG.info(
            "train complete updates=%s skipped=%s elapsed=%.2fs checkpoint=%s",
            step - first_step,
            skipped,
            time.perf_counter() - total_started,
            last_checkpoint,
        )
        return last_checkpoint


def train_from_config(
    path: str | Path,
    overrides: tuple[str, ...] = (),
    *,
    log_level: str = "INFO",
    model: str | Path | None = None,
) -> Path:
    with input_errors():
        cfg = load_run_config(path, overrides, model=model)
    return run_training(
        cfg,
        log_level=log_level,
        config_path=Path(path).resolve(),
    )
