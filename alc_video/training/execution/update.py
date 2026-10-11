from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from alc_video.data.schema import RawBatch
from alc_video.training.execution.metrics import MetricAccumulator

if TYPE_CHECKING:
    from alc_video.training.execution.session import TrainingSession
    from alc_video.training.execution.timing import TrainingTimingRecorder

from alc_video.training.execution.metrics import (
    TRAIN_LOSS_METRICS,
    deepspeed_global_grad_norm,
    metric_float,
)
from alc_video.utils.logging import (
    stage,
)


@dataclass(frozen=True)
class BatchUpdate:
    successful: bool
    grad_norm: float
    timings: dict[str, float]
    condition_timings: dict[str, float]


def train_batch(
    session: TrainingSession,
    batch: RawBatch,
    metrics: MetricAccumulator,
    *,
    step: int,
    context: dict[str, object],
    parameters: list[torch.nn.Parameter],
    clip_norm: float,
    clip_type: float,
    recorder: TrainingTimingRecorder | None,
) -> BatchUpdate:
    accelerator, model = session.accelerator, session.model
    optimizer, lr_scheduler = session.optimizer, session.lr_scheduler
    condition_timings = {}
    with (
        stage("train.update", level=logging.DEBUG, **context),
        accelerator.accumulate(model),
    ):
        with nullcontext() if recorder is None else recorder.cuda("forward"):
            loss, batch_metrics = model(
                batch,
                augmentation_step=step,
                collect_profile_metrics=recorder is not None,
                collect_timing_metrics=recorder is not None and recorder.active,
            )
        if recorder is not None:
            statistics = batch_metrics["profile_statistics"].detach()
            metrics.profile_statistics = (
                statistics
                if metrics.profile_statistics is None
                else metrics.profile_statistics + statistics
            )
            if recorder.active:
                condition_timings = {
                    name: metric_float(batch_metrics[f"profile/{name}"])
                    for name in ("cache_construct_ms", "cache_read_ms")
                }
        with nullcontext() if recorder is None else recorder.cuda("backward"):
            accelerator.backward(loss)
        for name, value in batch_metrics.items():
            if name == "profile_statistics" or name.startswith("profile/"):
                continue
            metrics.values[name] = metrics.values.get(name, 0.0) + metric_float(value)
        dataset = batch["dataset"]
        values = metrics.dataset_sums.setdefault(dataset, {})
        for name in TRAIN_LOSS_METRICS:
            values[name] = values.get(name, 0.0) + metric_float(batch_metrics[name])
        metrics.dataset_counts[dataset] = metrics.dataset_counts.get(dataset, 0) + 1
        metrics.count += 1
        grad_norm, deepspeed_clipped = 0.0, False
        with nullcontext() if recorder is None else recorder.cuda("optimizer"):
            if accelerator.sync_gradients:
                clipped = accelerator.clip_grad_norm_(
                    parameters, clip_norm, norm_type=clip_type
                )
                if clipped is None:
                    deepspeed_clipped = True
                else:
                    grad_norm = metric_float(clipped)
            optimizer.step()
            successful = (
                accelerator.sync_gradients
                and not accelerator.optimizer_step_was_skipped
            )
            if successful:
                lr_scheduler.step()
                if deepspeed_clipped:
                    grad_norm = deepspeed_global_grad_norm(model)
            optimizer.zero_grad(set_to_none=True)
    timings = {} if recorder is None else recorder.finish(successful=successful)
    return BatchUpdate(successful, grad_norm, timings, condition_timings)
