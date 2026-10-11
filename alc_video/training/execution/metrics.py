from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from alc_video.training.execution.timing import reduce_timing_metrics
from alc_video.training.objective.flow_matching import PROFILE_STATISTIC_NAMES

if TYPE_CHECKING:
    from alc_video.training.execution.session import TrainingSession
    from alc_video.training.execution.update import BatchUpdate


def deepspeed_global_grad_norm(model: Any) -> float:
    return metric_float(model.get_global_grad_norm())


class TrainingMetrics:
    def __init__(self, cfg: Any, output: Path) -> None:
        self.output = output
        self.config = cfg
        self.writer = None
        self.swanlab = None
        output.mkdir(parents=True, exist_ok=True)
        if cfg.tensorboard_enabled:
            from torch.utils.tensorboard import SummaryWriter

            relative = (
                "tensorboard"
                if cfg.tensorboard_logdir is None
                else cfg.tensorboard_logdir
            )
            self.writer = SummaryWriter(log_dir=str(output / relative))
        if cfg.swanlab_enabled:
            import swanlab

            swanlab.init(
                project=cfg.swanlab_project,
                experiment_name=cfg.swanlab_experiment_name,
                workspace=cfg.swanlab_workspace,
                mode=cfg.swanlab_mode,
                logdir=str(
                    output
                    / ("swanlog" if cfg.swanlab_logdir is None else cfg.swanlab_logdir)
                ),
            )
            self.swanlab = swanlab

    def log(self, metrics: Mapping[str, float], *, step: int, sample_id: str) -> None:
        with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"step": step, "sample_id": sample_id, **metrics},
                    ensure_ascii=False,
                )
                + "\n"
            )
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)
            self.writer.flush()
        if self.swanlab is not None:
            self.swanlab.log(dict(metrics), step=step)

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.swanlab is not None:
            self.swanlab.finish()


def metric_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"metric tensor must be scalar, got {tuple(value.shape)}")
        value = float(value.detach().float().cpu())
    else:
        value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"metric is non-finite: {value}")
    return value


def reduce_metrics(accelerator: Any, metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in sorted(metrics):
        value = metrics[name]
        tensor = (
            value.detach().float().reshape(())
            if isinstance(value, torch.Tensor)
            else torch.tensor(float(value), device=accelerator.device)
        )
        result[name] = float(
            accelerator.reduce(tensor.to(accelerator.device), reduction="mean").cpu()
        )
    return result


def reduce_profile_statistics(
    accelerator: Any,
    statistics: torch.Tensor,
    names: tuple[str, ...],
) -> dict[str, float]:
    if (
        not isinstance(statistics, torch.Tensor)
        or tuple(statistics.shape) != (len(names),)
        or not torch.is_floating_point(statistics)
    ):
        raise ValueError("profile statistics must be one floating-point vector")
    reduced = accelerator.reduce(
        statistics.detach().to(device=accelerator.device, dtype=torch.float64),
        reduction="sum",
    ).cpu()
    values = {name: float(reduced[index].item()) for index, name in enumerate(names)}
    position_count = values["position_count"]
    if position_count <= 0:
        raise ValueError("profile position count must be positive")
    result = {
        "profile/actual_hit_rate": values["actual_hit_weight"] / position_count,
    }
    hit_weight = values["hit_value_weight"]
    if hit_weight > 0:
        result["profile/hit_mse"] = values["hit_error_sum"] / hit_weight
        result["profile/hit_loss"] = values["hit_loss_sum"] / hit_weight
    non_hit_weight = values["non_hit_value_weight"]
    if non_hit_weight > 0:
        result["profile/non_hit_mse"] = values["non_hit_error_sum"] / non_hit_weight
        result["profile/non_hit_loss"] = values["non_hit_loss_sum"] / non_hit_weight
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("profile metrics are non-finite")
    return result


def format_summary(
    step: int,
    max_steps: int,
    metrics: Mapping[str, float],
    *,
    elapsed: float,
    completed: int,
) -> str:
    remaining = elapsed / completed * (max_steps - step)
    return (
        f"train progress update={step}/{max_steps} loss={metrics['loss']:.6g} "
        f"elapsed={elapsed:.1f}s eta={remaining:.1f}s"
    )


TRAIN_LOSS_METRICS = ("loss", "flow_mse")


def grouped_log_metrics(
    phase: str,
    metrics: Mapping[str, float],
    dataset_metrics: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, float]:
    if phase not in {"train", "eval", "eval_visualization"}:
        raise ValueError("metric phase must be train, eval, or eval_visualization")
    output = {f"{phase}/{name}": float(value) for name, value in metrics.items()}
    grouped = {} if dataset_metrics is None else dataset_metrics
    for dataset in sorted(grouped):
        if not dataset or "/" in dataset or "\\" in dataset:
            raise ValueError("dataset metric names must be path components")
        output.update(
            {
                f"{dataset}/{phase}/{name}": float(value)
                for name, value in grouped[dataset].items()
            }
        )
    if any(not math.isfinite(value) for value in output.values()):
        raise FloatingPointError("grouped metrics are non-finite")
    return output


def reduce_grouped_means(
    accelerator: Any,
    groups: tuple[str, ...],
    metric_names: tuple[str, ...],
    sums: Mapping[str, Mapping[str, float]],
    counts: Mapping[str, int],
) -> dict[str, dict[str, float]]:
    if not groups or not metric_names:
        raise ValueError("metric groups and names must be non-empty")
    packed: list[float] = []
    for group in groups:
        values = sums.get(group, {})
        packed.extend(float(values.get(name, 0.0)) for name in metric_names)
        packed.append(float(counts.get(group, 0)))
    reduced = accelerator.reduce(
        torch.tensor(
            packed,
            device=accelerator.device,
            dtype=torch.float64,
        ),
        reduction="sum",
    ).cpu()
    result: dict[str, dict[str, float]] = {}
    width = len(metric_names) + 1
    for group_index, group in enumerate(groups):
        offset = group_index * width
        count = int(reduced[offset + len(metric_names)].item())
        if count > 0:
            result[group] = {
                name: float(reduced[offset + metric_index].item() / count)
                for metric_index, name in enumerate(metric_names)
            }
    return result


def reduce_dataset_training_metrics(
    accelerator: Any,
    datasets: tuple[str, ...],
    sums: Mapping[str, Mapping[str, float]],
    counts: Mapping[str, int],
) -> dict[str, dict[str, float]]:
    return reduce_grouped_means(
        accelerator,
        datasets,
        TRAIN_LOSS_METRICS,
        sums,
        counts,
    )


def grouped_training_log_metrics(
    train: Mapping[str, Any],
    augmentation: Mapping[str, Any],
    per_dataset: Mapping[str, Mapping[str, Any]],
    augmentation_names: tuple[str, ...],
) -> dict[str, float]:
    result = grouped_log_metrics("train", train, per_dataset)
    result.update({name: float(augmentation[name]) for name in augmentation_names})
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("training metrics are non-finite")
    return result


@dataclass
class MetricAccumulator:
    values: dict[str, float] = field(default_factory=dict)
    count: int = 0
    dataset_sums: dict[str, dict[str, float]] = field(default_factory=dict)
    dataset_counts: dict[str, int] = field(default_factory=dict)
    profile_statistics: torch.Tensor | None = None
    profile_window: torch.Tensor | None = None

    def reset_update(self) -> None:
        """Reset update metrics while retaining the profiling window."""
        self.values.clear()
        self.count = 0
        self.dataset_sums.clear()
        self.dataset_counts.clear()
        self.profile_statistics = None


def summarize_update(
    session: TrainingSession,
    accumulated: MetricAccumulator,
    update: BatchUpdate,
    *,
    started: float,
) -> tuple[dict[str, float], dict[str, float]]:
    accelerator, optimizer, data = session.accelerator, session.optimizer, session.data
    metrics = {
        name: value / accumulated.count for name, value in accumulated.values.items()
    }
    metrics.update(
        grad_norm=update.grad_norm,
        lr=float(optimizer.param_groups[0]["lr"]),
        step_time=time.perf_counter() - started,
    )
    accumulated.values.clear()
    accumulated.count = 0
    reduced = reduce_metrics(accelerator, metrics)
    augmentation_names = tuple(
        name for name in reduced if name.startswith("augmentation/")
    )
    train_metrics = {
        name: value for name, value in reduced.items() if name not in augmentation_names
    }
    per_dataset = reduce_dataset_training_metrics(
        accelerator, data.datasets, accumulated.dataset_sums, accumulated.dataset_counts
    )
    accumulated.dataset_sums.clear()
    accumulated.dataset_counts.clear()
    last_metrics = grouped_training_log_metrics(
        train_metrics, reduced, per_dataset, augmentation_names
    )
    if update.timings:
        update.timings.update(update.condition_timings)
        last_metrics.update(reduce_timing_metrics(accelerator, update.timings))
        last_metrics.update(
            reduce_profile_statistics(
                accelerator, accumulated.profile_window, PROFILE_STATISTIC_NAMES
            )
        )
        accumulated.profile_window = None
    return last_metrics, train_metrics
