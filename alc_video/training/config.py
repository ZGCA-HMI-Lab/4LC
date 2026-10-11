from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from latent_cache import LatentCacheConfig

from alc_video.data.config import DataConfig
from alc_video.encoders.config import EncoderCatalog, load_encoder_catalog
from alc_video.models.config import ModelConfig, parse_model_config
from alc_video.runtime.profiles import (
    MODEL_PATH_FIELDS,
    load_cache_config,
    load_model_profile,
)
from alc_video.utils.config import (
    apply_overrides,
    as_mapping,
    default_values,
    finite_float,
    load_document,
    merge_defaults,
    optional_string,
    positive_int,
    reject_unknown_fields,
    resolve_document_paths,
    resolve_path,
)


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 0


@dataclass(frozen=True)
class CheckpointConfig:
    output_path: Path = Path("../outputs/training")
    resume_from: Path | None = None
    max_to_keep: int = 2
    initial_weights: Path | None = None


@dataclass(frozen=True)
class RunConfig:
    runtime: RuntimeConfig
    data: DataConfig
    model: ModelConfig
    encoder_catalog: EncoderCatalog
    representations: dict[str, str]
    worldmirror_checkpoint: Path
    wan_model_root: Path
    cache: LatentCacheConfig
    augmentation: Path | None
    propagate_rgb: bool
    training: TrainingConfig
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig


def load_run_config(
    path: str | Path,
    overrides: tuple[str, ...] = (),
    *,
    model: str | Path | None = None,
) -> RunConfig:
    paths = (
        "model",
        "cache",
        "data.root",
        "augmentation",
        "checkpoint.output_path",
        "checkpoint.resume_from",
        "checkpoint.initial_weights",
    )
    source, document = load_document(path, path_fields=paths)
    reject_unknown_fields(document, set(run_defaults()) | {"model", "cache"})
    profile = load_model_profile(model if model is not None else document["model"])
    value = merge_defaults(run_defaults(), document)
    for section in ("model", "encoders"):
        value[section] = profile[section]
    value = resolve_document_paths(value, root=source.parent, path_fields=paths[1:])
    value["cache"] = load_cache_config(value["cache"])
    value = apply_overrides(value, overrides)
    overridden_paths = tuple(
        item.partition("=")[0]
        for item in overrides
        if item.partition("=")[0] in (*paths[2:], *MODEL_PATH_FIELDS)
    )
    value = resolve_document_paths(value, root=Path.cwd(), path_fields=overridden_paths)
    root = source.parent

    def local_path(raw: str) -> Path:
        return resolve_path(raw, root=root)

    runtime = RuntimeConfig(seed=int(value["runtime"]["seed"]))
    reject_unknown_fields(
        value["runtime"], default_values(RuntimeConfig), name="runtime"
    )
    training = parse_training_config(value["training"])
    data = value["data"]
    reject_unknown_fields(data, set(run_defaults()["data"]) | {"root"}, name="data")
    model_config = parse_model_config(value["model"], root=root)
    encoders = value["encoders"]
    reject_unknown_fields(
        encoders,
        {"config", "representations", "worldmirror", "wan_model_root"},
        name="encoders",
    )
    catalog = load_encoder_catalog(local_path(encoders["config"]), require_assets=False)
    bindings = dict(encoders["representations"])
    for name in bindings.values():
        catalog.visual_encoder(name)
    worldmirror = catalog.worldmirror_encoder(encoders["worldmirror"])
    cache = dict(value["cache"])
    reject_unknown_fields(
        cache, (field.name for field in fields(LatentCacheConfig)), name="cache"
    )
    cache["image_size"] = tuple(cache["image_size"])
    checkpoint = value["checkpoint"]
    reject_unknown_fields(
        checkpoint, default_values(CheckpointConfig), name="checkpoint"
    )
    augmentation = value["augmentation"]
    if not isinstance(value["propagate_rgb"], bool):
        raise TypeError("propagate_rgb must be bool")
    return RunConfig(
        runtime=runtime,
        data=DataConfig(
            root=local_path(data["root"]),
            source=data["source"],
            sample_count=None
            if data["sample_count"] is None
            else positive_int(data["sample_count"], name="data.sample_count"),
            planning_seed=int(data["planning_seed"]),
            image_size=cache["image_size"],
        ),
        model=model_config,
        encoder_catalog=catalog,
        representations=bindings,
        worldmirror_checkpoint=worldmirror.checkpoint,
        wan_model_root=model_config.dit_path
        if encoders["wan_model_root"] is None
        else local_path(encoders["wan_model_root"]),
        cache=LatentCacheConfig(**cache),
        augmentation=None if augmentation is None else local_path(augmentation),
        propagate_rgb=value["propagate_rgb"],
        training=training,
        optimizer=parse_optimizer_config(value["optimizer"]),
        checkpoint=CheckpointConfig(
            output_path=local_path(checkpoint["output_path"]),
            resume_from=None
            if checkpoint["resume_from"] is None
            else local_path(checkpoint["resume_from"]),
            max_to_keep=positive_int(
                checkpoint["max_to_keep"], name="checkpoint.max_to_keep"
            ),
            initial_weights=None
            if checkpoint["initial_weights"] is None
            else local_path(checkpoint["initial_weights"]),
        ),
        logging=parse_logging_config(value["logging"]),
    )


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    lr: float = 1.0e-5
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1.0e-8
    schedule: str = "constant"
    clip_max_norm: float = 1.0
    clip_norm_type: float = 2.0


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    tensorboard_enabled: bool = False
    tensorboard_logdir: str | None = "tensorboard"
    swanlab_enabled: bool = False
    swanlab_project: str = "4lc-video"
    swanlab_experiment_name: str = "pretrain"
    swanlab_workspace: str | None = None
    swanlab_mode: str | None = None
    swanlab_logdir: str | None = "swanlog"
    train_log_enabled: bool = True
    train_log_path: str | None = "train.log"


def parse_optimizer_config(
    value: Any,
) -> OptimizerConfig:
    raw = as_mapping(merge_defaults(optimizer_defaults(), value), name="optimizer")
    reject_unknown_fields(raw, optimizer_defaults(), name="optimizer")
    lr = finite_float(raw["lr"], name="optimizer.lr")
    if lr <= 0.0:
        raise ValueError("optimizer.lr must be > 0")
    betas = raw["betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optimizer.betas must contain two values")
    parsed_betas = tuple(
        finite_float(item, name=f"optimizer.betas[{index}]")
        for index, item in enumerate(betas)
    )
    if any(item < 0.0 or item >= 1.0 for item in parsed_betas):
        raise ValueError("optimizer.betas values must be in [0,1)")
    weight_decay = finite_float(
        raw["weight_decay"],
        name="optimizer.weight_decay",
    )
    if weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be >= 0")
    eps = finite_float(raw["eps"], name="optimizer.eps")
    if eps <= 0.0:
        raise ValueError("optimizer.eps must be > 0")
    schedule = as_mapping(raw["schedule"], name="optimizer.schedule")
    reject_unknown_fields(schedule, {"type"}, name="optimizer.schedule")
    if schedule["type"] != "constant":
        raise ValueError("optimizer.schedule.type must be 'constant'")
    clip_max_norm = finite_float(
        raw["clip_max_norm"],
        name="optimizer.clip_max_norm",
    )
    clip_norm_type = finite_float(
        raw["clip_norm_type"],
        name="optimizer.clip_norm_type",
    )
    if clip_max_norm <= 0.0 or clip_norm_type <= 0.0:
        raise ValueError("optimizer gradient clip values must be > 0")
    return OptimizerConfig(
        lr=lr,
        betas=(parsed_betas[0], parsed_betas[1]),
        weight_decay=weight_decay,
        eps=eps,
        schedule="constant",
        clip_max_norm=clip_max_norm,
        clip_norm_type=clip_norm_type,
    )


def parse_logging_config(value: Any) -> LoggingConfig:
    raw = as_mapping(
        merge_defaults(default_values(LoggingConfig), value), name="logging"
    )
    reject_unknown_fields(raw, default_values(LoggingConfig), name="logging")
    for name in (
        "tensorboard_enabled",
        "swanlab_enabled",
        "train_log_enabled",
    ):
        if not isinstance(raw[name], bool):
            raise TypeError(f"logging.{name} must be bool")
    for name in ("swanlab_project", "swanlab_experiment_name"):
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise ValueError(f"logging.{name} must be non-empty")
    return LoggingConfig(
        tensorboard_enabled=raw["tensorboard_enabled"],
        tensorboard_logdir=optional_string(
            raw["tensorboard_logdir"],
            name="logging.tensorboard_logdir",
        ),
        swanlab_enabled=raw["swanlab_enabled"],
        swanlab_project=raw["swanlab_project"],
        swanlab_experiment_name=raw["swanlab_experiment_name"],
        swanlab_workspace=optional_string(
            raw["swanlab_workspace"],
            name="logging.swanlab_workspace",
        ),
        swanlab_mode=optional_string(
            raw["swanlab_mode"],
            name="logging.swanlab_mode",
        ),
        swanlab_logdir=optional_string(
            raw["swanlab_logdir"],
            name="logging.swanlab_logdir",
        ),
        train_log_enabled=raw["train_log_enabled"],
        train_log_path=optional_string(
            raw["train_log_path"],
            name="logging.train_log_path",
        ),
    )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    sample_exposure: float = 1.6
    max_updates: int | None = None
    max_steps: int = 0
    num_workers: int = 1
    prefetch_factor: int = 2
    shuffle_buffer_size: int = 1000
    gradient_accumulation_steps: int = 1
    zero_optimizer_offload: str = "none"
    print_freq: int = 50
    save_freq: int = 2000
    profile_freq: int = 50


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return cast(int, value)


def _positive_float(value: Any, *, name: str) -> float:
    result = finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def parse_training_config(value: Any) -> TrainingConfig:
    raw = as_mapping(merge_defaults(training_defaults(), value), name="training")
    reject_unknown_fields(raw, training_defaults(), name="training")
    sample_exposure = _positive_float(
        raw["sample_exposure"],
        name="training.sample_exposure",
    )
    max_updates = (
        None
        if raw["max_updates"] is None
        else positive_int(raw["max_updates"], name="training.max_updates")
    )
    gradient_accumulation_steps = positive_int(
        raw["gradient_accumulation_steps"],
        name="training.gradient_accumulation_steps",
    )
    zero_optimizer_offload = raw["zero_optimizer_offload"]
    if zero_optimizer_offload not in {"none", "cpu"}:
        raise ValueError("training.zero_optimizer_offload must be 'none' or 'cpu'")
    return TrainingConfig(
        sample_exposure=sample_exposure,
        max_updates=max_updates,
        max_steps=0,
        num_workers=_nonnegative_int(
            raw["num_workers"],
            name="training.num_workers",
        ),
        prefetch_factor=positive_int(
            raw["prefetch_factor"],
            name="training.prefetch_factor",
        ),
        shuffle_buffer_size=positive_int(
            raw["shuffle_buffer_size"],
            name="training.shuffle_buffer_size",
        ),
        gradient_accumulation_steps=gradient_accumulation_steps,
        zero_optimizer_offload=zero_optimizer_offload,
        print_freq=_nonnegative_int(
            raw["print_freq"],
            name="training.print_freq",
        ),
        save_freq=_nonnegative_int(
            raw["save_freq"],
            name="training.save_freq",
        ),
        profile_freq=_nonnegative_int(
            raw["profile_freq"],
            name="training.profile_freq",
        ),
    )


def optimizer_defaults() -> dict:
    values = default_values(OptimizerConfig)
    values["schedule"] = {"type": values["schedule"]}
    return values


def training_defaults() -> dict:
    values = default_values(TrainingConfig)
    del values["max_steps"]
    return values


def run_defaults() -> dict:
    return {
        "runtime": default_values(RuntimeConfig),
        "data": {"planning_seed": 0, "sample_count": None, "source": "spatialvid"},
        "augmentation": "augmentation.yaml",
        "propagate_rgb": False,
        "training": training_defaults(),
        "optimizer": optimizer_defaults(),
        "logging": default_values(LoggingConfig),
        "checkpoint": default_values(CheckpointConfig),
    }
