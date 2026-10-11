from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alc_video.utils.config import as_mapping, load_document, merge_defaults

AUGMENTATION_DEFAULTS = {
    "geometry": {
        "probability": 0.6,
        "source_drop": {"enabled": True, "max_drop_ratio": 0.3},
        "visibility_culling": {
            "enabled": True,
            "strategy": [
                "lateral_look_at_v1",
                "yaw_pan_v1",
                "pitch_pan_v1",
                "zoom_in_v1",
                "zoom_out_v1",
            ],
            "strategy_probability": [0.2, 0.2, 0.2, 0.2, 0.2],
            "lateral_translation": [0.1, 0.3],
            "pan_degrees": [5.0, 25.0],
            "pitch_degrees": [5.0, 25.0],
            "zoom_in_focal_scale": [1.1, 1.3],
            "zoom_out_focal_scale": [0.7, 0.9],
        },
    },
    "transport": {
        "first_frame": {"enabled": True, "probability": 0.1, "use_geometry": True},
        "motion_threshold": {"enabled": True, "range": [0.05, 0.2]},
    },
    "condition": {
        "prompt_drop": {
            "probability": 0.1,
            "unconditional_context": "../local-assets/augmentation/wan21-unconditional-context.safetensors",
        },
        "latent_drop": {"all_probability": 0.1, "per_source_probability": 0.1},
    },
}


VISIBILITY_CULLING_STRATEGIES = (
    "lateral_look_at_v1",
    "yaw_pan_v1",
    "pitch_pan_v1",
    "zoom_in_v1",
    "zoom_out_v1",
)


@dataclass(frozen=True, slots=True)
class SourceDropConfig:
    enabled: bool
    max_drop_ratio: float | None


@dataclass(frozen=True, slots=True)
class VisibilityCullingConfig:
    enabled: bool
    strategies: tuple[str, ...]
    strategy_probabilities: tuple[float, ...]
    lateral_translation_min: float | None
    lateral_translation_max: float | None
    pan_degrees_min: float | None
    pan_degrees_max: float | None
    pitch_degrees_min: float | None
    pitch_degrees_max: float | None
    zoom_in_focal_scale_min: float | None
    zoom_in_focal_scale_max: float | None
    zoom_out_focal_scale_min: float | None
    zoom_out_focal_scale_max: float | None

    @property
    def enabled_strategies(self) -> tuple[str, ...]:
        return tuple(
            strategy
            for strategy, probability in zip(
                self.strategies,
                self.strategy_probabilities,
                strict=True,
            )
            if probability > 0.0
        )


@dataclass(frozen=True, slots=True)
class GeometryAugmentationConfig:
    probability: float
    source_drop: SourceDropConfig
    visibility_culling: VisibilityCullingConfig


@dataclass(frozen=True, slots=True)
class MotionThresholdConfig:
    enabled: bool
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class FirstFrameConfig:
    enabled: bool
    probability: float
    use_geometry: bool


@dataclass(frozen=True, slots=True)
class TransportAugmentationConfig:
    first_frame: FirstFrameConfig
    motion_threshold: MotionThresholdConfig


@dataclass(frozen=True, slots=True)
class PromptDropConfig:
    probability: float
    unconditional_context: Path | None


@dataclass(frozen=True, slots=True)
class LatentDropConfig:
    all_probability: float
    per_source_probability: float


@dataclass(frozen=True, slots=True)
class ConditionAugmentationConfig:
    prompt_drop: PromptDropConfig
    latent_drop: LatentDropConfig


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    source_path: Path
    geometry: GeometryAugmentationConfig
    transport: TransportAugmentationConfig
    condition: ConditionAugmentationConfig

    @property
    def has_enabled_operators(self) -> bool:
        motion_threshold = self.transport.motion_threshold
        return any(
            (
                (self.geometry.probability > 0.0 and self.geometry.source_drop.enabled),
                (
                    self.geometry.probability > 0.0
                    and self.geometry.visibility_culling.enabled
                ),
                (motion_threshold.enabled),
                (
                    self.transport.first_frame.enabled
                    and self.transport.first_frame.probability > 0.0
                ),
                (self.condition.prompt_drop.probability > 0.0),
                (self.condition.latent_drop.all_probability > 0.0),
                (self.condition.latent_drop.per_source_probability > 0.0),
            )
        )


def _exact(value: Any, *, name: str, fields: set[str]) -> dict[str, Any]:
    result = as_mapping(value, name=name)
    if set(result) != fields:
        raise ValueError(
            f"{name} fields must be {sorted(fields)}, got {sorted(result)}"
        )
    return result


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _float(value: Any, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _probability(value: Any, *, name: str) -> float:
    result = _float(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must be <= 1")
    return result


def _asset_path(
    value: Any,
    *,
    name: str,
    root: Path,
    must_exist: bool,
) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be null or a non-empty local file path")
    if "://" in str(value):
        raise ValueError(f"{name} must be a local file path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _source_drop(value: Any) -> SourceDropConfig:
    name = "geometry.source_drop"
    raw = _exact(value, name=name, fields={"enabled", "max_drop_ratio"})
    enabled = _bool(raw["enabled"], name=f"{name}.enabled")
    ratio_value = raw["max_drop_ratio"]
    if ratio_value is not None:
        ratio = _float(ratio_value, name=f"{name}.max_drop_ratio")
        if ratio < 1.0 / 11.0 or ratio >= 1.0:
            raise ValueError(f"{name}.max_drop_ratio must satisfy 1/11 <= ratio < 1")
    elif enabled:
        raise ValueError(f"enabled {name} requires max_drop_ratio")
    else:
        ratio = None
    return SourceDropConfig(
        enabled=enabled,
        max_drop_ratio=ratio,
    )


def _first_frame(value: Any) -> FirstFrameConfig:
    name = "transport.first_frame"
    raw = _exact(
        value,
        name=name,
        fields={"enabled", "probability", "use_geometry"},
    )
    return FirstFrameConfig(
        enabled=_bool(raw["enabled"], name=f"{name}.enabled"),
        probability=_probability(raw["probability"], name=f"{name}.probability"),
        use_geometry=_bool(
            raw["use_geometry"],
            name=f"{name}.use_geometry",
        ),
    )


def _optional_range(
    value: Any,
    *,
    name: str,
    lower_exclusive: float,
    upper_exclusive: float | None = None,
) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{name} must be [minimum, maximum] or null")
    minimum = _float(value[0], name=f"{name}[0]")
    maximum = _float(value[1], name=f"{name}[1]")
    if minimum <= lower_exclusive:
        raise ValueError(f"{name} values must be > {lower_exclusive}")
    if maximum < minimum:
        raise ValueError(f"{name} maximum must be >= minimum")
    if upper_exclusive is not None and maximum >= upper_exclusive:
        raise ValueError(f"{name} values must be < {upper_exclusive}")
    return minimum, maximum


def _visibility_culling(value: Any) -> VisibilityCullingConfig:
    name = "geometry.visibility_culling"
    raw = _exact(
        value,
        name=name,
        fields={
            "enabled",
            "strategy",
            "strategy_probability",
            "lateral_translation",
            "pan_degrees",
            "pitch_degrees",
            "zoom_in_focal_scale",
            "zoom_out_focal_scale",
        },
    )
    enabled = _bool(raw["enabled"], name=f"{name}.enabled")
    strategies_raw = raw["strategy"]
    probabilities_raw = raw["strategy_probability"]
    if not isinstance(strategies_raw, list) or any(
        not isinstance(item, str) or not item for item in strategies_raw
    ):
        raise TypeError(f"{name}.strategy must be a list of strategy names")
    strategies = tuple(strategies_raw)
    if len(set(strategies)) != len(strategies):
        raise ValueError(f"{name}.strategy entries must be unique")
    unknown = set(strategies) - set(VISIBILITY_CULLING_STRATEGIES)
    if unknown:
        raise ValueError(
            f"{name}.strategy contains unsupported values: {sorted(unknown)}"
        )
    if not isinstance(probabilities_raw, list):
        raise TypeError(f"{name}.strategy_probability must be a list")
    probabilities = tuple(
        _float(value, name=f"{name}.strategy_probability[{index}]")
        for index, value in enumerate(probabilities_raw)
    )
    if len(probabilities) != len(strategies):
        raise ValueError(
            f"{name}.strategy and strategy_probability must have equal lengths"
        )
    if any(value < 0.0 for value in probabilities):
        raise ValueError(f"{name}.strategy_probability values must be >= 0")
    if enabled and not math.isclose(sum(probabilities), 1.0, abs_tol=1.0e-9):
        raise ValueError(f"{name}.strategy_probability must sum to 1")
    lateral = _optional_range(
        raw["lateral_translation"],
        name=f"{name}.lateral_translation",
        lower_exclusive=0.0,
    )
    pan = _optional_range(
        raw["pan_degrees"],
        name=f"{name}.pan_degrees",
        lower_exclusive=0.0,
        upper_exclusive=90.0,
    )
    pitch = _optional_range(
        raw["pitch_degrees"],
        name=f"{name}.pitch_degrees",
        lower_exclusive=0.0,
        upper_exclusive=90.0,
    )
    zoom_in = _optional_range(
        raw["zoom_in_focal_scale"],
        name=f"{name}.zoom_in_focal_scale",
        lower_exclusive=1.0,
    )
    zoom_out = _optional_range(
        raw["zoom_out_focal_scale"],
        name=f"{name}.zoom_out_focal_scale",
        lower_exclusive=0.0,
        upper_exclusive=1.0,
    )
    config = VisibilityCullingConfig(
        enabled=enabled,
        strategies=strategies,
        strategy_probabilities=probabilities,
        lateral_translation_min=lateral[0],
        lateral_translation_max=lateral[1],
        pan_degrees_min=pan[0],
        pan_degrees_max=pan[1],
        pitch_degrees_min=pitch[0],
        pitch_degrees_max=pitch[1],
        zoom_in_focal_scale_min=zoom_in[0],
        zoom_in_focal_scale_max=zoom_in[1],
        zoom_out_focal_scale_min=zoom_out[0],
        zoom_out_focal_scale_max=zoom_out[1],
    )
    if enabled and not config.enabled_strategies:
        raise ValueError(
            f"enabled {name} requires at least one positive-probability strategy"
        )
    required_ranges = {
        "lateral_look_at_v1": config.lateral_translation_min,
        "yaw_pan_v1": config.pan_degrees_min,
        "pitch_pan_v1": config.pitch_degrees_min,
        "zoom_in_v1": config.zoom_in_focal_scale_min,
        "zoom_out_v1": config.zoom_out_focal_scale_min,
    }
    missing = [
        strategy
        for strategy in config.enabled_strategies
        if required_ranges[strategy] is None
    ]
    if enabled and missing:
        raise ValueError(f"enabled {name} strategies lack parameter ranges: {missing}")
    return config


def _motion_threshold(value: Any) -> MotionThresholdConfig:
    name = "transport.motion_threshold"
    raw = _exact(value, name=name, fields={"enabled", "range"})
    enabled = _bool(raw["enabled"], name=f"{name}.enabled")
    range_value = raw["range"]
    if range_value is not None:
        if not isinstance(range_value, list) or len(range_value) != 2:
            raise TypeError(f"{name}.range must be [minimum, maximum]")
        minimum = _float(range_value[0], name=f"{name}.range[0]")
        maximum = _float(range_value[1], name=f"{name}.range[1]")
        if maximum < minimum:
            raise ValueError(f"{name}.range maximum must be >= minimum")
    elif enabled:
        raise ValueError(f"enabled {name} requires range")
    else:
        minimum = None
        maximum = None
    return MotionThresholdConfig(
        enabled=enabled,
        minimum=minimum,
        maximum=maximum,
    )


def _prompt_drop(
    value: Any,
    *,
    root: Path,
    require_assets: bool,
) -> PromptDropConfig:
    name = "condition.prompt_drop"
    raw = _exact(
        value,
        name=name,
        fields={"probability", "unconditional_context"},
    )
    probability = _probability(
        raw["probability"],
        name=f"{name}.probability",
    )
    path = _asset_path(
        raw["unconditional_context"],
        name=f"{name}.unconditional_context",
        root=root,
        must_exist=require_assets and probability > 0.0,
    )
    if probability > 0.0 and path is None:
        raise ValueError(f"enabled {name} requires unconditional_context")
    return PromptDropConfig(
        probability=probability,
        unconditional_context=path,
    )


def _latent_drop(value: Any) -> LatentDropConfig:
    name = "condition.latent_drop"
    raw = as_mapping(value, name=name)
    expected = {"all_probability", "per_source_probability"}
    if set(raw) != expected:
        raise ValueError(f"{name} fields must be {sorted(expected)}, got {sorted(raw)}")
    all_probability = _probability(
        raw["all_probability"],
        name=f"{name}.all_probability",
    )
    per_source_probability = _probability(
        raw["per_source_probability"],
        name=f"{name}.per_source_probability",
    )
    return LatentDropConfig(
        all_probability=all_probability,
        per_source_probability=per_source_probability,
    )


def load_augmentation_config(
    path: str | Path,
    *,
    require_assets: bool = True,
) -> AugmentationConfig:
    if not isinstance(require_assets, bool):
        raise TypeError("require_assets must be bool")
    source, document = load_document(path)
    value = merge_defaults(AUGMENTATION_DEFAULTS, document)
    raw = _exact(
        value,
        name="Augmentation config",
        fields={"geometry", "transport", "condition"},
    )

    geometry_raw = _exact(
        raw["geometry"],
        name="geometry",
        fields={
            "probability",
            "source_drop",
            "visibility_culling",
        },
    )
    geometry_probability = _probability(
        geometry_raw["probability"],
        name="geometry.probability",
    )
    source_drop = _source_drop(geometry_raw["source_drop"])
    visibility = _visibility_culling(geometry_raw["visibility_culling"])
    transport_raw = _exact(
        raw["transport"],
        name="transport",
        fields={"first_frame", "motion_threshold"},
    )
    first_frame = _first_frame(transport_raw["first_frame"])
    motion_threshold = _motion_threshold(transport_raw["motion_threshold"])
    condition_raw = _exact(
        raw["condition"],
        name="condition",
        fields={"prompt_drop", "latent_drop"},
    )
    prompt = _prompt_drop(
        condition_raw["prompt_drop"],
        root=source.parent,
        require_assets=require_assets,
    )
    latent = _latent_drop(condition_raw["latent_drop"])
    return AugmentationConfig(
        source_path=source,
        geometry=GeometryAugmentationConfig(
            probability=geometry_probability,
            source_drop=source_drop,
            visibility_culling=visibility,
        ),
        transport=TransportAugmentationConfig(
            first_frame=first_frame,
            motion_threshold=motion_threshold,
        ),
        condition=ConditionAugmentationConfig(
            prompt_drop=prompt,
            latent_drop=latent,
        ),
    )


__all__ = [
    "VISIBILITY_CULLING_STRATEGIES",
    "AugmentationConfig",
    "ConditionAugmentationConfig",
    "FirstFrameConfig",
    "GeometryAugmentationConfig",
    "LatentDropConfig",
    "PromptDropConfig",
    "SourceDropConfig",
    "TransportAugmentationConfig",
    "VisibilityCullingConfig",
    "load_augmentation_config",
]
