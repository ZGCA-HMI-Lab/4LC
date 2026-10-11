from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from alc_video.conditioning.config import DEFAULT_KEYFRAMES
from alc_video.inference.controls import CameraControl, RolloutAction, TimeControl
from alc_video.models.config import ModelConfig, parse_model_config
from alc_video.runtime.profiles import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    load_cache_config,
    load_model_profile,
)
from alc_video.utils.errors import input_errors

if TYPE_CHECKING:
    from latent_cache import LatentCacheConfig


@dataclass(frozen=True)
class SamplingConfig:
    steps: int = 4
    vae_tiled: bool = False
    control_scales: tuple[float, ...] | None = None


@dataclass(frozen=True)
class EncoderSelection:
    config: Path
    representations: tuple[tuple[str, str], ...]
    worldmirror: str
    wan_model_root: Path


@dataclass(frozen=True)
class GenerationConfig:
    model: ModelConfig
    encoders: EncoderSelection
    cache: LatentCacheConfig
    weights: Path | None
    lightx2v: Path | None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    seed: int = 0
    device: str = "cuda:0"


@dataclass(frozen=True)
class SourceInput:
    path: Path
    type: Literal["image", "video"] = "video"
    start: float = 0.0
    sampling_fps: float | None = None
    crop: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class InferenceRequest:
    generation: GenerationConfig
    source: SourceInput
    prompt: str
    output: Path = Path("outputs/inference/generated.mp4")
    camera: CameraControl = field(default_factory=CameraControl)
    time: TimeControl = field(default_factory=TimeControl)
    mode: Literal["control", "i2v", "geometry_i2v"] = "control"
    num_keyframes: int = DEFAULT_KEYFRAMES
    keyframes: tuple[int, ...] | None = None
    transport: Literal["velocity", "identity"] = "velocity"
    drop_source_frames: tuple[int, ...] = ()
    propagate_rgb: bool = False


@dataclass(frozen=True)
class RolloutRequest:
    generation: GenerationConfig
    source: SourceInput
    prompts: tuple[str, ...]
    actions: tuple[RolloutAction, ...]
    revisit: bool = False
    output: Path = Path("outputs/rollout/generated.mp4")
    mode: Literal["i2v", "geometry_i2v"] = "geometry_i2v"
    chunk_size: int = 81
    num_keyframes: int = DEFAULT_KEYFRAMES
    use_cache: bool = True
    order: Literal["earliest_first", "latest_first"] = "earliest_first"
    max_entries: int = 8
    coverage_threshold: float = 0.95


Request = InferenceRequest | RolloutRequest


def load_generation_config(
    model: str | Path = DEFAULT_MODEL, *, cache: str | Path = DEFAULT_CACHE
) -> GenerationConfig:
    from latent_cache import LatentCacheConfig

    with input_errors():
        profile = load_model_profile(model)
        encoders = profile["encoders"]
        cache_values = load_cache_config(cache)
        cache_values["image_size"] = tuple(cache_values["image_size"])
        assets = Path(__file__).resolve().parents[2] / "local-assets/models"
        return GenerationConfig(
            model=parse_model_config(profile["model"], root=Path.cwd()),
            encoders=EncoderSelection(
                config=Path(encoders["config"]),
                representations=tuple(encoders["representations"].items()),
                worldmirror=encoders["worldmirror"],
                wan_model_root=Path(encoders["wan_model_root"]),
            ),
            cache=LatentCacheConfig(**cache_values),
            weights=(assets / f"4lc-video-{Path(model).stem}.safetensors").resolve(),
            lightx2v=(
                assets
                / "lightx2v/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
            ).resolve(),
        )
