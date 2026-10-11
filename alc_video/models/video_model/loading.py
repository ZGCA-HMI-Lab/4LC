from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from diffsynth.models.wan_video_dit import WanModel


WAN_VIDEO_CODEC_RELATIVE_PATH = Path("Wan2.1_VAE.pth")
WAN_TEXT_ENCODER_RELATIVE_PATH = Path("models_t5_umt5-xxl-enc-bf16.pth")
WAN_TEXT_TOKENIZER_RELATIVE_PATH = Path("google/umt5-xxl")
WAN_WEIGHT_PATTERN = "diffusion_pytorch_model*.safetensors"


def resolve_wan_weights(
    value: Any,
    *,
    root: str | Path | None = None,
) -> tuple[Path, ...]:
    if (
        not isinstance(value, (str, Path))
        or not str(value).strip()
        or "://" in str(value)
    ):
        raise ValueError("model.dit_path must be a non-empty local directory")
    path = Path(value).expanduser()
    if not path.is_absolute():
        base = Path(__file__).resolve().parents[2] if root is None else Path(root)
        path = base / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"model.dit_path does not exist: {path}")
    weights = tuple(
        sorted(
            candidate
            for candidate in path.glob(WAN_WEIGHT_PATTERN)
            if candidate.is_file()
        )
    )
    if not weights:
        raise FileNotFoundError(
            f"Wan DiT weights not found: {path / WAN_WEIGHT_PATTERN}"
        )
    exact = path / "diffusion_pytorch_model.safetensors"
    if exact in weights and len(weights) != 1:
        raise ValueError(f"Wan directory contains complete and sharded weights: {path}")
    return weights


def resolve_wan_checkpoint(
    value: str | Path,
    *,
    name: str,
) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} checkpoint does not exist: {path}")
    return path


def load_wan_dit(dit_path: str | Path) -> WanModel:
    from diffsynth.core import ModelConfig
    from diffsynth.models.model_loader import ModelPool

    dit_files = [str(path) for path in resolve_wan_weights(dit_path)]
    model_config = ModelConfig(
        path=dit_files[0] if len(dit_files) == 1 else dit_files,
        skip_download=True,
    )
    vram_config = model_config.vram_config()
    vram_config["computation_dtype"] = torch.bfloat16
    vram_config["computation_device"] = "cpu"
    model_pool = ModelPool()
    model_pool.auto_load_model(
        model_config.path,
        vram_config=vram_config,
        clear_parameters=False,
        state_dict=model_config.state_dict,
    )
    base_dit = model_pool.fetch_model("wan_video_dit")
    if base_dit is None:
        raise RuntimeError("DiffSynth did not load a Wan DiT")
    return base_dit
