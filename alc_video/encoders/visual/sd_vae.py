from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from latent_cache import Representation

from alc_video.encoders.visual.base import (
    VisualEncoder,
)
from alc_video.encoders.visual.visualization import FeatureVisualization

SD_VAE_CHANNELS = 4
SD_VAE_DEFAULT_SCALING_FACTOR = 0.18215
SD_VAE_ENCODER = "sd_vae"
SD_VAE_GRID = (42, 70)
SD_VAE_IMAGE_SIZE = (336, 560)


SD_VAE_CONFIG_FILENAME = "config.json"
SD_VAE_SAFETENSORS_FILENAME = "diffusion_pytorch_model.safetensors"
SD_VAE_BIN_FILENAME = "diffusion_pytorch_model.bin"


def _select_sd_vae_weights(path: Path) -> Path:
    safetensors = path / SD_VAE_SAFETENSORS_FILENAME
    if safetensors.is_file():
        return safetensors
    binary = path / SD_VAE_BIN_FILENAME
    if binary.is_file():
        return binary
    raise FileNotFoundError(f"SD VAE weight is missing: {safetensors} or {binary}")


def resolve_sd_vae_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"SD VAE directory does not exist: {path}")
    config = path / SD_VAE_CONFIG_FILENAME
    if not config.is_file():
        raise FileNotFoundError(f"SD VAE config is missing: {config}")
    _select_sd_vae_weights(path)
    return path


def sd_vae_weight_path(value: str | Path) -> Path:
    return _select_sd_vae_weights(resolve_sd_vae_path(value))


def sd_vae_asset_files(value: str | Path) -> tuple[Path, Path]:
    path = resolve_sd_vae_path(value)
    return path / SD_VAE_CONFIG_FILENAME, _select_sd_vae_weights(path)


def load_sd_vae_model(
    autoencoder_kl: type[Any],
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    resolved = resolve_sd_vae_path(path)
    weights = _select_sd_vae_weights(resolved)
    model = autoencoder_kl.from_pretrained(
        str(resolved),
        local_files_only=True,
        torch_dtype=dtype,
        use_safetensors=weights.suffix == ".safetensors",
        device_map=torch.device(device),
        low_cpu_mem_usage=True,
        disable_mmap=True,
    )
    model.eval().requires_grad_(False)
    return model


def _read_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads((path / SD_VAE_CONFIG_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"SD VAE config is invalid: {path}") from error
    if not isinstance(value, dict):
        raise TypeError("SD VAE config must be an object")
    expected = {
        "_class_name": "AutoencoderKL",
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": SD_VAE_CHANNELS,
    }
    actual = {name: value.get(name) for name in expected}
    if actual != expected:
        raise ValueError(f"SD VAE config fields {actual} do not match {expected}")
    blocks = value.get("block_out_channels")
    if not isinstance(blocks, list) or len(blocks) != 4:
        raise ValueError(
            "SD VAE block_out_channels must describe four resolution levels"
        )
    scaling = value.get(
        "scaling_factor",
        SD_VAE_DEFAULT_SCALING_FACTOR,
    )
    if (
        isinstance(scaling, bool)
        or not isinstance(scaling, (int, float))
        or not math.isfinite(float(scaling))
        or float(scaling) <= 0.0
    ):
        raise ValueError("SD VAE scaling_factor must be finite and > 0")
    value["scaling_factor"] = float(scaling)
    return value


@dataclass(slots=True)
class _SDVAEVisualizer:
    encoder: SDVAEEncoder

    def render(
        self,
        readout_features: torch.Tensor,
        hit_mask: torch.Tensor,
    ) -> FeatureVisualization:
        expected_tail = (*SD_VAE_GRID, SD_VAE_CHANNELS)
        if (
            not isinstance(readout_features, torch.Tensor)
            or readout_features.ndim != 4
            or tuple(int(value) for value in readout_features.shape[1:])
            != expected_tail
        ):
            raise ValueError(
                "SD VAE query latents must be "
                f"[T,{SD_VAE_GRID[0]},{SD_VAE_GRID[1]},"
                f"{SD_VAE_CHANNELS}], got "
                f"{getattr(readout_features, 'shape', None)}"
            )
        expected_mask = tuple(int(value) for value in readout_features.shape[:-1])
        if (
            not isinstance(hit_mask, torch.Tensor)
            or tuple(int(value) for value in hit_mask.shape) != expected_mask
            or hit_mask.dtype != torch.bool
        ):
            raise ValueError(f"SD VAE hit mask must be bool {expected_mask}")
        mask = hit_mask.to(
            device=readout_features.device,
            dtype=torch.bool,
        ).unsqueeze(-1)
        masked = torch.where(mask, readout_features, torch.zeros_like(readout_features))
        frames = self.encoder.decode(masked)
        video = frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        return FeatureVisualization(
            "query target SD VAE decode",
            video,
            tuple(range(int(video.shape[2]))),
            "encoder_output",
        )


class SDVAEEncoder(VisualEncoder):
    path_kind = "directory"
    representation = Representation((42, 70), (8, 8), (7, 7), 4, "selected")

    encoder_type = SD_VAE_ENCODER

    def __init__(
        self,
        *,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 4,
    ) -> None:
        self.model_path = resolve_sd_vae_path(model_path)
        self.config = _read_config(self.model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self._model: torch.nn.Module | None = None
        self._on_device = False
        if self.batch_size <= 0:
            raise ValueError("SD VAE batch_size must be > 0")

    @classmethod
    def from_local_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        batch_size: int = 4,
    ) -> SDVAEEncoder:
        return cls(
            model_path=resolve_sd_vae_path(path),
            device=torch.device(device),
            dtype=dtype,
            batch_size=batch_size,
        )

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            from diffusers import AutoencoderKL

            self._model = load_sd_vae_model(
                AutoencoderKL,
                self.model_path,
                device=self.device,
                dtype=self.dtype,
            )
            self._on_device = True
        if not self._on_device:
            self._model.to(device=self.device, dtype=self.dtype)
            self._on_device = True
        return self._model

    def prepare(self) -> None:
        self._ensure_model()

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype in {
            torch.float16,
            torch.bfloat16,
        }:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    @staticmethod
    def _validate_frames(frames: torch.Tensor) -> None:
        expected = (3, *SD_VAE_IMAGE_SIZE)
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 4
            or tuple(int(value) for value in frames.shape[1:]) != expected
            or int(frames.shape[0]) <= 0
        ):
            raise ValueError(
                "SD VAE frames must be "
                f"[N,3,{SD_VAE_IMAGE_SIZE[0]},{SD_VAE_IMAGE_SIZE[1]}]"
            )
        if not torch.is_floating_point(frames):
            raise TypeError("SD VAE frames must be floating point RGB in [0,1]")
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("SD VAE frames must be finite RGB in [0,1]")

    @torch.inference_mode()
    def _encode(
        self,
        frames: torch.Tensor,
        *,
        output_device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        self._validate_frames(frames)
        output_device = torch.device(output_device)
        model = self._ensure_model()
        scaling = float(self.config["scaling_factor"])
        outputs: list[torch.Tensor] = []
        for start in range(0, int(frames.shape[0]), self.batch_size):
            images = frames[start : start + self.batch_size].to(
                device=self.device,
                dtype=self.dtype,
            )
            with self._autocast():
                encoded = model.encode(images.mul(2.0).sub(1.0)).latent_dist.mode()
            expected = (
                int(images.shape[0]),
                SD_VAE_CHANNELS,
                *SD_VAE_GRID,
            )
            if tuple(int(value) for value in encoded.shape) != expected:
                raise ValueError(
                    f"SD VAE encoded shape {tuple(encoded.shape)} != {expected}"
                )
            encoded = encoded.mul(scaling)
            if not bool(torch.isfinite(encoded).all()):
                raise FloatingPointError(
                    "SD VAE encoded latents contain non-finite values"
                )
            outputs.append(
                encoded.permute(0, 2, 3, 1)
                .to(device=output_device, dtype=torch.bfloat16)
                .contiguous()
            )
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        expected_tail = (*SD_VAE_GRID, SD_VAE_CHANNELS)
        if (
            not isinstance(latents, torch.Tensor)
            or latents.ndim != 4
            or tuple(int(value) for value in latents.shape[1:]) != expected_tail
            or int(latents.shape[0]) <= 0
        ):
            raise ValueError(
                "SD VAE latents must be "
                f"[N,{SD_VAE_GRID[0]},{SD_VAE_GRID[1]},"
                f"{SD_VAE_CHANNELS}]"
            )
        if not torch.is_floating_point(latents):
            raise TypeError("SD VAE latents must be floating point")
        if not bool(torch.isfinite(latents).all()):
            raise ValueError("SD VAE latents contain non-finite values")
        model = self._ensure_model()
        scaling = float(self.config["scaling_factor"])
        outputs: list[torch.Tensor] = []
        for start in range(0, int(latents.shape[0]), self.batch_size):
            values = (
                latents[start : start + self.batch_size]
                .permute(0, 3, 1, 2)
                .to(device=self.device, dtype=self.dtype)
            )
            with self._autocast():
                decoded = model.decode(values / scaling).sample
            expected = (
                int(values.shape[0]),
                3,
                *SD_VAE_IMAGE_SIZE,
            )
            if tuple(int(value) for value in decoded.shape) != expected:
                raise ValueError(
                    f"SD VAE decoded shape {tuple(decoded.shape)} != {expected}"
                )
            decoded = decoded.to(dtype=torch.float32).add(1.0).mul(0.5).clamp_(0.0, 1.0)
            if not bool(torch.isfinite(decoded).all()):
                raise FloatingPointError(
                    "SD VAE decoded frames contain non-finite values"
                )
            outputs.append(decoded.cpu())
        return torch.cat(outputs, dim=0)

    def _make_visualizer(
        self,
        source_latents: torch.Tensor,
    ) -> _SDVAEVisualizer:
        expected_tail = (*SD_VAE_GRID, SD_VAE_CHANNELS)
        if (
            not isinstance(source_latents, torch.Tensor)
            or source_latents.ndim != 4
            or int(source_latents.shape[0]) <= 0
            or tuple(int(value) for value in source_latents.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "SD VAE source latents must be "
                f"[K,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]}] "
                f"with K >= 1"
            )
        return _SDVAEVisualizer(self)

    def offload(self) -> None:
        if self._model is not None and self._on_device:
            self._model.to("cpu")
            self._on_device = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


__all__ = [
    "SD_VAE_BIN_FILENAME",
    "SD_VAE_CONFIG_FILENAME",
    "SD_VAE_SAFETENSORS_FILENAME",
    "SDVAEEncoder",
    "load_sd_vae_model",
    "resolve_sd_vae_path",
    "sd_vae_asset_files",
    "sd_vae_weight_path",
]
