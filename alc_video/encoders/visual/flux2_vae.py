from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from latent_cache import Representation

from alc_video.encoders.visual.base import (
    CONTEXT_MAX,
    CONTEXT_MIN,
    IMAGE_SIZE,
    FeatureVisualization,
    VisualEncoder,
    is_keyframe_count,
)

FLUX2_VAE_ENCODER = "flux2_vae"
FLUX2_VAE_MODEL = "black-forest-labs/FLUX.2-dev"
FLUX2_VAE_CONFIG_FILENAME = "config.json"
FLUX2_VAE_WEIGHTS_FILENAME = "diffusion_pytorch_model.safetensors"
FLUX2_VAE_IMAGE_SIZE = IMAGE_SIZE
FLUX2_VAE_DOWNSAMPLE_FACTOR = 8
FLUX2_VAE_GRID = (42, 70)
FLUX2_VAE_CHANNELS = 32
FLUX2_VAE_POSTERIOR = "mean"


def resolve_flux2_vae_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"FLUX.2 VAE directory does not exist: {path}")
    for filename in (FLUX2_VAE_CONFIG_FILENAME, FLUX2_VAE_WEIGHTS_FILENAME):
        asset = path / filename
        if not asset.is_file():
            raise FileNotFoundError(f"FLUX.2 VAE asset is missing: {asset}")
    return path


def flux2_vae_asset_files(value: str | Path) -> tuple[Path, Path]:
    path = resolve_flux2_vae_path(value)
    return (
        path / FLUX2_VAE_CONFIG_FILENAME,
        path / FLUX2_VAE_WEIGHTS_FILENAME,
    )


def _read_config(path: Path) -> dict[str, Any]:
    config_path = path / FLUX2_VAE_CONFIG_FILENAME
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"FLUX.2 VAE config is invalid: {config_path}") from error
    if not isinstance(value, dict):
        raise TypeError("FLUX.2 VAE config must be an object")
    expected = {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": FLUX2_VAE_CHANNELS,
        "layers_per_block": 2,
        "norm_num_groups": 32,
    }
    actual = {name: value.get(name) for name in expected}
    if actual != expected:
        raise ValueError(f"FLUX.2 VAE config fields {actual} do not match {expected}")
    patch_size = value.get("patch_size")
    if patch_size not in (2, [2, 2]):
        raise ValueError("FLUX.2 VAE patch_size must be 2 or [2,2]")
    blocks = value.get("block_out_channels")
    if (
        not isinstance(blocks, list)
        or len(blocks) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in blocks
        )
    ):
        raise ValueError("FLUX.2 VAE block_out_channels must contain four integers")
    return value


_MODEL_CONFIG_FIELDS = (
    "in_channels",
    "out_channels",
    "down_block_types",
    "up_block_types",
    "block_out_channels",
    "layers_per_block",
    "act_fn",
    "latent_channels",
    "norm_num_groups",
    "sample_size",
    "force_upcast",
    "use_quant_conv",
    "use_post_quant_conv",
    "mid_block_add_attention",
    "batch_norm_eps",
    "batch_norm_momentum",
    "patch_size",
)


def _load_model(
    path: Path,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from diffsynth.core.loader import load_model
    from diffsynth.models.flux2_vae import Flux2VAE

    model_config = {
        name: config[name] for name in _MODEL_CONFIG_FIELDS if name in config
    }
    if model_config.get("patch_size") == 2:
        model_config["patch_size"] = (2, 2)
    model = load_model(
        Flux2VAE,
        str(path / FLUX2_VAE_WEIGHTS_FILENAME),
        config=model_config,
        torch_dtype=dtype,
        device="cpu",
    )
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


@dataclass(slots=True)
class _Flux2VAEVisualizer:
    encoder: Flux2VAEEncoder

    def render(
        self,
        readout_features: torch.Tensor,
        coverage: torch.Tensor,
    ) -> FeatureVisualization:
        expected_tail = (*FLUX2_VAE_GRID, FLUX2_VAE_CHANNELS)
        if (
            not isinstance(readout_features, torch.Tensor)
            or readout_features.ndim != 4
            or tuple(int(value) for value in readout_features.shape[1:])
            != expected_tail
        ):
            raise ValueError(
                f"FLUX.2 VAE readout features must be [T,{expected_tail[0]},"
                f"{expected_tail[1]},{expected_tail[2]}]"
            )
        expected_mask = tuple(int(value) for value in readout_features.shape[:-1])
        if (
            not isinstance(coverage, torch.Tensor)
            or tuple(int(value) for value in coverage.shape) != expected_mask
            or coverage.dtype != torch.bool
        ):
            raise ValueError(f"FLUX.2 VAE coverage must be bool {expected_mask}")
        masked = readout_features.masked_fill(~coverage[..., None], 0)
        frames = self.encoder.decode(masked)
        return FeatureVisualization(
            "readout FLUX.2 VAE decode",
            frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
            tuple(range(int(frames.shape[0]))),
            "encoder_output",
        )


class Flux2VAEEncoder(VisualEncoder):
    encoder_type = FLUX2_VAE_ENCODER

    representation = Representation(
        grid=FLUX2_VAE_GRID,
        stride=(FLUX2_VAE_DOWNSAMPLE_FACTOR,) * 2,
        offset=(FLUX2_VAE_DOWNSAMPLE_FACTOR - 1,) * 2,
        dim=FLUX2_VAE_CHANNELS,
        frame_layout="selected",
    )
    path_kind = "directory"

    def __init__(
        self,
        *,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> None:
        self.model_path = resolve_flux2_vae_path(model_path)
        self.config = _read_config(self.model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self._model: torch.nn.Module | None = None
        self._on_device = False
        if self.batch_size <= 0:
            raise ValueError("FLUX.2 VAE batch_size must be > 0")

    @classmethod
    def from_local_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        batch_size: int = 4,
    ) -> Flux2VAEEncoder:
        return cls(
            model_path=resolve_flux2_vae_path(path),
            device=torch.device(device),
            dtype=dtype,
            batch_size=batch_size,
        )

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = _load_model(
                self.model_path,
                config=self.config,
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
        expected = (3, *FLUX2_VAE_IMAGE_SIZE)
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 4
            or tuple(int(value) for value in frames.shape[1:]) != expected
            or int(frames.shape[0]) <= 0
        ):
            raise ValueError(
                "FLUX.2 VAE frames must be "
                f"[N,3,{FLUX2_VAE_IMAGE_SIZE[0]},{FLUX2_VAE_IMAGE_SIZE[1]}]"
            )
        if not torch.is_floating_point(frames):
            raise TypeError("FLUX.2 VAE frames must be floating point RGB in [0,1]")
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("FLUX.2 VAE frames must be finite RGB in [0,1]")

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
        outputs: list[torch.Tensor] = []
        for start in range(0, int(frames.shape[0]), self.batch_size):
            images = frames[start : start + self.batch_size].to(
                device=self.device,
                dtype=self.dtype,
            )
            with self._autocast():
                moments = model.encoder(images.mul(2.0).sub(1.0))
                if model.quant_conv is not None:
                    moments = model.quant_conv(moments)
                encoded = moments[:, :FLUX2_VAE_CHANNELS]
            expected = (
                int(images.shape[0]),
                FLUX2_VAE_CHANNELS,
                *FLUX2_VAE_GRID,
            )
            if tuple(int(value) for value in encoded.shape) != expected:
                raise ValueError(
                    f"FLUX.2 VAE encoded shape {tuple(encoded.shape)} != {expected}"
                )
            if not bool(torch.isfinite(encoded).all()):
                raise FloatingPointError("FLUX.2 VAE returned non-finite latents")
            outputs.append(
                encoded.permute(0, 2, 3, 1)
                .to(device=output_device, dtype=torch.bfloat16)
                .contiguous()
            )
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        expected_tail = (*FLUX2_VAE_GRID, FLUX2_VAE_CHANNELS)
        if (
            not isinstance(latents, torch.Tensor)
            or latents.ndim != 4
            or tuple(int(value) for value in latents.shape[1:]) != expected_tail
            or int(latents.shape[0]) <= 0
        ):
            raise ValueError(
                f"FLUX.2 VAE latents must be [N,{expected_tail[0]},"
                f"{expected_tail[1]},{expected_tail[2]}]"
            )
        if not torch.is_floating_point(latents) or not bool(
            torch.isfinite(latents).all()
        ):
            raise ValueError("FLUX.2 VAE latents must be finite floating point")
        model = self._ensure_model()
        outputs: list[torch.Tensor] = []
        for start in range(0, int(latents.shape[0]), self.batch_size):
            values = (
                latents[start : start + self.batch_size]
                .permute(0, 3, 1, 2)
                .to(device=self.device, dtype=self.dtype)
            )
            with self._autocast():
                if model.post_quant_conv is not None:
                    values = model.post_quant_conv(values)
                decoded = model.decoder(values)
            expected = (int(values.shape[0]), 3, *FLUX2_VAE_IMAGE_SIZE)
            if tuple(int(value) for value in decoded.shape) != expected:
                raise ValueError(
                    f"FLUX.2 VAE decoded shape {tuple(decoded.shape)} != {expected}"
                )
            decoded = decoded.float().add(1.0).mul(0.5).clamp_(0.0, 1.0)
            if not bool(torch.isfinite(decoded).all()):
                raise FloatingPointError("FLUX.2 VAE returned non-finite frames")
            outputs.append(decoded.cpu())
        return torch.cat(outputs, dim=0)

    def _make_visualizer(
        self,
        features: torch.Tensor,
    ) -> _Flux2VAEVisualizer:
        expected_tail = (*FLUX2_VAE_GRID, FLUX2_VAE_CHANNELS)
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 4
            or not is_keyframe_count(int(features.shape[0]))
            or tuple(int(value) for value in features.shape[1:]) != expected_tail
        ):
            raise ValueError(
                f"FLUX.2 VAE features must be [K,{expected_tail[0]},"
                f"{expected_tail[1]},{expected_tail[2]}] with K=1 or "
                f"{CONTEXT_MIN} <= K <= {CONTEXT_MAX}"
            )
        return _Flux2VAEVisualizer(self)

    def offload(self) -> None:
        if self._model is not None and self._on_device:
            self._model.to("cpu")
            self._on_device = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


__all__ = [
    "FLUX2_VAE_ENCODER",
    "Flux2VAEEncoder",
    "flux2_vae_asset_files",
    "resolve_flux2_vae_path",
]
