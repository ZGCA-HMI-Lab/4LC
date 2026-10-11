from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import torch
from latent_cache import Representation

from alc_video.encoders.visual.base import (
    VisualEncoder,
)
from alc_video.encoders.visual.visualization import PCAFeatureVisualizer

TARGET_FRAMES = 81
VGGT_CHANNELS = 2048
VGGT_CONFIG_FILENAME = "config.json"
VGGT_DEFAULT_LEVELS = (1,)
VGGT_ENCODER = "vggt"
VGGT_GRID = (24, 40)
VGGT_IMAGE_SIZE = (336, 560)
VGGT_LEVEL_BLOCKS = (4, 11, 17, 23)
VGGT_MODEL = "facebook/VGGT-1B"
VGGT_PATCH_SIZE = 14
VGGT_PRETRAINED_IMAGE_SIZE = 518
VGGT_SPECIAL_TOKENS = 5
VGGT_WEIGHTS_FILENAME = "model.safetensors"


def resolve_vggt_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    required = (VGGT_CONFIG_FILENAME, VGGT_WEIGHTS_FILENAME)
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise FileNotFoundError(f"VGGT-1B path must contain {required}: {path}")
    try:
        config = json.loads((path / VGGT_CONFIG_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"VGGT-1B config.json is invalid: {path}") from error
    actual = (
        config.get("img_size"),
        config.get("patch_size"),
        config.get("embed_dim"),
    )
    expected = (VGGT_PRETRAINED_IMAGE_SIZE, VGGT_PATCH_SIZE, 1024)
    if actual != expected:
        raise ValueError(
            f"{path}: expected {VGGT_MODEL} image/patch/embed {expected}, got {actual}"
        )
    return path


def vggt_asset_files(path: str | Path) -> tuple[Path, Path]:
    resolved = resolve_vggt_path(path)
    return (
        resolved / VGGT_CONFIG_FILENAME,
        resolved / VGGT_WEIGHTS_FILENAME,
    )


def _load_vggt_aggregator(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained(
        str(path),
        local_files_only=True,
        strict=True,
        map_location="cpu",
    )
    aggregator = getattr(model, "aggregator", None)
    if not isinstance(aggregator, torch.nn.Module):
        raise TypeError(f"{VGGT_MODEL} must expose its aggregator")
    return aggregator.to(device=device, dtype=dtype).eval().requires_grad_(False)


class VGGTEncoder(VisualEncoder):
    path_kind = "directory"
    representation = Representation((24, 40), (14, 14), (7, 7), 2048, "selected")

    @classmethod
    def configured_representation(cls, levels=None):
        selected = validate_vggt_levels(levels)
        return Representation(
            (24, 40), (14, 14), (7, 7), 2048 * len(selected), "selected"
        )

    encoder_type = VGGT_ENCODER

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_path: Path,
        levels: tuple[int, ...] = VGGT_DEFAULT_LEVELS,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.levels = validate_vggt_levels(levels)
        self.representation = self.configured_representation(self.levels)
        self.layer_indices = tuple(VGGT_LEVEL_BLOCKS[level] for level in self.levels)
        self._model: torch.nn.Module | None = None
        self._on_device = False

    @classmethod
    def from_config(cls, config, *, device, dtype):
        cls.configured_representation(config.levels)
        return cls.from_local_pretrained(
            config.path, device=device, dtype=dtype, levels=config.levels
        )

    @classmethod
    def from_local_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        levels: tuple[int, ...] | list[int] | None = None,
    ) -> VGGTEncoder:
        return cls(
            device=torch.device(device),
            dtype=dtype,
            model_path=resolve_vggt_path(path),
            levels=validate_vggt_levels(levels),
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._ensure_model()

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = _load_vggt_aggregator(
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
            torch.bfloat16,
            torch.float16,
        }:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    @torch.no_grad()
    def _encode(
        self,
        frames: torch.Tensor,
        *,
        output_device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        expected = (3, *VGGT_IMAGE_SIZE)
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 4
            or tuple(int(size) for size in frames.shape[1:]) != expected
        ):
            raise ValueError(
                f"VGGT frames must be [V,3,{VGGT_IMAGE_SIZE[0]},"
                f"{VGGT_IMAGE_SIZE[1]}], got {getattr(frames, 'shape', None)}"
            )
        views = int(frames.shape[0])
        if not 1 <= views <= TARGET_FRAMES:
            raise ValueError(
                f"VGGT frames must contain between 1 and {TARGET_FRAMES} views, "
                f"got {views}"
            )
        if not torch.is_floating_point(frames):
            raise TypeError("VGGT frames must be floating point RGB in [0,1]")
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("VGGT frames must be finite RGB in [0,1]")

        pixels = frames.to(device=self.device, dtype=self.dtype).unsqueeze(0)
        with self._autocast():
            layers, patch_start = self._ensure_model()(pixels)
        if not isinstance(layers, list) or not layers:
            raise TypeError(f"{VGGT_MODEL} aggregator must return a layer list")
        if patch_start != VGGT_SPECIAL_TOKENS:
            raise ValueError(
                f"VGGT patch start must be {VGGT_SPECIAL_TOKENS}, got {patch_start}"
            )
        token_count = VGGT_GRID[0] * VGGT_GRID[1]
        expected_layer_shape = (
            1,
            views,
            token_count + VGGT_SPECIAL_TOKENS,
            VGGT_CHANNELS,
        )
        selected = []
        for level, layer_index in zip(
            self.levels,
            self.layer_indices,
            strict=True,
        ):
            if layer_index >= len(layers):
                raise ValueError(
                    f"VGGT level {level} requires layer {layer_index}, "
                    f"but aggregator returned {len(layers)} entries"
                )
            encoded = layers[layer_index]
            if not isinstance(encoded, torch.Tensor):
                raise TypeError(f"VGGT level {level} layer {layer_index} is missing")
            if tuple(int(size) for size in encoded.shape) != expected_layer_shape:
                raise ValueError(
                    f"VGGT level {level} tokens must be {expected_layer_shape}, "
                    f"got {tuple(encoded.shape)}"
                )
            patches = encoded[0, :, VGGT_SPECIAL_TOKENS:]
            if not bool(torch.isfinite(patches).all()):
                raise FloatingPointError(
                    f"VGGT level {level} returned non-finite patch tokens"
                )
            selected.append(patches.reshape(-1, *VGGT_GRID, VGGT_CHANNELS))
        return (
            torch.cat(selected, dim=-1)
            .to(device=torch.device(output_device), dtype=torch.bfloat16)
            .contiguous()
        )

    def _make_visualizer(
        self,
        source_latents: torch.Tensor,
    ) -> PCAFeatureVisualizer:
        expected = (*VGGT_GRID, VGGT_CHANNELS * len(self.levels))
        if (
            not isinstance(source_latents, torch.Tensor)
            or source_latents.ndim != 4
            or int(source_latents.shape[0]) <= 0
            or tuple(int(size) for size in source_latents.shape[1:]) != expected
        ):
            raise ValueError(
                f"VGGT source latents must be [K,{expected[0]},{expected[1]},"
                f"{expected[2]}] with K >= 1, "
                f"got {getattr(source_latents, 'shape', None)}"
            )
        return PCAFeatureVisualizer.fit(
            source_latents,
            label="query target VGGT PCA",
        )

    def offload(self) -> None:
        if self._model is not None and self._on_device:
            self._model.to("cpu")
            self._on_device = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


__all__ = [
    "VGGT_CHANNELS",
    "VGGT_CONFIG_FILENAME",
    "VGGT_ENCODER",
    "VGGT_GRID",
    "VGGT_IMAGE_SIZE",
    "VGGT_MODEL",
    "VGGT_PATCH_SIZE",
    "VGGT_WEIGHTS_FILENAME",
    "VGGTEncoder",
    "resolve_vggt_path",
    "vggt_asset_files",
]


def validate_vggt_levels(value):
    if value is None:
        return (1,)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("levels must be a nonempty sequence")
    levels = tuple(int(level) for level in value)
    if tuple(sorted(set(levels))) != levels or levels[0] < 0 or levels[-1] >= 4:
        raise ValueError("levels must be unique, increasing values in [0,3]")
    return levels
