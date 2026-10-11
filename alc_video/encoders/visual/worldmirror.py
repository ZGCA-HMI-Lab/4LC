from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from latent_cache import Representation
from omegaconf import OmegaConf

from alc_video.encoders.geometry.config import worldmirror_runtime_config
from alc_video.encoders.visual.base import (
    VisualEncoder,
)
from alc_video.encoders.visual.visualization import PCAFeatureVisualizer

WORLDMIRROR_CHANNELS = 2048
WORLDMIRROR_DEFAULT_LEVELS = (1,)
WORLDMIRROR_ENCODER = "worldmirror"
WORLDMIRROR_GRID = (24, 40)
WORLDMIRROR_IMAGE_SIZE = (336, 560)
WORLDMIRROR_MODEL = "WorldMirror"
WORLDMIRROR_RENDER_DIM = 2048
WORLDMIRROR_RENDER_LAYERS = (4, 11, 17, 23)


def resolve_worldmirror_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WorldMirror checkpoint does not exist: {path}")
    return path


def _load_worldmirror_token_encoder(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    layer_indices: tuple[int, ...],
) -> torch.nn.Module:
    config, _checkpoint = worldmirror_runtime_config(path, torch.device("cpu"))
    reconstructor = OmegaConf.to_container(config.reconstructor, resolve=True)
    if not isinstance(reconstructor, dict):
        raise TypeError("WorldMirror reconstructor config must be a mapping")
    from worldmirror.glue.loader import load_token_encoder

    encoder = load_token_encoder(
        reconstructor,
        torch.device("cpu"),
        layer_indices=layer_indices,
        batch_size=batch_size,
    )
    return encoder.to(device=device, dtype=dtype).eval()


class WorldMirrorFeatureEncoder(VisualEncoder):
    path_kind = "file"
    representation = Representation((24, 40), (14, 14), (7, 7), 2048, "selected")

    @classmethod
    def configured_representation(cls, levels=None):
        selected = validate_worldmirror_levels(levels)
        return Representation(
            (24, 40), (14, 14), (7, 7), 2048 * len(selected), "selected"
        )

    encoder_type = WORLDMIRROR_ENCODER

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_path: Path,
        batch_size: int = 4,
        levels: tuple[int, ...] = WORLDMIRROR_DEFAULT_LEVELS,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("WorldMirror encoder batch_size must be positive")
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.batch_size = int(batch_size)
        self.levels = validate_worldmirror_levels(levels)
        self.representation = self.configured_representation(self.levels)
        self.layer_indices = tuple(
            WORLDMIRROR_RENDER_LAYERS[level] for level in self.levels
        )
        self.channels = WORLDMIRROR_CHANNELS * len(self.levels)
        self._model: torch.nn.Module | None = None
        self._on_device = False

    @classmethod
    def from_config(cls, config, *, device, dtype):
        cls.configured_representation(config.levels)
        return cls.from_local_pretrained(
            config.path,
            device=device,
            dtype=dtype,
            levels=config.levels,
            batch_size=config.batch_size,
        )

    @classmethod
    def from_local_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        batch_size: int = 4,
        levels: tuple[int, ...] | list[int] | None = None,
    ) -> WorldMirrorFeatureEncoder:
        return cls(
            device=torch.device(device),
            dtype=dtype,
            model_path=resolve_worldmirror_path(path),
            batch_size=batch_size,
            levels=validate_worldmirror_levels(levels),
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._ensure_model()

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = _load_worldmirror_token_encoder(
                self.model_path,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
                layer_indices=self.layer_indices,
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
        expected = (3, *WORLDMIRROR_IMAGE_SIZE)
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 4
            or tuple(int(size) for size in frames.shape[1:]) != expected
        ):
            raise ValueError(
                f"WorldMirror frames must be [V,3,{WORLDMIRROR_IMAGE_SIZE[0]},"
                f"{WORLDMIRROR_IMAGE_SIZE[1]}], got {getattr(frames, 'shape', None)}"
            )
        views = int(frames.shape[0])
        if views <= 0:
            raise ValueError(f"WorldMirror frames require V >= 1, got {views}")
        if not torch.is_floating_point(frames):
            raise TypeError("WorldMirror frames must be floating point RGB in [0,1]")
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("WorldMirror frames must be finite RGB in [0,1]")

        pixels = frames.to(device=self.device, dtype=self.dtype).unsqueeze(0)
        model = self._ensure_model()
        encode_tokens = getattr(model, "encode_tokens", None)
        if not callable(encode_tokens):
            raise TypeError(f"{WORLDMIRROR_MODEL} token encoder lacks encode_tokens")
        with self._autocast():
            tokens = encode_tokens(
                pixels,
                num_layers=len(self.layer_indices),
                token_hw=WORLDMIRROR_GRID,
                token_dim=WORLDMIRROR_RENDER_DIM,
            )
        expected_shape = (
            1,
            views,
            len(self.layer_indices),
            *WORLDMIRROR_GRID,
            WORLDMIRROR_RENDER_DIM,
        )
        if (
            not isinstance(tokens, torch.Tensor)
            or tuple(tokens.shape) != expected_shape
        ):
            raise ValueError(
                f"WorldMirror tokens must be {expected_shape}, "
                f"got {getattr(tokens, 'shape', None)}"
            )
        if not bool(torch.isfinite(tokens).all()):
            raise FloatingPointError("WorldMirror returned non-finite patch tokens")
        return (
            tokens.permute(0, 1, 3, 4, 2, 5)
            .reshape(views, *WORLDMIRROR_GRID, self.channels)
            .to(device=torch.device(output_device), dtype=torch.bfloat16)
            .contiguous()
        )

    def _make_visualizer(
        self,
        source_latents: torch.Tensor,
    ) -> PCAFeatureVisualizer:
        expected = (*WORLDMIRROR_GRID, self.channels)
        if (
            not isinstance(source_latents, torch.Tensor)
            or source_latents.ndim != 4
            or int(source_latents.shape[0]) <= 0
            or tuple(int(size) for size in source_latents.shape[1:]) != expected
        ):
            raise ValueError(
                f"WorldMirror source latents must be [K,{expected[0]},"
                f"{expected[1]},{expected[2]}] with K >= 1, got {getattr(source_latents, 'shape', None)}"
            )
        return PCAFeatureVisualizer.fit(
            source_latents,
            label="query target WorldMirror PCA",
        )

    def offload(self) -> None:
        if self._model is not None and self._on_device:
            self._model.to("cpu")
            self._on_device = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


def validate_worldmirror_levels(value):
    if value is None:
        return (1,)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("levels must be a nonempty sequence")
    levels = tuple(int(level) for level in value)
    if tuple(sorted(set(levels))) != levels or levels[0] < 0 or levels[-1] >= 4:
        raise ValueError("levels must be unique, increasing values in [0,3]")
    return levels
