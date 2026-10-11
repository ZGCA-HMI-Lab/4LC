from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from latent_cache import Representation

from alc_video.encoders.visual.base import (
    CONTEXT_MAX,
    CONTEXT_MIN,
    VisualEncoder,
    is_keyframe_count,
)
from alc_video.encoders.visual.visualization import FeatureVisualization

DEPTH_ANYTHING_V2_CHANNELS = 1
DEPTH_ANYTHING_V2_ENCODER = "depth_anything_v2"
DEPTH_ANYTHING_V2_GRID = (24, 40)
DEPTH_ANYTHING_V2_IMAGE_SIZE = (336, 560)
DEPTH_ANYTHING_V2_MEAN = (0.485, 0.456, 0.406)
DEPTH_ANYTHING_V2_MODEL = "depth-anything/Depth-Anything-V2-Large"
DEPTH_ANYTHING_V2_PATCH_SIZE = 14
DEPTH_ANYTHING_V2_STD = (0.229, 0.224, 0.225)
DEPTH_ANYTHING_V2_WEIGHTS_FILENAME = "depth_anything_v2_vitl.pth"


def resolve_depth_anything_v2_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Depth Anything V2 Large weights must be the local "
            f"{DEPTH_ANYTHING_V2_WEIGHTS_FILENAME}: {path}"
        )
    if path.name != DEPTH_ANYTHING_V2_WEIGHTS_FILENAME:
        raise ValueError(
            "Depth Anything V2 Large weights filename must be "
            f"{DEPTH_ANYTHING_V2_WEIGHTS_FILENAME}"
        )
    return path


def _checkpoint_state(
    path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location=device, weights_only=True)
    if (
        not isinstance(value, Mapping)
        or not value
        or any(not isinstance(name, str) for name in value)
        or any(not isinstance(tensor, torch.Tensor) for tensor in value.values())
    ):
        raise ValueError(
            f"{path}: Depth Anything V2 checkpoint must be a non-empty state dict"
        )
    return dict(value)


def _load_depth_anything_v2_model(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from importlib import import_module

    DepthAnythingV2 = import_module(
        "third_party.Depth-Anything-V2.depth_anything_v2.dpt"
    ).DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )
    model.load_state_dict(_checkpoint_state(path, device), strict=True)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


class DepthAnythingV2Encoder(VisualEncoder):
    path_kind = "file"
    representation = Representation((24, 40), (14, 14), (7, 7), 1, "selected")

    encoder_type = DEPTH_ANYTHING_V2_ENCODER

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        model_path: Path,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.model_path = model_path
        self._model: torch.nn.Module | None = None
        self._on_device = False
        if self.batch_size <= 0:
            raise ValueError("Depth Anything V2 batch_size must be > 0")

    @classmethod
    def from_local_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        batch_size: int = 4,
    ) -> DepthAnythingV2Encoder:
        return cls(
            device=torch.device(device),
            dtype=dtype,
            batch_size=batch_size,
            model_path=resolve_depth_anything_v2_path(path),
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._ensure_model()

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = _load_depth_anything_v2_model(
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
        expected = (3, *DEPTH_ANYTHING_V2_IMAGE_SIZE)
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 4
            or tuple(int(size) for size in frames.shape[1:]) != expected
        ):
            raise ValueError(
                "Depth Anything V2 frames must be "
                f"[V,3,{DEPTH_ANYTHING_V2_IMAGE_SIZE[0]},"
                f"{DEPTH_ANYTHING_V2_IMAGE_SIZE[1]}], got "
                f"{getattr(frames, 'shape', None)}"
            )
        if int(frames.shape[0]) <= 0:
            raise ValueError("Depth Anything V2 frames must contain at least one view")
        if not torch.is_floating_point(frames):
            raise TypeError(
                "Depth Anything V2 frames must be floating point RGB in [0,1]"
            )
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("Depth Anything V2 frames must be finite RGB in [0,1]")

        mean = torch.tensor(
            DEPTH_ANYTHING_V2_MEAN,
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, 3, 1, 1)
        std = torch.tensor(
            DEPTH_ANYTHING_V2_STD,
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, 3, 1, 1)
        output_device = torch.device(output_device)
        model = self._ensure_model()
        outputs = []
        for start in range(0, int(frames.shape[0]), self.batch_size):
            pixels = frames[start : start + self.batch_size].to(
                device=self.device,
                dtype=self.dtype,
            )
            pixels = (pixels - mean) / std
            with self._autocast():
                depth = model(pixels)
                encoded = F.adaptive_avg_pool2d(
                    depth[:, None],
                    DEPTH_ANYTHING_V2_GRID,
                )
            expected_shape = (
                int(pixels.shape[0]),
                DEPTH_ANYTHING_V2_CHANNELS,
                *DEPTH_ANYTHING_V2_GRID,
            )
            if tuple(int(size) for size in encoded.shape) != expected_shape:
                raise ValueError(
                    "Depth Anything V2 depth latent must be "
                    f"{expected_shape}, got {tuple(encoded.shape)}"
                )
            if not bool(torch.isfinite(encoded).all()):
                raise FloatingPointError(
                    "Depth Anything V2 returned non-finite depth values"
                )
            outputs.append(
                encoded.permute(0, 2, 3, 1)
                .to(device=output_device, dtype=torch.bfloat16)
                .contiguous()
            )
        return torch.cat(outputs, dim=0)

    def _make_visualizer(
        self,
        source_latents: torch.Tensor,
    ) -> _DepthAnythingVisualizer:
        expected = (*DEPTH_ANYTHING_V2_GRID, DEPTH_ANYTHING_V2_CHANNELS)
        if (
            not isinstance(source_latents, torch.Tensor)
            or source_latents.ndim != 4
            or not is_keyframe_count(int(source_latents.shape[0]))
            or tuple(int(size) for size in source_latents.shape[1:]) != expected
        ):
            raise ValueError(
                "Depth Anything V2 source latents must be "
                f"[K,{expected[0]},{expected[1]},{expected[2]}] with "
                f"K=1 or {CONTEXT_MIN} <= K <= {CONTEXT_MAX}, got "
                f"{getattr(source_latents, 'shape', None)}"
            )
        values = source_latents.float()
        return _DepthAnythingVisualizer(
            lower=values.amin().cpu(),
            upper=values.amax().cpu(),
        )

    def offload(self) -> None:
        if self._model is not None and self._on_device:
            self._model.to("cpu")
            self._on_device = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


@dataclass(frozen=True, slots=True)
class _DepthAnythingVisualizer:
    lower: torch.Tensor
    upper: torch.Tensor

    def render(
        self,
        readout_features: torch.Tensor,
        hit_mask: torch.Tensor,
    ) -> FeatureVisualization:
        expected_tail = (*DEPTH_ANYTHING_V2_GRID, DEPTH_ANYTHING_V2_CHANNELS)
        if (
            not isinstance(readout_features, torch.Tensor)
            or readout_features.ndim != 4
            or tuple(int(value) for value in readout_features.shape[1:])
            != expected_tail
        ):
            raise ValueError(
                f"Depth Anything V2 query latents must be [T,{expected_tail[0]},"
                f"{expected_tail[1]},{expected_tail[2]}]"
            )
        expected_mask = tuple(int(value) for value in readout_features.shape[:-1])
        if (
            not isinstance(hit_mask, torch.Tensor)
            or tuple(int(value) for value in hit_mask.shape) != expected_mask
            or hit_mask.dtype != torch.bool
        ):
            raise ValueError(f"Depth Anything V2 hit mask must be bool {expected_mask}")
        values = readout_features.detach().float().cpu()[..., 0]
        span = (self.upper - self.lower).clamp_min(torch.finfo(torch.float32).eps)
        normalized = ((values - self.lower) / span).clamp(0.0, 1.0)
        normalized.masked_fill_(~hit_mask.detach().cpu(), 0.0)
        video = normalized.unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0)
        return FeatureVisualization(
            "query target Depth Anything V2",
            video,
            tuple(range(int(video.shape[2]))),
            "encoder_output",
        )


__all__ = [
    "DEPTH_ANYTHING_V2_CHANNELS",
    "DEPTH_ANYTHING_V2_ENCODER",
    "DEPTH_ANYTHING_V2_GRID",
    "DEPTH_ANYTHING_V2_IMAGE_SIZE",
    "DEPTH_ANYTHING_V2_MODEL",
    "DEPTH_ANYTHING_V2_PATCH_SIZE",
    "DEPTH_ANYTHING_V2_WEIGHTS_FILENAME",
    "DepthAnythingV2Encoder",
    "resolve_depth_anything_v2_path",
]
