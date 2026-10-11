from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
from latent_cache import Representation

from alc_video.encoders.visual.visualization import (
    FeatureVisualization,
    FeatureVisualizer,
    PCAFeatureVisualizer,
    VideoResolutionVisualizer,
)

if TYPE_CHECKING:
    from alc_video.encoders.config import VisualEncoderConfig


CONTEXT_MIN = 11
CONTEXT_MAX = 21
IMAGE_SIZE = (336, 560)


def is_keyframe_count(value: int) -> bool:
    return value == 1 or CONTEXT_MIN <= value <= CONTEXT_MAX


def _output_count(representation: Representation, frame_count: int) -> int:
    if representation.frame_layout == "selected":
        return frame_count
    if representation.frame_layout == "previous_current":
        if frame_count % 2:
            raise ValueError("previous_current input must contain 2K frames")
        return frame_count // 2
    stride = representation.temporal_stride
    if (frame_count - 1) % stride:
        raise ValueError("causal_hold input must contain 4m+1 frames")
    return 1 + (frame_count - 1) // stride


class VisualEncoder(ABC):
    encoder_type: ClassVar[str]
    representation: ClassVar[Representation]
    path_kind: ClassVar[Literal["directory", "file"]]
    image_size: ClassVar[tuple[int, int]] = IMAGE_SIZE

    @classmethod
    def configured_representation(cls, levels=None) -> Representation:
        if levels is not None:
            raise ValueError(f"encoder {cls.encoder_type!r} does not use layer levels")
        return cls.representation

    @classmethod
    def from_config(cls, config: VisualEncoderConfig, *, device, dtype):
        cls.configured_representation(config.levels)
        return cls.from_local_pretrained(
            config.path, device=device, dtype=dtype, batch_size=config.batch_size
        )

    @classmethod
    @abstractmethod
    def from_local_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> VisualEncoder: ...

    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def offload(self) -> None:
        pass

    @abstractmethod
    def _encode(
        self,
        frames: torch.Tensor,
        *,
        output_device: str | torch.device,
    ) -> torch.Tensor:
        pass

    def _make_visualizer(self, features: torch.Tensor) -> FeatureVisualizer:
        raise NotImplementedError(f"encoder {self.encoder_type!r} has no visualizer")

    @torch.no_grad()
    def encode(
        self,
        frames: torch.Tensor,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 5
            or int(frames.shape[-1]) != 3
            or any(int(size) <= 0 for size in frames.shape)
        ):
            raise ValueError("source RGB must be [B,F,H,W,3]")
        if not torch.is_floating_point(frames):
            raise TypeError("source RGB must be floating point")
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("source RGB must be finite and in [0,1]")
        batch, frame_count, height, width, _channels = (
            int(value) for value in frames.shape
        )
        if batch != 1:
            raise ValueError("visual encoder input batch size must be 1")
        if (height, width) != self.image_size:
            raise ValueError(
                "source RGB image size differs from encoder spatial contract"
            )
        selected_device = torch.device(device)
        representation = self.representation
        output_frames = _output_count(representation, frame_count)
        encoder_input = frames[0].permute(0, 3, 1, 2).contiguous()
        encoded = self._encode(
            encoder_input,
            output_device=device,
        )
        expected = (output_frames, *representation.grid, representation.dim)
        if (
            not isinstance(encoded, torch.Tensor)
            or tuple(int(value) for value in encoded.shape) != expected
        ):
            raise ValueError(
                f"visual encoder output must be {expected}, "
                f"got {getattr(encoded, 'shape', None)}"
            )
        if not torch.is_floating_point(encoded) or not bool(
            torch.isfinite(encoded).all()
        ):
            raise ValueError("visual encoder output must be finite floating point")
        if encoded.device != selected_device:
            raise ValueError(
                f"visual encoder returned {encoded.device}, expected {selected_device}"
            )
        return encoded.reshape(1, *expected).clone().contiguous()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        raise ValueError(
            f"visual encoder {self.encoder_type!r} does not support decode"
        )

    def make_visualizer(self, features: torch.Tensor) -> FeatureVisualizer:
        visualizer = self._make_visualizer(features)
        if not isinstance(visualizer, FeatureVisualizer):
            raise TypeError("visual encoder visualizer must implement render")
        return VideoResolutionVisualizer(visualizer, self.image_size)


__all__ = [
    "IMAGE_SIZE",
    "FeatureVisualization",
    "FeatureVisualizer",
    "PCAFeatureVisualizer",
    "VisualEncoder",
    "is_keyframe_count",
]
