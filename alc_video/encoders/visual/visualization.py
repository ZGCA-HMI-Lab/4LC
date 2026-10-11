from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class FeatureVisualization:
    label: str
    video: torch.Tensor
    timeline: tuple[int, ...]
    domain: Literal["encoder_output", "video_frame"]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("feature visualization label must be non-empty")
        if (
            not isinstance(self.video, torch.Tensor)
            or self.video.ndim != 5
            or tuple(int(value) for value in self.video.shape[:2]) != (1, 3)
        ):
            raise ValueError("feature visualization video must be [1,3,T,H,W]")
        if not torch.is_floating_point(self.video):
            raise TypeError("feature visualization video must be floating point")
        if (
            not bool(torch.isfinite(self.video).all())
            or float(self.video.min()) < 0.0
            or float(self.video.max()) > 1.0
        ):
            raise ValueError("feature visualization video must be finite RGB in [0,1]")
        frames = int(self.video.shape[2])
        timeline = self.timeline
        if (
            not isinstance(timeline, tuple)
            or len(timeline) != frames
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in timeline
            )
            or tuple(sorted(set(timeline))) != timeline
            or timeline[0] != 0
        ):
            raise ValueError(
                "feature visualization timeline must be a strictly increasing "
                "nonnegative integer tuple beginning at 0 with one entry per frame"
            )
        if self.domain not in {"encoder_output", "video_frame"}:
            raise ValueError("feature visualization domain is invalid")

    def with_timeline(self, timeline: tuple[int, ...]) -> FeatureVisualization:
        return FeatureVisualization(self.label, self.video, timeline, self.domain)


@runtime_checkable
class FeatureVisualizer(Protocol):
    def render(
        self,
        readout_features: torch.Tensor,
        coverage: torch.Tensor,
    ) -> FeatureVisualization: ...


@dataclass(frozen=True, slots=True)
class VideoResolutionVisualizer:
    visualizer: FeatureVisualizer
    image_size: tuple[int, int]

    def render(
        self,
        readout_features: torch.Tensor,
        coverage: torch.Tensor,
    ) -> FeatureVisualization:
        rendered = self.visualizer.render(readout_features, coverage)
        if tuple(int(value) for value in rendered.video.shape[-2:]) == self.image_size:
            return rendered
        frames = rendered.video[0].permute(1, 0, 2, 3)
        resized = F.interpolate(
            frames,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return FeatureVisualization(
            rendered.label,
            resized.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
            rendered.timeline,
            rendered.domain,
        )


@dataclass(frozen=True, slots=True)
class PCAFeatureVisualizer:
    label: str
    grid: tuple[int, int]
    channels: int
    mean: torch.Tensor
    basis: torch.Tensor
    lower: torch.Tensor
    ranges: torch.Tensor

    @classmethod
    def fit(
        cls,
        features: torch.Tensor,
        *,
        label: str,
    ) -> PCAFeatureVisualizer:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a tensor")
        if features.ndim != 4:
            raise ValueError(
                f"PCA features must be [V,H,W,C], got {tuple(features.shape)}"
            )
        views, height, width, channels = (int(value) for value in features.shape)
        if views <= 0 or height <= 0 or width <= 0 or channels < 3:
            raise ValueError("PCA feature shape is invalid")
        values = features.detach().to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("PCA features contain non-finite values")
        flattened = values.reshape(-1, channels)
        if int(flattened.shape[0]) < 3:
            raise ValueError("PCA requires at least three source tokens")
        mean = flattened.mean(dim=0, keepdim=True)
        centered = flattened - mean
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            _u, _s, basis = torch.pca_lowrank(
                centered,
                q=3,
                center=False,
            )
        component_indices = basis.abs().argmax(dim=0)
        component_values = basis[
            component_indices,
            torch.arange(3, dtype=torch.long),
        ]
        signs = torch.where(component_values < 0, -1.0, 1.0)
        basis = basis * signs
        projected = centered @ basis
        lower = projected.amin(dim=0)
        ranges = projected.amax(dim=0) - lower
        return cls(
            label=label,
            grid=(height, width),
            channels=channels,
            mean=mean.contiguous(),
            basis=basis.contiguous(),
            lower=lower.contiguous(),
            ranges=ranges.contiguous(),
        )

    def render(
        self,
        readout_features: torch.Tensor,
        coverage: torch.Tensor,
    ) -> FeatureVisualization:
        if not isinstance(readout_features, torch.Tensor):
            raise TypeError("readout features must be a tensor")
        expected_tail = (*self.grid, self.channels)
        if (
            readout_features.ndim != 4
            or tuple(int(value) for value in readout_features.shape[1:])
            != expected_tail
        ):
            raise ValueError(
                f"PCA readout features must be [T,{self.grid[0]},"
                f"{self.grid[1]},{self.channels}], "
                f"got {tuple(readout_features.shape)}"
            )
        expected_mask = tuple(int(value) for value in readout_features.shape[:-1])
        if (
            not isinstance(coverage, torch.Tensor)
            or tuple(int(value) for value in coverage.shape) != expected_mask
            or coverage.dtype != torch.bool
        ):
            raise ValueError(
                f"PCA coverage must be bool {expected_mask}, "
                f"got {getattr(coverage, 'shape', None)}"
            )
        values = readout_features.detach().to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        )
        mask = coverage.detach().to(device="cpu")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("PCA readout features contain non-finite values")
        flattened = values.reshape(-1, self.channels)
        projected = (flattened - self.mean) @ self.basis
        rgb = torch.zeros_like(projected)
        varying = self.ranges > torch.finfo(projected.dtype).eps
        rgb[:, varying] = (projected[:, varying] - self.lower[varying]) / self.ranges[
            varying
        ]
        rgb.clamp_(0.0, 1.0)
        rgb[~mask.reshape(-1)] = 0.0
        frames = int(readout_features.shape[0])
        video = (
            rgb.reshape(frames, *self.grid, 3)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .contiguous()
        )
        return FeatureVisualization(
            self.label,
            video,
            tuple(range(int(video.shape[2]))),
            "encoder_output",
        )
