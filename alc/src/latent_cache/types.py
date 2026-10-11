from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Literal, Protocol, runtime_checkable

from torch import Tensor

from latent_cache.geometry.spatial import SpatialGridSpec
from latent_cache.propagation.motion import MotionSelection

FrameLayout = Literal["selected", "previous_current", "causal_hold"]
_TEMPORAL_STRIDE = {"selected": 1, "previous_current": 2, "causal_hold": 4}


def _pair(value: tuple[int, int], *, name: str, minimum: int) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < minimum
            for item in value
        )
    ):
        raise ValueError(f"{name} must be a pair of integers >= {minimum}")


@dataclass(frozen=True)
class Representation:
    grid: tuple[int, int]
    stride: tuple[int, int]
    offset: tuple[int, int]
    dim: int
    frame_layout: FrameLayout

    def __post_init__(self) -> None:
        _pair(self.grid, name="grid", minimum=1)
        _pair(self.stride, name="stride", minimum=1)
        _pair(self.offset, name="offset", minimum=0)
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError("dim must be a positive integer")
        if self.frame_layout not in _TEMPORAL_STRIDE:
            raise ValueError(f"unsupported frame_layout {self.frame_layout!r}")

    @property
    def temporal_stride(self) -> int:
        return _TEMPORAL_STRIDE[self.frame_layout]

    def spatial_grid(self, image_size: tuple[int, int]) -> SpatialGridSpec:
        return SpatialGridSpec(
            image_size, "aligned", self.grid, self.stride, self.offset
        )

    def target_indices(self, num_targets: int) -> tuple[int, ...]:
        stride = self.temporal_stride
        if num_targets <= 0 or (num_targets - 1) % stride:
            raise ValueError(
                f"{num_targets} target frames do not fit temporal stride {stride}"
            )
        return tuple(range(0, num_targets, stride))

    def native_layout(
        self, frame_indices: tuple[int, ...], num_frames: int
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        if self.frame_layout == "selected":
            return frame_indices, tuple((index,) for index in frame_indices)
        if self.frame_layout == "previous_current":
            previous = (frame_indices[0], *frame_indices[:-1])
            return frame_indices, tuple(zip(previous, frame_indices, strict=True))
        if frame_indices == (0,):
            return (0,), ((0,),)
        endpoints = self.target_indices(num_frames)
        groups = ((0,),) + tuple(
            tuple(range(endpoint - 3, endpoint + 1)) for endpoint in endpoints[1:]
        )
        return endpoints, groups

    def keyframe_slots(self, frame_indices: tuple[int, ...]) -> tuple[int, ...]:
        if self.frame_layout in ("selected", "previous_current"):
            return tuple(range(len(frame_indices)))
        slots = tuple((index + 3) // 4 for index in frame_indices)
        if tuple(sorted(set(slots))) != slots:
            raise ValueError("keyframes map to repeated causal_hold outputs")
        return slots


@runtime_checkable
class VisualEncoder(Protocol):
    representation: Representation

    def encode(self, frames: Tensor, *, device) -> Tensor: ...


@dataclass(frozen=True)
class LatentCacheConfig:
    image_size: tuple[int, int] = (336, 560)
    motion_threshold: float = 0.1
    motion_mask_dilation: int = 4
    temporal_motion_propagation: bool = True
    temporal_motion_depth_tolerance: float = 0.1
    same_depth_abs_eps: float = 1.0e-3
    same_depth_rel_eps: float = 1.0e-3
    target_chunk_size: int = 81
    native_resolution: bool = True
    max_projection_window_frames: int | None = None

    def __post_init__(self) -> None:
        _pair(self.image_size, name="image_size", minimum=1)
        for name in ("target_chunk_size", "max_projection_window_frames"):
            value = getattr(self, name)
            if value is None and name == "max_projection_window_frames":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.motion_mask_dilation, bool)
            or not isinstance(self.motion_mask_dilation, int)
            or self.motion_mask_dilation < 0
        ):
            raise ValueError("motion_mask_dilation must be a nonnegative integer")
        for name in ("native_resolution", "temporal_motion_propagation"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in (
            "motion_threshold",
            "temporal_motion_depth_tolerance",
            "same_depth_abs_eps",
            "same_depth_rel_eps",
        ):
            value = float(getattr(self, name))
            minimum_ok = value > 0 if name.startswith("same_depth") else value >= 0
            if not minimum_ok or value == float("inf"):
                raise ValueError(
                    f"{name} must be finite and {'positive' if name.startswith('same_depth') else 'nonnegative'}"
                )


@dataclass(frozen=True)
class RepresentationFeatures:
    representation: Representation
    native_features: Tensor
    native_positions: tuple[int, ...]
    native_frame_groups: tuple[tuple[int, ...], ...]
    features: Tensor
    valid: Tensor
    keyframe_slots: Tensor
    native_endpoints: Tensor

    def to(self, device) -> RepresentationFeatures:
        return replace(
            self,
            **{
                item.name: getattr(self, item.name).to(device)
                for item in fields(self)
                if isinstance(getattr(self, item.name), Tensor)
            },
        )


@dataclass(frozen=True)
class CacheEntry:
    index: int
    tag: str | None
    num_keyframes: int
    positions: Tensor
    depth: Tensor
    valid: Tensor
    times: Tensor
    frame_indices: Tensor
    encoding_frame_indices: Tensor
    timeline_times: Tensor
    fps: Tensor
    intrinsics: Tensor
    c2w: Tensor
    velocity: tuple[Tensor, Tensor] | None
    dropped: Tensor | None
    motion: MotionSelection
    features: Mapping[str, RepresentationFeatures]

    @property
    def moving(self) -> Tensor:
        return self.motion.motion_mask

    @property
    def resolved(self) -> Tensor:
        return self.motion.anchor_mask | self.motion.motion_mask

    def to(self, device) -> CacheEntry:
        values = {
            item.name: getattr(self, item.name).to(device)
            for item in fields(self)
            if isinstance(getattr(self, item.name), Tensor)
        }
        return replace(
            self,
            **values,
            velocity=None
            if self.velocity is None
            else tuple(value.to(device) for value in self.velocity),
            motion=MotionSelection(
                **{
                    item.name: getattr(self.motion, item.name).to(device)
                    for item in fields(self.motion)
                }
            ),
            features={name: value.to(device) for name, value in self.features.items()},
        )


@dataclass(frozen=True)
class Readout:
    target_indices: tuple[int, ...]
    features: Tensor
    coverage: Tensor
    moving: Tensor
    depth: Tensor
    source_keyframe: Tensor
    source_token: Tensor
    time_offset: Tensor
    score: Tensor
    candidate_count: int
    target_frame_indices: Tensor
    times: Tensor
    fps: Tensor
    intrinsics: Tensor
    c2w: Tensor
    image_size: tuple[int, int]


@dataclass(frozen=True)
class CompositionStep:
    entry_index: int
    tag: str | None
    keyframe_indices: tuple[int, ...]
    coverage_before: Mapping[str, float]
    coverage_after: Mapping[str, float]
    new_hit_counts: Mapping[str, int]
    candidate_counts: Mapping[str, int]
    elapsed_seconds: float


@dataclass(frozen=True)
class CompositionTrace:
    order: str
    max_entries: int
    available_count: int
    selected_indices: tuple[int, ...]
    selected_entry_indices: tuple[int, ...]
    coverage_threshold: float
    steps: tuple[CompositionStep, ...]
    stop_reason: str
    coverage_rates: Mapping[str, float]


@dataclass(frozen=True)
class CompositionResult:
    readouts: Mapping[str, Readout]
    trace: CompositionTrace
