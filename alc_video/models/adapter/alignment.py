from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F

from alc_video.models.video_model.temporal import (
    target_latent_shape,
    wan_readout_frame_count,
    wan_readout_group_size,
)


def readout_stride(target_indices: tuple[int, ...], video_frames: int) -> int:
    stride = (
        1 if len(target_indices) == 1 else int(target_indices[1] - target_indices[0])
    )
    if stride not in {1, 2, 4} or tuple(target_indices) != tuple(
        range(0, video_frames, stride)
    ):
        raise ValueError(
            f"readout target indices {target_indices} do not cover {video_frames} "
            "frames at stride 1, 2 or 4"
        )
    return stride


def validate_feature_shape(
    grid: tuple[int, int],
    channels: int,
) -> tuple[tuple[int, int], int]:
    grid = tuple(int(size) for size in grid)
    channels = int(channels)
    if len(grid) != 2 or any(size <= 0 for size in grid) or channels <= 0:
        raise ValueError("feature grid and channels must be positive")
    return grid, channels


def validate_feature_shapes(
    feature_shapes: Sequence[tuple[tuple[int, int], int]],
) -> tuple[tuple[tuple[int, int], int], ...]:
    if not isinstance(feature_shapes, (tuple, list)):
        raise TypeError("feature_shapes must be a sequence")
    if not feature_shapes:
        raise ValueError("feature_shapes must contain at least one representation")
    validated = []
    for index, shape in enumerate(feature_shapes):
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or not isinstance(shape[0], tuple)
        ):
            raise TypeError(f"feature_shapes[{index}] must be (grid, channels)")
        validated.append(validate_feature_shape(shape[0], shape[1]))
    return tuple(validated)


def validate_readout_features(
    features: torch.Tensor,
    target_shape: torch.Tensor | torch.Size | tuple[int, ...] | list[int],
    *,
    grid: tuple[int, int],
    channels: int,
    latent_channels: int,
    readout_stride: int = 1,
) -> tuple[int, int, int, int, int]:
    target = target_latent_shape(target_shape)
    batch, target_channels, latent_frames, _height, _width = target
    if target_channels != int(latent_channels):
        raise ValueError(
            f"target latent channels {target_channels} != expected {int(latent_channels)}"
        )
    grid, channels = validate_feature_shape(grid, channels)
    expected = (
        batch,
        wan_readout_frame_count(latent_frames, readout_stride),
        *grid,
        channels,
    )
    if tuple(int(size) for size in features.shape) != expected:
        raise ValueError(
            f"readout features must be {expected}, got {tuple(features.shape)}"
        )
    if not torch.is_floating_point(features):
        raise TypeError("readout features must be floating point")
    return target


def validate_readout_coverage(
    features: torch.Tensor,
    coverage: torch.Tensor,
    target_shape: torch.Tensor | torch.Size | tuple[int, ...] | list[int],
    *,
    grid: tuple[int, int],
    channels: int,
    latent_channels: int,
    readout_stride: int = 1,
) -> tuple[int, int, int, int, int]:
    target = validate_readout_features(
        features,
        target_shape,
        grid=grid,
        channels=channels,
        latent_channels=latent_channels,
        readout_stride=readout_stride,
    )
    expected = tuple(int(size) for size in features.shape[:-1])
    if tuple(int(size) for size in coverage.shape) != expected:
        raise ValueError(f"coverage must be {expected}, got {tuple(coverage.shape)}")
    if coverage.dtype != torch.bool:
        raise TypeError("coverage must be bool")
    if coverage.device != features.device:
        raise ValueError("readout features and coverage must be on the same device")
    return target


def resize_bcthw_grid(
    value: torch.Tensor,
    *,
    size: tuple[int, int],
    mode: Literal["bilinear", "nearest"] = "bilinear",
) -> torch.Tensor:
    if value.ndim != 5:
        raise ValueError(f"value must be [B,C,F,H,W], got {tuple(value.shape)}")
    height, width = (int(value.shape[-2]), int(value.shape[-1]))
    size = tuple(int(item) for item in size)
    if len(size) != 2 or any(item <= 0 for item in size):
        raise ValueError("resize dimensions must be positive")
    if (height, width) == size:
        return value
    batch, channels, frames = (int(item) for item in value.shape[:3])
    flat = value.permute(0, 2, 1, 3, 4).reshape(
        batch * frames,
        channels,
        height,
        width,
    )
    if mode == "bilinear":
        resized = F.interpolate(
            flat.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    else:
        resized = F.interpolate(
            flat.float(),
            size=size,
            mode="nearest-exact",
        )
    return (
        resized.to(dtype=value.dtype)
        .reshape(
            batch,
            frames,
            channels,
            *size,
        )
        .permute(0, 2, 1, 3, 4)
    )


def select_wan_causal_endpoints(
    value: torch.Tensor,
    *,
    latent_frames: int,
    dim: int,
    readout_stride: int = 1,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a tensor")
    if value.ndim == 0:
        raise ValueError("value must have at least one dimension")
    dim = int(dim)
    if dim < 0:
        dim += value.ndim
    if dim < 0 or dim >= value.ndim:
        raise ValueError(f"dim {dim} is invalid for a {value.ndim}D tensor")
    expected_frames = wan_readout_frame_count(latent_frames, readout_stride)
    frames = int(value.shape[dim])
    if frames != expected_frames:
        raise ValueError(
            f"Wan endpoint selection expects {expected_frames} frames, got {frames}"
        )
    endpoints = torch.arange(
        0,
        expected_frames,
        wan_readout_group_size(readout_stride),
        device=value.device,
        dtype=torch.int64,
    )
    if int(endpoints.numel()) != int(latent_frames):
        raise RuntimeError("Wan endpoint count does not match latent_frames")
    return value.index_select(dim, endpoints)


def aggregate_wan_masked_groups(
    value: torch.Tensor,
    coverage: torch.Tensor,
    *,
    latent_frames: int,
    readout_stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 5:
        raise ValueError(f"value must be [B,C,F,H,W], got {tuple(value.shape)}")
    expected_mask_shape = (
        int(value.shape[0]),
        1,
        int(value.shape[2]),
        int(value.shape[3]),
        int(value.shape[4]),
    )
    if tuple(int(size) for size in coverage.shape) != expected_mask_shape:
        raise ValueError(
            f"coverage must be {expected_mask_shape}, got {tuple(coverage.shape)}"
        )
    if coverage.device != value.device:
        raise ValueError("value and coverage must be on the same device")
    if not torch.is_floating_point(value) or not torch.is_floating_point(coverage):
        raise TypeError("value and coverage must be floating point")

    batch, channels, frames, height, width = (int(item) for item in value.shape)
    expected_frames = wan_readout_frame_count(latent_frames, readout_stride)
    if frames != expected_frames:
        raise ValueError(
            f"Wan group aggregation expects {expected_frames} frames, got {frames}"
        )

    value = value * coverage.to(dtype=value.dtype)
    first_value = value[:, :, :1]
    first_coverage = coverage[:, :, :1]
    if latent_frames == 1:
        return first_value, first_coverage

    group_size = wan_readout_group_size(readout_stride)
    tail_value = value[:, :, 1:].reshape(
        batch,
        channels,
        latent_frames - 1,
        group_size,
        height,
        width,
    )
    tail_coverage = coverage[:, :, 1:].reshape(
        batch,
        1,
        latent_frames - 1,
        group_size,
        height,
        width,
    )
    covered_count = tail_coverage.sum(dim=3)
    tail_value = tail_value.sum(dim=3) / covered_count.clamp_min(1)
    group_coverage = tail_coverage.mean(dim=3)
    return (
        torch.cat((first_value, tail_value), dim=2),
        torch.cat((first_coverage, group_coverage), dim=2),
    )


def align_wan_coverage(
    coverage: torch.Tensor,
    target_shape: torch.Tensor | torch.Size | tuple[int, ...] | list[int],
    *,
    grid: tuple[int, int],
    readout_stride: int = 1,
) -> torch.Tensor:
    batch, _channels, latent_frames, height, width = target_latent_shape(target_shape)
    grid = tuple(int(size) for size in grid)
    expected = (
        batch,
        wan_readout_frame_count(latent_frames, readout_stride),
        *grid,
    )
    if not isinstance(coverage, torch.Tensor) or tuple(coverage.shape) != expected:
        raise ValueError(
            f"coverage must be {expected}, got {getattr(coverage, 'shape', None)}"
        )
    if coverage.dtype != torch.bool:
        raise TypeError("coverage must be bool")
    if (
        isinstance(target_shape, torch.Tensor)
        and coverage.device != target_shape.device
    ):
        raise ValueError("coverage and target latents must share one device")
    aligned = resize_bcthw_grid(
        coverage[:, None].float(),
        size=(height, width),
        mode="nearest",
    )
    _values, aligned_coverage = aggregate_wan_masked_groups(
        aligned,
        aligned,
        latent_frames=latent_frames,
        readout_stride=readout_stride,
    )
    return aligned_coverage
