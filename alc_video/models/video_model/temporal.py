from __future__ import annotations

import torch


def target_latent_shape(
    value: torch.Tensor | torch.Size | tuple[int, ...] | list[int],
) -> tuple[int, int, int, int, int]:
    raw = value.shape if isinstance(value, torch.Tensor) else value
    shape = tuple(int(size) for size in raw)
    if len(shape) != 5:
        raise ValueError(f"target latent shape must be [B,C,F,H,W], got {shape}")
    if any(size <= 0 for size in shape):
        raise ValueError(f"target latent dimensions must be positive, got {shape}")
    return shape


def wan_video_frame_count(latent_frames: int) -> int:
    latent_frames = int(latent_frames)
    if latent_frames <= 0:
        raise ValueError("latent_frames must be > 0")
    return 4 * (latent_frames - 1) + 1


def wan_vae_latent_frame_count(video_frames: int) -> int:
    video_frames = int(video_frames)
    if video_frames <= 0:
        raise ValueError("video_frames must be > 0")
    if (video_frames - 1) % 4 != 0:
        raise ValueError(
            f"Wan video frame count must satisfy T % 4 == 1, got {video_frames}"
        )
    return 1 + (video_frames - 1) // 4


def wan_readout_frame_count(latent_frames: int, readout_stride: int) -> int:
    latent_frames = int(latent_frames)
    readout_stride = int(readout_stride)
    if latent_frames <= 0:
        raise ValueError("latent_frames must be > 0")
    if readout_stride not in {1, 2, 4}:
        raise ValueError("readout_stride must be 1, 2, or 4")
    return 1 + 4 * (latent_frames - 1) // readout_stride


def wan_readout_group_size(readout_stride: int) -> int:
    readout_stride = int(readout_stride)
    if readout_stride not in {1, 2, 4}:
        raise ValueError("readout_stride must be 1, 2, or 4")
    return 4 // readout_stride
