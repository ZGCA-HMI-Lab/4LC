from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

import torch


class SampleIdentity(TypedDict):
    dataset: str
    sample_id: str
    sampling_seed: int


class RawBatch(SampleIdentity):
    text: str
    transport: str
    fps: torch.Tensor
    frame_indices: torch.Tensor
    source_positions: torch.Tensor
    target_positions: torch.Tensor
    source_video: torch.Tensor
    source_path: str


class MotionFields(TypedDict, total=False):
    velocity_fwd: torch.Tensor
    velocity_bwd: torch.Tensor


class EncodedBatch(SampleIdentity, MotionFields):
    transport: str
    fps: torch.Tensor
    source_frame_indices: torch.Tensor
    source_positions: torch.Tensor
    source_times: torch.Tensor
    source_camera_intr: torch.Tensor
    source_camera_c2w: torch.Tensor
    source_intr: torch.Tensor
    source_c2w: torch.Tensor
    source_depth: torch.Tensor
    source_valid: torch.Tensor
    source_rgb: torch.Tensor
    target_frame_indices: torch.Tensor
    target_positions: torch.Tensor
    observation_times: torch.Tensor
    target_times: torch.Tensor
    target_intr: torch.Tensor
    target_c2w: torch.Tensor
    target_latents: torch.Tensor
    context: torch.Tensor


TRANSPORTS = frozenset({"identity", "velocity"})
TARGET_FRAMES = 81
CONTEXT_MIN = 11
CONTEXT_MAX = 21
SAMPLE_BATCH_SIZE = 1
SOURCE_CAMERA_FRAMES = TARGET_FRAMES
WAN_LATENT_SHAPE = (16, 21, 42, 70)


def require_sample_batch_size(batch_size: int) -> None:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size != SAMPLE_BATCH_SIZE
    ):
        raise ValueError(
            f"4LC requires per-rank sample batch size 1, got B={batch_size}"
        )


__all__ = [
    "CONTEXT_MAX",
    "CONTEXT_MIN",
    "SAMPLE_BATCH_SIZE",
    "SOURCE_CAMERA_FRAMES",
    "TARGET_FRAMES",
    "TRANSPORTS",
    "WAN_LATENT_SHAPE",
    "require_sample_batch_size",
]


def collate_training_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    require_sample_batch_size(len(samples))

    def batch_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.unsqueeze(0)
        if isinstance(value, Mapping):
            return {name: batch_value(item) for name, item in value.items()}
        return value

    return {name: batch_value(value) for name, value in samples[0].items()}
