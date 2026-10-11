from __future__ import annotations

from collections.abc import Sequence

import torch


def build_source_drop_mask(
    dropped_source_ids: Sequence[int],
    *,
    batch_size: int,
    source_count: int,
    device: torch.device,
) -> torch.Tensor:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count < 2
    ):
        raise ValueError("source_count must be an integer >= 2")
    if not isinstance(dropped_source_ids, Sequence) or isinstance(
        dropped_source_ids,
        (str, bytes),
    ):
        raise TypeError("dropped_source_ids must be a sequence of integers")
    ids = tuple(dropped_source_ids)
    if not ids:
        raise ValueError("source_drop must remove at least one source")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise TypeError("dropped_source_ids must contain integers")
    if len(set(ids)) != len(ids):
        raise ValueError("dropped_source_ids must be unique")
    if any(value < 0 or value >= source_count for value in ids):
        raise ValueError("dropped_source_ids contains an out-of-range source")
    if len(ids) >= source_count:
        raise ValueError("source_drop must retain at least one source")
    result = torch.zeros(
        (batch_size, source_count),
        device=device,
        dtype=torch.bool,
    )
    result[:, torch.tensor(ids, device=device, dtype=torch.int64)] = True
    return result


def validate_source_drop_mask(
    dropped_sources: torch.Tensor,
    *,
    batch_size: int,
    source_count: int,
) -> None:
    if (
        not isinstance(dropped_sources, torch.Tensor)
        or dropped_sources.ndim != 2
        or dropped_sources.dtype != torch.bool
    ):
        raise TypeError("dropped_sources must be a bool [B,K] tensor")
    actual = tuple(int(size) for size in dropped_sources.shape)
    expected = (batch_size, source_count)
    if actual != expected:
        raise ValueError(f"dropped_sources must be {expected}, got {actual}")
    counts = dropped_sources.sum(dim=1)
    if bool((counts < 1).any()):
        raise ValueError("source_drop must remove at least one source per sample")
    if bool((counts >= source_count).any()):
        raise ValueError("source_drop must retain at least one source per sample")
    if batch_size > 1 and not bool((dropped_sources == dropped_sources[:1]).all()):
        raise ValueError(
            "source_drop mask must select one shared keyframe set per batch"
        )


__all__ = [
    "build_source_drop_mask",
    "validate_source_drop_mask",
]
