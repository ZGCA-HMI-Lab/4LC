from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from latent_cache.composition import _compose_readouts
from latent_cache.editing.data import CacheData, as_data
from latent_cache.editing.operations import merge
from latent_cache.editing.readout import _read
from latent_cache.readout import target_inputs, target_valid_masks
from latent_cache.types import CacheEntry, CompositionResult, LatentCacheConfig


@torch.no_grad()
def compose(
    items: Sequence[CacheEntry | CacheData],
    times: Tensor,
    intrinsics: Tensor,
    c2w: Tensor,
    *,
    config: LatentCacheConfig,
    fps: float | Tensor,
    frame_indices: Tensor | None = None,
    valid: Tensor | None = None,
    order: Literal["earliest_first", "latest_first"] = "earliest_first",
    max_entries: int = 8,
    coverage_threshold: float = 0.95,
    static: bool = True,
    source_frame_indices=None,
) -> CompositionResult:
    """Fill history gaps while preserving ordered editing exclusions.

    Each input is one history item, even if it contains several source blocks.
    Trace entry indices refer to input positions. Output source indices address
    the concatenation of each item's ``CacheData.sources`` table in input order.
    ``source_frame_indices`` supplies one read-compatible value per input item.
    """
    items = list(items)
    data = [as_data(item) for item in items]
    if not data:
        raise ValueError("composition needs at least one history item")
    # Validate common source layouts without flattening history boundaries.
    merge(data)
    first = data[0].blocks[0].entry
    frame_indices, valid = target_inputs(
        first, times, intrinsics, c2w, frame_indices, valid
    )
    masks = target_valid_masks(first, valid)
    if source_frame_indices is None:
        source_frames = (None,) * len(data)
    else:
        if isinstance(source_frame_indices, Tensor):
            raise TypeError("source_frame_indices needs one value per history item")
        source_frames = tuple(source_frame_indices)
        if len(source_frames) != len(data):
            raise ValueError("source_frame_indices must match the history item count")
    metadata = [
        (
            index,
            item.tag if isinstance(item, CacheEntry) else None,
            sum(source.entry.num_keyframes for source in edited.sources),
        )
        for index, (item, edited) in enumerate(zip(items, data, strict=True))
    ]

    def reader(index, fill):
        protected = {}
        readouts = _read(
            data[index],
            times,
            intrinsics,
            c2w,
            config=config,
            fps=fps,
            frame_indices=frame_indices,
            valid=valid,
            fill=fill,
            static=static,
            source_frame_indices=source_frames[index],
            protected=protected,
        )
        return readouts, protected

    return _compose_readouts(
        reader,
        metadata,
        masks,
        device=times.device,
        order=order,
        max_entries=max_entries,
        coverage_threshold=coverage_threshold,
    )
