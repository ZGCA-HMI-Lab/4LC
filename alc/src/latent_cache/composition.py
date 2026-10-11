from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from latent_cache.readout import target_inputs, target_valid_masks
from latent_cache.types import (
    CacheEntry,
    CompositionResult,
    CompositionStep,
    CompositionTrace,
)

if TYPE_CHECKING:
    from latent_cache.cache import LatentCache


def select_entries(count: int, max_entries: int) -> tuple[int, ...]:
    if (
        isinstance(max_entries, bool)
        or not isinstance(max_entries, int)
        or max_entries < 2
    ):
        raise ValueError("max_entries must be an integer >= 2")
    if count <= max_entries:
        return tuple(range(count))
    return tuple(
        index * (count - 1) // (max_entries - 1) for index in range(max_entries)
    )


def _coverage_rates(readouts, valid_masks) -> dict[str, float]:
    rates = {}
    for name, valid in valid_masks.items():
        denominator = int(valid.count_nonzero())
        if denominator == 0:
            raise ValueError(f"representation {name!r} has no valid targets")
        rates[name] = (
            int((readouts[name].coverage & valid).count_nonzero()) / denominator
        )
    return rates


def _synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compose_readouts(
    reader,
    items,
    valid_masks,
    *,
    device,
    order,
    max_entries,
    coverage_threshold,
):
    """Accumulate ordered readouts; readers may also reserve uncovered cells.

    Each item describes (index, tag, keyframe count). The reader returns readouts
    and optional per-representation masks that forbid later history filling.
    """
    if order not in ("earliest_first", "latest_first"):
        raise ValueError("order must be earliest_first or latest_first")
    if not 0.0 < float(coverage_threshold) <= 1.0:
        raise ValueError("coverage_threshold must be in (0,1]")
    names = tuple(valid_masks)
    selected = select_entries(len(items), max_entries)
    visit = selected if order == "earliest_first" else tuple(reversed(selected))
    offsets = []
    total = 0
    for _index, _tag, count in items:
        offsets.append(total)
        total += count
    protected = {name: torch.zeros_like(mask) for name, mask in valid_masks.items()}
    result, _ = reader(visit[0], protected)
    steps = []
    counts = dict.fromkeys(names, 0)
    stop_reason = "sampled_chunks_exhausted"
    for index in visit:
        before = _coverage_rates(result, valid_masks)
        fill = {
            name: valid_masks[name] & ~result[name].coverage & ~protected[name]
            if before[name] < coverage_threshold
            else torch.zeros_like(valid_masks[name])
            for name in names
        }
        _synchronize(device)
        started = time.perf_counter()
        delta, reserved = reader(index, fill)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        new_counts = {}
        for name in names:
            readout, output = delta[name], result[name]
            if reserved is not None:
                protected[name] |= reserved[name] & fill[name]
            new = readout.coverage & fill[name]
            output.features[new] = readout.features[new]
            output.coverage[new] = True
            for field in ("moving", "depth", "time_offset", "score"):
                getattr(output, field)[new] = getattr(readout, field)[new]
            output.source_keyframe[new] = readout.source_keyframe[new] + offsets[index]
            tokens_per_keyframe = output.features.shape[-3] * output.features.shape[-2]
            output.source_token[new] = (
                readout.source_token[new] + offsets[index] * tokens_per_keyframe
            )
            new_counts[name] = int(new.count_nonzero())
            counts[name] += readout.candidate_count
        after = _coverage_rates(result, valid_masks)
        entry_index, tag, count = items[index]
        steps.append(
            CompositionStep(
                entry_index,
                tag,
                tuple(range(offsets[index], offsets[index] + count)),
                before,
                after,
                new_counts,
                {name: delta[name].candidate_count for name in names},
                elapsed,
            )
        )
        del delta
        if all(rate >= coverage_threshold for rate in after.values()):
            stop_reason = "target_rate_reached"
            break
        if reserved is not None and not any(
            after[name] < coverage_threshold
            and bool(
                (valid_masks[name] & ~result[name].coverage & ~protected[name]).any()
            )
            for name in names
        ):
            stop_reason = "no_fillable_targets"
            break
    result = {
        name: replace(value, candidate_count=counts[name])
        for name, value in result.items()
    }
    trace = CompositionTrace(
        order,
        max_entries,
        len(items),
        selected,
        tuple(items[index][0] for index in selected),
        coverage_threshold,
        tuple(steps),
        stop_reason,
        _coverage_rates(result, valid_masks),
    )
    return CompositionResult(result, trace)


@torch.no_grad()
def compose(
    cache: LatentCache,
    entries: list[CacheEntry],
    times: Tensor,
    intrinsics: Tensor,
    c2w: Tensor,
    *,
    frame_indices: Tensor | None,
    valid: Tensor | None,
    order: str,
    max_entries: int,
    coverage_threshold: float,
) -> CompositionResult:
    if not entries:
        raise ValueError("composition needs at least one entry")
    if any(not isinstance(entry, CacheEntry) for entry in entries):
        raise TypeError("entries must be CacheEntry values")
    names = tuple(entries[0].features)
    specs = tuple(payload.representation for payload in entries[0].features.values())
    if any(
        tuple(entry.features) != names
        or tuple(payload.representation for payload in entry.features.values()) != specs
        for entry in entries
    ):
        raise ValueError("entries hold different representations")
    frame_indices, valid = target_inputs(
        entries[0], times, intrinsics, c2w, frame_indices, valid
    )
    valid_masks = target_valid_masks(entries[0], valid)

    def reader(index, fill):
        return cache.read(
            entries[index],
            times,
            intrinsics,
            c2w,
            frame_indices=frame_indices,
            valid=valid,
            fill=fill,
            static=True,
        ), None

    return _compose_readouts(
        reader,
        [(entry.index, entry.tag, entry.num_keyframes) for entry in entries],
        valid_masks,
        device=times.device,
        order=order,
        max_entries=max_entries,
        coverage_threshold=coverage_threshold,
    )
