from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields

import torch
from torch import Tensor

from latent_cache.construction import check_tensor
from latent_cache.geometry.projection import apply_plan, project
from latent_cache.propagation.motion import (
    MotionSelection,
    PropagationResult,
    infer_target_time_directions,
    propagate_classified_anchors,
)
from latent_cache.propagation.transport import (
    IdentityTransportBackend,
    VelocityTransportBackend,
)
from latent_cache.types import CacheEntry, LatentCacheConfig, Readout


def target_inputs(
    entry: CacheEntry,
    times: Tensor,
    intrinsics: Tensor,
    c2w: Tensor,
    frame_indices: Tensor | None,
    valid: Tensor | None,
) -> tuple[Tensor, Tensor]:
    batch = entry.times.shape[0]
    if (
        not isinstance(times, Tensor)
        or times.ndim != 2
        or times.shape[0] != batch
        or times.shape[1] <= 0
    ):
        raise ValueError("times must be nonempty [B,N]")
    count = times.shape[1]
    device = entry.positions.device
    check_tensor(times, (batch, count), "times", device=device)
    check_tensor(intrinsics, (batch, count, 3, 3), "intrinsics", device=device)
    check_tensor(c2w, (batch, count, 4, 4), "c2w", device=device)
    if frame_indices is None:
        frame_indices = torch.arange(count, device=device, dtype=torch.int64)[
            None
        ].expand(batch, -1)
    check_tensor(
        frame_indices, (batch, count), "frame_indices", device=device, dtype=torch.int64
    )
    if count > 1 and not bool((frame_indices[:, 1:] > frame_indices[:, :-1]).all()):
        raise ValueError("target frame_indices must be strictly increasing")
    if valid is None:
        valid = torch.ones((batch, count), device=device, dtype=torch.bool)
    check_tensor(valid, (batch, count), "valid", device=device, dtype=torch.bool)
    return frame_indices, valid


def target_valid_masks(entry: CacheEntry, valid: Tensor) -> dict[str, Tensor]:
    result = {}
    for name, payload in entry.features.items():
        indices = torch.tensor(
            payload.representation.target_indices(valid.shape[1]), device=valid.device
        )
        result[name] = (
            valid.index_select(1, indices)[..., None, None]
            .expand(valid.shape[0], len(indices), *payload.representation.grid)
            .clone()
        )
    return result


def _prepare_fills(entry, valid, fill):
    valid_masks = target_valid_masks(entry, valid)
    if fill is not None and set(fill) - set(entry.features):
        raise ValueError("fill names a representation the entry does not hold")
    fills = {}
    for name, target_valid in valid_masks.items():
        if fill is None or name not in fill:
            fills[name] = target_valid
        else:
            mask = fill[name]
            check_tensor(
                mask,
                tuple(target_valid.shape),
                f"fill[{name!r}]",
                device=valid.device,
                dtype=torch.bool,
            )
            if bool((mask & ~target_valid).any()):
                raise ValueError("fill exceeds valid target positions")
            fills[name] = mask
    return fills


def _representation_groups(entry, image_size, target_count, *, split_roles=False):
    groups = {}
    for name, payload in entry.features.items():
        representation = payload.representation
        key = (
            representation.target_indices(target_count),
            representation.spatial_grid(image_size),
            name if split_roles else None,
        )
        groups.setdefault(key, []).append(name)
    return [(indices, spec, names) for (indices, spec, _role), names in groups.items()]


def _readout_metadata(indices, times, intrinsics, c2w, frame_indices, fps, image_size):
    selected = torch.tensor(indices, device=times.device, dtype=torch.int64)
    return dict(
        target_indices=indices,
        target_frame_indices=frame_indices.index_select(1, selected),
        times=times.index_select(1, selected),
        fps=fps,
        intrinsics=intrinsics.index_select(1, selected),
        c2w=c2w.index_select(1, selected),
        image_size=image_size,
    )


def _prepare_geometry(entry, config, names, spec, *, static, mask=None):
    positions, valid = entry.positions, entry.valid
    classification, velocity = entry.motion, entry.velocity
    if mask is not None:
        valid = valid & mask
        classification = MotionSelection(
            **{
                item.name: getattr(classification, item.name) & valid
                if getattr(classification, item.name).dtype == torch.bool
                else getattr(classification, item.name)
                for item in fields(classification)
            }
        )
    if static:
        valid = valid & ~classification.motion_mask
        unavailable = torch.zeros_like(valid)
        classification = MotionSelection(
            torch.zeros_like(entry.depth),
            valid,
            unavailable,
            unavailable,
            unavailable,
            unavailable,
        )
        velocity = None
    if config.native_resolution:
        positions = spec.sample(positions, name="positions")
        valid = spec.sample(valid, name="valid")
        classification = MotionSelection(
            **{
                item.name: spec.sample(
                    getattr(classification, item.name), name=item.name
                )
                for item in fields(classification)
            }
        )
        if velocity is not None:
            velocity = tuple(spec.sample(value, name="velocity") for value in velocity)
        if any(
            not torch.equal(entry.features[names[0]].valid, entry.features[name].valid)
            for name in names[1:]
        ):
            raise ValueError(
                "representations sharing a native spatial grid must share feature_valid"
            )
    return positions, valid, classification, velocity


def _empty(entry: CacheEntry, name: str, count: int) -> dict[str, Tensor]:
    payload = entry.features[name]
    shape = (entry.times.shape[0], count, *payload.representation.grid)
    device = payload.features.device
    return {
        "features": torch.zeros(
            (*shape, payload.representation.dim),
            device=device,
            dtype=payload.features.dtype,
        ),
        "coverage": torch.zeros(shape, device=device, dtype=torch.bool),
        "moving": torch.zeros(shape, device=device, dtype=torch.bool),
        "depth": torch.zeros(shape, device=device, dtype=torch.float32),
        "source_keyframe": torch.full(shape, -1, device=device, dtype=torch.int64),
        "source_token": torch.full(shape, -1, device=device, dtype=torch.int64),
        "time_offset": torch.zeros(shape, device=device, dtype=torch.float32),
        "score": torch.zeros(shape, device=device, dtype=torch.float32),
    }


def _masked_values(projected, plan, fill: Tensor) -> dict[str, Tensor]:
    hit = projected.hit_mask & fill
    return {
        "features": torch.where(
            hit[..., None], projected.hit_latent, torch.zeros_like(projected.hit_latent)
        ),
        "coverage": hit,
        "moving": plan.motion_mask & hit,
        "depth": torch.where(hit, plan.depth, torch.zeros_like(plan.depth)),
        "source_keyframe": torch.where(
            hit, plan.source_view, torch.full_like(plan.source_view, -1)
        ),
        "source_token": torch.where(
            hit, plan.source_token, torch.full_like(plan.source_token, -1)
        ),
        "time_offset": torch.where(
            hit, plan.signed_dt, torch.zeros_like(plan.signed_dt)
        ),
        "score": torch.where(hit, projected.score, torch.zeros_like(projected.score)),
    }


def _read_group(
    config,
    entry,
    names,
    target_indices,
    spec,
    times,
    intrinsics,
    c2w,
    frame_indices,
    fills,
    static,
):
    device = entry.positions.device
    selected = torch.tensor(target_indices, device=device, dtype=torch.int64)
    target_times = times.index_select(1, selected)
    target_positions = frame_indices.index_select(1, selected)
    target_intr = intrinsics.index_select(1, selected)
    target_c2w = c2w.index_select(1, selected)
    directions = infer_target_time_directions(target_times)
    count = len(target_indices)
    outputs = {name: _empty(entry, name, count) for name in names}
    candidate_counts = dict.fromkeys(names, 0)
    group_fill = torch.zeros_like(fills[names[0]])
    for name in names:
        group_fill |= fills[name]
    if not bool(group_fill.any()):
        return outputs, candidate_counts

    positions, valid, classification, velocity = _prepare_geometry(
        entry,
        config,
        names,
        spec,
        static=static,
    )
    backend = (
        IdentityTransportBackend()
        if velocity is None
        else VelocityTransportBackend(*velocity)
    )
    grid_shape = config.image_size
    if config.native_resolution:
        target_intr = spec.adjust_intrinsics(target_intr)
        grid_shape = spec.grid_shape
        group_evidence = entry.features[names[0]].valid

    source_view_valid = None if entry.dropped is None else ~entry.dropped
    for start in range(0, count, config.target_chunk_size):
        end = min(start + config.target_chunk_size, count)
        active = torch.nonzero(
            group_fill[:, start:end].any(dim=(0, 2, 3)), as_tuple=False
        ).flatten()
        if active.numel() == 0:
            continue
        output_indices = active + start
        propagation = propagate_classified_anchors(
            positions,
            valid,
            entry.times,
            target_times.index_select(1, output_indices),
            target_time_direction=directions.index_select(1, output_indices),
            transport_backend=backend,
            classification=classification,
        )
        if config.max_projection_window_frames is not None:
            within_window = (
                target_positions.index_select(1, output_indices)[:, None]
                - entry.frame_indices[:, :, None]
            ).abs() <= config.max_projection_window_frames
            propagation = PropagationResult(
                propagation.points,
                propagation.valid & within_window[..., None, None],
                propagation.motion_mask,
                propagation.signed_dt,
                propagation.time_direction,
            )
        chunk_intr = target_intr.index_select(1, output_indices)
        chunk_pose = target_c2w.index_select(1, output_indices)
        options = dict(
            source_view_valid=source_view_valid,
            same_depth_abs_eps=config.same_depth_abs_eps,
            same_depth_rel_eps=config.same_depth_rel_eps,
        )
        if config.native_resolution:
            first = names[0]
            projected, plan = project(
                propagation,
                entry.features[first].features,
                entry.c2w,
                chunk_pose,
                chunk_intr,
                grid_shape,
                target_missing_mask=group_fill.index_select(1, output_indices),
                source_token_valid=group_evidence,
                **options,
            )
            for name in names:
                values = (
                    projected
                    if name == first
                    else apply_plan(entry.features[name].features, plan)
                )
                result = _masked_values(
                    values, plan, fills[name].index_select(1, output_indices)
                )
                for field, value in result.items():
                    outputs[name][field].index_copy_(1, output_indices, value)
                candidate_counts[name] += plan.candidate_count
        else:
            for name in names:
                projected, plan = project(
                    propagation,
                    entry.features[name].features,
                    entry.c2w,
                    chunk_pose,
                    chunk_intr,
                    grid_shape,
                    target_missing_mask=fills[name].index_select(1, output_indices),
                    source_token_valid=entry.features[name].valid,
                    **options,
                )
                result = _masked_values(
                    projected, plan, fills[name].index_select(1, output_indices)
                )
                for field, value in result.items():
                    outputs[name][field].index_copy_(1, output_indices, value)
                candidate_counts[name] += plan.candidate_count
    return outputs, candidate_counts


@torch.no_grad()
def read(
    config: LatentCacheConfig,
    entry: CacheEntry,
    times: Tensor,
    intrinsics: Tensor,
    c2w: Tensor,
    *,
    frame_indices: Tensor | None,
    valid: Tensor | None,
    fill: Mapping[str, Tensor] | None,
    static: bool,
) -> dict[str, Readout]:
    if not isinstance(entry, CacheEntry):
        raise TypeError("entry must be CacheEntry")
    frame_indices, valid = target_inputs(
        entry, times, intrinsics, c2w, frame_indices, valid
    )
    fills = _prepare_fills(entry, valid, fill)
    groups = _representation_groups(entry, config.image_size, times.shape[1])
    outputs = {}
    for indices, spec, names in groups:
        values, counts = _read_group(
            config,
            entry,
            names,
            indices,
            spec,
            times,
            intrinsics,
            c2w,
            frame_indices,
            fills,
            static,
        )
        for name in names:
            outputs[name] = Readout(
                **_readout_metadata(
                    indices,
                    times,
                    intrinsics,
                    c2w,
                    frame_indices,
                    entry.fps,
                    config.image_size,
                ),
                **values[name],
                candidate_count=counts[name],
            )
    return {name: outputs[name] for name in entry.features}
