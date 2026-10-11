from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace

import torch
from torch import Tensor

from latent_cache.construction import check_tensor
from latent_cache.editing.data import CacheData, _Transform, evaluate, per_block
from latent_cache.editing.operations import merge
from latent_cache.editing.projection import combine_candidates, joint_plan
from latent_cache.geometry.candidates import build_candidates
from latent_cache.geometry.projection import ProjectionResult
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
from latent_cache.readout import (
    _empty,
    _masked_values,
    _prepare_fills,
    _prepare_geometry,
    _readout_metadata,
    _representation_groups,
    target_inputs,
)
from latent_cache.types import CacheEntry, LatentCacheConfig, Readout


def _candidates(
    block,
    geometry,
    local_times,
    matrices,
    active,
    directions,
    source_frames,
    target_intr,
    target_pose,
    requested,
    config,
    names,
    source_index,
    source,
    total_views,
):
    entry = block.entry
    batch, queries = local_times.shape
    positions, valid, classification, velocity = geometry
    image_size = tuple(positions.shape[2:4])
    grid = entry.features[names[0]].representation.grid
    parts, gates, domains = [], [], []
    for batch_index in range(batch):
        indices = torch.nonzero(
            active[batch_index] & requested[batch_index].any(dim=(-2, -1)),
            as_tuple=False,
        ).flatten()
        if not indices.numel() or not bool(valid[batch_index].any()):
            continue
        row = slice(batch_index, batch_index + 1)
        motion = MotionSelection(
            **{
                field.name: getattr(classification, field.name)[row]
                for field in fields(classification)
            }
        )
        backend = (
            IdentityTransportBackend()
            if velocity is None
            else VelocityTransportBackend(*(value[row] for value in velocity))
        )
        propagation = propagate_classified_anchors(
            positions[row],
            valid[row],
            entry.times[row],
            local_times[row].index_select(1, indices),
            target_time_direction=directions[row].index_select(1, indices),
            transport_backend=backend,
            classification=motion,
        )
        for position, query in enumerate(indices.tolist()):
            matrix = matrices[batch_index, query]
            linear, translation = matrix[:3, :3], matrix[:3, 3]
            scale = linear.square().sum(-1).mean().sqrt()
            source_pose = entry.c2w[row].float().clone()
            source_pose[..., :3, :3] = (linear / scale) @ source_pose[..., :3, :3]
            source_pose[..., :3, 3] = source_pose[..., :3, 3] @ linear.T + translation
            point_valid = propagation.valid[:, :, position : position + 1]
            if config.max_projection_window_frames is not None:
                within = (
                    source_frames[row, query : query + 1] - entry.frame_indices[row]
                ).abs() <= config.max_projection_window_frames
                point_valid = point_valid & within[:, :, None, None, None]
            propagated = PropagationResult(
                propagation.points[:, :, position : position + 1].float() @ linear.T
                + translation,
                point_valid,
                propagation.motion_mask,
                propagation.signed_dt[:, :, position : position + 1],
                propagation.time_direction[:, position : position + 1],
            )
            candidate = build_candidates(
                propagated,
                source_pose,
                target_pose[row, query : query + 1],
                target_intr[row, query : query + 1],
                image_size,
                grid,
                source_view_valid=None
                if entry.dropped is None
                else ~entry.dropped[row],
                source_token_valid=entry.features[names[0]].valid[row],
            )
            if not len(candidate):
                continue
            evidence = {}
            for name in names:
                mask = block.feature_masks.get(name)
                evidence[name] = (
                    torch.ones_like(candidate.source_token, dtype=torch.bool)
                    if mask is None
                    else mask[batch_index].reshape(-1)[candidate.source_token]
                )
            offset = source.keyframe_offset
            parts.append(
                replace(
                    candidate,
                    batch_index=torch.full_like(candidate.batch_index, batch_index),
                    query_index=torch.full_like(candidate.query_index, query),
                    source_view=candidate.source_view + offset,
                    source_token=candidate.source_token + offset * grid[0] * grid[1],
                    anchor_id=candidate.anchor_id
                    + offset * image_size[0] * image_size[1],
                    batch_size=batch,
                    query_count=queries,
                    source_view_count=total_views,
                )
            )
            gates.append(evidence)
            domains.append(torch.full_like(candidate.source_token, source_index))
    return parts, gates, domains


def _gather(sources, entry, name, plan, candidate_index, gate, fill, *, protected=None):
    first = entry.features[name]
    hit = plan.hit_mask & fill
    if gate.numel():
        visible = gate[candidate_index.clamp_min(0)]
        if protected is not None:
            protected.copy_(hit & ~visible)
        hit = hit & visible
    shape = (*hit.shape, first.representation.dim)
    features = first.features.new_zeros(shape)
    for source in sources:
        payload = source.entry.features[name]
        offset = source.token_offset(name)
        count = (
            payload.features.shape[1]
            * payload.representation.grid[0]
            * payload.representation.grid[1]
        )
        owned = (
            hit & (plan.source_token >= offset) & (plan.source_token < offset + count)
        )
        batch, query, y, x = torch.where(owned)
        if batch.numel():
            tokens = plan.source_token[owned] - offset
            features[batch, query, y, x] = payload.features.reshape(
                payload.features.shape[0], count, -1
            )[batch, tokens]
    projected = ProjectionResult(features, hit, plan.score)
    return _masked_values(projected, plan, fill)


@torch.no_grad()
def _read(
    data: CacheEntry | CacheData,
    times: Tensor,
    intrinsics: Tensor,
    c2w: Tensor,
    *,
    config: LatentCacheConfig,
    fps: float | Tensor,
    frame_indices: Tensor | None = None,
    valid: Tensor | None = None,
    fill: Mapping[str, Tensor] | None = None,
    static: bool = False,
    source_frame_indices=None,
    protected=None,
) -> dict[str, Readout]:
    """Optionally collect visible editing exclusions for ordered composition."""
    if not isinstance(config, LatentCacheConfig):
        raise TypeError("config must be LatentCacheConfig")
    data = merge([data])
    first = data.blocks[0].entry
    if tuple(first.depth.shape[-2:]) != config.image_size:
        raise ValueError("config.image_size must match the source geometry")
    frame_indices, valid = target_inputs(
        first, times, intrinsics, c2w, frame_indices, valid
    )
    batch, target_count = times.shape
    output_fps = torch.as_tensor(fps, device=times.device, dtype=torch.float32)
    if output_fps.ndim == 0:
        output_fps = output_fps.expand(batch)
    check_tensor(output_fps, (batch,), "fps", device=times.device)
    if not bool((output_fps > 0).all()):
        raise ValueError("fps must be positive")
    fills = _prepare_fills(first, valid, fill)

    frames = per_block(source_frame_indices, len(data.blocks), "source_frame_indices")
    queries = []
    for block, source_frames in zip(data.blocks, frames, strict=True):
        local_times, matrices, active = evaluate(block, times)
        if source_frames is None:
            if config.max_projection_window_frames is not None and any(
                isinstance(op, _Transform) and op.time_map is not None
                for op in block.operations
            ):
                raise ValueError(
                    "retimed frame-window queries require source_frame_indices"
                )
            source_frames = frame_indices
        if (
            not isinstance(source_frames, Tensor)
            or source_frames.shape != times.shape
            or source_frames.device != times.device
            or not bool(torch.isfinite(source_frames).all())
        ):
            raise ValueError(
                "source_frame_indices must match [B,N] query times and device"
            )
        queries.append((local_times, matrices, active & valid, source_frames))

    groups = _representation_groups(
        first,
        config.image_size,
        target_count,
        split_roles=not config.native_resolution,
    )
    sources = data.sources
    source_by_id = {
        source.identity: (index, source) for index, source in enumerate(sources)
    }
    total_views = sum(source.entry.num_keyframes for source in sources)
    readouts = {}
    for indices, spec, names in groups:
        selected = torch.tensor(indices, dtype=torch.long, device=times.device)
        group_fill = torch.stack([fills[name] for name in names]).any(0)
        values = {name: _empty(first, name, len(indices)) for name in names}
        if protected is not None:
            for name in names:
                protected[name] = torch.zeros_like(values[name]["coverage"])
        count = 0
        geometry = []
        mapped = []
        for block, (local_times, matrices, active, source_frames) in zip(
            data.blocks, queries, strict=True
        ):
            geometry.append(
                _prepare_geometry(
                    block.entry,
                    config,
                    names,
                    spec,
                    static=static,
                    mask=block.mask,
                )
            )
            local = local_times.index_select(1, selected)
            mapped.append(
                (
                    local,
                    matrices.index_select(1, selected),
                    active.index_select(1, selected),
                    infer_target_time_directions(local),
                    source_frames.index_select(1, selected),
                )
            )
        target_intr = intrinsics.index_select(1, selected)
        if config.native_resolution:
            target_intr = spec.adjust_intrinsics(target_intr)
        target_pose = c2w.index_select(1, selected)
        for start in range(0, len(indices), config.target_chunk_size):
            end = min(start + config.target_chunk_size, len(indices))
            requested = group_fill[:, start:end]
            if not bool(requested.any()):
                continue
            parts, gates, domains = [], [], []
            for block, shape, query in zip(data.blocks, geometry, mapped, strict=True):
                source_index, source = source_by_id[block.identity]
                found, masks, ids = _candidates(
                    block,
                    shape,
                    *(value[:, start:end] for value in query),
                    target_intr[:, start:end],
                    target_pose[:, start:end],
                    requested,
                    config,
                    names,
                    source_index,
                    source,
                    total_views,
                )
                parts.extend(found)
                gates.extend(masks)
                domains.extend(ids)
            if not parts:
                continue
            candidates, gate, domain = combine_candidates(parts, gates, domains)
            plan, winner = joint_plan(
                candidates,
                domain,
                len(sources),
                spec.grid_shape,
                requested,
                depth_abs_eps=config.same_depth_abs_eps,
                depth_rel_eps=config.same_depth_rel_eps,
            )
            count += len(candidates)
            for name in names:
                result = _gather(
                    sources,
                    first,
                    name,
                    plan,
                    winner,
                    gate[name],
                    fills[name][:, start:end],
                    protected=None
                    if protected is None
                    else protected[name][:, start:end],
                )
                for field, value in result.items():
                    values[name][field][:, start:end] = value
        for name in names:
            readouts[name] = Readout(
                **_readout_metadata(
                    indices,
                    times,
                    intrinsics,
                    c2w,
                    frame_indices,
                    output_fps,
                    config.image_size,
                ),
                **values[name],
                candidate_count=count,
            )
    return {name: readouts[name] for name in first.features}


@torch.no_grad()
def read(
    data: CacheEntry | CacheData,
    times: Tensor,
    intrinsics: Tensor,
    c2w: Tensor,
    *,
    config: LatentCacheConfig,
    fps: float | Tensor,
    frame_indices: Tensor | None = None,
    valid: Tensor | None = None,
    fill: Mapping[str, Tensor] | None = None,
    static: bool = False,
    source_frame_indices=None,
) -> dict[str, Readout]:
    """Read edited sources through one joint geometric visibility selection.

    ``source_frame_indices`` is a [B,N] tensor for one block or a sequence for
    multiple blocks. Supply it when using a frame-distance window with retiming;
    it may pause/reverse independently of the increasing output frame indices.
    """
    return _read(
        data,
        times,
        intrinsics,
        c2w,
        config=config,
        fps=fps,
        frame_indices=frame_indices,
        valid=valid,
        fill=fill,
        static=static,
        source_frame_indices=source_frame_indices,
    )
