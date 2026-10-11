from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import torch
from latent_cache import CacheEntry
from latent_cache.geometry.spatial import unproject_depth
from latent_cache.propagation.motion import MotionSelection

from alc_video.conditioning.geometry import estimate_context_geometry
from alc_video.encoders.visual.depth import depth_to_rgb_frames
from alc_video.encoders.visual.loader import offload_visual_encoders
from alc_video.utils.logging import LOG


def _static_entry(entry, retained, *, start, index, tag):
    ids = torch.tensor(retained, device=entry.depth.device, dtype=torch.int64)
    count = len(retained)
    selected = {
        name: getattr(entry, name).index_select(1, ids)
        for name in ("depth", "valid", "intrinsics", "c2w", "frame_indices")
    }
    for name in ("depth", "intrinsics", "c2w"):
        selected[name] = selected[name].to(torch.bfloat16)
    selected["frame_indices"] = selected["frame_indices"] + start
    selected["positions"] = unproject_depth(
        selected["depth"], selected["intrinsics"], selected["c2w"]
    )
    zero = torch.zeros_like(selected["valid"])
    motion = MotionSelection(
        torch.zeros_like(selected["depth"]),
        selected["valid"],
        zero,
        zero,
        zero,
        zero,
    )
    features = {}
    for name, payload in entry.features.items():
        values = payload.features.index_select(1, ids).to(torch.bfloat16)
        endpoints = payload.native_endpoints.index_select(1, ids) + start
        positions = tuple(int(value) for value in endpoints[0])
        features[name] = replace(
            payload,
            features=values,
            native_features=values,
            valid=payload.valid.index_select(1, ids),
            keyframe_slots=torch.arange(count, device=ids.device)[None],
            native_endpoints=endpoints,
            native_positions=positions,
            native_frame_groups=tuple((value,) for value in positions),
        )
    return replace(
        entry,
        **selected,
        index=index,
        tag=tag,
        num_keyframes=count,
        times=torch.zeros(1, count, device=ids.device, dtype=torch.bfloat16),
        timeline_times=torch.zeros(
            1,
            start + entry.timeline_times.shape[1],
            device=ids.device,
            dtype=torch.bfloat16,
        ),
        fps=entry.fps.to(torch.bfloat16),
        motion=motion,
        velocity=None,
        dropped=None,
        features=features,
        encoding_frame_indices=selected["frame_indices"],
    )


@torch.no_grad()
def construct_static_entry(
    cache,
    rgb,
    depth,
    valid,
    moving,
    intrinsics,
    c2w,
    context_indices,
    *,
    num_frames,
    start,
    fps,
    depth_scale,
    index,
    tag,
):
    seed = num_frames == 1
    rows = torch.arange(0 if seed else 1, len(context_indices), device=rgb.device)
    keyframes = context_indices.index_select(0, rows)
    if not bool(valid.index_select(0, rows).any()):
        return None, tuple(int(value) + start for value in keyframes)
    streams = {
        "rgb": rgb[None],
        "depth": depth_to_rgb_frames(
            depth[None],
            valid[None, ..., None],
            normalization_scale=torch.tensor([depth_scale], device=rgb.device),
        ),
    }
    evidence = {
        name: ~encoder.representation.spatial_grid(cache.config.image_size).sample(
            moving.index_select(0, rows)[None], name=f"writer.{name}.motion"
        )
        for name, (encoder, _stream) in cache.encoders.items()
    }
    timeline = (torch.arange(num_frames, device=rgb.device)[None] + start).float() / fps
    entry = cache.construct(
        streams,
        depth.index_select(0, rows)[None],
        valid.index_select(0, rows)[None],
        intrinsics.index_select(0, rows)[None],
        c2w.index_select(0, rows)[None],
        timeline.index_select(1, keyframes),
        keyframes[None],
        num_frames,
        fps=fps,
        timeline_times=timeline,
        encoding_frame_indices=context_indices[None],
        encoder_frame_indices={
            name: (
                keyframes
                if encoder.representation.frame_layout == "selected"
                else context_indices
            )[None]
            for name, (encoder, _stream) in cache.encoders.items()
        },
        feature_valid=evidence,
    )
    cache.clear()
    retained = [
        row
        for row in range(entry.num_keyframes)
        if bool(valid[rows[row]].any())
        and any(
            bool(payload.valid[:, row].any()) for payload in entry.features.values()
        )
    ]
    rejected = tuple(
        int(keyframes[row]) + start
        for row in range(len(keyframes))
        if row not in retained
    )
    if not retained:
        return None, rejected
    return _static_entry(entry, retained, start=start, index=index, tag=tag), rejected


def merge_static_entries(seed, generated):
    tensors = {
        name: torch.cat((getattr(seed, name), getattr(generated, name)), dim=1)
        for name in (
            "positions",
            "depth",
            "valid",
            "times",
            "frame_indices",
            "encoding_frame_indices",
            "intrinsics",
            "c2w",
        )
    }
    count = seed.num_keyframes + generated.num_keyframes
    features = {}
    for name, right in generated.features.items():
        left = seed.features[name]
        features[name] = replace(
            right,
            **{
                field: torch.cat((getattr(left, field), getattr(right, field)), dim=1)
                for field in (
                    "features",
                    "valid",
                    "native_features",
                    "native_endpoints",
                )
            },
            keyframe_slots=torch.arange(count, device=right.features.device)[None],
            native_positions=left.native_positions + right.native_positions,
            native_frame_groups=left.native_frame_groups + right.native_frame_groups,
        )
    return replace(
        generated,
        **tensors,
        num_keyframes=count,
        features=features,
        motion=MotionSelection(
            **{
                item.name: torch.cat(
                    (
                        getattr(seed.motion, item.name),
                        getattr(generated.motion, item.name),
                    ),
                    dim=1,
                )
                for item in fields(seed.motion)
            }
        ),
    )


@dataclass
class RolloutState:
    overlap: torch.Tensor
    history: list[CacheEntry] = field(default_factory=list)
    depth_scale: float | None = None


def initialize_history(cache, components, state, camera):
    mirror, encoders = components.mirror, components.encoders
    overlap, history = state.overlap, state.history
    intr, c2w, fps = camera.intr, camera.c2w, camera.fps
    device = overlap.device
    indices = torch.zeros(1, device=device, dtype=torch.int64)
    depth, valid, moving = estimate_context_geometry(
        mirror,
        overlap[None],
        intr[:1].float(),
        c2w[:1].float(),
        indices,
        fps,
        cache.config,
    )
    values = depth[0].float().cpu()[valid[0].cpu()]
    state.depth_scale = float(torch.quantile(values, 0.98, interpolation="linear"))
    seed, rejected = construct_static_entry(
        cache,
        overlap[None],
        depth,
        valid,
        moving,
        intr[:1].float(),
        c2w[:1].float(),
        indices,
        num_frames=1,
        start=0,
        fps=fps,
        depth_scale=state.depth_scale,
        index=0,
        tag="seed",
    )
    if seed is None or rejected:
        raise RuntimeError("seed writer must create exactly one keyframe")
    history.append(seed)
    offload_visual_encoders(encoders)


def read_history(request, cache, state, camera, chunk_index):
    chunk_size = request.chunk_size
    start = chunk_index * (chunk_size - 1)
    stop = start + chunk_size
    intr, c2w = camera.intr, camera.c2w
    overlap, history = state.overlap, state.history
    device = overlap.device
    chunk_intr, chunk_c2w = intr[start:stop], c2w[start:stop]
    positions = torch.arange(start, stop, device=device)[None]
    chunk_intr, chunk_c2w = (
        chunk_intr.to(torch.bfloat16),
        chunk_c2w.to(torch.bfloat16),
    )
    scene_times = torch.zeros(1, chunk_size, device=device, dtype=torch.bfloat16)
    if chunk_index == 0:
        readouts = cache.read(
            history[0],
            scene_times,
            chunk_intr[None],
            chunk_c2w[None],
            frame_indices=positions,
            static=True,
        )

        rates = {
            name: int(value.coverage.count_nonzero()) / value.coverage.numel()
            for name, value in readouts.items()
        }
        selected, read_count, stop_reason = (
            (0,),
            1,
            "initialization_completed",
        )
    else:
        result = cache.compose(
            scene_times,
            chunk_intr[None],
            chunk_c2w[None],
            entries=history,
            frame_indices=positions,
            order=request.order,
            max_entries=request.max_entries,
            coverage_threshold=request.coverage_threshold,
        )
        readouts = result.readouts
        selected = result.trace.selected_indices
        read_count = len(result.trace.steps)
        stop_reason = result.trace.stop_reason
        rates = result.trace.coverage_rates
    LOG.info(
        "rollout.cache selected=%s read=%s stop=%s coverage=%s",
        selected,
        read_count,
        stop_reason,
        rates,
    )
    return readouts, chunk_intr, chunk_c2w


def write_history(
    request,
    components,
    cache,
    state,
    camera,
    keyframe_indices,
    generated,
    chunk_index,
):
    chunk_size = request.chunk_size
    start = chunk_index * (chunk_size - 1)
    stop = start + chunk_size
    intr, c2w, fps = camera.intr, camera.c2w, camera.fps
    overlap, history, depth_scale = state.overlap, state.history, state.depth_scale
    mirror, encoders = components.mirror, components.encoders
    device = overlap.device
    written, rejected = [], ()
    visible = generated.clone()
    visible[0] = overlap
    indices = keyframe_indices.to(device=device)
    context = visible.index_select(0, indices)
    depth, valid, moving = estimate_context_geometry(
        mirror,
        context,
        intr[start:stop].index_select(0, indices).float(),
        c2w[start:stop].index_select(0, indices).float(),
        indices + start,
        fps,
        cache.config,
    )
    entry, rejected = construct_static_entry(
        cache,
        context,
        depth,
        valid,
        moving,
        intr[start:stop].index_select(0, indices).float(),
        c2w[start:stop].index_select(0, indices).float(),
        indices,
        num_frames=chunk_size,
        start=start,
        fps=fps,
        depth_scale=depth_scale,
        index=chunk_index,
        tag=f"chunk-{chunk_index}:{start}-{stop - 1}",
    )
    offload_visual_encoders(encoders)
    if entry is not None:
        written = entry.frame_indices[0].tolist()
        if start == 0:
            history[0] = merge_static_entries(history[0], entry)
        else:
            history.append(entry)
    return written, rejected
