from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from latent_cache import CacheEntry, LatentCache, Readout

from alc_video.conditioning.geometry import estimate_context_geometry
from alc_video.encoders.geometry.types import CameraTrajectory
from alc_video.encoders.visual.depth import depth_to_rgb_frames
from alc_video.encoders.visual.loader import offload_visual_encoders
from alc_video.utils.logging import stage

if TYPE_CHECKING:
    from alc_video.encoders.geometry.worldmirror import WorldMirrorGeometryEncoder


def validate_first_frame_alignment(
    *,
    source_positions: torch.Tensor,
    source_times: torch.Tensor,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    target_times: torch.Tensor,
    target_intr: torch.Tensor,
    target_c2w: torch.Tensor,
) -> None:
    if bool((source_positions[:, 0] != 0).any()):
        raise ValueError("first_frame requires source position 0 to be F0")
    if not torch.equal(source_times[:, 0], target_times[:, 0]):
        raise ValueError("first_frame source and target F0 times differ")
    if not torch.equal(source_intr[:, 0], target_intr[:, 0]):
        raise ValueError("first_frame source and target F0 intrinsics differ")
    if not torch.equal(source_c2w[:, 0], target_c2w[:, 0]):
        raise ValueError("first_frame source and target F0 cameras differ")


def build_pure_first_frame_readouts(
    *,
    entry: CacheEntry,
    times: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    frame_indices: torch.Tensor,
) -> dict[str, Readout]:
    if not isinstance(entry, CacheEntry):
        raise TypeError("pure first_frame requires CacheEntry")
    if (
        tuple(entry.frame_indices.shape) != (1, 1)
        or int(entry.frame_indices[0, 0]) != 0
    ):
        raise ValueError("pure first_frame requires one F0 source keyframe")
    if int(times.shape[0]) != 1:
        raise ValueError("pure first_frame requires target batch size 1")
    readouts = {}
    for name, payload in entry.features.items():
        representation = payload.representation
        positions = representation.target_indices(int(times.shape[1]))
        if not positions or positions[0] != 0:
            raise ValueError(f"pure first_frame feature {name!r} has no F0 target slot")
        device = payload.features.device
        if times.device != device:
            raise ValueError("pure first_frame entry and target devices differ")
        batch = int(payload.features.shape[0])
        if batch != 1 or int(payload.features.shape[1]) != 1:
            raise ValueError(
                f"pure first_frame feature {name!r} must contain one keyframe"
            )
        height, width, channels = (int(value) for value in payload.features.shape[2:])
        scalar_shape = (batch, len(positions), height, width)
        values = torch.zeros(
            (*scalar_shape, channels), device=device, dtype=payload.features.dtype
        )
        hit_mask = torch.zeros(scalar_shape, device=device, dtype=torch.bool)
        values[:, 0] = payload.features[:, 0]
        hit_mask[:, 0] = payload.valid[:, 0]
        source_token = torch.full(scalar_shape, -1, device=device, dtype=torch.int64)
        source_view = source_token.clone()
        source_token[:, 0][hit_mask[:, 0]] = 0
        source_view[:, 0][hit_mask[:, 0]] = 0
        indices = torch.tensor(positions, device=device, dtype=torch.int64)
        readouts[name] = Readout(
            target_indices=positions,
            features=values,
            coverage=hit_mask,
            moving=torch.zeros_like(hit_mask),
            score=hit_mask.float(),
            source_token=source_token,
            source_keyframe=source_view,
            time_offset=torch.zeros(scalar_shape, device=device, dtype=torch.float32),
            depth=torch.zeros(scalar_shape, device=device, dtype=torch.float32),
            candidate_count=0,
            target_frame_indices=frame_indices.index_select(1, indices),
            times=times.index_select(1, indices),
            fps=entry.fps,
            intrinsics=intrinsics.index_select(1, indices),
            c2w=c2w.index_select(1, indices),
            image_size=(int(entry.depth.shape[2]), int(entry.depth.shape[3])),
        )
    return readouts


def construct_first_frame_entry(
    frame: torch.Tensor,
    intr: torch.Tensor,
    c2w: torch.Tensor,
    fps: float,
    mirror: WorldMirrorGeometryEncoder,
    cache: LatentCache,
    *,
    pure: bool = False,
) -> CacheEntry:
    with stage("geometry.first_frame"):
        depth, valid, _moving = estimate_context_geometry(
            mirror,
            frame[None],
            intr[:1].float(),
            c2w[:1].float(),
            torch.zeros(1, device=frame.device, dtype=torch.int64),
            fps,
            cache.config,
        )
    evidence = None
    if pure:
        evidence = {
            name: (
                encoder.representation.spatial_grid(cache.config.image_size).sample(
                    valid[None], name="i2v.depth.valid"
                )
                if name == "depth"
                else torch.ones(
                    1,
                    1,
                    *encoder.representation.grid,
                    device=frame.device,
                    dtype=torch.bool,
                )
            )
            for name, (encoder, _stream) in cache.encoders.items()
        }
    with stage("cache.construct", keyframes=1):
        entry = cache.construct(
            {
                "rgb": frame[None, None],
                "depth": depth_to_rgb_frames(depth[None], valid[None, ..., None]),
            },
            depth[None],
            valid[None],
            intr[:1][None].to(torch.bfloat16),
            c2w[:1][None].to(torch.bfloat16),
            torch.zeros(1, 1, device=frame.device),
            torch.zeros(1, 1, device=frame.device, dtype=torch.int64),
            1,
            fps=torch.tensor([fps], device=frame.device, dtype=torch.bfloat16),
            timeline_times=torch.zeros(1, 1, device=frame.device),
            feature_valid=evidence,
        )
    offload_visual_encoders({name: value[0] for name, value in cache.encoders.items()})
    return entry


def build_image_conditions(
    frames: torch.Tensor,
    camera: CameraTrajectory,
    *,
    mode: Literal["i2v", "geometry_i2v"],
    mirror: WorldMirrorGeometryEncoder,
    cache: LatentCache,
    device: torch.device,
) -> tuple[dict[str, Readout], torch.Tensor, torch.Tensor, torch.Tensor, float]:
    camera = camera.to(device)
    intr, c2w, times, fps = camera.intr, camera.c2w, camera.times, camera.fps
    assert times is not None and fps is not None
    entry = construct_first_frame_entry(
        frames[0].permute(1, 2, 0).to(device),
        intr,
        c2w,
        fps,
        mirror,
        cache,
        pure=mode == "i2v",
    )
    intr, c2w, times = (value.to(torch.bfloat16) for value in (intr, c2w, times))
    if mode == "i2v":
        readouts = build_pure_first_frame_readouts(
            entry=entry,
            times=times[None],
            intrinsics=intr[None],
            c2w=c2w[None],
            frame_indices=torch.arange(len(times), device=device)[None],
        )
    else:
        with stage("cache.read", targets=len(times)):
            readouts = cache.read(
                entry, times[None], intr[None], c2w[None], static=True
            )
    return readouts, intr, c2w, times, fps
