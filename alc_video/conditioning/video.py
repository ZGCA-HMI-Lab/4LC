from __future__ import annotations

import torch
from latent_cache import CacheEntry, LatentCache

from alc_video.conditioning.rgb import PixelEncoder, encode_rgb_readouts
from alc_video.encoders.visual.depth import depth_to_rgb_frames
from alc_video.encoders.visual.loader import offload_visual_encoders
from alc_video.utils.logging import stage

DIRECT_VIDEO_SOURCE_VIEWS = 21


MIN_DIRECT_VIDEO_FRAMES = 12


TARGET_FRAMES = 81


def sample_keyframe_indices(frame_count: int, num_keyframes: int) -> torch.Tensor:
    if not 2 <= num_keyframes <= frame_count:
        raise ValueError("num_keyframes must be between 2 and the target frame count")
    return (
        torch.arange(num_keyframes, dtype=torch.int64)
        * (frame_count - 1)
        // (num_keyframes - 1)
    )


def _evenly_spaced_frame_indices(
    frame_count: int,
    selected_count: int,
) -> torch.Tensor:
    if frame_count < selected_count or selected_count <= 0:
        raise ValueError("frame_count must be >= selected_count > 0")
    indices = (
        torch.linspace(
            0,
            frame_count - 1,
            steps=selected_count,
            dtype=torch.float64,
        )
        .round()
        .to(dtype=torch.int64)
    )
    if int(torch.unique(indices).numel()) != selected_count:
        raise RuntimeError("evenly spaced frame selection produced duplicates")
    return indices


def direct_video_frame_count(input_frame_count: int) -> int:
    if not MIN_DIRECT_VIDEO_FRAMES <= input_frame_count <= TARGET_FRAMES:
        raise ValueError(
            f"direct video frame count must be in "
            f"[{MIN_DIRECT_VIDEO_FRAMES},{TARGET_FRAMES}]"
        )
    latent_frames = 1 + (input_frame_count - 1) // 4
    return 4 * latent_frames - 3


@torch.no_grad()
def prepare_direct_video_observations(
    frames,
    input_fps,
    worldmirror,
    encoders,
    *,
    keyframes=None,
    num_keyframes=DIRECT_VIDEO_SOURCE_VIEWS,
    transport="velocity",
):
    input_frame_count = int(frames.shape[0])
    if transport not in {"identity", "velocity"}:
        raise ValueError(f"invalid source transport: {transport!r}")
    if any(encoder.encoder_type == "wan_vae" for encoder in encoders.values()):
        if input_frame_count != TARGET_FRAMES:
            raise ValueError(
                f"direct Wan source encoding requires exactly {TARGET_FRAMES} frames"
            )
    target_frame_count = direct_video_frame_count(input_frame_count)
    frame_indices = torch.arange(input_frame_count, dtype=torch.int64)
    input_times = frame_indices.to(torch.float32) / float(input_fps)
    target_indices = _evenly_spaced_frame_indices(input_frame_count, target_frame_count)
    source_positions = (
        sample_keyframe_indices(target_frame_count, num_keyframes)
        if keyframes is None
        else torch.tensor(keyframes, dtype=torch.int64)
    )
    source_indices = target_indices.index_select(0, source_positions)
    duration = float(input_frame_count - 1) / float(input_fps)
    target_times = torch.linspace(
        0.0, duration, steps=target_frame_count, dtype=torch.float32
    )
    output_fps = float(target_frame_count - 1) / duration
    worldmirror.reconstructor_module.to(worldmirror.device)
    state = worldmirror.encode_pair(
        frames,
        None,
        frame_indices=frame_indices,
        times=input_times,
        source_indices=source_indices,
        transport=transport,
        fps=input_fps,
    )
    worldmirror.offload()
    if transport == "velocity" and state.source.motion is None:
        raise RuntimeError("direct ST video preprocessing omitted source motion")
    result = {
        "transport": transport,
        "frame_count": target_frame_count,
        "fps": torch.tensor(output_fps, dtype=torch.float32),
        "source_positions": source_positions,
        "source_times": target_times.index_select(0, source_positions),
        "source_intr": state.source_camera.intr.index_select(0, source_indices),
        "source_c2w": state.source_camera.c2w.index_select(0, source_indices),
        "source_depth": state.source.depth,
        "source_valid": state.source.valid,
        "source_rgb": frames.index_select(0, source_indices)
        .permute(0, 2, 3, 1)
        .contiguous(),
        "observation_times": target_times,
        "target_intr": state.target_camera.intr.index_select(0, target_indices),
        "target_c2w": state.target_camera.c2w.index_select(0, target_indices),
    }
    if transport == "velocity":
        result.update(
            velocity_fwd=state.source.motion["v_fwd"],
            velocity_bwd=state.source.motion["v_bwd"],
        )
    return result


def validate_drop_source_frames(
    drop_source_frames: tuple[int, ...],
    *,
    views: int,
) -> tuple[int, ...]:
    if not isinstance(drop_source_frames, tuple) or any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in drop_source_frames
    ):
        raise TypeError("drop_source_frames must be a tuple of integers")
    if tuple(sorted(set(drop_source_frames))) != drop_source_frames:
        raise ValueError(
            "drop_source_frames must contain strictly increasing unique indices"
        )
    if views <= 0:
        raise ValueError("source latent view count must be > 0")
    if drop_source_frames and (
        drop_source_frames[0] < 0 or drop_source_frames[-1] >= views
    ):
        raise IndexError(
            f"drop_source_frames must be in [0,{views - 1}], got {drop_source_frames}"
        )
    return drop_source_frames


def source_drop_mask(
    source_condition: torch.Tensor,
    drop_source_frames: tuple[int, ...],
) -> torch.Tensor | None:
    if not drop_source_frames:
        return None
    if (
        not isinstance(source_condition, torch.Tensor)
        or source_condition.ndim != 5
        or int(source_condition.shape[0]) <= 0
        or int(source_condition.shape[1]) <= 0
    ):
        raise ValueError("source condition must be [B,V,H,W,C]")
    views = int(source_condition.shape[1])
    validate_drop_source_frames(drop_source_frames, views=views)
    mask = torch.zeros(
        tuple(int(size) for size in source_condition.shape[:2]),
        dtype=torch.bool,
        device=source_condition.device,
    )
    mask[:, drop_source_frames] = True
    return mask


def construct_video_entry(
    observation: dict,
    cache: LatentCache,
    device: torch.device,
    *,
    drop_source_frames: tuple[int, ...] = (),
) -> CacheEntry:
    """Encode prepared observations once, retaining the entry for repeated readout."""
    positions = observation["source_positions"][None].to(device)
    depth = observation["source_depth"][None].to(device)
    valid = observation["source_valid"][None].to(device)
    velocity = (
        tuple(
            observation[name][None].to(device)
            for name in ("velocity_fwd", "velocity_bwd")
        )
        if observation["transport"] == "velocity"
        else None
    )
    dropped = source_drop_mask(
        observation["source_rgb"][None].to(device),
        tuple(drop_source_frames),
    )
    with stage("cache.construct", keyframes=positions.shape[1]):
        entry = cache.construct(
            {
                "rgb": observation["source_rgb"][None].to(device),
                "depth": depth_to_rgb_frames(depth, valid),
            },
            depth,
            valid[..., 0],
            observation["source_intr"][None].to(device),
            observation["source_c2w"][None].to(device),
            observation["source_times"][None].to(device),
            positions,
            observation["frame_count"],
            fps=float(observation["fps"]),
            timeline_times=observation["observation_times"][None].to(device),
            velocity=velocity,
            dropped=dropped,
        )
    return entry


def build_video_conditions(
    frames,
    source_fps,
    camera,
    *,
    source_path,
    mirror,
    encoders,
    cache,
    device,
    keyframes,
    num_keyframes,
    transport,
    drop_source_frames,
    propagate_rgb,
):
    with stage("geometry.reconstruct", input=source_path):
        observation = prepare_direct_video_observations(
            frames,
            source_fps,
            mirror,
            encoders,
            keyframes=keyframes,
            num_keyframes=num_keyframes,
            transport=transport,
        )
    if camera is None:
        intr = observation["target_intr"].to(device)
        c2w = observation["target_c2w"].to(device)
        times = observation["observation_times"].to(device)
        fps = float(observation["fps"])
    else:
        camera = camera.to(device)
        intr, c2w = camera.intr, camera.c2w
        times = (
            observation["observation_times"].to(device)
            if camera.times is None
            else camera.times
        )
        fps = float(observation["fps"]) if camera.fps is None else camera.fps
    active_cache = cache
    if propagate_rgb:
        active_cache = LatentCache(
            cache.config,
            {
                "rgb": (PixelEncoder(cache.config.image_size), "rgb"),
                **(
                    {"depth": cache.encoders["depth"]}
                    if "depth" in cache.encoders
                    else {}
                ),
            },
        )
    entry = construct_video_entry(
        observation, active_cache, device, drop_source_frames=tuple(drop_source_frames)
    )
    offload_visual_encoders(encoders)
    with stage("cache.read", targets=len(times)):
        readouts = active_cache.read(entry, times[None], intr[None], c2w[None])
    if propagate_rgb:
        readouts = encode_rgb_readouts(readouts, encoders)
        offload_visual_encoders(encoders)

    return readouts, intr, c2w, times, fps
