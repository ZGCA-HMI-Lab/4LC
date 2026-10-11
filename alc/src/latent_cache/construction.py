from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from latent_cache.geometry.candidates import DEPTH_EPS
from latent_cache.geometry.spatial import unproject_depth
from latent_cache.propagation.motion import select_motion
from latent_cache.propagation.transport import (
    IdentityTransportBackend,
    VelocityTransportBackend,
)
from latent_cache.types import (
    CacheEntry,
    LatentCacheConfig,
    Representation,
    RepresentationFeatures,
    VisualEncoder,
)


def encoder_frames(
    representation: Representation,
    frames: Tensor,
    frame_indices: tuple[int, ...],
    num_frames: int,
) -> Tensor:
    batch, count = frames.shape[:2]
    if representation.frame_layout == "selected":
        return frames
    if representation.frame_layout == "previous_current":
        previous = torch.cat((frames[:, :1], frames[:, :-1]), dim=1)
        return (
            torch.stack((previous, frames), dim=2)
            .reshape(batch, 2 * count, *frames.shape[2:])
            .contiguous()
        )
    if count == 1:
        if frame_indices != (0,):
            raise ValueError("a single causal_hold keyframe must be frame 0")
        return frames
    if frame_indices[0] != 0 or frame_indices[-1] != num_frames - 1:
        raise ValueError(
            "causal_hold keyframes must begin at frame 0 and end at frame T-1"
        )
    indices = torch.tensor(frame_indices, device=frames.device, dtype=torch.int64)
    timeline = torch.arange(num_frames, device=frames.device, dtype=torch.int64)
    owner = torch.searchsorted(indices, timeline, right=True) - 1
    return frames.index_select(1, owner).contiguous()


def keyframe_slots(
    representation: Representation, frame_indices: tuple[int, ...], num_frames: int
) -> tuple[int, tuple[int, ...]]:
    native, _groups = representation.native_layout(frame_indices, num_frames)
    return len(native), representation.keyframe_slots(frame_indices)


def check_tensor(
    value: Tensor, shape: tuple[int, ...], name: str, *, device, dtype=None
) -> None:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be {shape}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")
    if dtype is not None:
        if value.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}")
    elif not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating point")


@torch.no_grad()
def construct(
    config: LatentCacheConfig,
    encoders: Mapping[str, tuple[VisualEncoder, str]],
    *,
    index: int,
    frames: Mapping[str, Tensor],
    depth: Tensor,
    valid: Tensor,
    intrinsics: Tensor,
    c2w: Tensor,
    times: Tensor,
    frame_indices: Tensor,
    num_frames: int,
    fps: float | Tensor,
    timeline_times: Tensor,
    encoding_frame_indices: Tensor | None,
    encoder_frame_indices: Mapping[str, Tensor] | None,
    velocity: tuple[Tensor, Tensor] | None,
    motion_mask: Tensor | None,
    dropped: Tensor | None,
    feature_valid: Mapping[str, Tensor] | None,
    motion_threshold: float | None,
    tag: str | None,
) -> CacheEntry:
    if not isinstance(depth, Tensor) or depth.ndim != 4:
        raise ValueError("depth must be [B,K,H,W]")
    batch, count, height, width = depth.shape
    if (height, width) != config.image_size or count <= 0:
        raise ValueError("depth must contain keyframes on image_size")
    device = depth.device
    for name, value, shape, dtype in (
        ("depth", depth, (batch, count, height, width), None),
        ("valid", valid, depth.shape, torch.bool),
        ("intrinsics", intrinsics, (batch, count, 3, 3), None),
        ("c2w", c2w, (batch, count, 4, 4), None),
        ("times", times, (batch, count), None),
        ("frame_indices", frame_indices, (batch, count), torch.int64),
        ("timeline_times", timeline_times, (batch, num_frames), None),
    ):
        check_tensor(value, tuple(shape), name, device=device, dtype=dtype)
    if bool((frame_indices < 0).any() or (frame_indices >= num_frames).any()) or (
        count > 1 and not bool((frame_indices[:, 1:] > frame_indices[:, :-1]).all())
    ):
        raise ValueError("frame_indices must be strictly increasing in [0,T)")
    if batch > 1 and not bool((frame_indices == frame_indices[:1]).all()):
        raise ValueError("all samples must use the same frame_indices")
    if num_frames > 1 and not bool(
        (timeline_times[:, 1:] > timeline_times[:, :-1]).all()
    ):
        raise ValueError("timeline_times must be strictly increasing")
    if not torch.equal(times, timeline_times.gather(1, frame_indices)):
        raise ValueError("times must equal timeline_times at frame_indices")
    fps_tensor = (
        fps
        if isinstance(fps, Tensor)
        else torch.full((batch,), float(fps), device=device, dtype=torch.float32)
    )
    check_tensor(fps_tensor, (batch,), "fps", device=device)
    if not bool((fps_tensor > 0).all()):
        raise ValueError("fps must be positive")
    if not isinstance(frames, Mapping) or not frames:
        raise ValueError("frames must be a nonempty stream mapping")
    encoding_positions = (
        frame_indices if encoding_frame_indices is None else encoding_frame_indices
    )
    if (
        not isinstance(encoding_positions, Tensor)
        or encoding_positions.ndim != 2
        or encoding_positions.shape[0] != batch
    ):
        raise ValueError("encoding_frame_indices must be [B,M]")
    encoding_count = encoding_positions.shape[1]
    check_tensor(
        encoding_positions,
        (batch, encoding_count),
        "encoding_frame_indices",
        device=device,
        dtype=torch.int64,
    )
    if encoding_count <= 0 or bool(
        (encoding_positions < 0).any() or (encoding_positions >= num_frames).any()
    ):
        raise ValueError("encoding_frame_indices must index timeline_times")
    if encoding_count > 1 and not bool(
        (encoding_positions[:, 1:] > encoding_positions[:, :-1]).all()
    ):
        raise ValueError("encoding_frame_indices must be strictly increasing")
    if batch > 1 and not bool((encoding_positions == encoding_positions[:1]).all()):
        raise ValueError("samples must share encoding_frame_indices")
    for name, values in frames.items():
        check_tensor(
            values,
            (batch, encoding_count, height, width, 3),
            f"frames[{name!r}]",
            device=device,
        )
        if float(values.min()) < 0.0 or float(values.max()) > 1.0:
            raise ValueError("frame streams must be in [0,1]")
    if dropped is not None:
        check_tensor(
            dropped, (batch, count), "dropped", device=device, dtype=torch.bool
        )
        if bool(dropped.all(dim=1).any()):
            raise ValueError("dropped must keep one keyframe per sample")
    backend = (
        IdentityTransportBackend()
        if velocity is None
        else VelocityTransportBackend(*velocity)
    )
    if velocity is not None:
        check_tensor(
            velocity[0], (batch, count, height, width, 3), "velocity", device=device
        )
        if count < 2 or not bool((times[:, 1:] > times[:, :-1]).all()):
            raise ValueError(
                "velocity requires increasing times and at least two keyframes"
            )
    if feature_valid is not None and set(feature_valid) != set(encoders):
        raise ValueError("feature_valid must name every representation")
    if encoder_frame_indices is not None and set(encoder_frame_indices) != set(
        encoders
    ):
        raise ValueError("encoder_frame_indices must name every representation")
    indices = tuple(int(value) for value in frame_indices[0].tolist())
    encoding_indices = tuple(int(value) for value in encoding_positions[0].tolist())
    encoded = {}
    features = {}
    for name, (encoder, stream) in encoders.items():
        if stream not in frames:
            raise ValueError(f"representation {name!r} needs frame stream {stream!r}")
        representation = encoder.representation
        selected_indices = encoding_indices
        selected_frames = frames[stream]
        if encoder_frame_indices is not None:
            selected = encoder_frame_indices[name]
            if not isinstance(selected, Tensor) or selected.ndim != 2:
                raise ValueError(f"encoder_frame_indices[{name!r}] must be [B,M]")
            check_tensor(
                selected,
                (batch, selected.shape[1]),
                f"encoder_frame_indices[{name!r}]",
                device=device,
                dtype=torch.int64,
            )
            if selected.shape[1] <= 0 or (
                selected.shape[1] > 1
                and not bool((selected[:, 1:] > selected[:, :-1]).all())
            ):
                raise ValueError(
                    "encoder frame indices must be nonempty and increasing"
                )
            if batch > 1 and not bool((selected == selected[:1]).all()):
                raise ValueError("samples must share encoder frame indices")
            rows = torch.searchsorted(encoding_positions[0], selected[0])
            if bool((rows >= encoding_count).any()) or not torch.equal(
                encoding_positions[0].index_select(0, rows), selected[0]
            ):
                raise ValueError(
                    "encoder frame indices must be in the encoding context"
                )
            selected_indices = tuple(int(value) for value in selected[0].tolist())
            selected_frames = frames[stream].index_select(1, rows)
        native_positions, groups = representation.native_layout(
            selected_indices, num_frames
        )
        if representation.frame_layout == "causal_hold":
            slots = representation.keyframe_slots(indices)
        else:
            slot_by_position = {
                position: slot for slot, position in enumerate(native_positions)
            }
            if any(position not in slot_by_position for position in indices):
                raise ValueError(
                    "geometry keyframes must be present in the encoding context"
                )
            slots = tuple(slot_by_position[position] for position in indices)
        key = (id(encoder), stream, selected_indices)
        if key not in encoded:
            native = encoder.encode(
                encoder_frames(
                    representation, selected_frames, selected_indices, num_frames
                ),
                device=device,
            )
            encoded[key] = native
        else:
            native = encoded[key]
        check_tensor(
            native,
            (batch, len(native_positions), *representation.grid, representation.dim),
            f"encoder output of {name!r}",
            device=device,
        )
        slot_tensor = torch.tensor(slots, device=device, dtype=torch.int64)
        evidence = (
            torch.ones(
                (batch, count, *representation.grid), device=device, dtype=torch.bool
            )
            if feature_valid is None
            else feature_valid[name]
        )
        check_tensor(
            evidence,
            (batch, count, *representation.grid),
            f"feature_valid[{name!r}]",
            device=device,
            dtype=torch.bool,
        )
        endpoints = torch.tensor(
            tuple(native_positions[slot] for slot in slots),
            device=device,
            dtype=torch.int64,
        )
        features[name] = RepresentationFeatures(
            representation,
            native,
            native_positions,
            groups,
            native.index_select(1, slot_tensor).contiguous(),
            evidence,
            slot_tensor[None].expand(batch, -1),
            endpoints[None].expand(batch, -1),
        )
    positions = unproject_depth(depth, intrinsics, c2w)
    geometry_valid = valid & (depth > DEPTH_EPS)
    motion = select_motion(
        positions,
        geometry_valid,
        depth,
        times,
        fps_tensor,
        transport_backend=backend,
        source_intr=intrinsics,
        source_c2w=c2w,
        image_size=config.image_size,
        threshold=config.motion_threshold
        if motion_threshold is None
        else motion_threshold,
        source_motion_mask=motion_mask,
        motion_mask_dilation_radius=config.motion_mask_dilation,
        temporal_motion_propagation=config.temporal_motion_propagation,
        temporal_motion_depth_tolerance=config.temporal_motion_depth_tolerance,
    )
    return CacheEntry(
        index=index,
        tag=tag,
        num_keyframes=count,
        positions=positions,
        depth=depth,
        valid=geometry_valid,
        times=times,
        frame_indices=frame_indices,
        encoding_frame_indices=encoding_positions,
        timeline_times=timeline_times,
        fps=fps_tensor,
        intrinsics=intrinsics,
        c2w=c2w,
        velocity=velocity,
        dropped=dropped,
        motion=motion,
        features=features,
    )
