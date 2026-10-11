from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from latent_cache.propagation.transport import (
    IdentityTransportBackend,
    TransportBackend,
    query_transport,
)

TEMPORAL_DEPTH_ABS_EPS = 1.0e-3


@dataclass(frozen=True)
class MotionSelection:
    magnitude: torch.Tensor
    anchor_mask: torch.Tensor
    motion_mask: torch.Tensor
    unresolved_mask: torch.Tensor
    available_fwd: torch.Tensor
    available_bwd: torch.Tensor


@dataclass(frozen=True)
class PropagationResult:
    points: torch.Tensor
    valid: torch.Tensor
    motion_mask: torch.Tensor
    signed_dt: torch.Tensor
    time_direction: torch.Tensor


def infer_target_time_directions(target_times: torch.Tensor) -> torch.Tensor:
    if target_times.ndim != 2 or int(target_times.shape[1]) == 0:
        raise ValueError("target_times must be non-empty [B,T]")
    if not torch.is_floating_point(target_times):
        raise TypeError("target_times must be floating point")
    if not bool(torch.isfinite(target_times).all()):
        raise ValueError("target_times contains non-finite values")
    batch, queries = (int(value) for value in target_times.shape)
    if queries == 1:
        return torch.ones(
            (batch, 1),
            device=target_times.device,
            dtype=torch.int8,
        )

    differences = target_times[:, 1:] - target_times[:, :-1]
    interval_direction = torch.sign(differences).to(dtype=torch.int8)

    for index in range(queries - 3, -1, -1):
        interval_direction[:, index] = torch.where(
            interval_direction[:, index] == 0,
            interval_direction[:, index + 1],
            interval_direction[:, index],
        )
    for index in range(1, queries - 1):
        interval_direction[:, index] = torch.where(
            interval_direction[:, index] == 0,
            interval_direction[:, index - 1],
            interval_direction[:, index],
        )
    interval_direction = torch.where(
        interval_direction == 0,
        torch.ones_like(interval_direction),
        interval_direction,
    )
    return torch.cat(
        (interval_direction, interval_direction[:, -1:]),
        dim=1,
    ).contiguous()


def _dilate_motion_mask(
    motion_mask: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    if isinstance(radius, bool) or not isinstance(radius, int):
        raise TypeError("motion mask dilation radius must be an integer")
    if radius < 0:
        raise ValueError("motion mask dilation radius must be >= 0")
    if radius == 0:
        return motion_mask
    batch, views, height, width = (int(value) for value in motion_mask.shape)
    pooled = F.max_pool2d(
        motion_mask.reshape(batch * views, 1, height, width).float(),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return pooled.reshape_as(motion_mask).bool()


def _validate_classification_inputs(
    source_anchor: torch.Tensor,
    source_valid: torch.Tensor,
    source_depth: torch.Tensor,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    source_times: torch.Tensor,
    image_size: tuple[int, int],
) -> None:
    if source_anchor.ndim != 5 or int(source_anchor.shape[-1]) != 3:
        raise ValueError(
            f"source_anchor must be [B,V,H,W,3], got {tuple(source_anchor.shape)}"
        )
    expected_scalar_shape = tuple(source_anchor.shape[:-1])
    if tuple(source_valid.shape) != expected_scalar_shape:
        raise ValueError(
            f"source_valid must be [B,V,H,W], got {tuple(source_valid.shape)}"
        )
    if source_valid.dtype != torch.bool:
        raise TypeError("source_valid must be bool")
    if tuple(source_depth.shape) != expected_scalar_shape:
        raise ValueError(
            f"source_depth must be [B,V,H,W], got {tuple(source_depth.shape)}"
        )
    batch, views = int(source_anchor.shape[0]), int(source_anchor.shape[1])
    if tuple(source_intr.shape) != (batch, views, 3, 3):
        raise ValueError(
            f"source_intr must be [B,V,3,3], got {tuple(source_intr.shape)}"
        )
    if tuple(source_c2w.shape) != (batch, views, 4, 4):
        raise ValueError(f"source_c2w must be [B,V,4,4], got {tuple(source_c2w.shape)}")
    if tuple(source_times.shape) != (batch, views):
        raise ValueError(
            f"source_times must be [B,V]={batch, views}, got {tuple(source_times.shape)}"
        )
    image_size = tuple(int(value) for value in image_size)
    if image_size != tuple(int(value) for value in source_anchor.shape[2:4]):
        raise ValueError(
            f"image_size {image_size} must match source grid "
            f"{tuple(source_anchor.shape[2:4])}"
        )
    floating = (
        source_anchor,
        source_depth,
        source_intr,
        source_c2w,
        source_times,
    )
    if not all(torch.is_floating_point(value) for value in floating):
        raise TypeError("motion classification tensors must be floating point")
    if not all(value.device == source_anchor.device for value in floating):
        raise ValueError("motion classification tensors must share one device")
    if source_valid.device != source_anchor.device:
        raise ValueError("source_valid and source points must share one device")
    if not all(bool(torch.isfinite(value).all()) for value in floating):
        raise ValueError("motion classification input contains non-finite values")
    if bool((source_times[:, 1:] < source_times[:, :-1]).any()):
        raise ValueError("source_times must be nondecreasing")
    if bool((source_depth <= 0).all()):
        raise ValueError("source_depth must contain positive values")


def _validate_propagation_inputs(
    source_anchor: torch.Tensor,
    source_valid: torch.Tensor,
    target_times: torch.Tensor,
    target_time_direction: torch.Tensor,
) -> None:
    if source_anchor.ndim != 5 or int(source_anchor.shape[-1]) != 3:
        raise ValueError(
            f"source_anchor must be [B,V,H,W,3], got {tuple(source_anchor.shape)}"
        )
    expected_scalar_shape = tuple(source_anchor.shape[:-1])
    if tuple(source_valid.shape) != expected_scalar_shape:
        raise ValueError(
            f"source_valid must be [B,V,H,W], got {tuple(source_valid.shape)}"
        )
    if source_valid.dtype != torch.bool:
        raise TypeError("source_valid must be bool")
    batch = int(source_anchor.shape[0])
    if target_times.ndim != 2 or int(target_times.shape[0]) != batch:
        raise ValueError(
            f"target_times must be [B,T] with B={batch}, got {tuple(target_times.shape)}"
        )
    if not torch.is_floating_point(target_times):
        raise TypeError("propagation target times must be floating point")
    if tuple(target_time_direction.shape) != tuple(target_times.shape):
        raise ValueError("target_time_direction must match target_times")
    if target_time_direction.dtype != torch.int8:
        raise TypeError("target_time_direction must be int8")
    if (
        source_valid.device != source_anchor.device
        or target_times.device != source_anchor.device
        or target_time_direction.device != source_anchor.device
    ):
        raise ValueError("propagation tensors must share one device")
    if not bool(torch.isfinite(target_times).all()):
        raise ValueError("propagation input contains non-finite values")
    if not bool(((target_time_direction == 1) | (target_time_direction == -1)).all()):
        raise ValueError("target_time_direction values must be -1 or 1")


def _validate_source_times(
    source_times: torch.Tensor,
    *,
    batch: int,
    views: int,
    source_anchor: torch.Tensor,
) -> None:
    if tuple(source_times.shape) != (batch, views):
        raise ValueError(
            f"source_times must be [B,V]={batch, views}, got {tuple(source_times.shape)}"
        )
    if not torch.is_floating_point(source_times):
        raise TypeError("source_times must be floating point")
    if source_times.device != source_anchor.device:
        raise ValueError("source_times and source points must share one device")
    if not bool(torch.isfinite(source_times).all()):
        raise ValueError("source_times contains non-finite values")


def _validate_motion_selection(
    classification: MotionSelection,
    *,
    source_anchor: torch.Tensor,
    source_valid: torch.Tensor,
) -> None:
    if not isinstance(classification, MotionSelection):
        raise TypeError("classification must be MotionSelection")
    expected = tuple(source_anchor.shape[:-1])
    if tuple(classification.magnitude.shape) != expected:
        raise ValueError("classification.magnitude must be [B,V,H,W]")
    if not torch.is_floating_point(classification.magnitude):
        raise TypeError("classification.magnitude must be floating point")
    if classification.magnitude.device != source_anchor.device:
        raise ValueError("classification and source points must share one device")
    if not bool(torch.isfinite(classification.magnitude).all()):
        raise ValueError("classification.magnitude contains non-finite values")
    masks = (
        classification.anchor_mask,
        classification.motion_mask,
        classification.unresolved_mask,
        classification.available_fwd,
        classification.available_bwd,
    )
    if any(tuple(mask.shape) != expected for mask in masks):
        raise ValueError("classification masks must be [B,V,H,W]")
    if any(mask.dtype != torch.bool for mask in masks):
        raise TypeError("classification masks must be bool")
    if any(mask.device != source_anchor.device for mask in masks):
        raise ValueError("classification and source points must share one device")
    classified = classification.anchor_mask | classification.motion_mask
    if bool((classification.anchor_mask & classification.motion_mask).any()):
        raise ValueError("reference and motion selection masks must be disjoint")
    if bool((classified & ~source_valid).any()):
        raise ValueError("classification marks an invalid source point")


def _bracketing_source_mask(
    source_times: torch.Tensor,
    target_times: torch.Tensor,
    required_batch: torch.Tensor,
) -> torch.Tensor:
    signed_dt = target_times[:, None].float() - source_times[:, :, None].float()
    infinity = torch.full_like(signed_dt, float("inf"))
    left_distance = torch.where(signed_dt >= 0, signed_dt, infinity)
    right_distance = torch.where(signed_dt <= 0, -signed_dt, infinity)
    bracketed = torch.isfinite(left_distance.amin(dim=1)) & torch.isfinite(
        right_distance.amin(dim=1)
    )
    if bool((required_batch[:, None] & ~bracketed).any()):
        raise ValueError("motion target_times must lie within the source time range")
    left_index = left_distance.argmin(dim=1, keepdim=True)
    right_index = right_distance.argmin(dim=1, keepdim=True)
    result = torch.zeros_like(signed_dt, dtype=torch.bool)
    result.scatter_(1, left_index, True)
    result.scatter_(1, right_index, True)
    return result & required_batch[:, None, None]


def _project_source_pixels(
    points: torch.Tensor,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_size
    rotation = source_c2w.float()[..., :3, :3]
    translation = source_c2w.float()[..., :3, 3]
    centered = points.float() - translation[:, :, None, None]
    camera = torch.einsum("bvhwc,bvcd->bvhwd", centered, rotation)
    depth = camera[..., 2]
    safe_depth = depth.clamp_min(1.0e-6)
    intr = source_intr.float()
    u = intr[..., 0, 0, None, None] * float(width) * camera[..., 0] / safe_depth + intr[
        ..., 0, 2, None, None
    ] * float(width)
    v = intr[..., 1, 1, None, None] * float(height) * camera[
        ..., 1
    ] / safe_depth + intr[..., 1, 2, None, None] * float(height)
    return torch.stack((u, v), dim=-1), depth


def _splat_projected_mask(
    mask: torch.Tensor,
    points: torch.Tensor,
    target_intr: torch.Tensor,
    target_c2w: torch.Tensor,
    target_depth: torch.Tensor,
    image_size: tuple[int, int],
    *,
    depth_tolerance: float,
) -> torch.Tensor:
    pixels, depth = _project_source_pixels(
        points,
        target_intr,
        target_c2w,
        image_size,
    )
    height, width = image_size
    rounded_x = pixels[..., 0].round()
    rounded_y = pixels[..., 1].round()
    valid = (
        mask
        & torch.isfinite(pixels).all(dim=-1)
        & torch.isfinite(depth)
        & (depth > 1.0e-6)
        & (rounded_x >= 0)
        & (rounded_x < width)
        & (rounded_y >= 0)
        & (rounded_y < height)
    )
    x = rounded_x.clamp(0, width - 1).long()
    y = rounded_y.clamp(0, height - 1).long()
    batch, views = (int(value) for value in mask.shape[:2])
    linear = y * width + x
    sampled_depth = (
        target_depth.float()
        .reshape(batch, views, -1)
        .gather(
            2,
            linear.reshape(batch, views, -1),
        )
        .reshape_as(depth)
    )
    allowed_depth_error = TEMPORAL_DEPTH_ABS_EPS + depth_tolerance * sampled_depth.abs()
    valid = (
        valid
        & torch.isfinite(sampled_depth)
        & (sampled_depth > 1.0e-6)
        & ((depth.float() - sampled_depth).abs() <= allowed_depth_error)
    )
    plane_offset = torch.arange(
        batch * views, device=mask.device, dtype=torch.long
    ).reshape(batch, views, 1, 1) * (height * width)
    indices = (plane_offset + linear)[valid]
    result = torch.zeros_like(mask)
    result.reshape(-1).scatter_(0, indices, True)
    return result


def _propagate_adjacent_motion_evidence(
    motion_mask: torch.Tensor,
    adjacent_points: torch.Tensor,
    available_fwd: torch.Tensor,
    available_bwd: torch.Tensor,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    source_depth: torch.Tensor,
    image_size: tuple[int, int],
    *,
    depth_tolerance: float,
) -> torch.Tensor:
    propagated = torch.zeros_like(motion_mask)
    propagated[:, 1:] = _splat_projected_mask(
        motion_mask[:, :-1] & available_fwd[:, :-1],
        adjacent_points[:, :-1, 0],
        source_intr[:, 1:],
        source_c2w[:, 1:],
        source_depth[:, 1:],
        image_size,
        depth_tolerance=depth_tolerance,
    )
    propagated[:, :-1] |= _splat_projected_mask(
        motion_mask[:, 1:] & available_bwd[:, 1:],
        adjacent_points[:, 1:, 1],
        source_intr[:, :-1],
        source_c2w[:, :-1],
        source_depth[:, :-1],
        image_size,
        depth_tolerance=depth_tolerance,
    )
    return propagated


def _adjacent_targets(
    source_times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    intervals = source_times[:, 1:] - source_times[:, :-1]
    forward_times = source_times.clone()
    backward_times = source_times.clone()
    forward_times[:, :-1] = source_times[:, 1:]
    backward_times[:, 1:] = source_times[:, :-1]
    available_fwd = torch.zeros_like(source_times, dtype=torch.bool)
    available_bwd = torch.zeros_like(source_times, dtype=torch.bool)
    available_fwd[:, :-1] = intervals > 0
    available_bwd[:, 1:] = intervals > 0
    return (
        torch.stack((forward_times, backward_times), dim=-1),
        available_fwd,
        available_bwd,
    )


def _projected_keyframe_displacement(
    source_pixels: torch.Tensor,
    adjacent_points: torch.Tensor,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    image_size: tuple[int, int],
    *,
    available_fwd: torch.Tensor,
    available_bwd: torch.Tensor,
    frame_gap_fwd: torch.Tensor,
    frame_gap_bwd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fwd_pixels, fwd_camera_depth = _project_source_pixels(
        adjacent_points[:, :, 0],
        source_intr,
        source_c2w,
        image_size,
    )
    bwd_pixels, bwd_camera_depth = _project_source_pixels(
        adjacent_points[:, :, 1],
        source_intr,
        source_c2w,
        image_size,
    )
    fwd = (fwd_pixels - source_pixels).norm(dim=-1)
    bwd = (bwd_pixels - source_pixels).norm(dim=-1)
    crossing = float(max(image_size))
    fwd = torch.where(
        available_fwd & (fwd_camera_depth <= 1.0e-6),
        crossing * frame_gap_fwd[:, :, None, None],
        fwd,
    )
    bwd = torch.where(
        available_bwd & (bwd_camera_depth <= 1.0e-6),
        crossing * frame_gap_bwd[:, :, None, None],
        bwd,
    )
    return fwd, bwd


@torch.no_grad()
def select_motion(
    source_anchor: torch.Tensor,
    source_valid: torch.Tensor,
    source_depth: torch.Tensor,
    source_times: torch.Tensor,
    fps: torch.Tensor,
    *,
    transport_backend: TransportBackend,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    image_size: tuple[int, int],
    threshold: float,
    motion_mask_dilation_radius: int = 0,
    temporal_motion_propagation: bool = True,
    temporal_motion_depth_tolerance: float = 0.05,
    source_motion_mask: torch.Tensor | None = None,
) -> MotionSelection:
    _validate_classification_inputs(
        source_anchor,
        source_valid,
        source_depth,
        source_intr,
        source_c2w,
        source_times,
        image_size,
    )
    if source_motion_mask is not None:
        if (
            source_motion_mask.shape != source_valid.shape
            or source_motion_mask.dtype != torch.bool
        ):
            raise ValueError(
                "source_motion_mask must be bool [B,K,H,W] on the source geometry grid"
            )
        if source_motion_mask.device != source_anchor.device:
            raise ValueError("source_motion_mask and geometry must share one device")
        if isinstance(transport_backend, IdentityTransportBackend) and bool(
            source_motion_mask.any()
        ):
            raise ValueError(
                "motion-selected elements require a motion transport backend"
            )
    batch = int(source_anchor.shape[0])
    if tuple(fps.shape) != (batch,):
        raise ValueError(f"fps must be [B]={batch}, got {tuple(fps.shape)}")
    if not torch.is_floating_point(fps):
        raise TypeError("fps must be floating point")
    if fps.device != source_anchor.device:
        raise ValueError("fps and source points must share one device")
    if not bool(torch.isfinite(fps).all()) or not bool((fps > 0).all()):
        raise ValueError("fps must contain finite positive values")
    threshold = float(threshold)
    if not 0.0 <= threshold < float("inf"):
        raise ValueError("motion threshold must be finite and >= 0")
    if isinstance(motion_mask_dilation_radius, bool) or not isinstance(
        motion_mask_dilation_radius, int
    ):
        raise TypeError("motion mask dilation radius must be an integer")
    if motion_mask_dilation_radius < 0:
        raise ValueError("motion mask dilation radius must be >= 0")
    if not isinstance(temporal_motion_propagation, bool):
        raise TypeError("temporal motion propagation must be bool")
    temporal_motion_depth_tolerance = float(temporal_motion_depth_tolerance)
    if not 0.0 <= temporal_motion_depth_tolerance < float("inf"):
        raise ValueError("temporal motion depth tolerance must be finite and >= 0")

    source_pixels, source_camera_depth = _project_source_pixels(
        source_anchor,
        source_intr,
        source_c2w,
        image_size,
    )
    base_valid = (
        source_valid
        & (source_depth > 0)
        & torch.isfinite(source_pixels).all(dim=-1)
        & torch.isfinite(source_camera_depth)
        & (source_camera_depth > 1.0e-6)
    )
    if (
        isinstance(transport_backend, IdentityTransportBackend)
        or int(source_anchor.shape[1]) == 1
    ):
        unavailable = torch.zeros_like(base_valid)
        return MotionSelection(
            magnitude=torch.zeros_like(source_depth, dtype=source_anchor.dtype),
            anchor_mask=base_valid
            if source_motion_mask is None
            else base_valid & ~source_motion_mask,
            motion_mask=unavailable
            if source_motion_mask is None
            else base_valid & source_motion_mask,
            unresolved_mask=unavailable,
            available_fwd=unavailable,
            available_bwd=unavailable,
        )
    adjacent_times, time_available_fwd, time_available_bwd = _adjacent_targets(
        source_times
    )
    adjacent = query_transport(
        transport_backend,
        source_anchor,
        source_times,
        adjacent_times,
    )
    available_fwd = (
        base_valid & time_available_fwd[:, :, None, None] & adjacent.valid[:, :, 0]
    )
    available_bwd = (
        base_valid & time_available_bwd[:, :, None, None] & adjacent.valid[:, :, 1]
    )
    interval_frames = (
        source_times[:, 1:].float() - source_times[:, :-1].float()
    ) * fps.float()[:, None]
    interval_frames = torch.where(
        interval_frames > 0,
        interval_frames,
        torch.ones_like(interval_frames),
    )
    frame_gap_fwd = torch.ones_like(source_times, dtype=torch.float32)
    frame_gap_bwd = torch.ones_like(source_times, dtype=torch.float32)
    frame_gap_fwd[:, :-1] = interval_frames
    frame_gap_bwd[:, 1:] = interval_frames
    displacement_fwd, displacement_bwd = _projected_keyframe_displacement(
        source_pixels,
        adjacent.points,
        source_intr,
        source_c2w,
        image_size,
        available_fwd=available_fwd,
        available_bwd=available_bwd,
        frame_gap_fwd=frame_gap_fwd,
        frame_gap_bwd=frame_gap_bwd,
    )
    magnitude_fwd = torch.where(
        available_fwd,
        displacement_fwd / frame_gap_fwd[:, :, None, None],
        torch.zeros_like(displacement_fwd),
    )
    magnitude_bwd = torch.where(
        available_bwd,
        displacement_bwd / frame_gap_bwd[:, :, None, None],
        torch.zeros_like(displacement_bwd),
    )
    magnitude = torch.maximum(magnitude_fwd, magnitude_bwd).to(
        dtype=source_anchor.dtype
    )
    observed = available_fwd | available_bwd
    if source_motion_mask is not None:
        return MotionSelection(
            magnitude=magnitude,
            anchor_mask=observed & ~source_motion_mask,
            motion_mask=observed & source_motion_mask,
            unresolved_mask=base_valid & ~observed,
            available_fwd=available_fwd,
            available_bwd=available_bwd,
        )
    motion_mask = (
        available_fwd & (displacement_fwd > threshold * frame_gap_fwd[:, :, None, None])
    ) | (
        available_bwd & (displacement_bwd > threshold * frame_gap_bwd[:, :, None, None])
    )
    anchor_mask = observed & ~motion_mask
    unresolved_mask = base_valid & ~observed
    expandable_anchor = anchor_mask
    if temporal_motion_propagation:
        temporal_motion = _propagate_adjacent_motion_evidence(
            motion_mask,
            adjacent.points,
            available_fwd,
            available_bwd,
            source_intr,
            source_c2w,
            source_depth,
            image_size,
            depth_tolerance=temporal_motion_depth_tolerance,
        )
        motion_mask = motion_mask | (temporal_motion & expandable_anchor)
    dilated = _dilate_motion_mask(
        motion_mask,
        motion_mask_dilation_radius,
    )
    motion_mask = motion_mask | (dilated & expandable_anchor)
    anchor_mask = anchor_mask & ~motion_mask
    return MotionSelection(
        magnitude=magnitude,
        anchor_mask=anchor_mask,
        motion_mask=motion_mask,
        unresolved_mask=unresolved_mask,
        available_fwd=available_fwd,
        available_bwd=available_bwd,
    )


@torch.no_grad()
def propagate_classified_anchors(
    source_anchor: torch.Tensor,
    source_valid: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor,
    *,
    target_time_direction: torch.Tensor,
    transport_backend: TransportBackend,
    classification: MotionSelection,
) -> PropagationResult:
    _validate_propagation_inputs(
        source_anchor,
        source_valid,
        target_times,
        target_time_direction,
    )
    batch, views = (int(value) for value in source_anchor.shape[:2])
    _validate_source_times(
        source_times,
        batch=batch,
        views=views,
        source_anchor=source_anchor,
    )
    _validate_motion_selection(
        classification,
        source_anchor=source_anchor,
        source_valid=source_valid,
    )
    motion_mask = classification.motion_mask

    queries = int(target_times.shape[1])
    motion_batch = motion_mask.flatten(1).any(dim=1)
    bracketing_source = _bracketing_source_mask(
        source_times,
        target_times,
        motion_batch,
    )
    expanded_target_times = target_times[:, None].expand(batch, views, queries)
    transported = query_transport(
        transport_backend,
        source_anchor,
        source_times,
        expanded_target_times,
    )
    points = torch.where(
        motion_mask[:, :, None, :, :, None],
        transported.points,
        source_anchor[:, :, None],
    )

    signed_dt = (target_times[:, None, :] - source_times[:, :, None]).to(
        dtype=source_anchor.dtype
    )
    motion_scalar = motion_mask[:, :, None]
    motion_valid = (
        transported.valid & motion_scalar & bracketing_source[:, :, :, None, None]
    )
    valid = torch.where(
        motion_scalar,
        motion_valid,
        classification.anchor_mask[:, :, None],
    )
    return PropagationResult(
        points=points,
        valid=valid,
        motion_mask=motion_mask,
        signed_dt=signed_dt,
        time_direction=target_time_direction,
    )


__all__ = [
    "MotionSelection",
    "PropagationResult",
    "infer_target_time_directions",
    "propagate_classified_anchors",
    "select_motion",
]
