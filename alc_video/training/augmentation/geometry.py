from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SourceGeometry:
    positions: torch.Tensor
    valid: torch.Tensor
    depth: torch.Tensor


_GEOMETRY_EPS = 1.0e-6


@dataclass(frozen=True, slots=True)
class VirtualTrajectory:
    intr: torch.Tensor
    c2w: torch.Tensor


@dataclass(frozen=True, slots=True)
class VisibilityCullingResult:
    trajectory: VirtualTrajectory
    valid: torch.Tensor
    metrics: dict[str, float | int]

    @property
    def all_empty(self) -> bool:
        return not bool(self.valid.any())


@dataclass(frozen=True, slots=True)
class _VisibilityRaster:
    projected_valid: torch.Tensor
    zbuffer: torch.Tensor
    winner: torch.Tensor
    output_valid: torch.Tensor


def _normalize(value: torch.Tensor, *, name: str) -> torch.Tensor:
    norm = value.float().norm(dim=-1, keepdim=True)
    if bool((norm <= _GEOMETRY_EPS).any()) or not bool(torch.isfinite(norm).all()):
        raise ValueError(f"virtual trajectory {name} is degenerate")
    return value.float() / norm


def _validate_geometry(geometry: SourceGeometry) -> tuple[int, int, int, int]:
    if not isinstance(geometry, SourceGeometry):
        raise TypeError("geometry must be SourceGeometry")
    if geometry.positions.ndim != 5 or int(geometry.positions.shape[-1]) != 3:
        raise ValueError("geometry positions must be [B,V,H,W,3]")
    batch, views, height, width = (int(value) for value in geometry.positions.shape[:4])
    if min(batch, views, height, width) <= 0:
        raise ValueError("geometry dimensions must be positive")
    if tuple(geometry.valid.shape) != (batch, views, height, width):
        raise ValueError("geometry valid must match position grid")
    if tuple(geometry.depth.shape) != (batch, views, height, width):
        raise ValueError("geometry depth must match position grid")
    if geometry.valid.dtype != torch.bool:
        raise TypeError("geometry valid must be bool")
    if not torch.is_floating_point(geometry.positions) or not torch.is_floating_point(
        geometry.depth
    ):
        raise TypeError("geometry positions and depth must be floating point")
    if any(
        value.device != geometry.positions.device
        for value in (geometry.valid, geometry.depth)
    ):
        raise ValueError("geometry tensors must share one device")
    if not bool(torch.isfinite(geometry.positions).all()) or not bool(
        torch.isfinite(geometry.depth).all()
    ):
        raise ValueError("geometry contains non-finite values")
    return batch, views, height, width


def _validate_camera(
    trajectory: VirtualTrajectory,
    *,
    batch: int,
    views: int,
    device: torch.device,
) -> None:
    if not isinstance(trajectory, VirtualTrajectory):
        raise TypeError("trajectory must be VirtualTrajectory")
    if tuple(trajectory.intr.shape) != (batch, views, 3, 3):
        raise ValueError("virtual intr must be [B,V,3,3]")
    if tuple(trajectory.c2w.shape) != (batch, views, 4, 4):
        raise ValueError("virtual c2w must be [B,V,4,4]")
    camera = (trajectory.intr, trajectory.c2w)
    if not all(torch.is_floating_point(value) for value in camera):
        raise TypeError("virtual camera tensors must be floating point")
    if not all(value.device == device for value in camera):
        raise ValueError("virtual camera and geometry must share one device")
    if not all(bool(torch.isfinite(value).all()) for value in camera):
        raise ValueError("virtual camera contains non-finite values")


@torch.no_grad()
def lateral_look_at_v1(
    geometry: SourceGeometry,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    *,
    side: str,
    translation: float,
) -> VirtualTrajectory:
    batch, views, _height, _width = _validate_geometry(geometry)
    if views < 2:
        raise ValueError("virtual trajectory requires at least two source views")
    source = VirtualTrajectory(intr=source_intr, c2w=source_c2w)
    _validate_camera(
        source,
        batch=batch,
        views=views,
        device=geometry.positions.device,
    )
    if side not in {"left", "right"}:
        raise ValueError("virtual trajectory side must be left or right")
    if not 0.0 < float(translation) < float("inf"):
        raise ValueError("virtual trajectory translation must be finite and > 0")

    valid = geometry.valid & (geometry.depth > _GEOMETRY_EPS)
    counts = valid.sum(dim=(2, 3))
    if bool((counts == 0).any()):
        raise ValueError("virtual trajectory requires valid depth in every source view")
    scene_depth = (geometry.depth.float() * valid).sum(dim=(2, 3)) / counts.float()
    positions = source_c2w.float()[..., :3, 3]
    rotations = source_c2w.float()[..., :3, :3]
    observation_centers = positions + scene_depth[..., None] * rotations[..., 2]
    movement = _normalize(
        positions[:, -1] - positions[:, 0],
        name="global movement",
    )
    world_down = torch.tensor(
        [0.0, 1.0, 0.0],
        device=positions.device,
        dtype=torch.float32,
    ).expand(batch, -1)
    lateral = (
        torch.cross(movement, world_down, dim=-1)
        if side == "left"
        else torch.cross(world_down, movement, dim=-1)
    )
    lateral = _normalize(lateral, name="lateral direction")
    virtual_positions = positions + lateral[:, None] * float(translation)
    virtual_forward = _normalize(
        observation_centers - virtual_positions,
        name="look-at direction",
    )
    virtual_down = world_down[:, None].expand(batch, views, -1)
    virtual_right = _normalize(
        torch.cross(virtual_down, virtual_forward, dim=-1),
        name="camera right axis",
    )
    virtual_down = torch.cross(virtual_forward, virtual_right, dim=-1)

    virtual_c2w = torch.zeros_like(source_c2w)
    virtual_c2w[..., :3, :3] = torch.stack(
        (virtual_right, virtual_down, virtual_forward),
        dim=-1,
    ).to(dtype=virtual_c2w.dtype)
    virtual_c2w[..., :3, 3] = virtual_positions.to(dtype=virtual_c2w.dtype)
    virtual_c2w[..., 3, 3] = 1
    if not bool(torch.isfinite(virtual_c2w).all()):
        raise ValueError("virtual trajectory contains non-finite values")
    return VirtualTrajectory(
        intr=source_intr.clone(),
        c2w=virtual_c2w,
    )


def _rotate_about_axis(
    value: torch.Tensor,
    axis: torch.Tensor,
    angle: float,
) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        value.float() * cosine
        + torch.cross(axis, value.float(), dim=-1) * sine
        + axis * (axis * value.float()).sum(dim=-1, keepdim=True) * (1.0 - cosine)
    )


@torch.no_grad()
def yaw_pan_v1(
    geometry: SourceGeometry,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    *,
    direction: str,
    degrees: float,
) -> VirtualTrajectory:
    batch, views, _height, _width = _validate_geometry(geometry)
    source = VirtualTrajectory(intr=source_intr, c2w=source_c2w)
    _validate_camera(
        source,
        batch=batch,
        views=views,
        device=geometry.positions.device,
    )
    if direction not in {"left", "right"}:
        raise ValueError("yaw pan direction must be left or right")
    if not 0.0 <= float(degrees) < 90.0 or not math.isfinite(float(degrees)):
        raise ValueError("yaw pan degrees must be finite and in [0,90)")
    signed_radians = math.radians(float(degrees)) * (
        -1.0 if direction == "left" else 1.0
    )
    world_down = torch.tensor(
        [0.0, 1.0, 0.0],
        device=source_c2w.device,
        dtype=torch.float32,
    ).expand(batch, views, -1)
    basis = source_c2w.float()[..., :3, :3].transpose(-1, -2)
    rotated_basis = _rotate_about_axis(
        basis,
        world_down[..., None, :],
        signed_radians,
    )
    virtual_c2w = source_c2w.clone()
    virtual_c2w[..., :3, :3] = rotated_basis.transpose(-1, -2).to(
        dtype=source_c2w.dtype
    )
    return VirtualTrajectory(intr=source_intr.clone(), c2w=virtual_c2w)


@torch.no_grad()
def pitch_pan_v1(
    geometry: SourceGeometry,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    *,
    direction: str,
    degrees: float,
) -> VirtualTrajectory:
    batch, views, _height, _width = _validate_geometry(geometry)
    source = VirtualTrajectory(intr=source_intr, c2w=source_c2w)
    _validate_camera(
        source,
        batch=batch,
        views=views,
        device=geometry.positions.device,
    )
    if direction not in {"up", "down"}:
        raise ValueError("pitch pan direction must be up or down")
    if not 0.0 <= float(degrees) < 90.0 or not math.isfinite(float(degrees)):
        raise ValueError("pitch pan degrees must be finite and in [0,90)")
    signed_radians = math.radians(float(degrees)) * (1.0 if direction == "up" else -1.0)
    basis = source_c2w.float()[..., :3, :3].transpose(-1, -2)
    camera_right = source_c2w.float()[..., :3, 0]
    rotated_basis = _rotate_about_axis(
        basis,
        camera_right[..., None, :],
        signed_radians,
    )
    virtual_c2w = source_c2w.clone()
    virtual_c2w[..., :3, :3] = rotated_basis.transpose(-1, -2).to(
        dtype=source_c2w.dtype
    )
    return VirtualTrajectory(intr=source_intr.clone(), c2w=virtual_c2w)


@torch.no_grad()
def focal_probe_v1(
    geometry: SourceGeometry,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    *,
    scale: float,
) -> VirtualTrajectory:
    batch, views, _height, _width = _validate_geometry(geometry)
    source = VirtualTrajectory(intr=source_intr, c2w=source_c2w)
    _validate_camera(
        source,
        batch=batch,
        views=views,
        device=geometry.positions.device,
    )
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("focal probe scale must be finite and > 0")
    virtual_intr = source_intr.clone()
    virtual_intr[..., 0, 0] *= float(scale)
    virtual_intr[..., 1, 1] *= float(scale)
    return VirtualTrajectory(intr=virtual_intr, c2w=source_c2w.clone())


@torch.no_grad()
def build_visibility_trajectory(
    geometry: SourceGeometry,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    *,
    strategy: str,
    direction: str | None = None,
    translation: float | None = None,
    degrees: float | None = None,
    focal_scale: float | None = None,
) -> VirtualTrajectory:
    if strategy == "lateral_look_at_v1":
        if direction is None or translation is None:
            raise ValueError("lateral visibility requires direction and translation")
        if degrees is not None or focal_scale is not None:
            raise ValueError("lateral visibility received unrelated parameters")
        return lateral_look_at_v1(
            geometry,
            source_intr,
            source_c2w,
            side=direction,
            translation=translation,
        )
    if strategy == "yaw_pan_v1":
        if direction is None or degrees is None:
            raise ValueError("pan visibility requires direction and degrees")
        if translation is not None or focal_scale is not None:
            raise ValueError("pan visibility received unrelated parameters")
        return yaw_pan_v1(
            geometry,
            source_intr,
            source_c2w,
            direction=direction,
            degrees=degrees,
        )
    if strategy == "pitch_pan_v1":
        if direction is None or degrees is None:
            raise ValueError("pitch visibility requires direction and degrees")
        if translation is not None or focal_scale is not None:
            raise ValueError("pitch visibility received unrelated parameters")
        return pitch_pan_v1(
            geometry,
            source_intr,
            source_c2w,
            direction=direction,
            degrees=degrees,
        )
    if strategy in {"zoom_in_v1", "zoom_out_v1"}:
        if focal_scale is None:
            raise ValueError(f"{strategy} visibility requires focal_scale")
        if direction is not None or translation is not None or degrees is not None:
            raise ValueError(f"{strategy} visibility received unrelated parameters")
        if strategy == "zoom_in_v1" and not float(focal_scale) > 1.0:
            raise ValueError("zoom_in visibility probe focal_scale must be > 1")
        if strategy == "zoom_out_v1" and not float(focal_scale) < 1.0:
            raise ValueError("zoom_out visibility probe focal_scale must be < 1")
        return focal_probe_v1(
            geometry,
            source_intr,
            source_c2w,
            scale=focal_scale,
        )
    raise ValueError(f"unsupported visibility strategy: {strategy!r}")


def _project_points(
    points: torch.Tensor,
    intr: torch.Tensor,
    c2w: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_size
    rotation = c2w.float()[..., :3, :3]
    translation = c2w.float()[..., :3, 3]
    centered = points.float() - translation[:, :, None, None]
    camera = torch.einsum("bvhwc,bvcd->bvhwd", centered, rotation)
    depth = camera[..., 2]
    safe_depth = depth.clamp_min(_GEOMETRY_EPS)
    intr = intr.float()
    u = intr[..., 0, 0, None, None] * float(width) * camera[..., 0] / safe_depth + intr[
        ..., 0, 2, None, None
    ] * float(width)
    v = intr[..., 1, 1, None, None] * float(height) * camera[
        ..., 1
    ] / safe_depth + intr[..., 1, 2, None, None] * float(height)
    return torch.stack((u, v), dim=-1), depth


def _rasterize_visibility(
    geometry: SourceGeometry,
    trajectory: VirtualTrajectory,
) -> _VisibilityRaster:
    batch, views, height, width = _validate_geometry(geometry)
    _validate_camera(
        trajectory,
        batch=batch,
        views=views,
        device=geometry.positions.device,
    )
    rows = batch * views
    pixels_per_view = height * width
    pixels, virtual_depth = _project_points(
        geometry.positions,
        trajectory.intr,
        trajectory.c2w,
        image_size=(height, width),
    )
    rounded_x = pixels[..., 0].round()
    rounded_y = pixels[..., 1].round()
    source_valid = geometry.valid & (geometry.depth > _GEOMETRY_EPS)
    projected_valid = (
        source_valid
        & torch.isfinite(pixels).all(dim=-1)
        & torch.isfinite(virtual_depth)
        & (virtual_depth > _GEOMETRY_EPS)
        & (rounded_x >= 0)
        & (rounded_x < width)
        & (rounded_y >= 0)
        & (rounded_y < height)
    )
    target_x = rounded_x.clamp(0, width - 1).long()
    target_y = rounded_y.clamp(0, height - 1).long()
    target_cell = (target_y * width + target_x).reshape(rows, pixels_per_view)
    flat_depth = virtual_depth.reshape(rows, pixels_per_view).float()
    flat_projected = projected_valid.reshape(rows, pixels_per_view)
    zbuffer = torch.full(
        (rows, pixels_per_view),
        float("inf"),
        device=geometry.positions.device,
        dtype=torch.float32,
    )
    zbuffer.scatter_reduce_(
        1,
        target_cell,
        torch.where(
            flat_projected, flat_depth, torch.full_like(flat_depth, float("inf"))
        ),
        reduce="amin",
        include_self=True,
    )
    nearest = zbuffer.gather(1, target_cell)
    nearest_candidate = flat_projected & (flat_depth == nearest)
    row_major_token = (
        torch.arange(
            pixels_per_view,
            device=geometry.positions.device,
            dtype=torch.long,
        )
        .reshape(1, pixels_per_view)
        .expand(rows, -1)
    )
    winner = torch.full(
        (rows, pixels_per_view),
        pixels_per_view,
        device=geometry.positions.device,
        dtype=torch.long,
    )
    winner.scatter_reduce_(
        1,
        target_cell,
        torch.where(
            nearest_candidate,
            row_major_token,
            torch.full_like(row_major_token, pixels_per_view),
        ),
        reduce="amin",
        include_self=True,
    )
    output_valid = winner < pixels_per_view
    return _VisibilityRaster(
        projected_valid=projected_valid,
        zbuffer=zbuffer,
        winner=winner,
        output_valid=output_valid,
    )


@torch.no_grad()
def build_visibility_culling(
    geometry: SourceGeometry,
    trajectory: VirtualTrajectory,
) -> VisibilityCullingResult:
    batch, views, height, width = _validate_geometry(geometry)
    raster = _rasterize_visibility(geometry, trajectory)
    pixels_per_view = height * width
    rows = batch * views
    keep = torch.zeros(
        (rows, pixels_per_view),
        dtype=torch.bool,
        device=geometry.positions.device,
    )
    row_indices = torch.arange(rows, device=geometry.positions.device)[:, None].expand(
        rows, pixels_per_view
    )
    keep[
        row_indices[raster.output_valid],
        raster.winner[raster.output_valid],
    ] = True
    source_valid = geometry.valid & (geometry.depth > _GEOMETRY_EPS)
    kept_valid = source_valid & keep.reshape(batch, views, height, width)
    valid_before = int(source_valid.sum().item())
    projected_candidates = int(raster.projected_valid.sum().item())
    valid_after = int(kept_valid.sum().item())
    metrics: dict[str, float | int] = {
        "valid_before": valid_before,
        "projected_candidates": projected_candidates,
        "valid_after": valid_after,
        "removed_count": valid_before - valid_after,
        "out_of_frame_count": valid_before - projected_candidates,
        "collision_count": projected_candidates - valid_after,
        "empty_view_count": int(
            (~kept_valid.reshape(batch, views, -1).any(dim=-1)).sum().item()
        ),
        "all_empty": int(valid_after == 0),
        "retained_fraction": valid_after / max(valid_before, 1),
    }
    return VisibilityCullingResult(
        trajectory=trajectory,
        valid=kept_valid[..., None],
        metrics=metrics,
    )


@torch.no_grad()
def apply_visibility_culling(
    geometry: SourceGeometry,
    source_intr: torch.Tensor,
    source_c2w: torch.Tensor,
    *,
    strategy: str,
    direction: str | None = None,
    translation: float | None = None,
    degrees: float | None = None,
    focal_scale: float | None = None,
) -> VisibilityCullingResult:
    trajectory = build_visibility_trajectory(
        geometry,
        source_intr,
        source_c2w,
        strategy=strategy,
        direction=direction,
        translation=translation,
        degrees=degrees,
        focal_scale=focal_scale,
    )
    return build_visibility_culling(geometry, trajectory)


__all__ = [
    "VirtualTrajectory",
    "VisibilityCullingResult",
    "apply_visibility_culling",
    "build_visibility_culling",
    "build_visibility_trajectory",
    "focal_probe_v1",
    "lateral_look_at_v1",
    "pitch_pan_v1",
    "yaw_pan_v1",
]
