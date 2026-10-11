from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from latent_cache.propagation.motion import PropagationResult

DEPTH_EPS = 1.0e-6


@dataclass(frozen=True)
class GeometryCandidates:
    batch_index: torch.Tensor
    query_index: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    z: torch.Tensor
    signed_dt: torch.Tensor
    time_direction: torch.Tensor
    motion_mask: torch.Tensor
    is_primary_time: torch.Tensor
    pose_dist: torch.Tensor
    source_view: torch.Tensor
    source_token: torch.Tensor
    anchor_id: torch.Tensor
    batch_size: int
    source_view_count: int
    query_count: int
    image_size: tuple[int, int]
    source_grid_shape: tuple[int, int]

    def __len__(self) -> int:
        return int(self.u.numel())


def _validate_propagation(
    propagation: PropagationResult,
) -> tuple[int, int, int, int, int]:
    points = propagation.points
    if points.ndim != 6 or int(points.shape[-1]) != 3:
        raise ValueError(
            f"propagation.points must be [B,V,T,H,W,3], got {tuple(points.shape)}"
        )
    if propagation.valid.shape != points.shape[:-1]:
        raise ValueError("propagation.valid must be [B,V,T,H,W]")
    if propagation.valid.dtype != torch.bool:
        raise TypeError("propagation.valid must be bool")
    batch, views, queries, height, width = (int(v) for v in points.shape[:5])
    source_shape = (batch, views, height, width)
    if tuple(propagation.motion_mask.shape) != source_shape:
        raise ValueError("propagation.motion_mask must be [B,V,H,W]")
    if propagation.motion_mask.dtype != torch.bool:
        raise TypeError("propagation.motion_mask must be bool")
    metadata_shape = (batch, views, queries)
    if tuple(propagation.signed_dt.shape) != metadata_shape:
        raise ValueError("propagation.signed_dt must be [B,V,T]")
    if not torch.is_floating_point(propagation.signed_dt):
        raise TypeError("propagation.signed_dt must be floating point")
    if tuple(propagation.time_direction.shape) != (batch, queries):
        raise ValueError("propagation.time_direction must be [B,T]")
    if propagation.time_direction.dtype != torch.int8:
        raise TypeError("propagation.time_direction must be int8")
    if not bool(
        ((propagation.time_direction == 1) | (propagation.time_direction == -1)).all()
    ):
        raise ValueError("propagation.time_direction values must be -1 or 1")
    return batch, views, queries, height, width


def _empty_candidates(
    *,
    device: torch.device,
    batch_size: int,
    source_view_count: int,
    query_count: int,
    image_size: tuple[int, int],
    source_grid_shape: tuple[int, int],
) -> GeometryCandidates:
    empty_float = torch.empty((0,), device=device, dtype=torch.float32)
    empty_long = torch.empty((0,), device=device, dtype=torch.long)
    return GeometryCandidates(
        batch_index=empty_long,
        query_index=empty_long.clone(),
        u=empty_float,
        v=empty_float.clone(),
        z=empty_float.clone(),
        signed_dt=empty_float.clone(),
        time_direction=torch.empty((0,), device=device, dtype=torch.int8),
        motion_mask=torch.empty((0,), device=device, dtype=torch.bool),
        is_primary_time=torch.empty((0,), device=device, dtype=torch.bool),
        pose_dist=empty_float.clone(),
        source_view=empty_long.clone(),
        source_token=empty_long.clone(),
        anchor_id=empty_long.clone(),
        batch_size=batch_size,
        source_view_count=source_view_count,
        query_count=query_count,
        image_size=image_size,
        source_grid_shape=source_grid_shape,
    )


@torch.no_grad()
def build_candidates(
    propagation: PropagationResult,
    source_poses: torch.Tensor,
    target_poses: torch.Tensor,
    target_intrs: torch.Tensor,
    image_size: tuple[int, int],
    source_grid_shape: tuple[int, int],
    *,
    source_view_valid: torch.Tensor | None = None,
    source_token_valid: torch.Tensor | None = None,
    depth_eps: float = DEPTH_EPS,
) -> GeometryCandidates:
    batch, views, queries, height, width = _validate_propagation(propagation)
    image_size = tuple(int(v) for v in image_size)
    source_grid_shape = tuple(int(v) for v in source_grid_shape)
    if image_size != (height, width):
        raise ValueError(
            f"propagation pixel grid {(height, width)} must match image_size {image_size}"
        )
    latent_h, latent_w = source_grid_shape
    if latent_h <= 0 or latent_w <= 0:
        raise ValueError("source_grid_shape values must be > 0")
    if height % latent_h != 0 or width % latent_w != 0:
        raise ValueError(
            f"image_size {image_size} must be divisible by source grid "
            f"{source_grid_shape}"
        )
    if tuple(source_poses.shape) != (batch, views, 4, 4):
        raise ValueError(
            f"source_poses must be [B,V,4,4], got {tuple(source_poses.shape)}"
        )
    if tuple(target_poses.shape) != (batch, queries, 4, 4):
        raise ValueError(
            f"target_poses must be [B,T,4,4], got {tuple(target_poses.shape)}"
        )
    if tuple(target_intrs.shape) != (batch, queries, 3, 3):
        raise ValueError(
            f"target_intrs must be [B,T,3,3], got {tuple(target_intrs.shape)}"
        )
    if depth_eps <= 0:
        raise ValueError("depth_eps must be > 0")
    device = propagation.points.device
    if source_view_valid is not None:
        if (
            not isinstance(source_view_valid, torch.Tensor)
            or tuple(source_view_valid.shape) != (batch, views)
            or source_view_valid.dtype != torch.bool
        ):
            raise TypeError("source_view_valid must be bool [B,V]")
        if source_view_valid.device != device:
            raise ValueError("source_view_valid and propagation must share one device")
    if source_token_valid is not None:
        expected_token_valid = (batch, views, latent_h, latent_w)
        if (
            not isinstance(source_token_valid, torch.Tensor)
            or source_token_valid.dtype != torch.bool
            or tuple(int(value) for value in source_token_valid.shape)
            != expected_token_valid
            or source_token_valid.device != device
        ):
            raise ValueError(
                "source_token_valid must be bool "
                f"{expected_token_valid} on the propagation device"
            )
    source_poses = source_poses.to(device=device, dtype=torch.float32)
    target_poses = target_poses.to(device=device, dtype=torch.float32)
    target_intrs = target_intrs.to(device=device, dtype=torch.float32)
    source_centers = source_poses[..., :3, 3]
    if views > 1:
        baseline = (
            (source_centers[:, 1:] - source_centers[:, :-1])
            .norm(dim=-1)
            .mean(dim=1)
            .clamp_min(1.0e-6)
        )
    else:
        baseline = torch.ones((batch,), device=device, dtype=torch.float32)

    stride_y = height // latent_h
    stride_x = width // latent_w
    pixel_y = torch.arange(height, device=device, dtype=torch.long)
    pixel_x = torch.arange(width, device=device, dtype=torch.long)
    token_grid = (pixel_y[:, None] // stride_y) * latent_w + pixel_x[
        None, :
    ] // stride_x
    source_view_grid = torch.arange(views, device=device, dtype=torch.long)[
        :, None, None
    ].expand(-1, height, width)
    source_token_grid = source_view_grid * (latent_h * latent_w) + token_grid[None]
    anchor_grid = torch.arange(
        views * height * width,
        device=device,
        dtype=torch.long,
    ).reshape(views, height, width)
    dt_abs = propagation.signed_dt.float().abs()
    nearest_dt = dt_abs.amin(dim=1, keepdim=True)
    nearest_time = dt_abs <= nearest_dt + 1.0e-8
    past = propagation.signed_dt >= 0
    nearest_past_exists = (nearest_time & past).any(dim=1, keepdim=True)
    primary_time = nearest_time & (~nearest_past_exists | past)

    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "batch_index",
            "query_index",
            "u",
            "v",
            "z",
            "signed_dt",
            "time_direction",
            "motion_mask",
            "is_primary_time",
            "pose_dist",
            "source_view",
            "source_token",
            "anchor_id",
        )
    }
    for batch_idx in range(batch):
        pose_distance = (
            source_centers[batch_idx] - target_poses[batch_idx, :, None, :3, 3]
        ).norm(dim=-1) / baseline[batch_idx]
        for query_idx in range(queries):
            points = propagation.points[batch_idx, :, query_idx].float()
            target_pose = target_poses[batch_idx, query_idx]
            rotation = target_pose[:3, :3]
            translation = target_pose[:3, 3]
            camera = (points - translation) @ rotation
            depth = camera[..., 2]
            intr = target_intrs[batch_idx, query_idx]
            u_pixel = intr[0, 0] * float(width) * camera[..., 0] / depth.clamp_min(
                depth_eps
            ) + intr[0, 2] * float(width)
            v_pixel = intr[1, 1] * float(height) * camera[..., 1] / depth.clamp_min(
                depth_eps
            ) + intr[1, 2] * float(height)
            source_ray = points - source_centers[batch_idx, :, None, None]
            target_ray = points - translation
            source_norm = source_ray.norm(dim=-1)
            target_norm = target_ray.norm(dim=-1)
            view_cos = (
                (
                    F.normalize(source_ray, dim=-1, eps=depth_eps)
                    * F.normalize(target_ray, dim=-1, eps=depth_eps)
                )
                .sum(dim=-1)
                .clamp(-1.0, 1.0)
            )
            angle_cost = 1.0 - view_cos
            valid = (
                propagation.valid[batch_idx, :, query_idx]
                & torch.isfinite(points).all(dim=-1)
                & torch.isfinite(depth)
                & torch.isfinite(u_pixel)
                & torch.isfinite(v_pixel)
                & torch.isfinite(angle_cost)
                & (depth > depth_eps)
                & (source_norm > depth_eps)
                & (target_norm > depth_eps)
            )
            if source_view_valid is not None:
                valid &= source_view_valid[batch_idx, :, None, None]
            if source_token_valid is not None:
                expanded_token_valid = (
                    source_token_valid[batch_idx]
                    .reshape(views, latent_h * latent_w)
                    .gather(
                        1,
                        token_grid.reshape(1, height * width).expand(views, -1),
                    )
                    .reshape(views, height, width)
                )
                valid &= expanded_token_valid
            selected = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
            count = int(selected.numel())
            if count == 0:
                continue

            def take(value: torch.Tensor, selected=selected) -> torch.Tensor:
                return value.reshape(-1).index_select(0, selected)

            collected["batch_index"].append(
                torch.full((count,), batch_idx, device=device, dtype=torch.long)
            )
            collected["query_index"].append(
                torch.full((count,), query_idx, device=device, dtype=torch.long)
            )
            collected["u"].append(take(u_pixel).float() / float(width))
            collected["v"].append(take(v_pixel).float() / float(height))
            collected["z"].append(take(depth).float())
            collected["signed_dt"].append(
                take(
                    propagation.signed_dt[batch_idx, :, query_idx, None, None].expand(
                        -1, height, width
                    )
                ).float()
            )
            collected["time_direction"].append(
                propagation.time_direction[batch_idx, query_idx].expand(count)
            )
            collected["motion_mask"].append(
                take(propagation.motion_mask[batch_idx]).bool()
            )
            collected["is_primary_time"].append(
                take(
                    primary_time[batch_idx, :, query_idx, None, None].expand(
                        -1, height, width
                    )
                ).bool()
            )
            collected["pose_dist"].append(
                take(
                    pose_distance[query_idx, :, None, None].expand(-1, height, width)
                ).float()
            )
            collected["source_view"].append(take(source_view_grid))
            collected["source_token"].append(take(source_token_grid))
            collected["anchor_id"].append(take(anchor_grid))

    if not collected["u"]:
        return _empty_candidates(
            device=device,
            batch_size=batch,
            source_view_count=views,
            query_count=queries,
            image_size=image_size,
            source_grid_shape=source_grid_shape,
        )
    values = {name: torch.cat(parts) for name, parts in collected.items()}
    return GeometryCandidates(
        **values,
        batch_size=batch,
        source_view_count=views,
        query_count=queries,
        image_size=image_size,
        source_grid_shape=source_grid_shape,
    )


__all__ = ["GeometryCandidates", "build_candidates"]
