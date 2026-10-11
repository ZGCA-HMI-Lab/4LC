from __future__ import annotations

from dataclasses import dataclass

import torch

from latent_cache.geometry.candidates import GeometryCandidates, build_candidates
from latent_cache.propagation.motion import PropagationResult


@dataclass(frozen=True)
class TransportPlan:
    source_token: torch.Tensor
    source_view: torch.Tensor
    signed_dt: torch.Tensor
    score: torch.Tensor
    depth: torch.Tensor
    hit_mask: torch.Tensor
    candidate_motion_mask: torch.Tensor
    motion_mask: torch.Tensor
    source_grid_shape: tuple[int, int]
    target_grid_shape: tuple[int, int]
    candidate_count: int


@dataclass(frozen=True)
class ProjectionResult:
    hit_latent: torch.Tensor
    hit_mask: torch.Tensor
    score: torch.Tensor


def _validate_candidates(candidates: GeometryCandidates) -> int:
    count = len(candidates)
    fields = (
        candidates.batch_index,
        candidates.query_index,
        candidates.u,
        candidates.v,
        candidates.z,
        candidates.signed_dt,
        candidates.time_direction,
        candidates.motion_mask,
        candidates.is_primary_time,
        candidates.pose_dist,
        candidates.source_view,
        candidates.source_token,
        candidates.anchor_id,
    )
    if any(int(value.numel()) != count for value in fields):
        raise ValueError("all GeometryCandidates fields must have equal length")
    if candidates.motion_mask.dtype != torch.bool:
        raise TypeError("candidates.motion_mask must be bool")
    if candidates.time_direction.dtype != torch.int8:
        raise TypeError("candidates.time_direction must be int8")
    if count and not bool(
        ((candidates.time_direction == 1) | (candidates.time_direction == -1)).all()
    ):
        raise ValueError("candidates.time_direction values must be -1 or 1")
    if candidates.is_primary_time.dtype != torch.bool:
        raise TypeError("candidates.is_primary_time must be bool")
    if count and (
        int(candidates.batch_index.min()) < 0
        or int(candidates.batch_index.max()) >= candidates.batch_size
    ):
        raise ValueError("candidate batch_index is out of range")
    if count and (
        int(candidates.query_index.min()) < 0
        or int(candidates.query_index.max()) >= candidates.query_count
    ):
        raise ValueError("candidate query_index is out of range")
    return count


def _group_best_positions(
    group_ids: torch.Tensor,
    scores: torch.Tensor,
    group_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(group_ids.numel()) == 0:
        empty = torch.empty((0,), device=group_ids.device, dtype=torch.long)
        return empty, empty
    best_score = torch.full(
        (group_count,),
        float("-inf"),
        device=scores.device,
        dtype=torch.float32,
    )
    best_score.scatter_reduce_(
        0,
        group_ids,
        scores.float(),
        reduce="amax",
        include_self=True,
    )
    positions = torch.arange(
        int(group_ids.numel()),
        device=group_ids.device,
        dtype=torch.long,
    )
    sentinel = int(group_ids.numel())
    tied = torch.where(
        scores.float() == best_score.index_select(0, group_ids),
        positions,
        torch.full_like(positions, sentinel),
    )
    best_position = torch.full(
        (group_count,),
        sentinel,
        device=group_ids.device,
        dtype=torch.long,
    )
    best_position.scatter_reduce_(
        0,
        group_ids,
        tied,
        reduce="amin",
        include_self=True,
    )
    active = torch.nonzero(best_position < sentinel, as_tuple=False).flatten()
    return active, best_position.index_select(0, active)


def _select_integer_token_representatives(
    candidates: GeometryCandidates,
    target_grid_shape: tuple[int, int],
    *,
    depth_abs_eps: float,
    depth_rel_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_h, image_w = candidates.image_size
    source_h, source_w = candidates.source_grid_shape
    target_h, target_w = target_grid_shape
    if image_h % source_h or image_w % source_w:
        raise ValueError("image_size must be divisible by source_grid_shape")
    if image_h % target_h or image_w % target_w:
        raise ValueError("image_size must be divisible by target_grid_shape")

    source_stride_y = image_h // source_h
    source_stride_x = image_w // source_w
    target_stride_y = image_h // target_h
    target_stride_x = image_w // target_w
    anchor_pixel = candidates.anchor_id % (image_h * image_w)
    anchor_y = torch.div(anchor_pixel, image_w, rounding_mode="floor")
    anchor_x = anchor_pixel % image_w
    local_y = (anchor_y.float() % source_stride_y + 0.5) / float(source_stride_y)
    local_x = (anchor_x.float() % source_stride_x + 0.5) / float(source_stride_x)
    center_prior = 0.5 + 0.5 * torch.minimum(
        1.0 - (local_y - 0.5).abs() * 2.0,
        1.0 - (local_x - 0.5).abs() * 2.0,
    )

    u_pixel = candidates.u.float() * float(image_w)
    v_pixel = candidates.v.float() * float(image_h)
    x_continuous = (u_pixel - 0.5 * float(target_stride_x - 1)) / float(target_stride_x)
    y_continuous = (v_pixel - 0.5 * float(target_stride_y - 1)) / float(target_stride_y)
    x = torch.floor(x_continuous + 0.5).long()
    y = torch.floor(y_continuous + 0.5).long()
    weight_x = (1.0 - (x_continuous - x.float()).abs()).clamp(0.0, 1.0)
    weight_y = (1.0 - (y_continuous - y.float()).abs()).clamp(0.0, 1.0)
    score = (center_prior * weight_x * weight_y).float()
    keep = (
        (x >= 0)
        & (x < target_w)
        & (y >= 0)
        & (y < target_h)
        & torch.isfinite(score)
        & (score > 0)
    )
    selected = torch.nonzero(keep, as_tuple=False).flatten()
    score = score.index_select(0, selected)
    x = x.index_select(0, selected)
    y = y.index_select(0, selected)

    query_group = candidates.batch_index.index_select(
        0, selected
    ) * candidates.query_count + candidates.query_index.index_select(0, selected)
    token_span = candidates.source_view_count * source_h * source_w
    token_group = (
        query_group * token_span + candidates.source_token.index_select(0, selected)
    ) * 2 + candidates.motion_mask.index_select(0, selected).long()
    token_group_count = candidates.batch_size * candidates.query_count * token_span * 2
    depth = candidates.z.index_select(0, selected).float()
    nearest_depth = torch.full(
        (token_group_count,),
        float("inf"),
        device=depth.device,
        dtype=torch.float32,
    )
    nearest_depth.scatter_reduce_(
        0,
        token_group,
        depth,
        reduce="amin",
        include_self=True,
    )
    tolerance = (
        depth_abs_eps + depth_rel_eps * nearest_depth.index_select(0, token_group).abs()
    )
    near = depth <= nearest_depth.index_select(0, token_group) + tolerance
    selected = selected[near]
    score = score[near]
    token_group = token_group[near]
    x = x[near]
    y = y[near]
    _active, token_winner = _group_best_positions(
        token_group,
        score,
        token_group_count,
    )
    selected = selected.index_select(0, token_winner)
    score = score.index_select(0, token_winner)
    x = x.index_select(0, token_winner)
    y = y.index_select(0, token_winner)
    query_group = candidates.batch_index.index_select(
        0, selected
    ) * candidates.query_count + candidates.query_index.index_select(0, selected)
    cell_group = query_group * (target_h * target_w) + y * target_w + x
    return selected, cell_group, score


def _stage_winners(
    cell_group: torch.Tensor,
    depth: torch.Tensor,
    score: torch.Tensor,
    priority: torch.Tensor,
    source_view: torch.Tensor,
    eligible: torch.Tensor,
    *,
    group_count: int,
    source_view_count: int,
    depth_abs_eps: float,
    depth_rel_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.nonzero(eligible, as_tuple=False).flatten()
    if int(selected.numel()) == 0:
        empty = torch.empty((0,), device=cell_group.device, dtype=torch.long)
        return empty, empty
    groups = cell_group.index_select(0, selected)
    selected_priority = priority.index_select(0, selected).float()
    best_priority = torch.full(
        (group_count,),
        float("-inf"),
        device=priority.device,
        dtype=torch.float32,
    )
    best_priority.scatter_reduce_(
        0,
        groups,
        selected_priority,
        reduce="amax",
        include_self=True,
    )
    keep = selected_priority >= best_priority.index_select(0, groups) - 1.0e-8
    selected = selected[keep]
    groups = groups[keep]
    views = source_view.index_select(0, selected)
    best_view = torch.full(
        (group_count,),
        source_view_count,
        device=source_view.device,
        dtype=torch.long,
    )
    best_view.scatter_reduce_(
        0,
        groups,
        views,
        reduce="amin",
        include_self=True,
    )
    keep = views == best_view.index_select(0, groups)
    selected = selected[keep]
    groups = groups[keep]
    selected_depth = depth.index_select(0, selected).float()
    nearest_depth = torch.full(
        (group_count,),
        float("inf"),
        device=depth.device,
        dtype=torch.float32,
    )
    nearest_depth.scatter_reduce_(
        0,
        groups,
        selected_depth,
        reduce="amin",
        include_self=True,
    )
    tolerance = (
        depth_abs_eps + depth_rel_eps * nearest_depth.index_select(0, groups).abs()
    )
    keep = selected_depth <= nearest_depth.index_select(0, groups) + tolerance
    selected = selected[keep]
    groups = groups[keep]
    active, winner = _group_best_positions(
        groups,
        score.index_select(0, selected),
        group_count,
    )
    return active, selected.index_select(0, winner)


def _empty_plan(
    candidates: GeometryCandidates,
    target_grid_shape: tuple[int, int],
) -> TransportPlan:
    height, width = target_grid_shape
    shape = (
        candidates.batch_size,
        candidates.query_count,
        height,
        width,
    )
    device = candidates.u.device
    return TransportPlan(
        source_token=torch.full(shape, -1, device=device, dtype=torch.long),
        source_view=torch.full(shape, -1, device=device, dtype=torch.long),
        signed_dt=torch.zeros(shape, device=device, dtype=torch.float32),
        score=torch.zeros(shape, device=device, dtype=torch.float32),
        depth=torch.zeros(shape, device=device, dtype=torch.float32),
        hit_mask=torch.zeros(shape, device=device, dtype=torch.bool),
        candidate_motion_mask=torch.zeros(shape, device=device, dtype=torch.bool),
        motion_mask=torch.zeros(shape, device=device, dtype=torch.bool),
        source_grid_shape=candidates.source_grid_shape,
        target_grid_shape=target_grid_shape,
        candidate_count=len(candidates),
    )


def _materialize_plan(
    candidates: GeometryCandidates,
    target_grid_shape: tuple[int, int],
    *,
    selected: torch.Tensor,
    candidate_score: torch.Tensor,
    depth: torch.Tensor,
    winner_position: torch.Tensor,
    candidate_motion: torch.Tensor,
    winner_motion: torch.Tensor,
) -> TransportPlan:
    group_count = int(winner_position.numel())
    hit_flat = winner_position >= 0
    active_groups = torch.nonzero(hit_flat, as_tuple=False).flatten()
    active_positions = winner_position.index_select(0, active_groups)
    active_candidates = selected.index_select(0, active_positions)

    token_flat = torch.full(
        (group_count,),
        -1,
        device=selected.device,
        dtype=torch.long,
    )
    view_flat = torch.full_like(token_flat, -1)
    signed_dt_flat = torch.zeros(
        (group_count,),
        device=selected.device,
        dtype=torch.float32,
    )
    score_flat = torch.zeros_like(signed_dt_flat)
    depth_flat = torch.zeros_like(signed_dt_flat)
    token_flat[active_groups] = candidates.source_token.index_select(
        0, active_candidates
    )
    view_flat[active_groups] = candidates.source_view.index_select(0, active_candidates)
    signed_dt_flat[active_groups] = candidates.signed_dt.index_select(
        0, active_candidates
    ).float()
    score_flat[active_groups] = candidate_score.index_select(0, active_positions)
    depth_flat[active_groups] = depth.index_select(0, active_positions)

    target_h, target_w = target_grid_shape
    plan_shape = (
        candidates.batch_size,
        candidates.query_count,
        target_h,
        target_w,
    )
    return TransportPlan(
        source_token=token_flat.reshape(plan_shape),
        source_view=view_flat.reshape(plan_shape),
        signed_dt=signed_dt_flat.reshape(plan_shape),
        score=score_flat.reshape(plan_shape),
        depth=depth_flat.reshape(plan_shape),
        hit_mask=hit_flat.reshape(plan_shape),
        candidate_motion_mask=candidate_motion.reshape(plan_shape),
        motion_mask=winner_motion.reshape(plan_shape),
        source_grid_shape=candidates.source_grid_shape,
        target_grid_shape=target_grid_shape,
        candidate_count=len(candidates),
    )


def _staged_winners(
    *,
    cell_group: torch.Tensor,
    depth: torch.Tensor,
    score: torch.Tensor,
    priority: torch.Tensor,
    source_view: torch.Tensor,
    stages: tuple[torch.Tensor, ...],
    group_count: int,
    source_view_count: int,
    depth_abs_eps: float,
    depth_rel_eps: float,
) -> torch.Tensor:
    winner_position = torch.full(
        (group_count,),
        -1,
        device=cell_group.device,
        dtype=torch.long,
    )
    claimed = torch.zeros(
        (group_count,),
        device=cell_group.device,
        dtype=torch.bool,
    )
    for stage in stages:
        eligible = stage & ~claimed.index_select(0, cell_group)
        groups, winners = _stage_winners(
            cell_group,
            depth,
            score,
            priority,
            source_view,
            eligible,
            group_count=group_count,
            source_view_count=source_view_count,
            depth_abs_eps=depth_abs_eps,
            depth_rel_eps=depth_rel_eps,
        )
        winner_position[groups] = winners
        claimed[groups] = True
    return winner_position


def _nearest_time_per_cell(
    cell_group: torch.Tensor,
    signed_dt: torch.Tensor,
    eligible: torch.Tensor,
    *,
    group_count: int,
) -> torch.Tensor:
    distance = signed_dt.abs().float()
    nearest = torch.full(
        (group_count,),
        float("inf"),
        device=distance.device,
        dtype=torch.float32,
    )
    nearest.scatter_reduce_(
        0,
        cell_group,
        torch.where(eligible, distance, torch.full_like(distance, float("inf"))),
        reduce="amin",
        include_self=True,
    )
    return eligible & (distance <= nearest.index_select(0, cell_group) + 1.0e-8)


@torch.no_grad()
def _observation_winners(
    cell_group: torch.Tensor,
    depth: torch.Tensor,
    score: torch.Tensor,
    pose_dist: torch.Tensor,
    source_view: torch.Tensor,
    motion_mask: torch.Tensor,
    signed_dt: torch.Tensor,
    time_direction: torch.Tensor,
    *,
    group_count: int,
    source_view_count: int,
    depth_abs_eps: float,
    depth_rel_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select observations within each group, independently of feature payloads."""
    motion_candidate_count = torch.zeros(
        (group_count,),
        device=cell_group.device,
        dtype=torch.long,
    )
    motion_candidate_count.scatter_add_(0, cell_group, motion_mask.long())
    has_motion_candidate = motion_candidate_count > 0
    preferred_side = signed_dt * time_direction <= 0
    primary_motion = _nearest_time_per_cell(
        cell_group,
        signed_dt,
        motion_mask & preferred_side,
        group_count=group_count,
    )
    secondary_motion = _nearest_time_per_cell(
        cell_group,
        signed_dt,
        motion_mask & ~preferred_side,
        group_count=group_count,
    )
    anchor_without_motion = ~motion_mask & ~has_motion_candidate.index_select(
        0, cell_group
    )
    winners = _staged_winners(
        cell_group=cell_group,
        depth=depth,
        score=score,
        priority=-pose_dist,
        source_view=source_view,
        stages=(primary_motion, secondary_motion, anchor_without_motion),
        group_count=group_count,
        source_view_count=source_view_count,
        depth_abs_eps=depth_abs_eps,
        depth_rel_eps=depth_rel_eps,
    )
    return winners, has_motion_candidate


@torch.no_grad()
def rasterize_plan(
    candidates: GeometryCandidates,
    target_grid_shape: tuple[int, int],
    *,
    target_missing_mask: torch.Tensor,
    same_depth_abs_eps: float = 1.0e-3,
    same_depth_rel_eps: float = 1.0e-3,
) -> TransportPlan:
    count = _validate_candidates(candidates)
    target_grid_shape = tuple(int(value) for value in target_grid_shape)
    target_h, target_w = target_grid_shape
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_grid_shape values must be > 0")
    expected_missing_shape = (
        candidates.batch_size,
        candidates.query_count,
        target_h,
        target_w,
    )
    if (
        not isinstance(target_missing_mask, torch.Tensor)
        or target_missing_mask.dtype != torch.bool
        or tuple(int(value) for value in target_missing_mask.shape)
        != expected_missing_shape
        or target_missing_mask.device != candidates.u.device
    ):
        raise ValueError(
            "target_missing_mask must be bool "
            f"{expected_missing_shape} on the candidate device"
        )
    for name, value in {
        "same_depth_abs_eps": same_depth_abs_eps,
        "same_depth_rel_eps": same_depth_rel_eps,
    }.items():
        if not 0.0 < float(value) < float("inf"):
            raise ValueError(f"{name} must be finite and > 0")
    if count == 0:
        return _empty_plan(candidates, target_grid_shape)

    target_count = target_h * target_w
    group_count = candidates.batch_size * candidates.query_count * target_count
    selected, cell_group, candidate_score = _select_integer_token_representatives(
        candidates,
        target_grid_shape,
        depth_abs_eps=same_depth_abs_eps,
        depth_rel_eps=same_depth_rel_eps,
    )
    if int(selected.numel()) == 0:
        return _empty_plan(candidates, target_grid_shape)
    missing_flat = target_missing_mask.reshape(-1)
    keep = missing_flat.index_select(0, cell_group)
    selected = selected[keep]
    cell_group = cell_group[keep]
    candidate_score = candidate_score[keep]
    if int(selected.numel()) == 0:
        return _empty_plan(candidates, target_grid_shape)

    depth = candidates.z.index_select(0, selected).float()
    source_view = candidates.source_view.index_select(0, selected)
    motion_mask = candidates.motion_mask.index_select(0, selected)
    signed_dt = candidates.signed_dt.index_select(0, selected).float()
    time_direction = candidates.time_direction.index_select(0, selected).float()
    winner_position, has_motion_candidate = _observation_winners(
        cell_group,
        depth,
        candidate_score,
        candidates.pose_dist.index_select(0, selected).float(),
        source_view,
        motion_mask,
        signed_dt,
        time_direction,
        group_count=group_count,
        source_view_count=candidates.source_view_count,
        depth_abs_eps=same_depth_abs_eps,
        depth_rel_eps=same_depth_rel_eps,
    )
    winner_motion = torch.zeros_like(has_motion_candidate)
    winner_groups = torch.nonzero(winner_position >= 0, as_tuple=False).flatten()
    winner_motion[winner_groups] = motion_mask.index_select(
        0,
        winner_position.index_select(0, winner_groups),
    )
    return _materialize_plan(
        candidates,
        target_grid_shape,
        selected=selected,
        candidate_score=candidate_score,
        depth=depth,
        winner_position=winner_position,
        candidate_motion=has_motion_candidate,
        winner_motion=winner_motion,
    )


def _validate_payload(
    source_payload: torch.Tensor,
    plan: TransportPlan,
) -> tuple[int, int, tuple[int, ...]]:
    if source_payload.ndim < 4:
        raise ValueError(
            "source_payload must be [B,V,h,w,...] with at least four dimensions"
        )
    batch, views, source_h, source_w = (int(v) for v in source_payload.shape[:4])
    if (source_h, source_w) != plan.source_grid_shape:
        raise ValueError(
            f"source payload grid {(source_h, source_w)} does not match plan "
            f"{plan.source_grid_shape}"
        )
    if int(plan.source_token.shape[0]) != batch:
        raise ValueError("source payload batch does not match plan")
    token_count = views * source_h * source_w
    valid_tokens = plan.source_token >= 0
    if (
        bool(valid_tokens.any())
        and int(plan.source_token[valid_tokens].max()) >= token_count
    ):
        raise ValueError("plan source_token exceeds source payload bounds")
    return batch, token_count, tuple(int(v) for v in source_payload.shape[4:])


@torch.no_grad()
def apply_plan(
    source_payload: torch.Tensor,
    plan: TransportPlan,
) -> ProjectionResult:
    batch, token_count, payload_shape = _validate_payload(source_payload, plan)
    flat_payload = source_payload.reshape(batch, token_count, *payload_shape)
    safe_tokens = plan.source_token.clamp_min(0)
    gathered = torch.stack(
        [
            flat_payload[batch_idx]
            .index_select(0, safe_tokens[batch_idx].reshape(-1))
            .reshape(*safe_tokens.shape[1:], *payload_shape)
            for batch_idx in range(batch)
        ],
        dim=0,
    )
    valid_payload = (plan.source_token >= 0)[(...,) + (None,) * len(payload_shape)]
    gathered = torch.where(valid_payload, gathered, torch.zeros_like(gathered))
    hard_valid = plan.hit_mask[(...,) + (None,) * len(payload_shape)]
    gathered = torch.where(hard_valid, gathered, torch.zeros_like(gathered))
    return ProjectionResult(
        hit_latent=gathered,
        hit_mask=plan.hit_mask,
        score=torch.where(plan.hit_mask, plan.score, torch.zeros_like(plan.score)),
    )


@torch.no_grad()
def project(
    propagation: PropagationResult,
    source_payload: torch.Tensor,
    source_poses: torch.Tensor,
    target_poses: torch.Tensor,
    target_intrs: torch.Tensor,
    image_size: tuple[int, int],
    *,
    target_missing_mask: torch.Tensor,
    source_view_valid: torch.Tensor | None = None,
    source_token_valid: torch.Tensor | None = None,
    target_grid_shape: tuple[int, int] | None = None,
    same_depth_abs_eps: float = 1.0e-3,
    same_depth_rel_eps: float = 1.0e-3,
) -> tuple[ProjectionResult, TransportPlan]:
    if source_payload.ndim < 4:
        raise ValueError("source_payload must be [B,V,h,w,...]")
    source_grid_shape = tuple(int(value) for value in source_payload.shape[2:4])
    if target_grid_shape is None:
        target_grid_shape = source_grid_shape
    else:
        target_grid_shape = tuple(int(value) for value in target_grid_shape)
    if int(propagation.points.shape[2]) <= 0:
        raise ValueError("propagation must contain at least one target query")
    candidates = build_candidates(
        propagation,
        source_poses,
        target_poses,
        target_intrs,
        image_size,
        source_grid_shape,
        source_view_valid=source_view_valid,
        source_token_valid=source_token_valid,
    )
    plan = rasterize_plan(
        candidates,
        target_grid_shape,
        target_missing_mask=target_missing_mask,
        same_depth_abs_eps=same_depth_abs_eps,
        same_depth_rel_eps=same_depth_rel_eps,
    )
    return apply_plan(source_payload, plan), plan


__all__ = [
    "ProjectionResult",
    "TransportPlan",
    "apply_plan",
    "project",
    "rasterize_plan",
]
