from __future__ import annotations

from dataclasses import fields, replace

import torch
from torch import Tensor

from latent_cache.geometry.candidates import GeometryCandidates
from latent_cache.geometry.projection import (
    TransportPlan,
    _empty_plan,
    _materialize_plan,
    _observation_winners,
    _select_integer_token_representatives,
)


def combine_candidates(
    parts: list[GeometryCandidates],
    masks: list[dict[str, Tensor]],
    domains: list[Tensor],
) -> tuple[GeometryCandidates, dict[str, Tensor], Tensor]:
    """Union observations, coalescing repeated units from same-source selections."""
    first = parts[0]
    values = {
        field.name: torch.cat([getattr(part, field.name) for part in parts])
        for field in fields(first)
        if isinstance(getattr(first, field.name), Tensor)
    }
    combined = replace(first, **values)
    domain = torch.cat(domains)
    gates = {name: torch.cat([mask[name] for mask in masks]) for name in masks[0]}
    if not len(combined):
        return combined, gates, domain

    # Stable source-pixel order also preserves the core's representative tie rule.
    pixels = first.image_size[0] * first.image_size[1]
    key = (
        combined.batch_index * first.query_count + combined.query_index
    ) * first.source_view_count * pixels + combined.anchor_id
    order = torch.argsort(key, stable=True)
    _, inverse, counts = torch.unique_consecutive(
        key[order], return_inverse=True, return_counts=True
    )
    keep = order[counts.cumsum(0) - counts]
    combined = replace(
        combined, **{name: value[keep] for name, value in values.items()}
    )
    for name, gate in gates.items():
        united = torch.zeros(len(counts), dtype=torch.long, device=gate.device)
        united.scatter_reduce_(
            0, inverse, gate[order].long(), reduce="amax", include_self=True
        )
        gates[name] = united.bool()
    return combined, gates, domain[keep]


def joint_plan(
    candidates: GeometryCandidates,
    domains: Tensor,
    source_count: int,
    grid: tuple[int, int],
    fill: Tensor,
    *,
    depth_abs_eps: float,
    depth_rel_eps: float,
) -> tuple[TransportPlan, Tensor]:
    """Resolve same-source observations, then cross-source geometric visibility.

    The second result addresses original candidates for post-visibility editing
    masks. No latent features are read until this joint selection is complete.
    """
    empty = _empty_plan(candidates, grid)
    candidate_index = torch.full_like(empty.source_token, -1)
    if not len(candidates):
        return empty, candidate_index
    selected, cells, scores = _select_integer_token_representatives(
        candidates,
        grid,
        depth_abs_eps=depth_abs_eps,
        depth_rel_eps=depth_rel_eps,
    )
    requested = fill.reshape(-1)[cells]
    selected, cells, scores = selected[requested], cells[requested], scores[requested]
    if not selected.numel():
        return empty, candidate_index

    cell_count = fill.numel()
    groups = domains[selected] * cell_count + cells
    depth = candidates.z[selected].float()
    motion = candidates.motion_mask[selected]
    local_winner, has_motion = _observation_winners(
        groups,
        depth,
        scores,
        candidates.pose_dist[selected].float(),
        candidates.source_view[selected],
        motion,
        candidates.signed_dt[selected].float(),
        candidates.time_direction[selected].float(),
        group_count=source_count * cell_count,
        source_view_count=candidates.source_view_count,
        depth_abs_eps=depth_abs_eps,
        depth_rel_eps=depth_rel_eps,
    )
    eligible = local_winner[local_winner >= 0]
    winner_cells = cells[eligible]
    nearest = torch.full((cell_count,), float("inf"), device=depth.device)
    nearest.scatter_reduce_(
        0, winner_cells, depth[eligible], reduce="amin", include_self=True
    )
    eligible = eligible[depth[eligible] == nearest[cells[eligible]]]

    # Each instance contributes at most one winner per cell. Its persistent
    # identity determines the exact-depth tie, independent of merge input order.
    winner_source = torch.full(
        (cell_count,), source_count, device=depth.device, dtype=torch.long
    )
    winner_source.scatter_reduce_(
        0,
        cells[eligible],
        domains[selected[eligible]],
        reduce="amin",
        include_self=True,
    )
    eligible = eligible[domains[selected[eligible]] == winner_source[cells[eligible]]]
    winners = torch.full((cell_count,), -1, device=depth.device, dtype=torch.long)
    winners[cells[eligible]] = eligible
    winner_motion = torch.zeros(cell_count, dtype=torch.bool, device=depth.device)
    winner_motion[cells[eligible]] = motion[eligible]
    plan = _materialize_plan(
        candidates,
        grid,
        selected=selected,
        candidate_score=scores,
        depth=depth,
        winner_position=winners,
        candidate_motion=has_motion.reshape(source_count, cell_count).any(0),
        winner_motion=winner_motion,
    )
    candidate_index.reshape(-1)[cells[eligible]] = selected[eligible]
    return plan, candidate_index
