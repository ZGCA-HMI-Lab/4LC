from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch
from torch import Tensor

from latent_cache.construction import check_tensor
from latent_cache.editing.data import (
    CacheData,
    SpatialMap,
    TimeMap,
    _Active,
    _Transform,
    as_data,
    new_identity,
    per_block,
)
from latent_cache.types import CacheEntry


def select(
    data: CacheEntry | CacheData,
    mask=None,
    *,
    feature_masks=None,
    active: TimeMap | None = None,
) -> CacheData:
    """Intersect source geometry/evidence masks without modifying stored features.

    Masks for one block use its native [B,K,H,W] geometry or [B,K,h,w]
    representation grids. Multiple blocks take equally sized sequences; None
    leaves a block unchanged. ``feature_masks`` items are name-to-mask mappings.
    """
    data = as_data(data)
    masks = per_block(mask, len(data.blocks), "mask")
    evidence = per_block(feature_masks, len(data.blocks), "feature_masks")
    if active is not None and not callable(active):
        raise TypeError("active must be callable")
    blocks = []
    for block, geometry, features in zip(data.blocks, masks, evidence, strict=True):
        entry = block.entry
        selected = block.mask
        if geometry is not None:
            check_tensor(
                geometry,
                tuple(entry.valid.shape),
                "mask",
                device=entry.valid.device,
                dtype=torch.bool,
            )
            selected = geometry.clone() if selected is None else selected & geometry
        selected_features = dict(block.feature_masks)
        if features is not None:
            if not isinstance(features, Mapping) or set(features) - set(entry.features):
                raise ValueError("feature_masks must name stored representations")
            for name, value in features.items():
                check_tensor(
                    value,
                    tuple(entry.features[name].valid.shape),
                    f"feature_masks[{name!r}]",
                    device=entry.valid.device,
                    dtype=torch.bool,
                )
                previous = selected_features.get(name)
                selected_features[name] = (
                    value.clone() if previous is None else previous & value
                )
        operations = block.operations + (() if active is None else (_Active(active),))
        blocks.append(
            replace(
                block,
                mask=selected,
                feature_masks=selected_features,
                operations=operations,
            )
        )
    return CacheData(tuple(blocks))


def transform(
    data: CacheEntry | CacheData,
    *,
    spatial: SpatialMap | None = None,
    time_map: TimeMap | None = None,
) -> CacheData:
    """Apply X_new(t) = spatial(t) X_input(time_map(t)).

    A fixed spatial matrix is [4,4]. A spatial callable returns [4,4] or
    [B,N,4,4]; a time callable returns [B,N] source times. Callables receive query
    times on the query device. Their captured state remains caller-owned.
    """
    data = as_data(data)
    if (
        spatial is not None
        and not isinstance(spatial, Tensor)
        and not callable(spatial)
    ):
        raise TypeError("spatial must be a tensor or callable")
    if time_map is not None and not callable(time_map):
        raise TypeError("time_map must be callable")
    if spatial is None and time_map is None:
        return data
    if isinstance(spatial, Tensor):
        spatial = spatial.clone()
    operation = _Transform(spatial, time_map)
    identities = {source.identity: new_identity() for source in data.sources}
    return CacheData(
        tuple(
            replace(
                block,
                identity=identities[block.identity],
                operations=block.operations + (operation,),
            )
            for block in data.blocks
        )
    )


def merge(items: Sequence[CacheEntry | CacheData]) -> CacheData:
    """Union source blocks; visibility is resolved by editing.read."""
    blocks = tuple(block for item in items for block in as_data(item).blocks)
    if not blocks:
        raise ValueError("merge needs at least one source")
    first = blocks[0].entry
    names = tuple(first.features)
    specs = tuple(payload.representation for payload in first.features.values())
    for block in blocks:
        entry = block.entry
        if (
            tuple(entry.features) != names
            or tuple(p.representation for p in entry.features.values()) != specs
        ):
            raise ValueError("sources hold different representations")
        if (
            entry.depth.shape[0] != first.depth.shape[0]
            or entry.depth.shape[2:] != first.depth.shape[2:]
        ):
            raise ValueError("sources must share batch and image dimensions")
        if entry.positions.device != first.positions.device:
            raise ValueError("sources must share one device")
    return CacheData(blocks)
