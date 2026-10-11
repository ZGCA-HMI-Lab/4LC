from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from itertools import count

import torch
from torch import Tensor

from latent_cache.types import CacheEntry

TimeMap = Callable[[Tensor], Tensor]
SpatialMap = Tensor | Callable[[Tensor], Tensor]

# Identities outlive data copies without retaining the original tensor storage.
_keys = count()
_origins: dict[int, tuple[weakref.ReferenceType, int]] = {}


def new_identity() -> int:
    return next(_keys)


def _identity(entry: CacheEntry) -> int:
    key = id(entry)
    if key not in _origins:
        reference = weakref.ref(entry, lambda _ref: _origins.pop(key, None))
        _origins[key] = reference, new_identity()
    return _origins[key][1]


@dataclass(frozen=True)
class _Transform:
    spatial: SpatialMap | None
    time_map: TimeMap | None


@dataclass(frozen=True)
class _Active:
    predicate: TimeMap


@dataclass(frozen=True)
class _Block:
    entry: CacheEntry
    identity: int
    mask: Tensor | None = None
    feature_masks: Mapping[str, Tensor] = field(default_factory=dict)
    operations: tuple[_Transform | _Active, ...] = ()


@dataclass(frozen=True)
class _Source:
    """One observation instance in the flattened Readout source address space."""

    identity: int
    entry: CacheEntry
    keyframe_offset: int

    def token_offset(self, representation: str) -> int:
        height, width = self.entry.features[representation].representation.grid
        return self.keyframe_offset * height * width


@dataclass(frozen=True)
class CacheData:
    """Source blocks with shared features, selection, and query-time transforms.

    ``sources`` maps Readout's global keyframe/token indices to source entries.
    Source order is stable when inputs are reordered or merges are nested.
    """

    blocks: tuple[_Block, ...]

    @property
    def sources(self) -> tuple[_Source, ...]:
        entries = {block.identity: block.entry for block in self.blocks}
        offset = 0
        sources = []
        for identity, entry in sorted(entries.items()):
            sources.append(_Source(identity, entry, offset))
            offset += entry.num_keyframes
        return tuple(sources)

    def to(self, device) -> CacheData:
        entries = {}
        matrices = {}
        blocks = []
        for block in self.blocks:
            key = id(block.entry)
            if key not in entries:
                entries[key] = block.entry.to(device)
            operations = []
            for operation in block.operations:
                if isinstance(operation, _Transform) and isinstance(
                    operation.spatial, Tensor
                ):
                    key_matrix = id(operation.spatial)
                    if key_matrix not in matrices:
                        matrices[key_matrix] = operation.spatial.to(device)
                    operation = replace(operation, spatial=matrices[key_matrix])
                operations.append(operation)
            blocks.append(
                replace(
                    block,
                    entry=entries[key],
                    mask=None if block.mask is None else block.mask.to(device),
                    feature_masks={
                        name: value.to(device)
                        for name, value in block.feature_masks.items()
                    },
                    operations=tuple(operations),
                )
            )
        return CacheData(tuple(blocks))


def as_data(data: CacheEntry | CacheData) -> CacheData:
    if isinstance(data, CacheEntry):
        return CacheData((_Block(data, _identity(data)),))
    if not isinstance(data, CacheData):
        raise TypeError("data must be CacheEntry or editing.CacheData")
    if not data.blocks:
        raise ValueError("CacheData needs at least one source block")
    return data


def per_block(value, count: int, name: str) -> tuple:
    if value is None:
        return (None,) * count
    if isinstance(value, (Tensor, Mapping)):
        if count != 1:
            raise ValueError(f"{name} needs one item per source block")
        return (value,)
    values = tuple(value)
    if len(values) != count:
        raise ValueError(f"{name} needs {count} source-block items")
    return values


def evaluate(block: _Block, times: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Pull queries through edits in reverse order; leave source data unchanged."""
    local = times
    matrix = torch.eye(4, device=times.device).expand(*times.shape, 4, 4)
    active = torch.ones_like(times, dtype=torch.bool)
    for operation in reversed(block.operations):
        if isinstance(operation, _Active):
            value = operation.predicate(local)
            if (
                not isinstance(value, Tensor)
                or value.shape != times.shape
                or value.dtype != torch.bool
                or value.device != times.device
            ):
                raise ValueError(
                    "active must return a bool tensor matching query times"
                )
            active = active & value
            continue
        if operation.spatial is not None:
            value = (
                operation.spatial(local)
                if callable(operation.spatial)
                else operation.spatial
            )
            if (
                not isinstance(value, Tensor)
                or value.device != times.device
                or not torch.is_floating_point(value)
            ):
                raise ValueError(
                    "spatial must return floating matrices on the query device"
                )
            if value.shape == (4, 4):
                value = value.expand(*times.shape, 4, 4)
            if value.shape != (*times.shape, 4, 4):
                raise ValueError("spatial must be [4,4] or [B,N,4,4]")
            _check_similarity(value)
            matrix = matrix @ value.float()
        if operation.time_map is not None:
            local = operation.time_map(local)
            if (
                not isinstance(local, Tensor)
                or local.shape != times.shape
                or local.device != times.device
                or not torch.is_floating_point(local)
                or not bool(torch.isfinite(local).all())
            ):
                raise ValueError(
                    "time_map must return finite floating times with shape [B,N]"
                )
    return local, matrix, active


def _check_similarity(matrix: Tensor) -> None:
    linear = matrix[..., :3, :3].float()
    gram = linear.transpose(-1, -2) @ linear
    scale2 = gram.diagonal(dim1=-2, dim2=-1).mean(-1)
    expected = torch.eye(3, device=matrix.device) * scale2[..., None, None]
    bottom = matrix.new_tensor([0, 0, 0, 1]).expand_as(matrix[..., 3, :])
    if (
        not bool(torch.isfinite(matrix).all())
        or not bool((scale2 > 0).all())
        or not bool((torch.linalg.det(linear) > 0).all())
        or not torch.allclose(gram, expected, atol=1e-5, rtol=1e-4)
        or not torch.allclose(matrix[..., 3, :], bottom, atol=1e-6, rtol=0)
    ):
        raise ValueError("spatial must be a positive-scale rotation and translation")
