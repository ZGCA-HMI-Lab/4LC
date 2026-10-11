from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, cast

import torch
from latent_cache import Representation

from alc_video.encoders.registry import BACKENDS

if TYPE_CHECKING:
    from alc_video.encoders.config import VisualEncoderConfig
    from alc_video.encoders.visual.base import VisualEncoder


def encoder_class(encoder_type: str) -> type[VisualEncoder]:
    try:
        backend = BACKENDS[encoder_type]
    except KeyError as error:
        raise ValueError(f"unsupported visual encoder: {encoder_type!r}") from error
    module = import_module(f"alc_video.encoders.visual.{backend.module}")
    return cast("type[VisualEncoder]", getattr(module, backend.class_name))


def load_visual_encoder(
    configured: VisualEncoderConfig,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> VisualEncoder:
    return encoder_class(configured.backend).from_config(
        configured, device=device, dtype=dtype
    )


def _unique_visual_encoders(
    encoders: Mapping[str, VisualEncoder],
) -> tuple[VisualEncoder, ...]:
    unique: list[VisualEncoder] = []
    seen: set[int] = set()
    for encoder in encoders.values():
        identity = id(encoder)
        if identity not in seen:
            seen.add(identity)
            unique.append(encoder)
    return tuple(unique)


def offload_visual_encoders(encoders: Mapping[str, VisualEncoder]) -> None:
    failures: list[str] = []
    for encoder in _unique_visual_encoders(encoders):
        try:
            encoder.offload()
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
    if failures:
        raise RuntimeError("visual encoder offload failed: " + "; ".join(failures))


def resolve_representations(
    configured_encoders: Mapping[str, VisualEncoderConfig],
    *,
    expected_names: tuple[str, ...],
) -> tuple[
    dict[str, tuple[int, int, int]],
    dict[str, Representation],
]:
    configured = dict(configured_encoders)
    if tuple(configured) != expected_names:
        raise ValueError("visual encoders differ from configuration")
    feature_shapes: dict[str, tuple[int, int, int]] = {}
    representations: dict[str, Representation] = {}
    for name, encoder_config in configured.items():
        representation = encoder_class(
            encoder_config.backend
        ).configured_representation(encoder_config.levels)
        feature_shapes[name] = (*representation.grid, representation.dim)
        representations[name] = representation
    return feature_shapes, representations


def load_visual_encoders(
    configured_encoders: Mapping[str, VisualEncoderConfig],
    *,
    expected_names: tuple[str, ...],
    device: str | torch.device,
    encoders: Mapping[str, VisualEncoder] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, VisualEncoder]:
    configured = dict(configured_encoders)
    if tuple(configured) != expected_names:
        raise ValueError("visual encoders differ from configuration")
    resolved: dict[str, VisualEncoder] = {}
    try:
        if encoders is None:
            loaded_by_asset: dict[tuple[str, object], VisualEncoder] = {}
            for name, encoder_config in configured.items():
                asset = (
                    encoder_config.backend,
                    encoder_config.path,
                    encoder_config.levels,
                    encoder_config.batch_size,
                )
                if asset not in loaded_by_asset:
                    loaded_by_asset[asset] = load_visual_encoder(
                        encoder_config,
                        device=device,
                        dtype=dtype,
                    )
                resolved[name] = loaded_by_asset[asset]
        else:
            resolved = dict(encoders)
        if tuple(resolved) != expected_names:
            raise ValueError("visual encoders must match configured names")
        for encoder in _unique_visual_encoders(resolved):
            encoder.prepare()
        return resolved
    except BaseException as error:
        if resolved:
            try:
                offload_visual_encoders(resolved)
            except Exception as cleanup_error:
                raise error from cleanup_error
        raise


__all__ = [
    "encoder_class",
    "load_visual_encoder",
    "load_visual_encoders",
    "offload_visual_encoders",
    "resolve_representations",
]
