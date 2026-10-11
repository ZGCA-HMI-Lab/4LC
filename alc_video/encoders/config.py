from __future__ import annotations

import re
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from alc_video.encoders.registry import BACKENDS
from alc_video.utils.config import (
    as_mapping,
    default_values,
    read_yaml,
    reject_unknown_fields,
)

_Profile = TypeVar("_Profile")

REPRESENTATION_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_-]*\Z")


def validate_representation_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or REPRESENTATION_NAME_PATTERN.fullmatch(value) is None
    ):
        raise ValueError("feature name must match [a-z][a-z0-9_-]*")
    return value


@dataclass(frozen=True, slots=True)
class VisualEncoderConfig:
    name: str
    backend: str
    path: Path
    levels: tuple[int, ...] | None = None
    batch_size: int = 4


@dataclass(frozen=True, slots=True)
class WorldMirrorEncoderConfig:
    name: str
    checkpoint: Path


@dataclass(frozen=True, slots=True)
class EncoderCatalog:
    path: Path
    visual_encoders: dict[str, VisualEncoderConfig]
    worldmirror: dict[str, WorldMirrorEncoderConfig]

    def visual_encoder(self, name: str) -> VisualEncoderConfig:
        validated = validate_representation_name(name)
        try:
            return self.visual_encoders[validated]
        except KeyError as error:
            raise ValueError(
                f"unknown feature Encoder Profile {validated!r}; "
                f"available={tuple(self.visual_encoders)}"
            ) from error

    def worldmirror_encoder(self, name: str) -> WorldMirrorEncoderConfig:
        return _profile(self.worldmirror, name, role="WorldMirror")


def _profile(profiles: dict[str, _Profile], name: str, *, role: str) -> _Profile:
    validated = validate_representation_name(name)
    try:
        return profiles[validated]
    except KeyError as error:
        raise ValueError(
            f"unknown {role} Profile {validated!r}; available={tuple(profiles)}"
        ) from error


def _required(
    value: Any,
    *,
    name: str,
    fields: set[str],
    optional: Set[str] = frozenset(),
) -> dict[str, Any]:
    result = as_mapping(value, name=name)
    reject_unknown_fields(result, fields | optional, name=name)
    missing = fields - set(result)
    if missing:
        raise ValueError(f"{name} is missing fields {sorted(missing)}")
    return result


def resolve_asset_path(
    value: Any,
    *,
    name: str,
    root: Path,
    kind: str,
    require_exists: bool,
) -> Path:
    if (
        not isinstance(value, (str, Path))
        or not str(value).strip()
        or "://" in str(value)
    ):
        raise ValueError(f"{name} must be a non-empty local {kind} path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    valid = path.is_file() if kind == "file" else path.is_dir()
    if require_exists and not valid:
        raise FileNotFoundError(f"{name} {kind} does not exist: {path}")
    return path


def _profiles(value: Any, *, name: str) -> dict[str, Any]:
    profiles = as_mapping(value, name=name)
    if not profiles:
        raise ValueError(f"{name} must not be empty")
    result: dict[str, Any] = {}
    for raw_name, profile in profiles.items():
        validated = validate_representation_name(raw_name)
        if validated in result:
            raise ValueError(f"{name} contains duplicate Profile {validated!r}")
        result[validated] = profile
    return result


def load_encoder_catalog(
    path: str | Path,
    *,
    require_assets: bool = True,
) -> EncoderCatalog:
    if not isinstance(require_assets, bool):
        raise TypeError("require_assets must be bool")
    config_path = Path(path).expanduser().resolve()
    document = read_yaml(config_path)
    value = _required(
        document,
        name="Encoder config",
        fields={"visual_encoders", "worldmirror"},
    )
    root = config_path.parent
    visual_encoders: dict[str, VisualEncoderConfig] = {}
    for name, raw in _profiles(
        value["visual_encoders"],
        name="visual_encoders",
    ).items():
        profile = _required(
            as_mapping(raw, name=f"visual_encoders.{name}"),
            name=f"visual_encoders.{name}",
            fields={"backend", "path"},
            optional=set(default_values(VisualEncoderConfig)),
        )
        backend = profile["backend"]
        if backend not in BACKENDS:
            raise ValueError(f"unsupported visual encoder backend: {backend!r}")
        visual_encoders[name] = VisualEncoderConfig(
            name=name,
            backend=str(backend),
            path=resolve_asset_path(
                profile["path"],
                name=f"visual_encoders.{name}.path",
                root=root,
                kind=BACKENDS[str(backend)].path_kind,
                require_exists=require_assets,
            ),
            levels=None if profile.get("levels") is None else tuple(profile["levels"]),
            batch_size=int(profile.get("batch_size", 4)),
        )

    worldmirror: dict[str, WorldMirrorEncoderConfig] = {}
    for name, raw in _profiles(
        value["worldmirror"],
        name="worldmirror",
    ).items():
        profile = _required(
            raw,
            name=f"worldmirror.{name}",
            fields={"checkpoint"},
        )
        worldmirror[name] = WorldMirrorEncoderConfig(
            name=name,
            checkpoint=resolve_asset_path(
                profile["checkpoint"],
                name=f"worldmirror.{name}.checkpoint",
                root=root,
                kind="file",
                require_exists=require_assets,
            ),
        )

    return EncoderCatalog(
        path=config_path,
        visual_encoders=visual_encoders,
        worldmirror=worldmirror,
    )


__all__ = [
    "EncoderCatalog",
    "VisualEncoderConfig",
    "WorldMirrorEncoderConfig",
    "load_encoder_catalog",
]
