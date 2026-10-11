from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, cast

import yaml


def reject_unknown_fields(
    value: Mapping[str, Any], allowed: Iterable[str], *, name: str = ""
) -> None:
    unknown = set(value) - set(allowed)
    if unknown:
        paths = [f"{name}.{key}" if name else str(key) for key in sorted(unknown)]
        raise ValueError(f"unknown configuration fields: {', '.join(paths)}")


def default_values(config_type: type[Any]) -> dict[str, Any]:
    result = {}
    for field in fields(config_type):
        if field.default is not MISSING:
            result[field.name] = copy.deepcopy(field.default)
        elif field.default_factory is not MISSING:
            result[field.name] = field.default_factory()
    return result


def merge_defaults(
    defaults: Mapping[str, Any], values: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("configuration must be a mapping")
    result = copy.deepcopy(dict(defaults))
    for name, value in values.items():
        if name == "representations":
            if not isinstance(value, Mapping) or not value:
                raise ValueError("representations must be a nonempty mapping")
            result[name] = copy.deepcopy(dict(value))
        elif name in result and isinstance(result[name], Mapping):
            result[name] = merge_defaults(result[name], value)
        else:
            result[name] = copy.deepcopy(value)
    return result


def resolve_path(value: str | Path, *, root: Path) -> Path:
    return (root / Path(value).expanduser()).resolve()


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise TypeError("configuration keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {path}")
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid YAML configuration: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError("configuration must be a mapping")
    return dict(value)


def _overlay(parent: object, child: object, *, path: str) -> object:
    if isinstance(parent, Mapping):
        if not isinstance(child, Mapping):
            raise TypeError(f"configuration override changes mapping type: {path}")
        if path == "encoders.representations":
            return dict(child)
        unknown = set(child) - set(parent)
        if unknown:
            raise ValueError(
                f"configuration override adds unknown fields at {path}: {sorted(unknown)}"
            )
        result = dict(parent)
        for key, value in child.items():
            name = f"{path}.{key}" if path else str(key)
            result[key] = _overlay(result[key], value, path=name)
        return result
    if isinstance(child, Mapping):
        raise TypeError(f"configuration override changes leaf type: {path}")
    return child


def resolve_document_paths(
    document: Mapping[str, Any], *, root: Path, path_fields: Iterable[str]
) -> dict[str, Any]:
    # YAML values are dynamic only during parsing; domain configs own runtime types.
    result = copy.deepcopy(dict(document))
    for field in path_fields:
        parts = field.split(".")
        value = result
        for part in parts[:-1]:
            value = value.get(part, {})
        if parts[-1] in value and value[parts[-1]] is not None:
            value[parts[-1]] = str(resolve_path(value[parts[-1]], root=root))
    return result


def load_document(
    path: str | Path, *, path_fields: Iterable[str] = ()
) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    child = read_yaml(source)
    parent_value = child.pop("parent", None)
    child = resolve_document_paths(child, root=source.parent, path_fields=path_fields)
    if parent_value is None:
        return source, child
    if not isinstance(parent_value, str) or not parent_value:
        raise TypeError("parent must be a nonempty path string")
    parent_path = (source.parent / parent_value).resolve()
    parent = read_yaml(parent_path)
    if "parent" in parent:
        raise ValueError("configuration parent nesting is forbidden")
    parent = resolve_document_paths(
        parent, root=parent_path.parent, path_fields=path_fields
    )
    merged = _overlay(parent, child, path="")
    if not isinstance(merged, dict):
        raise TypeError("resolved configuration must be a mapping")
    return source, merged


def apply_overrides(
    document: Mapping[str, object],
    overrides: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    seen: set[str] = set()
    for override in overrides:
        key, separator, raw = override.partition("=")
        if not separator or not key or key == "parent":
            raise ValueError(f"invalid configuration override: {override}")
        if key in seen:
            raise ValueError(f"duplicate configuration override: {key}")
        seen.add(key)
        parts = key.split(".")
        target: dict[str, Any] = result
        for part in parts[:-1]:
            value = target.get(part)
            if not isinstance(value, dict):
                raise ValueError(f"configuration override path does not exist: {key}")
            target = value
        leaf = parts[-1]
        if leaf not in target:
            raise ValueError(f"configuration override path does not exist: {key}")
        try:
            value = yaml.load(raw, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid configuration override value: {key}") from error
        target[leaf] = value
    return result


def as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return dict(value)


def positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return cast(int, value)


def finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty or null")
    return value
