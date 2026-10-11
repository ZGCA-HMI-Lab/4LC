from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

_PREFIX = "diffusion_model."
_DOWN_SUFFIX = ".lora_down.weight"
_UP_SUFFIX = ".lora_up.weight"
_BIAS_DIFF_SUFFIX = ".diff_b"
_WEIGHT_DIFF_SUFFIX = ".diff"


@dataclass(frozen=True, slots=True)
class LightX2VFusionReport:
    path: Path
    lora_pairs: int
    weight_diffs: int
    bias_diffs: int

    @property
    def updated_tensors(self) -> int:
        return self.lora_pairs + self.weight_diffs + self.bias_diffs

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "lora_pairs": self.lora_pairs,
            "weight_diffs": self.weight_diffs,
            "bias_diffs": self.bias_diffs,
            "updated_tensors": self.updated_tensors,
            "alpha": 1.0,
        }


def resolve_lightx2v_lora_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("model.lora_path must be a non-empty local file path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LightX2V LoRA does not exist: {path}")
    return path


def _target_name(key: str, suffix: str, target_suffix: str) -> str:
    return key[len(_PREFIX) : -len(suffix)] + target_suffix


def _fusion_plan(
    model: torch.nn.Module,
    path: Path,
) -> tuple[dict[str, tuple[str, str]], dict[str, str], dict[str, str]]:
    parameters = dict(model.named_parameters())
    if not parameters:
        raise ValueError("Wan backbone has no parameters")
    lora_parts: dict[str, dict[str, str]] = {}
    weight_diffs: dict[str, str] = {}
    bias_diffs: dict[str, str] = {}

    with safe_open(path, framework="pt", device="cpu") as tensors:
        keys = tuple(tensors.keys())
        for key in keys:
            if not key.startswith(_PREFIX):
                raise ValueError(f"LightX2V tensor has invalid prefix: {key}")
            if key.endswith(_DOWN_SUFFIX):
                target = _target_name(key, _DOWN_SUFFIX, ".weight")
                kind = "down"
                collection = lora_parts.setdefault(target, {})
                if kind in collection:
                    raise ValueError(f"duplicate LightX2V tensor target: {key}")
                collection[kind] = key
            elif key.endswith(_UP_SUFFIX):
                target = _target_name(key, _UP_SUFFIX, ".weight")
                kind = "up"
                collection = lora_parts.setdefault(target, {})
                if kind in collection:
                    raise ValueError(f"duplicate LightX2V tensor target: {key}")
                collection[kind] = key
            elif key.endswith(_BIAS_DIFF_SUFFIX):
                target = _target_name(key, _BIAS_DIFF_SUFFIX, ".bias")
                if target in bias_diffs:
                    raise ValueError(f"duplicate LightX2V tensor target: {key}")
                bias_diffs[target] = key
            elif key.endswith(_WEIGHT_DIFF_SUFFIX):
                target = _target_name(key, _WEIGHT_DIFF_SUFFIX, ".weight")
                if target in weight_diffs:
                    raise ValueError(f"duplicate LightX2V tensor target: {key}")
                weight_diffs[target] = key
            else:
                raise ValueError(f"unsupported LightX2V tensor key: {key}")

        lora_pairs: dict[str, tuple[str, str]] = {}
        for target, parts in lora_parts.items():
            if set(parts) != {"down", "up"}:
                raise ValueError(f"incomplete LightX2V LoRA pair for {target}")
            parameter = parameters.get(target)
            if parameter is None:
                raise ValueError(f"LightX2V target is absent from Wan: {target}")
            if parameter.ndim != 2:
                raise ValueError(f"LightX2V LoRA target must be a matrix: {target}")
            down_shape = tuple(tensors.get_slice(parts["down"]).get_shape())
            up_shape = tuple(tensors.get_slice(parts["up"]).get_shape())
            expected_shape = tuple(parameter.shape)
            if (
                len(down_shape) != 2
                or len(up_shape) != 2
                or down_shape[0] != up_shape[1]
                or down_shape[1] != expected_shape[1]
                or up_shape[0] != expected_shape[0]
            ):
                raise ValueError(
                    f"LightX2V LoRA shape differs for {target}: "
                    f"down={down_shape}, up={up_shape}, target={expected_shape}"
                )
            lora_pairs[target] = (parts["down"], parts["up"])

        for target, key in (*weight_diffs.items(), *bias_diffs.items()):
            parameter = parameters.get(target)
            if parameter is None:
                raise ValueError(f"LightX2V target is absent from Wan: {target}")
            tensor_shape = tuple(tensors.get_slice(key).get_shape())
            if tensor_shape != tuple(parameter.shape):
                raise ValueError(
                    f"LightX2V diff shape differs for {target}: "
                    f"{tensor_shape} != {tuple(parameter.shape)}"
                )

    return lora_pairs, weight_diffs, bias_diffs


def _apply_fusion_plan(
    model: torch.nn.Module,
    path: Path,
    lora_pairs: dict[str, tuple[str, str]],
    weight_diffs: dict[str, str],
    bias_diffs: dict[str, str],
) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad(), safe_open(path, framework="pt", device="cpu") as tensors:
        for target, (down_key, up_key) in sorted(lora_pairs.items()):
            parameter = parameters[target]
            down = tensors.get_tensor(down_key).to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            up = tensors.get_tensor(up_key).to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.add_(torch.mm(up, down))
        for target, key in sorted((*weight_diffs.items(), *bias_diffs.items())):
            parameter = parameters[target]
            diff = tensors.get_tensor(key).to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.add_(diff)


def fuse_lightx2v_distill_lora(
    model: torch.nn.Module,
    path: str | Path,
) -> LightX2VFusionReport:
    resolved = resolve_lightx2v_lora_path(path)
    lora_pairs, weight_diffs, bias_diffs = _fusion_plan(model, resolved)
    _apply_fusion_plan(
        model,
        resolved,
        lora_pairs,
        weight_diffs,
        bias_diffs,
    )
    return LightX2VFusionReport(
        path=resolved,
        lora_pairs=len(lora_pairs),
        weight_diffs=len(weight_diffs),
        bias_diffs=len(bias_diffs),
    )


__all__ = [
    "LightX2VFusionReport",
    "fuse_lightx2v_distill_lora",
    "resolve_lightx2v_lora_path",
]
