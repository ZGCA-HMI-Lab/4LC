from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

if TYPE_CHECKING:
    from alc_video.models.model import WanConditionedModel


TRAINABLE_MODULES = ("condition_adapter", "control_branch")


CHECKPOINT_PREFIX = "conditioned_model."
CHECKPOINT_STEP_PATTERN = re.compile(r"step_(\d{8})\Z")
TEMPORARY_CHECKPOINT_STEP_PATTERN = re.compile(r"\.step_(\d{8})\.tmp\Z")
DEEPSPEED_MODEL_TAG = "pytorch_model"


class CheckpointError(ValueError):
    pass


def _filter_state_dict(
    state: Mapping[str, torch.Tensor], parts: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    prefixes = tuple(f"{name}." for name in parts)
    return {name: value for name, value in state.items() if name.startswith(prefixes)}


def checkpoint_step(path_or_name: str | Path, *, allow_temporary: bool = False) -> int:
    name = Path(path_or_name).name
    match = CHECKPOINT_STEP_PATTERN.fullmatch(name)
    if match is None and allow_temporary:
        match = TEMPORARY_CHECKPOINT_STEP_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid checkpoint step name: {name!r}")
    step = int(match.group(1))
    if step <= 0:
        raise ValueError("checkpoint step must be a positive integer")
    return step


def _module_state(path: Path) -> Mapping[str, Any]:
    latest = path / "latest"
    if (
        not latest.is_file()
        or latest.is_symlink()
        or latest.read_text(encoding="utf-8").strip() != DEEPSPEED_MODEL_TAG
    ):
        raise CheckpointError(
            f"current-step DeepSpeed checkpoint tag is invalid: {latest}"
        )
    tagged = path / DEEPSPEED_MODEL_TAG
    if not tagged.is_dir() or tagged.is_symlink():
        raise CheckpointError(
            f"current-step DeepSpeed model state is unavailable: {tagged}"
        )
    model_state = tagged / "mp_rank_00_model_states.pt"
    if not model_state.is_file() or model_state.is_symlink():
        raise CheckpointError(
            f"current-step DeepSpeed module state is unavailable: {model_state}"
        )
    try:
        payload = torch.load(
            model_state, map_location="cpu", weights_only=False, mmap=True
        )
    except Exception as error:
        raise CheckpointError(
            "current-step DeepSpeed module state is invalid: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise CheckpointError("current-step DeepSpeed module payload is invalid")
    state = payload.get("module")
    if not isinstance(state, Mapping) or not state:
        raise CheckpointError("current-step DeepSpeed module state is empty")
    return state


def load_checkpoint(
    network: WanConditionedModel,
    checkpoint_path: str | Path,
    *,
    allow_temporary: bool = False,
) -> int:
    path = Path(checkpoint_path).expanduser().resolve()
    step = checkpoint_step(path, allow_temporary=allow_temporary)
    state = _module_state(path)
    expected = {
        CHECKPOINT_PREFIX + name: value
        for name, value in network.state_dict(keep_vars=True).items()
    }
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise CheckpointError(
            f"checkpoint model structure differs: missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, reference in expected.items():
            if reference.device.type != "cpu":
                raise ValueError(
                    "load the checkpoint before moving the network to CUDA"
                )
            source = state[name]
            if not isinstance(source, torch.Tensor):
                try:
                    source = source.contiguous()
                except Exception as error:
                    raise CheckpointError(
                        f"cannot materialize checkpoint tensor {name}: "
                        f"{type(error).__name__}: {error}"
                    ) from error
            if not isinstance(source, torch.Tensor):
                raise CheckpointError(f"checkpoint value is not a tensor: {name}")
            try:
                reference.copy_(source)
            except RuntimeError as error:
                raise CheckpointError(
                    f"checkpoint tensor differs for {name}: {error}"
                ) from error
    return step


def load_release(
    network: WanConditionedModel,
    path: str | Path,
    *,
    parts: tuple[str, ...] = TRAINABLE_MODULES,
) -> None:
    parameters = network.state_dict(keep_vars=True)
    expected = set(_filter_state_dict(parameters, parts))
    with safe_open(
        str(Path(path).expanduser()), framework="pt", device="cpu"
    ) as tensors:
        keys = set(tensors.keys())
        if keys != expected:
            raise CheckpointError(
                f"release tensors differ: missing={sorted(expected - keys)}, unexpected={sorted(keys - expected)}"
            )
        with torch.no_grad():
            for name in sorted(expected):
                source = tensors.get_tensor(name)
                if source.shape != parameters[name].shape:
                    raise CheckpointError(f"release shape differs for {name}")
                parameters[name].copy_(source)


def export_release(
    network: WanConditionedModel,
    path: str | Path,
    *,
    parts: tuple[str, ...] = TRAINABLE_MODULES,
) -> None:
    state = network.state_dict()
    save_file(
        {
            name: state[name].detach().cpu().contiguous()
            for name in sorted(_filter_state_dict(state, parts))
        },
        str(path),
    )


def export_checkpoint_release(
    checkpoint: str | Path,
    path: str | Path,
    *,
    parts: tuple[str, ...] = TRAINABLE_MODULES,
) -> None:
    state = _module_state(Path(checkpoint).expanduser().resolve())
    network_state = {
        name.removeprefix(CHECKPOINT_PREFIX): value for name, value in state.items()
    }
    learned = {
        name: value.detach().cpu().contiguous()
        for name, value in _filter_state_dict(network_state, parts).items()
    }
    if not learned:
        raise CheckpointError("checkpoint contains no adapter/control weights")
    save_file(learned, str(Path(path).expanduser()))


__all__ = [
    "CHECKPOINT_PREFIX",
    "CheckpointError",
    "checkpoint_step",
    "load_checkpoint",
    "load_release",
    "export_release",
    "export_checkpoint_release",
]
