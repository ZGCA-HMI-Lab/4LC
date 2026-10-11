from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file
from safetensors.torch import save as serialize_safetensors

UNCONDITIONAL_CONTEXT_SHAPE = (512, 4096)


def _write_safetensors(
    path: str | Path,
    tensors: Mapping[str, torch.Tensor],
) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Safetensors output already exists: {destination}")
    if not tensors:
        raise ValueError("Safetensors output must contain at least one tensor")

    payload: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Safetensors tensor names must be nonempty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Safetensors value {name!r} must be a tensor")
        payload[name] = value.detach().to(device="cpu", copy=True).contiguous()

    serialized = serialize_safetensors(payload)
    if not serialized:
        raise ValueError("serialized Safetensors output is empty")

    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with destination.open("xb") as stream:
            created = True
            written = stream.write(serialized)
            stream.flush()
        if written != len(serialized):
            raise OSError(
                f"short Safetensors write {written}/{len(serialized)}: {destination}"
            )
        loaded = load_file(str(destination), device="cpu")
        if set(loaded) != set(payload) or any(
            loaded[name].dtype != value.dtype
            or tuple(loaded[name].shape) != tuple(value.shape)
            for name, value in payload.items()
        ):
            raise ValueError(f"Safetensors output differs after writing: {destination}")
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return destination


def _validate_context(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("unconditional context must be a tensor")
    result = value.detach().cpu().contiguous()
    if tuple(result.shape) != UNCONDITIONAL_CONTEXT_SHAPE:
        raise ValueError(
            f"unconditional context shape {tuple(result.shape)} != "
            f"{UNCONDITIONAL_CONTEXT_SHAPE}"
        )
    if result.dtype != torch.bfloat16:
        raise TypeError("unconditional context must be bfloat16")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("unconditional context contains non-finite values")
    return result


def write_unconditional_context(
    path: str | Path,
    context: torch.Tensor,
) -> Path:
    output = Path(path).expanduser().resolve()
    if output.suffix != ".safetensors":
        raise ValueError("unconditional context output must use .safetensors")
    if output.exists():
        raise FileExistsError(f"unconditional context already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_safetensors(
        output,
        {"context": _validate_context(context)},
    )
    return output


def load_unconditional_context(
    path: str | Path,
) -> torch.Tensor:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"unconditional context does not exist: {source}")
    values = load_file(str(source), device="cpu")
    return _validate_context(values["context"])


__all__ = [
    "UNCONDITIONAL_CONTEXT_SHAPE",
    "load_unconditional_context",
    "write_unconditional_context",
]
