from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

from alc_video.models.checkpoint import (
    CHECKPOINT_STEP_PATTERN,
    CheckpointError,
    checkpoint_step,
)
from alc_video.training.execution.distributed import (
    run_main_process_stage,
    synchronize_process_error,
)
from alc_video.utils.logging import LOG


def resolve_resume_path(cfg: Any) -> Path | None:
    if cfg.checkpoint.resume_from is None:
        return None
    requested = cfg.checkpoint.resume_from
    if requested.name == "latest":
        path, _step = resolve_latest_checkpoint(requested)
        return path
    path = requested.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"training checkpoint does not exist: {path}")
    checkpoint_step(path)
    return path


def validate_training_output(cfg: Any) -> None:
    output = cfg.checkpoint.output_path
    if cfg.checkpoint.resume_from is None and output.is_dir():
        existing = next(
            (
                path
                for path in output.iterdir()
                if CHECKPOINT_STEP_PATTERN.fullmatch(path.name)
            ),
            None,
        )
        if existing is not None:
            raise FileExistsError(
                f"training checkpoint already exists: {existing}; "
                "set checkpoint.resume_from or choose another output_path"
            )


def save_training_checkpoint(
    accelerator: Any,
    cfg: Any,
    *,
    step: int,
    loader: Any,
    metrics: Mapping[str, float] | None = None,
) -> Path:
    output = cfg.checkpoint.output_path
    started = time.perf_counter()
    name = checkpoint_step_name(step)
    target, temporary = output / name, output / f".{name}.tmp"

    def setup() -> None:
        output.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"checkpoint already exists: {target}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()

    run_main_process_stage(accelerator, setup, name="checkpoint setup")
    LOG.info("checkpoint.save start update=%s path=%s", step, target)
    error = None
    try:
        accelerator.save_state(str(temporary), safe_serialization=True)
        _save_dataloader_state(accelerator, temporary, loader)
    except Exception as caught:
        error = caught
    if error is not None:
        LOG.error("checkpoint.save failed update=%s path=%s", step, target)
    synchronize_process_error(accelerator, error, operation="checkpoint state write")

    def publish() -> None:
        _write_json(
            temporary / "metadata.json", {"step": step, "metrics": dict(metrics or {})}
        )
        _write_json(temporary / "training.json", {"max_steps": cfg.training.max_steps})
        _fsync_dir(temporary)
        os.replace(temporary, target)
        _fsync_dir(output)

    run_main_process_stage(accelerator, publish, name="checkpoint publication")

    def maintain() -> None:
        update_latest_checkpoint(output, target)
        prune_checkpoints(output, max_to_keep=cfg.checkpoint.max_to_keep)

    run_main_process_stage(accelerator, maintain, name="checkpoint latest/retention")
    accelerator.wait_for_everyone()
    LOG.info(
        "checkpoint.save complete update=%s elapsed=%.2fs path=%s",
        step,
        time.perf_counter() - started,
        target,
    )
    return target


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def load_training_max_steps(checkpoint_path: str | Path) -> int:
    path = Path(checkpoint_path).expanduser().resolve() / "training.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(
            f"invalid checkpoint training metadata: {path}"
        ) from error
    if (
        not isinstance(payload, dict)
        or isinstance(payload["max_steps"], bool)
        or not isinstance(payload["max_steps"], int)
        or payload["max_steps"] <= 0
    ):
        raise CheckpointError(f"invalid checkpoint training metadata: {path}")
    return payload["max_steps"]


def _save_dataloader_state(
    accelerator: Any,
    root: Path,
    loader: Any,
) -> None:
    save_state = getattr(loader, "save_state_rank", None)
    if not callable(save_state):
        raise TypeError("training loader must provide save_state_rank()")
    directory = root / "dataloader"
    directory.mkdir(exist_ok=True)
    rank = int(accelerator.process_index)
    target = directory / f"rank-{rank:05d}.pt"
    temporary = directory / f".rank-{rank:05d}.tmp"
    torch.save(save_state(), temporary)
    os.replace(temporary, target)


def _restore_dataloader_state(
    accelerator: Any,
    root: Path,
    loader: Any,
) -> None:
    restore_state = getattr(loader, "restore_state_rank", None)
    if not callable(restore_state):
        raise TypeError("training loader must provide restore_state_rank()")
    rank = int(accelerator.process_index)
    path = root / "dataloader" / f"rank-{rank:05d}.pt"
    if not path.is_file():
        raise CheckpointError(f"rank loader state does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    restore_state(state)


def validate_resume_training_target(
    cfg: Any,
    checkpoint_path: str | Path,
) -> int:
    path = Path(checkpoint_path).resolve()
    step = checkpoint_step(path)
    checkpoint_max_steps = load_training_max_steps(path)
    if cfg.training.max_steps <= step:
        raise CheckpointError(
            "resume max_steps derived from current sample exposure must exceed "
            f"checkpoint step: target={cfg.training.max_steps}, step={step}, "
            f"checkpoint_max_steps={checkpoint_max_steps}"
        )
    return step


def load_training_checkpoint(
    accelerator: Any,
    cfg: Any,
    checkpoint_path: str | Path,
    *,
    loader: Any,
) -> int:
    path = Path(checkpoint_path).resolve()
    step = validate_resume_training_target(cfg, path)
    accelerator.load_state(str(path))
    _restore_dataloader_state(accelerator, path, loader)
    accelerator.wait_for_everyone()
    return step


def resume_training_iterator(
    loader: Any,
    torch_rng_state: torch.Tensor,
) -> Iterator[Any]:
    if (
        not isinstance(torch_rng_state, torch.Tensor)
        or torch_rng_state.dtype != torch.uint8
        or torch_rng_state.ndim != 1
        or torch_rng_state.device.type != "cpu"
    ):
        raise ValueError("resume Torch RNG state must be one-dimensional CPU uint8")
    iterator = iter(loader)
    try:
        first = next(iterator)
    except StopIteration:
        torch.set_rng_state(torch_rng_state)
        return iter(())
    torch.set_rng_state(torch_rng_state)

    def resumed() -> Iterator[Any]:
        yield first
        yield from iterator

    return resumed()


def validate_checkpoint_step(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("checkpoint step must be a positive integer")
    if value > 99_999_999:
        raise ValueError("checkpoint step must fit the eight-digit directory contract")
    return value


def checkpoint_step_name(step: int) -> str:
    return f"step_{validate_checkpoint_step(step):08d}"


def resolve_latest_checkpoint(path: str | Path) -> tuple[Path, int]:
    link = Path(path).expanduser().absolute()
    if link.name != "latest":
        raise ValueError(f"checkpoint must be a latest symlink: {link}")
    if not link.is_symlink():
        raise ValueError(f"checkpoint latest is not a symlink: {link}")
    try:
        checkpoint = link.resolve(strict=True)
        run = link.parent.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(
            f"checkpoint latest cannot be resolved: {link}"
        ) from error
    if checkpoint.parent != run:
        raise ValueError(f"checkpoint latest leaves its run directory: {link}")
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError(f"checkpoint latest target is invalid: {checkpoint}")
    return checkpoint, checkpoint_step(checkpoint)


def update_latest_checkpoint(root: str | Path, target: str | Path) -> None:
    output = Path(root)
    checkpoint = Path(target)
    if checkpoint.parent != output:
        raise ValueError("latest checkpoint target must be a direct child of its run")
    checkpoint_step(checkpoint)
    if not output.is_dir() or output.is_symlink():
        raise ValueError(f"checkpoint run directory is invalid: {output}")
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError(f"latest checkpoint target is invalid: {checkpoint}")
    link = output / "latest"
    temporary = output / ".latest.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(checkpoint.name)
    os.replace(temporary, link)
    descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prune_checkpoints(root: str | Path, *, max_to_keep: int) -> None:
    if (
        isinstance(max_to_keep, bool)
        or not isinstance(max_to_keep, int)
        or max_to_keep <= 0
    ):
        raise ValueError("checkpoint.max_to_keep must be a positive integer")
    output = Path(root)
    if not output.is_dir() or output.is_symlink():
        raise ValueError(f"checkpoint run directory is invalid: {output}")
    checkpoints = sorted(
        (
            path
            for path in output.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and CHECKPOINT_STEP_PATTERN.fullmatch(path.name)
        ),
        key=checkpoint_step,
    )
    for checkpoint in checkpoints[:-max_to_keep]:
        shutil.rmtree(checkpoint)
