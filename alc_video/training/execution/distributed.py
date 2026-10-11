from __future__ import annotations

import datetime
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from alc_video.data.schema import SAMPLE_BATCH_SIZE


def _topology(accelerator: Any) -> tuple[int, int]:
    world_size = int(accelerator.num_processes)
    if world_size <= 0:
        raise ValueError("accelerator.num_processes must be > 0")
    is_main = bool(accelerator.is_main_process)
    if world_size == 1:
        if not is_main:
            raise RuntimeError("single-process accelerator must be the main process")
        return 0, 1
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "multi-process accelerator requires an initialized process group"
        )
    rank = int(torch.distributed.get_rank())
    actual_world = int(torch.distributed.get_world_size())
    if actual_world != world_size or is_main != (rank == 0):
        raise RuntimeError("accelerator and process-group topology differ")
    return rank, world_size


def _collective_device(accelerator: Any) -> torch.device:
    value = accelerator.device
    return torch.device(value)


def broadcast_main_mapping(
    accelerator: Any,
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _rank, world_size = _topology(accelerator)
    if accelerator.is_main_process:
        if not isinstance(value, Mapping):
            raise TypeError("main process must provide a mapping")
        payload: list[Any] = [dict(value)]
    else:
        if value is not None:
            raise ValueError("non-main process must provide value=None")
        payload = [None]
    if world_size > 1:
        torch.distributed.broadcast_object_list(
            payload,
            src=0,
            device=_collective_device(accelerator),
        )
    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("rank 0 did not broadcast a mapping")
    return result


def synchronize_process_error(
    accelerator: Any,
    error: BaseException | None,
    *,
    operation: str,
) -> None:
    rank, world_size = _topology(accelerator)
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string")
    local = (
        None
        if error is None
        else {
            "rank": rank,
            "type": type(error).__name__,
            "message": str(error),
        }
    )
    gathered: list[dict[str, Any] | None]
    if world_size == 1:
        gathered = [local]
    else:
        gathered = [None] * world_size
        torch.distributed.all_gather_object(
            gathered,
            local,
        )
    failures = tuple(item for item in gathered if item is not None)
    if not failures:
        return
    if error is not None:
        raise error
    details = "; ".join(
        f"rank {item['rank']} {item['type']}: {item['message']}" for item in failures
    )
    raise RuntimeError(f"{operation} failed on another process: {details}")


def run_main_process_stage(
    accelerator: Any,
    operation: Callable[[], Any],
    *,
    name: str,
) -> Any:
    if not callable(operation):
        raise TypeError("operation must be callable")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("stage name must be non-empty")
    caught: BaseException | None = None
    payload = None
    if accelerator.is_main_process:
        try:
            result = operation()
        except BaseException as error:
            caught = error
            payload = {
                "error_type": type(error).__name__,
                "error": str(error),
                "result": None,
            }
        else:
            payload = {
                "error_type": None,
                "error": None,
                "result": result,
            }
    resolved = broadcast_main_mapping(accelerator, payload)
    if resolved.get("error") is None:
        return resolved.get("result")
    if caught is not None:
        raise caught
    raise RuntimeError(
        f"{name} failed on rank 0: {resolved.get('error_type')}: {resolved['error']}"
    )


def scale_run_max_steps(
    cfg: Any,
    *,
    sample_count: int,
    world_size: int,
) -> Any:
    sample_exposure = cfg.training.sample_exposure
    if sample_count <= 0 or sample_exposure <= 0.0 or world_size <= 0:
        raise ValueError("sample count, exposure, and world size must be positive")
    max_steps = (
        cfg.training.max_updates
        if cfg.training.max_updates is not None
        else math.ceil(
            sample_count
            * sample_exposure
            / (
                world_size
                * SAMPLE_BATCH_SIZE
                * cfg.training.gradient_accumulation_steps
            )
        )
    )
    return replace(cfg, training=replace(cfg.training, max_steps=max_steps))


def seed_everything(seed: int) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_deepspeed_config(
    cfg: Any,
) -> dict[str, Any]:
    return {
        "train_micro_batch_size_per_gpu": SAMPLE_BATCH_SIZE,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "gradient_clipping": cfg.optimizer.clip_max_norm,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": cfg.training.zero_optimizer_offload,
            },
            "offload_param": {"device": "none"},
        },
        "bf16": {"enabled": True},
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 10_000_000,
        "wall_clock_breakdown": False,
    }


def create_accelerator(cfg: Any):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan2.1 training")
    from accelerate import Accelerator, DeepSpeedPlugin, InitProcessGroupKwargs

    from alc_video.utils.logging import quiet_libraries

    quiet_libraries()
    plugin = DeepSpeedPlugin(hf_ds_config=build_deepspeed_config(cfg))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16",
        deepspeed_plugin=plugin,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2))],
    )
    _validate_deepspeed_runtime(accelerator)
    return accelerator


def _validate_deepspeed_runtime(accelerator: Any) -> None:
    if accelerator.distributed_type.name != "DEEPSPEED":
        raise RuntimeError(
            "model training requires the DeepSpeed launcher; use 4lc train"
        )
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("DeepSpeed process group was not initialized; use 4lc train")
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if world_size != int(accelerator.num_processes):
        raise RuntimeError(
            f"process-group world size {world_size} != Accelerate processes "
            f"{accelerator.num_processes}"
        )
    if rank != int(accelerator.process_index):
        raise RuntimeError(
            f"process-group rank {rank} != Accelerate process index "
            f"{accelerator.process_index}"
        )
    backend = str(torch.distributed.get_backend()).lower()
    if backend != "nccl":
        raise RuntimeError(f"model CUDA training requires NCCL, got {backend!r}")
