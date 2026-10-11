from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any


class TimedLoader:
    def __init__(self, loader: Any) -> None:
        self.loader = loader
        self.iterator: Iterator[Any] | None = None
        self.last_started = 0.0
        self.last_data_load_ms = 0.0

    def __iter__(self) -> TimedLoader:
        self.iterator = iter(self.loader)
        return self

    def __next__(self) -> Any:
        if self.iterator is None:
            raise RuntimeError("timed loader was not initialized")
        self.last_started = time.perf_counter()
        batch = next(self.iterator)
        self.last_data_load_ms = (time.perf_counter() - self.last_started) * 1000.0
        return batch


class TrainingTimingRecorder:
    def __init__(self, every_steps: int, device: Any) -> None:
        import torch

        if every_steps <= 0:
            raise ValueError("training timing interval must be positive")
        self.every_steps = every_steps
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("training timing profile requires a CUDA device")
        self.active = False
        self.iteration_started = 0.0
        self.train_started = 0.0
        self.data_load_ms = 0.0
        self.events: dict[str, tuple[Any, Any]] = {}

    def begin(
        self,
        *,
        next_step: int,
        iteration_started: float,
        data_load_ms: float,
    ) -> None:
        self.active = next_step % self.every_steps == 0
        self.iteration_started = iteration_started
        self.train_started = time.perf_counter()
        self.data_load_ms = data_load_ms
        self.events = {}

    @contextmanager
    def cuda(self, name: str):
        if not self.active:
            yield
            return
        import torch

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events[name] = (start, end)

    def finish(self, *, successful: bool) -> dict[str, float]:
        if not self.active:
            return {}
        self.active = False
        if not successful:
            self.events = {}
            return {}
        import torch

        torch.cuda.synchronize(self.device)
        result = {
            f"{name}_ms": float(start.elapsed_time(end))
            for name, (start, end) in self.events.items()
        }
        result.update(
            {
                "data_load_ms": self.data_load_ms,
                "train_step_wall_ms": (time.perf_counter() - self.train_started)
                * 1000.0,
                "iteration_wall_ms": (time.perf_counter() - self.iteration_started)
                * 1000.0,
            }
        )
        self.events = {}
        return result


def reduce_timing_metrics(
    accelerator: Any,
    metrics: Mapping[str, float],
) -> dict[str, float]:
    if not metrics:
        return {}
    import torch

    names = tuple(sorted(metrics))
    values = torch.tensor(
        [float(metrics[name]) for name in names],
        device=accelerator.device,
        dtype=torch.float32,
    )
    if int(accelerator.num_processes) > 1:
        if not torch.distributed.is_initialized():
            raise RuntimeError("training timing reduction requires a process group")
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    reduced = values.cpu().tolist()
    return {
        f"profile/{name}": float(value)
        for name, value in zip(names, reduced, strict=True)
    }
