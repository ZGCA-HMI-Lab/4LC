from __future__ import annotations

from typing import Any

import torch


def build_optimizer(
    model: torch.nn.Module,
    cfg: Any,
) -> torch.optim.AdamW:
    optimizer_cfg = cfg.optimizer
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("optimizer received no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=optimizer_cfg.lr,
        betas=optimizer_cfg.betas,
        weight_decay=optimizer_cfg.weight_decay,
        eps=optimizer_cfg.eps,
    )


class ConstantLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        base_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.base_lr = float(base_lr)
        self.step_count = 0
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.base_lr <= 0.0:
            raise ValueError("base_lr must be > 0")
        self._apply(self.base_lr)

    def lr_at(self, step: int) -> float:
        if not 0 <= int(step) <= self.total_steps:
            raise ValueError("scheduler step is out of range")
        return self.base_lr

    def _apply(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def step(self) -> None:
        if self.step_count >= self.total_steps:
            raise RuntimeError("learning-rate scheduler stepped past total_steps")
        self.step_count += 1
        self._apply(self.lr_at(self.step_count))

    def state_dict(self) -> dict[str, int]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        step_count = int(state_dict["step_count"])
        if not 0 <= step_count <= self.total_steps:
            raise ValueError(f"scheduler step_count is out of range: {step_count}")
        self.step_count = step_count
        self._apply(self.lr_at(step_count))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
) -> ConstantLRScheduler:
    total_steps = cfg.training.max_steps
    optimizer_cfg = cfg.optimizer
    if optimizer_cfg.schedule != "constant":
        raise ValueError("optimizer schedule must be constant")
    return ConstantLRScheduler(
        optimizer,
        total_steps=total_steps,
        base_lr=optimizer_cfg.lr,
    )


def resolve_gradient_clip(
    cfg: Any,
) -> tuple[float, float]:
    return cfg.optimizer.clip_max_norm, cfg.optimizer.clip_norm_type
