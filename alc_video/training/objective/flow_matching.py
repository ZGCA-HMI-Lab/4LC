from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from alc_video.models.model import WanConditionedModel


@dataclass(frozen=True, slots=True)
class FlowMatchingOutput:
    loss: torch.Tensor
    flow_mse: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor


def flow_matching_from_condition(
    model: WanConditionedModel,
    target_latents: torch.Tensor,
    text_context: torch.Tensor,
    condition: torch.Tensor,
    diffusion_scheduler: Any,
    *,
    gradient_checkpointing: bool,
) -> FlowMatchingOutput:
    timestep_index = torch.randint(
        0,
        len(diffusion_scheduler.timesteps),
        (1,),
        device="cpu",
    ).item()
    timestep = diffusion_scheduler.timesteps[timestep_index]
    noise = torch.randn_like(target_latents)
    noisy_latents = diffusion_scheduler.add_noise(target_latents, noise, timestep)
    target = diffusion_scheduler.training_target(target_latents, noise, timestep)
    prediction = model.denoise(
        noisy_latents,
        timestep.to(
            device=target_latents.device,
            dtype=target_latents.dtype,
        ).reshape(1),
        text_context,
        condition,
        use_gradient_checkpointing=gradient_checkpointing,
        use_gradient_checkpointing_offload=False,
    )
    flow_mse = torch.nn.functional.mse_loss(prediction.float(), target.float())
    weight = diffusion_scheduler.training_weight(timestep).to(
        device=target_latents.device,
        dtype=torch.float32,
    )
    loss = flow_mse * weight
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError(
            f"non-finite flow-matching loss: {loss.detach().cpu().item()}"
        )
    return FlowMatchingOutput(
        loss=loss,
        flow_mse=flow_mse,
        prediction=prediction,
        target=target,
        weight=weight,
    )


PROFILE_STATISTIC_NAMES = (
    "actual_hit_weight",
    "position_count",
    "hit_error_sum",
    "non_hit_error_sum",
    "hit_loss_sum",
    "non_hit_loss_sum",
    "hit_value_weight",
    "non_hit_value_weight",
)


def training_profile_statistics(
    prediction: torch.Tensor,
    flow_target: torch.Tensor,
    weight: torch.Tensor,
    actual_hit_coverage: torch.Tensor,
) -> torch.Tensor:
    expected_coverage_shape = (
        int(prediction.shape[0]),
        1,
        int(prediction.shape[2]),
        int(prediction.shape[3]),
        int(prediction.shape[4]),
    )
    if (
        prediction.shape != flow_target.shape
        or tuple(actual_hit_coverage.shape) != expected_coverage_shape
    ):
        raise ValueError("profile hit tensors have inconsistent shapes")
    if (
        not bool(torch.isfinite(actual_hit_coverage).all())
        or bool((actual_hit_coverage < 0).any())
        or bool((actual_hit_coverage > 1).any())
    ):
        raise ValueError("actual hit coverage must be finite in [0,1]")
    squared_error = (
        prediction.detach().float() - flow_target.detach().float()
    ).square()
    hit_error_sum = (squared_error * actual_hit_coverage).sum()
    non_hit_coverage = 1.0 - actual_hit_coverage
    non_hit_error_sum = (squared_error * non_hit_coverage).sum()
    channels = int(prediction.shape[1])
    hit_value_weight = actual_hit_coverage.sum() * channels
    non_hit_value_weight = non_hit_coverage.sum() * channels
    scalar_weight = weight.detach().float().reshape(())
    values = {
        "actual_hit_weight": actual_hit_coverage.sum(),
        "position_count": actual_hit_coverage.new_tensor(
            actual_hit_coverage.numel(),
            dtype=torch.float32,
        ),
        "hit_error_sum": hit_error_sum,
        "non_hit_error_sum": non_hit_error_sum,
        "hit_loss_sum": hit_error_sum * scalar_weight,
        "non_hit_loss_sum": non_hit_error_sum * scalar_weight,
        "hit_value_weight": hit_value_weight,
        "non_hit_value_weight": non_hit_value_weight,
    }
    return torch.stack(tuple(values[name] for name in PROFILE_STATISTIC_NAMES))
