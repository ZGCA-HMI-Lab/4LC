from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch
from latent_cache import Readout

from alc_video.utils.logging import LOG

if TYPE_CHECKING:
    from alc_video.models.model import WanConditionedModel


def validate_sampling_options(num_inference_steps, control_scales) -> None:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be > 0")
    if control_scales is not None and (
        not isinstance(control_scales, tuple)
        or len(control_scales) != num_inference_steps
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in control_scales
        )
    ):
        raise ValueError(
            "control_scales must contain one finite value per inference step"
        )


@torch.inference_mode()
def generate_latents_from_generator(
    network: WanConditionedModel,
    text_context: torch.Tensor,
    readouts: Mapping[str, Readout],
    target_intrinsics: torch.Tensor,
    target_c2w: torch.Tensor,
    *,
    image_size: tuple[int, int],
    latent_shape: tuple[int, ...],
    num_inference_steps: int,
    generator: torch.Generator,
    control_scales: tuple[float, ...] | None = None,
    feature_keep: tuple[bool, ...] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    from alc_video.models.model import WanConditionedModel

    if not isinstance(network, WanConditionedModel):
        raise TypeError(
            f"network must be WanConditionedModel, got {type(network).__name__}"
        )
    validate_sampling_options(num_inference_steps, control_scales)
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be torch.Generator")
    if generator.device.type != "cpu":
        raise ValueError("latent generator must use the CPU device")
    condition = network.prepare_condition(
        readouts,
        target_intrinsics,
        target_c2w,
        latent_shape,
        image_size=image_size,
        feature_keep=feature_keep,
    )
    latents = torch.randn(
        latent_shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=text_context.device, dtype=text_context.dtype)

    from diffsynth.diffusion import FlowMatchScheduler

    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(int(num_inference_steps), shift=5.0)
    original_control_scale: float | None = None
    if control_scales is not None:
        original_control_scale = float(network.control_scale)
    try:
        for step_index, timestep in enumerate(scheduler.timesteps):
            if progress is not None:
                progress(step_index, num_inference_steps)
            LOG.debug(
                "diffusion progress step=%s/%s", step_index + 1, num_inference_steps
            )
            if control_scales is not None:
                network.control_scale = control_scales[step_index]
            prediction = network.denoise(
                latents,
                timestep.to(
                    device=text_context.device,
                    dtype=text_context.dtype,
                ).reshape(1),
                text_context,
                condition,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
            latents = scheduler.step(prediction, timestep, latents)
    finally:
        if original_control_scale is not None:
            network.control_scale = original_control_scale
    if not bool(torch.isfinite(latents).all()):
        raise FloatingPointError("generated Wan latents contain non-finite values")
    if progress is not None:
        progress(num_inference_steps, num_inference_steps)
    return latents


@torch.inference_mode()
def generate_latents(
    network: WanConditionedModel,
    text_context: torch.Tensor,
    readouts: Mapping[str, Readout],
    target_intrinsics: torch.Tensor,
    target_c2w: torch.Tensor,
    *,
    image_size: tuple[int, int],
    latent_shape: tuple[int, ...],
    num_inference_steps: int,
    seed: int,
    feature_keep: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63))
    return generate_latents_from_generator(
        network,
        text_context,
        readouts,
        target_intrinsics,
        target_c2w,
        image_size=image_size,
        latent_shape=latent_shape,
        num_inference_steps=num_inference_steps,
        generator=generator,
        feature_keep=feature_keep,
    )


__all__ = ["generate_latents", "generate_latents_from_generator"]
