from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from latent_cache import LatentCacheConfig

from alc_video.data.schema import (
    CONTEXT_MAX,
    CONTEXT_MIN,
)
from alc_video.models.adapter.alignment import align_wan_coverage, readout_stride
from alc_video.models.checkpoint import TRAINABLE_MODULES
from alc_video.models.model import WanConditionedModel
from alc_video.training.augmentation import (
    AugmentationExecutor,
)
from alc_video.training.augmentation.trace import augmentation_training_metrics
from alc_video.training.objective.conditioning import prepare_training_readouts
from alc_video.training.objective.encoding import encode_training_batch
from alc_video.training.objective.flow_matching import (
    flow_matching_from_condition,
    training_profile_statistics,
)


def _validate_training_batch(
    conditioned_model: WanConditionedModel,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    source_times = batch["source_times"]
    if not isinstance(source_times, torch.Tensor):
        raise TypeError("source_times must be a tensor")
    if source_times.ndim != 2:
        raise ValueError("source_times must be [B,K]")
    batch_size, context_count = (
        int(source_times.shape[0]),
        int(source_times.shape[1]),
    )
    if not CONTEXT_MIN <= context_count <= CONTEXT_MAX:
        raise ValueError("training batch context count is outside the contract")
    if batch.get("transport") not in {"identity", "velocity"}:
        raise ValueError(
            f"training batch transport is invalid: {batch.get('transport')!r}"
        )
    target_latents = batch["target_latents"]
    context = batch["context"]
    if not isinstance(target_latents, torch.Tensor) or not isinstance(
        context, torch.Tensor
    ):
        raise TypeError("target_latents and context must be tensors")
    if (
        target_latents.ndim != 5
        or int(target_latents.shape[1]) != conditioned_model.latent_channels
    ):
        raise ValueError(
            f"target_latents must be [B,{conditioned_model.latent_channels},F,H,W], "
            f"got {tuple(target_latents.shape)}"
        )
    if context.ndim != 3 or int(context.shape[0]) != int(target_latents.shape[0]):
        raise ValueError(
            f"context must be [B,S,C] for B={int(target_latents.shape[0])}"
        )
    if int(context.shape[-1]) != conditioned_model.text_dim:
        raise ValueError(
            f"context hidden dim must be {conditioned_model.text_dim}, "
            f"got {int(context.shape[-1])}"
        )
    if not torch.is_floating_point(target_latents) or not torch.is_floating_point(
        context
    ):
        raise TypeError("target_latents and context must be floating-point tensors")
    source_rgb = batch["source_rgb"]
    if not isinstance(source_rgb, torch.Tensor):
        raise TypeError("source_rgb must be a tensor")
    views = int(source_rgb.shape[1]) if source_rgb.ndim == 5 else -1
    if (
        source_rgb.ndim != 5
        or int(source_rgb.shape[0]) != batch_size
        or not CONTEXT_MIN <= views <= CONTEXT_MAX
        or int(source_rgb.shape[-1]) != 3
    ):
        raise ValueError(
            f"source_rgb must be [B,K,H,W,3] with {CONTEXT_MIN} <= K <= {CONTEXT_MAX}"
        )
    return target_latents, context


class TrainingModule(torch.nn.Module):
    def __init__(
        self,
        *,
        conditioned_model: WanConditionedModel,
        cache_config: LatentCacheConfig,
        encoders: Mapping[str, tuple[Any, str]],
        worldmirror: Any,
        video_codec: Any,
        text_encoder: Any,
        augmentation: AugmentationExecutor | None = None,
        unconditional_context: torch.Tensor | None = None,
        diffusion_scheduler: Any | None = None,
        gradient_checkpointing: bool = True,
        propagate_rgb: bool = False,
    ) -> None:
        super().__init__()
        self.conditioned_model = conditioned_model
        self.cache_config = cache_config
        self.feature_names = tuple(encoders)
        object.__setattr__(self, "_encoders", dict(encoders))
        for name, resource in (
            ("_worldmirror", worldmirror),
            ("_video_codec", video_codec),
            ("_text_encoder", text_encoder),
            ("_augmentation", augmentation),
        ):
            object.__setattr__(self, name, resource)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if not self.gradient_checkpointing:
            raise ValueError("gradient checkpointing must be enabled")
        if diffusion_scheduler is None:
            from diffsynth.diffusion import FlowMatchScheduler

            diffusion_scheduler = FlowMatchScheduler("Wan")
        diffusion_scheduler.set_timesteps(1000, training=True)
        self.diffusion_scheduler = diffusion_scheduler
        if not isinstance(propagate_rgb, bool):
            raise TypeError("propagate_rgb must be bool")
        self.propagate_rgb = propagate_rgb
        if augmentation is not None and not isinstance(
            augmentation, AugmentationExecutor
        ):
            raise TypeError("augmentation must be AugmentationExecutor or None")
        prompt_enabled = (
            augmentation is not None
            and augmentation.config.condition.prompt_drop.probability > 0.0
        )
        if prompt_enabled:
            if (
                not isinstance(unconditional_context, torch.Tensor)
                or tuple(unconditional_context.shape) != (512, 4096)
                or unconditional_context.dtype != torch.bfloat16
                or not bool(torch.isfinite(unconditional_context).all())
            ):
                raise ValueError(
                    "enabled prompt drop requires finite bfloat16 unconditional context [512,4096]"
                )
        elif unconditional_context is not None:
            raise ValueError("unconditional_context requires enabled prompt drop")
        self.register_buffer(
            "_unconditional_context", unconditional_context, persistent=False
        )

    def forward(
        self,
        batch: dict[str, Any],
        *,
        augmentation_step: int | None = None,
        collect_profile_metrics: bool = False,
        collect_timing_metrics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(collect_profile_metrics, bool) or not isinstance(
            collect_timing_metrics, bool
        ):
            raise TypeError("profile and timing switches must be bool")
        batch = encode_training_batch(
            batch,
            worldmirror=self._worldmirror,
            video_codec=self._video_codec,
            text_encoder=self._text_encoder,
        )
        target_latents, context = _validate_training_batch(
            self.conditioned_model, batch
        )
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        target_latents = target_latents.to(
            device=device, dtype=dtype, non_blocking=True
        )
        context = context.to(device=device, dtype=dtype, non_blocking=True)
        executor = self._augmentation
        if augmentation_step is not None and (
            isinstance(augmentation_step, bool)
            or not isinstance(augmentation_step, int)
            or augmentation_step < 0
        ):
            raise ValueError("augmentation_step must be nonnegative")
        trace = None
        if executor is not None:
            if augmentation_step is None:
                raise ValueError(
                    "active augmentation requires a nonnegative augmentation_step"
                )
            trace = executor.decide(
                sample_id=batch["sample_id"],
                sampling_seed=batch["sampling_seed"],
                global_step=augmentation_step,
                source_count=int(batch["source_times"].shape[1]),
                source_latent_count=len(self.feature_names),
            )
            context = executor.apply_text_context(
                context, trace, unconditional_context=self._unconditional_context
            )
        readouts, trace, timings = prepare_training_readouts(
            batch,
            trace,
            visual_encoders=self._encoders,
            cache_config=self.cache_config,
            propagate_rgb=self.propagate_rgb,
            device=device,
            collect_timing_metrics=collect_timing_metrics,
        )
        state = None if executor is None else executor.condition_augmentation(trace)
        keep = None if state is None else state.feature_keep
        hit = None
        if collect_profile_metrics:
            available = tuple(
                align_wan_coverage(
                    readouts[name].coverage,
                    target_latents,
                    grid=tuple(readouts[name].features.shape[2:4]),
                    readout_stride=readout_stride(
                        readouts[name].target_indices,
                        int(batch["target_times"].shape[1]),
                    ),
                )
                for index, name in enumerate(self.feature_names)
                if keep is None or keep[index]
            )
            hit = (
                torch.stack(available, dim=0).amax(dim=0)
                if available
                else torch.zeros_like(
                    align_wan_coverage(
                        readouts[self.feature_names[0]].coverage,
                        target_latents,
                        grid=tuple(readouts[self.feature_names[0]].features.shape[2:4]),
                        readout_stride=readout_stride(
                            readouts[self.feature_names[0]].target_indices,
                            int(batch["target_times"].shape[1]),
                        ),
                    )
                )
            )
        condition = self.conditioned_model.prepare_condition(
            readouts,
            batch["target_intr"].to(device),
            batch["target_c2w"].to(device),
            tuple(target_latents.shape),
            image_size=self.cache_config.image_size,
            feature_keep=keep,
        )
        flow = flow_matching_from_condition(
            self.conditioned_model,
            target_latents,
            context,
            condition,
            self.diffusion_scheduler,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        loss = flow.loss
        metrics = {"loss": loss.detach(), "flow_mse": flow.flow_mse.detach()}
        metrics.update(
            {
                name: loss.detach().new_tensor(value)
                for name, value in augmentation_training_metrics(trace).items()
            }
        )
        if collect_timing_metrics:
            metrics.update(
                {
                    f"profile/{name}": loss.detach().new_tensor(value)
                    for name, value in timings.items()
                }
            )
        if collect_profile_metrics:
            metrics["profile_statistics"] = training_profile_statistics(
                flow.prediction, flow.target, flow.weight, hit
            )
        return loss, metrics


def configure_pretrain(network: torch.nn.Module) -> tuple[str, ...]:
    network.requires_grad_(False)
    for name in TRAINABLE_MODULES:
        network.get_submodule(name).requires_grad_(True)
    return TRAINABLE_MODULES
