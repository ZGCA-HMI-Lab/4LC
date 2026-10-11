from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
from diffsynth.models.wan_video_dit import WanModel
from latent_cache import Readout

from alc_video.models.adapter import ConditionAdapter
from alc_video.models.adapter.alignment import validate_feature_shapes
from alc_video.models.control_branch.vace import VaceControlBranch
from alc_video.models.video_model.dit import (
    denoise_with_blocks,
    validate_wan21_t2v,
    wan_architecture,
)
from alc_video.models.video_model.temporal import target_latent_shape


class WanConditionedModel(nn.Module):
    LATENT_CHANNELS = 16

    def __init__(
        self,
        dit: WanModel,
        *,
        architecture: str,
        feature_shapes: tuple[tuple[tuple[int, int], int], ...],
        feature_dims: tuple[int, ...],
        mask_dim: int,
        camera_dim: int,
        output_dim: int,
        control_scale: float,
        control_layers: tuple[int, ...],
        use_motion_mask: bool = False,
    ) -> None:
        super().__init__()
        self.wan_architecture = validate_wan21_t2v(dit, expected=architecture)
        self.dit = dit
        if isinstance(control_scale, bool) or not isinstance(
            control_scale, (int, float)
        ):
            raise TypeError("control_scale must be numeric")
        self.control_scale = float(control_scale)
        if not math.isfinite(self.control_scale):
            raise ValueError("control_scale must be finite")
        self.feature_shapes = validate_feature_shapes(feature_shapes)
        self.representation_count = len(self.feature_shapes)
        if not isinstance(use_motion_mask, bool):
            raise TypeError("use_motion_mask must be bool")
        self.use_motion_mask = bool(use_motion_mask)
        self.condition_adapter = ConditionAdapter(
            feature_shapes=self.feature_shapes,
            feature_dims=feature_dims,
            mask_dim=mask_dim,
            camera_dim=camera_dim,
            output_dim=output_dim,
            use_motion_mask=self.use_motion_mask,
        )
        self.control_in_dim = self.condition_adapter.control_in_dim
        architecture_shape = wan_architecture(dit)
        self.control_layers = control_layers
        self.control_branch = VaceControlBranch(
            dit,
            control_layers=control_layers,
            control_in_dim=self.control_in_dim,
            architecture=architecture_shape,
        )
        if wan_architecture(self.dit) != architecture_shape:
            raise RuntimeError("the control branch changed the Wan architecture")
        self.condition_adapter.to(
            device=self.dit.patch_embedding.weight.device,
            dtype=self.dit.patch_embedding.weight.dtype,
        )

    @property
    def latent_channels(self) -> int:
        return self.LATENT_CHANNELS

    @property
    def text_dim(self) -> int:
        return int(self.dit.text_embedding[0].in_features)

    def prepare_condition(
        self,
        readouts: Mapping[str, Readout],
        target_intrinsics: torch.Tensor,
        target_c2w: torch.Tensor,
        target_latent_shape: tuple[int, ...],
        *,
        image_size: tuple[int, int],
        feature_keep: tuple[bool, ...] | None = None,
    ) -> torch.Tensor:
        return self.condition_adapter(
            readouts,
            target_intrinsics,
            target_c2w,
            target_latent_shape,
            image_size=image_size,
            feature_keep=feature_keep,
        )

    def denoise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        text_context: torch.Tensor,
        condition: torch.Tensor,
        *,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> torch.Tensor:
        batch, _channels, frames, height, width = target_latent_shape(noisy_latents)
        expected_condition = (batch, self.control_in_dim, frames, height, width)
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a tensor")
        if not torch.is_floating_point(condition):
            raise TypeError("condition must be floating point")
        if tuple(int(size) for size in condition.shape) != expected_condition:
            raise ValueError(
                f"condition must be {expected_condition}, got {tuple(condition.shape)}"
            )
        if (
            condition.device != noisy_latents.device
            or condition.dtype != noisy_latents.dtype
        ):
            raise ValueError("condition must match noisy latents device and dtype")

        mapping = self.control_branch.vace_layers_mapping
        control_scale = self.control_scale

        def prepare_control(hidden, embedded_text, time_modulation, freqs):
            return self.control_branch(
                hidden,
                condition,
                embedded_text,
                time_modulation,
                freqs,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        def inject_control(index, hidden, hints):
            position = mapping.get(index)
            if position is not None:
                hidden = hidden + hints[position] * control_scale
            return hidden

        return denoise_with_blocks(
            self.dit,
            noisy_latents,
            timestep,
            text_context,
            latent_channels=self.LATENT_CHANNELS,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            prepare_control=prepare_control,
            inject_control=inject_control,
        )
