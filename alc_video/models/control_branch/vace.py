from __future__ import annotations

import torch
import torch.nn as nn
from diffsynth.core.gradient import gradient_checkpoint_forward
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.wan_video_vace import VaceWanModel


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class VaceControlBranch(VaceWanModel):
    def __init__(
        self,
        dit: WanModel,
        *,
        control_in_dim: int,
        control_layers: tuple[int, ...],
        architecture: dict[str, object],
    ) -> None:
        if not isinstance(control_layers, tuple) or any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in control_layers
        ):
            raise TypeError("control_layers must be a tuple of integers")
        num_layers = int(architecture["num_layers"])
        if not control_layers:
            raise ValueError("control_layers must not be empty")
        if any(layer < 0 or layer >= num_layers for layer in control_layers):
            raise ValueError(
                f"control_layers {control_layers} out of range for {num_layers} blocks"
            )
        if list(control_layers) != sorted(set(control_layers)):
            raise ValueError("control_layers must be strictly increasing and unique")
        if control_layers[0] != 0:
            raise ValueError(
                "control_layers must include layer 0 to seed the control branch"
            )
        super().__init__(
            vace_layers=control_layers,
            vace_in_dim=control_in_dim,
            patch_size=tuple(architecture["patch_size"]),
            has_image_input=False,
            dim=int(architecture["dim"]),
            num_heads=int(architecture["num_heads"]),
            ffn_dim=int(architecture["ffn_dim"]),
            eps=float(architecture["eps"]),
        )
        self._initialize_from_dit(dit)
        self.to(
            device=dit.patch_embedding.weight.device,
            dtype=dit.patch_embedding.weight.dtype,
        )

    def _initialize_from_dit(self, dit: WanModel) -> None:
        for layer_index, control_block in zip(
            self.vace_layers,
            self.vace_blocks,
            strict=True,
        ):
            source = dit.blocks[layer_index]
            missing, unexpected = control_block.load_state_dict(
                source.state_dict(), strict=False
            )
            invalid_missing = [
                name
                for name in missing
                if not (name.startswith("before_proj") or name.startswith("after_proj"))
            ]
            if invalid_missing or unexpected:
                raise RuntimeError(
                    f"control block for layer {layer_index} state mismatch: "
                    f"missing={invalid_missing}, unexpected={unexpected}"
                )
            if layer_index == 0:
                if not hasattr(control_block, "before_proj"):
                    raise RuntimeError("control block 0 is missing before_proj")
            _zero_module(control_block.after_proj)

    def _condition_tokens(
        self,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, int, int]:
        if not isinstance(condition, torch.Tensor) or condition.ndim != 5:
            raise ValueError("condition must be a 5D tensor")
        if not torch.is_floating_point(condition):
            raise TypeError("condition must be floating point")
        patch_weight = self.vace_patch_embedding.weight
        if (
            condition.device != patch_weight.device
            or condition.dtype != patch_weight.dtype
        ):
            raise ValueError("condition must match the control branch device and dtype")

        control = self.vace_patch_embedding(condition)
        batch, dim, frames, patch_height, patch_width = (
            int(size) for size in control.shape
        )
        spatial_tokens = patch_height * patch_width
        control = (
            control.permute(0, 2, 3, 4, 1)
            .reshape(batch, frames, spatial_tokens, dim)
            .contiguous()
        )

        return control, batch, frames, spatial_tokens, dim

    def _prepare_control_tokens(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        context: torch.Tensor,
        time_modulation: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, int, int]:
        control, batch, frames, spatial_tokens, dim = self._condition_tokens(condition)
        expected_hidden = (batch, frames * spatial_tokens, dim)
        if tuple(int(size) for size in hidden.shape) != expected_hidden:
            raise ValueError(
                f"control hidden must be {expected_hidden}, got {tuple(hidden.shape)}"
            )
        if (
            context.ndim != 3
            or int(context.shape[0]) != batch
            or int(context.shape[2]) != dim
        ):
            raise ValueError(
                f"control context must be [B,S,{dim}] with B={batch}, "
                f"got {tuple(context.shape)}"
            )
        expected_modulation = (batch, 6, dim)
        if tuple(int(size) for size in time_modulation.shape) != expected_modulation:
            raise ValueError(
                f"control time modulation must be {expected_modulation}, "
                f"got {tuple(time_modulation.shape)}"
            )
        return control, batch, frames, spatial_tokens, dim

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        context: torch.Tensor,
        time_modulation: torch.Tensor,
        freqs: torch.Tensor,
        *,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> tuple[torch.Tensor, ...]:
        control, batch, frames, spatial_tokens, dim = self._prepare_control_tokens(
            hidden,
            condition,
            context,
            time_modulation,
        )
        state = control.reshape(batch, frames * spatial_tokens, dim)
        for block in self.vace_blocks:
            state = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                state,
                hidden,
                context,
                time_modulation,
                freqs,
            )
        return tuple(torch.unbind(state)[:-1])
