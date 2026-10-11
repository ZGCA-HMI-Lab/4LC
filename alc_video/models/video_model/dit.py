from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from einops import rearrange

if TYPE_CHECKING:
    from diffsynth.models.wan_video_dit import WanModel


WAN21_T2V_1_3B = "wan21-t2v-1.3b"
WAN21_T2V_14B = "wan21-t2v-14b"
WAN21_T2V_ARCHITECTURES = (WAN21_T2V_1_3B, WAN21_T2V_14B)

WAN21_T2V_1_3B_ARCHITECTURE = {
    "in_dim": 16,
    "dim": 1536,
    "ffn_dim": 8960,
    "out_dim": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "eps": 1e-6,
    "patch_size": (1, 2, 2),
    "num_heads": 12,
    "num_layers": 30,
    "has_image_input": False,
}

WAN21_T2V_14B_ARCHITECTURE = {
    "in_dim": 16,
    "dim": 5120,
    "ffn_dim": 13824,
    "out_dim": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "eps": 1e-6,
    "patch_size": (1, 2, 2),
    "num_heads": 40,
    "num_layers": 40,
    "has_image_input": False,
}


def wan_architecture(model: WanModel) -> dict[str, object]:
    from diffsynth.models.wan_video_dit import WanModel

    if type(model) is not WanModel:
        actual = f"{type(model).__module__}.{type(model).__name__}"
        raise TypeError(f"Wan backbone must be DiffSynth WanModel, got {actual}")
    if not model.blocks:
        raise ValueError("Wan backbone has no transformer blocks")
    first_block = model.blocks[0]
    return {
        "in_dim": int(model.patch_embedding.in_channels),
        "dim": int(model.dim),
        "ffn_dim": int(first_block.ffn[0].out_features),
        "out_dim": int(model.head.head.out_features // math.prod(model.patch_size)),
        "text_dim": int(model.text_embedding[0].in_features),
        "freq_dim": int(model.time_embedding[0].in_features),
        "eps": float(first_block.norm1.eps),
        "patch_size": tuple(int(value) for value in model.patch_size),
        "num_heads": int(first_block.self_attn.num_heads),
        "num_layers": len(model.blocks),
        "has_image_input": bool(model.has_image_input),
    }


def validate_wan21_t2v(
    model: WanModel,
    *,
    expected: str,
) -> str:
    if expected not in WAN21_T2V_ARCHITECTURES:
        raise ValueError(
            f"Wan architecture must be one of {WAN21_T2V_ARCHITECTURES}, "
            f"got {expected!r}"
        )
    actual = wan_architecture(model)
    expected_shape = {
        WAN21_T2V_1_3B: WAN21_T2V_1_3B_ARCHITECTURE,
        WAN21_T2V_14B: WAN21_T2V_14B_ARCHITECTURE,
    }[expected]
    if actual != expected_shape:
        raise ValueError(
            f"configured Wan architecture {expected!r} does not match loaded model; "
            f"expected {expected_shape}, got {actual}"
        )
    unsupported_extensions = [
        name
        for name in (
            "has_ref_conv",
            "wantodance_enable_music_inject",
            "wantodance_enable_refimage",
            "wantodance_enable_refface",
            "wantodance_enable_global",
            "wantodance_enable_dynamicfps",
            "wantodance_enable_unimodel",
        )
        if getattr(model, name, False)
    ]
    if getattr(model, "control_adapter", None) is not None:
        unsupported_extensions.append("control_adapter")
    if unsupported_extensions:
        raise ValueError(
            "4LC-Video only accepts the plain Wan2.1 T2V backbone; "
            f"loaded extensions={unsupported_extensions}"
        )
    return expected


def validate_denoise_inputs(
    dit: WanModel,
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor,
    text_context: torch.Tensor,
    *,
    latent_channels: int,
) -> torch.Tensor:
    if noisy_latents.ndim != 5 or int(noisy_latents.shape[1]) != int(latent_channels):
        raise ValueError(
            f"noisy_latents must be [B,{latent_channels},F,H,W], "
            f"got {tuple(noisy_latents.shape)}"
        )
    if not torch.is_floating_point(noisy_latents):
        raise TypeError("noisy_latents must be floating point")
    bad_axes = [
        f"{axis}={int(size)}"
        for axis, size, patch in zip(
            ("F", "H", "W"), noisy_latents.shape[2:], dit.patch_size, strict=False
        )
        if int(size) % int(patch) != 0
    ]
    if bad_axes:
        raise ValueError(
            f"Wan latent shape is not patch-aligned: {', '.join(bad_axes)}"
        )
    batch = int(noisy_latents.shape[0])
    expected_text_dim = int(dit.text_embedding[0].in_features)
    if (
        text_context.ndim != 3
        or int(text_context.shape[0]) != batch
        or int(text_context.shape[-1]) != expected_text_dim
    ):
        raise ValueError(
            f"text_context must be [B,S,{expected_text_dim}] with B={batch}, "
            f"got {tuple(text_context.shape)}"
        )
    if not torch.is_floating_point(text_context) or not torch.is_floating_point(
        timestep
    ):
        raise TypeError("text_context and timestep must be floating point")
    timestep = timestep.flatten()
    if int(timestep.numel()) not in (1, batch):
        raise ValueError(f"timestep must contain one value or B={batch} values")
    return timestep


def patchify_wan(
    dit: WanModel, value: torch.Tensor
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    value = dit.patch_embedding(value)
    grid = tuple(int(size) for size in value.shape[2:])
    for axis, size, freqs in zip(("F", "H", "W"), grid, dit.freqs, strict=False):
        if size > int(freqs.shape[0]):
            raise ValueError(
                f"patch grid {axis}={size} exceeds Wan rotary cache length "
                f"{int(freqs.shape[0])}"
            )
    return rearrange(value, "b c f h w -> b (f h w) c").contiguous(), grid


def wan_rotary_frequencies(
    dit: WanModel,
    grid: tuple[int, int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    frames, height, width = grid
    return (
        torch.cat(
            [
                dit.freqs[0][:frames]
                .view(frames, 1, 1, -1)
                .expand(frames, height, width, -1),
                dit.freqs[1][:height]
                .view(1, height, 1, -1)
                .expand(frames, height, width, -1),
                dit.freqs[2][:width]
                .view(1, 1, width, -1)
                .expand(frames, height, width, -1),
            ],
            dim=-1,
        )
        .reshape(frames * height * width, 1, -1)
        .to(device)
    )


def denoise_with_blocks(
    dit: WanModel,
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor,
    text_context: torch.Tensor,
    *,
    latent_channels: int,
    use_gradient_checkpointing: bool,
    use_gradient_checkpointing_offload: bool,
    prepare_control: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Any
    ]
    | None = None,
    inject_control: Callable[[int, torch.Tensor, Any], torch.Tensor] | None = None,
) -> torch.Tensor:
    timestep = validate_denoise_inputs(
        dit,
        noisy_latents,
        timestep,
        text_context,
        latent_channels=latent_channels,
    )
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

    time_embedding = dit.time_embedding(
        sinusoidal_embedding_1d(dit.freq_dim, timestep).to(noisy_latents.dtype)
    )
    time_modulation = dit.time_projection(time_embedding).unflatten(1, (6, dit.dim))
    embedded_text = dit.text_embedding(text_context)
    hidden, grid = patchify_wan(dit, noisy_latents)
    freqs = wan_rotary_frequencies(dit, grid, device=hidden.device)

    control_state = (
        prepare_control(hidden, embedded_text, time_modulation, freqs)
        if prepare_control is not None
        else None
    )
    for index, block in enumerate(dit.blocks):
        if dit.training:
            from diffsynth.core.gradient import gradient_checkpoint_forward

            hidden = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                hidden,
                embedded_text,
                time_modulation,
                freqs,
            )
        else:
            hidden = block(hidden, embedded_text, time_modulation, freqs)
        if inject_control is not None:
            hidden = inject_control(index, hidden, control_state)
    return dit.unpatchify(dit.head(hidden, time_embedding), grid)


__all__ = [
    "WAN21_T2V_1_3B",
    "WAN21_T2V_1_3B_ARCHITECTURE",
    "WAN21_T2V_14B",
    "WAN21_T2V_14B_ARCHITECTURE",
    "WAN21_T2V_ARCHITECTURES",
    "denoise_with_blocks",
    "patchify_wan",
    "validate_denoise_inputs",
    "validate_wan21_t2v",
    "wan_architecture",
    "wan_rotary_frequencies",
]
