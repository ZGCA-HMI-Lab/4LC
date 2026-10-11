"""WorldMirror token packing + frozen visual token encoder (local integration).

Migrated from ``legacy_geometry.backends.worldmirror`` (token packing) and
``legacy_geometry.token_encoder`` (visual adapter). The visual adapter now wraps an
already-loaded WorldMirror reconstructor instead of a legacy geometry session.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


def _reshape_patch_tokens(tokens: torch.Tensor, height: int, width: int, patch_size: int) -> torch.Tensor:
    bsz, views, patches, channels = tokens.shape
    token_h, token_w = height // patch_size, width // patch_size
    expected = token_h * token_w
    if patches != expected:
        raise ValueError(
            f"Patch-token count mismatch: got {patches}, expected {expected} "
            f"from image {height}x{width} and patch size {patch_size}."
        )
    return tokens.reshape(bsz, views, token_h, token_w, channels)


@torch.no_grad()
def pack_worldmirror_token_list(
    token_list: list[torch.Tensor],
    patch_start_idx: int,
    height: int,
    width: int,
    patch_size: int,
    layer_indices: Iterable[int],
) -> dict[str, Any]:
    patch_groups = []
    for token_tensor in token_list:
        patch_tokens = token_tensor[:, :, patch_start_idx:]
        patch_groups.append(_reshape_patch_tokens(patch_tokens, height, width, patch_size))

    tokens = torch.stack(patch_groups, dim=2)

    return {
        "tokens": tokens,
        "layers": tuple(layer_indices),
        "token_hw": tuple(tokens.shape[-3:-1]),
        "token_dim": int(tokens.shape[-1]),
        "backend": "worldmirror",
    }


class WorldMirrorVisualAdapter(nn.Module):
    """Owns a frozen WorldMirror visual transformer extracted from a reconstructor."""

    def __init__(self, reconstructor: nn.Module) -> None:
        super().__init__()
        visual = getattr(reconstructor, "visual_geometry_transformer", None)
        if not isinstance(visual, nn.Module):
            raise TypeError("WorldMirror reconstructor must expose nn.Module visual_geometry_transformer")
        visual.eval()
        for param in visual.parameters():
            param.requires_grad_(False)
        object.__setattr__(self, "visual_geometry_transformer", visual)

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        self.visual_geometry_transformer._apply(fn)
        self.visual_geometry_transformer.eval()
        return self


class WorldMirrorSingleFrameTokenEncoder(WorldMirrorVisualAdapter):
    """Encode batched RGB frames by forcing WorldMirror VGT into S=1 mode."""

    def __init__(
        self,
        reconstructor: nn.Module,
        *,
        layer_indices: Iterable[int],
        batch_size: int = 16,
    ) -> None:
        super().__init__(reconstructor)
        self.layer_indices = [int(value) for value in layer_indices]
        if not self.layer_indices:
            raise ValueError("WorldMirrorSingleFrameTokenEncoder requires non-empty layer_indices")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"token encoder batch_size must be positive, got {batch_size}")
        self.last_encode_stats: dict[str, Any] = {}

    def _visual_device(self, fallback: torch.device) -> torch.device:
        visual = self.visual_geometry_transformer
        for param in visual.parameters():
            return param.device
        for buffer in visual.buffers():
            return buffer.device
        return fallback

    @torch.no_grad()
    def encode_tokens(self, rgb: torch.Tensor, *, num_layers: int, token_hw: tuple[int, int], token_dim: int) -> torch.Tensor:
        if rgb.ndim != 5:
            raise ValueError(f"rgb must be [B,T,3,H,W], got {tuple(rgb.shape)}")
        bsz, frames, channels, height, width = (int(v) for v in rgb.shape)
        if bsz <= 0 or frames <= 0:
            raise ValueError(f"rgb batch/frame dimensions must be positive, got {tuple(rgb.shape)}")
        if channels != 3:
            raise ValueError(f"rgb channel dimension must be 3, got {channels}")
        if not torch.is_floating_point(rgb):
            raise TypeError("rgb must be floating point")
        if not torch.isfinite(rgb.float()).all():
            raise ValueError("rgb contains non-finite values")
        token_h, token_w = (int(token_hw[0]), int(token_hw[1]))
        if token_h <= 0 or token_w <= 0:
            raise ValueError(f"token_hw must be positive, got {token_hw}")
        expected_patches = token_h * token_w
        if int(num_layers) != len(self.layer_indices):
            raise ValueError(f"num_layers={num_layers} does not match configured layer_indices={self.layer_indices}")

        visual = self.visual_geometry_transformer
        if not hasattr(visual, "intermediate_idxs"):
            raise TypeError("WorldMirror visual_geometry_transformer must expose intermediate_idxs")
        visual.eval()
        total = bsz * frames
        flat = rgb.float().clamp(0.0, 1.0).reshape(total, channels, height, width)
        flat = flat.to(device=self._visual_device(flat.device), dtype=torch.float32)
        old_idxs = list(visual.intermediate_idxs)
        call_shapes: list[tuple[int, ...]] = []
        chunks: list[torch.Tensor] = []
        try:
            visual.intermediate_idxs = list(self.layer_indices)
            for start in range(0, total, self.batch_size):
                chunk = flat[start:start + self.batch_size].unsqueeze(1).contiguous()
                call_shapes.append(tuple(chunk.shape))
                token_list, patch_start_idx, *_unused = visual(chunk, use_motion=False)
                if len(token_list) != int(num_layers):
                    raise ValueError(f"token encoder returned {len(token_list)} layers, expected {num_layers}")
                patch_start = int(patch_start_idx)
                chunk_layers = []
                for layer_idx, layer_tokens in enumerate(token_list):
                    if layer_tokens.ndim != 4:
                        raise ValueError(f"token layer {layer_idx} must be [N,1,P,C], got {tuple(layer_tokens.shape)}")
                    if int(layer_tokens.shape[0]) != int(chunk.shape[0]) or int(layer_tokens.shape[1]) != 1:
                        raise ValueError(
                            f"token layer {layer_idx} shape must start with {(int(chunk.shape[0]), 1)}, got {tuple(layer_tokens.shape)}"
                        )
                    if patch_start < 0 or patch_start > int(layer_tokens.shape[2]):
                        raise ValueError(f"patch_start_idx={patch_start} out of range for token layer {tuple(layer_tokens.shape)}")
                    patch = layer_tokens[:, :, patch_start:].float()
                    if int(patch.shape[2]) != expected_patches:
                        raise ValueError(f"patch tokens={patch.shape[2]} != expected {expected_patches} from token_hw={token_hw}")
                    if int(patch.shape[-1]) != int(token_dim):
                        raise ValueError(f"token dim {patch.shape[-1]} != configured token_dim {token_dim}")
                    if not torch.isfinite(patch).all():
                        raise ValueError(f"token layer {layer_idx} contains non-finite values")
                    chunk_layers.append(patch[:, 0].reshape(int(chunk.shape[0]), token_h, token_w, int(token_dim)))
                chunks.append(torch.stack(chunk_layers, dim=1))
        finally:
            visual.intermediate_idxs = old_idxs
        tokens = torch.cat(chunks, dim=0).reshape(bsz, frames, int(num_layers), token_h, token_w, int(token_dim))
        if not torch.isfinite(tokens.float()).all():
            raise ValueError("encoded tokens contain non-finite values")
        self.last_encode_stats = {
            "input": tuple(rgb.shape),
            "batch_size": self.batch_size,
            "calls": len(call_shapes),
            "call_shapes": call_shapes,
            "use_motion": False,
        }
        return tokens


class WorldMirrorDepthHeadAdapter(nn.Module):
    """Frozen WorldMirror DPT depth head over query-refined latent tokens.

    The wrapped depth head is intentionally kept out of this module's registered
    children: it should move with the local model, and gradients should flow through it to
    the input tokens, but its pretrained weights must not enter model optimizer,
    EMA, or checkpoints.
    """

    def __init__(
        self,
        reconstructor: nn.Module,
        *,
        layer_indices: Iterable[int],
        patch_size: int,
    ) -> None:
        super().__init__()
        module = getattr(reconstructor, "reconstructor", reconstructor)
        depth_head = getattr(module, "depth_head", None)
        if not isinstance(depth_head, nn.Module):
            raise TypeError("WorldMirror reconstructor must expose nn.Module depth_head")
        depth_head.eval()
        for param in depth_head.parameters():
            param.requires_grad_(False)
        object.__setattr__(self, "depth_head", depth_head)
        self.layer_indices = tuple(int(value) for value in layer_indices)
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if len(self.layer_indices) != len(depth_head.projects):
            raise ValueError(
                "WorldMirror depth_head requires exactly "
                f"{len(depth_head.projects)} token layers, got {self.layer_indices}"
            )

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        self.depth_head._apply(fn)
        self.depth_head.eval()
        return self

    def _head_device_dtype(self, fallback: torch.Tensor) -> tuple[torch.device, torch.dtype]:
        for param in self.depth_head.parameters():
            return param.device, param.dtype if torch.is_floating_point(param) else fallback.dtype
        for buffer in self.depth_head.buffers():
            return buffer.device, buffer.dtype if torch.is_floating_point(buffer) else fallback.dtype
        return fallback.device, fallback.dtype

    def _to_token_list(self, z_refined: torch.Tensor) -> list[torch.Tensor]:
        if z_refined.ndim != 6:
            raise ValueError(f"z_refined must be [B,T,G,h,w,C], got {tuple(z_refined.shape)}")
        batch, frames, layers, h, w, channels = (int(v) for v in z_refined.shape)
        if layers != len(self.layer_indices):
            raise ValueError(
                f"z_refined layer count {layers} must match WorldMirror depth layers {self.layer_indices}"
            )
        expected_dim = int(getattr(getattr(self.depth_head, "norm", None), "normalized_shape", (channels,))[0])
        if channels != expected_dim:
            raise ValueError(f"z_refined channel dim {channels} must match WorldMirror depth_head dim {expected_dim}")
        return [
            z_refined[:, :, layer].reshape(batch, frames, h * w, channels).contiguous()
            for layer in range(layers)
        ]

    @staticmethod
    def _scalar_map(value: torch.Tensor, *, name: str, batch: int, frames: int, height: int, width: int) -> torch.Tensor:
        if value.ndim == 5 and tuple(value.shape) == (batch, frames, height, width, 1):
            return value
        if value.ndim == 5 and tuple(value.shape) == (batch, frames, 1, height, width):
            return value.permute(0, 1, 3, 4, 2).contiguous()
        if value.ndim == 4 and tuple(value.shape) == (batch, frames, height, width):
            return value.unsqueeze(-1)
        raise ValueError(
            f"WorldMirror depth_head {name} must be [B,T,H,W,1], [B,T,1,H,W], or [B,T,H,W], "
            f"got {tuple(value.shape)} for {(batch, frames, height, width)}"
        )

    def forward(self, z_refined: torch.Tensor, *, image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        batch, frames, _layers, token_h, token_w, _channels = (int(v) for v in z_refined.shape)
        height, width = (int(image_size[0]), int(image_size[1]))
        expected = (token_h * self.patch_size, token_w * self.patch_size)
        if (height, width) != expected:
            raise ValueError(f"image_size {(height, width)} must equal token grid * patch_size {expected}")

        device, dtype = self._head_device_dtype(z_refined)
        if device != z_refined.device:
            self.depth_head.to(device=z_refined.device)
            device = z_refined.device
        token_list = [tokens.to(device=device, dtype=dtype) for tokens in self._to_token_list(z_refined)]
        images = torch.zeros((batch, frames, 3, height, width), device=device, dtype=dtype)
        depth, conf = self.depth_head(token_list, images=images, patch_start_idx=0)
        depth = self._scalar_map(depth, name="depth", batch=batch, frames=frames, height=height, width=width)
        conf = self._scalar_map(conf, name="conf", batch=batch, frames=frames, height=height, width=width)
        depth = depth.to(device=z_refined.device, dtype=torch.float32).contiguous()
        conf = conf.to(device=z_refined.device, dtype=torch.float32).contiguous()
        valid = torch.isfinite(depth) & (depth > 0.0) & torch.isfinite(conf) & (conf > 0.0)
        return {
            "depth": depth,
            "depth_conf": conf,
            "depth_valid": valid,
        }


__all__ = [
    "pack_worldmirror_token_list",
    "WorldMirrorVisualAdapter",
    "WorldMirrorSingleFrameTokenEncoder",
    "WorldMirrorDepthHeadAdapter",
]
