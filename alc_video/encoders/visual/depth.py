from __future__ import annotations

import torch


def depth_normalization_scale(
    depth: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    _validate_depth(depth, valid)
    visible = valid[..., 0]
    depth_float = depth.float()
    return torch.stack(
        [
            torch.quantile(depth_float[index][visible[index]], 0.98)
            for index in range(int(depth.shape[0]))
        ]
    ).clamp_min(torch.finfo(torch.float32).eps)


def depth_to_rgb_frames(
    depth: torch.Tensor,
    valid: torch.Tensor,
    *,
    normalization_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_depth(depth, valid)
    batch = int(depth.shape[0])
    if normalization_scale is None:
        scale = depth_normalization_scale(depth, valid)
    else:
        if (
            not isinstance(normalization_scale, torch.Tensor)
            or tuple(normalization_scale.shape) != (batch,)
            or not torch.is_floating_point(normalization_scale)
            or normalization_scale.device != depth.device
            or not bool(torch.isfinite(normalization_scale).all())
            or not bool((normalization_scale > 0).all())
        ):
            raise ValueError(
                "normalization_scale must be one positive finite value per sample"
            )
        scale = normalization_scale.float()
    visible = valid[..., 0]
    normalized = depth.float() / scale[:, None, None, None]
    normalized = normalized.masked_fill(~visible, 0).clamp_(0, 1)
    return normalized[..., None].expand(-1, -1, -1, -1, 3).contiguous()


def _validate_depth(depth: torch.Tensor, valid: torch.Tensor) -> None:
    if not isinstance(depth, torch.Tensor) or depth.ndim != 4:
        raise ValueError("source depth must be [B,V,H,W]")
    if (
        not isinstance(valid, torch.Tensor)
        or valid.ndim != 5
        or int(valid.shape[-1]) != 1
        or tuple(valid.shape[:-1]) != tuple(depth.shape)
        or valid.dtype != torch.bool
    ):
        raise ValueError("source depth validity must be bool [B,V,H,W,1]")
    if not torch.is_floating_point(depth):
        raise TypeError("source depth must be floating point")
    if depth.device != valid.device:
        raise ValueError("source depth and validity must share one device")
    if not bool(torch.isfinite(depth).all()):
        raise ValueError("source depth contains non-finite values")
    visible = valid[..., 0]
    batch = int(depth.shape[0])
    if any(not bool(visible[index].any()) for index in range(batch)):
        raise ValueError("each sample must contain at least one valid depth pixel")
    if not bool((depth[visible] > 0).all()):
        raise ValueError("valid source depth must be positive")


__all__ = ["depth_normalization_scale", "depth_to_rgb_frames"]
