"""Load a WorldMirror reconstructor for the local pipeline/datapipe glue use."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from worldmirror.glue.reconstructor import WorldMirrorReconstructor
from worldmirror.glue.tokens import WorldMirrorDepthHeadAdapter, WorldMirrorSingleFrameTokenEncoder


def _load_state_dict(path: str, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def load_reconstructor_module(recon_cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Instantiate + checkpoint-load the raw WorldMirror nn.Module."""
    from worldmirror.models.models.worldmirror import WorldMirror

    from accelerate import init_empty_weights

    path = recon_cfg.get("path", None)
    strict = bool(recon_cfg.get("strict_load", False))
    assign = bool(path) and strict
    # A complete checkpoint owns the parameters; avoid a second full allocation.
    with init_empty_weights(include_buffers=False) if assign else nullcontext():
        reconstructor = WorldMirror(**dict(recon_cfg.get("kwargs", {}) or {}))
    if path:
        state_dict = _load_state_dict(str(path), device="cpu")
        if assign:
            expected = reconstructor.state_dict()
            state_dict = {
                name: tensor.to(dtype=expected[name].dtype) if name in expected else tensor
                for name, tensor in state_dict.items()
            }
        reconstructor.load_state_dict(state_dict, strict=strict, assign=assign)
    return reconstructor.eval().to(device)


def load(recon_cfg: dict[str, Any], device: torch.device) -> WorldMirrorReconstructor:
    """Load WorldMirror and expose the local online reconstructor API."""
    module = load_reconstructor_module(recon_cfg, device)
    return WorldMirrorReconstructor(module, device=device)


def load_token_encoder(
    recon_cfg: dict[str, Any],
    device: torch.device,
    *,
    layer_indices,
    batch_size: int = 16,
) -> WorldMirrorSingleFrameTokenEncoder:
    """Load WorldMirror and expose only the frozen single-frame token encoder."""
    module = load_reconstructor_module(recon_cfg, device)
    return WorldMirrorSingleFrameTokenEncoder(module, layer_indices=layer_indices, batch_size=int(batch_size))


def load_depth_head(
    recon_cfg: dict[str, Any],
    device: torch.device,
    *,
    layer_indices,
    patch_size: int,
) -> WorldMirrorDepthHeadAdapter:
    """Load WorldMirror and expose only the frozen DPT depth head."""
    module = load_reconstructor_module(recon_cfg, device)
    return WorldMirrorDepthHeadAdapter(module, layer_indices=layer_indices, patch_size=int(patch_size))


__all__ = ["load", "load_depth_head", "load_reconstructor_module", "load_token_encoder"]
