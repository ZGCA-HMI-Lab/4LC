from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


def worldmirror_runtime_config(
    checkpoint: Path,
    device: torch.device,
    *,
    dynamic_threshold: float = 0.0,
    enable_global_motion_tracking: bool = False,
    dynamic_threshold2: float = 0.0,
) -> tuple[Any, Path]:
    thresholds = (float(dynamic_threshold), float(dynamic_threshold2))
    if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
        raise ValueError("WorldMirror dynamic thresholds must be finite and >= 0")
    if not isinstance(enable_global_motion_tracking, bool):
        raise TypeError("enable_global_motion_tracking must be bool")
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"WorldMirror checkpoint does not exist: {checkpoint}")
    runtime_config = {
        "reconstructor": {
            "name": "worldmirror",
            "strict_load": True,
            "kwargs": {
                "enable_norm": False,
                "dynamic_threshold": thresholds[0],
                "enable_global_motion_tracking": enable_global_motion_tracking,
                "dynamic_threshold2": thresholds[1],
            },
        },
        "runtime": {"dtype": "bfloat16"},
        "data": {
            "image_size": [336, 560],
            "patch_size": 14,
            "latent_grid": [24, 40],
        },
    }
    config = OmegaConf.create(runtime_config)
    config.reconstructor.path = str(checkpoint)
    config.runtime.device = str(device)
    return config, checkpoint
