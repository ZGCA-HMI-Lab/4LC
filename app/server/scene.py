from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from latent_cache import CacheEntry


@dataclass(frozen=True)
class SceneAsset:
    metadata: dict
    data: bytes


@torch.inference_mode()
def export_scene(entry: CacheEntry, colors: torch.Tensor) -> SceneAsset:
    """Pack RGB geometry once; retain every resolved point and its motion data."""
    static = entry.motion.anchor_mask[0]
    moving = entry.motion.motion_mask[0]
    points = entry.positions[0].float()
    resolved = static | moving
    selected = points[resolved]
    if not selected.numel():
        raise ValueError("The Cache contains no resolved scene points")
    chunks: list[bytes] = []
    size = 0

    def pack(value: torch.Tensor, dtype: str) -> dict:
        nonlocal size
        padding = (-size) % 4
        chunks.append(bytes(padding))
        size += padding
        array = np.ascontiguousarray(value.cpu().numpy(), dtype=dtype)
        result = {"offset": size, "length": array.size}
        chunks.append(array.tobytes())
        size += array.nbytes
        return result

    def cloud(mask):
        return {
            "positions": pack(points[mask], "<f4"),
            "colors": pack(colors[mask], "u1"),
        }

    statics = cloud(static)
    dynamics = cloud(moving)
    if bool(moving.any()):
        if entry.velocity is None:
            raise ValueError("Moving Cache points require velocity data")
        forward, backward = entry.velocity
        dynamics["forward"] = pack(forward[0][moving].float(), "<f4")
        dynamics["backward"] = pack(backward[0][moving].float(), "<f4")
        indices = torch.arange(entry.num_keyframes).reshape(-1, 1, 1).expand_as(moving)
        dynamics["sources"] = pack(indices[moving], "<u2")
    depth = entry.depth[0, 0][entry.valid[0, 0]]
    metadata = {
        "static": statics,
        "dynamic": dynamics,
        "source_times": entry.times[0].float().tolist(),
        "focus_distance": float(depth.float().median()),
        "bounds": [selected.amin(0).tolist(), selected.amax(0).tolist()],
    }
    return SceneAsset(metadata, b"".join(chunks))
