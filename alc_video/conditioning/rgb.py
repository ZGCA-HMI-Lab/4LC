from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch
from latent_cache import Readout, Representation
from latent_cache.construction import encoder_frames
from latent_cache.types import VisualEncoder


class PixelEncoder:
    def __init__(self, image_size: tuple[int, int]) -> None:
        self.representation = Representation(
            grid=image_size,
            stride=(1, 1),
            offset=(0, 0),
            dim=3,
            frame_layout="selected",
        )

    def encode(self, frames: torch.Tensor, *, device) -> torch.Tensor:
        return frames.to(device)


@torch.no_grad()
def encode_rgb_readouts(
    readouts: Mapping[str, Readout],
    encoders: Mapping[str, VisualEncoder],
) -> dict[str, Readout]:
    rgb = readouts["rgb"]
    frames = rgb.features.masked_fill(~rgb.coverage[..., None], 0).clamp(0, 1)
    _batch, frame_count = frames.shape[:2]
    observations = tuple(range(frame_count))
    result = {}
    for role, encoder in encoders.items():
        if role == "depth":
            result[role] = readouts[role]
            continue
        representation = encoder.representation
        adapted = encoder_frames(representation, frames, observations, frame_count)
        values = encoder.encode(adapted, device=frames.device)
        positions = representation.target_indices(frame_count)
        indices = torch.tensor(positions, device=frames.device)
        if representation.frame_layout != "causal_hold":
            values = values.index_select(1, indices)

        def sampled(name, representation=representation, indices=indices):
            return (
                representation.spatial_grid(rgb.image_size)
                .sample(getattr(rgb, name), name=f"propagated_rgb.{name}")
                .index_select(1, indices)
            )

        hit = sampled("coverage")
        result[role] = replace(
            rgb,
            target_indices=positions,
            features=values,
            coverage=hit,
            moving=sampled("moving") & hit,
            score=sampled("score"),
            source_token=sampled("source_token"),
            source_keyframe=sampled("source_keyframe"),
            time_offset=sampled("time_offset"),
            depth=sampled("depth"),
            target_frame_indices=rgb.target_frame_indices.index_select(1, indices),
            times=rgb.times.index_select(1, indices),
            intrinsics=rgb.intrinsics.index_select(1, indices),
            c2w=rgb.c2w.index_select(1, indices),
        )
    return result
