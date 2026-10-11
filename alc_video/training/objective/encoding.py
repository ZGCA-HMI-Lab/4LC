from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from alc_video.data.schema import (
    EncodedBatch,
    RawBatch,
    collate_training_samples,
    require_sample_batch_size,
)
from alc_video.utils.logging import stage

if TYPE_CHECKING:
    from alc_video.encoders.geometry.worldmirror import WorldMirrorGeometryEncoder
    from alc_video.models.video_model.text import WanTextEncoder
    from alc_video.models.video_model.vae import WanVideoCodec


@torch.no_grad()
def encode_training_clip(
    source_video: torch.Tensor,
    target_video: torch.Tensor | None,
    text: str,
    *,
    frame_indices: torch.Tensor,
    source_positions: torch.Tensor,
    target_positions: torch.Tensor,
    fps: float,
    transport: str,
    worldmirror: WorldMirrorGeometryEncoder,
    video_codec: WanVideoCodec,
    text_encoder: WanTextEncoder,
) -> dict[str, object]:
    target = source_video if target_video is None else target_video
    with stage("encode.target_video", level=logging.DEBUG):
        target_latents = video_codec.encode_video(
            target.index_select(0, target_positions)
        )
    video_codec.offload()
    times = torch.arange(len(frame_indices), dtype=torch.float32) / float(fps)
    with stage("encode.geometry", level=logging.DEBUG):
        worldmirror.reconstructor_module.to(worldmirror.device)
        state = worldmirror.encode_pair(
            source_video,
            target_video,
            frame_indices=frame_indices,
            times=times,
            source_indices=source_positions,
            transport=transport,
            fps=fps,
        )
    worldmirror.offload()
    with stage("encode.text", level=logging.DEBUG):
        text_encoder.text_encoder.to(text_encoder.device)
        context = text_encoder.encode_text(text)
    text_encoder.offload()
    sample = {
        "transport": transport,
        "fps": torch.tensor(float(fps), dtype=torch.float32),
        "source_frame_indices": frame_indices.index_select(0, source_positions),
        "source_positions": source_positions,
        "source_times": times.index_select(0, source_positions),
        "source_camera_intr": state.source_camera.intr,
        "source_camera_c2w": state.source_camera.c2w,
        "source_intr": state.source_camera.intr.index_select(0, source_positions),
        "source_c2w": state.source_camera.c2w.index_select(0, source_positions),
        "source_depth": state.source.depth,
        "source_valid": state.source.valid,
        "source_rgb": source_video.index_select(0, source_positions)
        .permute(0, 2, 3, 1)
        .contiguous(),
        "target_frame_indices": frame_indices.index_select(0, target_positions),
        "target_positions": target_positions,
        "observation_times": times.index_select(0, target_positions),
        "target_times": times.index_select(0, target_positions),
        "target_intr": state.target_camera.intr.index_select(0, target_positions),
        "target_c2w": state.target_camera.c2w.index_select(0, target_positions),
        "target_latents": target_latents,
        "context": context,
    }
    if transport == "velocity":
        if state.source.motion is None:
            raise ValueError("velocity pair omitted source motion")
        sample["velocity_fwd"] = state.source.motion["v_fwd"]
        sample["velocity_bwd"] = state.source.motion["v_bwd"]
    elif transport != "identity" or state.source.motion is not None:
        raise ValueError("pair transport and reconstructed motion differ")
    return sample


@torch.no_grad()
def encode_training_batch(
    batch: RawBatch,
    *,
    worldmirror: WorldMirrorGeometryEncoder,
    video_codec: WanVideoCodec,
    text_encoder: WanTextEncoder,
) -> EncodedBatch:
    require_sample_batch_size(int(batch["source_video"].shape[0]))
    sample = encode_training_clip(
        batch["source_video"][0],
        None,
        batch["text"],
        frame_indices=batch["frame_indices"][0],
        source_positions=batch["source_positions"][0],
        target_positions=batch["target_positions"][0],
        fps=float(batch["fps"][0]),
        transport=batch["transport"],
        worldmirror=worldmirror,
        video_codec=video_codec,
        text_encoder=text_encoder,
    )
    sample["sample_id"] = batch["sample_id"]
    sample["sampling_seed"] = batch["sampling_seed"]
    sample["dataset"] = batch["dataset"]
    return collate_training_samples([sample])
