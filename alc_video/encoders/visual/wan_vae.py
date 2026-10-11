from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from latent_cache import Representation

from alc_video.encoders.visual.base import VisualEncoder
from alc_video.encoders.visual.visualization import FeatureVisualization
from alc_video.models.video_model.vae import WanVideoCodec

WAN_VAE_GRID = (42, 70)
WAN_VAE_CHANNELS = 16
WAN_VAE_IMAGE_SIZE = (336, 560)


class WanVAESourceEncoder(VisualEncoder):
    encoder_type = "wan_vae"
    path_kind = "file"
    representation = Representation((42, 70), (8, 8), (7, 7), 16, "causal_hold")

    def __init__(self, codec: WanVideoCodec) -> None:
        self.codec = codec

    @classmethod
    def from_config(cls, config, *, device, dtype):
        cls.configured_representation(config.levels)
        return cls.from_local_pretrained(config.path, device=device, dtype=dtype)

    @classmethod
    def from_local_pretrained(cls, path, *, device, dtype):
        return cls(
            WanVideoCodec.from_local_checkpoint(path, device=device, dtype=dtype)
        )

    def prepare(self) -> None:
        self.codec.prepare()

    def offload(self) -> None:
        self.codec.offload()

    def _encode(self, frames, *, output_device):
        latent = self.codec.encode_video(frames)
        return (
            latent.permute(1, 2, 3, 0)
            .to(device=output_device, dtype=torch.bfloat16)
            .contiguous()
        )

    def decode(self, latents):
        values = latents.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        decoded = self.codec.decode_video(values, tiled=False)
        return (
            decoded[0]
            .permute(1, 0, 2, 3)
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )

    def _make_visualizer(self, features):
        return _WanVAEVisualizer(self)


@dataclass(slots=True)
class _WanVAEVisualizer:
    encoder: WanVAESourceEncoder

    def render(
        self,
        readout_features: torch.Tensor,
        hit_mask: torch.Tensor,
    ) -> FeatureVisualization:
        expected_tail = (*WAN_VAE_GRID, WAN_VAE_CHANNELS)
        if (
            not isinstance(readout_features, torch.Tensor)
            or readout_features.ndim != 4
            or tuple(int(size) for size in readout_features.shape[1:]) != expected_tail
        ):
            raise ValueError(
                f"Wan VAE query latents must be [T,{expected_tail[0]},"
                f"{expected_tail[1]},{expected_tail[2]}]"
            )
        expected_mask = tuple(int(size) for size in readout_features.shape[:-1])
        if (
            not isinstance(hit_mask, torch.Tensor)
            or hit_mask.dtype != torch.bool
            or tuple(int(size) for size in hit_mask.shape) != expected_mask
        ):
            raise ValueError(f"Wan VAE hit mask must be bool {expected_mask}")
        decoded = self.encoder.decode(readout_features)
        decoded_frames = int(decoded.shape[0])
        slot_endpoints = torch.arange(
            0,
            decoded_frames,
            4,
            device=hit_mask.device,
            dtype=torch.int64,
        )
        if int(slot_endpoints.numel()) != int(readout_features.shape[0]):
            raise RuntimeError("Wan VAE decoded timeline differs from latent slots")
        display_slots = torch.searchsorted(
            slot_endpoints,
            torch.arange(
                decoded_frames,
                device=hit_mask.device,
                dtype=torch.int64,
            ),
            right=False,
        )
        display_mask = hit_mask.index_select(0, display_slots).unsqueeze(1).float()
        display_mask = F.interpolate(
            display_mask,
            size=WAN_VAE_IMAGE_SIZE,
            mode="nearest",
        ).to(device=decoded.device, dtype=decoded.dtype)
        decoded = decoded * display_mask
        return FeatureVisualization(
            "query target Wan VAE decode",
            decoded.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
            tuple(range(int(decoded.shape[0]))),
            "video_frame",
        )


__all__ = ["WanVAESourceEncoder"]
