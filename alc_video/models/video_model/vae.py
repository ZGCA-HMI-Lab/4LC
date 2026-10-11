from __future__ import annotations

from pathlib import Path

import torch

from alc_video.models.video_model.loading import resolve_wan_checkpoint


class WanVideoCodec:
    def __init__(
        self,
        *,
        vae: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.vae = vae.eval().requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self._on_device = False

    @classmethod
    def from_local_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> WanVideoCodec:
        from diffsynth.models.model_loader import ModelPool

        vae_path = resolve_wan_checkpoint(
            checkpoint,
            name="Wan VAE",
        )
        resolved_device = torch.device(device)
        pool = ModelPool()
        vram = pool.default_vram_config()
        vram["computation_dtype"] = dtype
        vram["computation_device"] = "cpu"
        pool.auto_load_model(str(vae_path), vram_config=vram, clear_parameters=False)
        vae = pool.fetch_model("wan_video_vae")
        if vae is None:
            raise RuntimeError("DiffSynth failed to load the Wan VAE")
        return cls(vae=vae, device=resolved_device, dtype=dtype)

    def _ensure_device(self) -> None:
        if not self._on_device:
            self.vae.to(self.device)
            self._on_device = True

    def prepare(self) -> None:
        self._ensure_device()

    @torch.no_grad()
    def encode_video(
        self, frames: torch.Tensor, *, tiled: bool = False
    ) -> torch.Tensor:
        if (
            frames.ndim != 4
            or tuple(int(v) for v in frames.shape[1:]) != (3, 336, 560)
            or int(frames.shape[0]) <= 0
            or (int(frames.shape[0]) - 1) % 4 != 0
        ):
            raise ValueError(
                "Wan target frames must be [T,3,336,560] with T=4m+1, "
                f"got {tuple(frames.shape)}"
            )
        if not torch.is_floating_point(frames):
            raise TypeError("Wan target frames must be floating point RGB in [0,1]")
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("Wan target frames must be finite RGB in [0,1]")
        self._ensure_device()
        video = (
            frames.permute(1, 0, 2, 3)
            .to(device=self.device, dtype=self.dtype)
            .mul(2.0)
            .sub(1.0)
        )
        latent = self.vae.encode([video], device=self.device, tiled=bool(tiled))[0]
        latent_frames = (int(frames.shape[0]) - 1) // 4 + 1
        expected = (16, latent_frames, 42, 70)
        if tuple(int(v) for v in latent.shape) != expected:
            raise ValueError(
                f"Wan VAE latent shape {tuple(latent.shape)} != {expected}"
            )
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("Wan VAE latent contains non-finite values")
        return latent.to(device="cpu", dtype=torch.bfloat16).contiguous()

    @torch.inference_mode()
    def decode_video(
        self,
        latents: torch.Tensor,
        *,
        tiled: bool,
    ) -> torch.Tensor:
        if (
            latents.ndim != 5
            or int(latents.shape[0]) <= 0
            or int(latents.shape[1]) != 16
            or any(int(size) <= 0 for size in latents.shape[2:])
        ):
            raise ValueError(
                f"Wan decode latents must be [B,16,F,H,W], got {tuple(latents.shape)}"
            )
        if not torch.is_floating_point(latents):
            raise TypeError("Wan decode latents must be floating point")
        if not bool(torch.isfinite(latents).all()):
            raise ValueError("Wan decode latents contain non-finite values")
        self._ensure_device()
        video = self.vae.decode(
            latents,
            device=self.device,
            tiled=bool(tiled),
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )
        expected_frames = 4 * int(latents.shape[2]) - 3
        if (
            video.ndim != 5
            or tuple(int(size) for size in video.shape[:2])
            != (int(latents.shape[0]), 3)
            or int(video.shape[2]) != expected_frames
        ):
            raise ValueError(
                "Wan VAE returned invalid causal decode shape: "
                f"{tuple(video.shape)}; expected batch={int(latents.shape[0])}, "
                f"channels=3, frames={expected_frames}"
            )
        video = (
            video.to(device=self.device, dtype=torch.float32)
            .add(1.0)
            .mul(0.5)
            .clamp_(0.0, 1.0)
        )
        if not bool(torch.isfinite(video).all()):
            raise FloatingPointError("Wan VAE returned non-finite video values")
        return video

    def offload(self) -> None:
        if self._on_device:
            self.vae.to("cpu")
            self._on_device = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
