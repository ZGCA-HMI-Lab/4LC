from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import torch
from latent_cache import Representation

from alc_video.encoders.visual.base import (
    VisualEncoder,
)
from alc_video.encoders.visual.visualization import PCAFeatureVisualizer

VIT_CHANNELS = 1024
VIT_ENCODER = "vit"
VIT_GRID = (21, 35)
VIT_IMAGE_SIZE = (336, 560)
VIT_MEAN = (0.5, 0.5, 0.5)
VIT_MODEL = "google/vit-large-patch16-224"
VIT_PATCH_SIZE = 16
VIT_PRETRAINED_IMAGE_SIZE = 224
VIT_STD = (0.5, 0.5, 0.5)


VIT_CONFIG_FILENAME = "config.json"
VIT_WEIGHTS_FILENAME = "model.safetensors"


def _is_vit_checkpoint(path: Path) -> bool:
    return all(
        (path / filename).is_file()
        for filename in (VIT_CONFIG_FILENAME, VIT_WEIGHTS_FILENAME)
    )


def resolve_vit_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if _is_vit_checkpoint(path):
        return path
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(
            candidate
            for candidate in snapshots.iterdir()
            if candidate.is_dir() and _is_vit_checkpoint(candidate)
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                "ViT cache contains multiple snapshots; pass one exact "
                f"snapshot path: {candidates}"
            )
    raise FileNotFoundError(
        "ViT path must contain config.json and model.safetensors, or one "
        f"matching snapshots/* directory: {path}"
    )


class ViTEncoder(VisualEncoder):
    path_kind = "directory"
    representation = Representation((21, 35), (16, 16), (8, 8), 1024, "selected")

    encoder_type = VIT_ENCODER

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 4,
        model_path: Path,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.model_path = model_path
        self._model: torch.nn.Module | None = None
        self._on_device = False
        if self.batch_size <= 0:
            raise ValueError("ViT batch_size must be > 0")

    @classmethod
    def from_local_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        batch_size: int = 4,
    ) -> ViTEncoder:
        resolved = resolve_vit_path(path)
        try:
            config = json.loads(
                (resolved / VIT_CONFIG_FILENAME).read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"ViT config.json is invalid: {resolved}") from error
        if not isinstance(config, dict):
            raise ValueError(f"ViT config.json must be a JSON object: {resolved}")
        actual = (
            config.get("model_type"),
            int(config.get("hidden_size", -1)),
            int(config.get("patch_size", -1)),
            int(config.get("image_size", -1)),
        )
        expected = (
            VIT_ENCODER,
            VIT_CHANNELS,
            VIT_PATCH_SIZE,
            VIT_PRETRAINED_IMAGE_SIZE,
        )
        if actual != expected:
            raise ValueError(
                f"{resolved}: expected {VIT_MODEL} type/hidden/patch/image "
                f"{expected}, got {actual}"
            )
        return cls(
            device=torch.device(device),
            dtype=dtype,
            batch_size=batch_size,
            model_path=resolved,
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._ensure_model()

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            from transformers import ViTForImageClassification

            self._model = ViTForImageClassification.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                torch_dtype=self.dtype,
                device_map=self.device,
                use_safetensors=True,
            )
            self._model.eval().requires_grad_(False)
            self._on_device = True
        if not self._on_device:
            self._model.to(device=self.device, dtype=self.dtype)
            self._on_device = True
        return self._model

    def prepare(self) -> None:
        self._ensure_model()

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype in {
            torch.float16,
            torch.bfloat16,
        }:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    @torch.no_grad()
    def _encode(
        self,
        frames: torch.Tensor,
        *,
        output_device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        expected_frame = (3, *VIT_IMAGE_SIZE)
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 4
            or tuple(int(value) for value in frames.shape[1:]) != expected_frame
        ):
            raise ValueError(
                f"ViT frames must be [V,3,{VIT_IMAGE_SIZE[0]},"
                f"{VIT_IMAGE_SIZE[1]}], got {getattr(frames, 'shape', None)}"
            )
        if int(frames.shape[0]) <= 0:
            raise ValueError("ViT frames must contain at least one view")
        if not torch.is_floating_point(frames):
            raise TypeError("ViT frames must be floating point RGB in [0,1]")
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("ViT frames must be finite RGB in [0,1]")
        output_device = torch.device(output_device)

        mean = torch.tensor(
            VIT_MEAN,
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, 3, 1, 1)
        std = torch.tensor(
            VIT_STD,
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, 3, 1, 1)
        token_count = VIT_GRID[0] * VIT_GRID[1]
        outputs: list[torch.Tensor] = []
        model = self._ensure_model()
        backbone = getattr(model, "vit", None)
        if not isinstance(backbone, torch.nn.Module):
            raise TypeError(f"{VIT_MODEL} checkpoint must expose a ViT backbone")
        for start in range(0, int(frames.shape[0]), self.batch_size):
            pixels = frames[start : start + self.batch_size].to(
                device=self.device,
                dtype=self.dtype,
            )
            pixels = (pixels - mean) / std
            with self._autocast():
                encoded = backbone(
                    pixel_values=pixels,
                    interpolate_pos_encoding=True,
                ).last_hidden_state
            expected_shape = (
                int(pixels.shape[0]),
                token_count + 1,
                VIT_CHANNELS,
            )
            if tuple(int(value) for value in encoded.shape) != expected_shape:
                raise ValueError(
                    f"ViT last_hidden_state must be {expected_shape}, "
                    f"got {tuple(encoded.shape)}"
                )
            patch_tokens = encoded[:, 1:]
            if not bool(torch.isfinite(patch_tokens).all()):
                raise FloatingPointError("ViT returned non-finite patch tokens")
            outputs.append(
                patch_tokens.reshape(-1, *VIT_GRID, VIT_CHANNELS)
                .to(device=output_device, dtype=torch.bfloat16)
                .contiguous()
            )
        return torch.cat(outputs, dim=0)

    def _make_visualizer(
        self,
        source_latents: torch.Tensor,
    ) -> PCAFeatureVisualizer:
        expected_tail = (*VIT_GRID, VIT_CHANNELS)
        if (
            not isinstance(source_latents, torch.Tensor)
            or source_latents.ndim != 4
            or int(source_latents.shape[0]) <= 0
            or tuple(int(value) for value in source_latents.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "ViT source latents must be "
                f"[K,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]}] "
                f"with K >= 1, "
                f"got {getattr(source_latents, 'shape', None)}"
            )
        return PCAFeatureVisualizer.fit(
            source_latents,
            label="query target ViT PCA",
        )

    def offload(self) -> None:
        if self._model is not None and self._on_device:
            self._model.to("cpu")
            self._on_device = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


__all__ = [
    "VIT_CHANNELS",
    "VIT_CONFIG_FILENAME",
    "VIT_ENCODER",
    "VIT_GRID",
    "VIT_IMAGE_SIZE",
    "VIT_MODEL",
    "VIT_PATCH_SIZE",
    "VIT_WEIGHTS_FILENAME",
    "ViTEncoder",
    "resolve_vit_path",
]
