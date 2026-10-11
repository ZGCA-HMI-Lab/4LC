from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import torch
from latent_cache import Representation

from alc_video.encoders.visual.base import (
    CONTEXT_MAX,
    CONTEXT_MIN,
    IMAGE_SIZE,
    PCAFeatureVisualizer,
    VisualEncoder,
    is_keyframe_count,
)

VJEPA2_ENCODER = "vjepa2"
VJEPA2_MODEL = "vjepa2_1_vitl_dist_vitG_384"
VJEPA2_WEIGHTS_FILENAME = "vjepa2_1_vitl_dist_vitG_384.pt"
VJEPA2_PRETRAINED_IMAGE_SIZE = 384
VJEPA2_PATCH_SIZE = 16
VJEPA2_CHANNELS = 1024
VJEPA2_IMAGE_SIZE = IMAGE_SIZE
VJEPA2_GRID = (21, 35)
VJEPA2_MEAN = (0.485, 0.456, 0.406)
VJEPA2_STD = (0.229, 0.224, 0.225)
VJEPA2_CHECKPOINT_KEY = "ema_encoder"
VJEPA2_OUTPUT_TOKENS = "last_normalized_image_patch_tokens"


def resolve_vjepa2_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"V-JEPA2 weights must be the local {VJEPA2_WEIGHTS_FILENAME}: {path}"
        )
    if path.name != VJEPA2_WEIGHTS_FILENAME:
        raise ValueError(f"V-JEPA2 weights filename must be {VJEPA2_WEIGHTS_FILENAME}")
    return path


def _checkpoint_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(value, Mapping) or VJEPA2_CHECKPOINT_KEY not in value:
        raise ValueError(
            f"{path}: V-JEPA2 checkpoint must contain {VJEPA2_CHECKPOINT_KEY!r}"
        )
    raw = value[VJEPA2_CHECKPOINT_KEY]
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{path}: V-JEPA2 encoder state dict is empty")
    state: dict[str, torch.Tensor] = {}
    for raw_name, tensor in raw.items():
        if not isinstance(raw_name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{path}: V-JEPA2 encoder state dict is invalid")
        name = raw_name.replace("module.", "").replace("backbone.", "")
        if name in state:
            raise ValueError(f"{path}: duplicate V-JEPA2 tensor {name!r}")
        state[name] = tensor
    return state


def _load_vjepa2_encoder(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from src.hub.backbones import vjepa2_1_vit_large_384

    model, predictor = vjepa2_1_vit_large_384(pretrained=False)
    del predictor
    model = model.to(device=device, dtype=dtype)
    model.load_state_dict(_checkpoint_state(path, device), strict=True)
    return model.eval().requires_grad_(False)


class VJEPA2Encoder(VisualEncoder):
    encoder_type = VJEPA2_ENCODER

    representation = Representation(
        grid=VJEPA2_GRID,
        stride=(VJEPA2_PATCH_SIZE,) * 2,
        offset=(VJEPA2_PATCH_SIZE // 2,) * 2,
        dim=VJEPA2_CHANNELS,
        frame_layout="previous_current",
    )
    path_kind = "file"

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_path: Path,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self._model: torch.nn.Module | None = None
        self._on_device = False

    @classmethod
    def from_config(cls, config, *, device, dtype):
        cls.configured_representation(config.levels)
        return cls.from_local_pretrained(config.path, device=device, dtype=dtype)

    @classmethod
    def from_local_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> VJEPA2Encoder:
        return cls(
            device=torch.device(device),
            dtype=dtype,
            model_path=resolve_vjepa2_path(path),
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._ensure_model()

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = _load_vjepa2_encoder(
                self.model_path,
                device=self.device,
                dtype=self.dtype,
            )
            self._on_device = True
        if not self._on_device:
            self._model.to(device=self.device, dtype=self.dtype)
            self._on_device = True
        return self._model

    def prepare(self) -> None:
        self._ensure_model()

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype in {
            torch.bfloat16,
            torch.float16,
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
        expected = (3, *VJEPA2_IMAGE_SIZE)
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 4
            or tuple(int(size) for size in frames.shape[1:]) != expected
        ):
            raise ValueError(
                f"V-JEPA2 frames must be [V,3,{VJEPA2_IMAGE_SIZE[0]},"
                f"{VJEPA2_IMAGE_SIZE[1]}], got {getattr(frames, 'shape', None)}"
            )
        packed_frames = int(frames.shape[0])
        if packed_frames == 0 or packed_frames % 2:
            raise ValueError(
                "V-JEPA2 previous_current input must contain nonempty 2K frames"
            )
        output_frames = packed_frames // 2
        if not torch.is_floating_point(frames):
            raise TypeError("V-JEPA2 frames must be floating point RGB in [0,1]")
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("V-JEPA2 frames must be finite RGB in [0,1]")

        mean = torch.tensor(
            VJEPA2_MEAN,
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, 3, 1, 1)
        std = torch.tensor(
            VJEPA2_STD,
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, 3, 1, 1)
        output_device = torch.device(output_device)
        model = self._ensure_model()
        pixels = frames.to(device=self.device, dtype=self.dtype)
        pixels = (pixels - mean) / std
        video = pixels.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        with self._autocast():
            encoded = model(video)
        token_count = VJEPA2_GRID[0] * VJEPA2_GRID[1]
        expected_shape = (1, output_frames * token_count, VJEPA2_CHANNELS)
        if tuple(int(size) for size in encoded.shape) != expected_shape:
            raise ValueError(
                f"{VJEPA2_MODEL} patch tokens must be {expected_shape}, "
                f"got {tuple(encoded.shape)}"
            )
        if not bool(torch.isfinite(encoded).all()):
            raise FloatingPointError("V-JEPA2 returned non-finite patch tokens")
        return (
            encoded.reshape(output_frames, *VJEPA2_GRID, VJEPA2_CHANNELS)
            .to(device=output_device, dtype=torch.bfloat16)
            .contiguous()
        )

    def _make_visualizer(
        self,
        features: torch.Tensor,
    ) -> PCAFeatureVisualizer:
        expected = (*VJEPA2_GRID, VJEPA2_CHANNELS)
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 4
            or not is_keyframe_count(int(features.shape[0]))
            or tuple(int(size) for size in features.shape[1:]) != expected
        ):
            raise ValueError(
                f"V-JEPA2 features must be [K,{expected[0]},"
                f"{expected[1]},{expected[2]}] with K=1 or "
                f"{CONTEXT_MIN} <= K <= {CONTEXT_MAX}, got "
                f"{getattr(features, 'shape', None)}"
            )
        return PCAFeatureVisualizer.fit(
            features,
            label="readout V-JEPA2 PCA",
        )

    def offload(self) -> None:
        if self._model is not None and self._on_device:
            self._model.to("cpu")
            self._on_device = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


__all__ = [
    "VJEPA2_CHANNELS",
    "VJEPA2_ENCODER",
    "VJEPA2_GRID",
    "VJEPA2_IMAGE_SIZE",
    "VJEPA2_MODEL",
    "VJEPA2_PATCH_SIZE",
    "VJEPA2_WEIGHTS_FILENAME",
    "VJEPA2Encoder",
    "resolve_vjepa2_path",
]
