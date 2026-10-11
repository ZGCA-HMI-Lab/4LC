from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import torch
from latent_cache import Representation

from alc_video.encoders.visual.base import (
    CONTEXT_MAX,
    CONTEXT_MIN,
    VisualEncoder,
    is_keyframe_count,
)
from alc_video.encoders.visual.visualization import PCAFeatureVisualizer

RESNET50_CHANNELS = 512
RESNET50_ENCODER = "resnet50_layer2"
RESNET50_FEATURE_LAYER = "layer2"
RESNET50_GRID = (42, 70)
RESNET50_IMAGE_SIZE = (336, 560)
RESNET50_MEAN = (0.485, 0.456, 0.406)
RESNET50_MODEL = "torchvision/resnet50"
RESNET50_OUTPUT_STRIDE = 8
RESNET50_STD = (0.229, 0.224, 0.225)
RESNET50_WEIGHTS = "IMAGENET1K_V2"
RESNET50_WEIGHTS_FILENAME = "resnet50-11ad3fa6.pth"


def resolve_resnet50_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "ResNet-50 weights must be the local torchvision "
            f"{RESNET50_WEIGHTS} file {RESNET50_WEIGHTS_FILENAME}: {path}"
        )
    if path.name != RESNET50_WEIGHTS_FILENAME:
        raise ValueError(
            "ResNet-50 weights filename must identify torchvision "
            f"{RESNET50_WEIGHTS}: {RESNET50_WEIGHTS_FILENAME}"
        )
    return path


def _load_state_dict(
    path: Path,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    value = torch.load(path, map_location=device, weights_only=True)
    if (
        not isinstance(value, Mapping)
        or not value
        or any(not isinstance(name, str) for name in value)
        or any(not isinstance(tensor, torch.Tensor) for tensor in value.values())
    ):
        raise ValueError(f"ResNet-50 checkpoint must be a non-empty state dict: {path}")
    return value


def _load_feature_extractor(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from torchvision.models import resnet50

    model = resnet50(weights=None).to(device=device, dtype=dtype)
    model.load_state_dict(_load_state_dict(path, device), strict=True)
    return torch.nn.Sequential(
        OrderedDict(
            (
                ("conv1", model.conv1),
                ("bn1", model.bn1),
                ("relu", model.relu),
                ("maxpool", model.maxpool),
                ("layer1", model.layer1),
                ("layer2", model.layer2),
            )
        )
    )


class ResNet50Layer2Encoder(VisualEncoder):
    path_kind = "file"
    representation = Representation((42, 70), (8, 8), (0, 0), 512, "selected")

    encoder_type = RESNET50_ENCODER

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
            raise ValueError("ResNet-50 batch_size must be > 0")

    @classmethod
    def from_local_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        batch_size: int = 4,
    ) -> ResNet50Layer2Encoder:
        resolved = resolve_resnet50_path(path)
        resolved_device = torch.device(device)
        return cls(
            device=resolved_device,
            dtype=dtype,
            batch_size=batch_size,
            model_path=resolved,
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._ensure_model()

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = _load_feature_extractor(
                self.model_path,
                device=self.device,
                dtype=self.dtype,
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
        expected = (3, *RESNET50_IMAGE_SIZE)
        if not isinstance(frames, torch.Tensor):
            raise TypeError(
                "ResNet-50 frames must be a torch.Tensor with shape "
                f"[V,3,{RESNET50_IMAGE_SIZE[0]},{RESNET50_IMAGE_SIZE[1]}], "
                f"got {type(frames).__name__}"
            )
        if (
            frames.ndim != 4
            or tuple(int(value) for value in frames.shape[1:]) != expected
        ):
            raise ValueError(
                "ResNet-50 frames must be "
                f"[V,3,{RESNET50_IMAGE_SIZE[0]},{RESNET50_IMAGE_SIZE[1]}], "
                f"got {tuple(frames.shape)}"
            )
        if int(frames.shape[0]) <= 0:
            raise ValueError("ResNet-50 frames must contain at least one view")
        if not torch.is_floating_point(frames):
            raise TypeError("ResNet-50 frames must be floating point RGB in [0,1]")
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("ResNet-50 frames must be finite RGB in [0,1]")
        output_device = torch.device(output_device)
        mean = torch.tensor(
            RESNET50_MEAN,
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, 3, 1, 1)
        std = torch.tensor(
            RESNET50_STD,
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, 3, 1, 1)
        outputs: list[torch.Tensor] = []
        model = self._ensure_model()
        for start in range(0, int(frames.shape[0]), self.batch_size):
            pixels = frames[start : start + self.batch_size].to(
                device=self.device,
                dtype=self.dtype,
            )
            pixels = (pixels - mean) / std
            with self._autocast():
                encoded = model(pixels)
            expected_shape = (
                int(pixels.shape[0]),
                RESNET50_CHANNELS,
                *RESNET50_GRID,
            )
            if tuple(int(value) for value in encoded.shape) != expected_shape:
                raise ValueError(
                    f"ResNet-50 layer2 output must be {expected_shape}, "
                    f"got {tuple(encoded.shape)}"
                )
            if not bool(torch.isfinite(encoded).all()):
                raise FloatingPointError("ResNet-50 layer2 returned non-finite values")
            outputs.append(
                encoded.permute(0, 2, 3, 1)
                .to(device=output_device, dtype=torch.bfloat16)
                .contiguous()
            )
        return torch.cat(outputs, dim=0)

    def _make_visualizer(
        self,
        source_latents: torch.Tensor,
    ) -> PCAFeatureVisualizer:
        expected_tail = (*RESNET50_GRID, RESNET50_CHANNELS)
        if (
            not isinstance(source_latents, torch.Tensor)
            or source_latents.ndim != 4
            or not is_keyframe_count(int(source_latents.shape[0]))
            or tuple(int(value) for value in source_latents.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "ResNet-50 source latents must be "
                f"[K,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]}] "
                f"with K=1 or {CONTEXT_MIN} <= K <= {CONTEXT_MAX}, "
                f"got {getattr(source_latents, 'shape', None)}"
            )
        return PCAFeatureVisualizer.fit(
            source_latents,
            label="query target ResNet-50 layer2 PCA",
        )

    def offload(self) -> None:
        if self._model is not None and self._on_device:
            self._model.to("cpu")
            self._on_device = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


__all__ = [
    "RESNET50_CHANNELS",
    "RESNET50_ENCODER",
    "RESNET50_FEATURE_LAYER",
    "RESNET50_GRID",
    "RESNET50_IMAGE_SIZE",
    "RESNET50_MEAN",
    "RESNET50_MODEL",
    "RESNET50_OUTPUT_STRIDE",
    "RESNET50_STD",
    "RESNET50_WEIGHTS",
    "RESNET50_WEIGHTS_FILENAME",
    "ResNet50Layer2Encoder",
    "resolve_resnet50_path",
]
