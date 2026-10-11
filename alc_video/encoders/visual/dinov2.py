from __future__ import annotations

import json
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

DINOV2_ENCODER = "dinov2"
DINOV2_MODEL = "facebook/dinov2-large"
DINOV2_PATCH_SIZE = 14
DINOV2_CHANNELS = 1024
DINOV2_IMAGE_SIZE = IMAGE_SIZE
DINOV2_GRID = (24, 40)
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
DINOV2_CONFIG_FILENAME = "config.json"
DINOV2_SAFETENSORS_FILENAME = "model.safetensors"
DINOV2_SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"
DINOV2_BIN_FILENAME = "pytorch_model.bin"
DINOV2_BIN_INDEX_FILENAME = "pytorch_model.bin.index.json"


def _sharded_weight_files(path: Path, index_name: str) -> tuple[Path, ...]:
    index = path / index_name
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"DINOv2 shard index is invalid: {index}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"DINOv2 shard index must be a JSON object: {index}")
    weight_map = payload.get("weight_map")
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or any(not isinstance(key, str) or not key for key in weight_map)
        or any(not isinstance(value, str) or not value for value in weight_map.values())
    ):
        raise ValueError(
            f"DINOv2 shard index must contain a non-empty string weight_map: {index}"
        )
    names = tuple(sorted(set(weight_map.values())))
    expected_suffix = ".safetensors" if ".safetensors." in index_name else ".bin"
    if any(
        Path(name).name != name or not name.endswith(expected_suffix) for name in names
    ):
        raise ValueError(
            f"DINOv2 shard index contains an invalid shard filename: {index}"
        )
    shards = tuple(path / name for name in names)
    missing = tuple(shard for shard in shards if not shard.is_file())
    if missing:
        raise FileNotFoundError(
            f"DINOv2 shard index references missing weight files: {missing}"
        )
    return (index, *shards)


def _weight_asset_files(path: Path) -> tuple[Path, ...]:
    safetensors = path / DINOV2_SAFETENSORS_FILENAME
    safetensors_index = path / DINOV2_SAFETENSORS_INDEX_FILENAME
    if safetensors.is_file() and safetensors_index.is_file():
        raise ValueError(
            "DINOv2 checkpoint ambiguously contains both a monolithic and a "
            f"sharded safetensors checkpoint: {path}"
        )
    if safetensors.is_file():
        return (safetensors,)
    if safetensors_index.is_file():
        return _sharded_weight_files(path, DINOV2_SAFETENSORS_INDEX_FILENAME)

    binary = path / DINOV2_BIN_FILENAME
    binary_index = path / DINOV2_BIN_INDEX_FILENAME
    if binary.is_file() and binary_index.is_file():
        raise ValueError(
            "DINOv2 checkpoint ambiguously contains both a monolithic and a "
            f"sharded PyTorch checkpoint: {path}"
        )
    if binary.is_file():
        return (binary,)
    if binary_index.is_file():
        return _sharded_weight_files(path, DINOV2_BIN_INDEX_FILENAME)
    raise FileNotFoundError(
        "DINOv2 checkpoint has config.json but no Hugging Face weights; expected "
        "model.safetensors, model.safetensors.index.json, pytorch_model.bin, or "
        f"pytorch_model.bin.index.json under {path}"
    )


def dinov2_asset_files(path: str | Path) -> tuple[Path, ...]:
    resolved = Path(path).expanduser().resolve()
    config = resolved / DINOV2_CONFIG_FILENAME
    if not config.is_file():
        raise FileNotFoundError(f"DINOv2 checkpoint is missing config.json: {resolved}")
    return (config, *_weight_asset_files(resolved))


def resolve_dinov2_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if (path / DINOV2_CONFIG_FILENAME).is_file():
        dinov2_asset_files(path)
        return path
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        config_candidates = sorted(
            candidate
            for candidate in snapshots.iterdir()
            if candidate.is_dir() and (candidate / DINOV2_CONFIG_FILENAME).is_file()
        )
        candidates = []
        for candidate in config_candidates:
            try:
                dinov2_asset_files(candidate)
            except (FileNotFoundError, ValueError):
                continue
            candidates.append(candidate)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                "DINOv2 cache contains multiple snapshots; pass one exact "
                f"snapshot path: {candidates}"
            )
        if len(config_candidates) == 1:
            dinov2_asset_files(config_candidates[0])
    raise FileNotFoundError(
        "DINOv2 path must contain a complete Hugging Face checkpoint or one "
        f"complete snapshots/* checkpoint: {path}"
    )


class DINOv2Encoder(VisualEncoder):
    encoder_type = DINOV2_ENCODER

    representation = Representation(
        grid=DINOV2_GRID,
        stride=(DINOV2_PATCH_SIZE,) * 2,
        offset=(DINOV2_PATCH_SIZE // 2,) * 2,
        dim=DINOV2_CHANNELS,
        frame_layout="selected",
    )
    path_kind = "directory"

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
            raise ValueError("DINOv2 batch_size must be > 0")

    @classmethod
    def from_local_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        batch_size: int = 4,
    ) -> DINOv2Encoder:
        resolved = resolve_dinov2_path(path)
        import json

        try:
            config = json.loads((resolved / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"DINOv2 config.json is invalid: {resolved}") from error
        if not isinstance(config, dict):
            raise ValueError(f"DINOv2 config.json must be a JSON object: {resolved}")
        hidden = int(config.get("hidden_size", -1))
        patch = int(config.get("patch_size", -1))
        model_type = config.get("model_type")
        if hidden != DINOV2_CHANNELS or patch != DINOV2_PATCH_SIZE:
            raise ValueError(
                f"{resolved}: expected DINOv2-large hidden/patch "
                f"{DINOV2_CHANNELS}/{DINOV2_PATCH_SIZE}, got {hidden}/{patch}"
            )
        if model_type != "dinov2":
            raise ValueError(
                f"{resolved}: expected model_type='dinov2', got {model_type!r}"
            )
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
            from transformers import AutoModel

            weight_files = dinov2_asset_files(self.model_path)[1:]
            self._model = AutoModel.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                torch_dtype=self.dtype,
                device_map="cpu",
                use_safetensors=any(
                    path.name.endswith(".safetensors")
                    or path.name == DINOV2_SAFETENSORS_INDEX_FILENAME
                    for path in weight_files
                ),
            )
            self._model.eval().requires_grad_(False)
        if not self._on_device:
            self._model.to(device=self.device, dtype=self.dtype)
            self._on_device = True
        return self._model

    def prepare(self) -> None:
        self._ensure_model()

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    @torch.no_grad()
    def _encode(
        self,
        frames: torch.Tensor,
        *,
        output_device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 4
            or tuple(int(v) for v in frames.shape[1:]) != (3, *DINOV2_IMAGE_SIZE)
        ):
            raise ValueError(
                f"DINOv2 frames must be [V,3,{DINOV2_IMAGE_SIZE[0]},{DINOV2_IMAGE_SIZE[1]}], "
                f"got {getattr(frames, 'shape', None)}"
            )
        if int(frames.shape[0]) <= 0:
            raise ValueError("DINOv2 frames must contain at least one view")
        if not torch.is_floating_point(frames):
            raise TypeError("DINOv2 frames must be floating point RGB in [0,1]")
        if (
            not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or float(frames.max()) > 1.0
        ):
            raise ValueError("DINOv2 frames must be finite RGB in [0,1]")
        output_device = torch.device(output_device)
        mean = torch.tensor(DINOV2_MEAN, device=self.device, dtype=self.dtype).reshape(
            1, 3, 1, 1
        )
        std = torch.tensor(DINOV2_STD, device=self.device, dtype=self.dtype).reshape(
            1, 3, 1, 1
        )
        outputs: list[torch.Tensor] = []
        token_count = DINOV2_GRID[0] * DINOV2_GRID[1]
        model = self._ensure_model()
        for start in range(0, int(frames.shape[0]), self.batch_size):
            pixels = frames[start : start + self.batch_size].to(
                device=self.device, dtype=self.dtype
            )
            pixels = (pixels - mean) / std
            with self._autocast():
                encoded = model(
                    pixel_values=pixels,
                    interpolate_pos_encoding=True,
                ).last_hidden_state
            expected_shape = (
                int(pixels.shape[0]),
                token_count + 1,
                DINOV2_CHANNELS,
            )
            if tuple(int(value) for value in encoded.shape) != expected_shape:
                raise ValueError(
                    f"DINOv2 last_hidden_state must be {expected_shape} in official "
                    f"[CLS, row-major patch tokens] order, got {tuple(encoded.shape)}"
                )
            patch_tokens = encoded[:, 1:]
            outputs.append(
                patch_tokens.reshape(-1, *DINOV2_GRID, DINOV2_CHANNELS)
                .to(device=output_device, dtype=torch.bfloat16)
                .contiguous()
            )
        return torch.cat(outputs, dim=0)

    def _make_visualizer(
        self,
        features: torch.Tensor,
    ) -> PCAFeatureVisualizer:
        expected_tail = (*DINOV2_GRID, DINOV2_CHANNELS)
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 4
            or not is_keyframe_count(int(features.shape[0]))
            or tuple(int(value) for value in features.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "DINOv2 features must be "
                f"[K,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]}] "
                f"with K=1 or {CONTEXT_MIN} <= K <= {CONTEXT_MAX}, "
                f"got {getattr(features, 'shape', None)}"
            )
        return PCAFeatureVisualizer.fit(
            features,
            label="readout DINOv2 PCA",
        )

    def offload(self) -> None:
        if self._model is not None and self._on_device:
            self._model.to("cpu")
            self._on_device = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


__all__ = [
    "DINOV2_BIN_FILENAME",
    "DINOV2_BIN_INDEX_FILENAME",
    "DINOV2_CHANNELS",
    "DINOV2_CONFIG_FILENAME",
    "DINOV2_ENCODER",
    "DINOV2_GRID",
    "DINOV2_IMAGE_SIZE",
    "DINOV2_MODEL",
    "DINOV2_PATCH_SIZE",
    "DINOV2_SAFETENSORS_FILENAME",
    "DINOV2_SAFETENSORS_INDEX_FILENAME",
    "DINOv2Encoder",
    "dinov2_asset_files",
    "resolve_dinov2_path",
]
