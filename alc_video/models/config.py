from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from alc_video.utils.config import reject_unknown_fields, resolve_path


@dataclass(frozen=True)
class ModelConfig:
    dit_path: Path
    architecture: str
    feature_dims: tuple[int, ...]
    control_layers: tuple[int, ...]
    mask_dim: int = 32
    camera_dim: int = 64
    output_dim: int = 1024
    control_scale: float = 1.0
    use_motion_mask: bool = False

    def build_options(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def parse_model_config(value: Mapping, *, root: Path) -> ModelConfig:
    reject_unknown_fields(
        value, (field.name for field in fields(ModelConfig)), name="model"
    )
    options = dict(value)
    options["dit_path"] = resolve_path(options["dit_path"], root=root)
    options["feature_dims"] = tuple(options["feature_dims"])
    options["control_layers"] = tuple(options["control_layers"])
    return ModelConfig(**options)
