from dataclasses import fields
from pathlib import Path
from typing import Any

from alc_video.models.config import ModelConfig
from alc_video.utils.config import (
    default_values,
    load_document,
    merge_defaults,
    reject_unknown_fields,
)

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "configs/models/flux-all.yaml"
MODEL_PATH_FIELDS = ("model.dit_path", "encoders.config", "encoders.wan_model_root")


def load_model_profile(path: str | Path = DEFAULT_MODEL) -> dict:
    source, value = load_document(path, path_fields=MODEL_PATH_FIELDS)
    reject_unknown_fields(value, {"model", "encoders"})
    reject_unknown_fields(
        value["model"], (field.name for field in fields(ModelConfig)), name="model"
    )
    reject_unknown_fields(
        value["encoders"],
        {"config", "representations", "worldmirror", "wan_model_root"},
        name="encoders",
    )
    value["model"] = merge_defaults(default_values(ModelConfig), value["model"])
    value["encoders"] = merge_defaults(
        {"worldmirror": "default", "wan_model_root": None}, value["encoders"]
    )
    if value["encoders"]["wan_model_root"] is None:
        value["encoders"]["wan_model_root"] = value["model"]["dit_path"]
    if len(value["model"]["feature_dims"]) != len(value["encoders"]["representations"]):
        raise ValueError(
            f"{source}: feature_dims and representation bindings differ in length"
        )
    return value


DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "configs/cache.yaml"


def load_cache_config(path: str | Path = DEFAULT_CACHE) -> dict[str, Any]:
    from latent_cache import LatentCacheConfig

    _, value = load_document(path)
    reject_unknown_fields(
        value, (field.name for field in fields(LatentCacheConfig)), name="cache"
    )
    return merge_defaults(default_values(LatentCacheConfig), value)
