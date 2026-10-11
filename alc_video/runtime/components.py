from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from alc_video.encoders.config import EncoderCatalog
from alc_video.encoders.visual.loader import (
    load_visual_encoders,
    offload_visual_encoders,
    resolve_representations,
)
from alc_video.models.checkpoint import TRAINABLE_MODULES, load_checkpoint, load_release
from alc_video.models.config import ModelConfig
from alc_video.models.video_model.lightx2v import fuse_lightx2v_distill_lora
from alc_video.models.video_model.loading import (
    WAN_TEXT_ENCODER_RELATIVE_PATH,
    WAN_TEXT_TOKENIZER_RELATIVE_PATH,
    WAN_VIDEO_CODEC_RELATIVE_PATH,
    load_wan_dit,
    resolve_wan_weights,
)
from alc_video.runtime.resources import release_resource
from alc_video.utils.errors import input_errors
from alc_video.utils.logging import (
    LOG,
    stage,
)

if TYPE_CHECKING:
    from alc_video.encoders.geometry.worldmirror import WorldMirrorGeometryEncoder
    from alc_video.encoders.visual.base import VisualEncoder
    from alc_video.models.model import WanConditionedModel
    from alc_video.models.video_model.text import WanTextEncoder
    from alc_video.models.video_model.vae import WanVideoCodec


def build_network(
    *,
    dit_path: str | Path,
    architecture: str,
    feature_shapes: Mapping[str, tuple[int, int, int]],
    feature_dims: tuple[int, ...],
    mask_dim: int,
    camera_dim: int,
    output_dim: int,
    control_scale: float,
    control_layers: tuple[int, ...],
    use_motion_mask: bool,
) -> WanConditionedModel:
    from alc_video.models.model import WanConditionedModel

    shapes = dict(feature_shapes)
    if not shapes:
        raise ValueError("feature_shapes must not be empty")
    for name, shape in shapes.items():
        if (
            not isinstance(shape, tuple)
            or len(shape) != 3
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in shape
            )
        ):
            raise ValueError(
                f"feature shape of {name!r} must be three positive integers"
            )
    base_dit = load_wan_dit(dit_path)
    network = WanConditionedModel(
        base_dit,
        architecture=architecture,
        feature_shapes=tuple(
            ((int(shape[0]), int(shape[1])), int(shape[2])) for shape in shapes.values()
        ),
        feature_dims=tuple(int(dim) for dim in feature_dims),
        mask_dim=mask_dim,
        camera_dim=camera_dim,
        output_dim=output_dim,
        control_scale=control_scale,
        control_layers=control_layers,
        use_motion_mask=use_motion_mask,
    )
    del base_dit
    return network


@dataclass(frozen=True)
class RuntimeComponents:
    network: WanConditionedModel
    codec: WanVideoCodec
    text: WanTextEncoder
    mirror: WorldMirrorGeometryEncoder
    encoders: dict[str, VisualEncoder]

    @property
    def bindings(self) -> dict[str, tuple[VisualEncoder, str]]:
        return {
            role: (encoder, "depth" if role == "depth" else "rgb")
            for role, encoder in self.encoders.items()
        }


def load_runtime_components(
    model: ModelConfig,
    catalog: EncoderCatalog,
    representations: Mapping[str, str],
    *,
    wan_model_root: Path,
    worldmirror_checkpoint: Path,
    device: torch.device,
    resources: ExitStack,
    weights: Path | None = None,
    lightx2v: Path | None = None,
    trainable_modules: tuple[str, ...] = TRAINABLE_MODULES,
    prepare_network: Callable[[WanConditionedModel], object] | None = None,
) -> RuntimeComponents:
    from alc_video.encoders.config import resolve_asset_path
    from alc_video.encoders.geometry.config import worldmirror_runtime_config
    from alc_video.encoders.geometry.worldmirror import WorldMirrorGeometryEncoder
    from alc_video.encoders.registry import BACKENDS
    from alc_video.models.video_model.lightx2v import resolve_lightx2v_lora_path
    from alc_video.models.video_model.loading import resolve_wan_checkpoint
    from alc_video.models.video_model.text import WanTextEncoder
    from alc_video.models.video_model.vae import WanVideoCodec

    configured = {
        role: catalog.visual_encoder(name) for role, name in representations.items()
    }
    shapes, _specs = resolve_representations(
        configured, expected_names=tuple(configured)
    )
    with input_errors():
        resolve_wan_weights(model.dit_path)
        resolve_wan_checkpoint(
            wan_model_root / WAN_VIDEO_CODEC_RELATIVE_PATH, name="Wan VAE"
        )
        resolve_wan_checkpoint(
            wan_model_root / WAN_TEXT_ENCODER_RELATIVE_PATH, name="Wan text encoder"
        )
        tokenizer = wan_model_root / WAN_TEXT_TOKENIZER_RELATIVE_PATH
        if not tokenizer.is_dir():
            raise FileNotFoundError(f"Wan tokenizer directory is missing: {tokenizer}")
        worldmirror_runtime_config(worldmirror_checkpoint, device)
        for role, profile in configured.items():
            resolve_asset_path(
                profile.path,
                name=f"encoders.{role}.path",
                root=catalog.path.parent,
                kind=BACKENDS[profile.backend].path_kind,
                require_exists=True,
            )
        if lightx2v is not None:
            resolve_lightx2v_lora_path(lightx2v)
        if weights is not None and not weights.exists():
            raise FileNotFoundError(
                f"model weights do not exist: {weights}; set --weights"
            )
    LOG.info(
        "model architecture=%s device=%s dtype=bfloat16 representations=%s",
        model.architecture,
        device,
        dict(representations),
    )
    with stage("load.backbone", path=model.dit_path):
        network = build_network(feature_shapes=shapes, **model.build_options())
    resources.enter_context(release_resource(network.to, "cpu"))
    if weights is not None:
        with stage("load.learned_weights", path=weights):
            if weights.suffix == ".safetensors":
                load_release(network, weights, parts=trainable_modules)
            else:
                load_checkpoint(network, weights)
    if lightx2v is not None:
        with stage("load.lightx2v", path=lightx2v):
            network.to(device, dtype=torch.bfloat16)
            fuse_lightx2v_distill_lora(network.dit, lightx2v)
            network.to("cpu")
    if prepare_network is not None:
        prepare_network(network)
    with stage("load.video_codec", path=wan_model_root / WAN_VIDEO_CODEC_RELATIVE_PATH):
        codec = WanVideoCodec.from_local_checkpoint(
            wan_model_root / WAN_VIDEO_CODEC_RELATIVE_PATH, device=device
        )
    resources.enter_context(release_resource(codec.offload))
    with stage(
        "load.text_encoder", path=wan_model_root / WAN_TEXT_ENCODER_RELATIVE_PATH
    ):
        text = WanTextEncoder.from_local_checkpoint(
            wan_model_root / WAN_TEXT_ENCODER_RELATIVE_PATH,
            wan_model_root / WAN_TEXT_TOKENIZER_RELATIVE_PATH,
            device="cpu",
        )
    text.device = device
    resources.enter_context(release_resource(text.offload))
    with stage("load.geometry", path=worldmirror_checkpoint):
        mirror_config, _ = worldmirror_runtime_config(worldmirror_checkpoint, device)
        mirror = WorldMirrorGeometryEncoder.from_config(mirror_config, device=device)
        resources.enter_context(release_resource(mirror.offload))
        mirror.offload()
    with stage("load.representations", roles=",".join(configured)):
        encoders = load_visual_encoders(
            configured, expected_names=tuple(configured), device=device
        )
        resources.enter_context(release_resource(offload_visual_encoders, encoders))
        offload_visual_encoders(encoders)
    shared = {
        name: next(role for role, value in encoders.items() if value is encoder)
        for name, encoder in encoders.items()
    }
    LOG.info("representations ready shapes=%s shared_instances=%s", shapes, shared)
    return RuntimeComponents(network, codec, text, mirror, encoders)
