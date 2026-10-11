from __future__ import annotations

from contextlib import ExitStack

import torch
from latent_cache import LatentCache

from alc_video.encoders.config import load_encoder_catalog
from alc_video.inference.config import GenerationConfig
from alc_video.inference.sampling import generate_latents_from_generator
from alc_video.models.video_model.temporal import wan_vae_latent_frame_count
from alc_video.runtime.components import RuntimeComponents, load_runtime_components
from alc_video.utils.logging import stage


def load_inference_components(
    settings: GenerationConfig, resources: ExitStack
) -> RuntimeComponents:
    encoders = settings.encoders
    catalog = load_encoder_catalog(encoders.config, require_assets=False)
    return load_runtime_components(
        settings.model,
        catalog,
        dict(encoders.representations),
        wan_model_root=encoders.wan_model_root,
        worldmirror_checkpoint=catalog.worldmirror_encoder(
            encoders.worldmirror
        ).checkpoint,
        device=torch.device(settings.device),
        resources=resources,
        weights=settings.weights,
        lightx2v=settings.lightx2v,
        prepare_network=lambda network: network.eval().requires_grad_(False),
    )


def build_cache(
    settings: GenerationConfig, components: RuntimeComponents
) -> LatentCache:
    return LatentCache(settings.cache, components.bindings)


def generate_chunk(settings, components, readouts, intr, c2w, prompt, generator):
    network, codec, text = components.network, components.codec, components.text
    with stage("encode.prompt", characters=len(prompt)):
        text.text_encoder.to(text.device)
        context = text.encode_text(prompt)[None].to(text.device)
    text.offload()
    network.to(text.device, dtype=torch.bfloat16)
    height, width = settings.cache.image_size
    with stage("diffusion.sample", frames=len(intr), steps=settings.sampling.steps):
        latents = generate_latents_from_generator(
            network,
            context,
            readouts,
            intr[None],
            c2w[None],
            image_size=(height, width),
            latent_shape=(
                1,
                16,
                wan_vae_latent_frame_count(len(intr)),
                height // 8,
                width // 8,
            ),
            num_inference_steps=settings.sampling.steps,
            generator=generator,
            control_scales=settings.sampling.control_scales,
        )
    network.to("cpu")
    torch.cuda.empty_cache()
    with stage("video.decode"):
        video = codec.decode_video(latents, tiled=settings.sampling.vae_tiled)
    codec.offload()
    return video[0].permute(1, 2, 3, 0).contiguous()
