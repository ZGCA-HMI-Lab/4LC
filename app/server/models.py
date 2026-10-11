from __future__ import annotations

from contextlib import ExitStack

from alc_video.encoders.config import load_encoder_catalog
from alc_video.encoders.visual.loader import resolve_representations
from alc_video.inference.generation import load_inference_components
from alc_video.models.checkpoint import load_release
from alc_video.models.model import WanConditionedModel
from alc_video.runtime.resources import release_resource
from alc_video.utils.logging import stage


def load_models(config, profiles, resources: ExitStack):
    """Reuse one Wan backbone, text encoder, codec and visual encoder set."""
    components = load_inference_components(config, resources)
    networks = {}
    for name, profile in profiles.items():
        if profile is config:
            networks[name] = components.network
            continue
        catalog = load_encoder_catalog(profile.encoders.config, require_assets=False)
        configured = {
            role: catalog.visual_encoder(encoder)
            for role, encoder in profile.encoders.representations
        }
        shapes, _ = resolve_representations(
            configured, expected_names=tuple(configured)
        )
        options = profile.model.build_options()
        del options["dit_path"]
        with stage("load.profile", profile=name):
            network = (
                WanConditionedModel(
                    components.network.dit,
                    feature_shapes=tuple(
                        ((h, w), channels) for h, w, channels in shapes.values()
                    ),
                    **options,
                )
                .eval()
                .requires_grad_(False)
            )
            resources.enter_context(release_resource(network.to, "cpu"))
            load_release(network, profile.weights)
        networks[name] = network
    return components, networks
