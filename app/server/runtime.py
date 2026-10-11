from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Literal

import torch
from latent_cache import CacheEntry, LatentCache

from alc_video.conditioning.first_frame import construct_first_frame_entry
from alc_video.conditioning.video import (
    construct_video_entry,
    prepare_direct_video_observations,
)
from alc_video.encoders.visual.loader import offload_visual_encoders
from alc_video.encoders.visual.visualization import (
    PCAFeatureVisualizer,
    VideoResolutionVisualizer,
)
from alc_video.inference.config import GenerationConfig
from alc_video.inference.sampling import generate_latents_from_generator
from alc_video.models.video_model.temporal import wan_vae_latent_frame_count
from alc_video.runtime.resources import release_resource
from server.conditioning import Shot, move_readouts, read_conditions
from server.inputs import (
    DEFAULT_FPS,
    OUTPUT_FRAMES,
    MediaInfo,
    PreparedInput,
    prepare_input,
)

Progress = Callable[[str], None]


@dataclass(frozen=True)
class BuiltCache:
    """Session-owned CPU data; never relies on a previous GPU allocation."""

    entry: CacheEntry
    colors: torch.Tensor
    intrinsics: torch.Tensor
    c2w: torch.Tensor
    times: torch.Tensor
    fps: float
    source_type: Literal["image", "video"]
    image_size: tuple[int, int]
    frame_count: int
    timings: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        height, width = self.image_size
        return {
            "source_type": self.source_type,
            "width": width,
            "height": height,
            "frame_count": self.frame_count,
            "source_duration": float(self.times[-1]),
            "fps": self.fps,
            "keyframes": self.entry.num_keyframes,
            "representations": list(self.entry.features),
            "timings": self.timings,
            "camera": {
                "intrinsics": self.intrinsics.tolist(),
                "c2w": self.c2w.tolist(),
                "times": self.times.tolist(),
            },
        }


class CacheRuntime:
    """Load shared models once; construct a separate Cache for each input."""

    def __init__(
        self,
        config: GenerationConfig,
        *,
        profiles: dict[str, GenerationConfig],
        device: str,
        zerogpu: bool = False,
    ):
        self.zerogpu = zerogpu
        self.config = config
        self.profiles = profiles
        self.components = None
        self.networks = {}
        self.device = torch.device(device)
        self.resources = ExitStack()
        self.encoders = {}
        self.mirror = None

    def __enter__(self):
        from server.models import load_models

        try:
            profiles = (
                {
                    name: replace(profile, device="cpu")
                    for name, profile in self.profiles.items()
                }
                if self.zerogpu
                else self.profiles
            )
            config = max(
                profiles.values(),
                key=lambda profile: len(profile.encoders.representations),
            )
            self.components, self.networks = load_models(
                config, profiles, self.resources
            )
            if self.zerogpu:
                # Assemble once on CPU; register CUDA storage only after all weights are loaded.
                self.components.text.device = self.device
                self.components.codec.device = self.device
                self.components.mirror.device = self.device
                self.components.mirror.reconstructor.device = self.device
                for encoder in self.components.encoders.values():
                    encoder.device = self.device
            self.encoders = self.components.encoders
            self.mirror = self.components.mirror
        except BaseException:
            import sys

            self.resources.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *exc):
        return self.resources.__exit__(*exc)

    def prepare(self, media: MediaInfo, settings: dict) -> PreparedInput:
        return prepare_input(media, settings, self.config.cache.image_size)

    def read(self, built: BuiltCache, shot: Shot, timings: dict, progress: Progress):
        names = dict(self.profiles[shot.profile].encoders.representations)
        cache = LatentCache(
            self.config.cache,
            {
                name: (self.encoders[name], "depth" if name == "depth" else "rgb")
                for name in names
            },
        )
        with self.measure("cache.transfer_to_gpu", timings, progress):
            entry = replace(
                built.entry,
                features={name: built.entry.features[name] for name in names},
            ).to(self.device)
            camera = shot.camera.to(self.device)
        with self.measure("cache.read", timings, progress):
            return read_conditions(cache, entry, camera, shot.mode)

    @torch.inference_mode()
    def preview(
        self,
        built: BuiltCache,
        shot: Shot,
        pca: dict,
        progress: Progress,
        existing: dict | None = None,
    ):
        timings: dict[str, float] = {}
        if existing is None:
            readouts = self.read(built, shot, timings, progress)
        else:
            with self.measure("conditions.transfer_to_gpu", timings, progress):
                readouts = move_readouts(existing, self.device)
        visualizations = {}
        bases = dict(pca)
        with ExitStack() as resources:
            resources.enter_context(
                release_resource(offload_visual_encoders, self.encoders)
            )
            for name, readout in readouts.items():
                with self.measure(f"visualize.{name}", timings, progress):
                    if name in bases:
                        visualizer = VideoResolutionVisualizer(
                            bases[name], built.image_size
                        )
                    else:
                        visualizer = self.encoders[name].make_visualizer(
                            built.entry.features[name].features[0]
                        )
                        if isinstance(visualizer.visualizer, PCAFeatureVisualizer):
                            bases[name] = visualizer.visualizer
                    result = visualizer.render(readout.features[0], readout.coverage[0])
                    if result.domain == "video_frame":
                        positions = result.timeline
                    else:
                        positions = tuple(
                            readout.target_indices[i] for i in result.timeline
                        )
                    timeline = torch.tensor(positions, dtype=torch.int64)
                    slots = torch.searchsorted(
                        timeline, torch.arange(built.frame_count), right=True
                    ).sub(1)
                    frames = (
                        result.video[0].permute(1, 2, 3, 0).cpu().index_select(0, slots)
                    )
                    if shot.mode == "i2v":
                        frames[1:] = 0
                    visualizations[name] = frames.mul(255).round().clamp(0, 255).byte()
        return move_readouts(readouts, "cpu"), visualizations, bases, timings

    @torch.inference_mode()
    def generate(
        self,
        built: BuiltCache,
        shot: Shot,
        prompt: str,
        existing: dict | None,
        progress: Progress,
        sample_progress,
    ):
        if self.components is None:
            raise RuntimeError("Models are not loaded")
        settings = self.profiles[shot.profile]
        network = self.networks[shot.profile]
        text, codec = self.components.text, self.components.codec
        timings = {}
        with ExitStack() as resources:
            resources.enter_context(release_resource(network.to, "cpu"))
            resources.enter_context(release_resource(text.offload))
            resources.enter_context(release_resource(codec.offload))
            if existing is None:
                readouts = self.read(built, shot, timings, progress)
            else:
                with self.measure("conditions.transfer_to_gpu", timings, progress):
                    readouts = move_readouts(existing, self.device)
            cpu_readouts = move_readouts(readouts, "cpu")
            with self.measure("prompt.encode", timings, progress):
                text.text_encoder.to(self.device)
                context = text.encode_text(prompt)[None].to(self.device)
                text.offload()
            with self.measure("model.transfer_to_gpu", timings, progress):
                network.to(self.device, dtype=torch.bfloat16)
            camera = shot.camera.to(self.device)
            height, width = built.image_size
            generator = torch.Generator(device="cpu").manual_seed(
                settings.seed % (2**63)
            )
            with self.measure("diffusion.sample", timings, progress):
                latents = generate_latents_from_generator(
                    network,
                    context,
                    readouts,
                    camera.intr[None],
                    camera.c2w[None],
                    image_size=(height, width),
                    latent_shape=(
                        1,
                        16,
                        wan_vae_latent_frame_count(built.frame_count),
                        height // 8,
                        width // 8,
                    ),
                    num_inference_steps=settings.sampling.steps,
                    generator=generator,
                    control_scales=settings.sampling.control_scales,
                    progress=sample_progress,
                )
            with self.measure("model.transfer_to_cpu", timings, progress):
                network.to("cpu")
                torch.cuda.empty_cache()
            with self.measure("video.decode", timings, progress):
                video = codec.decode_video(latents, tiled=settings.sampling.vae_tiled)[
                    0
                ].permute(1, 2, 3, 0)
                frames = video.mul(255).round().clamp(0, 255).byte().cpu()
        return frames, cpu_readouts, timings

    @contextmanager
    def measure(self, name: str, timings: dict, progress: Progress):
        progress(name)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = perf_counter()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings[name] = perf_counter() - start

    @torch.inference_mode()
    def construct(self, prepared: PreparedInput, progress: Progress) -> BuiltCache:
        if self.mirror is None:
            raise RuntimeError("CacheRuntime must be started before construction")
        cache = LatentCache(
            self.config.cache,
            {
                name: (encoder, "depth" if name == "depth" else "rgb")
                for name, encoder in self.encoders.items()
            },
        )
        timings = {"input.decode": prepared.decode_seconds}
        frames, fps = prepared.frames, prepared.observation_fps
        with ExitStack() as resources:
            resources.enter_context(release_resource(self.mirror.offload))
            resources.enter_context(
                release_resource(offload_visual_encoders, self.encoders)
            )
            resources.callback(cache.clear)
            if prepared.kind == "video":
                with self.measure("geometry.reconstruct", timings, progress):
                    observation = prepare_direct_video_observations(
                        frames,
                        fps,
                        self.mirror,
                        self.encoders,
                    )
                with self.measure("cache.construct", timings, progress):
                    entry = construct_video_entry(observation, cache, self.device)
                intr, poses, times = (
                    observation[key]
                    for key in (
                        "target_intr",
                        "target_c2w",
                        "observation_times",
                    )
                )
                colors = observation["source_rgb"]
            else:
                fps = DEFAULT_FPS
                with self.measure("geometry.camera", timings, progress):
                    self.mirror.reconstructor_module.to(self.device)
                    camera = self.mirror.infer_camera(
                        frames,
                        frame_indices=torch.zeros(1, dtype=torch.int64),
                        times=torch.zeros(1),
                        fps=fps,
                    )
                with self.measure("geometry_and_cache.first_frame", timings, progress):
                    entry = construct_first_frame_entry(
                        frames[0].permute(1, 2, 0).to(self.device),
                        camera.intr.to(self.device),
                        camera.c2w.to(self.device),
                        fps,
                        self.mirror,
                        cache,
                        pure=True,
                    )
                intr = camera.intr
                poses = camera.c2w
                times = torch.zeros(1)
                colors = frames.permute(0, 2, 3, 1)
            with self.measure("cache.transfer_to_cpu", timings, progress):
                entry = entry.to("cpu")
            return BuiltCache(
                entry=entry,
                colors=(colors.cpu() * 255).round().clamp(0, 255).byte(),
                intrinsics=intr.float().cpu(),
                c2w=poses.float().cpu(),
                times=times.float().cpu(),
                fps=fps,
                frame_count=OUTPUT_FRAMES,
                source_type=prepared.kind,
                image_size=self.config.cache.image_size,
                timings=timings,
            )
