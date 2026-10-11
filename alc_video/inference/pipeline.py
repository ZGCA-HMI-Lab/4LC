from __future__ import annotations

import time
from contextlib import ExitStack

import torch

from alc_video.conditioning.first_frame import (
    read_image_conditions,
    reconstruct_image_entry,
)
from alc_video.conditioning.video import (
    build_video_conditions,
    prepare_direct_video_observations,
)
from alc_video.encoders.geometry.types import CameraTrajectory
from alc_video.inference.artifacts import request_config, save_result_config
from alc_video.inference.config import InferenceRequest
from alc_video.inference.generation import (
    build_cache,
    generate_chunk,
    load_inference_components,
)
from alc_video.inference.inputs import prepare_inference_inputs
from alc_video.inference.trajectory import generate_trajectory, scene_focus_distance
from alc_video.utils.errors import input_errors
from alc_video.utils.logging import LOG, configure_logging, stage
from alc_video.utils.video import write_raw_video


@torch.inference_mode()
def run_inference(request: InferenceRequest, *, log_level: str = "INFO"):
    if not isinstance(request, InferenceRequest):
        raise TypeError("run_inference requires InferenceRequest")
    configure_logging(log_level)
    with input_errors():
        if request.output.exists():
            raise FileExistsError(
                f"inference output already exists: {request.output}; choose another --output"
            )
    inputs = prepare_inference_inputs(request)
    configure_logging(log_level, path=request.output.with_suffix(".log"))
    started = time.perf_counter()
    settings = request.generation
    LOG.info(
        "inference start input=%s mode=%s seed=%s device=%s output=%s",
        request.source.path,
        request.mode,
        settings.seed,
        settings.device,
        request.output,
    )
    LOG.debug("effective generation config=%r", request_config(request))
    with ExitStack() as resources:
        components = load_inference_components(settings, resources)
        cache = build_cache(settings, components)
        device = torch.device(settings.device)
        if request.source.type == "image":
            entry, source_camera = reconstruct_image_entry(
                inputs.frames,
                inputs.settings.sampling_fps,
                components.mirror,
                cache,
                device,
            )
            camera = generate_trajectory(
                source_camera,
                request.camera,
                request.time,
                focus_distance=scene_focus_distance(
                    entry.depth[0, 0], entry.valid[0, 0]
                ),
                image=True,
            ).to(device)
            readouts = read_image_conditions(entry, camera, request.mode, cache)
            intr, c2w, times, fps = camera.intr, camera.c2w, camera.times, camera.fps
        else:
            with stage("geometry.reconstruct", input=request.source.path):
                observation = prepare_direct_video_observations(
                    inputs.frames,
                    inputs.observation_fps,
                    components.mirror,
                    components.encoders,
                    keyframes=request.keyframes,
                    num_keyframes=request.num_keyframes,
                    transport=request.transport,
                )
            source_camera = CameraTrajectory(
                observation["target_intr"],
                observation["target_c2w"],
                observation["observation_times"],
                inputs.settings.sampling_fps,
            )
            camera = generate_trajectory(
                source_camera,
                request.camera,
                request.time,
                focus_distance=scene_focus_distance(
                    observation["source_depth"][0], observation["source_valid"][0]
                ),
                image=False,
            )
            readouts, intr, c2w, times, fps = build_video_conditions(
                observation,
                camera,
                encoders=components.encoders,
                cache=cache,
                device=device,
                drop_source_frames=request.drop_source_frames,
                propagate_rgb=request.propagate_rgb,
            )
        generator = torch.Generator(device="cpu").manual_seed(settings.seed % (2**63))
        LOG.info(
            "request ready frames=%s image=%s fps=%s camera=%s sampling_steps=%s",
            len(times),
            settings.cache.image_size,
            fps,
            request.camera.preset,
            settings.sampling.steps,
        )
        frames = generate_chunk(
            settings, components, readouts, intr, c2w, request.prompt, generator
        )
        with stage("output.write", path=request.output):
            write_raw_video(request.output, frames.permute(3, 0, 1, 2)[None], fps=fps)
            save_result_config(request, len(frames))
        LOG.info(
            "inference complete frames=%s elapsed=%.2fs output=%s",
            len(frames),
            time.perf_counter() - started,
            request.output,
        )
    return request.output
