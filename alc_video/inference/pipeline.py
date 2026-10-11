from __future__ import annotations

import time
from contextlib import ExitStack

import torch

from alc_video.conditioning.first_frame import (
    build_image_conditions,
)
from alc_video.conditioning.video import (
    build_video_conditions,
)
from alc_video.inference.artifacts import request_config, save_result_config
from alc_video.inference.config import InferenceRequest
from alc_video.inference.generation import (
    build_cache,
    generate_chunk,
    load_inference_components,
)
from alc_video.inference.inputs import prepare_inference_inputs
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
            assert inputs.camera is not None
            readouts, intr, c2w, times, fps = build_image_conditions(
                inputs.frames,
                inputs.camera,
                mode=request.mode,
                mirror=components.mirror,
                cache=cache,
                device=device,
            )
        else:
            readouts, intr, c2w, times, fps = build_video_conditions(
                inputs.frames,
                inputs.fps,
                inputs.camera,
                source_path=request.source.path,
                mirror=components.mirror,
                encoders=components.encoders,
                cache=cache,
                device=device,
                keyframes=request.keyframes,
                num_keyframes=request.num_keyframes,
                transport=request.transport,
                drop_source_frames=request.drop_source_frames,
                propagate_rgb=request.propagate_rgb,
            )
        generator = torch.Generator(device="cpu").manual_seed(settings.seed % (2**63))
        LOG.info(
            "request ready frames=%s image=%s fps=%s camera=%s sampling_steps=%s",
            len(times),
            settings.cache.image_size,
            fps,
            request.camera or "reconstructed source",
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
