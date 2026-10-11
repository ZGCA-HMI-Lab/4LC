from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path

import torch

from alc_video.conditioning.first_frame import (
    build_image_conditions,
    infer_image_camera,
)
from alc_video.conditioning.geometry import estimate_context_geometry
from alc_video.encoders.geometry.types import CameraTrajectory
from alc_video.inference.artifacts import (
    completed_rollout_output,
    request_config,
    save_result_config,
)
from alc_video.inference.config import RolloutRequest
from alc_video.inference.generation import (
    build_cache,
    generate_chunk,
    load_inference_components,
)
from alc_video.inference.inputs import prepare_rollout_inputs
from alc_video.inference.rollout.history import (
    RolloutState,
    initialize_history,
    read_history,
    write_history,
)
from alc_video.inference.trajectory import (
    generate_rollout_trajectory,
    scene_focus_distance,
)
from alc_video.utils.errors import input_errors
from alc_video.utils.logging import LOG, configure_logging, stage
from alc_video.utils.video import VideoWriter


@torch.inference_mode()
def run_rollout(
    request: RolloutRequest, *, resume: bool = False, log_level: str = "INFO"
) -> Path:
    if not isinstance(request, RolloutRequest):
        raise TypeError("run_rollout requires RolloutRequest")
    configure_logging(log_level)
    output = request.output
    with input_errors():
        if not resume and (output.exists() or output.with_suffix(".json").exists()):
            raise FileExistsError(
                f"rollout output already exists: {output}; choose another --output or use --resume"
            )
    if resume and completed_rollout_output(request):
        configure_logging(log_level, path=output.with_suffix(".log"))
        LOG.info("rollout.skip reason=already_complete output=%s", output)
        return output
    inputs = prepare_rollout_inputs(request)
    settings = request.generation
    started = time.perf_counter()
    configure_logging(log_level, path=output.with_suffix(".log"))
    LOG.info(
        "rollout start seed=%s device=%s mode=%s use_cache=%s output=%s",
        settings.seed,
        settings.device,
        request.mode,
        request.use_cache,
        output,
    )
    LOG.debug("effective rollout config=%r", request_config(request))
    with ExitStack() as resources:
        components = load_inference_components(settings, resources)
        cache = build_cache(settings, components)
        device = torch.device(settings.device)
        count = len(inputs.prompts)
        state = RolloutState(inputs.first_frame.to(device))
        source_camera = infer_image_camera(
            inputs.first_frame.permute(2, 0, 1)[None],
            inputs.fps,
            components.mirror,
        ).to(device)
        if request.use_cache:
            initialize_history(cache, components, state, source_camera)
            seed = state.history[0]
            distance = scene_focus_distance(seed.depth[0, 0], seed.valid[0, 0])
        else:
            depth, valid, _moving = estimate_context_geometry(
                components.mirror,
                state.overlap[None],
                source_camera.intr,
                source_camera.c2w,
                torch.zeros(1, device=device, dtype=torch.int64),
                inputs.fps,
                settings.cache,
            )
            distance = scene_focus_distance(depth[0], valid[0])
        camera = generate_rollout_trajectory(
            source_camera,
            request.actions,
            focus_distance=distance,
            revisit=request.revisit,
        ).to(device)
        assert camera.times is not None and camera.fps is not None
        LOG.info(
            "rollout request frames=%s chunks=%s chunk_size=%s fps=%s num_keyframes=%s",
            len(camera.times),
            count,
            request.chunk_size,
            camera.fps,
            len(inputs.keyframe_indices),
        )
        generator = torch.Generator(device="cpu").manual_seed(settings.seed % (2**63))
        if resume:
            LOG.info(
                "rollout.restart reason=incomplete_output from=F0 output=%s", output
            )
            for path in (output, output.with_suffix(".json")):
                path.unlink(missing_ok=True)
        writer = resources.enter_context(VideoWriter(output, fps=camera.fps))
        if request.use_cache:
            writer(state.overlap[None], 0)
        for index, prompt in enumerate(inputs.prompts):
            chunk_started = time.perf_counter()
            LOG.info(
                "rollout.chunk start chunk=%s/%s history_entries=%s",
                index + 1,
                count,
                len(state.history),
            )
            if request.use_cache:
                readouts, intr, c2w = read_history(request, cache, state, camera, index)
            else:
                generator.manual_seed(settings.seed % (2**63))
                start = index * (request.chunk_size - 1)
                stop = start + request.chunk_size
                chunk_camera = CameraTrajectory(
                    camera.intr[start:stop],
                    camera.c2w[start:stop],
                    camera.times[start:stop],
                    camera.fps,
                )
                readouts, intr, c2w, _times, _fps = build_image_conditions(
                    state.overlap.permute(2, 0, 1)[None],
                    chunk_camera,
                    mode=request.mode,
                    mirror=components.mirror,
                    cache=cache,
                    device=device,
                )
                cache.clear()
            generated = generate_chunk(
                settings, components, readouts, intr, c2w, prompt, generator
            )
            if request.use_cache:
                written, rejected = write_history(
                    request,
                    components,
                    cache,
                    state,
                    camera,
                    inputs.keyframe_indices,
                    generated,
                    index,
                )
                writer(generated[1:], writer.frame_count)
            else:
                written, rejected = [], ()
                writer(generated if index == 0 else generated[1:], writer.frame_count)
            LOG.info(
                "rollout.chunk complete chunk=%s/%s written_keyframes=%s rejected_keyframes=%s output_frames=%s elapsed=%.2fs",
                index + 1,
                count,
                len(written),
                len(rejected),
                writer.frame_count,
                time.perf_counter() - chunk_started,
            )
            LOG.debug("rollout.write retained=%s rejected=%s", written, rejected)
            state.overlap = generated[-1].clone()
            del readouts, generated
        if writer.frame_count != len(camera.times):
            raise RuntimeError(
                "rollout emitted frame count differs from requested cameras"
            )
        with stage("output.finalize", path=output):
            writer.finish()
        save_result_config(request, writer.frame_count)
        LOG.info(
            "rollout complete frames=%s chunks=%s elapsed=%.2fs output=%s",
            writer.frame_count,
            count,
            time.perf_counter() - started,
            output,
        )
    return output
