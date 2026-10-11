from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from alc_video.conditioning.config import validate_keyframe_count
from alc_video.conditioning.video import (
    sample_keyframe_indices,
    validate_drop_source_frames,
)
from alc_video.inference.config import InferenceRequest, RolloutRequest, SourceInput
from alc_video.inference.media import (
    OUTPUT_FRAMES,
    PreparedInput,
    inspect_media,
    prepare_input,
)
from alc_video.inference.sampling import validate_sampling_options
from alc_video.utils.errors import input_errors
from alc_video.utils.logging import stage


@dataclass(frozen=True)
class RolloutInputs:
    first_frame: torch.Tensor
    fps: float
    prompts: tuple[str, ...]
    keyframe_indices: torch.Tensor


def read_source(source: SourceInput, image_size: tuple[int, int]) -> PreparedInput:
    media = inspect_media(str(source.path), source.type)
    settings = {
        key: value
        for key, value in asdict(source).items()
        if key not in ("path", "type")
    }
    return prepare_input(media, settings, image_size)


def prepare_inference_inputs(request: InferenceRequest) -> PreparedInput:
    with input_errors():
        validate_keyframe_count(request.num_keyframes)
        sampling = request.generation.sampling
        validate_sampling_options(sampling.steps, sampling.control_scales)
        image = request.source.type == "image"
        request.camera.validate(image=image)
        request.time.validate(image=image)
        if image:
            if request.drop_source_frames:
                raise ValueError("drop_source_frames requires video input")
            if request.mode not in ("i2v", "geometry_i2v"):
                raise ValueError(
                    "image input requires --mode i2v or --mode geometry_i2v"
                )
        elif request.mode != "control":
            raise ValueError("video input requires --mode control")
        with stage("input.decode", path=request.source.path):
            prepared = read_source(request.source, request.generation.cache.image_size)
        if not image:
            request.time.validate(
                duration=(OUTPUT_FRAMES - 1) / prepared.settings.sampling_fps
            )
            if request.keyframes is None:
                sample_keyframe_indices(OUTPUT_FRAMES, request.num_keyframes)
            validate_drop_source_frames(
                request.drop_source_frames,
                views=request.num_keyframes
                if request.keyframes is None
                else len(request.keyframes),
            )
        return prepared


def prepare_rollout_inputs(request: RolloutRequest) -> RolloutInputs:
    with input_errors():
        validate_keyframe_count(request.num_keyframes)
        if request.source.type != "image":
            raise ValueError("Rollout starts from a single image")
        if request.chunk_size != OUTPUT_FRAMES:
            raise ValueError(
                "Rollout uses 81-frame windows; --num-keyframes selects 11, 21, 41 or 81 "
                "samples per window. The shared first frame is omitted when writing history."
            )
        if not request.actions:
            raise ValueError("Rollout needs at least one camera action")
        for action in request.actions:
            action.camera.validate(image=True)
            if (
                isinstance(action.chunks, bool)
                or not isinstance(action.chunks, int)
                or action.chunks < 1
            ):
                raise ValueError(
                    "Action chunks must be a positive number of 81-frame windows"
                )
        if request.use_cache and request.mode != "geometry_i2v":
            raise ValueError("Cache rollout requires --mode geometry_i2v")
        if request.mode not in ("i2v", "geometry_i2v"):
            raise ValueError("rollout mode must be i2v or geometry_i2v")
        sampling = request.generation.sampling
        validate_sampling_options(sampling.steps, sampling.control_scales)
        count = sum(action.chunks for action in request.actions) * (
            2 if request.revisit else 1
        )
        prompts = (
            request.prompts * count if len(request.prompts) == 1 else request.prompts
        )
        if len(prompts) != count:
            raise ValueError(
                f"provide one --prompt or {count} per-window prompts, got {len(request.prompts)}"
            )
        with stage("input.decode", path=request.source.path):
            prepared = read_source(request.source, request.generation.cache.image_size)
        indices = sample_keyframe_indices(OUTPUT_FRAMES, request.num_keyframes)
        return RolloutInputs(
            prepared.frames[0].permute(1, 2, 0),
            prepared.settings.sampling_fps,
            prompts,
            indices,
        )
