from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from alc_video.conditioning.video import (
    direct_video_frame_count,
    sample_keyframe_indices,
    validate_drop_source_frames,
)
from alc_video.encoders.geometry.types import CameraTrajectory
from alc_video.encoders.visual.base import CONTEXT_MAX, CONTEXT_MIN, is_keyframe_count
from alc_video.inference.config import InferenceRequest, RolloutRequest, SourceInput
from alc_video.inference.sampling import validate_sampling_options
from alc_video.models.video_model.temporal import wan_vae_latent_frame_count
from alc_video.utils.errors import input_errors
from alc_video.utils.logging import stage
from alc_video.utils.video import fit_center_crop_rgb, read_video_clip


@dataclass(frozen=True)
class InferenceInputs:
    frames: torch.Tensor
    fps: float | None
    camera: CameraTrajectory | None


@dataclass(frozen=True)
class RolloutInputs:
    first_frame: torch.Tensor
    camera: CameraTrajectory
    prompts: tuple[str, ...]
    keyframe_indices: torch.Tensor


def load_cameras(path: Path) -> CameraTrajectory:
    camera = load_file(path)
    intr, c2w, times, fps = (camera[name] for name in ("intr", "c2w", "times", "fps"))
    count = times.numel()
    if (
        intr.shape != (count, 3, 3)
        or c2w.shape != (count, 4, 4)
        or times.shape != (count,)
    ):
        raise ValueError(
            "camera requires intr [T,3,3], c2w [T,4,4], times [T], fps scalar"
        )
    if fps.shape != () or float(fps) <= 0:
        raise ValueError("camera fps must be a positive scalar")
    if not all(
        torch.is_floating_point(x) and bool(torch.isfinite(x).all())
        for x in (intr, c2w, times, fps)
    ):
        raise ValueError("camera tensors must be finite floating point")
    return CameraTrajectory(intr, c2w, times, float(fps))


def read_source(
    source: SourceInput, image_size: tuple[int, int]
) -> tuple[torch.Tensor, float | None]:
    if source.type == "image":
        with Image.open(source.path) as image:
            frames = fit_center_crop_rgb(np.array(image.convert("RGB")), image_size)[
                None
            ]
        return frames, None
    if source.type != "video":
        raise ValueError("source.type must be image or video")
    return read_video_clip(
        source.path,
        start_frame=source.start_frame,
        min_frame_count=12,
        max_frame_count=source.num_frames,
        image_size=image_size,
    )


def prepare_inference_inputs(request: InferenceRequest) -> InferenceInputs:
    with input_errors():
        sampling = request.generation.sampling
        validate_sampling_options(sampling.steps, sampling.control_scales)
        with stage("input.decode", path=request.source.path):
            frames, fps = read_source(
                request.source, request.generation.cache.image_size
            )
        camera = None
        if request.source.type == "image":
            if request.drop_source_frames:
                raise ValueError("drop_source_frames requires video input")
            if request.mode not in ("i2v", "geometry_i2v"):
                raise ValueError(
                    "image input requires --mode i2v or --mode geometry_i2v"
                )
            if request.camera is None:
                raise ValueError("image input requires --camera")
            with stage("input.cameras"):
                camera = load_cameras(request.camera)
            assert camera.times is not None
            if (len(camera.times) - 1) % 4:
                raise ValueError("Wan target frame count must satisfy 4T+1")
        else:
            if request.mode != "control":
                raise ValueError("video input requires --mode control")
            count = direct_video_frame_count(len(frames))
            if request.keyframes is None:
                sample_keyframe_indices(count, request.num_keyframes)
            validate_drop_source_frames(
                request.drop_source_frames,
                views=request.num_keyframes
                if request.keyframes is None
                else len(request.keyframes),
            )
            if request.camera is not None:
                with stage("input.cameras"):
                    raw = load_file(request.camera)
                if (
                    raw["intr"].shape != (count, 3, 3)
                    or raw["c2w"].shape != (count, 4, 4)
                    or ("times" in raw and raw["times"].shape != (count,))
                ):
                    raise ValueError(
                        "direct-video camera must match the generated frame count"
                    )
                camera = CameraTrajectory(
                    raw["intr"],
                    raw["c2w"],
                    raw.get("times"),
                    None if "fps" not in raw else float(raw["fps"]),
                )
        return InferenceInputs(frames, fps, camera)


def prepare_rollout_inputs(request: RolloutRequest) -> RolloutInputs:
    with input_errors():
        size = request.chunk_size
        if size < 41 or (size - 1) % 4:
            raise ValueError(
                f"rollout chunk_size must be >=41 and 4n+1, got {size}. Use --chunk-size 81 for 21 sampled keyframes and up to 20 new Cache keyframes."
            )
        keyframes = wan_vae_latent_frame_count(size)
        if request.use_cache and not is_keyframe_count(keyframes):
            raise ValueError(
                f"rollout chunk_size={size} requires {keyframes} Cache keyframes; encoders support {CONTEXT_MIN}..{CONTEXT_MAX}. Use --chunk-size 81 or another 4n+1 length in 41..81."
            )
        indices = sample_keyframe_indices(size, keyframes)
        if request.use_cache and request.mode != "geometry_i2v":
            raise ValueError("Cache rollout requires --mode geometry_i2v")
        if request.mode not in ("i2v", "geometry_i2v"):
            raise ValueError("rollout mode must be i2v or geometry_i2v")
        sampling = request.generation.sampling
        validate_sampling_options(sampling.steps, sampling.control_scales)
        with stage("input.cameras"):
            camera = load_cameras(request.camera)
        assert camera.times is not None
        frames = len(camera.times)
        if frames < size or (frames - 1) % (size - 1):
            raise ValueError(
                f"rollout camera length={frames} must be 1 + ({size}-1)*n (n >= 1). Regenerate or resample the trajectory; 81-frame chunks require 81, 161, 241, ... camera frames."
            )
        with stage("input.decode"):
            source, _fps = read_source(
                request.source, request.generation.cache.image_size
            )
        count = (frames - 1) // (size - 1)
        prompts = (
            request.prompts * count if len(request.prompts) == 1 else request.prompts
        )
        if len(prompts) != count:
            raise ValueError(
                f"provide one --prompt or {count} per-chunk prompts, got {len(request.prompts)}"
            )
        return RolloutInputs(source[0].permute(1, 2, 0), camera, prompts, indices)
