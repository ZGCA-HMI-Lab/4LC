from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import torch
from latent_cache import LatentCache, Readout
from latent_cache.geometry.candidates import DEPTH_EPS
from latent_cache.geometry.spatial import unproject_depth

from alc_video.conditioning.first_frame import (
    build_pure_first_frame_readouts,
    validate_first_frame_alignment,
)
from alc_video.conditioning.rgb import PixelEncoder, encode_rgb_readouts
from alc_video.data.schema import (
    require_sample_batch_size,
)
from alc_video.encoders.visual.depth import depth_to_rgb_frames
from alc_video.training.augmentation import (
    FIRST_FRAME_ID,
    GEOMETRY_OPERATOR_IDS,
    MOTION_THRESHOLD_ID,
    SOURCE_DROP_ID,
    VISIBILITY_CULLING_ID,
    AugmentationTrace,
    apply_visibility_culling,
    build_source_drop_mask,
)
from alc_video.training.augmentation.geometry import SourceGeometry


def _validate_source_time_contract(
    *,
    values: Mapping[str, Any],
    source_rgb: torch.Tensor,
    source_positions: torch.Tensor,
) -> None:
    batch, count = (int(size) for size in source_positions.shape)
    source_times = values["source_times"]
    if (
        not isinstance(source_times, torch.Tensor)
        or tuple(int(size) for size in source_times.shape) != (batch, count)
        or not torch.is_floating_point(source_times)
        or not bool(torch.isfinite(source_times).all())
    ):
        raise ValueError("source_times must be finite floating [B,K]")
    if (
        source_rgb.ndim != 5
        or tuple(int(size) for size in source_rgb.shape[:2]) != (batch, count)
        or int(source_rgb.shape[-1]) != 3
    ):
        raise ValueError("source_rgb must be [B,K,H,W,3] aligned to source_positions")
    expected_prefix = (batch, count)
    for name in (
        "source_times",
        "source_depth",
        "source_valid",
        "source_intr",
        "source_c2w",
    ):
        value = values[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim < 2
            or tuple(int(size) for size in value.shape[:2]) != expected_prefix
        ):
            raise ValueError(f"{name} K axis must match source_positions")
    target_times = values["target_times"]
    observation_times = values["observation_times"]
    if (
        not isinstance(target_times, torch.Tensor)
        or target_times.ndim != 2
        or int(target_times.shape[0]) != batch
    ):
        raise ValueError("target_times must be [B,T]")
    if (
        not isinstance(observation_times, torch.Tensor)
        or observation_times.ndim != 2
        or tuple(int(size) for size in observation_times.shape)
        != tuple(int(size) for size in target_times.shape)
        or not torch.is_floating_point(observation_times)
        or not bool(torch.isfinite(observation_times).all())
    ):
        raise ValueError(
            "observation_times must be finite floating and match target_times [B,T]"
        )
    if int(observation_times.shape[1]) > 1 and not bool(
        (observation_times[:, 1:] > observation_times[:, :-1]).all()
    ):
        raise ValueError("observation_times must be strictly increasing")
    positions = tuple(int(value) for value in source_positions[0].tolist())
    if (
        not positions
        or tuple(sorted(set(positions))) != positions
        or positions[-1] >= int(target_times.shape[1])
    ):
        raise ValueError("source_positions must be strictly increasing target indices")
    expected_times = observation_times.gather(1, source_positions)
    if not torch.equal(source_times, expected_times):
        raise ValueError(
            "source_times must equal observation_times gathered at source_positions"
        )


def prepare_training_readouts(
    batch: Mapping[str, Any],
    trace: AugmentationTrace | None,
    *,
    visual_encoders,
    cache_config,
    propagate_rgb: bool,
    device: torch.device,
    collect_timing_metrics: bool,
) -> tuple[dict[str, Readout], AugmentationTrace | None, dict[str, float]]:
    events, cpu_metrics = {}, {}

    def profiled(name, operation):
        if not collect_timing_metrics:
            return operation()
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = operation()
            end.record()
            events[name] = (start, end)
            return result
        started = time.perf_counter()
        result = operation()
        cpu_metrics[name] = (time.perf_counter() - started) * 1000.0
        return result

    source_positions = batch["source_positions"]
    source_rgb = batch["source_rgb"]
    _validate_source_time_contract(
        values=batch, source_rgb=source_rgb, source_positions=source_positions
    )
    require_sample_batch_size(int(source_positions.shape[0]))
    names = (
        "source_times",
        "source_intr",
        "source_c2w",
        "observation_times",
        "target_times",
        "fps",
        "target_intr",
        "target_c2w",
        "source_depth",
        "source_valid",
    )
    values = {name: batch[name].to(device=device, non_blocking=True) for name in names}
    source_positions = source_positions.to(device=device, non_blocking=True)
    source_rgb = source_rgb.to(device=device, non_blocking=True)
    first_frame = trace is not None and trace.decision(FIRST_FRAME_ID).applied
    geometry_projection, velocity, dropped, motion_threshold = (
        True,
        None,
        None,
        None,
    )
    if first_frame:
        geometry_projection = trace.decision(FIRST_FRAME_ID).parameters["use_geometry"]
        validate_first_frame_alignment(
            source_positions=source_positions,
            source_times=values["source_times"],
            source_intr=values["source_intr"],
            source_c2w=values["source_c2w"],
            target_times=values["target_times"],
            target_intr=values["target_intr"],
            target_c2w=values["target_c2w"],
        )
        source_positions, source_rgb = source_positions[:, :1], source_rgb[:, :1]
        for name in (
            "source_times",
            "source_intr",
            "source_c2w",
            "source_depth",
            "source_valid",
        ):
            values[name] = values[name][:, :1]
    else:
        motion_names = ("velocity_fwd", "velocity_bwd")
        present = [name for name in motion_names if name in batch]
        if batch["transport"] == "identity" and present:
            raise KeyError("identity transport must not contain motion keys")
        if batch["transport"] == "velocity":
            if len(present) != len(motion_names):
                raise KeyError(
                    f"velocity transport requires all motion keys, got {present}"
                )
            velocity = tuple(
                batch[name].to(device=device, non_blocking=True)
                for name in motion_names
            )
        if trace is not None and trace.decision(MOTION_THRESHOLD_ID).applied:
            motion_threshold = float(
                trace.decision(MOTION_THRESHOLD_ID).parameters["motion_threshold"]
            )
    if trace is not None and trace.decision(VISIBILITY_CULLING_ID).applied:
        geometry = SourceGeometry(
            positions=unproject_depth(
                values["source_depth"], values["source_intr"], values["source_c2w"]
            ),
            valid=values["source_valid"][..., 0] & (values["source_depth"] > DEPTH_EPS),
            depth=values["source_depth"],
        )
        params = trace.decision(VISIBILITY_CULLING_ID).parameters
        culled = apply_visibility_culling(
            geometry,
            values["source_intr"],
            values["source_c2w"],
            strategy=params["strategy"],
            direction=params.get("direction"),
            translation=params.get("translation"),
            degrees=params.get("degrees"),
            focal_scale=params.get("probe_focal_scale"),
        )
        if culled.all_empty:
            raise ValueError(
                f"visibility culling removed all source conditions: {culled.metrics}"
            )
        else:
            values["source_valid"] = culled.valid
            trace = trace.record_metrics(VISIBILITY_CULLING_ID, culled.metrics)
    if trace is not None and trace.decision(SOURCE_DROP_ID).applied:
        dropped = build_source_drop_mask(
            trace.decision(SOURCE_DROP_ID).parameters["dropped_source_ids"],
            batch_size=int(source_positions.shape[0]),
            source_count=int(source_positions.shape[1]),
            device=device,
        )
    frames = {"rgb": source_rgb}
    if "depth" in visual_encoders:
        frames["depth"] = depth_to_rgb_frames(
            values["source_depth"], values["source_valid"]
        )
    encoders = visual_encoders
    if propagate_rgb:
        encoders = {
            "rgb": (PixelEncoder(cache_config.image_size), "rgb"),
            **(
                {"depth": visual_encoders["depth"]}
                if "depth" in visual_encoders
                else {}
            ),
        }
    cache = LatentCache(cache_config, encoders=encoders)
    entry = profiled(
        "cache_construct_ms",
        lambda: cache.construct(
            frames,
            values["source_depth"],
            values["source_valid"][..., 0],
            values["source_intr"],
            values["source_c2w"],
            values["source_times"],
            source_positions,
            int(values["observation_times"].shape[1]),
            fps=values["fps"],
            timeline_times=values["observation_times"],
            velocity=velocity,
            dropped=dropped,
            motion_threshold=motion_threshold,
        ),
    )
    target_positions = torch.arange(
        int(values["target_times"].shape[1]), device=device, dtype=torch.int64
    ).reshape(1, -1)
    readouts = profiled(
        "cache_read_ms",
        lambda: (
            cache.read(
                entry,
                values["target_times"],
                values["target_intr"],
                values["target_c2w"],
                frame_indices=target_positions,
            )
            if geometry_projection
            else build_pure_first_frame_readouts(
                entry=entry,
                times=values["target_times"],
                intrinsics=values["target_intr"],
                c2w=values["target_c2w"],
                frame_indices=target_positions,
            )
        ),
    )
    if propagate_rgb:
        readouts = profiled(
            "rgb_reencode_ms",
            lambda: encode_rgb_readouts(
                readouts,
                {name: encoder for name, (encoder, _stream) in visual_encoders.items()},
            ),
        )
    for encoder, _stream in visual_encoders.values():
        encoder.offload()
    if trace is not None:
        applied = next(
            (
                name
                for name in (*GEOMETRY_OPERATOR_IDS, FIRST_FRAME_ID)
                if trace.decision(name).applied
            ),
            None,
        )
        if applied is not None:
            hit_cells = sum(
                int(value.coverage.sum().item()) for value in readouts.values()
            )
            motion_cells = sum(
                int(value.moving.sum().item()) for value in readouts.values()
            )
            total_cells = sum(value.coverage.numel() for value in readouts.values())
            metrics = dict(trace.decision(applied).metrics or {})
            metrics.update(
                condition_hit_cells=hit_cells,
                condition_motion_cells=motion_cells,
                condition_hit_fraction=hit_cells / total_cells,
            )
            if applied == SOURCE_DROP_ID:
                metrics["dropped_source_count"] = int(dropped.sum().item())
            trace = trace.record_metrics(applied, metrics)
    if events:
        torch.cuda.synchronize(device)
    timings = {
        **cpu_metrics,
        **{
            name: float(start.elapsed_time(end))
            for name, (start, end) in events.items()
        },
    }
    return readouts, trace, timings
