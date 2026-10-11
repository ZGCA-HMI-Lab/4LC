from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from alc_video.data.config import DataConfig
from alc_video.data.schema import (
    CONTEXT_MAX,
    CONTEXT_MIN,
    TARGET_FRAMES,
    RawBatch,
)
from alc_video.utils.video import VideoInfo, iter_video_windows

SAMPLE_FPS = 16.0
DURATION_SECONDS = (TARGET_FRAMES - 1) / SAMPLE_FPS


def uniform_source_positions(context_count: int) -> tuple[int, ...]:
    if (
        isinstance(context_count, bool)
        or not isinstance(context_count, int)
        or not CONTEXT_MIN <= context_count <= CONTEXT_MAX
    ):
        raise ValueError(
            f"context_count must be an integer in [{CONTEXT_MIN},{CONTEXT_MAX}]"
        )
    return tuple(
        index * (TARGET_FRAMES - 1) // (context_count - 1)
        for index in range(context_count)
    )


def physical_time_frame_indices(
    frame_start: int,
    *,
    media_fps: float,
    sample_fps: float,
    frame_count: int,
) -> tuple[int, ...]:
    if (
        isinstance(frame_start, bool)
        or not isinstance(frame_start, int)
        or frame_start < 0
        or not math.isfinite(float(media_fps))
        or not math.isfinite(float(sample_fps))
        or media_fps <= 0.0
        or sample_fps <= 0.0
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise ValueError("physical-time sampling inputs are invalid")
    indices = tuple(
        math.floor(frame_start + position * media_fps / sample_fps + 0.5)
        for position in range(TARGET_FRAMES)
    )
    if (
        indices[0] != frame_start
        or indices[-1] >= frame_count
        or any(right <= left for left, right in zip(indices, indices[1:], strict=False))
    ):
        raise ValueError(
            f"physical-time sampling cannot produce {TARGET_FRAMES} distinct frames"
        )
    return indices


def sampling_seed(
    planning_seed: int,
    dataset: str,
    scene_id: str,
    clip_id: str,
) -> int:
    payload = f"{planning_seed}\0{dataset}\0{scene_id}\0{clip_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _read_metadata(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError as error:
        raise FileNotFoundError(f"missing SpatialVID metadata: {path}") from error
    if not rows:
        raise ValueError(f"SpatialVID metadata is empty: {path}")
    return rows


def _caption_data(path: Path) -> tuple[str, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FileNotFoundError(f"missing SpatialVID caption: {path}") from error
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid SpatialVID caption JSON: {path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"SpatialVID caption must contain a JSON object: {path}")
    value = data.get("SceneDescription")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SpatialVID caption must contain SceneDescription: {path}")
    return value, raw


def _metadata_float(row: dict[str, str], name: str, *, source_id: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"SpatialVID {source_id!r} metadata {name!r} must be numeric"
        ) from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"SpatialVID {source_id!r} metadata {name!r} must be positive")
    return value


def _metadata_int(row: dict[str, str], name: str, *, source_id: str) -> int:
    value = _metadata_float(row, name, source_id=source_id)
    integer = int(value)
    if value != integer:
        raise ValueError(
            f"SpatialVID {source_id!r} metadata {name!r} must be an integer"
        )
    return integer


def _metadata_resolution(
    row: dict[str, str],
    *,
    source_id: str,
) -> tuple[int, int]:
    raw = str(row.get("resolution", "")).strip().lower()
    parts = raw.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"SpatialVID {source_id!r} metadata resolution is invalid: {raw!r}"
        )
    try:
        width, height = (int(value) for value in parts)
    except ValueError as error:
        raise ValueError(
            f"SpatialVID {source_id!r} metadata resolution is invalid: {raw!r}"
        ) from error
    if width <= 0 or height <= 0:
        raise ValueError(
            f"SpatialVID {source_id!r} metadata resolution must be positive"
        )
    return width, height


def _source_path(root: Path, value: str, *, name: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{name} must be a relative path below the dataset root")
    return root / relative


@dataclass(frozen=True)
class SpatialVIDClip:
    sample_id: str
    video: Path
    caption: Path
    info: VideoInfo
    seed: int
    frame_indices: tuple[int, ...]
    source_positions: tuple[int, ...]

    @property
    def candidate_id(self) -> str:
        return hashlib.sha256(f"spatialvid\0{self.sample_id}\0{0}".encode()).hexdigest()


def _selection_rank(clip: SpatialVIDClip, seed: int) -> tuple[bytes, str]:
    identifier = clip.candidate_id
    return (
        hashlib.sha256(f"{seed}\0spatialvid\0train\0{identifier}".encode()).digest(),
        identifier,
    )


def select_clips(
    clips: list[SpatialVIDClip], count: int, seed: int
) -> tuple[SpatialVIDClip, ...]:
    population = len(clips)
    if count > population:
        raise ValueError(
            f"stratified selection requested {count} candidates from pool {population}"
        )
    grouped: dict[int, list[SpatialVIDClip]] = {}
    for clip in clips:
        grouped.setdefault(len(clip.source_positions), []).append(clip)
    sizes = {key: len(values) for key, values in grouped.items()}
    quotas = {key: count * size // population for key, size in sizes.items()}
    remaining = count - sum(quotas.values())
    order = sorted(
        sizes, key=lambda key: (-(count * sizes[key] % population), str(key))
    )
    for key in order[:remaining]:
        quotas[key] += 1
    selected = []
    for key in sorted(grouped, key=str):
        ranked = sorted(grouped[key], key=lambda clip: _selection_rank(clip, seed))
        selected.extend(ranked[: quotas[key]])

    return tuple(sorted(selected, key=lambda clip: _selection_rank(clip, seed)))


def scan_spatialvid(
    config: DataConfig,
) -> tuple[tuple[SpatialVIDClip, ...], dict[str, int]]:
    rows = _read_metadata(config.root / "data/train/SpatialVID_HQ_metadata.csv")
    raw_root = config.root / "SpatialVid/HQ"
    clips = []
    seen = set()
    counts = {"rows": len(rows), "short_clip": 0}
    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        video_rel = str(row.get("video path", "")).strip()
        annotation_rel = str(row.get("annotation path", "")).strip()
        if not sample_id or not video_rel or not annotation_rel:
            raise ValueError(
                "SpatialVID metadata row must include id, video path, and annotation path"
            )
        if sample_id in seen:
            raise ValueError(f"duplicate SpatialVID scene ID: {sample_id}")
        seen.add(sample_id)
        fps = _metadata_float(row, "fps", source_id=sample_id)
        frame_count = _metadata_int(row, "num frames", source_id=sample_id)
        width, height = _metadata_resolution(row, source_id=sample_id)
        video = _source_path(raw_root, video_rel, name="SpatialVID video path")
        caption = (
            _source_path(raw_root, annotation_rel, name="SpatialVID annotation path")
            / "caption.json"
        )
        remaining_frames = (frame_count - 1) - DURATION_SECONDS * fps
        if remaining_frames < 0.0:
            counts["short_clip"] += 1
            continue
        seed = sampling_seed(
            config.planning_seed, "spatialvid", sample_id, "clip000000"
        )
        rng = random.Random(seed)
        context_count = rng.randint(11, 21)
        frame_start = rng.randint(0, math.floor(remaining_frames))
        indices = physical_time_frame_indices(
            frame_start, media_fps=fps, sample_fps=SAMPLE_FPS, frame_count=frame_count
        )
        clips.append(
            SpatialVIDClip(
                sample_id,
                video,
                caption,
                VideoInfo(frame_count, fps, height, width, ""),
                seed,
                indices,
                uniform_source_positions(context_count),
            )
        )
    if not clips:
        raise ValueError(f"SpatialVID has no eligible clips: {counts}")
    counts["eligible"] = len(clips)
    selected = select_clips(
        clips,
        len(clips) if config.sample_count is None else config.sample_count,
        config.planning_seed,
    )
    counts["selected"] = len(selected)
    counts["not_selected"] = len(clips) - len(selected)
    return selected, counts


def decode_training_clip(clip: SpatialVIDClip, image_size: tuple[int, int]) -> RawBatch:
    text, _raw = _caption_data(clip.caption)
    frames = next(
        iter_video_windows(
            clip.video,
            (clip.frame_indices,),
            image_size,
            info=clip.info,
            max_timestamp_drift_seconds=1.0 / SAMPLE_FPS,
        )
    )
    return {
        "dataset": "spatialvid",
        "source_path": str(clip.video),
        "sample_id": clip.sample_id,
        "sampling_seed": clip.seed,
        "text": text,
        "transport": "velocity",
        "fps": torch.tensor(SAMPLE_FPS, dtype=torch.float32),
        "frame_indices": torch.tensor(clip.frame_indices, dtype=torch.int64),
        "source_positions": torch.tensor(clip.source_positions, dtype=torch.int64),
        "target_positions": torch.arange(TARGET_FRAMES, dtype=torch.int64),
        "source_video": frames,
    }
