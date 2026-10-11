from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch

from alc_video.utils.logging import LOG

VIDEO_IO_TIMEOUT_SECONDS = 30.0


def fit_center_crop_rgb(frame: Any, image_size: tuple[int, int]) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError(
            f"RGB frame must be a numpy HxWx3 array, got "
            f"{getattr(frame, 'shape', None)}"
        )
    height, width = (int(image_size[0]), int(image_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    image = Image.fromarray(frame[..., :3])
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"RGB frame dimensions must be positive, got {frame.shape}")
    if source_width * height >= source_height * width:
        resized_height = height
        resized_width = max(width, round(source_width * height / source_height))
    else:
        resized_width = width
        resized_height = max(height, round(source_height * width / source_width))
    if image.size != (resized_width, resized_height):
        image = image.resize(
            (resized_width, resized_height),
            Image.Resampling.BICUBIC,
        )
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    image = image.crop((left, top, left + width, top + height))
    values = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(values).permute(2, 0, 1).float().div_(255.0)


def read_video_clip(
    path: str | Path,
    *,
    start_frame: int,
    min_frame_count: int,
    max_frame_count: int,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, float]:
    import av

    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
        raise TypeError("start_frame must be an integer")
    if isinstance(min_frame_count, bool) or not isinstance(min_frame_count, int):
        raise TypeError("min_frame_count must be an integer")
    if isinstance(max_frame_count, bool) or not isinstance(max_frame_count, int):
        raise TypeError("max_frame_count must be an integer")
    if start_frame < 0:
        raise ValueError("start_frame must be nonnegative")
    if min_frame_count <= 0 or max_frame_count < min_frame_count:
        raise ValueError("frame counts must satisfy 0 < min <= max")
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"video does not exist or is empty: {source}")
    stop_frame = start_frame + max_frame_count
    frames: list[torch.Tensor] = []
    try:
        with av.open(str(source), mode="r") as container:
            formats = set(str(container.format.name).split(","))
            if "mp4" not in formats:
                raise ValueError(
                    f"container format must include mp4, got {sorted(formats)}"
                )
            streams = tuple(container.streams.video)
            if len(streams) != 1:
                raise ValueError(
                    f"MP4 must contain exactly one video stream, got {len(streams)}"
                )
            stream = streams[0]
            rate = stream.average_rate
            if rate is None or not math.isfinite(float(rate)) or float(rate) <= 0.0:
                raise ValueError(
                    "MP4 header must provide a positive average frame rate"
                )
            stream.thread_type = "SLICE"
            stream.codec_context.thread_count = 1
            for index, frame in enumerate(container.decode(video=0)):
                if index >= stop_frame:
                    break
                if index < start_frame:
                    continue
                array = frame.to_ndarray(format="rgb24")
                frames.append(fit_center_crop_rgb(array, image_size))
            fps = float(rate)
    except (av.error.FFmpegError, OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"video clip decode failed: {source}: {error}") from error
    if len(frames) < min_frame_count:
        raise ValueError(
            f"video has only {len(frames)} decodable frames from start_frame "
            f"{start_frame}; requires at least {min_frame_count}: {source}"
        )
    return torch.stack(frames).contiguous(), fps


def write_raw_video(
    path: str | Path,
    video: torch.Tensor,
    *,
    fps: float,
) -> None:
    if (
        not isinstance(video, torch.Tensor)
        or video.ndim != 5
        or tuple(int(value) for value in video.shape[:2]) != (1, 3)
        or int(video.shape[2]) <= 0
    ):
        raise ValueError("raw video must be [1,3,T,H,W]")
    if (
        not torch.is_floating_point(video)
        or not bool(torch.isfinite(video).all())
        or float(video.min()) < 0.0
        or float(video.max()) > 1.0
    ):
        raise ValueError("raw video must be finite floating RGB in [0,1]")
    destination = Path(path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"raw video already exists: {destination}")
    with VideoWriter(destination, fps=fps) as writer:
        writer(video[0].permute(1, 2, 3, 0), 0)
        writer.finish()


__all__ = [
    "fit_center_crop_rgb",
    "read_video_clip",
    "write_raw_video",
    "VideoWriter",
]


@dataclass(frozen=True)
class VideoInfo:
    frames: int
    fps: float
    height: int
    width: int
    codec: str


def probe_mp4(path: str | Path) -> VideoInfo:
    import av

    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing non-empty MP4: {path}")
    try:
        with av.open(
            str(path),
            mode="r",
            timeout=(VIDEO_IO_TIMEOUT_SECONDS, VIDEO_IO_TIMEOUT_SECONDS),
        ) as container:
            formats = set(str(container.format.name).split(","))
            if "mp4" not in formats:
                raise ValueError(
                    f"{path}: container format must include mp4, got {sorted(formats)}"
                )
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise ValueError(
                    f"{path}: expected one video stream, got {len(streams)}"
                )
            stream = streams[0]
            frames = int(stream.frames)
            rate = stream.average_rate
            if frames <= 0 or rate is None:
                raise ValueError(
                    f"{path}: MP4 header must provide frame count and average frame rate"
                )
            info = VideoInfo(
                frames=frames,
                fps=float(rate),
                height=int(stream.height),
                width=int(stream.width),
                codec=str(stream.codec_context.name),
            )
    except (av.error.FFmpegError, OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{path}: invalid MP4: {error}") from error
    if (
        info.height <= 0
        or info.width <= 0
        or not math.isfinite(info.fps)
        or info.fps <= 0.0
        or not info.codec
    ):
        raise ValueError(f"{path}: invalid MP4 metadata {info}")
    return info


class SourceMediaError(ValueError):
    pass


def validate_indices(info: VideoInfo, indices: Sequence[int]) -> None:
    if not indices or any(
        right <= left for left, right in zip(indices, indices[1:], strict=False)
    ):
        raise ValueError("frame indices must be non-empty and strictly increasing")
    if indices[0] < 0 or indices[-1] >= info.frames:
        raise ValueError(
            f"video frame range [{indices[0]},{indices[-1]}] outside [0,{info.frames})"
        )


def _iter_video_frames(
    path: str | Path,
) -> Iterator[tuple[np.ndarray, float | None, float | None]]:
    with av.open(
        str(path),
        mode="r",
        timeout=(VIDEO_IO_TIMEOUT_SECONDS, VIDEO_IO_TIMEOUT_SECONDS),
    ) as container:
        streams = tuple(container.streams.video)
        if len(streams) != 1:
            raise SourceMediaError(
                f"video {str(path)} must contain exactly one video stream, got "
                f"{len(streams)}"
            )
        stream = streams[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 1
        for frame in container.decode(video=0):
            pts = getattr(frame, "pts", None)
            time_base = getattr(frame, "time_base", None)
            timestamp = (
                None if pts is None or time_base is None else float(pts * time_base)
            )
            timestamp_resolution = None if time_base is None else float(time_base)
            yield (
                frame.to_ndarray(format="rgb24"),
                timestamp,
                timestamp_resolution,
            )


def _validate_presentation_timeline(
    media_path: str,
    timestamps: Sequence[float],
    timestamp_resolutions: Sequence[float],
    max_drift_seconds: float,
) -> None:
    if len(timestamps) != len(timestamp_resolutions):
        raise ValueError("timestamp values and resolutions differ")
    if (
        isinstance(max_drift_seconds, bool)
        or not isinstance(max_drift_seconds, (int, float))
        or not math.isfinite(float(max_drift_seconds))
        or max_drift_seconds <= 0.0
    ):
        raise ValueError("max timestamp drift must be finite and positive")
    if len(timestamps) < 3:
        return
    frame_duration = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    if frame_duration <= 0.0:
        raise SourceMediaError(f"video {media_path} has non-increasing PTS")
    tolerance = max(
        float(max_drift_seconds),
        1.01 * max(timestamp_resolutions),
    )
    for frame_index, timestamp in enumerate(timestamps[1:-1], start=1):
        expected = timestamps[0] + frame_index * frame_duration
        residual = abs(timestamp - expected)
        if residual > tolerance:
            raise SourceMediaError(
                f"video {media_path} presentation timeline drift exceeds limit "
                f"at frame {frame_index}: timestamp={timestamp:.9f}s, "
                f"expected={expected:.9f}s, residual={residual:.9f}s, "
                f"limit={tolerance:.9f}s"
            )


def iter_video_windows(
    path: str | Path,
    windows: Sequence[Sequence[int]],
    image_size: tuple[int, int],
    *,
    info: VideoInfo,
    max_timestamp_drift_seconds: float | None = None,
) -> Iterator[torch.Tensor]:
    if not windows:
        raise ValueError("video windows must not be empty")
    previous = -1
    for window in windows:
        validate_indices(info, window)
        if window[0] <= previous:
            raise ValueError("video windows must be ordered and non-overlapping")
        previous = window[-1]

    stream = _iter_video_frames(path)
    decoded_index = -1
    observed_size: tuple[int, int] | None = None
    decoded_timestamps: list[float] = []
    timestamp_resolutions: list[float] = []
    try:
        for window in windows:
            frames: list[torch.Tensor] = []
            for wanted_index in window:
                frame = None
                while decoded_index < wanted_index:
                    try:
                        frame, timestamp, timestamp_resolution = next(stream)
                    except StopIteration as error:
                        raise SourceMediaError(
                            f"video {str(path)} ended before requested frame "
                            f"{wanted_index}"
                        ) from error
                    except SourceMediaError:
                        raise
                    except (OSError, ValueError, av.FFmpegError) as error:
                        raise SourceMediaError(
                            f"video {str(path)} failed to decode frame "
                            f"{decoded_index + 1}: {error}"
                        ) from error
                    decoded_index += 1
                    if max_timestamp_drift_seconds is not None:
                        if timestamp is None:
                            raise SourceMediaError(
                                f"video {str(path)} frame {decoded_index} "
                                "has no presentation timestamp"
                            )
                        if (
                            timestamp_resolution is None
                            or not math.isfinite(timestamp_resolution)
                            or timestamp_resolution <= 0.0
                        ):
                            raise SourceMediaError(
                                f"video {str(path)} frame {decoded_index} "
                                "has no positive timestamp resolution"
                            )
                        if not math.isfinite(timestamp):
                            raise SourceMediaError(
                                f"video {str(path)} frame {decoded_index} "
                                "has a non-finite presentation timestamp"
                            )
                        if decoded_timestamps and timestamp <= decoded_timestamps[-1]:
                            raise SourceMediaError(
                                f"video {str(path)} has non-increasing PTS "
                                f"at frame {decoded_index}"
                            )
                        decoded_timestamps.append(timestamp)
                        timestamp_resolutions.append(timestamp_resolution)
                if frame is None:
                    raise RuntimeError(
                        "video decoder did not advance to the requested frame"
                    )
                current_size = (int(frame.shape[0]), int(frame.shape[1]))
                if observed_size is None:
                    observed_size = current_size
                elif current_size != observed_size:
                    raise SourceMediaError(
                        f"video resolution changed from {observed_size} to {current_size}"
                    )
                if current_size != (info.height, info.width):
                    raise SourceMediaError(
                        f"video dimensions differ from probed metadata: {current_size} != "
                        f"{(info.height, info.width)}"
                    )
                frames.append(fit_center_crop_rgb(frame, image_size))
            if max_timestamp_drift_seconds is not None:
                _validate_presentation_timeline(
                    str(path),
                    decoded_timestamps,
                    timestamp_resolutions,
                    max_timestamp_drift_seconds,
                )
            yield torch.stack(frames)
    finally:
        stream.close()


class VideoWriter:
    def __init__(self, output: str | Path, *, fps: float) -> None:
        if not math.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError("fps must be finite and positive")
        destination = Path(output).expanduser().resolve()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"rollout output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination
        self.fps = float(fps)
        self.video_path = destination
        self.frame_count = 0
        self.image_size: tuple[int, int] | None = None
        self._container: Any | None = None
        self._stream: Any | None = None
        self._finalized = False

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        try:
            self.close()
        except Exception:
            if _value is None:
                raise
            LOG.exception("video cleanup failed while handling an execution error")

    def _open_video(self, image_size: tuple[int, int]) -> None:
        import av

        height, width = image_size
        container = av.open(str(self.video_path), mode="w")
        self._container = container
        stream = container.add_stream(
            "libx264", rate=Fraction(str(self.fps)).limit_denominator(100000)
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        self._stream = stream

    def __call__(self, frames: torch.Tensor, start_index: int) -> None:
        if self._finalized:
            raise RuntimeError("rollout artifact writer is finalized")
        if start_index != self.frame_count:
            raise ValueError(
                f"rollout frame start {start_index} != next index {self.frame_count}"
            )
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 4
            or int(frames.shape[0]) <= 0
            or (int(frames.shape[-1]) != 3)
        ):
            raise ValueError("rollout frames must be non-empty [T,H,W,3]")
        if (
            not torch.is_floating_point(frames)
            or not bool(torch.isfinite(frames).all())
            or float(frames.min()) < 0.0
            or (float(frames.max()) > 1.0)
        ):
            raise ValueError("rollout frames must be finite floating RGB in [0,1]")
        image_size = (int(frames.shape[1]), int(frames.shape[2]))
        if self.image_size is None:
            self.image_size = image_size
            self._open_video(image_size)
        elif image_size != self.image_size:
            raise ValueError(
                f"rollout frame size {image_size} != initial size {self.image_size}"
            )
        values = (
            frames.detach()
            .mul(255.0)
            .round()
            .to(device="cpu", dtype=torch.uint8)
            .numpy()
        )
        import av

        if self._container is None or self._stream is None:
            raise RuntimeError("rollout video stream is unavailable")
        for value in values:
            frame = av.VideoFrame.from_ndarray(value, format="rgb24")
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        self.frame_count += int(frames.shape[0])

    def finish(self) -> None:
        if self._container is None or self._stream is None:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = None
        self._stream = None

        self._finalized = True

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
            self._stream = None
