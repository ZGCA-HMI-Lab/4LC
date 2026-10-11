from __future__ import annotations

from dataclasses import dataclass, fields
from time import perf_counter

import av
import numpy as np
import torch
from PIL import Image, ImageOps

OUTPUT_FRAMES = 81
DEFAULT_FPS = 16


@dataclass(frozen=True)
class MediaInfo:
    path: str
    kind: str
    width: int
    height: int
    times: tuple[float, ...] = ()
    fps: float | None = None

    @property
    def duration(self):
        return self.times[-1] - self.times[0] if self.times else 0.0

    def summary(self):
        return {
            "media_path": self.path,
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "fps": self.fps,
            "frames": len(self.times) if self.times else 1,
        }


def inspect_media(path: str, kind: str) -> MediaInfo:
    if kind == "image":
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            return MediaInfo(path, kind, *image.size)
    if kind != "video":
        raise ValueError("Choose an image or video.")
    times = []
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("This file has no video track.")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError("The video has no frame timestamps. Export it as MP4.")
            if times and frame.time <= times[-1]:
                raise ValueError(
                    "The video timestamps are not increasing. Export it as MP4."
                )
            times.append(float(frame.time))
            if len(times) == 1:
                width, height = frame.width, frame.height
                if frame.rotation % 180:
                    width, height = height, width
        if len(times) < 2:
            raise ValueError(
                "The video needs at least two frames. Upload a still as an image."
            )
        fps = (
            float(stream.average_rate)
            if stream.average_rate
            else (len(times) - 1) / (times[-1] - times[0])
        )
    return MediaInfo(path, kind, width, height, tuple(times), fps)


@dataclass(frozen=True)
class InputSettings:
    start: float
    sampling_fps: float
    crop: tuple[int, int, int, int]

    @property
    def end(self):
        return self.start + (OUTPUT_FRAMES - 1) / self.sampling_fps

    @classmethod
    def parse(cls, value: dict, media: MediaInfo, size: tuple[int, int]):
        unknown = set(value) - {item.name for item in fields(cls)} - {"media_path"}
        if unknown:
            raise ValueError(f"Unknown input settings: {', '.join(sorted(unknown))}")
        start, fps = float(value["start"]), float(value["sampling_fps"])
        crop = tuple(value["crop"])
        if not all(np.isfinite([start, fps])) or fps <= 0:
            raise ValueError("Choose a positive sampling FPS and a finite start time.")
        end = start + (OUTPUT_FRAMES - 1) / fps
        if media.kind == "video" and not 0 <= start < end <= media.duration + 1e-5:
            raise ValueError(
                f"Selected range {start:.3f}–{end:.3f} s must lie within "
                f"0–{media.duration:.3f} s. Adjust the start or sampling FPS."
            )
        if len(crop) != 4 or any(not isinstance(v, int) for v in crop):
            raise ValueError(
                "Crop coordinates must be integer pixels: x, y, width, height."
            )
        x, y, w, h = crop
        if (
            min(x, y) < 0
            or min(w, h) <= 0
            or x + w > media.width
            or y + h > media.height
        ):
            raise ValueError("Keep the crop inside the original image.")
        target_h, target_w = size
        if abs(w - h * target_w / target_h) > 1.5:
            raise ValueError("The crop must match the generation aspect ratio.")
        return cls(start, fps, crop)


@dataclass(frozen=True)
class PreparedInput:
    kind: str
    frames: torch.Tensor
    observation_fps: float | None
    settings: InputSettings
    decode_seconds: float


def prepare_input(
    media: MediaInfo, value: dict, size: tuple[int, int]
) -> PreparedInput:
    started = perf_counter()
    settings = InputSettings.parse(value, media, size)
    x, y, w, h = settings.crop

    def pixels(image):
        return np.array(
            image.convert("RGB")
            .crop((x, y, x + w, y + h))
            .resize(size[::-1], Image.Resampling.LANCZOS)
        )

    if media.kind == "image":
        with Image.open(media.path) as image:
            frames = [pixels(ImageOps.exif_transpose(image))]
        fps = None
    else:
        # Resample the selected time span to 81 frames using nearest timestamps.
        targets = (
            np.linspace(settings.start, settings.end, OUTPUT_FRAMES) + media.times[0]
        )
        times = np.asarray(media.times)
        right = np.searchsorted(times, targets).clip(0, len(times) - 1)
        left = np.maximum(right - 1, 0)
        selected = np.where(
            abs(times[left] - targets) <= abs(times[right] - targets), left, right
        )
        needed = set(selected.tolist())
        decoded = {}
        with av.open(media.path) as container:
            container.streams.video[0].thread_type = "AUTO"
            for i, frame in enumerate(container.decode(video=0)):
                if i in needed:
                    image = frame.to_image()
                    if frame.rotation:
                        image = image.rotate(frame.rotation, expand=True)
                    decoded[i] = pixels(image)
                if i >= selected[-1]:
                    break
        frames = [decoded[i] for i in selected]
        fps = settings.sampling_fps
    tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255)
    return PreparedInput(media.kind, tensor, fps, settings, perf_counter() - started)
