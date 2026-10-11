"""Lightweight shot settings shared by CLI requests and example descriptions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from alc_video.utils.config import reject_unknown_fields

CAMERA_PRESETS = (
    "original",
    "offset",
    "fixed",
    "dolly",
    "zoom",
    "truck",
    "pedestal",
    "pan",
    "tilt",
    "orbit",
)
TIME_MODES = ("normal", "freeze", "reverse")


def camera_directions(preset: str) -> tuple[str, str]:
    if preset in ("dolly", "zoom"):
        return "out", "in"
    if preset in ("pedestal", "tilt"):
        return "down", "up"
    return "left", "right"


@dataclass(frozen=True)
class CameraControl:
    preset: str = "original"
    direction: int = 1
    amount: float = 0.4

    @classmethod
    def parse(
        cls, value: dict[str, Any], *, default: str, name: str = "camera"
    ) -> CameraControl:
        reject_unknown_fields(value, (f.name for f in fields(cls)), name=name)
        preset = value.get("preset", default)
        if preset not in CAMERA_PRESETS:
            raise ValueError(
                f"{name}.preset={preset!r}; choose one of {CAMERA_PRESETS}"
            )
        negative, positive = camera_directions(preset)
        direction = value.get("direction", positive)
        if direction not in (negative, positive):
            raise ValueError(
                f"{name}.direction={direction!r}; {preset} uses {negative!r} or {positive!r}"
            )
        result = cls(
            preset, -1 if direction == negative else 1, float(value.get("amount", 0.4))
        )
        result.validate()
        return result

    def validate(self, *, image: bool = False) -> None:
        if self.preset not in CAMERA_PRESETS:
            raise ValueError(
                f"camera.preset={self.preset!r}; choose one of {CAMERA_PRESETS}"
            )
        if self.direction not in (-1, 1):
            raise ValueError("camera.direction must be -1 or 1")
        if not math.isfinite(self.amount) or not 0 <= self.amount <= 1:
            raise ValueError(f"camera.amount={self.amount}; use a value from 0 to 1")
        if image and self.preset in ("original", "offset"):
            raise ValueError(
                f"camera.preset={self.preset!r} needs a video input path; choose fixed or a camera move for an image"
            )


@dataclass(frozen=True)
class TimeControl:
    mode: str = "normal"
    at: float = 0.0

    @classmethod
    def parse(cls, value: dict[str, Any]) -> TimeControl:
        reject_unknown_fields(value, (f.name for f in fields(cls)), name="time")
        result = cls(str(value.get("mode", "normal")), float(value.get("at", 0)))
        result.validate()
        return result

    def validate(self, *, image: bool = False, duration: float | None = None) -> None:
        if self.mode not in TIME_MODES:
            raise ValueError(f"time.mode={self.mode!r}; choose one of {TIME_MODES}")
        if not math.isfinite(self.at) or self.at < 0:
            raise ValueError("time.at must be a nonnegative time in seconds")
        if image and self.mode != "normal":
            raise ValueError(
                "time.mode controls a source video; an image has no source timeline"
            )
        if self.mode == "freeze" and duration is not None and self.at > duration:
            raise ValueError(
                f"time.at={self.at} s exceeds the selected video's {duration:.3f} s span"
            )


@dataclass(frozen=True)
class RolloutAction:
    camera: CameraControl
    chunks: int = 1

    @classmethod
    def parse(cls, value: dict[str, Any], *, name: str) -> RolloutAction:
        reject_unknown_fields(
            value, ("preset", "direction", "amount", "chunks"), name=name
        )
        camera = CameraControl.parse(
            {k: v for k, v in value.items() if k != "chunks"},
            default="dolly",
            name=name,
        )
        chunks = value.get("chunks", 1)
        if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
            raise ValueError(
                f"{name}.chunks={chunks!r}; use a positive number of 81-frame windows"
            )
        camera.validate(image=True)
        return cls(camera, chunks)


def control_values(camera: CameraControl) -> dict[str, Any]:
    """Serialize the public direction names for saved configuration and examples."""
    result = asdict(camera)
    result["direction"] = camera_directions(camera.preset)[camera.direction == 1]
    return result
