"""Inference-time choices for constructing a video Cache."""

from dataclasses import dataclass, fields
from math import isfinite

from alc_video.utils.config import reject_unknown_fields

KEYFRAME_COUNTS = (11, 21, 41, 81)
DEFAULT_KEYFRAMES = 21


def validate_keyframe_count(value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in KEYFRAME_COUNTS
    ):
        raise ValueError(
            f"num_keyframes must be one of {KEYFRAME_COUNTS}, got {value!r}"
        )


@dataclass(frozen=True)
class CacheConstructionSettings:
    num_keyframes: int = DEFAULT_KEYFRAMES
    motion_threshold: float = 0.1

    def __post_init__(self) -> None:
        validate_keyframe_count(self.num_keyframes)
        if (
            isinstance(self.motion_threshold, bool)
            or not isinstance(self.motion_threshold, (int, float))
            or not isfinite(self.motion_threshold)
            or not 0 <= self.motion_threshold <= 1
        ):
            raise ValueError("motion_threshold must be a number from 0 to 1")

    @classmethod
    def parse(cls, values: dict) -> "CacheConstructionSettings":
        if not isinstance(values, dict):
            raise TypeError("cache settings must be a mapping")
        reject_unknown_fields(
            values, (field.name for field in fields(cls)), name="cache"
        )
        return cls(**values)
