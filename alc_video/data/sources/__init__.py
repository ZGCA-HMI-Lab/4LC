from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from alc_video.data.config import DataConfig
from alc_video.data.schema import RawBatch
from alc_video.data.sources.spatialvid import decode_training_clip, scan_spatialvid


class SourceRecord(Protocol):
    @property
    def candidate_id(self) -> str: ...


@dataclass(frozen=True)
class DataSource:
    name: str
    scan: Callable[[DataConfig], tuple[tuple[SourceRecord, ...], dict[str, int]]]
    decode: Callable[[SourceRecord, tuple[int, int]], RawBatch]


SOURCES = {
    "spatialvid": DataSource("spatialvid", scan_spatialvid, decode_training_clip)
}
