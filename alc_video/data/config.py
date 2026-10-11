from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    root: Path
    planning_seed: int
    image_size: tuple[int, int]
    sample_count: int | None = None
    source: str = "spatialvid"
