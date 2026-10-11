from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from megatron.energon import (
    MapDataset,
    Sample,
    SavableDataLoader,
    SavableDataset,
    TaskEncoder,
    WorkerConfig,
    get_savable_loader,
    stateless,
)
from megatron.energon.wrappers import RepeatDataset, ShuffleBufferDataset

from alc_video.data.config import DataConfig
from alc_video.data.schema import (
    SAMPLE_BATCH_SIZE,
    RawBatch,
    collate_training_samples,
)
from alc_video.data.sources import (
    SOURCES,
    DataSource,
    SourceRecord,
)


@dataclass
class VideoSample(Sample):
    clip: SourceRecord


class IndexedSamples(SavableDataset[VideoSample]):
    _savable_fields = ("_next_index",)

    def __init__(self, clips: tuple[SourceRecord, ...], worker_config: WorkerConfig):
        super().__init__(worker_config)
        self.clips = clips
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._next_index = 0

    def _indices(self, worker_idx: int | None = None) -> range:
        workers = max(1, self.worker_config.num_workers)
        if worker_idx is None:
            worker_idx = self.worker_config.rank_worker_id()
        worker = self.worker_config.rank * workers + worker_idx
        return range(worker, len(self.clips), self.worker_config.world_size * workers)

    def len_worker(self, worker_idx: int | None = None) -> int:
        return len(self._indices(worker_idx))

    def worker_has_samples(self) -> bool:
        return self.len_worker() > 0

    def __iter__(self):
        indices = self._indices()
        while self._next_index < len(indices):
            index = indices[self._next_index]
            self._next_index += 1
            yield self.restore_sample((index,))
        self._next_index = 0

    def can_restore_sample(self) -> bool:
        return True

    def restore_sample(self, restore_key) -> VideoSample:
        (index,) = restore_key
        clip = self.clips[index]
        return VideoSample(
            __key__=clip.candidate_id,
            __restore_key__=(index,),
            __subflavors__=None,
            __sources__=None,
            clip=clip,
        )

    def config(self) -> dict[str, Any]:
        return {"type": type(self).__qualname__, "sample_count": len(self.clips)}


def _raise_sample_error(error: Exception, _sample, *_sources) -> None:
    raise error


class VideoTaskEncoder(TaskEncoder[VideoSample, dict, dict, dict]):
    __default_failure_tolerance__ = 1

    def __init__(self, config: DataConfig, source: DataSource) -> None:
        super().__init__()
        self.data_config = config
        self.source = source

    def build_encode_sample(self, dataset, *, worker_config):
        return MapDataset(
            dataset,
            self.encode_sample,
            worker_config=worker_config,
            stateless_map_fn=True,
            error_handler=_raise_sample_error,
        )

    @stateless
    def encode_sample(self, sample: VideoSample) -> RawBatch:
        return self.source.decode(sample.clip, self.data_config.image_size)

    @stateless
    def batch(self, samples: list[dict]) -> dict:
        return collate_training_samples(samples)


@dataclass(frozen=True)
class DataBundle:
    loader: SavableDataLoader
    sample_count: int
    worker_config: WorkerConfig
    datasets: tuple[str, ...]
    selection_counts: dict[str, int]


def build_train_loader(cfg: Any, *, accelerator: Any) -> DataBundle:
    rank, world_size = int(accelerator.process_index), int(accelerator.num_processes)
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid data-parallel rank/world: {rank}/{world_size}")
    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=cfg.training.num_workers,
        seed_offset=cfg.runtime.seed,
    )
    selected_source = SOURCES[cfg.data.source]
    clips, counts = selected_source.scan(cfg.data)
    source = IndexedSamples(clips, worker_config)
    if any(
        source.len_worker(index) == 0 for index in range(worker_config.num_workers or 1)
    ):
        raise ValueError("selected clips cannot fill every rank/worker; reduce workers")

    dataset = RepeatDataset(source, worker_config=worker_config)
    if cfg.training.shuffle_buffer_size > 1:
        dataset = ShuffleBufferDataset(
            dataset, size=cfg.training.shuffle_buffer_size, worker_config=worker_config
        )
    encoder = VideoTaskEncoder(cfg.data, selected_source)
    dataset = encoder.build_encode_sample(dataset, worker_config=worker_config)
    dataset = encoder.build_batch(
        dataset,
        batch_size=SAMPLE_BATCH_SIZE,
        batch_drop_last=False,
        packing_buffer_size=None,
        worker_config=worker_config,
    )
    loader = get_savable_loader(dataset, prefetch_factor=cfg.training.prefetch_factor)
    return DataBundle(
        loader, len(clips), worker_config, (selected_source.name,), counts
    )


def close_loaders(loaders: Iterable[SavableDataLoader]) -> None:
    seen: set[int] = set()
    for loader in loaders:
        identity = id(loader)
        if identity in seen:
            continue
        seen.add(identity)
        if not isinstance(loader, SavableDataLoader):
            raise TypeError("loader must be an Energon SavableDataLoader")
        iterator = loader._epoch_iterator
        if iterator is None:
            continue
        iterator.close()
        loader._epoch_iterator = None
        loader._has_workers = False
