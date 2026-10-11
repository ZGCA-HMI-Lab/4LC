from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from torch import Tensor

from latent_cache.types import (
    CacheEntry,
    CompositionResult,
    LatentCacheConfig,
    Readout,
    Representation,
    VisualEncoder,
)


class LatentCache:
    def __init__(
        self,
        config: LatentCacheConfig,
        encoders: Mapping[str, tuple[VisualEncoder, str]],
    ) -> None:
        if not isinstance(config, LatentCacheConfig):
            raise TypeError("config must be LatentCacheConfig")
        if not isinstance(encoders, Mapping) or not encoders:
            raise ValueError("encoders must be a nonempty mapping")
        self.config = config
        self.encoders = dict(encoders)
        self.representations = {}
        for name, (encoder, stream) in self.encoders.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(stream, str)
                or not stream
            ):
                raise ValueError(
                    "representation and stream names must be nonempty strings"
                )
            if not isinstance(encoder.representation, Representation):
                raise TypeError("encoder must declare a Representation")
            encoder.representation.spatial_grid(config.image_size)
            self.representations[name] = encoder.representation
        self._entries: list[CacheEntry] = []

    @property
    def entries(self) -> list[CacheEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def construct(
        self,
        frames: Mapping[str, Tensor],
        depth: Tensor,
        valid: Tensor,
        intrinsics: Tensor,
        c2w: Tensor,
        times: Tensor,
        frame_indices: Tensor,
        num_frames: int,
        *,
        fps: float | Tensor,
        timeline_times: Tensor,
        encoding_frame_indices: Tensor | None = None,
        encoder_frame_indices: Mapping[str, Tensor] | None = None,
        velocity: tuple[Tensor, Tensor] | None = None,
        motion_mask: Tensor | None = None,
        dropped: Tensor | None = None,
        feature_valid: Mapping[str, Tensor] | None = None,
        motion_threshold: float | None = None,
        tag: str | None = None,
    ) -> CacheEntry:
        from latent_cache.construction import construct

        entry = construct(
            self.config,
            self.encoders,
            index=len(self._entries),
            frames=frames,
            depth=depth,
            valid=valid,
            intrinsics=intrinsics,
            c2w=c2w,
            times=times,
            frame_indices=frame_indices,
            num_frames=num_frames,
            fps=fps,
            timeline_times=timeline_times,
            encoding_frame_indices=encoding_frame_indices,
            encoder_frame_indices=encoder_frame_indices,
            velocity=velocity,
            motion_mask=motion_mask,
            dropped=dropped,
            feature_valid=feature_valid,
            motion_threshold=motion_threshold,
            tag=tag,
        )
        self._entries.append(entry)
        return entry

    def read(
        self,
        entry: CacheEntry,
        times: Tensor,
        intrinsics: Tensor,
        c2w: Tensor,
        *,
        frame_indices: Tensor | None = None,
        valid: Tensor | None = None,
        fill: Mapping[str, Tensor] | None = None,
        static: bool = False,
    ) -> dict[str, Readout]:
        from latent_cache.readout import read

        return read(
            self.config,
            entry,
            times,
            intrinsics,
            c2w,
            frame_indices=frame_indices,
            valid=valid,
            fill=fill,
            static=static,
        )

    def compose(
        self,
        times: Tensor,
        intrinsics: Tensor,
        c2w: Tensor,
        *,
        entries: Sequence[CacheEntry] | None = None,
        frame_indices: Tensor | None = None,
        valid: Tensor | None = None,
        order: Literal["earliest_first", "latest_first"] = "earliest_first",
        max_entries: int = 8,
        coverage_threshold: float = 0.95,
    ) -> CompositionResult:
        from latent_cache.composition import compose

        return compose(
            self,
            list(self._entries if entries is None else entries),
            times,
            intrinsics,
            c2w,
            frame_indices=frame_indices,
            valid=valid,
            order=order,
            max_entries=max_entries,
            coverage_threshold=coverage_threshold,
        )
