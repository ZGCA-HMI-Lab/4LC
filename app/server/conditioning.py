from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from latent_cache import LatentCache, Readout

from alc_video.conditioning.first_frame import build_pure_first_frame_readouts
from alc_video.encoders.geometry.types import CameraTrajectory


@dataclass(frozen=True)
class Shot:
    camera: CameraTrajectory
    mode: str
    profile: str

    @classmethod
    def parse(cls, value: dict, cache) -> Shot:
        intr = torch.as_tensor(value["intrinsics"], dtype=torch.float32)
        poses = torch.as_tensor(value["c2w"], dtype=torch.float32)
        times = torch.as_tensor(value["times"], dtype=torch.float32)
        count = cache.frame_count
        if (
            intr.shape != (count, 3, 3)
            or poses.shape != (count, 4, 4)
            or times.shape != (count,)
        ):
            raise ValueError(f"Camera requires {count} intrinsics, poses and times")
        if not all(
            bool(torch.isfinite(tensor).all()) for tensor in (intr, poses, times)
        ):
            raise ValueError("Camera parameters must be finite")
        if bool((intr[:, (0, 1), (0, 1)] <= 0).any()):
            raise ValueError("Camera focal lengths must be positive")
        mode = value["mode"]
        allowed = (
            ("i2v", "geometry_i2v") if cache.source_type == "image" else ("control",)
        )
        if mode not in allowed:
            raise ValueError(f"{cache.source_type} input requires mode {allowed}")
        return cls(
            CameraTrajectory(intr, poses, times, cache.fps),
            mode,
            value["profile"],
        )

    def matches(self, other: Shot) -> bool:
        return (
            self.profile == other.profile
            and self.mode == other.mode
            and all(
                torch.equal(getattr(self.camera, name), getattr(other.camera, name))
                for name in ("intr", "c2w", "times")
            )
        )


def move_readouts(readouts: dict[str, Readout], device) -> dict[str, Readout]:
    return {
        name: replace(
            readout,
            **{
                field.name: getattr(readout, field.name).to(device)
                for field in fields(readout)
                if isinstance(getattr(readout, field.name), torch.Tensor)
            },
        )
        for name, readout in readouts.items()
    }


@torch.inference_mode()
def read_conditions(
    cache: LatentCache, entry, camera: CameraTrajectory, mode: str
) -> dict[str, Readout]:
    if mode == "i2v":
        return build_pure_first_frame_readouts(
            entry=entry,
            times=camera.times[None],
            intrinsics=camera.intr[None],
            c2w=camera.c2w[None],
            frame_indices=torch.arange(
                len(camera.times), device=entry.positions.device
            )[None],
        )
    return cache.read(
        entry,
        camera.times[None],
        camera.intr[None],
        camera.c2w[None],
        static=mode == "geometry_i2v",
    )
