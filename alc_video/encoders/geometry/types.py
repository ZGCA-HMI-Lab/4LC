from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WorldMirrorCameraState:
    intr: torch.Tensor
    c2w: torch.Tensor


@dataclass(frozen=True)
class WorldMirrorCameraSceneState:
    camera: WorldMirrorCameraState
    scene_scale: float
    translation_drift: float


@dataclass(frozen=True)
class WorldMirrorCameraFirstDepthState:
    camera: WorldMirrorCameraState
    depth: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class WorldMirrorSourceState:
    depth: torch.Tensor
    valid: torch.Tensor
    motion: dict[str, torch.Tensor] | None


@dataclass(frozen=True)
class WorldMirrorPairState:
    source_camera: WorldMirrorCameraState
    target_camera: WorldMirrorCameraState
    source: WorldMirrorSourceState


@dataclass(frozen=True)
class CameraTrajectory:
    intr: torch.Tensor
    c2w: torch.Tensor
    times: torch.Tensor | None
    fps: float | None

    def to(self, device: torch.device) -> CameraTrajectory:
        return CameraTrajectory(
            self.intr.to(device),
            self.c2w.to(device),
            None if self.times is None else self.times.to(device),
            self.fps,
        )
