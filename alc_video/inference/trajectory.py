"""CPU camera presets matching the browser's camera and scene-time controls."""

from __future__ import annotations

import math

import torch

from alc_video.encoders.geometry.types import CameraTrajectory
from alc_video.inference.controls import CameraControl, RolloutAction, TimeControl
from alc_video.inference.media import OUTPUT_FRAMES


def scene_focus_distance(depth: torch.Tensor, valid: torch.Tensor) -> float:
    """Use the first observation's valid median depth, as the scene viewer does."""
    values = depth.float().reshape(-1)[valid.reshape(-1)]
    if not values.numel():
        raise ValueError("The first view has no valid depth for a camera move")
    distance = float(values.median())
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("The first view's median depth must be positive and finite")
    return distance


def _rotation(angles: torch.Tensor, axis: str) -> torch.Tensor:
    result = torch.eye(3, dtype=angles.dtype).repeat(len(angles), 1, 1)
    c, s = angles.cos(), angles.sin()
    if axis == "y":
        result[:, 0, 0] = result[:, 2, 2] = c
        result[:, 0, 2], result[:, 2, 0] = s, -s
    else:
        result[:, 1, 1] = result[:, 2, 2] = c
        result[:, 1, 2], result[:, 2, 1] = -s, s
    return result


def generate_trajectory(
    source: CameraTrajectory,
    camera: CameraControl,
    time: TimeControl,
    *,
    focus_distance: float,
    image: bool,
    frames: int = OUTPUT_FRAMES,
) -> CameraTrajectory:
    camera.validate(image=image)
    duration = 0.0 if image else float(source.times[-1])
    time.validate(image=image, duration=duration)
    intr = source.intr[:1].double().cpu().repeat(frames, 1, 1)
    poses = source.c2w[:1].double().cpu().repeat(frames, 1, 1)
    progress = torch.linspace(0, 1, frames, dtype=torch.float64)
    amount = progress * camera.direction * camera.amount
    start_rotation = poses[0, :3, :3].clone()
    start_position = poses[0, :3, 3].clone()
    forward, right = start_rotation[:, 2], start_rotation[:, 0]
    preset = camera.preset
    if preset in ("original", "offset"):
        if len(source.c2w) != frames or len(source.intr) != frames:
            raise ValueError(
                "The source camera path must match the prepared video frames"
            )
        intr = source.intr.double().cpu().clone()
        poses = source.c2w.double().cpu().clone()
        if preset == "offset":
            poses[:, :3, 3] += poses[:, :3, 0] * (
                camera.direction * camera.amount * focus_distance * 0.2
            )
    elif preset == "dolly":
        poses[:, :3, 3] += amount[:, None] * forward * (focus_distance * 0.5)
    elif preset == "zoom":
        # Scaling focal lengths preserves the principal point and pixel aspect.
        intr[:, :2, :2] *= (2**amount)[:, None, None]
    elif preset == "truck":
        poses[:, :3, 3] += amount[:, None] * right * (focus_distance * 0.4)
    elif preset == "pedestal":
        # OpenCV world Y is down; browser world Y is up.
        poses[:, 1, 3] -= amount * (focus_distance * 0.3)
    elif preset == "pan":
        poses[:, :3, :3] = start_rotation @ _rotation(amount * math.pi / 3, "y")
    elif preset == "tilt":
        poses[:, :3, :3] = start_rotation @ _rotation(amount * math.pi / 4, "x")
    elif preset == "orbit":
        rotation = start_rotation @ _rotation(-amount * math.pi / 2, "y")
        pivot = start_position + forward * focus_distance
        poses[:, :3, :3] = rotation
        poses[:, :3, 3] = pivot - rotation[:, :, 2] * focus_distance

    if image:
        times = torch.zeros(frames, dtype=torch.float32)
    elif time.mode == "freeze":
        times = torch.full((frames,), time.at, dtype=torch.float32)
    elif time.mode == "reverse":
        times = source.times.float().cpu().flip(0)
    else:
        times = source.times.float().cpu().clone()
    return CameraTrajectory(intr.float(), poses.float(), times, source.fps)


def generate_rollout_trajectory(
    source: CameraTrajectory,
    actions: tuple[RolloutAction, ...],
    *,
    focus_distance: float,
    revisit: bool = False,
) -> CameraTrajectory:
    """Expand actions once, carrying each endpoint into the following window."""
    if not actions:
        raise ValueError("Rollout needs at least one camera action")
    intr, poses = [source.intr[:1].float().cpu()], [source.c2w[:1].float().cpu()]
    current = source
    for action in actions:
        if action.chunks < 1:
            raise ValueError("Action chunks must be positive")
        target = generate_trajectory(
            current,
            action.camera,
            TimeControl(),
            focus_distance=focus_distance,
            image=True,
            frames=1 + (OUTPUT_FRAMES - 1) * action.chunks,
        )
        intr.append(target.intr[1:])
        poses.append(target.c2w[1:])
        current = CameraTrajectory(
            target.intr[-1:], target.c2w[-1:], torch.zeros(1), source.fps
        )
    all_intr, all_poses = torch.cat(intr), torch.cat(poses)
    if revisit:
        # Return through exactly the same camera poses and lenses, excluding the
        # turnaround frame already present at the end of the outgoing path.
        all_intr = torch.cat((all_intr, all_intr[:-1].flip(0)))
        all_poses = torch.cat((all_poses, all_poses[:-1].flip(0)))
    return CameraTrajectory(
        all_intr, all_poses, torch.zeros(len(all_poses)), source.fps
    )
