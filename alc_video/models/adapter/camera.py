from __future__ import annotations

import torch

from alc_video.models.adapter.alignment import select_wan_causal_endpoints
from alc_video.models.video_model.temporal import (
    target_latent_shape,
    wan_video_frame_count,
)


def wan_camera_condition(
    target_intr: torch.Tensor,
    target_c2w: torch.Tensor,
    target_shape: torch.Tensor | torch.Size | tuple[int, ...] | list[int],
    *,
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch, _channels, latent_frames, height, width = target_latent_shape(target_shape)
    frames = wan_video_frame_count(latent_frames)
    expected_intr = (batch, frames, 3, 3)
    expected_c2w = (batch, frames, 4, 4)
    if (
        not isinstance(target_intr, torch.Tensor)
        or tuple(int(size) for size in target_intr.shape) != expected_intr
    ):
        raise ValueError(
            f"target_intr must be {expected_intr}, "
            f"got {getattr(target_intr, 'shape', None)}"
        )
    if (
        not isinstance(target_c2w, torch.Tensor)
        or tuple(int(size) for size in target_c2w.shape) != expected_c2w
    ):
        raise ValueError(
            f"target_c2w must be {expected_c2w}, "
            f"got {getattr(target_c2w, 'shape', None)}"
        )
    if not torch.is_floating_point(target_intr) or not torch.is_floating_point(
        target_c2w
    ):
        raise TypeError("target camera tensors must be floating point")
    if target_intr.device != device or target_c2w.device != device:
        raise ValueError(
            "target camera tensors and readout features must share one device"
        )
    if not bool(torch.isfinite(target_intr).all()) or not bool(
        torch.isfinite(target_c2w).all()
    ):
        raise ValueError("target camera tensors contain non-finite values")

    image_size = tuple(int(size) for size in image_size)
    if len(image_size) != 2 or any(size <= 0 for size in image_size):
        raise ValueError("image_size must contain two positive dimensions")
    image_height, image_width = image_size
    if image_height % height or image_width % width:
        raise ValueError(
            f"image_size {image_size} must be divisible by camera grid "
            f"{(height, width)}"
        )
    target_intr = select_wan_causal_endpoints(
        target_intr.float(),
        latent_frames=latent_frames,
        dim=1,
        readout_stride=1,
    )
    target_c2w = select_wan_causal_endpoints(
        target_c2w.float(),
        latent_frames=latent_frames,
        dim=1,
        readout_stride=1,
    )
    frames = latent_frames

    stride_y = image_height // height
    stride_x = image_width // width
    yy, xx = torch.meshgrid(
        (
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
            * float(stride_y)
            - 0.5
        )
        / float(image_height),
        (
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5)
            * float(stride_x)
            - 0.5
        )
        / float(image_width),
        indexing="ij",
    )
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(
        1,
        1,
        height * width,
        3,
    )
    intr = target_intr.reshape(batch * frames, 3, 3)
    rhs = (
        pixels.expand(batch, frames, -1, -1)
        .reshape(batch * frames, height * width, 3)
        .transpose(1, 2)
    )
    camera_rays, solve_info = torch.linalg.solve_ex(intr, rhs)
    if bool((solve_info != 0).any()):
        raise ValueError("target_intr contains a singular camera matrix")
    camera_rays = camera_rays.transpose(1, 2)
    poses = target_c2w.reshape(batch * frames, 4, 4)
    direction = camera_rays @ poses[:, :3, :3].transpose(1, 2)
    direction_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if not bool(torch.isfinite(direction_norm).all()) or bool(
        (direction_norm <= torch.finfo(torch.float32).eps).any()
    ):
        raise ValueError("target camera produced an invalid ray direction")
    direction = direction / direction_norm
    origin = poses[:, None, :3, 3].expand_as(direction)
    moment = torch.linalg.cross(origin, direction, dim=-1)
    plucker = (
        torch.cat((direction, moment), dim=-1)
        .reshape(batch, frames, height, width, 6)
        .permute(0, 1, 4, 2, 3)
        .contiguous()
    )
    if not bool(torch.isfinite(plucker).all()):
        raise ValueError("target camera produced non-finite Plücker rays")
    return plucker.permute(0, 2, 1, 3, 4).to(dtype=dtype).contiguous()
