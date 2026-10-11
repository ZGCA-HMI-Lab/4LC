from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

NEAREST_EXACT = "nearest_exact"
ROW_MAJOR = "row_major"


def _integer_pair(
    value: tuple[int, int],
    *,
    name: str,
    positive: bool,
) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TypeError(f"{name} must be a pair of integers")
    result = tuple(int(item) for item in value)
    lower = 1 if positive else 0
    if any(item < lower for item in result):
        operator = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} values must be {operator}")
    return result


@dataclass(frozen=True, slots=True)
class SpatialGridSpec:
    input_image_size: tuple[int, int]
    preprocessing: str
    grid_shape: tuple[int, int]
    stride: tuple[int, int]
    offset: tuple[int, int]
    anchor_sampling: str = NEAREST_EXACT
    token_layout: str = ROW_MAJOR

    def __post_init__(self) -> None:
        for name in ("input_image_size", "grid_shape", "stride"):
            object.__setattr__(
                self,
                name,
                _integer_pair(getattr(self, name), name=name, positive=True),
            )
        object.__setattr__(
            self,
            "offset",
            _integer_pair(self.offset, name="offset", positive=False),
        )
        if not isinstance(self.preprocessing, str) or not self.preprocessing:
            raise ValueError("preprocessing must be a non-empty string")
        if self.anchor_sampling != NEAREST_EXACT:
            raise ValueError(f"anchor_sampling must be {NEAREST_EXACT!r}")
        if self.token_layout != ROW_MAJOR:
            raise ValueError(f"token_layout must be {ROW_MAJOR!r}")
        image_h, image_w = self.input_image_size
        grid_h, grid_w = self.grid_shape
        stride_y, stride_x = self.stride
        offset_y, offset_x = self.offset
        last_y = offset_y + (grid_h - 1) * stride_y
        last_x = offset_x + (grid_w - 1) * stride_x
        if last_y >= image_h or last_x >= image_w:
            raise ValueError("SpatialGridSpec samples outside input_image_size")

    def sample(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim < 4:
            raise ValueError(f"{name} must be [B,V,H,W,...]")
        if tuple(int(size) for size in value.shape[2:4]) != self.input_image_size:
            raise ValueError(
                f"{name} image grid {tuple(value.shape[2:4])} does not match "
                f"SpatialGridSpec input {self.input_image_size}"
            )
        grid_h, grid_w = self.grid_shape
        stride_y, stride_x = self.stride
        offset_y, offset_x = self.offset
        stop_y = offset_y + grid_h * stride_y
        stop_x = offset_x + grid_w * stride_x
        return value[
            :,
            :,
            offset_y:stop_y:stride_y,
            offset_x:stop_x:stride_x,
        ].contiguous()

    def adjust_intrinsics(self, intr: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(intr, torch.Tensor)
            or intr.ndim < 2
            or tuple(int(size) for size in intr.shape[-2:]) != (3, 3)
        ):
            raise ValueError("intr must end with [3,3]")
        if not torch.is_floating_point(intr):
            raise TypeError("intr must be floating point")
        if not bool(torch.isfinite(intr).all()):
            raise ValueError("intr contains non-finite values")
        image_h, image_w = self.input_image_size
        grid_h, grid_w = self.grid_shape
        stride_y, stride_x = self.stride
        offset_y, offset_x = self.offset
        pixel = intr.float().clone()
        pixel[..., 0, :] *= float(image_w)
        pixel[..., 1, :] *= float(image_h)
        inverse_grid = torch.tensor(
            (
                (1.0 / stride_x, 0.0, -offset_x / stride_x),
                (0.0, 1.0 / stride_y, -offset_y / stride_y),
                (0.0, 0.0, 1.0),
            ),
            device=intr.device,
            dtype=torch.float32,
        )
        adjusted = inverse_grid @ pixel
        adjusted[..., 0, :] /= float(grid_w)
        adjusted[..., 1, :] /= float(grid_h)
        return adjusted.to(dtype=intr.dtype)

    def to_payload(self) -> dict[str, Any]:
        return {
            "input_image_size": list(self.input_image_size),
            "preprocessing": self.preprocessing,
            "grid_shape": list(self.grid_shape),
            "stride": list(self.stride),
            "offset": list(self.offset),
            "anchor_sampling": self.anchor_sampling,
            "token_layout": self.token_layout,
        }


__all__ = [
    "NEAREST_EXACT",
    "ROW_MAJOR",
    "SpatialGridSpec",
]


def unproject_depth(
    depth: torch.Tensor,
    intr: torch.Tensor,
    c2w: torch.Tensor,
) -> torch.Tensor:
    batch, views, height, width = (int(v) for v in depth.shape)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=torch.float32),
        torch.arange(width, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(
        1, 1, height * width, 3
    )
    intr_px = intr.float().clone()
    intr_px[..., 0, :] *= float(width)
    intr_px[..., 1, :] *= float(height)
    rays = torch.linalg.solve(
        intr_px.reshape(batch * views, 3, 3),
        pixels.expand(batch, views, -1, -1)
        .reshape(batch * views, height * width, 3)
        .transpose(1, 2),
    ).transpose(1, 2)
    camera_points = rays * depth.float().reshape(batch * views, height * width, 1)
    poses = c2w.float().reshape(batch * views, 4, 4)
    world = camera_points @ poses[:, :3, :3].transpose(1, 2) + poses[:, None, :3, 3]
    return world.reshape(batch, views, height, width, 3)
