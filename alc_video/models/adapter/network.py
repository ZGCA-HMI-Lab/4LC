from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from latent_cache import Readout

from alc_video.models.adapter.alignment import (
    aggregate_wan_masked_groups,
    readout_stride,
    resize_bcthw_grid,
    validate_feature_shapes,
    validate_readout_coverage,
)
from alc_video.models.adapter.camera import wan_camera_condition
from alc_video.models.video_model.temporal import (
    target_latent_shape,
    wan_readout_frame_count,
    wan_video_frame_count,
)


class ConditionProjection(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        normalize_input: bool,
    ) -> None:
        super().__init__()
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (in_dim, out_dim)
        ):
            raise TypeError("condition projection dimensions must be integers")
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("condition projection dimensions must be positive")
        if not isinstance(normalize_input, bool):
            raise TypeError("normalize_input must be bool")
        self.in_dim = in_dim
        self.out_dim = out_dim
        layers: list[nn.Module] = []
        if normalize_input:
            layers.append(nn.LayerNorm(in_dim))
        layers.extend((nn.Linear(in_dim, out_dim), nn.GELU(approximate="tanh")))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim != 5:
            raise ValueError("condition projection input must be [B,C,F,H,W]")
        if int(value.shape[1]) != self.in_dim:
            raise ValueError(
                f"condition projection expected {self.in_dim} channels, "
                f"got {int(value.shape[1])}"
            )
        if not torch.is_floating_point(value):
            raise TypeError("condition projection input must be floating point")
        parameter = next(self.parameters())
        if value.device != parameter.device or value.dtype != parameter.dtype:
            raise ValueError(
                "condition projection input must match its parameter device and dtype"
            )
        batch, _channels, frames, height, width = value.shape
        result = self.net(value.permute(0, 2, 3, 4, 1))
        return result.permute(0, 4, 1, 2, 3).reshape(
            batch,
            self.out_dim,
            frames,
            height,
            width,
        )


def _positive_dim(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


class ConditionAdapter(nn.Module):
    LATENT_CHANNELS = 16

    def __init__(
        self,
        *,
        feature_shapes: tuple[tuple[tuple[int, int], int], ...],
        feature_dims: tuple[int, ...],
        mask_dim: int,
        camera_dim: int,
        output_dim: int,
        use_motion_mask: bool,
    ) -> None:
        super().__init__()
        self.feature_shapes = validate_feature_shapes(feature_shapes)
        self.representation_count = len(self.feature_shapes)
        if not isinstance(use_motion_mask, bool):
            raise TypeError("use_motion_mask must be bool")
        self.use_motion_mask = use_motion_mask
        if (
            not isinstance(feature_dims, tuple)
            or len(feature_dims) != self.representation_count
        ):
            raise ValueError("feature_dims must match feature_shapes")
        self.feature_dims = tuple(
            _positive_dim(dim, name=f"feature_dims[{index}]")
            for index, dim in enumerate(feature_dims)
        )
        self.mask_dim = _positive_dim(mask_dim, name="mask_dim")
        self.camera_dim = _positive_dim(camera_dim, name="camera_dim")
        self.output_dim = _positive_dim(output_dim, name="output_dim")

        self.source_projectors = nn.ModuleList(
            ConditionProjection(
                shape[1],
                dim,
                normalize_input=False,
            )
            for shape, dim in zip(
                self.feature_shapes,
                self.feature_dims,
                strict=True,
            )
        )
        self.mask_projector = ConditionProjection(
            2 if self.use_motion_mask else 1,
            self.mask_dim,
            normalize_input=False,
        )
        self.camera_projector = ConditionProjection(
            6,
            self.camera_dim,
            normalize_input=False,
        )
        compressed_dim = sum((*self.feature_dims, self.mask_dim, self.camera_dim))
        self.fusion = nn.Conv3d(compressed_dim, self.output_dim, kernel_size=1)
        self.control_in_dim = self.output_dim

    def _validate_readouts(
        self,
        readouts: tuple[Readout, ...],
        *,
        latent_frames: int,
    ) -> tuple[int, ...]:
        if (
            not isinstance(readouts, tuple)
            or len(readouts) != self.representation_count
        ):
            raise ValueError("readouts must match the trained representation count")
        if any(not isinstance(value, Readout) for value in readouts):
            raise TypeError("readout values must be latent_cache.Readout")
        reference = readouts[0]
        nominal_frames = wan_video_frame_count(latent_frames)
        strides = []
        for index, value in enumerate(readouts):
            expected_grid, expected_channels = self.feature_shapes[index]
            if not isinstance(value.features, torch.Tensor) or value.features.ndim != 5:
                raise ValueError(f"readout {index} features must be [B,|J|,h,w,D]")
            grid = tuple(int(size) for size in value.features.shape[2:4])
            if (
                grid != expected_grid
                or int(value.features.shape[4]) != expected_channels
            ):
                raise ValueError(
                    f"readout {index} feature shape differs from the trained network"
                )
            try:
                stride = readout_stride(tuple(value.target_indices), nominal_frames)
            except ValueError as error:
                raise ValueError(
                    f"readout {index} target indices differ from its cadence"
                ) from error
            readout_frames = int(value.features.shape[1])
            expected_readout_frames = wan_readout_frame_count(latent_frames, stride)
            if readout_frames != expected_readout_frames:
                raise ValueError(
                    f"readout {index} has {readout_frames} frames; "
                    f"its cadence requires {expected_readout_frames}"
                )
            strides.append(stride)
        for value in readouts[1:]:
            if (
                value.features.device != reference.features.device
                or value.features.dtype != reference.features.dtype
            ):
                raise ValueError("readout features must share device and dtype")
        return tuple(strides)

    def _align_readout(
        self,
        index: int,
        readout: Readout,
        stride: int,
        target_latent_shape_value: torch.Size | tuple[int, ...] | list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid, channels = self.feature_shapes[index]
        _batch, _channels, latent_frames, latent_h, latent_w = (
            validate_readout_coverage(
                readout.features,
                readout.coverage,
                target_latent_shape_value,
                grid=grid,
                channels=channels,
                latent_channels=self.LATENT_CHANNELS,
                readout_stride=stride,
            )
        )
        dtype = readout.features.dtype
        covered = readout.coverage[:, None]
        interpolation_coverage = covered.float()
        features = readout.features.permute(0, 4, 1, 2, 3).float().contiguous()
        features = resize_bcthw_grid(
            features * interpolation_coverage,
            size=(latent_h, latent_w),
        )
        interpolation_coverage = resize_bcthw_grid(
            interpolation_coverage,
            size=(latent_h, latent_w),
        )
        features = features / interpolation_coverage.clamp_min(
            torch.finfo(interpolation_coverage.dtype).eps
        )
        aligned_frame_coverage = resize_bcthw_grid(
            covered.to(dtype=dtype),
            size=(latent_h, latent_w),
            mode="nearest",
        )
        features = (
            features * (aligned_frame_coverage > 0).to(dtype=features.dtype)
        ).to(dtype=dtype)
        return aggregate_wan_masked_groups(
            features,
            aligned_frame_coverage,
            latent_frames=latent_frames,
            readout_stride=stride,
        )

    def _align_moving(
        self,
        readout: Readout,
        stride: int,
        target_latent_shape_value: torch.Size | tuple[int, ...] | list[int],
    ) -> torch.Tensor:
        moving = readout.moving
        if not isinstance(moving, torch.Tensor):
            raise TypeError("readout moving mask must be a tensor")
        if tuple(moving.shape) != tuple(readout.coverage.shape):
            raise ValueError(
                f"moving mask must be {tuple(readout.coverage.shape)}, "
                f"got {tuple(moving.shape)}"
            )
        if moving.dtype != torch.bool:
            raise TypeError("moving mask must be bool")
        if moving.device != readout.coverage.device:
            raise ValueError("moving mask and coverage must share one device")
        if bool((moving & ~readout.coverage).any()):
            raise ValueError("moving mask must be a subset of coverage")
        _batch, _channels, latent_frames, latent_h, latent_w = target_latent_shape(
            target_latent_shape_value
        )
        aligned = resize_bcthw_grid(
            moving[:, None].to(dtype=readout.features.dtype),
            size=(latent_h, latent_w),
            mode="nearest",
        )
        _values, aligned_moving = aggregate_wan_masked_groups(
            aligned,
            aligned,
            latent_frames=latent_frames,
            readout_stride=stride,
        )
        return aligned_moving

    @staticmethod
    def _validate_coverage(
        coverage: torch.Tensor,
        *,
        name: str,
        upper: torch.Tensor | float = 1.0,
    ) -> None:
        if (
            not bool(torch.isfinite(coverage).all())
            or bool((coverage < 0).any())
            or bool((coverage > upper).any())
        ):
            raise RuntimeError(f"{name} coverage is outside its valid range")

    def forward(
        self,
        readouts: Mapping[str, Readout],
        target_intrinsics: torch.Tensor,
        target_c2w: torch.Tensor,
        target_latent_shape_value: torch.Size | tuple[int, ...] | list[int],
        *,
        image_size: tuple[int, int],
        feature_keep: tuple[bool, ...] | None = None,
    ) -> torch.Tensor:
        if not isinstance(readouts, Mapping):
            raise TypeError("readouts must map representation names to Readout")
        values = tuple(readouts.values())
        _batch, _channels, latent_frames, _height, _width = target_latent_shape(
            target_latent_shape_value
        )
        strides = self._validate_readouts(values, latent_frames=latent_frames)
        if feature_keep is None:
            feature_keep = (True,) * self.representation_count
        if (
            not isinstance(feature_keep, tuple)
            or len(feature_keep) != self.representation_count
            or any(not isinstance(keep, bool) for keep in feature_keep)
        ):
            raise ValueError("feature_keep must provide one bool per representation")

        aligned = tuple(
            self._align_readout(index, readout, stride, target_latent_shape_value)
            for index, (readout, stride) in enumerate(zip(values, strides, strict=True))
        )
        projected: list[torch.Tensor] = []
        for projector, (features, coverage), keep in zip(
            self.source_projectors,
            aligned,
            feature_keep,
            strict=True,
        ):
            value = projector(features)
            support = (coverage > 0).to(dtype=value.dtype)
            gate = value.new_tensor(float(keep))
            projected.append(value * support * gate)

        reference = values[0]
        kept_coverage = tuple(
            coverage
            for (_features, coverage), keep in zip(aligned, feature_keep, strict=True)
            if keep
        )
        coverage = (
            torch.stack(kept_coverage, dim=0).amax(dim=0)
            if kept_coverage
            else torch.zeros_like(aligned[0][1])
        )
        self._validate_coverage(coverage, name="aligned")
        mask = coverage
        if self.use_motion_mask:
            kept_moving = tuple(
                self._align_moving(readout, stride, target_latent_shape_value)
                for readout, stride, keep in zip(
                    values, strides, feature_keep, strict=True
                )
                if keep
            )
            moving = (
                torch.stack(kept_moving, dim=0).amax(dim=0)
                if kept_moving
                else torch.zeros_like(coverage)
            )
            self._validate_coverage(moving, name="aligned moving", upper=coverage)
            mask = torch.cat((coverage, moving), dim=1)
        mask_gate = mask.new_tensor(float(any(feature_keep)))
        mask_value = self.mask_projector(mask * mask_gate)
        projected.append(mask_value * mask_gate)

        camera = wan_camera_condition(
            target_intrinsics,
            target_c2w,
            target_latent_shape_value,
            image_size=image_size,
            device=reference.features.device,
            dtype=reference.features.dtype,
        )
        projected.append(self.camera_projector(camera))
        return self.fusion(torch.cat(projected, dim=1)).contiguous()
