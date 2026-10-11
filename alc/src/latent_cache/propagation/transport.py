from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class TransportBackendResult:
    points: torch.Tensor
    valid: torch.Tensor


class TransportBackend(Protocol):
    def __call__(
        self,
        source_anchor: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
    ) -> TransportBackendResult: ...


def _validate_backend_inputs(
    source_anchor: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor,
) -> tuple[int, int, int]:
    if source_anchor.ndim != 5 or int(source_anchor.shape[-1]) != 3:
        raise ValueError(
            f"source_anchor must be [B,V,H,W,3], got {tuple(source_anchor.shape)}"
        )
    batch, views = (int(value) for value in source_anchor.shape[:2])
    if tuple(source_times.shape) != (batch, views):
        raise ValueError(
            f"source_times must be [B,V]={batch, views}, got {tuple(source_times.shape)}"
        )
    if target_times.ndim != 3 or tuple(target_times.shape[:2]) != (batch, views):
        raise ValueError(
            "transport backend target_times must be [B,V,T], got "
            f"{tuple(target_times.shape)}"
        )
    queries = int(target_times.shape[2])
    if queries <= 0:
        raise ValueError("transport backend requires at least one target time")
    values = (source_anchor, source_times, target_times)
    if not all(torch.is_floating_point(value) for value in values):
        raise TypeError("transport backend inputs must be floating point")
    if any(value.dtype == torch.float16 for value in values):
        raise TypeError("transport backend inputs must use BF16 or FP32, not FP16")
    if not all(value.device == source_anchor.device for value in values):
        raise ValueError("transport backend inputs must share one device")
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("transport backend input contains non-finite values")
    return batch, views, queries


@dataclass(frozen=True)
class IdentityTransportBackend:
    def __call__(
        self,
        source_anchor: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
    ) -> TransportBackendResult:
        batch, views, queries = _validate_backend_inputs(
            source_anchor,
            source_times,
            target_times,
        )
        height, width = (int(value) for value in source_anchor.shape[2:4])
        return TransportBackendResult(
            points=source_anchor[:, :, None].expand(
                batch,
                views,
                queries,
                height,
                width,
                3,
            ),
            valid=torch.ones(
                (batch, views, queries, height, width),
                device=source_anchor.device,
                dtype=torch.bool,
            ),
        )


@dataclass(frozen=True)
class VelocityTransportBackend:
    velocity_fwd: torch.Tensor
    velocity_bwd: torch.Tensor

    def __post_init__(self) -> None:
        if self.velocity_fwd.ndim != 5 or int(self.velocity_fwd.shape[-1]) != 3:
            raise ValueError("velocity_fwd must be [B,V,H,W,3]")
        if self.velocity_bwd.shape != self.velocity_fwd.shape:
            raise ValueError("velocity_bwd must match velocity_fwd")
        if self.velocity_bwd.dtype != self.velocity_fwd.dtype:
            raise TypeError("transport velocities must share one dtype")
        if not torch.is_floating_point(
            self.velocity_fwd
        ) or not torch.is_floating_point(self.velocity_bwd):
            raise TypeError("transport velocities must be floating point")
        if self.velocity_fwd.dtype == torch.float16:
            raise TypeError("transport velocities must use BF16 or FP32, not FP16")
        if self.velocity_bwd.device != self.velocity_fwd.device:
            raise ValueError("transport velocities must share one device")
        if not bool(torch.isfinite(self.velocity_fwd).all()) or not bool(
            torch.isfinite(self.velocity_bwd).all()
        ):
            raise ValueError("transport velocities contain non-finite values")

    def __call__(
        self,
        source_anchor: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
    ) -> TransportBackendResult:
        batch, views, queries = _validate_backend_inputs(
            source_anchor,
            source_times,
            target_times,
        )
        if self.velocity_fwd.shape != source_anchor.shape:
            raise ValueError("transport velocities must match source_anchor")
        if self.velocity_fwd.device != source_anchor.device:
            raise ValueError(
                "transport velocities and source points must share one device"
            )

        signed_dt = target_times - source_times[:, :, None]
        expanded_dt = signed_dt[..., None, None, None]
        velocity = torch.where(
            expanded_dt >= 0,
            self.velocity_fwd[:, :, None],
            self.velocity_bwd[:, :, None],
        ).to(dtype=source_anchor.dtype)
        points = source_anchor[:, :, None] + velocity * expanded_dt.abs().to(
            dtype=source_anchor.dtype
        )

        view_index = torch.arange(views, device=source_anchor.device).reshape(
            1, views, 1
        )
        available = torch.where(
            signed_dt == 0,
            torch.ones_like(signed_dt, dtype=torch.bool),
            torch.where(
                signed_dt > 0,
                view_index < views - 1,
                view_index > 0,
            ),
        )
        height, width = (int(value) for value in source_anchor.shape[2:4])
        return TransportBackendResult(
            points=points,
            valid=available[:, :, :, None, None].expand(
                batch,
                views,
                queries,
                height,
                width,
            ),
        )


def query_transport(
    backend: TransportBackend,
    source_anchor: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor,
) -> TransportBackendResult:
    _validate_backend_inputs(source_anchor, source_times, target_times)
    if not callable(backend):
        raise TypeError("transport_backend must be callable")
    result = backend(source_anchor, source_times, target_times)
    if not isinstance(result, TransportBackendResult):
        raise TypeError("transport_backend must return TransportBackendResult")
    expected_points = (
        *source_anchor.shape[:2],
        int(target_times.shape[2]),
        *source_anchor.shape[2:],
    )
    expected_valid = expected_points[:-1]
    if tuple(result.points.shape) != tuple(expected_points):
        raise ValueError(
            "transport backend points must be [B,V,T,H,W,3], got "
            f"{tuple(result.points.shape)}"
        )
    if tuple(result.valid.shape) != tuple(expected_valid):
        raise ValueError(
            "transport backend valid must be [B,V,T,H,W], got "
            f"{tuple(result.valid.shape)}"
        )
    if not torch.is_floating_point(result.points):
        raise TypeError("transport backend points must be floating point")
    if result.points.dtype == torch.float16:
        raise TypeError("transport backend points must use BF16 or FP32, not FP16")
    if result.valid.dtype != torch.bool:
        raise TypeError("transport backend valid must be bool")
    if (
        result.points.device != source_anchor.device
        or result.valid.device != source_anchor.device
    ):
        raise ValueError("transport backend output must share the source-point device")
    if not bool(torch.isfinite(result.points).all()):
        raise ValueError("transport backend points contain non-finite values")
    return result


__all__ = [
    "IdentityTransportBackend",
    "TransportBackend",
    "TransportBackendResult",
    "VelocityTransportBackend",
    "query_transport",
]
