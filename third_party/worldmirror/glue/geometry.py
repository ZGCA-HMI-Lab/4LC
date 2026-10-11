"""WorldMirror geometry helpers (local integration layer).

Migrated from ``legacy_geometry.geometry``. Self-contained: depends only on torch and
the vendored WorldMirror model. Rendering helpers are intentionally absent —
The local pipeline no longer renders novel views.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch

# Same value, different meanings: patch confidence-sum floor vs scene-scale floor.
EPS_CONF_SUM = 1.0e-6
MIN_TRANSLATION_SCALE = 1.0e-4
MIN_SCENE_SCALE = 1.0e-6
DEFAULT_FPS = 16.0
DENSE_HEAD_LAYER_INDICES = (4, 11, 17, 23)


def cfg_get(cfg: Any, section: str, key: str, default=None):
    values = cfg.get(section, {}) if hasattr(cfg, "get") else {}
    if values is None:
        values = {}
    return values.get(key, default) if hasattr(values, "get") else default


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def torch_dtype_from_cfg(cfg: Any) -> torch.dtype | None:
    runtime = cfg.get("runtime", {}) if hasattr(cfg, "get") else {}
    raw = runtime.get("dtype", None) if hasattr(runtime, "get") else None
    if raw in {None, "", "none", "null", "None"}:
        return None
    if isinstance(raw, torch.dtype):
        return raw
    return getattr(torch, str(raw))


def fps_from_cfg(cfg: Any) -> float:
    value = cfg_get(cfg, "data", "fps", None)
    if value in {None, "", "null", "None"}:
        value = DEFAULT_FPS
    return float(value)


def image_grid_from_cfg(cfg: Any, image_size: tuple[int, int]) -> tuple[int, int]:
    data = cfg.get("data", {}) if hasattr(cfg, "get") else {}
    raw = data.get("latent_grid", None) if hasattr(data, "get") else None
    if raw is not None:
        out = tuple(int(v) for v in raw)
        if len(out) != 2:
            raise ValueError(f"data.latent_grid must be [Hq,Wq], got {raw!r}")
        return out  # type: ignore[return-value]
    patch_size = patch_size_from_cfg(cfg)
    return int(image_size[0]) // patch_size, int(image_size[1]) // patch_size


def patch_size_from_cfg(cfg: Any) -> int:
    data = cfg.get("data", {}) if hasattr(cfg, "get") else {}
    value = data.get("patch_size", 14) if hasattr(data, "get") else 14
    return int(value)


def layer_indices_from_cfg(cfg: Any) -> tuple[int, ...]:
    token = cfg.get("token", {}) if hasattr(cfg, "get") else {}
    raw = token.get("layer_indices", None) if hasattr(token, "get") else None
    if raw is None:
        worldmirror = token.get("worldmirror", {}) if hasattr(token, "get") else {}
        raw = worldmirror.get("layer_indices", (4, 11, 17, 23)) if hasattr(worldmirror, "get") else (4, 11, 17, 23)
    return tuple(int(v) for v in raw)


def dense_compatible_layer_indices(layer_indices: Iterable[int] | None = None) -> tuple[int, ...]:
    requested = tuple(int(v) for v in (layer_indices or ()))
    out = list(DENSE_HEAD_LAYER_INDICES)
    for value in requested:
        if value not in out:
            out.append(value)
    return tuple(out)


def build_views(
    frames: torch.Tensor,
    *,
    fps: float,
    device: torch.device,
    dtype: torch.dtype | None,
    pose_c2w: torch.Tensor | None = None,
    intr_px: torch.Tensor | None = None,
    timestamps: torch.Tensor | None = None,
) -> dict[str, Any]:
    if frames.ndim != 4 or int(frames.shape[1]) != 3:
        raise ValueError(f"frames must be [T,3,H,W], got {tuple(frames.shape)}")
    model_dtype = dtype or torch.float32
    frames = frames.to(device=device, dtype=model_dtype, non_blocking=True).clamp(0.0, 1.0)
    count = int(frames.shape[0])
    if timestamps is None:
        view_timestamps = torch.arange(count, device=device, dtype=torch.float32) / float(fps)
    else:
        view_timestamps = timestamps.to(device=device, dtype=torch.float32)
        if view_timestamps.ndim != 1 or int(view_timestamps.numel()) != count:
            raise ValueError(f"timestamps must be [T] matching frames, got {tuple(view_timestamps.shape)} for T={count}")
        if not torch.isfinite(view_timestamps).all():
            raise ValueError("timestamps must be finite")
    if count <= 1:
        is_static = torch.ones((count,), device=device, dtype=torch.bool)
    else:
        is_static_value = bool(torch.allclose(view_timestamps, view_timestamps[:1].expand_as(view_timestamps)))
        is_static = torch.full((count,), is_static_value, device=device, dtype=torch.bool)
    views: dict[str, Any] = {
        "img": frames.unsqueeze(0),
        "timestamp": view_timestamps.unsqueeze(0),
        "is_static": is_static.unsqueeze(0),
    }
    if pose_c2w is not None:
        views["camera_poses"] = pose_c2w.to(device=device, dtype=torch.float32).unsqueeze(0)
    if intr_px is not None:
        views["camera_intrs"] = intr_px.to(device=device, dtype=torch.float32).unsqueeze(0)
    return views


def _as_unbatched(value: Any, *, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"prediction {key!r} must be a tensor")
    tensor = value.detach()
    if tensor.ndim >= 1 and int(tensor.shape[0]) == 1:
        tensor = tensor[0]
    return tensor.float().cpu()


def _homogeneous_pose(value: torch.Tensor, *, key: str) -> torch.Tensor:
    if value.ndim == 3 and tuple(value.shape[-2:]) == (4, 4):
        return value
    if value.ndim == 3 and tuple(value.shape[-2:]) == (3, 4):
        bottom = torch.zeros((*value.shape[:-2], 1, 4), dtype=value.dtype)
        bottom[..., 0, 3] = 1.0
        return torch.cat([value, bottom], dim=-2)
    raise ValueError(f"{key} must be [T,4,4] or [T,3,4], got {tuple(value.shape)}")


@torch.no_grad()
def extract_camera_predictions(
    predictions: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    pose_value = predictions.get("camera_poses")
    intr_value = predictions.get("camera_intrs")
    pose = _homogeneous_pose(_as_unbatched(pose_value, key="camera_poses"), key="camera_poses")
    intr = _as_unbatched(intr_value, key="camera_intrs")
    if intr.ndim != 3 or tuple(intr.shape[-2:]) != (3, 3):
        raise ValueError(f"camera_intrs must be [T,3,3], got {tuple(intr.shape)}")
    if pose.shape[0] != intr.shape[0]:
        raise ValueError("predicted pose and intrinsics frame counts differ")
    if not bool(torch.isfinite(pose).all()) or not bool(torch.isfinite(intr).all()):
        raise ValueError("predicted camera contains non-finite values")
    return pose, intr


@torch.no_grad()
def infer_worldmirror_camera(
    reconstructor: torch.nn.Module,
    views: dict[str, Any],
    *,
    dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = infer_reconstruction(
        reconstructor,
        views,
        dtype=dtype,
        use_motion=False,
        return_token_list=False,
        skip_unused_heads=True,
        force_motion_tokens=False,
        camera_only=True,
    )
    return extract_camera_predictions(predictions)


def _temporary_worldmirror_layers(reconstructor: torch.nn.Module, layer_indices: Iterable[int] | None):
    class Scope:
        def __enter__(self):
            self.vgt = getattr(reconstructor, "visual_geometry_transformer", None)
            self.old = None
            if self.vgt is not None and layer_indices is not None:
                if not hasattr(self.vgt, "intermediate_idxs"):
                    raise TypeError("WorldMirror visual_geometry_transformer must expose intermediate_idxs")
                self.old = list(self.vgt.intermediate_idxs)
                self.vgt.intermediate_idxs = [int(value) for value in layer_indices]

        def __exit__(self, exc_type, exc, tb):
            if self.old is not None:
                self.vgt.intermediate_idxs = self.old
            return False

    return Scope()


@torch.no_grad()
def infer_reconstruction(
    reconstructor: torch.nn.Module,
    views: dict[str, Any],
    *,
    dtype: torch.dtype | None,
    use_motion: bool = True,
    is_inference: bool = True,
    return_token_list: bool = False,
    skip_unused_heads: bool = False,
    force_motion_tokens: bool = False,
    camera_only: bool = False,
    layer_indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    device = views["img"].device
    with _temporary_worldmirror_layers(reconstructor, layer_indices):
        with autocast_context(device, dtype):
            out = reconstructor(
                views,
                is_inference=is_inference,
                use_motion=use_motion,
                return_token_list=return_token_list,
                skip_unused_heads=skip_unused_heads,
                force_motion_tokens=force_motion_tokens,
                camera_only=camera_only,
            )
    if not isinstance(out, dict):
        raise ValueError("reconstructor must return a mapping payload")
    return out


def _first_tensor(predictions: dict[str, Any], keys: Iterable[str]) -> tuple[str, torch.Tensor] | tuple[None, None]:
    for key in keys:
        value = predictions.get(key)
        if isinstance(value, torch.Tensor):
            return key, _as_unbatched(value, key=key)
    return None, None


def drop_batch(value: torch.Tensor, *, key: str) -> torch.Tensor:
    if value.ndim >= 1 and int(value.shape[0]) == 1:
        return value[0]
    raise ValueError(f"{key} must have batch size 1, got {tuple(value.shape)}")


def as_map(value: torch.Tensor, *, channels: int, key: str) -> torch.Tensor:
    if channels == 1 and value.ndim == 3:
        value = value.unsqueeze(-1)
    if value.ndim == 4 and value.shape[1] == channels and value.shape[-1] != channels:
        value = value.permute(0, 2, 3, 1)
    if value.ndim != 4 or int(value.shape[-1]) != int(channels):
        raise ValueError(f"{key} must be [T,H,W,{channels}], got {tuple(value.shape)}")
    return value.float()


def unproject_depth_px(depth: torch.Tensor, intr_px: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 4 and int(depth.shape[-1]) == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"depth must be [T,H,W] or [T,H,W,1], got {tuple(depth.shape)}")
    frames, height, width = (int(v) for v in depth.shape)
    if intr_px.shape[0] != frames or c2w.shape[0] != frames:
        raise ValueError("depth, intr_px, and c2w frame counts must match")
    intr_px = intr_px.to(device=depth.device, dtype=depth.dtype)
    c2w = c2w.to(device=depth.device, dtype=depth.dtype)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pix = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)
    points = []
    for idx in range(frames):
        rays = torch.linalg.solve(intr_px[idx], pix.T).T
        cam = rays * depth[idx].reshape(-1, 1)
        world = (cam @ c2w[idx, :3, :3].T) + c2w[idx, :3, 3]
        points.append(world.reshape(height, width, 3))
    return torch.stack(points, dim=0)


def extract_gs_depth_point_conf(predictions: dict[str, Any], *, pose_c2w: torch.Tensor, intr_px: torch.Tensor):
    depth_key, depth = _first_tensor(predictions, ("gs_depth", "depth"))
    conf_key, conf = _first_tensor(predictions, ("gs_depth_conf", "depth_conf", "pts3d_conf"))
    if depth is None or conf is None:
        raise ValueError("WorldMirror prediction must include gs_depth/depth and a confidence map")
    depth_map = as_map(depth, channels=1, key=str(depth_key))
    conf_map = as_map(conf, channels=1, key=str(conf_key))
    if depth_map.shape[0] != pose_c2w.shape[0] or conf_map.shape[0] != pose_c2w.shape[0]:
        raise ValueError("source GS depth frame count does not match camera frame count")
    point_map = unproject_depth_px(depth_map, intr_px=intr_px, c2w=pose_c2w)
    for name, value in (("depth", depth_map), ("point", point_map), ("conf", conf_map)):
        if not torch.isfinite(value).all():
            raise ValueError(f"source GS {name} contains non-finite values")
    if not bool((conf_map >= 0.0).all().item()):
        raise ValueError("source GS confidence contains negative values")
    return depth_map, point_map, conf_map


def _motion_map(value: torch.Tensor, *, frames: int, channels: int, key: str, direction: str | None = None):
    out = _as_unbatched(value, key=key)
    out = as_map(out, channels=channels, key=key)
    if int(out.shape[0]) == int(frames):
        return out
    if direction in {"fwd", "bwd"} and int(out.shape[0]) == int(frames) - 1:
        pad = torch.zeros_like(out[:1])
        return torch.cat([out, pad], dim=0) if direction == "fwd" else torch.cat([pad, out], dim=0)
    raise ValueError(f"{key} frame count must be {frames} or {frames - 1}, got {out.shape[0]}")


def extract_motion(predictions: dict[str, Any], *, frames: int) -> dict[str, torch.Tensor]:
    mapping = {
        "velocity_fwd": ("v_fwd", 3, "fwd"),
        "velocity_bwd": ("v_bwd", 3, "bwd"),
    }
    motion: dict[str, torch.Tensor] = {}
    for src, (dst, channels, direction) in mapping.items():
        value = predictions.get(src)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"WorldMirror prediction is missing required motion field {src!r}")
        motion[dst] = _motion_map(value, frames=frames, channels=channels, key=src, direction=direction)
    return motion


def _patch_view(value: torch.Tensor, target_grid: tuple[int, int], patch_size: int, *, key: str) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(f"{key} must be [V,H,W,C], got {tuple(value.shape)}")
    views, height, width, channels = value.shape
    out_h, out_w = (int(target_grid[0]), int(target_grid[1]))
    expected = (out_h * int(patch_size), out_w * int(patch_size))
    if (int(height), int(width)) != expected:
        raise ValueError(f"{key} spatial shape must be {expected}, got {(int(height), int(width))}")
    return (
        value.float()
        .reshape(int(views), out_h, int(patch_size), out_w, int(patch_size), int(channels))
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(int(views), out_h, out_w, int(patch_size) * int(patch_size), int(channels))
    )


def pool_to_latent_grid(point: torch.Tensor, conf: torch.Tensor, target_grid: tuple[int, int], patch_size: int):
    point_patch = _patch_view(point, target_grid, patch_size, key="point")
    conf_patch = _patch_view(conf, target_grid, patch_size, key="conf")
    conf_sum = conf_patch.sum(dim=3)
    normal = conf_sum > EPS_CONF_SUM
    if not bool(normal.all().item()):
        raise ValueError(f"latent pooling found {int((~normal).sum().item())} patches with confidence sum <= {EPS_CONF_SUM}")
    point_mean = (point_patch * conf_patch).sum(dim=3) / conf_sum
    conf_mean = conf_patch.mean(dim=3)
    return point_mean.contiguous(), conf_mean.contiguous()


def normalize_intrinsics_px(intr_px: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = (int(image_size[0]), int(image_size[1]))
    intr = intr_px.detach().float().clone()
    intr[..., 0, :] = intr[..., 0, :] / float(width)
    intr[..., 1, :] = intr[..., 1, :] / float(height)
    return intr


def denormalize_intrinsics(intr: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = (int(image_size[0]), int(image_size[1]))
    out = intr.detach().float().clone()
    out[..., 0, :] = out[..., 0, :] * float(width)
    out[..., 1, :] = out[..., 1, :] * float(height)
    return out


def normalize_pose_c2w(pose_c2w_raw: torch.Tensor, anchor_c2w_raw: torch.Tensor, scale: float) -> torch.Tensor:
    anchor_inv = torch.linalg.inv(anchor_c2w_raw.to(device=pose_c2w_raw.device, dtype=torch.float32))
    pose = anchor_inv[None] @ pose_c2w_raw.to(dtype=torch.float32)
    pose = pose.clone()
    pose[..., :3, 3] = pose[..., :3, 3] / float(scale)
    return pose.to(dtype=pose_c2w_raw.dtype)


def denormalize_pose_c2w(pose: torch.Tensor, anchor_c2w_raw: torch.Tensor, scale: float) -> torch.Tensor:
    local = pose.detach().float().clone()
    local[..., :3, 3] = local[..., :3, 3] * float(scale)
    anchor = anchor_c2w_raw.to(device=local.device, dtype=torch.float32)
    return anchor.unsqueeze(0) @ local


def camera_scale(source_pose_c2w_raw: torch.Tensor, anchor_c2w_raw: torch.Tensor) -> float:
    rel = normalize_pose_c2w(source_pose_c2w_raw, anchor_c2w_raw, 1.0)
    scale = rel[..., :3, 3].norm(dim=-1).amax().item()
    return max(float(scale), MIN_TRANSLATION_SCALE)


def scene_scale_from_depth(
    depth: torch.Tensor,
    *,
    intr_px: torch.Tensor,
    pose_c2w_raw: torch.Tensor,
    conf: torch.Tensor | None = None,
) -> float:
    depth_map = as_map(depth, channels=1, key="depth")
    point_raw = unproject_depth_px(depth_map, intr_px=intr_px, c2w=pose_c2w_raw)
    anchor_inv = torch.linalg.inv(pose_c2w_raw[0].to(device=point_raw.device, dtype=torch.float32))
    flat = point_raw.float().reshape(-1, 3)
    homo = torch.cat([flat, torch.ones((flat.shape[0], 1), device=flat.device, dtype=flat.dtype)], dim=-1)
    local = (homo @ anchor_inv.T)[..., :3].reshape_as(point_raw)
    dist = local.norm(dim=-1)
    valid = torch.isfinite(dist) & torch.isfinite(depth_map[..., 0]) & (depth_map[..., 0] > 0.0)
    if conf is not None:
        conf_map = as_map(conf, channels=1, key="depth_conf")
        if conf_map.shape != depth_map.shape:
            raise ValueError(f"depth_conf shape {tuple(conf_map.shape)} != depth {tuple(depth_map.shape)}")
        valid = valid & torch.isfinite(conf_map[..., 0]) & (conf_map[..., 0] > 0.0)
    if not bool(valid.any().item()):
        raise ValueError("scene scale requires at least one valid positive depth sample")
    scale = dist[valid].mean()
    if not bool(torch.isfinite(scale).item()) or float(scale.item()) <= 0.0:
        raise ValueError(f"scene scale must be positive and finite, got {float(scale.item())}")
    return max(float(scale.item()), MIN_SCENE_SCALE)


__all__ = [
    "DEFAULT_FPS",
    "DENSE_HEAD_LAYER_INDICES",
    "as_map",
    "build_views",
    "camera_scale",
    "cfg_get",
    "denormalize_intrinsics",
    "denormalize_pose_c2w",
    "dense_compatible_layer_indices",
    "drop_batch",
    "extract_camera_predictions",
    "extract_gs_depth_point_conf",
    "extract_motion",
    "fps_from_cfg",
    "image_grid_from_cfg",
    "infer_reconstruction",
    "infer_worldmirror_camera",
    "layer_indices_from_cfg",
    "normalize_intrinsics_px",
    "normalize_pose_c2w",
    "patch_size_from_cfg",
    "pool_to_latent_grid",
    "scene_scale_from_depth",
    "torch_dtype_from_cfg",
    "unproject_depth_px",
]
