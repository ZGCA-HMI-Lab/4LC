"""WorldMirror reconstructor wrapper (local integration, online encode + motion).

Migrated from ``legacy_geometry.backends.worldmirror.WorldMirrorBackend`` with all
rendering removed. WorldMirror now serves two roles for the local pipeline, both online:

* fixed motion estimator (``estimate_velocity``) — always available;
* optional visual encoder (``encode_source`` / ``encode_source_pixel_geometry``).

Camera information is passed as a plain :class:`CameraParams` (raw tensors +
normalization scalars) so this layer does not depend on datapipe types.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from worldmirror.glue import geometry
from worldmirror.glue.tokens import pack_worldmirror_token_list


@dataclass
class CameraParams:
    """Normalized per-frame camera + world normalization scalars.

    ``c2w`` / ``intr`` are anchor-normalized (``c2w`` scaled by ``scale``);
    ``anchor_c2w_raw`` + ``scale`` recover the raw metric camera.
    """

    c2w: torch.Tensor
    intr: torch.Tensor
    image_size: tuple[int, int]
    timestamps: torch.Tensor
    frame_indices: torch.Tensor
    anchor_c2w_raw: torch.Tensor
    scale: float

    def index_select(self, indices: torch.Tensor) -> "CameraParams":
        idx = indices if isinstance(indices, torch.Tensor) else torch.as_tensor(indices, dtype=torch.long)
        idx = idx.to(dtype=torch.long)
        return replace(
            self,
            c2w=self.c2w.index_select(0, idx.to(self.c2w.device)),
            intr=self.intr.index_select(0, idx.to(self.intr.device)),
            timestamps=self.timestamps.index_select(0, idx.to(self.timestamps.device)),
            frame_indices=self.frame_indices.index_select(0, idx.to(self.frame_indices.device)),
        )


def _normalization(camera: CameraParams) -> tuple[torch.Tensor, float]:
    anchor = camera.anchor_c2w_raw
    if not isinstance(anchor, torch.Tensor):
        raise ValueError("camera.anchor_c2w_raw must be a tensor")
    return anchor.float(), float(camera.scale)


def _normalize_points(points: torch.Tensor, *, anchor: torch.Tensor, scale: float) -> torch.Tensor:
    device = points.device
    dtype = points.dtype
    anchor_inv = torch.linalg.inv(anchor.to(device=device, dtype=torch.float32))
    flat = points.float().reshape(-1, 3)
    homo = torch.cat([flat, torch.ones((flat.shape[0], 1), device=device)], dim=-1)
    local = (homo @ anchor_inv.T)[..., :3] / float(scale)
    return local.reshape_as(points).to(dtype=dtype)


class WorldMirrorReconstructor:
    """Online WorldMirror encode + motion. Owns the loaded reconstructor module."""

    def __init__(self, reconstructor: torch.nn.Module, device: torch.device):
        self.reconstructor = reconstructor
        self.device = device

    @property
    def visual_geometry_transformer(self):
        return self.reconstructor.visual_geometry_transformer

    def parameters(self):
        return self.reconstructor.parameters()

    def eval(self):
        self.reconstructor.eval()
        return self

    def to(self, *args, **kwargs):
        self.reconstructor.to(*args, **kwargs)
        return self

    def _raw_camera(self, camera: CameraParams) -> tuple[torch.Tensor, torch.Tensor]:
        anchor, scale = _normalization(camera)
        raw_pose = geometry.denormalize_pose_c2w(
            camera.c2w.to(device=self.device, dtype=torch.float32),
            anchor.to(device=self.device, dtype=torch.float32),
            scale,
        )
        raw_intr = geometry.denormalize_intrinsics(camera.intr, camera.image_size).to(device=self.device, dtype=torch.float32)
        return raw_pose, raw_intr

    def _infer(
        self,
        frames: torch.Tensor,
        camera: CameraParams,
        cfg: Any,
        *,
        return_token_list: bool,
        use_motion: bool = True,
        force_motion_tokens: bool = True,
        fps: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]:
        raw_pose, raw_intr = self._raw_camera(camera)
        dtype = geometry.torch_dtype_from_cfg(cfg)
        views = geometry.build_views(
            frames,
            fps=geometry.fps_from_cfg(cfg) if fps is None else float(fps),
            device=self.device,
            dtype=dtype,
            pose_c2w=raw_pose,
            intr_px=raw_intr,
            timestamps=camera.timestamps,
        )
        requested_layers = geometry.layer_indices_from_cfg(cfg) if return_token_list else None
        inference_layers = geometry.dense_compatible_layer_indices(requested_layers) if return_token_list else None
        predictions = geometry.infer_reconstruction(
            self.reconstructor,
            views,
            dtype=dtype,
            use_motion=bool(use_motion),
            return_token_list=return_token_list,
            skip_unused_heads=True,
            force_motion_tokens=bool(force_motion_tokens),
            layer_indices=inference_layers,
        )
        if return_token_list:
            predictions["_inference_token_layers"] = inference_layers
        return predictions, views, raw_pose, raw_intr

    def _token_payload_from_predictions(self, predictions: dict[str, Any], views: dict[str, Any], cfg: Any) -> dict[str, Any]:
        token_list = predictions.get("context_token_list", predictions.get("token_list"))
        if not isinstance(token_list, list) or not token_list:
            raise ValueError("WorldMirror prediction must include token_list when return_token_list=true")
        requested_layers = geometry.layer_indices_from_cfg(cfg)
        available_layers = tuple(int(v) for v in predictions.get("_inference_token_layers", requested_layers))
        if len(token_list) != len(available_layers):
            raise ValueError(
                f"WorldMirror token_list length {len(token_list)} does not match layer list {available_layers}"
            )
        tokens_by_layer = dict(zip(available_layers, token_list))
        missing = [layer for layer in requested_layers if layer not in tokens_by_layer]
        if missing:
            raise ValueError(f"WorldMirror token_list missing requested layer(s) {missing}; available={available_layers}")
        selected_token_list = [tokens_by_layer[layer] for layer in requested_layers]
        patch_start_idx = predictions.get("context_patch_start_idx", predictions.get("patch_start_idx"))
        if patch_start_idx is None:
            raise ValueError("WorldMirror prediction must include patch_start_idx when return_token_list=true")
        _batch, _frames, _channels, height, width = (int(v) for v in views["img"].shape)
        return pack_worldmirror_token_list(
            selected_token_list,
            patch_start_idx=int(patch_start_idx),
            height=height,
            width=width,
            patch_size=geometry.patch_size_from_cfg(cfg),
            layer_indices=requested_layers,
        )

    def _source_pixel_geometry_from_predictions(
        self,
        *,
        predictions: dict[str, Any],
        raw_pose: torch.Tensor,
        raw_intr: torch.Tensor,
        camera: CameraParams,
    ) -> dict[str, torch.Tensor]:
        _depth, point_raw, conf = geometry.extract_gs_depth_point_conf(
            predictions,
            pose_c2w=raw_pose.cpu(),
            intr_px=raw_intr.cpu(),
        )
        anchor, scale = _normalization(camera)
        point_raw = point_raw.to(device=self.device)
        point_norm = _normalize_points(point_raw, anchor=anchor.to(device=self.device), scale=scale).cpu().float()
        conf_cpu = conf.float().cpu()
        if conf_cpu.ndim == 3:
            conf_cpu = conf_cpu.unsqueeze(-1)
        if conf_cpu.ndim != 4 or int(conf_cpu.shape[-1]) != 1:
            raise ValueError(f"WorldMirror pixel confidence must be [V,H,W,1], got {tuple(conf_cpu.shape)}")
        return {"point": point_norm.contiguous(), "conf": conf_cpu.contiguous()}

    def _encode(self, frames: torch.Tensor, camera: CameraParams, cfg: Any) -> dict[str, Any]:
        predictions, views, raw_pose, raw_intr = self._infer(frames, camera, cfg, return_token_list=True)
        token_payload = self._token_payload_from_predictions(predictions, views, cfg)
        source_pixel_geometry = self._source_pixel_geometry_from_predictions(
            predictions=predictions,
            raw_pose=raw_pose,
            raw_intr=raw_intr,
            camera=camera,
        )
        motion = geometry.extract_motion(predictions, frames=int(camera.frame_indices.shape[0]))
        return {
            "tokens": geometry.drop_batch(token_payload["tokens"].detach().cpu(), key="tokens").contiguous(),
            "pixel_point": source_pixel_geometry["point"],
            "pixel_conf": source_pixel_geometry["conf"],
            "motion": {key: value.detach().cpu().contiguous() for key, value in motion.items()},
        }

    # -- Public online API ---------------------------------------------------

    def encode_source(self, frames: torch.Tensor, camera: CameraParams, cfg: Any) -> dict[str, Any]:
        with torch.inference_mode():
            return self._encode(frames, camera, cfg)

    def encode_source_pixel_geometry(self, frames: torch.Tensor, camera: CameraParams, cfg: Any) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            predictions, _views, raw_pose, raw_intr = self._infer(
                frames,
                camera,
                cfg,
                return_token_list=False,
                use_motion=False,
                force_motion_tokens=False,
            )
            source_pixel_geometry = self._source_pixel_geometry_from_predictions(
                predictions=predictions,
                raw_pose=raw_pose,
                raw_intr=raw_intr,
                camera=camera,
            )
            return {
                "pixel_point": source_pixel_geometry["point"],
                "pixel_conf": source_pixel_geometry["conf"],
            }

    def estimate_velocity(self, frames: torch.Tensor, camera: CameraParams, cfg: Any) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            predictions, _views, _raw_pose, _raw_intr = self._infer(frames, camera, cfg, return_token_list=False)
            motion = geometry.extract_motion(predictions, frames=int(frames.shape[0]))
            return {
                "v_fwd": motion["v_fwd"].cpu().contiguous(),
                "v_bwd": motion["v_bwd"].cpu().contiguous(),
            }

    def predict_depth(self, frames: torch.Tensor, camera: CameraParams, cfg: Any) -> dict[str, torch.Tensor]:
        """Predict per-frame metric depth + valid mask for all window frames."""
        with torch.inference_mode():
            predictions, _views, raw_pose, raw_intr = self._infer(
                frames,
                camera,
                cfg,
                return_token_list=False,
                use_motion=False,
                force_motion_tokens=False,
            )
            depth_map, _point, conf = geometry.extract_gs_depth_point_conf(
                predictions, pose_c2w=raw_pose.cpu(), intr_px=raw_intr.cpu()
            )
            depth = depth_map.float().cpu().contiguous()
            conf_cpu = conf.float().cpu()
            if conf_cpu.ndim == 4 and int(conf_cpu.shape[-1]) == 1:
                conf_cpu = conf_cpu[..., 0]
            depth_hw = depth[..., 0] if depth.ndim == 4 else depth
            valid = (conf_cpu > 0.0) & torch.isfinite(depth_hw) & (depth_hw > 0.0)
            return {
                "depth": depth,
                "valid": valid.unsqueeze(-1).contiguous(),
            }

    def predict_depth_and_velocity(
        self,
        frames: torch.Tensor,
        camera: CameraParams,
        cfg: Any,
    ) -> dict[str, torch.Tensor]:
        """Predict depth, validity and bidirectional velocity in one forward."""
        with torch.inference_mode():
            predictions, _views, raw_pose, raw_intr = self._infer(
                frames,
                camera,
                cfg,
                return_token_list=False,
                use_motion=True,
                force_motion_tokens=True,
            )
            depth_map, _point, conf = geometry.extract_gs_depth_point_conf(
                predictions,
                pose_c2w=raw_pose.cpu(),
                intr_px=raw_intr.cpu(),
            )
            depth = depth_map.float().cpu().contiguous()
            depth_hw = depth[..., 0] if depth.ndim == 4 else depth
            confidence = conf.float().cpu()
            if confidence.ndim == 4 and int(confidence.shape[-1]) == 1:
                confidence = confidence[..., 0]
            valid = (
                (confidence > 0.0)
                & torch.isfinite(depth_hw)
                & (depth_hw > 0.0)
            )
            motion = geometry.extract_motion(predictions, frames=int(frames.shape[0]))
            return {
                "depth": depth,
                "valid": valid.unsqueeze(-1).contiguous(),
                "v_fwd": motion["v_fwd"].float().cpu().contiguous(),
                "v_bwd": motion["v_bwd"].float().cpu().contiguous(),
            }

__all__ = ["CameraParams", "WorldMirrorReconstructor"]
