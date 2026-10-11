from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from worldmirror.glue.reconstructor import CameraParams, WorldMirrorReconstructor

from alc_video.encoders.geometry.types import (
    WorldMirrorCameraFirstDepthState,
    WorldMirrorCameraSceneState,
    WorldMirrorCameraState,
    WorldMirrorPairState,
    WorldMirrorSourceState,
)
from alc_video.encoders.visual.base import (
    CONTEXT_MAX,
    CONTEXT_MIN,
)


def to_world_normalized_velocity(
    velocity: torch.Tensor,
    *,
    source_poses: torch.Tensor,
    source_times: torch.Tensor,
    world_scale: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    batch, views, _height, _width, _xyz = (int(value) for value in velocity.shape)
    if not torch.is_floating_point(velocity) or not bool(
        torch.isfinite(velocity).all()
    ):
        raise ValueError("velocity must be finite and floating point")
    if tuple(source_poses.shape) != (batch, views, 4, 4):
        raise ValueError(
            f"source_poses must be [B,V,4,4], got {tuple(source_poses.shape)}"
        )
    if tuple(source_times.shape) != (batch, views):
        raise ValueError(f"source_times must be [B,V], got {tuple(source_times.shape)}")
    intervals = source_times[:, 1:] - source_times[:, :-1]
    if not bool((intervals > 0.0).all()):
        raise ValueError("velocity conversion requires increasing source times")
    dt = torch.ones_like(source_times)
    available = torch.ones(
        (*velocity.shape[:-1], 1),
        device=velocity.device,
        dtype=torch.bool,
    )
    if direction == "fwd":
        dt[:, :-1] = intervals
        available[:, -1] = False
    elif direction == "bwd":
        dt[:, 1:] = intervals
        available[:, 0] = False
    else:
        raise ValueError(f"unknown velocity direction {direction!r}")
    scale = world_scale.to(device=velocity.device, dtype=velocity.dtype).reshape(-1)
    if int(scale.numel()) == 1:
        scale = scale.expand(batch)
    if (
        int(scale.numel()) != batch
        or not bool(torch.isfinite(scale).all())
        or not bool((scale > 0).all())
    ):
        raise ValueError("world_scale must contain B positive finite values")
    rotation = source_poses[:, :, :3, :3].to(
        device=velocity.device, dtype=velocity.dtype
    )
    rotated = torch.einsum("bvij,bvhwj->bvhwi", rotation, velocity)
    normalized = rotated / scale.reshape(batch, 1, 1, 1, 1)
    normalized = normalized / dt.to(velocity).reshape(batch, views, 1, 1, 1)
    return torch.where(
        available.expand_as(normalized),
        normalized,
        torch.zeros_like(normalized),
    )


class WorldMirrorGeometryEncoder:
    def __init__(
        self,
        reconstructor: torch.nn.Module,
        *,
        device: torch.device,
        dtype: torch.dtype | None,
        config: Any,
    ) -> None:
        self.reconstructor_module = reconstructor
        self.device = device
        self.dtype = dtype
        self.config = config
        self.reconstructor = WorldMirrorReconstructor(reconstructor, device)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        device: str | torch.device,
    ) -> WorldMirrorGeometryEncoder:

        plain = (
            OmegaConf.to_container(config, resolve=True)
            if OmegaConf.is_config(config)
            else config
        )
        if not isinstance(plain, dict):
            raise TypeError("WorldMirror config must be a mapping")
        raw_recon_cfg = plain.get("reconstructor")
        if not isinstance(raw_recon_cfg, dict):
            raise TypeError("WorldMirror reconstructor config must be a mapping")
        recon_cfg = dict(raw_recon_cfg)
        if str(recon_cfg.get("name", "worldmirror")).strip().lower() != "worldmirror":
            raise ValueError(
                "geometry reconstruction requires reconstructor.name=worldmirror"
            )
        if recon_cfg.get("strict_load") is not True:
            raise ValueError(
                "geometry reconstruction requires reconstructor.strict_load=true"
            )
        raw_path = recon_cfg.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip() or "://" in raw_path:
            raise ValueError(
                "reconstructor.path must be a non-empty local checkpoint path"
            )
        checkpoint = Path(raw_path).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[2] / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"reconstructor.path local checkpoint does not exist: {checkpoint}"
            )
        recon_cfg["path"] = str(checkpoint)

        import worldmirror.glue.geometry as geometry
        from worldmirror.glue.loader import load_reconstructor_module

        resolved = torch.device(device)
        return cls(
            load_reconstructor_module(recon_cfg, resolved),
            device=resolved,
            dtype=geometry.torch_dtype_from_cfg(config),
            config=config,
        )

    @staticmethod
    def _validate_camera_inputs(
        tracks: tuple[torch.Tensor, ...],
        frame_indices: torch.Tensor,
        times: torch.Tensor,
        fps: float,
    ) -> int:
        if not tracks:
            raise ValueError("WorldMirror scene must contain at least one track")
        count = int(tracks[0].shape[0])
        if count == 0:
            raise ValueError("WorldMirror tracks must contain at least one frame")
        for track_index, frames in enumerate(tracks):
            if frames.ndim != 4 or tuple(int(v) for v in frames.shape[1:]) != (
                3,
                336,
                560,
            ):
                raise ValueError(
                    f"WorldMirror track {track_index} frames must be [T,3,336,560], "
                    f"got {tuple(frames.shape)}"
                )
            if int(frames.shape[0]) != count:
                raise ValueError(
                    "WorldMirror synchronized tracks must have the same frame count"
                )
            if not torch.is_floating_point(frames):
                raise TypeError(
                    "WorldMirror track frames must be floating point RGB in [0,1]"
                )
            if (
                not bool(torch.isfinite(frames).all())
                or float(frames.min()) < 0.0
                or float(frames.max()) > 1.0
            ):
                raise ValueError("WorldMirror track frames must be finite RGB in [0,1]")
        if tuple(frame_indices.shape) != (count,) or frame_indices.dtype != torch.int64:
            raise ValueError(f"frame_indices must be int64 [{count}]")
        if int(frame_indices[0]) < 0 or not bool(
            (frame_indices[1:] > frame_indices[:-1]).all()
        ):
            raise ValueError(
                "frame_indices must be non-negative and strictly increasing"
            )
        if tuple(times.shape) != (count,) or times.dtype != torch.float32:
            raise ValueError(f"times must be float32 [{count}]")
        if not math.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError(f"fps must be > 0, got {fps}")
        expected_times = torch.arange(count, dtype=torch.float32) / float(fps)
        if not torch.allclose(times.cpu(), expected_times, rtol=0.0, atol=1.0e-6):
            raise ValueError("track times must use the declared FPS grid")
        return count

    @classmethod
    def _validate_inputs(
        cls,
        tracks: tuple[torch.Tensor, ...],
        frame_indices: torch.Tensor,
        times: torch.Tensor,
        source_indices: torch.Tensor,
        transport: str,
        fps: float,
    ) -> int:
        count = cls._validate_camera_inputs(tracks, frame_indices, times, fps)
        if source_indices.dtype != torch.int64 or source_indices.ndim != 1:
            raise ValueError("source_indices must be an int64 vector")
        source_count = int(source_indices.numel())
        if not CONTEXT_MIN <= source_count <= CONTEXT_MAX:
            raise ValueError(
                f"source_indices count must be in [{CONTEXT_MIN}, {CONTEXT_MAX}]"
            )
        if not bool((source_indices[1:] > source_indices[:-1]).all()):
            raise ValueError("source_indices must be strictly increasing")
        if int(source_indices[0]) < 0 or int(source_indices[-1]) >= count:
            raise ValueError("source_indices are outside the track")
        if transport not in {"identity", "velocity"}:
            raise ValueError(f"unknown transport: {transport!r}")
        return count

    def _infer_cameras_joint(
        self,
        tracks: tuple[torch.Tensor, ...],
        times: torch.Tensor,
        *,
        fps: float,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        import worldmirror.glue.geometry as geometry

        frames_per_track = int(tracks[0].shape[0])
        track_count = len(tracks)
        frames = torch.stack(tracks, dim=1).flatten(0, 1).contiguous()
        timestamps = times.repeat_interleave(track_count)
        views = geometry.build_views(
            frames,
            fps=fps,
            device=self.device,
            dtype=self.dtype,
            timestamps=timestamps,
        )
        try:
            poses, intrinsics = geometry.infer_worldmirror_camera(
                self.reconstructor_module,
                views,
                dtype=self.dtype,
            )
        finally:
            del views
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        expected = track_count * frames_per_track
        if int(poses.shape[0]) != expected or int(intrinsics.shape[0]) != expected:
            raise ValueError(
                "WorldMirror joint camera output length mismatch: "
                f"pose={tuple(poses.shape)}, intr={tuple(intrinsics.shape)}, "
                f"tracks={track_count}, frames={frames_per_track}"
            )
        poses = poses.reshape(frames_per_track, track_count, *poses.shape[1:])
        intrinsics = intrinsics.reshape(
            frames_per_track,
            track_count,
            *intrinsics.shape[1:],
        )
        return tuple(
            (
                poses[:, track_index],
                intrinsics[:, track_index],
            )
            for track_index in range(track_count)
        )

    @staticmethod
    def _normalize_camera_tracks(
        raw_tracks: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[tuple[WorldMirrorCameraState, ...], torch.Tensor, float]:
        import worldmirror.glue.geometry as geometry

        source_pose_raw = raw_tracks[0][0].float()
        reference_pose = source_pose_raw[0]
        scale = float(geometry.camera_scale(source_pose_raw, reference_pose))
        states = tuple(
            WorldMirrorCameraState(
                intr=geometry.normalize_intrinsics_px(intr_px, (336, 560))
                .float()
                .cpu()
                .contiguous(),
                c2w=geometry.normalize_pose_c2w(pose_raw.float(), reference_pose, scale)
                .cpu()
                .contiguous(),
            )
            for pose_raw, intr_px in raw_tracks
        )
        return states, reference_pose, scale

    @torch.no_grad()
    def infer_camera(
        self,
        frames: torch.Tensor,
        *,
        frame_indices: torch.Tensor,
        times: torch.Tensor,
        fps: float,
    ) -> WorldMirrorCameraState:
        self._validate_camera_inputs((frames,), frame_indices, times, fps)
        states, _reference_pose, _scale = self._normalize_camera_tracks(
            self._infer_cameras_joint((frames,), times, fps=fps)
        )
        return states[0]

    @torch.no_grad()
    def infer_camera_scene(
        self,
        frames: torch.Tensor,
        *,
        frame_indices: torch.Tensor,
        times: torch.Tensor,
        fps: float,
    ) -> WorldMirrorCameraSceneState:
        import worldmirror.glue.geometry as geometry

        self._validate_camera_inputs((frames,), frame_indices, times, fps)
        views = geometry.build_views(
            frames,
            fps=fps,
            device=self.device,
            dtype=self.dtype,
            timestamps=times,
        )
        try:
            predictions = geometry.infer_reconstruction(
                self.reconstructor_module,
                views,
                dtype=self.dtype,
                use_motion=False,
                return_token_list=False,
                skip_unused_heads=False,
                force_motion_tokens=False,
                camera_only=False,
            )
            pose_raw, intr_raw = geometry.extract_camera_predictions(predictions)
            depth, _points, confidence = geometry.extract_gs_depth_point_conf(
                predictions,
                pose_c2w=pose_raw,
                intr_px=intr_raw,
            )
            scene_scale = geometry.scene_scale_from_depth(
                depth,
                intr_px=intr_raw,
                pose_c2w_raw=pose_raw,
                conf=confidence,
            )
        finally:
            del views
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        reference_pose = pose_raw[0].float()
        camera_scale = float(geometry.camera_scale(pose_raw.float(), reference_pose))
        camera = WorldMirrorCameraState(
            intr=geometry.normalize_intrinsics_px(intr_raw, (336, 560))
            .float()
            .cpu()
            .contiguous(),
            c2w=geometry.normalize_pose_c2w(
                pose_raw.float(), reference_pose, camera_scale
            )
            .cpu()
            .contiguous(),
        )
        relative = torch.matmul(torch.linalg.inv(reference_pose), pose_raw.float())
        translation_drift = float(relative[:, :3, 3].norm(dim=-1).mean())
        if not math.isfinite(translation_drift):
            raise FloatingPointError("WorldMirror translation drift is non-finite")
        return WorldMirrorCameraSceneState(
            camera=camera,
            scene_scale=scene_scale,
            translation_drift=translation_drift,
        )

    @torch.no_grad()
    def infer_camera_with_first_depth(
        self,
        frames: torch.Tensor,
        *,
        frame_indices: torch.Tensor,
        times: torch.Tensor,
        fps: float,
    ) -> WorldMirrorCameraFirstDepthState:
        self._validate_camera_inputs((frames,), frame_indices, times, fps)
        camera_states, reference_pose, scale = self._normalize_camera_tracks(
            self._infer_cameras_joint((frames,), times, fps=fps)
        )
        camera = camera_states[0]
        first_camera = CameraParams(
            c2w=camera.c2w[:1],
            intr=camera.intr[:1],
            image_size=(336, 560),
            timestamps=times[:1].cpu().contiguous(),
            frame_indices=frame_indices[:1].cpu().contiguous(),
            anchor_c2w_raw=reference_pose.cpu().contiguous(),
            scale=scale,
        )
        prediction = self.reconstructor.predict_depth(
            frames[:1],
            first_camera,
            self.config,
        )
        depth = prediction["depth"]
        if tuple(depth.shape) == (1, 336, 560, 1):
            depth = depth[..., 0]
        if tuple(depth.shape) != (1, 336, 560):
            raise ValueError("WorldMirror F0 depth has an unexpected shape")
        valid = prediction["valid"]
        if tuple(valid.shape) == (1, 336, 560, 1):
            valid = valid[..., 0]
        if tuple(valid.shape) != (1, 336, 560) or valid.dtype != torch.bool:
            raise ValueError("WorldMirror F0 validity has an unexpected shape")
        depth = (depth[0].float() / scale).cpu().contiguous()
        valid = (valid[0].cpu() & torch.isfinite(depth) & (depth > 0.0)).contiguous()
        if not bool(valid.any()):
            raise ValueError("WorldMirror F0 depth contains no valid pixels")
        return WorldMirrorCameraFirstDepthState(
            camera=camera,
            depth=depth,
            valid=valid,
        )

    @torch.no_grad()
    def encode_pair(
        self,
        source_frames: torch.Tensor,
        target_frames: torch.Tensor | None,
        *,
        frame_indices: torch.Tensor,
        times: torch.Tensor,
        source_indices: torch.Tensor,
        transport: str,
        fps: float,
    ) -> WorldMirrorPairState:
        tracks = (
            (source_frames,)
            if target_frames is None
            else (source_frames, target_frames)
        )
        self._validate_inputs(
            tracks,
            frame_indices,
            times,
            source_indices,
            transport,
            fps,
        )
        camera_states, reference_pose, scale = self._normalize_camera_tracks(
            self._infer_cameras_joint(tracks, times, fps=fps)
        )
        source_camera = camera_states[0]
        target_camera = camera_states[-1]
        source_state = self._infer_source_state(
            frames=source_frames,
            frame_indices=frame_indices,
            times=times,
            source_indices=source_indices,
            c2w=source_camera.c2w,
            intr=source_camera.intr,
            reference_pose=reference_pose,
            scale=scale,
            transport=transport,
            fps=fps,
        )
        return WorldMirrorPairState(
            source_camera=source_camera,
            target_camera=target_camera,
            source=source_state,
        )

    def _infer_source_state(
        self,
        *,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        times: torch.Tensor,
        source_indices: torch.Tensor,
        c2w: torch.Tensor,
        intr: torch.Tensor,
        reference_pose: torch.Tensor,
        scale: float,
        transport: str,
        fps: float,
    ) -> WorldMirrorSourceState:
        import worldmirror.glue.geometry as geometry

        source_cpu = source_indices.cpu()
        source_count = int(source_cpu.numel())
        source_times = times.cpu().index_select(0, source_cpu).float().contiguous()
        camera = CameraParams(
            c2w=c2w.index_select(0, source_cpu),
            intr=intr.index_select(0, source_cpu),
            image_size=(336, 560),
            timestamps=source_times,
            frame_indices=frame_indices.cpu()
            .index_select(0, source_cpu)
            .long()
            .contiguous(),
            anchor_c2w_raw=reference_pose.cpu().contiguous(),
            scale=scale,
        )
        source_frames = frames.index_select(0, source_indices.to(frames.device))
        extract_motion = transport == "velocity"
        predictions, _views, raw_pose, raw_intr = self.reconstructor._infer(
            source_frames,
            camera,
            self.config,
            return_token_list=False,
            use_motion=extract_motion,
            force_motion_tokens=extract_motion,
            fps=fps,
        )

        depth_raw, _points, _conf = geometry.extract_gs_depth_point_conf(
            predictions,
            pose_c2w=raw_pose.cpu(),
            intr_px=raw_intr.cpu(),
        )
        depth = depth_raw[..., 0] if depth_raw.ndim == 4 else depth_raw
        depth = (depth.float() / scale).cpu().contiguous()
        valid = (torch.isfinite(depth) & (depth > 0.0)).contiguous()
        if not bool(valid.reshape(source_count, -1).any(dim=1).all()):
            raise ValueError(
                "WorldMirror produced a source view with no valid depth pixels"
            )
        motion = None
        if extract_motion:
            raw_motion = geometry.extract_motion(
                predictions,
                frames=source_count,
            )
            velocity_fwd = to_world_normalized_velocity(
                raw_motion["v_fwd"].unsqueeze(0).float(),
                source_poses=camera.c2w.unsqueeze(0),
                source_times=source_times.unsqueeze(0),
                world_scale=torch.tensor([scale], dtype=torch.float32),
                direction="fwd",
            )
            velocity_bwd = to_world_normalized_velocity(
                raw_motion["v_bwd"].unsqueeze(0).float(),
                source_poses=camera.c2w.unsqueeze(0),
                source_times=source_times.unsqueeze(0),
                world_scale=torch.tensor([scale], dtype=torch.float32),
                direction="bwd",
            )
            motion = {
                "v_fwd": velocity_fwd[0].to(dtype=torch.bfloat16).cpu().contiguous(),
                "v_bwd": velocity_bwd[0].to(dtype=torch.bfloat16).cpu().contiguous(),
            }
        return WorldMirrorSourceState(
            depth=depth.to(dtype=torch.bfloat16),
            valid=valid.unsqueeze(-1),
            motion=motion,
        )

    def offload(self) -> None:
        self.reconstructor_module.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
