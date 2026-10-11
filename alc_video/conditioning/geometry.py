from __future__ import annotations

import torch
from latent_cache.geometry.spatial import unproject_depth
from latent_cache.propagation.motion import select_motion
from latent_cache.propagation.transport import VelocityTransportBackend


@torch.no_grad()
def estimate_context_geometry(
    worldmirror, rgb, intrinsics, c2w, frame_ids, fps, config
):
    from worldmirror.glue.reconstructor import CameraParams

    from alc_video.encoders.geometry.worldmirror import to_world_normalized_velocity

    count, height, width = rgb.shape[:3]
    camera = CameraParams(
        c2w=c2w.float().cpu().contiguous(),
        intr=intrinsics.float().cpu().contiguous(),
        image_size=(height, width),
        timestamps=frame_ids.float().cpu() / float(fps),
        frame_indices=frame_ids.cpu().contiguous(),
        anchor_c2w_raw=torch.eye(4, dtype=torch.float32),
        scale=1.0,
    )
    worldmirror.reconstructor_module.to(worldmirror.device)
    frames = rgb.permute(0, 3, 1, 2).contiguous()
    prediction = (
        worldmirror.reconstructor.predict_depth(frames, camera, worldmirror.config)
        if count == 1
        else worldmirror.reconstructor.predict_depth_and_velocity(
            frames, camera, worldmirror.config
        )
    )
    depth = prediction["depth"].reshape(count, height, width)
    valid = prediction["valid"].reshape(count, height, width)
    valid = valid & torch.isfinite(depth) & (depth > 0)
    if not bool(valid.reshape(count, -1).any(dim=1).all()):
        raise ValueError("WorldMirror produced a frame with no valid depth")
    depth = depth.float().to(rgb.device).contiguous()
    valid = valid.to(rgb.device).contiguous()
    moving = torch.zeros_like(valid)
    if count > 1:
        times = frame_ids.float().cpu().reshape(1, count) / float(fps)
        velocities = tuple(
            to_world_normalized_velocity(
                prediction[f"v_{direction}"].reshape(1, count, height, width, 3),
                source_poses=c2w.float().cpu().reshape(1, count, 4, 4),
                source_times=times,
                world_scale=torch.ones(1, dtype=torch.float32),
                direction=direction,
            ).to(rgb.device)
            for direction in ("fwd", "bwd")
        )
        intr, c2w = intrinsics.float()[None], c2w.float()[None]
        classified = select_motion(
            unproject_depth(depth[None], intr, c2w),
            valid[None],
            depth[None],
            times.to(rgb.device),
            torch.tensor([float(fps)], device=rgb.device),
            transport_backend=VelocityTransportBackend(*velocities),
            source_intr=intr,
            source_c2w=c2w,
            image_size=(height, width),
            threshold=config.motion_threshold,
            motion_mask_dilation_radius=config.motion_mask_dilation,
            temporal_motion_propagation=config.temporal_motion_propagation,
            temporal_motion_depth_tolerance=config.temporal_motion_depth_tolerance,
        )
        moving = classified.motion_mask[0] & valid
    worldmirror.offload()
    return depth, valid, moving
