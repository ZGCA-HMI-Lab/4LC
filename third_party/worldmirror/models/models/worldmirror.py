import hashlib
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_transformer import VisualGeometryTransformer
from ..heads.camera_head import CameraHead
from ..heads.dense_head import DPTHead
from .rasterization import GaussianSplatRenderer
from ..utils.camera_utils import vector_to_camera_matrices, extrinsics_to_vector
from ..utils.priors import normalize_depth, normalize_poses

from huggingface_hub import PyTorchModelHubMixin


def _convert_state_dict_keys_to_single_str(state_dict, with_shape: bool = True) -> str:
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + _convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    return ",".join(keys)


def hash_state_dict_keys(state_dict, with_shape: bool = True) -> str:
    keys_str = _convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    return hashlib.md5(keys_str.encode("UTF-8")).hexdigest()


class WorldMirror(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 img_size=518,
                 patch_size=14,
                 embed_dim=1024,
                 gs_dim=256,
                 enable_cond=True,
                 enable_cam=True,
                 enable_pts=True,
                 enable_depth=True,
                 enable_norm=True,
                 enable_motion=True,
                 enable_gs=True,
                 enable_dynamic_gs_attr=True,
                 life_span_gamma=10.0,
                 dynamic_threshold=0.0,
                 enable_global_motion_tracking=False,
                 dynamic_threshold2=0.0,
                 occlusion_threshold=0.05,
                 bidirection=True,
                 patch_embed="dinov2_vitl14_reg",
                 fixed_patch_embed=False,
                 sampling_strategy="uniform",
                 dpt_gradient_checkpoint=False,
                 condition_strategy=["token", "pow3r", "token"],
                 **kwargs):

        super().__init__()
        # Configuration flags
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.gs_dim = gs_dim
        self.enable_cam = enable_cam
        self.enable_pts = enable_pts
        self.enable_depth = enable_depth
        self.enable_cond = enable_cond
        self.enable_norm = enable_norm
        self.enable_motion = enable_motion
        self.enable_gs = enable_gs
        self.enable_dynamic_gs_attr = enable_dynamic_gs_attr
        self.life_span_gamma = life_span_gamma
        self.dynamic_threshold = dynamic_threshold
        self.enable_global_motion_tracking = enable_global_motion_tracking
        self.dynamic_threshold2 = dynamic_threshold2
        self.occlusion_threshold = occlusion_threshold
        self.bidirection = bidirection
        self.patch_embed = patch_embed
        self.sampling = sampling_strategy
        self.dpt_checkpoint = dpt_gradient_checkpoint
        self.cond_methods = condition_strategy
        self.config = self._store_config()

        # Visual geometry transformer
        self.visual_geometry_transformer = VisualGeometryTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_cond=enable_cond,
            enable_motion=enable_motion,
            sampling_strategy=sampling_strategy,
            patch_embed=patch_embed,
            fixed_patch_embed=fixed_patch_embed,
            condition_strategy=condition_strategy
        )

        # Initialize prediction heads
        self._init_heads(embed_dim, patch_size, gs_dim)

    def _store_config(self):
        """Save the model configuration"""
        return {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "gs_dim": self.gs_dim,
            "enable_cam": self.enable_cam,
            "enable_pts": self.enable_pts,
            "enable_depth": self.enable_depth,
            "enable_norm": self.enable_norm,
            "enable_gs": self.enable_gs,
            "patch_embed": self.patch_embed,
            "sampling_strategy": self.sampling,
            "dpt_checkpoint": self.dpt_checkpoint,
            "condition_strategy": self.cond_methods,
        }

    def _init_heads(self, dim, patch_size, gs_dim):
        """Initialize all prediction heads"""

        # Camera pose prediction head
        if self.enable_cam:
            self.cam_head = CameraHead(dim_in=2 * dim)

        # 3D point prediction head
        if self.enable_pts:
            self.pts_head = DPTHead(
                dim_in=2 * dim,
                output_dim=4,
                patch_size=patch_size,
                activation="inv_log+expp1"
            )

        # Depth prediction head
        if self.enable_depth:
            self.depth_head = DPTHead(
                dim_in=2 * dim,
                output_dim=2,
                patch_size=patch_size,
                activation="exp+expp1",
            )

        # Surface normal prediction head
        if self.enable_norm:
            self.norm_head = DPTHead(
                dim_in=2 * dim,
                output_dim=4,
                patch_size=patch_size,
                activation="norm+expp1",
            )

        # Velocity prediction heads
        if self.enable_motion:
            self.velocity_fwd_head = DPTHead(
                dim_in=dim,
                output_dim=4,
                patch_size=patch_size,
                activation="inv_log+expp1",
            )
            self.velocity_bwd_head = DPTHead(
                dim_in=dim,
                output_dim=4,
                patch_size=patch_size,
                activation="inv_log+expp1",
            )

        # Gaussian splatting feature head and renderer
        if self.enable_gs:
            self.gs_head = DPTHead(
                dim_in=2 * dim,
                output_dim=2,
                patch_size=patch_size,
                features=gs_dim,
                is_gsdpt=True,
                activation="exp+expp1"
            )
            self.gs_renderer = GaussianSplatRenderer(
                sh_degree=0,
                enable_prune=True,
                voxel_size=0.002,
                is_4dgs=self.enable_dynamic_gs_attr,
                life_span_gamma=self.life_span_gamma,
                dynamic_threshold=self.dynamic_threshold,
                global_motion_tracking=self.enable_global_motion_tracking,
                dynamic_threshold2=self.dynamic_threshold2,
                occlusion_threshold=self.occlusion_threshold,
                bidirection=self.bidirection,
            )
            # Dynamic Gaussian splatting attribute heads
            if self.enable_dynamic_gs_attr:
                self.gs_fwd_attr_head = DPTHead(
                    dim_in=dim,
                    output_dim=3,
                    patch_size=patch_size,
                    activation="rotation+none", # use 'none' to disable confidence prediction
                )
                self.gs_bwd_attr_head = DPTHead(
                    dim_in=dim,
                    output_dim=3,
                    patch_size=patch_size,
                    activation="rotation+none", # use 'none' to disable confidence prediction
                )

    def forward(
        self,
        views: Dict[str, torch.Tensor],
        cond_flags: List[int]=[0, 0, 0],
        is_inference=True,
        use_motion=True,
        return_token_list: bool = False,
        skip_unused_heads: bool = False,
        force_motion_tokens: bool = False,
        camera_only: bool = False,
    ):
        """
        Execute forward pass through the WorldMirror model.

        Args:
            views: Input data dictionary
            cond_flags: Conditioning flags [depth, rays, camera]

        Returns:
            dict: Prediction results dictionary
        """
        imgs = views['img']

        # Enable conditional input during training if enabled, or during inference if any cond_flags are set
        use_cond = sum(cond_flags) > 0
        if (imgs.shape[1] == 1):
            use_motion = False

        # Extract priors and process features based on conditional input
        if use_cond:
            priors = self.extract_priors(views)
            token_list, patch_start_idx, fwd_token_list, bwd_token_list = self.visual_geometry_transformer(
                imgs, priors, cond_flags=cond_flags, use_motion=(use_motion and (is_inference or force_motion_tokens))
            )
        else:
            token_list, patch_start_idx, fwd_token_list, bwd_token_list = self.visual_geometry_transformer(
                imgs,
                use_motion=(use_motion and (is_inference or force_motion_tokens)),
            )

        # Generate all predictions
        preds = self._gen_all_preds(
            token_list, imgs, patch_start_idx, views, cond_flags, is_inference, use_motion,
            fwd_token_list, bwd_token_list,
            return_token_list=return_token_list,
            skip_unused_heads=skip_unused_heads,
            camera_only=camera_only,
        )

        for key, value in preds.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
                preds[key] = value.to(torch.float32)
            elif isinstance(value, list) and key not in {"token_list", "context_token_list"}:
                for batch_value in value:
                    for frame_value in batch_value:
                        frame_value.to(torch.float32)
        return preds

    def _gen_all_preds(self, token_list, imgs, patch_start_idx,
                        views, cond_flags, is_inference, use_motion,
                       fwd_token_list=[], bwd_token_list=[], return_token_list: bool = False,
                       skip_unused_heads: bool = False,
                       camera_only: bool = False):
        """Generate all enabled predictions"""
        preds = {}

        # Camera pose prediction
        if self.enable_cam:
            cam_seq = self.cam_head(token_list)
            cam_params = cam_seq[-1]
            preds["camera_params"] = cam_params
            c2w_mat, int_mat = self.transform_camera_vector(cam_params, imgs.shape[-2], imgs.shape[-1])
            preds["camera_poses"] = c2w_mat  # C2W pose (OpenCV) in world coordinates: [B, S, 4, 4]
            preds["camera_intrs"] = int_mat  # Camera intrinsic matrix: [B, S, 3, 3]

        if camera_only:
            return preds

        # Depth prediction
        if self.enable_depth and not skip_unused_heads:
            depth, depth_conf = self.depth_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["depth"] = depth
            preds["depth_conf"] = depth_conf

        # 3D point prediction
        if self.enable_pts and not skip_unused_heads:
            pts, pts_conf = self.pts_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["pts3d"] = pts
            preds["pts3d_conf"] = pts_conf

        # Normal prediction
        if self.enable_norm and not skip_unused_heads:
            normals, norm_conf = self.norm_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["normals"] = normals
            preds["normals_conf"] = norm_conf

        # Prepare context predictions for motion and GS heads
        if self.enable_motion or self.enable_gs:
            context_preds = self.prepare_contexts(
                views,
                cond_flags,
                is_inference,
                use_motion,
                token_list=token_list,
                fwd_token_list=fwd_token_list,
                bwd_token_list=bwd_token_list,
            )
        else:
            context_preds = {}

        fwd_token_list = context_preds.get("fwd_token_list", fwd_token_list)
        bwd_token_list = context_preds.get("bwd_token_list", bwd_token_list)

        # Velocity prediction
        if self.enable_motion and use_motion:
            assert len(fwd_token_list) > 0 and len(bwd_token_list) > 0
            vel_fwd, vel_fwd_conf = self.velocity_fwd_head(
                fwd_token_list,
                images=context_preds.get("imgs", imgs)[:, :-1],
                patch_start_idx=patch_start_idx
            )
            vel_bwd, vel_bwd_conf = self.velocity_bwd_head(
                bwd_token_list,
                images=context_preds.get("imgs", imgs)[:, 1:],
                patch_start_idx=patch_start_idx
            )
            preds["velocity_fwd"] = vel_fwd
            preds["velocity_fwd_conf"] = vel_fwd_conf
            preds["velocity_bwd"] = vel_bwd
            preds["velocity_bwd_conf"] = vel_bwd_conf

        # 3D Gaussian Splatting
        if self.enable_gs:
            gs_feat, gs_depth, gs_depth_conf = self.gs_head(
                context_preds.get("token_list", token_list),
                images=context_preds.get("imgs", imgs),
                patch_start_idx=patch_start_idx
            )
            preds["gs_depth"] = gs_depth
            preds["gs_depth_conf"] = gs_depth_conf

            # Dynamic GS attributes
            if self.enable_dynamic_gs_attr and use_motion:
                assert len(fwd_token_list) > 0 and len(bwd_token_list) > 0
                gs_fwd_attr, _ = self.gs_fwd_attr_head(
                    fwd_token_list,
                    images=context_preds.get("imgs", imgs)[:, :-1],
                    patch_start_idx=patch_start_idx
                )
                gs_bwd_attr, _ = self.gs_bwd_attr_head(
                    bwd_token_list,
                    images=context_preds.get("imgs", imgs)[:, 1:],
                    patch_start_idx=patch_start_idx
                )
                preds["gs_fwd_attr"] = gs_fwd_attr
                preds["gs_bwd_attr"] = gs_bwd_attr

            preds = self.gs_renderer.render(
                gs_feats=gs_feat,
                images=imgs,
                predictions=preds,
                views=views,
                context_predictions=context_preds,
                is_inference=is_inference
            )
        if return_token_list:
            if "token_list" in context_preds:
                preds["context_token_list"] = context_preds["token_list"]
                preds["context_patch_start_idx"] = patch_start_idx
            else:
                preds["token_list"] = token_list
                preds["patch_start_idx"] = patch_start_idx
        return preds

    def extract_priors(self, views):
        """
        Extract and normalize geometric priors.

        Args:
            views: Input view data dictionary.

        Returns:
            tuple: (depths, rays, poses) Normalized priors.
        """
        h, w = views['img'].shape[-2:]

        # Initialize prior variables
        depths = rays = poses = None

        # Extract camera pose
        if 'camera_poses' in views:
            extrinsics = views['camera_poses'][:, :, :3]
            extrinsics = normalize_poses(extrinsics)
            cam_params = extrinsics_to_vector(extrinsics)
            poses = cam_params[:, :, :7]  # Shape: [B, S, 7]

        # Extract depth map
        if 'depthmap' in views:
            depth_h, depth_w = views['depthmap'].shape[-2:]
            depths = views['depthmap']
            if depth_h != h or depth_w != w:  # Check if depth dimensions match target resolution
                try:
                    depths = F.interpolate(depths, size=(h, w), mode='bilinear', align_corners=False)
                except:
                    import pdb; pdb.set_trace()
            depths = normalize_depth(depths)  # Shape: [B, S, H, W]

        # Extract ray directions
        if 'camera_intrs' in views:
            intrinsics = views['camera_intrs'][:, :, :3, :3]
            fx, fy = intrinsics[:, :, 0, 0] / w, intrinsics[:, :, 1, 1] / h
            cx, cy = intrinsics[:, :, 0, 2] / w, intrinsics[:, :, 1, 2] / h
            rays = torch.stack([fx, fy, cx, cy], dim=-1)  # Shape: [B, S, 4]

        return (depths, rays, poses)

    def transform_camera_vector(self, camera_params, h, w):
        ext_mat, int_mat = vector_to_camera_matrices(
            camera_params, image_hw=(h, w)
        )
        # Create homogeneous transformation matrix
        homo_row = torch.tensor([0, 0, 0, 1], device=ext_mat.device).view(1, 1, 1, 4)
        homo_row = homo_row.repeat(ext_mat.shape[0], ext_mat.shape[1], 1, 1)
        w2c_mat = torch.cat([ext_mat, homo_row], dim=2)
        c2w_mat = torch.linalg.inv(w2c_mat)
        return c2w_mat, int_mat

    def prepare_contexts(
        self,
        views,
        cond_flags,
        is_inference,
        use_motion,
        token_list=None,
        fwd_token_list=None,
        bwd_token_list=None,
    ):
        # Generate context views predictions
        context_preds = {}
        # only for training or evaluation
        if is_inference:
            return context_preds

        assert self.enable_cam and (self.enable_motion or self.enable_gs)
        if 'is_target' not in views:
            context_nums = views['img'].shape[1]
            context_indices = torch.arange(
                context_nums,
                device=views["img"].device,
                dtype=torch.long,
            ).unsqueeze(0).expand(views["img"].shape[0], -1)
        else:
            context_mask = ~views['is_target'].bool()
            context_counts = context_mask.sum(dim=1)
            if not bool(torch.all(context_counts == context_counts[0]).item()):
                raise ValueError("WorldMirror expects the same number of context views in every batch item.")
            context_nums = int(context_counts[0].item())
            context_indices = torch.stack(
                [torch.nonzero(mask, as_tuple=False).flatten() for mask in context_mask],
                dim=0,
            ).to(device=views["img"].device, dtype=torch.long)
        context_imgs = self._gather_sequence(views['img'], context_indices)

        use_cond = sum(cond_flags) > 0
        all_views_are_context = context_nums == views["img"].shape[1]
        has_motion_tokens = fwd_token_list is not None and bwd_token_list is not None and len(fwd_token_list) > 0 and len(bwd_token_list) > 0
        can_reuse_tokens = token_list is not None and all_views_are_context and (not (self.enable_motion and use_motion) or has_motion_tokens)

        # Extract context priors and process features based on context views
        if can_reuse_tokens:
            context_token_list = token_list
            context_fwd_token_list = fwd_token_list
            context_bwd_token_list = bwd_token_list
        elif use_cond:
            priors = self.extract_priors(views)
            context_priors = (self._gather_sequence(prior, context_indices) if prior is not None else None for prior in priors)
            context_token_list, _, context_fwd_token_list, context_bwd_token_list = self.visual_geometry_transformer(
                context_imgs, context_priors, cond_flags=cond_flags, use_motion=use_motion
            )
        else:
            context_token_list, _, context_fwd_token_list, context_bwd_token_list = self.visual_geometry_transformer(context_imgs, use_motion=use_motion)

        # Execute predictions
        # Context camera pose prediction
        context_cam_seq = self.cam_head(context_token_list)
        context_cam_params = context_cam_seq[-1]
        context_c2w_mat, context_int_mat = self.transform_camera_vector(context_cam_params, context_imgs.shape[-2], context_imgs.shape[-1])
        context_preds['camera_poses'] = context_c2w_mat  # C2W pose (OpenCV) in world coordinates: [B, S, 4, 4]
        context_preds['camera_intrs'] = context_int_mat  # Camera intrinsic matrix: [B, S, 3, 3]
        context_preds['token_list'] = context_token_list
        context_preds['imgs'] = context_imgs
        context_preds['fwd_token_list'] = context_fwd_token_list
        context_preds['bwd_token_list'] = context_bwd_token_list
        context_preds['view_indices'] = context_indices
        for key in ("timestamp", "is_static"):
            value = views.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == views["img"].shape[1]:
                context_preds[key] = self._gather_sequence(value, context_indices)

        return context_preds

    @staticmethod
    def _gather_sequence(tensor, indices):
        index = indices.to(device=tensor.device, dtype=torch.long)
        index = index.reshape(index.shape[0], index.shape[1], *([1] * (tensor.ndim - 2)))
        index = index.expand(-1, -1, *tensor.shape[2:])
        return torch.gather(tensor, dim=1, index=index)

    @staticmethod
    def state_dict_converter():
        return ModelDictConverter()


class ModelDictConverter:
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == '1a1d001a35f78f3a7796a1e719ead340':
            config = {
                "enable_norm": False,
                "strict_load": True,
                # "upcast_to_float32": True,
            }
        else:
            config = {}
        return state_dict, config
