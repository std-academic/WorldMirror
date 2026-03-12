from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.models.visual_transformer import VisualGeometryTransformer
from src.models.heads.camera_head import CameraHead
from src.models.heads.dense_head import DPTHead
from src.models.models.rasterization import GaussianSplatRenderer
from src.models.utils.camera_utils import vector_to_camera_matrices, extrinsics_to_vector
from src.models.utils.priors import normalize_depth, normalize_poses

from huggingface_hub import PyTorchModelHubMixin


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
                 enable_gs=True,
                 patch_embed="dinov2_vitl14_reg", 
                 fixed_patch_embed=False, 
                 sampling_strategy="uniform",
                 dpt_gradient_checkpoint=False, 
                 condition_strategy=["token", "pow3r", "token"]):

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
        self.enable_gs = enable_gs
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
            )

    def forward(self, views: Dict[str, torch.Tensor], cond_flags: List[int]=[0, 0, 0], is_inference=True):
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
        
        # Extract priors and process features based on conditional input
        if use_cond:
            priors = self.extract_priors(views)
            token_list, patch_start_idx = self.visual_geometry_transformer(
                imgs, priors, cond_flags=cond_flags
            )
        else:
            token_list, patch_start_idx = self.visual_geometry_transformer(imgs)
        
        # Execute predictions
        with torch.amp.autocast('cuda', dtype=torch.float32):
            # Generate all predictions
            preds = self._gen_all_preds(
                token_list, imgs, patch_start_idx, views, cond_flags, is_inference
            )

        return preds

    def _gen_all_preds(self, token_list, imgs, patch_start_idx, 
                       views, cond_flags, is_inference):
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
            
        # Depth prediction
        if self.enable_depth:
            depth, depth_conf = self.depth_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx, 
            )
            preds["depth"] = depth
            preds["depth_conf"] = depth_conf

        # 3D point prediction
        if self.enable_pts:
            pts, pts_conf = self.pts_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["pts3d"] = pts
            preds["pts3d_conf"] = pts_conf
        
        # Normal prediction
        if self.enable_norm:
            normals, norm_conf = self.norm_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["normals"] = normals
            preds["normals_conf"] = norm_conf
        
        # 3D Gaussian Splatting
        if self.enable_gs:
            context_preds = self.prepare_contexts(views, cond_flags, is_inference)
            gs_feat, gs_depth, gs_depth_conf = self.gs_head(
                context_preds.get("token_list", token_list), 
                images=context_preds.get("imgs", imgs), 
                patch_start_idx=patch_start_idx
            )

            preds["gs_depth"] = gs_depth
            preds["gs_depth_conf"] = gs_depth_conf
            preds = self.gs_renderer.render(
                gs_feats=gs_feat,
                images=imgs,
                predictions=preds,
                views=views,
                context_predictions=context_preds,
                is_inference=is_inference
            )
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
            
        # Extract ray directions (intrinsics summary)
        if 'camera_intrs' in views:
            intrinsics = views['camera_intrs'][:, :, :3, :3]
            fx, fy = intrinsics[:, :, 0, 0] / w, intrinsics[:, :, 1, 1] / h
            cx, cy = intrinsics[:, :, 0, 2] / w, intrinsics[:, :, 1, 2] / h
            rays = torch.stack([fx, fy, cx, cy], dim=-1)  # Shape: [B, S, 4]

        # Compute patch-wise spherical ray direction maps for cross-view spatial awareness
        # Each pixel gets (θ, φ) in world space; this helps the model understand
        # which patches across different cubemap faces correspond to the same 3D direction.
        spherical_rays = None
        if 'camera_intrs' in views and 'camera_poses' in views:
            spherical_rays = self._compute_spherical_ray_maps(
                views['camera_intrs'], views['camera_poses'], h, w
            )  # Shape: [B, S, 2, H, W]

        return (depths, rays, poses, spherical_rays)
    
    @staticmethod
    def _compute_spherical_ray_maps(camera_intrs, camera_poses, h, w):
        """
        Compute per-pixel spherical ray direction maps (θ, φ) in world space.

        For each pixel (u, v), we:
          1. Compute the camera-space ray direction via pinhole unprojection.
          2. Rotate to world space using the camera-to-world rotation.
          3. Convert to spherical coordinates (θ=longitude, φ=latitude).
          4. Normalize to [-1, 1] range for stable embedding.

        Args:
            camera_intrs: [B, S, 3, 3] intrinsic matrices.
            camera_poses: [B, S, 4, 4] camera-to-world extrinsic matrices.
            h, w: Image height and width in pixels.

        Returns:
            spherical_maps: [B, S, 2, H, W] where channel 0 = θ/π, channel 1 = φ/(π/2).
                            Both in [-1, 1].
        """
        B, S = camera_intrs.shape[:2]
        device = camera_intrs.device
        dtype = camera_intrs.dtype

        # Build pixel grid [H, W]
        v_coords, u_coords = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing='ij',
        )  # both [H, W]

        # Unproject to camera-space rays: [H, W, 3]
        # For each frame, use its own intrinsics
        fx = camera_intrs[:, :, 0, 0]  # [B, S]
        fy = camera_intrs[:, :, 1, 1]
        cx = camera_intrs[:, :, 0, 2]
        cy = camera_intrs[:, :, 1, 2]

        # Expand pixel grid for batch: [1, 1, H, W]
        u = u_coords.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        v = v_coords.unsqueeze(0).unsqueeze(0)

        # Camera-space ray directions: [B, S, H, W, 3]
        dirs_x = (u - cx[:, :, None, None]) / fx[:, :, None, None].clamp(min=1e-6)
        dirs_y = (v - cy[:, :, None, None]) / fy[:, :, None, None].clamp(min=1e-6)
        dirs_z = torch.ones(B, S, h, w, device=device, dtype=dtype)
        dirs_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # [B, S, H, W, 3]

        # Normalize to unit vectors
        dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Rotate to world space using C2W rotation
        R_c2w = camera_poses[:, :, :3, :3]  # [B, S, 3, 3]
        # dirs_cam: [B, S, H, W, 3] → [B, S, H*W, 3]
        dirs_flat = dirs_cam.reshape(B, S, h * w, 3)
        # Batched matmul: [B, S, H*W, 3] @ [B, S, 3, 3]^T → [B, S, H*W, 3]
        dirs_world = torch.einsum('bsnc,bsdc->bsnd', dirs_flat, R_c2w)
        dirs_world = dirs_world.reshape(B, S, h, w, 3)

        # World direction → spherical coordinates (OpenCV convention: Y-down)
        x_w, y_w, z_w = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]
        theta = torch.atan2(x_w, z_w)                              # longitude ∈ [-π, π]
        phi = torch.asin((-y_w).clamp(-1.0, 1.0))                  # latitude  ∈ [-π/2, π/2]

        # Normalize to [-1, 1]
        theta_norm = theta / torch.pi          # [-1, 1]
        phi_norm = phi / (torch.pi / 2.0)      # [-1, 1]

        # Stack as [B, S, 2, H, W]
        spherical_maps = torch.stack([theta_norm, phi_norm], dim=2)

        return spherical_maps

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
    
    def prepare_contexts(self, views, cond_flags, is_inference):
        # Generate context views predictions
        context_preds = {}
        # only for training or evaluation
        if is_inference:
            return context_preds
        
        assert self.enable_cam and self.enable_gs
        if 'is_target' not in views:
            context_nums = views['img'].shape[1]
        else:
            context_nums = (views['is_target'][0] == False).sum().item()
        context_imgs = views['img'][:, :context_nums]

        use_cond = sum(cond_flags) > 0
        
        # Extract context priors and process features based on context views
        with torch.amp.autocast('cuda', enabled=True):
            if use_cond:
                priors = self.extract_priors(views)
                context_priors = (prior[:, :context_nums] if prior is not None else None for prior in priors)
                context_token_list, _ = self.visual_geometry_transformer(
                    context_imgs, context_priors, cond_flags=cond_flags
                )
            else:
                context_token_list, _ = self.visual_geometry_transformer(context_imgs)

        # Execute predictions
        with torch.amp.autocast('cuda', enabled=False):
            # Context camera pose prediction
            context_cam_seq = self.cam_head(context_token_list)
            context_cam_params = context_cam_seq[-1]
            context_c2w_mat, context_int_mat = self.transform_camera_vector(context_cam_params, context_imgs.shape[-2], context_imgs.shape[-1])
            context_preds['camera_poses'] = context_c2w_mat  # C2W pose (OpenCV) in world coordinates: [B, S, 4, 4]
            context_preds['camera_intrs'] = context_int_mat  # Camera intrinsic matrix: [B, S, 3, 3]
            context_preds['token_list'] = context_token_list
            context_preds['imgs'] = context_imgs

        return context_preds

    
if __name__ == "__main__":
    device = "cuda"
    model = WorldMirror().to(device).eval()
    x = torch.rand(1, 1, 3, 518, 518).to(device)
    out = model({'img': x})
    import pdb; pdb.set_trace()
    