"""
ERP (Equirectangular Projection) support for HunyuanWorld-Mirror.

Converts 360° panoramic images (ERP format) into perspective cubemap views
that can be processed by the existing pinhole-based WorldMirror pipeline,
and provides utilities to render ERP output from reconstructed 3D Gaussians.

Pipeline (single ERP):
    1. Detect ERP input:     is_erp_image("panorama.jpg")
    2. Convert to cubemap:   views = erp_to_cubemap(erp_img, face_size=518)
    3. Prepare model input:  imgs, c2w, K = cubemap_views_to_model_input(views, device)
    4. Run WorldMirror:      preds = model(
                                 views={'img': imgs, 'camera_poses': c2w, 'camera_intrs': K},
                                 cond_flags=[1, 0, 1],
                             )
    5. Render ERP output:    erp = render_erp_from_splats(renderer, preds['splats'])

Pipeline (multiple ERPs):
    1. Detect ERP inputs:    all(is_erp_image(p) for p in paths)
    2. Convert each to cube: views, groups = multi_erp_to_cubemap(erp_images)
    3. Prepare model input:  imgs, c2w, K = cubemap_views_to_model_input(views)
       - Rotations are known within each ERP group; translations are unknown across groups.
       - For multi-ERP: only condition on intrinsics (cond_flags=[0, 0, 1]).
"""

import numpy as np
import torch
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
from torchvision import transforms


# ============================= Rotation Helpers =============================

def _rotation_x(angle: float) -> np.ndarray:
    """Rotation matrix around X axis (OpenCV convention)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    """Rotation matrix around Y axis (OpenCV convention)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


# Cubemap face definitions: face name → Camera-to-World rotation matrix (R_c2w)
# Convention: OpenCV camera coords (X-right, Y-down, Z-forward into scene)
CUBEMAP_FACES = {
    'front': np.eye(3, dtype=np.float64),           # Camera looks along +Z
    'right': _rotation_y(np.pi / 2),                # Camera looks along +X
    'back':  _rotation_y(np.pi),                     # Camera looks along -Z
    'left':  _rotation_y(-np.pi / 2),               # Camera looks along -X
    'up':    _rotation_x(np.pi / 2),                # Camera looks along -Y (world up)
    'down':  _rotation_x(-np.pi / 2),               # Camera looks along +Y (world down)
}

# Horizontal faces in circular order (for smooth video trajectories)
HORIZONTAL_FACE_ORDER = ['front', 'right', 'back', 'left']


# ============================= ERP Detection =============================

def is_erp_image(image_path: str, tolerance: float = 0.15) -> bool:
    """
    Heuristically detect whether an image is in ERP format by checking
    for an approximately 2:1 width-to-height aspect ratio.

    Args:
        image_path: Path to the image file.
        tolerance: Maximum deviation from 2.0 aspect ratio to be considered ERP.

    Returns:
        True if the image appears to be in ERP format.
    """
    img = Image.open(image_path)
    w, h = img.size
    return abs(w / max(h, 1) - 2.0) < tolerance


# ==================== ERP → Perspective Conversion ====================

def _sample_perspective_from_erp(
    erp_image: np.ndarray,
    R_c2w: np.ndarray,
    focal_length: float,
    out_size: int,
) -> np.ndarray:
    """
    Sample a single perspective view from an ERP image via bilinear interpolation.

    For each output pixel (u, v):
      1. Compute unit ray direction in camera space (pinhole model).
      2. Rotate to world space using R_c2w.
      3. Convert to ERP (longitude θ, latitude φ) coordinates.
      4. Sample from the ERP image using cv2.remap.

    Args:
        erp_image: Source ERP image [H_erp, W_erp, 3], uint8.
        R_c2w: Camera-to-world rotation matrix [3, 3].
        focal_length: Focal length in pixels.
        out_size: Square output image side length.

    Returns:
        Perspective image [out_size, out_size, 3], same dtype as input.
    """
    H_erp, W_erp = erp_image.shape[:2]
    cx = cy = out_size / 2.0

    # Pixel grid for perspective image
    u = np.arange(out_size, dtype=np.float32)
    v = np.arange(out_size, dtype=np.float32)
    u, v = np.meshgrid(u, v)  # [out_size, out_size]

    # Camera-space unit ray directions (pinhole model)
    dirs = np.stack([
        (u - cx) / focal_length,
        (v - cy) / focal_length,
        np.ones_like(u),
    ], axis=-1)  # [H, W, 3]
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # Rotate to world space
    dirs_world = dirs @ R_c2w.T  # [H, W, 3]
    x, y, z = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]

    # World direction → spherical coordinates
    # OpenCV convention: Y-down, so negate Y for standard latitude
    theta = np.arctan2(x, z)                          # longitude ∈ [-π, π]
    phi = np.arcsin(np.clip(-y, -1.0, 1.0))           # latitude  ∈ [-π/2, π/2]

    # Spherical → ERP pixel coordinates
    map_x = ((theta / np.pi + 1.0) / 2.0 * W_erp).astype(np.float32)
    map_y = ((0.5 - phi / np.pi) * H_erp).astype(np.float32)

    # Longitude wraps horizontally; latitude clamps at poles
    map_x = map_x % W_erp
    map_y = np.clip(map_y, 0, H_erp - 1)

    return cv2.remap(
        erp_image, map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
    )


def erp_to_cubemap(
    erp_image: np.ndarray,
    face_size: int = 518,
) -> List[Dict]:
    """
    Convert an ERP panoramic image to 6 cubemap perspective views (90° FOV each).

    Each view is a standard pinhole image suitable for WorldMirror processing.
    Together, the 6 faces cover the full 360° × 180° sphere with no gaps.

    Args:
        erp_image: ERP panorama [H, W, 3], uint8 (expects H:W ≈ 1:2).
        face_size: Output face size in pixels (should be divisible by 14 for ViT patches).

    Returns:
        List of 6 dicts, each containing:
            'name':  str           – face identifier ('front', 'right', etc.)
            'image': np.ndarray    – [face_size, face_size, 3] uint8
            'R_c2w': np.ndarray    – [3, 3] camera-to-world rotation
            'fov':   float         – field of view in radians (π/2)
    """
    fov = np.pi / 2.0
    f = face_size / (2.0 * np.tan(fov / 2.0))  # = face_size / 2

    views = []
    for name, R_c2w in CUBEMAP_FACES.items():
        face_img = _sample_perspective_from_erp(erp_image, R_c2w, f, face_size)
        views.append({
            'name': name,
            'image': face_img,
            'R_c2w': R_c2w.copy(),
            'fov': fov,
        })
    return views


def erp_to_perspective(
    erp_image: np.ndarray,
    face_size: int = 518,
    fov_deg: float = 90.0,
    yaw_steps: int = 8,
    pitch_angles_deg: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Convert an ERP image to multiple overlapping perspective views at custom orientations.

    Args:
        erp_image: ERP panorama [H, W, 3], uint8.
        face_size: Square output size.
        fov_deg: Horizontal FOV per view in degrees.
        yaw_steps: Number of equally-spaced yaw divisions around the horizon.
        pitch_angles_deg: List of pitch angles in degrees. Default: [0].

    Returns:
        List of dicts (same format as erp_to_cubemap output).
    """
    fov = np.radians(fov_deg)
    f = face_size / (2.0 * np.tan(fov / 2.0))
    if pitch_angles_deg is None:
        pitch_angles_deg = [0.0]

    views = []
    for yi in range(yaw_steps):
        yaw = yi * 2 * np.pi / yaw_steps
        for pitch_deg in pitch_angles_deg:
            pitch = np.radians(pitch_deg)
            R_c2w = _rotation_y(yaw) @ _rotation_x(pitch)
            face_img = _sample_perspective_from_erp(erp_image, R_c2w, f, face_size)
            views.append({
                'name': f'yaw{np.degrees(yaw):.0f}_pitch{pitch_deg:.0f}',
                'image': face_img,
                'R_c2w': R_c2w.copy(),
                'fov': fov,
            })
    return views


def multi_erp_to_cubemap(
    erp_images: List[np.ndarray],
    face_size: int = 518,
    erp_names: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[int]]:
    """
    Convert multiple ERP panoramic images to perspective cubemap views.

    Each ERP produces 6 cubemap faces. All views are concatenated into a single
    list that can be fed to the model as one batch of perspective images.

    Args:
        erp_images: List of N ERP images, each [H, W, 3] uint8.
        face_size: Output square face size in pixels.
        erp_names: Optional names for each ERP (for face naming). Default: ["erp0", "erp1", ...].

    Returns:
        all_views: Flat list of N*6 view dicts (same format as erp_to_cubemap).
                   Each dict has an additional 'erp_index' key (0-based).
        group_sizes: List of ints indicating how many views came from each ERP.
                     For standard cubemap this is always [6, 6, ..., 6].
    """
    if erp_names is None:
        erp_names = [f"erp{i}" for i in range(len(erp_images))]

    all_views = []
    group_sizes = []
    for idx, (erp_img, name) in enumerate(zip(erp_images, erp_names)):
        views = erp_to_cubemap(erp_img, face_size=face_size)
        for v in views:
            v['erp_index'] = idx
            v['name'] = f"{name}_{v['name']}"  # e.g. "erp0_front"
        all_views.extend(views)
        group_sizes.append(len(views))

    return all_views, group_sizes


# ==================== Tensor Preparation ====================

def cubemap_views_to_model_input(
    views: List[Dict],
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert perspective view dicts into model-ready tensors with known camera params.

    All cubemap views share the same centre (zero translation), only the orientation
    differs. This is provided as conditioning priors to guide the model.

    Args:
        views: List of view dicts from erp_to_cubemap() or erp_to_perspective().
        device: Target device for tensors.

    Returns:
        images:       [1, S, 3, H, W]  float32 in [0, 1]
        camera_poses: [1, S, 4, 4]     Camera-to-world matrices
        camera_intrs: [1, S, 3, 3]     Pinhole intrinsic matrices
    """
    converter = transforms.ToTensor()
    S = len(views)

    img_list, c2w_list, K_list = [], [], []
    for v in views:
        img = v['image']
        face_size = img.shape[0]

        # Image → [3, H, W] float32 in [0, 1]
        if img.dtype in (np.float32, np.float64):
            t = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        else:
            t = converter(Image.fromarray(img))
        img_list.append(t)

        # C2W 4×4: rotation from cubemap face, zero translation (panorama centre)
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = torch.from_numpy(v['R_c2w']).float()
        c2w_list.append(c2w)

        # Intrinsics: derived from known FOV and image size
        fov = v['fov']
        f = face_size / (2.0 * np.tan(fov / 2.0))
        K = torch.zeros(3, 3, dtype=torch.float32)
        K[0, 0] = f              # fx
        K[1, 1] = f              # fy
        K[0, 2] = face_size / 2.0  # cx
        K[1, 2] = face_size / 2.0  # cy
        K[2, 2] = 1.0
        K_list.append(K)

    images = torch.stack(img_list).unsqueeze(0).to(device)   # [1, S, 3, H, W]
    c2w = torch.stack(c2w_list).unsqueeze(0).to(device)      # [1, S, 4, 4]
    K = torch.stack(K_list).unsqueeze(0).to(device)          # [1, S, 3, 3]

    return images, c2w, K


# ==================== ERP Rendering from 3DGS ====================

def _build_erp_to_cubemap_lut(
    erp_h: int,
    erp_w: int,
    face_size: int,
    focal_length: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a look-up table that maps each ERP pixel to its best cubemap face
    and the corresponding pixel coordinate within that face.

    "Best" is defined as the face where the pixel's camera-space z is largest
    (most frontal to the face camera), reducing projection distortion.

    Returns:
        face_idx: [erp_h, erp_w] int32   – face index (0–5)
        face_u:   [erp_h, erp_w] float32 – u coordinate within the face
        face_v:   [erp_h, erp_w] float32 – v coordinate within the face
    """
    u_erp = np.arange(erp_w, dtype=np.float64)
    v_erp = np.arange(erp_h, dtype=np.float64)
    u_erp, v_erp = np.meshgrid(u_erp, v_erp)

    # ERP pixel → spherical coordinates
    theta = (u_erp / erp_w) * 2.0 * np.pi - np.pi    # longitude [-π, π]
    phi = (0.5 - v_erp / erp_h) * np.pi                # latitude  [π/2, −π/2]

    # Spherical → Cartesian (OpenCV: Y-down)
    x = np.cos(phi) * np.sin(theta)
    y = -np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    dirs = np.stack([x, y, z], axis=-1)  # [erp_h, erp_w, 3]

    face_rotations = list(CUBEMAP_FACES.values())
    cx = cy = face_size / 2.0

    best_face = np.full((erp_h, erp_w), -1, dtype=np.int32)
    best_z = np.full((erp_h, erp_w), -1.0, dtype=np.float64)
    best_u = np.zeros((erp_h, erp_w), dtype=np.float64)
    best_v = np.zeros((erp_h, erp_w), dtype=np.float64)

    for fi, R_c2w in enumerate(face_rotations):
        R_w2c = R_c2w.T
        dirs_cam = dirs @ R_w2c.T  # [erp_h, erp_w, 3]

        z_cam = dirs_cam[..., 2]
        valid = z_cam > 0.01

        u_px = dirs_cam[..., 0] / np.maximum(z_cam, 1e-8) * focal_length + cx
        v_px = dirs_cam[..., 1] / np.maximum(z_cam, 1e-8) * focal_length + cy

        in_bounds = valid & (u_px >= 0) & (u_px < face_size) & (v_px >= 0) & (v_px < face_size)
        better = in_bounds & (z_cam > best_z)

        best_face[better] = fi
        best_z[better] = z_cam[better]
        best_u[better] = u_px[better]
        best_v[better] = v_px[better]

    return best_face, best_u.astype(np.float32), best_v.astype(np.float32)


def render_erp_from_splats(
    gs_renderer,
    splats: Dict,
    erp_h: int = 1024,
    erp_w: int = 2048,
    face_size: int = 518,
) -> np.ndarray:
    """
    Render an ERP panoramic image from 3D Gaussian Splatting parameters.

    Process:
      1. Render 6 cubemap faces from the scene centre using the gsplat rasterizer.
      2. Stitch the rendered faces into an ERP image using a pre-computed LUT.

    Args:
        gs_renderer: GaussianSplatRenderer instance (only used to access rasterizer config).
        splats: Dict of Gaussian parameters from model predictions.
        erp_h: Output ERP height in pixels.
        erp_w: Output ERP width in pixels (should be 2× height).
        face_size: Cubemap face render resolution.

    Returns:
        ERP image [erp_h, erp_w, 3] float32 in [0, 1].
    """
    from src.models.models.rasterization import Rasterizer

    # Determine device from splat tensors
    m = splats["means"]
    m = m[0] if isinstance(m, list) else m
    if m.ndim == 3:
        m = m[0]  # Remove batch dim
    device = m.device

    fov = np.pi / 2.0
    f = face_size / (2.0 * np.tan(fov / 2.0))

    # Build 6 cubemap cameras at the scene centre
    K = torch.tensor([
        [f, 0, face_size / 2.0],
        [0, f, face_size / 2.0],
        [0, 0, 1.0],
    ], dtype=torch.float32, device=device)

    face_names = list(CUBEMAP_FACES.keys())
    c2w_all = []
    for name in face_names:
        c2w = torch.eye(4, dtype=torch.float32, device=device)
        c2w[:3, :3] = torch.from_numpy(CUBEMAP_FACES[name]).float().to(device)
        c2w_all.append(c2w)
    c2w_all = torch.stack(c2w_all)                      # [6, 4, 4]
    K_all = K.unsqueeze(0).expand(6, -1, -1).contiguous()  # [6, 3, 3]

    # Extract splat tensors (handle both list-of-tensors and batched formats)
    def _get(key):
        v = splats[key]
        if isinstance(v, list):
            return v[0]
        if v.ndim >= 2 and v.shape[0] == 1:
            return v[0]
        return v

    means_ = _get("means")
    quats_ = _get("quats")
    scales_ = _get("scales")
    opacities_ = _get("opacities")
    sh_key = "sh" if "sh" in splats else "colors"
    colors_ = _get(sh_key)

    # Render all 6 cubemap faces in one call
    rasterizer = Rasterizer()
    with torch.no_grad():
        rendered_colors, _, _ = rasterizer.rasterize_splats(
            means_, quats_, scales_, opacities_, colors_,
            c2w_all, K_all,
            width=face_size, height=face_size,
            sh_degree=0 if "sh" in splats else None,
        )
    # rendered_colors: [6, face_size, face_size, 3]
    face_images = rendered_colors.detach().cpu().numpy()

    # Build ERP ← cubemap LUT and stitch
    face_idx, face_u, face_v = _build_erp_to_cubemap_lut(erp_h, erp_w, face_size, f)

    erp = np.zeros((erp_h, erp_w, 3), dtype=np.float32)

    for fi in range(6):
        mask = face_idx == fi
        if not mask.any():
            continue

        u_coords = np.clip(face_u[mask], 0, face_size - 1.001)
        v_coords = np.clip(face_v[mask], 0, face_size - 1.001)

        # Bilinear interpolation within the face
        u0 = np.floor(u_coords).astype(np.int32)
        v0 = np.floor(v_coords).astype(np.int32)
        u1 = np.minimum(u0 + 1, face_size - 1)
        v1 = np.minimum(v0 + 1, face_size - 1)
        du = (u_coords - u0)[:, None]
        dv = (v_coords - v0)[:, None]

        img = face_images[fi]
        erp[mask] = (
            img[v0, u0] * (1 - du) * (1 - dv) +
            img[v0, u1] * du * (1 - dv) +
            img[v1, u0] * (1 - du) * dv +
            img[v1, u1] * du * dv
        )

    return np.clip(erp, 0.0, 1.0)


# ==================== Horizontal Rotation Video ====================

def generate_horizontal_rotation_cameras(
    num_frames: int = 120,
    fov: float = np.pi / 2,
    face_size: int = 518,
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate camera matrices for a smooth 360° horizontal rotation at eye level.
    Useful for rendering orbit-style videos from ERP-reconstructed 3DGS scenes.

    Args:
        num_frames: Number of frames in the full rotation.
        fov: FOV for each rendered frame in radians.
        face_size: Image size for intrinsic computation.
        device: Target device.

    Returns:
        camtoworlds: [1, num_frames, 4, 4] C2W matrices
        intrinsics:  [1, num_frames, 3, 3] intrinsic matrices
    """
    f = face_size / (2.0 * np.tan(fov / 2.0))
    K = torch.zeros(3, 3, dtype=torch.float32, device=device)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = face_size / 2.0
    K[1, 2] = face_size / 2.0
    K[2, 2] = 1.0

    c2w_list = []
    for i in range(num_frames):
        yaw = 2.0 * np.pi * i / num_frames
        R_c2w = _rotation_y(yaw)
        c2w = torch.eye(4, dtype=torch.float32, device=device)
        c2w[:3, :3] = torch.from_numpy(R_c2w).float().to(device)
        c2w_list.append(c2w)

    camtoworlds = torch.stack(c2w_list).unsqueeze(0)  # [1, N, 4, 4]
    intrinsics = K.unsqueeze(0).unsqueeze(0).expand(1, num_frames, -1, -1).contiguous()  # [1, N, 3, 3]

    return camtoworlds, intrinsics
