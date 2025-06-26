"""
This script converts a COLMAP model with dual fisheye images into a cubemap model.
This script is primarily used to get colmap export format after aligning spherical images in Agisoft.
Step 1: Align dual fisheye images in Agisoft.
Step 2: In Agisoft Metashape 2.2.1, export the aligned cameras to COLMAP format.
Step 3: Run this script to convert the exported COLMAP model to a COLMAP model made with cubemaps.
"""

import argparse, math, numpy as np, cv2
import py360convert as pyc
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from tqdm import tqdm
import open3d as o3d

# ---- import COLMAP parsers ---------------------------------------------------

from read_write_model import read_model, write_model, \
                             CAMERA_MODEL_NAMES, Camera, Image, Point3D

XYZ_dict = {}

def equirect_to_fisheye(eq_img: np.ndarray,
                        coeffs: dict,
                        out_size=None,           # (w, h) of fisheye
                        iterations: int = 15):
    """
    Warp full-sphere equirectangular panorama into a distorted
    equidistant-fisheye image that follows the Metashape model.
    """

    # --- unpack -------------------------------------------------
    f, cx, cy = coeffs['f'], coeffs['cx'], coeffs['cy']
    K1, K2, K3, K4 = (coeffs[k] for k in ('K1','K2','K3','K4'))
    P1, P2 = coeffs['P1'], coeffs['P2']
    B1, B2 = coeffs['B1'], coeffs['B2']
    W_f, H_f = coeffs['w'], coeffs['h'] if out_size is None else out_size
    out_size = (W_f, H_f)
    W_eq, H_eq = eq_img.shape[1], eq_img.shape[0]

    key = (tuple(coeffs.values()), out_size, iterations)
    if key not in XYZ_dict:
        # --- 1. fisheye pixel grid ---------------------------------
        u, v = np.meshgrid(np.arange(W_f, dtype=np.float32),
                        np.arange(H_f, dtype=np.float32))
        dx = u - (0.5*W_f + cx)
        dy = v - (0.5*H_f + cy)

        y_p = dy / f
        x_p = (dx - B2 * y_p) / (f + B1)

        # --- 2. invert Brown-Conrady -------------------------------
        x, y = x_p.copy(), y_p.copy()
        for _ in range(iterations):
            r2  = x*x + y*y
            r4, r6, r8 = r2*r2, r2*r2*r2, r2*r2*r2*r2
            D   = 1 + K1*r2 + K2*r4 + K3*r6 + K4*r8
            dx  = P1*(r2 + 2*x*x) + 2*P2*x*y
            dy  = P2*(r2 + 2*y*y) + 2*P1*x*y
            x   -= (x*D + dx) - x_p
            y   -= (y*D + dy) - y_p

        # --- 3. ray direction --------------------------------------
        theta = np.sqrt(x*x + y*y)
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_phi = np.where(theta > 1e-8, x/theta, 1)
            sin_phi = np.where(theta > 1e-8, y/theta, 0)
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        X = sin_t * cos_phi
        Y = sin_t * sin_phi
        Z = cos_t
        # store XYZ for later use
        XYZ_dict[key] = (X, Y, Z)
    else:
        # use precomputed XYZ
        X, Y, Z = XYZ_dict[key]

    rays = np.stack((X, Y, Z), axis=-1)  # (H_f, W_f, 3)

    # --- 4. lon/lat to equirect coords -------------------------
    lon = np.arctan2(X, Z)
    lat = np.arcsin(Y)
    map_x = ((lon + np.pi) / (2*np.pi) * W_eq).astype(np.float32)
    map_y = ((lat + np.pi/2) / np.pi  * H_eq).astype(np.float32)

    # keep modulo so right edge wraps cleanly
    map_x %= W_eq
    # set undefined rays (behind sensor) to -1 → border fill
    # map_y[cos_t < 0] = -1

    # --- 5. resample -------------------------------------------
    resampled = cv2.remap(eq_img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)
    return resampled, rays


def render_depth_from_mesh(mesh: o3d.geometry.TriangleMesh,
                         qvec: np.ndarray, tvec: np.ndarray,
                         face_px: int = 512):
    """
    Render depth map from a mesh using the camera pose defined by qvec and tvec.
    """
    # mesh.compute_vertex_normals()
    # mesh.paint_uniform_color([1.0, 1.0, 1.0])  # white color

    # Convert qvec (quaternion) to rotation matrix
    R_mat = R.from_quat(qvec, scalar_first=True).as_matrix()

    # Set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = R_mat
    camera_pose[:3, 3] = tvec

    # Render depth map
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=face_px, height=face_px)
    vis.add_geometry(mesh)
    
    ctr = vis.get_view_control()
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic.set_intrinsics(face_px, face_px,
                                    face_px / 2, face_px / 2,
                                    face_px / 2 - 0.5, face_px / 2 - 0.5)
    
    depth_cube = {}
    scene = o3d.t.geometry.RaycastingScene()
    rays = scene.create_rays_pinhole(
        o3d.core.Tensor([   [face_px / 2, 0.0, face_px / 2 - 0.5], 
                            [0.0, face_px / 2, face_px / 2 - 0.5],
                            [0, 0, 1]],
                         dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.eye(4), dtype=o3d.core.Dtype.Float32),
        face_px,
        face_px,
    )
    rays = rays.numpy()[:, :, 3:]
    rays = np.linalg.norm(rays, axis=-1)
    for face, face_rot in FACE_ROT.items():
        rotmat = np.eye(4)
        rotmat[:3, :3] = face_rot.as_matrix().T
        params.extrinsic= rotmat @ camera_pose
        ctr.convert_from_pinhole_camera_parameters(params)
        depth_map = vis.capture_depth_float_buffer(do_render=True)
        depth_map = np.asarray(depth_map)
        depth_cube[FACE_KEY[face]] = depth_map * rays

    
    depth_equi = pyc.c2e(depth_cube, 1000, 2000, cube_format="dict")
    # import matplotlib.pyplot as plt
    # plt.imshow(depth_equi)
    # plt.show()
    vis.destroy_window()
    return np.asarray(depth_equi)

FACE_ROT = {
    "front":  R.from_euler("zyx", [0, 0, 0], degrees=True),
    "back":   R.from_euler("zyx", [0, 180, 0], degrees=True),
    "right":  R.from_euler("zyx", [0, 90, 0], degrees=True),
    "left":   R.from_euler("zyx", [0, -90, 0], degrees=True),
    "up":     R.from_euler("zyx", [0, 0, 90], degrees=True),   # look up
    "down":   R.from_euler("zyx", [0, 0, -90], degrees=True),    # look down
}

FACE_IDX = {
    "front": 0,
    "back":  1,
    "right": 2,
    "left":  3,
    "up":    4,
    "down":  5
}

FACE_KEY = {
    "front": "F",
    "right": "R",
    "back":  "B",
    "left":  "L",
    "up":    "U",
    "down":  "D"
}

def to_param_dict(camera):
    assert camera.model == "THIN_PRISM_FISHEYE", \
        f"Unsupported camera model: {camera.model}. Expected THIN_PRISM_FISHEYE."
    params = {
        "f": camera.params[0],
        "cx": -camera.width/2 + camera.params[2],
        "cy": -camera.height/2 + camera.params[3],
        "K1": camera.params[4],
        "K2": camera.params[5],
        "P1": camera.params[6],
        "P2": camera.params[7],
        "B1": 0.0,
        "B2": 0.0,
        "K3": camera.params[8],
        "K4": camera.params[9],
        "w": camera.width,
        "h": camera.height
    }
    return params

def main(args):
    in_model  = args.input_model
    depth_dir = args.depth_dir

    cameras, images, points3D = read_model(in_model, ext=".txt")
    mesh = o3d.io.read_triangle_mesh(args.mesh)

    for k, images in tqdm(images.items()):
        equi_depth = render_depth_from_mesh(mesh,
                                            images.qvec, images.tvec,
                                            face_px=512)
        # breakpoint()
        # img = cv2.imread("/media/sanskar/runs/shreyansh/data/gsplat/yelloskye/equirect_tile1.jpg")
        # w = img.shape[1]
        # img[:, w//4:w*3//4, :] = 0  # crop to center
        # img = np.roll(img, shift=-int(img.shape[1] // 2), axis=1)  # shift to center
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        # img_fisheye = equirect_to_fisheye(
        #     eq_img=img,
        #     coeffs=to_param_dict(cameras[images.camera_id]),
        # )
        # plt.imshow(img_fisheye)
        # plt.show()

        dist_fisheye, rays = equirect_to_fisheye(
            eq_img=equi_depth,
            coeffs=to_param_dict(cameras[images.camera_id]),
        )
        rays_norm = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        depth_fisheye = dist_fisheye * rays_norm[..., 2]  # depth = distance / cos(theta)
        depth_fisheye[depth_fisheye < 0] = 0  # set negative depths to 0
        # import matplotlib.pyplot as plt
        # plt.imshow(depth_fisheye)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(rays_norm)
        # plt.colorbar()
        # plt.show()
        cv2.imwrite(
            str(Path(depth_dir) / f"{images.name[:-4]}.tiff"),
            depth_fisheye.astype(np.float32)
        )
        
        print (f"Rendering depth for image {k} ({images.name})")
        # save depth map
        # depth

# -----------------------------------------------------------------------------    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-model",  required=True,
                    help="Folder containing original *.txt model")
    ap.add_argument("--mesh", required=True, 
                    help="Path to the mesh file (e.g., .obj) for depth rendering")
    ap.add_argument("--depth-dir", required=True,
                    help="Output directory for depth maps")
    args = ap.parse_args()
    main(args)


# ---------- quick demo ----------------------------------------
# if __name__ == "__main__":
#     pano = cv2.imread("/media/sanskar/runs/shreyansh/data/gsplat/yelloskye/equirect_tile3.jpg")           # full-sphere pano


#     fish = equirect_to_fisheye(pano, calib2)
#     cv2.imwrite("/media/sanskar/runs/shreyansh/data/gsplat/yelloskye/fisheye_tile3_diff.jpg", fish)