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

# ---- import COLMAP parsers ---------------------------------------------------

from read_write_model import read_model, write_model, \
                             CAMERA_MODEL_NAMES, Camera, Image, Point3D

def fisheye_to_equirect(
        src_img: np.ndarray,
        coeffs: dict,
        out_size=(1600, 800),          # (width, height) of the equirect tile
        lon_lat_bounds=(0,        2*np.pi,   # lon_min, lon_max   (rad)
                        -np.pi/2, np.pi/2)): # lat_min, lat_max   (rad)
    """
    Warp a fisheye image into an equirectangular patch.

    :param src_img:     BGR or Gray fisheye frame (uint8/float32)
    :param coeffs:      dict with keys f,cx,cy,K1-4,P1,P2,B1,B2,w,h
    :param out_size:    target tile size (WxH pixels)
    :param lon_lat_bounds:
                        tuple (lon_min, lon_max, lat_min, lat_max) in radians
    :return:            equirectangular tile as np.ndarray
    """

    # --- 0. Unpack camera intrinsics ---------------------------
    f, cx, cy = coeffs['f'], coeffs['cx'], coeffs['cy']
    K1, K2, K3, K4 = (coeffs[k] for k in ('K1', 'K2', 'K3', 'K4'))
    P1, P2 = coeffs['P1'], coeffs['P2']
    B1, B2 = coeffs['B1'], coeffs['B2']
    w, h = coeffs['w'], coeffs['h']          # original image size

    # --- 1. Build lon/lat grid for the output ------------------
    W_out, H_out = out_size
    lon_min, lon_max, lat_min, lat_max = lon_lat_bounds
    lon = np.linspace(lon_min, lon_max, W_out, dtype=np.float32)
    lat = np.linspace(lat_min, lat_max, H_out, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)   # shape HxW

    # --- 2. Convert to 3-D rays in camera space ----------------
    # Using Z forward, X right, Y down:
    X = np.cos(lat_grid) * np.sin(lon_grid)
    Y = np.sin(lat_grid)
    Z = np.cos(lat_grid) * np.cos(lon_grid)

    # Rays pointing behind the camera will produce Z<=0; mark them
    # mask = Z > 1e-6
    # Z[~mask] = 1e-6        # avoid division by zero

    # --- 3. Ideal equidistant projection -----------------------
    x0 = X / Z
    y0 = Y / Z
    r0 = np.hypot(x0, y0)
    theta = np.arctan(r0)
    theta[Z <= 0] = -np.pi + theta[Z <= 0]  # flip theta for rays behind the camera
    scale = theta / r0 #np.where(r0 > 1e-8, theta / r0, 1.0)
    x = x0 * scale
    y = y0 * scale
    r2 = x * x + y * y

    # --- 4. Apply Brown-Conrady radial + tangential distortion --
    radial = 1 + K1*r2 + K2*r2**2 + K3*r2**3 + K4*r2**4
    x_prime = x * radial + (P1*(r2 + 2*x*x) + 2*P2*x*y)
    y_prime = y * radial + (P2*(r2 + 2*y*y) + 2*P1*x*y)

    # --- 5. Affinity / skew and pixel shift ---------------------
    u = 0.5*w + cx + f*x_prime + B1*x_prime + B2*y_prime
    v = 0.5*h + cy + f*y_prime

    # --- 6. Prepare remap maps, mask pixels outside FOV ---------
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    # map_x[~mask] = -1      # trigger borderValue fill
    # map_y[~mask] = -1

    # breakpoint()
    # import matplotlib.pyplot as plt
    # plt.plot(map_x[::8, ::8].reshape(-1), map_y[::8, ::8].reshape(-1), '.', markersize=1)
    # plt.title("Remap map points")
    # plt.xlabel("map_x")
    # plt.ylabel("map_y")
    # plt.xlim(0, w)
    # plt.ylim(0, h)
    # plt.grid()
    # plt.imshow(src_img)
    # plt.show()

    # --- 7. Warp -----------------------------------------------
    dst = cv2.remap(src_img, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
    return dst

def generate_mask(
        coeffs: dict,
        out_size=(1600, 800),          # (width, height) of the equirect tile
        max_angle=95 * np.pi/180,  # max angle from the camera center
        lon_lat_bounds=(0, 2*np.pi,   # lon_min, lon_max   (rad)
                        -np.pi/2, np.pi/2)): # lat_min, lat_max   (rad)
    """
    Generate a mask for the equirectangular tile, where pixels
    outside the field of view are set to 0.
    :param coeffs:      dict with keys f,cx,cy,K1-4,P1,P2,B1,B2,w,h
    :param out_size:    target tile size (WxH pixels)
    :param max_angle:   maximum angle from the camera center (rad)
    :param lon_lat_bounds:
                        tuple (lon_min, lon_max, lat_min, lat_max) in radians
    :return:            mask as np.ndarray of shape (H, W)
    """

    # --- 0. Unpack camera intrinsics ---------------------------
    f, cx, cy = coeffs['f'], coeffs['cx'], coeffs['cy']
    K1, K2, K3, K4 = (coeffs[k] for k in ('K1', 'K2', 'K3', 'K4'))
    P1, P2 = coeffs['P1'], coeffs['P2']
    B1, B2 = coeffs['B1'], coeffs['B2']
    w, h = coeffs['w'], coeffs['h']          # original image size

    # --- 1. Build lon/lat grid for the output ------------------
    W_out, H_out = out_size
    lon_min, lon_max, lat_min, lat_max = lon_lat_bounds
    lon = np.linspace(lon_min, lon_max, W_out, dtype=np.float32)
    lat = np.linspace(lat_min, lat_max, H_out, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)   # shape HxW

    # --- 2. Convert to 3-D rays in camera space ----------------
    # Using Z forward, X right, Y down:
    X = np.cos(lat_grid) * np.sin(lon_grid)
    Y = np.sin(lat_grid)
    Z = np.cos(lat_grid) * np.cos(lon_grid)

    # Rays pointing behind the camera will produce Z<=0; mark them
    # mask = Z > 1e-6
    # Z[~mask] = 1e-6        # avoid division by zero

    # --- 3. Ideal equidistant projection -----------------------
    x0 = X / Z
    y0 = Y / Z
    r0 = np.hypot(x0, y0)
    theta = np.arctan(r0)
    theta[Z <= 0] = -np.pi + theta[Z <= 0]  # flip theta for rays behind the camera

    # --- 4. Calculate mask based on max_angle -------------------
    mask = np.zeros_like(theta, dtype=np.float32)
    # Linearly vary mask from 1 to 0 as theta varies from np.pi to max_angle
    diff = max_angle - np.pi/2
    theta_diff = max_angle - np.abs(theta)
    mask = np.clip(theta_diff / diff, 0, 1)  # 1 inside FOV, 0 outside
    return mask

def dual_fisheye_to_equirect(
        left_img: np.ndarray,
        left_coeffs: dict,
        right_img: np.ndarray,
        right_coeffs: dict,
        out_size=(1600, 800),          # (width, height) of the equirect tile
    ):
    """
    Warp a dual-fisheye image into an equirectangular patch.
    """    
    left_equi = fisheye_to_equirect(left_img, left_coeffs, out_size)
    right_equi = fisheye_to_equirect(right_img, right_coeffs, out_size)

    left_mask = generate_mask(left_coeffs, out_size)
    right_mask = generate_mask(right_coeffs, out_size)

    left_equi_rotated = np.roll(left_equi, shift=out_size[0] // 2, axis=1)  # Shift left image to the right
    left_mask_rotated = np.roll(left_mask, shift=out_size[0] // 2, axis=1)
    
    left_equi_rotated, left_mask_rotated, right_equi, right_mask = \
        np.atleast_3d(left_equi_rotated, left_mask_rotated, right_equi, right_mask)

    # Combine left and right images
    combined = left_equi_rotated * left_mask_rotated + right_equi * right_mask
    combined = combined / (left_mask_rotated + right_mask)  # Normalize by mask
    combined[np.isnan(combined)] = 0  # Replace NaNs with 0
    combined[np.isinf(combined)] = 0  # Replace inf with 0
    combined = np.clip(combined, 0, 255).astype(np.uint8)  # Ensure valid pixel values

    return combined


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

def rotate_image(qvec, tvec, face):
    w2c = np.eye(4)
    w2c[:3, :3] = R.from_quat(qvec, scalar_first=True).as_matrix()
    w2c[:3, 3] = tvec
    c2w = np.linalg.inv(w2c)
    rotmat = np.eye(4)
    rotmat[:3, :3] = FACE_ROT[face].as_matrix()
    c2w_new = c2w @ rotmat
    w2c_new = np.linalg.inv(c2w_new)
    new_tvec = w2c_new[:3, 3]
    new_qvec = R.from_matrix(w2c_new[:3, :3]).as_quat(scalar_first=True)
    return new_qvec, new_tvec

# def is_point_in_face(point3D, qvec, tvec, face):
#     """Check if a 3D point is in the specified cubemap face."""
#     w2c = np.eye(4)
#     w2c[:3, :3] = R.from_quat(qvec, scalar_first=True).as_matrix()
#     w2c[:3, 3] = tvec
#     x, y, z, _ = (w2c @ np.array([point3D.x, point3D.y, point3D.z, 1.0]).T).T
#     rotation_mat = FACE_ROT[face].as_matrix().T
#     rotated_vec = rotation_mat @ np.array([x, y, z]).T
#     x, y, z = rotated_vec
#     return abs(z) >= abs(y) and abs(z) >= abs(x) and z > 0

def xy_spherical_to_face(spherical_xy, sphere_img_size, face, face_size):
    """Convert spherical pixel coordinates to cubemap face."""
    longitude, latitude = [spherical_xy[0] / sphere_img_size[0] * 2 * np.pi - np.pi, spherical_xy[1] * np.pi / sphere_img_size[1] - np.pi / 2]
    bearing_vec = np.array([
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
        np.cos(latitude) * np.cos(longitude)
    ])
    rotation_mat = FACE_ROT[face].as_matrix().T
    rotated_vec = rotation_mat @ bearing_vec
    rx, ry, rz = rotated_vec
    if abs(rz) >= abs(ry) and abs(rz) >= abs(rx) and rz > 0:
        # The point is in the face        
        x = rotated_vec[0] / abs(rotated_vec[2]) * face_size / 2 + face_size / 2
        y = rotated_vec[1] / abs(rotated_vec[2]) * face_size / 2 + face_size / 2
        return int(x), int(y)
    else:
        # The point is not in the face
        return None

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
    out_model = args.output_model
    img_dir = args.image_dir
    face_px = args.face_px
    cube_dir = args.cube_dir
    mask_dir = args.mask_dir
    cm_mask_dir = args.cube_mask_dir
    if cm_mask_dir is None:
        cm_mask_dir = Path(cube_dir).parent / "masks"
    Path(cube_dir).mkdir(parents=True, exist_ok=True)
    Path(out_model).mkdir(parents=True, exist_ok=True)
    Path(cm_mask_dir).mkdir(parents=True, exist_ok=True)

    cameras, images, points3D = read_model(in_model, ext=".txt")
    cm_images = {}
    cm_points3D = {}
    for point3D in points3D.values():
        cm_points3D[point3D.id] = Point3D(
            id=point3D.id,
            xyz=point3D.xyz,
            rgb=point3D.rgb,
            error=point3D.error,
            point2D_idxs=[],
            image_ids=[]
        )
    cm_cameras = {}
    cm_cameras[0] = Camera(
        id=0,
        model='SIMPLE_PINHOLE',
        width=face_px,
        height=face_px,
        params=np.array([
            face_px / 2,  # fx
            face_px / 2,  # cx
            face_px / 2,  # cy
        ], dtype=np.float64)  # fx, cx, cy
    )

    image_name_to_id = {image.name: image.id for image in images.values()}
    image_pairs = {}
    for image_name in image_name_to_id.keys():
        assert Path(image_name).stem.endswith("_00") or Path(image_name).stem.endswith("_10"), \
            f"Image {image_name} does not have a valid stem ending with _00 or _10."
        if Path(image_name).stem.endswith("_00"):
            pair_name = Path(image_name).stem[:-3] + "_10" + Path(image_name).suffix
            im_id = image_name_to_id[image_name]
            pair_id = image_name_to_id[pair_name]
            image_pairs[(im_id, pair_id)] = (images[im_id], images[pair_id])
    if len(image_pairs) == 0:
        raise ValueError("No image pairs found with the expected naming convention (_00 and _10).")

    image_id_count = 0
    for ks, images in tqdm(image_pairs.items()):
        k1, k2 = ks
        l_img, r_img = images
        left_fisheye = cv2.imread(str(Path(img_dir) / l_img.name), cv2.IMREAD_UNCHANGED)
        right_fisheye = cv2.imread(str(Path(img_dir) / r_img.name), cv2.IMREAD_UNCHANGED)
        pano = dual_fisheye_to_equirect(
            left_fisheye, to_param_dict(cameras[l_img.camera_id]),
            right_fisheye, to_param_dict(cameras[r_img.camera_id]),
            out_size=(int(face_px * 3.2), int(face_px * 1.6))
        )

        
        left_mask = cv2.imread(str(Path(mask_dir) / l_img.name), cv2.IMREAD_UNCHANGED) if mask_dir else None
        right_mask = cv2.imread(str(Path(mask_dir) / r_img.name), cv2.IMREAD_UNCHANGED) if mask_dir else None
        mask = dual_fisheye_to_equirect(
            left_mask, to_param_dict(cameras[l_img.camera_id]),
            right_mask, to_param_dict(cameras[r_img.camera_id]),
            out_size=(int(face_px * 3.2), int(face_px * 1.6))
        )

        # cv2.imshow("pano", pano)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # convert to cubemap faces
        cube_faces = pyc.e2c(pano[:, :, ::-1], face_w=face_px, cube_format="dict")  # BGR→RGB
        # py360convert returns dict {'F','R','B','L','U','D'}
        
        if mask_dir is not None:
            mask_cube_faces = pyc.e2c(mask[:, :, ::-1], face_w=face_px, cube_format="dict")

        image_name = f"{Path(l_img.name).stem[:-3]}{Path(l_img.name).suffix}"

        for face, quat in FACE_ROT.items():
            # if face == "down" or face == "up":
            #     continue

            face_id = f"{Path(image_name).stem}_{face}{Path(image_name).suffix}"
            image_id = image_id_count #len(FACE_ROT.values())*k + FACE_IDX[face]
            image_id_count += 1
            new_qvec, new_tvec = rotate_image(l_img.qvec, l_img.tvec, face)
            new_xys = []
            new_point3D_ids = []
            # # Check if points3D are in the face
            # for xy_id, point3D_id in enumerate(image.point3D_ids):
            #     img_size = [cameras[image.camera_id].width, cameras[image.camera_id].height]
            #     face_xy = xy_spherical_to_face(
            #         spherical_xy=image.xys[xy_id],
            #         sphere_img_size=img_size,
            #         face=face,
            #         face_size=face_px
            #     )
            #     if face_xy is not None:
            #         x, y = face_xy
            #         cm_points3D[point3D_id].image_ids.append(image_id)
            #         cm_points3D[point3D_id].point2D_idxs.append(len(new_xys))
            #         new_xys.append([x, y])
            #         new_point3D_ids.append(point3D_id)
            
            # if len(new_xys) == 0:
            #     print(f"Image {image_id} ({face_id}) has no points in the face {face}, skipping.")
            #     continue
            # cm_images[image_id] = Image(
            #     id=image_id,
            #     qvec=new_qvec,
            #     tvec=new_tvec,
            #     camera_id=image.camera_id,
            #     name=face_id,
            #     xys=np.array(new_xys, dtype=np.float64),
            #     point3D_ids=np.array(new_point3D_ids, dtype=np.int64)
            # )

            cm_images[image_id] = Image(
                id=image_id,
                qvec=new_qvec,
                tvec=new_tvec,
                camera_id=0,
                name=face_id,
                xys=l_img.xys,
                point3D_ids=l_img.point3D_ids
            )

            face_key = FACE_KEY[face]
            cv2.imwrite(str(Path(cube_dir) / face_id), cube_faces[face_key][:, :, ::-1])
            if mask_dir is not None:
                cv2.imwrite(str(Path(cm_mask_dir) / face_id), mask_cube_faces[face_key][:, :, ::-1])

    pt_ids = list(cm_points3D.keys())
    for point3D_id in pt_ids:
        point3D = cm_points3D[point3D_id]
        if len(point3D.image_ids) < 2:
            print(f"Point {point3D.id} has less than 2 images, removing it.")
            del cm_points3D[point3D.id]

    write_model(
        cameras=cm_cameras,
        images=cm_images,
        points3D=points3D,
        path=out_model,
        ext=".txt"
    )
    print(f"Cube-map model written to {out_model}")

# -----------------------------------------------------------------------------    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-model",  required=True,
                    help="Folder containing original *.txt model")
    ap.add_argument("--image-dir",    required=True,
                    help="Folder with the equirectangular panoramas")
    ap.add_argument("--mask-dir",     default=None,
                    help="Folder with the masks for the panoramas")
    ap.add_argument("--output-model", default="model_cubemap",
                    help="Folder for the new cube-map model")
    ap.add_argument("--cube-dir",     default="cube_images",
                    help="Where to write the six face PNGs")
    ap.add_argument("--cube-mask-dir",    default=None,
                    help="Where to write the six face mask PNGs")
    ap.add_argument("--face-px",      type=int, default=1024,
                    help="Face resolution (square)")
    args = ap.parse_args()
    main(args)