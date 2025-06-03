"""
This script converts a COLMAP model with spherical images into a cubemap model.
This script is primarily used to get colmap export format after aligning spherical images in Agisoft.
Step 1: Align spherical images in Agisoft.
Step 2: Set the camera type to "Frame" in Agisoft Metashape 2.2.1 (otherwise COLMAP export is blocked). Export the aligned cameras to COLMAP format.
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

def quat_from_euler(yaw, pitch, roll):
    """Return w,x,y,z quaternion from intrinsic ZYX (yaw-pitch-roll) angles (rad)."""
    cy, sy = np.cos(yaw*0.5), np.sin(yaw*0.5)
    cp, sp = np.cos(pitch*0.5), np.sin(pitch*0.5)
    cr, sr = np.cos(roll*0.5), np.sin(roll*0.5)
    return np.array([
        cr*cp*cy + sr*sp*sy,  # w
        sr*cp*cy - cr*sp*sy,  # x
        cr*sp*cy + sr*cp*sy,  # y
        cr*cp*sy - sr*sp*cy   # z
    ])

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
    for camera_id, camera in cameras.items():
        # if camera.model not in CAMERA_MODEL_NAMES:
        #     raise ValueError(f"Unsupported camera model: {camera.model}")
        cm_cameras[camera_id] = Camera(
            id=camera.id,
            model='SIMPLE_PINHOLE',
            width=face_px,
            height=face_px,
            params=np.array([
                face_px / 2,  # fx
                face_px / 2,  # cx
                face_px / 2,  # cy
            ], dtype=np.float64)  # fx, cx, cy
        )

    image_id_count = 0
    for k, image in tqdm(images.items()):
        if image.id < 114 or image.id > 154:
            continue
        # load panorama once
        pano = cv2.imread(str(Path(img_dir)/ image.name), cv2.IMREAD_UNCHANGED)
        if pano is None:
            raise FileNotFoundError(image.name)

        # convert to cubemap faces
        cube_faces = pyc.e2c(pano[:, :, ::-1], face_w=face_px, cube_format="dict")  # BGR→RGB
        # py360convert returns dict {'F','R','B','L','U','D'}
        
        if mask_dir is not None:
            mask = cv2.imread(str(Path(mask_dir) / f"{Path(image.name).stem}.png"))
            if mask is None:
                raise FileNotFoundError(image.name)
            mask_cube_faces = pyc.e2c(mask[:, :, ::-1], face_w=face_px, cube_format="dict")



        for face, quat in FACE_ROT.items():
            # if face == "down" or face == "up":
            #     continue

            face_id = f"{Path(image.name).stem}_{face}{Path(image.name).suffix}"
            image_id = image_id_count #len(FACE_ROT.values())*k + FACE_IDX[face]
            image_id_count += 1
            print (f"Writing {image_id} originating from {k} ({image.name}) to {face_id}")
            new_qvec, new_tvec = rotate_image(image.qvec, image.tvec, face)
            new_xys = []
            new_point3D_ids = []
            # Check if points3D are in the face
            for xy_id, point3D_id in enumerate(image.point3D_ids):
                img_size = [cameras[image.camera_id].width, cameras[image.camera_id].height]
                face_xy = xy_spherical_to_face(
                    spherical_xy=image.xys[xy_id],
                    sphere_img_size=img_size,
                    face=face,
                    face_size=face_px
                )
                if face_xy is not None:
                    x, y = face_xy
                    cm_points3D[point3D_id].image_ids.append(image_id)
                    cm_points3D[point3D_id].point2D_idxs.append(len(new_xys))
                    new_xys.append([x, y])
                    new_point3D_ids.append(point3D_id)
            
            if len(new_xys) == 0:
                print(f"Image {image_id} ({face_id}) has no points in the face {face}, skipping.")
                continue
            cm_images[image_id] = Image(
                id=image_id,
                qvec=new_qvec,
                tvec=new_tvec,
                camera_id=image.camera_id,
                name=face_id,
                xys=np.array(new_xys, dtype=np.float64),
                point3D_ids=np.array(new_point3D_ids, dtype=np.int64)
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
        points3D=cm_points3D,
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