#!/usr/bin/env python3
"""
mask_human.py
Rotate → segment → inverse-rotate → save human masks for every
equirectangular panorama in a directory.

Usage:
    python mask_human.py --src panoramas/ --dst masks/ --engine clips
"""
import argparse, os, cv2, torch, numpy as np
from tqdm import tqdm
from equilib import equi2equi
import matplotlib.pyplot as plt
# from equilib.sampling import grid_sample_equirect           # warps tensor

# ----------------------------------------------------------------------
# Choose segmentation backend -------------------------------------------------
def load_clipseg(device="cuda"):
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    ).to(device).eval()
    return proc, model

@torch.inference_mode()
def run_clipseg(proc, model, img_bgr, device="cuda"):
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = proc(text=["person"], images=[pil], return_tensors="pt").to(device)
    out = model(**inputs).logits[0]      # (H,W) probability map
    mask = (torch.sigmoid(out)).cpu().numpy()
    return mask
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with *.jpg / *.png")
    ap.add_argument("--dst", required=True, help="Output mask folder")
    ap.add_argument("--engine", choices=["clips", "gsam"], default="clips")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fixed rotations
    # R_down2front = rot_x(torch.tensor([90.0]))   # +90° about x
    # R_front2down = rot_x(torch.tensor([-90.0]))  # inverse

    # load segmentation model
    if args.engine == "clips":
        proc, model = load_clipseg(device)
        segment = lambda img: run_clipseg(proc, model, img, device)
    # else:
    #     from grounded_sam import load_model, segment_image
    #     gsam = load_model("cuda")
    #     segment = lambda img: segment_image(gsam, img, "human")[0]  # binary mask

    def get_mask(img):
        # 2. segment people
        mask_gray = segment(img)              # uint8 {0,255}
        mask_gray = cv2.resize(mask_gray, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask_gray
    
    def get_mask_on_rotated(img):
        # 1. rotate so -z becomes +z
        rot = {"yaw": 0.0, "pitch": 90.0, "roll": 0.0}
        img = np.transpose(img, (2, 0, 1))  # HWC→CHW
        rot_img = equi2equi(img, rot)
        rot_img = np.transpose(rot_img, (1, 2, 0))  # CHW→HWC
        # cv2.imshow("rot_img", rot_img)
        # cv2.waitKey(0)
        
        # 2. segment people
        mask_rot = segment(rot_img)              # uint8 {0,255}

        # 3. rotate mask back
        mask_rgb = cv2.cvtColor(mask_rot, cv2.COLOR_GRAY2BGR)
        rot = {"yaw": 0.0, "pitch": -90.0, "roll": 0.0}
        mask_rgb = np.transpose(mask_rgb, (2, 0, 1))  # HWC→CHW
        mask_orig = equi2equi(mask_rgb, rot)
        mask_orig = np.transpose(mask_orig, (1, 2, 0))

        mask_gray = cv2.cvtColor(mask_orig, cv2.COLOR_BGR2GRAY)
        mask_gray = cv2.resize(mask_gray, (rot_img.shape[1], rot_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        return mask_gray
    
    

    # cv2.namedWindow("rot_img", cv2.WINDOW_NORMAL)
    # iterate over panoramas
    for fn in tqdm(sorted(os.listdir(args.src))):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(args.src, fn)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        orig_size = img.shape[:2]
        img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_LINEAR)
        if img is None:
            continue

        mask = get_mask(img)
        mask_rot = get_mask_on_rotated(img)
        # plt.imshow(mask_rot + mask)
        # plt.show()
        mask_comb = ((mask_rot + mask) > 0.7).astype(np.uint8) * 255
        mask_comb = cv2.dilate(mask_comb, np.ones((5, 5), np.uint8), iterations=10)
        # mask_rgb = cv2.cvtColor(mask_comb, cv2.COLOR_GRAY2BGR)
        # mask_rgb = cv2.addWeighted(mask_rgb, 0.5, img, 0.5, 0)
        mask_comb = cv2.resize(mask_comb, orig_size[::-1], interpolation=cv2.INTER_NEAREST)
        mask_inv = 255 - mask_comb
        cv2.imwrite(os.path.join(args.dst, fn.rsplit(".",1)[0]+"_mask.png"),
                    mask_inv)

if __name__ == "__main__":
    main()
