import os
import glob
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# Import methods for DRR generation from the appropriate module
from src.vis.drr import drr_ap, drr_lat  # Use drr_ap and drr_lat for DRR generation

# Normalize image to range [0,1]
def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)  # Prevent division by zero
    return x

# Function to save images as PNG
def save_png(img01: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(img01)).save(out_path)

# Convert to uint8 for saving
def to_uint8(img01: np.ndarray) -> np.ndarray:
    img = np.clip(img01, 0.0, 1.0)  # Ensure values are within the [0,1] range
    return (img * 255.0).round().astype(np.uint8)

# Function to apply reorientation similar to geo_smoke.py (for AP and LAT projections)
def apply_reorientation(img: np.ndarray, flip_ud: bool = False, flip_lr: bool = False, rot_k: int = 0) -> np.ndarray:
    """ 
    Apply orientation adjustments to match visualization in geo_smoke.
    flip_ud: Flip up-down (vertical axis)
    flip_lr: Flip left-right (horizontal axis)
    rot_k: Rotate the image k times (90° each rotation)
    """
    if flip_ud:
        img = np.flipud(img)
    if flip_lr:
        img = np.fliplr(img)
    if rot_k:
        img = np.rot90(img, k=rot_k)
    return img

# Main function to create DRR pairs
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default=r"runs\runs_final\data\drr_pairs_fixed\npz")  # Path to NPZ files
    ap.add_argument("--out_dir", default=r"runs/runs_final/data/drr_pairs")  # Output directory
    ap.add_argument("--n", type=int, default=30, help="Number of cases to export")  # How many cases to export
    ap.add_argument("--seed", type=int, default=0)  # Random seed for selection
    ap.add_argument("--flip_ud", action="store_true", help="Flip images vertically")  # Option to flip images
    ap.add_argument("--flip_lr", action="store_true", help="Flip images horizontally")  # Option to flip images
    ap.add_argument("--rot_k", type=int, default=0, help="Rotate image k times (90° each)")  # Rotation argument
    args = ap.parse_args()

    np.random.seed(args.seed)

    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))  # Load all .npz files
    if not files:
        raise SystemExit(f"No .npz files found in {args.npz_dir}")

    # Select a subset of files if needed
    if args.n < len(files):
        idx = np.random.choice(len(files), size=args.n, replace=False)
        files = [files[i] for i in sorted(idx)]

    out_dir = Path(args.out_dir)
    (out_dir / "npz").mkdir(parents=True, exist_ok=True)  # Create output directories
    (out_dir / "png").mkdir(parents=True, exist_ok=True)

    exported = 0

    for p in files:
        d = np.load(p)
        if "ct_zyx" not in d.files:
            print(f"Skipping (no ct_zyx): {p}")
            continue

        ct = d["ct_zyx"]  # Load CT data (Z, Y, X)

        # Use drr_ap and drr_lat methods to generate DRRs
        ap_img = drr_ap(ct).astype(np.float32)  # AP view (Y, X)
        lat_img = drr_lat(ct).astype(np.float32)  # LAT view (Z, Y)

        # Apply reorientation based on the args (flip, rotate)
        ap_img = apply_reorientation(ap_img, flip_ud=args.flip_ud, flip_lr=args.flip_lr, rot_k=args.rot_k)
        lat_img = apply_reorientation(lat_img, flip_ud=args.flip_ud, flip_lr=args.flip_lr, rot_k=args.rot_k)

        # Ensure both views are 256x256 by cropping/padding as needed
        if ap_img.shape != (256, 256):
            ap_img = ap_img[:256, :256]  # Crop AP to 256x256

        if lat_img.shape[1] != 256:
            lat_img = lat_img[:, :256]

        # Pad LAT to 256x256 if needed
        if lat_img.shape[0] < 256:
            pad = 256 - lat_img.shape[0]
            lat_img = np.pad(lat_img, ((pad//2, pad - pad//2), (0, 0)), mode="constant", constant_values=0.0)
        elif lat_img.shape[0] > 256:
            start = (lat_img.shape[0] - 256)//2
            lat_img = lat_img[start:start+256, :]

        case_id = Path(p).stem

        # Save as npz (already exists and no need for re-export of the NPZ data)
        out_npz = out_dir / "npz" / f"{case_id}.npz"
        np.savez_compressed(out_npz, ap=ap_img.astype(np.float16), lat=lat_img.astype(np.float16), src=np.array([str(p)]))

        # Save PNG images for quick visualization (AP and LAT as per geo_smoke)
        out_png_dir = out_dir / "png" / case_id
        save_png(ap_img, out_png_dir / "ap.png")
        save_png(lat_img, out_png_dir / "lat_gt.png")

        exported += 1

    print(f"Exported {exported} DRR pairs to: {out_dir}")

if __name__ == "__main__":
    main()
