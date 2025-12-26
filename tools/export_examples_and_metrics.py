# tools/export_examples_and_metrics.py
#
# Run inference on the test set, save PNG triplets:
#   {idx:03d}_ap.png, {idx:03d}_lat_pred.png, {idx:03d}_lat_gt.png
# and write metrics.json with PSNR / SSIM.

import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data_drr_pairs import DRRPairsDataset
from src.models.unet3d_min import UNet3DMin
from src.projection_simple import forward_project_lat_from_ct


def seed_everything(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n, test_frac=0.33, seed=0):
    import random
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_test = max(1, int(n * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def to_numpy(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: (1, H, W) or (H, W)
    -> float32 numpy, normalized 0..1
    """
    if img_t.ndim == 3:
        img_t = img_t[0]
    x = img_t.detach().cpu().float().numpy()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def save_png(x: np.ndarray, path: Path):
    """Save grayscale image 0..1 -> 0..255 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    x = np.clip(x, 0.0, 1.0)
    arr = (x * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(str(path))


def psnr(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def ssim(gt, pred):
    # Simple SSIM implementation for 0..1 images.
    # Not perfect, but fine for this challenge.
    from math import sqrt

    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    mu_x = gt.mean()
    mu_y = pred.mean()
    sigma_x = gt.var()
    sigma_y = pred.var()
    sigma_xy = ((gt - mu_x) * (pred - mu_y)).mean()

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    if den <= 1e-12:
        return 1.0
    return float(num / den)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to ckpt_best.pt")
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write PNG triplets + metrics.json",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    ckpt_path = Path(args.ckpt).resolve()
    run_dir = ckpt_path.parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training config to get drr_dir, seed, test_frac, base, etc.
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = json.load(open(cfg_path, "r"))

    drr_dir = cfg.get("drr_dir", "data/drr_pairs/npz")
    seed = cfg.get("seed", 0)
    test_frac = cfg.get("test_frac", 0.33)
    base = cfg.get("base", 16)

    seed_everything(seed)

    # Dataset + test subset
    ds = DRRPairsDataset(drr_dir)
    _, test_idx = split_indices(len(ds), test_frac=test_frac, seed=seed)
    test_ds = Subset(ds, test_idx)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Model
    ck = torch.load(ckpt_path, map_location=device)
    model = UNet3DMin(base=base).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    psnrs = []
    ssims = []

    print(f"Running inference on {len(test_ds)} test samples...", flush=True)

    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            ap_img = batch["ap"].to(device)              # (B,1,H,W)
            bp = batch["bp"].to(device)                  # (B,1,Z,H,W)
            lat_gt = batch["lat"].to(device)             # (B,1,H,W)

            ct_pred = model(bp)
            lat_pred = forward_project_lat_from_ct(ct_pred)

            # Convert to 2D numpy
            ap_np = to_numpy(ap_img[0])
            lat_gt_np = to_numpy(lat_gt[0])
            lat_pred_np = to_numpy(lat_pred[0])

            # Metrics
            ps = psnr(lat_gt_np, lat_pred_np)
            ss = ssim(lat_gt_np, lat_pred_np)
            psnrs.append(ps)
            ssims.append(ss)

            # Save triplet PNGs
            base_name = f"{i:03d}"
            save_png(ap_np, out_dir / f"{base_name}_ap.png")
            save_png(lat_pred_np, out_dir / f"{base_name}_lat_pred.png")
            save_png(lat_gt_np, out_dir / f"{base_name}_lat_gt.png")

    metrics = {
    "n_test": len(psnrs),
    "psnr_mean": float(np.mean(psnrs)) if psnrs else None,
    "ssim_mean": float(np.mean(ssims)) if ssims else None,
    "psnr_all": [float(x) for x in psnrs],
    "ssim_all": [float(x) for x in ssims],
        }


    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Wrote examples to:", out_dir)
    print("Metrics:", metrics)
    print("metrics.json:", metrics_path)


if __name__ == "__main__":
    main()
