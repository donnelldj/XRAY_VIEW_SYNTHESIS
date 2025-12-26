# scripts/viz_ap_lat_triplets.py
# Save side-by-side AP / Pred LAT / GT LAT PNGs for inspection & report.

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
    img_t: (C,H,W) or (1,H,W) or (H,W) tensor -> normalized float32 numpy [0,1]
    """
    x = img_t.detach().cpu().float()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]          # (H,W)
    x = x.numpy()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/ap2lat_baseline/ckpt_best.pt")
    parser.add_argument("--out_dir", default="runs/ap2lat_baseline/viz_triplets")
    parser.add_argument("--num", type=int, default=10, help="how many triplets to dump")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    run_dir = ckpt_path.parent

    # Load training config so we use the same data split
    cfg_path = run_dir / "config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    drr_dir = cfg.get("drr_dir", "data/drr_pairs/npz")
    seed = cfg.get("seed", 0)
    test_frac = cfg.get("test_frac", 0.33)
    base = cfg.get("base", 16)

    seed_everything(seed)

    # Dataset + test split (same logic as training)
    ds = DRRPairsDataset(drr_dir)
    _, test_idx = split_indices(len(ds), test_frac=test_frac, seed=seed)
    test_ds = Subset(ds, test_idx)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    model = UNet3DMin(base=base).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("saving PNGs to:", out_dir, flush=True)

    # Headless matplotlib backend so this works from CLI
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            if i >= args.num:
                break

            ap = batch["ap"].to(device)      # (1,1,H,W)
            bp = batch["bp"].to(device)      # (1,1,Z,H,W)
            lat_gt = batch["lat"].to(device) # (1,1,H,W)

            ct_pred = model(bp)
            lat_pred = forward_project_lat_from_ct(ct_pred)

            ap_np = to_numpy(ap[0])         # (H,W)
            lat_pred_np = to_numpy(lat_pred[0])
            lat_gt_np = to_numpy(lat_gt[0])

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(ap_np, cmap="gray")
            axes[0].set_title("AP input")
            axes[1].imshow(lat_pred_np, cmap="gray")
            axes[1].set_title("Pred LAT")
            axes[2].imshow(lat_gt_np, cmap="gray")
            axes[2].set_title("GT LAT")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()

            out_path = out_dir / f"triplet_{i:03d}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
