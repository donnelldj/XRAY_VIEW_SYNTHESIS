from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import from your training script (same folder)
from tools.train_ap2lat import DRRPairDataset, SmallUNet2D


def _as_numpy_2d(x) -> np.ndarray:
    """Accept torch tensor (1,H,W) or (H,W) or numpy and return (H,W) float32."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3:  # (1,H,W)
        x = x[0]
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to val.csv (or train.csv)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
    ap.add_argument("--n", type=int, default=10, help="# samples to visualize")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out_dir", type=str, default="", help="If set, saves images here")
    ap.add_argument("--save", action="store_true", help="Save PNGs instead of interactive show")
    ap.add_argument("--show", action="store_true", help="Show interactive windows")
    ap.add_argument("--flip_ud", action="store_true", help="Flip images vertically for display (fix upside-down)")
    ap.add_argument("--seed", type=int, default=0, help="If >0, random sample instead of linspace")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    ds = DRRPairDataset(csv_path)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = SmallUNet2D().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n = min(args.n, len(ds))
    if args.seed > 0:
        rng = np.random.RandomState(args.seed)
        idxs = rng.choice(len(ds), size=n, replace=False)
        idxs = np.sort(idxs)
    else:
        idxs = np.linspace(0, len(ds) - 1, n, dtype=int)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.save:
        if out_dir is None:
            raise ValueError("--save requires --out_dir")
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(idxs, 1):
        ap_img_t, lat_img_t = ds[idx]  # each is torch (1,H,W)
        ap_in = ap_img_t.to(device)[None, ...]  # (1,1,H,W)

        lat_gt = _as_numpy_2d(lat_img_t)  # (H,W)

        with torch.no_grad():
            pred_t = model(ap_in)
            # For visualization: clamp to [0,1]
            pred_t = torch.clamp(pred_t, 0.0, 1.0)

        ap_np = _as_numpy_2d(ap_img_t)          # (H,W)
        pred_np = _as_numpy_2d(pred_t[0])       # (H,W)
        err = np.abs(pred_np - lat_gt)

        if args.flip_ud:
            ap_np = np.flipud(ap_np)
            lat_gt = np.flipud(lat_gt)
            pred_np = np.flipud(pred_np)
            err = np.flipud(err)

        fig, ax = plt.subplots(1, 4, figsize=(14, 4))
        ax[0].imshow(ap_np, cmap="gray", vmin=0, vmax=1)
        ax[0].set_title("AP input")
        ax[1].imshow(lat_gt, cmap="gray", vmin=0, vmax=1)
        ax[1].set_title("LAT GT")
        ax[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        ax[2].set_title("LAT Pred")
        ax[3].imshow(err, cmap="magma")
        ax[3].set_title("|Error|")

        for a in ax:
            a.axis("off")

        plt.tight_layout()

        if args.save:
            fn = out_dir / f"triplet_{i:02d}_idx{idx}.png"
            fig.savefig(fn, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("saved:", fn)
        if args.show:
            plt.show()

        # If neither save nor show, default to show
        if (not args.save) and (not args.show):
            plt.show()


if __name__ == "__main__":
    main()
