from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from train_ap2lat import DRRPairDataset, SmallUNet2D  # assumes same folder; adjust if needed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/drp_pairs")
    ap.add_argument("--ckpt", type=str, default="runs/ap2lat_rtx/best.pt")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    val_csv = Path(args.data_dir) / "val.csv"
    ds = DRRPairDataset(val_csv)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = SmallUNet2D().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    idxs = np.linspace(0, len(ds)-1, args.n, dtype=int)

    for i, idx in enumerate(idxs, 1):
        ap_img, lat_img = ds[idx]
        ap_img = ap_img.to(device)[None, ...]      # (1,1,H,W)
        lat_img = lat_img.numpy()[0]               # (H,W)

        with torch.no_grad():
            pred = model(ap_img).cpu().numpy()[0,0]

        err = np.abs(pred - lat_img)

        fig, ax = plt.subplots(1, 4, figsize=(14, 4))
        ax[0].imshow(ap_img.cpu().numpy()[0,0], cmap="gray")
        ax[0].set_title("AP input")
        ax[1].imshow(lat_img, cmap="gray")
        ax[1].set_title("LAT GT")
        ax[2].imshow(pred, cmap="gray")
        ax[2].set_title("LAT Pred")
        ax[3].imshow(err, cmap="magma")
        ax[3].set_title("|Error|")

        for a in ax: a.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
