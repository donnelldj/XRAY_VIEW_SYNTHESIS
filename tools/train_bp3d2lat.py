from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from src.geo.backproject import backproject_parallel_beam
from src.models_unet3d import UNet3D


EPS = 1e-8


def minmax01_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def minmax01_t(x: torch.Tensor) -> torch.Tensor:
    # per-sample min/max over spatial dims
    # expects x: (B,1,H,W) or (B,1,Z,Y) etc.
    mn = x.amin(dim=tuple(range(2, x.ndim)), keepdim=True)
    mx = x.amax(dim=tuple(range(2, x.ndim)), keepdim=True)
    return (x - mn) / (mx - mn + EPS)


def psnr_t(pred01: torch.Tensor, gt01: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred01 - gt01) ** 2, dim=tuple(range(2, pred01.ndim)))
    return 20.0 * torch.log10(torch.tensor(1.0, device=pred01.device)) - 10.0 * torch.log10(mse + EPS)


def _gaussian_window_2d(win: int = 11, sigma: float = 1.5, device: Optional[torch.device] = None) -> torch.Tensor:
    coords = torch.arange(win, dtype=torch.float32, device=device) - (win - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,win,win)
    return w


def ssim_t(pred01: torch.Tensor, gt01: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    pred01, gt01: (B,1,H,W) OR (B,1,Z,Y) treated as 2D images per sample
    Returns: (B,)
    """
    device = pred01.device
    w = _gaussian_window_2d(win, sigma, device=device)
    padding = win // 2

    mu1 = F.conv2d(pred01, w, padding=padding)
    mu2 = F.conv2d(gt01, w, padding=padding)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred01 * pred01, w, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(gt01 * gt01, w, padding=padding) - mu2_sq
    sigma12 = F.conv2d(pred01 * gt01, w, padding=padding) - mu12

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + EPS)
    return ssim_map.mean(dim=(2, 3)).squeeze(1)  # (B,)


class BP3DToLatDataset(Dataset):
    """
    Uses your NEW dataset convention:
      ap  : (Z,X)
      lat : (Z,Y)

    Backproject AP into volume (Z,Y,X) by smearing along Y => axis=1 in (Z,Y,X).
    """

    def __init__(self, csv_path: Path, y_size: Optional[int] = None):
        self.df = pd.read_csv(csv_path)
        self.y_size = y_size

        if "npz_path" not in self.df.columns:
            raise ValueError(f"CSV missing 'npz_path': {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        p = Path(str(self.df.iloc[idx]["npz_path"]))
        d = np.load(p, allow_pickle=True)

        ap = d["ap"].astype(np.float32)   # (Z,X)
        lat = d["lat"].astype(np.float32) # (Z,Y)

        # normalize for stability (your drr is already 0..1; this just defends against drift)
        ap = minmax01_np(ap)
        lat = minmax01_np(lat)

        z, x = ap.shape
        z2, y = lat.shape
        if z != z2:
            raise ValueError(f"Z mismatch ap={ap.shape} lat={lat.shape} in {p}")

        y_use = int(self.y_size) if self.y_size is not None else int(y)

        vbp = backproject_parallel_beam(ap, out_zyx=(z, y_use, x), axis=1).astype(np.float32)  # (Z,Y,X)

        x_t = torch.from_numpy(vbp)[None, ...]  # (1,Z,Y,X)
        y_t = torch.from_numpy(lat)[None, ...]  # (1,Z,Y)

        meta = {"npz_path": str(p.as_posix())}
        return x_t, y_t, meta


def save_triplet_png(
    out_path: Path,
    ap_zx: np.ndarray,
    lat_gt_zy: np.ndarray,
    lat_pred_zy: np.ndarray,
    flip_ud_viz: bool = True,
):
    ap_show = minmax01_np(ap_zx)
    gt_show = minmax01_np(lat_gt_zy)
    pr_show = minmax01_np(lat_pred_zy)
    err = np.abs(pr_show - gt_show)

    if flip_ud_viz:
        ap_show = np.flipud(ap_show)
        gt_show = np.flipud(gt_show)
        pr_show = np.flipud(pr_show)
        err = np.flipud(err)

    fig, ax = plt.subplots(1, 4, figsize=(14, 4))
    ax[0].imshow(ap_show, cmap="gray"); ax[0].set_title("AP (Z,X)")
    ax[1].imshow(gt_show, cmap="gray"); ax[1].set_title("LAT GT (Z,Y)")
    ax[2].imshow(pr_show, cmap="gray"); ax[2].set_title("LAT Pred (Z,Y)")
    ax[3].imshow(err, cmap="magma"); ax[3].set_title("|diff|")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/bp3d2lat")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1)  # 3D is heavy
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--base", type=int, default=16, help="UNet3D base channels (16 is safe)")
    ap.add_argument("--amp", action="store_true", help="use mixed precision")
    ap.add_argument("--y_size", type=int, default=None, help="optional Y used for backprojection volume")

    ap.add_argument("--save_triplets", action="store_true")
    ap.add_argument("--n_triplets", type=int, default=10)
    ap.add_argument("--flip_ud_viz", action="store_true", help="flip images for display (fix upside-down)")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = BP3DToLatDataset(Path(args.train_csv), y_size=args.y_size)
    val_ds = BP3DToLatDataset(Path(args.val_csv), y_size=args.y_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    model = UNet3D(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    
    # Inside your training loop, after computing va_loss

    best_val_loss = float('inf')  # Initialize best validation loss
    best_epoch = 0  # Optional: Track which epoch had the best model

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0

        for vbp, lat_gt, _meta in train_loader:
            vbp = vbp.to(device, non_blocking=True)
            lat_gt = lat_gt.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                v_hat = model(vbp)  # (B,1,Z,Y,X)
                lat_pred = v_hat.sum(dim=4)  # (B,1,Z,Y)

                lat_pred01 = minmax01_t(lat_pred)
                lat_gt01 = minmax01_t(lat_gt)

                loss = loss_fn(lat_pred01, lat_gt01)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += float(loss.item()) * vbp.size(0)
            n_tr += vbp.size(0)

        tr_loss /= max(1, n_tr)

        # ---- validation ----
        model.eval()
        va_loss = 0.0
        psnr_list: List[float] = []
        ssim_list: List[float] = []
        n_va = 0

        with torch.no_grad():
            for j, (vbp, lat_gt, meta) in enumerate(val_loader):
                vbp = vbp.to(device, non_blocking=True)
                lat_gt = lat_gt.to(device, non_blocking=True)

                v_hat = model(vbp)
                lat_pred = v_hat.sum(dim=4)

                lat_pred01 = minmax01_t(lat_pred)
                lat_gt01 = minmax01_t(lat_gt)

                loss = loss_fn(lat_pred01, lat_gt01)
                va_loss += float(loss.item())
                n_va += 1

                # metrics
                psnr_b = psnr_t(lat_pred01, lat_gt01)  # (B,)
                ssim_b = ssim_t(lat_pred01, lat_gt01)  # (B,)
                psnr_list.append(float(psnr_b.mean().item()))
                ssim_list.append(float(ssim_b.mean().item()))

            va_loss /= max(1, n_va)
            psnr_mean = float(np.mean(psnr_list)) if psnr_list else 0.0
            ssim_mean = float(np.mean(ssim_list)) if ssim_list else 0.0

            print(f"epoch {epoch:03d} | train {tr_loss:.5f} | val {va_loss:.5f} | "
                f"PSNR {psnr_mean:.2f} | SSIM {ssim_mean:.3f}")

            # Only save the model if it achieves the best validation loss so far
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_epoch = epoch
                print(f"  Saving best model at epoch {epoch}")
                torch.save(model.state_dict(), out_dir / "best_model.pth")

        # Optional: Save checkpoints after every epoch (can still save the state_dict)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "val_loss": va_loss,
            "psnr": psnr_mean,
            "ssim": ssim_mean,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / f"epoch_{epoch}.pt")

        # After training, you can also save the best model info
        if epoch == best_epoch:
            print(f"Final Best Model from Epoch {best_epoch} saved!")

    print("done. Best validation loss:", best_val_loss)



if __name__ == "__main__":
    main()
