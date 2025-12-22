from __future__ import annotations

# --- IMPORTANT: allow imports from repo root (src/) ---------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------------------------------------

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from src.data.bp_dataset import BackprojectToLatDataset


# ---------------- Metrics ----------------

def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse < 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)

def ssim_simple(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """
    Lightweight SSIM (global) to avoid extra deps.
    Not windowed-SSIM, but good enough for progress + sanity.
    """
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    mu_x = pred.mean()
    mu_y = gt.mean()
    sig_x = pred.var()
    sig_y = gt.var()
    sig_xy = ((pred - mu_x) * (gt - mu_y)).mean()

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2)
    return float(num / (den + 1e-12))


# ---------------- Model ----------------

class ConvBlock3D(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1),
            nn.InstanceNorm3d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.InstanceNorm3d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Small3DUNetToLat(nn.Module):
    """
    Input:  (B,1,Z,Y,X)
    Output: (B,1,Y,X)   (Lat prediction)

    We do encoder in 3D, then collapse Z with mean pooling, then 2D head.
    """
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1 = ConvBlock3D(1, base)
        self.down1 = nn.MaxPool3d(2)  # /2
        self.enc2 = ConvBlock3D(base, base * 2)
        self.down2 = nn.MaxPool3d(2)  # /4
        self.enc3 = ConvBlock3D(base * 2, base * 4)

        # up
        self.up1 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up2 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)

        # collapse Z -> 2D
        self.head2d = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1),
            nn.Sigmoid(),  # outputs 0..1
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        u2 = self.up1(e3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up2(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # collapse Z
        d1_2d = d1.mean(dim=2)  # (B,C,Y,X)
        return self.head2d(d1_2d)


# ---------------- Utils ----------------

def to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)

def save_grid(out_path: Path, aps: list[np.ndarray], preds: list[np.ndarray], gts: list[np.ndarray]):
    # stack rows: each row is [AP | PRED | GT]
    strips = []
    for ap, pr, gt in zip(aps, preds, gts):
        strips.append(np.concatenate([ap, pr, gt], axis=1))
    grid = np.concatenate(strips, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(grid), mode="L").save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="data/drr_pairs/train.csv")
    p.add_argument("--val_csv", type=str, default="data/drr_pairs/val.csv")
    p.add_argument("--crop_z", type=int, default=96)
    p.add_argument("--crop_y", type=int, default=256)
    p.add_argument("--crop_x", type=int, default=256)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="results/bp_to_lat")
    p.add_argument("--export_n", type=int, default=10)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_zyx = (args.crop_z, args.crop_y, args.crop_x)

    train_ds = BackprojectToLatDataset(args.train_csv, crop_zyx=crop_zyx, normalize_mode="minmax01", return_meta=False)
    val_ds = BackprojectToLatDataset(args.val_csv, crop_zyx=crop_zyx, normalize_mode="minmax01", return_meta=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = Small3DUNetToLat(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    print(f"[train_bp_to_lat] device={device} train={len(train_ds)} val={len(val_ds)}")
    print(f"[train_bp_to_lat] out_dir={out_dir} epochs={args.epochs} batch={args.batch} lr={args.lr}")

    best_val = -1.0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)  # (B,1,Z,Y,X)
            yb = yb.to(device)  # (B,1,Y,X)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            run_loss += float(loss.item())

        train_loss = run_loss / max(1, len(train_loader))

        # ---- val metrics ----
        model.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pr = model(xb).clamp(0, 1)
                pr_np = pr[0, 0].cpu().numpy()
                gt_np = yb[0, 0].cpu().numpy()

                psnrs.append(psnr(pr_np, gt_np, data_range=1.0))
                ssims.append(ssim_simple(pr_np, gt_np, data_range=1.0))

        m_psnr = float(np.mean(psnrs))
        m_ssim = float(np.mean(ssims))
        dt = time.time() - t0

        print(f"[ep {ep:02d}] train_l1={train_loss:.4f}  val_psnr={m_psnr:.2f}  val_ssim={m_ssim:.3f}  ({dt:.1f}s)")

        # Save best
        score = m_psnr + 100.0 * m_ssim
        if score > best_val:
            best_val = score
            ckpt = out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
            print(f"  [saved] {ckpt}")

    # Export 10 examples grid from val
    model.eval()
    aps, preds, gts = [], [], []
    with torch.no_grad():
        for i in range(min(args.export_n, len(val_ds))):
            xb, yb = val_ds[i]
            xb = xb[None, ...].to(device)
            pr = model(xb).clamp(0, 1)[0, 0].cpu().numpy()
            gt = yb[0].numpy()

            # AP image for display = middle slice of Vbp? Better: show original AP embedded in Vbp (same every z)
            # We can recover AP by taking first slice:
            ap_img = xb[0, 0, 0].cpu().numpy()

            aps.append(ap_img)
            preds.append(pr)
            gts.append(gt)

    grid_path = out_dir / "examples_AP_PRED_GT.png"
    save_grid(grid_path, aps, preds, gts)
    print(f"[export] {grid_path}")


if __name__ == "__main__":
    main()
