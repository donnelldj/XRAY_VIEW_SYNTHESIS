from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Dataset
# -------------------------
class DRRPairDataset(Dataset):
    """
    Loads NPZs from a CSV that contains at least: npz_path
    NPZ must contain: ap, lat (both 2D float arrays, already sized consistently)
    """

    def __init__(self, csv_path: Path, normalize: str = "minmax01"):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        if "npz_path" not in self.df.columns:
            raise ValueError(f"CSV missing required column 'npz_path': {self.csv_path}")
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _minmax01(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mn = float(x.min())
        mx = float(x.max())
        if mx - mn < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)

    @staticmethod
    def _meanstd(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mu = float(x.mean())
        sd = float(x.std())
        if sd < 1e-8:
            return x - mu
        return (x - mu) / sd

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return x.astype(np.float32)
        if self.normalize == "minmax01":
            return self._minmax01(x)
        if self.normalize == "meanstd":
            return self._meanstd(x)
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = Path(str(self.df.iloc[idx]["npz_path"]))
        if not p.exists():
            raise FileNotFoundError(f"Missing npz: {p}")

        d = np.load(p, allow_pickle=True)
        ap = d["ap"].astype(np.float32)    # (H,W) but your convention may be (Z,X)
        lat = d["lat"].astype(np.float32)  # (H,W) but your convention may be (Z,Y)

        ap = self._normalize(ap)
        lat = self._normalize(lat)

        # add channel dim -> (1,H,W)
        ap = torch.from_numpy(ap)[None, ...]
        lat = torch.from_numpy(lat)[None, ...]
        return ap, lat


# -------------------------
# Model
# -------------------------
class SmallUNet2D(nn.Module):
    # Tiny UNet-like model: fast baseline
    def __init__(self, c_in: int = 1, c_out: int = 1, base: int = 32):
        super().__init__()

        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.SiLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.SiLU(),
            )

        self.enc1 = C(c_in, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = C(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bott = C(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = C(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = C(base * 2, base)

        self.out = nn.Conv2d(base, c_out, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bott(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


# -------------------------
# Train
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out_dir", type=str, default="runs/ap2lat")

    ap.add_argument("--normalize", type=str, default="minmax01", choices=["none", "minmax01", "meanstd"])
    ap.add_argument("--loss", type=str, default="l1", choices=["l1", "mse"])
    ap.add_argument("--base", type=int, default=32)

    args = ap.parse_args()

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"missing {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"missing {val_csv}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = DRRPairDataset(train_csv, normalize=args.normalize)
    val_ds = DRRPairDataset(val_csv, normalize=args.normalize)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SmallUNet2D(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.loss == "l1":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0

        for ap_img, lat_img in train_loader:
            ap_img = ap_img.to(device, non_blocking=True)
            lat_img = lat_img.to(device, non_blocking=True)

            pred = model(ap_img)
            # keep predictions in [0,1] since targets are [0,1] (minmax01)
            pred = pred.clamp(0.0, 1.0)

            loss = loss_fn(pred, lat_img)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * ap_img.size(0)

        tr_loss /= max(1, len(train_ds))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for ap_img, lat_img in val_loader:
                ap_img = ap_img.to(device, non_blocking=True)
                lat_img = lat_img.to(device, non_blocking=True)

                pred = model(ap_img).clamp(0.0, 1.0)
                loss = loss_fn(pred, lat_img)
                va_loss += loss.item() * ap_img.size(0)

        va_loss /= max(1, len(val_ds))
        print(f"epoch {epoch:03d} | train {tr_loss:.5f} | val {va_loss:.5f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "val_loss": va_loss,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, out_dir / "best.pt")
            print("  saved best.pt")

    print("done. best_val:", best_val)


if __name__ == "__main__":
    main()
