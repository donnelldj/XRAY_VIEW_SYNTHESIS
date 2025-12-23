from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DRRPairDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        p = self.df.iloc[idx]["npz_path"]
        d = np.load(p, allow_pickle=True)
        ap = d["ap"].astype(np.float32)   # (H,W)
        lat = d["lat"].astype(np.float32) # (H,W)

        # add channel dim -> (1,H,W)
        ap = torch.from_numpy(ap)[None, ...]
        lat = torch.from_numpy(lat)[None, ...]
        return ap, lat


class SmallUNet2D(nn.Module):
    # Tiny UNet-like model: fast + good baseline
    def __init__(self, c_in=1, c_out=1, base=32):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/drp_pairs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default="runs/ap2lat")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    assert train_csv.exists(), f"missing {train_csv}"
    assert val_csv.exists(), f"missing {val_csv}"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = DRRPairDataset(train_csv)
    val_ds = DRRPairDataset(val_csv)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = SmallUNet2D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

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
            loss = loss_fn(pred, lat_img)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * ap_img.size(0)

        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for ap_img, lat_img in val_loader:
                ap_img = ap_img.to(device, non_blocking=True)
                lat_img = lat_img.to(device, non_blocking=True)
                pred = model(ap_img)
                loss = loss_fn(pred, lat_img)
                va_loss += loss.item() * ap_img.size(0)

        va_loss /= len(val_ds)
        print(f"epoch {epoch:03d} | train {tr_loss:.5f} | val {va_loss:.5f}")

        # save checkpoints
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "val_loss": va_loss,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, out_dir / "best.pt")
            print("  saved best.pt")

    print("done. best_val:", best_val)


if __name__ == "__main__":
    main()
