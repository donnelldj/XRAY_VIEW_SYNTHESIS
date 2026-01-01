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
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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

    Encoder/decoder in 3D, then collapse Z via mean pooling, then 2D head.
    """
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1 = ConvBlock3D(1, base)
        self.down1 = nn.MaxPool3d(2)  # /2
        self.enc2 = ConvBlock3D(base, base * 2)
        self.down2 = nn.MaxPool3d(2)  # /4
        self.enc3 = ConvBlock3D(base * 2, base * 4)

        self.up1 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up2 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)

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

        d1_2d = d1.mean(dim=2)  # (B,C,Y,X)
        return self.head2d(d1_2d)


# ---------------- NPZ Dataset ----------------

VOL_KEYS = ["vol_bp", "bp_vol", "vbp", "bp", "vol"]
LAT_KEYS = ["lat", "lat_gt", "drr_lat", "lat_drr", "lat_img"]
AP_KEYS  = ["ap", "ap_img", "drr_ap", "ap_drr", "ap0"]

def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + 1e-8)

def _find_first_key(d: np.lib.npyio.NpzFile, keys: list[str]) -> str | None:
    for k in keys:
        if k in d.files:
            return k
    return None

def _center_crop_3d(vol_zyx: np.ndarray, crop_zyx: tuple[int,int,int]) -> np.ndarray:
    z, y, x = vol_zyx.shape
    cz, cy, cx = crop_zyx
    if (z, y, x) == (cz, cy, cx):
        return vol_zyx
    if z < cz or y < cy or x < cx:
        raise ValueError(f"Volume smaller than crop. vol={vol_zyx.shape} crop={crop_zyx}")
    z0 = (z - cz) // 2
    y0 = (y - cy) // 2
    x0 = (x - cx) // 2
    return vol_zyx[z0:z0+cz, y0:y0+cy, x0:x0+cx]

def _center_crop_2d(img: np.ndarray, crop_yx: tuple[int,int]) -> np.ndarray:
    y, x = img.shape
    cy, cx = crop_yx
    if (y, x) == (cy, cx):
        return img
    if y < cy or x < cx:
        raise ValueError(f"Image smaller than crop. img={img.shape} crop={crop_yx}")
    y0 = (y - cy) // 2
    x0 = (x - cx) // 2
    return img[y0:y0+cy, x0:x0+cx]

@dataclass
class SampleMeta:
    path: str

class BackprojectNPZDataset(Dataset):
    """
    Expects each npz to contain:
      - a backprojected volume in Z,Y,X (one of VOL_KEYS)
      - a GT lateral DRR in Y,X or Z,Y depending on how you stored it (we handle both)
      - optionally AP for display (one of AP_KEYS). If missing, we take vol[0] as AP display.
    """
    def __init__(self, npz_paths: list[Path], crop_zyx: tuple[int,int,int], normalize_mode: str = "minmax01"):
        self.paths = list(npz_paths)
        self.crop_zyx = crop_zyx
        self.normalize_mode = normalize_mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        d = np.load(str(p), allow_pickle=True)

        k_vol = _find_first_key(d, VOL_KEYS)
        k_lat = _find_first_key(d, LAT_KEYS)
        k_ap  = _find_first_key(d, AP_KEYS)

        if k_vol is None:
            raise KeyError(f"No volume key found in {p.name}. Have keys={d.files}")
        if k_lat is None:
            raise KeyError(f"No lateral key found in {p.name}. Have keys={d.files}")

        vol = d[k_vol].astype(np.float32)   # expected (Z,Y,X)
        lat = d[k_lat].astype(np.float32)

        # Normalize
        if self.normalize_mode == "minmax01":
            vol = _normalize01(vol)
            lat = _normalize01(lat)
        else:
            raise ValueError(f"Unknown normalize_mode={self.normalize_mode}")

        # Crop volume
        vol = _center_crop_3d(vol, self.crop_zyx)  # (Z,Y,X)

        # Ensure lat is (Y,X). Some pipelines store lat as (Z,Y) for "integrate along X".
        # If your saved lat is (Z,Y), we center-crop to (crop_z,crop_y) then resize/collapse to (crop_y,crop_x) is ambiguous.
        # For this trainer we assume your lat_gt is already a square (Y,X) = (crop_y,crop_x).
        if lat.ndim != 2:
            raise ValueError(f"Expected 2D lat, got {lat.shape} in {p.name}")

        lat = _center_crop_2d(lat, (self.crop_zyx[1], self.crop_zyx[2]))  # (Y,X)

        # AP for display
        if k_ap is not None:
            ap = d[k_ap].astype(np.float32)
            if self.normalize_mode == "minmax01":
                ap = _normalize01(ap)
            if ap.ndim == 2:
                ap_disp = _center_crop_2d(ap, (self.crop_zyx[1], self.crop_zyx[2]))
            else:
                ap_disp = vol[0]
        else:
            ap_disp = vol[0]

        # Torch tensors
        xb = torch.from_numpy(vol[None, ...])         # (1,Z,Y,X)
        yb = torch.from_numpy(lat[None, ...])         # (1,Y,X)
        ap = torch.from_numpy(ap_disp.astype(np.float32))  # (Y,X) display only

        return xb, yb, ap, SampleMeta(path=str(p))


# ---------------- Utils ----------------

def to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)

def save_grid(out_path: Path, aps: list[np.ndarray], preds: list[np.ndarray], gts: list[np.ndarray]):
    strips = []
    for ap, pr, gt in zip(aps, preds, gts):
        strips.append(np.concatenate([ap, pr, gt], axis=1))
    grid = np.concatenate(strips, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(grid), mode="L").save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_dir", type=str, required=True, help="Folder containing .npz files")
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--crop_z", type=int, default=256)
    p.add_argument("--crop_y", type=int, default=256)
    p.add_argument("--crop_x", type=int, default=256)

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--out_dir", type=str, default="runs/bp_to_lat_npz")
    p.add_argument("--export_n", type=int, default=10)
    args = p.parse_args()

    npz_dir = Path(args.npz_dir)
    assert npz_dir.exists(), f"npz_dir not found: {npz_dir}"

    all_npz = sorted(npz_dir.glob("*.npz"))
    if len(all_npz) == 0:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(all_npz)

    n_val = max(1, int(round(len(all_npz) * args.val_frac)))
    val_npz = all_npz[:n_val]
    train_npz = all_npz[n_val:]

    crop_zyx = (args.crop_z, args.crop_y, args.crop_x)

    train_ds = BackprojectNPZDataset(train_npz, crop_zyx=crop_zyx, normalize_mode="minmax01")
    val_ds   = BackprojectNPZDataset(val_npz,   crop_zyx=crop_zyx, normalize_mode="minmax01")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Small3DUNetToLat(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    print(f"[train_bp_to_lat_npz] device={device}")
    print(f"[train_bp_to_lat_npz] npz_dir={npz_dir}")
    print(f"[train_bp_to_lat_npz] train={len(train_ds)} val={len(val_ds)} val_frac={args.val_frac}")
    print(f"[train_bp_to_lat_npz] crop(Z,Y,X)={crop_zyx} epochs={args.epochs} batch={args.batch} lr={args.lr}")
    print(f"[train_bp_to_lat_npz] out_dir={out_dir}")

    best_val = -1.0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = 0.0

        for xb, yb, _ap, _meta in train_loader:
            xb = xb.to(device)  # (B,1,Z,Y,X) after we add channel below
            yb = yb.to(device)  # (B,1,Y,X)

            # Dataset returns xb as (B,1,Z,Y,X)? Actually xb is (1,Z,Y,X) per item.
            # DataLoader stacks -> (B,1,Z,Y,X). Good.
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
            for xb, yb, _ap, _meta in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pr = model(xb).clamp(0, 1)
                pr_np = pr[0, 0].detach().cpu().numpy()
                gt_np = yb[0, 0].detach().cpu().numpy()

                psnrs.append(psnr(pr_np, gt_np, data_range=1.0))
                ssims.append(ssim_simple(pr_np, gt_np, data_range=1.0))

        m_psnr = float(np.mean(psnrs))
        m_ssim = float(np.mean(ssims))
        dt = time.time() - t0

        print(f"[ep {ep:02d}] train_l1={train_loss:.4f}  val_psnr={m_psnr:.2f}  val_ssim={m_ssim:.3f}  ({dt:.1f}s)")

        score = m_psnr + 100.0 * m_ssim
        if score > best_val:
            best_val = score
            ckpt = out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
            print(f"  [saved] {ckpt}")

    # Export N examples grid from val
    model.eval()
    aps, preds, gts = [], [], []
    with torch.no_grad():
        for i in range(min(args.export_n, len(val_ds))):
            xb, yb, ap_disp, meta = val_ds[i]
            xb = xb[None, ...].to(device)  # (1,1,Z,Y,X)

            pr = model(xb).clamp(0, 1)[0, 0].cpu().numpy()
            gt = yb[0].numpy()
            ap = ap_disp.numpy()

            aps.append(ap)
            preds.append(pr)
            gts.append(gt)

    grid_path = out_dir / "examples_AP_PRED_GT.png"
    save_grid(grid_path, aps, preds, gts)
    print(f"[export] {grid_path}")

    # Save a simple metrics summary too
    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"npz_dir: {npz_dir}\n")
        f.write(f"train: {len(train_ds)} val: {len(val_ds)}\n")
        f.write(f"crop_zyx: {crop_zyx}\n")
        f.write(f"epochs: {args.epochs} batch: {args.batch} lr: {args.lr} base: {args.base}\n")
        f.write(f"best_score(psnr+100*ssim): {best_val:.4f}\n")
    print(f"[write] {metrics_path}")


if __name__ == "__main__":
    main()
