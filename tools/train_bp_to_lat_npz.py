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
    Output: (B,1,Y,X)
    """
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1 = ConvBlock3D(1, base)
        self.down1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.down2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)

        self.up1 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up2 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)

        self.head2d = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        u2 = self.up1(e3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up2(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        d1_2d = d1.mean(dim=2)
        return self.head2d(d1_2d)


# ---------------- Dataset ----------------

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

def _center_crop_2d(img: np.ndarray, crop_hw: tuple[int, int]) -> np.ndarray:
    h, w = img.shape
    ch, cw = crop_hw
    if (h, w) == (ch, cw):
        return img
    if h < ch or w < cw:
        raise ValueError(f"Image smaller than crop. img={img.shape} crop={crop_hw}")
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img[y0:y0+ch, x0:x0+cw]

class BackprojectFromAPNPZDataset(Dataset):
    """
    NPZ has ap, lat.
    ap: (Z,X). Build Vbp: (Z,Y,X) by repeating along Y.
    lat: either (Z,Y) (then transpose -> (Y,Z) and crop to (Y,X)) OR already (Y,X).
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

        k_ap  = _find_first_key(d, AP_KEYS)
        k_lat = _find_first_key(d, LAT_KEYS)

        if k_ap is None:
            raise KeyError(f"No AP key found in {p.name}. Have keys={d.files}")
        if k_lat is None:
            raise KeyError(f"No LAT key found in {p.name}. Have keys={d.files}")

        ap = d[k_ap].astype(np.float32)      # (Z,X)
        lat_raw = d[k_lat].astype(np.float32)

        if self.normalize_mode == "minmax01":
            ap = _normalize01(ap)
            lat_raw = _normalize01(lat_raw)
        else:
            raise ValueError(f"Unknown normalize_mode={self.normalize_mode}")

        cz, cy, cx = self.crop_zyx

        if ap.ndim != 2:
            raise ValueError(f"Expected 2D ap, got {ap.shape} in {p.name}")
        ap = _center_crop_2d(ap, (cz, cx))  # (Z,X)

        # Vbp: (Z,Y,X)
        vol = np.repeat(ap[:, None, :], repeats=cy, axis=1).astype(np.float32)

        if lat_raw.ndim != 2:
            raise ValueError(f"Expected 2D lat, got {lat_raw.shape} in {p.name}")

        # detect lat mode
        lat_mode = "YX"
        if lat_raw.shape == (cz, cy):
            lat_yx = lat_raw.T  # (Y,Z)
            lat_mode = "ZY->YX"
        else:
            lat_yx = lat_raw

        lat = _center_crop_2d(lat_yx, (cy, cx)).astype(np.float32)  # (Y,X)

        # AP display: use mid-Z row and repeat along Y
        ap_mid = ap[cz // 2]  # (X,)
        ap_disp = np.repeat(ap_mid[None, :], repeats=cy, axis=0).astype(np.float32)  # (Y,X)

        xb = torch.from_numpy(vol[None, ...])  # (1,Z,Y,X)
        yb = torch.from_numpy(lat[None, ...])  # (1,Y,X)
        apd = torch.from_numpy(ap_disp)

        meta = {
            "path": str(p),
            "ap_shape": tuple(d[k_ap].shape),
            "lat_shape": tuple(d[k_lat].shape),
            "lat_mode": lat_mode,
        }
        return xb, yb, apd, meta


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
    p.add_argument("--npz_dir", type=str, required=True)
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

    train_ds = BackprojectFromAPNPZDataset(train_npz, crop_zyx=crop_zyx)
    val_ds   = BackprojectFromAPNPZDataset(val_npz,   crop_zyx=crop_zyx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_bp_to_lat_npz] device={device}")
    print(f"[train_bp_to_lat_npz] npz_dir={npz_dir}")
    print(f"[train_bp_to_lat_npz] train={len(train_ds)} val={len(val_ds)} val_frac={args.val_frac}")
    print(f"[train_bp_to_lat_npz] crop(Z,Y,X)={crop_zyx} epochs={args.epochs} batch={args.batch} lr={args.lr}")
    print(f"[train_bp_to_lat_npz] out_dir={out_dir}")

    xb0, yb0, ap0, meta0 = train_ds[0]
    print(f"[sample0] ap_npz_shape={meta0['ap_shape']} lat_npz_shape={meta0['lat_shape']} lat_mode={meta0['lat_mode']}")
    print(f"[sample0] xb(vol)={tuple(xb0.shape)} yb(lat)={tuple(yb0.shape)} ap_disp={tuple(ap0.shape)}")

    model = Small3DUNetToLat(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    best_val = -1.0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = 0.0

        for xb, yb, _ap, _meta in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss.item())

        train_loss = run_loss / max(1, len(train_loader))

        model.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for xb, yb, _ap, _meta in val_loader:
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

        score = m_psnr + 100.0 * m_ssim
        if score > best_val:
            best_val = score
            ckpt = out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
            print(f"  [saved] {ckpt}")

    model.eval()
    aps, preds, gts = [], [], []
    with torch.no_grad():
        for i in range(min(args.export_n, len(val_ds))):
            xb, yb, ap_disp, _meta = val_ds[i]
            xb = xb[None, ...].to(device)

            pr = model(xb).clamp(0, 1)[0, 0].cpu().numpy()
            gt = yb[0].numpy()
            ap = ap_disp.numpy()

            aps.append(ap)
            preds.append(pr)
            gts.append(gt)

    grid_path = out_dir / "examples_AP_PRED_GT.png"
    save_grid(grid_path, aps, preds, gts)
    print(f"[export] {grid_path}")

    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"npz_dir: {npz_dir}\n")
        f.write(f"train: {len(train_ds)} val: {len(val_ds)} val_frac: {args.val_frac}\n")
        f.write(f"crop_zyx: {crop_zyx}\n")
        f.write(f"epochs: {args.epochs} batch: {args.batch} lr: {args.lr} base: {args.base}\n")
        f.write(f"best_score(psnr+100*ssim): {best_val:.4f}\n")
        f.write("note: Vbp synthesized from AP by replicating along Y.\n")
    print(f"[write] {metrics_path}")


if __name__ == "__main__":
    main()
