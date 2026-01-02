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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import SimpleITK as sitk

from src.geo.backproject import backproject_parallel_beam


# ---------------- Metrics ----------------

def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse < 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim_simple(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """
    Lightweight global SSIM (not windowed). Dependency-free.
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


# ---------------- Utils ----------------

def _normalize(img: np.ndarray, mode: str) -> np.ndarray:
    img = img.astype(np.float32)
    if mode == "none":
        return img
    if mode == "minmax01":
        mn = float(img.min())
        mx = float(img.max())
        if mx - mn < 1e-8:
            return np.zeros_like(img, dtype=np.float32)
        return (img - mn) / (mx - mn)
    if mode == "meanstd":
        mu = float(img.mean())
        sd = float(img.std())
        if sd < 1e-8:
            return img - mu
        return (img - mu) / sd
    raise ValueError(f"Unknown normalize_mode: {mode}")


def center_crop_or_pad_zyx(vol: np.ndarray, out_zyx: Tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    """
    Center crop or pad a (Z,Y,X) volume to out_zyx.
    No resampling; assumes roughly comparable spacing.
    """
    z, y, x = out_zyx
    vz, vy, vx = vol.shape

    # pad if needed
    pad_z0 = max((z - vz) // 2, 0)
    pad_z1 = max(z - vz - pad_z0, 0)
    pad_y0 = max((y - vy) // 2, 0)
    pad_y1 = max(y - vy - pad_y0, 0)
    pad_x0 = max((x - vx) // 2, 0)
    pad_x1 = max(x - vx - pad_x0, 0)

    if any(p > 0 for p in [pad_z0, pad_z1, pad_y0, pad_y1, pad_x0, pad_x1]):
        vol = np.pad(
            vol,
            ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)),
            mode="constant",
            constant_values=float(pad_value),
        )

    # crop if needed
    vz, vy, vx = vol.shape
    z0 = max((vz - z) // 2, 0)
    y0 = max((vy - y) // 2, 0)
    x0 = max((vx - x) // 2, 0)
    vol = vol[z0:z0 + z, y0:y0 + y, x0:x0 + x]
    return vol.astype(np.float32)


def forward_project_lat(ct_zyx: np.ndarray) -> np.ndarray:
    """
    Simple DRR-style forward projection for LAT:
    integrate along X -> (Z,Y)
    """
    return ct_zyx.astype(np.float32).sum(axis=2)


def to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


def save_grid(out_path: Path, aps: List[np.ndarray], preds: List[np.ndarray], gts: List[np.ndarray]) -> None:
    """
    stacks rows: each row is [AP | PRED | GT]
    expects each image already (H,W) float in 0..1
    """
    strips = []
    for ap, pr, gt in zip(aps, preds, gts):
        strips.append(np.concatenate([ap, pr, gt], axis=1))
    grid = np.concatenate(strips, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(grid), mode="L").save(out_path)


# ---------------- Models ----------------

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentPredictor3D(nn.Module):
    """
    Predicts a compressed 'latent' volume from Vbp.
    Output shape is downsampled by factor `down`.
    """
    def __init__(self, base: int = 16, down: int = 8):
        super().__init__()
        assert down in (2, 4, 8, 16), "down should be a power-of-two like 4 or 8"

        self.down = down

        self.enc1 = ConvBlock3D(1, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)

        if down >= 8:
            self.pool3 = nn.MaxPool3d(2)
            self.enc4 = ConvBlock3D(base * 4, base * 4)
        else:
            self.pool3 = None
            self.enc4 = None

        if down >= 16:
            self.pool4 = nn.MaxPool3d(2)
            self.enc5 = ConvBlock3D(base * 4, base * 4)
        else:
            self.pool4 = None
            self.enc5 = None

        self.out = nn.Conv3d(base * 4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(self.pool1(x))
        x = self.enc3(self.pool2(x))
        if self.pool3 is not None:
            x = self.enc4(self.pool3(x))
        if self.pool4 is not None:
            x = self.enc5(self.pool4(x))
        # latent in 0..1 (keeps things stable)
        return torch.sigmoid(self.out(x))


class SimpleVQStyleDecoder3D(nn.Module):
    """
    A learnable decoder mapping compressed latent -> CT volume (pixel space).
    This is the "decode z -> CT" step in the paper (they use a VQGAN decoder).
    Here: upsample+conv blocks (fast, stable, trains with small data).
    """
    def __init__(self, base: int = 32, up: int = 8):
        super().__init__()
        assert up in (2, 4, 8, 16), "up should match latent down factor"

        self.up = up
        c = base

        self.in_conv = nn.Sequential(
            nn.Conv3d(1, c, 3, padding=1),
            nn.InstanceNorm3d(c),
            nn.ReLU(inplace=True),
        )

        blocks = []
        # each stage doubles spatial dims
        stages = int(round(math.log2(up)))
        for _ in range(stages):
            blocks += [
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(c, c, 3, padding=1),
                nn.InstanceNorm3d(c),
                nn.ReLU(inplace=True),
                nn.Conv3d(c, c, 3, padding=1),
                nn.InstanceNorm3d(c),
                nn.ReLU(inplace=True),
            ]
        self.blocks = nn.Sequential(*blocks)

        self.out = nn.Sequential(
            nn.Conv3d(c, 1, 1),
            nn.Sigmoid(),  # CT assumed normalized 0..1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(z)
        x = self.blocks(x)
        return self.out(x)


# ---------------- Dataset ----------------

@dataclass
class Meta:
    case_id: str
    npz_path: str
    mhd_path: str


class End2EndDataset(Dataset):
    """
    Uses CSV with 'npz_path' column (like your existing manifests).

    Loads:
      - ap from npz['ap']
      - lat from npz['lat']

    For CT + latent:
      - If npz contains CT key, use it
      - Else load CT from npz['mhd_path'] (SimpleITK) if enabled
      - If npz contains latent key, use it
      - Else create latent_gt by avgpool downsampling CT (acts as compressed latent target)

    Returns:
      vbp:  (1,Z,Y,X)
      lat:  (1,H,W)   (we treat whatever 2D array as HxW after normalize)
      z_gt: (1,z',y',x')  (latent gt)
      ct_gt:(1,Z,Y,X)     (ct gt in 0..1)
      meta: dict (collate-safe)
    """

    CT_KEYS = ("ct", "ct_zyx", "ct_vol", "ct_gt")
    LATENT_KEYS = ("ct_latent", "latent", "z", "z_gt", "ct_latent_gt")

    def __init__(
        self,
        csv_path: str | Path,
        crop_zyx: Tuple[int, int, int],
        bp_axis: int,
        ap_mode: str,
        normalize_mode: str,
        latent_down: int,
        load_ct_from_mhd: bool = True,
        hu_clip: Tuple[float, float] = (-1000.0, 400.0),
    ):
        import pandas as pd

        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if "npz_path" not in self.df.columns:
            raise ValueError(f"Expected 'npz_path' column in {self.csv_path}")

        self.crop_zyx = tuple(map(int, crop_zyx))
        self.bp_axis = int(bp_axis)
        self.ap_mode = str(ap_mode).upper()
        self.normalize_mode = normalize_mode
        self.latent_down = int(latent_down)
        self.load_ct_from_mhd = bool(load_ct_from_mhd)
        self.hu_clip = hu_clip

    def __len__(self) -> int:
        return len(self.df)

    def _load_ct_from_mhd(self, mhd_path: str) -> np.ndarray:
        p = Path(mhd_path)
        if not p.exists():
            raise FileNotFoundError(f"CT mhd not found: {p}")

        img = sitk.ReadImage(str(p))
        ct = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)

        # HU clip then normalize to 0..1
        lo, hi = map(float, self.hu_clip)
        ct = np.clip(ct, lo, hi)
        ct = (ct - lo) / (hi - lo + 1e-8)
        return ct.astype(np.float32)

    def _find_key(self, npz: np.lib.npyio.NpzFile, keys: Tuple[str, ...]) -> Optional[str]:
        for k in keys:
            if k in npz.files:
                return k
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npz_path = Path(str(row["npz_path"]))
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing npz: {npz_path}")

        npz = np.load(npz_path, allow_pickle=True)

        if "ap" not in npz.files or "lat" not in npz.files:
            raise KeyError(f"NPZ missing ap/lat: {npz_path.name}. Have keys={npz.files}")

        ap = npz["ap"].astype(np.float32)
        lat = npz["lat"].astype(np.float32)

        # normalize 2D views
        ap = _normalize(ap, self.normalize_mode)
        lat = _normalize(lat, self.normalize_mode)

        # build Vbp from AP
        Z, Y, X = self.crop_zyx

        # AP conventions:
        # - ap_mode="YX" expects ap shape (Y,X) and bp_axis=0
        # - ap_mode="ZX" expects ap shape (Z,X) and bp_axis=1
        if self.ap_mode == "YX":
            # ensure correct shape (best effort)
            if ap.shape != (Y, X):
                # center crop/pad to (Y,X)
                ap2 = center_crop_or_pad_zyx(ap[None, ...], (1, Y, X), pad_value=0.0)[0]
                ap = ap2.astype(np.float32)
        elif self.ap_mode == "ZX":
            if ap.shape != (Z, X):
                ap2 = center_crop_or_pad_zyx(ap[None, ...], (1, Z, X), pad_value=0.0)[0]
                ap = ap2.astype(np.float32)
        else:
            raise ValueError(f"--ap_mode must be YX or ZX, got {self.ap_mode}")

        vbp = backproject_parallel_beam(ap, out_zyx=self.crop_zyx, axis=self.bp_axis)
        vbp = _normalize(vbp, self.normalize_mode)

        # CT GT
        ct_key = self._find_key(npz, self.CT_KEYS)
        if ct_key is not None:
            ct = npz[ct_key].astype(np.float32)
            # best effort crop/pad to crop_zyx
            if ct.shape != self.crop_zyx:
                ct = center_crop_or_pad_zyx(ct, self.crop_zyx, pad_value=0.0)
            ct = _normalize(ct, "minmax01")  # keep CT in 0..1
        else:
            # load from mhd_path
            if not self.load_ct_from_mhd:
                raise KeyError(
                    f"Missing CT in {npz_path.name} (keys={npz.files}) and --load_ct_from_mhd is false."
                )
            if "mhd_path" not in npz.files:
                # sometimes stored in CSV
                mhd_path = str(row.get("mhd_path", ""))
            else:
                mhd_path = str(npz["mhd_path"])
            if not mhd_path:
                raise KeyError(
                    f"No CT key in npz AND no mhd_path available (npz keys={npz.files}; csv cols={list(self.df.columns)})"
                )
            ct = self._load_ct_from_mhd(mhd_path)
            ct = center_crop_or_pad_zyx(ct, self.crop_zyx, pad_value=0.0)

        # latent GT
        z_key = self._find_key(npz, self.LATENT_KEYS)
        if z_key is not None:
            zgt = npz[z_key].astype(np.float32)
            zgt = _normalize(zgt, "minmax01")
        else:
            # Create a "latent" target by downsampling CT (proxy for compressed latent).
            # This is the practical fallback when latent isn't stored in your NPZ.
            d = self.latent_down
            # ensure divisible by d via adaptive pooling
            zt = torch.from_numpy(ct)[None, None, ...]  # (1,1,Z,Y,X)
            zt = F.avg_pool3d(zt, kernel_size=d, stride=d)
            zgt = zt[0, 0].numpy().astype(np.float32)  # (Z/d, Y/d, X/d)

        meta = {
            "case_id": str(npz["case_id"]) if "case_id" in npz.files else str(row.get("case_id", "unknown")),
            "npz_path": str(npz_path),
            "mhd_path": str(npz["mhd_path"]) if "mhd_path" in npz.files else str(row.get("mhd_path", "")),
            "npz_keys": ",".join(list(npz.files)),
        }

        # tensors
        xb = torch.from_numpy(vbp)[None, ...]          # (1,Z,Y,X)
        ylat = torch.from_numpy(lat)[None, ...]        # (1,H,W)
        zgt_t = torch.from_numpy(zgt)[None, ...]       # (1,z',y',x')
        ct_t = torch.from_numpy(ct)[None, ...]         # (1,Z,Y,X)

        # AP for display (always 2D)
        ap_disp = ap.astype(np.float32)

        return xb, ylat, zgt_t, ct_t, ap_disp, meta


# ---------------- Train / Eval ----------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)

    p.add_argument("--crop_z", type=int, default=256)
    p.add_argument("--crop_y", type=int, default=256)
    p.add_argument("--crop_x", type=int, default=256)

    # BP / AP conventions
    p.add_argument("--ap_mode", type=str, default="ZX", choices=["YX", "ZX"])
    p.add_argument("--bp_axis", type=int, default=1, choices=[0, 1])

    # latent compression
    p.add_argument("--latent_down", type=int, default=8, choices=[2, 4, 8, 16])

    # models
    p.add_argument("--latent_base", type=int, default=16)
    p.add_argument("--decoder_base", type=int, default=32)

    # training
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--export_n", type=int, default=10)

    # data / normalization
    p.add_argument("--normalize_mode", type=str, default="minmax01", choices=["none", "minmax01", "meanstd"])
    p.add_argument("--load_ct_from_mhd", action="store_true")
    p.add_argument("--no_load_ct_from_mhd", dest="load_ct_from_mhd", action="store_false")
    p.set_defaults(load_ct_from_mhd=True)
    p.add_argument("--hu_lo", type=float, default=-1000.0)
    p.add_argument("--hu_hi", type=float, default=400.0)

    # weights for losses
    p.add_argument("--w_latent", type=float, default=1.0)
    p.add_argument("--w_ct", type=float, default=1.0)
    p.add_argument("--w_lat", type=float, default=1.0)

    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_zyx = (args.crop_z, args.crop_y, args.crop_x)

    train_ds = End2EndDataset(
        args.train_csv,
        crop_zyx=crop_zyx,
        bp_axis=args.bp_axis,
        ap_mode=args.ap_mode,
        normalize_mode=args.normalize_mode,
        latent_down=args.latent_down,
        load_ct_from_mhd=args.load_ct_from_mhd,
        hu_clip=(args.hu_lo, args.hu_hi),
    )
    val_ds = End2EndDataset(
        args.val_csv,
        crop_zyx=crop_zyx,
        bp_axis=args.bp_axis,
        ap_mode=args.ap_mode,
        normalize_mode=args.normalize_mode,
        latent_down=args.latent_down,
        load_ct_from_mhd=args.load_ct_from_mhd,
        hu_clip=(args.hu_lo, args.hu_hi),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # models
    latent_model = LatentPredictor3D(base=args.latent_base, down=args.latent_down).to(device)
    decoder = SimpleVQStyleDecoder3D(base=args.decoder_base, up=args.latent_down).to(device)

    # optim
    params = list(latent_model.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    l1 = nn.L1Loss()

    print(f"[end2end] device={device}")
    print(f"[end2end] crop_zyx={crop_zyx} latent_down={args.latent_down} ap_mode={args.ap_mode} bp_axis={args.bp_axis}")
    print(f"[end2end] epochs={args.epochs} batch={args.batch} lr={args.lr}")
    print(f"[end2end] out_dir={out_dir}")
    print(f"[end2end] train={len(train_ds)} val={len(val_ds)}")
    print(f"[end2end] decoder=SimpleVQStyleDecoder3D (learned decode step)")

    # quick sample print
    xb0, ylat0, zgt0, ctgt0, ap0, meta0 = train_ds[0]
    print(f"[sample0] ap_disp={tuple(ap0.shape)} vbp={tuple(xb0.shape)} lat={tuple(ylat0.shape)} z_gt={tuple(zgt0.shape)} ct_gt={tuple(ctgt0.shape)}")
    print(f"[sample0] keys={meta0.get('npz_keys','')}")

    best_val = -1e9

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        latent_model.train()
        decoder.train()

        run_loss = 0.0

        for xb, ylat, zgt, ctgt, _ap, _meta in train_loader:
            xb = xb.to(device)         # (B,1,Z,Y,X)
            ylat = ylat.to(device)     # (B,1,H,W)
            zgt = zgt.to(device)       # (B,1,z',y',x')
            ctgt = ctgt.to(device)     # (B,1,Z,Y,X)

            z_pred = latent_model(xb)  # (B,1,z',y',x') (should match zgt dims)
            # If shapes don't match (edge cases), align to gt via interpolate
            if z_pred.shape[-3:] != zgt.shape[-3:]:
                z_pred = F.interpolate(z_pred, size=zgt.shape[-3:], mode="trilinear", align_corners=False)

            ct_pred = decoder(z_pred)  # (B,1,Z,Y,X)
            if ct_pred.shape[-3:] != ctgt.shape[-3:]:
                ct_pred = F.interpolate(ct_pred, size=ctgt.shape[-3:], mode="trilinear", align_corners=False)

            # forward project -> LAT
            # lat_pred_zy = sum over X => (B,1,Z,Y)
            lat_pred_zy = ct_pred.sum(dim=4)

            # make it comparable to ylat (B,1,H,W)
            # In your current pipeline you often treat (Z,Y) as (H,W) and keep 256x256.
            lat_pred = lat_pred_zy

            # If target lat is (Y,X) but pred is (Z,Y), both are 256x256 so it still trains.
            # If shapes differ, align.
            if lat_pred.shape[-2:] != ylat.shape[-2:]:
                lat_pred = F.interpolate(lat_pred, size=ylat.shape[-2:], mode="bilinear", align_corners=False)

            loss_latent = l1(z_pred, zgt)
            loss_ct = l1(ct_pred, ctgt)
            loss_lat = l1(lat_pred, ylat)

            loss = args.w_latent * loss_latent + args.w_ct * loss_ct + args.w_lat * loss_lat

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            run_loss += float(loss.item())

        train_loss = run_loss / max(1, len(train_loader))

        # ---- val metrics on LAT ----
        latent_model.eval()
        decoder.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for xb, ylat, zgt, ctgt, _ap, _meta in val_loader:
                xb = xb.to(device)
                ylat = ylat.to(device)
                zgt = zgt.to(device)
                ctgt = ctgt.to(device)

                z_pred = latent_model(xb)
                if z_pred.shape[-3:] != zgt.shape[-3:]:
                    z_pred = F.interpolate(z_pred, size=zgt.shape[-3:], mode="trilinear", align_corners=False)

                ct_pred = decoder(z_pred)
                if ct_pred.shape[-3:] != ctgt.shape[-3:]:
                    ct_pred = F.interpolate(ct_pred, size=ctgt.shape[-3:], mode="trilinear", align_corners=False)

                lat_pred_zy = ct_pred.sum(dim=4)  # (1,1,Z,Y)
                lat_pred = lat_pred_zy
                if lat_pred.shape[-2:] != ylat.shape[-2:]:
                    lat_pred = F.interpolate(lat_pred, size=ylat.shape[-2:], mode="bilinear", align_corners=False)

                pr_np = lat_pred[0, 0].clamp(0, 1).cpu().numpy()
                gt_np = ylat[0, 0].clamp(0, 1).cpu().numpy()

                psnrs.append(psnr(pr_np, gt_np, data_range=1.0))
                ssims.append(ssim_simple(pr_np, gt_np, data_range=1.0))

        m_psnr = float(np.mean(psnrs))
        m_ssim = float(np.mean(ssims))
        dt = time.time() - t0

        print(f"[ep {ep:02d}] train_loss={train_loss:.4f}  val_psnr={m_psnr:.2f}  val_ssim={m_ssim:.3f}  ({dt:.1f}s)")

        # Save best
        score = m_psnr + 100.0 * m_ssim
        if score > best_val:
            best_val = score
            ckpt = out_dir / "best.pt"
            torch.save(
                {
                    "latent_model": latent_model.state_dict(),
                    "decoder": decoder.state_dict(),
                    "epoch": ep,
                    "args": vars(args),
                },
                ckpt,
            )
            print(f"  [saved] {ckpt}")

    # Export examples grid from val
    latent_model.eval()
    decoder.eval()
    aps, preds, gts = [], [], []
    with torch.no_grad():
        for i in range(min(args.export_n, len(val_ds))):
            xb, ylat, zgt, ctgt, ap_disp, meta = val_ds[i]

            xb_t = xb[None, ...].to(device)  # (1,1,Z,Y,X)
            z_pred = latent_model(xb_t)
            ct_pred = decoder(z_pred)
            lat_pred = ct_pred.sum(dim=4)  # (1,1,Z,Y)

            pr = lat_pred[0, 0].clamp(0, 1).cpu().numpy()
            gt = ylat[0].clamp(0, 1).numpy()
            ap_img = _normalize(ap_disp, "minmax01")

            # Align export shapes if needed
            if pr.shape != gt.shape:
                pr_t = torch.from_numpy(pr)[None, None, ...]
                pr_t = F.interpolate(pr_t, size=gt.shape, mode="bilinear", align_corners=False)
                pr = pr_t[0, 0].numpy()

            aps.append(ap_img)
            preds.append(pr)
            gts.append(gt)

    grid_path = out_dir / "examples_AP_PRED_GT.png"
    save_grid(grid_path, aps, preds, gts)
    print(f"[export] {grid_path}")


if __name__ == "__main__":
    main()
