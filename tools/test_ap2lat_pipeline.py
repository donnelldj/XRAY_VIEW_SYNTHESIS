# tools/final_ap2lat_pipeline.py
from __future__ import annotations

import os
import json
import math
import time
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Utils
# -------------------------
def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def to_uint8_img(x01: np.ndarray) -> Image.Image:
    x = np.clip(x01, 0.0, 1.0)
    x8 = (x * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(x8)


def resize_hw(x: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    x: (H,W) float32
    returns (outH,outW) float32
    """
    img = to_uint8_img(normalize01(x))
    img = img.resize((out_hw[1], out_hw[0]), resample=Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def center_crop_or_pad_2d(x: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    x: (H,W)
    """
    H, W = x.shape
    outH, outW = out_hw
    y = x

    # pad if needed
    pad_top = max(0, (outH - H) // 2)
    pad_bottom = max(0, outH - H - pad_top)
    pad_left = max(0, (outW - W) // 2)
    pad_right = max(0, outW - W - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        y = np.pad(y, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=float(np.min(y)))

    # crop if needed
    H2, W2 = y.shape
    start_y = max(0, (H2 - outH) // 2)
    start_x = max(0, (W2 - outW) // 2)
    y = y[start_y:start_y + outH, start_x:start_x + outW]
    return y


def save_triplet(ap01: np.ndarray, lat_gt01: np.ndarray, lat_pred01: np.ndarray, out_path: str) -> None:
    """
    Save a 3-panel horizontal triplet image.
    """
    ap_img = to_uint8_img(ap01)
    gt_img = to_uint8_img(lat_gt01)
    pr_img = to_uint8_img(lat_pred01)

    w, h = ap_img.size
    canvas = Image.new("L", (w * 3, h))
    canvas.paste(ap_img, (0, 0))
    canvas.paste(gt_img, (w, 0))
    canvas.paste(pr_img, (w * 2, 0))
    canvas.save(out_path)


# -------------------------
# Geometry: Eq (1) Backprojection
# -------------------------
def backproject_ap_to_volume(ap01: np.ndarray, vol_dhw: Tuple[int, int, int]) -> np.ndarray:
    """
    Parallel/orthographic backprojection from a single AP view:
      Given AP image I(y,x), assign every voxel along the ray the same value.
      For AP (0°) with integration along Z, the backprojection is replication along Z.

    ap01: (H,W) normalized in [0,1]
    returns bp: (D,H,W) float32 normalized [0,1]
    """
    D, H, W = vol_dhw
    if ap01.shape != (H, W):
        raise ValueError(f"AP shape {ap01.shape} must equal target (H,W)=({H},{W})")
    bp = np.repeat(ap01[None, :, :], D, axis=0).astype(np.float32)
    return bp


# -------------------------
# Projection: Eq (9) Forward projection to LAT
# -------------------------
def forward_project_lat_from_volume(vol: torch.Tensor) -> torch.Tensor:
    """
    vol: (B,1,D,H,W) where axes are (Z,Y,X) = (D,H,W)
    LAT DRR for our setup: integrate along X -> output (B,1,D,H) which we treat as (H,W) after mapping.
    We'll output (B,1,H,W) by interpreting (D,H) as image plane and resizing to (H,W).
    """
    # sum along X (last axis)
    lat = torch.sum(vol, dim=-1)  # (B,1,D,H)

    # normalize per-sample to stabilize training loss
    B = lat.shape[0]
    lat2 = lat.view(B, -1)
    mn = lat2.min(dim=1).values.view(B, 1, 1, 1)
    mx = lat2.max(dim=1).values.view(B, 1, 1, 1)
    lat = (lat - mn) / (mx - mn + 1e-8)

    return lat  # (B,1,D,H)


def lat_image_from_lat_tensor(lat_dh: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """
    lat_dh: (B,1,D,H)
    Convert to (B,1,H,W) by treating (D,H) as image and resizing to out_hw.
    """
    B, C, D, H = lat_dh.shape
    x = lat_dh  # (B,1,D,H)
    x = x.permute(0, 1, 3, 2)  # (B,1,H,D) so "W" becomes D
    x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    return x  # (B,1,outH,outW)


# -------------------------
# Model: 3D U-Net (clean, real)
# -------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        b = base
        self.enc1 = ConvBlock3D(1, b)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(b, b * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(b * 2, b * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.mid = ConvBlock3D(b * 4, b * 8)

        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock3D(b * 8, b * 4)
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(b * 4, b * 2)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock3D(b * 2, b)

        self.out = nn.Conv3d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)      # (B,b, D,H,W)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        m  = self.mid(self.pool3(e3))

        d3 = self.up3(m)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        y = self.out(d1)
        return y


# -------------------------
# Dataset (reads your NPZs directly)
# -------------------------
@dataclass
class Sample:
    ap01: np.ndarray          # (H,W) float32 [0,1]
    lat01: np.ndarray         # (H,W) float32 [0,1]
    bp01: np.ndarray          # (D,H,W) float32 [0,1]
    case_id: str


class NPZPairsDataset(Dataset):
    def __init__(self, npz_dir: str, vol_dhw: Tuple[int, int, int], img_hw: Tuple[int, int]):
        super().__init__()
        self.paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"No .npz found in: {npz_dir}")
        self.vol_dhw = vol_dhw
        self.img_hw = img_hw

    @staticmethod
    def _get_any(d: np.lib.npyio.NpzFile, keys: List[str]) -> Optional[np.ndarray]:
        for k in keys:
            if k in d.files:
                return d[k]
        return None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        with np.load(p, allow_pickle=True) as d:
            case_id = None
            if "case_id" in d.files:
                case_id = str(d["case_id"])
            else:
                case_id = os.path.basename(p).replace(".npz", "")

            # Accept a few common key names (no guessing beyond reasonable fallbacks)
            ap = self._get_any(d, ["ap", "ap_img", "ap_drr", "drr_ap"])
            lat = self._get_any(d, ["lat", "lat_img", "lat_drr", "drr_lat", "lat_gt"])
            if ap is None or lat is None:
                raise KeyError(
                    f"{os.path.basename(p)} missing AP/LAT arrays. "
                    f"Available keys: {d.files}"
                )

            ap = np.asarray(ap).astype(np.float32)
            lat = np.asarray(lat).astype(np.float32)

            # Force 2D
            ap = ap.squeeze()
            lat = lat.squeeze()
            if ap.ndim != 2 or lat.ndim != 2:
                raise ValueError(f"{os.path.basename(p)} AP/LAT must be 2D after squeeze. Got ap{ap.shape} lat{lat.shape}")

            # Normalize + standardize size to img_hw (so the volume is consistent)
            ap01 = normalize01(ap)
            lat01 = normalize01(lat)

            ap01 = center_crop_or_pad_2d(ap01, self.img_hw)
            lat01 = center_crop_or_pad_2d(lat01, self.img_hw)

            # Eq(1) backprojection
            bp01 = backproject_ap_to_volume(ap01, self.vol_dhw)

        # Torch expects (C, D, H, W)
        bp_t = torch.from_numpy(bp01[None, ...])         # (1,D,H,W)
        ap_t = torch.from_numpy(ap01[None, ...])         # (1,H,W)
        lat_t = torch.from_numpy(lat01[None, ...])       # (1,H,W)

        return {
            "bp": bp_t.float(),
            "ap": ap_t.float(),
            "lat": lat_t.float(),
            "case_id": case_id,
            "path": p,
        }


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, img_hw: Tuple[int, int]) -> Dict[str, float]:
    model.eval()
    l1_sum, n = 0.0, 0
    for batch in loader:
        bp = batch["bp"].to(device)  # (B,1,D,H,W)
        lat_gt = batch["lat"].to(device)  # (B,1,H,W)

        ct_pred = model(bp)  # (B,1,D,H,W)

        lat_dh = forward_project_lat_from_volume(ct_pred)         # (B,1,D,H)
        lat_pred = lat_image_from_lat_tensor(lat_dh, img_hw)      # (B,1,H,W)

        l1 = torch.mean(torch.abs(lat_pred - lat_gt)).item()
        l1_sum += l1
        n += 1
    return {"val_l1": l1_sum / max(n, 1)}


def train(
    npz_dir: str,
    run_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    num_workers: int,
    device: str,
    vol_dhw: Tuple[int, int, int],
    img_hw: Tuple[int, int],
    val_frac: float,
    export_n: int,
    base_channels: int,
) -> None:
    seed_all(seed)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "examples"))
    ensure_dir(os.path.join(run_dir, "triplets"))

    # Dataset split
    ds = NPZPairsDataset(npz_dir=npz_dir, vol_dhw=vol_dhw, img_hw=img_hw)
    N = len(ds)
    n_val = max(1, int(round(N * val_frac)))
    n_train = max(1, N - n_val)

    # deterministic split
    idxs = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idxs)
    tr_idxs = idxs[:n_train]
    va_idxs = idxs[n_train:]

    tr_ds = torch.utils.data.Subset(ds, tr_idxs.tolist())
    va_ds = torch.utils.data.Subset(ds, va_idxs.tolist())

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = UNet3D(base=base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Loss is defined on the synthesized LAT view (Eq 9 supervision)
    best_val = float("inf")
    log_path = os.path.join(run_dir, "log.jsonl")
    ckpt_best = os.path.join(run_dir, "best.pt")
    ckpt_last = os.path.join(run_dir, "last.pt")
    metrics_path = os.path.join(run_dir, "metrics.json")
    config_path = os.path.join(run_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(
            {
                "npz_dir": npz_dir,
                "run_dir": run_dir,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "num_workers": num_workers,
                "device": device,
                "vol_dhw": vol_dhw,
                "img_hw": img_hw,
                "val_frac": val_frac,
                "export_n": export_n,
                "base_channels": base_channels,
            },
            f,
            indent=2,
        )

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        steps = 0

        for batch in tr_loader:
            bp = batch["bp"].to(device, non_blocking=True)     # (B,1,D,H,W)
            lat_gt = batch["lat"].to(device, non_blocking=True)  # (B,1,H,W)

            ct_pred = model(bp)

            lat_dh = forward_project_lat_from_volume(ct_pred)       # (B,1,D,H)
            lat_pred = lat_image_from_lat_tensor(lat_dh, img_hw)    # (B,1,H,W)

            loss = F.l1_loss(lat_pred, lat_gt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += float(loss.item())
            steps += 1

        train_l1 = ep_loss / max(steps, 1)
        val_metrics = evaluate(model, va_loader, device=device, img_hw=img_hw)
        val_l1 = val_metrics["val_l1"]

        rec = {
            "epoch": ep,
            "train_l1": train_l1,
            "val_l1": val_l1,
            "time_s": time.time() - t0,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        # Save last
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "rec": rec}, ckpt_last)

        # Save best
        if val_l1 < best_val:
            best_val = val_l1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "rec": rec}, ckpt_best)

        print(f"epoch {ep:03d} | train_l1={train_l1:.6f} | val_l1={val_l1:.6f} | best={best_val:.6f}")

    # Final metrics summary
    with open(metrics_path, "w") as f:
        json.dump({"best_val_l1": best_val}, f, indent=2)

    # Export example images + triplets from val set (or train if tiny)
    export_examples(model, ds, va_idxs if len(va_idxs) else tr_idxs, run_dir, device, img_hw, export_n)


@torch.no_grad()
def export_examples(
    model: nn.Module,
    full_ds: NPZPairsDataset,
    eval_indices: np.ndarray,
    run_dir: str,
    device: str,
    img_hw: Tuple[int, int],
    export_n: int,
) -> None:
    model.eval()
    ex_dir = os.path.join(run_dir, "examples")
    tri_dir = os.path.join(run_dir, "triplets")
    ensure_dir(ex_dir)
    ensure_dir(tri_dir)

    take = eval_indices[:export_n].tolist()
    for i, idx in enumerate(take):
        batch = full_ds[idx]
        bp = batch["bp"].unsqueeze(0).to(device)     # (1,1,D,H,W)
        ap = batch["ap"].squeeze(0).numpy()          # (H,W)
        lat_gt = batch["lat"].squeeze(0).numpy()     # (H,W)

        ct_pred = model(bp)  # (1,1,D,H,W)

        lat_dh = forward_project_lat_from_volume(ct_pred)         # (1,1,D,H)
        lat_pred = lat_image_from_lat_tensor(lat_dh, img_hw)      # (1,1,H,W)
        lat_pred = lat_pred.squeeze().cpu().numpy().astype(np.float32)

        # already normalized [0,1] from pipeline
        ap01 = np.clip(ap, 0.0, 1.0)
        gt01 = np.clip(lat_gt, 0.0, 1.0)
        pr01 = np.clip(lat_pred, 0.0, 1.0)

        to_uint8_img(ap01).save(os.path.join(ex_dir, f"{i:03d}_ap.png"))
        to_uint8_img(gt01).save(os.path.join(ex_dir, f"{i:03d}_lat_gt.png"))
        to_uint8_img(pr01).save(os.path.join(ex_dir, f"{i:03d}_lat_pred.png"))

        save_triplet(ap01, gt01, pr01, os.path.join(tri_dir, f"triplet_{i:03d}.png"))


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, default="runs/runs_final/data/drr_pairs_fixed/npz/")
    ap.add_argument("--run_dir", type=str, default="runs/final_ap2lat")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--export_n", type=int, default=12)
    ap.add_argument("--base_channels", type=int, default=16)
    return ap.parse_args()


def main():
    args = parse_args()
    vol_dhw = (args.D, args.H, args.W)
    img_hw = (args.H, args.W)

    train(
        npz_dir=args.npz_dir,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        vol_dhw=vol_dhw,
        img_hw=img_hw,
        val_frac=args.val_frac,
        export_n=args.export_n,
        base_channels=args.base_channels,
    )


if __name__ == "__main__":
    main()
