from __future__ import annotations

# --- allow imports from repo root (src/) ---
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ------------------------------------------

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Optional: CT reading from MHD via SimpleITK
try:
    import SimpleITK as sitk
    _HAS_SITK = True
except Exception:
    _HAS_SITK = False


# ------------------------- utils -------------------------
def collate_samples(batch):
    """
    Convert List[Sample] -> dict of batched tensors that default_collate can’t do.
    """
    ap = torch.stack([b.ap_zx for b in batch], dim=0)      # (B,1,Z,X)
    lat = torch.stack([b.lat_zy for b in batch], dim=0)    # (B,1,Z,Y)

    if batch[0].ct_zyx is None:
        ct = None
    else:
        ct = torch.stack([b.ct_zyx for b in batch], dim=0) # (B,1,Z,Y,X)

    case_id = [b.case_id for b in batch]
    return {"ap_zx": ap, "lat_zy": lat, "ct_zyx": ct, "case_id": case_id}

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse < 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)

def ssim_simple(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """
    Lightweight SSIM (single-scale, global stats). Not a perfect SSIM,
    but stable + dependency-free for “compare vs. yourself” reporting.
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_x = float(pred.mean())
    mu_y = float(gt.mean())
    sigma_x = float(pred.var())
    sigma_y = float(gt.var())
    sigma_xy = float(((pred - mu_x) * (gt - mu_y)).mean())

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    return float(num / (den + 1e-12))

def center_crop_or_pad_zyx(vol: np.ndarray, out_zyx: Tuple[int, int, int]) -> np.ndarray:
    """
    Center crop if larger, center pad with zeros if smaller. Input/Output Z,Y,X.
    """
    z, y, x = vol.shape
    oz, oy, ox = out_zyx

    # crop
    z0 = max(0, (z - oz) // 2); z1 = z0 + min(oz, z)
    y0 = max(0, (y - oy) // 2); y1 = y0 + min(oy, y)
    x0 = max(0, (x - ox) // 2); x1 = x0 + min(ox, x)
    cropped = vol[z0:z1, y0:y1, x0:x1]

    # pad
    cz, cy, cx = cropped.shape
    pad_z0 = max(0, (oz - cz) // 2)
    pad_y0 = max(0, (oy - cy) // 2)
    pad_x0 = max(0, (ox - cx) // 2)
    pad_z1 = oz - cz - pad_z0
    pad_y1 = oy - cy - pad_y0
    pad_x1 = ox - cx - pad_x0

    out = np.pad(
        cropped,
        ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)),
        mode="constant",
        constant_values=0.0,
    )
    return out.astype(np.float32)

def read_mhd_zyx(mhd_path: str) -> np.ndarray:
    if not _HAS_SITK:
        raise RuntimeError("SimpleITK not available. Install: pip install SimpleITK")

    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)  # Z,Y,X
    return arr.astype(np.float32)

def resolve_path(p: str) -> Path:
    p2 = p.replace("\\", "/")
    cand = Path(p2)
    if cand.exists():
        return cand
    # try relative to repo root
    cand2 = PROJECT_ROOT / p2
    if cand2.exists():
        return cand2
    return cand  # return original; caller will handle existence

def load_npz_paths_from_csv(csv_path: str) -> List[str]:
    """
    Robust: accepts either a single column CSV of paths, or a header row containing 'npz_path'.
    """
    csv_path = str(csv_path)
    paths: List[str] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return paths

    # If header contains npz_path
    header = [c.strip() for c in rows[0]]
    if any(h.lower() == "npz_path" for h in header):
        idx = [i for i, h in enumerate(header) if h.lower() == "npz_path"][0]
        for r in rows[1:]:
            if not r:
                continue
            paths.append(r[idx].strip())
        return paths

    # Otherwise: take first column, skipping blanks
    for r in rows:
        if not r:
            continue
        p = r[0].strip()
        if p and (p.endswith(".npz") or "npz" in p.lower()):
            paths.append(p)
    return paths

def save_triplet(ap_2d: np.ndarray, lat_pred_2d: np.ndarray, lat_gt_2d: np.ndarray, out_path: Path) -> None:
    ap = (normalize01(ap_2d) * 255.0).astype(np.uint8)
    pr = (normalize01(lat_pred_2d) * 255.0).astype(np.uint8)
    gt = (normalize01(lat_gt_2d) * 255.0).astype(np.uint8)

    ap_im = Image.fromarray(ap, mode="L")
    pr_im = Image.fromarray(pr, mode="L")
    gt_im = Image.fromarray(gt, mode="L")

    w, h = ap_im.size
    canvas = Image.new("L", (w * 3, h))
    canvas.paste(ap_im, (0, 0))
    canvas.paste(pr_im, (w, 0))
    canvas.paste(gt_im, (w * 2, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ------------------------- dataset -------------------------

@dataclass
class Sample:
    ap_zx: torch.Tensor          # (1, Z, X)
    lat_zy: torch.Tensor         # (1, Z, Y)
    ct_zyx: Optional[torch.Tensor]  # (1, Z, Y, X) or None
    case_id: str

class NpzAPLatCTDataset(Dataset):
    """
    NPZ keys: case_id, mhd_path, spacing_zyx, ap (Z,X), lat (Z,Y)
    CT is read from mhd_path (train-time supervision). Inference uses only AP.
    """
    def __init__(
        self,
        npz_paths: List[str],
        target_zyx: Tuple[int, int, int] = (256, 256, 256),
        hu_clip: Tuple[float, float] = (-1000.0, 400.0),
        load_ct: bool = True,
    ):
        self.npz_paths = npz_paths
        self.target_zyx = target_zyx
        self.hu_clip = hu_clip
        self.load_ct = load_ct

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> Sample:
        p = resolve_path(self.npz_paths[idx])
        d = np.load(str(p), allow_pickle=True)

        case_id = str(d["case_id"])
        ap = d["ap"].astype(np.float32)   # (Z,X)
        lat = d["lat"].astype(np.float32) # (Z,Y)

        ap_t = torch.from_numpy(ap)[None, ...]   # (1,Z,X)
        lat_t = torch.from_numpy(lat)[None, ...] # (1,Z,Y)

        ct_t: Optional[torch.Tensor] = None
        if self.load_ct:
            mhd_path = str(d["mhd_path"])
            mhd_resolved = resolve_path(mhd_path)
            if not mhd_resolved.exists():
                raise FileNotFoundError(
                    f"CT mhd_path not found for case_id={case_id}: {mhd_path} (resolved={mhd_resolved})"
                )
            ct = read_mhd_zyx(str(mhd_resolved))  # Z,Y,X (HU typically)
            # clip HU -> normalize to 0..1
            lo, hi = self.hu_clip
            ct = np.clip(ct, lo, hi)
            ct = (ct - lo) / (hi - lo + 1e-8)  # 0..1
            ct = center_crop_or_pad_zyx(ct, self.target_zyx)  # (256,256,256)
            ct_t = torch.from_numpy(ct)[None, ...]  # (1,Z,Y,X)

        return Sample(ap_t.float(), lat_t.float(), ct_t.float() if ct_t is not None else None, case_id)


# ------------------------- model -------------------------

class ConvBlock3D(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(),
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UNet3D(nn.Module):
    """
    Small 3D UNet operating in latent space (e.g., 64^3).
    Input:  (B,1, Zl,Yl,Xl)   (backprojected+pooled)
    Output: (B,1, Zl,Yl,Xl)   (CT latent)
    """
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1 = ConvBlock3D(1, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)

        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ------------------------- physics-ish ops -------------------------

def backproject_ap_to_volume(ap_zx: torch.Tensor, out_y: int) -> torch.Tensor:
    """
    AP is (B,1,Z,X). Backproject -> (B,1,Z,Y,X) by repeating across Y.
    This matches your repo convention: ap(Z,X), vol(Z,Y,X).
    """
    B, C, Z, X = ap_zx.shape
    ap_zyx = ap_zx.unsqueeze(3)          # (B,1,Z,1,X)
    vol = ap_zyx.repeat(1, 1, 1, out_y, 1)  # (B,1,Z,Y,X)
    return vol

def forward_project_ct_to_lat(ct_zyx: torch.Tensor) -> torch.Tensor:
    """
    CT is (B,1,Z,Y,X). LAT is (B,1,Z,Y) by integrating along X.
    Then per-sample normalize to 0..1 to match stored lat.
    """
    lat = ct_zyx.sum(dim=-1)  # (B,1,Z,Y)
    # normalize per-sample
    B = lat.shape[0]
    lat_norm = []
    for b in range(B):
        x = lat[b]
        mn = x.min()
        mx = x.max()
        lat_norm.append((x - mn) / (mx - mn + 1e-8))
    return torch.stack(lat_norm, dim=0)

def avg_pool_latent(vol_zyx: torch.Tensor, down: int) -> torch.Tensor:
    """
    Downsample (B,1,Z,Y,X) -> (B,1,Zl,Yl,Xl)
    """
    if down == 1:
        return vol_zyx
    return F.avg_pool3d(vol_zyx, kernel_size=down, stride=down)


# ------------------------- train / eval -------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    latent_down: int,
    w_latent: float,
    w_lat: float,
    amp: bool,
) -> float:
    model.train()
    total = 0.0
    n = 0
    amp_enabled = amp and (device.type == "cuda")

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)


    for batch in loader:
        ap = batch["ap_zx"].to(device)
        lat_gt = batch["lat_zy"].to(device)
        ct = batch["ct_zyx"]

        if ct is None:
            raise RuntimeError("CT is required for latent supervision, but dataset returned ct=None.")
        ct = ct.to(device)  # (B,1,Z,Y,X)

        B, _, Z, Y, X = ct.shape

        # Eq.1-style step (in your simplified repo sense): backproject
        bp = backproject_ap_to_volume(ap, out_y=Y)  # (B,1,Z,Y,X)

        # latent-space inputs/targets
        bp_lat = avg_pool_latent(bp, latent_down)     # (B,1,Zl,Yl,Xl)
        ct_lat_gt = avg_pool_latent(ct, latent_down)  # (B,1,Zl,Yl,Xl)

        with torch.autocast(device_type="cuda", enabled=amp_enabled):

            ct_lat_pred = model(bp_lat)
            loss_latent = F.mse_loss(ct_lat_pred, ct_lat_gt)

            # decode latent -> CT (fixed upsample)
            ct_pred = F.interpolate(ct_lat_pred, scale_factor=latent_down, mode="trilinear", align_corners=False)

            # Eq.9-style step (your simplified DRR sense): forward-project CT -> LAT
            lat_pred = forward_project_ct_to_lat(ct_pred)

            loss_lat = F.mse_loss(lat_pred, lat_gt)
            loss = w_latent * loss_latent + w_lat * loss_lat

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        total += float(loss.detach().cpu().item())
        n += 1

    return total / max(1, n)

@torch.no_grad()
def eval_and_save_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    latent_down: int,
    out_dir: Path,
    num_examples: int,
) -> dict:
    model.eval()

    metrics = []
    saved = 0
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        ap = batch["ap_zx"].to(device)
        lat_gt = batch["lat_zy"].to(device)
        ct = batch["ct_zyx"].to(device)
        case_id = batch["case_id"][0]


        B, _, Z, Y, X = ct.shape

        bp = backproject_ap_to_volume(ap, out_y=Y)
        bp_lat = avg_pool_latent(bp, latent_down)

        ct_lat_pred = model(bp_lat)
        ct_pred = F.interpolate(ct_lat_pred, scale_factor=latent_down, mode="trilinear", align_corners=False)
        lat_pred = forward_project_ct_to_lat(ct_pred)

        ap_np = ap[0, 0].detach().cpu().numpy()       # (Z,X)
        lat_gt_np = lat_gt[0, 0].detach().cpu().numpy()   # (Z,Y)
        lat_pr_np = lat_pred[0, 0].detach().cpu().numpy() # (Z,Y)

        p = psnr(lat_pr_np, lat_gt_np, data_range=1.0)
        s = ssim_simple(lat_pr_np, lat_gt_np, data_range=1.0)
        metrics.append({"case_id": case_id, "psnr": p, "ssim": s})

        if saved < num_examples:
            out_path = ex_dir / f"triplet_{saved:03d}__{case_id.replace('/', '_')}.png"
            save_triplet(ap_np, lat_pr_np, lat_gt_np, out_path)
            saved += 1

    psnrs = [m["psnr"] for m in metrics]
    ssims = [m["ssim"] for m in metrics]
    summary = {
        "count": len(metrics),
        "psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
        "psnr_std": float(np.std(psnrs)) if psnrs else 0.0,
        "ssim_mean": float(np.mean(ssims)) if ssims else 0.0,
        "ssim_std": float(np.std(ssims)) if ssims else 0.0,
        "per_case": metrics,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--latent_down", type=int, default=4, help="Downsample factor for CT latent (256 -> 64 if 4).")
    ap.add_argument("--base", type=int, default=16, help="UNet base channels.")
    ap.add_argument("--w_latent", type=float, default=1.0)
    ap.add_argument("--w_lat", type=float, default=0.1)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    ap.add_argument("--num_examples", type=int, default=10)
    ap.add_argument("--ckpt", type=str, default="", help="If set, skip training and only eval this checkpoint.")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_npzs = load_npz_paths_from_csv(args.train_csv)
    val_npzs = load_npz_paths_from_csv(args.val_csv)
    if not train_npzs:
        raise RuntimeError(f"No train npz paths found in {args.train_csv}")
    if not val_npzs:
        raise RuntimeError(f"No val npz paths found in {args.val_csv}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; switching to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # CT is required for latent supervision
    if not _HAS_SITK:
        raise RuntimeError("SimpleITK is required to read CT from mhd_path. Install: pip install SimpleITK")

    ds_train = NpzAPLatCTDataset(train_npzs, load_ct=True)
    ds_val = NpzAPLatCTDataset(val_npzs, load_ct=True)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate_samples)
    dl_val   = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_samples)


    model = UNet3D(base=args.base).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_path = out_dir / "checkpoint.pt"

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ck["model"])
        print(f"[eval] loaded checkpoint: {args.ckpt}")
    elif ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ck["model"])
        print(f"[resume] loaded checkpoint: {ckpt_path}")
    else:
        print(f"[train] epochs={args.epochs} batch={args.batch} lr={args.lr} latent_down={args.latent_down} amp={args.amp}")
        for ep in range(1, args.epochs + 1):
            loss = train_one_epoch(
                model=model,
                loader=dl_train,
                optim=optim,
                device=device,
                latent_down=args.latent_down,
                w_latent=args.w_latent,
                w_lat=args.w_lat,
                amp=args.amp and device.type == "cuda",
            )
            print(f"[train] epoch {ep:03d}/{args.epochs}  loss={loss:.6f}")

        torch.save({"model": model.state_dict(), "args": vars(args)}, str(ckpt_path))
        print(f"[train] saved: {ckpt_path}")

    summary = eval_and_save_examples(
        model=model,
        loader=dl_val,
        device=device,
        latent_down=args.latent_down,
        out_dir=out_dir,
        num_examples=args.num_examples,
    )

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "metrics.txt").write_text(
        f"count={summary['count']}\n"
        f"psnr_mean={summary['psnr_mean']:.4f}  psnr_std={summary['psnr_std']:.4f}\n"
        f"ssim_mean={summary['ssim_mean']:.4f}  ssim_std={summary['ssim_std']:.4f}\n"
    )

    print("[eval] " + (out_dir / "metrics.txt").read_text().strip())
    print(f"[eval] examples saved in: {out_dir / 'examples'}")


if __name__ == "__main__":
    main()
