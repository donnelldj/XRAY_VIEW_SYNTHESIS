# scripts/viz_ap_lat_triplets.py
# Visualize AP -> LAT prediction triplets from an ap2lat checkpoint (best.pt / last.pt)

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0).round().astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(img01)).save(out_path)


def _stack_triplet(ap01: np.ndarray, gt01: np.ndarray, pr01: np.ndarray) -> np.ndarray:
    # AP | GT | PRED
    return np.concatenate([ap01, gt01, pr01], axis=1)


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + 1e-8)


def load_state_dict_any(ckpt_path: str, device: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        for k in ("state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


def infer_base(sd: dict) -> int:
    w = sd.get("enc1.0.weight", None)  # (base, 1, 3, 3)
    if w is None:
        raise SystemExit("[ERROR] Could not find enc1.0.weight in checkpoint.")
    return int(w.shape[0])


def _double_conv_gn(in_ch: int, out_ch: int, groups: int = 8) -> nn.Sequential:
    g = min(groups, out_ch)
    # MUST be flat Sequential so keys match:
    #   enc1.0, enc1.1, enc1.2, enc1.3, enc1.4, enc1.5
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),  # 0
        nn.GroupNorm(g, out_ch),                 # 1
        nn.ReLU(inplace=True),                   # 2
        nn.Conv2d(out_ch, out_ch, 3, padding=1), # 3
        nn.GroupNorm(g, out_ch),                 # 4
        nn.ReLU(inplace=True),                   # 5
    )


class Ap2LatUNet2D_GN(nn.Module):
    """
    Matches checkpoint keys:
      enc1.*, enc2.*, bott.*, dec2.*, dec1.*, up1.*, up2.*, out.*
    """
    def __init__(self, base: int = 32):
        super().__init__()
        b = base

        self.enc1 = _double_conv_gn(1, b)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = _double_conv_gn(b, 2 * b)
        self.pool2 = nn.MaxPool2d(2)

        self.bott = _double_conv_gn(2 * b, 4 * b)

        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, kernel_size=2, stride=2)
        self.dec2 = _double_conv_gn(4 * b, 2 * b)

        self.up1 = nn.ConvTranspose2d(2 * b, b, kernel_size=2, stride=2)
        self.dec1 = _double_conv_gn(2 * b, b)

        self.out = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        bt = self.bott(p2)

        u2 = self.up2(bt)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to ap2lat best.pt / last.pt / ckpt_best.pt")
    p.add_argument("--npz_dir", default="data/drp_pairs/npz")
    p.add_argument("--out_dir", default="runs/ap2lat_viz_triplets")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)
    (out_dir / "viz_triplets").mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(Path(args.npz_dir).glob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"[ERROR] No npz found in: {args.npz_dir}")

    rng = np.random.default_rng(args.seed)
    if args.n < len(npz_paths):
        pick = rng.choice(len(npz_paths), size=args.n, replace=False)
        npz_paths = [npz_paths[i] for i in sorted(pick)]

    sd = load_state_dict_any(args.ckpt, device=device)
    base = infer_base(sd)

    model = Ap2LatUNet2D_GN(base=base).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    wrote = 0
    with torch.no_grad():
        for i, npz_path in enumerate(npz_paths):
            d = np.load(npz_path)
            if "ap" not in d.files or "lat" not in d.files:
                continue

            ap = d["ap"].astype(np.float32)
            lat_gt = d["lat"].astype(np.float32)

            ap01 = _norm01(ap)
            gt01 = _norm01(lat_gt)

            ap_t = torch.from_numpy(ap01)[None, None].to(device)
            lat_pred = model(ap_t)[0, 0].detach().cpu().numpy()
            pr01 = _norm01(lat_pred)

            _save_png(ap01, out_dir / "examples" / f"{i:03d}_ap.png")
            _save_png(gt01, out_dir / "examples" / f"{i:03d}_lat_gt.png")
            _save_png(pr01, out_dir / "examples" / f"{i:03d}_lat_pred.png")

            trip = _stack_triplet(ap01, gt01, pr01)
            _save_png(trip, out_dir / "viz_triplets" / f"triplet_{i:03d}.png")
            wrote += 1

    print(f"[OK] base={base} wrote={wrote} -> {out_dir}")


if __name__ == "__main__":
    main()
