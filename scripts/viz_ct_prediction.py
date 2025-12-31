# scripts/viz_ct_prediction.py
# Visualize BP volume vs GT CT vs Predicted CT (3D stage ONLY)

from __future__ import annotations

import argparse
import numpy as np
import torch
import pyvista as pv

from src.data_drr_pairs import DRRPairsDataset
from src.models_unet3d import UNet3D_CT


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def volume_to_mesh(vol_zyx: np.ndarray, iso: float = 0.35):
    z, h, w = vol_zyx.shape
    grid = pv.ImageData(
        dimensions=(w, h, z),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
    )
    grid.point_data["v"] = vol_zyx.ravel(order="F")
    return grid.contour([iso])


def load_ckpt_state_dict(ckpt_path: str, device: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)

    # common patterns:
    # - {"model": state_dict, ...}
    # - raw state_dict
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    # Guardrail: ap2lat checkpoints have keys like "enc1.0.weight"
    # UNet3D_CT expects keys like "inc.block.0.conv.weight"
    if any(k.startswith("enc1.") or k.startswith("enc2.") for k in sd.keys()):
        raise SystemExit(
            "[ERROR] This checkpoint is AP→LAT (2D) weights (enc1/enc2/* keys).\n"
            "Use: python scripts/viz_ap_lat_triplets.py --ckpt <best.pt>\n"
            "NOT viz_ct_prediction.py."
        )

    return sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="3D CT checkpoint (ckpt_best.pt) from BP->CT training")
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--npz_dir", default="data/drr_pairs/npz", help="NPZ dir that includes bp + ct volumes")
    p.add_argument("--iso", type=float, default=0.40)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = DRRPairsDataset(args.npz_dir)
    if args.index < 0 or args.index >= len(ds):
        raise SystemExit(f"[ERROR] index out of range: {args.index} (len={len(ds)})")

    sample = ds[args.index]

    # (Z,H,W)
    bp = sample["bp"][0].cpu().numpy()
    ct_gt = sample["ct"][0].cpu().numpy()

    # model + ckpt
    model = UNet3D_CT(base=args.base).to(device)
    sd = load_ckpt_state_dict(args.ckpt, device=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    with torch.no_grad():
        bp_t = sample["bp"].unsqueeze(1).to(device)  # (1,1,Z,H,W)
        ct_pred = model(bp_t)[0, 0].detach().cpu().numpy()

    bp_n = normalize01(bp)
    gt_n = normalize01(ct_gt)
    pr_n = normalize01(ct_pred)

    m_bp = volume_to_mesh(bp_n, iso=args.iso)
    m_gt = volume_to_mesh(gt_n, iso=args.iso)
    m_pr = volume_to_mesh(pr_n, iso=args.iso)

    pl = pv.Plotter(shape=(1, 3), window_size=(1600, 600))

    pl.subplot(0, 0)
    pl.add_text("Back-Projected Volume", font_size=10)
    pl.add_mesh(m_bp, opacity=0.6)

    pl.subplot(0, 1)
    pl.add_text("Ground-Truth CT", font_size=10)
    pl.add_mesh(m_gt, opacity=0.6)

    pl.subplot(0, 2)
    pl.add_text("Predicted CT", font_size=10)
    pl.add_mesh(m_pr, opacity=0.6)

    pl.show()


if __name__ == "__main__":
    main()
