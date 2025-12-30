# scripts/viz_ct_vs_pred_volume.py

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_drr_pairs import DRRPairsDataset
from src.projection import load_ct          # already exists in your repo
from src.models_unet3d import UNet3D_CT     # the 3D UNet we defined


def get_bp_and_ct(case_idx: int, drr_dir: str):
    """
    Use DRRPairsDataset to get:
      - back-projected volume (BP) as numpy [D, H, W]
      - ground-truth CT volume as numpy [D, H, W]

    This mirrors scripts/viz_bp_volume.py for BP,
    and uses the matching .npz to load CT via mhd_path.
    """
    ds = DRRPairsDataset(drr_dir)

    # ---- 1) BP volume (same as viz_bp_volume.py) ----
    sample = ds[case_idx]
    bp_t = sample["bp"]              # torch tensor, [1, Z, H, W]
    bp = bp_t[0].cpu().numpy()       # -> [Z, H, W]

    # ---- 2) Matching .npz for CT path ----
    # In data_drr_pairs.py you almost certainly have self.files = sorted(glob.glob(...))
    # If the attribute is named differently, just change "files" to that name.
    npz_path = Path(ds.files[case_idx])
    meta = np.load(npz_path, allow_pickle=True)

    # mhd_path is stored in the .npz as seen earlier
    mhd_arr = meta["mhd_path"]
    if mhd_arr.shape == ():          # 0-d array with a single string
        mhd_path = mhd_arr.item()
    else:
        mhd_path = mhd_arr[()]

    # ---- 3) Load CT volume with existing helper ----
    ct_vol = load_ct(mhd_path)       # expects to return [D, H, W] numpy

    # Optional: sanity check shapes (you can comment this out later)
    print("BP shape:", bp.shape)
    print("CT shape:", ct_vol.shape)

    return bp.astype(np.float32), ct_vol.astype(np.float32)


def save_slice_pair(ct_gt: np.ndarray, ct_pred: np.ndarray, out_dir: Path, prefix: str):
    """
    Save central axial / coronal / sagittal slices of GT & prediction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    assert ct_gt.ndim == 3
    assert ct_pred.ndim == 3

    D, H, W = ct_gt.shape
    z_mid = D // 2
    y_mid = H // 2
    x_mid = W // 2

    planes = [
        ("axial",    ct_gt[z_mid, :, :],   ct_pred[z_mid, :, :]),
        ("coronal",  ct_gt[:, y_mid, :],   ct_pred[:, y_mid, :]),
        ("sagittal", ct_gt[:, :, x_mid],   ct_pred[:, :, x_mid]),
    ]

    for name, gt_slice, pred_slice in planes:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(gt_slice, cmap="gray")
        axes[0].set_title("GT CT")
        axes[0].axis("off")

        axes[1].imshow(pred_slice, cmap="gray")
        axes[1].set_title("Pred CT")
        axes[1].axis("off")

        fig.suptitle(f"{prefix} – {name}")
        fig.tight_layout()

        out_path = out_dir / f"{prefix}_{name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print("saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to BP→CT UNet3D checkpoint")
    parser.add_argument("--drr_dir", type=str, default="data/drr_pairs/npz",
                        help="Directory with DRR pair npz files")
    parser.add_argument("--idx", type=int, default=0,
                        help="Index of case in DRRPairsDataset")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save slice visualizations")
    parser.add_argument("--base", type=int, default=16,
                        help="Base channels of UNet (must match training)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1) BP and GT CT
    bp_vol, ct_gt = get_bp_and_ct(args.idx, args.drr_dir)  # [D, H, W]

    # Normalize BP if you normalized during training
    bp_norm = (bp_vol - bp_vol.min()) / (bp_vol.max() - bp_vol.min() + 1e-8)

    # [1, 1, D, H, W] for UNet3D
    bp_t = torch.from_numpy(bp_norm[None, None, ...]).to(device)

    # 2) Build model and load checkpoint
    model = UNet3D_CT(in_ch=1, out_ch=1, base=args.base).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    if "model" in ck:
        model.load_state_dict(ck["model"])
        print("Loaded ckpt['model'], epoch:", ck.get("epoch"))
    else:
        model.load_state_dict(ck)
        print("Loaded ckpt as plain state_dict")

    model.eval()
    with torch.no_grad():
        ct_pred_t = model(bp_t)      # [1, 1, D, H, W]

    ct_pred = ct_pred_t[0, 0].detach().cpu().numpy()  # [D, H, W]
    print("Pred CT shape:", ct_pred.shape)

    # 3) Save comparison slices
    out_dir = Path(args.out_dir)
    prefix = f"case_{args.idx}"
    save_slice_pair(ct_gt, ct_pred, out_dir, prefix)


if __name__ == "__main__":
    main()
