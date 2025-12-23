# tools/geo_smoke.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.geo.backproject import backproject_parallel_beam

# Your repo has these
try:
    from src.projection import project_lat
except Exception as e:
    project_lat = None


def minmax01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def forward_project_fallback(vol_zyx: np.ndarray, view: str) -> np.ndarray:
    """
    vol_zyx: (Z,Y,X)
    Parallel-beam DRR baseline:
      AP  = sum over Z -> (Y,X)
      LAT = sum over X -> (Z,Y) then transpose -> (Y,Z)
    """
    view = view.upper()
    if view == "AP":
        return vol_zyx.sum(axis=0).astype(np.float32)  # (Y,X)
    if view == "LAT":
        lat_zy = vol_zyx.sum(axis=2).astype(np.float32)  # (Z,Y)
        return lat_zy  # (Z,Y) caller can transpose if they want (Y,Z)
    raise ValueError("view must be 'AP' or 'LAT'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="Path to one .npz containing keys: ap, lat")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    if "ap" not in npz.files or "lat" not in npz.files:
        raise ValueError(f"NPZ missing ap/lat. Keys: {npz.files}")

    ap_img = npz["ap"].astype(np.float32)      # (Y,X)
    lat_gt = npz["lat"].astype(np.float32)     # expected (Y,Z) in your dataset

    if ap_img.ndim != 2 or lat_gt.ndim != 2:
        raise ValueError(f"Expected ap/lat as 2D arrays. Got ap={ap_img.shape}, lat={lat_gt.shape}")

    y, x = ap_img.shape
    z = lat_gt.shape[1]  # treat GT LAT width as Z

    # 1) backproject AP -> volume (Z,Y,X)
    vol = backproject_parallel_beam(ap_img, out_zyx=(z, y, x), axis=0)

    # 2) forward project -> LAT using your function if available
    if project_lat is not None:
        # your project_lat sums axis=2 => (Z,Y)
        lat_pred_zy = project_lat(vol).astype(np.float32)  # (Z,Y)
    else:
        lat_pred_zy = forward_project_fallback(vol, "LAT")  # (Z,Y)

    # convert to (Y,Z) to match GT
    lat_pred_yz = lat_pred_zy.T  # (Y,Z)

    # crop to match exactly
    yy = min(lat_pred_yz.shape[0], lat_gt.shape[0])
    zz = min(lat_pred_yz.shape[1], lat_gt.shape[1])
    lat_pred_yz = lat_pred_yz[:yy, :zz]
    lat_gt_yz = lat_gt[:yy, :zz]

    # normalize both consistently so diff is interpretable
    lat_pred_n = minmax01(lat_pred_yz)
    lat_gt_n = minmax01(lat_gt_yz)

    diff = np.abs(lat_pred_n - lat_gt_n)

    mae = float(diff.mean())
    mse = float((diff ** 2).mean())

    print(f"AP: {ap_img.shape}  VOL: {vol.shape}  LAT_GT: {lat_gt.shape}  LAT_PRED: {lat_pred_yz.shape}")
    print(f"MAE: {mae:.6f}  MSE: {mse:.6f}")
    print(f"npz: {args.npz}")

    if args.show:
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(minmax01(ap_img), cmap="gray"); ax[0].set_title("AP (input)")
        ax[1].imshow(lat_gt_n, cmap="gray"); ax[1].set_title("LAT (GT) [norm]")
        ax[2].imshow(lat_pred_n, cmap="gray"); ax[2].set_title("LAT (geometry) [norm]")
        ax[3].imshow(diff, cmap="magma"); ax[3].set_title("|diff|")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
