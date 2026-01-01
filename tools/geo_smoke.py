from __future__ import annotations

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
except Exception:
    project_lat = None
# src/geo/backproject.py




def backproject_parallel_beam(ap_2d: np.ndarray, out_zyx: tuple[int, int, int], axis: int = 0) -> np.ndarray:
    """
    Simple parallel-beam backprojection baseline: "smear" a 2D projection back into a 3D volume.

    out_zyx: desired output volume shape (Z,Y,X)
    axis:
      0 -> assumes ap_2d is (Y,X) and represents sum over Z (classic AP in older code)
           vol[z,y,x] = ap_2d[y,x]
      1 -> assumes ap_2d is (Z,X) and represents sum over Y (your new AP convention)
           vol[z,y,x] = ap_2d[z,x]
      2 -> assumes ap_2d is (Z,Y) and represents sum over X
           vol[z,y,x] = ap_2d[z,y]
    """
    ap = ap_2d.astype(np.float32)
    Z, Y, X = [int(v) for v in out_zyx]

    if axis == 0:
        # ap: (Y,X) -> tile across Z
        if ap.shape != (Y, X):
            raise ValueError(f"axis=0 expects ap shape (Y,X)=({Y},{X}), got {ap.shape}")
        vol = np.broadcast_to(ap[None, :, :], (Z, Y, X)).copy()
        return vol.astype(np.float32)

    if axis == 1:
        # ap: (Z,X) -> tile across Y
        if ap.shape != (Z, X):
            raise ValueError(f"axis=1 expects ap shape (Z,X)=({Z},{X}), got {ap.shape}")
        vol = np.broadcast_to(ap[:, None, :], (Z, Y, X)).copy()
        return vol.astype(np.float32)

    if axis == 2:
        # ap: (Z,Y) -> tile across X
        if ap.shape != (Z, Y):
            raise ValueError(f"axis=2 expects ap shape (Z,Y)=({Z},{Y}), got {ap.shape}")
        vol = np.broadcast_to(ap[:, :, None], (Z, Y, X)).copy()
        return vol.astype(np.float32)

    raise ValueError(f"axis must be 0, 1, or 2. Got {axis}")


def minmax01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def forward_project_fallback(vol_zyx: np.ndarray, view: str) -> np.ndarray:
    """
    vol_zyx: (Z,Y,X)

    IMPORTANT: This matches your NEW dataset convention from src/vis/drr.py:
      AP  = integrate along Y -> (Z,X)
      LAT = integrate along X -> (Z,Y)
    """
    view = view.upper()
    if view == "AP":
        return vol_zyx.sum(axis=1).astype(np.float32)  # (Z,X)
    if view == "LAT":
        return vol_zyx.sum(axis=2).astype(np.float32)  # (Z,Y)
    raise ValueError("view must be 'AP' or 'LAT'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="Path to one .npz containing keys: ap, lat")
    p.add_argument("--show", action="store_true")
    p.add_argument("--y_size", type=int, default=None,
                   help="Optional Y size for backprojection volume. Defaults to lat.shape[1].")
    args = p.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    if "ap" not in npz.files or "lat" not in npz.files:
        raise ValueError(f"NPZ missing ap/lat. Keys: {npz.files}")

    # NEW dataset convention:
    # ap:  (Z,X)
    # lat: (Z,Y)
    ap_img = npz["ap"].astype(np.float32)
    lat_gt_zy = npz["lat"].astype(np.float32)

    if ap_img.ndim != 2 or lat_gt_zy.ndim != 2:
        raise ValueError(f"Expected ap/lat as 2D arrays. Got ap={ap_img.shape}, lat={lat_gt_zy.shape}")

    z_ap, x_ap = ap_img.shape
    z_lat, y_lat = lat_gt_zy.shape

    if z_ap != z_lat:
        # If these disagree, something upstream is inconsistent
        raise ValueError(f"Z mismatch: ap(Z,X)={ap_img.shape} vs lat(Z,Y)={lat_gt_zy.shape}")

    z = z_ap
    x = x_ap
    y = args.y_size if args.y_size is not None else y_lat

    # 1) backproject AP (Z,X) into volume (Z,Y,X) by "smearing" along Y
    #    axis=1 corresponds to the Y dimension in (Z,Y,X)
    vol = backproject_parallel_beam(ap_img, out_zyx=(z, y, x), axis=1)

    # 2) forward project -> LAT (Z,Y)
    if project_lat is not None:
        lat_pred_zy = project_lat(vol).astype(np.float32)  # expected (Z,Y)
    else:
        lat_pred_zy = forward_project_fallback(vol, "LAT")  # (Z,Y)

    # 3) crop to match GT exactly (Z,Y)
    zz = min(lat_pred_zy.shape[0], lat_gt_zy.shape[0])
    yy = min(lat_pred_zy.shape[1], lat_gt_zy.shape[1])
    lat_pred_zy = lat_pred_zy[:zz, :yy]
    lat_gt_zy = lat_gt_zy[:zz, :yy]

    # normalize for interpretable diff
    ap_n = minmax01(ap_img)
    lat_pred_n = minmax01(lat_pred_zy)
    lat_gt_n = minmax01(lat_gt_zy)

    diff = np.abs(lat_pred_n - lat_gt_n)
    mae = float(diff.mean())
    mse = float((diff ** 2).mean())

    print(f"AP: {ap_img.shape}  VOL: {vol.shape}  LAT_GT(Z,Y): {lat_gt_zy.shape}  LAT_PRED(Z,Y): {lat_pred_zy.shape}")
    print(f"MAE: {mae:.6f}  MSE: {mse:.6f}")
    print(f"npz: {args.npz}")

    if args.show:
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(np.flipud(ap_n), cmap="gray")
        ax[1].imshow(np.flipud(lat_gt_n), cmap="gray")
        ax[2].imshow(np.flipud(lat_pred_n), cmap="gray")
        ax[3].imshow(np.flipud(diff), cmap="magma")

        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
