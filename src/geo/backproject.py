from __future__ import annotations
import numpy as np


def backproject_parallel_beam(
    ap_img: np.ndarray,
    out_zyx: tuple[int, int, int],
    axis: int = 0,
) -> np.ndarray:
    """
    Fast backprojection consistent with our SIMPLE DRR definition (sum along an axis).

    Supported conventions:

    (A) Old convention:
        ap_img: (Y, X)
        out_zyx: (Z, Y, X)
        axis=1 means AP was created by sum(volume, axis=1) over Z
        -> backproject replicates ap_img along Z

    (B) New convention (your current pipeline):
        ap_img: (Z, X)
        out_zyx: (Z, Y, X)
        axis=1 means AP was created by sum(volume, axis=1) over Y
        -> backproject replicates ap_img along Y

    Returns:
      V_bp: (Z, Y, X) float32
    """
    ap_img = np.asarray(ap_img, dtype=np.float32)
    z, y, x = map(int, out_zyx)

    if axis == 0:
        # ap_img expected (Y,X)
        if ap_img.shape != (y, x):
            raise ValueError(f"[axis=0] Expected ap_img (Y,X)=({y},{x}), got {ap_img.shape}")
        v = np.repeat(ap_img[None, :, :], repeats=z, axis=0)  # (Z,Y,X)
        return v.astype(np.float32)

    if axis == 1:
        # ap_img expected (Z,X)
        if ap_img.shape != (z, x):
            raise ValueError(f"[axis=1] Expected ap_img (Z,X)=({z},{x}), got {ap_img.shape}")
        v = np.repeat(ap_img[:, None, :], repeats=y, axis=1)  # (Z,Y,X)
        return v.astype(np.float32)

    raise NotImplementedError("Only axis=0 (smear along Z) and axis=1 (smear along Y) are supported.")
