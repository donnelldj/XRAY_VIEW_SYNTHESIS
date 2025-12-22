from __future__ import annotations

import numpy as np


def backproject_parallel_beam(
    ap_img: np.ndarray,
    out_zyx: tuple[int, int, int],
    axis: int = 0,
) -> np.ndarray:
    """
    Fast, simple backprojection for our current DRR definition.

    Your current DRR AP is basically: ap = sum(volume, axis=0) -> (Y,X)
    So the corresponding "backprojection" (Eq.1 conceptually) is to
    replicate the AP image along the summed axis to create a 3D volume.

    Returns:
      V_bp: (Z,Y,X) float32
    """
    ap_img = ap_img.astype(np.float32)
    z, y, x = out_zyx

    if ap_img.shape != (y, x):
        # If someone changes DRR size later, fail loudly.
        raise ValueError(f"Expected ap_img shape (Y,X)=({y},{x}), got {ap_img.shape}")

    if axis != 0:
        raise NotImplementedError("This helper assumes AP sums along Z (axis=0).")

    v = np.repeat(ap_img[None, :, :], repeats=z, axis=0)  # (Z,Y,X)
    return v.astype(np.float32)
