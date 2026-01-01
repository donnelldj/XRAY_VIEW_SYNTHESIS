import numpy as np


def normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(np.min(x))
    d = float(np.max(x)) + eps
    return x / d


def _resize_nearest(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Nearest-neighbor resize (no extra deps).
    img: (H, W)
    returns: (out_h, out_w)
    """
    h, w = img.shape
    yy = (np.linspace(0, h - 1, out_h)).astype(np.int32)
    xx = (np.linspace(0, w - 1, out_w)).astype(np.int32)
    return img[np.ix_(yy, xx)]


def drr_ap(ct_zyx: np.ndarray) -> np.ndarray:
    """
    AP-ish DRR:
      Sum along Z -> (Y, X)
    ct_zyx: (Z, Y, X)
    """
    proj = np.sum(ct_zyx, axis=0)  # (Y, X)
    return normalize01(proj)


def drr_lat(ct_zyx: np.ndarray) -> np.ndarray:
    """
    LAT-ish DRR (fixed to match AP shape):
      1) Sum along Y -> (Z, X)  (a side view)
      2) Resize to (Y, X) so it matches AP dimensions for training

    ct_zyx: (Z, Y, X)
    returns: (Y, X)
    """
    z, y, x = ct_zyx.shape

    # Side projection: collapse Y
    proj_zx = np.sum(ct_zyx, axis=1)  # (Z, X)
    proj_zx = normalize01(proj_zx)

    # Resize (Z, X) -> (Y, X) for consistent supervision target
    proj_yx = _resize_nearest(proj_zx, out_h=y, out_w=x)  # (Y, X)
    return normalize01(proj_yx)
