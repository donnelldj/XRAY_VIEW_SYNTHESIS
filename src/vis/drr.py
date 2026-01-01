from __future__ import annotations

import numpy as np


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(np.min(x))
    x = x / (float(np.max(x)) + 1e-8)
    return x


def _maybe_invert(x01: np.ndarray, invert: bool) -> np.ndarray:
    return (1.0 - x01) if invert else x01


def _apply_orient(
    x: np.ndarray,
    rot_k: int = 0,
    flip_ud: bool = False,
    flip_lr: bool = False,
) -> np.ndarray:
    """
    OPTIONAL display/orientation transforms.
    Keep defaults as no-ops for dataset correctness.

    rot_k: rotate by 90 degrees k times (counter-clockwise), k in {0,1,2,3}
    """
    k = int(rot_k) % 4
    if k:
        x = np.rot90(x, k=k)
    if flip_ud:
        x = np.flipud(x)
    if flip_lr:
        x = np.fliplr(x)
    return x


def drr_ap(
    ct_zyx: np.ndarray,
    *,
    invert: bool = True,
    rot_k: int = 0,
    flip_ud: bool = False,
    flip_lr: bool = False,
) -> np.ndarray:
    """
    AP-like DRR from CT volume.

    Input:
      ct_zyx: (Z, Y, X)
        Z: superior-inferior
        Y: anterior-posterior
        X: left-right

    AP projection integrates along Y -> image plane (Z, X).
    """
    ct = ct_zyx.astype(np.float32)
    img = ct.sum(axis=1)  # (Z, X)
    img01 = _normalize01(img)
    img01 = _maybe_invert(img01, invert)
    img01 = _apply_orient(img01, rot_k=rot_k, flip_ud=flip_ud, flip_lr=flip_lr)
    return img01.astype(np.float32)


def drr_lat(
    ct_zyx: np.ndarray,
    *,
    invert: bool = True,
    rot_k: int = 0,
    flip_ud: bool = False,
    flip_lr: bool = False,
) -> np.ndarray:
    """
    LAT-like DRR from CT volume.

    Input:
      ct_zyx: (Z, Y, X)

    LAT projection integrates along X -> image plane (Z, Y).
    """
    ct = ct_zyx.astype(np.float32)
    img = ct.sum(axis=2)  # (Z, Y)
    img01 = _normalize01(img)
    img01 = _maybe_invert(img01, invert)
    img01 = _apply_orient(img01, rot_k=rot_k, flip_ud=flip_ud, flip_lr=flip_lr)
    return img01.astype(np.float32)


def drr_from_ct(
    ct_zyx: np.ndarray,
    view: str,
    *,
    invert: bool = True,
    rot_k: int = 0,
    flip_ud: bool = False,
    flip_lr: bool = False,
) -> np.ndarray:
    """
    Convenience wrapper: view in {"ap","lat"}.
    """
    v = view.lower().strip()
    if v == "ap":
        return drr_ap(ct_zyx, invert=invert, rot_k=rot_k, flip_ud=flip_ud, flip_lr=flip_lr)
    if v == "lat":
        return drr_lat(ct_zyx, invert=invert, rot_k=rot_k, flip_ud=flip_ud, flip_lr=flip_lr)
    raise ValueError(f"view must be 'ap' or 'lat', got: {view!r}")
