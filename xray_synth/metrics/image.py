from __future__ import annotations

import math
import numpy as np


def normalize01(x: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] for visualization only."""
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (PSNR) for images in [0, data_range]."""
    mse = float(np.mean((pred - gt) ** 2))
    if mse < 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim_simple(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """
    Lightweight SSIM proxy (single-scale, global stats). Dependency-free.

    NOTE: This is not windowed SSIM; it is intended as a stable quick proxy.
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
