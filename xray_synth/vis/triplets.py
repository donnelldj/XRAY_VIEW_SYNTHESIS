from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from xray_synth.metrics.image import normalize01


def save_triplet(ap_2d: np.ndarray, lat_pred_2d: np.ndarray, lat_gt_2d: np.ndarray, out_path: Path) -> None:
    """
    Save a side-by-side PNG:
      [ AP | Pred LAT | GT LAT ]
    """
    ap = (normalize01(ap_2d) * 255.0).astype(np.uint8)
    pr = (normalize01(lat_pred_2d) * 255.0).astype(np.uint8)
    gt = (normalize01(lat_gt_2d) * 255.0).astype(np.uint8)

    ap_im = Image.fromarray(ap, mode="L")
    pr_im = Image.fromarray(pr, mode="L")
    gt_im = Image.fromarray(gt, mode="L")

    w, h = ap_im.size
    canvas = Image.new("L", (w * 3, h))
    canvas.paste(ap_im, (0, 0))
    canvas.paste(pr_im, (w, 0))
    canvas.paste(gt_im, (w * 2, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
