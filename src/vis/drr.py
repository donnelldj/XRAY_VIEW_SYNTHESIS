import numpy as np
import SimpleITK as sitk

def normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(x.min())
    return x / (float(x.max()) + eps)

def pad_or_crop_center(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape
    y0 = max(0, (h - out_h) // 2)
    x0 = max(0, (w - out_w) // 2)
    y1 = min(h, y0 + out_h)
    x1 = min(w, x0 + out_w)
    cropped = img[y0:y1, x0:x1]

    out = np.zeros((out_h, out_w), dtype=cropped.dtype)
    ch, cw = cropped.shape
    oy = (out_h - ch) // 2
    ox = (out_w - cw) // 2
    out[oy:oy+ch, ox:ox+cw] = cropped
    return out

def drr_ap_from_sitk(img: sitk.Image, out_hw=(256, 256)) -> np.ndarray:
    """
    AP DRR using physical axes from SITK direction.
    We integrate along patient Anterior-Posterior axis ("P/A") to get a frontal projection.
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)
    # Heuristic: for LUNA/LIDC, GetArrayFromImage gives axial stack, so:
    # AP ≈ integrate along Y to get (Z,X), then pad/crop to square
    proj = np.sum(arr, axis=1)  # (Z, X)
    proj = normalize01(proj)
    proj = pad_or_crop_center(proj, out_hw[0], out_hw[1])
    return normalize01(proj)

def drr_lat_from_sitk(img: sitk.Image, out_hw=(256, 256)) -> np.ndarray:
    """
    LAT DRR using physical axes from SITK direction.
    We integrate along patient Left-Right axis to get a lateral projection.
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)
    proj = np.sum(arr, axis=2)  # (Z, Y)
    proj = normalize01(proj)
    proj = pad_or_crop_center(proj, out_hw[0], out_hw[1])
    return normalize01(proj)
