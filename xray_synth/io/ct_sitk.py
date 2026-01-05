from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import SimpleITK as sitk
    _HAS_SITK = True
except Exception:
    sitk = None  # type: ignore
    _HAS_SITK = False


def has_sitk() -> bool:
    return _HAS_SITK


def resample_to_size_preserve_extent(img: "sitk.Image", out_size_xyz: Tuple[int, int, int]) -> "sitk.Image":
    """
    Resample to target voxel grid size (X,Y,Z) while preserving physical extent.

    This matches the Streamlit exporter behavior:
    - keep origin + direction
    - adjust spacing so physical extent stays constant
    - linear interpolation
    - default background HU
    """
    in_size = np.array(list(img.GetSize()), dtype=np.float64)           # (X,Y,Z)
    in_spacing = np.array(list(img.GetSpacing()), dtype=np.float64)     # (sx,sy,sz)
    out_size = np.array(list(out_size_xyz), dtype=np.int64)

    extent = in_size * in_spacing
    out_spacing = extent / np.maximum(out_size.astype(np.float64), 1.0)

    r = sitk.ResampleImageFilter()
    r.SetSize([int(x) for x in out_size.tolist()])
    r.SetOutputSpacing([float(x) for x in out_spacing.tolist()])
    r.SetOutputOrigin(img.GetOrigin())
    r.SetOutputDirection(img.GetDirection())
    r.SetTransform(sitk.Transform())
    r.SetInterpolator(sitk.sitkLinear)
    r.SetDefaultPixelValue(-1024.0)
    return r.Execute(img)


def read_ct_preprocessed_zyx(
    mhd_path: str,
    target_zyx: Tuple[int, int, int] = (256, 256, 256),
    hu_clip: Tuple[float, float] = (-1000.0, 400.0),
) -> np.ndarray:
    """
    Fallback CT load:
      Read -> resample preserve extent -> GetArray(Z,Y,X) -> HU clip -> normalize [0,1]
    """
    if not _HAS_SITK:
        raise RuntimeError("SimpleITK is required for CT fallback. Install: pip install SimpleITK")

    img = sitk.ReadImage(str(mhd_path))
    tz, ty, tx = target_zyx
    img = resample_to_size_preserve_extent(img, out_size_xyz=(tx, ty, tz))  # expects (X,Y,Z)

    ct_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)

    lo, hi = hu_clip
    ct_zyx = np.clip(ct_zyx, lo, hi)
    ct_zyx = (ct_zyx - lo) / (hi - lo + 1e-8)
    return ct_zyx.astype(np.float32)
