from __future__ import annotations

import os
import numpy as np
import SimpleITK as sitk
import streamlit as st

from xray_synth.utils.sitk_resample import resample_to_fixed_cube_preserve_extent

def normalize_ct_hu(ct_hu_zyx: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    v = np.clip(ct_hu_zyx, hu_min, hu_max)
    v = (v - hu_min) / (hu_max - hu_min + 1e-6)
    return v.astype(np.float32)

@st.cache_data
def load_original_volume(path: str):
    itk_img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(itk_img), itk_img

def resample_volume(itk_img: sitk.Image, hu_min: float, hu_max: float):
    """
    Resamples volume to fixed 256³ cube preserving physical extent and generates AP/LAT projections.

    Returns:
      ct_hu_zyx:    (256,256,256) float32 HU
      ct_norm_zyx:  (256,256,256) float32 in [0,1]
      xray_ap:      (256,256) float32
      xray_lat:     (256,256) float32
      spacing_zyx:  (sz,sy,sx) float32 AFTER resample
    """
    itk_img_resampled = resample_to_fixed_cube_preserve_extent(itk_img, out_size_xyz=(256, 256, 256))

    # HU volume (Z,Y,X)
    ct_hu_zyx = sitk.GetArrayFromImage(itk_img_resampled).astype(np.float32)

    # spacing AFTER resample (SITK: (sx,sy,sz))
    sx, sy, sz = itk_img_resampled.GetSpacing()
    spacing_zyx = np.array([sz, sy, sx], dtype=np.float32)

    # Normalize HU -> [0,1] for pipeline consistency
    ct_norm_zyx = normalize_ct_hu(ct_hu_zyx, hu_min=float(hu_min), hu_max=float(hu_max))

    # Projection convention (GT CONTRACT):
    # - AP  = mean over Y  -> (Z,X), then flipud over Z
    # - LAT = mean over X  -> (Z,Y), then flipud over Z
    attenuation_data = ct_norm_zyx
    xray_ap  = np.flipud(np.mean(attenuation_data, axis=1)).astype(np.float32)  # (Z,X)
    xray_lat = np.flipud(np.mean(attenuation_data, axis=2)).astype(np.float32)  # (Z,Y)

    return ct_hu_zyx, ct_norm_zyx, xray_ap, xray_lat, spacing_zyx

def abs_path(p: str) -> str:
    return os.path.abspath(p)
