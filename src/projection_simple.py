# src/projection_simple.py
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8

# How to map (Z,Y) -> (256,256) when Z != 256:
#   "resize" : interpolate to 256x256
#   "pad"    : center pad/crop Z to 256 (keeps Y fixed, assumes Y==256)
LAT_GEOMETRY = "resize"   # "resize" or "pad"
RESIZE_MODE = "bicubic"   # "bilinear" | "bicubic" | "nearest" | "area"
NORM_ORDER = "after"      # "before" or "after"


def _minmax01(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min/max normalize to [0,1] over HxW (dims 2,3)."""
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + EPS)


def _center_pad_or_crop_z(lat_zy: torch.Tensor, target_z: int = 256) -> torch.Tensor:
    """
    lat_zy: (B,1,Z,Y) -> (B,1,target_z,Y) via center pad/crop on Z.
    """
    B, C, Z, Y = lat_zy.shape
    if Z < target_z:
        pad = target_z - Z
        pad0 = pad // 2
        pad1 = pad - pad0
        # Pad format for NCHW: (W_left, W_right, H_top, H_bottom)
        return F.pad(lat_zy, (0, 0, pad0, pad1), mode="constant", value=0.0)
    if Z > target_z:
        start = (Z - target_z) // 2
        return lat_zy[:, :, start:start + target_z, :]
    return lat_zy


def _resize_to_256(lat_zy: torch.Tensor, mode: str) -> torch.Tensor:
    """
    lat_zy: (B,1,Z,Y) -> (B,1,256,256)
    """
    if mode in ("nearest", "area"):
        return F.interpolate(lat_zy, size=(256, 256), mode=mode)
    return F.interpolate(lat_zy, size=(256, 256), mode=mode, align_corners=False)


def project_lat(ct_zyx: torch.Tensor) -> torch.Tensor:
    """
    Forward project LAT from CT volume (dataset convention):
      ct_zyx: (B,1,Z,Y,X)
      LAT = sum over X -> (B,1,Z,Y)

    Returns:
      lat_256: (B,1,256,256) in [0,1]
    """
    if ct_zyx.ndim != 5:
        raise ValueError(f"Expected ct_zyx as (B,1,Z,Y,X). Got shape={tuple(ct_zyx.shape)}")

    # (B,1,Z,Y)
    lat_zy = ct_zyx.sum(dim=4)

    if NORM_ORDER == "before":
        lat_zy = _minmax01(lat_zy)

    Z = int(lat_zy.shape[2])
    Y = int(lat_zy.shape[3])

    # If already 256x256, keep it clean.
    if Z == 256 and Y == 256:
        lat_256 = lat_zy
    else:
        if LAT_GEOMETRY == "pad":
            # pad/crop Z -> 256, then resize Y if needed
            lat_pad = _center_pad_or_crop_z(lat_zy, target_z=256)
            if int(lat_pad.shape[3]) != 256:
                lat_256 = _resize_to_256(lat_pad, mode=RESIZE_MODE)
            else:
                lat_256 = lat_pad
        elif LAT_GEOMETRY == "resize":
            lat_256 = _resize_to_256(lat_zy, mode=RESIZE_MODE)
        else:
            raise ValueError(f"LAT_GEOMETRY must be 'pad' or 'resize', got {LAT_GEOMETRY!r}")

    if NORM_ORDER == "after":
        lat_256 = _minmax01(lat_256)

    return lat_256
