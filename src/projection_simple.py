import torch
import torch.nn.functional as F

# =============================================================================
# Single source of truth for CT->LAT forward projection used in:
# - metrics
# - visualization triplets
#
# Your dataset GT LAT is generated in prepare_dataset.py via:
#   lat_img = drr_lat(ct_norm)
# and saved into each NPZ as "lat".
# =============================================================================

# Choose how we map (Z=96,Y=256) -> (256,256)
#   - "resize": interpolate Z from 96 -> 256 (most likely what your GT uses)
#   - "pad"   : zero-pad Z from 96 -> 256 (your older debug/export style)
LAT_GEOMETRY = "resize"  # "resize" or "pad"

# If LAT_GEOMETRY == "resize", choose interpolation kernel:
# "bilinear", "bicubic", "nearest", "area"
RESIZE_MODE = "bicubic"

# Normalization placement:
#   - "before": normalize after sum, before resize/pad
#   - "after" : resize/pad first, then normalize
NORM_ORDER = "after"

EPS = 1e-8


def _minmax01(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min/max normalize to [0,1] over HxW."""
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + EPS)


def _center_pad_to_256(lat_zy: torch.Tensor) -> torch.Tensor:
    """
    lat_zy: (B,1,Z,Y) -> pad/crop Z to 256, keep Y=256
    returns (B,1,256,256)
    """
    B, C, Z, Y = lat_zy.shape
    assert Y == 256, f"Expected Y=256, got Y={Y}"

    if Z < 256:
        pad = 256 - Z
        pad0 = pad // 2
        pad1 = pad - pad0
        # pad format for 4D (N,C,H,W): (W_left, W_right, H_top, H_bottom)
        lat_zy = F.pad(lat_zy, (0, 0, pad0, pad1), mode="constant", value=0.0)
    elif Z > 256:
        start = (Z - 256) // 2
        lat_zy = lat_zy[:, :, start:start + 256, :]
    return lat_zy


def _resize_to_256(lat_zy: torch.Tensor, mode: str) -> torch.Tensor:
    """
    lat_zy: (B,1,Z,Y) -> (B,1,256,256) by interpolation
    """
    if mode in ("nearest", "area"):
        return F.interpolate(lat_zy, size=(256, 256), mode=mode)
    # bilinear/bicubic
    return F.interpolate(lat_zy, size=(256, 256), mode=mode, align_corners=False)


def forward_project_lat_from_ct(ct_zyx: torch.Tensor) -> torch.Tensor:
    """
    Args:
        ct_zyx: (B,1,Z,Y,X) where your dataset uses Z=96, Y=X=256 (ct_zyx saved as ct_zyx in NPZ)
    Returns:
        lat: (B,1,256,256) in [0,1]
    """
    # Integrate along X (axis=4): (B,1,Z,Y) = (B,1,96,256)
    lat = ct_zyx.sum(dim=4)

    if NORM_ORDER == "before":
        lat = _minmax01(lat)

    if LAT_GEOMETRY == "pad":
        lat = _center_pad_to_256(lat)
    elif LAT_GEOMETRY == "bilinear":
        lat = _resize_to_256(lat, RESIZE_MODE)
    else:
        raise ValueError(f"LAT_GEOMETRY must be 'pad' or 'resize', got {LAT_GEOMETRY}")

    if NORM_ORDER == "after":
        lat = _minmax01(lat)

    return lat
