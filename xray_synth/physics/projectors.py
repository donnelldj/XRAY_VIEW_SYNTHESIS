from __future__ import annotations

import torch
import torch.nn.functional as F


def backproject_ap_to_volume(ap_zx: torch.Tensor, out_y: int) -> torch.Tensor:
    """
    Backprojection approximation (Eq. 1-ish):
      AP is (B,1,Z,X). Create volume by repeating AP across Y.
    Returns (B,1,Z,Y,X).
    """
    ap_zyx = ap_zx.unsqueeze(3)              # (B,1,Z,1,X)
    vol = ap_zyx.repeat(1, 1, 1, out_y, 1)   # (B,1,Z,Y,X)
    return vol


def avg_pool_latent(vol_zyx: torch.Tensor, down: int) -> torch.Tensor:
    """Downsample (B,1,Z,Y,X) -> (B,1,Zl,Yl,Xl) via avg pooling."""
    if down == 1:
        return vol_zyx
    return F.avg_pool3d(vol_zyx, kernel_size=down, stride=down)


def forward_project_ct_to_lat_export_match(
    ct_zyx: torch.Tensor,
    *,
    clamp01: bool = True,
    export_flipud: bool = True,
    invert: bool = False,
    rot_k: int = 0,
    flip_lr: bool = False,
) -> torch.Tensor:
    """
    Forward projection (Eq. 9-ish) that MATCHES Streamlit exporter GT:

      Streamlit:
        lat = flipud(mean(ct_norm_zyx, axis=X))   # mean over X then flip along Z

    Torch:
      lat = ct.mean(dim=-1)                       # (B,1,Z,Y)
      if export_flipud: flip Z dimension          # torch.flip(..., dims=(-2,))

    Debug transforms exist only for diagnosis; keep defaults for real runs.
    """
    if clamp01:
        ct_zyx = ct_zyx.clamp(0.0, 1.0)

    lat = ct_zyx.mean(dim=-1)  # (B,1,Z,Y)

    if export_flipud:
        lat = torch.flip(lat, dims=(-2,))  # flip Z

    if invert:
        lat = 1.0 - lat

    k = int(rot_k) % 4
    if k:
        lat = torch.rot90(lat, k=k, dims=(-2, -1))  # rotate in (Z,Y)

    if flip_lr:
        lat = torch.flip(lat, dims=(-1,))  # flip Y

    return lat
