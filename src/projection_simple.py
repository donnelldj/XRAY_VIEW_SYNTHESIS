import torch

def forward_project_lat_from_ct(ct_zyx: torch.Tensor) -> torch.Tensor:
    """
    ct_zyx: (B,1,Z,Y,X) in 0..1
    returns lat: (B,1,256,256) in 0..1
    Simple DRR: sum along X axis => (Z,Y) then padded/cropped to (256,256)
    """
    # Sum along X -> (B,1,Z,Y)
    lat = ct_zyx.sum(dim=4)

    # Normalize per-sample to 0..1
    lat_min = lat.amin(dim=(2,3), keepdim=True)
    lat_max = lat.amax(dim=(2,3), keepdim=True)
    lat = (lat - lat_min) / (lat_max - lat_min + 1e-8)

    # lat is (B,1,Z,Y) where Z=96, Y=256 -> pad Z to 256
    B, C, Z, Y = lat.shape
    if Z < 256:
        pad = 256 - Z
        pad0 = pad // 2
        pad1 = pad - pad0
        lat = torch.nn.functional.pad(lat, (0, 0, pad0, pad1), mode="constant", value=0.0)
    elif Z > 256:
        start = (Z - 256)//2
        lat = lat[:, :, start:start+256, :]

    # Now (B,1,256,256)
    return lat
