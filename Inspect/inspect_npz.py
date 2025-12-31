import glob
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


def minmax01_torch(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + EPS)


def center_pad_to_256(lat_zy: torch.Tensor) -> torch.Tensor:
    B, C, Z, Y = lat_zy.shape
    if Y != 256:
        raise ValueError(f"Expected Y=256, got {Y}")

    if Z < 256:
        pad = 256 - Z
        pad0 = pad // 2
        pad1 = pad - pad0
        lat_zy = F.pad(lat_zy, (0, 0, pad0, pad1), mode="constant", value=0.0)
    elif Z > 256:
        start = (Z - 256) // 2
        lat_zy = lat_zy[:, :, start:start + 256, :]
    return lat_zy


def resize_to_256(lat_zy: torch.Tensor, mode: str) -> torch.Tensor:
    if mode in ("nearest", "area"):
        return F.interpolate(lat_zy, size=(256, 256), mode=mode)
    return F.interpolate(lat_zy, size=(256, 256), mode=mode, align_corners=False)


def project_variant(ct_t: torch.Tensor, geometry: str, resize_mode: str, norm_order: str) -> torch.Tensor:
    """
    ct_t: (1,1,96,256,256)
    returns lat: (1,1,256,256)
    """
    lat = ct_t.sum(dim=4)  # (1,1,96,256)

    if norm_order == "before":
        lat = minmax01_torch(lat)

    if geometry == "pad":
        lat = center_pad_to_256(lat)
    elif geometry == "resize":
        lat = resize_to_256(lat, resize_mode)
    else:
        raise ValueError("geometry must be 'pad' or 'resize'")

    if norm_order == "after":
        lat = minmax01_torch(lat)

    return lat


def main():
    npz_files = sorted(glob.glob(r"data/drp_pairs/npz/*.npz"))
    if not npz_files:
        raise FileNotFoundError("No NPZ files found in data/drp_pairs/npz/*.npz")

    p = npz_files[0]
    d = np.load(p, allow_pickle=True)

    if "ct_zyx" not in d.files:
        raise KeyError(f"'ct_zyx' not found in NPZ keys: {d.files}")
    if "lat" not in d.files:
        raise KeyError(f"'lat' not found in NPZ keys: {d.files}")

    ct = d["ct_zyx"].astype(np.float32)   # (96,256,256)
    lat_gt = d["lat"].astype(np.float32)  # (256,256)

    ct_t = torch.from_numpy(ct)[None, None]         # (1,1,96,256,256)
    lat_gt_t = torch.from_numpy(lat_gt)[None, None] # (1,1,256,256)

    geometries = ["resize", "pad"]
    resize_modes = ["bilinear", "bicubic", "nearest", "area"]
    norm_orders = ["after", "before"]

    best = None

    print(f"NPZ: {p}")
    print(f"ct: {ct.shape} lat_gt: {lat_gt.shape}")
    print(f"lat_gt min/max: {float(lat_gt.min())} {float(lat_gt.max())}")
    print("")

    with torch.no_grad():
        for g in geometries:
            for ro in norm_orders:
                for rm in resize_modes:
                    # rm only matters for resize; for pad we still print it for uniformity
                    lat_pred_t = project_variant(ct_t, geometry=g, resize_mode=rm, norm_order=ro)

                    diff = lat_pred_t - lat_gt_t
                    mse = float((diff * diff).mean().item())
                    mae = float(diff.abs().mean().item())

                    tag = f"[{g:6s} | {rm:8s} | norm={ro:5s}]"
                    print(f"{tag}  MSE={mse:.6f}  MAE={mae:.6f}")

                    if best is None or mse < best["mse"]:
                        best = {"geometry": g, "resize_mode": rm, "norm_order": ro, "mse": mse, "mae": mae}

    print("\nBEST SETTINGS TO COPY INTO src/projection_simple.py:")
    print(f'LAT_GEOMETRY = "{best["geometry"]}"')
    print(f'RESIZE_MODE  = "{best["resize_mode"]}"')
    print(f'NORM_ORDER   = "{best["norm_order"]}"')
    print(f'MSE={best["mse"]:.6f}  MAE={best["mae"]:.6f}')


if __name__ == "__main__":
    main()
