# scripts/luna_smoke_test.py
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import kagglehub
import SimpleITK as sitk

from src.projection import normalize_hu, project_ap, project_lat


def find_first_mhd(root: Path) -> Path:
    # LUNA16 is typically subset0..subset9, each contains .mhd + .raw
    mhd_files = list(root.rglob("*.mhd"))
    if not mhd_files:
        raise FileNotFoundError(f"No .mhd files found under: {root}")
    return mhd_files[0]


def load_luna_ct(mhd_path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(mhd_path))
    vol = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    return vol.astype(np.float32)


def main():
    # 1) Download
    data_root = Path(kagglehub.dataset_download("avc0706/luna16"))
    print(f"[kagglehub] LUNA16 root: {data_root}")

    # 2) Pick one volume
    mhd_path = find_first_mhd(data_root)
    print(f"[data] Using volume: {mhd_path}")

    # 3) Load CT
    ct = load_luna_ct(mhd_path)
    print(f"[ct] shape={ct.shape}, dtype={ct.dtype}, min={ct.min():.1f}, max={ct.max():.1f}")

    # 4) Normalize + project
    ct = normalize_hu(ct)
    ap = project_ap(ct)
    lat = project_lat(ct)

    # 5) Show
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("AP")
    plt.imshow(ap, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("LAT")
    plt.imshow(lat, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
