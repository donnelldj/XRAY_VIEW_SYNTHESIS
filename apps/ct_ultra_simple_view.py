from pathlib import Path
import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def load_ct(mhd_path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(mhd_path))
    ct = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)
    return ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhd", required=True, help="Path to .mhd file")
    ap.add_argument("--slice", type=int, default=None, help="Axial slice index")
    args = ap.parse_args()

    ct = load_ct(Path(args.mhd))
    z, y, x = ct.shape

    # pick middle slice by default
    k = args.slice if args.slice is not None else z // 2

    print(f"CT shape (Z,Y,X): {ct.shape}")
    print(f"HU min/max: {ct.min():.1f} / {ct.max():.1f}")
    print(f"Showing axial slice z={k}")

    plt.figure(figsize=(6, 6))
    plt.imshow(ct[k], cmap="gray", vmin=-1000, vmax=400)
    plt.title(f"Axial slice z={k}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()

