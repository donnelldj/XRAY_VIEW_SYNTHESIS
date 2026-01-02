import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def show_one(npz_path: str, save_dir: Path):
    d = np.load(npz_path)
    ap = d["ap"]
    lat = d["lat"]
    ct = d["ct_zyx"] if "ct_zyx" in d.files else None

    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------- AP ----------
    plt.figure(figsize=(4, 4))
    plt.imshow(ap, cmap="gray")
    plt.title("AP")
    plt.axis("off")
    plt.savefig(save_dir / "ap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- LAT ----------
    plt.figure(figsize=(4, 4))
    plt.imshow(lat, cmap="gray")
    plt.title("LAT")
    plt.axis("off")
    plt.savefig(save_dir / "lat.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- CT sanity views ----------
    if ct is not None:
        z, y, x = ct.shape
        mz, my, mx = z // 2, y // 2, x // 2

        # Axial (Z)
        plt.figure(figsize=(4, 4))
        plt.imshow(ct[mz, :, :], cmap="gray")
        plt.title("CT axial  ct[z,:,:]")
        plt.axis("off")
        plt.savefig(save_dir / "ct_axial.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Coronal (Y)
        plt.figure(figsize=(4, 4))
        plt.imshow(ct[:, my, :], cmap="gray")
        plt.title("CT coronal  ct[:,y,:]")
        plt.axis("off")
        plt.savefig(save_dir / "ct_coronal.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Sagittal (X)
        plt.figure(figsize=(4, 4))
        plt.imshow(ct[:, :, mx], cmap="gray")
        plt.title("CT sagittal  ct[:,:,x]")
        plt.axis("off")
        plt.savefig(save_dir / "ct_sagittal.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    npzs = glob.glob("runs/final_runs_1/data/drr_pairs_fixed/npz/subset0__1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.npz")
    if not npzs:
        raise SystemExit("No npz files found. Check your out_dir.")

    p = random.choice(npzs)
    print("showing:", p)

    case_id = Path(p).stem
    out_dir = Path("runs/debug_viz") / case_id

    show_one(p, out_dir)
    print("saved debug images to:", out_dir.resolve())


if __name__ == "__main__":
    main()
