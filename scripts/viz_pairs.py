import glob
import random
import numpy as np
import matplotlib.pyplot as plt

def show_one(npz_path: str):
    d = np.load(npz_path)
    ap = d["ap"]
    lat = d["lat"]
    ct = d["ct_zyx"] if "ct_zyx" in d.files else None

    plt.figure()
    plt.imshow(ap, cmap="gray")
    plt.title("AP")
    plt.axis("off")

    plt.figure()
    plt.imshow(lat, cmap="gray")
    plt.title("LAT")
    plt.axis("off")

    if ct is not None:
        z, y, x = ct.shape
        mids = (z//2, y//2, x//2)

        plt.figure()
        plt.imshow(ct[mids[0]], cmap="gray")
        plt.title("CT axial (Z mid)  ct[z,:,:]")
        plt.axis("off")

        plt.figure()
        plt.imshow(ct[:, mids[1], :], cmap="gray")
        plt.title("CT coronal (Y mid)  ct[:,y,:]")
        plt.axis("off")

        plt.figure()
        plt.imshow(ct[:, :, mids[2]], cmap="gray")
        plt.title("CT sagittal (X mid)  ct[:,:,x]")
        plt.axis("off")

    plt.show()

def main():
    npzs = glob.glob("runs/drr_debug/npz/*.npz") + glob.glob("data/drr_pairs/npz/*.npz")
    if not npzs:
        raise SystemExit("No npz files found. Check your out_dir.")
    p = random.choice(npzs)
    print("showing:", p)
    show_one(p)

if __name__ == "__main__":
    main()
