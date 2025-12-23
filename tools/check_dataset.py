import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def stats(arr: np.ndarray):
    arr = arr.astype(np.float32)
    return dict(
        shape=list(arr.shape),
        dtype=str(arr.dtype),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        p1=float(np.percentile(arr, 1)),
        p50=float(np.percentile(arr, 50)),
        p99=float(np.percentile(arr, 99)),
        nan=int(np.isnan(arr).sum()),
        inf=int(np.isinf(arr).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to selected_200.csv (contains npz_path column)")
    ap.add_argument("--n", type=int, default=32, help="num samples to probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--normalize", default="minmax01", choices=["minmax01", "meanstd", "none"])
    args = ap.parse_args()

    from src.data.drr_dataset import DRRPairDataset

    ds = DRRPairDataset(args.csv, normalize_mode=args.normalize, return_meta=False)
    N = len(ds)
    print(f"[OK] dataset len: {N}")
    if N == 0:
        raise RuntimeError("Dataset length is 0. Check CSV contents.")

    random.seed(args.seed)
    idxs = [random.randrange(N) for _ in range(min(args.n, N))]

    ap_stats = []
    lat_stats = []
    pair_mae = []

    for k, idx in enumerate(idxs):
        ap_img, lat_img = ds[idx]  # (1,H,W), (1,H,W)
        ap_np = to_np(ap_img).squeeze()
        lat_np = to_np(lat_img).squeeze()

        if ap_np.shape != lat_np.shape:
            raise ValueError(f"Shape mismatch at idx {idx}: AP {ap_np.shape} vs LAT {lat_np.shape}")

        ap_stats.append(stats(ap_np))
        lat_stats.append(stats(lat_np))
        pair_mae.append(float(np.mean(np.abs(ap_np - lat_np))))

        if args.show and k < 8:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(ap_np, cmap="gray")
            ax[0].set_title(f"AP (idx={idx})")
            ax[1].imshow(lat_np, cmap="gray")
            ax[1].set_title("LAT GT")
            ax[2].imshow(np.abs(ap_np - lat_np), cmap="magma")
            ax[2].set_title("|AP-LAT|")
            for a in ax:
                a.axis("off")
            plt.tight_layout()
            plt.show()

    def summarize(all_stats):
        mins = [d["min"] for d in all_stats]
        maxs = [d["max"] for d in all_stats]
        means = [d["mean"] for d in all_stats]
        stds = [d["std"] for d in all_stats]
        nans = sum(d["nan"] for d in all_stats)
        infs = sum(d["inf"] for d in all_stats)
        return {
            "min_range": [float(np.min(mins)), float(np.max(mins))],
            "max_range": [float(np.min(maxs)), float(np.max(maxs))],
            "mean_range": [float(np.min(means)), float(np.max(means))],
            "std_range": [float(np.min(stds)), float(np.max(stds))],
            "nan_total": int(nans),
            "inf_total": int(infs),
        }

    out = {
        "csv": args.csv,
        "normalize_mode": args.normalize,
        "dataset_len": N,
        "ap_summary": summarize(ap_stats),
        "lat_summary": summarize(lat_stats),
        "pair_mae_mean": float(np.mean(pair_mae)),
        "pair_mae_std": float(np.std(pair_mae)),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
