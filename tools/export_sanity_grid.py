from __future__ import annotations

# --- IMPORTANT: allow imports from repo root (src/) ---------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (xray_view_synthesis)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------------------------------------

import argparse

import numpy as np
from PIL import Image

from src.data.drr_dataset import DRRPairDataset


def to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


def save_png(path: Path, img01: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(to_uint8(img01), mode="L")
    im.save(path)


def make_triplet_strip(ap01: np.ndarray, pred01: np.ndarray, gt01: np.ndarray) -> np.ndarray:
    # concat horizontally: (H, 3W)
    return np.concatenate([ap01, pred01, gt01], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/drr_pairs/val.csv")
    ap.add_argument("--out_dir", type=str, default="results/sanity")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset returns normalized 0..1 by default
    ds = DRRPairDataset(args.csv, normalize_mode="minmax01", return_meta=True)

    # Baseline predictor: predicted lat = ap (identity)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist()

    strips = []
    for k, idx in enumerate(idxs):
        ap_t, gt_t, meta = ds[idx]  # (1,H,W)
        ap01 = ap_t[0].numpy()
        gt01 = gt_t[0].numpy()

        pred01 = ap01.copy()

        case_id = meta["case_id"].replace("/", "__").replace("\\", "__")
        save_png(out_dir / f"{k:02d}_{case_id}_ap.png", ap01)
        save_png(out_dir / f"{k:02d}_{case_id}_pred.png", pred01)
        save_png(out_dir / f"{k:02d}_{case_id}_gt.png", gt01)

        strips.append(make_triplet_strip(ap01, pred01, gt01))

    grid = np.concatenate(strips, axis=0)
    save_png(out_dir / "sanity_grid_AP_PRED_GT.png", grid)

    print(f"[export_sanity_grid] wrote {len(strips)} triplets to: {out_dir}")
    print(f"[export_sanity_grid] grid: {out_dir / 'sanity_grid_AP_PRED_GT.png'}")
    print("Columns are: AP | PRED (baseline) | GT LAT")


if __name__ == "__main__":
    main()
