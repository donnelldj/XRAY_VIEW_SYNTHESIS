from __future__ import annotations

# --- IMPORTANT: allow imports from repo root (src/) ---------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (xray_view_synthesis)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------------------------------------

import argparse
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.vis.drr import drr_ap, drr_lat


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    mhd_path: str


def read_selected(json_path: Path) -> List[CaseInfo]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    cases: List[CaseInfo] = []
    for row in payload:
        cases.append(CaseInfo(case_id=row["case_id"], mhd_path=row["mhd_path"]))
    return cases


def read_ct(mhd_path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Returns:
      ct_zyx: (Z,Y,X) float32
      spacing_zyx: (Z,Y,X) mm
    """
    img = sitk.ReadImage(mhd_path)
    spacing_xyz = img.GetSpacing()  # (X,Y,Z)
    ct_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    return ct_zyx, spacing_zyx


def center_crop_zyx(ct: np.ndarray, target_zyx: Tuple[int, int, int]) -> np.ndarray:
    """
    Center-crop / pad to target shape (Z,Y,X). Pads with zeros if smaller.
    """
    zt, yt, xt = target_zyx
    z, y, x = ct.shape

    cz, cy, cx = z // 2, y // 2, x // 2

    z0 = max(0, cz - zt // 2)
    y0 = max(0, cy - yt // 2)
    x0 = max(0, cx - xt // 2)

    z1 = min(z, z0 + zt)
    y1 = min(y, y0 + yt)
    x1 = min(x, x0 + xt)

    cropped = ct[z0:z1, y0:y1, x0:x1]

    out = np.zeros((zt, yt, xt), dtype=np.float32)
    zz, yy, xx = cropped.shape
    out[:zz, :yy, :xx] = cropped
    return out


def normalize_ct_hu(ct: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
    """
    Simple HU window -> [0,1].
    """
    v = np.clip(ct, hu_min, hu_max)
    v = (v - hu_min) / (hu_max - hu_min + 1e-6)
    return v.astype(np.float32)


def safe_filename(case_id: str) -> str:
    # "subset0/1.2.3" -> "subset0__1.2.3"
    return case_id.replace("/", "__").replace("\\", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected_json", type=str, default="data/selected_200.json")
    ap.add_argument("--out_dir", type=str, default="data/drr_pairs")
    ap.add_argument("--crop_z", type=int, default=96)
    ap.add_argument("--crop_y", type=int, default=256)
    ap.add_argument("--crop_x", type=int, default=256)
    ap.add_argument("--hu_min", type=float, default=-1000.0)
    ap.add_argument("--hu_max", type=float, default=400.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save_ct",
        action="store_true",
        help="Also save the cropped CT volume (normalized) in each npz (bigger files).",
    )
    args = ap.parse_args()

    selected_json = Path(args.selected_json)
    if not selected_json.exists():
        raise SystemExit(f"[ERROR] selected_json not found: {selected_json}")

    out_dir = Path(args.out_dir)
    out_npz = out_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz.mkdir(parents=True, exist_ok=True)

    cases = read_selected(selected_json)
    if len(cases) == 0:
        raise SystemExit(f"[ERROR] No cases found in {selected_json}")

    target_zyx = (args.crop_z, args.crop_y, args.crop_x)

    rows: List[Dict[str, object]] = []
    print(f"[prepare_dataset] cases={len(cases)} out={out_dir}")
    print(f"[prepare_dataset] crop(Z,Y,X)={target_zyx} hu=({args.hu_min},{args.hu_max}) val_frac={args.val_frac}")

    written = 0
    skipped = 0

    for i, c in enumerate(cases, start=1):
        mhd_path = Path(c.mhd_path)
        if not mhd_path.exists():
            print(f"[WARN] missing: {c.mhd_path} (skipping)")
            skipped += 1
            continue

        ct, spacing_zyx = read_ct(str(mhd_path))
        ct_crop = center_crop_zyx(ct, target_zyx=target_zyx)
        ct_norm = normalize_ct_hu(ct_crop, hu_min=args.hu_min, hu_max=args.hu_max)

        # DRR projections (your src/vis/drr.py functions)
        ap_img = drr_ap(ct_norm).astype(np.float32)
        lat_img = drr_lat(ct_norm).astype(np.float32)

        fn = safe_filename(c.case_id) + ".npz"
        npz_path = out_npz / fn

        save_kwargs = dict(
            case_id=c.case_id,
            mhd_path=str(mhd_path.as_posix()),
            spacing_zyx=np.array(spacing_zyx, dtype=np.float32),
            ap=ap_img,
            lat=lat_img,
        )
        if args.save_ct:
            save_kwargs["ct_zyx"] = ct_norm.astype(np.float16)  # smaller than float32

        np.savez_compressed(npz_path, **save_kwargs)

        rows.append(
            {
                "case_id": c.case_id,
                "npz_path": str(npz_path.as_posix()),
                "mhd_path": str(mhd_path.as_posix()),
                "z": int(target_zyx[0]),
                "y": int(target_zyx[1]),
                "x": int(target_zyx[2]),
                "spacing_z": float(spacing_zyx[0]),
                "spacing_y": float(spacing_zyx[1]),
                "spacing_x": float(spacing_zyx[2]),
            }
        )

        written += 1
        if written % 10 == 0 or i == len(cases):
            print(f"  wrote {written}/{len(cases)} (seen={i}, skipped={skipped})")

    if written == 0:
        raise SystemExit("[ERROR] Wrote 0 cases. Check that paths inside selected_200.json are valid on this machine.")

    df = pd.DataFrame(rows)
    manifest_csv = out_dir / "manifest.csv"
    df.to_csv(manifest_csv, index=False)
    print(f"[prepare_dataset] manifest -> {manifest_csv} rows={len(df)}")

    # Train/val split
    rng = random.Random(args.seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)

    n_val = max(1, int(round(len(df) * float(args.val_frac))))
    val_set = set(idxs[:n_val])

    df_split = df.copy()
    df_split["split"] = ["val" if i in val_set else "train" for i in range(len(df))]

    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    df_split[df_split["split"] == "train"].to_csv(train_csv, index=False)
    df_split[df_split["split"] == "val"].to_csv(val_csv, index=False)

    print(f"[prepare_dataset] train -> {train_csv} ({(df_split['split']=='train').sum()})")
    print(f"[prepare_dataset] val   -> {val_csv} ({(df_split['split']=='val').sum()})")
    print("[prepare_dataset] done")


if __name__ == "__main__":
    main()
