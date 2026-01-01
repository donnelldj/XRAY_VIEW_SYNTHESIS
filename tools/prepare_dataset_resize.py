from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    return [CaseInfo(case_id=r["case_id"], mhd_path=r["mhd_path"]) for r in payload]


def read_ct_sitk(mhd_path: str) -> sitk.Image:
    return sitk.ReadImage(mhd_path)


def resample_to_size(img: sitk.Image, out_size_xyz: Tuple[int, int, int]) -> sitk.Image:
    """
    Resize (resample) to a target voxel grid size (X,Y,Z) while preserving physical extent.
    No cropping.
    """
    in_size = np.array(list(img.GetSize()), dtype=np.float64)         # (X,Y,Z)
    in_spacing = np.array(list(img.GetSpacing()), dtype=np.float64)   # (sx,sy,sz)
    out_size = np.array(list(out_size_xyz), dtype=np.int64)

    extent = in_size * in_spacing
    out_spacing = extent / np.maximum(out_size.astype(np.float64), 1.0)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(x) for x in out_size.tolist()])
    resampler.SetOutputSpacing([float(x) for x in out_spacing.tolist()])
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024.0)
    return resampler.Execute(img)


def sitk_to_np_zyx(img: sitk.Image) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Returns:
      ct_zyx: (Z,Y,X) float32
      spacing_zyx: (sz,sy,sx) mm
    """
    spacing_xyz = img.GetSpacing()  # (sx,sy,sz)
    ct_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    return ct_zyx, spacing_zyx


def normalize_ct_hu(ct: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    v = np.clip(ct, hu_min, hu_max)
    v = (v - hu_min) / (hu_max - hu_min + 1e-6)
    return v.astype(np.float32)


def safe_filename(case_id: str) -> str:
    return case_id.replace("/", "__").replace("\\", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected_json", type=str, default="data/selected_10.json")
    ap.add_argument("--out_dir", type=str, required=True)

    # Target RESIZE volume shape (Z,Y,X) expressed as args (X,Y,Z) for SITK
    ap.add_argument("--size_x", type=int, default=256)
    ap.add_argument("--size_y", type=int, default=256)
    ap.add_argument("--size_z", type=int, default=96)

    ap.add_argument("--hu_min", type=float, default=-1000.0)
    ap.add_argument("--hu_max", type=float, default=400.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_ct", action="store_true")
    args = ap.parse_args()

    selected_json = Path(args.selected_json)
    if not selected_json.exists():
        raise SystemExit(f"[ERROR] selected_json not found: {selected_json}")

    out_dir = Path(args.out_dir)
    out_npz = out_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz.mkdir(parents=True, exist_ok=True)

    cases = read_selected(selected_json)
    if not cases:
        raise SystemExit(f"[ERROR] No cases in {selected_json}")

    target_xyz = (args.size_x, args.size_y, args.size_z)

    rows: List[Dict[str, object]] = []
    written = 0
    skipped = 0

    print(f"[prepare_dataset_resize] cases={len(cases)} out={out_dir}")
    print(f"[prepare_dataset_resize] RESIZE to (X,Y,Z)={target_xyz} hu=({args.hu_min},{args.hu_max})")

    for i, c in enumerate(cases, start=1):
        mhd_path = Path(c.mhd_path)
        if not mhd_path.exists():
            print(f"[WARN] missing: {c.mhd_path} (skipping)")
            skipped += 1
            continue

        img = read_ct_sitk(str(mhd_path))
        img_rs = resample_to_size(img, out_size_xyz=target_xyz)

        ct_zyx, spacing_zyx = sitk_to_np_zyx(img_rs)  # (Z,Y,X)
        ct_norm = normalize_ct_hu(ct_zyx, args.hu_min, args.hu_max)

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
            save_kwargs["ct_zyx"] = ct_norm.astype(np.float16)

        np.savez_compressed(npz_path, **save_kwargs)

        rows.append(
            dict(
                case_id=c.case_id,
                npz_path=str(npz_path.as_posix()),
                mhd_path=str(mhd_path.as_posix()),
                z=int(ct_norm.shape[0]),
                y=int(ct_norm.shape[1]),
                x=int(ct_norm.shape[2]),
                spacing_z=float(spacing_zyx[0]),
                spacing_y=float(spacing_zyx[1]),
                spacing_x=float(spacing_zyx[2]),
            )
        )

        written += 1
        if written % 10 == 0 or i == len(cases):
            print(f"  wrote {written}/{len(cases)} (seen={i}, skipped={skipped})")

    if written == 0:
        raise SystemExit("[ERROR] Wrote 0 cases. selected_200.json paths not valid on this machine.")

    df = pd.DataFrame(rows)
    manifest_csv = out_dir / "manifest.csv"
    df.to_csv(manifest_csv, index=False)

    rng = random.Random(args.seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)

    n_val = max(1, int(round(len(df) * float(args.val_frac))))
    val_set = set(idxs[:n_val])

    df["split"] = ["val" if i in val_set else "train" for i in range(len(df))]

    (out_dir / "train.csv").write_text(df[df["split"] == "train"].to_csv(index=False), encoding="utf-8")
    (out_dir / "val.csv").write_text(df[df["split"] == "val"].to_csv(index=False), encoding="utf-8")

    print(f"[prepare_dataset_resize] manifest -> {manifest_csv} rows={len(df)}")
    print("[prepare_dataset_resize] done")


if __name__ == "__main__":
    main()
