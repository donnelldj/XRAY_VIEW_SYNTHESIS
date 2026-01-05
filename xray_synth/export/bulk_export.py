from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import streamlit as st

from xray_synth.processing.volume import resample_volume, abs_path
from xray_synth.utils.image_utils import save_16bit_png

def run_randomized_export(
    mhd_files: list[str],
    export_limit: int,
    export_folder: str,
    val_frac: float,
    seed: int,
    hu_min: float,
    hu_max: float,
    save_ct: bool,
) -> None:
    export_list = random.sample(mhd_files, int(export_limit))

    export_root = Path(export_folder)
    npz_dir = export_root / "npz_registry"
    audit_root = export_root / "audit_samples"
    npz_dir.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    rows = []
    n = int(export_limit)
    rng = random.Random(int(seed))

    for i, target_path in enumerate(export_list):
        p_id = os.path.basename(target_path).replace(".mhd", "")
        status_text.text(f"Processing ({i+1}/{n}): {p_id}")

        v_orig_itk = sitk.ReadImage(target_path)
        ct_hu, ct_norm, ap_img, lat_img, spacing_zyx = resample_volume(
            v_orig_itk, hu_min=float(hu_min), hu_max=float(hu_max)
        )

        npz_path = npz_dir / f"{p_id}.npz"

        save_kwargs = dict(
            case_id=p_id,
            mhd_path=abs_path(target_path),
            size_zyx=np.array(ct_norm.shape, dtype=np.int32),   # (256,256,256)
            spacing_zyx=spacing_zyx,                            # AFTER resample
            hu_min=np.float32(hu_min),
            hu_max=np.float32(hu_max),
            ap=ap_img.astype(np.float32),                       # (256,256)
            lat=lat_img.astype(np.float32),                     # (256,256)
        )
        if save_ct:
            save_kwargs["ct_zyx_norm"] = ct_norm.astype(np.float16)

        np.savez_compressed(str(npz_path), **save_kwargs)

        rows.append(
            dict(
                case_id=p_id,
                npz_path=str(npz_path.as_posix()),
                mhd_path=str(Path(target_path).as_posix()),
                z=int(ct_norm.shape[0]),
                y=int(ct_norm.shape[1]),
                x=int(ct_norm.shape[2]),
                spacing_z=float(spacing_zyx[0]),
                spacing_y=float(spacing_zyx[1]),
                spacing_x=float(spacing_zyx[2]),
            )
        )

        # Audits (first 10)
        if i < 10:
            case_audit_dir = audit_root / p_id
            case_audit_dir.mkdir(parents=True, exist_ok=True)

            save_16bit_png(ap_img, str(case_audit_dir / "xray_ap.png"))
            save_16bit_png(lat_img, str(case_audit_dir / "xray_lat_gt.png"))

            z, y, x = ct_hu.shape
            save_16bit_png(np.clip(ct_hu[z // 2, :, :], -1000, 400), str(case_audit_dir / "ct_axial.png"))
            save_16bit_png(np.clip(np.flipud(ct_hu[:, y // 2, :]), -1000, 400), str(case_audit_dir / "ct_coronal.png"))
            save_16bit_png(np.clip(np.flipud(ct_hu[:, :, x // 2]), -1000, 400), str(case_audit_dir / "ct_sagittal.png"))

        progress_bar.progress((i + 1) / n)

    df = pd.DataFrame(rows)
    export_root.mkdir(parents=True, exist_ok=True)
    (export_root / "manifest.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    n_val = max(1, int(round(len(df) * float(val_frac))))
    val_set = set(idxs[:n_val])

    df["split"] = ["val" if i in val_set else "train" for i in range(len(df))]
    (export_root / "train.csv").write_text(df[df["split"] == "train"].to_csv(index=False), encoding="utf-8")
    (export_root / "val.csv").write_text(df[df["split"] == "val"].to_csv(index=False), encoding="utf-8")

    st.sidebar.success(
        f"Exported {n} NPZs -> {npz_dir}\n"
        f"Audits (10) -> {audit_root}\n"
        f"manifest.csv/train.csv/val.csv -> {export_root}"
    )
    st.balloons()
