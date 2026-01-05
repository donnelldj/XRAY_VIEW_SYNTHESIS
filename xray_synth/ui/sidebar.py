from __future__ import annotations

import os
import glob
import random
from pathlib import Path

import streamlit as st

from xray_synth.export.bulk_export import run_randomized_export

def discover_luna_mhds() -> tuple[list[str], dict[str, int], str]:
    """
    Returns (mhd_files, subset_stats, base_data_path)
    Assumes repo layout:
      <root>/
        app.py  (or this lives one level down)
        xray_synth/data/luna16/subset0/subset0/*.mhd
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # __file__ is xray_synth/ui/sidebar.py -> go up to repo root (two parents to reach project root)
    root_dir = str(Path(current_dir).resolve().parents[1])
    base_data_path = os.path.join(root_dir, "data", "luna16")

    mhd_files: list[str] = []
    subset_stats: dict[str, int] = {}

    for i in range(5):
        subset_folder = os.path.join(base_data_path, f"subset{i}", f"subset{i}")
        found = glob.glob(os.path.join(subset_folder, "*.mhd"))
        mhd_files.extend(found)
        subset_stats[f"Subset {i}"] = len(found)

    return mhd_files, subset_stats, base_data_path

def sidebar_controls() -> dict:
    st.sidebar.title("🩻 Holobeam Control")

    mhd_files, subset_stats, _ = discover_luna_mhds()

    with st.sidebar.expander("📂 Dataset Explorer", expanded=False):
        for name, count in subset_stats.items():
            st.write(f"{'✅' if count > 0 else '❌'} {name}: {count} cases")

    if not mhd_files:
        st.sidebar.error("Data folders not found. Check path structure.")
        st.stop()

    if st.sidebar.button("🔀 Shuffle Registry Case", use_container_width=True):
        st.session_state.selected_index = random.randint(0, len(mhd_files) - 1)

    sorted_files = sorted(mhd_files)
    selected_path = st.sidebar.selectbox(
        "Select Active Case",
        options=sorted_files,
        index=st.session_state.selected_index,
        format_func=lambda x: os.path.basename(x),
    )
    selected_filename = os.path.basename(selected_path)

    # --- SIDEBAR: EXPORT SETTINGS ---
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Export Settings")
    hu_min = st.sidebar.number_input("HU min (clip)", value=-1000.0, step=50.0)
    hu_max = st.sidebar.number_input("HU max (clip)", value=400.0, step=50.0)
    save_ct = st.sidebar.checkbox("Include CT volume in NPZ (ct_zyx_norm float16)", value=False)

    st.sidebar.divider()
    st.sidebar.subheader("📦 Bulk Triplet Export")
    export_limit = st.sidebar.number_input(
        "Max cases to export",
        min_value=1,
        max_value=len(mhd_files),
        value=min(200, len(mhd_files)),
        step=1,
    )
    export_folder = st.sidebar.text_input("Export Root", "training_triplets")
    val_frac = st.sidebar.number_input("Val fraction", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    seed = st.sidebar.number_input("Split seed", min_value=0, max_value=999999, value=42, step=1)

    # --- SIDEBAR: RANDOMIZED BULK EXPORT ---
    if st.sidebar.button("🚀 Execute Randomized Export", use_container_width=True):
        run_randomized_export(
            mhd_files=mhd_files,
            export_limit=int(export_limit),
            export_folder=str(export_folder),
            val_frac=float(val_frac),
            seed=int(seed),
            hu_min=float(hu_min),
            hu_max=float(hu_max),
            save_ct=bool(save_ct),
        )

    return dict(
        mhd_files=mhd_files,
        selected_path=selected_path,
        selected_filename=selected_filename,
        hu_min=float(hu_min),
        hu_max=float(hu_max),
        save_ct=bool(save_ct),
        export_limit=int(export_limit),
        export_folder=str(export_folder),
        val_frac=float(val_frac),
        seed=int(seed),
    )
