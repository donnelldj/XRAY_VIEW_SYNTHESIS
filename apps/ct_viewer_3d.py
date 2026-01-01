from __future__ import annotations

# --- IMPORTANT: Streamlit import fix for src/ package -------------------------
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------------------------------------


from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import streamlit as st

import SimpleITK as sitk
import pyvista as pv
from stpyvista import stpyvista  # embeds pyvista in streamlit


DATA_ROOT_DEFAULT = Path("data/luna16")


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    mhd_path: str


def find_cases(data_root: Path, exclude_seg_masks: bool = True) -> List[CaseInfo]:
    mhd_files = list(data_root.rglob("*.mhd"))
    if exclude_seg_masks:
        mhd_files = [p for p in mhd_files if "seg-lungs-LUNA16" not in str(p).replace("\\", "/")]
    mhd_files = sorted(mhd_files)

    cases: List[CaseInfo] = []
    for p in mhd_files:
        case_id = f"{p.parent.name}/{p.stem}"
        cases.append(CaseInfo(case_id=case_id, mhd_path=str(p.as_posix())))
    return cases


@st.cache_data(show_spinner=False)
def load_ct_mhd(mhd_path: str) -> dict:
    """
    Load .mhd via SimpleITK and return:
      - vol: np.ndarray in (Z, Y, X)
      - spacing: (sx, sy, sz)
      - origin: (ox, oy, oz)
    """
    img = sitk.ReadImage(mhd_path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)
    spacing = img.GetSpacing()  # (sx,sy,sz)
    origin = img.GetOrigin()
    return {"vol": vol, "spacing": spacing, "origin": origin}


def make_uniform_grid(vol_zyx: np.ndarray, spacing_xyz: tuple[float, float, float]) -> pv.UniformGrid:
    """
    Create a PyVista UniformGrid from a volume shaped (Z,Y,X).
    PyVista grid dims must be (X,Y,Z), and flattened in Fortran order.
    """
    sz, sy, sx = vol_zyx.shape  # Z,Y,X
    sx_mm, sy_mm, sz_mm = spacing_xyz  # (sx,sy,sz) in mm

    grid = pv.UniformGrid()
    grid.dimensions = (sx, sy, sz)  # (X,Y,Z)
    grid.spacing = (sx_mm, sy_mm, sz_mm)

    # Flatten volume in Fortran order to align with dimensions
    grid.point_data["HU"] = vol_zyx.transpose(2, 1, 0).ravel(order="F")
    return grid


def ensure_state():
    if "cases" not in st.session_state:
        st.session_state.cases = []
    if "view_index" not in st.session_state:
        st.session_state.view_index = 0


def main():
    st.set_page_config(page_title="CT 3D Viewer", layout="wide")
    ensure_state()

    st.title("CT 3D Viewer — PyVista (toggle + arrows)")

    with st.sidebar:
        st.header("Data")
        data_root = Path(st.text_input("LUNA16 root", str(DATA_ROOT_DEFAULT)))
        exclude_seg = st.checkbox("Exclude seg-lungs-LUNA16", value=True)
        if st.button("🔄 Scan for .mhd"):
            if not data_root.exists():
                st.error(f"Path not found: {data_root}")
                return
            st.session_state.cases = find_cases(data_root, exclude_seg_masks=exclude_seg)
            st.session_state.view_index = 0

        st.divider()
        st.header("3D Render Settings")
        show_ct = st.checkbox("Show CT Volume", value=True)

        # Light downsample so interaction stays snappy
        stride = st.slider("Downsample stride", 1, 6, 2, help="Higher = faster, lower = sharper")

        # Volume render via opacity mapping
        render_mode = st.selectbox("Mode", ["Isosurface", "Volume"], index=0)

        # HU window-ish controls (good defaults for lung-ish)
        hu_min = st.slider("HU min (clip)", -1200, 0, -1000)
        hu_max = st.slider("HU max (clip)", -500, 3000, 400)

        if render_mode == "Isosurface":
            iso = st.slider("Isosurface HU", -1200, 1000, -600, help="Try -600 lung, -300 soft tissue")
        else:
            opacity = st.slider("Opacity", 0.01, 0.5, 0.08)

        st.divider()
        st.caption("Use Prev/Next to switch cases.")

    # Auto-scan on first load
    if not st.session_state.cases:
        if data_root.exists():
            st.session_state.cases = find_cases(data_root, exclude_seg_masks=True)

    cases: List[CaseInfo] = st.session_state.cases
    if not cases:
        st.warning("No .mhd files found. Click Scan in the sidebar.")
        return

    # --- navigation controls (arrows) ---
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        if st.button("⬅ Prev"):
            st.session_state.view_index = max(0, st.session_state.view_index - 1)
            st.rerun()
    with c2:
        if st.button("Next ➡"):
            st.session_state.view_index = min(len(cases) - 1, st.session_state.view_index + 1)
            st.rerun()
    with c3:
        st.session_state.view_index = int(
            st.number_input(
                "Index",
                min_value=0,
                max_value=len(cases) - 1,
                value=st.session_state.view_index,
                step=1,
            )
        )

    case = cases[st.session_state.view_index]
    st.subheader(f"Case: {case.case_id}")
    st.code(case.mhd_path)

    # Load CT
    with st.spinner("Loading CT..."):
        ct = load_ct_mhd(case.mhd_path)
        vol = ct["vol"]  # (Z,Y,X)
        spacing = ct["spacing"]  # (sx,sy,sz)

    # Basic stats
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Shape (Z,Y,X)", str(vol.shape))
    s2.metric("Spacing (sx,sy,sz)", f"{spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}")
    s3.metric("Min HU", f"{float(vol.min()):.1f}")
    s4.metric("Max HU", f"{float(vol.max()):.1f}")

    st.divider()

    # --- build pyvista scene ---
    plotter = pv.Plotter(window_size=(1100, 750))
    plotter.set_background("black")

    if show_ct:
        vol_ds = vol[::stride, ::stride, ::stride]
        vol_clip = np.clip(vol_ds, hu_min, hu_max)

        grid = make_uniform_grid(vol_clip, spacing_xyz=spacing)

        if render_mode == "Isosurface":
            # Extract iso-surface
            surf = grid.contour([float(iso)], scalars="HU")
            plotter.add_mesh(surf, opacity=0.9)
        else:
            # Volume render
            plotter.add_volume(grid, scalars="HU", opacity=opacity)

    plotter.add_axes()
    plotter.camera_position = "iso"

    stpyvista(plotter, key=f"pv_{case.case_id}")

    st.caption("Tip: Isosurface is usually faster + clearer than full volume for quick sanity checks.")


if __name__ == "__main__":
    main()
