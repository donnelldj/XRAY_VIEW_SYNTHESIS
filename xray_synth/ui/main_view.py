from __future__ import annotations

import streamlit as st
import numpy as np

from xray_synth.processing.volume import load_original_volume, resample_volume
from xray_synth.utils.image_utils import normalize_for_display

def render_main(selected_path: str, selected_filename: str, hu_min: float, hu_max: float) -> None:
    st.header("🩻 Holobeam: Clinical-to-Synthesis Pipeline")

    vol_orig, itk_obj = load_original_volume(selected_path)

    col_meta, col_btn = st.columns([3, 1])
    with col_meta:
        st.markdown(f"**Case:** `{selected_filename}` | **Dim:** {itk_obj.GetSize()} | **Spacing:** {itk_obj.GetSpacing()}")
    with col_btn:
        if st.button("🔄 Resample & Generate DRRs", use_container_width=True, type="primary"):
            st.session_state.resampled_list.add(selected_filename)
            st.toast("Volume Resampled to fixed 256³ (preserve extent)", icon="✅")
            st.rerun()

    tabs_list = ["3D Volume Slicer"]
    if selected_filename in st.session_state.resampled_list:
        tabs_list.append("Synthesized X-Rays (DRR)")

    tabs = st.tabs(tabs_list)

    # --- TAB 0: Volume slicer ---
    with tabs[0]:
        if selected_filename in st.session_state.resampled_list:
            ct_hu, ct_norm, _, _, _ = resample_volume(itk_obj, hu_min=float(hu_min), hu_max=float(hu_max))
            display_vol = ct_hu
            st.info("💡 Viewing Resampled Volume (fixed 256³, preserve extent)")
        else:
            display_vol = vol_orig
            st.warning("⚠️ Viewing Original Raw CT (Non-Isotropic)")

        z, y, x = display_vol.shape
        c1, c2, c3 = st.columns(3)

        with c1:
            iz = st.slider("Axial Plane", 0, z - 1, z // 2)
            st.image(
                normalize_for_display(np.clip(display_vol[iz, :, :], -1000, 400)),
                use_container_width=True,
                caption="Axial (Superior-Inferior)",
            )
        with c2:
            iy = st.slider("Coronal Plane", 0, y - 1, y // 2)
            st.image(
                normalize_for_display(np.clip(np.flipud(display_vol[:, iy, :]), -1000, 400)),
                use_container_width=True,
                caption="Coronal (Anterior-Posterior)",
            )
        with c3:
            ix = st.slider("Sagittal Plane", 0, x - 1, x // 2)
            st.image(
                normalize_for_display(np.clip(np.flipud(display_vol[:, :, ix]), -1000, 400)),
                use_container_width=True,
                caption="Sagittal (Left-Right)",
            )

    # --- TAB 1: DRRs ---
    if "Synthesized X-Rays (DRR)" in tabs_list:
        with tabs[1]:
            st.subheader("Dual-View Guided Synthesis")
            st.write("These views represent the 0° and 90° projections used for Triplet training.")

            _, _, ap_img, lat_img, _ = resample_volume(itk_obj, hu_min=float(hu_min), hu_max=float(hu_max))

            col1, col2 = st.columns(2)
            with col1:
                st.image(normalize_for_display(ap_img), caption="Input AP (0°)", use_container_width=True)
                st.caption("Target for Equation 1: Back-projection")
            with col2:
                st.image(normalize_for_display(lat_img), caption="Target Lateral (90°)", use_container_width=True)
                st.caption("Target for Equation 9: New View Synthesis Loss")
