"""
xray_synth

End-to-end baseline for AP -> LAT synthesis using a latent 3D UNet.

This package is intentionally structured for:
- readability (clean module boundaries)
- reproducibility (explicit config + deterministic seeds)
- reviewer ergonomics (thin CLI, testable units)
"""
import streamlit as st

def set_streamlit_config() -> None:
    st.set_page_config(
        layout="wide",
        page_title="Holobeam X-Ray Synthesis",
        page_icon="🩻",
    )
