import streamlit as st

from xray_synth.app_config import set_streamlit_config
from xray_synth.ui.sidebar import sidebar_controls
from xray_synth.ui.main_view import render_main

def main() -> None:
    set_streamlit_config()

    # ---- SESSION STATE ----
    if "resampled_list" not in st.session_state:
        st.session_state.resampled_list = set()
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = 0

    # Sidebar (discover files, choose case, export settings, run bulk export)
    sidebar_state = sidebar_controls()

    # Main UI
    render_main(
        selected_path=sidebar_state["selected_path"],
        selected_filename=sidebar_state["selected_filename"],
        hu_min=sidebar_state["hu_min"],
        hu_max=sidebar_state["hu_max"],
    )

if __name__ == "__main__":
    main()
