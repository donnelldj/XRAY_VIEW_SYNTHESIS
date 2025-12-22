from __future__ import annotations

# --- IMPORTANT: Streamlit import fix for src/ package -------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (xray_view_synthesis)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------------------------------------

import json
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import streamlit as st


# Change this if your unzipped LUNA16 lives elsewhere
DATA_ROOT_DEFAULT = Path("data/luna16")

SAVE_JSON_DEFAULT = Path("data/selected_200.json")
SAVE_CSV_DEFAULT  = Path("data/selected_200.csv")


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    mhd_path: str


def find_cases(data_root: Path, exclude_seg_masks: bool = True) -> List[CaseInfo]:
    """
    Find all .mhd volumes under data_root.
    LUNA16 includes seg-lungs-LUNA16 masks; exclude by default.
    """
    mhd_files = list(data_root.rglob("*.mhd"))
    if exclude_seg_masks:
        mhd_files = [p for p in mhd_files if "seg-lungs-LUNA16" not in str(p).replace("\\", "/")]
    mhd_files = sorted(mhd_files)

    cases: List[CaseInfo] = []
    for p in mhd_files:
        case_id = f"{p.parent.name}/{p.stem}"
        cases.append(CaseInfo(case_id=case_id, mhd_path=str(p.as_posix())))
    return cases


def ensure_state():
    if "selected" not in st.session_state:
        st.session_state.selected = set()
    if "cases" not in st.session_state:
        st.session_state.cases = []
    if "view_index" not in st.session_state:
        st.session_state.view_index = 0
    if "last_filter_key" not in st.session_state:
        st.session_state.last_filter_key = ""


def save_selection(cases: List[CaseInfo], selected_ids: set[str], json_path: Path, csv_path: Path):
    selected = [c for c in cases if c.case_id in selected_ids]
    payload: List[Dict[str, Any]] = [{"case_id": c.case_id, "mhd_path": c.mhd_path} for c in selected]

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(payload).to_csv(csv_path, index=False)


def main():
    st.set_page_config(page_title="CT Volume Selector (Stable)", layout="wide")
    ensure_state()

    st.title("CT Volume Selector (Stable) — pick 200 + save")

    with st.sidebar:
        st.header("Data")
        data_root = Path(st.text_input("LUNA16 root", str(DATA_ROOT_DEFAULT)))
        exclude_seg = st.checkbox("Exclude seg-lungs-LUNA16 (masks)", value=True)
        reload_btn = st.button("🔄 Scan for .mhd files")

        st.divider()

        st.header("Selection")
        target_n = st.number_input("Target count", min_value=1, max_value=5000, value=200, step=1)
        st.write(f"Selected: **{len(st.session_state.selected)} / {target_n}**")

        st.divider()

        st.header("Save")
        json_out = Path(st.text_input("Save JSON to", str(SAVE_JSON_DEFAULT)))
        csv_out  = Path(st.text_input("Save CSV to", str(SAVE_CSV_DEFAULT)))

        if st.button("💾 Save selection now"):
            if not st.session_state.cases:
                st.error("No cases loaded yet.")
            else:
                save_selection(st.session_state.cases, st.session_state.selected, json_out, csv_out)
                st.success(f"Saved {len(st.session_state.selected)} cases.")
                st.info(f"JSON: {json_out}")
                st.info(f"CSV:  {csv_out}")

        st.divider()

        if st.button("✅ Fill to target (first N)"):
            if st.session_state.cases:
                st.session_state.selected = set([c.case_id for c in st.session_state.cases[: int(target_n)]])
                st.rerun()

        if st.button("🧹 Clear selection"):
            st.session_state.selected = set()
            st.rerun()

    # Load cases
    if reload_btn or not st.session_state.cases:
        if not data_root.exists():
            st.error(f"Path not found: {data_root}")
            return
        st.session_state.cases = find_cases(data_root, exclude_seg_masks=exclude_seg)
        st.session_state.view_index = 0
        st.success(f"Found {len(st.session_state.cases)} volumes.")

    cases: List[CaseInfo] = st.session_state.cases
    if not cases:
        st.warning("No .mhd files found under that folder.")
        return

    # Filter
    c1, c2, c3 = st.columns([2.2, 1.2, 1.6])
    with c1:
        search = st.text_input("Search case_id contains", "")
    with c2:
        only_selected = st.checkbox("Show only selected", value=False)
    with c3:
        st.caption("Prev/Next navigates the filtered list")

    view_cases = cases
    if only_selected:
        view_cases = [c for c in view_cases if c.case_id in st.session_state.selected]
    if search.strip():
        s = search.lower().strip()
        view_cases = [c for c in view_cases if s in c.case_id.lower()]

    if not view_cases:
        st.warning("No cases match your filter.")
        return

    # Reset view index if filter changed
    filter_key = f"{search.strip().lower()}|onlysel={only_selected}"
    if filter_key != st.session_state.last_filter_key:
        st.session_state.view_index = 0
        st.session_state.last_filter_key = filter_key

    # Jump
    st.session_state.view_index = int(
        st.number_input(
            "Jump to view index",
            min_value=0,
            max_value=max(0, len(view_cases) - 1),
            value=min(st.session_state.view_index, len(view_cases) - 1),
            step=1,
        )
    )

    # Nav + toggle
    ctrl1, ctrl2, ctrl3 = st.columns([1.0, 1.0, 2.0])
    with ctrl1:
        if st.button("⬅ Prev"):
            st.session_state.view_index = max(0, st.session_state.view_index - 1)
            st.rerun()
    with ctrl2:
        if st.button("Next ➡"):
            st.session_state.view_index = min(len(view_cases) - 1, st.session_state.view_index + 1)
            st.rerun()

    case = view_cases[st.session_state.view_index]
    is_selected = case.case_id in st.session_state.selected
    with ctrl3:
        toggle = st.checkbox("Selected", value=is_selected, key=f"sel::{case.case_id}")
        if toggle and not is_selected:
            st.session_state.selected.add(case.case_id)
        if (not toggle) and is_selected:
            st.session_state.selected.remove(case.case_id)

    st.subheader(f"Case: {case.case_id}")
    st.code(case.mhd_path)

    # Stats + mini table
    m1, m2, m3 = st.columns(3)
    m1.metric("Filtered volumes", f"{len(view_cases)}")
    m2.metric("Total volumes", f"{len(cases)}")
    m3.metric("Selected", f"{len(st.session_state.selected)}")

    st.divider()

    if st.session_state.selected:
        st.subheader("Selected (top 50 preview)")
        sel_rows = [{"case_id": c.case_id, "mhd_path": c.mhd_path} for c in cases if c.case_id in st.session_state.selected]
        st.dataframe(pd.DataFrame(sel_rows[:50]), use_container_width=True, height=240)
        st.caption("Save from the sidebar to write the full JSON/CSV.")


if __name__ == "__main__":
    main()
