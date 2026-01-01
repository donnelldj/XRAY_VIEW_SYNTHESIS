from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import SimpleITK as sitk
import pyvista as pv


# -----------------------------
# Data structs / selection utils
# -----------------------------

@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    mhd_path: str


def read_selected(json_path: Path) -> List[CaseInfo]:
    """
    Reads a JSON list like:
      [{"case_id": "...", "mhd_path": "..."}, ...]
    """
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    cases: List[CaseInfo] = []
    for row in payload:
        cases.append(CaseInfo(case_id=row["case_id"], mhd_path=row["mhd_path"]))
    return cases


# -----------------------------
# CT load + PyVista conversion
# -----------------------------

def load_ct_mhd(mhd_path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Returns:
      vol_zyx: float32 ndarray shaped (Z, Y, X)
      spacing_xyz: (sx, sy, sz) in mm
    """
    img = sitk.ReadImage(mhd_path)
    vol_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)
    spacing_xyz = img.GetSpacing()  # (sx,sy,sz)
    return vol_zyx, spacing_xyz


def make_image_data(vol_zyx: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> pv.ImageData:
    """
    PyVista expects grid.dimensions = (X, Y, Z).
    We store HU scalars on points. Flatten in Fortran order.
    """
    z, y, x = vol_zyx.shape
    sx, sy, sz = spacing_xyz

    grid = pv.ImageData()
    grid.dimensions = (x, y, z)  # (X,Y,Z)
    grid.spacing = (sx, sy, sz)  # mm

    # Convert volume to (X,Y,Z) then flatten Fortran for correct mapping
    scalars = vol_zyx.transpose(2, 1, 0).ravel(order="F")
    grid.point_data["HU"] = scalars
    return grid


# -----------------------------
# Main
# -----------------------------

def resolve_mhd_path(args) -> Tuple[str, Optional[str]]:
    """
    Returns (mhd_path, case_id)
    - If --mhd is provided, use it.
    - Else if --selected_json and --index are provided, load and select that case.
    """
    if args.mhd:
        return str(Path(args.mhd)), None

    if args.selected_json is None or args.index is None:
        raise SystemExit("Provide either --mhd OR (--selected_json AND --index).")

    selected_path = Path(args.selected_json)
    cases = read_selected(selected_path)
    if not cases:
        raise SystemExit(f"No cases found in {selected_path}")

    idx = int(args.index)
    if idx < 0 or idx >= len(cases):
        raise SystemExit(f"--index out of range: {idx} (0..{len(cases)-1})")

    c = cases[idx]
    return str(Path(c.mhd_path)), c.case_id


def main():
    ap = argparse.ArgumentParser(description="Single CT 3D viewer (PyVista) for LUNA16 .mhd volumes.")
    ap.add_argument("--mhd", default=None, help="Path to a .mhd file")

    # Optional: choose from selected_200.json by index
    ap.add_argument("--selected_json", default=None, help="Path to selected_200.json")
    ap.add_argument("--index", type=int, default=None, help="Index into selected_json list (0-based)")

    ap.add_argument("--mode", choices=["iso", "volume"], default="iso")
    ap.add_argument("--stride", type=int, default=2, help="Downsample stride for speed (>=1)")

    ap.add_argument("--hu_min", type=int, default=-1000, help="Clip lower HU for viewing")
    ap.add_argument("--hu_max", type=int, default=400, help="Clip upper HU for viewing")

    ap.add_argument("--iso", type=int, default=-600, help="Iso value if mode=iso (e.g., -600 lung)")
    ap.add_argument("--opacity", type=float, default=0.08, help="Opacity if mode=volume")

    ap.add_argument("--no_clip", action="store_true", help="Disable HU clipping (use raw values)")

    args = ap.parse_args()

    mhd_path, case_id = resolve_mhd_path(args)

    print(f"[viewer] mhd: {mhd_path}")
    if case_id:
        print(f"[viewer] case_id: {case_id}")

    vol, spacing = load_ct_mhd(mhd_path)
    print(f"[viewer] shape (Z,Y,X): {vol.shape}")
    print(f"[viewer] spacing (sx,sy,sz): {spacing}")
    print(f"[viewer] HU min/max: {float(vol.min()):.1f} / {float(vol.max()):.1f}")

    # downsample for speed
    s = max(1, int(args.stride))
    vol_ds = vol[::s, ::s, ::s]

    if not args.no_clip:
        vol_ds = np.clip(vol_ds, args.hu_min, args.hu_max)

    grid = make_image_data(vol_ds, spacing_xyz=spacing)

    p = pv.Plotter()
    p.set_background("black")
    p.add_axes()
    p.camera_position = "iso"

    if args.mode == "iso":
        surf = grid.contour([float(args.iso)], scalars="HU")
        p.add_mesh(surf, opacity=0.9)
    else:
        p.add_volume(grid, scalars="HU", opacity=args.opacity)

    p.show()


if __name__ == "__main__":
    main()
