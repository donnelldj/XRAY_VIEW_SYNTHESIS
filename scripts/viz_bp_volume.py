# scripts/viz_bp_volume.py
# Visualize a back-projected (BP) volume in 3D using PyVista

import numpy as np
import pyvista as pv
from src.data_drr_pairs import DRRPairsDataset


def main():
    # Load one sample from the DRR pairs dataset
    ds = DRRPairsDataset(r"data/drr_pairs/npz")
    sample = ds[0]

    # sample["bp"] is a torch tensor with shape (1, Z, H, W)
    bp_t = sample["bp"]          # torch tensor
    bp = bp_t[0].cpu().numpy()   # -> (Z, H, W) numpy array

    # Normalize to [0, 1]
    v = (bp - bp.min()) / (bp.max() - bp.min() + 1e-8)

    z, h, w = v.shape
    # PyVista expects (nx, ny, nz) = (W, H, Z)
    grid = pv.ImageData(
        dimensions=(w, h, z),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )

    # Flatten with Fortran order to align memory layout
    grid.point_data["intensity"] = v.ravel(order="F")

    iso_level = 0.3
    surf = grid.contour([iso_level])

    pl = pv.Plotter()
    pl.add_text("Back-projected volume (BP)", font_size=12)
    pl.add_mesh(surf, opacity=0.6)
    pl.show_axes()
    pl.show()


if __name__ == "__main__":
    main()
