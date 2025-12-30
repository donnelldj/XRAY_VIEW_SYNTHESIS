# scripts/viz_ct_prediction.py
# Visualize BP volume vs GT CT vs Predicted CT

import numpy as np
import torch
import pyvista as pv

from src.data_drr_pairs import DRRPairsDataset
from src.models_unet3d import UNet3D_CT


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def volume_to_mesh(vol, title, iso=0.35):
    z, h, w = vol.shape
    grid = pv.ImageData(
        dimensions=(w, h, z),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
    )
    grid.point_data["v"] = vol.ravel(order="F")
    surf = grid.contour([iso])
    return surf


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    ds = DRRPairsDataset("data/drr_pairs/npz")
    sample = ds[0]

    # Load volumes
    bp = sample["bp"][0].cpu().numpy()      # (Z,H,W)
    ct_gt = sample["ct"][0].cpu().numpy()   # (Z,H,W)

    # Load model
    model = UNet3D_CT(base=16).to(device)
    ckpt = torch.load(
        "runs/ap2lat_rtx_e40_b16/ckpt_best.pt",
        map_location=device,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Predict CT
    with torch.no_grad():
        bp_t = sample["bp"].unsqueeze(1).to(device)  # [1,1,Z,H,W]
        ct_pred = model(bp_t)[0, 0].cpu().numpy()

    # Normalize
    bp_n = normalize(bp)
    ct_gt_n = normalize(ct_gt)
    ct_pred_n = normalize(ct_pred)

    # Build meshes
    m_bp = volume_to_mesh(bp_n, "BP", iso=0.4)
    m_gt = volume_to_mesh(ct_gt_n, "GT CT", iso=0.4)
    m_pr = volume_to_mesh(ct_pred_n, "Pred CT", iso=0.4)

    # Plot side-by-side
    pl = pv.Plotter(shape=(1, 3), window_size=(1600, 600))

    pl.subplot(0, 0)
    pl.add_text("Back-Projected Volume", font_size=10)
    pl.add_mesh(m_bp, opacity=0.6)

    pl.subplot(0, 1)
    pl.add_text("Ground-Truth CT", font_size=10)
    pl.add_mesh(m_gt, opacity=0.6)

    pl.subplot(0, 2)
    pl.add_text("Predicted CT", font_size=10)
    pl.add_mesh(m_pr, opacity=0.6)

    pl.show()


if __name__ == "__main__":
    main()
