# tools/train_step_smoke.py
# Smoke test: one batch, one optimizer step.
# Fixes Windows imports by adding repo root to sys.path.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import torch
from torch.utils.data import DataLoader

from src.data_drr_pairs import DRRPairsDataset
from src.models.unet3d_min import UNet3DMin
from src.projection_simple import forward_project_lat_from_ct


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = DRRPairsDataset(r"data/drr_pairs/npz")
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    model = UNet3DMin(base=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch = next(iter(dl))
    bp = batch["bp"].to(device)        # (B,1,96,256,256)
    ct = batch["ct"].to(device)        # (B,1,96,256,256)  (not used in loss yet)
    lat_gt = batch["lat"].to(device)   # (B,1,256,256)

    # Predict CT from BP
    ct_pred = model(bp)

    # Project predicted CT to Lat
    lat_pred = forward_project_lat_from_ct(ct_pred)

    # Loss on Lat only (matches the view-synthesis task: AP -> Lat)
    loss = torch.nn.functional.mse_loss(lat_pred, lat_gt)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    print("loss:", float(loss.item()))
    print("bp:", tuple(bp.shape), "ct_pred:", tuple(ct_pred.shape), "lat_pred:", tuple(lat_pred.shape))

    # Save a quick tensor dump for debugging/inspection
    os.makedirs("runs/smoke", exist_ok=True)
    torch.save(
        {
            "loss": float(loss.item()),
            "lat_pred": lat_pred.detach().cpu(),
            "lat_gt": lat_gt.detach().cpu(),
            "ap": batch["ap"],  # already CPU
            "pair_path": batch["pair_path"][0],
            "src_path": batch["src_path"][0],
        },
        "runs/smoke/step.pt",
    )
    print("saved runs/smoke/step.pt")


if __name__ == "__main__":
    main()
