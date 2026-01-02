# tools/rain_ap2lat_baseline.py
# Baseline training:
# AP -> BP volume -> 3D UNet -> CT_pred -> forward-project -> Lat_pred
# Loss = w_ct*MSE(CT_pred, CT_gt) + w_lat*MSE(Lat_pred, Lat_gt)
#git status
git commit -m "Ignore configs/docs and prune runs + drp_pairs artifacts"
git push
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
import time
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data_drr_pairs import DRRPairsDataset
from src.models.unet3d_min import UNet3DMin
from src.projection_simple import forward_project_lat_from_ct


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n, test_frac=0.33, seed=0):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_test = max(1, int(n * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


@torch.no_grad()
def eval_epoch(model, dl, device, w_ct, w_lat):
    model.eval()
    total_loss = 0.0
    total_ct = 0.0
    total_lat = 0.0
    n = 0

    for batch in dl:
        bp = batch["bp"].to(device, non_blocking=True)
        ct_gt = batch["ct"].to(device, non_blocking=True)
        lat_gt = batch["lat"].to(device, non_blocking=True)

        ct_pred = model(bp)
        lat_pred = forward_project_lat_from_ct(ct_pred)

        loss_ct = torch.nn.functional.mse_loss(ct_pred, ct_gt)
        loss_lat = torch.nn.functional.mse_loss(lat_pred, lat_gt)
        loss = w_ct * loss_ct + w_lat * loss_lat

        bs = bp.shape[0]
        total_loss += float(loss.item()) * bs
        total_ct += float(loss_ct.item()) * bs
        total_lat += float(loss_lat.item()) * bs
        n += bs

    return {
        "loss": total_loss / max(1, n),
        "loss_ct": total_ct / max(1, n),
        "loss_lat": total_lat / max(1, n),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--drr_dir", default=r"data/drr_pairs/npz")
    ap.add_argument("--out_dir", default=r"runs/ap2lat_baseline")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=16, help="UNet base channels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.33)
    ap.add_argument("--w_ct", type=float, default=0.5)
    ap.add_argument("--w_lat", type=float, default=1.0)
    args = ap.parse_args()

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    ds = DRRPairsDataset(args.drr_dir)
    train_idx, test_idx = split_indices(len(ds), test_frac=args.test_frac, seed=args.seed)

    train_ds = Subset(ds, train_idx)
    test_ds = Subset(ds, test_idx)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = UNet3DMin(base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {"epoch": -1, "loss": float("inf")}

    log_path = os.path.join(args.out_dir, "log.jsonl")
    print("writing log to:", log_path, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        running_loss = 0.0
        running_ct = 0.0
        running_lat = 0.0
        n = 0

        for batch in train_dl:
            bp = batch["bp"].to(device, non_blocking=True)
            ct_gt = batch["ct"].to(device, non_blocking=True)
            lat_gt = batch["lat"].to(device, non_blocking=True)

            ct_pred = model(bp)
            lat_pred = forward_project_lat_from_ct(ct_pred)

            loss_ct = torch.nn.functional.mse_loss(ct_pred, ct_gt)
            loss_lat = torch.nn.functional.mse_loss(lat_pred, lat_gt)
            loss = args.w_ct * loss_ct + args.w_lat * loss_lat

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = bp.shape[0]
            running_loss += float(loss.item()) * bs
            running_ct += float(loss_ct.item()) * bs
            running_lat += float(loss_lat.item()) * bs
            n += bs

        train_metrics = {
            "loss": running_loss / max(1, n),
            "loss_ct": running_ct / max(1, n),
            "loss_lat": running_lat / max(1, n),
        }

        val_metrics = eval_epoch(model, test_dl, device, args.w_ct, args.w_lat)

        dt = time.time() - t0
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "sec": dt}
        print(row, flush=True)

        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        # Save latest
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "val_loss": val_metrics["loss"]},
            os.path.join(args.out_dir, "ckpt_last.pt"),
        )

        # Save best
        if val_metrics["loss"] < best["loss"]:
            best = {"epoch": epoch, "loss": val_metrics["loss"]}
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": val_metrics["loss"]},
                os.path.join(args.out_dir, "ckpt_best.pt"),
            )
            with open(os.path.join(args.out_dir, "best.json"), "w") as f:
                json.dump(best, f, indent=2)

    print("DONE. best:", best, flush=True)
    print("best ckpt:", os.path.join(args.out_dir, "ckpt_best.pt"), flush=True)


if __name__ == "__main__":
    main()
