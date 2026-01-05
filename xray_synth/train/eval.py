from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from xray_synth.metrics.image import psnr, ssim_simple
from xray_synth.physics.projectors import (
    avg_pool_latent,
    backproject_ap_to_volume,
    forward_project_ct_to_lat_export_match,
)
from xray_synth.vis.triplets import save_triplet


@torch.no_grad()
def eval_and_save_examples(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    latent_down: int,
    out_dir: Path,
    num_examples: int,
    export_flipud: bool,
    invert: bool,
    rot_k: int,
    flip_lr: bool,
) -> Dict:
    """
    Evaluate val set:
    - compute PSNR/SSIM per case
    - save example triplets to out_dir/examples
    """
    model.eval()
    metrics = []
    saved = 0

    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(loader, start=1):
        ap = batch["ap_zx"].to(device)
        lat_gt = batch["lat_zy"].to(device)
        ct = batch["ct_zyx"].to(device)
        case_id = batch["case_id"][0]

        _, _, _, Y, _ = ct.shape

        bp = backproject_ap_to_volume(ap, out_y=Y)
        bp_lat = avg_pool_latent(bp, latent_down)

        ct_lat_pred = torch.sigmoid(model(bp_lat))
        ct_pred = F.interpolate(ct_lat_pred, scale_factor=latent_down, mode="trilinear", align_corners=False)

        lat_pred = forward_project_ct_to_lat_export_match(
            ct_pred,
            clamp01=True,
            export_flipud=export_flipud,
            invert=invert,
            rot_k=rot_k,
            flip_lr=flip_lr,
        )

        ap_np = ap[0, 0].detach().cpu().numpy()
        lat_gt_np = lat_gt[0, 0].detach().cpu().numpy()
        lat_pr_np = lat_pred[0, 0].detach().cpu().numpy()

        p = psnr(lat_pr_np, lat_gt_np, data_range=1.0)
        s = ssim_simple(lat_pr_np, lat_gt_np, data_range=1.0)
        metrics.append({"case_id": case_id, "psnr": p, "ssim": s})

        if saved < num_examples:
            out_path = ex_dir / f"triplet_{saved:03d}__{case_id.replace('/', '_')}.png"
            save_triplet(ap_np, lat_pr_np, lat_gt_np, out_path)
            saved += 1

        if i % 20 == 0:
            print(f"[eval] processed {i} cases...")

    psnrs = [m["psnr"] for m in metrics]
    ssims = [m["ssim"] for m in metrics]

    return {
        "count": len(metrics),
        "psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
        "psnr_std": float(np.std(psnrs)) if psnrs else 0.0,
        "ssim_mean": float(np.mean(ssims)) if ssims else 0.0,
        "ssim_std": float(np.std(ssims)) if ssims else 0.0,
        "per_case": metrics,
    }
