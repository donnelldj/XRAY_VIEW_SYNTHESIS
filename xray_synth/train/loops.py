from __future__ import annotations

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from xray_synth.physics.projectors import (
    avg_pool_latent,
    backproject_ap_to_volume,
    forward_project_ct_to_lat_export_match,
)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    latent_down: int,
    w_latent: float,
    w_lat: float,
    amp: bool,
    export_flipud: bool,
    invert: bool,
    rot_k: int,
    flip_lr: bool,
    log_every: int = 50,
) -> float:
    """
    Train for one epoch:
    - latent supervision: MSE(ct_lat_pred, ct_lat_gt)
    - projection supervision: MSE(lat_pred, lat_gt)
    """
    model.train()
    total = 0.0
    n = 0

    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        ap = batch["ap_zx"].to(device)        # (B,1,Z,X)
        lat_gt = batch["lat_zy"].to(device)   # (B,1,Z,Y) already export-matched
        ct = batch["ct_zyx"]
        if ct is None:
            raise RuntimeError("CT required for latent supervision, but dataset returned ct=None.")
        ct = ct.to(device)                    # (B,1,Z,Y,X)

        _, _, _, Y, _ = ct.shape

        bp = backproject_ap_to_volume(ap, out_y=Y)
        bp_lat = avg_pool_latent(bp, latent_down)
        ct_lat_gt = avg_pool_latent(ct, latent_down)

        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            ct_lat_pred = torch.sigmoid(model(bp_lat))  # keep in [0,1] (matches normalized CT)
            loss_latent = F.mse_loss(ct_lat_pred, ct_lat_gt)

            ct_pred = F.interpolate(
                ct_lat_pred, scale_factor=latent_down, mode="trilinear", align_corners=False
            )

            lat_pred = forward_project_ct_to_lat_export_match(
                ct_pred,
                clamp01=True,
                export_flipud=export_flipud,
                invert=invert,
                rot_k=rot_k,
                flip_lr=flip_lr,
            )

            loss_lat = F.mse_loss(lat_pred, lat_gt)
            loss = w_latent * loss_latent + w_lat * loss_lat

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        total += float(loss.detach().cpu().item())
        n += 1

        if log_every > 0 and (step % log_every == 0):
            dt = time.time() - t0
            print(
                f"[train] step {step:05d} "
                f"loss={loss.item():.6f} latent={loss_latent.item():.6f} lat={loss_lat.item():.6f} "
                f"time={dt:.1f}s"
            )

    return total / max(1, n)
