# from __future__ import annotations

# # =============================================================================
# #.py
# #
# # Purpose
# # -------
# # End-to-end "Section III-D: New X-ray View Synthesis" baseline:
# #   1) Input: single AP DRR (0°)   -> ap(z,x) in [0,1]
# #   2) Back-project (Eq. 1-ish):   -> replicate AP across Y to form volume V(z,y,x)
# #   3) Train a 3D UNet in latent space to predict CT-latent (supervised by CT)
# #   4) Forward-project predicted CT to LAT DRR (Eq. 9-ish): mean over X -> lat(z,y)
# #   5) Evaluate with PSNR/SSIM and save qualitative triplets: AP | Pred LAT | GT LAT
# #
# # Key Contract (MOST IMPORTANT)
# # -----------------------------
# # Your Streamlit exporter defines GT:
# #   ap  = flipud(mean(ct_norm, axis=Y))  # (Z,X)
# #   lat = flipud(mean(ct_norm, axis=X))  # (Z,Y)
# #
# # Therefore, to compare apples-to-apples, we MUST apply the same flip in the
# # forward projector used during training and evaluation:
# #   lat_pred = flipud(mean(ct_pred, axis=X))
# #
# # This script hard-matches that behavior by default:
# #   forward_project_ct_to_lat_export_match(..., export_flipud=True)
# #
# # Optional "lat transforms" flags exist for debugging only (keep defaults).
# # =============================================================================

# # --- allow imports from repo root (src/) ---
# import sys
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
# # ------------------------------------------

# import argparse
# import csv
# import json
# import math
# import random
# import time
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset

# # Optional: CT reading from MHD via SimpleITK (only needed if NPZ has no ct_zyx_norm)
# try:
#     import SimpleITK as sitk

#     _HAS_SITK = True
# except Exception:
#     _HAS_SITK = False


# # =============================================================================
# # Utils
# # =============================================================================

# def collate_samples(batch):
#     """
#     Convert List[Sample] -> dict of batched tensors.
#     """
#     ap = torch.stack([b.ap_zx for b in batch], dim=0)  # (B,1,Z,X)
#     lat = torch.stack([b.lat_zy for b in batch], dim=0)  # (B,1,Z,Y)
#     ct = None if batch[0].ct_zyx is None else torch.stack([b.ct_zyx for b in batch], dim=0)  # (B,1,Z,Y,X)
#     case_id = [b.case_id for b in batch]
#     return {"ap_zx": ap, "lat_zy": lat, "ct_zyx": ct, "case_id": case_id}


# def normalize01(x: np.ndarray) -> np.ndarray:
#     """
#     Normalize an array to [0,1] for visualization only.
#     """
#     x = x.astype(np.float32)
#     mn = float(np.min(x))
#     mx = float(np.max(x))
#     if mx - mn < 1e-8:
#         return np.zeros_like(x, dtype=np.float32)
#     return (x - mn) / (mx - mn)


# def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
#     """
#     Peak Signal-to-Noise Ratio (PSNR) for images in [0, data_range].
#     """
#     mse = float(np.mean((pred - gt) ** 2))
#     if mse < 1e-12:
#         return 99.0
#     return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


# def ssim_simple(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
#     """
#     Lightweight SSIM (single-scale, global stats). Stable + dependency-free.

#     NOTE: This is not a windowed SSIM; it's intended as a quick proxy.
#     """
#     pred = pred.astype(np.float32)
#     gt = gt.astype(np.float32)

#     C1 = (0.01 * data_range) ** 2
#     C2 = (0.03 * data_range) ** 2

#     mu_x = float(pred.mean())
#     mu_y = float(gt.mean())
#     sigma_x = float(pred.var())
#     sigma_y = float(gt.var())
#     sigma_xy = float(((pred - mu_x) * (gt - mu_y)).mean())

#     num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
#     den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
#     return float(num / (den + 1e-12))


# def resolve_path(p: str) -> Path:
#     """
#     Robust path resolver:
#       - accept Windows paths in CSV
#       - try absolute first
#       - fallback to repo-relative
#     """
#     p2 = p.replace("\\", "/")
#     cand = Path(p2)
#     if cand.exists():
#         return cand
#     cand2 = PROJECT_ROOT / p2
#     if cand2.exists():
#         return cand2
#     return cand


# def load_npz_paths_from_csv(csv_path: str) -> List[str]:
#     """
#     Robust: accepts either a single-column CSV of paths, or a header row containing 'npz_path'.
#     """
#     paths: List[str] = []
#     with open(str(csv_path), "r", newline="") as f:
#         reader = csv.reader(f)
#         rows = list(reader)

#     if not rows:
#         return paths

#     header = [c.strip() for c in rows[0]]
#     if any(h.lower() == "npz_path" for h in header):
#         idx = [i for i, h in enumerate(header) if h.lower() == "npz_path"][0]
#         for r in rows[1:]:
#             if not r:
#                 continue
#             v = r[idx].strip()
#             if v:
#                 paths.append(v)
#         return paths

#     # No header: assume first col is a path
#     for r in rows:
#         if not r:
#             continue
#         p = r[0].strip()
#         if p and p.endswith(".npz"):
#             paths.append(p)

#     return paths


# def save_triplet(ap_2d: np.ndarray, lat_pred_2d: np.ndarray, lat_gt_2d: np.ndarray, out_path: Path) -> None:
#     """
#     Save a side-by-side 3-panel PNG:
#       [ AP | Pred LAT | GT LAT ]
#     """
#     ap = (normalize01(ap_2d) * 255.0).astype(np.uint8)
#     pr = (normalize01(lat_pred_2d) * 255.0).astype(np.uint8)
#     gt = (normalize01(lat_gt_2d) * 255.0).astype(np.uint8)

#     ap_im = Image.fromarray(ap, mode="L")
#     pr_im = Image.fromarray(pr, mode="L")
#     gt_im = Image.fromarray(gt, mode="L")

#     w, h = ap_im.size
#     canvas = Image.new("L", (w * 3, h))
#     canvas.paste(ap_im, (0, 0))
#     canvas.paste(pr_im, (w, 0))
#     canvas.paste(gt_im, (w * 2, 0))

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     canvas.save(out_path)


# # =============================================================================
# # CT loading (fallback if NPZ has no ct_zyx_norm)
# # =============================================================================

# def resample_to_size_preserve_extent(img: "sitk.Image", out_size_xyz: Tuple[int, int, int]) -> "sitk.Image":
#     """
#     Resample to target voxel grid size (X,Y,Z) while preserving physical extent.
#     Matches your Streamlit export logic.

#     Note: This is not the same as isotropic resampling; it preserves world extent
#     but adjusts spacing so the new grid covers the same physical volume.
#     """
#     in_size = np.array(list(img.GetSize()), dtype=np.float64)  # (X,Y,Z)
#     in_spacing = np.array(list(img.GetSpacing()), dtype=np.float64)  # (sx,sy,sz)
#     out_size = np.array(list(out_size_xyz), dtype=np.int64)

#     extent = in_size * in_spacing
#     out_spacing = extent / np.maximum(out_size.astype(np.float64), 1.0)

#     r = sitk.ResampleImageFilter()
#     r.SetSize([int(x) for x in out_size.tolist()])
#     r.SetOutputSpacing([float(x) for x in out_spacing.tolist()])
#     r.SetOutputOrigin(img.GetOrigin())
#     r.SetOutputDirection(img.GetDirection())
#     r.SetTransform(sitk.Transform())
#     r.SetInterpolator(sitk.sitkLinear)
#     r.SetDefaultPixelValue(-1024.0)
#     return r.Execute(img)


# def read_ct_preprocessed_zyx(
#     mhd_path: str,
#     target_zyx: Tuple[int, int, int] = (256, 256, 256),
#     hu_clip: Tuple[float, float] = (-1000.0, 400.0),
# ) -> np.ndarray:
#     """
#     Fallback CT load:
#       Read -> resample preserve extent -> GetArray(Z,Y,X) -> HU clip -> normalize 0..1
#     """
#     if not _HAS_SITK:
#         raise RuntimeError("SimpleITK is required. Install: pip install SimpleITK")

#     img = sitk.ReadImage(str(mhd_path))
#     tz, ty, tx = target_zyx
#     img = resample_to_size_preserve_extent(img, out_size_xyz=(tx, ty, tz))  # (X,Y,Z)

#     ct_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)

#     lo, hi = hu_clip
#     ct_zyx = np.clip(ct_zyx, lo, hi)
#     ct_zyx = (ct_zyx - lo) / (hi - lo + 1e-8)
#     return ct_zyx.astype(np.float32)


# # =============================================================================
# # Dataset
# # =============================================================================

# @dataclass
# class Sample:
#     ap_zx: torch.Tensor               # (1,Z,X)
#     lat_zy: torch.Tensor              # (1,Z,Y)
#     ct_zyx: Optional[torch.Tensor]    # (1,Z,Y,X) or None
#     case_id: str


# class NpzAPLatCTDataset(Dataset):
#     """
#     Expected NPZ keys from your Streamlit exporter:
#       - case_id (str)
#       - mhd_path (str)
#       - ap:  (Z,X) float32  = flipud(mean(ct_norm, axis=1))
#       - lat: (Z,Y) float32  = flipud(mean(ct_norm, axis=2))
#       - ct_zyx_norm (Z,Y,X) float16/float32 OPTIONAL (if you checked "Include CT volume")

#     Training uses CT supervision:
#       - Prefer ct_zyx_norm if present (perfect match to export)
#       - Else fallback to reading mhd_path and applying same resample+normalize
#     """
#     def __init__(
#         self,
#         npz_paths: List[str],
#         target_zyx: Tuple[int, int, int] = (256, 256, 256),
#         hu_clip: Tuple[float, float] = (-1000.0, 400.0),
#         load_ct: bool = True,
#     ):
#         self.npz_paths = npz_paths
#         self.target_zyx = target_zyx
#         self.hu_clip = hu_clip
#         self.load_ct = load_ct

#     def __len__(self) -> int:
#         return len(self.npz_paths)

#     def __getitem__(self, idx: int) -> Sample:
#         p = resolve_path(self.npz_paths[idx])
#         d = np.load(str(p), allow_pickle=True)

#         case_id = str(d["case_id"])
#         ap = d["ap"].astype(np.float32)    # (Z,X) already flipud in export
#         lat = d["lat"].astype(np.float32)  # (Z,Y) already flipud in export

#         if ap.ndim != 2 or lat.ndim != 2:
#             raise RuntimeError(f"Bad shapes in {p}: ap={ap.shape} lat={lat.shape}")

#         ap_t = torch.from_numpy(ap)[None, ...]     # (1,Z,X)
#         lat_t = torch.from_numpy(lat)[None, ...]   # (1,Z,Y)

#         ct_t: Optional[torch.Tensor] = None
#         if self.load_ct:
#             if "ct_zyx_norm" in d.files:
#                 ct = d["ct_zyx_norm"].astype(np.float32)  # (Z,Y,X) in [0,1]
#             else:
#                 mhd_path = str(d["mhd_path"])
#                 mhd_resolved = resolve_path(mhd_path)
#                 if not mhd_resolved.exists():
#                     raise FileNotFoundError(
#                         f"CT mhd_path not found for case_id={case_id}: {mhd_path} (resolved={mhd_resolved})"
#                     )
#                 ct = read_ct_preprocessed_zyx(
#                     str(mhd_resolved),
#                     target_zyx=self.target_zyx,
#                     hu_clip=self.hu_clip,
#                 )

#             if ct.shape != self.target_zyx:
#                 raise RuntimeError(f"CT shape mismatch for {case_id}: got {ct.shape}, expected {self.target_zyx}")

#             ct_t = torch.from_numpy(ct)[None, ...]  # (1,Z,Y,X)

#         return Sample(ap_t.float(), lat_t.float(), ct_t.float() if ct_t is not None else None, case_id)


# # =============================================================================
# # Model (latent 3D UNet)
# # =============================================================================

# class ConvBlock3D(nn.Module):
#     """
#     Simple Conv3D -> GN -> SiLU block, repeated twice.
#     GroupNorm is more stable than BatchNorm for small batch sizes (common here).
#     """
#     def __init__(self, c_in: int, c_out: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv3d(c_in, c_out, 3, padding=1),
#             nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
#             nn.SiLU(),
#             nn.Conv3d(c_out, c_out, 3, padding=1),
#             nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
#             nn.SiLU(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


# class UNet3D(nn.Module):
#     """
#     Small 3D UNet operating in latent space (e.g., 64^3 if latent_down=4).

#     Input:  (B,1,Zl,Yl,Xl) backprojected+pooled volume
#     Output: (B,1,Zl,Yl,Xl) predicted CT latent (0..1 after sigmoid)
#     """
#     def __init__(self, base: int = 16):
#         super().__init__()
#         self.enc1 = ConvBlock3D(1, base)
#         self.pool1 = nn.MaxPool3d(2)
#         self.enc2 = ConvBlock3D(base, base * 2)
#         self.pool2 = nn.MaxPool3d(2)
#         self.enc3 = ConvBlock3D(base * 2, base * 4)

#         self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
#         self.dec2 = ConvBlock3D(base * 4, base * 2)
#         self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
#         self.dec1 = ConvBlock3D(base * 2, base)

#         self.out = nn.Conv3d(base, 1, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool1(e1))
#         e3 = self.enc3(self.pool2(e2))

#         d2 = self.up2(e3)
#         d2 = torch.cat([d2, e2], dim=1)
#         d2 = self.dec2(d2)

#         d1 = self.up1(d2)
#         d1 = torch.cat([d1, e1], dim=1)
#         d1 = self.dec1(d1)

#         return self.out(d1)


# # =============================================================================
# # Physics-ish ops (must match export conventions)
# # =============================================================================

# def backproject_ap_to_volume(ap_zx: torch.Tensor, out_y: int) -> torch.Tensor:
#     """
#     Backprojection step (Eq. 1-ish approximation):
#       AP is (B,1,Z,X). We create a volume by repeating AP across the Y axis.

#     Output: (B,1,Z,Y,X)
#     """
#     ap_zyx = ap_zx.unsqueeze(3)               # (B,1,Z,1,X)
#     vol = ap_zyx.repeat(1, 1, 1, out_y, 1)    # (B,1,Z,Y,X)
#     return vol


# def forward_project_ct_to_lat_export_match(
#     ct_zyx: torch.Tensor,
#     *,
#     clamp01: bool = True,
#     export_flipud: bool = True,
#     invert: bool = False,
#     rot_k: int = 0,
#     flip_lr: bool = False,
# ) -> torch.Tensor:
#     """
#     Forward projection (Eq. 9-ish) that MATCHES your Streamlit export:

#       Streamlit GT:
#         lat = np.flipud(np.mean(ct_norm_zyx, axis=2))   # mean over X, then flip Z

#       Here:
#         lat = mean(ct, dim=-1)     # (B,1,Z,Y)
#         if export_flipud: flip Z   # torch.flip(..., dims=(-2,))

#     Optional transforms are for debugging ONLY; keep defaults for best results.
#     CT:  (B,1,Z,Y,X)
#     LAT: (B,1,Z,Y)
#     """
#     if clamp01:
#         ct_zyx = ct_zyx.clamp(0.0, 1.0)

#     # Mean over X -> (B,1,Z,Y)
#     lat = ct_zyx.mean(dim=-1)

#     # Match Streamlit: np.flipud on (Z,Y) == flip Z dimension
#     if export_flipud:
#         lat = torch.flip(lat, dims=(-2,))

#     # Debug-only transforms (leave off unless you have a known mismatch)
#     if invert:
#         lat = 1.0 - lat

#     k = int(rot_k) % 4
#     if k:
#         lat = torch.rot90(lat, k=k, dims=(-2, -1))  # rotate in (Z,Y)

#     if flip_lr:
#         lat = torch.flip(lat, dims=(-1,))  # flip Y

#     return lat


# def avg_pool_latent(vol_zyx: torch.Tensor, down: int) -> torch.Tensor:
#     """
#     Downsample (B,1,Z,Y,X) -> (B,1,Zl,Yl,Xl)
#     """
#     if down == 1:
#         return vol_zyx
#     return F.avg_pool3d(vol_zyx, kernel_size=down, stride=down)


# # =============================================================================
# # Train / Eval
# # =============================================================================

# def train_one_epoch(
#     model: nn.Module,
#     loader: DataLoader,
#     optim: torch.optim.Optimizer,
#     device: torch.device,
#     latent_down: int,
#     w_latent: float,
#     w_lat: float,
#     amp: bool,
#     export_flipud: bool,
#     invert: bool,
#     rot_k: int,
#     flip_lr: bool,
#     log_every: int = 50,
# ) -> float:
#     """
#     One epoch of training:
#       - latent supervision (MSE in CT latent space)
#       - optional LAT supervision (MSE on forward projection)
#     """
#     model.train()
#     total = 0.0
#     n = 0

#     amp_enabled = bool(amp and device.type == "cuda")
#     scaler = torch.amp.GradScaler(enabled=amp_enabled)

#     t0 = time.time()

#     for step, batch in enumerate(loader, start=1):
#         ap = batch["ap_zx"].to(device)         # (B,1,Z,X)
#         lat_gt = batch["lat_zy"].to(device)    # (B,1,Z,Y)  (already Streamlit-flipped)
#         ct = batch["ct_zyx"]
#         if ct is None:
#             raise RuntimeError("CT is required for latent supervision, but dataset returned ct=None.")
#         ct = ct.to(device)                     # (B,1,Z,Y,X)

#         _, _, _, Y, _ = ct.shape

#         # Backproject AP -> volume
#         bp = backproject_ap_to_volume(ap, out_y=Y)  # (B,1,Z,Y,X)

#         # Latent volumes
#         bp_lat = avg_pool_latent(bp, latent_down)   # (B,1,Zl,Yl,Xl)
#         ct_lat_gt = avg_pool_latent(ct, latent_down)

#         with torch.autocast(device_type="cuda", enabled=amp_enabled):
#             # Predict latent; sigmoid keeps in [0,1] to match normalized CT target
#             ct_lat_pred = torch.sigmoid(model(bp_lat))
#             loss_latent = F.mse_loss(ct_lat_pred, ct_lat_gt)

#             # Upsample latent back to full resolution for forward projection supervision
#             ct_pred = F.interpolate(
#                 ct_lat_pred, scale_factor=latent_down, mode="trilinear", align_corners=False
#             )

#             lat_pred = forward_project_ct_to_lat_export_match(
#                 ct_pred,
#                 clamp01=True,
#                 export_flipud=export_flipud,
#                 invert=invert,
#                 rot_k=rot_k,
#                 flip_lr=flip_lr,
#             )

#             loss_lat = F.mse_loss(lat_pred, lat_gt)
#             loss = w_latent * loss_latent + w_lat * loss_lat

#         optim.zero_grad(set_to_none=True)
#         scaler.scale(loss).backward()
#         scaler.step(optim)
#         scaler.update()

#         total += float(loss.detach().cpu().item())
#         n += 1

#         if log_every > 0 and (step % log_every == 0):
#             dt = time.time() - t0
#             print(
#                 f"[train] step {step:05d}  "
#                 f"loss={loss.item():.6f}  "
#                 f"latent={loss_latent.item():.6f}  lat={loss_lat.item():.6f}  "
#                 f"time={dt:.1f}s"
#             )

#     return total / max(1, n)


# @torch.no_grad()
# def eval_and_save_examples(
#     model: nn.Module,
#     loader: DataLoader,
#     device: torch.device,
#     latent_down: int,
#     out_dir: Path,
#     num_examples: int,
#     export_flipud: bool,
#     invert: bool,
#     rot_k: int,
#     flip_lr: bool,
# ) -> Dict:
#     """
#     Evaluate on the val set, compute PSNR/SSIM, and save example triplets.
#     """
#     model.eval()
#     metrics = []
#     saved = 0
#     ex_dir = out_dir / "examples"
#     ex_dir.mkdir(parents=True, exist_ok=True)

#     for i, batch in enumerate(loader, start=1):
#         ap = batch["ap_zx"].to(device)
#         lat_gt = batch["lat_zy"].to(device)
#         ct = batch["ct_zyx"].to(device)
#         case_id = batch["case_id"][0]

#         _, _, _, Y, _ = ct.shape

#         bp = backproject_ap_to_volume(ap, out_y=Y)
#         bp_lat = avg_pool_latent(bp, latent_down)

#         ct_lat_pred = torch.sigmoid(model(bp_lat))
#         ct_pred = F.interpolate(
#             ct_lat_pred, scale_factor=latent_down, mode="trilinear", align_corners=False
#         )

#         lat_pred = forward_project_ct_to_lat_export_match(
#             ct_pred,
#             clamp01=True,
#             export_flipud=export_flipud,
#             invert=invert,
#             rot_k=rot_k,
#             flip_lr=flip_lr,
#         )

#         ap_np = ap[0, 0].detach().cpu().numpy()
#         lat_gt_np = lat_gt[0, 0].detach().cpu().numpy()
#         lat_pr_np = lat_pred[0, 0].detach().cpu().numpy()

#         p = psnr(lat_pr_np, lat_gt_np, data_range=1.0)
#         s = ssim_simple(lat_pr_np, lat_gt_np, data_range=1.0)
#         metrics.append({"case_id": case_id, "psnr": p, "ssim": s})

#         if saved < num_examples:
#             out_path = ex_dir / f"triplet_{saved:03d}__{case_id.replace('/', '_')}.png"
#             save_triplet(ap_np, lat_pr_np, lat_gt_np, out_path)
#             saved += 1

#         if i % 20 == 0:
#             print(f"[eval] processed {i} cases...")

#     psnrs = [m["psnr"] for m in metrics]
#     ssims = [m["ssim"] for m in metrics]
#     summary = {
#         "count": len(metrics),
#         "psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
#         "psnr_std": float(np.std(psnrs)) if psnrs else 0.0,
#         "ssim_mean": float(np.mean(ssims)) if ssims else 0.0,
#         "ssim_std": float(np.std(ssims)) if ssims else 0.0,
#         "per_case": metrics,
#     }
#     return summary


# # =============================================================================
# # Main
# # =============================================================================

# def main():
#     ap = argparse.ArgumentParser()

#     # Inputs / outputs
#     ap.add_argument("--train_csv", type=str, required=True)
#     ap.add_argument("--val_csv", type=str, required=True)
#     ap.add_argument("--out_dir", type=str, required=True)

#     # Lat projection behavior
#     # IMPORTANT: export_flipud=True matches your Streamlit GT convention.
#     ap.add_argument("--no_export_flipud", action="store_true",
#                     help="DEBUG: disable the Streamlit flipud(Z) in forward projection.")
#     ap.add_argument("--lat_invert", action="store_true", help="DEBUG: invert LAT (1-lat).")
#     ap.add_argument("--lat_rot_k", type=int, default=0, help="DEBUG: rot90 k times on (Z,Y).")
#     ap.add_argument("--lat_flip_lr", action="store_true", help="DEBUG: flip Y axis (left-right).")

#     # CT preprocessing fallback (only used if NPZ has no ct_zyx_norm)
#     ap.add_argument("--hu_min", type=float, default=-1000.0)
#     ap.add_argument("--hu_max", type=float, default=400.0)
#     ap.add_argument("--size_z", type=int, default=256)
#     ap.add_argument("--size_y", type=int, default=256)
#     ap.add_argument("--size_x", type=int, default=256)

#     # Train hyperparams
#     ap.add_argument("--epochs", type=int, default=16)
#     ap.add_argument("--batch", type=int, default=1)
#     ap.add_argument("--lr", type=float, default=2e-4)
#     ap.add_argument("--seed", type=int, default=123)

#     ap.add_argument("--latent_down", type=int, default=4,
#                     help="Downsample factor for CT latent (256 -> 64 if 4).")
#     ap.add_argument("--base", type=int, default=16, help="UNet base channels.")
#     ap.add_argument("--w_latent", type=float, default=1.0, help="Weight for latent CT supervision (MSE).")
#     ap.add_argument("--w_lat", type=float, default=0.1, help="Weight for LAT projection supervision (MSE).")

#     # Runtime
#     ap.add_argument("--device", type=str, default="cuda")
#     ap.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
#     ap.add_argument("--num_examples", type=int, default=10)
#     ap.add_argument("--ckpt", type=str, default="",
#                     help="If set, skip training and only eval this checkpoint.")
#     ap.add_argument("--log_every", type=int, default=50)

#     args = ap.parse_args()

#     # Reproducibility
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     train_npzs = load_npz_paths_from_csv(args.train_csv)
#     val_npzs = load_npz_paths_from_csv(args.val_csv)
#     if not train_npzs:
#         raise RuntimeError(f"No train npz paths found in {args.train_csv}")
#     if not val_npzs:
#         raise RuntimeError(f"No val npz paths found in {args.val_csv}")

#     # Device
#     if args.device == "cuda" and not torch.cuda.is_available():
#         print("[warn] CUDA requested but not available; switching to CPU.")
#         device = torch.device("cpu")
#     else:
#         device = torch.device(args.device)

#     target_zyx = (args.size_z, args.size_y, args.size_x)
#     hu_clip = (args.hu_min, args.hu_max)

#     # If SimpleITK isn't installed, you MUST have ct_zyx_norm in NPZ.
#     if not _HAS_SITK:
#         print("[warn] SimpleITK not available. This is fine ONLY if your NPZs include ct_zyx_norm.")
#         print("       If not, install it: pip install SimpleITK")

#     export_flipud = not args.no_export_flipud

#     print(f"[info] device={device}  amp={bool(args.amp and device.type=='cuda')}")
#     print(f"[info] target_zyx={target_zyx}  hu_clip={hu_clip}")
#     print(
#         f"[info] lat_transform: export_flipud={export_flipud} "
#         f"invert={args.lat_invert} rot_k={args.lat_rot_k} flip_lr={args.lat_flip_lr}"
#     )
#     print(f"[info] train_npz={len(train_npzs)}  val_npz={len(val_npzs)}")

#     # Data
#     ds_train = NpzAPLatCTDataset(train_npzs, target_zyx=target_zyx, hu_clip=hu_clip, load_ct=True)
#     ds_val = NpzAPLatCTDataset(val_npzs, target_zyx=target_zyx, hu_clip=hu_clip, load_ct=True)

#     dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate_samples)
#     dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_samples)

#     # Model + optimizer
#     model = UNet3D(base=args.base).to(device)
#     optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

#     ckpt_path = out_dir / "checkpoint.pt"

#     # Load checkpoint if requested / resume if present
#     if args.ckpt:
#         ck = torch.load(args.ckpt, map_location="cpu")
#         model.load_state_dict(ck["model"])
#         print(f"[eval] loaded checkpoint: {args.ckpt}")
#     elif ckpt_path.exists():
#         ck = torch.load(str(ckpt_path), map_location="cpu")
#         model.load_state_dict(ck["model"])
#         print(f"[resume] loaded checkpoint: {ckpt_path}")
#     else:
#         print(
#             f"[train] epochs={args.epochs} batch={args.batch} lr={args.lr} "
#             f"latent_down={args.latent_down} w_latent={args.w_latent} w_lat={args.w_lat}"
#         )
#         for ep in range(1, args.epochs + 1):
#             ep_t0 = time.time()
#             loss = train_one_epoch(
#                 model=model,
#                 loader=dl_train,
#                 optim=optim,
#                 device=device,
#                 latent_down=args.latent_down,
#                 w_latent=args.w_latent,
#                 w_lat=args.w_lat,
#                 amp=args.amp and device.type == "cuda",
#                 export_flipud=export_flipud,
#                 invert=args.lat_invert,
#                 rot_k=args.lat_rot_k,
#                 flip_lr=args.lat_flip_lr,
#                 log_every=args.log_every,
#             )
#             dt = time.time() - ep_t0
#             print(f"[train] epoch {ep:03d}/{args.epochs}  loss={loss:.6f}  time={dt:.1f}s")

#         torch.save({"model": model.state_dict(), "args": vars(args)}, str(ckpt_path))
#         print(f"[train] saved: {ckpt_path}")

#     # Eval + examples
#     summary = eval_and_save_examples(
#         model=model,
#         loader=dl_val,
#         device=device,
#         latent_down=args.latent_down,
#         out_dir=out_dir,
#         num_examples=args.num_examples,
#         export_flipud=export_flipud,
#         invert=args.lat_invert,
#         rot_k=args.lat_rot_k,
#         flip_lr=args.lat_flip_lr,
#     )

#     # Write metrics
#     (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
#     (out_dir / "metrics.txt").write_text(
#         f"count={summary['count']}\n"
#         f"psnr_mean={summary['psnr_mean']:.4f}  psnr_std={summary['psnr_std']:.4f}\n"
#         f"ssim_mean={summary['ssim_mean']:.4f}  ssim_std={summary['ssim_std']:.4f}\n",
#         encoding="utf-8",
#     )

#     print("[eval] " + (out_dir / "metrics.txt").read_text(encoding="utf-8").strip())
#     print(f"[eval] examples saved in: {out_dir / 'examples'}")


# if __name__ == "__main__":
#     main()
