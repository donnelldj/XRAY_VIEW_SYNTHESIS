from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from xray_synth.config.defaults import DataConfig, LatTransformConfig, RuntimeConfig, TrainConfig
from xray_synth.data.csv_io import load_npz_paths_from_csv
from xray_synth.data.dataset import NpzAPLatCTDataset, collate_samples
from xray_synth.io.ct_sitk import has_sitk
from xray_synth.models.unet3d import UNet3D
from xray_synth.train.eval import eval_and_save_examples
from xray_synth.train.loops import train_one_epoch


def _pick_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; switching to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AP -> LAT synthesis baseline (latent 3D UNet)")

    # Inputs/Outputs
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    # Lat transforms
    p.add_argument("--no_export_flipud", action="store_true",
                   help="DEBUG: disable Streamlit flipud(Z) in forward projection.")
    p.add_argument("--lat_invert", action="store_true", help="DEBUG: invert LAT (1-lat).")
    p.add_argument("--lat_rot_k", type=int, default=0, help="DEBUG: rot90 k times on (Z,Y).")
    p.add_argument("--lat_flip_lr", action="store_true", help="DEBUG: flip Y axis.")

    # CT fallback preprocessing
    p.add_argument("--hu_min", type=float, default=-1000.0)
    p.add_argument("--hu_max", type=float, default=400.0)
    p.add_argument("--size_z", type=int, default=256)
    p.add_argument("--size_y", type=int, default=256)
    p.add_argument("--size_x", type=int, default=256)

    # Train
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--latent_down", type=int, default=4)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--w_latent", type=float, default=1.0)
    p.add_argument("--w_lat", type=float, default=0.1)

    # Runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_examples", type=int, default=10)
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--log_every", type=int, default=50)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    project_root = Path(__file__).resolve().parents[3]  # .../src/xray_synth/cli -> repo root

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_npzs = load_npz_paths_from_csv(args.train_csv)
    val_npzs = load_npz_paths_from_csv(args.val_csv)
    if not train_npzs:
        raise RuntimeError(f"No train npz paths found in {args.train_csv}")
    if not val_npzs:
        raise RuntimeError(f"No val npz paths found in {args.val_csv}")

    data_cfg = DataConfig(
        target_zyx=(args.size_z, args.size_y, args.size_x),
        hu_clip=(args.hu_min, args.hu_max),
        load_ct=True,
    )

    lat_cfg = LatTransformConfig(
        export_flipud=not args.no_export_flipud,
        invert=args.lat_invert,
        rot_k=args.lat_rot_k,
        flip_lr=args.lat_flip_lr,
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
        latent_down=args.latent_down,
        base_channels=args.base,
        w_latent=args.w_latent,
        w_lat=args.w_lat,
        log_every=args.log_every,
    )

    runtime_cfg = RuntimeConfig(device=args.device, amp=bool(args.amp))

    if not has_sitk():
        print("[warn] SimpleITK not available. OK only if NPZs include ct_zyx_norm.")
        print("       If missing CT in NPZ, install: pip install SimpleITK")

    device = _pick_device(runtime_cfg.device)
    amp_enabled = bool(runtime_cfg.amp and device.type == "cuda")

    print(f"[info] device={device} amp={amp_enabled}")
    print(f"[info] target_zyx={data_cfg.target_zyx} hu_clip={data_cfg.hu_clip}")
    print(f"[info] lat_transform: export_flipud={lat_cfg.export_flipud} invert={lat_cfg.invert} rot_k={lat_cfg.rot_k} flip_lr={lat_cfg.flip_lr}")
    print(f"[info] train_npz={len(train_npzs)} val_npz={len(val_npzs)}")

    ds_train = NpzAPLatCTDataset(
        project_root=project_root,
        npz_paths=train_npzs,
        target_zyx=data_cfg.target_zyx,
        hu_clip=data_cfg.hu_clip,
        load_ct=data_cfg.load_ct,
    )
    ds_val = NpzAPLatCTDataset(
        project_root=project_root,
        npz_paths=val_npzs,
        target_zyx=data_cfg.target_zyx,
        hu_clip=data_cfg.hu_clip,
        load_ct=data_cfg.load_ct,
    )

    dl_train = DataLoader(ds_train, batch_size=train_cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_samples)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_samples)

    model = UNet3D(base=train_cfg.base_channels).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    ckpt_path = out_dir / "checkpoint.pt"

    # Load checkpoint if requested / resume if present
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ck["model"])
        print(f"[eval] loaded checkpoint: {args.ckpt}")
    elif ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ck["model"])
        print(f"[resume] loaded checkpoint: {ckpt_path}")
    else:
        print(f"[train] epochs={train_cfg.epochs} batch={train_cfg.batch_size} lr={train_cfg.lr} latent_down={train_cfg.latent_down}")
        for ep in range(1, train_cfg.epochs + 1):
            ep_t0 = time.time()
            loss = train_one_epoch(
                model=model,
                loader=dl_train,
                optim=optim,
                device=device,
                latent_down=train_cfg.latent_down,
                w_latent=train_cfg.w_latent,
                w_lat=train_cfg.w_lat,
                amp=amp_enabled,
                export_flipud=lat_cfg.export_flipud,
                invert=lat_cfg.invert,
                rot_k=lat_cfg.rot_k,
                flip_lr=lat_cfg.flip_lr,
                log_every=train_cfg.log_every,
            )
            dt = time.time() - ep_t0
            print(f"[train] epoch {ep:03d}/{train_cfg.epochs} loss={loss:.6f} time={dt:.1f}s")

        torch.save({"model": model.state_dict(), "args": vars(args)}, str(ckpt_path))
        print(f"[train] saved: {ckpt_path}")

    summary = eval_and_save_examples(
        model=model,
        loader=dl_val,
        device=device,
        latent_down=train_cfg.latent_down,
        out_dir=out_dir,
        num_examples=args.num_examples,
        export_flipud=lat_cfg.export_flipud,
        invert=lat_cfg.invert,
        rot_k=lat_cfg.rot_k,
        flip_lr=lat_cfg.flip_lr,
    )

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "metrics.txt").write_text(
        f"count={summary['count']}\n"
        f"psnr_mean={summary['psnr_mean']:.4f}  psnr_std={summary['psnr_std']:.4f}\n"
        f"ssim_mean={summary['ssim_mean']:.4f}  ssim_std={summary['ssim_std']:.4f}\n",
        encoding="utf-8",
    )

    print("[eval] " + (out_dir / "metrics.txt").read_text(encoding="utf-8").strip())
    print(f"[eval] examples saved in: {out_dir / 'examples'}")


if __name__ == "__main__":
    main()
