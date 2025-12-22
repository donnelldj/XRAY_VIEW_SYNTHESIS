from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.geo.backproject import backproject_parallel_beam


class BackprojectToLatDataset(Dataset):
    """
    Loads AP/LAT from .npz and returns:
      x: backprojected volume V_bp (1, Z, Y, X)
      y: GT lat image             (1, H, W)

    This is the first training target we use to prove:
      AP -> BP volume -> 3D UNet -> Lat
    """

    def __init__(
        self,
        csv_path: str | Path,
        crop_zyx: tuple[int, int, int] = (96, 256, 256),
        normalize_mode: str = "minmax01",
        return_meta: bool = False,
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.crop_zyx = crop_zyx
        self.normalize_mode = normalize_mode
        self.return_meta = return_meta

        if "npz_path" not in self.df.columns:
            raise ValueError(f"Expected 'npz_path' column in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _normalize(img: np.ndarray, mode: str) -> np.ndarray:
        img = img.astype(np.float32)
        if mode == "none":
            return img
        if mode == "minmax01":
            mn = float(img.min())
            mx = float(img.max())
            if mx - mn < 1e-8:
                return np.zeros_like(img, dtype=np.float32)
            return (img - mn) / (mx - mn)
        if mode == "meanstd":
            mu = float(img.mean())
            sd = float(img.std())
            if sd < 1e-8:
                return img - mu
            return (img - mu) / sd
        raise ValueError(f"Unknown normalize_mode: {mode}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npz_path = Path(str(row["npz_path"]))
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing npz: {npz_path}")

        npz = np.load(npz_path, allow_pickle=True)
        ap = npz["ap"].astype(np.float32)   # (H,W)
        lat = npz["lat"].astype(np.float32) # (H,W)

        ap = self._normalize(ap, self.normalize_mode)
        lat = self._normalize(lat, self.normalize_mode)

        # Backproject AP into a 3D volume (Z,Y,X)
        vbp = backproject_parallel_beam(ap, out_zyx=self.crop_zyx, axis=0)
        vbp = self._normalize(vbp, self.normalize_mode)

        x = torch.from_numpy(vbp)[None, ...]  # (1,Z,Y,X)
        y = torch.from_numpy(lat)[None, ...]  # (1,H,W)

        if self.return_meta:
            meta: Dict[str, str] = {
                "case_id": str(npz["case_id"]) if "case_id" in npz.files else str(row.get("case_id", "unknown")),
                "npz_path": str(npz_path.as_posix()),
            }
            return x, y, meta

        return x, y
