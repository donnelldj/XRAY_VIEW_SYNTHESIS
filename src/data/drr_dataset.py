from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DRRSample:
    case_id: str
    ap: torch.Tensor     # (1, H, W)
    lat: torch.Tensor    # (1, H, W)
    npz_path: str


class DRRPairDataset(Dataset):
    """
    Loads AP/LAT DRR pairs from the .npz files created by tools/prepare_dataset.py.

    Each .npz contains:
      - ap: (H, W) float32
      - lat: (H, W) float32
      - case_id: str
      - mhd_path: str
      - spacing_zyx: (3)
    """

    def __init__(
        self,
        csv_path: str | Path,
        normalize_mode: str = "minmax01",
        return_meta: bool = False,
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if "npz_path" not in self.df.columns:
            raise ValueError(f"Expected 'npz_path' column in {self.csv_path}")

        self.normalize_mode = normalize_mode
        self.return_meta = return_meta

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

        ap = npz["ap"].astype(np.float32)
        lat = npz["lat"].astype(np.float32)

        ap = self._normalize(ap, self.normalize_mode)
        lat = self._normalize(lat, self.normalize_mode)

        ap_t = torch.from_numpy(ap)[None, ...]   # (1,H,W)
        lat_t = torch.from_numpy(lat)[None, ...] # (1,H,W)

        case_id = str(npz["case_id"]) if "case_id" in npz.files else str(row.get("case_id", "unknown"))

        if self.return_meta:
            meta: Dict[str, str] = {
                "case_id": case_id,
                "npz_path": str(npz_path.as_posix()),
                "mhd_path": str(npz["mhd_path"]) if "mhd_path" in npz.files else str(row.get("mhd_path", "")),
            }
            return ap_t, lat_t, meta

        return ap_t, lat_t
