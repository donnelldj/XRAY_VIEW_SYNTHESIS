from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from xray_synth.io.npz import resolve_path
from xray_synth.io.ct_sitk import read_ct_preprocessed_zyx


@dataclass
class Sample:
    """Single training sample."""
    ap_zx: torch.Tensor               # (1,Z,X)
    lat_zy: torch.Tensor              # (1,Z,Y)
    ct_zyx: Optional[torch.Tensor]    # (1,Z,Y,X) or None
    case_id: str


def collate_samples(batch: List[Sample]):
    """Convert List[Sample] -> dict of batched tensors."""
    ap = torch.stack([b.ap_zx for b in batch], dim=0)    # (B,1,Z,X)
    lat = torch.stack([b.lat_zy for b in batch], dim=0)  # (B,1,Z,Y)
    ct = None if batch[0].ct_zyx is None else torch.stack([b.ct_zyx for b in batch], dim=0)  # (B,1,Z,Y,X)
    case_id = [b.case_id for b in batch]
    return {"ap_zx": ap, "lat_zy": lat, "ct_zyx": ct, "case_id": case_id}


class NpzAPLatCTDataset(Dataset):
    """
    Expected NPZ keys from Streamlit exporter:
      - case_id (str)
      - mhd_path (str)
      - ap:  (Z,X) float32  = flipud(mean(ct_norm, axis=Y))
      - lat: (Z,Y) float32  = flipud(mean(ct_norm, axis=X))
      - ct_zyx_norm (Z,Y,X) OPTIONAL (if exporter includes CT)

    Supervision:
      - Prefer ct_zyx_norm if present (perfect match to exporter)
      - Else fallback to mhd_path + SITK read/resample/normalize
    """
    def __init__(
        self,
        project_root: Path,
        npz_paths: List[str],
        target_zyx: Tuple[int, int, int] = (256, 256, 256),
        hu_clip: Tuple[float, float] = (-1000.0, 400.0),
        load_ct: bool = True,
    ):
        self.project_root = project_root
        self.npz_paths = npz_paths
        self.target_zyx = target_zyx
        self.hu_clip = hu_clip
        self.load_ct = load_ct

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> Sample:
        p = resolve_path(self.project_root, self.npz_paths[idx])
        d = np.load(str(p), allow_pickle=True)

        case_id = str(d["case_id"])
        ap = d["ap"].astype(np.float32)    # (Z,X) already flipud in exporter
        lat = d["lat"].astype(np.float32)  # (Z,Y) already flipud in exporter

        if ap.ndim != 2 or lat.ndim != 2:
            raise RuntimeError(f"Bad shapes in {p}: ap={ap.shape} lat={lat.shape}")

        ap_t = torch.from_numpy(ap)[None, ...]     # (1,Z,X)
        lat_t = torch.from_numpy(lat)[None, ...]   # (1,Z,Y)

        ct_t: Optional[torch.Tensor] = None
        if self.load_ct:
            if "ct_zyx_norm" in d.files:
                ct = d["ct_zyx_norm"].astype(np.float32)  # (Z,Y,X) in [0,1]
            else:
                mhd_path = str(d["mhd_path"])
                mhd_resolved = resolve_path(self.project_root, mhd_path)
                if not mhd_resolved.exists():
                    raise FileNotFoundError(
                        f"CT mhd_path not found for case_id={case_id}: {mhd_path} (resolved={mhd_resolved})"
                    )
                ct = read_ct_preprocessed_zyx(
                    str(mhd_resolved),
                    target_zyx=self.target_zyx,
                    hu_clip=self.hu_clip,
                )

            if ct.shape != self.target_zyx:
                raise RuntimeError(f"CT shape mismatch for {case_id}: got {ct.shape}, expected {self.target_zyx}")

            ct_t = torch.from_numpy(ct)[None, ...]  # (1,Z,Y,X)

        return Sample(ap_t.float(), lat_t.float(), ct_t.float() if ct_t is not None else None, case_id)
