from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DataConfig:
    """Dataset + preprocessing configuration."""
    target_zyx: Tuple[int, int, int] = (256, 256, 256)
    hu_clip: Tuple[float, float] = (-1000.0, 400.0)
    load_ct: bool = True  # CT supervision required for this baseline


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 16
    batch_size: int = 1
    lr: float = 2e-4
    seed: int = 123
    latent_down: int = 4
    base_channels: int = 16
    w_latent: float = 1.0
    w_lat: float = 0.1
    log_every: int = 50


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime / device configuration."""
    device: str = "cuda"
    amp: bool = False


@dataclass(frozen=True)
class LatTransformConfig:
    """
    IMPORTANT: export_flipud=True MUST match Streamlit exporter GT convention.

    Streamlit GT contract:
      ap  = flipud(mean(ct_norm, axis=Y))  # (Z,X)
      lat = flipud(mean(ct_norm, axis=X))  # (Z,Y)

    Therefore, our forward projector must do:
      lat_pred = flipud(mean(ct_pred, axis=X))
    """
    export_flipud: bool = True

    # Debug-only knobs (keep defaults for “real” results)
    invert: bool = False
    rot_k: int = 0
    flip_lr: bool = False
