import numpy as np
import SimpleITK as sitk


def load_ct(path: str) -> np.ndarray:
    """
    Load CT volume (DICOM series or NIfTI) as numpy array
    Returns array in Z,Y,X order
    """
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    return vol


def normalize_hu(vol: np.ndarray, min_hu=-1000, max_hu=1000):
    vol = np.clip(vol, min_hu, max_hu)
    vol = (vol - min_hu) / (max_hu - min_hu)
    return vol


def project_ap(vol: np.ndarray) -> np.ndarray:
    """
    AP projection (0°): sum over Z axis
    """
    ap = np.sum(vol, axis=0)
    ap = ap / (ap.max() + 1e-8)
    return ap


def project_lat(vol: np.ndarray) -> np.ndarray:
    """
    Lateral projection (90°): sum over X axis
    """
    lat = np.sum(vol, axis=2)
    lat = lat / (lat.max() + 1e-8)
    return lat
