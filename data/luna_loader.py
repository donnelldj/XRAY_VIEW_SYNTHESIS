from pathlib import Path
import SimpleITK as sitk
import numpy as np

def load_ct(mhd_path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(mhd_path))
    vol = sitk.GetArrayFromImage(img)
    return vol.astype(np.float32)
