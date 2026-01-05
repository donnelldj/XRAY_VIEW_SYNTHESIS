from __future__ import annotations

import numpy as np
import cv2

def normalize_for_display(image_array: np.ndarray) -> np.ndarray:
    """Safe normalization for 0..1 Streamlit display."""
    img_min, img_max = float(np.min(image_array)), float(np.max(image_array))
    if img_max - img_min == 0:
        return image_array
    return (image_array - img_min) / (img_max - img_min)

def save_16bit_png(array: np.ndarray, path: str) -> None:
    """Saves a 2D array as a 16-bit PNG safely."""
    array = array.astype(np.float64)
    img_min, img_max = float(array.min()), float(array.max())
    if img_max - img_min == 0:
        normalized = np.zeros_like(array, dtype=np.uint16)
    else:
        normalized = (65535.0 * (array - img_min) / (img_max - img_min)).astype(np.uint16)
    cv2.imwrite(path, normalized)
