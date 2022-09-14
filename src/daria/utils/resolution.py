"""
Utiliy functions for transforming resolution.
"""

import cv2
import numpy as np


def resize(img: np.ndarray, scale_percent: float = 100) -> np.ndarray:
    """
    Resizes image while keeping the aspect ratio.

    Args:
        img (np.ndarray): image array
        scale_percent (float): positive scaling factor (in %)

    Returns:
        np.ndarray: resized image
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
