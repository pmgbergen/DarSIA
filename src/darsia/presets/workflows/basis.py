"""Shared calibration/analysis basis utilities."""

from __future__ import annotations

import numpy as np


def label_ids_from_image(labels_img) -> list[int]:
    """Extract sorted non-negative label ids from an image-like labels container."""

    return sorted([int(label) for label in np.unique(labels_img.img) if label >= 0])
