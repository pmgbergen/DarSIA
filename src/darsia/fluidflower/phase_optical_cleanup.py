"""Experimental optical support cleanup in physical units; no learned completion.

Gas pixels retain their archived raw classification. This is an explicit preview,
not a production decision or a concentration estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

VERSION = "optical-cleanup-preview-v1"


@dataclass(frozen=True)
class Settings:
    threshold: float = 0.5
    sigma_mm: float = 0.0
    hole_diameter_mm: float = 0.0

    def __post_init__(self):
        bounds = ((self.threshold, .01, .99), (self.sigma_mm, 0, 3),
                  (self.hole_diameter_mm, 0, 5))
        if any(not math.isfinite(v) or not lo <= v <= hi for v, lo, hi in bounds):
            raise ValueError("Invalid optical cleanup settings")


def apply(presence, raw_labels, domain, spacing_m, settings=Settings()):
    """Smooth scores, threshold, then fill only small enclosed water components.

    Gaussian sigma is specified in mm, not pixels. Normalization avoids treating
    pixels outside the analysis domain as water evidence. Filling never closes
    an open water channel or a component touching the domain/image boundary.
    Both component area and physical bounding-box diagonal must fit the chosen
    hole diameter, preventing long thin channels from passing an area-only gate.
    """
    p = np.asarray(presence, dtype=np.float32)
    raw = np.asarray(raw_labels)
    valid = np.asarray(domain, dtype=bool)
    spacing = np.asarray(spacing_m, dtype=float) * 1000
    if p.ndim != 2 or p.shape != raw.shape or p.shape != valid.shape:
        raise ValueError("Optical scores, labels and domain must share a 2D grid")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Positive y/x pixel spacing required")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Optical scores must be finite and in [0, 1]")
    if not np.isin(raw, (0, 1, 2)).all():
        raise ValueError("Unknown raw phase label")
    if settings.sigma_mm:
        sy, sx = settings.sigma_mm / spacing
        weights = valid.astype(np.float32)
        den = cv2.GaussianBlur(weights, (0, 0), sigmaX=float(sx), sigmaY=float(sy),
                               borderType=cv2.BORDER_CONSTANT)
        num = cv2.GaussianBlur(p * weights, (0, 0), sigmaX=float(sx), sigmaY=float(sy),
                               borderType=cv2.BORDER_CONSTANT)
        p = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-7)
    active = (p >= settings.threshold) & valid
    gas = (raw == 2) & valid
    active |= gas
    filled = 0
    if settings.hole_diameter_mm:
        water = (valid & ~active).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=8)
        # Invalid domain cells count as exterior, even for internal obstacles.
        exterior = cv2.dilate((~valid).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        exterior[[0, -1], :] = True
        exterior[:, [0, -1]] = True
        blocked = np.unique(components[exterior])
        d = settings.hole_diameter_mm
        allowed = (stats[:, cv2.CC_STAT_AREA] * np.prod(spacing) <= math.pi * d * d / 4)
        allowed &= np.hypot(stats[:, cv2.CC_STAT_HEIGHT] * spacing[0],
                            stats[:, cv2.CC_STAT_WIDTH] * spacing[1]) <= d
        allowed[blocked] = False
        allowed[0] = False
        holes = allowed[components]
        filled = int(holes.sum())
        active |= holes
    labels = np.where(gas, 2, active).astype(np.uint8)
    # Cleanup has no authority outside its domain. In particular, a zero-effect
    # preview must reproduce the archived raw map, including its exterior.
    labels[~valid] = raw[~valid]
    report = dict(added_pixels=int(((labels > 0) & (raw == 0)).sum()),
                  removed_pixels=int(((labels == 0) & (raw > 0)).sum()),
                  filled_pixels=filled, gas_changed_pixels=int(((labels == 2) != (raw == 2)).sum()))
    return labels, report
