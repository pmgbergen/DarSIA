"""Retained photograph-coordinate helpers, without source or QA decisions.

The point transform is the original ``phase_ffac_rig_geometry`` implementation.
The top-mask implementation is the original ``build_grain_top_revision``
``top_curve_mask`` including its complete-canvas origin, endpoint continuation
and ceil rule. Neither helper fits observations or certifies their accuracy.
"""
from __future__ import annotations

import numpy as np


def homography(value):
    matrix = np.asarray(value, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Rig transform must be a finite 3x3 homography")
    if abs(np.linalg.det(matrix)) < 1e-14:
        raise ValueError("Rig transform is singular")
    return matrix


def transform_points(points, matrix):
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Rig points must be finite Nx2 native coordinates")
    projected = homography(matrix) @ np.column_stack((points, np.ones(len(points)))).T
    if np.any(np.abs(projected[2]) < 1e-12):
        raise ValueError("Rig transform crosses a projective horizon")
    return (projected[:2] / projected[2]).T


def top_curve_mask(curve, H, shape, origin):
    curve = np.asarray(curve, float)
    if curve.ndim != 1 or not np.isfinite(curve).all():
        raise ValueError("Finite native photographic curve required")
    pts = transform_points(np.column_stack((np.arange(len(curve)), curve)), H)
    order = np.argsort(pts[:, 0])
    x = pts[order, 0]
    y = pts[order, 1]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Surface folds under frozen homography")
    boundary = np.interp(np.arange(shape[1]) + origin[0], x, y)
    keep = np.arange(shape[0])[:, None] + origin[1] >= np.ceil(boundary)[None, :]
    if not keep[-1].all() or np.any(keep[:-1] & ~keep[1:]):
        raise ValueError("Top is not column-prefix-only removal")
    return keep, boundary
