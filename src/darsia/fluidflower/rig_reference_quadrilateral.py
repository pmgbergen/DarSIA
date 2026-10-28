"""Calibrate a rectangular rig domain from four measured reference points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
_FORMAT = "darsia-rig-reference-quadrilateral-v1"


def _positive(measurements: Mapping[str, Any], name: str) -> float:
    try:
        value = float(measurements[name])
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(f"Enter a positive physical {name} measurement") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Enter a positive physical {name} measurement")
    return value


def _crosses(points: np.ndarray) -> np.ndarray:
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    return edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]


def _convex(points: np.ndarray, *, positive: bool) -> bool:
    scale = float(np.max(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        return False
    crosses = _crosses(points) * (1 if positive else -1)
    return bool(np.all(crosses > 1e-10 * scale * scale))


def _normalization(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    distance = float(np.mean(np.linalg.norm(points - centre, axis=1)))
    scale = np.sqrt(2.0) / distance
    transform = np.array(
        [[scale, 0.0, -scale * centre[0]],
         [0.0, scale, -scale * centre[1]],
         [0.0, 0.0, 1.0]]
    )
    normalized = (np.column_stack((points, np.ones(4))) @ transform.T)[:, :2]
    return transform, normalized


def _homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit four points using normalized DLT, avoiding pixel-scale conditioning."""
    source_transform, source_n = _normalization(source)
    target_transform, target_n = _normalization(target)
    rows = []
    for (x, y), (u, v) in zip(source_n, target_n):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, singular, vectors = np.linalg.svd(np.asarray(rows, dtype=float))
    if singular[-1] <= 1e-12 * singular[0]:
        raise ValueError("Reference points do not define a stable perspective map")
    normalized = vectors[-1].reshape(3, 3)
    result = np.linalg.inv(target_transform) @ normalized @ source_transform
    return result / np.linalg.norm(result)


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    mapped = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    denominator = mapped[:, 2]
    scale = float(np.max(np.abs(denominator)))
    if scale <= 0 or not (
        np.all(denominator > 1e-10 * scale)
        or np.all(denominator < -1e-10 * scale)
    ):
        raise ValueError("The rectangular domain crosses a perspective horizon")
    result = mapped[:, :2] / denominator[:, None]
    if not np.all(np.isfinite(result)):
        raise ValueError("Reference points produce non-finite image geometry")
    return result


def _custom_metric(measurements: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    lengths = {
        name: _positive(measurements, name)
        for name in ("top", "right", "bottom", "left", "diagonal_ac")
    }
    bottom, left, diagonal = (lengths[name] for name in ("bottom", "left", "diagonal_ac"))
    tolerance = 1e-10 * max(lengths.values())
    if not abs(left - bottom) + tolerance < diagonal < left + bottom - tolerance:
        raise ValueError("The left, bottom and diagonal measurements cannot form a triangle")
    ax = (left * left + bottom * bottom - diagonal * diagonal) / (2 * bottom)
    ay = np.sqrt(max(0.0, left * left - ax * ax))
    a, c, d = np.array([ax, ay]), np.array([bottom, 0.0]), np.zeros(2)
    top, right = lengths["top"], lengths["right"]
    if not abs(top - right) + tolerance < diagonal < top + right - tolerance:
        raise ValueError("The top, right and diagonal measurements cannot form a triangle")
    direction = (c - a) / diagonal
    along = (top * top - right * right + diagonal * diagonal) / (2 * diagonal)
    height = np.sqrt(max(0.0, top * top - along * along))
    foot = a + along * direction
    normal = np.array([-direction[1], direction[0]])
    candidates = [np.array([a, foot + sign * height * normal, c, d]) for sign in (-1, 1)]
    valid = [points for points in candidates if _convex(points, positive=False)]
    if len(valid) != 1:
        raise ValueError("These measurements do not define a convex quadrilateral")
    points = valid[0]
    points -= np.min(points, axis=0)
    return points, lengths


def reference_frame_geometry(
    photo_corners_rc: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    *,
    shape: str = "rectangle",
    measurements_m: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a rig frame and its observed, physically measured references.

    Points are supplied in top-left, top-right, bottom-right, bottom-left order.
    These names describe the ordering, not a requirement to use the actual rig
    corners. Rectangle dimensions use ``width`` and ``height`` (or four equal
    opposite sides). A general quadrilateral requires all four side lengths and
    diagonal ``diagonal_ac`` from top-left to bottom-right, in metres.

    A nonrectangular physical reference is embedded in its metric bounding box.
    The returned frame corners describe that rectangular domain, preserving the
    existing rig-coordinate and native-crop semantics without stretching the
    measured quadrilateral into a false rectangle.
    """
    try:
        ordered = [photo_corners_rc[name] for name in _NAMES] if isinstance(photo_corners_rc, Mapping) else photo_corners_rc
        photo_rc = np.asarray(ordered, dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Supply four finite image points in TL, TR, BR, BL order") from error
    if photo_rc.shape != (4, 2) or not np.all(np.isfinite(photo_rc)):
        raise ValueError("Supply four finite image points in TL, TR, BR, BL order")
    photo_xy = photo_rc[:, ::-1]
    if not _convex(photo_xy, positive=True):
        raise ValueError("Image points must form a convex, uncrossed TL, TR, BR, BL quadrilateral")
    if shape == "rectangle":
        if "width" in measurements_m or "height" in measurements_m:
            width = _positive(measurements_m, "width")
            height = _positive(measurements_m, "height")
            lengths = {"top": width, "right": height, "bottom": width, "left": height}
            for name in lengths.keys() & measurements_m.keys():
                if not np.isclose(_positive(measurements_m, name), lengths[name], rtol=1e-8, atol=1e-10):
                    raise ValueError("Rectangle side measurements must agree with width and height")
        else:
            lengths = {name: _positive(measurements_m, name) for name in ("top", "right", "bottom", "left")}
            if not (np.isclose(lengths["top"], lengths["bottom"], rtol=1e-8, atol=1e-10)
                    and np.isclose(lengths["left"], lengths["right"], rtol=1e-8, atol=1e-10)):
                raise ValueError("A rectangle requires equal opposite side measurements")
            width = (lengths["top"] + lengths["bottom"]) / 2
            height = (lengths["left"] + lengths["right"]) / 2
        metric = np.array([[0.0, height], [width, height], [width, 0.0], [0.0, 0.0]])
        domain_rc = photo_rc.copy()
    elif shape == "quadrilateral":
        metric, lengths = _custom_metric(measurements_m)
        width, height = np.ptp(metric, axis=0)
        domain_xy = np.array([[0.0, height], [width, height], [width, 0.0], [0.0, 0.0]])
        matrix = _homography(photo_xy, metric)
        domain_rc = _project(domain_xy, np.linalg.inv(matrix))[:, ::-1]
        if not _convex(domain_rc[:, ::-1], positive=True):
            raise ValueError("The derived rectangular image domain is invalid")
    else:
        raise ValueError("Reference shape must be rectangle or quadrilateral")
    if not np.isfinite(width) or not np.isfinite(height) or min(width, height) <= 0:
        raise ValueError("Reference measurements must define positive finite bounds")
    record = {
        "format": _FORMAT,
        "shape": shape,
        "photo_corners_rc": {name: photo_rc[index].tolist() for index, name in enumerate(_NAMES)},
        "metric_corners_xy": {name: metric[index].tolist() for index, name in enumerate(_NAMES)},
        "measurements_m": lengths,
    }
    return {
        "corners": {name: tuple(domain_rc[index]) for index, name in enumerate(_NAMES)},
        "width_m": float(width),
        "height_m": float(height),
        "reference_geometry": record,
    }
