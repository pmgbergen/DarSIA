"""Presentation coordinates for observed references, with point 4 as origin."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from darsia.fluidflower.series_geometry import RigFrame


REFERENCE_POINT_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


def _point(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        return None
    return point if point.shape == (2,) and np.all(np.isfinite(point)) else None


def _record(frame: RigFrame) -> Mapping[str, Any]:
    record = frame.reference_geometry
    return record if isinstance(record, Mapping) else {}


def _relative_mm(points: Mapping[str, np.ndarray]) -> dict[str, list[float] | None]:
    origin = points.get("bottom_left")
    result = dict.fromkeys(REFERENCE_POINT_NAMES)
    if origin is None:
        return result
    for name, point in points.items():
        xy = (point - origin) * 1000.0
        xy[np.abs(xy) < 1e-9] = 0.0
        result[name] = xy.tolist()
    return result


def reference_point_coordinates_mm(frame: RigFrame) -> dict[str, list[float] | None]:
    """Return observed reference x/y in mm relative to the bottom-left point.

    This is a display coordinate system. The rectangular native domain, saved
    frame and measurement records are never modified. In particular, a general
    quadrilateral may have a negative top-left x even though its internal
    rectangular domain starts at the metric bounding box's leftmost edge.
    """
    record = _record(frame)
    metric = record.get("metric_corners_xy")
    if isinstance(metric, Mapping):
        metric_points = {name: _point(metric.get(name)) for name in REFERENCE_POINT_NAMES}
        if all(point is not None for point in metric_points.values()):
            return _relative_mm(metric_points)

    recorded_photo = record.get("photo_corners_rc")
    photo = recorded_photo if isinstance(recorded_photo, Mapping) else frame.corners
    photo_points = {
        name: point for name in REFERENCE_POINT_NAMES
        if (point := _point(photo.get(name))) is not None
    }
    result = dict.fromkeys(REFERENCE_POINT_NAMES)
    if "bottom_left" not in photo_points:
        return result
    result["bottom_left"] = [0.0, 0.0]

    try:
        dimensions = np.asarray([frame.width_m, frame.height_m], dtype=float)
    except (TypeError, ValueError):
        return result
    if not np.all(np.isfinite(dimensions)) or not np.all(dimensions > 0):
        return result

    mapped = {}
    for name, point in photo_points.items():
        try:
            value = _point(frame.pixel_to_rig(*point))
        except (ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
            value = None
        if value is not None:
            mapped[name] = value
    if "bottom_left" in mapped:
        return _relative_mm(mapped)

    # An incomplete legacy rectangular frame can still state known reference
    # coordinates, but cannot provide a projective map for arbitrary points.
    if not record or record.get("shape", "rectangle") == "rectangle":
        width, height = dimensions
        rectangular = {
            "top_left": np.array([0.0, height]),
            "top_right": np.array([width, height]),
            "bottom_right": np.array([width, 0.0]),
            "bottom_left": np.zeros(2),
        }
        return _relative_mm({name: rectangular[name] for name in photo_points})
    return result


def point_description(frame: RigFrame, name: str) -> str:
    """Read a per-point description key without inferring it from estimates.

    The initial UI choices are ``rig_frame`` and ``top_of_free_water``. Keep
    nonempty custom or future keys intact rather than rewriting saved data.
    """
    descriptions = _record(frame).get("point_descriptions")
    description = descriptions.get(name) if isinstance(descriptions, Mapping) else None
    return description.strip() if isinstance(description, str) and description.strip() else "rig_frame"


__all__ = ["REFERENCE_POINT_NAMES", "reference_point_coordinates_mm", "point_description"]
