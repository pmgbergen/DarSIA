"""Human-confirmed, display-only water lines on an explicitly bound photograph.

The caller verifies the actual photo and existing geometry descriptors before
providing a basis. No helper here writes files or modifies a production model.
    Three observed bolt points define an optional separate display cross-check,
    never a new registration. The primary height uses the existing own geometry;
    missing bolt observations remain unavailable and are never inferred from it.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import re

import numpy as np

BASIS_FORMAT = "darsia-water-surface-basis-v1"
ACTION_FORMAT = "darsia-water-surface-confirmation-v1"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _points(value, count, name):
    if (not isinstance(value, (list, tuple)) or len(value) != count
            or any(not isinstance(row, (list, tuple)) or len(row) != 2
                   or any(type(v) not in (int, float) for v in row) for row in value)):
        raise ValueError(f"Explicit finite {name} photo points are required")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite {name} photo points")
    return result


def _basis(value):
    keys = {"format", "run_id", "photo", "photo_shape", "photo_to_metric_h",
            "left_bolts_photo_xy", "bolt_heights_mm", "bottom_edge_photo_xy",
            "own_top_photo_xy", "transported_top_photo_xy", "port_photo_xy", "provenance",
            "own_geometry_origin_metric_mm", "own_geometry_up_metric_xy"}
    if (not isinstance(value, dict) or set(value) != keys
            or value["format"] != BASIS_FORMAT
            or not isinstance(value["run_id"], str) or not value["run_id"]
            or not isinstance(value["provenance"], dict)):
        raise ValueError("An exact bound water-surface display basis is required")
    result = json.loads(_json(value))
    photo = result["photo"]
    if (not isinstance(photo, dict) or set(photo) != {"path", "sha256", "size_bytes"}
            or not isinstance(photo["path"], str) or not photo["path"]
            or re.fullmatch("[0-9a-f]{64}", str(photo["sha256"])) is None
            or type(photo["size_bytes"]) is not int or photo["size_bytes"] < 0):
        raise ValueError("An exact actual photograph descriptor is required")
    shape = result["photo_shape"]
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(v) is not int or v < 2 for v in shape)):
        raise ValueError("An explicit photo pixel shape is required")
    if _json(result["bolt_heights_mm"]) != _json([0, 195, 390]):
        raise ValueError("Only the existing 0/195/390 mm bolt observations are supported")
    for name, count in (("left_bolts_photo_xy", 3), ("bottom_edge_photo_xy", 2),
                        ("own_top_photo_xy", 2), ("transported_top_photo_xy", 2)):
        if result[name] is not None:
            _points(result[name], count, name)
    if result["port_photo_xy"] is not None:
        _points([result["port_photo_xy"]], 1, "port")
    for name in ("own_geometry_origin_metric_mm", "own_geometry_up_metric_xy"):
        if result[name] is not None:
            vector = _points([result[name]], 1, name)[0]
            if name == "own_geometry_up_metric_xy" and np.dot(vector, vector) == 0:
                raise ValueError("The existing geometry's up direction cannot be zero")
    h = result["photo_to_metric_h"]
    if h is not None:
        if (not isinstance(h, list) or len(h) != 3
                or any(not isinstance(row, list) or len(row) != 3
                       or any(type(v) not in (int, float) for v in row) for row in h)):
            raise ValueError("An existing photo-to-metric homography is required")
        matrix = np.asarray(h, dtype=float)
        if not np.isfinite(matrix).all() or np.linalg.det(matrix) == 0:
            raise ValueError("The existing photo-to-metric homography is singular")
    return result


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf8")).hexdigest()


def _transform(points, h):
    result = np.c_[points, np.ones(len(points))] @ np.asarray(h, dtype=float).T
    if np.any(result[:, 2] == 0):
        raise ValueError("A display point lies at a projective horizon")
    result = result[:, :2] / result[:, 2:3]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite display transform")
    return result


def _line(points):
    if np.array_equal(points[0], points[1]):
        raise ValueError("Two distinct water-line endpoints are required")
    return np.cross(np.r_[points[0], 1.0], np.r_[points[1], 1.0])


def _intersection(first, second):
    point = np.cross(_line(first), _line(second))
    if point[2] == 0:
        return None
    result = point[:2] / point[2]
    return result if np.isfinite(result).all() else None


def _clip(points, shape):
    line = _line(points)
    height, width = shape
    candidates = []
    for x in (0.0, float(width - 1)):
        if line[1] != 0:
            y = -(line[0] * x + line[2]) / line[1]
            if 0 <= y <= height - 1:
                candidates.append([x, float(y)])
    for y in (0.0, float(height - 1)):
        if line[0] != 0:
            x = -(line[1] * y + line[2]) / line[0]
            if 0 <= x <= width - 1:
                candidates.append([float(x), y])
    unique = np.unique(np.asarray(candidates, dtype=float).reshape(-1, 2), axis=0)
    if len(unique) < 2:
        raise ValueError("The moved line does not cross the bound photograph")
    distances = ((unique[:, None] - unique[None, :]) ** 2).sum(axis=2)
    a, b = np.unravel_index(np.argmax(distances), distances.shape)
    return unique[[a, b]].tolist()


def propose_water_surface(photo, *, basis):
    """Propose a visible image gradient; no physical classification or approval."""
    binding = _basis(basis)
    pixels = np.asarray(photo)
    if (pixels.shape[:2] != tuple(binding["photo_shape"])
            or pixels.ndim not in (2, 3)
            or (pixels.ndim == 3 and pixels.shape[2] != 3)
            or pixels.dtype.kind not in "fiu" or not np.isfinite(pixels).all()):
        raise ValueError("The exact finite bound grayscale/RGB photo is required")
    gray = pixels.astype(float, copy=False)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    if not np.isfinite(gray).all():
        raise ValueError("Non-finite display grayscale conversion")
    gradient = np.abs(np.diff(gray, axis=0))
    profile = np.median(gradient, axis=1)
    row = int(np.argmax(profile))
    diagnostic = dict(method="unconfirmed-image-gradient", human_confirmation_required=True,
                      physical_classification=False, gradient_score=float(profile[row]))
    if profile[row] <= 0:
        return dict(line_photo_xy=None, diagnostics=diagnostic)
    # A horizontal candidate is deliberately only a starting line for the human.
    return dict(line_photo_xy=[[0.0, row + .5], [float(gray.shape[1] - 1), row + .5]],
                diagnostics=diagnostic)


def move_water_surface_to_photo_point(line_photo_xy, point_photo_xy, *, basis):
    """Translate through a click while preserving angle in the existing metric plane."""
    binding = _basis(basis)
    if binding["photo_to_metric_h"] is None:
        raise ValueError("Existing metric-plane binding is unavailable; no angle-preserving move")
    h = binding["photo_to_metric_h"]
    line = _transform(_points(line_photo_xy, 2, "water line"), h)
    _line(line)
    click = _transform(_points([point_photo_xy], 1, "clicked"), h)[0]
    delta = line[1] - line[0]
    normal = np.array([-delta[1], delta[0]])
    moved = line + normal * (np.dot(click - line[0], normal) / np.dot(normal, normal))
    return _clip(_transform(moved, np.linalg.inv(np.asarray(h))), binding["photo_shape"])


def make_water_surface_action(basis, line_photo_xy, *, actor, confirmed_utc):
    """Called only by the actual GUI confirmation action; this is not authentication."""
    binding = _basis(basis)
    line = _points(line_photo_xy, 2, "water line")
    _line(line)
    height, width = binding["photo_shape"]
    if (np.any(line < 0) or np.any(line[:, 0] > width - 1)
            or np.any(line[:, 1] > height - 1)):
        raise ValueError("Confirmed endpoints must lie on the bound photograph")
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("An explicit human confirmation actor is required")
    try:
        when = datetime.fromisoformat(confirmed_utc.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("An explicit timezone-aware human confirmation time is required") from exc
    if when.tzinfo is None or when.utcoffset().total_seconds() != 0:
        raise ValueError("The human confirmation time must be UTC")
    return dict(format=ACTION_FORMAT, basis_sha256=_digest(binding), run_id=binding["run_id"],
                line_photo_xy=line.tolist(), actor=actor.strip(), confirmed_utc=confirmed_utc)


def _action(value, basis):
    if (not isinstance(value, dict) or set(value) != {
            "format", "basis_sha256", "run_id", "line_photo_xy", "actor", "confirmed_utc"}):
        raise ValueError("An explicit human water-line confirmation is required")
    expected = make_water_surface_action(basis, value["line_photo_xy"],
        actor=value["actor"], confirmed_utc=value["confirmed_utc"])
    if _json(value) != _json(expected):
        raise ValueError("Water-line confirmation no longer matches its exact photo/geometry basis")
    return expected


def water_surface_measurements(action, *, expected_basis):
    binding = _basis(expected_basis)
    confirmed = _action(action, binding)
    result = dict(run_id=binding["run_id"], basis_sha256=_digest(binding),
                  display_only=True, production_geometry_changed=False,
                  height_mm=None, own_geometry_height_mm=None,
                  height_basis="existing-own-geometry", height_location=None, angle_degrees=None,
                  bolt_height_mm=None, bolt_height_location=None,
                  own_top_minus_water_mm=None, transported_top_minus_water_mm=None,
                  ruler_lateral_residual=None, ruler_extrapolated=None, unavailable=[])
    h = binding["photo_to_metric_h"]
    line = np.asarray(confirmed["line_photo_xy"], dtype=float)
    if h is None:
        result["unavailable"].append("existing-metric-plane-binding")
        return result
    line = _transform(line, h)
    bottom = binding["bottom_edge_photo_xy"]
    if bottom is None:
        result["unavailable"].append("observed-bottom-edge")
    else:
        edge = _transform(np.asarray(bottom), h)
        _line(edge)
        a, b = line[1] - line[0], edge[1] - edge[0]
        angle = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b)))
        # A line is unoriented; endpoint reversal must not report a 180-degree tilt.
        result["angle_degrees"] = float((angle + 90) % 180 - 90)
    # Primary cross-run check: the existing model's explicitly declared origin,
    # up direction and millimetre transform. This is computed, not a bolt measure.
    origin_value = binding["own_geometry_origin_metric_mm"]
    up_value = binding["own_geometry_up_metric_xy"]
    port = (None if binding["port_photo_xy"] is None else
            _transform(np.asarray([binding["port_photo_xy"]]), h)[0])
    if origin_value is None or up_value is None:
        result["unavailable"].append("existing-own-geometry-origin-or-up-convention")
    else:
        origin = np.asarray(origin_value, dtype=float)
        up = np.asarray(up_value, dtype=float)
        up /= np.linalg.norm(up)
        column_origin = origin if port is None else port
        column = np.asarray([column_origin, column_origin + up])
        water = _intersection(line, column)
        if water is None:
            result["unavailable"].append("water-line-parallel-to-own-height-column")
        else:
            result["height_mm"] = result["own_geometry_height_mm"] = float(np.dot(water - origin, up))
            result["height_location"] = "own-geometry-left-origin-column" if port is None else "bound-port-column"
            for name, output in (("own_top_photo_xy", "own_top_minus_water_mm"),
                                 ("transported_top_photo_xy", "transported_top_minus_water_mm")):
                if binding[name] is None:
                    result["unavailable"].append(name)
                    continue
                top = _intersection(_transform(np.asarray(binding[name]), h), column)
                if top is None:
                    result["unavailable"].append(name + ":parallel")
                else:
                    result[output] = float(np.dot(top - water, up))
    # An independent, optional one-dimensional ruler from actual observed bolts.
    if binding["left_bolts_photo_xy"] is None:
        result["unavailable"].append("observed-0-195-390-mm-left-bolts")
        return json.loads(_json(result))
    bolts = _transform(np.asarray(binding["left_bolts_photo_xy"]), h)
    axis = bolts[2] - bolts[0]
    denominator = float(np.dot(axis, axis))
    if denominator == 0:
        raise ValueError("Observed bolt ruler endpoints coincide")
    middle = float(np.dot(bolts[1] - bolts[0], axis) / denominator)
    if not 0 < middle < 1:
        raise ValueError("Observed bolt ruler order must be 0, 195, 390 mm")
    c = (1 - 2 * middle) / middle
    a = 390 * (c + 1)
    result["ruler_lateral_residual"] = float(np.linalg.norm(
        bolts[1] - bolts[0] - middle * axis))
    def height_at(point):
        t = float(np.dot(point - bolts[0], axis) / denominator)
        if c * t + 1 <= 0:
            raise ValueError("Requested display height crosses the observed ruler horizon")
        height = a * t / (c * t + 1)
        if not np.isfinite(height):
            raise ValueError("Non-finite observed-ruler display height")
        return float(height), not 0 <= t <= 1
    ruler_origin = bolts[0]
    location = "left-bolt-column"
    if port is not None:
        ruler_origin = port
        location = "bound-port-column"
    column = np.asarray([ruler_origin, ruler_origin + axis])
    water = _intersection(line, column)
    if water is None:
        result["unavailable"].append("water-line-parallel-to-height-column")
        return result
    try:
        result["bolt_height_mm"], result["ruler_extrapolated"] = height_at(water)
        result["bolt_height_location"] = location
    except ValueError:
        result["unavailable"].append("observed-bolt-ruler-projective-horizon")
    return json.loads(_json(result))


def water_surface_findings(actions, bases, *, height_tolerance_mm=3.0, angle_tolerance_degrees=.5):
    """Compare confirmed run heights to their series median, with editable advisories."""
    for value in (height_tolerance_mm, angle_tolerance_degrees):
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("Finite nonnegative display-warning tolerances are required")
    if not isinstance(actions, dict) or not isinstance(bases, dict) or set(actions) - set(bases):
        raise ValueError("Each confirmed action needs its exact run basis")
    measured = {run: water_surface_measurements(action, expected_basis=bases[run])
                for run, action in actions.items()}
    if any(run != row["run_id"] for run, row in measured.items()):
        raise ValueError("Run key and bound water-line action disagree")
    heights = [row["height_mm"] for row in measured.values() if row["height_mm"] is not None]
    median = float(np.median(heights)) if len(heights) >= 2 else None
    rows = []
    for run, row in measured.items():
        delta = None if median is None or row["height_mm"] is None else row["height_mm"] - median
        angle = row["angle_degrees"]
        rows.append(dict(run_id=run, severity="warning" if (
            (delta is not None and abs(delta) > height_tolerance_mm)
            or (angle is not None and abs(angle) > angle_tolerance_degrees)) else "information",
            height_delta_from_confirmed_median_mm=delta, confirmed_series_median_mm=median,
            confirmed_height_count=len(heights), total_basis_count=len(bases),
            height_tolerance_mm=height_tolerance_mm, angle_tolerance_degrees=angle_tolerance_degrees,
            measurements=deepcopy(row), display_only=True, physical_verdict=False))
    return rows
