"""Descriptive preview gaps inside the placed *drawing* frame only.

This is a GUI consumer of already checked placement metadata. It grants no
adoption and does not define an independently measured top or rig boundary.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np

from . import layer_painting as lp
from . import map_placement as placement

KEY = "drawing_frame_display"
EDGE_STRIP_MM = 5.0
_SIDES = ("top", "right", "bottom", "left")


def _matrix(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("The preview has no finite measured coordinate transform.")
    np.linalg.inv(matrix)
    return matrix


def _project(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    denominator = homogeneous[:, 2]
    if (not np.isfinite(homogeneous).all() or np.any(np.abs(denominator) < 1.e-12)
            or (np.any(denominator < 0.) and np.any(denominator > 0.))):
        raise ValueError("The placed drawing frame crosses an undefined coordinate transform.")
    return homogeneous[:, :2] / denominator[:, None]


def _check_frame_polygon(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("The reference rig frame needs four finite corners.")
    if np.max(np.abs(points)) > (np.iinfo(np.int32).max // 16) - 1:
        raise ValueError("The placed rig frame exceeds the existing raster coordinate range.")
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if np.any(np.sum(edges * edges, axis=1) <= 0.) or not (np.all(crosses > 0.) or np.all(crosses < 0.)):
        raise ValueError("The reference rig frame must be a noncollapsed convex quadrilateral.")


def _frame_at(frame, corners_xy):
    placed = deepcopy(frame)
    placed.corners = {
        name: (float(point[1]), float(point[0]))
        for name, point in zip(("top_left", "top_right", "bottom_right", "bottom_left"), corners_xy)
    }
    return placed


def _geometry(run, row):
    shape = row.get("shape_hw")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError("The measured preview model has no exact native photograph grid.")
    return placement.RunGeometry(
        run, _matrix(row["metric_to_grid"]), deepcopy(row["reference_image"]),
        tuple(shape), str(row["model"]["path"]), entry=deepcopy(row["entry"]))


def _extend_bbox(bounds, values):
    lower, upper = np.min(values, axis=0), np.max(values, axis=0)
    if bounds is None:
        return [lower.tolist(), upper.tolist()]
    return [np.minimum(bounds[0], lower).tolist(), np.maximum(bounds[1], upper).tolist()]


def _statistics(gaps, target_to_mm, frame_metric):
    count, area_mm2, beyond = int(np.count_nonzero(gaps)), 0., 0
    regions = {name: dict(pixels=0, bbox_native_rc=None, bbox_metric_mm=None) for name in _SIDES}
    for start in range(0, gaps.shape[0], 128):
        rows, cols = np.nonzero(gaps[start:start + 128])
        rows += start
        for offset in range(0, len(rows), 4096):
            rc = np.column_stack((rows[offset:offset + 4096], cols[offset:offset + 4096]))
            xy = rc[:, ::-1].astype(np.float64)
            metric = _project(xy, target_to_mm)
            corners = xy[:, None, :] + np.asarray([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
            cell = _project(corners.reshape(-1, 2), target_to_mm).reshape(-1, 4, 2)
            cross = cell[:, :, 0] * np.roll(cell[:, :, 1], -1, axis=1) - cell[:, :, 1] * np.roll(cell[:, :, 0], -1, axis=1)
            area_mm2 += float(np.sum(np.abs(np.sum(cross, axis=1)) * .5))
            distances = []
            for first, second in zip(frame_metric, np.roll(frame_metric, -1, axis=0)):
                delta = second - first
                length = float(np.dot(delta, delta))
                if length <= 0:
                    raise ValueError("The reference drawing frame has a collapsed edge.")
                along = np.clip(((metric - first) @ delta) / length, 0., 1.)
                distances.append(np.linalg.norm(metric - first - along[:, None] * delta, axis=1))
            distances = np.column_stack(distances)
            nearest = np.argmin(distances, axis=1)
            far = np.min(distances, axis=1) > EDGE_STRIP_MM
            beyond += int(np.count_nonzero(far))
            for index, name in enumerate(_SIDES):
                selected = far & (nearest == index)
                if np.any(selected):
                    region = regions[name]
                    region["pixels"] += int(np.count_nonzero(selected))
                    region["bbox_native_rc"] = _extend_bbox(region["bbox_native_rc"], rc[selected])
                    region["bbox_metric_mm"] = _extend_bbox(region["bbox_metric_mm"], metric[selected])
    if not np.isfinite(area_mm2):
        raise ValueError("The uncovered pixel-footprint area is not finite.")
    return dict(inside_uncovered_pixels=count, inside_uncovered_cm2=area_mm2 / 100.,
        beyond_edge_strip_pixels=beyond, edge_strip_mm=EDGE_STRIP_MM,
        beyond_edge_strip_regions={name: row for name, row in regions.items() if row["pixels"]})


def frame_domain(result, session, frame, native_layers):
    """Compute a closed in-memory display domain; no photos or files are read.

    Corners are transformed before rasterisation. Warping a source-clipped
    frame mask would incorrectly hide real in-frame gaps beyond that photo.
    Counts use native pixels; area sums their projective metric footprints,
    without an optical/volume factor. Edge-region boxes are aggregate bounds.
    """
    if frame is None or not frame.complete():
        raise ValueError("Choose a complete reference rig frame to show inside-frame gaps.")
    binding = result.geometry_binding
    if not isinstance(binding, dict) or not isinstance(binding.get("models"), dict):
        raise ValueError("Refresh the preview to load bound measured frame metadata.")
    models = binding["models"]
    source_run, target_run = session.run_id.casefold(), result.run_id.casefold()
    if source_run not in models or target_run not in models:
        raise ValueError("The preview needs bound models for both the reference and target run.")
    if result.source_run != source_run:
        raise ValueError("The preview drawing frame belongs to another reference run.")
    source, target = _geometry(source_run, models[source_run]), _geometry(target_run, models[target_run])
    native_layers = np.asarray(native_layers)
    if (native_layers.dtype != np.uint8 or native_layers.shape != source.shape_hw
            or tuple(session.image_size or ())[::-1] != source.shape_hw
            or tuple(result.native_shape_hw or ()) != target.shape_hw):
        raise ValueError("The preview drawing, measured models and native photograph grids differ.")
    from .painting_run_preview import _same_path
    if str(frame.run_id).casefold() != source_run:
        raise ValueError("The reference rig frame belongs to another run.")
    if not frame.image or not _same_path(frame.image, session.image_path):
        raise ValueError("The reference rig frame is not bound to the drawing photograph.")
    if frame.image_size is not None and tuple(frame.image_size)[::-1] != source.shape_hw:
        raise ValueError("The reference rig frame belongs to another native photograph grid.")
    if (not _same_path(session.image_path, source.reference_image["path"])
            or not _same_path(result.photo_path, target.reference_image["path"])):
        raise ValueError("The preview drawing frame is not bound to these photographs.")
    display_shape = tuple(result.display_shape_hw or ())
    if (display_shape != tuple(result.layers.shape) or display_shape != tuple(result.photo.shape[:2])
            or result.uncovered.shape != result.layers.shape):
        raise ValueError("The preview display grids differ.")
    display = _matrix(result.H_native_to_display)
    sy, sx = display_shape[0] / target.shape_hw[0], display_shape[1] / target.shape_hw[1]
    expected = np.asarray(((sx, 0., (sx - 1.) / 2.), (0., sy, (sy - 1.) / 2.), (0., 0., 1.)))
    if not np.array_equal(display, expected):
        raise ValueError("The preview has no exact producer resize mapping.")
    original = lp.frame_polygon(frame, (1., 1.))
    if original is None or not np.isfinite(original).all():
        raise ValueError("The reference rig frame is incomplete or invalid.")
    _check_frame_polygon(original)
    placed_corners = _project(original, target.metric_to_grid @ np.linalg.inv(source.metric_to_grid))
    _check_frame_polygon(placed_corners)
    inside = lp.frame_mask(_frame_at(frame, placed_corners), target.shape_hw, (1., 1.))
    placed, filled = placement.place_layer_map(native_layers, source, target, target.shape_hw)
    gaps = inside & (filled | (placed == lp.OUTSIDE))
    display_corners = _project(placed_corners, display)
    _check_frame_polygon(display_corners)
    display_inside = lp.frame_mask(_frame_at(frame, display_corners), display_shape, (1., 1.))
    display_gaps = display_inside & (result.uncovered | (result.layers == lp.OUTSIDE))
    display_inside.setflags(write=False)
    display_gaps.setflags(write=False)
    target_to_mm = np.linalg.inv(target.metric_to_grid)
    statistics = _statistics(gaps, target_to_mm, _project(placed_corners, target_to_mm))
    frame_payload = deepcopy(frame.to_dict())
    frame_sha = hashlib.sha256(json.dumps(frame_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return dict(status="available", adopted=False, authority="Placed current reference drawing frame; not an independently measured rig top",
        frame=frame_payload, frame_sha256=frame_sha, projected_frame_native_xy=placed_corners.tolist(),
        outside_pixels=int(inside.size - np.count_nonzero(inside)),
        display_inside=display_inside, display_gaps=display_gaps, **statistics)


def attach_frame_domain(result, session, frame, native_layers):
    """Attach display-only metadata without changing the placement packet arrays."""
    try:
        domain = frame_domain(result, session, frame, native_layers)
    except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as error:
        domain = dict(status="unavailable", adopted=False, reason=str(error))
    measurements = dict(result.measurements)
    measurements[KEY] = domain
    return replace(result, measurements=measurements)
