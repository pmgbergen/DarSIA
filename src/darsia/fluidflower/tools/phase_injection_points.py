"""Project protocol wells into review canvases without changing native evidence."""
from __future__ import annotations
from darsia.fluidflower.review_storage import resolve_review_path

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _pixels(values: Sequence, size: int, name: str) -> tuple[int, ...]:
    array = np.asarray(values, dtype=float)
    if (array.shape != (size,) or not np.isfinite(array).all()
            or not np.equal(array, np.floor(array)).all()):
        raise ValueError(f"Invalid {name} geometry")
    return tuple(map(int, array))


def project_injection_points(
    stages: Sequence[Mapping[str, Any]], *, x_m: Sequence, y_m: Sequence,
    canvas_wh: Sequence, content_box: Sequence,
) -> list[dict[str, Any]]:
    """Use the nearest registered pixel centre, including the canvas margins."""
    width, height = _pixels(canvas_wh, 2, "canvas")
    left, top, image_width, image_height = _pixels(content_box, 4, "content box")
    if (min(width, height, image_width, image_height) <= 0 or min(left, top) < 0
            or left + image_width > width or top + image_height > height):
        raise ValueError("Content box does not fit the native canvas")
    x_axis, y_axis = np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float)
    for axis, length in ((x_axis, image_width), (y_axis, image_height)):
        if axis.shape != (length,) or not np.isfinite(axis).all():
            raise ValueError("Coordinate axes do not match the native analysis image")
        differences = np.diff(axis)
        if differences.size and not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError("Registered coordinate axes must be monotonic")

    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for stage in stages:
        physical_x, physical_y = float(stage["x_m"]), float(stage["y_m"])
        start, end = float(stage["start_h"]), float(stage["end_h"])
        if not np.isfinite([physical_x, physical_y, start, end]).all() or end < start:
            raise ValueError("Invalid injection coordinates or active window")
        column = int(np.argmin(np.abs(x_axis - physical_x)))
        row = int(np.argmin(np.abs(y_axis - physical_y)))
        point = grouped.setdefault((column, row), {
            "x": (left + column + .5) / width,
            "y": (top + row + .5) / height,
            "analysis_xy_px": [column, row],
            "x_m": physical_x, "y_m": physical_y, "active_windows_h": [],
            "coordinate_source": str(stage.get("coordinate_source", "protocol_and_rig_coordinate_system")),
        })
        if "protocol_x_m" in stage and "protocol_y_m" in stage:
            point["protocol_x_m"] = float(stage["protocol_x_m"])
            point["protocol_y_m"] = float(stage["protocol_y_m"])
        if "uncertainty_m" in stage:
            point["uncertainty_m"] = float(stage["uncertainty_m"])
        point["active_windows_h"].append([start, end])
    points = sorted(grouped.values(), key=lambda point: min(window[0] for window in point["active_windows_h"]))
    for index, point in enumerate(points, 1):
        point["label"] = f"Injector {index}"
    return points


def review_injection_points(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recover omitted markers only from the native package's bound physical grid.

    Legacy packages without enough metadata remain unavailable. Contradictory
    hashes or geometry are errors; no assumed tank size or coordinate flip is used.
    Explicit marker positions, including an explicitly empty list, take precedence.
    """
    if manifest.get("injection_points") is not None:
        return [dict(row) for row in manifest["injection_points"] if isinstance(row, Mapping)]
    stages = (manifest.get("phase_topology") or {}).get("injection_stages")
    domain = manifest.get("domain") or {}
    if not stages or not domain.get("path") or not domain.get("sha256"):
        return []
    if any(manifest.get(key) is None for key in ("canvas_wh", "content_box_xywh", "analysis_image_shape_hw")):
        return []
    path = resolve_review_path(domain["path"])
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != str(domain["sha256"]).lower():
            raise ValueError("Injection coordinate context SHA256 does not match the native manifest")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            if "x_m" not in archive or "y_m" not in archive:
                return []
            x_axis, y_axis = archive["x_m"], archive["y_m"]
    shape = _pixels(manifest["analysis_image_shape_hw"], 2, "analysis image")
    if shape != (y_axis.size, x_axis.size):
        raise ValueError("Coordinate axes do not match the native analysis image")
    return project_injection_points(stages, x_m=x_axis, y_m=y_axis,
        canvas_wh=manifest["canvas_wh"], content_box=manifest["content_box_xywh"])
