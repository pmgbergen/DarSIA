"""Measured analysis-box drafts for the existing photograph painting tools.

Only rectangular box drafts change here. Layer polygons are passive drawing
guides, and neither a geometry revision nor an image is read or written.
Shared rectangle sides move together; polygon Follow/Join tools are excluded.
The existing rectangle-only store and mass ABI stay unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from shapely.geometry import Polygon

from . import metric_analysis_domains as domains


class NonRectangularBox(ValueError):
    """An edit would leave the existing measured rectangle contract."""


_SIDES = {"x_min": (0, 0), "x_max": (0, 1), "y_min": (1, 2), "y_max": (1, 3)}


def _outline(points):
    raw = list(points)
    if (len(raw) < 3 or any(len(point) != 2 for point in raw)
            or any(isinstance(value, bool) or not isinstance(value, Real)
                   for point in raw for value in point)):
        raise ValueError("An outline needs at least three finite coordinate pairs")
    value = np.asarray(raw, np.float64)
    if not np.isfinite(value).all():
        raise ValueError("An outline needs at least three finite coordinate pairs")
    if np.array_equal(value[0], value[-1]):
        value = value[:-1]
    polygon = Polygon(value)
    if len(value) < 3 or not polygon.is_valid or not math.isfinite(polygon.area) or polygon.area <= 0:
        raise ValueError("An analysis-box outline must be a simple polygon with positive area")
    return tuple((float(x), float(y)) for x, y in value)


def _matrix(value):
    matrix = np.asarray(value, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("An exact finite invertible painting-frame homography is required")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("An exact finite invertible painting-frame homography is required") from error
    if not np.isfinite(inverse).all():
        raise ValueError("An exact finite invertible painting-frame homography is required")
    return matrix


def _project(points, matrix):
    values = np.asarray(points, np.float64)
    mapped = matrix @ np.vstack((values.T, np.ones(len(values))))
    denominator = mapped[2]
    if (not np.isfinite(mapped).all() or np.any(np.abs(denominator) < 1e-12)
            or (np.any(denominator < 0) and np.any(denominator > 0))):
        raise ValueError("The outline crosses the painting frame's projective horizon")
    mapped = (mapped[:2] / denominator).T
    if not np.isfinite(mapped).all():
        raise ValueError("The projected outline is not finite")
    return mapped


def metric_to_raw(points_mm, metric_to_grid):
    """Measured x/y millimetres to raw row/column, without display scaling."""
    points = _outline(points_mm)
    mapped = _project(points, _matrix(metric_to_grid))[:, ::-1]
    return tuple((float(row), float(col)) for row, col in mapped)


def raw_to_metric(points_rc, metric_to_grid):
    """Raw row/column to measured x/y millimetres using the same exact frame."""
    points = _outline(points_rc)
    mapped = _project(np.asarray(points)[:, ::-1], np.linalg.inv(_matrix(metric_to_grid)))
    return _outline(mapped)


def _bounds(points):
    value = np.asarray(points, np.float64)
    return [float(value[:, 0].min()), float(value[:, 0].max()),
            float(value[:, 1].min()), float(value[:, 1].max())]


def _rectangle_bounds(points):
    x0, x1, y0, y1 = _bounds(points)
    for first, second in zip(points, (*points[1:], points[0])):
        x, y = first
        if not (x in (x0, x1) or y in (y0, y1)):
            raise NonRectangularBox("Analysis boxes must stay rectangles in measured millimetres; polygon Follow/Join is unavailable")
        if first[0] != second[0] and first[1] != second[1]:
            raise NonRectangularBox("Analysis boxes must stay rectangles in measured millimetres; polygon Follow/Join is unavailable")
    return [x0, x1, y0, y1]


def _rectangle_points(bounds):
    x0, x1, y0, y1 = bounds
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def raw_point_to_metric(point_rc, metric_to_grid):
    """One passive layer-edge point, using the same measured frame as the boxes."""
    if (len(point_rc) != 2 or any(isinstance(v, bool) or not isinstance(v, Real)
                                 or not math.isfinite(v) for v in point_rc)):
        raise ValueError("A layer-edge point needs two finite raw row/column values")
    mapped = _project([(point_rc[1], point_rc[0])], np.linalg.inv(_matrix(metric_to_grid)))[0]
    return float(mapped[0]), float(mapped[1])


@dataclass(frozen=True)
class BoxOutline:
    domain_id: str
    name: str
    role: str
    enabled: bool
    points_mm: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CanvasBoxOutline:
    """A box's own index in a combined layer/box canvas; layers have no token."""

    canvas_index: int
    domain_id: str
    points_mm: tuple[tuple[float, float], ...]
    points_rc: tuple[tuple[float, float], ...]
    metric_to_grid: tuple[tuple[float, ...], ...]


class AnalysisBoxesDraft:
    """A separate measured-box draft, leaving the supplied configuration intact."""

    def __init__(self, configuration):
        expected = dict(format=domains.FORMAT, units="mm", x_axis="right", y_axis="up",
                        origin="inner left-wall and bottom intersection")
        if not isinstance(configuration, Mapping) or any(configuration.get(k) != v for k, v in expected.items()):
            raise ValueError("The existing measured-frame analysis-box configuration is required")
        self._configuration = deepcopy(dict(configuration))
        self._rows = {}
        self._points = {}
        for row in configuration["domains"]:
            canonical = domains.box(row["name"], row["bounds_mm"], role=row["role"],
                domain_id=row["domain_id"], enabled=row["enabled"], legacy=row.get("legacy_import"))
            if canonical != row:
                raise ValueError("An analysis-box row differs from the existing rectangle schema")
            self._check_identity(canonical)
            self._rows[canonical["domain_id"]] = deepcopy(canonical)
            x0, x1, y0, y1 = canonical["bounds_mm"]
            self._points[canonical["domain_id"]] = _outline(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))

    def _check_identity(self, row, *, replacing=None):
        for key, existing in self._rows.items():
            if key != replacing and (key == row["domain_id"] or existing["name"].casefold() == row["name"].casefold()):
                raise ValueError("Analysis-box identifiers and names must be unique")

    def outlines(self):
        return tuple(BoxOutline(key, row["name"], row["role"], row["enabled"], self._points[key])
                     for key, row in self._rows.items())

    def clone(self):
        """An independent undo snapshot of all rectangle sides and metadata."""
        return deepcopy(self)

    def configuration(self):
        """Existing rectangle DATA only; never replace a polygon by its bounds."""
        result = deepcopy(self._configuration)
        rows = []
        for key, row in self._rows.items():
            bounds = _rectangle_bounds(self._points[key])
            value = deepcopy(row)
            if bounds != row["bounds_mm"]:
                value["bounds_mm"] = bounds
                if value.get("legacy_import"):
                    value["legacy_import"]["unmodified"] = False
            rows.append(value)
        result["domains"] = rows
        return result

    def mass_boxes(self):
        return domains.mass_boxes(self.configuration())

    def add_outline(self, name, points_mm, *, role="", domain_id=None, enabled=True):
        points = _outline(points_mm)
        row = domains.box(name, _rectangle_bounds(points), role=role, domain_id=domain_id, enabled=enabled)
        self._check_identity(row)
        self._rows[row["domain_id"]], self._points[row["domain_id"]] = row, points
        return row["domain_id"]

    def add_rectangle(self, name, bounds_mm, **kwargs):
        value = domains.box(name, bounds_mm, **kwargs)
        x0, x1, y0, y1 = value["bounds_mm"]
        return self.add_outline(name, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
            role=value["role"], domain_id=value["domain_id"], enabled=value["enabled"])

    def update_metadata(self, domain_id, *, name=None, role=None, enabled=None):
        old = self._rows[domain_id]
        row = domains.box(old["name"] if name is None else name, old["bounds_mm"],
            domain_id=domain_id, role=old["role"] if role is None else role,
            enabled=old["enabled"] if enabled is None else enabled, legacy=old.get("legacy_import"))
        self._check_identity(row, replacing=domain_id)
        self._rows[domain_id] = row

    def remove(self, domain_id):
        """Remove a draft row only; no file or revision is deleted."""
        del self._rows[domain_id]
        del self._points[domain_id]

    def import_missing_legacy(self, values, *, source):
        """Import absent roles only, preserving original coordinates and pending review."""
        roles = {row["role"] for row in self._rows.values()}
        if {"box1", "box2"} <= roles:
            return ()
        added = []
        for value in domains.import_legacy_boxes(values, source=source):
            if value["role"] in roles:
                continue
            base, number = value["name"], 1
            names = {row["name"].casefold() for row in self._rows.values()}
            while value["name"].casefold() in names:
                number += 1
                value["name"] = f"{base} (imported {number})"
            self._check_identity(value)
            key = value["domain_id"]
            x0, x1, y0, y1 = value["bounds_mm"]
            self._rows[key] = deepcopy(value)
            self._points[key] = _outline(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
            added.append(key)
            roles.add(value["role"])
        return tuple(added)

    def shared_side_group(self, domain_id, side):
        """Exact coincident sides with positive overlap, including connected boxes.

        A point contact alone is not a shared side. No tolerance rounds or
        merges separately authored measured coordinates.
        """
        if side not in _SIDES:
            raise ValueError("Choose x_min, x_max, y_min or y_max")
        axis, position = _SIDES[side]
        bounds = {key: _rectangle_bounds(points) for key, points in self._points.items()}
        coordinate = bounds[domain_id][position]
        candidates = []
        for key, values in bounds.items():
            for name, (other_axis, other_position) in _SIDES.items():
                if axis == other_axis and values[other_position] == coordinate:
                    interval = values[2:4] if axis == 0 else values[0:2]
                    candidates.append(((key, name), interval))
        group = {(domain_id, side)}
        changed = True
        while changed:
            changed = False
            intervals = [interval for identity, interval in candidates if identity in group]
            for identity, interval in candidates:
                if identity not in group and any(max(interval[0], other[0]) < min(interval[1], other[1])
                                                 for other in intervals):
                    group.add(identity)
                    changed = True
        return tuple(identity for identity, _interval in candidates if identity in group)

    def edit_sides(self, changes, *, move_shared=True):
        """Move rectangle sides atomically, for one caller-controlled undo step."""
        if type(move_shared) is not bool:
            raise ValueError("Shared-side movement must be an explicit boolean")
        requested = {}
        for identity, value in changes.items():
            key, side = identity
            if key not in self._rows or side not in _SIDES:
                raise ValueError("The selected analysis-box side does not exist")
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("A rectangle side needs a finite measured millimetre coordinate")
            group = self.shared_side_group(key, side) if move_shared else ((key, side),)
            for target in group:
                if target in requested and requested[target] != value:
                    raise ValueError("One shared rectangle side received conflicting positions")
                requested[target] = float(value)
        bounds = {key: _rectangle_bounds(points) for key, points in self._points.items()}
        for (key, side), value in requested.items():
            bounds[key][_SIDES[side][1]] = value
        updates = {}
        for key, values in bounds.items():
            old = _rectangle_bounds(self._points[key])
            if values != old:
                row = self._rows[key]
                domains.box(row["name"], values, domain_id=key, role=row["role"], enabled=row["enabled"])
                updates[key] = _rectangle_points(values)
        self._points.update(updates)
        return tuple(updates)

    def move_side(self, domain_id, side, coordinate_mm, *, move_shared=True):
        return self.edit_sides({(domain_id, side): coordinate_mm}, move_shared=move_shared)

    def move_box(self, domain_id, delta_x_mm, delta_y_mm, *, move_shared=False):
        """Translate a whole measured rectangle; all four sides are one change."""
        for value in (delta_x_mm, delta_y_mm):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("Box movement needs finite measured millimetre offsets")
        bounds = _rectangle_bounds(self._points[domain_id])
        return self.edit_sides({(domain_id, side): bounds[position] + (delta_x_mm if axis == 0 else delta_y_mm)
            for side, (axis, position) in _SIDES.items()}, move_shared=move_shared)

    def snap_side_to_box(self, domain_id, side, neighbour_id, neighbour_side, *, move_shared=True):
        if side not in _SIDES or neighbour_side not in _SIDES or _SIDES[side][0] != _SIDES[neighbour_side][0]:
            raise ValueError("Only rectangle sides on the same x or y axis can snap together")
        value = _rectangle_bounds(self._points[neighbour_id])[_SIDES[neighbour_side][1]]
        return self.move_side(domain_id, side, value, move_shared=move_shared)

    def snap_side_to_layer_point(self, domain_id, side, point_mm, *, move_shared=True):
        """Use the x/y of an already selected nearest layer-edge point only."""
        if side not in _SIDES:
            raise ValueError("Choose x_min, x_max, y_min or y_max")
        if (len(point_mm) != 2 or any(isinstance(v, bool) or not isinstance(v, Real)
                                     or not math.isfinite(v) for v in point_mm)):
            raise ValueError("A layer-edge point needs two finite measured millimetre coordinates")
        return self.move_side(domain_id, side, point_mm[_SIDES[side][0]], move_shared=move_shared)

    def canvas_outlines(self, metric_to_grid, *, first_index=0):
        if type(first_index) is not int or first_index < 0:
            raise ValueError("The first canvas box index must be a nonnegative integer")
        matrix = _matrix(metric_to_grid)
        binding = tuple(tuple(float(v) for v in row) for row in matrix)
        return tuple(CanvasBoxOutline(first_index + number, outline.domain_id, outline.points_mm,
            metric_to_raw(outline.points_mm, matrix), binding)
            for number, outline in enumerate(self.outlines()))

    def apply_canvas_edits(self, changes, projections):
        """Apply box indices atomically; layer-only edits remain passive guides.

        Exact unchanged projected vertices bypass inverse projection so a click
        or a metadata change cannot perturb the original measured coordinates.
        Each projection token also refuses edits made against a stale box draft.
        """
        tokens = {row.canvas_index: row for row in projections}
        if len(tokens) != len(projections):
            raise ValueError("Canvas box indices must be unique")
        updates = {}
        for index, points in changes.items():
            if index not in tokens:
                continue
            token = tokens[index]
            if token.domain_id not in self._points or self._points[token.domain_id] != token.points_mm:
                raise ValueError("The canvas box draft changed; redraw before editing")
            supplied = _outline(points)
            if supplied != token.points_rc:
                updated = raw_to_metric(supplied, token.metric_to_grid)
                _rectangle_bounds(updated)
                updates[token.domain_id] = updated
        self._points.update(updates)
        return tuple(updates)
