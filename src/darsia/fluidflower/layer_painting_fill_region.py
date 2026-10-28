"""Plan a polygon in an uncovered region, without changing any marks.

All coordinates are raw photograph pixels, in (row, column) order. Polygon
interiors of every layer bound the gap. The complete frame bounds its outside.
The result contains only points and indexed edits for the caller's one undo
transaction; it does not read images, sessions, proposals or settings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral

from shapely.errors import ShapelyError
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import split, unary_union
from shapely.validation import explain_validity

PointRC = tuple[float, float]
_EPS = 1.0e-7  # Numerical checks only; never a snapping or gap-closing distance.


class FillRegionError(ValueError):
    """A refused fill. No input has been changed."""


@dataclass(frozen=True)
class PolygonUpdate:
    """Only inserted shared vertices; every old vertex retains its position."""

    index: int
    before: tuple[PointRC, ...]
    after: tuple[PointRC, ...]


@dataclass(frozen=True)
class FillRegionPlan:
    """A new mark and neighbour edits to apply together in one undo step."""

    layer_id: int
    polygon_points: tuple[PointRC, ...]
    updates: tuple[PolygonUpdate, ...]
    area_px2: float


def _points(values: Sequence[PointRC], description: str) -> tuple[PointRC, ...]:
    try:
        points = tuple((float(row), float(col)) for row, col in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise FillRegionError(f"{description} needs finite (row, column) points.") from error
    if not all(math.isfinite(value) for point in points for value in point):
        raise FillRegionError(f"{description} needs finite (row, column) points.")
    return points


def _polygon(points: tuple[PointRC, ...], description: str) -> Polygon:
    if len(points) < 3:
        raise FillRegionError(f"{description} needs at least three points.")
    try:
        polygon = Polygon(points)
    except ShapelyError as error:
        raise FillRegionError(f"{description} cannot form a polygon.") from error
    if not polygon.is_valid or not math.isfinite(polygon.area) or polygon.area <= 0:
        raise FillRegionError(f"{description}: {explain_validity(polygon)}. Nothing was changed.")
    return polygon


def _parts(geometry):
    if geometry.geom_type == "Polygon":
        if not geometry.is_empty:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _parts(part)


def _containing(geometry, click: Point) -> Polygon:
    found = [part for part in _parts(geometry) if part.contains(click)]
    if len(found) != 1:
        raise FillRegionError("Click inside one uncovered region, away from its boundary.")
    return found[0]


def _insert_vertices(
    outline: tuple[PointRC, ...], candidates: Sequence[PointRC]
) -> tuple[PointRC, ...]:
    """Keep the original ordering, inserting points on its existing edges."""
    result: list[PointRC] = []
    unique = tuple(dict.fromkeys(candidates))
    for index, start in enumerate(outline):
        result.append(start)
        end = outline[(index + 1) % len(outline)]
        dr, dc = end[0] - start[0], end[1] - start[1]
        length2 = dr * dr + dc * dc
        if length2 == 0:
            continue
        length = math.sqrt(length2)
        inside = []
        for point in unique:
            if point == start or point == end:
                continue
            fraction = ((point[0] - start[0]) * dr + (point[1] - start[1]) * dc) / length2
            if not 0 < fraction < 1:
                continue
            distance = abs((point[0] - start[0]) * dc - (point[1] - start[1]) * dr) / length
            if distance <= _EPS:
                inside.append((fraction, point))
        result.extend(point for _fraction, point in sorted(inside))
    return tuple(result)


def _same_region(first: Polygon, second: Polygon) -> bool:
    # GEOS intersections use floating point. Insertions may differ from an
    # analytic segment by its last few bits, but may not move an outline.
    allowance = _EPS * max(1.0, first.length, second.length)
    return (first.hausdorff_distance(second) <= _EPS
            and first.symmetric_difference(second).area <= allowance)


def fill_region(
    polygons: Mapping[int, Sequence[PointRC]],
    frame_points: Sequence[PointRC],
    clicked_point: PointRC,
    *,
    layer_id: int,
    cut_line: Sequence[PointRC] | None = None,
) -> FillRegionPlan:
    """Plan an exact vector fill of the clicked gap, or refuse it explicitly.

    ``polygons`` maps existing polygon indices to their complete raw outlines,
    including hidden polygons and polygons of the selected layer. No polygon
    is a donor to be moved: their union is covered space. ``frame_points`` is
    the complete frame's raw outline, not an inferred image rectangle.

    A cut must divide this gap, not just end inside it. Remaining holes are
    refused because a mark has one simple ring. No rasterisation, buffering,
    rounding, padding, repair or automatic closing of gaps is performed.
    Input points are never mutated. The caller checks that each ``before``
    still matches before applying all updates and the new mark together.
    """
    if isinstance(layer_id, bool) or not isinstance(layer_id, Integral) or layer_id <= 0:
        raise FillRegionError("Choose a layer before filling a region.")
    if not isinstance(polygons, Mapping):
        raise FillRegionError("Existing polygons must be indexed outlines.")
    rows: dict[int, tuple[PointRC, ...]] = {}
    for index, values in polygons.items():
        if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
            raise FillRegionError("Existing polygon indices must be non-negative integers.")
        rows[int(index)] = _points(values, f"Polygon {index + 1}")
    frame = _points(frame_points, "Rig frame")
    frame_geometry = _polygon(frame, "Rig frame")
    click_values = _points([clicked_point], "Fill point")
    click = Point(click_values[0])
    if not frame_geometry.contains(click):
        raise FillRegionError("Click inside the rig frame, away from its boundary.")
    geometries = [_polygon(points, f"Polygon {index + 1}")
                  for index, points in sorted(rows.items())]
    cut = None
    cut_points: tuple[PointRC, ...] = ()
    if cut_line is not None:
        cut_points = _points(cut_line, "Cut line")
        if len(cut_points) < 2:
            raise FillRegionError("A cut line needs at least two distinct points.")
        cut = LineString(cut_points)
        if not cut.is_simple or not cut.is_valid or not math.isfinite(cut.length) or cut.length <= 0:
            raise FillRegionError("The cut line must be a simple open line with distinct endpoints.")
        if cut.is_closed:
            raise FillRegionError("The cut line must be open.")
        if cut.distance(click) <= _EPS:
            raise FillRegionError("Click away from the cut line.")
    try:
        covered = unary_union(geometries)
        if covered.covers(click):
            raise FillRegionError("This point is already covered or on a polygon boundary.")
        region = _containing(frame_geometry.difference(covered), click)
        if cut is not None:
            pieces = list(_parts(split(region, cut)))
            if len(pieces) < 2:
                raise FillRegionError(
                    "The cut does not divide this gap. Extend it between boundaries."
                )
            # Do not union these again: that would remove the cut.
            region = _containing(GeometryCollection(pieces), click)
            # A line partly hanging into the chosen part is not a barrier.
            if cut.intersection(region).difference(region.boundary).length > _EPS:
                raise FillRegionError(
                    "The cut ends inside the chosen region. Extend it between boundaries."
                )
        if region.interiors:
            raise FillRegionError(
                "This region contains an enclosed polygon. Use a cut to select a part without holes."
            )
        outline = tuple((float(row), float(col)) for row, col in region.exterior.coords[:-1])
        # Keep donor/frame/cut corners even when an overlay drops a collinear
        # vertex. This is necessary for the existing shared-edge point model.
        donors = [point for points in rows.values() for point in points]
        outline = _insert_vertices(outline, [*frame, *donors, *cut_points])
        new_geometry = _polygon(outline, "Filled polygon")
        if not _same_region(region, new_geometry):
            raise FillRegionError("Shared vertices could not be retained without changing the region.")
        updates = []
        for index, before in sorted(rows.items()):
            after = _insert_vertices(before, outline)
            if after == before:
                continue
            if not _same_region(_polygon(before, f"Polygon {index + 1}"),
                                _polygon(after, f"Polygon {index + 1} after insertion")):
                raise FillRegionError("A shared point would change an existing polygon.")
            updates.append(PolygonUpdate(index, before, after))
        return FillRegionPlan(int(layer_id), outline, tuple(updates), float(region.area))
    except ShapelyError as error:
        raise FillRegionError("The boundaries could not form a simple fill. Nothing was changed.") from error
