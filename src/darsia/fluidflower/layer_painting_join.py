"""Preview a polygon following a neighbouring outline, without changing it.

Coordinates and distances are raw photograph pixels. A neighbouring polygon
is the master: its vertices stay exactly as supplied; only the two end points
of each joined stretch may be inserted on its edges. No session setting,
classification or image is changed here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.validation import explain_validity

PointRC = tuple[float, float]
_EPS = 1.0e-7
# Two endpoints inside the existing half-pixel shared-point tolerance are
# one point, not a shared stretch. In particular, a transverse crossing
# must not collapse a long target edge onto a subpixel piece of the master.
_MIN_SHARED_ARC_PX = 1.0


class JoinEdgesError(ValueError):
    """A refused join, with its raw (row, column) when available.

    When a polygon check refused it, ``outline`` names the refused outline
    (for example "Joined polygon") and ``reason`` is the check's own
    explanation (for example "Self-intersection[12.5 40.0]"); both are None
    for other refusals, so that a dialog can say in plain words what failed.
    """

    def __init__(
        self,
        message: str,
        location: PointRC | None = None,
        *,
        outline: str | None = None,
        reason: str | None = None,
    ):
        super().__init__(message)
        self.location = location
        self.outline = outline
        self.reason = reason


@dataclass(frozen=True)
class JoinedStretch:
    before: tuple[PointRC, ...]
    after: tuple[PointRC, ...]
    length_px: float


@dataclass(frozen=True)
class JoinPreview:
    target: tuple[PointRC, ...]
    neighbour: tuple[PointRC, ...]
    stretches: tuple[JoinedStretch, ...]

    @property
    def length_px(self) -> float:
        return sum(row.length_px for row in self.stretches)


def _valid(points: Sequence[PointRC], description: str) -> Polygon:
    if len(points) < 3 or not np.isfinite(points).all():
        raise JoinEdgesError(
            f"{description} needs three finite points.",
            outline=description,
            reason="Too few finite points",
        )
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= _EPS:
        reason = explain_validity(polygon)
        match = re.search(r"\[([^\]]+)\]", reason)
        location = None
        if match:
            try:
                row, col = map(float, match.group(1).split())
                location = (row, col)
            except ValueError:
                pass
        where = "" if location is None else f" at raw row {location[0]:.2f}, column {location[1]:.2f}"
        raise JoinEdgesError(
            f"{description}: {reason}{where}. Nothing was changed.",
            location,
            outline=description,
            reason=reason,
        )
    return polygon


class _Outline:
    def __init__(self, points: Sequence[PointRC]):
        self.points = tuple((float(r), float(c)) for r, c in points)
        closed = np.asarray((*self.points, self.points[0]), dtype=float)
        self.lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        self.cumulative = np.r_[0.0, np.cumsum(self.lengths)]
        self.length = float(self.cumulative[-1])
        self.line = LineString(closed)

    def at(self, distance: float) -> PointRC:
        distance %= self.length
        index = min(int(np.searchsorted(self.cumulative, distance, side="right")) - 1,
                    len(self.points) - 1)
        start, end = self.points[index], self.points[(index + 1) % len(self.points)]
        fraction = (distance - self.cumulative[index]) / max(self.lengths[index], _EPS)
        if fraction <= _EPS:
            return start
        if fraction >= 1.0 - _EPS:
            return end
        return (start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]))

    def vertices_between(self, start: float, end: float) -> list[PointRC]:
        result = []
        for turn in range(math.floor(start / self.length), math.floor(end / self.length) + 1):
            for distance, point in zip(self.cumulative[:-1], self.points):
                if start + _EPS < distance + turn * self.length < end - _EPS:
                    result.append(point)
        return result

    def path(self, start: float, end: float) -> list[PointRC]:
        return [self.at(start), *self.vertices_between(start, end), self.at(end)]


def _lines(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _lines(part)


def _near_intervals(target: _Outline, neighbour: _Outline, distance: float):
    band = neighbour.line.buffer(distance, quad_segs=16)
    intervals = []
    for index, point in enumerate(target.points):
        edge = LineString((point, target.points[(index + 1) % len(target.points)]))
        if edge.length <= _EPS:
            continue
        for part in _lines(edge.intersection(band)):
            a = float(target.cumulative[index]) + edge.project(Point(part.coords[0]))
            b = float(target.cumulative[index]) + edge.project(Point(part.coords[-1]))
            a, b = sorted((a, b))
            if b - a > _EPS:
                intervals.append((a, b))
    merged: list[tuple[float, float]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1] + _EPS:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    if len(merged) > 1 and merged[0][0] <= _EPS and merged[-1][1] >= target.length - _EPS:
        merged = [(merged[-1][0], merged[0][1] + target.length), *merged[1:-1]]
    return merged


def _plan(
    target_points: Sequence[PointRC],
    neighbour_points: Sequence[PointRC],
    distance_px: float,
):
    """The validated outlines, the cut and the joinable stretches in preview order.

    Returns ``(target, neighbour, cut, rows)``. Each row is ``(a, b, new,
    stretch, ends)``: the stretch's span along the target's outline from the
    cut onwards, the master's path that replaces it, its JoinedStretch, and
    the master positions of its two end points. No rows: nothing to join.
    The joined polygon itself is neither built nor checked here.
    """
    if not math.isfinite(distance_px) or distance_px < 0:
        raise JoinEdgesError("Distance must be a finite, non-negative number.")
    _valid(target_points, "Selected polygon")
    _valid(neighbour_points, "Neighbour polygon")
    target, neighbour = _Outline(target_points), _Outline(neighbour_points)
    if distance_px == 0 or target.line.distance(neighbour.line) > distance_px:
        return target, neighbour, 0.0, []
    intervals = _near_intervals(target, neighbour, distance_px)
    if any(b - a >= target.length - _EPS for a, b in intervals):
        raise JoinEdgesError("The whole outline is within the distance. Choose a smaller distance.")
    accepted = []
    for a, b in intervals:
        old = target.path(a, b)
        start = float(neighbour.line.project(Point(old[0])))
        end = float(neighbour.line.project(Point(old[-1])))
        forward = (end - start) % neighbour.length
        if min(forward, neighbour.length - forward) <= _EPS:
            continue
        ways = [neighbour.path(start, start + forward),
                list(reversed(neighbour.path(end, end + neighbour.length - forward)))]
        old_line = LineString(old)
        ways = [way for way in ways
                if LineString(way).hausdorff_distance(old_line) <= distance_px + _EPS]
        if not ways:
            continue
        new = min(ways, key=lambda way: LineString(way).length)
        if LineString(new).length < _MIN_SHARED_ARC_PX:
            continue
        # Already shared, including all master's vertices: leave the session alone.
        if old_line.equals(LineString(new)) and all(point in target.points for point in new):
            continue
        stretch = JoinedStretch(tuple(old), tuple(new), b - a)
        accepted.append((a, b, new, stretch, (start, end)))
    if not accepted:
        return target, neighbour, 0.0, []
    # Start just beyond a joined interval, in a gap, so no stretch is split
    # across the beginning of the new list of vertices.
    first_end = accepted[0][1] % target.length
    rebased = sorted(((a - first_end) % target.length,
                      (a - first_end) % target.length + b - a, new, stretch)
                     for a, b, new, stretch, _ends in accepted)
    gap = rebased[0][0]
    if gap <= _EPS:
        raise JoinEdgesError("The joined stretches leave no unchanged outline. Choose a smaller distance.")
    cut = first_end + gap / 2.0
    for vertex_distance in target.cumulative[:-1]:
        if not any((vertex_distance - a) % target.length <= b - a + _EPS
                   for a, b, _new, _stretch, _ends in accepted):
            cut = float(vertex_distance)
            break
    # Spans are distinct, so the order never compares the later fields.
    rows = sorted(((a - cut) % target.length + cut,
                   (a - cut) % target.length + cut + b - a, new, stretch, ends)
                  for a, b, new, stretch, ends in accepted)
    return target, neighbour, cut, rows


def _included(include: Iterable[int], count: int) -> set[int]:
    """The stretch numbers in ``include``; refused unless each is 0 to count - 1."""
    chosen = set()
    for number in include:
        if (isinstance(number, (bool, np.bool_))
                or not isinstance(number, (int, np.integer))
                or not 0 <= number < count):
            raise JoinEdgesError(
                f"Stretch {number!r} is not one of the {count} stretches at this distance."
            )
        chosen.add(int(number))
    return chosen


def join_stretches(
    target_points: Sequence[PointRC],
    neighbour_points: Sequence[PointRC],
    distance_px: float,
) -> tuple[JoinedStretch, ...]:
    """The stretches that ``join_to_neighbour`` joins, in its preview's order.

    Their positions are the stretch numbers that its ``include`` takes. Only
    the stretches are found: the joined polygon is neither built nor
    checked, so the stretches of a refused join are listed too. Invalid
    polygons or distances, and distances that would join the whole outline,
    are refused as by ``join_to_neighbour``. Neither input is mutated.
    """
    return tuple(row[3] for row in _plan(target_points, neighbour_points, distance_px)[3])


def join_to_neighbour(
    target_points: Sequence[PointRC],
    neighbour_points: Sequence[PointRC],
    distance_px: float,
    *,
    include: Iterable[int] | None = None,
) -> JoinPreview:
    """Return a complete proposed edit, or refuse a non-simple polygon.

    Each stretch of the target within ``distance_px`` is replaced by the
    master's own outline. A connecting arc must stay within that distance of
    the old stretch in both directions; a mere near corner is not an arc.
    ``include`` takes the numbers (0, 1, ...) of the stretches to join, in the
    order of ``join_stretches``; the others keep their old outline, and the
    same check refuses a polygon that would still cross itself. None (the
    default) joins every stretch. The master never moves. This function
    never mutates either input.
    """
    target, neighbour, cut, rows = _plan(target_points, neighbour_points, distance_px)
    unchanged = JoinPreview(target.points, neighbour.points, ())
    if include is not None:
        chosen = _included(include, len(rows))
        rows = [row for number, row in enumerate(rows) if number in chosen]
    if not rows:
        return unchanged
    result = [target.at(cut)]
    cursor = cut
    for a, b, new, _stretch, _ends in rows:
        result.extend(target.vertices_between(cursor, a))
        result.extend(new)
        cursor = b
    result.extend(target.vertices_between(cursor, cut + target.length))
    result = _deduplicate(result)
    _valid(result, "Joined polygon")
    # Insert endpoints in the original master order, retaining every vertex.
    insertions = [value for row in rows for value in row[4]]
    master = []
    for index, point in enumerate(neighbour.points):
        master.append(point)
        for value in sorted(set(insertions)):
            if neighbour.cumulative[index] + _EPS < value < neighbour.cumulative[index + 1] - _EPS:
                master.append(neighbour.at(value))
    _valid(master, "Neighbour polygon after endpoint insertion")
    if (Polygon(result).equals(Polygon(target.points))
            and tuple(master) == neighbour.points
            and all(point in target.points for row in rows for point in row[2])):
        return unchanged
    return JoinPreview(tuple(result), tuple(master), tuple(row[3] for row in rows))


def _outcome(
    target_points: Sequence[PointRC],
    neighbour_points: Sequence[PointRC],
    distance_px: float,
) -> str:
    """The full join's outcome there: "works", "nothing" (no stretch) or "refused"."""
    try:
        preview = join_to_neighbour(target_points, neighbour_points, distance_px)
    except JoinEdgesError:
        return "refused"
    return "works" if preview.stretches else "nothing"


def verified_working_distance(
    target_points: Sequence[PointRC],
    neighbour_points: Sequence[PointRC],
    distance_px: float,
    tolerance_px: float,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> float | None:
    """A distance at or below ``distance_px`` that is checked to give a valid join.

    A distance works when ``join_to_neighbour`` (all stretches) joins at least
    one stretch there without a refusal. The chosen distance is returned when
    it works. Otherwise whole multiples of ``tolerance_px`` below it are tried
    by bisection: halving from the largest while they are refused, then
    bisecting until a working distance lies one ``tolerance_px`` below one
    that does not work; a distance that joins nothing counts as too small.
    Whether a distance works need not be monotone in the distance, because
    the stretches and the shape of the joined outline change with it, so the
    result is a verified working distance, never claimed to be the largest
    one. Every returned distance comes from a real ``join_to_neighbour``
    call, and one found by bisection is checked once more just before it is
    returned. None when no tried distance works, or when ``cancelled`` (asked
    before each try) returns True. Invalid polygons, distances or tolerances
    are refused.
    """
    if not math.isfinite(distance_px) or distance_px < 0:
        raise JoinEdgesError("Distance must be a finite, non-negative number.")
    if not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise JoinEdgesError("Tolerance must be a finite, positive number.")
    _valid(target_points, "Selected polygon")
    _valid(neighbour_points, "Neighbour polygon")
    stop = cancelled if cancelled is not None else (lambda: False)

    def outcome(step: int) -> str:
        """The full join's outcome at ``step`` whole tolerances."""
        return _outcome(target_points, neighbour_points, step * tolerance_px)

    if stop():
        return None
    if _outcome(target_points, neighbour_points, distance_px) == "works":
        return float(distance_px)
    # Steps (multiples of the tolerance) strictly below the chosen distance.
    # ``high`` is a step that does not work (at first: the chosen distance),
    # ``low`` one that joins nothing (at first: zero).
    top = math.floor(distance_px / tolerance_px)
    if top * tolerance_px >= distance_px:
        top -= 1
    low, high, good = 0, top + 1, None
    step = top
    while step > low:
        if stop():
            return None
        found = outcome(step)
        if found == "works":
            good = step
            break
        if found == "nothing":
            low = step
            break
        high, step = step, step // 2
    while good is None and high - low > 1:
        if stop():
            return None
        middle = (low + high) // 2
        found = outcome(middle)
        if found == "works":
            good = middle
        elif found == "nothing":
            low = middle
        else:
            high = middle
    if good is None:
        return None
    while high - good > 1:
        if stop():
            return None
        middle = (good + high) // 2
        if outcome(middle) == "works":
            good = middle
        else:
            high = middle
    distance = good * tolerance_px
    if stop() or _outcome(target_points, neighbour_points, distance) != "works":
        return None
    return distance


def _deduplicate(points: Sequence[PointRC]) -> list[PointRC]:
    result = []
    for point in points:
        if not result or math.dist(point, result[-1]) > _EPS:
            result.append(point)
    if len(result) > 1 and math.dist(result[0], result[-1]) <= _EPS:
        result.pop()
    return result
