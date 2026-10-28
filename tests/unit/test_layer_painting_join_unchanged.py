"""The default join is exactly the join before stretches could be left out.

``_base_join_to_neighbour`` is the earlier ``join_to_neighbour``, kept here
verbatim as the oracle. Both run on the same inputs in the same process, so
the comparison is exact: every float of both polygons and of every stretch,
and every refusal's message and location.
"""

import math

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from darsia.fluidflower import layer_painting_join as join
from darsia.fluidflower.layer_painting_join import (
    JoinEdgesError, JoinedStretch, JoinPreview, join_stretches, join_to_neighbour,
)

_EPS = join._EPS


def _base_join_to_neighbour(target_points, neighbour_points, distance_px):
    """The earlier join_to_neighbour, unchanged, with the module's helpers."""
    if not math.isfinite(distance_px) or distance_px < 0:
        raise JoinEdgesError("Distance must be a finite, non-negative number.")
    join._valid(target_points, "Selected polygon")
    join._valid(neighbour_points, "Neighbour polygon")
    target, neighbour = join._Outline(target_points), join._Outline(neighbour_points)
    unchanged = JoinPreview(target.points, neighbour.points, ())
    if distance_px == 0 or target.line.distance(neighbour.line) > distance_px:
        return unchanged
    intervals = join._near_intervals(target, neighbour, distance_px)
    if any(b - a >= target.length - _EPS for a, b in intervals):
        raise JoinEdgesError("The whole outline is within the distance. Choose a smaller distance.")
    accepted = []
    insertions = []
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
        if LineString(new).length < join._MIN_SHARED_ARC_PX:
            continue
        if old_line.equals(LineString(new)) and all(point in target.points for point in new):
            continue
        accepted.append((a, b, new, JoinedStretch(tuple(old), tuple(new), b - a)))
        insertions.extend((start, end))
    if not accepted:
        return unchanged
    first_end = accepted[0][1] % target.length
    rebased = sorted(((a - first_end) % target.length,
                      (a - first_end) % target.length + b - a, new, stretch)
                     for a, b, new, stretch in accepted)
    gap = rebased[0][0]
    if gap <= _EPS:
        raise JoinEdgesError("The joined stretches leave no unchanged outline. Choose a smaller distance.")
    cut = first_end + gap / 2.0
    for vertex_distance in target.cumulative[:-1]:
        if not any((vertex_distance - a) % target.length <= b - a + _EPS
                   for a, b, _new, _stretch in accepted):
            cut = float(vertex_distance)
            break
    ordered = sorted(((a - cut) % target.length + cut,
                      (a - cut) % target.length + cut + b - a, new, stretch)
                     for a, b, new, stretch in accepted)
    result = [target.at(cut)]
    cursor = cut
    for a, b, new, stretch in ordered:
        result.extend(target.vertices_between(cursor, a))
        result.extend(new)
        cursor = b
    result.extend(target.vertices_between(cursor, cut + target.length))
    result = join._deduplicate(result)
    join._valid(result, "Joined polygon")
    master = []
    for index, point in enumerate(neighbour.points):
        master.append(point)
        for value in sorted(set(insertions)):
            if neighbour.cumulative[index] + _EPS < value < neighbour.cumulative[index + 1] - _EPS:
                master.append(neighbour.at(value))
    join._valid(master, "Neighbour polygon after endpoint insertion")
    if (Polygon(result).equals(Polygon(target.points))
            and tuple(master) == neighbour.points
            and all(point in target.points for row in accepted for point in row[2])):
        return unchanged
    return JoinPreview(tuple(result), tuple(master), tuple(row[3] for row in ordered))


def _outcome(function, *args, **kwargs):
    """The whole preview, or the refusal's message and location."""
    try:
        return function(*args, **kwargs)
    except JoinEdgesError as error:
        return ("refused", str(error), error.location)


def _rectangle(top, left, bottom, right):
    """The four corners of an axis-aligned rectangle, clockwise from top left."""
    return [(top, left), (top, right), (bottom, right), (bottom, left)]


def _star(rng, center, count):
    """A valid star-shaped polygon with irregular radii around ``center``."""
    angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, count))
    radii = 40.0 * (1.0 + 0.25 * rng.uniform(-1.0, 1.0, count))
    return [(float(center[0] + r * math.sin(a)), float(center[1] + r * math.cos(a)))
            for a, r in zip(angles, radii)]


def _band(rng, count):
    """A wiggly layer and the polygon below it, painted with their own points."""
    cols = np.linspace(0.0, 600.0, count)
    rows = 50.0 + 6.0 * np.sin(cols / 50.0) + rng.normal(0.0, 0.4, cols.size)
    thickness = float(rng.uniform(6.0, 14.0))
    layer = [(float(r), float(c)) for r, c in zip(rows, cols)]
    layer += [(float(r + thickness + rng.normal(0.0, 0.5)), float(c))
              for r, c in zip(rows[::-1], cols[::-1])]
    below_cols = np.linspace(-5.0, 605.0, 90)
    below = [(float(r + thickness + 1.0 + rng.normal(0.0, 1.0)), float(c))
             for r, c in zip(np.interp(below_cols, cols, rows), below_cols)]
    below += [(140.0, 605.0), (140.0, -5.0)]
    return layer, below


def _pairs():
    """Hand-made cases (gaps, overlaps, a thin layer, refusals) and generated ones."""
    pairs = [
        (_rectangle(0, 0, 8, 20), [(10, 0), (10, 8), (10, 15), (10, 20), (30, 20), (30, 0)]),
        (_rectangle(0, 0, 12, 20), [(10, 0), (10, 8), (10, 15), (10, 20), (30, 20), (30, 0)]),
        (_rectangle(0, 6, 8, 14), _rectangle(10, 0, 30, 20)),
        ([(8, 0), (0, 0), (0, 20), (8, 20)], _rectangle(10, 0, 30, 20)),
        (_rectangle(0, 6, 8, 14), [(10, 0), (10, 0), (10, 20), (30, 20), (30, 0)]),
        ([(0, 0), (0, 100), (10, 100), (10, 0)],
         [(-10, 40), (30, 40.2), (30, 60.2), (-10, 60)]),
        ([(10, 5), (10, 28), (22, 40), (10, 52), (10, 75), (25, 75), (25, 5)],
         [(27, 2), (27, 78), (55, 78), (55, 2)]),
        ([(5, 15), (5, 50), (28, 50), (28, 38), (20, 38), (20, 27), (28, 27), (28, 15)],
         _rectangle(30, 5, 55, 70)),
        ([(10, 10), (10, 10.6), (10.5, 10.3)], _rectangle(12, 0, 30, 30)),
        ([(0, 0), (10, 10), (0, 10), (10, 0)], _rectangle(10, 0, 30, 20)),
        (_rectangle(0, 0, 8, 20), [(0, 0), (10, 10), (0, 10), (10, 0)]),
    ]
    rng = np.random.default_rng(1234)
    while len(pairs) < 31:
        a = _star(rng, (0.0, 0.0), int(rng.integers(8, 60)))
        b = _star(rng, (0.0, float(rng.uniform(70.0, 95.0))), int(rng.integers(8, 60)))
        if Polygon(a).is_valid and Polygon(b).is_valid:
            pairs.append((a, b))
    for count in (40, 77, 120):
        layer, below = _band(rng, count)
        if Polygon(layer).is_valid and Polygon(below).is_valid:
            pairs.append((layer, below))
    return pairs


PAIRS = _pairs()
DISTANCES = (0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0, 14.0, 20.0)


@pytest.mark.parametrize("number", range(len(PAIRS)))
def test_default_join_is_exactly_the_earlier_join(number):
    target, master = PAIRS[number]
    for distance in DISTANCES:
        expected = _outcome(_base_join_to_neighbour, target, master, distance)
        assert _outcome(join_to_neighbour, target, master, distance) == expected
        assert _outcome(join_to_neighbour, target, master, distance, include=None) == expected


@pytest.mark.parametrize("number", range(len(PAIRS)))
def test_every_stretch_included_is_exactly_the_default(number):
    target, master = PAIRS[number]
    for distance in DISTANCES:
        expected = _outcome(_base_join_to_neighbour, target, master, distance)
        try:
            listed = join_stretches(target, master, distance)
        except JoinEdgesError as error:
            assert expected == ("refused", str(error), error.location)
            continue
        if isinstance(expected, JoinPreview) and expected.stretches:
            assert listed == expected.stretches
        every = range(len(listed))
        for include in (set(every), list(every), tuple(every), every):
            outcome = _outcome(join_to_neighbour, target, master, distance, include=include)
            assert outcome == expected


def test_the_cases_cover_joins_refusals_and_untouched_outlines():
    joined = refused = untouched = 0
    for target, master in PAIRS:
        for distance in DISTANCES:
            result = _outcome(_base_join_to_neighbour, target, master, distance)
            if not isinstance(result, JoinPreview):
                refused += 1
            elif result.stretches:
                joined += 1
            else:
                untouched += 1
    assert joined >= 100 and refused >= 20 and untouched >= 20
