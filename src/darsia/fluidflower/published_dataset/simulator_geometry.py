"""The simulator geometry of a published S record: the complete porous domain inside the inner frame.

The analysis keeps the sand behind fixtures hidden (N16). Only this published
geometry is completed. Its source is the series' geometry (a rig-corrected
facies map with region roles), together with explicit configuration:

1. **Fixtures.** The fixture regions are removed from the map. They are the
   regions with a fixture role (by default ``colorchecker``) and any fixture
   polygons given in metres. The hidden sand is filled by continuing each
   layer boundary across it:
   - a boundary seen on both sides is interpolated linearly between them;
   - a boundary seen on one side only keeps that side's height.
2. **Edges.** The layers are extended to the inner walls, the floor and the
   sand surface by edge replication (as Q43). An undefined cell takes the
   facies of the nearest defined cell in its column, or, where its column has
   none, in its row. Original cells are never changed.
3. **Flags.** Every filled cell carries a flag: ``fixture_filled`` or
   ``edge_filled``.
4. **Grid.** The result is reported on the explicitly requested simulator grid:
   - majority facies and facies area fractions;
   - porosity (pore volume over bulk volume), depth, pore volume and bulk
     volume;
   - the fill flags, as area fractions and as cell flags.

   The facies table and the depth measurement points go with it as data.

Depth follows the production rule: linear between the measured points inside
their hull, nearest outside. Porosity comes from the facies table. Nothing
here is imported by a production worker.

Missing inputs refuse the export with a clear message; nothing is inferred:
- undeclared fixtures;
- a fixture polygon that covers no sand;
- an undeclared inner frame or sand surface;
- a porous facies without porosity or permeability;
- an unstated permeability unit;
- missing depth points;
- a cell that no rule can fill.

The rule-based choices that remain are counted and reported: surface columns
without free water, and gaps filled by edge replication by their role.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import io
import json
import math
from pathlib import Path, PureWindowsPath

import numpy as np

GEOMETRY_FORMAT = "darsia-simulator-geometry-v1"
REQUEST_FORMAT = "darsia-simulator-geometry-request-v1"
POROUS_ROLES = frozenset({"porous_medium", "seal"})
WATER_ROLES = frozenset({"free_water"})
FRAME_ROLES = frozenset({"frame"})
DEFAULT_FIXTURE_ROLES = ("colorchecker",)
RASTER_SHIFT = 8
FILE_STEM = "simulator_geometry"
INNER_FRAMES = ("canvas", "frame_role")
SAND_SURFACES = ("free_water", "highest_sand")
UNDEFINED = "undefined"
OPTIONAL_COLUMNS = ("swi", "grain_size", "source")
UNSTATED_UNITS = frozenset({"", "source_unit"})


@dataclass(frozen=True)
class CompletedGeometry:
    """The completed facies map on its canvas, with the domain and the fill flags (N18 steps 1-3)."""

    facies: np.ndarray
    domain: np.ndarray
    original: np.ndarray
    fixture_filled: np.ndarray
    edge_filled: np.ndarray
    unfilled: np.ndarray
    counts: dict
    drawn_behind: np.ndarray | None = None


def canvas_coordinates(shape, width_m: float, height_m: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Cell-centre coordinates of a rig-corrected canvas (origin lower left, y up) and the cell area."""
    rows, columns = (int(value) for value in shape)
    if rows < 1 or columns < 1 or not width_m > 0 or not height_m > 0:
        raise ValueError("A nonempty canvas with a positive width and height in metres is required")
    dx, dy = float(width_m) / columns, float(height_m) / rows
    x = (np.arange(columns, dtype=np.float64) + 0.5) * dx
    y = float(height_m) - (np.arange(rows, dtype=np.float64) + 0.5) * dy
    x_m, y_m = np.meshgrid(x, y)
    return x_m, y_m, dx * dy


def fixture_mask(shape, width_m: float, height_m: float, polygons_m, *, raster_shift: int = RASTER_SHIFT) -> np.ndarray:
    """The canvas cells inside any fixture polygon given in metres (the union of ``fixture_masks``)."""
    result = np.zeros(tuple(int(value) for value in shape), bool)
    for mask in fixture_masks(shape, width_m, height_m, polygons_m, raster_shift=raster_shift).values():
        result |= mask
    return result


def fixture_masks(shape, width_m: float, height_m: float, polygons_m, *,
                  raster_shift: int = RASTER_SHIFT) -> dict[str, np.ndarray]:
    """The canvas cells of each fixture polygon given in metres (the existing fixed-point rasterizer)."""
    from ..geometry.pixels import polygon_pixel_domain

    rows, columns = (int(value) for value in shape)
    if not polygons_m:
        return {}
    dx, dy = float(width_m) / columns, float(height_m) / rows

    def pixels(points):
        """Metric polygon vertices as canvas pixel coordinates (column, row; centres on integers)."""
        array = np.asarray(points, np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3 or not np.isfinite(array).all():
            raise ValueError("Every fixture polygon needs at least three finite x/y points in metres")
        return np.column_stack((array[:, 0] / dx - 0.5, (float(height_m) - array[:, 1]) / dy - 0.5))

    canvas = [[-0.5, -0.5], [columns - 0.5, -0.5], [columns - 0.5, rows - 0.5], [-0.5, rows - 0.5]]
    _domain, _inside, fixtures = polygon_pixel_domain(
        canvas, {str(name): pixels(points) for name, points in dict(polygons_m).items()}, [rows, columns],
        native_origin_xy=(0.0, 0.0), raster_scale=float(2 ** raster_shift), raster_shift=raster_shift)
    return fixtures


def porous_domain(porous, fixture, water, frame, *, sand_surface: str) -> tuple[np.ndarray, dict]:
    """The complete porous domain and how its surface was found, per column.

    The domain lies inside the frame and runs from the sand surface down to
    the column's lowest cell inside the frame (the floor).
    - With ``sand_surface="free_water"``, the surface is below the free water
      at the top of the interior. A column that starts without free water
      starts at its highest sand or fixture cell.
    - With ``"highest_sand"``, every column starts at its highest sand or
      fixture cell.

    A column with neither takes the surface of the nearest column that has
    one. Free water below the surface stays outside the domain. The counts say
    how many columns used each of the three rules.
    """
    if sand_surface not in SAND_SURFACES:
        raise ValueError(f"The sand surface must be one of {', '.join(SAND_SURFACES)}")
    porous, fixture, water, frame = (np.asarray(value, bool) for value in (porous, fixture, water, frame))
    rows, columns = porous.shape
    interior = ~frame
    top = np.full(columns, -1, np.int64)
    floor = np.full(columns, -1, np.int64)
    solid = (porous | fixture) & interior
    counts = {"surface_columns_below_free_water": 0, "surface_columns_at_highest_sand": 0,
              "surface_columns_from_neighbour": 0}
    for column in range(columns):
        inside = np.flatnonzero(interior[:, column])
        if not inside.size:
            continue
        floor[column] = inside[-1]
        start = inside[0]
        run = 0
        while (sand_surface == "free_water" and start + run < rows and interior[start + run, column]
               and water[start + run, column]):
            run += 1
        if run:
            top[column] = start + run
            counts["surface_columns_below_free_water"] += 1
        else:
            filled = np.flatnonzero(solid[:, column])
            if filled.size:
                top[column] = filled[0]
                counts["surface_columns_at_highest_sand"] += 1
    known = np.flatnonzero(top >= 0)
    if not known.size:
        return np.zeros(porous.shape, bool), counts
    missing = np.flatnonzero((top < 0) & (floor >= 0))
    if missing.size:
        nearest = known[np.abs(known[None, :] - missing[:, None]).argmin(axis=1)]
        top[missing] = top[nearest]
        counts["surface_columns_from_neighbour"] = int(missing.size)
    index = np.arange(rows)[:, None]
    domain = interior & ~water & (index >= top[None, :]) & (index <= floor[None, :]) & (floor[None, :] >= 0)
    return domain, counts


def _boundaries(facies, defined, y_m, column: int) -> list[tuple[float, int, int]]:
    """A column's layer boundaries from top to bottom: (height, facies above, facies below)."""
    rows = np.flatnonzero(defined[:, column])
    values = facies[rows, column]
    changes = np.flatnonzero(values[1:] != values[:-1])
    return [(0.5 * (y_m[rows[i], column] + y_m[rows[i + 1], column]), int(values[i]), int(values[i + 1]))
            for i in changes]


def _match(left: list, right: list) -> list[tuple[int, int]]:
    """Pair boundaries of the two sides that separate the same facies, in order (a longest common subsequence)."""
    a, b = [(upper, lower) for _h, upper, lower in left], [(upper, lower) for _h, upper, lower in right]
    table = np.zeros((len(a) + 1, len(b) + 1), np.int64)
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            table[i, j] = table[i + 1, j + 1] + 1 if a[i] == b[j] else max(table[i + 1, j], table[i, j + 1])
    pairs, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif table[i + 1, j] >= table[i, j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def continue_layers(facies, defined, hidden, y_m) -> tuple[np.ndarray, np.ndarray]:
    """Fill hidden cells by continuing each layer boundary across the hidden region (N18 step 1).

    ``defined`` marks the cells whose facies is known, ``hidden`` the cells to
    fill and ``y_m`` the cell heights. Each connected hidden region is filled
    from the nearest defined column on each side:
    - boundaries that separate the same facies on both sides are paired in
      order and interpolated linearly across the region;
    - a boundary found on one side only keeps its height.

    In each hidden column, a cell takes the facies above the region and
    switches at every boundary above it. A region with no defined column on
    either side stays unfilled. Returns the facies and the mask of filled cells.
    """
    from scipy import ndimage

    labels = np.asarray(facies)
    known = np.asarray(defined, bool)
    targets = np.asarray(hidden, bool) & ~known
    heights = np.asarray(y_m, np.float64)
    result, filled = labels.copy(), np.zeros(labels.shape, bool)
    regions, count = ndimage.label(targets, structure=np.ones((3, 3), bool))
    columns_with_cells = known.any(axis=0)
    for region in range(1, count + 1):
        rows, columns = np.nonzero(regions == region)
        first, last = int(columns.min()), int(columns.max())
        left = next((c for c in range(first - 1, -1, -1) if columns_with_cells[c]), None)
        right = next((c for c in range(last + 1, labels.shape[1]) if columns_with_cells[c]), None)
        if left is None and right is None:
            continue
        sides = {side: _boundaries(labels, known, heights, side) for side in (left, right) if side is not None}
        pairs = _match(sides[left], sides[right]) if left is not None and right is not None else []
        for column in range(first, last + 1):
            cells = rows[columns == column]
            if not cells.size:
                continue
            lines = []
            if left is not None and right is not None:
                t = (column - left) / (right - left)
                paired_left, paired_right = {i for i, _ in pairs}, {j for _, j in pairs}
                lines += [((1 - t) * sides[left][i][0] + t * sides[right][j][0], sides[left][i][1],
                           sides[left][i][2]) for i, j in pairs]
                lines += [row for k, row in enumerate(sides[left]) if k not in paired_left]
                lines += [row for k, row in enumerate(sides[right]) if k not in paired_right]
            else:
                lines += sides[left if left is not None else right]
            lines.sort(key=lambda row: -row[0])
            above = int(cells.min()) - 1
            start = None
            if above >= 0 and known[above, column]:
                # The column's own cell above wins; only boundaries below it apply.
                start = int(labels[above, column])
                lines = [row for row in lines if row[0] < heights[above, column]]
            side = left if right is None or (left is not None and column - left <= right - column) else right
            donors = np.flatnonzero(known[:, side])
            for row in cells:
                crossed = [line for line in lines if line[0] > heights[row, column]]
                if crossed:
                    value = crossed[-1][2]
                elif start is not None:
                    value = start
                elif lines:
                    value = lines[0][1]
                else:
                    value = int(labels[donors[np.abs(donors - row).argmin()], side])
                result[row, column] = value
                filled[row, column] = True
    return result, filled


def replicate_edges(facies, donors, targets, x_m, y_m) -> tuple[np.ndarray, np.ndarray]:
    """Fill target cells by edge replication (N18 step 2, as Q43): nearest donor in the column, then in the row.

    This uses the existing column rule (``geometry.pixels.extend_column_geometry``).
    The second pass runs on the transposed grid, so that a cell whose column
    has no donor takes the nearest donor in its row. On a tie, the higher
    cell, or the cell to the left, wins. Cells filled in the first pass may
    donate in the second. Returns the facies and the mask of filled cells.
    """
    from ..geometry.pixels import extend_column_geometry

    labels = np.asarray(facies)
    source, wanted = np.asarray(donors, bool), np.asarray(targets, bool) & ~np.asarray(donors, bool)
    domain = source | wanted
    if not source.any() or not wanted.any():
        return labels.copy(), np.zeros(labels.shape, bool)
    table = {int(value): 1.0 for value in np.unique(labels[source])}
    first = extend_column_geometry(labels, source, domain, y_m, porosity_by_facies=table, swi_by_facies={})
    vertical = first.geometry_filled & wanted
    stage = np.where(vertical, first.facies, labels)
    second = extend_column_geometry(stage.T, (source | vertical).T, domain.T, -np.asarray(x_m, np.float64).T,
                                    porosity_by_facies=table, swi_by_facies={})
    horizontal = second.geometry_filled.T & wanted & ~vertical
    result = np.where(horizontal, second.facies.T, stage)
    return result, vertical | horizontal


def complete_geometry(facies, roles: dict, *, width_m: float, height_m: float, inner_frame: str,
                      sand_surface: str, fixture_polygons_m=None, fixture_roles=DEFAULT_FIXTURE_ROLES,
                      raster_shift: int = RASTER_SHIFT, drawn_underlay: dict | None = None) -> CompletedGeometry:
    """Complete a rig-corrected facies map (N18 steps 1-3); ``roles`` maps each facies id to its role.

    The inputs must be declared, and a missing one raises ValueError:
    - ``inner_frame``: ``"canvas"`` (the canvas edges are the inner walls and
      the floor) or ``"frame_role"`` (cells with the frame role lie outside);
    - ``sand_surface``: see ``porous_domain``;
    - ``fixture_polygons_m``: a mapping, possibly empty, or None when the map's
      fixture-role regions are the fixtures. Each polygon must cover sand in
      the domain.

    Hidden sand is filled only by continuing its layers. Gaps (cells without a
    porous role) are filled by edge replication and counted by their role.
    Cells that no rule fills stay ``unfilled``.
    """
    labels = np.asarray(facies)
    if labels.ndim != 2 or labels.dtype.kind not in "iu" or not all(labels.shape):
        raise ValueError("The facies map must be a nonempty two-dimensional integer grid")
    if inner_frame not in INNER_FRAMES:
        raise ValueError(f"The inner frame must be declared as one of {', '.join(INNER_FRAMES)}")
    role = {int(key): str(value) for key, value in dict(roles).items()}

    def has(names):
        """Cells whose facies has one of the roles ``names``."""
        ids = [key for key, value in role.items() if value in set(names)]
        return np.isin(labels, ids) if ids else np.zeros(labels.shape, bool)

    frame, water = has(FRAME_ROLES), has(WATER_ROLES)
    if inner_frame == "frame_role" and not frame.any():
        raise ValueError("The inner frame is declared by the frame role, but the map has no frame cell")
    if inner_frame == "canvas" and frame.any():
        raise ValueError("The map has frame cells; declare the inner frame as 'frame_role'")
    if sand_surface == "free_water" and not water.any():
        raise ValueError("The sand surface is declared by the free water, but the map has no free-water cell")
    roled = has(fixture_roles)
    if fixture_polygons_m is None and not roled.any():
        raise ValueError("No fixtures are declared: give fixture polygons in metres, or an explicit empty mapping "
                         "for a rig without fixtures")
    polygons = fixture_masks(labels.shape, width_m, height_m, fixture_polygons_m or {}, raster_shift=raster_shift)
    fixture = roled & ~frame
    if drawn_underlay is None:
        for mask in polygons.values():
            fixture |= mask & ~frame
    porous_ids = sorted(key for key, value in role.items() if value in POROUS_ROLES)
    if 0 in porous_ids and np.any(labels == 0):
        raise ValueError("Facies id 0 cannot be porous: the production depth and porosity rule skips it")
    porous = has(POROUS_ROLES) & ~fixture
    domain, counts = porous_domain(porous, fixture, water, frame, sand_surface=sand_surface)
    if drawn_underlay is None:
        for name, mask in polygons.items():
            if not (mask & domain).any():
                raise ValueError(f"The fixture polygon {name!r} covers no cell of the porous domain")
    original = porous & domain
    x_m, y_m, _area = canvas_coordinates(labels.shape, width_m, height_m)
    hidden = fixture & domain
    drawn_behind = None
    seeds, known = labels, original
    if drawn_underlay is not None:
        footprint = np.asarray(drawn_underlay["fixture_mask"], bool)
        drawn = np.asarray(drawn_underlay["drawn_facies"])
        if (footprint.shape != labels.shape or drawn.shape != labels.shape
                or drawn.dtype.kind not in "iu" or not np.array_equal(footprint, roled)):
            raise ValueError("The drawn underlay must match this map's fixture-role grid")
        drawn_behind = np.asarray(drawn_underlay.get("drawn_mask", drawn != 0), bool)
        allowed = [key for key, value in role.items() if value in POROUS_ROLES | WATER_ROLES | {"excluded"}]
        if (drawn_behind.shape != labels.shape or np.any(drawn_behind & ~footprint)
                or np.any(drawn_behind & ~np.isin(drawn, allowed))):
            raise ValueError("Drawn layers need an explicit non-fixture role inside the fixture")
        drawn_porous = drawn_behind & np.isin(drawn, porous_ids)
        # An explicit drawing wins even beyond an inferred sand surface. A
        # drawn excluded or free-water layer stays outside the porous domain.
        domain = (domain | drawn_porous) & ~(drawn_behind & ~drawn_porous)
        original = porous & domain
        hidden = fixture & domain
        seeds = labels.copy()
        seeds[drawn_behind] = drawn[drawn_behind]
        known = original | drawn_porous
    targets = hidden if drawn_behind is None else hidden & ~drawn_behind
    stage, fixture_filled = continue_layers(seeds, known, targets, y_m)
    gaps = domain & ~original & ~hidden
    if drawn_behind is not None:
        gaps &= ~drawn_behind
    result, edge_filled = replicate_edges(stage, known | fixture_filled, gaps, x_m, y_m)
    unfilled = domain & ~original & ~fixture_filled & ~edge_filled
    if drawn_behind is not None:
        unfilled &= ~drawn_behind
    by_role = {}
    for value in np.unique(labels[gaps]):
        name = role.get(int(value), UNDEFINED)
        by_role[name] = by_role.get(name, 0) + int(np.count_nonzero(gaps & (labels == value)))
    counts = dict(counts, domain_cells=int(domain.sum()), original_cells=int(original.sum()),
                  fixture_filled_cells=int(fixture_filled.sum()), edge_filled_cells=int(edge_filled.sum()),
                  unfilled_cells=int(unfilled.sum()), edge_filled_cells_by_role=by_role,
                  fixture_polygons=sorted(polygons), fixture_role_cells=int((roled & domain).sum()))
    if drawn_behind is not None:
        counts["drawn_behind_cells"] = int(drawn_behind.sum())
    output_mask = domain if drawn_behind is None else domain | drawn_behind
    return CompletedGeometry(np.where(output_mask, result, labels), domain, original, fixture_filled, edge_filled,
                             unfilled, counts, drawn_behind)


def _simulator_resolution_mm(value):
    """An explicit simulator spacing, independent of saved-map choices."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("An explicit finite positive simulator resolution in millimetres is required")
    if float(value) / 1000.0 == 0:
        raise ValueError("The simulator resolution cannot be represented in metres")
    return float(value)


def covering_extent(width_m: float, height_m: float, resolution_mm: float) -> tuple[float, float, float, float]:
    """The smallest simulator-lattice extent covering the canvas (exact decimal steps)."""
    resolution_mm = _simulator_resolution_mm(resolution_mm)
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0
           for value in (width_m, height_m)):
        raise ValueError("The simulator canvas needs finite positive width and height in metres")
    step = Decimal(str(resolution_mm)) / Decimal(1000)

    def edge(value):
        """The first lattice edge at or above ``value``."""
        count = (Decimal(str(value)) / step).to_integral_value(rounding="ROUND_CEILING")
        return float(count * step)

    return 0.0, edge(width_m), 0.0, edge(height_m)


def _simulator_grid(x_m, y_m, pixel_area_m2, *, include, extent_m, resolution_mm,
                    precision="float32"):
    """Assign the unchanged canvas to its independent simulator lattice.

    Saved-map spacings retain the original constructor. Other explicit
    simulator spacings use the same grid data and reducers, without widening
    the saved-storage policy or rescaling/clamping any native coordinates.
    """
    from ..storage_resolution import MetricStorageGrid, RESOLUTIONS_MM, make_grid

    step_mm = _simulator_resolution_mm(resolution_mm)
    if step_mm in RESOLUTIONS_MM:
        return make_grid(x_m, y_m, pixel_area_m2, include=include, extent_m=extent_m,
                         storage_resolution_mm=step_mm, precision=precision)
    if precision not in {"float32", "float64"}:
        raise ValueError("One declared storage precision, float32 or float64, is required.")
    x, y, area = (np.asarray(value) for value in (x_m, y_m, pixel_area_m2))
    included = np.asarray(include)
    if x.ndim != 2 or x.shape != y.shape or x.shape != area.shape or x.shape != included.shape:
        raise ValueError("Metric coordinates, area and include must share one native 2D grid.")
    if included.dtype != np.bool_ or not np.any(included):
        raise ValueError("An explicit nonempty boolean geometric coverage is required.")
    if len(extent_m) != 4 or any(isinstance(value, bool) or not math.isfinite(value) for value in extent_m):
        raise ValueError("Explicit finite metric extent x0,x1,y0,y1 is required.")
    # Preserve exact origin-zero lattice and cell-count admission from the
    # storage constructor, while the simulator owns its spacing choice.
    step_n, step_d = (Decimal(str(step_mm)) / Decimal(1000)).as_integer_ratio()
    edges = []
    for value in extent_m:
        value_n, value_d = Decimal(str(value)).as_integer_ratio()
        index, remainder = divmod(value_n * step_d, value_d * step_n)
        if remainder:
            raise ValueError("Simulator extent endpoints must lie on the exact origin-zero metric lattice.")
        edges.append(index)
    height, width = edges[3] - edges[2], edges[1] - edges[0]
    if width < 1 or height < 1:
        raise ValueError("Metric simulator extent must have positive width and height.")
    if width * height > 100_000_000:
        raise ValueError("Declared metric simulator grid exceeds the bounded cell count.")
    invalid_count = int(np.count_nonzero(~(np.isfinite(x) & np.isfinite(y))))
    if invalid_count:
        raise ValueError("Unassignable native simulator coordinates: invalid=" + str(invalid_count))
    x0, x1, y0, y1 = map(float, extent_m)
    if np.any((x < x0) | (x >= x1) | (y < y0) | (y >= y1)):
        raise ValueError("Full-canvas native pixel centre lies outside declared half-open metric extent.")
    if not np.isfinite(area[included]).all() or np.any(area[included] < 0):
        raise ValueError("Included physical pixel areas must be finite and nonnegative.")
    step = step_mm / 1000.0
    col, row = np.floor((x - x0) / step).astype(np.int64), np.floor((y - y0) / step).astype(np.int64)
    if np.any((col < 0) | (col >= width) | (row < 0) | (row >= height)):
        raise ValueError("Native pixel centre has an unassignable half-open cell index; no clamp is allowed.")
    full_ids = row * width + col
    ids = np.where(included, full_ids, -1)
    size = height * width
    cell_area = np.bincount(ids[included], weights=area[included].astype(np.float64), minlength=size)
    count = np.bincount(ids[included], minlength=size)
    full_count = np.bincount(full_ids.ravel(), minlength=size)
    coordinate_counts = dict(native_pixels=int(x.size), inside_extent=int(x.size),
                             outside_extent=0, invalid_coordinates=0)
    return MetricStorageGrid(x.shape, (height, width), (x0, x1, y0, y1), step_mm, precision,
                             ids, included.copy(), area.astype(np.float64), cell_area, count,
                             full_ids, full_count, coordinate_counts)


def facies_table(properties: list, extra_rows=None) -> list[dict]:
    """The facies table as published: porosity, permeability, Swi, grain size and source per facies id."""
    extra = {int(row["facies_id"]): row for row in (extra_rows or ())}
    rows = []
    for item in properties:
        facies_id = int(item["id"])
        more = extra.get(facies_id, {})
        rows.append({"facies_id": facies_id, "name": str(item.get("name") or f"Facies {facies_id}"),
                     "analysis_role": item.get("analysis_role"), "porosity": item.get("porosity"),
                     "permeability": item.get("permeability"),
                     "permeability_unit": item.get("permeability_unit"), "swi": more.get("swi"),
                     "grain_size": more.get("grain_size"), "source": more.get("source")})
    return rows


def table_problems(table: list, present_ids, *, permeability_unit: str | None = None) -> tuple[list, list, list]:
    """Check the facies table for the facies present in the domain; return refusals, warnings and the table.

    Every porous facies needs a porosity in (0, 1] and a positive
    permeability. The permeability unit must be stated, in the table or by
    ``permeability_unit``; an unstated unit refuses, and two different units
    refuse. Missing Swi, grain size or source values are reported as warnings
    and published as null. The returned table carries the stated unit.
    """
    refusals, warnings, rows = [], [], []
    present = {int(value) for value in present_ids}
    for row in table:
        row = dict(row)
        if row["facies_id"] in present and row.get("analysis_role") in POROUS_ROLES:
            porosity, permeability = row.get("porosity"), row.get("permeability")
            if porosity is None or not 0 < float(porosity) <= 1:
                refusals.append(f"Facies {row['facies_id']} has no porosity in (0, 1] in the series table")
            if permeability is None or not float(permeability) > 0:
                refusals.append(f"Facies {row['facies_id']} has no positive permeability in the series table")
            unit = str(row.get("permeability_unit") or "")
            if unit in UNSTATED_UNITS:
                if not permeability_unit:
                    refusals.append(f"The permeability unit of facies {row['facies_id']} is not stated; state it "
                                    "in the request (permeability_unit)")
                unit = permeability_unit or unit
            elif permeability_unit and permeability_unit != unit:
                refusals.append(f"Facies {row['facies_id']}: the table states {unit}, the request "
                                f"{permeability_unit}")
            row["permeability_unit"] = unit
            missing = [name for name in OPTIONAL_COLUMNS if row.get(name) is None]
            if missing:
                warnings.append(f"Facies {row['facies_id']} has no {', '.join(missing)} in the series table; "
                                "published as null")
        rows.append(row)
    return sorted(set(refusals), key=refusals.index), warnings, rows


def read_property_columns(path) -> list[dict]:
    """The optional Swi, grain size and source columns of a facies properties table (CSV)."""
    import csv

    if path is None or not str(path) or not Path(path).is_file():
        return []
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            columns = {str(key).strip().casefold(): value for key, value in row.items() if key is not None}
            identity = columns.get("facies_id") or columns.get("id")
            if identity in (None, ""):
                continue

            def number(name):
                """A finite number from a column, or None."""
                try:
                    value = float(columns.get(name))
                except (TypeError, ValueError):
                    return None
                return value if math.isfinite(value) else None

            rows.append({"facies_id": int(float(identity)), "swi": number("swi"),
                         "grain_size": number("grain_size") if number("grain_size") is not None
                         else columns.get("grain_size") or None,
                         "source": columns.get("source") or None})
    return rows


def grid_products(completed: CompletedGeometry, *, width_m: float, height_m: float, depth_points, table: list,
                  resolution_mm: float, precision: str = "float32") -> tuple[dict, dict]:
    """The completed geometry on the requested simulator grid (N18 step 4): arrays and grid metadata."""
    from ..mass.n1_geometry import measured_geometry
    from ..storage_resolution import aggregate_static

    x_m, y_m, area = canvas_coordinates(completed.facies.shape, width_m, height_m)
    porous_ids = {row["facies_id"] for row in table if row.get("analysis_role") in POROUS_ROLES}
    porosity = {row["facies_id"]: row["porosity"] for row in table
                if row["facies_id"] in porous_ids and row.get("porosity") is not None}
    swi = {row["facies_id"]: row["swi"] for row in table
           if row["facies_id"] in porous_ids and row.get("swi") is not None}
    facies = np.where(completed.domain, completed.facies, 0).astype(np.int64)
    measured = measured_geometry(x_m, y_m, facies, depth_points, porosity_by_facies=porosity, swi_by_facies=swi)
    cell = np.full(facies.shape, area, np.float64)
    grid = _simulator_grid(x_m, y_m, cell, include=completed.domain,
                          extent_m=covering_extent(width_m, height_m, resolution_mm),
                          resolution_mm=resolution_mm, precision=precision)
    dtype = np.dtype(precision)
    ids, included = grid.cell_index[grid.included], grid.included
    size = int(np.prod(grid.shape))

    def total(value):
        """The sum of a canvas field over each grid cell (domain cells only)."""
        return np.bincount(ids, weights=np.asarray(value, np.float64)[included], minlength=size)

    def per_area(numerator, denominator):
        """A ratio per grid cell, NaN where the denominator is zero."""
        out = np.full(size, np.nan, np.float64)
        np.divide(numerator, denominator, out=out, where=denominator > 0)
        return out.reshape(grid.shape).astype(dtype)

    depth = np.where(completed.domain, measured["depth_m"], 0.0)
    phi = np.where(completed.domain & np.isfinite(measured["porosity"]), measured["porosity"], 0.0)
    domain_area = total(cell)
    bulk = total(cell * depth)
    pore = total(cell * depth * phi)
    static = aggregate_static(grid, depth_m=depth, porosity=np.where(completed.domain, measured["porosity"], np.nan),
                              facies=facies)
    present = sorted(int(value) for value in np.unique(facies[completed.domain]))
    fractions = np.stack([per_area(total(cell * (facies == value)), domain_area) for value in present]) if present \
        else np.zeros((0, *grid.shape), dtype)
    step = float(resolution_mm) / 1000.0
    arrays = {
        "facies_majority": static["facies_majority"].astype(np.int32),
        "facies_majority_area_fraction": static["facies_majority_area_fraction"],
        "facies_ids": np.asarray(present, np.int32),
        "facies_area_fractions": fractions,
        "porosity": per_area(pore, bulk),
        "depth_m": per_area(bulk, domain_area),
        "pore_volume_m3": pore.reshape(grid.shape).astype(dtype),
        "bulk_volume_m3": bulk.reshape(grid.shape).astype(dtype),
        "domain_area_m2": domain_area.reshape(grid.shape).astype(dtype),
        "domain_area_fraction": (domain_area / (step * step)).reshape(grid.shape).astype(dtype),
        "fixture_filled_area_fraction": per_area(total(cell * completed.fixture_filled), domain_area),
        "edge_filled_area_fraction": per_area(total(cell * completed.edge_filled), domain_area),
        "unfilled_area_fraction": per_area(total(cell * completed.unfilled), domain_area),
        "depth_outside_hull_area_fraction": per_area(total(cell * measured["depth_nearest_outside_hull"]),
                                                     domain_area),
    }
    for name in ("fixture_filled", "edge_filled", "unfilled"):
        arrays[name] = np.nan_to_num(arrays[name + "_area_fraction"]) > 0
    if completed.drawn_behind is not None:
        # Drawn excluded/free-water pixels lie outside the porous domain. Use
        # the existing full-canvas cell indices so their flags and roles are
        # retained, without admitting them to depth, porosity or pore volume.
        full_ids = grid.probability_cell_index.ravel()
        def drawn_area(mask):
            return np.bincount(full_ids, weights=(cell * mask).ravel(), minlength=size)
        drawn_ids = sorted(int(v) for v in np.unique(completed.facies[completed.drawn_behind]))
        arrays["drawn_behind_area_fraction"] = (drawn_area(completed.drawn_behind) / (step * step)).reshape(grid.shape).astype(dtype)
        arrays["drawn_behind"] = np.nan_to_num(arrays["drawn_behind_area_fraction"]) > 0
        arrays["drawn_behind_facies_ids"] = np.asarray(drawn_ids, np.int32)
        arrays["drawn_behind_facies_area_fractions"] = np.stack([
            (drawn_area(completed.drawn_behind & (completed.facies == value)) / (step * step)).reshape(grid.shape).astype(dtype)
            for value in drawn_ids]) if drawn_ids else np.zeros((0, *grid.shape), dtype)
    return arrays, grid.metadata()


UNITS = {"facies_majority": "facies id", "facies_majority_area_fraction": "1", "facies_ids": "facies id",
         "facies_area_fractions": "1 (one layer per facies id, same order)", "porosity": "1 (pore over bulk volume)",
         "depth_m": "m", "pore_volume_m3": "m3", "bulk_volume_m3": "m3", "domain_area_m2": "m2",
         "domain_area_fraction": "1 of the cell", "fixture_filled_area_fraction": "1 of the domain area",
         "edge_filled_area_fraction": "1 of the domain area", "unfilled_area_fraction": "1 of the domain area",
         "depth_outside_hull_area_fraction": "1 of the domain area", "fixture_filled": "flag",
         "edge_filled": "flag", "unfilled": "flag"}
UNITS.update(drawn_behind="flag", drawn_behind_area_fraction="1 of the cell",
             drawn_behind_facies_ids="facies id",
             drawn_behind_facies_area_fractions="1 of the cell (one layer per drawn facies id, same order)")


def export_geometry(facies, roles: dict, table: list, *, width_m: float, height_m: float, depth_points,
                    resolution_mm: float, inner_frame: str, sand_surface: str, fixture_polygons_m=None,
                    fixture_roles=DEFAULT_FIXTURE_ROLES, permeability_unit: str | None = None,
                    depth_source: str = "", precision: str = "float32",
                    source: dict | None = None, drawn_underlay: dict | None = None) -> tuple[bytes, bytes, dict, list]:
    """The simulator geometry of one run (N18): the NPZ bytes, the JSON bytes, the counts and the warnings.

    A missing or unusable input raises ValueError with the reason, and nothing
    is inferred in its place. That covers the declarations of
    ``complete_geometry``, the facies table (``table_problems``), at least
    three depth points with positive depths, and every cell of the domain
    being filled.
    """
    source = dict(source or {})
    stored_underlay = source.pop("_drawn_fixture_underlay", None)
    if drawn_underlay is None:
        drawn_underlay = stored_underlay
    elif stored_underlay is not None:
        raise ValueError("Supply one explicit drawn underlay, not two competing bindings")
    completed = complete_geometry(facies, roles, width_m=width_m, height_m=height_m, inner_frame=inner_frame,
                                  sand_surface=sand_surface, fixture_polygons_m=fixture_polygons_m,
                                  fixture_roles=fixture_roles, drawn_underlay=drawn_underlay)
    counts = completed.counts
    if counts["unfilled_cells"]:
        raise ValueError(f"{counts['unfilled_cells']} cell(s) of the porous domain have no geometry: a hidden "
                         "region without a defined column on either side, or a gap without a donor")
    present = np.unique(completed.facies[completed.domain])
    refusals, warnings, table = table_problems(table, present, permeability_unit=permeability_unit)
    if refusals:
        raise ValueError("; ".join(refusals))
    points = np.asarray(depth_points, np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3 or not np.isfinite(points).all() \
            or np.any(points[:, 2] <= 0):
        raise ValueError("At least three measured depth points (x, y, depth) with positive depths are required")
    arrays, grid = grid_products(completed, width_m=width_m, height_m=height_m, depth_points=points, table=table,
                                 resolution_mm=resolution_mm, precision=precision)
    if counts["surface_columns_at_highest_sand"] and sand_surface == "free_water":
        warnings.append(f"{counts['surface_columns_at_highest_sand']} column(s) start without free water; their "
                        "sand surface is their highest sand cell")
    if counts["surface_columns_from_neighbour"]:
        warnings.append(f"{counts['surface_columns_from_neighbour']} column(s) have neither free water nor sand; "
                        "their sand surface is the nearest column's")
    for name, cells in sorted(counts["edge_filled_cells_by_role"].items()):
        warnings.append(f"{cells} gap cell(s) with the role {name!r} filled by edge replication (edge_filled)")
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    document = {
        "format": GEOMETRY_FORMAT,
        "grid": grid,
        "canvas": {"shape": list(completed.facies.shape), "width_m": float(width_m), "height_m": float(height_m),
                   "coordinate_convention": "origin_lower_left_x_right_y_up"},
        "method": {
            "fixtures": "removed; each layer boundary continued across the hidden region (linear between its two "
                        "sides, a one-sided boundary kept at its height)",
            "edges": "edge replication to the walls, the floor and the sand surface: the nearest defined cell in "
                     "the column, else in the row",
            "fixture_roles": list(fixture_roles),
            "fixture_polygons": sorted(str(name) for name in (fixture_polygons_m or {})),
            "fixtures_declared": "fixture-role regions only" if fixture_polygons_m is None else (
                "polygons in metres" if fixture_polygons_m else "none (explicitly)"),
            "inner_frame": inner_frame,
            "sand_surface": sand_surface,
            "depth": "linear between the measured points inside their hull, nearest outside",
            "depth_points_source": depth_source,
            "porosity": "from the facies table; reported as pore volume over bulk volume",
            "analysis_unchanged": "the analysis keeps the hidden sand hidden (N16); only this geometry is completed",
        },
        "counts": counts,
        "warnings": warnings,
        "arrays": {name: {"dtype": str(value.dtype), "shape": list(value.shape), "unit": UNITS[name]}
                   for name, value in arrays.items()},
        "facies_table": table,
        "depth_points": [[float(x), float(y), float(d)] for x, y, d in points],
        "source": source or {},
    }
    if drawn_underlay is not None:
        from ..layer_painting_fixtures import fixture_boundary_agreement
        document["method"]["drawn_behind"] = "every explicit drawn role retained; only undrawn fixture cells continued as sand"
        document["method"]["fixtures_declared"] = "frozen fixture-role regions; measured polygons compared only"
        document["source"]["fixture_underlay"] = dict(drawn_underlay.get("provenance") or {})
        document["fixture_boundary_agreement"] = fixture_boundary_agreement(
            drawn_underlay["fixture_mask"], fixture_polygons_m, width_m=width_m, height_m=height_m)
    text = (json.dumps(document, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return stream.getvalue(), text, counts, warnings


def load_request(path) -> dict:
    """A simulator geometry request: the geometry revision, the fixtures and the depth points to use.

    Format ``darsia-simulator-geometry-request-v1`` (JSON)::

        {"format": "darsia-simulator-geometry-request-v1",
         "geometry": "<the geometry revision or configuration JSON, relative to the series folder>",
         "inner_frame": "canvas" or "frame_role",
         "sand_surface": "free_water" or "highest_sand",
         "fixture_roles": ["colorchecker"],
         "fixtures_m": {"<name>": [[x, y], ...]},
         "runs": {"<run>": {"fixtures_m": {...}}},
         "permeability_unit": "<the unit, when the series table does not state it>",
         "depth_points": [[x, y, depth], ...] or "<a CSV with x, y, depth, relative to the series folder>"}

    ``format``, ``geometry``, ``inner_frame`` and ``sand_surface`` are required.
    The fixtures come from ``fixtures_m`` (a run's own entry replaces it, and an
    empty mapping declares no fixtures) and from the fixture-role regions; a
    run with neither refuses. Without ``depth_points`` the run configuration's
    own measured depth points are used.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("format") != REQUEST_FORMAT:
        raise ValueError(f"A simulator geometry request ({REQUEST_FORMAT}) is required")
    unknown = set(value) - {"format", "geometry", "inner_frame", "sand_surface", "fixture_roles", "fixtures_m",
                            "runs", "depth_points", "permeability_unit"}
    if unknown or not isinstance(value.get("geometry"), str):
        raise ValueError("The request names its geometry and only known fields")
    if value.get("inner_frame") not in INNER_FRAMES:
        raise ValueError(f"The request must declare inner_frame as one of {', '.join(INNER_FRAMES)}")
    if value.get("sand_surface") not in SAND_SURFACES:
        raise ValueError(f"The request must declare sand_surface as one of {', '.join(SAND_SURFACES)}")
    return value


def series_file(root, value: str) -> Path:
    """A path named by a request, inside the series folder (a path that leaves it is refused)."""
    from .layout import inside

    return inside(root, str(value).replace("\\", "/"))


def run_geometry(revision: dict, run: str) -> tuple[np.ndarray, dict, list, dict]:
    """One run's facies map, roles by facies id, facies table and source identity from a geometry revision."""
    from ..geometry_facies import _canonical_facies, _load_array, resolve_run_geometry_artifacts

    configuration = dict(revision.get("configuration") or revision)
    if configuration.get("coordinate_convention") != "origin_lower_left_x_right_y_up":
        raise ValueError("The geometry must be rig-corrected with its origin at the lower left")
    continuation_row = next((row for row in (configuration.get("run_geometry") or {}).get("runs") or []
        if str(row.get("run_id") or "").casefold() == str(run).casefold()
        and (row.get("registration") or {}).get("kind") == "painted-map-placement-v2"), None)
    if continuation_row is not None:
        from ..geometry_raster_sources import canonical_parent_payload
        canonical_parent_payload(configuration, run_id=run)
    artifacts = resolve_run_geometry_artifacts(revision, run) if "artifacts" in revision else {
        "scope": "series_reference", "labels_path": configuration.get("labels_path", "")}
    labels_path = str(artifacts.get("labels_path") or configuration.get("labels_path") or "")
    if not labels_path:
        raise ValueError(f"The geometry of run {run} has no label map")
    facies, properties = _canonical_facies(dict(configuration, labels_path=labels_path))
    labels = np.squeeze(_load_array(Path(labels_path)))
    known = [int(row["label_id"]) for row in configuration.get("regions") or ()]
    undefined = ~np.isin(labels, known)
    roles = {int(row["id"]): row["analysis_role"] for row in properties}
    if undefined.any():
        # A label without a region is undefined, never facies 0 (which may be a real facies).
        facies = np.where(undefined, max([*roles, 0]) + 1, facies)
    extra = read_property_columns(artifacts.get("facies_properties_path")
                                  or (revision.get("artifacts") or {}).get("facies_properties_path"))
    source = {"scope": artifacts.get("scope"), "labels": PureWindowsPath(labels_path).name,
              "revision": revision.get("revision_id")}
    if continuation_row is not None:
        from ..photo_edge_continuation import load_contract, verify_product, POLICY
        from ..geometry_raster_sources import _integer_map
        registration = continuation_row["registration"]
        contract = load_contract(registration["continuation"]["path"], str(run).casefold())
        replay = verify_product(labels_path, _integer_map(registration["source_map"]["path"]), contract)
        source["photo_edge_continuation"] = dict(policy=POLICY, **replay["counts"],
            labels_with_masks=labels_path, adopted_by_this_export=False)
    if (configuration.get("painted_fixture_underlays")
            or any((row.get("registration") or {}).get("kind") in {"painted-map-placement-v1", "painted-map-placement-v2"}
                   for row in (configuration.get("run_geometry") or {}).get("runs") or [])):
        from ..layer_painting_fixtures import adopted_underlay
        underlay = adopted_underlay(configuration, run.casefold(), labels)
        if underlay is not None:
            source["_drawn_fixture_underlay"] = underlay
    table = facies_table(properties, extra)
    if "painted_adoption_table" in configuration:
        from ..geometry_raster_sources import canonical_parent_payload
        import csv
        import io
        payload = canonical_parent_payload(configuration, run_id=run)
        path = artifacts.get("facies_properties_path") or (revision.get("artifacts") or {}).get("facies_properties_path")
        if not path or Path(path).read_bytes() != payload["painted_table_bytes"]:
            raise ValueError("Simulator geometry lost its exact derived painting CSV")
        reader = csv.DictReader(io.StringIO(payload["painted_table_bytes"].decode("utf8"), newline=""))
        columns = list(reader.fieldnames)
        names = {column.strip().casefold(): column for column in columns}
        identity = names.get("id") or names["facies_id"]
        full = {int(row[identity]): row for row in reader}
        for row in table:
            row["source_columns"] = columns
            row["source_cells"] = [full[row["facies_id"]][column] for column in columns]
        source["facies_table_sha256"] = configuration["painted_adoption_table"]["table"]["sha256"]
    return np.asarray(facies), roles, table, source


def depth_points_for(request: dict, series_root, config: dict) -> np.ndarray:
    """The measured depth points of a run: the request's, else the run configuration's own."""
    from ..mass.n1_pipeline import _depth_points

    value = request.get("depth_points")
    if value is None:
        value = ((((config or {}).get("mass") or {}).get("template") or {}).get("shared") or {}).get("depth_points")
    if value is None:
        raise ValueError("No measured depth points: name them in the request or in the run configuration")
    if isinstance(value, str):
        value = {"path": value}
    reader = type("Reader", (), {"path": staticmethod(lambda bound: series_file(series_root, bound["path"]))})
    return _depth_points(value, reader)
