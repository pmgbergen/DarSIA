"""Per-pixel rig support and same-column physical-field transport."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .context import _raster_polygon


def _boolean_grid(value, name, shape=None):
    result = np.asarray(value)
    if result.dtype != np.bool_ or result.ndim != 2 or not all(result.shape):
        raise ValueError(name + " must be a nonempty two-dimensional boolean grid")
    if shape is not None and result.shape != shape:
        raise ValueError(name + " must share the exact pixel grid")
    return result


def _height_grid(value, shape):
    result = np.asarray(value)
    if (
        result.shape != shape
        or result.dtype.kind not in "fiu"
        or not np.isfinite(result).all()
    ):
        raise ValueError("Physical heights must be finite numbers on the exact pixel grid")
    result = np.asarray(result, np.float64)
    if np.any(np.diff(result, axis=0) >= 0):
        raise ValueError("Physical height must decrease down every image column")
    return result


def _column_neighbours(mask):
    """Return previous/following same-column categorical donors."""
    rows = np.arange(mask.shape[0], dtype=np.int32)[:, None]
    previous = np.maximum.accumulate(np.where(mask, rows, -1), axis=0)
    following = np.minimum.accumulate(np.where(mask, rows, mask.shape[0])[::-1], axis=0)[::-1]
    return previous, following


def rig_pixel_domain(inner_mask, fixture_masks):
    """Return the rig interior with all explicitly supplied hardware removed.

    ``inner_mask`` and every named mask in ``fixture_masks`` must be boolean
    arrays on one grid. An empty mapping explicitly declares no fixtures.
    Fixtures may overlap or extend outside the interior. No photo, sand,
    floor-line, or phase classification changes this physical domain.
    """
    inside = _boolean_grid(inner_mask, "Rig interior")
    if not isinstance(fixture_masks, Mapping):
        raise TypeError("Fixture masks must be an explicit named mapping")
    domain = inside.copy()
    for name, value in fixture_masks.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Every fixture mask requires a nonempty name")
        domain &= ~_boolean_grid(value, "Fixture " + name, inside.shape)
    return domain


def polygon_pixel_domain(
    inner_polygon, fixture_polygons, shape_hw, *, native_origin_xy,
    raster_scale, raster_shift,
):
    """Place measured frame-native polygons on the complete reporting canvas.

    Subtract ``native_origin_xy`` once to convert native coordinates to canvas
    coordinates. Interior and named hardware use the existing fixed-point
    polygon rasterizer; its scale and shift are explicit configuration data.
    Return the physical domain, the interior mask and named fixture masks.
    No photo top, floor, sand classifier or phase label clips the result.
    The caller must supply all intended hardware; this function cannot infer
    an omitted fitting from a mask name or an image.
    """
    if (not isinstance(shape_hw, (tuple, list)) or len(shape_hw) != 2
            or any(type(n) is not int or n <= 0 for n in shape_hw)):
        raise ValueError("Canvas shape must contain two positive integer dimensions")
    origin = np.asarray(native_origin_xy, np.float64)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("Native origin must contain two finite coordinates")
    if (type(raster_shift) is not int or not 0 <= raster_shift <= 16
            or not np.isfinite(raster_scale) or raster_scale != 2 ** raster_shift):
        raise ValueError("Raster scale must equal the configured fixed-point shift")
    if not isinstance(fixture_polygons, Mapping):
        raise TypeError("Explicit named fixture polygons are required")

    def raster(value):
        points = np.asarray(value, np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all()):
            raise ValueError("Every physical polygon requires at least three finite x/y points")
        points = points - origin
        fixed = np.rint(points * raster_scale)
        limits = np.iinfo(np.int32)
        if np.any(fixed < limits.min) or np.any(fixed > limits.max):
            raise ValueError("Polygon exceeds fixed-point coordinate range")
        return _raster_polygon(points, tuple(shape_hw), raster_scale=raster_scale,
                               raster_shift=raster_shift)

    inside = raster(inner_polygon)
    fixtures = {}
    for name, polygon in fixture_polygons.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Every fixture polygon requires a nonempty name")
        fixtures[name] = raster(polygon) & inside
    return rig_pixel_domain(inside, fixtures), inside, fixtures


def visible_pixels(domain, source_valid):
    """Select pixels whose source supplies valid colour inside the rig.

    Both arguments are boolean grids. Visibility uses the source validity
    directly; a historical sand classifier or imputed colour is not evidence
    that a pixel was observed.
    """
    rig = _boolean_grid(domain, "Rig domain")
    valid = _boolean_grid(source_valid, "Source validity", rig.shape)
    return rig & valid


@dataclass(frozen=True)
class ColumnGeometry:
    """Facies/property transport with explicit source rows and missing support.

    ``geometry_unknown`` concerns facies/porosity support inside the domain.
    Depth is deliberately absent: evaluate measured depth at each target's
    own physical coordinates, then combine its validity with this support.
    Missing supplied water saturation stays NaN for gas-specific admission.
    """

    facies: np.ndarray
    porosity: np.ndarray
    swi: np.ndarray
    defined: np.ndarray
    geometry_filled: np.ndarray
    geometry_unknown: np.ndarray
    donor_rows: np.ndarray
    donor_distance_m: np.ndarray


def _property_table(value, name, *, positive):
    if not isinstance(value, Mapping):
        raise TypeError(name + " must be an explicit facies mapping")
    result = {}
    for label, number in value.items():
        if isinstance(label, (bool, np.bool_)):
            raise ValueError(name + " facies keys must be nonnegative integers")
        try:
            key = int(label)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(name + " facies keys must be nonnegative integers") from error
        if key < 0 or (isinstance(label, str) and label != str(key)) or (
            not isinstance(label, str) and key != label
        ):
            raise ValueError(name + " facies keys must be nonnegative integers")
        if key in result:
            raise ValueError(name + " contains duplicate facies keys")
        if isinstance(number, (bool, np.bool_)) or not isinstance(
            number, (int, float, np.integer, np.floating)
        ):
            raise ValueError(name + " values must be finite physical fractions")
        number = float(number)
        if not np.isfinite(number) or not 0 <= number <= 1 or (positive and number == 0):
            raise ValueError(name + " values must be finite physical fractions")
        result[key] = number
    return result


def extend_column_geometry(
    facies,
    defined,
    domain,
    y_m,
    *,
    porosity_by_facies,
    swi_by_facies,
):
    """Extend undefined facies from the nearest measured pixel in that column.

    ``defined`` declares the original categorical map, ``domain`` excludes
    hardware, and ``y_m`` supplies physical heights decreasing down each
    column. Donors must be original defined pixels inside the domain with
    supplied valid porosity. Distances use metres, with an equal-distance
    tie selecting the higher pixel. Newly filled pixels never become donors.

    Defined facies and all input arrays remain unchanged. No donor in a
    column means explicit unknown geometry; neither another column nor a
    global nearest pixel is used. Porosity and water saturation are looked
    up from the supplied tables. Depth is not transported from the donor.
    """
    labels = np.asarray(facies)
    if labels.ndim != 2 or not all(labels.shape) or labels.dtype.kind not in "iu":
        raise ValueError("Facies must be a nonempty two-dimensional integer grid")
    if np.any(labels < 0):
        raise ValueError("Facies values must be nonnegative")
    rig = _boolean_grid(domain, "Rig domain", labels.shape)
    original = _boolean_grid(defined, "Defined facies", labels.shape)
    heights = _height_grid(y_m, labels.shape)
    porosity = _property_table(porosity_by_facies, "Porosity", positive=True)
    water = _property_table(swi_by_facies, "Water saturation", positive=False)
    phi = np.full(labels.shape, np.nan, np.float64)
    swi = np.full(labels.shape, np.nan, np.float64)
    for table, target in ((porosity, phi), (water, swi)):
        for label, number in table.items():
            target[original & rig & (labels == label)] = number
    donors = original & rig & np.isfinite(phi)
    previous, following = _column_neighbours(donors)
    columns = np.arange(labels.shape[1])[None, :]
    previous_distance = np.where(
        previous >= 0, np.abs(heights[np.maximum(previous, 0), columns] - heights), np.inf
    )
    following_distance = np.where(
        following < labels.shape[0],
        np.abs(heights[np.minimum(following, labels.shape[0] - 1), columns] - heights),
        np.inf,
    )
    chosen = np.where(previous_distance <= following_distance, previous, following)
    available = (chosen >= 0) & (chosen < labels.shape[0])
    filled = rig & ~original & available
    result = labels.copy()
    donor = np.clip(chosen, 0, labels.shape[0] - 1)
    result[filled] = labels[donor, columns][filled]
    phi[filled] = phi[donor, columns][filled]
    swi[filled] = swi[donor, columns][filled]
    known = donors | filled
    donor_rows = np.where(known, chosen, -1).astype(np.int32)
    distance = np.where(
        known, np.abs(heights[donor, columns] - heights), np.nan
    )
    return ColumnGeometry(
        facies=result,
        porosity=phi,
        swi=swi,
        defined=original.copy(),
        geometry_filled=filled,
        geometry_unknown=rig & ~known,
        donor_rows=donor_rows,
        donor_distance_m=distance,
    )


def column_co2_donors(visible_co2, y_m, *, max_distance_m):
    """Find the nearest strictly higher observed CO2 pixel in the same column.

    ``visible_co2`` is the caller's admitted donor mask, ``y_m`` is the
    physical-height grid, and the required positive ``max_distance_m`` is
    inclusive. Return row indices (-1 if unavailable), availability, and
    distances in metres (NaN if unavailable). No row-based distance, hidden
    tolerance, horizontal borrowing, or chaining of estimated pixels is used.
    """
    donors = _boolean_grid(visible_co2, "Visible CO2")
    heights = _height_grid(y_m, donors.shape)
    if isinstance(max_distance_m, (bool, np.bool_)) or not isinstance(
        max_distance_m, (int, float, np.integer, np.floating)
    ):
        raise ValueError("Maximum column distance must be an explicit positive finite number")
    limit = float(max_distance_m)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Maximum column distance must be an explicit positive finite number")
    previous, _ = _column_neighbours(donors)
    previous = np.vstack(
        (np.full((1, donors.shape[1]), -1, np.int32), previous[:-1])
    )
    columns = np.arange(donors.shape[1])[None, :]
    distance = heights[np.maximum(previous, 0), columns] - heights
    available = (previous >= 0) & (distance > 0) & (distance <= limit)
    return (
        np.where(available, previous, -1).astype(np.int32),
        available,
        np.where(available, distance, np.nan),
    )


def hidden_boundary_adjacency(phase_labels, visible, x_m, y_m, *,
                              inner_polygon_m, fixture_polygons_m):
    """Measure N16 contact with the union of the three hidden polygons.

    Source pixel centres and measured polygon vertices are in metres. Contact
    is the part of the hidden union boundary within 20 mm of a visible label
    1/2 centre. Segment/circle intersections are solved directly; no buffer
    approximation, raster dilation, distance tolerance or hidden phase is used.
    Overlapping fixtures contribute only their union's external boundary.
    This function does not compute or admit a hidden mass capacity.
    """
    from scipy.spatial import cKDTree
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    labels = np.asarray(phase_labels)
    if (labels.dtype != np.uint8 or labels.ndim != 2 or not all(labels.shape)
            or not np.isin(labels, (0, 1, 2, 255)).all()):
        raise ValueError("Original uint8 phase labels are required on the native grid")
    observed = _boolean_grid(visible, "Visible pixels", labels.shape)
    coordinates = []
    for name, value in (("Physical x", x_m), ("Physical y", y_m)):
        array = np.asarray(value)
        if (array.shape != labels.shape or array.dtype.kind not in "fiu"
                or not np.isfinite(array).all()):
            raise ValueError(name + " must be finite on the native grid")
        coordinates.append(np.asarray(array, np.float64))
    if not isinstance(fixture_polygons_m, Mapping) or len(fixture_polygons_m) != 3:
        raise ValueError("N16 requires the three explicitly bound hidden polygons")

    def polygon(points):
        points = np.asarray(points, np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all()):
            raise ValueError("Hidden geometry needs finite measured metre vertices")
        result = Polygon(points)
        if not result.is_valid or result.is_empty or result.area <= 0:
            raise ValueError("Hidden polygons must be valid without geometry repair")
        return result

    inside = polygon(inner_polygon_m)
    hidden = unary_union([polygon(points) for points in fixture_polygons_m.values()]).intersection(inside)
    selected = observed & np.isin(labels, (1, 2))
    points = np.column_stack([axis[selected] for axis in coordinates])
    radius = .020
    tree = cKDTree(points) if len(points) else None
    boundary = hidden.boundary
    lines = ([] if hidden.is_empty else
             list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary])
    touched, length = False, 0.0
    for line in lines:
        vertices = np.asarray(line.coords, np.float64)
        for start, end in zip(vertices[:-1], vertices[1:]):
            vector = end - start
            segment_length = float(np.linalg.norm(vector))
            if tree is None or segment_length == 0:
                continue
            direction = vector / segment_length
            indices = tree.query_ball_point((start + end) / 2,
                                           segment_length / 2 + radius)
            if not indices:
                continue
            relative = points[indices] - start
            along = relative @ direction
            perpendicular = relative - along[:, None] * direction
            distance_squared = np.sum(perpendicular * perpendicular, axis=1)
            nearby = distance_squared <= radius * radius
            half_width = np.sqrt(np.maximum(radius * radius - distance_squared[nearby], 0))
            low = np.maximum(along[nearby] - half_width, 0)
            high = np.minimum(along[nearby] + half_width, segment_length)
            intervals = sorted((float(a), float(b)) for a, b in zip(low, high) if a <= b)
            if not intervals:
                continue
            touched = True
            first, last = intervals[0]
            for low, high in intervals[1:]:
                if low > last:
                    length += last - first
                    first, last = low, high
                else:
                    last = max(last, high)
            length += last - first
    return dict(co2_adjacent=touched, touching_length_mm=length * 1000,
                distance_limit_mm=20.0, visible_co2_pixels=int(selected.sum()),
                boundary_definition="union of three hidden polygons clipped to inner rig",
                distance_definition="visible label 1/2 pixel centre to physical boundary",
                touching_length_definition="union boundary within 20 mm of visible label 1/2 centres")


def hidden_physical_geometry(facies, defined, inner_mask, fixture_masks, y_m, *,
                             porosity_by_facies, swi_by_facies):
    """Continue Q52 properties through hidden sand by the original column rule.

    Hidden pixels never supply categorical donors. The analysis domain still
    excludes the fixtures; the separate full physical result includes them.
    Undefined columns remain undefined. Measured target depth and original
    Geometry pore weights must still be supplied by the production caller.
    """
    inside = _boolean_grid(inner_mask, "Rig interior")
    analysis = rig_pixel_domain(inside, fixture_masks)
    original = _boolean_grid(defined, "Defined facies", inside.shape)
    columns = extend_column_geometry(
        facies, original & analysis, inside, y_m,
        porosity_by_facies=porosity_by_facies, swi_by_facies=swi_by_facies,
    )
    return dict(physical_mask=inside.copy(), analysis_mask=analysis,
                hidden_mask=inside & ~analysis, geometry=columns,
                physical_geometry_admitted=False)


def below_photo_column_support(below_photo_mask, visible, phase_labels, y_m):
    """Classify the explicitly bound AC17-38 8 mm strip as estimate or zero.

    This is the original inclusive 20 mm, strictly higher same-column donor
    search. Only visible label 1/2 pixels donate; neither hidden pixels nor
    estimates are chained. Labels and mass fields are not rewritten here.
    The caller binds the actual below-photo strip, never a below-floor mask.
    """
    strip = _boolean_grid(below_photo_mask, "Bound below-photo strip")
    observed = _boolean_grid(visible, "Visible pixels", strip.shape)
    labels = np.asarray(phase_labels)
    if (labels.dtype != np.uint8 or labels.shape != strip.shape
            or not np.isin(labels, (0, 1, 2, 255)).all()):
        raise ValueError("Original phase labels must share the bound below-photo grid")
    donors = observed & ~strip & np.isin(labels, (1, 2))
    rows, available, distances = column_co2_donors(donors, y_m, max_distance_m=.020)
    estimated = strip & available
    return dict(estimated_mask=estimated, zero_mask=strip & ~available,
                unknown_mask=np.zeros(strip.shape, bool),
                donor_rows=np.where(estimated, rows, -1).astype(np.int32),
                donor_distance_m=np.where(estimated, distances, np.nan),
                phase_labels_modified=False)


def hidden_boundary_observation(phase_labels, visible, x_m, y_m, *,
                                inner_polygon_m, fixture_polygons_m):
    """N16 CO2 contact plus Q85 report-only gas contact, with no branch switch."""
    keywords = dict(inner_polygon_m=inner_polygon_m, fixture_polygons_m=fixture_polygons_m)
    co2 = hidden_boundary_adjacency(phase_labels, visible, x_m, y_m, **keywords)
    gas = hidden_boundary_adjacency(phase_labels,
        np.asarray(visible) & (np.asarray(phase_labels) == 2), x_m, y_m, **keywords)
    return dict(co2, gas_label2_within_20mm=gas["co2_adjacent"],
                gas_touching_length_mm=gas["touching_length_mm"],
                gas_report_only=True, capacity_branch_automatically_switched=False)
