"""Registered scientific frame geometry, independent of campaign execution."""

from __future__ import annotations
from dataclasses import dataclass, replace
from copy import deepcopy
import math
import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PreparedInputs:
    """Registered optical/physical inputs before the measured correction."""

    run: str
    frame_index: int
    elapsed_h: float
    current_rgb: np.ndarray
    baseline_rgb: np.ndarray
    clear_water_rgb: np.ndarray
    clear_water_supported_mask: np.ndarray
    phase_labels: np.ndarray
    facies: np.ndarray
    domain: np.ndarray
    supported_mask: np.ndarray
    mass_eligible_mask: np.ndarray
    bottom_strip_mask: np.ndarray
    below_photo_filled_mask: np.ndarray
    unknown_physical_mask: np.ndarray
    unquantified_facies_mask: np.ndarray
    depth_m: np.ndarray
    porosity: np.ndarray
    pixel_area_m2: np.ndarray
    recipe: Mapping
    atmosphere: Mapping
    nominal_injected_mass_g: float
    dimensions_m: tuple[float, float]
    provenance: Mapping


def _mask(value, shape, name):
    """Require an exact boolean mask on the supplied native grid."""
    value = np.asarray(value)
    if value.dtype != bool or value.shape != shape:
        raise ValueError(name + " must be a boolean mask on the native grid")
    return value


def _points(value, name):
    """Require finite coordinate pairs on the original measured geometry."""
    value = np.asarray(value, np.float64)
    if (
        value.ndim != 2
        or value.shape[1] != 2
        or not len(value)
        or not np.isfinite(value).all()
    ):
        raise ValueError(name + " requires finite Nx2 pixel coordinates")
    return value


def fill_to_edges(facies, rig_domain, *, sand_labels):
    """Fill facies0 from ORIGINAL sand: nearest same row, column, then global.

    Equal row distances choose left; equal column distances choose above. The
    global Euclidean nearest-pixel tie follows SciPy's deterministic EDT. Known
    facies, including D/E/G, and all pixels outside the rig remain unchanged.
    """
    labels = np.asarray(facies)
    if (
        labels.ndim != 2
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any(labels < 0)
    ):
        raise ValueError("Facies must be nonnegative native categorical values")
    rig = _mask(rig_domain, labels.shape, "Rig domain")
    allowed = tuple(map(int, sand_labels))
    if not allowed or any(label <= 0 or label == 255 for label in allowed):
        raise ValueError("Measured positive sand IDs are required")
    sand = rig & np.isin(labels, allowed)
    pending = rig & (labels == 0)
    result, method = labels.copy(), np.zeros(labels.shape, np.uint8)
    if pending.any() and not sand.any():
        raise ValueError("No measured sand pixel exists for the prescribed fill")
    for row in np.flatnonzero(pending.any(axis=1)):
        donors, targets = np.flatnonzero(sand[row]), np.flatnonzero(pending[row])
        if not len(donors):
            continue
        right = np.searchsorted(donors, targets)
        a, b = (
            donors[np.clip(right - 1, 0, len(donors) - 1)],
            donors[np.clip(right, 0, len(donors) - 1)],
        )
        selected = np.where(np.abs(targets - a) <= np.abs(b - targets), a, b)
        result[row, targets], method[row, targets], pending[row, targets] = (
            labels[row, selected],
            1,
            False,
        )
    for column in np.flatnonzero(pending.any(axis=0)):
        donors, targets = np.flatnonzero(sand[:, column]), np.flatnonzero(
            pending[:, column]
        )
        if not len(donors):
            continue
        below = np.searchsorted(donors, targets)
        a, b = (
            donors[np.clip(below - 1, 0, len(donors) - 1)],
            donors[np.clip(below, 0, len(donors) - 1)],
        )
        selected = np.where(np.abs(targets - a) <= np.abs(b - targets), a, b)
        result[targets, column], method[targets, column], pending[targets, column] = (
            labels[selected, column],
            2,
            False,
        )
    if pending.any():
        yy, xx = distance_transform_edt(
            ~sand, return_distances=False, return_indices=True
        )
        result[pending], method[pending] = labels[yy[pending], xx[pending]], 3
    return result, method


def measured_depth(x_m, y_m, depth_points, *, chunk_pixels=250000):
    """Original linear measured depth and nearest fallback; no facies inference.

    This numerical operation supplies no domain, porosity or mass eligibility.
    """
    x, y = np.asarray(x_m, np.float64), np.asarray(y_m, np.float64)
    points = np.asarray(depth_points, np.float64)
    if (
        x.ndim != 2
        or x.shape != y.shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 3
        or not np.isfinite(points).all()
        or np.any(points[:, 2] <= 0)
        or type(chunk_pixels) is not int
        or chunk_pixels < 1
    ):
        raise ValueError(
            "Registered metre coordinates and positive measured depth triples are required"
        )
    linear, nearest = LinearNDInterpolator(points[:, :2], points[:, 2]), cKDTree(
        points[:, :2]
    )
    depth, outside = np.empty(x.size, np.float64), np.zeros(x.size, bool)
    for start in range(0, x.size, chunk_pixels):
        stop = min(x.size, start + chunk_pixels)
        query = np.column_stack((x.ravel()[start:stop], y.ravel()[start:stop]))
        values = np.asarray(linear(query), np.float64)
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = points[nearest.query(query[bad])[1], 2]
        depth[start:stop], outside[start:stop] = values, bad
    return dict(depth_m=depth.reshape(x.shape),
                depth_nearest_outside_hull=outside.reshape(x.shape))


def measured_geometry(
    x_m,
    y_m,
    facies,
    depth_points,
    *,
    porosity_by_facies,
    swi_by_facies,
    chunk_pixels=250000,
):
    """Measured phi/Swi and linear depth inside the hull, nearest outside."""
    x, y = np.asarray(x_m, np.float64), np.asarray(y_m, np.float64)
    labels, points = np.asarray(facies), np.asarray(depth_points, np.float64)
    if (
        x.ndim != 2
        or x.shape != y.shape
        or labels.shape != x.shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 3
        or not np.isfinite(points).all()
        or np.any(points[:, 2] <= 0)
        or type(chunk_pixels) is not int
        or chunk_pixels < 1
    ):
        raise ValueError(
            "Registered metre coordinates and positive measured depth triples are required"
        )
    measured = measured_depth(x, y, points, chunk_pixels=chunk_pixels)
    depth = measured["depth_m"]
    phi, swi = np.full(x.shape, np.nan, np.float64), np.full(
        x.shape, np.nan, np.float64
    )
    for table, output, name in (
        (porosity_by_facies, phi, "porosity"),
        (swi_by_facies, swi, "Swi"),
    ):
        for label, value in table.items():
            if int(label) == 0:
                continue
            value = float(value)
            if (
                not np.isfinite(value)
                or not 0 <= value <= 1
                or name == "porosity"
                and value == 0
            ):
                raise ValueError("Measured " + name + " is outside its physical range")
            output[labels == int(label)] = value
    known = np.isfinite(phi) & (phi > 0) & np.isfinite(depth) & (depth > 0)
    return dict(
        depth_m=depth,
        porosity=phi,
        swi=swi,
        known=known,
        depth_nearest_outside_hull=measured["depth_nearest_outside_hull"],
    )


def correct_prepared(
    prepared,
    *,
    x_m,
    y_m,
    floor_curve_canvas_y,
    depth_points,
    porosity_by_facies,
    swi_by_facies,
    optical_support,
    transported_facies=None,
    source_binding=None,
    unquantified_facies=(),
    report_contract="darsia-n1-corrected-geometry-v1",
):
    """Round4 physical frame input, before ANY native optics or fit summary."""
    shape = prepared.domain.shape
    rig = _mask(prepared.domain, shape, "Original rig domain")
    optical = _mask(optical_support, shape, "Original light/colour/clearwater support")
    floor = np.asarray(floor_curve_canvas_y, np.float64)
    if floor.shape != (shape[1],) or not np.isfinite(floor).all():
        raise ValueError("The original transported sand-floor line is required")
    below = rig & (np.arange(shape[0])[:, None] > floor[None, :])
    if np.any(below & np.isin(prepared.phase_labels, (1, 2))):
        raise ValueError(
            "Positive locked phase labels exist below the bound floor; cannot discard them"
        )
    domain = rig & ~below
    labels = prepared.facies if transported_facies is None else transported_facies
    corrected, fill = fill_to_edges(labels, domain, sand_labels=porosity_by_facies)
    physical = measured_geometry(
        x_m,
        y_m,
        corrected,
        depth_points,
        porosity_by_facies=porosity_by_facies,
        swi_by_facies=swi_by_facies,
    )
    known = domain & physical["known"]
    supported = known & optical
    eligible = supported.copy()
    # Former bottom additions above the floor are admitted only with the SAME
    # existing optical support. Remaining old excluded classes keep their names.
    bottom = prepared.bottom_strip_mask & domain & ~eligible
    below_photo = prepared.below_photo_filled_mask & domain & ~eligible & ~bottom
    old_main = prepared.mass_eligible_mask & prepared.supported_mask
    # Report the actual quantifiable main partition; the original D/E/G masks
    # and numerical eligibility fields below retain their treatment.
    quantifiable = ~np.isin(corrected, unquantified_facies) | (
        prepared.phase_labels == 2
    ) & np.isfinite(physical["swi"])
    current_main = eligible & quantifiable
    previous_finite_swi = np.isin(
        prepared.facies,
        [int(label) for label, value in swi_by_facies.items() if np.isfinite(value)],
    )
    previous_main = old_main & (
        ~prepared.unquantified_facies_mask
        | (prepared.phase_labels == 2)
        & ~prepared.unknown_physical_mask
        & previous_finite_swi
    )
    added = current_main & ~previous_main
    filled = fill != 0
    blocked = domain & filled & ~optical
    area = np.broadcast_to(np.asarray(prepared.pixel_area_m2, np.float64), shape)
    if not np.isfinite(area[rig]).all() or np.any(area[rig] <= 0):
        raise ValueError("Original measured per-frame pixel area is required")
    co2 = np.isin(prepared.phase_labels, (1, 2))

    def areas(mask):
        """Integrate original pixel areas separately from CO2 support."""
        return dict(
            area_cm2=float(np.sum(area[mask])) * 1.0e4,
            co2_area_cm2=float(np.sum(area[mask & co2])) * 1.0e4,
        )

    report = dict(
        contract=report_contract,
        source_binding=dict(source_binding or {}),
        run=prepared.run,
        frame_index=prepared.frame_index,
        elapsed_h=prepared.elapsed_h,
        added_to_M=areas(added),
        filled_without_optical_support=areas(blocked),
        below_sand_floor=areas(below),
        removed_from_M=areas(previous_main & ~current_main),
        filled_pixels_by_method={
            name: int(np.count_nonzero(fill == code))
            for name, code in (
                ("same_row", 1),
                ("same_column", 2),
                ("global_nearest", 3),
            )
        },
        depth_nearest_outside_hull_pixels=int(
            np.count_nonzero(domain & physical["depth_nearest_outside_hull"])
        ),
        phase_labels_changed=False,
        RAW_changed=False,
        D_E_G_treatment_changed=False,
    )
    provenance = dict(
        prepared.provenance,
        original_physical_provenance=dict(prepared.provenance),
        round4_sandmap=report,
        undefined_facies_not_borrowed=False,
        facies_zero_fill_rule="same row, same column, global nearest sand",
        physical_unknown_pixels=int(np.count_nonzero(domain & ~known)),
        supported_pixels=int(np.count_nonzero(supported)),
        mass_eligible_pixels=int(np.count_nonzero(eligible)),
        quantifiable_main_pixels=int(np.count_nonzero(current_main)),
        bottom_mass_allowed_above_sand_floor_with_existing_optical_support=True,
        optical_unresolved_pixels=int(np.count_nonzero(domain & ~optical)),
    )
    value = replace(
        prepared,
        facies=corrected,
        domain=domain,
        supported_mask=supported,
        mass_eligible_mask=eligible,
        depth_m=physical["depth_m"],
        porosity=physical["porosity"],
        unknown_physical_mask=domain & ~known,
        bottom_strip_mask=bottom,
        below_photo_filled_mask=below_photo,
        unquantified_facies_mask=domain & np.isin(corrected, unquantified_facies),
        provenance=provenance,
    )
    return value, report, physical["swi"]


def prepare_build_physical(
    prepared, *, x_m, y_m, floor_curve_canvas_y, depth_points, facies,
    porosity, swi, analysis_domain, reporting_domain, material_supported_mask,
    excluded_masks, source_binding,
):
    """Prepare explicitly bound Build material without filling any category.

    The caller authenticates the material/role masks, recipes and coordinates.
    This array operation does not infer roles or approve a source. Full report
    membership is retained; only active analysis is cut by the original floor.
    ``prepared.supported_mask`` is the original optical-support mask here.
    """
    shape = np.asarray(reporting_domain).shape
    if len(shape) != 2 or not all(shape):
        raise ValueError("Build reporting domain must be a nonempty native grid")
    reporting = _mask(reporting_domain, shape, "Build reporting domain")
    analysis = _mask(analysis_domain, shape, "Build analysis domain")
    material = _mask(material_supported_mask, shape, "Build material support")
    optical = _mask(prepared.supported_mask, shape, "Original optical support")
    if (not np.array_equal(_mask(prepared.domain, shape, "Prepared analysis domain"), analysis)
            or np.any(analysis & ~reporting) or np.any(material & ~reporting)):
        raise ValueError("Build analysis/material support differs from its full reporting domain")
    labels, phi, water = np.asarray(facies), np.asarray(porosity), np.asarray(swi)
    if (labels.shape != shape or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or phi.shape != shape or water.shape != shape
            or phi.dtype.kind not in "fiu" or water.dtype.kind not in "fiu"):
        raise ValueError("Explicit unchanged native Build facies, porosity and Swi arrays are required")
    if (np.any(~np.isfinite(phi[material])) or np.any(phi[material] < 0)
            or np.any(phi[material] > 1) or np.isinf(water).any()
            or np.any(np.isfinite(water) & ((water < 0) | (water > 1)))):
        raise ValueError("Bound Build material properties are outside their physical range")
    if not isinstance(excluded_masks, dict) or not isinstance(source_binding, dict):
        raise ValueError("Explicit named Build exclusions and source binding are required")
    exclusions, covered = {}, np.zeros(shape, bool)
    for name, value in excluded_masks.items():
        if not isinstance(name, str) or not name or name == "below_sand_floor":
            raise ValueError("Build exclusions need distinct names; below_sand_floor is derived")
        mask = _mask(value, shape, "Build exclusion " + name)
        if np.any(mask & (~reporting | analysis | covered)):
            raise ValueError("Build exclusions must be disjoint, in the full report and outside analysis")
        exclusions[name] = mask.copy()
        covered |= mask
    # Zero porosity is an explicitly excluded material value, not missing sand.
    if np.any(material & (phi == 0) & ~covered):
        raise ValueError("Known zero-porosity Build material requires an explicit exclusion")
    phase = np.asarray(prepared.phase_labels)
    if phase.shape != shape or phase.dtype != np.uint8 or not np.isin(phase, (0, 1, 2, 255)).all():
        raise ValueError("Original native phase labels are required for the Build floor check")
    floor = np.asarray(floor_curve_canvas_y, np.float64)
    if floor.shape != (shape[1],) or not np.isfinite(floor).all():
        raise ValueError("The original transported sand-floor line is required")
    below = analysis & (np.arange(shape[0])[:, None] > floor[None, :])
    if np.any(below & np.isin(phase, (1, 2))):
        raise ValueError("Positive locked phase labels exist below the bound floor; cannot discard them")
    active = analysis & ~below
    if below.any():
        exclusions["below_sand_floor"] = below.copy()
    measured = measured_depth(x_m, y_m, depth_points)
    if measured["depth_m"].shape != shape:
        raise ValueError("Measured Build coordinates differ from its native material grid")
    known = active & material & np.isfinite(phi) & (phi > 0)
    known &= np.isfinite(measured["depth_m"]) & (measured["depth_m"] > 0)
    supported = known & optical
    area = np.broadcast_to(np.asarray(prepared.pixel_area_m2, np.float64), shape)
    if not np.isfinite(area[reporting]).all() or np.any(area[reporting] <= 0):
        raise ValueError("Original measured per-frame pixel area is required")
    unquantified = _mask(prepared.unquantified_facies_mask, shape, "Explicit unquantified facies")
    report = dict(
        contract="darsia-build-role-physical-geometry-v1", run=prepared.run,
        frame_index=prepared.frame_index, elapsed_h=prepared.elapsed_h,
        source_binding=deepcopy(source_binding),
        reporting_pixels=int(reporting.sum()), analysis_pixels=int(active.sum()),
        material_supported_pixels=int(material.sum()),
        known_zero_porosity_pixels=int(np.count_nonzero(material & (phi == 0))),
        missing_material_support_pixels=int(np.count_nonzero(reporting & ~material)),
        unknown_physical_pixels=int(np.count_nonzero(reporting & ~known)),
        excluded_pixels={name: int(mask.sum()) for name, mask in exclusions.items()},
        below_sand_floor=dict(pixels=int(below.sum()),
                             area_cm2=float(np.sum(area[below])) * 1.0e4),
        depth_nearest_outside_hull_pixels=int(np.count_nonzero(
            reporting & measured["depth_nearest_outside_hull"])),
        facies_filled=False, material_values_changed=False,
        reporting_domain_changed=False, phase_labels_changed=False, RAW_changed=False,
    )
    provenance = dict(prepared.provenance, build_physical_preparation=deepcopy(report))
    value = replace(prepared, facies=labels.copy(), porosity=phi.copy(),
        depth_m=measured["depth_m"], domain=reporting.copy(),
        supported_mask=supported, mass_eligible_mask=supported.copy(),
        unknown_physical_mask=reporting & ~known,
        bottom_strip_mask=np.zeros(shape, bool), below_photo_filled_mask=np.zeros(shape, bool),
        unquantified_facies_mask=reporting & unquantified, provenance=provenance)
    fields = dict(x_m=np.asarray(x_m, np.float64).copy(), y_m=np.asarray(y_m, np.float64).copy(),
        facies=value.facies, porosity=value.porosity, swi=water.copy(), depth_m=value.depth_m,
        valid=known.copy(), depth_nearest_outside_hull=measured["depth_nearest_outside_hull"],
        analysis_domain=active.copy(), reporting_domain=reporting.copy(),
        material_supported_mask=material.copy(), excluded_masks=exclusions,
        depth_points=np.asarray(depth_points, np.float64).copy(),
        source_binding=deepcopy(source_binding))
    return value, report, fields


def project_points(matrix, x, y):
    """Project native pixel centres with the exact homogeneous denominator."""
    matrix = np.asarray(matrix, np.float64)
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if not np.isfinite(denominator).all() or np.any(np.abs(denominator) < 1e-12):
        raise ValueError("Source-bound coordinate transform reaches infinity")
    return (
        (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator,
        (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator,
    )


def frame_coordinates(reference_matrix, geometry, revision, *, fields):
    """Retain the original float32 metre coordinates on the native canvas."""
    shape = tuple(map(int, revision["phase_shape_hw"]))
    ox, oy = map(float, revision["native_origin_xy"])
    matrix = np.asarray(reference_matrix, np.float64) @ np.linalg.inv(
        np.asarray(geometry[fields["source_to_native"]], np.float64)
    )
    x = np.arange(shape[1], dtype=np.float64)[None, :] + ox
    y = np.arange(shape[0], dtype=np.float64)[:, None] + oy
    xm, ym = project_points(matrix, x, y)
    return xm.astype(np.float32), ym.astype(np.float32)


def project_curve(H, x, y):
    """Original curve homography arithmetic and 1e-10 infinity bound."""
    H = np.asarray(H, dtype=np.float64)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if H.shape != (3, 3) or np.any(~np.isfinite(den)) or np.any(np.abs(den) < 1e-10):
        raise ValueError("Bottom report metric homography is invalid")
    return (
        (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den,
        (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den,
    )


def transport_curve(
    curve, *, H_source_to_target, source_origin_xy, target_origin_xy, target_width
):
    """Transport only observed curve segments; never bridge missing evidence."""
    good = np.isfinite(curve)
    result = np.full(target_width, np.nan)
    cuts = np.flatnonzero(np.diff(np.r_[False, good, False].astype(np.int8)))
    for a, b in zip(cuts[::2], cuts[1::2]):
        xs = np.arange(a, b, dtype=float) + source_origin_xy[0]
        tx, ty = project_curve(H_source_to_target, xs, curve[a:b] + source_origin_xy[1])
        tx, ty = tx - target_origin_xy[0], ty - target_origin_xy[1]
        order = np.argsort(tx)
        tx, ty = tx[order], ty[order]
        if len(tx) > 1 and np.all(np.diff(tx) > 0):
            low, high = max(0, math.ceil(tx[0])), min(
                target_width - 1, math.floor(tx[-1])
            )
            result[low : high + 1] = np.interp(np.arange(low, high + 1), tx, ty)
        elif len(tx) == 1 and 0 <= round(tx[0]) < target_width:
            result[round(tx[0])] = ty[0]
    return result


def _carried_domain(row, current_matrix, shape, reader, *, fields):
    """Carry only historical domain membership with nearest-neighbour geometry."""
    if "membership" in row:
        descriptor = row["membership"]
        membership = reader.array(descriptor)
        if membership.dtype != bool:
            raise ValueError("Explicit physical membership must be a boolean mask")
    else:
        descriptor = row["parent_labels_map"]
        # Only membership is retained. Aqueous/gas categories never enter the new map.
        with Image.open(reader.path(descriptor)) as image:
            membership = np.asarray(image) != 255
    if tuple(membership.shape) != tuple(row["native_shape_hw"]):
        raise ValueError("Original domain and source registration use different grids")
    original_h = np.asarray(row[fields["original_domain_to_native"]], np.float64)
    if (
        original_h.shape != (3, 3)
        or not np.isfinite(original_h).all()
        or abs(np.linalg.det(original_h)) < 1e-12
    ):
        raise ValueError("Original domain source homography is invalid")
    warp = np.asarray(current_matrix) @ np.linalg.inv(original_h)
    carried = cv2.warpPerspective(
        membership.astype(np.uint8),
        warp,
        shape[::-1],
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return carried, dict(
        original_labels=descriptor,
        original_frame_index=int(row["frame_index"]),
        **{fields["carried_homography_provenance"]: original_h.tolist()},
        original_domain_to_current=warp.tolist(),
        prediction_categories_used=False,
    )


def _bottom_side(domain, photo_top, polygon, origin):
    """Established nearest raster-axis boundary partition, unchanged tie order."""
    h, w = domain.shape
    if photo_top.dtype != bool or photo_top.shape != domain.shape:
        raise ValueError("Bottom strip requires the selected image's photographed top")
    polygon = np.asarray(polygon, np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
        raise ValueError("Selected inner-rig polygon is invalid")
    outer = np.zeros(domain.shape, np.uint8)
    cv2.fillPoly(
        outer,
        [np.rint((polygon - np.asarray(origin)) * 256).astype(np.int32)],
        1,
        shift=8,
    )
    outer = outer.astype(bool)
    left, right = outer.argmax(1), w - 1 - outer[:, ::-1].argmax(1)
    floor, surface = h - 1 - outer[::-1].argmax(0), photo_top.argmax(0)
    x = np.arange(w, dtype=np.float32)[None]
    bottom = np.zeros(domain.shape, bool)
    for first in range(0, h, 128):
        last = min(h, first + 128)
        y = np.arange(first, last, dtype=np.float32)[:, None]
        side = np.stack(
            np.broadcast_arrays(
                y - surface[None],
                floor[None] - y,
                x - left[first:last, None],
                right[first:last, None] - x,
            )
        ).argmin(0)
        bottom[first:last] = domain[first:last] & (side == 1)
    return bottom


def compose_raw(descriptor, reader, math):
    """Original scientific assets/shape/bounds/storage/support checks, typed exp."""
    import numpy as np

    shape = tuple(descriptor["phase_shape_hw"])
    centre = descriptor["native_centre"]
    hc, wc = tuple(centre["native_shape_hw"])
    ox, oy = descriptor["native_origin_xy"]
    x, y = -int(ox), -int(oy)
    if min(x, y) < 0 or x + wc > shape[1] or y + hc > shape[0]:
        raise ValueError("Scientific RAW centre lies outside the phase canvas")
    rgb = np.zeros((*shape, 3), math.dtype)
    valid = np.zeros(shape, bool)
    covered = np.zeros(shape, bool)
    log = reader.array(centre["log_linear_rgb"])
    mask = reader.array(centre["valid_mask"])
    if (
        log.dtype != np.float32
        or log.shape != (3, hc, wc)
        or mask.dtype != bool
        or mask.shape != (hc, wc)
    ):
        raise ValueError("Scientific centre encoding is not exact float32 log RGB")

    def place(values, box, support=None):
        """Decode one disjoint log-RGB window without clipping or support extension."""
        a, b, c, d = map(int, box)
        if (
            not 0 <= a < c <= shape[1]
            or not 0 <= b < d <= shape[0]
            or covered[b:d, a:c].any()
            or values.dtype != np.float32
            or values.shape != (3, d - b, c - a)
        ):
            raise ValueError("Scientific windows overlap or use another native grid")
        good = np.isfinite(values).all(0)
        if support is not None:
            good &= support
        for channel in range(3):
            target = rgb[b:d, a:c, channel]
            target[good] = math.decode_log_rgb(values[channel][good])
        valid[b:d, a:c] = good
        covered[b:d, a:c] = True

    place(log, (x, y, x + wc, y + hc), mask)
    for window in descriptor["ring_windows"]:
        place(reader.array(window["log_linear_rgb"]), window["canvas_xyxy"])
    if not np.isfinite(rgb[valid]).all() or np.any(rgb[valid] < 0):
        raise ValueError("Scientific RAW overflows; no clipping is permitted")
    return rgb, valid


def transport_physical_reference(reference, matrix, shape_hw, math, cv2):
    """Original shape/categorical/physical support checks; typed continuous warp."""
    import numpy as np

    h, w = map(int, shape_hw)
    source_domain = np.asarray(reference["domain"])
    if source_domain.ndim != 2 or source_domain.dtype != bool:
        raise ValueError("Physical reference needs exact boolean domain")
    result = {}
    for name in ("facies", "domain", "porosity", "depth_m"):
        values = np.asarray(reference[name])
        if values.shape != source_domain.shape:
            raise ValueError("Physical reference fields use different geometries")
        categorical = name in ("facies", "domain")
        if name == "facies" and not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Facies is categorical")
        if categorical:
            transformed = cv2.warpPerspective(
                values.astype(np.uint8),
                np.asarray(matrix, np.float64),
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            transformed = math.warp_continuous(values, matrix, (h, w), cv2_module=cv2)
        result[name] = transformed.astype(bool) if name == "domain" else transformed
    coverage = math.warp_continuous(source_domain, matrix, (h, w), cv2_module=cv2)
    valid = result["domain"] & (coverage >= 1 - 1e-6)
    valid &= np.isfinite(result["depth_m"]) & (result["depth_m"] > 0)
    valid &= (
        np.isfinite(result["porosity"])
        & (result["porosity"] > 0)
        & (result["porosity"] <= 1)
    )
    result["physical_supported_mask"] = valid
    result["unknown_physical_mask"] = ~valid
    return result
