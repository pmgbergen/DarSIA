"""Physical area validation and categorical component measurements.

Frozen reference areas and gas cutoffs are always provided by the caller.
"""

from __future__ import annotations

import hashlib

from hashlib import sha256

import cv2
import numpy as np


def array_identity(array):
    """Hash exact C-order values together with dtype and little-endian shape."""
    array = np.ascontiguousarray(array)
    return sha256(
        str(array.dtype).encode()
        + np.asarray(array.shape, "<i8").tobytes()
        + array.tobytes()
    ).hexdigest()


def array_pair_identity(actual, expected):
    """Bitwise identity, including invalid values and their stored encodings."""
    first, second = np.asarray(actual), np.asarray(expected)
    same_grid = first.shape == second.shape and first.dtype == second.dtype
    if same_grid:
        byte_view = first.view(np.dtype((np.void, first.dtype.itemsize)))
        other_view = second.view(np.dtype((np.void, second.dtype.itemsize)))
        different = byte_view != other_view
        changed = int(np.count_nonzero(different))
        locations = np.argwhere(different)
        first_location = None if changed == 0 else locations[0].tolist()
    else:
        changed, first_location = None, None
    return dict(
        bit_identical=bool(same_grid and changed == 0),
        changed_elements=changed,
        first_different_index=first_location,
        actual_shape=list(first.shape),
        expected_shape=list(second.shape),
        actual_dtype=str(first.dtype),
        expected_dtype=str(second.dtype),
        actual_array_sha256=hashlib.sha256(
            np.ascontiguousarray(first).tobytes()
        ).hexdigest(),
        expected_array_sha256=hashlib.sha256(
            np.ascontiguousarray(second).tobytes()
        ).hexdigest(),
    )


def _shape(shape):
    result = tuple(int(v) for v in shape)
    if len(result) != 2 or min(result) <= 0 or tuple(shape) != result:
        raise ValueError("Pixel area requires a positive integer grid shape")
    return result


def validate_pixel_area(value, shape_hw):
    """Return a positive finite scalar or matching floating area grid."""
    shape = _shape(shape_hw)
    array = np.asarray(value)
    if array.ndim == 0:
        number = float(array)
        if not np.isfinite(number) or number <= 0:
            raise ValueError("Pixel area must be positive and finite")
        return number
    if (
        array.shape != shape
        or not np.issubdtype(array.dtype, np.floating)
        or not np.isfinite(array).all()
        or np.any(array <= 0)
    ):
        raise ValueError(
            "Pixel area grid must be positive finite floating values on the exact grid"
        )
    return array


def sum_pixel_area(mask, pixel_area, *, unit="mm2"):
    """Sum supported physical areas in float64 and convert the requested unit."""
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 2:
        raise ValueError("Area support must be a two-dimensional bool mask")
    area = validate_pixel_area(pixel_area, mask.shape)
    total = (
        float(np.count_nonzero(mask)) * area
        if np.isscalar(area)
        else float(np.sum(area, where=mask, dtype=np.float64))
    )
    factors = {"mm2": 1.0, "cm2": 0.01, "m2": 1e-6}
    if unit not in factors:
        raise ValueError("Area output unit must be mm2, cm2 or m2")
    return total * factors[unit]


def filter_components_by_area(mask, pixel_area_mm2, *, minimum_area_mm2, connectivity):
    """Filter components by the sum of physical pixel areas, never pixel count."""
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 2 or connectivity != 4:
        raise ValueError("Gas minimum requires a bool grid and four-connectivity")
    area = validate_pixel_area(pixel_area_mm2, mask.shape)
    minimum = float(minimum_area_mm2)
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("Gas minimum area must be finite and nonnegative")
    if minimum == 0:
        return mask.copy(), dict(
            withdrawn_components=0, withdrawn_pixels=0, minimum_area_mm2=minimum
        )
    count, components = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=connectivity
    )
    weights = (
        np.bincount(components.ravel(), minlength=count) * area
        if np.isscalar(area)
        else np.bincount(components.ravel(), weights=area.ravel(), minlength=count)
    )
    small = weights < minimum
    small[0] = False
    result = mask & ~small[components]
    return result, dict(
        withdrawn_components=int(small.sum()),
        withdrawn_pixels=int(np.count_nonzero(mask & ~result)),
        minimum_area_mm2=minimum,
        connectivity=connectivity,
        component_areas_mm2=weights[1:].tolist(),
    )


def area_filtered_gas(mask, area_mm2_array, *, minimum_area_mm2, connectivity):
    """Production inference API for a native physically weighted gas cutoff."""
    return filter_components_by_area(
        mask,
        area_mm2_array,
        minimum_area_mm2=minimum_area_mm2,
        connectivity=connectivity,
    )


def categorical_replay(
    parent_labels, replay_labels, pixel_area_mm2, *, identity, parameters
):
    """Compare phase areas against a frozen categorical acquisition.

    A zero parent phase permits zero changed pixels only. Areas retain the
    original selected-value float64 reduction and division order. Explicit
    physical acceptance limits are diagnostic data; this function neither
    selects evidence nor alters labels, source support or pixel areas.
    """
    a, b, area = map(np.asarray, (parent_labels, replay_labels, pixel_area_mm2))
    if (
        a.dtype != np.uint8
        or b.dtype != np.uint8
        or a.ndim != 2
        or a.shape != b.shape
        or area.shape != a.shape
        or not np.isfinite(area).all()
        or np.any(area < 0)
        or not np.isin(a, parameters["allowed_labels"]).all()
        or not np.isin(b, parameters["allowed_labels"]).all()
    ):
        raise ValueError("Replay QA requires exact categorical and measured-area grids")
    result = dict(identity, categorical_changed_pixels=int((a != b).sum()))
    passes = []
    for name, before, after, limit in (
        ("co2", np.isin(a, (1, 2)), np.isin(b, (1, 2)), parameters["co2_replay_limit"]),
        ("gas", a == 2, b == 2, parameters["gas_replay_limit"]),
    ):
        changed = before != after
        measure = lambda mask: float(
            area[mask].sum(dtype=np.float64)
            / parameters["square_millimetres_per_square_centimetre"]
        )
        old_area, changed_area = measure(before), measure(changed)
        relative = (
            changed_area / old_area
            if old_area > 0
            else (0.0 if not changed.any() else None)
        )
        passed = not changed.any() if old_area == 0 else relative <= limit
        result[name] = dict(
            parent_pixels=int(before.sum()),
            replay_pixels=int(after.sum()),
            changed_pixels=int(changed.sum()),
            added_pixels=int((after & ~before).sum()),
            removed_pixels=int((before & ~after).sum()),
            parent_cm2=old_area,
            replay_cm2=measure(after),
            changed_cm2=changed_area,
            added_cm2=measure(after & ~before),
            removed_cm2=measure(before & ~after),
            changed_area_fraction=relative,
            zero_parent_area=old_area == 0,
            acceptance_limit=limit,
            pass_gate=passed,
        )
        passes.append(passed)
    result.update(
        all_pass=all(passes),
        bit_identical=bool(np.array_equal(a, b)),
        support_changed_pixels=int(((a == 255) != (b == 255)).sum()),
    )
    return result


def _categorical_coarse_grids(parent, current, area, *, parameters):
    """Retain categorical grid and nonnegative measured-area admission."""
    a, b, area = (np.asarray(parent), np.asarray(current), np.asarray(area))
    if (
        a.ndim != 2
        or a.dtype != np.uint8
        or b.dtype != np.uint8
        or a.shape != b.shape
        or area.shape != a.shape
        or not np.isfinite(area).all()
        or np.any(area < 0)
        or not np.isin(a, parameters["allowed_labels"]).all()
        or not np.isin(b, parameters["allowed_labels"]).all()
    ):
        raise ValueError("Coarse QA requires exact categorical and measured-area grids")
    return a, b, area


def categorical_coarse_gate(parent, current, pixel_area_mm2, *, parameters):
    """Measure coarse CO2 and gas changes on an unchanged physical area grid.

    Selected pixel areas are summed in float64 and divided into cm2 in the
    original order. Each phase passes when its symmetric-difference area is
    at most the larger of the relative parent-area limit and absolute cm2
    floor. Zero parent area retains the same explicit absolute-floor rule.
    Parameters supply the original limits, categories and receipt descriptors;
    this diagnostic neither changes labels nor selects evidence.
    """
    a, b, area = _categorical_coarse_grids(
        parent, current, pixel_area_mm2, parameters=parameters
    )
    result = dict(
        contract=parameters["contract"],
        categorical_changed_pixels=int(np.count_nonzero(a != b)),
        support_changed_pixels=int(np.count_nonzero((a == 255) != (b == 255))),
        acceptance_backend=parameters["acceptance_backend"],
    )
    measure = lambda mask: float(
        np.sum(area[mask], dtype=np.float64) / parameters["square_mm_per_cm2"]
    )
    for phase, fraction, old, new in (
        (
            "co2",
            parameters["co2_relative_limit"],
            np.isin(a, (1, 2)),
            np.isin(b, (1, 2)),
        ),
        ("gas", parameters["gas_relative_limit"], a == 2, b == 2),
    ):
        changed, added, removed = old ^ new, new & ~old, old & ~new
        parent_area, xor_area = measure(old), measure(changed)
        limit = max(fraction * parent_area, parameters["absolute_floor_cm2"])
        result[phase] = dict(
            parent_cm2=parent_area,
            current_cm2=measure(new),
            xor_cm2=xor_area,
            relative_xor=None if parent_area == 0 else xor_area / parent_area,
            added_cm2=measure(added),
            removed_cm2=measure(removed),
            changed_pixels=int(changed.sum()),
            added_pixels=int(added.sum()),
            removed_pixels=int(removed.sum()),
            relative_limit=fraction,
            absolute_floor_cm2=parameters["absolute_floor_cm2"],
            acceptance_limit_cm2=limit,
            pass_gate=bool(xor_area <= limit),
        )
    result["all_pass"] = result["co2"]["pass_gate"] and result["gas"]["pass_gate"]
    return result
