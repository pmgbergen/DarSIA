"""Pure extensive quantities from explicit native arrays, never domain inference.

This module does not admit a physical domain, create N15 estimates, calculate
N16 adjacency/capacity, or publish a Main/N1 result. Missing channels stay
explicitly missing. Original coarse-v1, label-area and probability-count APIs
are neither imported nor modified.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
import zipfile

import numpy as np

from .storage_resolution import metric_grid_shape

FORMAT = "darsia-extensive-cell-quantities-v3"
RELATIVE_LIMIT = 1e-6
COVERAGE = ("measured", "estimated", "hidden")
INTENSIVE = ("depth_m_mean", "porosity_mean", "concentration_aqueous", "saturation_free_gas")
MASS = ("total", "aqueous", "gas")
EXTENSIVE = ("physical_area_m2", "bulk_volume_m3", "pore_volume_m3", "gas_pore_volume_m3")
CHANNELS = ("M_maalt", "M_kant", "U")
PENDING = "not supplied; no producer or zero value inferred"
KNOWN_ZERO = "known_zero_by_adjacency_rule"


def _known_zero_channel(name, row):
    """The actual N16 zero decision is metadata, never a published U array."""
    if (name != "U" or not isinstance(row, dict)
            or set(row) != {"status", "native_total_g", "producer_binding"}
            or row["status"] != KNOWN_ZERO or type(row["native_total_g"]) not in (int, float)
            or row["native_total_g"] != 0
            or not isinstance(row["producer_binding"], dict)):
        raise ValueError("Known-zero U requires the exact actual N16 adjacency receipt")
    n16 = row["producer_binding"].get("n16")
    if (not isinstance(n16, dict) or n16.get("status") != "zero by adjacency rule"
            or n16.get("co2_adjacent") is not False
            or n16.get("touching_length_mm") != 0
            or n16.get("distance_limit_mm") != 20.0
            or n16.get("U_layer_present") is not False
            or type(n16.get("U_total_g")) not in (int, float) or n16["U_total_g"] != 0
            or n16.get("capacity_selector") != "literal-R3-hidden255-solubility"
            or type(n16.get("elapsed_h")) not in (int, float)
            or not math.isfinite(n16["elapsed_h"]) or n16["elapsed_h"] < 0):
        raise ValueError("Known-zero U conflicts with its actual original adjacency decision")
    # The producer bound the decision record by SHA; here it must still be named.
    from .series_decisions import recorded_decision
    try:
        recorded_decision(n16.get("authority_record"))
    except ValueError:
        raise ValueError("Known-zero U conflicts with its actual original adjacency decision") from None
    return copy.deepcopy(row)


def _exact(first, second):
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(_exact(first[k], second[k]) for k in first)
    if isinstance(first, list):
        return len(first) == len(second) and all(_exact(a, b) for a, b in zip(first, second))
    return first == second


def _identity(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _array(grid, value, name, *, boolean=False, integer=False):
    value = np.asarray(value)
    if value.shape != tuple(grid.native_shape) or value.dtype.hasobject:
        raise ValueError(name + " must retain the explicit native grid")
    if boolean:
        valid = value.dtype == np.bool_
    elif integer:
        valid = value.dtype.kind in "iu"
    else:
        valid = value.dtype.kind in "fiu"
    if not valid:
        raise ValueError(name + " has the wrong explicit array type")
    return value


def _count(grid, mask):
    return np.bincount(grid.probability_cell_index[mask],
                       minlength=math.prod(grid.shape)).reshape(grid.shape).astype(np.int64)


def _sum(grid, value, support):
    return np.bincount(grid.probability_cell_index[support],
                       weights=np.asarray(value, np.float64)[support],
                       minlength=math.prod(grid.shape)).reshape(grid.shape)


def _ratio(numerator, denominator, has_pv, dtype):
    value = np.full(denominator.shape, np.nan, np.float64)
    np.divide(np.asarray(numerator, np.float64), np.asarray(denominator, np.float64),
              out=value, where=has_pv & (denominator > 0))
    return value.astype(dtype)


def _total(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Explicit finite nonnegative native total required: " + name)
    return value


def _check(cells, expected, name):
    _total(expected, name)
    actual = float(cells.sum(dtype=np.float64))
    if not math.isfinite(actual) or abs(actual - expected) > RELATIVE_LIMIT * abs(expected):
        raise ValueError("Cell sum differs from native " + name + " by more than 1e-6")
    return dict(native_total=expected, stored_cell_sum=actual,
                absolute_difference=abs(actual - expected), relative_limit=RELATIVE_LIMIT)


def _fields(facies_ids, channel_names):
    fields = {
        "physical_area_m2": ("m2", "sum native pixel area over explicit physical support"),
        "bulk_volume_m3": ("m3", "sum area * depth over explicit physical support"),
        "pore_volume_m3": ("m3", "sum supplied original Geometry pixel pore-volume weights over explicit physical support"),
        "gas_pore_volume_m3": ("m3", "sum original Sg * supplied pixel pore volume where physical support and final label == 2"),
        "canvas_pixel_count": ("count", "count original full-canvas cell membership, independent of physical/PV support"),
        "physical_pixel_count": ("count", "count explicit physical support, including zero-area pixels"),
        "physical_properties_valid_pixel_count": ("count", "count explicit property-valid support within physical support"),
        "mass_support_pixel_count": ("count", "count explicit original native mass support"),
        "hidden_pixel_count": ("count", "count explicit hidden mask, independent of physical-domain convention"),
        "hidden_area_m2": ("m2", "sum original pixel area over explicit hidden mask; not a physical-PV definition"),
        "cell_has_pore_volume": ("boolean", "stored pore_volume_m3 > 0"),
        "depth_m_mean": ("m", "bulk volume / full physical area; NaN where PV is zero or no caller-valid input exists"),
        "porosity_mean": ("1", "pore volume / full bulk volume; NaN where PV is zero or no caller-valid input exists"),
        "concentration_aqueous": ("g/L", "independent aqueous mass / (full physical PV in m3 * 1000); NaN without caller-valid input"),
        "saturation_free_gas": ("1", "gas pore volume / full physical PV; NaN without caller-valid input"),
        "facies_majority": ("category", "original float64 area-weighted mode; lowest ID wins an exact positive-area tie; empty placeholder 0"),
    }
    for name in INTENSIVE:
        fields[name + "_valid_native_pixel_count"] = ("count", "count explicitly supplied valid native pixels within physical support; ratio denominator is unchanged")
        fields[name + "_has_valid_input"] = ("boolean", "physical PV > 0 and at least one explicitly supplied valid native pixel")
    for name in MASS:
        fields["cell_mass_" + name + "_g"] = ("g", "sum supplied independent original native pixel masses")
    for name in COVERAGE:
        fields["coverage_" + name + "_pore_volume_m3"] = ("m3", "sum supplied pixel PV over explicit " + name + " mask")
        fields["coverage_" + name + "_pore_volume_fraction"] = ("1", name + " PV / physical PV; NaN where PV is zero")
        fields["coverage_" + name + "_pixel_count"] = ("count", "count explicit " + name + " mask")
    for value in facies_ids:
        fields["facies_area_m2_" + str(value)] = ("m2", "sum original pixel area of literal facies ID over physical support")
        fields["facies_area_fraction_" + str(value)] = ("1", "literal facies area / physical area; NaN where PV is zero")
    for name in channel_names:
        fields["cell_" + name + "_g"] = ("g", "sum explicitly supplied " + name + " pixel channel on its declared support; no N15/N16 calculation")
        fields["channel_" + name + "_support_pixel_count"] = ("count", "count explicitly supplied " + name + " support")
    return {key: dict(units=value[0], weighting=value[1]) for key, value in fields.items()}


def aggregate_extensive(grid, *, depth_m, porosity, pixel_pore_volume_m3, facies,
                        physical_mask, physical_properties_valid_mask, mass_support_mask,
                        hidden_mask, coverage_masks, intensive_validity_masks, phase_labels, saturation_free_gas,
                        saturation_valid_mask, native_pixel_masses_g, native_mass_totals_g,
                        native_extensive_totals, channels, provenance):
    """Reduce explicit quantities only; require missing evidence, never infer it.

    ``channels`` must name M_maalt/M_kant/U explicitly. A None entry means the
    actual producer channel is unavailable and creates no array or zero. A
    supplied entry contains pixel_values_g, support_mask, native_total_g and
    producer_binding; this API only sums it and never derives the channel.
    """
    if grid is None or grid.precision not in {"float32", "float64"}:
        raise ValueError("An explicit existing metric output grid is required")
    ids = np.asarray(grid.probability_cell_index)
    if (ids.shape != tuple(grid.native_shape) or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids >= math.prod(grid.shape))):
        raise ValueError("Every canvas centre must retain its original assigned cell")
    if not isinstance(provenance, dict):
        raise ValueError("Explicit caller provenance is required, including explicit missing clock facts")
    json.dumps(provenance, allow_nan=False)
    physical = _array(grid, physical_mask, "physical support", boolean=True)
    property_valid = _array(grid, physical_properties_valid_mask, "physical property support", boolean=True)
    mass_support = _array(grid, mass_support_mask, "native mass support", boolean=True)
    hidden = _array(grid, hidden_mask, "hidden mask", boolean=True)
    saturation_valid = _array(grid, saturation_valid_mask, "saturation support", boolean=True)
    if np.any(physical & ~property_valid):
        raise ValueError("Missing physical properties: complete physical PV cannot be fabricated")
    if np.any(mass_support & ~physical):
        raise ValueError("Native mass support is outside the supplied physical domain")
    if not isinstance(coverage_masks, dict) or set(coverage_masks) != set(COVERAGE):
        raise ValueError("Explicit measured/estimated/hidden masks required")
    coverage, union = {}, np.zeros(grid.native_shape, bool)
    for name in COVERAGE:
        mask = _array(grid, coverage_masks[name], name, boolean=True)
        if np.any(mask & ~physical) or np.any(mask & union):
            raise ValueError("Coverage masks must be disjoint within explicit physical support")
        coverage[name] = mask
        union |= mask
    if not np.array_equal(union, physical):
        raise ValueError("Coverage masks must cover the explicit physical support")
    if not isinstance(intensive_validity_masks, dict) or set(intensive_validity_masks) != set(INTENSIVE):
        raise ValueError("Every intensive output needs its explicit caller-owned validity mask")
    intensive_validity = {}
    for name in INTENSIVE:
        mask = _array(grid, intensive_validity_masks[name], name + " validity", boolean=True)
        if np.any(mask & ~physical):
            raise ValueError("Intensive validity must be within explicit physical support")
        intensive_validity[name] = mask
    depth = _array(grid, depth_m, "depth")
    phi = _array(grid, porosity, "porosity")
    area = _array(grid, grid.native_pixel_area_m2, "pixel area")
    pv = _array(grid, pixel_pore_volume_m3, "original Geometry pixel pore volume")
    for name, value in (("area", area), ("pore volume", pv)):
        if not np.isfinite(value[physical]).all() or np.any(value[physical] < 0):
            raise ValueError("Physical " + name + " must be finite and nonnegative")
    if (not np.isfinite(depth[physical]).all() or np.any(depth[physical] <= 0)
            or not np.isfinite(phi[physical]).all() or np.any(phi[physical] <= 0)
            or np.any(phi[physical] > 1)):
        raise ValueError("Physical properties must be actual finite measurements; no capacity imputation")
    if (not np.isfinite(area[hidden]).all() or np.any(area[hidden] < 0)
            or np.any((pv[physical] == 0) != (area[physical] == 0))):
        raise ValueError("Explicit hidden area/original positive-PV support is inconsistent")
    categories = _array(grid, facies, "facies", integer=True)
    if np.any(categories[physical] < 0):
        raise ValueError("Actual nonnegative integer facies IDs required")
    labels = _array(grid, phase_labels, "phase labels", integer=True)
    if labels.dtype != np.uint8 or not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("Original uint8 phase labels 0/1/2/255 required")
    saturation = _array(grid, saturation_free_gas, "gas saturation")
    gas = physical & (labels == 2)
    if (np.any(gas & ~saturation_valid) or not np.isfinite(saturation[gas]).all()
            or np.any(saturation[gas] < 0) or np.any(saturation[gas] > 1)):
        raise ValueError("Physical gas pixels need explicit valid original saturation; missing is not zero")
    if (not isinstance(native_pixel_masses_g, dict) or set(native_pixel_masses_g) != set(MASS)
            or not isinstance(native_mass_totals_g, dict) or set(native_mass_totals_g) != set(MASS)
            or not isinstance(native_extensive_totals, dict) or set(native_extensive_totals) != set(EXTENSIVE)):
        raise ValueError("All independent original mass and extensive native references required")
    if not isinstance(channels, dict) or set(channels) != set(CHANNELS):
        raise ValueError("Every N15/N16 channel must be explicitly supplied or None=pending")
    dtype = np.dtype(grid.precision)
    bulk = np.zeros(grid.native_shape, np.float64)
    bulk[physical] = area[physical].astype(np.float64) * depth[physical].astype(np.float64)
    gas_pv = np.zeros(grid.native_shape, np.float64)
    gas_pv[gas] = saturation[gas].astype(np.float64) * pv[gas].astype(np.float64)
    arrays = {key: _sum(grid, value, physical).astype(dtype) for key, value in
              (("physical_area_m2", area), ("bulk_volume_m3", bulk),
               ("pore_volume_m3", pv), ("gas_pore_volume_m3", gas_pv))}
    arrays.update(canvas_pixel_count=_count(grid, np.ones(grid.native_shape, bool)),
                  physical_pixel_count=_count(grid, physical),
                  physical_properties_valid_pixel_count=_count(grid, physical & property_valid),
                  mass_support_pixel_count=_count(grid, mass_support),
                  hidden_pixel_count=_count(grid, hidden),
                  hidden_area_m2=_sum(grid, area, hidden).astype(dtype),
                  cell_has_pore_volume=arrays["pore_volume_m3"] > 0)
    checks = {key: _check(arrays[key], native_extensive_totals[key], key) for key in EXTENSIVE}
    inputs = {key: _identity(value) for key, value in
              (("pixel_area_m2", area), ("depth_m", depth), ("porosity", phi),
               ("pixel_pore_volume_m3", pv), ("physical_mask", physical),
               ("physical_properties_valid_mask", property_valid),
               ("mass_support_mask", mass_support), ("hidden_mask", hidden),
               ("saturation_valid_mask", saturation_valid), ("facies", categories),
               ("phase_labels", labels), ("saturation_free_gas", saturation))}
    for name in MASS:
        value = _array(grid, native_pixel_masses_g[name], name + " pixel mass")
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value[~mass_support] != 0):
            raise ValueError("Native mass must be finite/nonnegative and zero outside its explicit support")
        key = "cell_mass_" + name + "_g"
        arrays[key] = _sum(grid, value, mass_support).astype(dtype)
        checks[key] = _check(arrays[key], native_mass_totals_g[name], key)
        inputs[key] = _identity(value)
    has_pv = arrays["cell_has_pore_volume"]
    for key, numerator, denominator in (
            ("depth_m_mean", arrays["bulk_volume_m3"], arrays["physical_area_m2"]),
            ("porosity_mean", arrays["pore_volume_m3"], arrays["bulk_volume_m3"]),
            ("concentration_aqueous", arrays["cell_mass_aqueous_g"], arrays["pore_volume_m3"].astype(np.float64) * 1000),
            ("saturation_free_gas", arrays["gas_pore_volume_m3"], arrays["pore_volume_m3"])):
        count = _count(grid, intensive_validity[key])
        valid = has_pv & (count > 0)
        arrays[key + "_valid_native_pixel_count"] = count
        arrays[key + "_has_valid_input"] = valid
        inputs["intensive_" + key + "_validity_mask"] = _identity(intensive_validity[key])
        # Validity controls NaN only. Ratios retain full physical denominators.
        arrays[key] = _ratio(numerator, denominator, valid, dtype)
    for name, mask in coverage.items():
        key = "coverage_" + name
        arrays[key + "_pore_volume_m3"] = _sum(grid, pv, mask).astype(dtype)
        arrays[key + "_pore_volume_fraction"] = _ratio(arrays[key + "_pore_volume_m3"], arrays["pore_volume_m3"], has_pv, dtype)
        arrays[key + "_pixel_count"] = _count(grid, mask)
        inputs[key + "_mask"] = _identity(mask)
    facies_ids = list(map(int, np.unique(categories[physical])))
    majority, maximum = np.zeros(grid.shape, categories.dtype), np.zeros(grid.shape, np.float64)
    for value in facies_ids:
        weights = _sum(grid, area, physical & (categories == value))
        selected = weights > maximum
        majority[selected], maximum[selected] = value, weights[selected]
        key = "facies_area_m2_" + str(value)
        arrays[key] = weights.astype(dtype)
        arrays["facies_area_fraction_" + str(value)] = _ratio(arrays[key], arrays["physical_area_m2"], has_pv, dtype)
    arrays["facies_majority"] = majority
    channel_metadata = {}
    for name in CHANNELS:
        channel = channels[name]
        if channel is None:
            channel_metadata[name] = dict(status=PENDING)
            continue
        if isinstance(channel, dict) and channel.get("status") == KNOWN_ZERO:
            channel_metadata[name] = _known_zero_channel(name, channel)
            continue
        if not isinstance(channel, dict) or set(channel) != {"pixel_values_g", "support_mask", "native_total_g", "producer_binding"}:
            raise ValueError("Explicit closed native channel ABI required: " + name)
        if not isinstance(channel["producer_binding"], dict) or not channel["producer_binding"]:
            raise ValueError("A supplied channel needs its actual producer binding")
        json.dumps(channel["producer_binding"], allow_nan=False)
        mask = _array(grid, channel["support_mask"], name + " support", boolean=True)
        value = _array(grid, channel["pixel_values_g"], name + " values")
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value[~mask] != 0):
            raise ValueError("Explicit channel values must be finite/nonnegative and retain their support")
        key = "cell_" + name + "_g"
        arrays[key] = _sum(grid, value, mask).astype(dtype)
        arrays["channel_" + name + "_support_pixel_count"] = _count(grid, mask)
        checks[key] = _check(arrays[key], channel["native_total_g"], key)
        inputs[key] = _identity(value)
        inputs["channel_" + name + "_support_mask"] = _identity(mask)
        channel_metadata[name] = dict(status="supplied, output-only sum", native_total_g=channel["native_total_g"],
                                      producer_binding=copy.deepcopy(channel["producer_binding"]))
    metadata = dict(format=FORMAT, precision=grid.precision, reduction_dtype="float64",
                    grid=grid.metadata(), facies_ids=facies_ids,
                    fields=_fields(facies_ids, [n for n in CHANNELS if channel_metadata[n]["status"] == "supplied, output-only sum"]),
                    arrays={key: _identity(value) for key, value in arrays.items()},
                    input_identities=inputs, native_mass_totals_g=dict(native_mass_totals_g),
                    native_extensive_totals=dict(native_extensive_totals), native_checks=checks,
                    channels=channel_metadata, provenance=copy.deepcopy(provenance),
                    domain_authority="explicit caller masks only; physical/hidden convention not inferred",
                    scope="pure extensive arrays; physical/producer admission and N15/N16 rule execution are not performed")
    _validate_arrays(arrays, metadata, native_mass_totals_g, native_extensive_totals)
    return dict(arrays=arrays, metadata=metadata)


def _relative_close(actual, expected):
    return bool(np.all(np.abs(np.asarray(actual, np.float64) - expected)
                       <= RELATIVE_LIMIT * np.abs(np.asarray(expected, np.float64))))


def _not_greater(actual, expected):
    return bool(np.all(np.asarray(actual, np.float64) - expected
                       <= RELATIVE_LIMIT * np.abs(np.asarray(expected, np.float64))))


def _validate_arrays(arrays, metadata, mass_totals, extensive_totals):
    facies_ids, channel_rows = metadata["facies_ids"], metadata["channels"]
    channels = [n for n in CHANNELS if channel_rows[n]["status"] == "supplied, output-only sum"]
    for name, row in channel_rows.items():
        if row.get("status") == KNOWN_ZERO:
            _known_zero_channel(name, row)
    expected_fields = _fields(facies_ids, channels)
    if not _exact(metadata["fields"], expected_fields) or set(arrays) != set(expected_fields):
        raise ValueError("Extensive units/weighting/array closure differs")
    shape, dtype = tuple(metadata["grid"]["shape"]), np.dtype(metadata["precision"])
    for key, value in arrays.items():
        if value.shape != shape or value.dtype.hasobject or np.isinf(value).any():
            raise ValueError("Extensive shape/type/infinity differs")
        if key == "cell_has_pore_volume" or key.endswith("_has_valid_input"):
            valid = value.dtype == np.bool_
        elif key.endswith("_pixel_count"):
            valid = value.dtype == np.int64 and np.all(value >= 0)
        elif key == "facies_majority":
            valid = (value.dtype.kind in "iu"
                     and value.dtype.str == metadata["input_identities"]["facies"]["dtype"])
        else:
            valid = value.dtype == dtype
        if not valid:
            raise ValueError("Extensive dtype/count differs: " + key)
    count = arrays["canvas_pixel_count"]
    native_count = math.prod(metadata["grid"]["native_shape"])
    if np.any(count > native_count) or sum(map(int, count.flat)) != native_count:
        raise ValueError("Full-canvas count differs from original native shape")
    if any(np.any(value > count) for key, value in arrays.items()
           if key.endswith("_pixel_count") and key != "canvas_pixel_count"):
        raise ValueError("A declared support count exceeds its original full-canvas cell")
    physical = arrays["physical_pixel_count"]
    if (np.any(physical > count) or np.any(arrays["hidden_pixel_count"] > count)
            or not np.array_equal(arrays["physical_properties_valid_pixel_count"], physical)
            or np.any(arrays["mass_support_pixel_count"] > physical)):
        raise ValueError("Explicit support count closure differs")
    if not np.array_equal(sum(arrays["coverage_" + n + "_pixel_count"] for n in COVERAGE), physical):
        raise ValueError("Coverage count partition differs")
    extensive_names = [*EXTENSIVE, "hidden_area_m2", *("cell_mass_" + n + "_g" for n in MASS),
                       *("coverage_" + n + "_pore_volume_m3" for n in COVERAGE),
                       *("facies_area_m2_" + str(n) for n in facies_ids),
                       *("cell_" + n + "_g" for n in channels)]
    if any(not np.isfinite(arrays[k]).all() or np.any(arrays[k] < 0) for k in extensive_names):
        raise ValueError("Extensive quantities must be finite and nonnegative")
    checks = {}
    for key in EXTENSIVE:
        checks[key] = _check(arrays[key], extensive_totals[key], key)
    for name in MASS:
        key = "cell_mass_" + name + "_g"
        checks[key] = _check(arrays[key], mass_totals[name], key)
    for name in channels:
        key = "cell_" + name + "_g"
        checks[key] = _check(arrays[key], channel_rows[name]["native_total_g"], key)
        if np.any(arrays["channel_" + name + "_support_pixel_count"] > count):
            raise ValueError("Channel support count exceeds full canvas")
        if np.any(arrays[key][arrays["channel_" + name + "_support_pixel_count"] == 0] != 0):
            raise ValueError("Channel mass is outside its declared support")
    if not _exact(metadata["native_checks"], checks):
        raise ValueError("Independent native scalar closure differs")
    pv, area, bulk = (arrays[k] for k in ("pore_volume_m3", "physical_area_m2", "bulk_volume_m3"))
    has_pv = pv > 0
    if not np.array_equal(arrays["cell_has_pore_volume"], has_pv):
        raise ValueError("Pore-volume validity differs")
    zero = ~has_pv
    zero_extensive = [*EXTENSIVE, *("cell_mass_" + n + "_g" for n in MASS),
                      *("coverage_" + n + "_pore_volume_m3" for n in COVERAGE),
                      *("facies_area_m2_" + str(n) for n in facies_ids)]
    if (any(np.any(arrays[k][zero] != 0) for k in zero_extensive)
            or np.any(area[has_pv] <= 0) or np.any(bulk[has_pv] <= 0)
            or not _not_greater(pv, bulk) or not _not_greater(arrays["gas_pore_volume_m3"], pv)):
        raise ValueError("Zero-PV extensive support or physical volume relation differs")
    if (np.any(arrays["physical_area_m2"][physical == 0] != 0)
            or any(np.any(arrays["cell_mass_" + n + "_g"][arrays["mass_support_pixel_count"] == 0] != 0) for n in MASS)):
        raise ValueError("Physical/mass support counts omit nonzero quantities")
    for key, numerator, denominator in (
            ("depth_m_mean", bulk, area), ("porosity_mean", pv, bulk),
            ("concentration_aqueous", arrays["cell_mass_aqueous_g"], pv.astype(np.float64) * 1000),
            ("saturation_free_gas", arrays["gas_pore_volume_m3"], pv)):
        valid_count = arrays[key + "_valid_native_pixel_count"]
        valid = has_pv & (valid_count > 0)
        if (np.any(valid_count > physical)
                or not np.array_equal(arrays[key + "_has_valid_input"], valid)
                or not np.array_equal(arrays[key], _ratio(numerator, denominator, valid, dtype), equal_nan=True)):
            raise ValueError("Explicit intensive validity/full-denominator ratio differs: " + key)
    coverage_pv = sum(arrays["coverage_" + n + "_pore_volume_m3"].astype(np.float64) for n in COVERAGE)
    if not _relative_close(coverage_pv, pv):
        raise ValueError("Coverage PV partition differs by more than 1e-6 relative")
    for name in COVERAGE:
        key = "coverage_" + name
        value = arrays[key + "_pore_volume_m3"]
        if np.any(value[arrays[key + "_pixel_count"] == 0] != 0):
            raise ValueError("Coverage PV is outside its declared support count")
        if not np.array_equal(arrays[key + "_pore_volume_fraction"], _ratio(value, pv, has_pv, dtype), equal_nan=True):
            raise ValueError("Coverage fraction denominator differs")
    facies_area = sum((arrays["facies_area_m2_" + str(n)].astype(np.float64) for n in facies_ids), start=np.zeros(shape))
    if not _relative_close(facies_area, area):
        raise ValueError("Facies area fractions omit the physical area")
    for value in facies_ids:
        key = str(value)
        if not np.array_equal(arrays["facies_area_fraction_" + key],
                              _ratio(arrays["facies_area_m2_" + key], area, has_pv, dtype), equal_nan=True):
            raise ValueError("Facies area fraction denominator differs")
    majority = arrays["facies_majority"]
    if (np.any(majority[zero] != 0) or not np.isin(majority[has_pv], facies_ids).all()):
        raise ValueError("Facies majority category/empty placeholder differs")
    # Rounded stored areas cannot prove the native float64 ordering of a tie.
    # Require the selected category to be a stored maximum; the source/outer
    # producer binding retains the exact original native tie decision.
    if facies_ids:
        largest = np.maximum.reduce([arrays["facies_area_m2_" + str(n)] for n in facies_ids])
        selected = np.zeros(shape, dtype)
        for value in facies_ids:
            mask = majority == value
            selected[mask] = arrays["facies_area_m2_" + str(value)][mask]
        if np.any(selected[has_pv] != largest[has_pv]):
            raise ValueError("Facies majority has no maximal stored area")


def read_extensive_archive(data, metadata, *, expected_input_identities,
                           native_mass_totals_g, native_extensive_totals,
                           expected_channels, expected_provenance, expected_grid_metadata):
    """Pure bytes reader; outer producer/physical admission stays mandatory."""
    keys = {"format", "precision", "reduction_dtype", "grid", "facies_ids", "fields", "arrays",
            "input_identities", "native_mass_totals_g", "native_extensive_totals", "native_checks",
            "channels", "provenance", "domain_authority", "scope"}
    if (type(data) is not bytes or not isinstance(metadata, dict) or set(metadata) != keys
            or metadata["format"] != FORMAT or metadata["precision"] not in {"float32", "float64"}
            or metadata["reduction_dtype"] != "float64"
            or not _exact(metadata["input_identities"], expected_input_identities)
            or not _exact(metadata["native_mass_totals_g"], native_mass_totals_g)
            or not _exact(metadata["native_extensive_totals"], native_extensive_totals)
            or not _exact(metadata["channels"], expected_channels)
            or not _exact(metadata["provenance"], expected_provenance)
            or not isinstance(metadata["provenance"], dict)
            or not _exact(metadata["grid"], expected_grid_metadata)
            or metadata["domain_authority"] != "explicit caller masks only; physical/hidden convention not inferred"
            or metadata["scope"] != "pure extensive arrays; physical/producer admission and N15/N16 rule execution are not performed"):
        raise ValueError("Extensive source/native/version/provenance identity differs")
    if (not isinstance(native_mass_totals_g, dict) or set(native_mass_totals_g) != set(MASS)
            or not isinstance(native_extensive_totals, dict) or set(native_extensive_totals) != set(EXTENSIVE)):
        raise ValueError("Independent native scalar keys differ")
    for key, value in {**native_mass_totals_g, **native_extensive_totals}.items():
        _total(value, key)
    grid, facies_ids, channels = metadata["grid"], metadata["facies_ids"], metadata["channels"]
    if (not isinstance(grid, dict) or grid.get("format") != "darsia-metric-storage-grid-v1"
            or grid.get("storage_precision") != metadata["precision"]
            or not isinstance(grid.get("shape"), list) or any(type(n) is not int or n < 1 for n in grid["shape"])
            or len(grid["shape"]) != 2
            or not isinstance(grid.get("native_shape"), list) or len(grid["native_shape"]) != 2
            or any(type(n) is not int or n < 1 for n in grid["native_shape"])
            or tuple(grid["shape"]) != metric_grid_shape(grid["extent_m"], grid["resolution_mm"])
            or not isinstance(facies_ids, list) or any(type(n) is not int or n < 0 for n in facies_ids)
            or facies_ids != sorted(set(facies_ids))
            or not isinstance(channels, dict) or set(channels) != set(CHANNELS)):
        raise ValueError("Extensive explicit grid/facies/channel schema differs")
    supplied = []
    for name in CHANNELS:
        row = channels[name]
        if not isinstance(row, dict):
            raise ValueError("Explicit channel state required")
        if row.get("status") == PENDING:
            if set(row) != {"status"}:
                raise ValueError("Pending channel cannot carry fabricated values")
        elif row.get("status") == KNOWN_ZERO:
            _known_zero_channel(name, row)
        else:
            if (set(row) != {"status", "native_total_g", "producer_binding"}
                    or row["status"] != "supplied, output-only sum"
                    or not isinstance(row["producer_binding"], dict) or not row["producer_binding"]):
                raise ValueError("Actual channel producer identity required")
            _total(row["native_total_g"], name)
            supplied.append(name)
    inputs = metadata["input_identities"]
    input_keys = {"pixel_area_m2", "depth_m", "porosity", "pixel_pore_volume_m3",
                  "physical_mask", "physical_properties_valid_mask", "mass_support_mask",
                  "hidden_mask", "saturation_valid_mask", "facies", "phase_labels", "saturation_free_gas"}
    input_keys |= {"cell_mass_" + name + "_g" for name in MASS}
    input_keys |= {"coverage_" + name + "_mask" for name in COVERAGE}
    input_keys |= {"intensive_" + name + "_validity_mask" for name in INTENSIVE}
    input_keys |= {"cell_" + name + "_g" for name in supplied}
    input_keys |= {"channel_" + name + "_support_mask" for name in supplied}
    if not isinstance(inputs, dict) or set(inputs) != input_keys:
        raise ValueError("Extensive native input identity closure differs")
    for key, row in inputs.items():
        if (not isinstance(row, dict) or set(row) != {"shape", "dtype", "sha256"}
                or not _exact(row["shape"], grid["native_shape"])
                or not isinstance(row["dtype"], str) or not isinstance(row["sha256"], str)
                or re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("Explicit typed native input identity required")
        actual_type = np.dtype(row["dtype"])
        allowed = actual_type.kind in "fiu"
        if key.endswith("_mask"):
            allowed = actual_type == np.bool_
        elif key == "phase_labels":
            allowed = actual_type == np.uint8
        elif key == "facies":
            allowed = actual_type.kind in "iu"
        if not allowed:
            raise ValueError("Extensive native input dtype differs")
    identities = metadata["arrays"]
    if (not isinstance(identities, dict) or set(identities) != set(_fields(facies_ids, supplied))
            or not _exact(metadata["fields"], _fields(facies_ids, supplied))):
        raise ValueError("Extensive exact array/field closure differs")
    for key, row in identities.items():
        if (not isinstance(row, dict) or set(row) != {"shape", "dtype", "sha256"}
                or not _exact(row["shape"], grid["shape"])
                or not isinstance(row["dtype"], str)
                or not isinstance(row["sha256"], str) or re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("Explicit typed stored-array identity required")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if (len(members) != len({m.filename for m in members})
                or {m.filename for m in members} != {key + ".npy" for key in identities}):
            raise ValueError("Extensive ZIP members differ")
        for row in members:
            dtype = np.dtype(identities[row.filename[:-4]]["dtype"])
            if dtype.hasobject or dtype.kind not in "buif" or row.file_size > math.prod(grid["shape"]) * dtype.itemsize + 65536:
                raise ValueError("Extensive archive exceeds its declared numeric array size")
    arrays = {}
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != set(identities):
            raise ValueError("Extensive logical member closure differs")
        for key, identity in identities.items():
            value = archive[key]
            if not _exact(_identity(value), identity):
                raise ValueError("Extensive stored array identity differs")
            arrays[key] = value
    _validate_arrays(arrays, metadata, native_mass_totals_g, native_extensive_totals)
    for value in arrays.values():
        value.setflags(write=False)
    return dict(format=FORMAT, arrays=arrays, metadata=copy.deepcopy(metadata),
                physical_binding_admitted=False, producer_identity_admitted=False)
