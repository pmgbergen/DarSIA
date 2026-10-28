"""Inert coarser data from already verified one-millimetre extensive cells.

This is a postprocessing format, separate from every analysis-worker format.
It neither reads native pixels nor proves the native-to-one-millimetre replay
gate. Missing channels remain missing, and no simulator format is selected.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import math

import numpy as np

from . import storage_extensive as extensive
from .storage_resolution import metric_grid_shape

FORMAT = "darsia-coarsened-extensive-data-v1"
GRID_FORMAT = "darsia-coarsened-metric-grid-v1"
_TARGETS = (1, 2, 5, 10)
_AXES = "row=y increasing, column=x increasing"


def _target(value):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value not in _TARGETS):
        raise ValueError("Choose an explicit 1, 2, 5 or 10 mm output grid")
    return int(value)


def _source(payload):
    if (not isinstance(payload, dict) or payload.get("format") != extensive.FORMAT
            or not isinstance(payload.get("arrays"), dict)
            or not isinstance(payload.get("metadata"), dict)):
        raise ValueError("An already verified complete extensive payload is required")
    metadata, arrays = payload["metadata"], payload["arrays"]
    if (metadata.get("format") != extensive.FORMAT
            or metadata.get("precision") not in {"float32", "float64"}
            or metadata.get("reduction_dtype") != "float64"):
        raise ValueError("The original extensive version and precision must be retained")
    grid = metadata["grid"]
    if (grid.get("format") != "darsia-metric-storage-grid-v1"
            or grid.get("resolution_mm") != 1.0 or type(grid.get("resolution_mm")) is bool
            or grid.get("axes") != _AXES
            or grid.get("storage_precision") != metadata["precision"]
            or not isinstance(grid.get("shape"), list) or len(grid["shape"]) != 2
            or any(type(value) is not int or value < 1 for value in grid["shape"])
            or not isinstance(grid.get("native_shape"), list) or len(grid["native_shape"]) != 2
            or any(type(value) is not int or value < 1 for value in grid["native_shape"])
            or tuple(grid["shape"]) != metric_grid_shape(grid["extent_m"], 1.0)):
        raise ValueError("The source must be a complete origin-zero aligned 1 mm metric grid")
    ids, channels = metadata["facies_ids"], metadata["channels"]
    if (not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids)
            or ids != sorted(set(ids)) or not isinstance(channels, dict)
            or set(channels) != set(extensive.CHANNELS)):
        raise ValueError("The source facies and channel schema differs")
    for name, row in channels.items():
        if not isinstance(row, dict):
            raise ValueError("Every source channel needs its explicit original state")
        if row.get("status") == extensive.PENDING:
            if set(row) != {"status"}:
                raise ValueError("A pending channel cannot acquire a fabricated value")
        elif row.get("status") == extensive.KNOWN_ZERO:
            extensive._known_zero_channel(name, row)
        elif (row.get("status") != "supplied, output-only sum"
                or set(row) != {"status", "native_total_g", "producer_binding"}):
            raise ValueError("A supplied channel needs its original explicit producer state")
    if set(arrays) != set(metadata["arrays"]):
        raise ValueError("The source extensive array closure differs")
    for name, value in arrays.items():
        if (not isinstance(value, np.ndarray)
                or extensive._identity(value) != metadata["arrays"][name]):
            raise ValueError("The source array differs from its verified identity: " + name)
    # The source identities describe logical C-order values, not an ndarray's
    # physical strides. Retain that same order for validation and every sum.
    canonical = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    extensive._validate_arrays(canonical, metadata,
        metadata["native_mass_totals_g"], metadata["native_extensive_totals"])
    return canonical, metadata


def _shape(grid, factor):
    step = Decimal(factor) / Decimal(1000)
    # Decimal JSON endpoints retain the exact origin-zero lattice rule. No
    # snapping, trailing partial cell, clipping or unrequested padding occurs.
    for edge in grid["extent_m"]:
        if Decimal(str(edge)) % step:
            raise ValueError("Every extent edge must lie on the requested origin-zero output lattice")
    height, width = grid["shape"]
    if height % factor or width % factor:
        raise ValueError("Coarsening requires complete cells; padding and clipping are not performed")
    return height // factor, width // factor


def _sum_cells(value, factor):
    value = np.ascontiguousarray(value)
    height, width = value.shape
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = value.reshape(height // factor, factor, width // factor, factor).sum(
                axis=(1, 3), dtype=np.float64)
    except FloatingPointError as error:
        raise ValueError("A coarsened extensive sum is not finite") from error
    if not np.isfinite(result).all():
        raise ValueError("A coarsened extensive sum is not finite")
    return result


def _sum_counts(value, factor):
    value = np.ascontiguousarray(value)
    if value.dtype != np.int64 or np.any(value < 0):
        raise ValueError("Source counts must be nonnegative int64 values")
    height, width = value.shape
    blocks = value.reshape(height // factor, factor, width // factor, factor)
    limit = np.iinfo(np.int64).max
    if int(value.max(initial=0)) <= limit // (factor * factor):
        return blocks.sum(axis=(1, 3), dtype=np.int64)
    # Only unusually large counts need Python integer arithmetic. Checking
    # before casting avoids both signed wraparound and float rounding at 2**63.
    result = np.empty((height // factor, width // factor), np.int64)
    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            total = sum(map(int, blocks[row, :, col, :].flat))
            if total > limit:
                raise ValueError("A coarsened native-pixel count exceeds int64")
            result[row, col] = total
    return result


def _totals(source, target, names):
    result = {}
    for name in names:
        if source[name].dtype == np.int64:
            before = sum(map(int, source[name].flat))
            after = sum(map(int, target[name].flat))
            if before != after:
                raise ValueError("A coarsened count did not conserve its source cells")
        else:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    before = float(np.ascontiguousarray(source[name]).sum(dtype=np.float64))
                    after = float(np.ascontiguousarray(target[name]).sum(dtype=np.float64))
            except FloatingPointError as error:
                raise ValueError("A source or coarsened extensive total is not finite") from error
            if not math.isfinite(before) or not math.isfinite(after):
                raise ValueError("A source or coarsened extensive total is not finite")
        result[name] = dict(source_sum=before, output_sum=after, difference=after - before)
    return result


def coarsen_verified_extensive(payload, *, target_resolution_mm):
    """Sum verified 1 mm cells, preserving every explicit extensive channel.

    Targets 2, 5 and 10 mm use float64 block sums and float64 derived ratios;
    source float32 rounding is never inverted. Counts retain checked int64.
    An explicit 1 mm request is an identity view with unchanged array bytes,
    dtypes and original tie decisions. Neither route is native re-analysis.
    """
    try:
        factor = _target(target_resolution_mm)
        original, metadata = _source(payload)
        shape = _shape(metadata["grid"], factor)
        facies_ids = metadata["facies_ids"]
        channels = [name for name in extensive.CHANNELS
                    if metadata["channels"][name]["status"] == "supplied, output-only sum"]
        counts = sorted(name for name in original if name.endswith("_pixel_count"))
        summed = [*extensive.EXTENSIVE, "hidden_area_m2",
            *("cell_mass_" + name + "_g" for name in extensive.MASS),
            *("coverage_" + name + "_pore_volume_m3" for name in extensive.COVERAGE),
            *("facies_area_m2_" + str(value) for value in facies_ids),
            *("cell_" + name + "_g" for name in channels)]
        fields = deepcopy(metadata["fields"])
        if factor == 1:
            arrays = {name: value.view() for name, value in original.items()}
            mode = "identity: source array bytes, dtypes and original majority retained"
        else:
            arrays = {name: _sum_cells(original[name], factor) for name in summed}
            arrays.update({name: _sum_counts(original[name], factor) for name in counts})
            pv, area, bulk = (arrays[name] for name in
                             ("pore_volume_m3", "physical_area_m2", "bulk_volume_m3"))
            has_pv = arrays["cell_has_pore_volume"] = pv > 0
            with np.errstate(over="raise", invalid="raise"):
                litres = pv * 1000
            for name, numerator, denominator in (
                    ("depth_m_mean", bulk, area), ("porosity_mean", pv, bulk),
                    ("concentration_aqueous", arrays["cell_mass_aqueous_g"], litres),
                    ("saturation_free_gas", arrays["gas_pore_volume_m3"], pv)):
                valid = has_pv & (arrays[name + "_valid_native_pixel_count"] > 0)
                arrays[name + "_has_valid_input"] = valid
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    arrays[name] = extensive._ratio(numerator, denominator, valid, np.float64)
            for name in extensive.COVERAGE:
                key = "coverage_" + name
                arrays[key + "_pore_volume_fraction"] = extensive._ratio(
                    arrays[key + "_pore_volume_m3"], pv, has_pv, np.float64)
            majority = np.zeros(shape, dtype=original["facies_majority"].dtype)
            maximum = np.zeros(shape, np.float64)
            for value in facies_ids:
                key = str(value)
                weights = arrays["facies_area_m2_" + key]
                arrays["facies_area_fraction_" + key] = extensive._ratio(weights, area, has_pv, np.float64)
                selected = weights > maximum
                majority[selected], maximum[selected] = value, weights[selected]
            arrays["facies_majority"] = majority
            mode = "float64 sum of stored 1 mm extensive cells; no additional float32 rounding"
            for name in summed:
                fields[name]["weighting"] = "sum stored 1 mm cells: " + fields[name]["weighting"]
            for name in extensive.INTENSIVE:
                fields[name]["weighting"] = "derive from summed stored numerators and full denominators; retain explicit validity counts"
            fields["facies_majority"]["weighting"] = (
                "largest summed stored 1 mm facies area; lowest ID wins exact positive-area tie; empty placeholder 0")
        if set(arrays) != set(original):
            raise ValueError("The coarsening did not preserve the complete explicit field set")
        for name, value in arrays.items():
            if value.dtype.kind == "f" and np.isinf(value).any():
                raise ValueError("A derived coarsened value is not finite")
            value.setflags(write=False)
        conservation = _totals(original, arrays, [*summed, *counts])
        # Preserve only supplied outer-envelope evidence. This snapshot is
        # source provenance, never a new producer or production admission.
        envelope_fields = ("format", "state", "original_frame_identity", "producer_binding",
            "loaded_assets", "physical_binding_verified", "producer_identity_verified",
            "physical_binding_admitted", "producer_identity_admitted",
            "production_permission_granted", "U_layer_present")
        source_binding = {key: deepcopy(payload[key]) for key in envelope_fields if key in payload}
        result_metadata = dict(format=FORMAT, source_format=extensive.FORMAT,
            source_precision=metadata["precision"], output_float_dtype=metadata["precision"] if factor == 1 else "float64",
            reduction_dtype="none" if factor == 1 else "float64", operation=mode,
            grid=dict(format=GRID_FORMAT, resolution_mm=factor, shape=list(shape),
                extent_m=deepcopy(metadata["grid"]["extent_m"]), axes=_AXES, origin_m=[0, 0]),
            source_grid=deepcopy(metadata["grid"]), source_metadata=deepcopy(metadata),
            source_binding=source_binding,
            fields=fields, facies_ids=list(facies_ids), channels=deepcopy(metadata["channels"]),
            arrays={name: extensive._identity(value) for name, value in arrays.items()},
            conservation=conservation,
            summation_scope="each output cell is the deterministic float64 sum of its stored source cells; whole-grid floating sums may differ by reduction order",
            native_to_1mm_reexport_gate="NOT_RUN", native_fields_reconstructed=False,
            simulator_format_selected=False, physical_binding_admitted=False,
            producer_identity_admitted=False, production_permission_granted=False)
        return dict(format=FORMAT, arrays=arrays, metadata=result_metadata,
                    complete=True, adopted=False, production_permission_granted=False)
    except (KeyError, TypeError, IndexError, AttributeError, ArithmeticError) as error:
        raise ValueError("The complete explicit extensive payload cannot be coarsened") from error
