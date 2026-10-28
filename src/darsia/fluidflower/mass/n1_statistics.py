"""Native paired exclusion estimates and corrected-map U_rest capacities."""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from .phase_expert_knowledge import expert_knowledge_of
from .concentration_joint import dilution_factor
from .n1_arithmetic import (ANALYSIS_GAP, analysis_gap_scope, main_partition,
                            scoped_result, validate_parameters)
from ..geometry.pixels import _column_neighbours
INTERVALS = 32768
MAX_COLUMN_DISTANCE_M = .020
CONTRACT = "darsia-n1-estimates-v1"

def reduce_weighted_codes(codes, weights, *, backend="cpu"):
    """Original sorted int64 histogram and CPU64 bincount reduction."""
    codes, weights = np.asarray(codes), np.asarray(weights, np.float64)
    if codes.ndim != 1 or codes.dtype.kind not in "iu" or weights.shape != codes.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Finite nonnegative histogram weights and integer codes required")
    codes = codes.astype(np.int64, copy=False)
    unique, inverse = np.unique(codes, return_inverse=True)
    sums = np.bincount(inverse, weights=weights, minlength=len(unique))
    return unique, sums, {"backend":"cpu-float64", "input_rows":len(codes), "downsampled":False, "optical_response_changed":False, "reference_backend":"cpu-float64"}

def baseline_main(response, parameters, anchor, pixel_math):
    """Original raw-summary N1 baseline, separate from the selected gain."""
    frame, physics, analysis = response["frame"],response["physics"],response["analysis"]
    partition=main_partition(response)
    factor=dilution_factor(frame.elapsed_h,partition["affected_pore_volume_m3"],anchor,0.)
    maps=pixel_math.selected_n1_fields(response["prepared"].aqueous_relative,gain=parameters["gain"],factor=factor,swi=response["swi"],aqueous_known=partition["aq_known"],gas_known=partition["gas_known"],solubility=physics.solubility_co2,gas_density=physics.density_gaseous_co2)
    eligible=frame.mass_eligible_mask & frame.known_physical_domain
    def integrate(field,mask):
        """Use original float64 native geometry weights and axis order."""
        analysis.geometry._prepare_cached_voxel_volume(list(field.shape[:analysis.geometry.space_dim]))
        if np.asarray(analysis.geometry.cached_voxel_volume).dtype != np.float64:
            raise TypeError("Original integration weights must remain float64")
        return float(analysis.geometry.integrate(np.where(mask,field,0.)))*1000
    gas=eligible & partition["gas_known"]
    result = dict(total_mass_g=integrate(maps["mass_total"],eligible),gas_mass_g=integrate(maps["mass_gas"],gas),gas_water_mass_g=integrate(maps["mass_aqueous"],gas))
    return scoped_result(result, partition.get("analysis_scope"))

@dataclass(frozen=True)
class EstimateGeometry:
    """Measured exclusion-estimate geometry, with undefined support explicit."""
    x_m: np.ndarray
    y_m: np.ndarray
    depth_m: np.ndarray
    porosity: np.ndarray
    swi: np.ndarray
    facies_for_geometry: np.ndarray
    valid: np.ndarray
    depth_nearest_outside_hull: np.ndarray
    porosity_from_column: np.ndarray
    source_binding: dict


def _coordinates(x_m, y_m):
    """Require finite registered metre grids with descending native height."""
    x, y = np.asarray(x_m, np.float64), np.asarray(y_m, np.float64)
    if x.ndim != 2 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite source-bound physical x/y grids are required")
    if np.any(np.diff(y, axis=0) >= 0):
        raise ValueError("Physical height must decrease down each image column; no automatic axis flip")
    return x, y


def _prepared_build_estimate(x_m, y_m, facies, depth_points, fields, source_binding,
                             swi_by_facies):
    """Reuse the explicit Build preparation without filling excluded material."""
    from .n1_geometry import _mask
    expected = {"x_m", "y_m", "facies", "porosity", "swi", "depth_m", "valid",
                "depth_nearest_outside_hull", "analysis_domain", "reporting_domain",
                "material_supported_mask", "excluded_masks", "depth_points", "source_binding"}
    if not isinstance(fields, dict) or set(fields) != expected:
        raise ValueError("Exact already prepared Build geometry fields are required")
    x, y = _coordinates(x_m, y_m)
    labels, points = np.asarray(facies), np.asarray(depth_points, np.float64)
    if (labels.shape != x.shape or not np.issubdtype(labels.dtype, np.integer)
            or points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
            or not np.isfinite(points).all() or np.any(points[:, 2] <= 0)
            or not np.array_equal(x, fields["x_m"]) or not np.array_equal(y, fields["y_m"])
            or not np.array_equal(labels, fields["facies"])
            or labels.dtype != np.asarray(fields["facies"]).dtype
            or not np.array_equal(points, fields["depth_points"])):
        raise ValueError("Prepared Build geometry differs from the exact frame/depth inputs")
    if (not isinstance(fields["source_binding"], dict)
            or source_binding is not None and source_binding != fields["source_binding"]):
        raise ValueError("Prepared Build geometry has a different source binding")
    shape = x.shape
    analysis = _mask(fields["analysis_domain"], shape, "Prepared Build analysis domain")
    reporting = _mask(fields["reporting_domain"], shape, "Prepared Build reporting domain")
    material = _mask(fields["material_supported_mask"], shape, "Prepared Build material support")
    valid = _mask(fields["valid"], shape, "Prepared Build known geometry")
    outside = _mask(fields["depth_nearest_outside_hull"], shape, "Prepared depth hull fallback")
    if np.any((analysis | material) & ~reporting) or not isinstance(fields["excluded_masks"], dict):
        raise ValueError("Prepared Build masks differ from the full reporting domain")
    covered = np.zeros(shape, bool)
    for name, value in fields["excluded_masks"].items():
        mask = _mask(value, shape, "Prepared Build exclusion")
        if (not isinstance(name, str) or not name
                or np.any(mask & (~reporting | analysis | covered))):
            raise ValueError("Prepared Build exclusions must stay disjoint and outside analysis")
        covered |= mask
    depth, phi, swi = (np.asarray(fields[name]) for name in ("depth_m", "porosity", "swi"))
    if (any(value.shape != shape for value in (depth, phi, swi))
            or depth.dtype != np.float64 or phi.dtype.kind not in "fiu" or swi.dtype.kind not in "fiu"
            or not np.isfinite(depth).all() or np.any(depth <= 0)
            or np.any(~np.isfinite(phi[material])) or np.any(phi[material] < 0)
            or np.any(phi[material] > 1) or np.isinf(swi).any()
            or np.any(np.isfinite(swi) & ((swi < 0) | (swi > 1)))
            or np.any(material & (phi == 0) & ~covered)):
        raise ValueError("Prepared Build properties no longer match explicit physical support")
    known = analysis & material & np.isfinite(phi) & (phi > 0)
    if not np.array_equal(valid, known):
        raise ValueError("Prepared Build known support cannot be expanded or imputed")
    if swi.dtype.kind != "f":
        raise ValueError("Prepared Swi must preserve explicit floating unknown values")
    canonical_swi = np.full(shape, np.nan, swi.dtype)
    for label, value in swi_by_facies.items():
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Measured Swi must be in [0,1]")
        canonical_swi[labels == int(label)] = value
    if not np.array_equal(swi, canonical_swi, equal_nan=True):
        raise ValueError("Prepared Build Swi differs from the unchanged optical response recipe")
    return EstimateGeometry(x.copy(), y.copy(), depth.copy(), phi.copy(), swi.copy(),
        labels.copy(), valid.copy(), outside.copy(), np.zeros(shape, bool),
        deepcopy(fields["source_binding"]))


def build_estimate_geometry(x_m, y_m, facies, depth_points, *, porosity_by_facies,
        swi_by_facies, source_binding=None, chunk_pixels=250000, prepared_fields=None):
    """Linear measured depth, nearest outside hull, nearest sand in one column.

    Missing sand in a column remains explicitly undefined. No neighbouring
    column, maximum depth or maximum porosity is substituted into an estimate.
    Explicit Build preparation instead retains its exact fields/support and
    performs no column borrowing or repeated depth interpolation.
    """
    if prepared_fields is not None:
        return _prepared_build_estimate(x_m, y_m, facies, depth_points,
                                        prepared_fields, source_binding, swi_by_facies)
    x, y = _coordinates(x_m, y_m)
    labels = np.asarray(facies)
    points = np.asarray(depth_points, np.float64)
    if (labels.shape != x.shape or not np.issubdtype(labels.dtype, np.integer)
            or points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3
            or not np.isfinite(points).all() or np.any(points[:, 2] <= 0)):
        raise ValueError("Registered facies and measured x/y/depth triples are required")
    linear, nearest = LinearNDInterpolator(points[:, :2], points[:, 2]), cKDTree(points[:, :2])
    depth, outside = np.empty(x.size, np.float64), np.zeros(x.size, bool)
    for start in range(0, x.size, chunk_pixels):
        stop = min(x.size, start + chunk_pixels)
        query = np.column_stack((x.ravel()[start:stop], y.ravel()[start:stop]))
        values = np.asarray(linear(query), np.float64)
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = points[nearest.query(query[bad])[1], 2]
        depth[start:stop], outside[start:stop] = values, bad
    phi, swi = np.full(x.shape, np.nan), np.full(x.shape, np.nan)
    for label, value in porosity_by_facies.items():
        if int(label) == 0:
            continue
        if not np.isfinite(value) or not 0 < float(value) <= 1:
            raise ValueError("Measured sand porosity must be positive and no greater than one")
        phi[labels == int(label)] = float(value)
    for label, value in swi_by_facies.items():
        if int(label) == 0:
            continue
        if not np.isfinite(value) or not 0 <= float(value) <= 1:
            raise ValueError("Measured sand Swi must lie in [0,1]")
        swi[labels == int(label)] = float(value)
    sand = np.isfinite(phi)
    previous, following = _column_neighbours(sand)
    cols = np.arange(x.shape[1])[None, :]
    prev_y = y[np.maximum(previous, 0), cols]
    next_y = y[np.minimum(following, x.shape[0]-1), cols]
    prev_dist = np.where(previous >= 0, np.abs(prev_y-y), np.inf)
    next_dist = np.where(following < x.shape[0], np.abs(next_y-y), np.inf)
    chosen = np.where(prev_dist <= next_dist, previous, following)
    available = (chosen >= 0) & (chosen < x.shape[0])
    donor = np.clip(chosen, 0, x.shape[0]-1)
    filled = ~sand & available
    phi[filled] = phi[donor, cols][filled]
    swi[filled] = swi[donor, cols][filled]
    geometry_facies = labels.copy()
    geometry_facies[filled] = labels[donor, cols][filled]
    depth = depth.reshape(x.shape)
    valid = np.isfinite(phi) & np.isfinite(depth) & (depth > 0)
    return EstimateGeometry(x, y, depth, phi, swi, geometry_facies, valid,
        outside.reshape(x.shape), filled, dict(source_binding or {}))


def nearest_main_above(main_positive, y_m, *, max_distance_m=MAX_COLUMN_DISTANCE_M):
    """Indices of the nearest strictly higher main CO2 pixel in the same column."""
    mask, y = np.asarray(main_positive, bool), np.asarray(y_m, np.float64)
    if mask.shape != y.shape or y.ndim != 2 or np.any(np.diff(y, axis=0) >= 0):
        raise ValueError("Main mask must share a downward-decreasing physical-height grid")
    if max_distance_m != MAX_COLUMN_DISTANCE_M:
        raise ValueError("The column estimate distance is fixed at 20 mm")
    previous, _ = _column_neighbours(mask)
    previous = np.vstack((np.full((1, mask.shape[1]), -1, np.int32), previous[:-1]))
    cols = np.arange(mask.shape[1])[None, :]
    distance = y[np.maximum(previous, 0), cols]-y
    available = (previous >= 0) & (distance > 0) & (distance <= max_distance_m + 1e-12)
    return previous, available, np.where(available, distance, np.nan)


def _parts(frame, missing):
    """Partition missing support using original sorted exclusion masks."""
    parts, covered = {}, np.zeros(missing.shape, bool)
    for name, mask in sorted((frame.excluded_masks or {}).items()):
        selection = missing & mask & ~covered
        if name != ANALYSIS_GAP:
            parts[name] = selection
        covered |= mask
    parts["other_not_mass_eligible"] = missing & ~frame.mass_eligible_mask & ~covered
    parts["unquantified_facies"] = missing & frame.mass_eligible_mask & ~covered
    return parts


def _weighted_median(values, weights):
    """Compute the original area-weighted median for descriptive distances."""
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    value, weight = values[valid], weights[valid]
    order = np.argsort(value, kind="stable")
    return float(value[order[min(np.searchsorted(np.cumsum(weight[order]), weight.sum()/2), len(order)-1)]])


def _native_histogram(relative, capacity, mask, backend):
    """Quantize unchanged native relative signal with full CPU64 capacities."""
    codes = np.rint(np.clip(relative[mask],0,1)*INTERVALS).astype(np.int64)
    codes,weights,audit = reduce_weighted_codes(codes,capacity[mask],backend=backend)
    return dict(native_relative=(codes/INTERVALS).tolist(),capacity_g=weights.tolist(),
        quantization_half_width=.5/INTERVALS,reduction_audit=audit)


def _pairs(own, column, solubility_ratio, source_gas, own_available, column_available, capacity, mask, backend):
    # E1 is diluted aqueous. E2 is either diluted aqueous or undiluted water in
    # a source gas pixel. Recipient gas is aggregated separately, never fitted.
    """Aggregate coupled own/column estimates without losing their pairing."""
    a1, a2, ratio, weights = own[mask], column[mask],solubility_ratio[mask],capacity[mask]
    scale = max(1., float(np.max(ratio, initial=0)))
    q1 = np.rint(np.clip(a1, 0, 1)*INTERVALS).astype(np.int64)
    q2 = np.rint(np.clip(a2,0,1)*INTERVALS).astype(np.int64)
    qr = np.rint(ratio/scale*INTERVALS).astype(np.int64)
    kinds = (own_available[mask].astype(np.int64)
        + 2*column_available[mask].astype(np.int64) + 4*source_gas[mask].astype(np.int64))
    codes = ((q1*(INTERVALS+1)+q2)*(INTERVALS+1)+qr)*8+kinds
    codes, sums, audit = reduce_weighted_codes(codes, weights, backend=backend)
    kind, pair = codes % 8, codes // 8
    qr,pair = pair%(INTERVALS+1),pair//(INTERVALS+1)
    return dict(own_native_relative=(pair//(INTERVALS+1)/INTERVALS).tolist(),
        column_native_relative=(pair%(INTERVALS+1)/INTERVALS).tolist(),
        solubility_ratio=(qr/INTERVALS*scale).tolist(),
        kind=kind.tolist(), aqueous_capacity_g=sums.tolist(),
        quantization_half_width_own=.5/INTERVALS,
        quantization_half_width_column=.5*scale/INTERVALS,
        aqueous_capacity_total_g=float(np.sum(weights)),
        ratio_scale=scale,
        reduction_audit=audit)


def physical_partition(response, *, max_depth_m, max_porosity):
    """Original main/missing measured-volume partition and capacity only."""
    frame, physics = response["frame"], response["physics"]
    shape = frame.domain.shape
    approved_gap, scope = analysis_gap_scope(frame)
    positive = frame.domain & expert_knowledge_of(frame).co2
    if approved_gap is not None:
        positive &= ~approved_gap
    known = frame.known_physical_domain
    area = np.broadcast_to(np.asarray(frame.pixel_area_m2, np.float64), shape)
    depth = np.broadcast_to(np.asarray(frame.depth_m, np.float64), shape)
    phi = np.broadcast_to(np.asarray(frame.porosity, np.float64), shape)
    volume = np.where(known, area*depth*phi, 0.)
    aq_known = known & response["quantifiable"] & (expert_knowledge_of(frame).dissolved)
    gas_known = known & np.isfinite(response["swi"]) & (expert_knowledge_of(frame).free_gas)
    eligible = frame.mass_eligible_mask & frame.domain
    main_aq, main_gas = eligible & aq_known, eligible & gas_known
    missing = positive & ~(main_aq | main_gas)
    # These maxima provide a capacity only. The measured-volume array stays zero
    # on missing geometry and is never replaced by the maximum-volume array.
    bound_volume = np.where(known, volume, area*max_depth_m*max_porosity)
    sol = np.asarray(physics.solubility_co2, np.float64)
    density = np.asarray(physics.density_gaseous_co2, np.float64)
    if (not np.isfinite(sol[positive]).all() or not np.isfinite(density[positive]).all()
            or np.any(sol[positive] < 0) or np.any(density[positive] < 0)):
        raise ValueError("Native thermodynamic capacity must be finite across the locked phase domain")
    maximum = np.where(expert_knowledge_of(frame).free_gas, np.maximum(sol, density), sol)
    parts, covered = {}, np.zeros(shape, bool)
    for name, mask in sorted((frame.excluded_masks or {}).items()):
        selection = missing & mask
        covered |= mask
        if name == ANALYSIS_GAP and scope is not None:
            parts[name] = dict(capacity_upper_g=None, phase_pixels=int((mask & expert_knowledge_of(frame).co2).sum()),
                unknown_geometry_phase_pixels=int((mask & expert_knowledge_of(frame).co2).sum()),
                pixel_count=scope["approved_analysis_gap_pixels"], phase255_pixels=scope["phase255_pixels"],
                area_cm2=None, unknown_mass_g=None, meaning="outside-analysis-unquantified")
            continue
        parts[name] = {"capacity_upper_g": float(np.sum(bound_volume[selection]*maximum[selection]))*1000,
            "phase_pixels": int(selection.sum()), "unknown_geometry_phase_pixels": int((selection & ~known).sum())}
    masks = {"other_not_mass_eligible": missing & ~eligible & ~covered,
        "unquantified_main": missing & eligible}
    for name, selection in masks.items():
        parts[name] = {"capacity_upper_g": float(np.sum(bound_volume[selection]*maximum[selection]))*1000,
            "phase_pixels": int(selection.sum()), "unknown_geometry_phase_pixels": int((selection & ~known).sum())}
    upper = sum(item["capacity_upper_g"] for item in parts.values() if item["capacity_upper_g"] is not None)
    expected = float(np.sum(bound_volume[missing]*maximum[missing]))*1000
    if not np.isclose(upper, expected, rtol=1e-12, atol=1e-12):
        raise ValueError("Missing capacity masks overlap or omit positive phase support")
    result = dict(volume=volume, bound_volume=bound_volume, main_aq=main_aq, main_gas=main_gas,
        aq_known=aq_known, gas_known=gas_known, missing=missing, solubility=sol, density=density,
        parts=parts, missing_capacity_upper_g=upper,
        affected_pore_volume_m3=float(np.sum(volume[positive])),
        affected_unknown_geometry_pixels=int((positive & ~known).sum()),
        affected_unknown_geometry_area_cm2=float(np.sum(area[positive & ~known]))*1e4)
    if scope is not None:
        result.update(quantified_missing_capacity_upper_g=upper, missing_capacity_upper_g=None)
    return scoped_result(result, scope)


def summarize_round3_response(response, geometry, *, own_colour, photo_support,
        no_outflow_evidence, light_epoch, source_binding=None, distances_m=None,
        backend="cpu", scope="frozen_evaluation", pixel_math, summary_settings):
    """Extract native main and correlated excluded estimates once per frame."""
    frame, physics = response["frame"], response["physics"]
    run=frame.run_id
    partition=physical_partition(response,max_depth_m=summary_settings["max_depth_m"],max_porosity=summary_settings["max_porosity"])
    # These full-grid partition intermediates have no later summary consumer.
    # Keep the measured volume and main masks; release only this call's copies.
    del partition["bound_volume"], partition["aq_known"], partition["gas_known"]
    parameters=dict(candidate="N1",gain=float(summary_settings["baseline_gain"]),beta=0.)
    anchor=dict(elapsed_h=float(summary_settings.get("anchor_target_h",2.25)),affected_pore_volume_m3=max(partition["affected_pore_volume_m3"],1e-20))
    main=baseline_main(response,parameters,anchor,pixel_math)
    shape = frame.domain.shape
    area = np.broadcast_to(np.asarray(frame.pixel_area_m2, np.float64), shape)
    sol, density = partition["solubility"], partition["density"]
    volume = np.where(geometry.valid, area*geometry.depth_m*geometry.porosity, 0.)
    capacity = volume*sol*1000
    missing = partition["missing"]
    is_gas = expert_knowledge_of(frame).free_gas
    gas_available = missing & is_gas & geometry.valid & np.isfinite(geometry.swi)
    fixed_gas = volume*density*(1-geometry.swi)*1000
    fixed_water = volume*sol*geometry.swi*1000
    gas_constant = np.where(gas_available, fixed_gas+fixed_water, 0.)
    gas_constant_g = float(gas_constant[gas_available].sum())
    del gas_constant
    own_fraction = np.asarray(own_colour["aqueous_relative"],np.float64)
    own_available = missing & ~is_gas & own_colour["available"] & np.asarray(photo_support, bool) & geometry.valid
    main_positive = partition["main_aq"] | partition["main_gas"]
    donor, column_available, column_distance = nearest_main_above(main_positive, geometry.y_m, max_distance_m=summary_settings["max_column_distance_m"])
    del main_positive
    cols = np.arange(shape[1])[None, :]
    donor = np.maximum(donor, 0)
    source_gas = partition["main_gas"][donor, cols]
    source_relative = response["prepared"].aqueous_relative[donor,cols].astype(np.float64)
    source_sol = sol[donor,cols]
    solubility_ratio = np.divide(source_sol,sol,out=np.zeros(shape),where=sol>0)
    del source_sol, donor, cols
    column_fraction = source_relative
    column_available &= missing & ~is_gas & geometry.valid & np.isfinite(source_relative) & np.isfinite(solubility_ratio)
    source_gas &= column_available
    pair_mask = missing & ~is_gas & (own_available | column_available)
    own_fraction = np.where(own_available, own_fraction, 0.)
    column_fraction = np.where(column_available, column_fraction, 0.)
    del source_relative
    neither = missing & ~(gas_available | own_available | column_available)
    undefined = neither & ~geometry.valid
    # The where result owns its buffer. Reuse it for the same two float64
    # multiplies, with the original operand order and no input-array mutation.
    urest = np.where(is_gas, np.maximum(sol, density), sol)
    np.multiply(volume, urest, out=urest)
    np.multiply(urest, 1000, out=urest)
    del volume
    parts = _parts(frame, missing)
    part_records = {}
    distance_arrays = {name: np.broadcast_to(np.asarray(value, np.float64), shape)
        for name, value in (distances_m or {}).items()}
    for name, mask in parts.items():
        labels = {}
        for label in np.unique(frame.facies[mask]):
            selection = mask & (frame.facies == label)
            labels[str(int(label))] = dict(co2_area_cm2=float(area[selection].sum()*10000),
                without_main_geometry_area_cm2=float(area[selection & ~frame.known_physical_domain].sum()*10000),
                estimate_geometry_undefined_area_cm2=float(area[selection & ~geometry.valid].sum()*10000),
                capacity_g=None if (selection & ~geometry.valid).any() else float(urest[selection].sum()),
                own_colour_histogram=_native_histogram(own_fraction,capacity,selection & own_available,backend),
                fixed_gas_mass_g=float(fixed_gas[selection & gas_available].sum()),
                fixed_gas_water_mass_g=float(fixed_water[selection & gas_available].sum()))
        part_records[name] = dict(by_facies=labels, co2_area_cm2=float(area[mask].sum()*10000),
            without_main_geometry_area_cm2=float(area[mask & ~frame.known_physical_domain].sum()*10000),
            geometry_extended_area_cm2=float(area[mask & geometry.valid & ~frame.known_physical_domain].sum()*10000),
            depth_nearest_outside_hull_area_cm2=float(area[mask & geometry.depth_nearest_outside_hull].sum()*10000),
            porosity_column_extended_area_cm2=float(area[mask & geometry.porosity_from_column].sum()*10000),
            capacity_g=None if (mask & ~geometry.valid).any() else float(urest[mask].sum()),
            own_colour_histogram=_native_histogram(own_fraction,capacity,mask & own_available,backend),
            fixed_gas_mass_g=float(fixed_gas[mask & gas_available].sum()),
            fixed_gas_water_mass_g=float(fixed_water[mask & gas_available].sum()),
            urest_g=None if (mask & undefined).any() else float(urest[mask & neither].sum()),
            median_distances_mm={key: None if (value := _weighted_median(array[mask],area[mask])) is None
                else value*1000 for key,array in distance_arrays.items()})
    row = dict(contract=summary_settings.get("contract", CONTRACT), run=run, frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
        light_epoch=light_epoch, injected_mass_g=frame.nominal_injected_mass_g,
        source_binding=dict(source_binding or frame.provenance), geometry_source_binding=geometry.source_binding,
        no_outflow_evidence=dict(no_outflow_evidence), scope=scope,
        main_aqueous_histogram=_native_histogram(response["prepared"].aqueous_relative,
            partition["volume"]*sol*1000,partition["main_aq"],backend),
        baseline_undiluted_main_mass_g=main["total_mass_g"],main_gas_mass_g=main["gas_mass_g"],
        main_gas_water_mass_g=main["gas_water_mass_g"], affected_pore_volume_m3=partition["affected_pore_volume_m3"],
        estimate_fixed_gas_mass_g=float(fixed_gas[gas_available].sum()),
        estimate_fixed_gas_water_mass_g=float(fixed_water[gas_available].sum()),
        estimate_gas_constant_g=gas_constant_g,
        urest_g=None if undefined.any() else float(urest[neither].sum()),
        estimate_geometry_undefined_co2_pixels=int(undefined.sum()),
        estimate_geometry_extended_area_cm2=float(area[missing & geometry.valid & ~frame.known_physical_domain].sum()*10000),
        excluded_parts=part_records, own_colour_audit=own_colour["audit"],
        own_estimate_pixels=int(own_available.sum()), column_estimate_pixels=int(column_available.sum()),
        both_estimates_pixels=int((own_available & column_available).sum()),
        fixed_gas_estimate_pixels=int(gas_available.sum()), neither_estimate_pixels=int(neither.sum()),
        column_distance_median_mm=None if not column_available.any() else float(np.nanmedian(column_distance[column_available])*1000),
        main_geometry_modified=False, main_labels_modified=False, gas_branch_fitted=False,
        paired_estimates=_pairs(own_fraction,column_fraction,solubility_ratio,source_gas,
            own_available,column_available,capacity,pair_mask,backend))
    if partition.get("analysis_scope") is not None:
        scope_record = partition["analysis_scope"]
        row["excluded_parts"][ANALYSIS_GAP] = dict(meaning="outside-analysis-unquantified",
            pixel_count=scope_record["approved_analysis_gap_pixels"], phase255_pixels=scope_record["phase255_pixels"],
            area_cm2=None, co2_area_cm2=None, unknown_mass_g=None, capacity_g=None, by_facies={})
        row = scoped_result(row, scope_record)
    return row


def interval_residual(mass, dose, upper):
    """Return the original signed residual outside a physical interval."""
    if not np.isfinite([mass, dose, upper]).all() or min(mass, upper) < 0 or dose <= 0:
        raise ValueError("A two-sided residual requires finite physical bounds and a positive dose")
    return float((mass-dose)/dose if mass > dose else (mass+upper-dose)/dose if mass+upper < dose else 0.)


def evaluate_summary(row, parameters, anchor):
    """Evaluate the literal per-pixel estimate interval at one shared beta."""
    parameters = dict(parameters)
    validate_parameters(parameters)
    if parameters["candidate"] != "N1" or not isinstance(row.get("contract"), str):
        raise ValueError("Round-three evaluation uses the unchanged native N1 response")
    gain,beta = parameters["gain"],parameters["beta"]
    factor = dilution_factor(row["elapsed_h"],row["affected_pore_volume_m3"],anchor,beta)
    paired = row["paired_estimates"]
    own,column,ratio,kind,weight = (np.asarray(paired[key],np.float64 if key != "kind" else np.int64)
        for key in ("own_native_relative","column_native_relative","solubility_ratio","kind","aqueous_capacity_g"))
    e1 = np.clip(own*gain,0,1)*factor
    e2 = ratio*np.where((kind & 4) != 0,1.,np.clip(column*gain,0,1)*factor)
    has1,has2 = (kind & 1) != 0,(kind & 2) != 0
    both = has1 & has2
    single = np.where(has1,e1,e2)
    lo = float(np.sum(weight*np.where(both,np.minimum(e1,e2),single)))+row["estimate_gas_constant_g"]
    hi = float(np.sum(weight*np.where(both,np.maximum(e1,e2),single)))+row["estimate_gas_constant_g"]
    def own_mass(hist):
        """Evaluate one saved native-relative capacity histogram."""
        return float(np.dot(np.asarray(hist["capacity_g"],np.float64),
            np.clip(np.asarray(hist["native_relative"],np.float64)*gain,0,1)))*factor
    main_aq = own_mass(row["main_aqueous_histogram"])
    main = main_aq+row["main_gas_mass_g"]+row["main_gas_water_mass_g"]
    urest = row["urest_g"]
    width = None if urest is None else hi-lo+urest
    dose = row["injected_mass_g"]
    residual = None if width is None or dose is None or dose <= 0 else interval_residual(main+lo,dose,width)
    def evaluated_part(part):
        """Attach own-colour and fixed-gas masses to one exclusion partition."""
        if part.get("meaning") == "outside-analysis-unquantified":
            return dict(part)
        own_aq = own_mass(part["own_colour_histogram"])
        gas_total = part["fixed_gas_mass_g"]+part["fixed_gas_water_mass_g"]
        return dict(part,own_colour_aqueous_mass_g=own_aq,gas_rule_total_mass_g=gas_total,
            own_colour_mass_g=own_aq+gas_total)
    parts = {name: dict(evaluated_part(part),by_facies={label:evaluated_part(value)
        for label,value in part["by_facies"].items()})
        for name,part in row.get("excluded_parts",{}).items()}
    prediction = dict(contract=row["contract"],run=row["run"],frame_index=row["frame_index"],elapsed_h=row["elapsed_h"],
        light_epoch=row.get("light_epoch"),source_binding=row.get("source_binding"),parameters=parameters,
        main_mass_g=main,total_mass_g=main,main_aqueous_nongas_mass_g=main_aq,
        aqueous_mass_g=main_aq+row["main_gas_water_mass_g"],gas_mass_g=row["main_gas_mass_g"],
        gas_water_mass_g=row["main_gas_water_mass_g"],gas_share=0. if main == 0 else row["main_gas_mass_g"]/main,
        E_lo_g=lo,E_hi_g=hi,U_rest_g=urest,W_g=width,
        interval_lower_g=main+lo,interval_upper_g=None if urest is None else main+hi+urest,
        injected_mass_g=dose,W_over_d=None if width is None or dose is None or dose <= 0 else width/dose,
        signed_interval_residual=residual,delta=None if residual is None else abs(residual),
        dilution_factor=factor,estimate_gas_mass_g=row["estimate_fixed_gas_mass_g"],
        estimate_gas_water_mass_g=row["estimate_fixed_gas_water_mass_g"],
        estimate_geometry_extended_area_cm2=row["estimate_geometry_extended_area_cm2"],excluded_parts=parts,
        summary_quantization_bound_g=(paired["aqueous_capacity_total_g"]*.5/INTERVALS*
            (gain*(1+paired["ratio_scale"])+paired["ratio_scale"])
            +sum(row["main_aqueous_histogram"]["capacity_g"])*gain*.5/INTERVALS),
        main_geometry_modified=False,
        main_labels_modified=False,map_mass_rescaling=False,overflow_in_objective=False)
    return scoped_result(prediction, row.get("analysis_scope"))


def exact_prediction(row, result, parameters, anchor, *, omitted=None):
    """Combine exact retained M with original E and corrected-map U_rest."""
    prediction=evaluate_summary(row,parameters,anchor)
    SPEC_SHA256=row.get("round4_spec_sha256")
    AMENDMENT_SHA256=row.get("round4_amendment_sha256")
    main = float(result["total_mass_g"])
    urest = row["urest_g"]
    lo = main+prediction["E_lo_g"]
    hi = None if urest is None else main+prediction["E_hi_g"]+urest
    dose = prediction["injected_mass_g"]
    width = None if urest is None else prediction["E_hi_g"]-prediction["E_lo_g"]+urest
    residual = None if width is None or dose is None or dose <= 0 else interval_residual(lo, dose, width)
    prediction.update(main_mass_g=main, total_mass_g=main,
        aqueous_mass_g=result["aqueous_mass_g"], gas_mass_g=result["gas_mass_g"],
        gas_share=0. if main == 0 else result["gas_mass_g"]/main,
        U_rest_g=urest, W_g=width, W_over_d=None if width is None or dose is None or dose <= 0 else width/dose,
        interval_lower_g=lo, interval_upper_g=hi, signed_interval_residual=residual,
        delta=None if residual is None else abs(residual), exact_native_main_replay=True,
        selected_target_h=row["selected_target_h"], scope=row["scope"],
        round4_spec_sha256=SPEC_SHA256, precision=result["precision"], heldout_fit_run=omitted,
        evaluation_is_leave_one_run_out=omitted is not None, main_geometry_modified=True,
        amendment_sha256=AMENDMENT_SHA256)
    def scalar_part(part):
        """Retain scalar/by-facies metadata while omitting histogram payloads."""
        return {key: ({label:scalar_part(value) for label,value in item.items()}
                      if key == "by_facies" else item)
                for key,item in part.items() if key != "own_colour_histogram"}
    prediction["excluded_parts"] = {name:scalar_part(part)
                                    for name,part in prediction.get("excluded_parts", {}).items()}
    return scoped_result(prediction, row.get("analysis_scope") or result.get("analysis_scope"))
