"""Frozen categorical phase rules on an explicitly supplied native grid.

Only decision fields are changed. Source geometry, fitted heads and camera-time
history are supplied by callers. Rule order, component connectivity, nominal
physical distances and array casts retain their original definitions.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field, InitVar, asdict
import time
import hashlib
import json
import cv2
import numpy as np
from scipy import ndimage as cpu_ndimage
from scipy.special import logit
from . import heads, regional
from .metrics import validate_pixel_area, area_filtered_gas, array_identity

@dataclass
class TopRuleResult:
    """Labels, removed masks, previous categories and physical top-rule diagnostics."""
    labels: np.ndarray
    after_floor_removal: np.ndarray
    removed_top: np.ndarray
    previous_top_labels: np.ndarray
    qa: dict

def _select_backend(requested='auto', *, parameters: Mapping):
    """Apply the original select backend operation with explicit rule parameters."""
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('frozen backend must be auto, cuda or cpu')
    if requested == 'cpu':
        return 'cpu'
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            cp.zeros(1).sum().item()
            return 'cuda'
    except (ImportError, RuntimeError, OSError) as error:
        if requested == 'cuda':
            raise RuntimeError('Requested CUDA backend is unavailable') from error
    if requested == 'cuda':
        raise RuntimeError('Requested CUDA backend is unavailable')
    return 'cpu'

def _top_mask(labels, backend, *, parameters: Mapping):
    """Apply the original top mask operation with explicit rule parameters."""
    BAND_PIXELS = parameters['top']['band_pixels']
    COLUMN_GATE_PIXELS = parameters['top']['column_gate_pixels']
    if backend == 'cuda':
        import cupy as xp
        from cupyx.scipy.ndimage import label
        source = xp.asarray(labels)
    else:
        xp = np
        source = labels
    domain = source != parameters['literal_values']['top_mask_int_255']
    has = domain.any(axis=0)
    top = xp.where(has, domain.argmax(axis=0), labels.shape[0]).astype(xp.int32)
    median = xp.median(top[has]) if bool(has.any()) else xp.asarray(labels.shape[0])
    acting = has & (xp.abs(top - median) <= COLUMN_GATE_PIXELS)
    depth = xp.arange(labels.shape[0], dtype=xp.int32)[:, None] - top[None, :]
    co2 = (source == 1) | (source == 2)
    if backend == 'cuda':
        components, count = label(co2, structure=xp.ones(tuple(parameters['literal_values']['top_mask_structure_hw']), dtype=xp.int8))
    else:
        count_plus_one, components = cv2.connectedComponents(co2.astype(np.uint8), connectivity=parameters['literal_values']['top_mask_int_8'])
        count = count_plus_one - 1
    inside = co2 & acting[None, :] & (depth >= 0) & (depth <= BAND_PIXELS)
    deeper = co2 & (depth > BAND_PIXELS)

    def present(mask):
        selected = components[mask]
        return xp.bincount(selected, minlength=int(count) + 1) > 0 if selected.size else xp.zeros(int(count) + 1, dtype=bool)
    in_ids = present(inside)
    deep_ids = present(deeper)
    remove_ids = in_ids & ~deep_ids
    remove_ids[0] = False
    removed = co2 & remove_ids[components]
    audit = dict(removed_top_components=int(remove_ids.sum()), excluded_columns=int((~acting).sum()), acting_columns=int(acting.sum()), top_median_row=float(median))
    if backend == 'cuda':
        return (xp.asnumpy(removed), xp.asnumpy(top), xp.asnumpy(acting), audit)
    return (removed, top, acting, audit)

def apply_top_rule(p1_labels, base_labels, floor_mask, pixel_area_mm2, *, backend=None, parameters: Mapping):
    """Restore floor fill, then remove detached CO2 components near the sand top.

    Removal requires contact with the acting top band and no CO2 deeper than
    that band anywhere in the component. parameters supplies nominal mm/pixel,
    pixel distances and connectivity. Measured pixel_area_mm2 weights reported
    areas only. The original gas, component-closure and outside-mask checks
    remain exact."""
    if backend is None:
        backend = parameters['literal_values']['apply_top_rule_default_backend']
    NOMINAL_AREA_MM2 = parameters['nominal_pixel_area_mm2']
    MM_PER_PIXEL = parameters['nominal_mm_per_pixel']
    BAND_PIXELS = parameters['top']['band_pixels']
    wall, cpu = (time.perf_counter(), time.process_time())
    p1, base, floor, area = map(np.asarray, (p1_labels, base_labels, floor_mask, pixel_area_mm2))
    if p1.dtype != np.uint8 or p1.ndim != 2 or base.dtype != np.uint8 or (base.shape != p1.shape) or (floor.dtype != bool) or (floor.shape != p1.shape) or (area.shape != p1.shape) or (not np.isfinite(area).all()) or np.any(area < 0) or (not set(np.unique(p1)).issubset({0, 1, 2, parameters['literal_values']['apply_top_rule_int_255']})):
        raise ValueError('frozen requires exact categorical labels, source-bound flags and measured areas')
    if np.any(floor & ((p1 != 1) | (base != 0))):
        raise ValueError('Floor provenance is not exclusively original no-CO2 to aqueous fill')
    after_a = p1.copy()
    after_a[floor] = base[floor]
    union_exact = np.array_equal(np.isin(after_a, (1, 2)), np.isin(base, (1, 2)))
    gas_exact = np.array_equal(after_a == 2, p1 == 2)
    if not union_exact or not gas_exact:
        raise ValueError('Floor removal failed the exact BASE union / P1 gas identity')
    actual = _select_backend(backend, parameters=parameters)
    gpu_seconds = 0.0
    if actual == 'cuda':
        import cupy as cp
        begin, end = (cp.cuda.Event(), cp.cuda.Event())
        begin.record()
    removed, top, acting, audit = _top_mask(after_a, actual, parameters=parameters)
    if actual == 'cuda':
        end.record()
        end.synchronize()
        gpu_seconds = float(cp.cuda.get_elapsed_time(begin, end)) / parameters['literal_values']['apply_top_rule_int_1000']
    final = after_a.copy()
    final[removed] = 0
    previous = np.where(removed, after_a, 0).astype(np.uint8)
    depth = np.arange(p1.shape[0], dtype=np.int32)[:, None] - top[None, :]
    changed = final != after_a
    closed = not np.any(cv2.dilate(removed.astype(np.uint8), np.ones(tuple(parameters['literal_values']['apply_top_rule_structure_hw']), np.uint8)).astype(bool) & np.isin(after_a, (1, 2)) & ~removed)
    components = cv2.connectedComponents(removed.astype(np.uint8), connectivity=parameters['literal_values']['apply_top_rule_int_8'])[0] - 1
    exact_outside = np.array_equal(final[~(floor | removed)], p1[~(floor | removed)])
    passed = np.array_equal(changed, removed) and closed and (components == audit['removed_top_components']) and (not np.any(removed & (depth > BAND_PIXELS))) and exact_outside and np.array_equal(final == parameters['literal_values']['apply_top_rule_int_255'], p1 == parameters['literal_values']['apply_top_rule_int_255'])
    if not passed:
        raise ValueError('frozen fixed component/flag/outside-mask QA failed')
    measure = lambda mask: float(area[mask].sum(dtype=np.float64) / parameters['literal_values']['apply_top_rule_int_100'])
    nominal = lambda mask: float(np.count_nonzero(mask) * NOMINAL_AREA_MM2 / parameters['literal_values']['apply_top_rule_int_100'])
    remains = np.isin(final, (1, 2)) & acting[None, :] & (depth >= 0)
    qa = dict(audit, all_pass=True, after_a_co2_union_equals_base=union_exact, after_a_gas_equals_p1=gas_exact, changed_pixels_exactly_top_flags=True, removed_components_closed=True, removed_components_within_fixed_band=True, labels_outside_masks_identical=True, rig_domain_unchanged=True, removed_floor_pixels=int(floor.sum()), removed_top_pixels=int(removed.sum()), removed_floor_cm2=measure(floor), removed_top_aqueous_cm2=measure(removed & (after_a == 1)), removed_top_gas_cm2=measure(removed & (after_a == 2)), nominal_removed_top_aqueous_cm2=nominal(removed & (after_a == 1)), nominal_removed_top_gas_cm2=nominal(removed & (after_a == 2)), remaining_co2_60mm_cm2=measure(remains & (depth <= BAND_PIXELS)), remaining_co2_90mm_cm2=measure(remains & (depth <= round(parameters['literal_values']['apply_top_rule_int_90'] / MM_PER_PIXEL))), retained_60mm_reason='connected to a component deeper than 60 mm' if np.any(remains & (depth <= BAND_PIXELS)) else 'none', backend=actual, gpu_seconds=gpu_seconds, cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall, training=False, inference=False)
    return TopRuleResult(final, after_a, removed, previous, qa)

@dataclass
class FloorRuleResult:
    """Separate arrays and diagnostics for ordered floor, contact and envelope stages."""
    labels: np.ndarray
    step_masks: dict[str, np.ndarray]
    previous_labels: dict[str, np.ndarray]
    fill_reason_a: np.ndarray
    fill_reason_b: np.ndarray
    qa: dict

def _floor_bool(value, shape, name, *, parameters: Mapping):
    """Apply the original floor bool operation with explicit rule parameters."""
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f'{name} must be a bool array on the exact phase grid')
    return array

def _floor_labels(value, *, parameters: Mapping):
    """Apply the original floor labels operation with explicit rule parameters."""
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 2 or (not all(array.shape)):
        raise ValueError('Parent labels must be a nonempty two-dimensional uint8 grid')
    if not set(np.unique(array)).issubset({0, 1, 2, parameters['literal_values']['floor_labels_int_255']}):
        raise ValueError('Parent labels contain an unsupported category')
    return array

def _modules(backend, *, parameters: Mapping):
    """Apply the original modules operation with explicit rule parameters."""
    if backend == 'cuda':
        import cupy as xp
        import cupyx.scipy.ndimage as ndi
        return (xp, ndi)
    return (np, cpu_ndimage)

def _numpy(value, xp):
    """Apply the original numpy operation with explicit rule parameters."""
    return xp.asnumpy(value) if xp is not np else np.asarray(value).copy()

def _is_gpu_failure(error):
    """Apply the original is gpu failure operation with explicit rule parameters."""
    return type(error).__module__.startswith(('cupy', 'cuda'))

def _edt(mask, xp, ndi, *, parameters: Mapping):
    """Apply the original edt operation with explicit rule parameters."""
    return ndi.distance_transform_edt(mask, float64_distances=True) if xp is not np else ndi.distance_transform_edt(mask)

def _dilate(mask, radius_pixels, xp, ndi, *, parameters: Mapping):
    """Apply the original dilate operation with explicit rule parameters."""
    if not bool(mask.any()):
        return xp.zeros(mask.shape, dtype=bool)
    return _edt(~mask, xp, ndi, parameters=parameters) <= radius_pixels

def _label(mask, xp, ndi, *, parameters: Mapping):
    """Apply the original label operation with explicit rule parameters."""
    if xp is not np:
        labels, count = ndi.label(mask, structure=xp.ones(tuple(parameters['literal_values']['label_structure_hw']), dtype=xp.int8))
        return (labels, int(count))
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=parameters['literal_values']['label_int_8'])
    return (labels, int(count) - 1)

def apply_floor_boundary(labels, below_floor_mask, *, parameters: Mapping):
    """Mark the supplied below-sand-floor mask as unresolved.

    The Boolean mask must share the uint8 label grid. parameters supplies the
    unresolved category; every other category is copied unchanged."""
    source = _floor_labels(labels, parameters=parameters)
    below = _floor_bool(below_floor_mask, source.shape, 'below_floor_mask', parameters=parameters)
    result = source.copy()
    result[below] = parameters['literal_values']['apply_floor_boundary_int_255']
    return result

def _envelopes_core(union1, union2, actual, *, parameters: Mapping):
    """Apply the original envelopes core operation with explicit rule parameters."""
    MM_PER_PIXEL = parameters['nominal_mm_per_pixel']
    xp, ndi = _modules(actual, parameters=parameters)
    unions = (xp.asarray(union1), xp.asarray(union2))
    return tuple((_numpy(_dilate(mask, parameters['literal_values']['envelopes_core_float_5_0'] / MM_PER_PIXEL, xp, ndi, parameters=parameters), xp) for mask in unions))

def build_envelopes(union1, union2, backend=None, *, parameters: Mapping):
    """Expand two inclusive gas unions by an explicit physical Euclidean disc.

    The distance transform retains float64 distances. parameters supplies
    radius in millimetres and nominal mm/pixel. An empty union has an empty
    envelope; the caller supplies the original inclusive history selection."""
    if backend is None:
        backend = parameters['literal_values']['build_envelopes_default_backend']
    MM_PER_PIXEL = parameters['nominal_mm_per_pixel']
    first = np.asarray(union1)
    if first.dtype != bool or first.ndim != 2 or (not all(first.shape)):
        raise ValueError('Gas unions must be nonempty bool grids')
    second = _floor_bool(union2, first.shape, 'union2', parameters=parameters)
    wall, cpu = (time.perf_counter(), time.process_time())
    actual = _select_backend(backend, parameters=parameters)
    gpu, failure = (0.0, None)
    try:
        if actual == 'cuda':
            import cupy as cp
            begin, end = (cp.cuda.Event(), cp.cuda.Event())
            begin.record()
        env1, env2 = _envelopes_core(first, second, actual, parameters=parameters)
        if actual == 'cuda':
            end.record()
            end.synchronize()
            gpu = float(cp.cuda.get_elapsed_time(begin, end)) / parameters['literal_values']['build_envelopes_float_1000_0']
    except Exception as error:
        if actual != 'cuda' or not _is_gpu_failure(error):
            raise
        failure = f'{type(error).__name__}: {error}'
        actual = 'cpu'
        env1, env2 = _envelopes_core(first, second, actual, parameters=parameters)
    audit = dict(backend=actual, requested_backend=backend, cuda_fallback_reason=failure, union1_pixels=int(first.sum()), union2_pixels=int(second.sum()), envelope1_pixels=int(env1.sum()), envelope2_pixels=int(env2.sum()), radius_mm=parameters['literal_values']['build_envelopes_float_5_0'], radius_pixels=parameters['literal_values']['build_envelopes_float_5_0'] / MM_PER_PIXEL, empty_union_envelope_policy='empty, not an implicit outside-grid source', cpu_seconds=time.process_time() - cpu, gpu_seconds=gpu, wall_seconds=time.perf_counter() - wall)
    return (env1, env2, audit)

def _floor_core(parent, below, support, carryover, l1, env1, env2, elapsed_h, frame_index, u1_index, u2_index, corrected_domain, actual, *, parameters: Mapping):
    """Apply the original floor core operation with explicit rule parameters."""
    MM_PER_PIXEL = parameters['nominal_mm_per_pixel']
    FLOOR_BAND_PIXELS = parameters['floor']['floor_band_pixels']
    SHORT_FILL_PIXELS = parameters['contact']['short_fill_limit_pixels']
    COLUMN_GATE_PIXELS = parameters['floor']['column_gate_pixels']
    xp, ndi = _modules(actual, parameters=parameters)
    source = xp.asarray(parent).copy()
    height, width = source.shape
    rows = xp.arange(height, dtype=xp.int32)[:, None]
    columns = xp.arange(width, dtype=xp.int32)
    below = xp.asarray(below)
    mask_a = below & (source != parameters['literal_values']['floor_core_int_255'])
    prior_a = xp.where(mask_a, source, 0).astype(xp.uint8)
    source[below] = parameters['literal_values']['floor_core_int_255']
    domain = source != parameters['literal_values']['floor_core_int_255'] if corrected_domain is None else xp.asarray(corrected_domain) & ~below
    has = domain.any(axis=0)
    floor = xp.where(has, height - 1 - domain[::-1].argmax(axis=0), -1).astype(xp.int32)
    median = float(xp.median(floor[has])) if bool(has.any()) else None
    acting = has & (xp.abs(floor - (median if median is not None else -1)) <= COLUMN_GATE_PIXELS)
    floor_height = floor[None, :] - rows
    co2 = (source == 1) | (source == 2)
    dilated = _dilate(co2, parameters['literal_values']['floor_core_float_1_5'] / MM_PER_PIXEL, xp, ndi, parameters=parameters)
    components, count = _label(dilated, xp, ndi, parameters=parameters)
    touches = co2 & acting[None, :] & (floor_height >= 0) & (floor_height <= FLOOR_BAND_PIXELS)
    deeper = co2 & (floor_height > FLOOR_BAND_PIXELS)

    def present(mask):
        selected = components[mask]
        return xp.bincount(selected, minlength=count + 1) > 0 if selected.size else xp.zeros(count + 1, dtype=bool)
    remove_ids = present(touches) & ~present(deeper)
    remove_ids[0] = False
    mask_b = co2 & remove_ids[components]
    prior_b = xp.where(mask_b, source, 0).astype(xp.uint8)
    kept_co2 = co2 & ~mask_b
    removed_ids = _numpy(xp.flatnonzero(remove_ids), xp).tolist()
    group_distances = []
    if removed_ids and bool(kept_co2.any()):
        distances = _edt(~kept_co2, xp, ndi, parameters=parameters) * MM_PER_PIXEL
        minima = ndi.minimum(xp.where(mask_b, distances, xp.inf), labels=components, index=xp.asarray(removed_ids))
        group_distances = _numpy(minima, xp).tolist()
    source[mask_b] = 0
    co2_after_b = (source == 1) | (source == 2)
    lowest = xp.max(xp.where(co2_after_b, rows, -1), axis=0)
    spans = floor - lowest
    trigger = source[xp.maximum(lowest, 0), columns]
    eligible = acting & (lowest >= 0) & (spans > 0) & (trigger == 1)
    run = (rows > lowest[None, :]) & (rows <= floor[None, :])
    intact = ~xp.any(run & ((source != 0) | ~xp.asarray(support)), axis=0)
    short = eligible & intact & (spans <= SHORT_FILL_PIXELS)
    colour = xp.zeros(width, dtype=bool)
    if carryover is not None and elapsed_h > 0:
        tolerated = ndi.binary_dilation(xp.asarray(carryover), structure=xp.ones(tuple(parameters['literal_values']['floor_core_structure_hw']), dtype=bool), iterations=1, border_value=0)
        colour = eligible & intact & ~xp.any(run & ~tolerated, axis=0)
    if elapsed_h <= 0:
        short[:] = False
        colour[:] = False
    reason_a = run & short[None, :]
    reason_b = run & (colour & ~short)[None, :]
    mask_c = reason_a | reason_b
    prior_c = xp.where(mask_c, source, 0).astype(xp.uint8)
    source[mask_c] = 1
    gas = source == 2
    mask_d = xp.zeros(source.shape, dtype=bool)
    if frame_index > u1_index:
        mask_d |= gas & xp.asarray(l1) & ~xp.asarray(env1)
    if frame_index > u2_index:
        mask_d |= gas & ~xp.asarray(env2)
    prior_d = xp.where(mask_d, source, 0).astype(xp.uint8)
    source[mask_d] = 1
    diagnostics = dict(floor_median_row=median, acting_columns=int(acting.sum()), excluded_columns=int((~acting).sum()), co2_connectivity_groups=count, floor_connectivity_removed_groups=len(removed_ids), floor_connectivity_removed_group_min_distance_to_kept_co2_mm=group_distances, floor_connectivity_distance_status='known' if group_distances else 'unknown: no kept CO2' if removed_ids else 'not applicable: no removed group', floor_connectivity_distance_median_mm=float(np.median(group_distances)) if group_distances else None, floor_connectivity_distance_max_mm=float(np.max(group_distances)) if group_distances else None, floor_connectivity_removed_gas_pixels=int((mask_b & (prior_b == 2)).sum()), fill_reason_a_columns=int(short.sum()), fill_reason_b_columns=int((colour & ~short).sum()), carryover_mask_known=carryover is not None, filled_under_gas_pixels=int((mask_c & (trigger == 2)[None, :]).sum()), fill_invalid_trigger_columns=int((mask_c.any(axis=0) & (trigger != 1)).sum()), fill_unsupported_pixels=int((mask_c & ~xp.asarray(support)).sum()), preinjection_fill_pixels=int(mask_c.sum()) if elapsed_h <= 0 else 0)
    arrays = (source, mask_a, mask_b, mask_c, mask_d, prior_a, prior_b, prior_c, prior_d, reason_a, reason_b)
    return (tuple((_numpy(array, xp) for array in arrays)), diagnostics)

def apply_floor_round(parent_labels, below_floor_mask, supported_mask, carryover_mask_orNone, l1_zone, env1, env2, elapsed_h, frame_index, u1_index, u2_index, pixel_area_mm2, backend=None, corrected_domain_mask=None, *, parameters: Mapping):
    """Apply sand-floor exclusion, connectivity removal, contact fill and envelopes.

    Stages execute in that order. Nominal distances in parameters define the
    floor band, connectivity disc and short contact span; measured areas
    weight reports only. Supported aqueous contact runs can fill after
    injection, with short-run reason preceding carryover. Gas is restricted
    only strictly after each inclusive envelope frame index. No minimum-area
    filter is repeated."""
    if backend is None:
        backend = parameters['literal_values']['apply_floor_round_default_backend']
    parent = _floor_labels(parent_labels, parameters=parameters)
    shape = parent.shape
    below = _floor_bool(below_floor_mask, shape, 'below_floor_mask', parameters=parameters)
    supported = _floor_bool(supported_mask, shape, 'supported_mask', parameters=parameters)
    carryover = None if carryover_mask_orNone is None else _floor_bool(carryover_mask_orNone, shape, 'carryover_mask', parameters=parameters)
    corrected_domain = None if corrected_domain_mask is None else _floor_bool(corrected_domain_mask, shape, 'corrected_domain_mask', parameters=parameters)
    l1, first, second = (_floor_bool(value, shape, name, parameters=parameters) for value, name in ((l1_zone, 'l1_zone'), (env1, 'env1'), (env2, 'env2')))
    area = np.asarray(pixel_area_mm2)
    if area.ndim not in (0, 2) or (area.ndim == 2 and area.shape != shape) or (not np.issubdtype(area.dtype, np.number)) or (not np.isfinite(area).all()) or np.any(area < 0):
        raise ValueError('Pixel areas must be finite nonnegative weights on the exact phase grid')
    if not np.isfinite(elapsed_h) or any((type(value) is not int or value < 0 for value in (frame_index, u1_index, u2_index))) or u2_index < u1_index:
        raise ValueError('Frame time and inclusive injection-end indices must be valid and ordered')
    wall, cpu = (time.perf_counter(), time.process_time())
    actual, gpu_seconds, failure = (_select_backend(backend, parameters=parameters), 0.0, None)
    arguments = (parent, below, supported, carryover, l1, first, second, float(elapsed_h), frame_index, u1_index, u2_index, corrected_domain)
    try:
        if actual == 'cuda':
            import cupy as cp
            begin, end = (cp.cuda.Event(), cp.cuda.Event())
            begin.record()
        arrays, audit = _floor_core(*arguments, actual, parameters=parameters)
        if actual == 'cuda':
            end.record()
            end.synchronize()
            gpu_seconds = float(cp.cuda.get_elapsed_time(begin, end)) / parameters['literal_values']['apply_floor_round_float_1000_0']
    except Exception as error:
        if actual != 'cuda' or not _is_gpu_failure(error):
            raise
        failure = f'{type(error).__name__}: {error}'
        actual = 'cpu'
        arrays, audit = _floor_core(*arguments, actual, parameters=parameters)
    labels, *parts = arrays
    masks = dict(zip('ABCD', parts[:parameters['literal_values']['apply_floor_round_int_4']]))
    previous = dict(zip('ABCD', parts[parameters['literal_values']['apply_floor_round_int_4']:parameters['literal_values']['apply_floor_round_int_8']]))
    reason_a, reason_b = parts[parameters['literal_values']['apply_floor_round_int_8']:]
    overlaps = {a + b: int(np.count_nonzero(masks[a] & masks[b])) for i, a in enumerate('ABCD') for b in 'ABCD'[i + 1:]}
    flagged = np.logical_or.reduce(list(masks.values()))
    changed = parent != labels
    overlap_changed = {a + b: int(np.count_nonzero(masks[a] & masks[b] & changed)) for i, a in enumerate('ABCD') for b in 'ABCD'[i + 1:]}
    gas_changes = (parent == 2) != (labels == 2)
    co2_changes = np.isin(parent, (1, 2)) != np.isin(labels, (1, 2))

    def measure(mask):
        return float(np.count_nonzero(mask) * area) / parameters['literal_values']['apply_floor_round_float_100_0'] if area.ndim == 0 else float(np.sum(area, where=mask, dtype=np.float64)) / parameters['literal_values']['apply_floor_round_float_100_0']
    qa = dict(audit, backend=actual, requested_backend=backend, cuda_fallback_reason=failure, step_pixels={name: int(mask.sum()) for name, mask in masks.items()}, step_area_cm2={name: measure(mask) for name, mask in masks.items()}, step_previous_aqueous_cm2={name: measure(mask & (previous[name] == 1)) for name, mask in masks.items()}, step_previous_gas_cm2={name: measure(mask & (previous[name] == 2)) for name, mask in masks.items()}, fill_reason_a_cm2=measure(reason_a), fill_reason_b_cm2=measure(reason_b), mask_overlaps_pixels=overlaps, mask_overlaps_on_changed_pixels=overlap_changed, unchanged_stage_restoration_pixels=int((flagged & ~changed).sum()), changed_pixels=int(changed.sum()), flag_not_final_changed_pixels=int((flagged & ~changed).sum()), changed_unflagged_pixels=int((changed & ~flagged).sum()), labels_outside_masks_identical=bool(np.array_equal(parent[~flagged], labels[~flagged])), gas_changed_outside_a_d_pixels=int((gas_changes & ~(masks['A'] | masks['D'])).sum()), co2_union_changed_outside_a_b_c_pixels=int((co2_changes & ~(masks['A'] | masks['B'] | masks['C'])).sum()), domain_changed_outside_a_pixels=int(((parent == parameters['literal_values']['apply_floor_round_int_255']) != (labels == parameters['literal_values']['apply_floor_round_int_255']))[~masks['A']].sum()), floor_boundary_matches=bool(np.array_equal(labels == parameters['literal_values']['apply_floor_round_int_255'], (parent == parameters['literal_values']['apply_floor_round_int_255']) | below)), fill_flags_are_new_aqueous=bool(np.all(previous['C'][masks['C']] == 0) and np.all(labels[masks['C']] == 1)), gas_flags_are_gas_to_aqueous=bool(np.all(previous['D'][masks['D']] == 2) and np.all(labels[masks['D']] == 1)), size_filter_reapplied=False, training=False, inference=False, cpu_seconds=time.process_time() - cpu, gpu_seconds=gpu_seconds, wall_seconds=time.perf_counter() - wall)
    qa['all_pass'] = bool(not any(overlap_changed.values()) and (not (changed & ~flagged).any()) and (qa['floor_connectivity_removed_gas_pixels'] == 0) and qa['labels_outside_masks_identical'] and qa['floor_boundary_matches'] and (qa['gas_changed_outside_a_d_pixels'] == 0) and (qa['co2_union_changed_outside_a_b_c_pixels'] == 0) and (qa['domain_changed_outside_a_pixels'] == 0) and (qa['filled_under_gas_pixels'] == 0) and (qa['fill_invalid_trigger_columns'] == 0) and (qa['fill_unsupported_pixels'] == 0) and (qa['preinjection_fill_pixels'] == 0) and qa['fill_flags_are_new_aqueous'] and qa['gas_flags_are_gas_to_aqueous'])
    return FloorRuleResult(labels, masks, previous, reason_a, reason_b, qa)

def remove_contact_fill_b(labels, reason_a, reason_b, parent_contact_mask, *, parameters: Mapping):
    """Undo only inherited-colour reason-b aqueous contact fill.

    Reason masks must be disjoint and their union must equal the parent
    contact mask. Reason-b label 1 becomes water; short supported reason-a
    fill and all other categories remain byte-identical."""
    labels = np.asarray(labels)
    masks = [np.asarray(value) for value in (reason_a, reason_b, parent_contact_mask)]
    if labels.dtype != np.uint8 or labels.ndim != 2:
        raise ValueError('Production labels must be a uint8 image')
    if any((mask.dtype != np.bool_ or mask.shape != labels.shape for mask in masks)):
        raise ValueError('Contact flags must be Boolean masks on the exact label grid')
    a, b, contact = masks
    if np.any(a & b) or not np.array_equal(contact, a | b):
        raise ValueError('Parent contact mask differs from its disjoint reason partition')
    if np.any(labels[b] != 1):
        raise ValueError('Every contact-fill reason (b) pixel must be parent label 1')
    if np.any(~np.isin(labels, (0, 1, 2, parameters['literal_values']['remove_contact_fill_b_int_255']))):
        raise ValueError('Unexpected saved production category')
    result = labels.copy()
    result[b] = 0
    return result


def _validate_components(open_components, shape=None, barrier_mask=None, *, parameters: Mapping):
    """Apply the original validate components operation with explicit rule parameters."""
    components = np.asarray(open_components)
    if components.ndim != 2 or not components.size or components.dtype.kind not in 'ui' or (shape is not None and components.shape != tuple(shape)) or (components.min() < 0):
        raise ValueError('Rule B requires nonnegative integer open components on the native grid')
    if int(components.max()) > parameters['literal_values']['validate_components_int_255']:
        raise ValueError('Rule B supports at most 255 reference component IDs')
    if barrier_mask is not None:
        barrier = np.asarray(barrier_mask)
        if barrier.dtype != np.bool_ or barrier.shape != components.shape or np.any(barrier & (components > 0)):
            raise ValueError('Barrier and open sand must be disjoint native boolean/integer grids')
    return components

def apply_barrier_rule_b(labels, open_components, source_component_ids, *, pixel_area_mm2, barrier_mask=None, parameters: Mapping):
    """Restrict gas to open-sand regions reachable through four-connected CO2.

    Reachability expands from explicit source-component IDs through CO2
    bridges. Unreachable open-region gas becomes aqueous, preserving the CO2
    union and barrier labels. pixel_area_mm2 is the nominal diagnostic area;
    parameters supplies connectivity and graph bounds. No minimum-area
    filter is applied here."""
    VERSION = parameters['barrier']['rule_contract']
    labels = np.asarray(labels)
    if labels.dtype != np.uint8 or labels.ndim != 2 or (not labels.size) or (not np.isin(labels, (0, 1, 2, parameters['literal_values']['apply_barrier_rule_b_int_255'])).all()):
        raise ValueError('Rule B labels must be uint8 water0/aqueous1/gas2/unresolved255')
    components = _validate_components(open_components, labels.shape, barrier_mask, parameters=parameters)
    area = float(pixel_area_mm2)
    if not np.isfinite(area) or area <= 0:
        raise ValueError('Rule B requires a positive finite nominal pixel area')
    sources = sorted(set((int(value) for value in source_component_ids)))
    if not sources or any((value <= 0 or value > parameters['literal_values']['apply_barrier_rule_b_int_255'] for value in sources)):
        raise ValueError('Rule B requires positive protocol source component IDs')
    co2 = (labels == 1) | (labels == 2)
    count, cc, stats, _centroids = cv2.connectedComponentsWithStats(np.asarray(co2, np.uint8), connectivity=parameters['literal_values']['apply_barrier_rule_b_int_4'])
    stride = max(int(components.max()), max(sources)) + 1
    touched = co2 & (components > 0)
    encoded = cc[touched].astype(np.int64) * stride + components[touched].astype(np.int64)
    if count * stride > parameters['literal_values']['apply_barrier_rule_b_int_32000000']:
        raise ValueError('Rule B touch graph exceeds its bounded CPU allocation')
    touches = np.bincount(encoded, minlength=count * stride).reshape(count, stride) > 0
    touches[0] = False
    allowed = np.zeros(stride, bool)
    allowed[sources] = True
    bridges, bridge_ids, breach_events = ([], set(), [])
    iteration = 0
    while True:
        active_cc = np.any(touches[:, allowed], axis=1)
        reached = np.any(touches[active_cc], axis=0) if np.any(active_cc) else np.zeros(stride, bool)
        new = np.flatnonzero(reached & ~allowed)
        new = new[new != 0]
        if not new.size:
            break
        iteration += 1
        for component in new.tolist():
            triggers = np.flatnonzero(active_cc & touches[:, component])
            breach_events.append(dict(component_id=component, iteration=iteration, bridging_co2_component_ids=triggers.tolist(), touched_previously_allowed_components=sorted({int(value) for identifier in triggers for value in np.flatnonzero(touches[identifier] & allowed)})))
            for identifier in triggers.tolist():
                if identifier not in bridge_ids:
                    bridge_ids.add(identifier)
                    stat = stats[identifier]
                    bridges.append(dict(co2_component_id=identifier, co2_component_pixels=int(stat[cv2.CC_STAT_AREA]), open_component_ids=np.flatnonzero(touches[identifier]).tolist(), bbox_xywh=[int(value) for value in stat[:parameters['literal_values']['apply_barrier_rule_b_int_4']]]))
        allowed[new] = True
    removed = (labels == 2) & (components > 0) & ~allowed[components]
    revised = labels.copy()
    revised[removed] = 1
    union_after = (revised == 1) | (revised == 2)
    if not np.array_equal(co2, union_after) or np.any((revised != labels) & ~removed):
        raise AssertionError('Rule B must change only gas to aqueous, preserving the CO2 union')
    gas_before, gas_after = (labels == 2, revised == 2)
    audit = dict(contract=VERSION, source_component_ids=sources, allowed_component_ids=np.flatnonzero(allowed).tolist(), breached_components=sorted(set(np.flatnonzero(allowed).tolist()) - set(sources)), breach_events=breach_events, bridging_co2_components=sorted(bridges, key=lambda row: row['co2_component_id']), co2_connected_components=count - 1, closure_iterations=iteration, before_gas_pixels=int(gas_before.sum()), after_gas_pixels=int(gas_after.sum()), removed_gas_pixels=int(removed.sum()), removed_gas_cm2=float(removed.sum() * area / parameters['literal_values']['apply_barrier_rule_b_float_100_0']), co2_union_changed_pixels=0, only_gas_to_aqueous=True, gas_minimum_reapplied=False, open_components_array_sha256=array_identity(components), component_gas_before={str(identifier): int(np.count_nonzero(gas_before & (components == identifier))) for identifier in np.unique(components)}, component_gas_after={str(identifier): int(np.count_nonzero(gas_after & (components == identifier))) for identifier in np.unique(components)})
    if barrier_mask is not None:
        audit['barrier_gas_before'] = int(np.count_nonzero(gas_before & barrier_mask))
        audit['barrier_gas_after'] = int(np.count_nonzero(gas_after & barrier_mask))
        audit['barrier_gas_changed_pixels'] = 0
    return (revised, audit)

def frozen_rule_b_forbidden_mask(open_components, barrier_mask, rule_b_receipt, *, parameters: Mapping):
    """Recover forbidden regions from an existing base reachability receipt.

    The receipt binds the exact component grid, source IDs and allowed IDs.
    It supplies a spatial constraint without recomputing breakthroughs from
    revised gas. The disjoint barrier grid is recorded separately."""
    RULE_B_CONTRACT = parameters['barrier']['rule_contract']
    components, barrier = (np.asarray(open_components), np.asarray(barrier_mask))
    if components.ndim != 2 or not components.size or components.dtype.kind not in 'ui' or (components.min() < 0) or (components.max() > parameters['literal_values']['frozen_rule_b_forbidden_mask_int_255']):
        raise ValueError('Frozen Rule B requires a nonnegative component grid with IDs <=255')
    if barrier.dtype != np.bool_ or barrier.shape != components.shape or np.any(barrier & (components > 0)):
        raise ValueError('Frozen Rule B barrier must be a disjoint boolean grid of the same shape')
    if not isinstance(rule_b_receipt, dict) or rule_b_receipt.get('contract') != RULE_B_CONTRACT or rule_b_receipt.get('co2_union_changed_pixels') != 0 or (rule_b_receipt.get('only_gas_to_aqueous') is not True):
        raise ValueError('Missing or incompatible frozen base Rule B decision')
    actual = array_identity(components)
    if rule_b_receipt.get('open_components_array_sha256') != actual:
        raise ValueError('Frozen Rule B spatial decision is bound to different component geometry')

    def ids(key):
        values = rule_b_receipt.get(key)
        if not isinstance(values, list) or not values or any((type(v) is not int or not 0 < v <= parameters['literal_values']['frozen_rule_b_forbidden_mask_int_255'] for v in values)) or (len(set(values)) != len(values)):
            raise ValueError(f'Frozen Rule B requires explicit positive {key}')
        return sorted(values)
    sources, allowed = (ids('source_component_ids'), ids('allowed_component_ids'))
    if not set(sources).issubset(allowed):
        raise ValueError('Frozen Rule B allowed components omit a protocol source component')
    forbidden = (components > 0) & ~np.isin(components, allowed)
    audit = dict(contract=parameters['literal_values']['frozen_rule_b_forbidden_mask_metadata_9c7fee89'], open_components_array_sha256=actual, barrier_array_sha256=array_identity(barrier), forbidden_array_sha256=array_identity(forbidden), source_component_ids=sources, allowed_component_ids=allowed, forbidden_pixels=int(forbidden.sum()), barrier_pixels=int(barrier.sum()), rule_b_decision_recomputed=False, revised_gas_used=False)
    return (forbidden, audit)

@dataclass
class PlacementContext:
    """Prepared scores and geometry for placement; this object loads no models.

    The light epoch, optical validity, base probability and frozen heads are
    supplied explicitly. Cached neighbourhoods bind only the frozen base gas."""
    light_epoch_id: str
    domain: np.ndarray
    flat_valid: np.ndarray | None = None
    base_pixel: np.ndarray | None = None
    logits: dict = field(default_factory=dict)
    regional_models: dict = field(default_factory=dict)
    shared_regional: object | None = None
    metadata: dict = field(default_factory=dict)
    base_fraction: np.ndarray | None = None
    p2_score: np.ndarray | None = None
    _base_gas_sha256: str | None = None

def _placement_labels(value, *, parameters: Mapping):
    """Apply the original placement labels operation with explicit rule parameters."""
    labels = np.asarray(value)
    if labels.dtype != np.uint8 or labels.ndim != 2 or (not np.isin(labels, (0, 1, 2, parameters['literal_values']['placement_labels_int_255'])).all()):
        raise ValueError('frozen requires unchanged categorical base labels on a two-dimensional grid')
    return labels

def _placement_bool(value, shape, name, *, parameters: Mapping):
    """Apply the original placement bool operation with explicit rule parameters."""
    array = np.asarray(value)
    if array.dtype != bool or array.shape != shape:
        raise ValueError(f'frozen {name} must be a boolean mask on the exact base grid')
    return array

def base_gas_fraction(base_labels, *, parameters: Mapping):
    """Blur the frozen base gas indicator over an explicit neighbourhood.

    The indicator and result are float32. parameters supplies kernel height
    and width; OpenCV default border handling is unchanged. Revised gas
    never replaces the neighbourhood source."""
    P2_FRACTION_KERNEL = tuple(parameters['placement']['fraction_kernel_hw'])
    return cv2.blur((_placement_labels(base_labels, parameters=parameters) == 2).astype(np.float32), P2_FRACTION_KERNEL)

def _cached_fraction(context, base_labels, *, parameters: Mapping):
    """Apply the original cached fraction operation with explicit rule parameters."""
    identity = array_identity(base_labels == 2)
    if context._base_gas_sha256 is None:
        context.base_fraction = base_gas_fraction(base_labels, parameters=parameters)
        context._base_gas_sha256 = identity
        if context.base_pixel is not None:
            p = np.asarray(context.base_pixel, np.float64)
            if p.shape != base_labels.shape:
                raise ValueError('frozen cached base probability belongs to another native grid')
            context.p2_score = logit(np.clip(p, parameters['literal_values']['cached_fraction_float_1eminus_06'], 1.0 - parameters['literal_values']['cached_fraction_float_1eminus_06'])) + logit(np.clip(np.asarray(context.base_fraction, np.float64), parameters['literal_values']['cached_fraction_float_0_01'], parameters['literal_values']['cached_fraction_float_0_99']))
    elif context._base_gas_sha256 != identity:
        raise ValueError('frozen cached neighbourhood belongs to another frozen base gas mask')
    return context.base_fraction

def prepare_p2_allowed(base_labels, zones, barrier_mask, open_components, frozen_base_rule_b, *, parameters: Mapping):
    """Select base aqueous cap-zone candidates outside frozen barriers.

    Candidates must also avoid the forbidden regions of the base Rule-B
    decision. The returned Boolean masks preserve that spatial decision
    without reevaluating optical or regional heads."""
    CONTRACT = parameters['source_contracts']['prepare_p2_allowed']
    labels = _placement_labels(base_labels, parameters=parameters)
    zone = np.asarray(zones)
    if zone.shape != labels.shape or zone.dtype.kind not in 'ui' or (not np.isin(zone, (0, 1, 2, 3, parameters['literal_values']['prepare_p2_allowed_int_4'])).all()):
        raise ValueError('frozen P2 needs the transported frozen QA zone grid')
    barrier = _placement_bool(barrier_mask, labels.shape, 'P2 barrier facies', parameters=parameters)
    forbidden, decision = frozen_rule_b_forbidden_mask(open_components, barrier, frozen_base_rule_b, parameters=parameters)
    if forbidden.shape != labels.shape:
        raise ValueError('frozen P2 frozen Rule B belongs to another grid')
    caps = np.isin(zone, (1, 3))
    allowed = (labels == 1) & caps & ~barrier & ~forbidden
    audit = dict(contract=CONTRACT, placement=parameters['placement']['placement_modes']['constrained_neighbourhood'], candidates='base aqueous in frozen l1_cap/l2_caps, outside barriers and frozen Rule B forbidden', p2_allowed_array_sha256=array_identity(allowed), p2_barrier_array_sha256=array_identity(barrier), p2_rule_b_forbidden_array_sha256=array_identity(forbidden), zone_array_sha256=array_identity(zone), allowed_pixels=int(allowed.sum()), regional_head_used=False, distance_eligibility_used=False, frozen_rule_b=decision)
    return (allowed, audit, dict(p2_allowed=allowed, p2_barrier=barrier, p2_rule_b_forbidden=forbidden))

def _minimum_additions(base_gas, selected_additions, pixel_area_mm2, *, parameters: Mapping):
    """Apply the original minimum additions operation with explicit rule parameters."""
    MINIMUM_AREA_MM2 = parameters['placement']['minimum_area_mm2']
    area = validate_pixel_area(pixel_area_mm2, base_gas.shape)
    final = base_gas | selected_additions
    count, components = cv2.connectedComponents(final.astype(np.uint8), connectivity=parameters['literal_values']['minimum_additions_int_4'])
    sizes = np.bincount(components.ravel(), minlength=count)
    areas = sizes * area if np.isscalar(area) else np.bincount(components.ravel(), weights=area.ravel(), minlength=count)
    protected = np.zeros(count, bool)
    protected[np.unique(components[base_gas])] = True
    keep = (areas >= MINIMUM_AREA_MM2) | protected
    keep[0] = False
    retained = final & keep[components]
    return (retained, dict(minimum_area_mm2=MINIMUM_AREA_MM2, connectivity=parameters['literal_values']['minimum_additions_int_4'], scope='final gas components; every component containing frozen base gas protected', components=int(count - 1), protected_base_components=int(protected[1:].sum()), removed_isolated_addition_pixels=int((final & ~retained).sum()), base_gas_removed_pixels=int((base_gas & ~retained).sum()), variable_pixel_area=not np.isscalar(area)))

def evaluate_placement(context, placement, delta, base_labels, modelled, pixel_area_mm2, *, filled_support=None, top_keep, open_components, barrier_mask, frozen_base_rule_b, p2_allowed=None, source_component_ids=None, parameters: Mapping, heads_parameters: Mapping, regional_parameters: Mapping, output_capture=None):
    """Place gas on a frozen CO2 union using explicit prepared optical scores.

    The old-light epoch returns before optical or regional evaluation.
    Neighbourhood placement protects base gas, filters isolated additions by
    measured mm2 area, then clips new gas at the top. Regional placement
    thresholds pixel and regional scores, filters the combined fresh-and-fill
    graph, clips the top, then applies barrier reachability. parameters
    supplies thresholds and epoch/placement identifiers. The separate head
    and regional mappings are explicit dependencies; no CO2 union is fitted."""
    PIXEL_THRESHOLD = parameters['placement']['pixel_threshold']
    P2_FRACTION_KERNEL = tuple(parameters['placement']['fraction_kernel_hw'])
    NOMINAL_BARRIER_PIXEL_AREA_MM2 = parameters['barrier']['diagnostic_pixel_area_mm2']
    CONTRACT = parameters['source_contracts']['evaluate_placement']
    labels = _placement_labels(base_labels, parameters=parameters)
    if placement not in (parameters['placement']['placement_modes']['regional_base'], parameters['placement']['placement_modes']['constrained_neighbourhood'], parameters['placement']['placement_modes']['regional_joint_epoch']) or not np.isfinite(delta):
        raise ValueError('frozen needs a declared placement and finite logit shift')
    audit = dict(context.metadata, contract=CONTRACT, placement=placement, delta=float(delta), floor_rule_applied=False, frozen_co2_union_reused=True, co2_union_changed_pixels=0)
    if context.light_epoch_id == parameters['placement']['epoch_modes']['bypass']:
        audit.update(rig_light_labels_bit_identical=True, changed_phase_pixels=0, gas_added_pixels=0, gas_removed_pixels=0, optical_evaluation_bypassed=True)
        if output_capture is not None:
            output_capture['bypassed'] = True
        return (labels.copy(), audit, {})
    if context.light_epoch_id != parameters['placement']['epoch_modes']['evaluate'] or context.domain.shape != labels.shape:
        raise ValueError('frozen shift context does not match the declared new-lamp frame')
    support = _placement_bool(modelled, labels.shape, 'fresh modelled support', parameters=parameters) & context.domain & context.flat_valid
    top = _placement_bool(top_keep, labels.shape, 'top keep', parameters=parameters)
    forbidden, decision = frozen_rule_b_forbidden_mask(open_components, barrier_mask, frozen_base_rule_b, parameters=parameters)
    if source_component_ids is not None and sorted(source_component_ids) != decision['source_component_ids']:
        raise ValueError('frozen source components disagree with the frozen base Rule B decision')
    fraction = _cached_fraction(context, labels, parameters=parameters)
    masks = dict(base_fraction=fraction, frozen_co2=np.isin(labels, (1, 2)))
    if output_capture is not None:
        from .probability_storage import require_pair_mode
        require_pair_mode(placement, parameters)
    if placement == parameters['placement']['placement_modes']['constrained_neighbourhood']:
        allowed = _placement_bool(p2_allowed, labels.shape, 'P2 allowed', parameters=parameters)
        barrier = _placement_bool(barrier_mask, labels.shape, 'P2 barrier facies', parameters=parameters)
        if np.any(allowed & ((labels != 1) | barrier | forbidden)):
            raise ValueError('frozen P2 allowed mask violates the frozen base constraints')
        p = np.asarray(context.base_pixel, np.float64)
        if p.shape != labels.shape:
            raise ValueError('frozen P2 lost the unshifted base G3 probability grid')
        score = context.p2_score
        if score is None:
            raise ValueError('frozen P2 lost its cached unshifted optical/neighbourhood score')
        additions = allowed & np.isfinite(score) & (score + float(delta) >= logit(PIXEL_THRESHOLD))
        final, minimum = _minimum_additions(labels == 2, additions, pixel_area_mm2, parameters=parameters)
        result = labels.copy()
        result[additions & final] = 2
        clipped = (result == 2) & (labels != 2) & ~top
        result[clipped] = labels[clipped]
        audit.update(regional_head_used=False, shifted_pixel_probability_used=False, neighbourhood_computed_from='frozen base gas only; never updated', neighbourhood_kernel=list(P2_FRACTION_KERNEL), neighbourhood_border='OpenCV default BORDER_DEFAULT', neighbourhood_fraction_clip=[parameters['literal_values']['evaluate_placement_float_0_01'], parameters['literal_values']['evaluate_placement_float_0_99']], base_pixel_probability_clip=[parameters['literal_values']['evaluate_placement_float_1eminus_06'], 1.0 - parameters['literal_values']['evaluate_placement_float_1eminus_06']], pixel_threshold=PIXEL_THRESHOLD, p2_allowed_array_sha256=array_identity(allowed), allowed_pixels=int(allowed.sum()), allowed_nonfinite_probability_pixels=int((allowed & ~np.isfinite(p)).sum()), selected_pre_minimum_pixels=int(additions.sum()), gas_minimum=minimum, top_clip_removed_pixels=int(clipped.sum()), top_clip_applied=True, frozen_rule_b=decision, rule_b_decision_recomputed=False, rule_b_applied=True)
        masks.update(p2_allowed=allowed, p2_barrier=barrier, p2_rule_b_forbidden=forbidden, pre_minimum_additions=additions, post_minimum_gas=final)
        if np.any((labels == 2) & (result != 2)) or not np.array_equal(result[~allowed], labels[~allowed]):
            raise AssertionError('frozen P2 changed frozen base gas or a non-candidate label')
    else:
        if placement not in context.logits or placement not in context.regional_models or context.shared_regional is None:
            raise ValueError('frozen placement lost its exact frozen pixel/regional pair')
        pixel = heads.shifted_probability(context.logits[placement], delta, context.flat_valid)
        prepared = regional.prepare_conditional_gas(context.shared_regional, np.nan_to_num(pixel, nan=0.0), parameters=regional_parameters)
        model = context.regional_models[placement]
        regional_score = regional.probability(prepared['features'], model['heads']['gas_within_co2'], parameters=regional_parameters)
        regional_score = cv2.resize(regional_score, labels.shape[::-1], interpolation=cv2.INTER_LINEAR)
        result, threshold_audit, threshold_masks = replace_gas_from_scores(labels, pixel, regional_score, support, pixel_area_mm2, filled_support=filled_support, parameters=parameters)
        if output_capture is not None:
            from .probability_storage import capture_placement, state
            capture_placement(output_capture, locals())
        clipped = (result == 2) & ~top
        result[clipped] = 1
        if output_capture is not None:
            state(output_capture, 'placement-top', result, eligible_rule='placement_top')
        result, applied_b = apply_barrier_rule_b(result, open_components, decision['source_component_ids'] if source_component_ids is None else source_component_ids, pixel_area_mm2=NOMINAL_BARRIER_PIXEL_AREA_MM2, barrier_mask=barrier_mask, parameters=parameters)
        if output_capture is not None:
            state(output_capture, 'placement-Rule-B', result, eligible_rule='placement_B')
        audit.update(threshold_audit)
        audit.update(context.metadata, contract=CONTRACT, placement=placement, delta=float(delta), regional_head_used=True, shifted_pixel_probability_used=True, floor_rule_applied=False, top_clip_removed_pixels=int(clipped.sum()), top_clip_applied=True, rule_b_application=applied_b, rule_b_decision_recomputed=True, rule_b_applied=True, rule_b_removed_pixels=applied_b['removed_gas_pixels'], t4b_indicator_applied_once=placement == parameters['placement']['placement_modes']['regional_joint_epoch'])
        masks.update(threshold_masks, frozen_rule_b_forbidden=forbidden)
    if not np.array_equal(np.isin(result, (1, 2)), np.isin(labels, (1, 2))):
        raise AssertionError('frozen gas placement changed the frozen CO2 union')
    audit.update(changed_phase_pixels=int((result != labels).sum()), gas_added_pixels=int(((result == 2) & (labels != 2)).sum()), gas_removed_pixels=int(((result != 2) & (labels == 2)).sum()), base_fraction_array_sha256=array_identity(fraction))
    return (result, audit, masks)

def frozen_native_partition(presence, conditional_gas, valid, *, parameters: Mapping):
    """Assign water, aqueous CO2 and gas using separate frozen score thresholds.

    Presence is thresholded first on the supplied valid native grid; gas
    requires both presence and its conditional score threshold. This is
    the original hierarchical partition, rather than a three-class argmax.
    Thresholds are explicit in parameters and output labels are uint8.
    """
    presence, conditional_gas = (np.asarray(presence), np.asarray(conditional_gas))
    valid = np.asarray(valid, bool)
    if presence.shape != valid.shape or conditional_gas.shape != valid.shape:
        raise ValueError('Frozen partition inputs do not share the native grid')
    active = (presence >= parameters['native_partition']['presence_threshold']) & valid
    labels = active.astype(np.uint8)
    labels[active & (conditional_gas >= parameters['native_partition']['conditional_gas_threshold'])] = 2
    return labels


def require_categorical_feature_support(labels, optical_valid, gas_feature_valid, *, parameters: Mapping):
    """Require both frozen feature stages before assigning a categorical class.

    Missing gas evidence is unresolved rather than aqueous. The explicit
    unresolved category is applied before any separately flagged nearest-
    label filling; counts retain optical support and missing gas evidence.
    """
    labels = np.asarray(labels)
    optical_valid, gas_feature_valid = (np.asarray(mask) for mask in (optical_valid, gas_feature_valid))
    if labels.dtype != np.uint8 or optical_valid.dtype != bool or gas_feature_valid.dtype != bool or (labels.shape != optical_valid.shape) or (labels.shape != gas_feature_valid.shape):
        raise ValueError('Categorical feature support must share the native label grid')
    supported = optical_valid & gas_feature_valid
    result = labels.copy()
    result[~supported] = parameters['native_partition']['unresolved_category']
    return (result, supported, dict(optical_supported_pixels=int(optical_valid.sum()), gas_feature_invalid_optical_supported_pixels=int(np.count_nonzero(optical_valid & ~gas_feature_valid)), final_modelled_feature_pixels=int(supported.sum()), invalid_feature_policy='unresolved before explicitly flagged nearest-label filling'))


def preminimum_gas_selection(decoded_g3, decoded_regional, *, parameters: Mapping, output_capture=None):
    """Threshold decoded gas evidence before intersecting a corrected CO2 union.

    Both supplied probability grids must already be float32. The regional
    grid is resized to the pixel grid with OpenCV linear interpolation,
    then the explicit pixel and regional thresholds are applied. Keeping
    this evidence independent of an earlier CO2 mask allows a later seam
    release to use its original gas evidence without a new prediction.
    """
    pixel, regional = (np.asarray(decoded_g3), np.asarray(decoded_regional))
    if pixel.ndim != 2 or regional.ndim != 2 or pixel.dtype != np.float32 or (regional.dtype != np.float32):
        raise ValueError('Pre-minimum selection needs decoded float32 probability grids')
    native = cv2.resize(regional, pixel.shape[::-1], interpolation=cv2.INTER_LINEAR)
    if output_capture is not None:
        from .probability_storage import capture_centre_selection
        capture_centre_selection(output_capture, locals())
    return (pixel >= parameters['centre_selection']['pixel_threshold']) & (native >= parameters['centre_selection']['regional_threshold'])


def replace_gas_from_scores(base_labels, pixel_score, regional_score, modelled, pixel_area_mm2, *, filled_support=None, parameters: Mapping):
    """Partition frozen CO2 using pixel and regional gas score thresholds.

    Declared inherited fill takes its nearest modelled conditional partition.
    The complete selected gas graph is filtered by measured mm2 area and
    connectivity before barrier processing. Unsupported, unfilled categories
    remain exact; thresholds and minimum area are supplied in parameters."""
    CONTRACT = parameters['source_contracts']['replace_gas_from_scores']
    labels, pixel, score, support = map(np.asarray, (base_labels, pixel_score, regional_score, modelled))
    if labels.dtype != np.uint8 or labels.ndim != 2 or support.dtype != bool or any((v.shape != labels.shape for v in (pixel, score, support))) or (not np.isin(labels, (0, 1, 2, parameters['literal_values']['replace_gas_from_scores_int_255'])).all()) or (not np.isfinite(score[support]).all()) or (not np.isfinite(pixel[support]).all()):
        raise ValueError('Gas replacement lost the frozen label/support/probability grid')
    area = validate_pixel_area(pixel_area_mm2, labels.shape)
    co2 = np.isin(labels, (1, 2))
    selected = co2 & support & (pixel >= parameters['literal_values']['replace_gas_from_scores_float_0_7']) & (score >= parameters['literal_values']['replace_gas_from_scores_float_0_6'])
    filled = np.zeros(labels.shape, bool) if filled_support is None else np.asarray(filled_support)
    if filled.dtype != bool or filled.shape != labels.shape or np.any(filled & support):
        raise ValueError('Inherited fill flags must be disjoint from fresh modelled gas support')
    filled_co2 = filled & co2
    if filled_co2.any():
        from scipy.ndimage import distance_transform_edt
        if not support.any():
            raise ValueError('Gas partition filling needs a supported source donor')
        _distance, indices = distance_transform_edt(~support, return_indices=True)
        nearest = selected[tuple(indices)]
        selected[filled_co2] = nearest[filled_co2]
    gas, minimum = area_filtered_gas(selected, area, minimum_area_mm2=parameters['literal_values']['replace_gas_from_scores_float_2_5'], connectivity=parameters['literal_values']['replace_gas_from_scores_int_4'])
    result = labels.copy()
    revised = co2 & (support | filled)
    result[revised] = np.where(gas[revised], 2, 1)
    if not np.array_equal(np.isin(result, (1, 2)), co2) or not np.array_equal(result[~revised], labels[~revised]):
        raise AssertionError('Gas-only replacement changed frozen CO2 or unsupported categories')
    audit = dict(contract=CONTRACT, frozen_co2_union_reused=True, co2_union_changed_pixels=0, unsupported_unfilled_categories_preserved=True, unsupported_gas_pixels_preserved=int(np.count_nonzero((labels == 2) & ~(support | filled))), inherited_filled_co2_pixels=int(filled_co2.sum()), fill_policy='nearest fresh modelled conditional partition inside frozen CO2 only; support flags unchanged', minimum_scope='full combined modelled-and-declared-filled gas graph before B', pixel_threshold=parameters['literal_values']['replace_gas_from_scores_float_0_7'], regional_threshold=parameters['literal_values']['replace_gas_from_scores_float_0_6'], gas_minimum=minimum, gas_minimum_before_rule_b=True, rule_b_applied=False, top_clip_applied=False, changed_phase_pixels=int(np.count_nonzero(result != labels)), selected_pre_minimum_pixels=int(selected.sum()), selected_post_minimum_pixels=int(gas.sum()))
    return (result, audit, dict(frozen_co2=co2, modelled_gas_support=support, inherited_gas_fill=filled_co2, pre_minimum_gas=selected, post_minimum_gas=gas))

@dataclass(frozen=True)
class GasDecisionSettings:
    """Explicit thresholds for gas partition within an unchanged CO2 union.

    The six fields keep the original serialized configuration identity.
    Required parameters supplies the allowed configuration grid; it is an
    InitVar and is not part of the numerical settings or their hash.
    """
    format: str
    pixel_threshold: float
    regional_mode: str
    regional_threshold: float
    minimum_area_mm2: float
    connectivity: int
    parameters: InitVar[Mapping] = field(kw_only=True)

    def __post_init__(self, parameters):
        grid = parameters['gas_configuration']
        if self.format != grid['format'] or self.pixel_threshold not in tuple(grid['allowed_pixel_thresholds']) or self.regional_mode not in tuple(grid['allowed_regional_modes']) or (self.regional_threshold != grid['regional_threshold']) or (self.minimum_area_mm2 not in tuple(grid['allowed_minimum_area_mm2'])) or (self.connectivity != grid['connectivity']):
            raise ValueError('Gas decision settings differ from the explicit configuration grid')


def gas_configuration_identity(settings):
    """Hash the six original serialized gas-decision settings fields.

    Canonical sorted JSON and compact separators preserve the original
    identity; the explicit parameter Mapping is not serialized.
    """
    return hashlib.sha256(json.dumps(asdict(settings), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def _packed_mask_identity(mask):
    """Retain the gas-decision receipt's shape plus little-endian packed bits."""
    shape = np.asarray(mask.shape, dtype='<i8')
    return hashlib.sha256(shape.tobytes() + np.packbits(np.asarray(mask, bool).reshape(-1), bitorder='little').tobytes()).hexdigest()

def _check_scores(values, name, *, shape=None, allow_nan=False):
    """Check floating probabilities without casting or replacing missing evidence."""
    array = np.asarray(values)
    if array.dtype not in (np.dtype('float16'), np.dtype('float32'), np.dtype('float64')) or array.ndim != 2 or (not array.size) or (shape is not None and array.shape != shape):
        raise ValueError(f'{name} requires floating-point probabilities on its declared grid')
    finite = np.isfinite(array)
    if np.isinf(array).any() or (not allow_nan and (not finite.all())):
        raise ValueError(f'{name} has unsupported non-finite probabilities')
    if ((array[finite] < 0) | (array[finite] > 1)).any():
        raise ValueError(f'{name} probabilities are outside [0,1]')
    return array

def apply_gas_configuration(frozen_labels, g3_pixel, regional_probability, pixel_area_mm2, settings, *, parameters: Mapping, output_capture=None):
    """Apply thresholds and native four-connected minimum inside frozen CO2.

    Regional continuous probabilities are converted to float32 and resized
    with OpenCV INTER_LINEAR *before* thresholding, as in ``confirm_gas``.
    G3 NaNs are invalid evidence and fail its threshold; they never substitute
    the old gas head or restore a frozen gas pixel inside the CO2 union.
    """
    if not isinstance(settings, GasDecisionSettings):
        raise TypeError('G3 decisions require an immutable settings object')
    labels = np.asarray(frozen_labels)
    if labels.dtype != np.uint8 or labels.ndim != 2 or (not labels.size) or (not np.isin(labels, (0, 1, 2, parameters['gas_configuration']['unresolved_category'])).all()):
        raise ValueError('Frozen labels must be a native uint8 grid with values 0,1,2,255')
    g3 = _check_scores(g3_pixel, 'G3 pixel', shape=labels.shape, allow_nan=True)
    area = float(pixel_area_mm2)
    if not np.isfinite(area) or area <= 0:
        raise ValueError('G3 gas minimum requires a positive nominal pixel area')
    co2 = np.isin(labels, (1, 2))
    pixel_gas = co2 & (g3 >= settings.pixel_threshold)
    if settings.regional_mode == parameters['gas_configuration']['disabled_regional_mode']:
        regional_gas = pixel_gas.copy()
        source_shape = None
    else:
        values = _check_scores(regional_probability, 'Regional gas')
        source_shape = list(values.shape)
        scores = cv2.resize(np.asarray(values, np.float32), labels.shape[::-1], interpolation=cv2.INTER_LINEAR)
        regional_gas = pixel_gas & (scores >= settings.regional_threshold)
    gas = regional_gas.copy()
    withdrawn_components = 0
    if settings.minimum_area_mm2:
        _, components, statistics, _ = cv2.connectedComponentsWithStats(regional_gas.astype(np.uint8), connectivity=settings.connectivity)
        small = statistics[:, cv2.CC_STAT_AREA] * area < settings.minimum_area_mm2
        small[0] = False
        withdrawn_components = int(np.count_nonzero(small))
        gas[small[components]] = False
    result = labels.copy()
    result[co2] = np.where(gas[co2], 2, 1).astype(np.uint8)
    if not np.array_equal(np.isin(result, (1, 2)), co2):
        raise AssertionError('Approved G3 decision changed the immutable CO2 union')
    if not np.array_equal(result[~co2], labels[~co2]):
        raise AssertionError('Approved G3 decision changed frozen water or unresolved pixels')
    gas_count = int(np.count_nonzero(gas))
    audit = dict(format=parameters['gas_configuration']['format'], settings=asdict(settings), configuration_sha256=gas_configuration_identity(settings), co2_union_mask_sha256=_packed_mask_identity(co2), co2_support_exact=True, outside_co2_labels_exact=True, co2_pixels=int(np.count_nonzero(co2)), gas_pixels_before_regional=int(np.count_nonzero(pixel_gas)), gas_pixels_before_minimum=int(np.count_nonzero(regional_gas)), gas_pixels=gas_count, gas_area_mm2=gas_count * area, pixel_area_mm2=area, withdrawn_gas_components=withdrawn_components, withdrawn_gas_pixels=int(np.count_nonzero(regional_gas & ~gas)), invalid_g3_pixels_inside_co2=int(np.count_nonzero(co2 & ~np.isfinite(g3))), regional_source_shape_hw=source_shape, regional_resize='float32 cv2.INTER_LINEAR to native grid before threshold', minimum_policy='4-connected component area strictly smaller than minimum is aqueous', optical_presence_gate_used=False, feature_eligibility_gate_used=False, legacy_gas_fallback_used=False, mass_eligible=False)
    if output_capture is not None:
        from .probability_storage import capture_centre
        capture_centre(output_capture, locals())
    return (result, audit)
