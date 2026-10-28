"""Saved before/after free-gas accounting; display products never drive a fit."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np


FORMAT = "darsia-gas-contribution-review-v1"
_FIELDS = ("cell_model_retained_g", "cell_overflow_added_g", "cell_overflow_removed_g")


def capture_gas_contributions(frame, before, after, *, max_axis=512):
    """Conservatively reduce existing native gas densities, without reinterpreting RGB.

    Minimum/positive differences are taken on native pixels BEFORE pooling.
    Thus simultaneous additions and removals cannot cancel into a false origin.
    The exact Geometry weights are those already admitted by the split producer.
    """
    shape = frame.population.eligible.shape
    if (before.shape != shape or after.shape != shape or before.dtype != np.float32
            or after.dtype != np.float32 or type(max_axis) is not int or max_axis < 1):
        raise ValueError("Existing same-grid float32 gas fields are required")
    keep = frame.population.eligible
    volume = frame.population.pore_volume_m3
    steps = [max(1, int(np.ceil(n/max_axis))) for n in shape]
    row_edges = list(range(0, shape[0], steps[0])) + [shape[0]]
    col_edges = list(range(0, shape[1], steps[1])) + [shape[1]]
    coarse_shape = (len(row_edges)-1, len(col_edges)-1)
    arrays = {name: np.zeros(coarse_shape, np.float64) for name in _FIELDS}
    arrays["cell_has_support"] = np.zeros(coarse_shape, bool)
    starts = np.asarray(col_edges[:-1])
    for row, (start, stop) in enumerate(zip(row_edges[:-1], row_edges[1:])):
        mask = keep[start:stop]
        weights = np.where(mask, volume[start:stop], 0.)*1000
        old = before[start:stop].astype(np.float64)
        new = after[start:stop].astype(np.float64)
        old = np.where(mask, old, 0.)
        new = np.where(mask, new, 0.)
        for name, value in zip(_FIELDS, (np.minimum(old, new), np.maximum(new-old, 0.),
                                        np.maximum(old-new, 0.))):
            arrays[name][row] = np.add.reduceat((value*weights).sum(axis=0), starts)
        arrays["cell_has_support"][row] = np.logical_or.reduceat(mask.any(axis=0), starts)
    height, width = map(float, frame.prepared["frame"].dimensions_m)
    grid = dict(format="darsia-gas-contribution-review-grid-v1", shape=list(coarse_shape),
        native_shape_hw=list(shape), row_edges_px=row_edges, column_edges_px=col_edges,
        extent_m=[0., width, 0., height], imshow_origin="upper", display_only=True,
        axes="native image rows top to bottom, columns left to right",
        units="g per review cell", reduction="sum of original native gas density times Geometry pore-volume weights")
    return dict(grid=grid, arrays=arrays)


def contribution_summary(report, target=None, *, model_only=False):
    """Keep original producer scalars literal, and distinguish measured estimates."""
    before, after = report.get("baseline_gas_mass_g"), report.get("gas_mass_g")
    if any(type(n) not in (int, float) or not np.isfinite(n) or n < 0 for n in (before, after)):
        return None
    target = target or {}
    kind = target.get("kind", "model_phase_split" if model_only else "legacy_overflow")
    estimated = target.get("estimated", False if model_only else None)
    label = {"own_measurement": "Measured overflow", "pressure_matched_estimate": "ESTIMATED overflow",
             "model_phase_split": "Phase model; no overflow reference",
             "legacy_overflow": "Historical overflow adjustment"}.get(kind, str(kind))
    change = float(after-before)
    return dict(model_gas_mass_g=float(before), final_gas_mass_g=float(after),
        overflow_change_g=change, overflow_change_share_of_final=None if after == 0 else change/after,
        reference_kind=kind, estimated=estimated, reference_label=label,
        scope="quantified main physical support; excluded and unknown contributions are unchanged")


def save_gas_contributions(destination, frame, result, target=None):
    """Save only small review cells while native fields are already in memory."""
    from .n1_pipeline import asset
    model_only = result.get("statistics", {}).get("overflow_steering", {}).get("kind") == "model_phase_split"
    summary = contribution_summary(result["report"], target, model_only=model_only)
    record = dict(format=FORMAT, summary=summary, maps=None, grid=None,
        original_statistics=deepcopy(frame.statistics_descriptor),
        probability_roles=deepcopy(frame.probability_roles), phase_source=deepcopy(frame.phase_source),
        raw_decoded=False, phase_inference_run=False, mass_fit_run=False)
    if summary is not None:
        value = frame.prepared["frame"]
        summary.update(frame_index=value.frame_index, elapsed_h=float(value.elapsed_h))
    captured = result.get("gas_contribution_maps")
    if captured is None:
        return record
    arrays = captured["arrays"]
    path = Path(destination)/"gas_contributions.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    record.update(maps=asset(path), grid=deepcopy(captured["grid"]))
    retained, added, removed = [arrays[name] for name in _FIELDS]
    if summary is not None:
        summary.update(model_retained_g=float(retained.sum()), overflow_added_g=float(added.sum()),
            overflow_removed_g=float(removed.sum()),
            retained_share_of_final=None if summary["final_gas_mass_g"] == 0 else float(retained.sum())/summary["final_gas_mass_g"],
            added_share_of_final=None if summary["final_gas_mass_g"] == 0 else float(added.sum())/summary["final_gas_mass_g"])
    record["display_scale_g_per_cell"] = dict(
        positive=float(max((retained+added).max(), (retained+removed).max())),
        signed=float(np.abs(added-removed).max()))
    return record


def contribution_scales(rows):
    """A single saved scale for every frame; no frame-wise colour stretching."""
    return dict(format=FORMAT, display_scale_g_per_cell={key: max((
        row.get("gas_contribution", {}).get("display_scale_g_per_cell", {}).get(key, 0.)
        for row in rows), default=0.) for key in ("positive", "signed")})


def read_gas_contributions(preparation, run, case_id, *, reader=None):
    """Read an exact recorded split; old reports retain numbers without invented maps."""
    from .n1_pipeline import DataReader
    rows = [row for row in preparation.get("frames", []) if row.get("run") == run and row.get("case_id") == case_id]
    if len(rows) != 1:
        return dict(status="unavailable", summary=None, maps=None, grid=None,
                    display_scale_g_per_cell={}, reason="No unique saved before/after gas split for this image")
    row = rows[0]
    record = row.get("gas_contribution") or {}
    report = row.get("report", {})
    target = row.get("phase_split_target") or row.get("distribution", {}).get("phase_split_target")
    summary = record.get("summary") or contribution_summary(report, target,
        model_only=report.get("overflow_reference_available") is False)
    scales = preparation.get("gas_contribution_review", {}).get("display_scale_g_per_cell", {})
    result = dict(status="spatial_unavailable" if summary else "unavailable", summary=deepcopy(summary),
        maps=None, grid=None, display_scale_g_per_cell=deepcopy(scales),
        reason="This saved result contains no spatial gas-contribution product")
    if record.get("maps") is None:
        return result
    if record.get("format") != FORMAT or summary is None:
        raise ValueError("Unsupported saved gas contribution record")
    reader = reader or DataReader(unc_only=False)
    arrays = reader.archive(record["maps"], (*_FIELDS, "cell_has_support"))
    grid = record["grid"]
    shape = tuple(grid["shape"])
    for name in _FIELDS:
        field = arrays[name]
        if (field.shape != shape or field.dtype != np.float64 or not np.isfinite(field).all() or np.any(field < 0)):
            raise ValueError("Saved gas contribution cells differ from their original grid")
    if arrays["cell_has_support"].shape != shape or arrays["cell_has_support"].dtype != bool:
        raise ValueError("Saved gas contribution coverage differs from its original grid")
    retained, added, removed = [arrays[name] for name in _FIELDS]
    arrays.update(cell_model_before_g=retained+removed, cell_final_gas_g=retained+added,
                  cell_overflow_change_g=added-removed)
    result.update(status="available", maps=arrays, grid=deepcopy(grid), reason=None,
        display_scale_g_per_cell=deepcopy(scales or record["display_scale_g_per_cell"]))
    return result
