"""Export independent per-acquisition masses from verified saved coarse cells.

This separate saved-output branch never reads native fields, images or models,
collects an asset, reconstructs a mass component, or sums acquisition times.
"""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .storage_saved_products import read_stored_frame, read_stored_run


_COMPONENTS = (("total", "total_mass_g"), ("aqueous", "aqueous_mass_g"),
               ("gas", "gas_mass_g"))
_COLUMNS = ("row_index", "run", "case_id", "precision", "source_sha256", "source_json",
            *("cell_mass_" + component + "_sum_g" for component, _ in _COMPONENTS),
            *("native_" + name for _, name in _COMPONENTS),
            *("stored_minus_native_" + component + "_g" for component, _ in _COMPONENTS))
_SCOPE_COLUMNS = ("analysis_scope_json", "unknown_mass_g",
    *("quantified_cell_mass_" + component + "_sum_g" for component, _ in _COMPONENTS),
    *("quantified_native_" + name for _, name in _COMPONENTS),
    *("quantified_stored_minus_native_" + component + "_g" for component, _ in _COMPONENTS))


def _destination(destination):
    destination = Path(destination)
    receipt = destination.with_name(destination.name + ".receipt.json")
    if destination.suffix.lower() != ".csv":
        raise ValueError("Choose a new saved-mass CSV path")
    if destination.exists() or receipt.exists():
        raise FileExistsError("Saved mass tables and receipts are immutable")
    return destination, receipt


def _asset(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size)


def _row(frame, index):
    # The qualified reader independently sums all three cell-mass arrays in
    # float64 and validates each against its original saved native scalar.
    totals = frame["mass_totals_g"]
    reference = frame["native_scalar_reference"]
    row = dict(row_index=index, run=frame["run"], case_id=frame["case_id"],
               precision=frame["precision"], source_sha256=frame["source"]["source_sha256"],
               source_json=json.dumps(frame["source"], ensure_ascii=False, allow_nan=False,
                                      sort_keys=True, separators=(",", ":")))
    native, differences = {}, {}
    scope = frame.get("analysis_scope")
    quantified_native, quantified_differences = {}, {}
    for component, name in _COMPONENTS:
        value, target = totals[component], reference[name]
        difference = value - target if scope is None else None
        row["cell_mass_" + component + "_sum_g"] = value
        row["native_" + name] = target
        row["stored_minus_native_" + component + "_g"] = difference
        native[component], differences[component] = target, difference
        if scope is not None:
            subtotal = frame["quantified_mass_totals_g"][component]
            native_subtotal = reference["quantified_" + name]
            delta = subtotal - native_subtotal
            row["quantified_cell_mass_" + component + "_sum_g"] = subtotal
            row["quantified_native_" + name] = native_subtotal
            row["quantified_stored_minus_native_" + component + "_g"] = delta
            quantified_native[component], quantified_differences[component] = native_subtotal, delta
    document = frame["storage_document"]
    provenance = dict(row_index=index, run=frame["run"], case_id=frame["case_id"],
        precision=frame["precision"], source=deepcopy(frame["source"]),
        storage_receipt=deepcopy(frame["storage_receipt"]),
        configuration=deepcopy(document["configuration"]), code_commit=document["code_commit"],
        calibration_sha256=document["calibration_sha256"], before_rules=deepcopy(frame["before_rules"]),
        independent_cell_mass_sums_g=deepcopy(totals), native_saved_mass_targets_g=native,
        stored_minus_native_g=differences, verified_relative_limit=1e-6,
        loaded_assets=deepcopy(frame["loaded_assets"]))
    if scope is not None:
        row.update(analysis_scope_json=json.dumps(scope, ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":")), unknown_mass_g=None)
        provenance.update(analysis_scope=deepcopy(scope), unknown_mass_g=None,
            quantified_independent_cell_mass_sums_g=deepcopy(frame["quantified_mass_totals_g"]),
            quantified_native_saved_mass_targets_g=quantified_native,
            quantified_stored_minus_native_g=quantified_differences)
    return row, provenance


def _write(destination, receipt, frames, *, manifest_assets):
    rows, provenance = [], []
    for index, frame in enumerate(frames):
        row, origin = _row(frame, index)
        rows.append(row)
        provenance.append(origin)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = _COLUMNS + _SCOPE_COLUMNS if any("analysis_scope" in frame for frame in frames) else _COLUMNS
    with destination.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    document = dict(format="darsia-saved-coarse-mass-csv-receipt-v1", complete=True,
        row_count=len(rows), ordered_case_ids=[frame["case_id"] for frame in frames],
        columns=list(columns), row_index="zero-based CSV row; run export preserves original completion order",
        mass_unit="g", mass_sums="three independent saved cell-array sums, float64 reduction",
        time_aggregation=False, reconstructed_mass_components=False,
        additional_legacy_calibration_columns_inferred=False, load_before=False,
        frames=provenance, loaded_manifest_assets=deepcopy(manifest_assets), table=_asset(destination))
    with receipt.open("xb") as stream:
        stream.write((json.dumps(document, ensure_ascii=False, allow_nan=False,
                                 sort_keys=True, indent=2) + "\n").encode("utf8"))
    return dict(table=document["table"], receipt=_asset(receipt), row_count=len(rows),
                ordered_case_ids=document["ordered_case_ids"])


def export_stored_frame_csv(destination, native_frame_metadata, *, resolve_asset, expected=None):
    """Export one acquisition; strict reader/collect errors propagate unchanged."""
    destination, receipt = _destination(destination)
    frame = read_stored_frame(native_frame_metadata, resolve_asset=resolve_asset,
                              load_before=False, expected=expected)
    return _write(destination, receipt, [frame], manifest_assets=[])


def export_stored_run_csv(destination, completed_metadata, *, resolve_asset):
    """Export one complete run in original order, with one row per acquisition.

    The existing run reader validates manifest/order/producer identities before
    any table is created. Its memory use is retained: one run's saved coarse
    arrays are resident together; this function does not read other runs.
    """
    destination, receipt = _destination(destination)
    run = read_stored_run(completed_metadata, resolve_asset=resolve_asset, load_before=False)
    return _write(destination, receipt, run["frames"], manifest_assets=run["loaded_manifest_assets"])
