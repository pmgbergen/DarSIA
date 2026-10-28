"""Sparse, source-bound historical V5 metadata; no ranking or readiness inference."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


FORMAT = "series-saved-v5-normalisation-v1"
MAX_METADATA_BYTES = 1024 * 1024
MAX_RUNS = 128
PHOTOMETRY_POINTER = "/settings/v5_raw_photometry"
MATERIALIZATION_FIELDS = (
    "format", "source_cohort_contract", "canonical_runs", "timeline_frames",
    "training_eligibility", "run_geometry_contract",
    "accepted_run_native_geometry_revisions", "run_geometry_validation_only",
)


def _series(value):
    return str(value).casefold().replace("_", "").replace("-", "")


def _read(bound):
    if not isinstance(bound, dict) or not isinstance(bound.get("path"), str):
        raise ValueError("A saved metadata path and SHA are required.")
    if not isinstance(bound.get("sha256"), str) or re.fullmatch(r"[a-f0-9]{64}", bound["sha256"]) is None:
        raise ValueError("A saved metadata SHA-256 is required.")
    from .review_storage import resolve_review_path
    path = resolve_review_path(bound["path"])
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Saved import metadata exceeds the display limit.")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != bound["sha256"] or ("size_bytes" in bound and len(raw) != bound["size_bytes"]):
        raise ValueError("The saved source-import metadata changed: " + str(path))
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate saved metadata key: " + key)
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("Nonfinite saved metadata value: " + value)
    document = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(document, dict):
        raise ValueError("A saved metadata object is required.")
    return document, {"path": bound["path"], "sha256": digest, "size_bytes": len(raw)}


def build_source_import(training_request, series_id):
    """Metadata-only import candidate, built only from explicit bound sources.

    This is not a SavedV5Snapshot: no ranking, suggested reference, majority class,
    selection mode, completion marker or numerical default is manufactured.
    """
    request, request_source = _read(training_request)
    settings = request.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("The saved training request has no settings object.")
    scope = settings.get("series")
    if not isinstance(scope, list) or len(scope) != 1 or _series(scope[0]) != _series(series_id):
        raise ValueError("The saved training request identifies another series.")
    block = settings.get("v5_raw_photometry")
    if not isinstance(block, dict) or block.get("format") != "ffac-v5-raw-full-native-training-v1":
        raise ValueError("The bound request has no saved V5 raw-photometry block.")
    materialization_source = {
        "path": block.get("materialization_receipt_path"),
        "sha256": block.get("materialization_receipt_sha256"),
    }
    materialization, materialization_source = _read(materialization_source)
    if materialization.get("format") != "ffac-v5-native-materialization-batch-v1":
        raise ValueError("The bound receipt is not a saved V5 materialization document.")
    mapping = materialization.get("canonical_mapping")
    if not isinstance(mapping, list) or not 0 < len(mapping) <= MAX_RUNS:
        raise ValueError("The bound materialization has no bounded canonical run list.")
    if "canonical_runs" in materialization and materialization["canonical_runs"] != len(mapping):
        raise ValueError("The bound canonical run count differs from its mapping.")
    runs = [row.get("run") if isinstance(row, dict) else None for row in mapping]
    if any(not isinstance(run, str) or not run for run in runs) or len(set(runs)) != len(runs):
        raise ValueError("The bound canonical run identities are absent or repeated.")
    return {
        "format": FORMAT,
        "series_id": str(series_id),
        "runs": runs,
        "v5_raw_photometry": deepcopy(block),
        "materialization": {key: deepcopy(materialization[key]) for key in MATERIALIZATION_FIELDS if key in materialization},
        "sources": [
            dict(request_source, pointer=PHOTOMETRY_POINTER),
            dict(materialization_source, pointer="/canonical_mapping"),
        ],
        "field_sources": {
            "v5_raw_photometry": {"source": request_source, "pointer": PHOTOMETRY_POINTER},
            "runs": {"source": materialization_source, "pointer": "/canonical_mapping/*/run"},
            "materialization": {"source": materialization_source, "pointer": "/"},
        },
    }


def is_source_import(record):
    return isinstance(record, dict) and record.get("format") == FORMAT and isinstance(record.get("v5_raw_photometry"), dict)


def read_source_import(record, series_id, runs):
    """Revalidate the stored import against its original exact sources, read only."""
    if not is_source_import(record):
        raise ValueError("This record is not a saved V5 source import.")
    fields = record.get("field_sources")
    if not isinstance(fields, dict) or not isinstance(fields.get("v5_raw_photometry"), dict):
        raise ValueError("The saved import lacks field-source metadata.")
    source = fields["v5_raw_photometry"]
    if source.get("pointer") != PHOTOMETRY_POINTER:
        raise ValueError("The saved import lacks its exact training-request pointer.")
    saved_series = record.get("series_id")
    if not isinstance(saved_series, str) or _series(saved_series) != _series(series_id):
        raise ValueError("The saved V5 import identifies another series.")
    expected = build_source_import(source.get("source"), saved_series)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("The saved V5 import differs from its bound source fields.")
    # A historical result keeps its original cohort after current exclusions.
    current = list(runs)
    if (not current or len(set(current)) != len(current)
            or current != [run for run in expected["runs"] if run in set(current)]):
        raise ValueError("The imported V5 run list differs from the selected series run list.")
    return expected


def import_label(record):
    source = record["field_sources"]["v5_raw_photometry"]["source"]
    if record["v5_raw_photometry"].get("ac28_light_anchor") is True:
        anchor = "AC28 light anchor (saved ac28_light_anchor=true; no reference_run field)"
    else:
        anchor = "reference not available from the bound sources"
    return f"Imported (v5, {anchor}, bound in training request {source['sha256'][:8]}…)."
