"""Explicit series and reviewer isolation for training and prepared model validation.

Old bundles without a scope retain their historical cross-series behavior.
Series-specific models share algorithms, never another series' training rows.
A reviewer scope keeps only the chosen people's annotations, blind answers and
evidence actions; without one, everyone's evidence is used as before.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from darsia.fluidflower.reviewer_preferences import reviewer_identifier

# Training settings that bind a request to one frozen set of annotation IDs.
FROZEN_ANNOTATION_SET_SETTINGS = ("v5_chemistry_fit_scope", "v5_review_fit_partition")
FROZEN_ANNOTATION_SET_REFUSAL = ("This training route is bound to a frozen annotation set; "
                                 "a reviewer subset cannot be used with it")


def series_id(value):
    value = str(value).strip().casefold().replace("_", "")
    if not value or not value.isalnum():
        raise ValueError("A nonempty experimental-series ID is required")
    return value


def requested_series(settings):
    mode = settings.get("model_scope", "shared")
    if mode not in {"shared", "single-series"}:
        raise ValueError(f"Unknown model scope: {mode}")
    if mode == "shared":
        return None
    selected = {series_id(value) for value in settings.get("series") or []}
    if len(selected) != 1:
        raise ValueError("A series-specific model requires exactly one experimental series")
    return next(iter(selected))


def partition_bundle(decisions, manifest, selected):
    """Return a derived subset; source dictionaries and annotations stay intact."""
    selected = series_id(selected)
    assignments = {}
    for job in manifest.get("jobs") or []:
        run = str(job.get("run") or "").casefold()
        dataset = series_id(job.get("dataset") or "")
        if run in assignments and assignments[run] != dataset:
            raise ValueError(f"Ambiguous series assignment for {run}")
        assignments[run] = dataset
    kept = {}
    for run, decision in (decisions.get("decisions") or {}).items():
        if str(run).casefold() not in assignments:
            raise ValueError(f"No source job identifies series for {run}")
        if assignments[str(run).casefold()] == selected:
            kept[run] = decision
    jobs = [job for job in manifest.get("jobs") or []
            if series_id(job["dataset"]) == selected]
    return dict(decisions, decisions=kept), dict(manifest, jobs=jobs)


def requested_reviewers(settings):
    """Reviewer IDs selected by training settings; None (absent or empty) means everyone."""
    values = settings.get("reviewers")
    if not values:
        return None
    if isinstance(values, str):
        values = [values]
    return tuple(sorted({reviewer_identifier(value) for value in values}))


def evidence_reviewer(item, decision):
    """Reviewer ID of one annotation or answer.

    The item's reviewer_id, else the record's, else the slug of the reviewer
    name (the item's, else the record's).
    """
    for value in (item.get("reviewer_id"), decision.get("reviewer_id")):
        if str(value or "").strip():
            return reviewer_identifier(value)
    return reviewer_identifier(item.get("reviewer") or decision.get("reviewer") or "")


def partition_reviewers(decisions, selected):
    """Return a derived subset with only the selected reviewers' evidence.

    Annotations, blind answers and frame verdicts are kept per item; runs left
    with none of them are dropped. Source dictionaries stay intact, and
    ``selected`` None returns ``decisions`` itself.
    """
    if selected is None:
        return decisions
    chosen = set(selected)
    kept = {}
    for run, decision in (decisions.get("decisions") or {}).items():
        record = dict(decision)
        for field in ("annotations", "phase_quiz_answers", "frame_choices"):
            if field not in decision:
                continue
            items = decision.get(field) or []
            if isinstance(items, Mapping):
                # Historical {frame: verdict} maps name only the record's reviewer.
                record[field] = dict(items) if evidence_reviewer({}, decision) in chosen else {}
            else:
                record[field] = [item for item in items if isinstance(item, Mapping)
                                 and evidence_reviewer(item, decision) in chosen]
        if any(record.get(field) for field in ("annotations", "phase_quiz_answers", "frame_choices")):
            kept[run] = record
    return dict(decisions, decisions=kept)


def partition_review_events(snapshot, selected):
    """Keep only the selected reviewers' withdraw/reclassify events (by actor).

    Replacement annotations belong to their event, so other people's
    replacements leave with their events. ``selected`` None keeps ``snapshot``.
    """
    if selected is None:
        return snapshot
    from .. import phase_review_retractions as withdrawals
    checked = withdrawals.validate_snapshot(snapshot)
    chosen = set(selected)
    return withdrawals.snapshot_from_events(checked["series"], [
        event for event in checked["events"] if reviewer_identifier(event["actor"]) in chosen])


def require_reviewer_scope_allowed(settings):
    """Refuse a reviewer subset on a route bound to a frozen annotation set."""
    if requested_reviewers(settings) is not None and any(
            settings.get(key) for key in FROZEN_ANNOTATION_SET_SETTINGS):
        raise ValueError(FROZEN_ANNOTATION_SET_REFUSAL)


def validate_training_series(sources, regions, selected):
    if selected is None:
        return
    if not sources or not regions:
        raise ValueError(f"No reviewed evidence for {selected}")
    actual = {series_id(row.dataset) for row in (*sources, *regions)}
    if actual != {series_id(selected)}:
        raise ValueError(f"Series isolation failed: expected {selected}, found {sorted(actual)}")


def validation_jobs(model_root, jobs):
    """Prepared campaigns for a scoped model contain only its declared series."""
    root = Path(model_root)
    scopes = []
    for head in ("co2_presence", "gas_within_co2"):
        path = root / "models/generic" / f"{head}_signal.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("training_provenance") or {}
        scopes.append(provenance.get("series_scope"))
    if not any(scopes):
        return jobs
    if scopes[0] != scopes[1] or not scopes[0]:
        raise ValueError("Phase heads have inconsistent experimental-series scope")
    selected = series_id(scopes[0])
    allowed = [job for job in jobs if series_id(job.get("dataset") or "") == selected]
    if not allowed:
        raise ValueError(f"This model is restricted to {selected}; no matching experiments were supplied")
    return allowed
