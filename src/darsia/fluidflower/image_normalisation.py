"""Explicit per-series lighting choices, separate from saved scientific records.

These records express a choice, not a completed correction or training gate.
The historical ``phase_normalisation`` record is never rewritten here.
"""

from __future__ import annotations

from copy import deepcopy
import datetime as dt
import getpass
import math
import os
from pathlib import Path
import re
import tomllib
import uuid


FORMAT = "series-image-normalisation-choice-v1"
OFF = "off"
MATCH_REFERENCE = "match_reference_lighting"
EQUALISE_SAND = "match_reference_lighting_equalise_sand"
CHOICE_DECISION_REF = "image-normalisation-method-choice"
METHODS = (
    (OFF, "Off"),
    (MATCH_REFERENCE, "Per-photo light fit on a band along the frame (old)"),
    (EQUALISE_SAND, "Match lighting and equalise sand types"),
)
# Sand-type equalisation: one smooth light field per run, fitted on the start
# photo's same-facies sand outside a user-chosen band along the rig frame.
EXCLUSION_BAND_KIND = "frame-facing-sand-band"
LIGHT_FIELD = dict(kind="start-photo-same-facies", degree=2)
COLOR_OFF = "off"
COLOR_DARSIA = "darsia_original"
COLOR_UNIFORM24 = "uniform24"
COLOR_SAVED_NATIVE = "saved_native"
COLOR_METHODS = (
    (COLOR_OFF, "Off"),
    (COLOR_DARSIA, "DarSIA original"),
    (COLOR_UNIFORM24, "Uniform24"),
    (COLOR_SAVED_NATIVE, "Saved native correction (existing)"),
)


def selected_corrections(choice):
    """Read independent choices without relabelling historical pixel methods.

    Old records enabled the native matrix and lighting together. Their absent
    colour choice retains that exact behaviour; it does not select DarSIA's
    separate ColorCorrection implementation or Uniform24 retroactively.
    """
    if not isinstance(choice, dict) or choice.get("method") not in (OFF, MATCH_REFERENCE, EQUALISE_SAND):
        raise ValueError("Choose a supported lighting method.")
    lighting = choice["method"] != OFF
    equalise = choice["method"] == EQUALISE_SAND
    if equalise and choice.get("color_method") in (None, COLOR_SAVED_NATIVE):
        raise ValueError("Equalising sand types needs Off, DarSIA original or Uniform24 colour correction.")
    color = choice.get("color_method", COLOR_SAVED_NATIVE if lighting else COLOR_OFF)
    if color not in dict(COLOR_METHODS):
        raise ValueError("Choose a supported ColorChecker correction.")
    result = dict(lighting=lighting, color_method=color)
    if equalise:
        # Only this method adds the key, so older choices keep their exact value.
        result["light_field"] = LIGHT_FIELD["kind"]
    return result


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _decision_record(value):
    """Validate a retained human choice without manufacturing a new event."""
    keys = {"actor", "approved_utc", "reason", "decision_ref"}
    if (not isinstance(value, dict) or set(value) != keys
            or any(not isinstance(value[key], str) or not value[key].strip() for key in keys)):
        raise ValueError("A saved method decision needs its human actor, time and reason.")
    if value["decision_ref"] != CHOICE_DECISION_REF:
        raise ValueError("The method decision needs its ordinary application choice reference.")
    try:
        approved = dt.datetime.fromisoformat(value["approved_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("The saved method decision needs a valid UTC time.") from error
    if approved.tzinfo is None or approved.utcoffset() != dt.timedelta(0):
        raise ValueError("The saved method decision needs a valid UTC time.")
    return deepcopy(value)


def choice_record(series_id, runs, method, reference=None, *, decision=None, color_method=None,
                  stable_band_mm=None, exclusion_band_mm=None):
    """Record a series choice; prepared images and models must bind it separately.

    ``stable_band_mm`` belongs to the old per-photo band fit. Sand-type
    equalisation instead takes ``exclusion_band_mm``: the band along the rig
    frame left out of its light fit (0 leaves nothing out).
    """
    if not isinstance(series_id, str) or not _key(series_id):
        raise ValueError("Choose an experimental series first.")
    if not isinstance(runs, (tuple, list)) or not all(isinstance(run, str) for run in runs):
        raise ValueError("Run identities must be explicit strings.")
    runs = tuple(run.casefold() for run in runs)
    if not runs or len({_key(r) for r in runs}) != len(runs) or any(not _key(r) for r in runs):
        raise ValueError("Choose a series with distinct nonempty run identities.")
    if method not in (OFF, MATCH_REFERENCE, EQUALISE_SAND):
        raise ValueError("Choose a supported lighting method.")
    corrections = selected_corrections(dict(method=method, **(
        {} if color_method is None else dict(color_method=color_method))))
    needs_reference = corrections["lighting"] or corrections["color_method"] != COLOR_OFF
    if not needs_reference:
        if reference is not None:
            raise ValueError("Off has no reference run.")
    elif reference not in runs:
        raise ValueError("Choose a reference belonging to this series.")
    record = dict(format=FORMAT, series_id=str(series_id), method=method,
                  runs=list(runs), selection="explicit", numerical_status="not_run")
    if color_method is not None:
        record["color_method"] = color_method
    if stable_band_mm is not None:
        if method == EQUALISE_SAND:
            raise ValueError("Equalising sand types leaves out a band along the frame; it fits no stable band.")
        if (isinstance(stable_band_mm, bool) or not isinstance(stable_band_mm, (int, float))
                or not math.isfinite(stable_band_mm) or stable_band_mm <= 0):
            raise ValueError("Choose a positive stable-area width in millimetres.")
        record["stable_area"] = dict(kind="sand-boundary-inward-band", width_mm=float(stable_band_mm))
    if method == EQUALISE_SAND:
        if (exclusion_band_mm is None or isinstance(exclusion_band_mm, bool)
                or not isinstance(exclusion_band_mm, (int, float))
                or not math.isfinite(exclusion_band_mm) or exclusion_band_mm < 0):
            raise ValueError("Choose the exclusion band along the frame in millimetres (0 leaves nothing out).")
        record["exclusion_band"] = dict(kind=EXCLUSION_BAND_KIND, width_mm=float(exclusion_band_mm))
        record["light_field"] = dict(LIGHT_FIELD)
    elif exclusion_band_mm is not None:
        raise ValueError("Only sand-type equalisation leaves out a band along the frame.")
    if reference is not None:
        record["reference_run"] = reference
    if decision is not None:
        record["decision"] = _decision_record(decision)
    return record


def selected_source_family(document):
    """Describe Setup's saved image selection without scanning source folders."""
    from .image_formats import image_family
    families = set()
    for run in (document.get("image_selection", {}).get("runs", {}) or {}).values():
        for row in run.get("selected", []):
            path = row.get("source_path") or row.get("image_id")
            family = image_family(path) if path else None
            if family in {"raw", "processed"}:
                families.add(family)
    return next(iter(families)) if len(families) == 1 else "mixed" if families else "unknown"


def with_choice(document, series_id, runs, method, reference=None, *, decision=None, color_method=None,
                stable_band_mm=None, exclusion_band_mm=None):
    """Return a candidate document; all existing values remain unchanged."""
    result = deepcopy(document)
    result["image_normalisation"] = choice_record(series_id, runs, method, reference,
        decision=decision, color_method=color_method, stable_band_mm=stable_band_mm,
        exclusion_band_mm=exclusion_band_mm)
    return result


def saved_choice(document, series_id, runs):
    """Read only a complete matching choice; never infer completion from it."""
    record = document.get("image_normalisation")
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("The image normalisation choice must be a table.")
    recorded_series = record.get("series_id")
    if not isinstance(recorded_series, str) or _key(recorded_series) != _key(series_id):
        raise ValueError("The image normalisation choice differs from this series.")
    decision = _decision_record(record["decision"]) if "decision" in record else None
    stable = record.get("stable_area")
    if stable is not None and (not isinstance(stable, dict)
            or set(stable) != {"kind", "width_mm"} or stable["kind"] != "sand-boundary-inward-band"):
        raise ValueError("Unknown stable image area.")
    excluded = record.get("exclusion_band")
    if excluded is not None and (not isinstance(excluded, dict)
            or set(excluded) != {"kind", "width_mm"} or excluded["kind"] != EXCLUSION_BAND_KIND):
        raise ValueError("Unknown exclusion band along the frame.")
    expected = choice_record(recorded_series, runs, record.get("method"), record.get("reference_run"),
                             decision=decision, color_method=record.get("color_method"),
                             stable_band_mm=stable["width_mm"] if stable else None,
                             exclusion_band_mm=excluded["width_mm"] if excluded else None)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("The image normalisation choice differs from this series.")
    return expected


def save_choice(series_root, series_id, runs, method, reference=None, *, expected_bytes,
                color_method=None, stable_band_mm=None, exclusion_band_mm=None):
    """Explicit metadata save with the same stale-manifest guard as series setup."""
    import toml
    from .series_method import _same_values

    path = Path(series_root) / "series.toml"
    original = path.read_bytes()
    if original != expected_bytes:
        raise ValueError("The series changed; reload before saving the method.")
    document = tomllib.loads(original.decode("utf-8-sig"))
    actual_runs = (document.get("runs") or {}).get("ids")
    if (_key(document.get("series_id")) != _key(series_id)
            or not isinstance(actual_runs, list)
            or tuple(r.casefold() for r in actual_runs) != tuple(runs)):
        raise ValueError("The selected method belongs to another series or run list.")
    reason = ("Selected Off for image normalisation in this series." if method == OFF
              else "Selected one start-photo light field per run from same-facies sand and reference run "
                   + str(reference).upper() + " for this series." if method == EQUALISE_SAND
              else "Selected reference lighting from run " + str(reference).upper() + " for this series.")
    if method == EQUALISE_SAND:
        reason += f" Band along the frame left out: {float(exclusion_band_mm):g} mm."
    if color_method is not None:
        reason += " ColorChecker correction: " + dict(COLOR_METHODS).get(color_method, str(color_method)) + "."
    decision = dict(actor=getpass.getuser(), approved_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    reason=reason, decision_ref=CHOICE_DECISION_REF)
    updated = with_choice(document, series_id, runs, method, reference,
                          decision=decision, color_method=color_method, stable_band_mm=stable_band_mm,
                          exclusion_band_mm=exclusion_band_mm)
    text = toml.dumps(updated)
    if b"\r\n" in original:
        text = text.replace("\n", "\r\n")
    raw = (b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b"") + text.encode("utf-8")
    if not _same_values(updated, tomllib.loads(raw.decode("utf-8-sig"))):
        raise ValueError("Saving the method would change unrelated series values.")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
        if path.read_bytes() != original:
            raise ValueError("The series changed while saving; reload the method.")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return raw
