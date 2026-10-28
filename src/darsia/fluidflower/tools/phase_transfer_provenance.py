"""Audit fitted reference-calibration exposure without using context registries.

This covers the recorded supervised reference gate and facies-threshold fits.
It does not certify scientific independence or audit every historical design
decision. Missing inherited evidence never establishes that a run was unseen.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


PROFILE_FILES = ("phase_native_reference.json", "phase_facies_reference.json",
    "phase_reviewed_reference.json", "phase_reference_support.json", "phase_presence_context.json",
    "phase_plume_support.json", "phase_physics.json")
ANCESTRY_FILES = ("training_request.json", "physics_inheritance.json", "revision_provenance.json",
    "summary.json", "reference_calibration/NATIVE_BOUNDARY_README.json",
    "parent_provenance/training_request.json", "parent_provenance/revision_provenance.json")
KINDS = {"negative", "pure_positive", "regional_positive"}


def _fit_key(calibration):
    if not isinstance(calibration, dict) or not calibration.get("series") or not calibration.get("feature_version"):
        return None
    fit = calibration.get("fit") or {}
    required = ("mean", "scale", "coefficients", "intercept")
    if not isinstance(fit, dict) or not all(key in fit for key in required):
        return None
    try:
        arrays = [fit[key] for key in required[:-1]]
        if (not all(isinstance(values, list) and values for values in arrays)
                or len({len(values) for values in arrays}) != 1
                or not all(math.isfinite(float(value)) for values in arrays for value in values)
                or not math.isfinite(float(fit["intercept"]))):
            return None
    except (TypeError, ValueError):
        return None
    return json.dumps({"series": calibration.get("series"),
        "feature_version": calibration.get("feature_version"),
        "fit": {key: fit[key] for key in required}}, sort_keys=True, allow_nan=False)


def _run(value):
    return str(value or "").strip().casefold()


def _ancestor_paths(document, origin):
    """Follow declared model/calibration lineage, never scan unrelated models."""
    paths = []
    fields = ("parent", "parent_model", "calibration_parent", "inherited_from", "model_root")
    def add(value, file_parent=0):
        if not isinstance(value, str) or not value.strip():
            return
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = origin / path
        for _ in range(file_parent):
            path = path.parent
        paths.append(path.resolve())
    for field in fields:
        add(document.get(field))
    add(document.get("inherited_training_request"), 1)
    add(document.get("reference_calibration_evidence"), 2)
    for field in ("settings", "evaluation_parent"):
        value = document.get(field)
        if isinstance(value, dict):
            for name in fields:
                add(value.get(name))
    for name in (document.get("protected_sources") or {}):
        path = Path(name)
        if path.name in PROFILE_FILES:
            add(str(path), 1)
        elif path.name in {"region_features.json", "facies_regions.json"}:
            add(str(path), 2)
    return paths


def audit_calibration_scope(model_root):
    """Return positive exposure plus completeness of the selected frozen fits."""
    root = Path(model_root).expanduser().resolve()
    evidence, reasons, fitted_runs = {}, [], set()
    def read(path, role):
        path = Path(path)
        try:
            content = path.read_bytes()
            result = json.loads(content.decode("utf-8-sig"))
        except (OSError, ValueError, TypeError) as error:
            reasons.append(f"Unavailable {role}: {path} ({type(error).__name__})")
            return None
        evidence[str(path)] = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(), role=role)
        return result
    def profile_at(directory):
        available = [directory / name for name in PROFILE_FILES if (directory / name).is_file()]
        if len(available) != 1:
            return None
        value = read(available[0], "frozen_physics_profile")
        return value if isinstance(value, dict) else None
    selected = profile_at(root)
    calibration = (selected or {}).get("calibration") or {}
    try:
        expected_fit = _fit_key(calibration)
    except (ValueError, TypeError):
        expected_fit = None
    if expected_fit is None:
        return dict(fitted_runs=[], status="unverified", evidence=list(evidence.values()),
            reasons=[*reasons, "No auditable fitted reference-gate contract in the selected model"],
            audit_scope="recorded reference-gate and facies-threshold fits", independent_validation=False)
    needs_facies = bool(calibration.get("facies_thresholds"))
    gate_complete, facies_complete = False, not needs_facies
    pending, visited = [root], set()
    while pending and len(visited) < 32:
        directory = pending.pop(0)
        identity = str(directory).casefold()
        if identity in visited:
            continue
        visited.add(identity)
        profile = selected if directory == root else profile_at(directory)
        source_calibration = (profile or {}).get("calibration") or {}
        try:
            same_fit = _fit_key(source_calibration) == expected_fit
        except (ValueError, TypeError):
            same_fit = False
        if same_fit:
            folder = directory / "reference_calibration"
            inventory = read(folder / "evidence.json", "calibration_inventory") if (folder / "evidence.json").is_file() else None
            report = read(folder / "report.json", "calibration_report") if (folder / "report.json").is_file() else None
            # Positive exposure is useful even if original fitted-row files are
            # missing. These are calibration evidence inventories, not runs maps.
            if isinstance(inventory, dict):
                for row in inventory.get("regions") or []:
                    if isinstance(row, dict) and _run(row.get("run")):
                        fitted_runs.add(_run(row["run"]))
                for key in inventory.get("counts") or {}:
                    parts = key.split("/")
                    if len(parts) >= 2 and parts[-1] in KINDS:
                        fitted_runs.add(_run(parts[-2]))
                fitted_runs.update(_run(run) for run, count in
                    (inventory.get("old_negative_runs_without_registered_context") or {}).items() if count)
            for filename, role in (("region_features.json", "gate_fit_regions"),
                                   ("facies_regions.json", "facies_threshold_regions")):
                path = folder / filename
                if not path.is_file():
                    continue
                rows = read(path, role)
                if not isinstance(rows, list) or not rows:
                    continue
                valid = [row for row in rows if isinstance(row, dict) and _run(row.get("run"))
                    and row.get("kind") in KINDS
                    and row.get("series", calibration["series"]) == calibration["series"]]
                fitted_runs.update(_run(row["run"]) for row in valid)
                if len(valid) != len(rows) or not isinstance(report, dict) or not isinstance(inventory, dict):
                    continue
                report_calibration = report.get("calibration") or {}
                if (_fit_key(report_calibration) != expected_fit
                        or report_calibration.get("threshold") != source_calibration.get("threshold")):
                    continue
                if filename == "region_features.json":
                    counts = dict(Counter(f"{_run(row['run'])}/{row['kind']}" for row in rows))
                    kind_counts = dict(Counter(row["kind"] for row in rows))
                    report_counts = {key: value for key, value in (report.get("counts") or {}).items() if value}
                    threshold_matches = needs_facies or source_calibration.get("threshold") == calibration.get("threshold")
                    if (inventory.get("calibration_regions") == len(rows) and inventory.get("counts") == counts
                            and report_counts == kind_counts and threshold_matches):
                        gate_complete = True
                elif (needs_facies and source_calibration.get("facies_thresholds") == calibration["facies_thresholds"]
                        and source_calibration.get("threshold") == calibration.get("threshold")
                        and report.get("facies_thresholds") == calibration["facies_thresholds"]
                        and report.get("threshold") == calibration.get("threshold")
                        and inventory.get("calibration_uses_all_available_registered_rows") is True):
                    # Each material's report must cover every run represented in
                    # its threshold rows; a truncated inventory is not complete.
                    by_facies = {str(int(row["facies"])) for row in rows}
                    reports = report.get("facies") or {}
                    if by_facies == set(calibration["facies_thresholds"]) and all(
                            {_run(fold.get("excluded_run")) for fold in reports.get(label, {}).get("run_excluded_development_checks", [])}
                            == {_run(row["run"]) for row in rows if str(int(row["facies"])) == label}
                            for label in by_facies):
                        facies_complete = True
        if gate_complete and facies_complete:
            break
        for relative in ANCESTRY_FILES:
            path = directory / relative
            if not path.is_file():
                continue
            document = read(path, "declared_model_lineage")
            if isinstance(document, dict):
                pending.extend(_ancestor_paths(document, directory))
    if not gate_complete:
        reasons.append("Original gate fit/threshold evidence could not be fully matched to the selected frozen parameters")
    if not facies_complete:
        reasons.append("Inherited facies-threshold evidence could not be fully matched")
    if pending and len(visited) >= 32:
        reasons.append("Declared ancestry exceeded the bounded 32-model audit")
    return dict(fitted_runs=sorted(fitted_runs), status="audited" if gate_complete and facies_complete else "unverified",
        evidence=list(evidence.values()), reasons=reasons, gate_fit_evidence_complete=gate_complete,
        facies_threshold_evidence_complete=facies_complete,
        audit_scope="recorded reference-gate and facies-threshold fits", independent_validation=False)


def target_exposure(model_root, run, trained_runs, audit=None):
    audit = audit_calibration_scope(model_root) if audit is None else audit
    target = _run(run)
    optical = "seen" if target in {_run(value) for value in trained_runs} else "unseen"
    calibration = ("seen" if target in {_run(value) for value in audit["fitted_runs"]}
        else "unseen" if audit["status"] == "audited" else "unverified")
    optical_label = "In optical training" if optical == "seen" else "Not in listed optical training runs"
    history = {"seen": "Used in recorded reference calibration",
        "unseen": "Not in audited reference-calibration evidence",
        "unverified": "Calibration history not fully verified"}[calibration]
    return dict(optical_training_status=optical, calibration_evidence_status=calibration,
        exposure_label=f"{optical_label}; {history}", calibration_history=history)
