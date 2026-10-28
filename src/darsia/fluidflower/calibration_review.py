"""Versioned scientific review of FluidFlower mass-calibration results."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from darsia.fluidflower.calibration import CalibrationPlanStore
from darsia.fluidflower.phase_interpretation import PhaseInterpretationRevisionStore
from darsia.fluidflower.project import sha256_file, utc_now

CALIBRATION_REVIEW_CONFIG_FORMAT = "darsia-fluidflower-calibration-review-config-v1"
CALIBRATION_REVIEW_REVISION_FORMAT = "darsia-fluidflower-calibration-review-v1"
CALIBRATION_REVIEW_STATUSES = ("pending", "approved", "needs_recalibration")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _append_jsonl as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Invalid calibration-review revision at line {line_number}"
                )
            rows.append(payload)
    return rows


def _json_digest(payload: Any) -> str:
    from darsia.fluidflower.analysis_domains import _json_digest as _canonical_ascii_digest
    return _canonical_ascii_digest(payload)


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    **details: Any,
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            **details,
        }
    )


def _expected_runs(project: FluidFlowerProject) -> list[str]:
    phase = PhaseInterpretationRevisionStore(project).current()
    selected = dict((phase.get("configuration") or {}).get("selected_candidates") or {})
    return sorted({str(run_id).strip().casefold() for run_id in selected if str(run_id).strip()})


def default_calibration_review_config(project: FluidFlowerProject) -> dict[str, Any]:
    """Return a draft bound to the current calibration plan."""

    calibration = CalibrationPlanStore(project).current()
    return {
        "schema": 1,
        "format": CALIBRATION_REVIEW_CONFIG_FORMAT,
        "calibration_plan_id": str(calibration.get("plan_id") or ""),
        "calibration_plan_sha256": str(calibration.get("content_sha256") or ""),
        "calibration_result_manifest_path": "",
        "run_reviews": {},
    }


def _asset_reference(raw_path: Any, *, relative_to: Path | None) -> dict[str, Any]:
    text = str(raw_path or "").strip()
    if not text:
        return {}
    path = Path(text).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    path = path.resolve()
    if not path.is_file():
        return {"path": str(path), "available": False}
    return {
        "path": str(path),
        "available": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_calibration_review_config(
    project: FluidFlowerProject,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one review without modifying calibration products."""

    candidate = deepcopy(dict(config))
    issues: list[dict[str, Any]] = []
    if candidate.get("format") != CALIBRATION_REVIEW_CONFIG_FORMAT:
        _issue(
            issues,
            "error",
            "unsupported_format",
            "Unsupported calibration-review configuration.",
        )

    calibration = CalibrationPlanStore(project).current()
    plan_id = str(calibration.get("plan_id") or "")
    plan_sha256 = str(calibration.get("content_sha256") or "")
    if not plan_id:
        _issue(
            issues,
            "error",
            "calibration_plan_missing",
            "Save a calibration plan before reviewing its result.",
        )
    elif (
        str(candidate.get("calibration_plan_id") or "") != plan_id
        or str(candidate.get("calibration_plan_sha256") or "") != plan_sha256
    ):
        _issue(
            issues,
            "error",
            "calibration_plan_changed",
            "The review no longer references the current calibration plan.",
        )

    # Local import avoids coupling the calibration planner to mass-analysis UI code.
    from darsia.fluidflower.mass_analysis import inspect_calibration_result

    expected_runs = _expected_runs(project)
    result = inspect_calibration_result(
        str(candidate.get("calibration_result_manifest_path") or ""),
        calibration,
        run_ids=expected_runs,
    )
    result_reference = dict(result.get("reference") or {})
    result_path = Path(str(result_reference.get("path") or "."))
    result_parent = result_path.parent if result_reference.get("path") else None
    result_sha256 = str(result_reference.get("sha256") or "")

    reviews = {
        str(run_id).strip().casefold(): dict(review)
        for run_id, review in dict(candidate.get("run_reviews") or {}).items()
        if isinstance(review, Mapping)
    }
    approved_runs: list[str] = []
    needs_recalibration: list[str] = []
    reviewed_runs: list[str] = []
    asset_references: dict[str, dict[str, Any]] = {}
    for run_id, review in sorted(reviews.items()):
        if run_id not in expected_runs:
            _issue(
                issues,
                "error",
                "unknown_reviewed_run",
                f"Calibration review references unknown run {run_id!r}.",
            )
            continue
        status = str(review.get("status") or "pending")
        if status not in CALIBRATION_REVIEW_STATUSES:
            _issue(
                issues,
                "error",
                "invalid_review_status",
                f"Unsupported calibration-review status for {run_id}: {status!r}.",
            )
            continue
        if status == "pending":
            continue
        reviewed_runs.append(run_id)
        reviewer = str(review.get("reviewer") or "").strip()
        if not reviewer:
            _issue(
                issues,
                "error",
                "reviewer_missing",
                f"Calibration review for {run_id} has no reviewer.",
            )
        if str(review.get("calibration_result_sha256") or "") != result_sha256:
            _issue(
                issues,
                "error",
                "reviewed_result_changed",
                f"The calibration result reviewed for {run_id} has changed.",
            )
        concentration = _asset_reference(
            review.get("concentration_preview_path"), relative_to=result_parent
        )
        mass_plot = _asset_reference(
            review.get("mass_plot_path"), relative_to=result_parent
        )
        asset_references[run_id] = {
            "concentration_preview": concentration,
            "mass_plot": mass_plot,
        }
        if status == "approved":
            for key, label, reference in (
                ("concentration_preview", "concentration map", concentration),
                ("mass_plot", "mass plot", mass_plot),
            ):
                if not reference.get("available"):
                    _issue(
                        issues,
                        "error",
                        f"{key}_missing",
                        f"Approved run {run_id} has no available {label}.",
                    )
            approved_runs.append(run_id)
        elif status == "needs_recalibration":
            needs_recalibration.append(run_id)
            _issue(
                issues,
                "warning",
                "run_needs_recalibration",
                f"Mass calibration needs revision for {run_id}.",
            )

    for run_id in sorted(set(expected_runs) - set(reviewed_runs)):
        _issue(
            issues,
            "warning",
            "run_not_mass_approved",
            f"Concentration and mass calibration have not been approved for {run_id}.",
        )

    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    all_runs_approved = bool(expected_runs) and len(approved_runs) == len(expected_runs)
    return {
        "schema": 1,
        "config_sha256": _json_digest(candidate),
        "valid": errors == 0,
        "ready": bool(result.get("valid")) and errors == 0 and all_runs_approved,
        "summary": {
            "error_count": errors,
            "warning_count": warnings,
            "run_count": len(expected_runs),
            "reviewed_run_count": len(reviewed_runs),
            "approved_run_count": len(approved_runs),
            "needs_recalibration_run_count": len(needs_recalibration),
            "calibration_result_valid": bool(result.get("valid")),
            "all_runs_approved": all_runs_approved,
        },
        "calibration_result": result,
        "asset_references": asset_references,
        "issues": issues,
    }


class CalibrationResultReviewStore:
    """Append-only reviews with a replaceable current pointer."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "calibration_review"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        from .series_paths import series_folder_path
        self.artifact_root = series_folder_path(project.root, "calibration_reviews")

    def revisions(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.revisions_path)

    def current(self) -> dict[str, Any]:
        from .analysis_domains import AnalysisDomainStore
        return AnalysisDomainStore.current(self)

    def save_review(
        self,
        config: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        candidate = deepcopy(dict(config))
        report = validate_calibration_review_config(self.project, candidate)
        if int((report.get("summary") or {}).get("error_count") or 0) > 0:
            raise ValueError("Calibration-result review has blocking errors")
        review_id = uuid.uuid4().hex
        revision = {
            "schema": 1,
            "format": CALIBRATION_REVIEW_REVISION_FORMAT,
            "review_id": review_id,
            "recorded_utc": utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "configuration": candidate,
            "validation": report,
            "scientific_scope": {
                "reviews_visible_2d_concentration": True,
                "reviews_mass_conversion": True,
                "modifies_calibration_products": False,
                "starts_mass_analysis": False,
            },
        }
        revision["content_sha256"] = _json_digest(revision)
        destination = self.artifact_root / review_id
        destination.mkdir(parents=True, exist_ok=False)
        _atomic_json(destination / "review.json", revision)
        _append_jsonl(self.revisions_path, revision)
        _atomic_json(self.current_path, revision)
        self.project._audit(
            "calibration_result_review_saved",
            actor,
            {
                "review_id": review_id,
                "calibration_plan_id": candidate.get("calibration_plan_id"),
                "approved_run_count": report["summary"]["approved_run_count"],
                "run_count": report["summary"]["run_count"],
            },
        )
        return deepcopy(revision)


if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject
