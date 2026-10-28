"""Versioned phase-product intake for FluidFlower experiments.

This module deliberately does not train a phase model. It records existing
phase campaigns, candidate selections, and manual-review files against the
exact upstream project state used to interpret them.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from darsia.fluidflower.frame_quality import FrameQualityStore
from darsia.fluidflower.geometry_facies import GeometryRevisionStore
from darsia.fluidflower.image_correction import CorrectionRevisionStore
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.timeline import TimelineStore

PHASE_INTERPRETATION_CONFIG_FORMAT = "darsia-fluidflower-phase-interpretation-config-v1"
PHASE_INTERPRETATION_REVISION_FORMAT = (
    "darsia-fluidflower-phase-interpretation-revision-v1"
)
PHASE_CLASSES = (
    "untouched_water",
    "aqueous_co2",
    "free_co2_gas",
    "mixed",
    "artifact",
    "uncertain",
)
PHASE_RUN_REVIEW_STATUSES = (
    "pending",
    "approved",
    "needs_local_correction",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _append_jsonl as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(
                    "Invalid phase-interpretation revision at line " f"{line_number}"
                )
            records.append(payload)
    return records


def _json_digest(payload: Any) -> str:
    from darsia.fluidflower.analysis_domains import _json_digest as _canonical_ascii_digest
    return _canonical_ascii_digest(payload)


def _file_reference(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "modified_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved),
    }


def default_phase_interpretation_config() -> dict[str, Any]:
    """Return a neutral intake configuration without model assumptions."""

    return {
        "schema": 1,
        "format": PHASE_INTERPRETATION_CONFIG_FORMAT,
        "campaign_manifest_path": "",
        "review_package_path": "",
        "selected_candidates": {},
        "run_reviews": {},
        "manual_review_files": [],
        "interpretation_scope": "visible_registered_camera_plane_2d",
        "phase_classes": list(PHASE_CLASSES),
        "training_requested": False,
    }


def inspect_phase_campaign(path: Path) -> dict[str, Any]:
    """Read one campaign manifest without modifying its files."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("The phase campaign manifest must contain one JSON object")
    jobs: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("jobs") or []):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Phase campaign job {index + 1} is not an object")
        run_id = str(raw.get("run") or "").strip().casefold()
        candidate_id = str(raw.get("candidate_id") or "").strip()
        role = str(raw.get("role") or raw.get("variant") or "").strip()
        if not run_id or not candidate_id or not role:
            raise ValueError(
                f"Phase campaign job {index + 1} lacks run, role, or candidate_id"
            )
        quality_flags = dict(raw.get("quality_flags") or {})
        jobs.append(
            {
                "job_id": str(raw.get("job_id") or f"job-{index + 1}"),
                "run": run_id,
                "role": role,
                "variant": str(raw.get("variant") or ""),
                "family": str(raw.get("family") or ""),
                "candidate_id": candidate_id,
                "dataset": str(
                    quality_flags.get("dataset") or payload.get("dataset") or ""
                ),
                "quality_flags": quality_flags,
                "numerical_products": deepcopy(
                    dict(raw.get("numerical_products") or {})
                ),
            }
        )
    if not jobs:
        raise ValueError("The phase campaign contains no candidate jobs")
    runs = sorted({job["run"] for job in jobs})
    candidates_by_run = {
        run_id: [job for job in jobs if job["run"] == run_id] for run_id in runs
    }
    review = dict(payload.get("review") or {})
    design = dict(payload.get("design") or {})
    return {
        "campaign_id": str(payload.get("campaign_id") or manifest_path.stem),
        "schema": payload.get("schema"),
        "dataset": str(payload.get("dataset") or ""),
        "runs": runs,
        "jobs": jobs,
        "candidates_by_run": candidates_by_run,
        "models": deepcopy(dict(payload.get("models") or {})),
        "design": deepcopy(design),
        "comparison": str(design.get("comparison") or ""),
        "interpretation_scope": str(
            design.get("interpretation_scope") or "visible_registered_camera_plane_2d"
        ),
        "mass_calibration_active": bool(design.get("mass_calibration_active")),
        "depth_used": bool(design.get("depth_used")),
        "deployment_allowed": bool(design.get("deployment_allowed")),
        "review_output_dir": str(review.get("output_dir") or ""),
        "manifest": _file_reference(manifest_path),
    }


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
            "details": details,
        }
    )


def validate_phase_interpretation_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate phase-product references and candidate assignments."""

    candidate = deepcopy(dict(config))
    issues: list[dict[str, Any]] = []
    if candidate.get("format") != PHASE_INTERPRETATION_CONFIG_FORMAT:
        _issue(
            issues,
            "error",
            "unsupported_format",
            "Unsupported phase-interpretation configuration",
        )
    if bool(candidate.get("training_requested")):
        _issue(
            issues,
            "error",
            "training_not_supported",
            "This intake revision cannot request model training.",
        )

    summary: dict[str, Any] = {}
    manifest_path = Path(str(candidate.get("campaign_manifest_path") or ""))
    try:
        summary = inspect_phase_campaign(manifest_path)
    except Exception as exc:  # noqa: BLE001
        _issue(
            issues,
            "error",
            "campaign_unavailable",
            f"Campaign manifest is unavailable: {type(exc).__name__}: {exc}",
        )

    review_path_text = str(candidate.get("review_package_path") or "").strip()
    if review_path_text:
        review_path = Path(review_path_text).expanduser()
        if not review_path.is_dir():
            _issue(
                issues,
                "warning",
                "review_package_unavailable",
                "The referenced manual-review folder is currently unavailable.",
                path=str(review_path),
            )

    available: dict[str, set[str]] = {}
    for run_id, rows in (summary.get("candidates_by_run") or {}).items():
        available[str(run_id)] = {str(row.get("candidate_id") or "") for row in rows}
    selected = {
        str(raw_run).strip().casefold(): str(raw_candidate).strip()
        for raw_run, raw_candidate in dict(
            candidate.get("selected_candidates") or {}
        ).items()
    }
    for run_id, candidate_id in selected.items():
        if run_id not in available:
            _issue(
                issues,
                "error",
                "unknown_selected_run",
                f"Selected run {run_id!r} is absent from the campaign.",
            )
        elif candidate_id not in available[run_id]:
            _issue(
                issues,
                "error",
                "unknown_selected_candidate",
                f"Selected candidate for {run_id} is absent from the campaign.",
                candidate_id=candidate_id,
            )
    for run_id in sorted(set(available) - set(selected)):
        _issue(
            issues,
            "warning",
            "candidate_not_selected",
            f"No final phase candidate is selected for {run_id}.",
        )

    run_reviews = {
        str(run_id).strip().casefold(): dict(review)
        for run_id, review in dict(candidate.get("run_reviews") or {}).items()
        if isinstance(review, Mapping)
    }
    approved_runs: list[str] = []
    correction_runs: list[str] = []
    reviewed_runs: list[str] = []
    for run_id, review in sorted(run_reviews.items()):
        if run_id not in available:
            _issue(
                issues,
                "error",
                "unknown_reviewed_run",
                f"Reviewed run {run_id!r} is absent from the campaign.",
            )
            continue
        status = str(review.get("status") or "pending")
        if status not in PHASE_RUN_REVIEW_STATUSES:
            _issue(
                issues,
                "error",
                "invalid_run_review_status",
                f"Unsupported 2D phase-review status for {run_id}: {status!r}.",
            )
            continue
        reviewed_candidate = str(review.get("candidate_id") or "")
        selected_candidate = str(selected.get(run_id) or "")
        if status != "pending" and reviewed_candidate != selected_candidate:
            _issue(
                issues,
                "error",
                "run_review_candidate_changed",
                f"The reviewed candidate for {run_id} is no longer selected.",
                reviewed_candidate=reviewed_candidate,
                selected_candidate=selected_candidate,
            )
        reviewer = str(review.get("reviewer") or "").strip()
        if status != "pending" and not reviewer:
            _issue(
                issues,
                "error",
                "run_review_reviewer_missing",
                f"The 2D phase review for {run_id} has no reviewer.",
            )
        if status == "approved":
            approved_runs.append(run_id)
            reviewed_runs.append(run_id)
        elif status == "needs_local_correction":
            correction_runs.append(run_id)
            reviewed_runs.append(run_id)
            _issue(
                issues,
                "warning",
                "run_needs_local_correction",
                f"The selected phase model needs local correction for {run_id}.",
            )
    for run_id in sorted(set(available) - set(reviewed_runs)):
        _issue(
            issues,
            "warning",
            "run_not_phase_approved",
            f"The visible 2D phase interpretation has not been approved for {run_id}.",
        )

    manual_files: list[dict[str, Any]] = []
    for raw_path in candidate.get("manual_review_files") or []:
        path = Path(str(raw_path)).expanduser()
        if not path.is_file():
            _issue(
                issues,
                "warning",
                "manual_review_file_unavailable",
                "A manual-review file is currently unavailable.",
                path=str(path),
            )
            continue
        manual_files.append(_file_reference(path))

    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    report = {
        "schema": 1,
        "config_sha256": _json_digest(candidate),
        "valid": error_count == 0,
        "summary": {
            "error_count": error_count,
            "warning_count": warning_count,
            "run_count": len(summary.get("runs") or []),
            "candidate_count": len(summary.get("jobs") or []),
            "selected_run_count": len(selected),
            "reviewed_run_count": len(reviewed_runs),
            "approved_run_count": len(approved_runs),
            "needs_correction_run_count": len(correction_runs),
            "manual_review_file_count": len(manual_files),
        },
        "campaign": summary,
        "manual_review_files": manual_files,
        "issues": issues,
    }
    return report


def _timeline_digest(project: FluidFlowerProject) -> dict[str, Any]:
    events = TimelineStore(project).effective_events()
    return {
        "event_count": len(events),
        "content_sha256": _json_digest(events),
    }


def _upstream_references(project: FluidFlowerProject) -> dict[str, Any]:
    geometry = GeometryRevisionStore(project).current()
    correction = CorrectionRevisionStore(project).current()
    frame_quality = FrameQualityStore(project).dependency_reference()
    return {
        "geometry_revision_id": str(geometry.get("revision_id") or ""),
        "geometry_config_sha256": str(
            ((geometry.get("validation") or {}).get("config_sha256")) or ""
        ),
        "image_correction_revision_id": str(correction.get("revision_id") or ""),
        "image_correction_revision_sha256": str(correction.get("content_sha256") or ""),
        "image_correction_dependency_sha256": (
            CorrectionRevisionStore.dependency_sha256(correction)
        ),
        "image_correction_algorithm": str(correction.get("algorithm") or ""),
        "frame_quality": frame_quality,
        "timeline": _timeline_digest(project),
        "module_states": {
            key: {
                "status": str(project.module_states[key].get("status") or ""),
                "updated_utc": project.module_states[key].get("updated_utc"),
            }
            for key in ("geometry_facies", "image_correction", "temporal_sync")
        },
    }


class PhaseInterpretationRevisionStore:
    """Append-only phase-intake revisions with a replaceable current pointer."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "phase_interpretation"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        self.artifact_root = project.root / "results" / "phase_interpretation"

    def revisions(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.revisions_path)

    def current(self) -> dict[str, Any]:
        from .analysis_domains import AnalysisDomainStore
        return AnalysisDomainStore.current(self)

    def save_revision(
        self,
        config: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
        validation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous = self.current()
        candidate = deepcopy(dict(config))
        report = deepcopy(dict(validation or {}))
        if report.get("config_sha256") != _json_digest(candidate):
            report = validate_phase_interpretation_config(candidate)
        if int((report.get("summary") or {}).get("error_count") or 0) > 0:
            raise ValueError(
                "Phase-interpretation configuration has blocking validation errors"
            )

        revision_id = uuid.uuid4().hex
        destination = self.artifact_root / revision_id
        destination.mkdir(parents=True, exist_ok=False)
        upstream = _upstream_references(self.project)
        campaign = dict(report.get("campaign") or {})
        revision = {
            "schema": 1,
            "format": PHASE_INTERPRETATION_REVISION_FORMAT,
            "revision_id": revision_id,
            "recorded_utc": utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "configuration": candidate,
            "campaign": campaign,
            "upstream": upstream,
            "validation": report,
            "scientific_scope": {
                "domain": "visible_registered_camera_plane_2d",
                "contains_depth_or_3d_extrapolation": False,
                "contains_concentration_or_mass": False,
                "trains_model": False,
            },
        }
        revision["content_sha256"] = _json_digest(
            {
                "configuration": candidate,
                "campaign_manifest_sha256": (
                    (campaign.get("manifest") or {}).get("sha256")
                ),
                "upstream": upstream,
                "manual_review_sha256": [
                    row.get("sha256") for row in report.get("manual_review_files") or []
                ],
            }
        )
        _atomic_json(destination / "revision.json", revision)
        _append_jsonl(self.revisions_path, revision)
        _atomic_json(self.current_path, revision)
        revision_changed = str(previous.get("content_sha256") or "") != str(
            revision.get("content_sha256") or ""
        )
        invalidated: tuple[str, ...] = ()
        if revision_changed:
            invalidated = self.project.set_module_status(
                "phase_interpretation",
                "review_required",
                updated_by=str(actor).strip(),
                reason=(
                    f"Phase revision {revision_id[:8]} changed the active color-path "
                    "or phase interpretation."
                ),
            )
            registered_run_ids = {
                str(row.get("run_id") or "")
                for row in self.project.manifest.get("runs") or []
                if str(row.get("run_id") or "")
            }
            selected_run_ids = [
                str(run_id)
                for run_id in (candidate.get("selected_candidates") or {})
                if str(run_id) in registered_run_ids
            ]
            if selected_run_ids:
                self.project.set_run_module_status(
                    selected_run_ids,
                    "phase_interpretation",
                    "review_required",
                    updated_by=str(actor).strip(),
                    reason=(
                        f"Phase revision {revision_id[:8]} changed the selected "
                        "2D interpretation."
                    ),
                )
        self.project._audit(
            "phase_interpretation_revision_saved",
            actor,
            {
                "revision_id": revision_id,
                "campaign_id": campaign.get("campaign_id"),
                "selected_candidates": candidate.get("selected_candidates") or {},
                "content_sha256": revision["content_sha256"],
                "invalidated_modules": list(invalidated),
            },
        )
        return deepcopy(revision)


def parse_review_file_list(value: str | Sequence[str]) -> list[str]:
    """Normalize semicolon/newline separated manual-review paths."""

    if isinstance(value, str):
        raw = value.replace("\n", ";").split(";")
    else:
        raw = [str(item) for item in value]
    output: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        normalized = str(Path(text).expanduser().resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject
