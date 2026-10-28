"""Versioned quality-control and reporting plans for FluidFlower projects."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from darsia.fluidflower.modules import (
    MODULE_DEFINITIONS,
    SATISFIED_MODULE_STATUSES,
)
from darsia.fluidflower.post_analysis import (
    POST_ANALYSIS_REVISION_FORMAT,
    PostAnalysisPlanStore,
    inspect_post_analysis_result,
    validate_post_analysis_plan,
)
from darsia.fluidflower.project import sha256_file, utc_now

REPORTING_PLAN_FORMAT = "darsia-fluidflower-reporting-plan-v1"
REPORTING_REVISION_FORMAT = "darsia-fluidflower-reporting-revision-v1"
REPORTING_TASK_CONTRACT_FORMAT = "darsia-fluidflower-reporting-task-v1"

REPORT_EXPORT_FORMATS = (
    "html",
    "pdf",
    "docx",
    "pptx",
    "machine_readable_json",
    "provenance_bundle_zip",
)
REQUIRED_REPORT_EXPORTS = {"machine_readable_json", "provenance_bundle_zip"}
HUMAN_READABLE_REPORT_EXPORTS = {"html", "pdf", "docx", "pptx"}

REPORT_SECTIONS = (
    "executive_summary",
    "experimental_setup",
    "data_integrity",
    "geometry_facies",
    "image_correction",
    "temporal_alignment",
    "phase_interpretation",
    "calibration",
    "mass_analysis",
    "wasserstein_analysis",
    "post_analysis",
    "uncertainty_limitations",
    "reproducibility_provenance",
    "appendices",
)
REQUIRED_REPORT_SECTIONS = {
    "executive_summary",
    "experimental_setup",
    "data_integrity",
    "geometry_facies",
    "image_correction",
    "temporal_alignment",
    "phase_interpretation",
    "calibration",
    "mass_analysis",
    "post_analysis",
    "uncertainty_limitations",
    "reproducibility_provenance",
}

EVIDENCE_KINDS = (
    "protocol_or_source",
    "manual_review",
    "calibration_history",
    "uncertainty_evidence",
    "wasserstein_result",
    "supplementary_figure",
    "supplementary_movie",
    "supplementary_table",
    "other",
)

UNCERTAINTY_DEFINITIONS = (
    (
        "optical_correction",
        "Optical correction",
        "Residual illumination, ColorChecker, registration, and camera variation.",
    ),
    (
        "phase_interpretation",
        "Phase interpretation",
        "Uncertainty separating untouched water, aqueous CO2, and free gas in the visible slice.",
    ),
    (
        "chemistry_concentration",
        "Chemistry and concentration",
        "Indicator chemistry, pH, buffering, solubility, and transfer from optical signal to concentration.",
    ),
    (
        "geometry_volume",
        "Geometry and 2D-to-3D volume",
        "Depth, porosity, active area, and extrapolation from the camera-visible slice.",
    ),
    (
        "injection_overflow",
        "Injection and overflow",
        "Injection timing/rate and overflow-derived phase constraints or imputation.",
    ),
    (
        "temporal_alignment",
        "Temporal alignment",
        "Synchronization of images, injection, overflow, and interventions.",
    ),
    (
        "numerical_backend",
        "Numerical backend",
        "Precision, backend parity, optimization stochasticity, and numerical tolerances.",
    ),
    (
        "cross_run_transport",
        "Cross-run and transport comparison",
        "Resolution, detected-mass differences, normalization, and Wasserstein interpretation.",
    ),
)
UNCERTAINTY_STATUSES = (
    "unreviewed",
    "reviewed",
    "accepted_with_limitation",
    "not_applicable",
)

# Retained as empty compatibility exports for callers reading older plans. Report
# generation no longer has a sign-off workflow or publication watermark.
SIGNOFF_ROLES: tuple[str, ...] = ()
SIGNOFF_DECISIONS: tuple[str, ...] = ()


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
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _json_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


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


def _quality_snapshot(project: FluidFlowerProject) -> list[dict[str, Any]]:
    return [
        {
            "module": definition.key,
            "title": definition.title,
            "status": str(
                project.module_states[definition.key].get("status") or "not_started"
            ),
            "updated_utc": project.module_states[definition.key].get("updated_utc"),
            "updated_by": str(
                project.module_states[definition.key].get("updated_by") or ""
            ),
            "reason": str(project.module_states[definition.key].get("reason") or ""),
        }
        for definition in MODULE_DEFINITIONS
        if definition.key != "reporting"
    ]


def _default_uncertainties() -> list[dict[str, Any]]:
    return [
        {
            "uncertainty_id": uncertainty_id,
            "category": category,
            "statement": statement,
            "effect_on_results": "",
            "mitigation_or_evidence": "",
            "review_status": "unreviewed",
            "reviewed_by": "",
        }
        for uncertainty_id, category, statement in UNCERTAINTY_DEFINITIONS
    ]


def default_reporting_plan(project: FluidFlowerProject) -> dict[str, Any]:
    """Return a non-executing report-package plan for the current project."""

    post_plan = PostAnalysisPlanStore(project).current()
    author = str(project.manifest.get("created_by") or "").strip()
    return {
        "schema": 1,
        "format": REPORTING_PLAN_FORMAT,
        "post_analysis_plan_id": str(post_plan.get("plan_id") or ""),
        "post_analysis_plan_sha256": str(post_plan.get("content_sha256") or ""),
        "post_analysis_task_contract_sha256": str(
            (post_plan.get("task_contract") or {}).get("content_sha256") or ""
        ),
        "post_analysis_result_manifest_path": "",
        "identity": {
            "title": f"{project.name} analysis report",
            "subtitle": "Image-based CO2 phase interpretation and mass analysis",
            "report_version": "draft-1",
            "language": "English",
            "audience": "scientific_internal_review",
            "authors": [author] if author else [],
            "organization": "",
        },
        "sections": [
            section for section in REPORT_SECTIONS if section != "wasserstein_analysis"
        ],
        "exports": {
            "formats": list(REPORT_EXPORT_FORMATS),
            "large_media_policy": "link_with_hash_and_thumbnail",
            "small_artifact_policy": "copy_verified",
            "maximum_embedded_media_bytes": 25 * 1024 * 1024,
        },
        "evidence_files": [],
        "uncertainty_register": _default_uncertainties(),
        "scientific_contract": {
            "raw_detected_mass_is_authoritative": True,
            "injected_mass_is_reference_only": True,
            "force_mass_conservation": False,
            "visible_phase_scope": "camera_visible_2d_slice",
            "physical_volume_source": "approved_geometry_depth_porosity",
            "separate_phase_and_mass_uncertainty": True,
            "report_negative_and_null_findings": True,
        },
        "execution_profile": {
            "renderer": "isolated-process",
            "local_read_through_cache": True,
            "retry_transient_io": True,
            "maximum_io_retries": 8,
            "atomic_publish": True,
        },
        "output_root": str(project.root / "results" / "reporting" / "packages"),
        "execution_requested": False,
    }


def _inspect_evidence_files(raw_rows: Any) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    unavailable = False
    seen_ids: set[str] = set()
    if not isinstance(raw_rows, list):
        _issue(
            issues,
            "error",
            "evidence_invalid",
            "Evidence files must be represented as a list.",
        )
        raw_rows = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            _issue(
                issues,
                "error",
                "evidence_entry_invalid",
                "Every evidence entry must be a JSON object.",
                index=index,
            )
            continue
        row = dict(raw)
        evidence_id = str(row.get("evidence_id") or "").strip()
        kind = str(row.get("kind") or "").strip()
        title = str(row.get("title") or "").strip()
        raw_path = str(row.get("path") or "").strip()
        expected_sha256 = str(row.get("sha256") or "").strip()
        if not evidence_id or evidence_id in seen_ids:
            _issue(
                issues,
                "error",
                "evidence_id_invalid",
                "Evidence IDs must be present and unique.",
                index=index,
            )
            continue
        seen_ids.add(evidence_id)
        if kind not in EVIDENCE_KINDS or not title or not raw_path:
            _issue(
                issues,
                "error",
                "evidence_metadata_invalid",
                "Every evidence file requires a supported type, title, and path.",
                evidence_id=evidence_id,
            )
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            unavailable = True
            _issue(
                issues,
                "warning",
                "evidence_unavailable",
                "An evidence file is temporarily unavailable.",
                evidence_id=evidence_id,
                path=str(path),
            )
            continue
        actual_sha256 = sha256_file(path)
        if expected_sha256 and expected_sha256 != actual_sha256:
            _issue(
                issues,
                "error",
                "evidence_hash_mismatch",
                "An evidence file changed after it was selected.",
                evidence_id=evidence_id,
            )
            continue
        references.append(
            {
                "evidence_id": evidence_id,
                "kind": kind,
                "title": title,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": actual_sha256,
            }
        )
    errors = sum(issue["severity"] == "error" for issue in issues)
    return {
        "valid": errors == 0 and not unavailable,
        "references": references,
        "issues": issues,
    }


def reporting_task_contract(
    config: Mapping[str, Any],
    *,
    post_result_reference: Mapping[str, Any] | None = None,
    evidence_references: list[dict[str, Any]] | None = None,
    quality_snapshot: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    contract = {
        "schema": 1,
        "format": REPORTING_TASK_CONTRACT_FORMAT,
        "post_analysis_plan_id": str(config.get("post_analysis_plan_id") or ""),
        "post_analysis_plan_sha256": str(config.get("post_analysis_plan_sha256") or ""),
        "post_analysis_task_contract_sha256": str(
            config.get("post_analysis_task_contract_sha256") or ""
        ),
        "post_analysis_result": deepcopy(dict(post_result_reference or {})),
        "identity": deepcopy(dict(config.get("identity") or {})),
        "sections": deepcopy(list(config.get("sections") or [])),
        "exports": deepcopy(dict(config.get("exports") or {})),
        "evidence_files": deepcopy(list(evidence_references or [])),
        "uncertainty_register": deepcopy(
            list(config.get("uncertainty_register") or [])
        ),
        "scientific_contract": deepcopy(dict(config.get("scientific_contract") or {})),
        "quality_snapshot": deepcopy(list(quality_snapshot or [])),
        "execution_profile": deepcopy(dict(config.get("execution_profile") or {})),
        "output_root": str(config.get("output_root") or ""),
    }
    contract["content_sha256"] = _json_digest(contract)
    return contract


def validate_reporting_plan(
    project: FluidFlowerProject,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate report generation and publication readiness independently."""

    candidate = deepcopy(dict(config))
    issues: list[dict[str, Any]] = []
    if candidate.get("format") != REPORTING_PLAN_FORMAT:
        _issue(issues, "error", "unsupported_format", "Unsupported reporting plan")
    if bool(candidate.get("execution_requested")):
        _issue(
            issues,
            "error",
            "execution_not_supported",
            "Saving a reporting plan cannot start document rendering.",
        )

    post_plan = PostAnalysisPlanStore(project).current()
    post_plan_valid = False
    if post_plan.get("format") != POST_ANALYSIS_REVISION_FORMAT:
        _issue(
            issues,
            "error",
            "post_analysis_plan_missing",
            "Save a post-analysis plan before reporting.",
        )
    else:
        expected = (
            str(post_plan.get("plan_id") or ""),
            str(post_plan.get("content_sha256") or ""),
            str((post_plan.get("task_contract") or {}).get("content_sha256") or ""),
        )
        actual = (
            str(candidate.get("post_analysis_plan_id") or ""),
            str(candidate.get("post_analysis_plan_sha256") or ""),
            str(candidate.get("post_analysis_task_contract_sha256") or ""),
        )
        if actual != expected:
            _issue(
                issues,
                "error",
                "post_analysis_plan_changed",
                "The reporting plan does not reference the current post-analysis plan.",
            )
        post_report = validate_post_analysis_plan(
            project, dict(post_plan.get("configuration") or {})
        )
        post_plan_valid = bool(post_report.get("valid"))
        if not post_plan_valid:
            _issue(
                issues,
                "error",
                "post_analysis_plan_invalid",
                "The current post-analysis plan no longer matches its upstream state.",
            )

    identity = dict(candidate.get("identity") or {})
    if not str(identity.get("title") or "").strip():
        _issue(issues, "error", "report_title_missing", "Enter a report title.")
    if not str(identity.get("report_version") or "").strip():
        _issue(
            issues,
            "error",
            "report_version_missing",
            "Enter a report version.",
        )
    if identity.get("language") not in {"English", "Norwegian Bokmal"}:
        _issue(
            issues,
            "error",
            "report_language_invalid",
            "Select English or Norwegian Bokmal.",
        )
    if identity.get("audience") not in {
        "scientific_internal_review",
        "supervisor_and_developer_review",
        "publication_supplement",
        "public_release",
    }:
        _issue(
            issues,
            "error",
            "report_audience_invalid",
            "Select a supported report audience.",
        )
    authors = [str(value).strip() for value in identity.get("authors") or []]
    if (
        not authors
        or any(not value for value in authors)
        or len(authors) != len(set(authors))
    ):
        _issue(
            issues,
            "error",
            "report_authors_invalid",
            "Provide one or more unique named authors.",
        )

    sections = [str(value) for value in candidate.get("sections") or []]
    unsupported_sections = sorted(set(sections) - set(REPORT_SECTIONS))
    missing_sections = sorted(REQUIRED_REPORT_SECTIONS - set(sections))
    if not sections or len(sections) != len(set(sections)) or unsupported_sections:
        _issue(
            issues,
            "error",
            "report_sections_invalid",
            "Report sections must be unique and supported.",
            unsupported=unsupported_sections,
        )
    if missing_sections:
        _issue(
            issues,
            "error",
            "required_report_sections_missing",
            "Methods, uncertainty, and provenance cannot be omitted.",
            sections=missing_sections,
        )

    exports = dict(candidate.get("exports") or {})
    formats = [str(value) for value in exports.get("formats") or []]
    if (
        not formats
        or len(formats) != len(set(formats))
        or set(formats) - set(REPORT_EXPORT_FORMATS)
    ):
        _issue(
            issues,
            "error",
            "report_exports_invalid",
            "Select unique supported report formats.",
        )
    if not set(formats) & HUMAN_READABLE_REPORT_EXPORTS:
        _issue(
            issues,
            "error",
            "human_report_missing",
            "Select at least one human-readable report format.",
        )
    if not REQUIRED_REPORT_EXPORTS.issubset(formats):
        _issue(
            issues,
            "error",
            "provenance_exports_missing",
            "Machine-readable JSON and a provenance bundle are mandatory.",
        )
    if exports.get("large_media_policy") != "link_with_hash_and_thumbnail":
        _issue(
            issues,
            "error",
            "large_media_policy_invalid",
            "Large movies must remain hash-verified links with thumbnails.",
        )
    if exports.get("small_artifact_policy") != "copy_verified":
        _issue(
            issues,
            "error",
            "small_artifact_policy_invalid",
            "Small report artifacts must be copied after verification.",
        )
    maximum_embedded = exports.get("maximum_embedded_media_bytes")
    if (
        not isinstance(maximum_embedded, int)
        or isinstance(maximum_embedded, bool)
        or maximum_embedded < 0
    ):
        _issue(
            issues,
            "error",
            "embedded_media_limit_invalid",
            "The embedded-media limit must be a non-negative byte count.",
        )

    evidence = _inspect_evidence_files(candidate.get("evidence_files") or [])
    issues.extend(deepcopy(evidence["issues"]))
    if "wasserstein_analysis" in sections and not any(
        row["kind"] == "wasserstein_result" for row in evidence["references"]
    ):
        _issue(
            issues,
            "warning",
            "wasserstein_evidence_missing",
            "The Wasserstein section needs a verified result or should be deselected.",
        )

    uncertainties = candidate.get("uncertainty_register") or []
    uncertainty_ready = True
    expected_uncertainties = {row[0] for row in UNCERTAINTY_DEFINITIONS}
    observed_uncertainties: set[str] = set()
    if not isinstance(uncertainties, list):
        uncertainties = []
        _issue(
            issues,
            "error",
            "uncertainty_register_invalid",
            "The uncertainty register must be a list.",
        )
    for row in uncertainties:
        if not isinstance(row, Mapping):
            _issue(
                issues,
                "error",
                "uncertainty_entry_invalid",
                "Every uncertainty entry must be a JSON object.",
            )
            continue
        uncertainty_id = str(row.get("uncertainty_id") or "")
        if uncertainty_id in observed_uncertainties:
            _issue(
                issues,
                "error",
                "uncertainty_id_duplicate",
                "Uncertainty IDs must be unique.",
                uncertainty_id=uncertainty_id,
            )
        observed_uncertainties.add(uncertainty_id)
        if not str(row.get("statement") or "").strip():
            _issue(
                issues,
                "error",
                "uncertainty_statement_missing",
                "Every uncertainty requires a scientific statement.",
                uncertainty_id=uncertainty_id,
            )
        review_status = str(row.get("review_status") or "")
        if review_status not in UNCERTAINTY_STATUSES:
            _issue(
                issues,
                "error",
                "uncertainty_status_invalid",
                "Select a supported uncertainty review status.",
                uncertainty_id=uncertainty_id,
            )
        if review_status == "unreviewed" or (
            review_status != "not_applicable"
            and not str(row.get("reviewed_by") or "").strip()
        ):
            uncertainty_ready = False
    missing_uncertainties = sorted(expected_uncertainties - observed_uncertainties)
    if missing_uncertainties:
        _issue(
            issues,
            "error",
            "required_uncertainties_missing",
            "The report must address every standard uncertainty category.",
            uncertainty_ids=missing_uncertainties,
        )
    if not uncertainty_ready:
        _issue(
            issues,
            "warning",
            "uncertainties_not_reviewed",
            "One or more uncertainty statements still require named review.",
        )

    scientific = dict(candidate.get("scientific_contract") or {})
    required_scientific = {
        "raw_detected_mass_is_authoritative": True,
        "injected_mass_is_reference_only": True,
        "force_mass_conservation": False,
        "visible_phase_scope": "camera_visible_2d_slice",
        "physical_volume_source": "approved_geometry_depth_porosity",
        "separate_phase_and_mass_uncertainty": True,
        "report_negative_and_null_findings": True,
    }
    if scientific != required_scientific:
        _issue(
            issues,
            "error",
            "scientific_reporting_contract_changed",
            "Reporting cannot change mass semantics or hide uncertainty and null findings.",
        )

    execution = dict(candidate.get("execution_profile") or {})
    required_execution = {
        "renderer": "isolated-process",
        "local_read_through_cache": True,
        "retry_transient_io": True,
        "atomic_publish": True,
    }
    for key, expected_value in required_execution.items():
        if execution.get(key) != expected_value:
            _issue(
                issues,
                "error",
                "report_execution_contract_changed",
                "Report generation must remain isolated and interruption safe.",
                field=key,
            )
    retries = execution.get("maximum_io_retries")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 1:
        _issue(
            issues,
            "error",
            "report_io_retries_invalid",
            "At least one transient-I/O retry is required.",
        )
    if not str(candidate.get("output_root") or "").strip():
        _issue(
            issues,
            "error",
            "report_output_root_missing",
            "Choose a report-package output directory.",
        )

    post_result = inspect_post_analysis_result(
        str(candidate.get("post_analysis_result_manifest_path") or ""), post_plan
    )
    issues.extend(deepcopy(post_result["issues"]))

    quality = _quality_snapshot(project)
    required_quality_modules = {
        "setup",
        "data_integrity",
        "geometry_facies",
        "image_correction",
        "temporal_sync",
        "phase_interpretation",
        "calibration",
        "mass_analysis",
        "post_analysis",
    }
    if "wasserstein_analysis" in sections:
        required_quality_modules.add("wasserstein")
    unsatisfied = [
        row["module"]
        for row in quality
        if row["module"] in required_quality_modules
        and row["status"] not in SATISFIED_MODULE_STATUSES
    ]
    if unsatisfied:
        _issue(
            issues,
            "warning",
            "quality_gates_incomplete",
            "Publication is blocked by incomplete or outdated quality gates.",
            modules=unsatisfied,
        )

    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    valid = errors == 0
    generation_ready = (
        valid
        and post_plan_valid
        and bool(post_result.get("valid"))
        and bool(evidence.get("valid"))
        and (
            "wasserstein_analysis" not in sections
            or any(
                row["kind"] == "wasserstein_result" for row in evidence["references"]
            )
        )
    )
    publication_ready = (
        generation_ready
        and not unsatisfied
        and uncertainty_ready
    )
    contract = reporting_task_contract(
        candidate,
        post_result_reference=post_result.get("reference") or {},
        evidence_references=evidence["references"],
        quality_snapshot=quality,
    )
    return {
        "schema": 1,
        "config_sha256": _json_digest(candidate),
        "valid": valid,
        "execution_ready": generation_ready,
        "generation_ready": generation_ready,
        "publication_ready": publication_ready,
        "summary": {
            "error_count": errors,
            "warning_count": warnings,
            "section_count": len(sections),
            "export_count": len(formats),
            "evidence_count": len(evidence["references"]),
            "post_artifact_count": int(
                (post_result.get("reference") or {}).get("artifact_count") or 0
            ),
            "unreviewed_uncertainty_count": sum(
                str(row.get("review_status") or "") == "unreviewed"
                for row in uncertainties
                if isinstance(row, Mapping)
            ),
            "unsatisfied_quality_gate_count": len(unsatisfied),
        },
        "task_contract": contract,
        "post_analysis_result": post_result,
        "evidence": evidence,
        "quality_snapshot": quality,
        "unsatisfied_quality_modules": unsatisfied,
        "issues": issues,
    }


class ReportingPlanStore:
    """Append-only reporting plans; no document rendering occurs here."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "reporting"
        self.result_root = project.root / "results" / "reporting"
        self.artifact_root = self.result_root / "plans"
        self.current_path = self.override_root / "current.json"
        self.revisions_path = self.override_root / "revisions.jsonl"

    def revisions(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.revisions_path)

    def current(self) -> dict[str, Any]:
        from .analysis_domains import AnalysisDomainStore
        return AnalysisDomainStore.current(self)

    def save_plan(
        self,
        config: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
        validation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate = deepcopy(dict(config))
        report = validate_reporting_plan(self.project, candidate)
        if int((report.get("summary") or {}).get("error_count") or 0) > 0:
            raise ValueError("Reporting plan has blocking validation errors")
        canonical_evidence = [
            {
                "evidence_id": row["evidence_id"],
                "kind": row["kind"],
                "title": row["title"],
                "path": row["path"],
                "sha256": row["sha256"],
            }
            for row in report["evidence"]["references"]
        ]
        if len(canonical_evidence) == len(candidate.get("evidence_files") or []):
            candidate["evidence_files"] = canonical_evidence
            report = validate_reporting_plan(self.project, candidate)
        plan_id = uuid.uuid4().hex
        post_plan = PostAnalysisPlanStore(self.project).current()
        revision = {
            "schema": 1,
            "format": REPORTING_REVISION_FORMAT,
            "plan_id": plan_id,
            "recorded_utc": utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "configuration": candidate,
            "task_contract": deepcopy(report["task_contract"]),
            "upstream": {
                "post_analysis_plan_id": post_plan.get("plan_id"),
                "post_analysis_plan_sha256": post_plan.get("content_sha256"),
                "post_analysis_task_contract_sha256": (
                    post_plan.get("task_contract") or {}
                ).get("content_sha256"),
                "post_analysis_result": deepcopy(
                    report["post_analysis_result"].get("reference") or {}
                ),
            },
            "validation": report,
            "generation": {
                "ready": bool(report.get("generation_ready")),
                "requested": False,
                "started": False,
                "task_id": None,
            },
            "publication": {
                "ready": bool(report.get("publication_ready")),
                "published": False,
                "package_id": None,
            },
            "scientific_scope": {
                "contains_generated_documents": False,
                "changes_scientific_results": False,
                "raw_detected_mass_is_authoritative": True,
                "injected_mass_is_reference_only": True,
                "preserves_negative_and_null_findings": True,
                "freezes_quality_snapshot": True,
            },
        }
        revision["content_sha256"] = _json_digest(revision)
        destination = self.artifact_root / plan_id
        destination.mkdir(parents=True, exist_ok=False)
        _atomic_json(destination / "plan.json", revision)
        _append_jsonl(self.revisions_path, revision)
        _atomic_json(self.current_path, revision)
        self.project._audit(
            "reporting_plan_saved",
            actor,
            {
                "plan_id": plan_id,
                "task_contract_sha256": report["task_contract"]["content_sha256"],
                "generation_ready": bool(report.get("generation_ready")),
                "publication_ready": bool(report.get("publication_ready")),
                "generation_started": False,
            },
        )
        return deepcopy(revision)


if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject
