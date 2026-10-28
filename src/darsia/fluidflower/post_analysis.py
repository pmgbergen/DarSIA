"""Versioned post-analysis plans for FluidFlower mass products."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from darsia.fluidflower.analysis_domains import (
    ANALYSIS_DOMAIN_REVISION_FORMAT,
    AnalysisDomainStore,
    validate_analysis_domain_config,
)
from darsia.fluidflower.mass_analysis import (
    MASS_ANALYSIS_REVISION_FORMAT,
    MassAnalysisPlanStore,
    validate_mass_analysis_plan,
)
from darsia.fluidflower.modules import SATISFIED_MODULE_STATUSES
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.wasserstein import inspect_mass_analysis_result

POST_ANALYSIS_PLAN_FORMAT = "darsia-fluidflower-post-analysis-plan-v1"
POST_ANALYSIS_REVISION_FORMAT = "darsia-fluidflower-post-analysis-revision-v1"
POST_ANALYSIS_TASK_CONTRACT_FORMAT = "darsia-fluidflower-post-analysis-task-v1"
POST_ANALYSIS_RESULT_FORMAT = "darsia-fluidflower-post-analysis-result-v1"

POST_ANALYSIS_PRODUCTS = (
    "phase_map_movie_corrected_background",
    "phase_map_movie_black_background",
    "phase_contour_movie",
    "phase_contour_contact_sheet",
    "mass_curve_full_domain",
    "mass_curve_analysis_domains",
    "cross_run_mass_grid",
    "cross_run_phase_grid_movie",
    "integrated_mass_csv",
    "per_frame_diagnostics_csv",
    "provenance_json",
)

REQUIRED_POST_ANALYSIS_PRODUCTS = (
    "integrated_mass_csv",
    "per_frame_diagnostics_csv",
    "provenance_json",
)

DEFAULT_POST_ANALYSIS_PRODUCTS = tuple(
    product
    for product in POST_ANALYSIS_PRODUCTS
    if product != "phase_map_movie_black_background"
)

PHASE_VISUAL_PRODUCTS = {
    "phase_map_movie_corrected_background",
    "phase_map_movie_black_background",
    "phase_contour_movie",
    "phase_contour_contact_sheet",
    "cross_run_phase_grid_movie",
}

CROSS_RUN_POST_ANALYSIS_PRODUCTS = {
    "cross_run_mass_grid",
    "cross_run_phase_grid_movie",
}
GLOBAL_POST_ANALYSIS_PRODUCTS = {"provenance_json"}


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
                raise TypeError(f"Invalid post-analysis revision at line {line_number}")
            rows.append(payload)
    return rows


def _json_digest(payload: Any) -> str:
    from darsia.fluidflower.analysis_domains import _json_digest as _canonical_ascii_digest
    return _canonical_ascii_digest(payload)


def _number(value: Any) -> float | None:
    from darsia.fluidflower.mass_inputs import _number as _canonical_finite_number
    return _canonical_finite_number(value)


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


def _mass_run_ids(mass_plan: Mapping[str, Any]) -> list[str]:
    bindings = (mass_plan.get("configuration") or {}).get("run_bindings") or []
    return sorted(str(row.get("run_id") or "") for row in bindings if row.get("run_id"))


def _current_domain_state(project: FluidFlowerProject) -> tuple[dict, dict]:
    revision = AnalysisDomainStore(project).current()
    if revision.get("format") == ANALYSIS_DOMAIN_REVISION_FORMAT:
        return revision, deepcopy(dict(revision.get("configuration") or {}))
    return {}, {}


def default_post_analysis_plan(project: FluidFlowerProject) -> dict[str, Any]:
    """Return a non-executing post-analysis plan for current mass products."""

    mass_plan = MassAnalysisPlanStore(project).current()
    domain_revision, domain_config = _current_domain_state(project)
    domains = [
        str(row.get("domain_id") or "")
        for row in domain_config.get("domains") or []
        if row.get("domain_id") and bool(row.get("enabled", True))
    ]
    return {
        "schema": 1,
        "format": POST_ANALYSIS_PLAN_FORMAT,
        "mass_analysis_plan_id": str(mass_plan.get("plan_id") or ""),
        "mass_analysis_plan_sha256": str(mass_plan.get("content_sha256") or ""),
        "mass_task_contract_sha256": str(
            (mass_plan.get("task_contract") or {}).get("content_sha256") or ""
        ),
        "mass_result_manifest_path": "",
        "analysis_domain_revision_id": str(domain_revision.get("revision_id") or ""),
        "analysis_domain_revision_sha256": str(
            domain_revision.get("content_sha256") or ""
        ),
        "run_ids": _mass_run_ids(mass_plan),
        "products": list(DEFAULT_POST_ANALYSIS_PRODUCTS),
        "domains": ["whole_active_domain", *domains],
        "rendering": {
            "frame_policy": "all_mass_analysis_frames_chronological",
            "background_source": "approved_corrected_registered_experiment_image",
            "include_pre_injection_reference": True,
            "phase_map_representation": "filled_mutually_exclusive_phase_map",
            "phase_overlay_alpha": 0.58,
            "aqueous_levels_c_over_csat": [0.05, 0.30, 0.50],
            "free_gas_levels_saturation": [0.05],
            "show_injectors": True,
            "show_elapsed_time": True,
            "show_injected_mass": False,
            "movie_width_pixels": 1600,
            "movie_fps": 6.0,
            "movie_codec": "auto-compatible-mp4",
        },
        "mass_curves": {
            "components": ["mass_total", "mass_aqueous", "mass_free_gas"],
            "injected_mass_reference": "effective_synchronized_injection_protocol",
            "detected_mass_is_authoritative": True,
            "force_mass_conservation": False,
            "show_injected_mass_projection": False,
            "units": {"time": "hours", "mass": "grams"},
        },
        "execution_profile": {
            "frame_renderer": "cpu-parallel",
            "frame_workers": 24,
            "video_encoder": "auto-hardware-if-available",
            "local_read_through_cache": True,
            "cache_corrected_frames": True,
            "checkpoint_each_run": True,
            "resume_completed_runs": True,
            "retry_transient_io": True,
            "maximum_io_retries": 8,
            "atomic_publish": True,
        },
        "output_root": str(project.root / "results" / "post_analysis"),
        "execution_requested": False,
    }


def _required_mass_products(products: set[str]) -> set[str]:
    required = {"mass_total", "mass_aqueous", "mass_free_gas"}
    if products & PHASE_VISUAL_PRODUCTS:
        required.update({"concentration_aqueous", "saturation_free_gas"})
    if products & {
        "phase_map_movie_corrected_background",
        "phase_contour_movie",
        "phase_contour_contact_sheet",
        "cross_run_phase_grid_movie",
    }:
        required.add("preview_images")
    return required


def _resolved_artifact_path(raw: str, *, manifest_path: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(manifest_path).expanduser().resolve().parent / path
    return path.resolve()


def inspect_post_analysis_inputs(
    mass_result: Mapping[str, Any],
    *,
    selected_runs: set[str],
    required_products: set[str],
) -> dict[str, Any]:
    """Verify post-analysis-only mass products for selected runs."""

    issues: list[dict[str, Any]] = []
    if not mass_result.get("available"):
        return {"valid": False, "artifacts": [], "issues": issues}
    manifest = dict(mass_result.get("manifest") or {})
    manifest_path = str((mass_result.get("reference") or {}).get("path") or "")
    artifacts: list[dict[str, Any]] = []
    unavailable = False
    rows = {
        str(row.get("run_id") or ""): row
        for row in manifest.get("run_results") or []
        if isinstance(row, Mapping)
    }
    for run_id in sorted(selected_runs):
        row = rows.get(run_id)
        if row is None:
            _issue(
                issues,
                "error",
                "selected_run_result_missing",
                f"Mass result does not contain selected run {run_id}.",
            )
            continue
        products = row.get("products") or {}
        if not isinstance(products, Mapping):
            products = {}
        for product in sorted(required_products):
            if product in {"mass_total", "mass_aqueous", "mass_free_gas"}:
                continue
            reference = products.get(product)
            if not isinstance(reference, Mapping):
                _issue(
                    issues,
                    "error",
                    "post_analysis_input_missing",
                    f"{run_id} lacks required product {product}.",
                )
                continue
            raw_path = str(reference.get("path") or "").strip()
            expected_sha256 = str(reference.get("sha256") or "").strip()
            if not raw_path or not expected_sha256:
                _issue(
                    issues,
                    "error",
                    "post_analysis_input_reference_invalid",
                    f"{run_id}/{product} has no complete path and hash reference.",
                )
                continue
            path = _resolved_artifact_path(raw_path, manifest_path=manifest_path)
            if not path.is_file():
                unavailable = True
                _issue(
                    issues,
                    "warning",
                    "post_analysis_input_unavailable",
                    f"{run_id}/{product} is temporarily unavailable.",
                    path=str(path),
                )
                continue
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_sha256:
                _issue(
                    issues,
                    "error",
                    "post_analysis_input_hash_mismatch",
                    f"{run_id}/{product} failed its hash check.",
                )
                continue
            artifacts.append(
                {
                    "run_id": run_id,
                    "product": product,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": actual_sha256,
                }
            )
    errors = sum(issue["severity"] == "error" for issue in issues)
    return {
        "valid": errors == 0 and not unavailable,
        "artifacts": artifacts,
        "issues": issues,
    }


def expected_post_analysis_artifacts(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the exact product/run coverage required by a post-analysis plan."""

    run_ids = sorted({str(value) for value in config.get("run_ids") or []})
    expected: list[dict[str, Any]] = []
    for product in sorted({str(value) for value in config.get("products") or []}):
        if product in GLOBAL_POST_ANALYSIS_PRODUCTS:
            expected.append({"product": product, "run_ids": []})
        elif product in CROSS_RUN_POST_ANALYSIS_PRODUCTS:
            expected.append({"product": product, "run_ids": run_ids})
        else:
            expected.extend(
                {"product": product, "run_ids": [run_id]} for run_id in run_ids
            )
    return expected


def inspect_post_analysis_result(
    raw_path: str,
    post_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect a post-analysis result without accepting unverified artifacts."""

    issues: list[dict[str, Any]] = []
    path_text = str(raw_path or "").strip()
    if not path_text:
        _issue(
            issues,
            "warning",
            "post_analysis_result_missing",
            "Attach a post-analysis result manifest before downstream use.",
        )
        return {
            "available": False,
            "valid": False,
            "complete": False,
            "manifest": {},
            "reference": {},
            "issues": issues,
        }
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        _issue(
            issues,
            "warning",
            "post_analysis_result_unavailable",
            "The post-analysis result is temporarily unavailable.",
            path=str(path),
        )
        return {
            "available": False,
            "valid": False,
            "complete": False,
            "manifest": {},
            "reference": {"path": str(path)},
            "issues": issues,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "error",
            "post_analysis_result_invalid_json",
            "The post-analysis result is not valid JSON.",
            error=f"{type(exc).__name__}: {exc}",
        )
        payload = {}
    if not isinstance(payload, dict):
        _issue(
            issues,
            "error",
            "post_analysis_result_invalid",
            "The post-analysis result must be a JSON object.",
        )
        payload = {}
    if payload.get("format") != POST_ANALYSIS_RESULT_FORMAT:
        _issue(
            issues,
            "error",
            "post_analysis_result_format_invalid",
            "Unsupported post-analysis result format.",
        )
    expected_pointer = (
        str(post_plan.get("plan_id") or ""),
        str((post_plan.get("task_contract") or {}).get("content_sha256") or ""),
    )
    actual_pointer = (
        str(payload.get("post_analysis_plan_id") or ""),
        str(payload.get("post_analysis_task_contract_sha256") or ""),
    )
    if actual_pointer != expected_pointer:
        _issue(
            issues,
            "error",
            "post_analysis_result_plan_mismatch",
            "The result was not produced from the current post-analysis task.",
        )
    complete = payload.get("complete") is True
    if not complete:
        _issue(
            issues,
            "warning",
            "post_analysis_result_incomplete",
            "Post-analysis has not published a complete result yet.",
        )

    config = dict(post_plan.get("configuration") or {})
    expected_rows = expected_post_analysis_artifacts(config)
    expected = {(row["product"], tuple(row["run_ids"])) for row in expected_rows}
    observed: set[tuple[str, tuple[str, ...]]] = set()
    verified: list[dict[str, Any]] = []
    unavailable = False
    artifact_ids: set[str] = set()
    for index, raw in enumerate(payload.get("artifacts") or []):
        if not isinstance(raw, Mapping):
            _issue(
                issues,
                "error",
                "post_analysis_artifact_invalid",
                "Every post-analysis artifact must be a JSON object.",
                index=index,
            )
            continue
        artifact = dict(raw)
        artifact_id = str(artifact.get("artifact_id") or "").strip()
        product = str(artifact.get("product") or "").strip()
        run_ids = tuple(sorted({str(value) for value in artifact.get("run_ids") or []}))
        signature = (product, run_ids)
        if not artifact_id or artifact_id in artifact_ids:
            _issue(
                issues,
                "error",
                "post_analysis_artifact_id_invalid",
                "Artifact IDs must be present and unique.",
                index=index,
            )
            continue
        artifact_ids.add(artifact_id)
        if signature not in expected:
            _issue(
                issues,
                "warning",
                "post_analysis_artifact_unplanned",
                "The result contains an artifact outside the current plan.",
                artifact_id=artifact_id,
                product=product,
                run_ids=list(run_ids),
            )
        if signature in observed:
            _issue(
                issues,
                "error",
                "post_analysis_artifact_duplicate",
                "The same planned product coverage appears more than once.",
                product=product,
                run_ids=list(run_ids),
            )
            continue
        observed.add(signature)
        raw_artifact_path = str(artifact.get("path") or "").strip()
        expected_sha256 = str(artifact.get("sha256") or "").strip()
        if not raw_artifact_path or not expected_sha256:
            _issue(
                issues,
                "error",
                "post_analysis_artifact_reference_invalid",
                "Every artifact requires a path and SHA-256 digest.",
                artifact_id=artifact_id,
            )
            continue
        artifact_path = _resolved_artifact_path(
            raw_artifact_path, manifest_path=str(path)
        )
        if not artifact_path.is_file():
            unavailable = True
            _issue(
                issues,
                "warning",
                "post_analysis_artifact_unavailable",
                "A post-analysis artifact is temporarily unavailable.",
                artifact_id=artifact_id,
                path=str(artifact_path),
            )
            continue
        actual_sha256 = sha256_file(artifact_path)
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "error",
                "post_analysis_artifact_hash_mismatch",
                "A post-analysis artifact failed its hash check.",
                artifact_id=artifact_id,
            )
            continue
        verified.append(
            {
                "artifact_id": artifact_id,
                "product": product,
                "run_ids": list(run_ids),
                "path": str(artifact_path),
                "media_type": str(artifact.get("media_type") or ""),
                "size_bytes": artifact_path.stat().st_size,
                "sha256": actual_sha256,
            }
        )
    missing = expected - observed
    for product, run_ids in sorted(missing):
        _issue(
            issues,
            "error" if complete else "warning",
            "post_analysis_artifact_missing",
            "The result does not contain every artifact required by the plan.",
            product=product,
            run_ids=list(run_ids),
        )
    errors = sum(issue["severity"] == "error" for issue in issues)
    reference = {
        "path": str(path),
        "sha256": sha256_file(path),
        "artifact_count": len(verified),
        "artifacts": verified,
    }
    return {
        "available": True,
        "valid": errors == 0 and complete and not unavailable and not missing,
        "complete": complete,
        "manifest": payload,
        "reference": reference,
        "expected_artifacts": expected_rows,
        "issues": issues,
    }


def post_analysis_task_contract(
    config: Mapping[str, Any],
    *,
    mass_result_reference: Mapping[str, Any] | None = None,
    analysis_domain_reference: Mapping[str, Any] | None = None,
    input_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    contract = {
        "schema": 1,
        "format": POST_ANALYSIS_TASK_CONTRACT_FORMAT,
        "mass_analysis_plan_id": str(config.get("mass_analysis_plan_id") or ""),
        "mass_analysis_plan_sha256": str(config.get("mass_analysis_plan_sha256") or ""),
        "mass_task_contract_sha256": str(config.get("mass_task_contract_sha256") or ""),
        "mass_result": deepcopy(dict(mass_result_reference or {})),
        "analysis_domains": deepcopy(dict(analysis_domain_reference or {})),
        "input_artifacts": deepcopy(list(input_artifacts or [])),
        "run_ids": deepcopy(list(config.get("run_ids") or [])),
        "products": deepcopy(list(config.get("products") or [])),
        "domains": deepcopy(list(config.get("domains") or [])),
        "rendering": deepcopy(dict(config.get("rendering") or {})),
        "mass_curves": deepcopy(dict(config.get("mass_curves") or {})),
        "execution_profile": deepcopy(dict(config.get("execution_profile") or {})),
        "output_root": str(config.get("output_root") or ""),
    }
    contract["content_sha256"] = _json_digest(contract)
    return contract


def _validate_levels(
    issues: list[dict[str, Any]], raw: Any, *, code: str, label: str
) -> list[float]:
    if not isinstance(raw, list):
        _issue(issues, "error", code, f"{label} must be a list.")
        return []
    values = [_number(value) for value in raw]
    if (
        not values
        or any(value is None for value in values)
        or any(not (0.0 < float(value) <= 1.0) for value in values)
        or len(values) != len(set(values))
    ):
        _issue(
            issues,
            "error",
            code,
            f"{label} must contain unique fractions in (0, 1].",
        )
        return []
    return sorted(float(value) for value in values if value is not None)


def validate_post_analysis_plan(
    project: FluidFlowerProject,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a post-analysis plan without rendering frames or movies."""

    candidate = deepcopy(dict(config))
    issues: list[dict[str, Any]] = []
    if candidate.get("format") != POST_ANALYSIS_PLAN_FORMAT:
        _issue(issues, "error", "unsupported_format", "Unsupported post-analysis plan")
    if bool(candidate.get("execution_requested")):
        _issue(
            issues,
            "error",
            "execution_not_supported",
            "Saving a post-analysis plan cannot start rendering.",
        )

    mass_plan = MassAnalysisPlanStore(project).current()
    mass_plan_ready = False
    if mass_plan.get("format") != MASS_ANALYSIS_REVISION_FORMAT:
        _issue(
            issues,
            "error",
            "mass_plan_missing",
            "Save a mass-analysis plan before post-analysis planning.",
        )
    else:
        expected = (
            str(mass_plan.get("plan_id") or ""),
            str(mass_plan.get("content_sha256") or ""),
            str((mass_plan.get("task_contract") or {}).get("content_sha256") or ""),
        )
        actual = (
            str(candidate.get("mass_analysis_plan_id") or ""),
            str(candidate.get("mass_analysis_plan_sha256") or ""),
            str(candidate.get("mass_task_contract_sha256") or ""),
        )
        if actual != expected:
            _issue(
                issues,
                "error",
                "mass_plan_changed",
                "Post-analysis plan does not reference the current mass plan.",
            )
        mass_report = validate_mass_analysis_plan(
            project, dict(mass_plan.get("configuration") or {})
        )
        if not mass_report.get("valid"):
            _issue(
                issues,
                "error",
                "mass_plan_invalid",
                "Current mass plan no longer matches its upstream state.",
            )
        mass_plan_ready = bool(mass_report.get("execution_ready"))
        if not mass_plan_ready:
            _issue(
                issues,
                "warning",
                "mass_plan_not_execution_ready",
                "Current mass plan is valid but not execution ready.",
            )

    expected_runs = set(_mass_run_ids(mass_plan))
    run_ids = sorted({str(value) for value in candidate.get("run_ids") or []})
    if not run_ids:
        _issue(issues, "error", "runs_missing", "Select at least one run.")
    if not set(run_ids).issubset(expected_runs):
        _issue(
            issues,
            "error",
            "run_set_invalid",
            "Selected post-analysis runs must come from the current mass plan.",
        )

    products = [str(value) for value in candidate.get("products") or []]
    if not products or len(products) != len(set(products)):
        _issue(
            issues,
            "error",
            "products_invalid",
            "Select one or more unique post-analysis products.",
        )
    unsupported_products = sorted(set(products) - set(POST_ANALYSIS_PRODUCTS))
    if unsupported_products:
        _issue(
            issues,
            "error",
            "products_unsupported",
            "Post-analysis plan contains unsupported products.",
            products=unsupported_products,
        )
    missing_required = sorted(set(REQUIRED_POST_ANALYSIS_PRODUCTS) - set(products))
    if missing_required:
        _issue(
            issues,
            "error",
            "required_products_missing",
            "Integrated mass, frame diagnostics, and provenance are mandatory.",
            products=missing_required,
        )
    cross_run_products = {
        "cross_run_mass_grid",
        "cross_run_phase_grid_movie",
    }
    if len(run_ids) < 2 and set(products) & cross_run_products:
        _issue(
            issues,
            "error",
            "cross_run_product_without_runs",
            "Cross-run products require at least two selected runs.",
        )

    domain_revision, domain_config = _current_domain_state(project)
    domain_valid = True
    expected_domain_pointer = (
        str(domain_revision.get("revision_id") or ""),
        str(domain_revision.get("content_sha256") or ""),
    )
    actual_domain_pointer = (
        str(candidate.get("analysis_domain_revision_id") or ""),
        str(candidate.get("analysis_domain_revision_sha256") or ""),
    )
    if actual_domain_pointer != expected_domain_pointer:
        domain_valid = False
        _issue(
            issues,
            "error",
            "analysis_domains_changed",
            "Post-analysis plan does not reference the current analysis boxes.",
        )
    if domain_revision:
        domain_report = validate_analysis_domain_config(project, domain_config)
        domain_valid = domain_valid and bool(domain_report.get("valid"))
        if not domain_report.get("valid"):
            _issue(
                issues,
                "error",
                "analysis_domains_invalid",
                "Current analysis boxes no longer match geometry or correction.",
            )
    domain_rows = {
        str(row.get("domain_id") or ""): deepcopy(dict(row))
        for row in domain_config.get("domains") or []
        if row.get("domain_id") and bool(row.get("enabled", True))
    }
    domains = [str(value) for value in candidate.get("domains") or []]
    if not domains or len(domains) != len(set(domains)):
        _issue(
            issues,
            "error",
            "domains_invalid",
            "Select one or more unique analysis domains.",
        )
    if "whole_active_domain" not in domains:
        _issue(
            issues,
            "error",
            "whole_domain_missing",
            "The whole active rig domain is required for quality control.",
        )
    unsupported_domains = sorted(set(domains) - {"whole_active_domain", *domain_rows})
    if unsupported_domains:
        _issue(
            issues,
            "error",
            "domains_unsupported",
            "Post-analysis plan contains unavailable analysis boxes.",
            domains=unsupported_domains,
        )
    analysis_domain_reference = {
        "revision_id": str(domain_revision.get("revision_id") or ""),
        "revision_sha256": str(domain_revision.get("content_sha256") or ""),
        "coordinate_system": "corrected_registered_rig_grid",
        "domains": [
            domain_rows[domain_id] for domain_id in domains if domain_id in domain_rows
        ],
    }

    rendering = dict(candidate.get("rendering") or {})
    required_rendering = {
        "frame_policy": "all_mass_analysis_frames_chronological",
        "background_source": "approved_corrected_registered_experiment_image",
        "include_pre_injection_reference": True,
        "phase_map_representation": "filled_mutually_exclusive_phase_map",
        "show_injected_mass": False,
    }
    for key, expected_value in required_rendering.items():
        if rendering.get(key) != expected_value:
            _issue(
                issues,
                "error",
                "rendering_contract_changed",
                "Post-analysis rendering semantics changed.",
                field=key,
            )
    alpha = _number(rendering.get("phase_overlay_alpha"))
    if alpha is None or not 0.0 <= alpha <= 1.0:
        _issue(
            issues,
            "error",
            "phase_alpha_invalid",
            "Phase-map opacity must be between zero and one.",
        )
    width = rendering.get("movie_width_pixels")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not 640 <= width <= 7680
    ):
        _issue(
            issues,
            "error",
            "movie_width_invalid",
            "Movie width must be an integer between 640 and 7680 pixels.",
        )
    fps = _number(rendering.get("movie_fps"))
    if fps is None or not 0.25 <= fps <= 60.0:
        _issue(
            issues,
            "error",
            "movie_fps_invalid",
            "Movie frame rate must be between 0.25 and 60 fps.",
        )
    _validate_levels(
        issues,
        rendering.get("aqueous_levels_c_over_csat"),
        code="aqueous_levels_invalid",
        label="Aqueous c/csat contour levels",
    )
    _validate_levels(
        issues,
        rendering.get("free_gas_levels_saturation"),
        code="gas_levels_invalid",
        label="Free-gas saturation contour levels",
    )

    mass_curves = dict(candidate.get("mass_curves") or {})
    required_mass_curve_contract = {
        "components": ["mass_total", "mass_aqueous", "mass_free_gas"],
        "injected_mass_reference": "effective_synchronized_injection_protocol",
        "detected_mass_is_authoritative": True,
        "force_mass_conservation": False,
        "show_injected_mass_projection": False,
        "units": {"time": "hours", "mass": "grams"},
    }
    if mass_curves != required_mass_curve_contract:
        _issue(
            issues,
            "error",
            "mass_curve_contract_changed",
            "Mass curves must preserve raw detected phase masses and use injection only as reference.",
        )

    execution = dict(candidate.get("execution_profile") or {})
    workers = execution.get("frame_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        _issue(
            issues,
            "error",
            "workers_invalid",
            "Frame worker count must be a positive integer.",
        )
    required_execution = {
        "frame_renderer": "cpu-parallel",
        "local_read_through_cache": True,
        "cache_corrected_frames": True,
        "checkpoint_each_run": True,
        "resume_completed_runs": True,
        "retry_transient_io": True,
        "atomic_publish": True,
    }
    for key, expected_value in required_execution.items():
        if execution.get(key) != expected_value:
            _issue(
                issues,
                "error",
                "execution_contract_changed",
                "Post-analysis execution must remain resumable and interruption safe.",
                field=key,
            )
    retries = execution.get("maximum_io_retries")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 1:
        _issue(
            issues,
            "error",
            "io_retries_invalid",
            "At least one transient-I/O retry is required.",
        )
    if execution.get("video_encoder") not in {
        "auto-hardware-if-available",
        "cpu-software",
    }:
        _issue(
            issues,
            "error",
            "video_encoder_invalid",
            "Select automatic hardware encoding or CPU software encoding.",
        )
    output_root = str(candidate.get("output_root") or "").strip()
    if not output_root:
        _issue(
            issues,
            "error",
            "output_root_missing",
            "Choose a post-analysis output directory.",
        )

    mass_result = inspect_mass_analysis_result(
        str(candidate.get("mass_result_manifest_path") or ""), mass_plan
    )
    issues.extend(deepcopy(mass_result.get("issues") or []))
    required_inputs = _required_mass_products(set(products))
    post_inputs = inspect_post_analysis_inputs(
        mass_result,
        selected_runs=set(run_ids),
        required_products=required_inputs,
    )
    issues.extend(deepcopy(post_inputs["issues"]))

    gate_status = str(project.module_states["mass_analysis"].get("status") or "")
    gate_ready = gate_status in SATISFIED_MODULE_STATUSES
    if not gate_ready:
        _issue(
            issues,
            "warning",
            "mass_gate_incomplete",
            "Mass analysis must be approved or complete before rendering.",
            status=gate_status,
        )

    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    valid = errors == 0
    execution_ready = (
        valid
        and mass_plan_ready
        and bool(mass_result.get("valid"))
        and bool(post_inputs.get("valid"))
        and domain_valid
        and gate_ready
    )
    contract = post_analysis_task_contract(
        candidate,
        mass_result_reference=mass_result.get("reference") or {},
        analysis_domain_reference=analysis_domain_reference,
        input_artifacts=post_inputs["artifacts"],
    )
    return {
        "schema": 1,
        "config_sha256": _json_digest(candidate),
        "valid": valid,
        "execution_ready": execution_ready,
        "summary": {
            "error_count": errors,
            "warning_count": warnings,
            "run_count": len(run_ids),
            "product_count": len(products),
            "domain_count": len(domains),
            "required_input_count": len(required_inputs),
            "verified_input_artifact_count": len(post_inputs["artifacts"]),
            "mass_result_ready": bool(mass_result.get("valid")),
            "post_inputs_ready": bool(post_inputs.get("valid")),
            "mass_gate_ready": gate_ready,
        },
        "task_contract": contract,
        "mass_result": mass_result,
        "post_inputs": post_inputs,
        "analysis_domain_reference": analysis_domain_reference,
        "mass_gate_status": gate_status,
        "issues": issues,
    }


class PostAnalysisPlanStore:
    """Append-only post-analysis plans; workers perform the actual rendering."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "post_analysis"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        self.artifact_root = project.root / "results" / "post_analysis" / "plans"

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
        report = validate_post_analysis_plan(self.project, candidate)
        if int((report.get("summary") or {}).get("error_count") or 0) > 0:
            raise ValueError("Post-analysis plan has blocking validation errors")
        plan_id = uuid.uuid4().hex
        mass_plan = MassAnalysisPlanStore(self.project).current()
        revision = {
            "schema": 1,
            "format": POST_ANALYSIS_REVISION_FORMAT,
            "plan_id": plan_id,
            "recorded_utc": utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "configuration": candidate,
            "task_contract": deepcopy(report["task_contract"]),
            "upstream": {
                "mass_analysis_plan_id": mass_plan.get("plan_id"),
                "mass_analysis_plan_sha256": mass_plan.get("content_sha256"),
                "mass_task_contract_sha256": (mass_plan.get("task_contract") or {}).get(
                    "content_sha256"
                ),
                "mass_upstream": deepcopy(mass_plan.get("upstream") or {}),
                "mass_result": deepcopy(report["mass_result"].get("reference") or {}),
                "analysis_domains": deepcopy(report["analysis_domain_reference"]),
            },
            "validation": report,
            "execution": {
                "ready": bool(report.get("execution_ready")),
                "requested": False,
                "started": False,
                "task_id": None,
            },
            "scientific_scope": {
                "contains_rendered_products": False,
                "recomputes_mass": False,
                "changes_phase_interpretation": False,
                "raw_detected_mass_is_authoritative": True,
                "injected_mass_is_reference_only": True,
                "trains_facies_model": False,
                "trains_phase_model": False,
            },
        }
        revision["content_sha256"] = _json_digest(revision)
        destination = self.artifact_root / plan_id
        destination.mkdir(parents=True, exist_ok=False)
        _atomic_json(destination / "plan.json", revision)
        _append_jsonl(self.revisions_path, revision)
        _atomic_json(self.current_path, revision)
        self.project._audit(
            "post_analysis_plan_saved",
            actor,
            {
                "plan_id": plan_id,
                "task_contract_sha256": report["task_contract"]["content_sha256"],
                "execution_ready": bool(report.get("execution_ready")),
                "execution_started": False,
            },
        )
        return deepcopy(revision)


if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject
