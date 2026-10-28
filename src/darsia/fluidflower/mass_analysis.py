"""Versioned mass-analysis task plans for FluidFlower projects."""

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

from darsia.fluidflower.calibration import (
    CALIBRATION_REVISION_FORMAT,
    NUMERICAL_BACKENDS,
    CalibrationPlanStore,
    validate_calibration_plan,
)
from darsia.fluidflower.frame_quality import FrameQualityStore
from darsia.fluidflower.geometry_facies import GeometryRevisionStore
from darsia.fluidflower.image_correction import CorrectionRevisionStore
from darsia.fluidflower.mass_inputs import (
    inspect_mass_input_manifest,
    validate_mass_parameter_artifact,
)
from darsia.fluidflower.modules import SATISFIED_MODULE_STATUSES
from darsia.fluidflower.phase_interpretation import PhaseInterpretationRevisionStore
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.quality_budget import (
    default_quality_profile,
    default_time_selection_budget,
    estimate_mass_publication_bytes,
    numerical_dtype_for_backend,
    resolve_time_selection,
    summarize_quality_contract,
    validate_quality_profile,
)

MASS_ANALYSIS_PLAN_FORMAT = "darsia-fluidflower-mass-analysis-plan-v1"
MASS_ANALYSIS_REVISION_FORMAT = "darsia-fluidflower-mass-analysis-revision-v1"
MASS_ANALYSIS_TASK_CONTRACT_FORMAT = "darsia-fluidflower-mass-task-v1"
CALIBRATION_RESULT_FORMAT = "darsia-fluidflower-calibration-result-v1"

MASS_PRODUCTS = (
    "mass_total",
    "mass_aqueous",
    "mass_free_gas",
    "transport_weight",
    "concentration_aqueous",
    "saturation_free_gas",
    "integrated_mass_csv",
    "preview_images",
    "diagnostic_injected_mass_projection",
)


def synchronized_image_selection_configuration(target_hours, *, manifest_path,
        manifest_sha256, early_end_h=3.0, early_tolerance_seconds=15.0,
        late_tolerance_seconds=150.0):
    """Non-executing mass-plan selection bound to an actual camera manifest.

    Missing/replaced acquisitions are recorded in that manifest; the target
    times stay identical across runs. This helper does not approve a phase or
    concentration calibration and does not launch production.
    """
    times = [float(value) for value in target_hours]
    if not times or any(not math.isfinite(t) for t in times) or any(a>=b for a,b in zip(times,times[1:])):
        raise ValueError("Representative times must increase strictly")
    if len(str(manifest_sha256)) != 64:
        raise ValueError("Selection manifest needs an exact SHA-256")
    return dict(policy="synchronized_representative_times", target_hours=times,
        maximum_images_per_run=len(times), chronological=True, require_same_target_times=True,
        drift_tolerance_seconds=float(late_tolerance_seconds),
        time_tolerances=[dict(end_hour=float(early_end_h),seconds=float(early_tolerance_seconds)),
                         dict(end_hour=max(times),seconds=float(late_tolerance_seconds))],
        manifest=dict(path=str(manifest_path),sha256=str(manifest_sha256)),
        budget=default_time_selection_budget(maximum_time_points=len(times)),
        execution_authorized=False)


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
                raise TypeError(f"Invalid mass-analysis revision at line {line_number}")
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


def _resolved_path(raw: str, *, relative_to: Path | None = None) -> Path:
    from darsia.fluidflower.mass_inputs import _resolved_path as _canonical_resolved_path
    return _canonical_resolved_path(raw, relative_to=relative_to)


def _expected_run_bindings(project: FluidFlowerProject) -> list[dict[str, str]]:
    phase = PhaseInterpretationRevisionStore(project).current()
    selected = dict((phase.get("configuration") or {}).get("selected_candidates") or {})
    return [
        {"run_id": str(run_id), "phase_candidate_id": str(candidate_id)}
        for run_id, candidate_id in sorted(selected.items())
    ]


def current_optical_branch(project: FluidFlowerProject) -> dict[str, Any]:
    """Describe the active, checksum-linked correction and color-path pair."""

    correction = CorrectionRevisionStore(project).current()
    frame_quality = FrameQualityStore(project).dependency_reference()
    phase = PhaseInterpretationRevisionStore(project).current()
    phase_upstream = dict(phase.get("upstream") or {})
    correction_id = str(correction.get("revision_id") or "")
    correction_sha256 = str(correction.get("content_sha256") or "")
    phase_id = str(phase.get("revision_id") or "")
    phase_sha256 = str(phase.get("content_sha256") or "")
    phase_frame_quality = dict(phase_upstream.get("frame_quality") or {})
    frame_quality_compatible = str(
        phase_frame_quality.get("blacklist_sha256") or ""
    ) == str(frame_quality.get("blacklist_sha256") or "")
    compatible = bool(
        correction_id
        and correction_sha256
        and phase_id
        and phase_sha256
        and str(phase_upstream.get("image_correction_revision_id") or "")
        == correction_id
        and str(phase_upstream.get("image_correction_revision_sha256") or "")
        == correction_sha256
        and frame_quality_compatible
    )
    return {
        "schema": 1,
        "correction_revision_id": correction_id,
        "correction_revision_sha256": correction_sha256,
        "correction_dependency_sha256": (
            CorrectionRevisionStore.dependency_sha256(correction)
        ),
        "correction_algorithm": str(correction.get("algorithm") or ""),
        "frame_quality": frame_quality,
        "frame_quality_compatible": frame_quality_compatible,
        "phase_revision_id": phase_id,
        "phase_revision_sha256": phase_sha256,
        "compatible": compatible,
        "selection_confirmed": False,
    }


def default_mass_analysis_plan(project: FluidFlowerProject) -> dict[str, Any]:
    """Return an editable, non-executing task plan."""

    calibration = CalibrationPlanStore(project).current()
    from darsia.fluidflower.calibration_review import CalibrationResultReviewStore

    calibration_review = CalibrationResultReviewStore(project).current()
    execution = dict(
        (calibration.get("configuration") or {}).get("execution_profile") or {}
    )
    primary_backend = str(execution.get("primary_backend") or "cpu-float64")
    numerical_dtype = numerical_dtype_for_backend(primary_backend)
    quality_profile = default_quality_profile("production")
    quality_profile["precision"]["compute_dtype"] = numerical_dtype
    return {
        "schema": 1,
        "format": MASS_ANALYSIS_PLAN_FORMAT,
        "calibration_plan_id": str(calibration.get("plan_id") or ""),
        "calibration_plan_sha256": str(calibration.get("content_sha256") or ""),
        "calibration_review_id": str(calibration_review.get("review_id") or ""),
        "calibration_review_sha256": str(
            calibration_review.get("content_sha256") or ""
        ),
        "calibration_result_manifest_path": "",
        "mass_input_manifest_path": "",
        "optical_branch": current_optical_branch(project),
        "run_bindings": _expected_run_bindings(project),
        "image_selection": {
            "policy": "synchronized_representative_times",
            "maximum_images_per_run": 100,
            "drift_tolerance_seconds": 300.0,
            "require_same_target_times": True,
            "chronological": True,
            "budget": default_time_selection_budget(maximum_time_points=100),
        },
        "quality_profile": quality_profile,
        "physical_conversion": {
            "phase_support_source": "selected_phase_candidate_probability_maps",
            "continuous_signal_source": "approved_corrected_optical_signal",
            "chemistry_engine": "darsia_co2_mass_analysis",
            "phase_partition": "exclusive_aqueous_and_free_gas",
            "depth_source": "approved_geometry_revision",
            "porosity_source": "approved_geometry_revision",
            "two_d_to_three_d": "pixel_area_times_local_depth_times_porosity",
            "mass_reference": "injection_protocol",
            "force_mass_conservation": False,
            "injected_mass_projection": "diagnostic_only",
        },
        "exports": {
            "products": list(MASS_PRODUCTS),
            "numerical_format": "npz",
            "numerical_dtype": numerical_dtype,
            "wasserstein_cache": {
                "enabled": False,
                "resolution_policy": "defer_to_wasserstein_plan",
                "target_cell_size_m": None,
                "mass_preserving": True,
            },
        },
        "execution_profile": {
            "primary_backend": primary_backend,
            "reference_backend": "cpu-float64",
            "workers": max(1, int(execution.get("workers") or 1)),
            "prefetch_images": 1,
            "csv_checkpoint_images": 10,
            "local_read_through_cache": True,
            "atomic_publish": True,
        },
        "execution_requested": False,
    }


def inspect_calibration_result(
    raw_path: str,
    calibration_plan: Mapping[str, Any],
    *,
    run_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Inspect a calibration-result manifest without modifying it."""

    issues: list[dict[str, Any]] = []
    text = str(raw_path).strip()
    if not text:
        _issue(
            issues,
            "warning",
            "calibration_result_missing",
            "No calibration-result manifest is attached; execution remains blocked.",
        )
        return {
            "available": False,
            "valid": False,
            "reference": {},
            "manifest": {},
            "parameter_validation": {},
            "issues": issues,
        }

    path = _resolved_path(text)
    if not path.is_file():
        _issue(
            issues,
            "warning",
            "calibration_result_unavailable",
            "The calibration-result manifest is currently unavailable.",
            path=str(path),
        )
        return {
            "available": False,
            "valid": False,
            "reference": {"path": str(path)},
            "manifest": {},
            "parameter_validation": {},
            "issues": issues,
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError) as exc:
        _issue(
            issues,
            "error",
            "calibration_result_unreadable",
            f"Could not read calibration result: {type(exc).__name__}: {exc}",
        )
        payload = {}
    if not isinstance(payload, dict):
        _issue(
            issues,
            "error",
            "calibration_result_invalid",
            "Calibration-result manifest must contain a JSON object.",
        )
        payload = {}

    if payload.get("format") != CALIBRATION_RESULT_FORMAT:
        _issue(
            issues,
            "error",
            "calibration_result_format",
            "Unsupported calibration-result format.",
        )
    if str(payload.get("calibration_plan_id") or "") != str(
        calibration_plan.get("plan_id") or ""
    ):
        _issue(
            issues,
            "error",
            "calibration_result_plan_mismatch",
            "Calibration result belongs to a different calibration plan.",
        )
    expected_objective = str(calibration_plan.get("selection_contract_sha256") or "")
    if str(payload.get("objective_contract_sha256") or "") != expected_objective:
        _issue(
            issues,
            "error",
            "calibration_result_objective_mismatch",
            "Calibration result used a different optimization/selection objective.",
        )
    if str(payload.get("selected_model") or "") not in {"simple_flash", "titration"}:
        _issue(
            issues,
            "error",
            "calibration_result_model",
            "Calibration result must select Simple Flash or Titration.",
        )

    parameter = dict(payload.get("parameter_artifact") or {})
    parameter_path_text = str(parameter.get("path") or "").strip()
    parameter_reference: dict[str, Any] = {}
    parameter_validation: dict[str, Any] = {}
    if not parameter_path_text:
        _issue(
            issues,
            "error",
            "parameter_artifact_missing",
            "Calibration result does not reference calibrated parameters.",
        )
    else:
        parameter_path = _resolved_path(parameter_path_text, relative_to=path.parent)
        if not parameter_path.is_file():
            _issue(
                issues,
                "warning",
                "parameter_artifact_unavailable",
                "The calibrated-parameter artifact is currently unavailable.",
                path=str(parameter_path),
            )
        else:
            actual_sha256 = sha256_file(parameter_path)
            parameter_reference = {
                "path": str(parameter_path),
                "size_bytes": parameter_path.stat().st_size,
                "sha256": actual_sha256,
            }
            expected_sha256 = str(parameter.get("sha256") or "")
            if not expected_sha256 or expected_sha256 != actual_sha256:
                _issue(
                    issues,
                    "error",
                    "parameter_artifact_hash_mismatch",
                    "Calibrated-parameter artifact does not match its recorded hash.",
                )
            try:
                parameter_payload = json.loads(
                    parameter_path.read_text(encoding="utf-8-sig")
                )
                if not isinstance(parameter_payload, Mapping):
                    raise TypeError("parameter artifact must contain one JSON object")
                parameter_validation = validate_mass_parameter_artifact(
                    parameter_payload,
                    run_ids=list(run_ids or []),
                    selected_model=str(payload.get("selected_model") or ""),
                )
                issues.extend(deepcopy(parameter_validation.get("issues") or []))
                parameter_reference["format"] = str(
                    parameter_payload.get("format") or ""
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                _issue(
                    issues,
                    "error",
                    "parameter_artifact_invalid",
                    f"Could not validate calibrated parameters: "
                    f"{type(exc).__name__}: {exc}",
                )

    parity = dict(payload.get("parity") or {})
    if str(parity.get("reference_backend") or "") != "cpu-float64":
        _issue(
            issues,
            "error",
            "calibration_parity_reference",
            "Calibration result must use CPU float64 as parity reference.",
        )
    if parity.get("passed") is not True:
        _issue(
            issues,
            "error",
            "calibration_parity_failed",
            "Calibration backend has not passed tolerance-based parity control.",
        )

    errors = sum(issue["severity"] == "error" for issue in issues)
    unavailable = any(
        issue["code"] in {"parameter_artifact_unavailable"} for issue in issues
    )
    return {
        "available": True,
        "valid": errors == 0 and not unavailable,
        "reference": {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "parameter_artifact": parameter_reference,
        },
        "manifest": payload,
        "parameter_validation": parameter_validation,
        "issues": issues,
    }


def mass_task_contract(
    config: Mapping[str, Any],
    calibration_result_reference: Mapping[str, Any] | None = None,
    calibration_review_reference: Mapping[str, Any] | None = None,
    mass_input_reference: Mapping[str, Any] | None = None,
    resolved_time_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the immutable scientific and numerical task contract."""

    contract = {
        "schema": 1,
        "format": MASS_ANALYSIS_TASK_CONTRACT_FORMAT,
        "calibration_plan_id": str(config.get("calibration_plan_id") or ""),
        "calibration_plan_sha256": str(config.get("calibration_plan_sha256") or ""),
        "calibration_result": deepcopy(dict(calibration_result_reference or {})),
        "calibration_review": deepcopy(dict(calibration_review_reference or {})),
        "mass_inputs": deepcopy(dict(mass_input_reference or {})),
        "optical_branch": deepcopy(dict(config.get("optical_branch") or {})),
        "run_bindings": deepcopy(list(config.get("run_bindings") or [])),
        "image_selection": deepcopy(dict(config.get("image_selection") or {})),
        "resolved_time_selection": deepcopy(dict(resolved_time_selection or {})),
        "quality_profile": deepcopy(dict(config.get("quality_profile") or {})),
        "physical_conversion": deepcopy(dict(config.get("physical_conversion") or {})),
        "exports": deepcopy(dict(config.get("exports") or {})),
        "execution_profile": deepcopy(dict(config.get("execution_profile") or {})),
    }
    contract["content_sha256"] = _json_digest(contract)
    return contract


def validate_mass_analysis_plan(
    project: FluidFlowerProject,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a mass-analysis task without starting computation."""

    candidate = deepcopy(dict(config))
    issues: list[dict[str, Any]] = []
    if candidate.get("format") != MASS_ANALYSIS_PLAN_FORMAT:
        _issue(issues, "error", "unsupported_format", "Unsupported mass plan")
    if bool(candidate.get("execution_requested")):
        _issue(
            issues,
            "error",
            "execution_not_supported",
            "Saving a mass plan cannot start mass analysis.",
        )

    selected_branch = dict(candidate.get("optical_branch") or {})
    expected_branch = current_optical_branch(project)
    if not bool(selected_branch.get("selection_confirmed")):
        _issue(
            issues,
            "error",
            "optical_branch_not_selected",
            "Select the correction and color-path branch explicitly before mass analysis.",
        )
    branch_keys = (
        "correction_revision_id",
        "correction_revision_sha256",
        "correction_dependency_sha256",
        "correction_algorithm",
        "phase_revision_id",
        "phase_revision_sha256",
    )
    if not bool(expected_branch.get("compatible")):
        _issue(
            issues,
            "error",
            "optical_branch_incompatible",
            "The active color paths were not built from the active image correction.",
        )
    elif any(
        str(selected_branch.get(key) or "") != str(expected_branch.get(key) or "")
        for key in branch_keys
    ):
        _issue(
            issues,
            "error",
            "optical_branch_changed",
            "The selected correction and color-path branch is no longer active.",
            expected={key: expected_branch.get(key) for key in branch_keys},
        )

    calibration = CalibrationPlanStore(project).current()
    if calibration.get("format") != CALIBRATION_REVISION_FORMAT:
        _issue(
            issues,
            "error",
            "calibration_plan_missing",
            "Save a calibration plan before mass-analysis planning.",
        )
    else:
        if str(candidate.get("calibration_plan_id") or "") != str(
            calibration.get("plan_id") or ""
        ) or str(candidate.get("calibration_plan_sha256") or "") != str(
            calibration.get("content_sha256") or ""
        ):
            _issue(
                issues,
                "error",
                "calibration_plan_changed",
                "Mass plan does not reference the current calibration plan.",
            )
        calibration_validation = validate_calibration_plan(
            project, dict(calibration.get("configuration") or {})
        )
        if not calibration_validation.get("valid"):
            _issue(
                issues,
                "error",
                "calibration_plan_invalid",
                "Current calibration plan no longer matches its upstream phase state.",
            )

    expected_bindings = _expected_run_bindings(project)
    run_bindings = [dict(row) for row in candidate.get("run_bindings") or []]
    if run_bindings != expected_bindings or not run_bindings:
        _issue(
            issues,
            "error",
            "run_bindings_changed",
            "Run-to-phase-candidate bindings must match the current phase revision.",
        )

    image_selection = dict(candidate.get("image_selection") or {})
    if image_selection.get("policy") != "synchronized_representative_times":
        _issue(
            issues,
            "error",
            "image_selection_policy",
            "Mass analysis requires synchronized representative times.",
        )
    maximum_images = image_selection.get("maximum_images_per_run")
    if (
        not isinstance(maximum_images, int)
        or isinstance(maximum_images, bool)
        or maximum_images < 1
    ):
        _issue(
            issues,
            "error",
            "maximum_images_invalid",
            "Maximum images per run must be a positive integer.",
        )
    tolerance = _number(image_selection.get("drift_tolerance_seconds"))
    if tolerance is None or tolerance < 0.0:
        _issue(
            issues,
            "error",
            "drift_tolerance_invalid",
            "Time drift tolerance must be finite and non-negative.",
        )
    if image_selection.get("require_same_target_times") is not True:
        _issue(
            issues,
            "error",
            "target_times_not_shared",
            "Every run must be sampled against the same target times.",
        )
    if image_selection.get("chronological") is not True:
        _issue(
            issues,
            "error",
            "non_chronological_analysis",
            "Mass analysis must publish images chronologically.",
        )
    selection_budget = dict(image_selection.get("budget") or {})
    if selection_budget.get("maximum_time_points") != maximum_images:
        _issue(
            issues,
            "error",
            "image_budget_mismatch",
            "Maximum images and the deterministic time-selection budget must agree.",
        )

    physical = dict(candidate.get("physical_conversion") or {})
    expected_physical = {
        "phase_support_source": "selected_phase_candidate_probability_maps",
        "continuous_signal_source": "approved_corrected_optical_signal",
        "chemistry_engine": "darsia_co2_mass_analysis",
        "phase_partition": "exclusive_aqueous_and_free_gas",
        "depth_source": "approved_geometry_revision",
        "porosity_source": "approved_geometry_revision",
        "two_d_to_three_d": "pixel_area_times_local_depth_times_porosity",
        "mass_reference": "injection_protocol",
        "force_mass_conservation": False,
        "injected_mass_projection": "diagnostic_only",
    }
    if physical != expected_physical:
        _issue(
            issues,
            "error",
            "physical_conversion_changed",
            "Mass conversion must preserve exclusive phases, approved geometry, "
            "and raw detected mass as the scientific result.",
        )

    exports = dict(candidate.get("exports") or {})
    products = [str(value) for value in exports.get("products") or []]
    unknown_products = sorted(set(products) - set(MASS_PRODUCTS))
    if unknown_products:
        _issue(
            issues,
            "error",
            "unknown_mass_products",
            "Mass plan contains unsupported output products.",
            products=unknown_products,
        )
    required_products = {
        "mass_total",
        "mass_aqueous",
        "mass_free_gas",
        "transport_weight",
        "integrated_mass_csv",
    }
    if not required_products.issubset(products):
        _issue(
            issues,
            "error",
            "required_mass_products_missing",
            "Raw total, aqueous, free-gas, transport-weight, and integrated mass "
            "products are required.",
        )
    if exports.get("numerical_format") != "npz":
        _issue(
            issues,
            "error",
            "numerical_format_invalid",
            "The supported immutable numerical format is NPZ.",
        )

    execution = dict(candidate.get("execution_profile") or {})
    backend = str(execution.get("primary_backend") or "")
    if backend not in NUMERICAL_BACKENDS:
        _issue(issues, "error", "invalid_backend", "Unknown numerical backend")
    if str(execution.get("reference_backend") or "") != "cpu-float64":
        _issue(
            issues,
            "error",
            "invalid_reference_backend",
            "CPU float64 is required as the authoritative parity reference.",
        )
    try:
        expected_dtype = numerical_dtype_for_backend(backend)
    except ValueError:
        expected_dtype = ""
    if str(exports.get("numerical_dtype") or "") != expected_dtype:
        _issue(
            issues,
            "error",
            "backend_precision_mismatch",
            f"{backend} requires {expected_dtype} output in this contract.",
        )
    workers = execution.get("workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        _issue(issues, "error", "invalid_worker_count", "Worker count must be positive")
    if execution.get("prefetch_images") not in {0, 1}:
        _issue(
            issues,
            "error",
            "invalid_prefetch_count",
            "The bounded image prefetch count must be zero or one.",
        )
    checkpoint = execution.get("csv_checkpoint_images")
    if (
        not isinstance(checkpoint, int)
        or isinstance(checkpoint, bool)
        or checkpoint < 1
    ):
        _issue(
            issues,
            "error",
            "invalid_checkpoint_interval",
            "CSV checkpoint interval must be a positive integer.",
        )
    if execution.get("atomic_publish") is not True:
        _issue(
            issues,
            "error",
            "atomic_publish_required",
            "Verified results must be published atomically.",
        )

    quality_report = validate_quality_profile(
        dict(candidate.get("quality_profile") or {}), backend=backend
    )
    issues.extend(deepcopy(quality_report["issues"]))
    quality_contract = (
        summarize_quality_contract(
            quality_report["configuration"],
            backend=backend,
        )
        if backend in NUMERICAL_BACKENDS
        else {}
    )

    cache = dict(exports.get("wasserstein_cache") or {})
    cache_enabled = cache.get("enabled") is True
    resolution_policy = str(cache.get("resolution_policy") or "")
    if cache.get("mass_preserving") is not True:
        _issue(
            issues,
            "error",
            "wasserstein_cache_not_mass_preserving",
            "Any Wasserstein cache must preserve integrated mass.",
        )
    if cache_enabled:
        if resolution_policy != "explicit_physical_cell_size":
            _issue(
                issues,
                "error",
                "wasserstein_resolution_not_physical",
                "Cache generation requires an explicit physical cell size.",
            )
        cell_size = _number(cache.get("target_cell_size_m"))
        if cell_size is None or cell_size <= 0.0:
            _issue(
                issues,
                "error",
                "wasserstein_cell_size_invalid",
                "Physical Wasserstein cell size must be positive.",
            )
    elif resolution_policy != "defer_to_wasserstein_plan":
        _issue(
            issues,
            "error",
            "wasserstein_cache_policy_invalid",
            "Disabled cache generation must defer resolution to Wasserstein planning.",
        )

    calibration_result = inspect_calibration_result(
        str(candidate.get("calibration_result_manifest_path") or ""),
        calibration,
        run_ids=[str(row.get("run_id") or "") for row in run_bindings],
    )
    issues.extend(deepcopy(calibration_result["issues"]))
    if (calibration_result.get("parameter_validation") or {}).get("requires_native_fixed_executor"):
        _issue(issues, "error", "native_fixed_executor_required",
               "This frozen native calibration is reviewed through the calibration pages. "
               "Its saved locked-label fields use the native fixed-phase executor; "
               "the probability-transfer mass executor cannot apply them.")

    from darsia.fluidflower.calibration_review import (
        CalibrationResultReviewStore,
        validate_calibration_review_config,
    )

    calibration_review_revision = CalibrationResultReviewStore(project).current()
    calibration_review_report: dict[str, Any] = {}
    calibration_review_reference: dict[str, Any] = {}
    if not calibration_review_revision:
        _issue(
            issues,
            "warning",
            "calibration_review_missing",
            "Review concentration maps and mass plots before mass analysis.",
        )
    else:
        expected_review_id = str(candidate.get("calibration_review_id") or "")
        expected_review_sha256 = str(candidate.get("calibration_review_sha256") or "")
        current_review_id = str(calibration_review_revision.get("review_id") or "")
        current_review_sha256 = str(
            calibration_review_revision.get("content_sha256") or ""
        )
        if (
            expected_review_id != current_review_id
            or expected_review_sha256 != current_review_sha256
        ):
            _issue(
                issues,
                "error",
                "calibration_review_changed",
                "Mass plan does not reference the current calibration-result review.",
            )
        calibration_review_report = validate_calibration_review_config(
            project,
            dict(calibration_review_revision.get("configuration") or {}),
        )
        if not calibration_review_report.get("ready"):
            _issue(
                issues,
                "warning",
                "calibration_review_incomplete",
                "Every experiment requires approved concentration and mass "
                "calibration before execution.",
            )
        from .series_paths import calibration_artifact_path
        review_path = calibration_artifact_path(
            project.root, "calibration_reviews", current_review_id, "review.json")
        calibration_review_reference = {
            "review_id": current_review_id,
            "content_sha256": current_review_sha256,
            "path": str(review_path.resolve()),
            "sha256": sha256_file(review_path) if review_path.is_file() else "",
        }

    phase_revision = PhaseInterpretationRevisionStore(project).current()
    mass_inputs = inspect_mass_input_manifest(
        str(candidate.get("mass_input_manifest_path") or ""),
        run_bindings=run_bindings,
        maximum_images_per_run=(
            int(maximum_images) if isinstance(maximum_images, int) else 0
        ),
        phase_revision=phase_revision,
        correction_revision=CorrectionRevisionStore(project).current(),
        geometry_revision=GeometryRevisionStore(project).current(),
        require_preview_images="preview_images" in products,
        allow_time_subset=True,
    )
    issues.extend(deepcopy(mass_inputs["issues"]))

    resolved_time_selection: dict[str, Any] = {
        "valid": False,
        "source_time_count": 0,
        "selected_time_count": 0,
        "selected_time_indices": [],
        "selected_times_hours": [],
        "estimated_publication_bytes": 0,
        "issues": [],
    }
    shared_times = list(mass_inputs.get("shared_target_times_hours") or [])
    if shared_times:
        preview = dict((candidate.get("quality_profile") or {}).get("preview") or {})
        estimate = estimate_mass_publication_bytes(
            mass_inputs.get("manifest") or {},
            products=products,
            numerical_dtype=str(exports.get("numerical_dtype") or "float64"),
            preview_maximum_dimension_px=int(
                preview.get("maximum_dimension_px") or 1920
            ),
            run_shapes=mass_inputs.get("run_shapes") or {},
            analysis_profile=dict(candidate.get("quality_profile") or {}),
        )
        resolved_time_selection = resolve_time_selection(
            shared_times,
            selection_budget,
            fixed_publication_bytes=estimate["fixed_bytes"],
            publication_bytes_per_time=estimate["bytes_per_time"],
        )
        issues.extend(deepcopy(resolved_time_selection["issues"]))

    required_gates = (
        "geometry_facies",
        "image_correction",
        "temporal_sync",
        "phase_interpretation",
        "calibration",
    )
    gate_status = {
        key: str(project.module_states[key].get("status") or "")
        for key in required_gates
    }
    gates_ready = all(
        status in SATISFIED_MODULE_STATUSES for status in gate_status.values()
    )
    if not gates_ready:
        _issue(
            issues,
            "warning",
            "upstream_gates_incomplete",
            "All upstream quality gates must be approved before execution.",
            statuses=gate_status,
        )

    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    valid = errors == 0
    execution_ready = (
        valid
        and calibration_result["valid"]
        and bool(calibration_review_report.get("ready"))
        and mass_inputs["valid"]
        and resolved_time_selection["valid"]
        and gates_ready
    )
    return {
        "schema": 1,
        "config_sha256": _json_digest(candidate),
        "valid": valid,
        "execution_ready": execution_ready,
        "summary": {
            "error_count": errors,
            "warning_count": warnings,
            "run_count": len(run_bindings),
            "maximum_images_per_run": maximum_images,
            "primary_backend": backend,
            "numerical_dtype": str(quality_contract.get("compute_dtype") or ""),
            "scientific_grid": str(quality_contract.get("scientific_grid") or ""),
            "quality_label": str(quality_contract.get("label") or ""),
            "calibration_result_ready": calibration_result["valid"],
            "calibration_review_ready": bool(calibration_review_report.get("ready")),
            "mass_inputs_ready": mass_inputs["valid"],
            "target_time_count": len(
                mass_inputs.get("shared_target_times_hours") or []
            ),
            "selected_time_count": int(
                resolved_time_selection.get("selected_time_count") or 0
            ),
            "estimated_publication_bytes": int(
                resolved_time_selection.get("estimated_publication_bytes") or 0
            ),
            "upstream_gates_ready": gates_ready,
        },
        "task_contract": mass_task_contract(
            candidate,
            calibration_result["reference"],
            calibration_review_reference,
            mass_inputs["reference"],
            resolved_time_selection,
        ),
        "calibration_result": calibration_result,
        "calibration_review": {
            "revision": calibration_review_revision,
            "validation": calibration_review_report,
            "reference": calibration_review_reference,
        },
        "mass_inputs": mass_inputs,
        "resolved_time_selection": resolved_time_selection,
        "quality_profile": quality_report,
        "quality_contract": quality_contract,
        "upstream_gate_status": gate_status,
        "issues": issues,
    }


class MassAnalysisPlanStore:
    """Append-only mass plans; compute is handled by a later task runner."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "mass_analysis"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        self.artifact_root = project.root / "results" / "mass_analysis" / "plans"

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
        # Revalidate current pointers and network-backed artifacts at commit time.
        report = validate_mass_analysis_plan(self.project, candidate)
        if int((report.get("summary") or {}).get("error_count") or 0) > 0:
            raise ValueError("Mass-analysis plan has blocking validation errors")
        plan_id = uuid.uuid4().hex
        calibration = CalibrationPlanStore(self.project).current()
        phase = PhaseInterpretationRevisionStore(self.project).current()
        revision = {
            "schema": 1,
            "format": MASS_ANALYSIS_REVISION_FORMAT,
            "plan_id": plan_id,
            "recorded_utc": utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "configuration": candidate,
            "task_contract": deepcopy(report["task_contract"]),
            "upstream": {
                "calibration_plan_id": calibration.get("plan_id"),
                "calibration_plan_sha256": calibration.get("content_sha256"),
                "calibration_objective_sha256": calibration.get(
                    "selection_contract_sha256"
                ),
                "phase_revision_id": phase.get("revision_id"),
                "phase_revision_sha256": phase.get("content_sha256"),
                "phase_upstream": deepcopy(phase.get("upstream") or {}),
                "optical_branch": deepcopy(candidate.get("optical_branch") or {}),
                "calibration_result": deepcopy(
                    report["calibration_result"]["reference"]
                ),
                "calibration_review": deepcopy(
                    report["calibration_review"]["reference"]
                ),
                "mass_inputs": deepcopy(report["mass_inputs"]["reference"]),
            },
            "validation": report,
            "execution": {
                "ready": bool(report.get("execution_ready")),
                "requested": False,
                "started": False,
                "task_id": None,
            },
            "scientific_scope": {
                "contains_mass_results": False,
                "performs_2d_to_3d_conversion": False,
                "raw_mass_is_authoritative": True,
                "injected_mass_projection_is_diagnostic": True,
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
            "mass_analysis_plan_saved",
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
