"""Versioned Wasserstein-analysis plans for FluidFlower projects."""

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
from darsia.fluidflower.quality_budget import (
    default_time_selection_budget,
    estimate_wasserstein_publication_bytes,
    resolve_time_selection,
)

WASSERSTEIN_PLAN_FORMAT = "darsia-fluidflower-wasserstein-plan-v1"
WASSERSTEIN_REVISION_FORMAT = "darsia-fluidflower-wasserstein-revision-v1"
WASSERSTEIN_TASK_CONTRACT_FORMAT = "darsia-fluidflower-wasserstein-task-v1"
MASS_ANALYSIS_RESULT_FORMAT = "darsia-fluidflower-mass-analysis-result-v1"

WASSERSTEIN_MASS_COMPONENTS = (
    "mass_total",
    "mass_aqueous",
    "mass_free_gas",
)
WASSERSTEIN_REQUIRED_INPUT_PRODUCTS = (
    *WASSERSTEIN_MASS_COMPONENTS,
    "transport_weight",
)
WASSERSTEIN_DOMAINS = ("whole_active_domain",)
WASSERSTEIN_METRICS = (
    "transport_cost_mass_m",
    "mass_based_distance_m",
    "rig_based_normalized_distance",
    "rig_diagonal_normalized_distance",
    "plume_based_normalized_distance",
    "absolute_mass_difference_g",
    "relative_mass_difference",
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
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Invalid Wasserstein revision at line {line_number}")
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


def _mass_run_ids(mass_plan: Mapping[str, Any]) -> list[str]:
    from .post_analysis import _mass_run_ids as shared_mass_run_ids
    return shared_mass_run_ids(mass_plan)


def _all_run_pairs(run_ids: list[str]) -> list[dict[str, str]]:
    ordered = sorted({str(value) for value in run_ids if str(value)})
    return [
        {"run_1": run_1, "run_2": run_2}
        for index, run_1 in enumerate(ordered)
        for run_2 in ordered[index + 1 :]
    ]


def _current_domain_state(project: FluidFlowerProject) -> tuple[dict, dict]:
    from .post_analysis import _current_domain_state as shared_domain_state
    return shared_domain_state(project)


def _mass_result_reference(
    path: Path,
    product_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "product_artifacts": deepcopy(product_artifacts),
    }


def default_wasserstein_plan(project: FluidFlowerProject) -> dict[str, Any]:
    """Return an editable plan that cannot start Wasserstein computation."""

    mass_plan = MassAnalysisPlanStore(project).current()
    domain_revision, domain_config = _current_domain_state(project)
    mass_config = dict(mass_plan.get("configuration") or {})
    image_selection = dict(mass_config.get("image_selection") or {})
    run_ids = _mass_run_ids(mass_plan)
    dynamic_domains = [
        str(row.get("domain_id"))
        for row in domain_config.get("domains") or []
        if row.get("domain_id") and bool(row.get("enabled", True))
    ]
    return {
        "schema": 1,
        "format": WASSERSTEIN_PLAN_FORMAT,
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
        "run_ids": run_ids,
        "comparison": {
            "pairing_policy": "explicit_unordered_run_pairs",
            "pairs": _all_run_pairs(run_ids),
            "time_policy": "shared_synchronized_target_times",
            "require_same_target_times": True,
            "recorded_mass_time_tolerance_seconds": float(
                image_selection.get("drift_tolerance_seconds") or 0.0
            ),
            "mass_components": list(WASSERSTEIN_MASS_COMPONENTS),
            "domains": ["whole_active_domain", *dynamic_domains],
            "time_selection_budget": default_time_selection_budget(
                maximum_time_points=None
            ),
        },
        "spatial_discretization": {
            "input_preference": (
                "matching_mass_analysis_cache_else_native_mass_fields"
            ),
            "resolution_policy": "explicit_common_physical_cell_size",
            "target_cell_size_m": None,
            "interpolation": "conservative_area",
            "preserve_integrated_mass": True,
            "forbid_upsampling": True,
            "dimensionless_resize_factor": None,
        },
        "transport": {
            "method": "variational_wasserstein_l1",
            "mobility": "cell_based",
            "flux_discretization": "raviart_thomas",
            "spatial_cost": "inverse_relative_permeability",
            "permeability_floor_relative": 1.0e-6,
            "equal_mass_policy": (
                "rescale_each_pair_to_mean_detected_mass_for_transport_only"
            ),
            "preserve_raw_masses_in_output": True,
            "mass_difference_policy": "always_report_never_hide",
            "normalizations": {
                "mass_based": "cost_divided_by_pair_mean_detected_mass",
                "rig_based": ("cost_divided_by_injected_mass_times_mean_rig_dimension"),
                "rig_diagonal": ("mass_based_distance_divided_by_rig_diagonal"),
                "plume_based": (
                    "cost_divided_by_injected_mass_times_reference_plume_radius"
                ),
            },
        },
        "outputs": {
            "metrics": list(WASSERSTEIN_METRICS),
            "intermediate_format": "json",
            "table_formats": ["csv", "xlsx"],
            "store_transport_diagnostics": False,
        },
        "execution_profile": {
            "backend": "cpu-float64",
            "workers": 12,
            "cache_timepoints_per_worker": 2,
            "task_order": "time_pair_domain_component",
            "reuse_loaded_mass_fields": True,
            "local_read_through_cache": True,
            "checkpoint_each_task": True,
            "resume_completed_tasks": True,
            "retry_transient_io": True,
            "maximum_io_retries": 8,
            "atomic_publish": True,
        },
        "execution_requested": False,
    }


def inspect_mass_analysis_result(
    raw_path: str,
    mass_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect and fingerprint a completed mass-analysis result manifest."""

    issues: list[dict[str, Any]] = []
    text = str(raw_path).strip()
    if not text:
        _issue(
            issues,
            "warning",
            "mass_result_missing",
            "No mass-result manifest is attached; execution remains blocked.",
        )
        return {
            "available": False,
            "valid": False,
            "reference": {},
            "manifest": {},
            "shared_target_times_hours": [],
            "available_domains": [],
            "native_cell_sizes_m": [],
            "issues": issues,
        }

    path = _resolved_path(text)
    if not path.is_file():
        _issue(
            issues,
            "warning",
            "mass_result_unavailable",
            "The mass-result manifest is currently unavailable.",
            path=str(path),
        )
        return {
            "available": False,
            "valid": False,
            "reference": {"path": str(path)},
            "manifest": {},
            "shared_target_times_hours": [],
            "available_domains": [],
            "native_cell_sizes_m": [],
            "issues": issues,
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError) as exc:
        _issue(
            issues,
            "error",
            "mass_result_unreadable",
            f"Could not read mass result: {type(exc).__name__}: {exc}",
        )
        payload = {}
    if not isinstance(payload, dict):
        _issue(
            issues,
            "error",
            "mass_result_invalid",
            "Mass-result manifest must contain a JSON object.",
        )
        payload = {}

    if payload.get("format") != MASS_ANALYSIS_RESULT_FORMAT:
        _issue(
            issues,
            "error",
            "mass_result_format",
            "Unsupported mass-result format.",
        )
    if str(payload.get("mass_analysis_plan_id") or "") != str(
        mass_plan.get("plan_id") or ""
    ):
        _issue(
            issues,
            "error",
            "mass_result_plan_mismatch",
            "Mass result belongs to a different mass-analysis plan.",
        )
    expected_contract = str(
        (mass_plan.get("task_contract") or {}).get("content_sha256") or ""
    )
    if str(payload.get("mass_task_contract_sha256") or "") != expected_contract:
        _issue(
            issues,
            "error",
            "mass_result_contract_mismatch",
            "Mass result used a different immutable mass task contract.",
        )
    if payload.get("complete") is not True:
        _issue(
            issues,
            "error",
            "mass_result_incomplete",
            "Mass-result manifest is not marked complete.",
        )

    expected_runs = _mass_run_ids(mass_plan)
    rows = payload.get("run_results") or []
    if not isinstance(rows, list):
        _issue(
            issues,
            "error",
            "mass_run_results_invalid",
            "Mass-result run_results must be a list.",
        )
        rows = []

    run_ids = [str(row.get("run_id") or "") for row in rows if isinstance(row, dict)]
    if len(run_ids) != len(set(run_ids)):
        _issue(
            issues,
            "error",
            "duplicate_mass_runs",
            "Mass-result manifest contains duplicate run identifiers.",
        )
    if sorted(run_ids) != expected_runs:
        _issue(
            issues,
            "error",
            "mass_result_run_mismatch",
            "Mass-result runs do not exactly match the current mass plan.",
            expected=expected_runs,
            actual=sorted(run_ids),
        )

    time_grids: list[list[float]] = []
    domain_sets: list[set[str]] = []
    native_cell_sizes: list[float] = []
    product_artifacts: list[dict[str, Any]] = []
    unavailable_artifact = False
    for row in rows:
        if not isinstance(row, dict):
            _issue(
                issues,
                "error",
                "mass_run_result_invalid",
                "Every run result must be a JSON object.",
            )
            continue
        run_id = str(row.get("run_id") or "")
        if row.get("complete") is not True:
            _issue(
                issues,
                "error",
                "mass_run_incomplete",
                f"Mass products for {run_id or 'unknown run'} are incomplete.",
            )

        raw_times = row.get("target_times_hours") or []
        parsed_times = [_number(value) for value in raw_times]
        if (
            not parsed_times
            or any(value is None for value in parsed_times)
            or [float(value) for value in parsed_times if value is not None]
            != sorted({float(value) for value in parsed_times if value is not None})
        ):
            _issue(
                issues,
                "error",
                "mass_target_times_invalid",
                f"Target times for {run_id or 'unknown run'} must be finite, unique, "
                "and strictly increasing.",
            )
        else:
            time_grids.append(
                [float(value) for value in parsed_times if value is not None]
            )

        domains = {str(value) for value in row.get("domains") or [] if str(value)}
        if "whole_active_domain" not in domains:
            _issue(
                issues,
                "error",
                "whole_domain_missing",
                f"Mass result for {run_id or 'unknown run'} lacks the active rig domain.",
            )
        domain_sets.append(domains)

        geometry = dict(row.get("geometry") or {})
        width = _number(geometry.get("width_m"))
        height = _number(geometry.get("height_m"))
        native_cell = _number(geometry.get("native_cell_size_m"))
        if width is None or width <= 0.0 or height is None or height <= 0.0:
            _issue(
                issues,
                "error",
                "rig_dimensions_invalid",
                f"Physical rig dimensions are missing for {run_id or 'unknown run'}.",
            )
        if native_cell is None or native_cell <= 0.0:
            _issue(
                issues,
                "error",
                "native_cell_size_invalid",
                f"Native physical cell size is missing for {run_id or 'unknown run'}.",
            )
        else:
            native_cell_sizes.append(native_cell)

        products = row.get("products") or {}
        if not isinstance(products, dict):
            products = {}
        for product in WASSERSTEIN_REQUIRED_INPUT_PRODUCTS:
            artifact = products.get(product)
            if not isinstance(artifact, dict):
                _issue(
                    issues,
                    "error",
                    "mass_product_missing",
                    f"{run_id or 'Unknown run'} lacks {product}.",
                )
                continue
            raw_artifact_path = str(artifact.get("path") or "").strip()
            expected_sha256 = str(artifact.get("sha256") or "")
            if not raw_artifact_path or not expected_sha256:
                _issue(
                    issues,
                    "error",
                    "mass_product_reference_invalid",
                    f"{run_id or 'Unknown run'} has an incomplete {product} reference.",
                )
                continue
            artifact_path = _resolved_path(raw_artifact_path, relative_to=path.parent)
            if not artifact_path.is_file():
                unavailable_artifact = True
                _issue(
                    issues,
                    "warning",
                    "mass_product_unavailable",
                    f"{product} for {run_id or 'unknown run'} is unavailable.",
                    path=str(artifact_path),
                )
                continue
            actual_sha256 = sha256_file(artifact_path)
            if actual_sha256 != expected_sha256:
                _issue(
                    issues,
                    "error",
                    "mass_product_hash_mismatch",
                    f"{product} for {run_id or 'unknown run'} failed its hash check.",
                )
                continue
            product_artifacts.append(
                {
                    "run_id": run_id,
                    "product": product,
                    "component": product,
                    "path": str(artifact_path),
                    "size_bytes": artifact_path.stat().st_size,
                    "sha256": actual_sha256,
                }
            )

        for field, label in (
            ("injected_mass_g", "Injected-mass reference"),
            ("reference_plume_radius_m", "Reference plume radius"),
        ):
            raw_values = row.get(field) or []
            values = [_number(value) for value in raw_values]
            if len(values) != len(raw_times) or any(
                value is None or value < 0.0 for value in values
            ):
                _issue(
                    issues,
                    "error",
                    f"{field}_invalid",
                    f"{label} for {run_id or 'unknown run'} must contain one finite, "
                    "non-negative value per target time.",
                )

    shared_times: list[float] = []
    if time_grids:
        shared_times = time_grids[0]
        if any(grid != shared_times for grid in time_grids[1:]):
            _issue(
                issues,
                "error",
                "mass_time_grids_differ",
                "Every run must expose the same synchronized target-time grid.",
            )

    available_domains = sorted(set.intersection(*domain_sets)) if domain_sets else []
    errors = sum(issue["severity"] == "error" for issue in issues)
    valid = errors == 0 and not unavailable_artifact
    return {
        "available": True,
        "valid": valid,
        "reference": _mass_result_reference(path, product_artifacts),
        "manifest": payload,
        "shared_target_times_hours": shared_times,
        "available_domains": available_domains,
        "native_cell_sizes_m": native_cell_sizes,
        "issues": issues,
    }


def wasserstein_task_contract(
    config: Mapping[str, Any],
    mass_result_reference: Mapping[str, Any] | None = None,
    analysis_domain_reference: Mapping[str, Any] | None = None,
    resolved_time_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the immutable numerical and scientific comparison contract."""

    contract = {
        "schema": 1,
        "format": WASSERSTEIN_TASK_CONTRACT_FORMAT,
        "mass_analysis_plan_id": str(config.get("mass_analysis_plan_id") or ""),
        "mass_analysis_plan_sha256": str(config.get("mass_analysis_plan_sha256") or ""),
        "mass_task_contract_sha256": str(config.get("mass_task_contract_sha256") or ""),
        "mass_result": deepcopy(dict(mass_result_reference or {})),
        "analysis_domains": deepcopy(dict(analysis_domain_reference or {})),
        "run_ids": deepcopy(list(config.get("run_ids") or [])),
        "comparison": deepcopy(dict(config.get("comparison") or {})),
        "resolved_time_selection": deepcopy(dict(resolved_time_selection or {})),
        "spatial_discretization": deepcopy(
            dict(config.get("spatial_discretization") or {})
        ),
        "transport": deepcopy(dict(config.get("transport") or {})),
        "outputs": deepcopy(dict(config.get("outputs") or {})),
        "execution_profile": deepcopy(dict(config.get("execution_profile") or {})),
    }
    contract["content_sha256"] = _json_digest(contract)
    return contract


def validate_wasserstein_plan(
    project: FluidFlowerProject,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a Wasserstein plan without loading fields or starting a solver."""

    candidate = deepcopy(dict(config))
    issues: list[dict[str, Any]] = []
    if candidate.get("format") != WASSERSTEIN_PLAN_FORMAT:
        _issue(issues, "error", "unsupported_format", "Unsupported Wasserstein plan")
    if bool(candidate.get("execution_requested")):
        _issue(
            issues,
            "error",
            "execution_not_supported",
            "Saving a Wasserstein plan cannot start computation.",
        )

    mass_plan = MassAnalysisPlanStore(project).current()
    mass_plan_valid = False
    mass_plan_ready = False
    if mass_plan.get("format") != MASS_ANALYSIS_REVISION_FORMAT:
        _issue(
            issues,
            "error",
            "mass_plan_missing",
            "Save a mass-analysis plan before Wasserstein planning.",
        )
    else:
        expected_pointers = (
            str(mass_plan.get("plan_id") or ""),
            str(mass_plan.get("content_sha256") or ""),
            str((mass_plan.get("task_contract") or {}).get("content_sha256") or ""),
        )
        actual_pointers = (
            str(candidate.get("mass_analysis_plan_id") or ""),
            str(candidate.get("mass_analysis_plan_sha256") or ""),
            str(candidate.get("mass_task_contract_sha256") or ""),
        )
        if actual_pointers != expected_pointers:
            _issue(
                issues,
                "error",
                "mass_plan_changed",
                "Wasserstein plan does not reference the current mass-analysis plan.",
            )
        mass_report = validate_mass_analysis_plan(
            project, dict(mass_plan.get("configuration") or {})
        )
        mass_plan_valid = bool(mass_report.get("valid"))
        mass_plan_ready = bool(mass_report.get("execution_ready"))
        if not mass_plan_valid:
            _issue(
                issues,
                "error",
                "mass_plan_invalid",
                "Current mass-analysis plan no longer matches its upstream state.",
            )
        elif not mass_plan_ready:
            _issue(
                issues,
                "warning",
                "mass_plan_not_execution_ready",
                "Current mass-analysis plan is valid but not yet execution ready.",
            )

    domain_revision, domain_config = _current_domain_state(project)
    domain_revision_valid = True
    expected_domain_pointers = (
        str(domain_revision.get("revision_id") or ""),
        str(domain_revision.get("content_sha256") or ""),
    )
    actual_domain_pointers = (
        str(candidate.get("analysis_domain_revision_id") or ""),
        str(candidate.get("analysis_domain_revision_sha256") or ""),
    )
    if actual_domain_pointers != expected_domain_pointers:
        domain_revision_valid = False
        _issue(
            issues,
            "error",
            "analysis_domains_changed",
            "Wasserstein plan does not reference the current analysis-domain revision.",
        )
    if domain_revision:
        domain_report = validate_analysis_domain_config(project, domain_config)
        domain_revision_valid = domain_revision_valid and bool(domain_report["valid"])
        if not domain_report["valid"]:
            _issue(
                issues,
                "error",
                "analysis_domains_invalid",
                "Current analysis domains no longer match image correction or geometry.",
            )

    expected_runs = _mass_run_ids(mass_plan)
    run_ids = sorted(str(value) for value in candidate.get("run_ids") or [])
    if not set(run_ids).issubset(expected_runs):
        _issue(
            issues,
            "error",
            "run_set_invalid",
            "Selected Wasserstein runs must come from the current mass-analysis plan.",
        )
    if len(run_ids) < 2:
        _issue(
            issues,
            "error",
            "insufficient_runs",
            "At least two runs are required for pairwise Wasserstein analysis.",
        )

    comparison = dict(candidate.get("comparison") or {})
    if comparison.get("pairing_policy") != "explicit_unordered_run_pairs":
        _issue(
            issues,
            "error",
            "pairing_policy_invalid",
            "Wasserstein comparisons require an explicit unordered pair list.",
        )
    raw_pairs = comparison.get("pairs") or []
    pairs: list[tuple[str, str]] = []
    for row in raw_pairs:
        if not isinstance(row, Mapping):
            continue
        run_1 = str(row.get("run_1") or "")
        run_2 = str(row.get("run_2") or "")
        pairs.append(tuple(sorted((run_1, run_2))))
    valid_pair_set = bool(pairs) and all(
        run_1 and run_2 and run_1 != run_2 and run_1 in run_ids and run_2 in run_ids
        for run_1, run_2 in pairs
    )
    if not valid_pair_set or len(pairs) != len(set(pairs)):
        _issue(
            issues,
            "error",
            "pair_selection_invalid",
            "Pairs must be unique, unordered, and use selected runs.",
        )
    participating_runs = {run_id for pair in pairs for run_id in pair}
    if participating_runs != set(run_ids):
        _issue(
            issues,
            "error",
            "selected_run_without_pair",
            "Every selected run must participate in at least one comparison.",
        )
    if (
        comparison.get("time_policy") != "shared_synchronized_target_times"
        or comparison.get("require_same_target_times") is not True
    ):
        _issue(
            issues,
            "error",
            "time_policy_invalid",
            "All runs must be compared at the same synchronized target times.",
        )
    tolerance = _number(comparison.get("recorded_mass_time_tolerance_seconds"))
    expected_tolerance = _number(
        ((mass_plan.get("configuration") or {}).get("image_selection") or {}).get(
            "drift_tolerance_seconds"
        )
    )
    if tolerance is None or tolerance < 0.0 or tolerance != expected_tolerance:
        _issue(
            issues,
            "error",
            "time_tolerance_changed",
            "Recorded drift tolerance must be inherited from mass analysis.",
        )

    components = [str(value) for value in comparison.get("mass_components") or []]
    if not components or len(components) != len(set(components)):
        _issue(
            issues,
            "error",
            "mass_components_invalid",
            "Select one or more unique mass components.",
        )
    unsupported_components = sorted(set(components) - set(WASSERSTEIN_MASS_COMPONENTS))
    if unsupported_components:
        _issue(
            issues,
            "error",
            "mass_components_unsupported",
            "Wasserstein plan contains unsupported mass components.",
            components=unsupported_components,
        )

    domains = [str(value) for value in comparison.get("domains") or []]
    if not domains or len(domains) != len(set(domains)):
        _issue(
            issues,
            "error",
            "domains_invalid",
            "Select one or more unique analysis domains.",
        )
    domain_rows = {
        str(row.get("domain_id") or ""): deepcopy(dict(row))
        for row in domain_config.get("domains") or []
        if row.get("domain_id") and bool(row.get("enabled", True))
    }
    allowed_domains = {"whole_active_domain", *domain_rows}
    unsupported_domains = sorted(set(domains) - allowed_domains)
    if unsupported_domains:
        _issue(
            issues,
            "error",
            "domains_unsupported",
            "Wasserstein plan contains unsupported analysis domains.",
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

    spatial = dict(candidate.get("spatial_discretization") or {})
    expected_spatial = {
        "input_preference": "matching_mass_analysis_cache_else_native_mass_fields",
        "resolution_policy": "explicit_common_physical_cell_size",
        "interpolation": "conservative_area",
        "preserve_integrated_mass": True,
        "forbid_upsampling": True,
        "dimensionless_resize_factor": None,
    }
    for key, expected in expected_spatial.items():
        if spatial.get(key) != expected:
            _issue(
                issues,
                "error",
                "spatial_contract_changed",
                "Wasserstein inputs must use one mass-preserving physical grid; "
                "dimensionless resize factors are not accepted.",
                field=key,
            )
    target_cell_size = _number(spatial.get("target_cell_size_m"))
    if spatial.get("target_cell_size_m") is None:
        _issue(
            issues,
            "warning",
            "physical_cell_size_missing",
            "Choose a common physical cell size before execution.",
        )
    elif target_cell_size is None or target_cell_size <= 0.0:
        _issue(
            issues,
            "error",
            "physical_cell_size_invalid",
            "Physical cell size must be finite and positive.",
        )

    expected_transport = default_wasserstein_plan(project)["transport"]
    if dict(candidate.get("transport") or {}) != expected_transport:
        _issue(
            issues,
            "error",
            "transport_contract_changed",
            "Transport, equal-mass, permeability, and normalization semantics changed.",
        )

    outputs = dict(candidate.get("outputs") or {})
    metrics = [str(value) for value in outputs.get("metrics") or []]
    if set(metrics) != set(WASSERSTEIN_METRICS) or len(metrics) != len(
        WASSERSTEIN_METRICS
    ):
        _issue(
            issues,
            "error",
            "metrics_incomplete",
            "Raw cost, normalized distances, and mass differences are all required.",
        )
    if outputs.get("intermediate_format") != "json" or outputs.get("table_formats") != [
        "csv",
        "xlsx",
    ]:
        _issue(
            issues,
            "error",
            "output_contract_changed",
            "Checkpoint JSON and final CSV/XLSX outputs are required.",
        )

    execution = dict(candidate.get("execution_profile") or {})
    if execution.get("backend") != "cpu-float64":
        _issue(
            issues,
            "error",
            "backend_invalid",
            "The validated variational Wasserstein solver uses CPU float64.",
        )
    workers = execution.get("workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        _issue(issues, "error", "workers_invalid", "Worker count must be positive.")
    cache_timepoints = execution.get("cache_timepoints_per_worker")
    if (
        not isinstance(cache_timepoints, int)
        or isinstance(cache_timepoints, bool)
        or cache_timepoints < 1
    ):
        _issue(
            issues,
            "error",
            "cache_timepoints_invalid",
            "Each worker must retain at least one target time in memory.",
        )
    required_execution = {
        "task_order": "time_pair_domain_component",
        "reuse_loaded_mass_fields": True,
        "checkpoint_each_task": True,
        "resume_completed_tasks": True,
        "retry_transient_io": True,
        "atomic_publish": True,
    }
    for key, expected in required_execution.items():
        if execution.get(key) != expected:
            _issue(
                issues,
                "error",
                "execution_contract_changed",
                "Wasserstein execution must be resumable and interruption safe.",
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

    mass_result = inspect_mass_analysis_result(
        str(candidate.get("mass_result_manifest_path") or ""), mass_plan
    )
    issues.extend(deepcopy(mass_result["issues"]))
    source_time_count = len(mass_result["shared_target_times_hours"])
    if (
        mass_result["valid"]
        and target_cell_size is not None
        and mass_result["native_cell_sizes_m"]
    ):
        coarsest_native = max(mass_result["native_cell_sizes_m"])
        if target_cell_size < coarsest_native:
            _issue(
                issues,
                "error",
                "physical_grid_would_upsample",
                "The common grid is finer than at least one source field and "
                "would create false spatial precision.",
                target_cell_size_m=target_cell_size,
                coarsest_native_cell_size_m=coarsest_native,
            )

    gate_status = str(project.module_states["mass_analysis"].get("status") or "")
    gate_ready = gate_status in SATISFIED_MODULE_STATUSES
    if not gate_ready:
        _issue(
            issues,
            "warning",
            "mass_gate_incomplete",
            "Mass analysis must be approved or complete before execution.",
            status=gate_status,
        )

    pair_count = len(pairs)
    tasks_per_time = pair_count * len(domains) * len(components)
    resolved_time_selection: dict[str, Any] = {
        "valid": False,
        "source_time_count": source_time_count,
        "selected_time_count": 0,
        "selected_time_indices": [],
        "selected_times_hours": [],
        "selected_solver_task_count": 0,
        "estimated_publication_bytes": 0,
        "issues": [],
    }
    if source_time_count:
        estimate = estimate_wasserstein_publication_bytes(
            tasks_per_time=tasks_per_time,
            store_transport_diagnostics=bool(
                outputs.get("store_transport_diagnostics")
            ),
        )
        resolved_time_selection = resolve_time_selection(
            mass_result["shared_target_times_hours"],
            dict(comparison.get("time_selection_budget") or {}),
            tasks_per_time=tasks_per_time,
            fixed_publication_bytes=estimate["fixed_bytes"],
            publication_bytes_per_time=estimate["bytes_per_time"],
        )
        issues.extend(deepcopy(resolved_time_selection["issues"]))
    time_count = int(resolved_time_selection.get("selected_time_count") or 0)
    task_count = int(resolved_time_selection.get("selected_solver_task_count") or 0)
    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    valid = errors == 0
    execution_ready = (
        valid
        and mass_plan_ready
        and mass_result["valid"]
        and domain_revision_valid
        and target_cell_size is not None
        and resolved_time_selection["valid"]
        and gate_ready
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
            "pair_count": pair_count,
            "time_count": time_count,
            "source_time_count": source_time_count,
            "domain_count": len(domains),
            "mass_component_count": len(components),
            "task_count": task_count,
            "estimated_publication_bytes": int(
                resolved_time_selection.get("estimated_publication_bytes") or 0
            ),
            "backend": execution.get("backend"),
            "mass_result_ready": mass_result["valid"],
            "mass_plan_ready": mass_plan_ready,
            "mass_gate_ready": gate_ready,
            "analysis_domains_ready": domain_revision_valid,
        },
        "task_contract": wasserstein_task_contract(
            candidate,
            mass_result["reference"],
            analysis_domain_reference,
            resolved_time_selection,
        ),
        "mass_result": mass_result,
        "analysis_domain_reference": analysis_domain_reference,
        "resolved_time_selection": resolved_time_selection,
        "mass_gate_status": gate_status,
        "issues": issues,
    }


class WassersteinPlanStore:
    """Append-only Wasserstein plans; execution belongs to a separate runner."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "wasserstein"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        self.artifact_root = project.root / "results" / "wasserstein" / "plans"

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
        report = validate_wasserstein_plan(self.project, candidate)
        if int((report.get("summary") or {}).get("error_count") or 0) > 0:
            raise ValueError("Wasserstein plan has blocking validation errors")
        plan_id = uuid.uuid4().hex
        mass_plan = MassAnalysisPlanStore(self.project).current()
        revision = {
            "schema": 1,
            "format": WASSERSTEIN_REVISION_FORMAT,
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
                "mass_result": deepcopy(report["mass_result"]["reference"]),
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
                "contains_wasserstein_results": False,
                "loads_mass_fields": False,
                "balances_mass_for_transport_only": True,
                "preserves_raw_mass_difference": True,
                "uses_physical_spatial_resolution": True,
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
            "wasserstein_plan_saved",
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
