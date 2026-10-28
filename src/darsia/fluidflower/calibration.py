"""Versioned, execution-free calibration plans for FluidFlower projects."""

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

from darsia.fluidflower.phase_interpretation import (
    PHASE_INTERPRETATION_REVISION_FORMAT,
    PhaseInterpretationRevisionStore,
)
from darsia.fluidflower.indicator_response import (
    CHEMICAL_PARAMETER_KEYS,
    OPTICAL_PARAMETERS_REPLACED_BY_RESPONSE_MODEL,
    inspect_indicator_response_model_reference,
)
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.quality_budget import (
    default_quality_profile,
    numerical_dtype_for_backend,
    summarize_quality_contract,
    validate_quality_profile,
)

CALIBRATION_PLAN_FORMAT = "darsia-fluidflower-calibration-plan-v1"
CALIBRATION_REVISION_FORMAT = "darsia-fluidflower-calibration-revision-v1"
CALIBRATION_PARAMETER_SCHEMA_FORMAT = (
    "darsia-fluidflower-calibration-parameter-schema-v1"
)
CALIBRATION_EVALUATION_MANIFEST_FORMAT = (
    "darsia-fluidflower-calibration-evaluation-manifest-v1"
)
CHEMISTRY_MODELS = (
    "simple_flash",
    "titration",
    "compare_simple_flash_and_titration",
)
NUMERICAL_BACKENDS = (
    "cpu-float64",
    "cpu-float32",
    "cuda-float32",
    "cuda-float64",
    "opencl-float32",
    "opencl-float64",
    "metal-float32",
)
OBJECTIVE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "key": "injected_mass_balance",
        "label": "Injected-versus-detected mass",
        "default_weight": 1.0,
        "default_enabled": True,
        "direction": "minimize_penalty",
        "normalization": "absolute_mass_error_divided_by_injected_mass",
    },
    {
        "key": "overflow_phase_constraint",
        "label": "Overflow-derived phase constraint",
        "default_weight": 1.0,
        "default_enabled": True,
        "direction": "minimize_penalty",
        "normalization": "phase_mass_error_divided_by_available_overflow_mass",
    },
    {
        "key": "manual_phase_review",
        "label": "Manual phase-review agreement",
        "default_weight": 1.0,
        "default_enabled": True,
        "direction": "minimize_penalty",
        "normalization": "review_region_error_divided_by_review_region_count",
    },
    {
        "key": "temporal_physics",
        "label": "Temporal physical plausibility",
        "default_weight": 0.5,
        "default_enabled": True,
        "direction": "minimize_penalty",
        "normalization": "weighted_violation_sum_divided_by_evaluated_frame_count",
    },
    {
        "key": "spatial_artifacts",
        "label": "Spatial artifact penalty",
        "default_weight": 0.5,
        "default_enabled": True,
        "direction": "minimize_penalty",
        "normalization": "artifact_area_divided_by_active_porous_area",
    },
)

OBJECTIVE_CONTRACT_VERSION = "fluidflower-calibration-objective-v1"

PARAMETER_SCOPES = ("shared_across_series", "per_run", "hierarchical")
PARAMETER_SCALES = ("linear", "log")
PARAMETER_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "key": "aqueous_transition_low_fraction",
        "label": "Aqueous transition start",
        "default_enabled": True,
        "default_lower": 0.0,
        "default_upper": 1.0,
        "default_scale": "linear",
        "default_scope": "per_run",
        "unit": "fraction of run-specific optical range",
        "models": list(CHEMISTRY_MODELS),
        "transform": "run_optical_quantile_window_v1",
        "scientific_role": "Lower optical bound of the aqueous CO2 response.",
    },
    {
        "key": "aqueous_transition_width_fraction",
        "label": "Aqueous transition width",
        "default_enabled": True,
        "default_lower": 0.0,
        "default_upper": 1.0,
        "default_scale": "linear",
        "default_scope": "per_run",
        "unit": "fraction of remaining optical range",
        "models": list(CHEMISTRY_MODELS),
        "transform": "positive_run_optical_width_v1",
        "scientific_role": "Positive width of the aqueous CO2 transition.",
    },
    {
        "key": "gas_transition_onset",
        "label": "Free-gas transition onset",
        "default_enabled": True,
        "default_lower": 0.02,
        "default_upper": 0.70,
        "default_scale": "linear",
        "default_scope": "per_run",
        "unit": "gas probability",
        "models": list(CHEMISTRY_MODELS),
        "transform": "direct_probability_v1",
        "scientific_role": "Gas-model probability where free-gas saturation starts.",
    },
    {
        "key": "gas_transition_width_fraction",
        "label": "Free-gas transition width",
        "default_enabled": True,
        "default_lower": 0.0,
        "default_upper": 1.0,
        "default_scale": "linear",
        "default_scope": "per_run",
        "unit": "fraction of remaining gas-probability range",
        "models": list(CHEMISTRY_MODELS),
        "transform": "positive_probability_width_v1",
        "scientific_role": "Positive width from gas onset to full saturation response.",
    },
    {
        "key": "maximum_gas_saturation",
        "label": "Maximum free-gas saturation",
        "default_enabled": True,
        "default_lower": 0.05,
        "default_upper": 0.85,
        "default_scale": "linear",
        "default_scope": "hierarchical",
        "unit": "pore-volume fraction",
        "models": list(CHEMISTRY_MODELS),
        "transform": "direct_fraction_v1",
        "scientific_role": "Upper saturation represented by gas-dominant pixels.",
    },
    {
        "key": "initial_ph",
        "label": "Initial pH",
        "default_enabled": True,
        "default_lower": 6.0,
        "default_upper": 10.0,
        "default_scale": "linear",
        "default_scope": "per_run",
        "unit": "pH",
        "models": ["titration", "compare_simple_flash_and_titration"],
        "transform": "direct_ph_v1",
        "scientific_role": "Initial pH controlling the titration response curve.",
    },
    {
        "key": "aqueous_titration_blend",
        "label": "Titration/linear aqueous blend",
        "default_enabled": True,
        "default_lower": 0.0,
        "default_upper": 1.0,
        "default_scale": "linear",
        "default_scope": "shared_across_series",
        "unit": "fraction linear response",
        "models": ["titration", "compare_simple_flash_and_titration"],
        "transform": "direct_fraction_v1",
        "scientific_role": (
            "Sensitivity between the physical titration curve and a linear response."
        ),
    },
    {
        "key": "alkalinity_M",
        "label": "Effective alkalinity",
        "default_enabled": False,
        "default_lower": 5.0e-4,
        "default_upper": 3.0e-3,
        "default_scale": "linear",
        "default_scope": "per_run",
        "unit": "mol/litre",
        "models": ["titration", "compare_simple_flash_and_titration"],
        "transform": "reviewed_chemical_interval_v1",
        "scientific_role": (
            "Effective alkalinity; searchable only inside a reviewed measured bound."
        ),
    },
    {
        "key": "indicator_concentration_M",
        "label": "Indicator concentration",
        "default_enabled": False,
        "default_lower": 1.0e-4,
        "default_upper": 2.0e-3,
        "default_scale": "linear",
        "default_scope": "shared_across_series",
        "unit": "mol/litre",
        "models": ["titration", "compare_simple_flash_and_titration"],
        "transform": "reviewed_chemical_interval_v1",
        "scientific_role": (
            "Indicator concentration; searchable only inside a reviewed measured bound."
        ),
    },
    {
        "key": "co2_saturation_M",
        "label": "Aqueous CO2 saturation concentration",
        "default_enabled": False,
        "default_lower": 0.02,
        "default_upper": 0.06,
        "default_scale": "linear",
        "default_scope": "shared_across_series",
        "unit": "mol/litre",
        "models": ["titration", "compare_simple_flash_and_titration"],
        "transform": "reviewed_chemical_interval_v1",
        "scientific_role": (
            "Saturation scale; searchable only inside a reviewed measured bound."
        ),
    },
    {
        "key": "diffusion_coefficient_m2_s",
        "label": "Aqueous spatial diffusion",
        "default_enabled": False,
        "default_lower": 1.0e-10,
        "default_upper": 1.0e-8,
        "default_scale": "log",
        "default_scope": "shared_across_series",
        "unit": "m2/s",
        "models": list(CHEMISTRY_MODELS),
        "transform": "mass_conserving_spatial_diffusion_v1",
        "scientific_role": (
            "Optional mass-conserving redistribution; requires a spatial evaluator."
        ),
    },
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
                raise TypeError(f"Invalid calibration revision at line {line_number}")
            rows.append(payload)
    return rows


def _json_digest(payload: Any) -> str:
    from darsia.fluidflower.analysis_domains import _json_digest as _canonical_ascii_digest
    return _canonical_ascii_digest(payload)


def _number(value: Any) -> float | None:
    from darsia.fluidflower.mass_inputs import _number as _canonical_finite_number
    return _canonical_finite_number(value)


def default_calibration_parameter_space(
    initial_ph_bounds: tuple[float, float] | list[float] = (6.0, 10.0),
) -> dict[str, Any]:
    """Return the complete, explicit Optuna search-space contract."""

    ph_values = list(initial_ph_bounds)
    if len(ph_values) != 2:
        ph_values = [6.0, 10.0]
    ph_low = _number(ph_values[0])
    ph_high = _number(ph_values[1])
    if ph_low is None or ph_high is None or not 0.0 <= ph_low < ph_high <= 14.0:
        ph_low, ph_high = 6.0, 10.0
    parameters: list[dict[str, Any]] = []
    for definition in PARAMETER_DEFINITIONS:
        lower = float(definition["default_lower"])
        upper = float(definition["default_upper"])
        if definition["key"] == "initial_ph":
            lower, upper = float(ph_low), float(ph_high)
        parameters.append(
            {
                "key": str(definition["key"]),
                "label": str(definition["label"]),
                "enabled": bool(definition["default_enabled"]),
                "lower": lower,
                "upper": upper,
                "scale": str(definition["default_scale"]),
                "scope": str(definition["default_scope"]),
                "unit": str(definition["unit"]),
                "models": list(definition["models"]),
                "transform": str(definition["transform"]),
                "scientific_role": str(definition["scientific_role"]),
            }
        )
    return {
        "schema": 1,
        "format": CALIBRATION_PARAMETER_SCHEMA_FORMAT,
        "parameters": parameters,
        "constraints": [
            "aqueous transition width is strictly positive after transformation",
            "gas transition width is strictly positive after transformation",
            "all concentration and saturation fractions are clipped to [0, 1]",
            "spatial diffusion must conserve total mass",
            "measured indicator response is immutable during Optuna optimization",
            "chemical parameters remain inside reviewed measured bounds",
        ],
    }


def _canonical_parameter_space(
    raw: Any,
    *,
    initial_ph_bounds: tuple[float, float] | list[float] = (6.0, 10.0),
) -> dict[str, Any]:
    default = default_calibration_parameter_space(initial_ph_bounds)
    if not isinstance(raw, Mapping):
        return default
    by_key = {
        str(row.get("key") or ""): dict(row)
        for row in raw.get("parameters") or []
        if isinstance(row, Mapping)
    }
    parameters: list[dict[str, Any]] = []
    for fallback in default["parameters"]:
        row = by_key.get(str(fallback["key"]), {})
        parameters.append(
            {
                **fallback,
                "enabled": bool(row.get("enabled", fallback["enabled"])),
                "lower": row.get("lower", fallback["lower"]),
                "upper": row.get("upper", fallback["upper"]),
                "scale": str(row.get("scale") or fallback["scale"]),
                "scope": str(row.get("scope") or fallback["scope"]),
            }
        )
    return {
        "schema": 1,
        "format": CALIBRATION_PARAMETER_SCHEMA_FORMAT,
        "parameters": parameters,
        "constraints": list(default["constraints"]),
    }


def canonical_calibration_parameter_space(
    raw: Any,
    *,
    initial_ph_bounds: tuple[float, float] | list[float] = (6.0, 10.0),
) -> dict[str, Any]:
    """Upgrade an older explicit parameter table to the current schema."""

    return _canonical_parameter_space(raw, initial_ph_bounds=initial_ph_bounds)


def apply_indicator_response_constraints(
    config: Mapping[str, Any],
    model_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind measured optical response and reviewed chemistry bounds to a plan."""

    inspected = inspect_indicator_response_model_reference(model_reference)
    if not inspected.get("valid"):
        raise ValueError(" ".join(str(value) for value in inspected.get("issues") or []))
    constrained = deepcopy(dict(config))
    chemistry = dict(constrained.get("chemistry") or {})
    chemistry["indicator_response_model"] = {
        "schema": 1,
        "format": str(model_reference.get("format") or ""),
        "model_id": str(inspected.get("model_id") or ""),
        "path": str(inspected.get("path") or ""),
        "sha256": str(inspected.get("sha256") or ""),
        "status": str(inspected.get("status") or ""),
    }
    response_chemistry = dict(inspected.get("chemistry") or {})
    chemistry["chemical_parameter_values"] = {
        key: response_chemistry.get(source_key)
        for key, source_key in CHEMICAL_PARAMETER_KEYS.items()
        if response_chemistry.get(source_key) is not None
    }
    chemistry["measured_uncertainty_bounds"] = deepcopy(
        dict(response_chemistry.get("measured_bounds") or {})
    )
    chemistry["optical_response_policy"] = "fixed_measured_indicator_response"
    constrained["chemistry"] = chemistry

    parameter_space = _canonical_parameter_space(
        constrained.get("parameter_space"),
        initial_ph_bounds=list(chemistry.get("initial_ph_bounds") or [6.0, 10.0]),
    )
    measured_bounds = dict(response_chemistry.get("measured_bounds") or {})
    initial_bounds = list(measured_bounds.get("initial_ph") or [])
    if len(initial_bounds) == 2:
        initial_low, initial_high = (
            _number(initial_bounds[0]),
            _number(initial_bounds[1]),
        )
        if (
            initial_low is not None
            and initial_high is not None
            and initial_low < initial_high
        ):
            chemistry["initial_ph_bounds"] = [initial_low, initial_high]
            constrained["chemistry"] = chemistry
    for row in parameter_space["parameters"]:
        key = str(row["key"])
        if key in OPTICAL_PARAMETERS_REPLACED_BY_RESPONSE_MODEL:
            row["enabled"] = False
            continue
        source_key = CHEMICAL_PARAMETER_KEYS.get(key)
        if source_key is None:
            continue
        bounds = list(measured_bounds.get(source_key) or [])
        if len(bounds) != 2:
            row["enabled"] = False
            continue
        low, high = (_number(bounds[0]), _number(bounds[1]))
        if low is None or high is None or low > high:
            raise ValueError(f"Invalid measured bounds for {source_key}")
        if low < high:
            row.update({"enabled": True, "lower": low, "upper": high})
        else:
            row["enabled"] = False
    constrained["parameter_space"] = parameter_space
    fixed = dict(constrained.get("fixed_transfer_contract") or {})
    fixed.update(
        {
            "aqueous_response": "fixed_measured_indicator_response",
            "indicator_response_model_sha256": str(inspected.get("sha256") or ""),
        }
    )
    constrained["fixed_transfer_contract"] = fixed
    return constrained


def inspect_calibration_evaluation_manifest(path_value: Any) -> dict[str, Any]:
    """Inspect the prepared evaluator contract without executing calibration."""

    text = str(path_value or "").strip()
    if not text:
        return {
            "available": False,
            "valid": False,
            "path": "",
            "supported_objectives": [],
            "supported_parameters": [],
            "run_count": 0,
            "issues": ["No prepared evaluation manifest is attached."],
        }
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        return {
            "available": False,
            "valid": False,
            "path": str(path),
            "supported_objectives": [],
            "supported_parameters": [],
            "run_count": 0,
            "issues": [f"Evaluation manifest is unavailable: {path}"],
        }
    issues: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "available": True,
            "valid": False,
            "path": str(path),
            "supported_objectives": [],
            "supported_parameters": [],
            "run_count": 0,
            "issues": [f"Could not read evaluation manifest: {exc}"],
        }
    if not isinstance(payload, Mapping):
        issues.append("Evaluation manifest must contain one JSON object.")
        payload = {}
    if payload.get("format") != CALIBRATION_EVALUATION_MANIFEST_FORMAT:
        issues.append("Unsupported calibration evaluation manifest format.")
    supported_objectives = sorted(
        {
            str(value).strip()
            for value in payload.get("supported_objectives") or []
            if str(value).strip()
        }
    )
    supported_parameters = sorted(
        {
            str(value).strip()
            for value in payload.get("supported_parameters") or []
            if str(value).strip()
        }
    )
    known_objectives = {str(row["key"]) for row in OBJECTIVE_DEFINITIONS}
    known_parameters = {str(row["key"]) for row in PARAMETER_DEFINITIONS}
    unknown_objectives = sorted(set(supported_objectives) - known_objectives)
    unknown_parameters = sorted(set(supported_parameters) - known_parameters)
    if unknown_objectives:
        issues.append(
            "Unknown supported objectives: " + ", ".join(unknown_objectives)
        )
    if unknown_parameters:
        issues.append(
            "Unknown supported parameters: " + ", ".join(unknown_parameters)
        )
    runs = [row for row in payload.get("runs") or [] if isinstance(row, Mapping)]
    if not runs:
        issues.append("Evaluation manifest contains no run entries.")
    run_ids: set[str] = set()
    for index, run in enumerate(runs):
        run_id = str(run.get("run_id") or "").strip()
        if not run_id:
            issues.append(f"Evaluation run {index + 1} has no run_id.")
        elif run_id in run_ids:
            issues.append(f"Evaluation run_id is duplicated: {run_id}")
        run_ids.add(run_id)
        data_text = str(run.get("data_path") or "").strip()
        if not data_text:
            issues.append(f"Evaluation run {run_id or index + 1} has no data_path.")
            continue
        data_path = Path(data_text).expanduser()
        if not data_path.is_absolute():
            data_path = path.parent / data_path
        data_path = data_path.resolve()
        if not data_path.is_file():
            issues.append(f"Evaluation data is unavailable: {data_path}")
            continue
        expected_sha256 = str(run.get("data_sha256") or "").strip()
        if not expected_sha256:
            issues.append(f"Evaluation data has no SHA-256: {data_path}")
        elif hashlib.sha256(data_path.read_bytes()).hexdigest() != expected_sha256:
            issues.append(f"Evaluation data failed its SHA-256 check: {data_path}")
    return {
        "available": True,
        "valid": not issues,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evaluator": str(payload.get("evaluator") or ""),
        "phase_revision_sha256": str(payload.get("phase_revision_sha256") or ""),
        "supported_objectives": supported_objectives,
        "supported_parameters": supported_parameters,
        "run_count": len(runs),
        "issues": issues,
    }


def default_calibration_plan(project: FluidFlowerProject) -> dict[str, Any]:
    """Return an editable plan bound to the current phase revision."""

    phase_revision = PhaseInterpretationRevisionStore(project).current()
    chemistry = dict(project.manifest.get("chemistry") or {})
    initial_ph = _number(chemistry.get("initial_ph"))
    temperature = _number(chemistry.get("temperature"))
    pressure = _number(chemistry.get("pressure"))
    indicator_response_reference = deepcopy(
        dict(chemistry.get("indicator_response_reference") or {})
    )
    ph_low = max(0.0, initial_ph - 0.75) if initial_ph is not None else 6.0
    ph_high = min(14.0, initial_ph + 0.75) if initial_ph is not None else 10.0
    overflow_mode = str((project.manifest.get("overflow") or {}).get("mode") or "none")
    objectives = []
    for definition in OBJECTIVE_DEFINITIONS:
        enabled = bool(definition["default_enabled"])
        if definition["key"] == "overflow_phase_constraint" and overflow_mode == "none":
            enabled = False
        objectives.append(
            {
                "key": definition["key"],
                "enabled": enabled,
                "weight": float(definition["default_weight"]),
            }
        )
    quality_profile = default_quality_profile("production")
    quality_profile["precision"]["compute_dtype"] = "float64"
    return {
        "schema": 1,
        "format": CALIBRATION_PLAN_FORMAT,
        "phase_revision_id": str(phase_revision.get("revision_id") or ""),
        "phase_revision_sha256": str(phase_revision.get("content_sha256") or ""),
        "chemistry": {
            "indicator_system": str(chemistry.get("indicator_system") or "unspecified"),
            "model": "compare_simple_flash_and_titration",
            "initial_ph_bounds": [ph_low, ph_high],
            "temperature_c": 20.0 if temperature is None else temperature,
            "pressure_bar": 1.0 if pressure is None else pressure,
            "indicator_response_reference": indicator_response_reference,
        },
        "parameter_space": default_calibration_parameter_space([ph_low, ph_high]),
        "fixed_transfer_contract": {
            "transfer_shape": "smoothstep",
            "co2_support_onset": 0.15,
            "co2_support_high": 0.50,
            "phase_probabilities_are_visible_2d_support": True,
        },
        "evaluation_manifest_path": "",
        "geometry_policy": {
            "depth": "fixed_current_geometry_revision",
            "porosity": "fixed_current_geometry_revision",
            "two_d_to_three_d": "deferred_to_mass_analysis",
        },
        "objectives": objectives,
        "quality_profile": quality_profile,
        "optimizer": {
            "engine": "optuna",
            "trial_budget_per_model": 5000,
            "sampler_seed": 20_260_901,
            "selection_uses_identical_objective": True,
        },
        "execution_profile": {
            "primary_backend": "cpu-float64",
            "reference_backend": "cpu-float64",
            "workers": 1,
            "absolute_parity_tolerance": 1.0e-5,
            "relative_parity_tolerance": 1.0e-4,
        },
        "overflow_policy": "use_when_available",
        "execution_requested": False,
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


def _canonical_objectives(raw: Any) -> list[dict[str, Any]]:
    allowed = {str(row["key"]) for row in OBJECTIVE_DEFINITIONS}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw or []:
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("key") or "").strip()
        if key not in allowed or key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "key": key,
                "enabled": bool(row.get("enabled")),
                "weight": _number(row.get("weight")),
            }
        )
    return sorted(output, key=lambda row: str(row["key"]))


def calibration_objective_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact contract shared by optimization and final selection."""

    chemistry = dict(config.get("chemistry") or {})
    parameter_space = _canonical_parameter_space(
        config.get("parameter_space"),
        initial_ph_bounds=list(chemistry.get("initial_ph_bounds") or [6.0, 10.0]),
    )
    evaluation = inspect_calibration_evaluation_manifest(
        config.get("evaluation_manifest_path")
    )
    definition_by_key = {
        str(definition["key"]): definition for definition in OBJECTIVE_DEFINITIONS
    }
    objective_terms = []
    for row in _canonical_objectives(config.get("objectives")):
        definition = definition_by_key[str(row["key"])]
        objective_terms.append(
            {
                **row,
                "direction": str(definition["direction"]),
                "normalization": str(definition["normalization"]),
            }
        )
    contract = {
        "schema": 1,
        "format": OBJECTIVE_CONTRACT_VERSION,
        "score_direction": "minimize",
        "chemistry": deepcopy(chemistry),
        "parameter_space": parameter_space,
        "fixed_transfer_contract": deepcopy(
            dict(config.get("fixed_transfer_contract") or {})
        ),
        "evaluation_manifest": {
            "path": str(evaluation.get("path") or ""),
            "sha256": str(evaluation.get("sha256") or ""),
            "evaluator": str(evaluation.get("evaluator") or ""),
            "phase_revision_sha256": str(
                evaluation.get("phase_revision_sha256") or ""
            ),
            "supported_objectives": list(
                evaluation.get("supported_objectives") or []
            ),
            "supported_parameters": list(
                evaluation.get("supported_parameters") or []
            ),
            "run_count": int(evaluation.get("run_count") or 0),
        },
        "objectives": objective_terms,
        "phase_revision_id": str(config.get("phase_revision_id") or ""),
        "phase_revision_sha256": str(config.get("phase_revision_sha256") or ""),
        "geometry_policy": deepcopy(dict(config.get("geometry_policy") or {})),
        "quality_profile": deepcopy(dict(config.get("quality_profile") or {})),
        "overflow_policy": str(config.get("overflow_policy") or ""),
    }
    if config.get("native_fixed_phase_contract"):
        # Native fixed-label calibration extends this same plan/review workflow.
        # Its response scales must be included in the selection fingerprint.
        contract["native_fixed_phase_contract"] = deepcopy(
            dict(config["native_fixed_phase_contract"])
        )
    contract["content_sha256"] = _json_digest(contract)
    return contract


def validate_calibration_plan(
    project: FluidFlowerProject,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one execution-free calibration plan."""

    candidate = deepcopy(dict(config))
    issues: list[dict[str, Any]] = []
    native_contract = dict(candidate.get("native_fixed_phase_contract") or {})
    native_round2 = native_contract.get("contract") == "native-round2-fixed-phase-v1"
    native_fixed = native_round2 or native_contract.get("contract") == "native-darsia-fixed-phase-mass-v1"
    if native_contract and not native_fixed:
        _issue(issues, "error", "invalid_native_fixed_contract", "Unknown native fixed-phase calibration contract.")
    if native_round2:
        from .phase_saved_review_metadata import configured_parameters
        cohort_parameters = configured_parameters("calibration_cohorts")
        CALIBRATION = frozenset(cohort_parameters["CALIBRATION"])
        HELDOUT = frozenset(cohort_parameters["HELDOUT"])
        EXCLUDED = frozenset(cohort_parameters["EXCLUDED"])
        DESCRIPTIVE = frozenset(cohort_parameters["DESCRIPTIVE"])
        if (set(native_contract.get("calibration_runs") or []) != CALIBRATION
                or set(native_contract.get("heldout_runs") or []) != HELDOUT
                or set(native_contract.get("excluded_runs") or []) != EXCLUDED
                or set(native_contract.get("descriptive_runs") or []) != DESCRIPTIVE):
            _issue(issues, "error", "invalid_native_round2_cohort", "Round 2 requires the exact 26/12/AC38/4 split.")
        parameters = dict(native_contract.get("candidate_parameters") or {})
        required = {"N1": ["gain", "beta"], "N2": ["gain", "beta"], "J": ["endpoint", "amplitude", "gamma", "beta"]}
        if any(parameters.get(key) != value for key, value in required.items()) or set(parameters) - {"N1", "N2", "N3", "J"}:
            _issue(issues, "error", "invalid_native_round2_parameters", "The round-2 shared candidates must have only their declared aqueous parameters and beta.")
        if "N3" in parameters and parameters["N3"] != ["blend", "gain", "beta"]:
            _issue(issues, "error", "invalid_native_round2_blend", "The optional native blend has only blend, gain and beta.")
        if native_contract.get("parameter_scope") != "shared_across_series" or native_contract.get("gas_rule") != "fixed 1-Swi; remaining water at local solubility; neither diluted nor fitted":
            _issue(issues, "error", "invalid_native_round2_gas", "All round-2 candidates use the same source-bound physical gas rule without any gas gain.")
        stages = dict(native_contract.get("stages") or {})
        if stages != {"injection_h": [.497, 2.247], "post_injection_h": [2.747, 24.], "stage2_fitted": ["beta"], "exclude_quarter_hour": True}:
            _issue(issues, "error", "invalid_native_round2_stages", "The two pre-registered fit stages may not drift.")
        if native_contract.get("selection_rule") != "round2-spec-section3" or native_contract.get("freeze_before_heldout") is not True or native_contract.get("heldout_final_check_before_production") is not True:
            _issue(issues, "error", "invalid_native_round2_selection", "Round 2 must select on calibration-run LORO and freeze before held-out pixels.")
        if native_contract.get("capacity_policy") != "finite two-sided dose interval; measured max depth/porosity only bound missing geometry; fit U/d<=0.05":
            _issue(issues, "error", "invalid_native_round2_capacity", "Missing physical support is only a finite capacity bound; never main mass.")
        for key in ("locked_phase_binding", "input_plan", "round1_freeze", "swi_binding", "physical_limits_binding"):
            binding = dict(native_contract.get(key) or {})
            source = Path(str(binding.get("path") or ""))
            if len(str(binding.get("sha256") or "")) != 64 or not source.is_file() or sha256_file(source) != binding["sha256"]:
                _issue(issues, "error", "native_round2_source_binding", f"Round-2 source binding is missing or changed: {key}")
        if native_contract.get("map_rescaling_allowed") is not False or native_contract.get("overflow_fit_weight") != 0:
            _issue(issues, "error", "native_rescaling_or_overflow_fit", "Round 2 forbids map-to-dose rescaling and overflow fitting.")
    elif native_fixed:
        fit_runs = set(native_contract.get("calibration_runs") or [])
        heldout_runs = set(native_contract.get("heldout_runs") or [])
        excluded_runs = set(native_contract.get("excluded_runs") or [])
        if (len(fit_runs) != 27 or len(heldout_runs) != 12 or len(excluded_runs) != 4
                or fit_runs & heldout_runs or (fit_runs | heldout_runs) & excluded_runs):
            _issue(issues, "error", "invalid_native_cohort", "The pre-registered native FF_AC split requires 27 calibration, 12 hold-out and 4 excluded runs without overlap.")
        expected_scales = {"aqueous_scale", "gas_scale"}
        scales = list(native_contract.get("shared_response_parameters") or [])
        if {row.get("key") for row in scales} != expected_scales or len(scales) != 2:
            _issue(issues, "error", "invalid_native_shared_pair", "Native calibration requires one aqueous/gas response-scale pair.")
        for row in scales:
            low, high = _number(row.get("lower")), _number(row.get("upper"))
            positive_log_domain = (row.get("domain") == "positive"
                and row.get("parameterization") == "log"
                and low == 0 and row.get("upper") is None)
            finite_positive_bounds = low is not None and high is not None and 0 < low < high
            if row.get("scope") != "shared_across_series" or not (positive_log_domain or finite_positive_bounds):
                _issue(issues, "error", "invalid_native_shared_scope", "Every native response scale needs a positive domain and shared_across_series scope.")
        binding = dict(native_contract.get("locked_phase_binding") or {})
        if not binding.get("path") or len(str(binding.get("sha256") or "")) != 64:
            _issue(issues, "error", "native_phase_binding_missing", "A SHA-bound locked categorical phase manifest is required.")
        elif not Path(str(binding["path"])).is_file() or sha256_file(Path(str(binding["path"]))) != binding["sha256"]:
            _issue(issues, "error", "native_phase_binding_changed", "The locked categorical phase manifest failed its source hash.")
        if native_contract.get("map_rescaling_allowed") is not False or native_contract.get("overflow_fit_weight") != 0:
            _issue(issues, "error", "native_rescaling_or_overflow_fit", "Native shared calibration forbids map-to-dose rescaling and overflow fitting.")
    if candidate.get("format") != CALIBRATION_PLAN_FORMAT:
        _issue(issues, "error", "unsupported_format", "Unsupported calibration plan")
    if bool(candidate.get("execution_requested")):
        _issue(
            issues,
            "error",
            "execution_not_supported",
            "Saving a calibration plan cannot start optimization.",
        )

    phase_revision = PhaseInterpretationRevisionStore(project).current()
    phase_id = str(candidate.get("phase_revision_id") or "")
    phase_sha256 = str(candidate.get("phase_revision_sha256") or "")
    if phase_revision.get("format") != PHASE_INTERPRETATION_REVISION_FORMAT:
        _issue(
            issues,
            "error",
            "phase_revision_missing",
            "Save a phase-interpretation revision before calibration planning.",
        )
    else:
        current_id = str(phase_revision.get("revision_id") or "")
        current_sha256 = str(phase_revision.get("content_sha256") or "")
        if phase_id != current_id or phase_sha256 != current_sha256:
            _issue(
                issues,
                "error",
                "phase_revision_changed",
                "The plan does not reference the current phase interpretation.",
                expected_revision_id=current_id,
            )
        phase_summary = dict(
            (phase_revision.get("validation") or {}).get("summary") or {}
        )
        if int(phase_summary.get("selected_run_count") or 0) < int(
            phase_summary.get("run_count") or 0
        ):
            _issue(
                issues,
                "error",
                "phase_selection_incomplete",
                "Every phase-campaign run requires a selected candidate.",
            )
        if int(phase_summary.get("approved_run_count") or 0) < int(
            phase_summary.get("run_count") or 0
        ):
            _issue(
                issues,
                "error",
                "phase_2d_approval_incomplete",
                "Every run requires an approved visible-2D phase interpretation "
                "before mass calibration can be planned.",
            )

    chemistry = dict(candidate.get("chemistry") or {})
    model = str(chemistry.get("model") or "")
    if model not in CHEMISTRY_MODELS:
        _issue(issues, "error", "invalid_chemistry_model", "Unknown chemistry model")
    ph_bounds = list(chemistry.get("initial_ph_bounds") or [])
    if len(ph_bounds) != 2:
        _issue(
            issues, "error", "invalid_ph_bounds", "Provide lower and upper pH bounds"
        )
    else:
        low, high = (_number(ph_bounds[0]), _number(ph_bounds[1]))
        if low is None or high is None or not (0.0 <= low < high <= 14.0):
            _issue(
                issues,
                "error",
                "invalid_ph_bounds",
                "pH bounds must satisfy 0 <= lower < upper <= 14.",
            )
    temperature = _number(chemistry.get("temperature_c"))
    if temperature is None or not (-20.0 <= temperature <= 100.0):
        _issue(
            issues,
            "error",
            "invalid_temperature",
            "Temperature is outside the supported range",
        )
    pressure = _number(chemistry.get("pressure_bar"))
    if pressure is None or pressure <= 0.0:
        _issue(issues, "error", "invalid_pressure", "Pressure must be positive")

    response_reference = chemistry.get("indicator_response_model")
    response_inspection: dict[str, Any] = {}
    if isinstance(response_reference, Mapping) and response_reference:
        response_inspection = inspect_indicator_response_model_reference(
            response_reference
        )
        if not response_inspection.get("valid"):
            _issue(
                issues,
                "error",
                "invalid_indicator_response_model",
                "The measured indicator-response model is unavailable or not deployable.",
                model_issues=list(response_inspection.get("issues") or []),
            )
        else:
            if model == "simple_flash":
                _issue(
                    issues,
                    "error",
                    "indicator_response_requires_titration",
                    "Measured pH response requires the Titration chemistry model.",
                )
            expected_indicator = str(chemistry.get("indicator_system") or "").casefold()
            model_indicator = str(
                response_inspection.get("indicator_system") or ""
            ).casefold()
            if expected_indicator and model_indicator and expected_indicator != model_indicator:
                _issue(
                    issues,
                    "error",
                    "indicator_response_system_mismatch",
                    "The fitted response belongs to a different indicator system.",
                    expected=expected_indicator,
                    observed=model_indicator,
                )
            if response_inspection.get("status") == "ready_limited_holdout":
                _issue(
                    issues,
                    "warning",
                    "indicator_response_limited_holdout",
                    "The measured response is fitted, but leave-one-image-out validation "
                    "was not possible with the available pH images.",
                )

    parameter_space_missing = not isinstance(
        candidate.get("parameter_space"), Mapping
    )
    parameter_space = _canonical_parameter_space(
        candidate.get("parameter_space"),
        initial_ph_bounds=ph_bounds if len(ph_bounds) == 2 else [6.0, 10.0],
    )
    if parameter_space_missing:
        _issue(
            issues,
            "warning",
            "legacy_parameter_space_defaulted",
            "The plan predates the explicit parameter schema; review all bounds.",
        )
    raw_parameter_rows = (
        list((candidate.get("parameter_space") or {}).get("parameters") or [])
        if isinstance(candidate.get("parameter_space"), Mapping)
        else []
    )
    known_parameter_keys = {str(row["key"]) for row in PARAMETER_DEFINITIONS}
    raw_parameter_keys = {
        str(row.get("key") or "")
        for row in raw_parameter_rows
        if isinstance(row, Mapping)
    }
    if raw_parameter_rows and raw_parameter_keys != known_parameter_keys:
        _issue(
            issues,
            "error",
            "parameter_contract_incomplete",
            "Every defined calibration parameter must be represented explicitly.",
            missing=sorted(known_parameter_keys - raw_parameter_keys),
            unknown=sorted(raw_parameter_keys - known_parameter_keys),
        )
    active_parameters: list[dict[str, Any]] = []
    for row in parameter_space["parameters"]:
        key = str(row["key"])
        lower = _number(row.get("lower"))
        upper = _number(row.get("upper"))
        if lower is None or upper is None or lower >= upper:
            _issue(
                issues,
                "error",
                "invalid_parameter_bounds",
                f"Calibration parameter {key} requires finite lower < upper bounds.",
            )
        if str(row.get("scale") or "") not in PARAMETER_SCALES:
            _issue(
                issues,
                "error",
                "invalid_parameter_scale",
                f"Calibration parameter {key} has an unsupported scale.",
            )
        if str(row.get("scope") or "") not in PARAMETER_SCOPES:
            _issue(
                issues,
                "error",
                "invalid_parameter_scope",
                f"Calibration parameter {key} has an unsupported sharing scope.",
            )
        if str(row.get("scale") or "") == "log" and (
            lower is None or lower <= 0.0
        ):
            _issue(
                issues,
                "error",
                "invalid_log_parameter_bounds",
                f"Log-scaled calibration parameter {key} must be strictly positive.",
            )
        applies = model in set(row.get("models") or [])
        if bool(row.get("enabled")) and applies:
            active_parameters.append(row)
    if native_fixed and active_parameters:
        _issue(issues, "error", "native_extra_search_parameters", "The native fixed-label procedure fits only its shared response-scale pair.")
    if not active_parameters and not native_fixed:
        _issue(
            issues,
            "error",
            "no_active_parameters",
            "Enable at least one parameter applicable to the selected chemistry model.",
        )
    if response_inspection.get("valid"):
        active_by_key = {
            str(row["key"]): row for row in active_parameters
        }
        optical_drift = sorted(
            OPTICAL_PARAMETERS_REPLACED_BY_RESPONSE_MODEL & set(active_by_key)
        )
        if optical_drift:
            _issue(
                issues,
                "error",
                "measured_optical_response_is_searchable",
                "Optuna cannot relearn a measured indicator-color response.",
                parameters=optical_drift,
            )
        response_chemistry = dict(response_inspection.get("chemistry") or {})
        measured_bounds = dict(response_chemistry.get("measured_bounds") or {})
        for parameter_key, source_key in CHEMICAL_PARAMETER_KEYS.items():
            row = active_by_key.get(parameter_key)
            if row is None:
                continue
            allowed = list(measured_bounds.get(source_key) or [])
            if len(allowed) != 2:
                _issue(
                    issues,
                    "error",
                    "chemical_parameter_has_no_measured_bound",
                    f"{parameter_key} is searchable without a reviewed measured bound.",
                )
                continue
            allowed_low, allowed_high = (_number(allowed[0]), _number(allowed[1]))
            requested_low = _number(row.get("lower"))
            requested_high = _number(row.get("upper"))
            if (
                allowed_low is None
                or allowed_high is None
                or requested_low is None
                or requested_high is None
                or requested_low < allowed_low
                or requested_high > allowed_high
            ):
                _issue(
                    issues,
                    "error",
                    "chemical_parameter_exceeds_measured_bound",
                    f"{parameter_key} exceeds its reviewed measured uncertainty.",
                    allowed=allowed,
                    requested=[requested_low, requested_high],
                )
    initial_ph_parameter = next(
        (
            row
            for row in parameter_space["parameters"]
            if row["key"] == "initial_ph"
        ),
        {},
    )
    if len(ph_bounds) == 2 and all(_number(value) is not None for value in ph_bounds):
        if (
            _number(initial_ph_parameter.get("lower")) != _number(ph_bounds[0])
            or _number(initial_ph_parameter.get("upper")) != _number(ph_bounds[1])
        ):
            _issue(
                issues,
                "error",
                "ph_parameter_bounds_mismatch",
                "Initial-pH search bounds must match the chemistry contract.",
            )
    fixed_transfer = dict(candidate.get("fixed_transfer_contract") or {})
    transfer_shape = str(fixed_transfer.get("transfer_shape") or "")
    co2_onset = _number(fixed_transfer.get("co2_support_onset"))
    co2_high = _number(fixed_transfer.get("co2_support_high"))
    if native_fixed:
        if transfer_shape != "locked_categorical" or fixed_transfer.get("source") != "production_labels" or fixed_transfer.get("c_aq_zero_labels") != [0, 255] or fixed_transfer.get("s_g_positive_labels") != [2] or fixed_transfer.get("phase_probabilities_are_visible_2d_support") is not False:
            _issue(issues, "error", "invalid_locked_phase_support", "Native support must use locked production labels; probability transfer is disabled.")
    elif transfer_shape not in {"linear", "smoothstep"}:
        _issue(
            issues,
            "error",
            "invalid_fixed_transfer_shape",
            "The fixed CO2-support transfer must be linear or smoothstep.",
        )
    if not native_fixed and (
        co2_onset is None
        or co2_high is None
        or not 0.0 <= co2_onset < co2_high <= 1.0
    ):
        _issue(
            issues,
            "error",
            "invalid_fixed_co2_support",
            "Fixed CO2-support bounds must satisfy 0 <= onset < high <= 1.",
        )
    if not native_fixed and not bool(fixed_transfer.get("phase_probabilities_are_visible_2d_support")):
        _issue(
            issues,
            "error",
            "invalid_phase_support_scope",
            "Calibration phase probabilities must describe visible 2D support only.",
        )

    expected_geometry_policy = {
        "depth": "fixed_current_geometry_revision",
        "porosity": "fixed_current_geometry_revision",
        "two_d_to_three_d": "deferred_to_mass_analysis",
    }
    if dict(candidate.get("geometry_policy") or {}) != expected_geometry_policy:
        _issue(
            issues,
            "error",
            "invalid_geometry_policy",
            "Calibration must keep depth and porosity fixed and defer 3D conversion.",
        )
    if str(candidate.get("overflow_policy") or "") != ("compare_only" if native_fixed else "use_when_available"):
        _issue(
            issues,
            "error",
            "invalid_overflow_policy",
            "The supported overflow policy is use_when_available.",
        )

    objectives = _canonical_objectives(candidate.get("objectives"))
    if len(objectives) != len(OBJECTIVE_DEFINITIONS):
        _issue(
            issues,
            "error",
            "objective_contract_incomplete",
            "Every defined objective must be represented explicitly.",
        )
    active = [row for row in objectives if row["enabled"]]
    if not active:
        _issue(issues, "error", "no_active_objective", "Enable at least one objective")
    for row in objectives:
        if row["weight"] is None or float(row["weight"]) < 0.0:
            _issue(
                issues,
                "error",
                "invalid_objective_weight",
                f"Objective {row['key']} has an invalid weight.",
            )
        elif row["enabled"] and float(row["weight"]) == 0.0:
            _issue(
                issues,
                "warning",
                "zero_weight_objective",
                f"Enabled objective {row['key']} has zero influence.",
            )
    overflow_mode = str((project.manifest.get("overflow") or {}).get("mode") or "none")
    overflow_active = any(
        row["key"] == "overflow_phase_constraint" and row["enabled"]
        for row in objectives
    )
    if overflow_active and overflow_mode == "none":
        _issue(
            issues,
            "warning",
            "overflow_unavailable",
            "Overflow objective is enabled although this series has no overflow source.",
        )
    manual_review_active = any(
        row["key"] == "manual_phase_review" and row["enabled"] for row in objectives
    )
    manual_review_count = int(
        ((phase_revision.get("validation") or {}).get("summary") or {}).get(
            "manual_review_file_count"
        )
        or 0
    )
    if manual_review_active and manual_review_count == 0:
        _issue(
            issues,
            "warning",
            "manual_review_unavailable",
            "Manual-review objective is enabled without attached review files.",
        )

    evaluation = inspect_calibration_evaluation_manifest(
        candidate.get("evaluation_manifest_path")
    )
    unsupported_objectives: list[str] = []
    unsupported_parameters: list[str] = []
    if not evaluation.get("available"):
        _issue(
            issues,
            "info",
            "evaluation_manifest_missing",
            "Attach a prepared evaluation manifest before creating an Optuna task.",
        )
    elif not evaluation.get("valid"):
        _issue(
            issues,
            "error",
            "invalid_evaluation_manifest",
            "The prepared calibration evaluation manifest is invalid.",
            manifest_issues=list(evaluation.get("issues") or []),
        )
    else:
        expected_phase_sha256 = str(candidate.get("phase_revision_sha256") or "")
        evaluator_phase_sha256 = str(
            evaluation.get("phase_revision_sha256") or ""
        )
        if evaluator_phase_sha256 != expected_phase_sha256:
            _issue(
                issues,
                "error",
                "evaluation_phase_revision_mismatch",
                "The evaluation data was not prepared from this phase revision.",
            )
        supported_objectives = set(evaluation.get("supported_objectives") or [])
        unsupported_objectives = sorted(
            str(row["key"])
            for row in active
            if str(row["key"]) not in supported_objectives
        )
        supported_parameters = set(evaluation.get("supported_parameters") or [])
        unsupported_parameters = sorted(
            str(row["key"])
            for row in active_parameters
            if str(row["key"]) not in supported_parameters
        )
        if unsupported_objectives:
            _issue(
                issues,
                "error",
                "objective_not_evaluable",
                "The evaluator cannot calculate every enabled objective.",
                unsupported=unsupported_objectives,
            )
        if unsupported_parameters:
            _issue(
                issues,
                "error",
                "parameter_not_evaluable",
                "The evaluator cannot apply every enabled parameter.",
                unsupported=unsupported_parameters,
            )

    optimizer = dict(candidate.get("optimizer") or {})
    trials = optimizer.get("trial_budget_per_model")
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        _issue(
            issues,
            "error",
            "invalid_trial_budget",
            "Trial budget must be a positive integer",
        )
    if not bool(optimizer.get("selection_uses_identical_objective")):
        _issue(
            issues,
            "error",
            "selection_objective_mismatch",
            "Final selection must use the same fingerprinted objective as optimization.",
        )

    execution = dict(candidate.get("execution_profile") or {})
    primary_backend = str(execution.get("primary_backend") or "")
    if primary_backend not in NUMERICAL_BACKENDS:
        _issue(issues, "error", "invalid_backend", "Unknown numerical backend")
    if str(execution.get("reference_backend") or "") != "cpu-float64":
        _issue(
            issues,
            "error",
            "invalid_reference_backend",
            "CPU float64 is required as the authoritative parity reference.",
        )
    workers = execution.get("workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        _issue(issues, "error", "invalid_worker_count", "Worker count must be positive")
    for key in ("absolute_parity_tolerance", "relative_parity_tolerance"):
        value = _number(execution.get(key))
        if value is None or value <= 0.0:
            _issue(
                issues, "error", "invalid_parity_tolerance", f"{key} must be positive"
            )
    quality_report = validate_quality_profile(
        dict(candidate.get("quality_profile") or {}), backend=primary_backend
    )
    for row in quality_report["issues"]:
        _issue(
            issues,
            str(row.get("severity") or "error"),
            str(row.get("code") or "quality_profile"),
            str(row.get("message") or "Invalid quality profile"),
        )
    quality_contract = (
        summarize_quality_contract(
            quality_report["configuration"],
            backend=primary_backend,
        )
        if primary_backend in NUMERICAL_BACKENDS
        else {}
    )

    contract = calibration_objective_contract(candidate)
    errors = sum(row["severity"] == "error" for row in issues)
    warnings = sum(row["severity"] == "warning" for row in issues)
    execution_ready = bool(
        errors == 0
        and evaluation.get("available")
        and evaluation.get("valid")
        and not unsupported_objectives
        and not unsupported_parameters
    )
    return {
        "schema": 1,
        "config_sha256": _json_digest(candidate),
        "valid": errors == 0,
        "summary": {
            "error_count": errors,
            "warning_count": warnings,
            "active_objective_count": len(active),
            "active_parameter_count": len(active_parameters) + (2 if native_fixed else 0),
            "trial_budget_per_model": trials,
            "primary_backend": primary_backend,
            "numerical_dtype": (
                numerical_dtype_for_backend(primary_backend)
                if primary_backend in NUMERICAL_BACKENDS
                else ""
            ),
            "analysis_resolution_policy": str(
                (candidate.get("quality_profile") or {})
                .get("numerical_resolution", {})
                .get("policy", "")
            ),
            "scientific_grid": str(quality_contract.get("scientific_grid") or ""),
            "quality_label": str(quality_contract.get("label") or ""),
            "evaluation_run_count": int(evaluation.get("run_count") or 0),
            "execution_ready": execution_ready,
        },
        "objective_contract": contract,
        "quality_contract": quality_contract,
        "issues": issues,
    }


class CalibrationPlanStore:
    """Append-only calibration plans; execution is a later module sequence."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "calibration"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        from .series_paths import series_folder_path
        self.artifact_root = series_folder_path(project.root, "calibration_plans")

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
        # Recheck the current phase pointer at commit time. It can be replaced
        # by another process after a GUI preview validation.
        report = validate_calibration_plan(self.project, candidate)
        if int((report.get("summary") or {}).get("error_count") or 0) > 0:
            raise ValueError("Calibration plan has blocking validation errors")
        plan_id = uuid.uuid4().hex
        phase_revision = PhaseInterpretationRevisionStore(self.project).current()
        revision = {
            "schema": 1,
            "format": CALIBRATION_REVISION_FORMAT,
            "plan_id": plan_id,
            "recorded_utc": utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "configuration": candidate,
            "objective_contract": deepcopy(report["objective_contract"]),
            "selection_contract_sha256": report["objective_contract"]["content_sha256"],
            "upstream": {
                "phase_revision_id": phase_revision.get("revision_id"),
                "phase_revision_sha256": phase_revision.get("content_sha256"),
                "phase_upstream": deepcopy(phase_revision.get("upstream") or {}),
                "indicator_response_model": deepcopy(
                    (candidate.get("chemistry") or {}).get(
                        "indicator_response_model"
                    )
                    or {}
                ),
            },
            "validation": report,
            "execution": {
                "requested": False,
                "started": False,
                "study_id": None,
            },
            "scientific_scope": {
                "contains_parameter_estimates": False,
                "contains_mass_results": False,
                "contains_3d_extrapolation": False,
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
            "calibration_plan_saved",
            actor,
            {
                "plan_id": plan_id,
                "objective_contract_sha256": report["objective_contract"][
                    "content_sha256"
                ],
                "primary_backend": (candidate.get("execution_profile") or {}).get(
                    "primary_backend"
                ),
                "execution_started": False,
            },
        )
        return deepcopy(revision)


if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject
