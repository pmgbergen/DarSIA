"""Immutable numerical inputs for generic FluidFlower mass conversion."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from darsia.fluidflower.mass_artifacts import (
    MASS_INPUT_DYNAMIC_PRODUCTS,
    MASS_INPUT_STATIC_PRODUCTS,
    IndexedFieldProduct,
    PreviewProduct,
    load_field_index,
    load_preview_index,
)
from darsia.fluidflower.project import sha256_file

MASS_INPUT_MANIFEST_FORMAT = "darsia-fluidflower-mass-input-manifest-v1"
MASS_PARAMETER_ARTIFACT_FORMAT = "darsia-fluidflower-mass-parameters-v1"

MASS_INPUT_PRODUCTS = (
    *MASS_INPUT_DYNAMIC_PRODUCTS,
    *MASS_INPUT_STATIC_PRODUCTS,
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _resolved_path(raw: str, *, relative_to: Path | None = None) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return path.resolve()


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


def _strict_times(raw: Sequence[Any]) -> list[float] | None:
    values = [_number(value) for value in raw]
    if not values or any(value is None for value in values):
        return None
    result = [float(value) for value in values if value is not None]
    if result != sorted(set(result)):
        return None
    return result


def _finite_series(
    raw: Sequence[Any],
    count: int,
    *,
    nonnegative: bool = False,
) -> list[float] | None:
    values = [_number(value) for value in raw]
    if len(values) != count or any(value is None for value in values):
        return None
    result = [float(value) for value in values if value is not None]
    if nonnegative and any(value < 0.0 for value in result):
        return None
    return result


def _unit_key(value: str) -> str:
    return str(value).strip().casefold().replace(" ", "")


def _unit_valid(product: str, unit: str) -> bool:
    key = _unit_key(unit)
    if product in {
        "optical_acidification",
        "phase_co2_probability",
        "phase_free_gas_probability",
        "porosity",
    }:
        return key in {"1", "dimensionless"}
    if product == "depth":
        return key in {"m", "meter", "metre"}
    if product == "permeability":
        return bool(key)
    return False


def _upstream_identity(revision: Mapping[str, Any] | None) -> tuple[str, str]:
    row = dict(revision or {})
    return (
        str(row.get("revision_id") or ""),
        str(row.get("content_sha256") or ""),
    )


def inspect_mass_input_manifest(
    raw_path: str,
    *,
    run_bindings: Sequence[Mapping[str, Any]],
    maximum_images_per_run: int,
    phase_revision: Mapping[str, Any] | None = None,
    correction_revision: Mapping[str, Any] | None = None,
    geometry_revision: Mapping[str, Any] | None = None,
    require_preview_images: bool = False,
    allow_time_subset: bool = False,
) -> dict[str, Any]:
    """Validate one prepared, transitive mass-input manifest."""

    issues: list[dict[str, Any]] = []
    text = str(raw_path).strip()
    empty = {
        "available": False,
        "valid": False,
        "reference": {},
        "manifest": {},
        "shared_target_times_hours": [],
        "run_shapes": {},
        "issues": issues,
    }
    if not text:
        _issue(
            issues,
            "warning",
            "mass_input_manifest_missing",
            "No prepared mass-input manifest is attached; execution remains blocked.",
        )
        return empty

    path = _resolved_path(text)
    if not path.is_file():
        _issue(
            issues,
            "warning",
            "mass_input_manifest_unavailable",
            "The prepared mass-input manifest is currently unavailable.",
            path=str(path),
        )
        empty["reference"] = {"path": str(path)}
        return empty

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError) as exc:
        _issue(
            issues,
            "error",
            "mass_input_manifest_unreadable",
            f"Could not read mass inputs: {type(exc).__name__}: {exc}",
        )
        payload = {}
    if not isinstance(payload, dict):
        _issue(
            issues,
            "error",
            "mass_input_manifest_invalid",
            "Mass-input manifest must contain one JSON object.",
        )
        payload = {}
    if payload.get("format") != MASS_INPUT_MANIFEST_FORMAT:
        _issue(
            issues,
            "error",
            "mass_input_manifest_format",
            "Unsupported mass-input manifest format.",
        )

    upstream = dict(payload.get("upstream") or {})
    for prefix, revision in (
        ("phase", phase_revision),
        ("image_correction", correction_revision),
        ("geometry", geometry_revision),
    ):
        expected_id, expected_sha256 = _upstream_identity(revision)
        actual_id = str(upstream.get(f"{prefix}_revision_id") or "")
        actual_sha256 = str(upstream.get(f"{prefix}_revision_sha256") or "")
        if expected_id and actual_id != expected_id:
            _issue(
                issues,
                "error",
                f"{prefix}_revision_mismatch",
                f"Mass inputs do not use the current {prefix.replace('_', ' ')} revision.",
                expected_revision_id=expected_id,
                actual_revision_id=actual_id,
            )
        if expected_sha256 and actual_sha256 != expected_sha256:
            _issue(
                issues,
                "error",
                f"{prefix}_revision_hash_mismatch",
                f"Mass inputs do not match the current {prefix.replace('_', ' ')} hash.",
            )

    expected_bindings = {
        str(row.get("run_id") or ""): str(row.get("phase_candidate_id") or "")
        for row in run_bindings
    }
    raw_runs = payload.get("run_inputs") or []
    if not isinstance(raw_runs, list):
        _issue(
            issues,
            "error",
            "mass_input_runs_invalid",
            "Mass-input run_inputs must be a list.",
        )
        raw_runs = []
    run_rows = [dict(row) for row in raw_runs if isinstance(row, Mapping)]
    actual_ids = [str(row.get("run_id") or "") for row in run_rows]
    if len(actual_ids) != len(set(actual_ids)):
        _issue(
            issues,
            "error",
            "mass_input_runs_duplicated",
            "Mass-input manifest contains duplicate run identifiers.",
        )
    if sorted(actual_ids) != sorted(expected_bindings):
        _issue(
            issues,
            "error",
            "mass_input_run_mismatch",
            "Prepared mass inputs do not exactly match the selected phase runs.",
            expected=sorted(expected_bindings),
            actual=sorted(actual_ids),
        )

    product_artifacts: list[dict[str, Any]] = []
    preview_artifacts: list[dict[str, Any]] = []
    time_grids: list[list[float]] = []
    unavailable_artifact = False
    run_shapes: dict[str, list[int]] = {}
    for row in run_rows:
        run_id = str(row.get("run_id") or "")
        candidate_id = str(row.get("phase_candidate_id") or "")
        expected_candidate = expected_bindings.get(run_id)
        if expected_candidate is not None and candidate_id != expected_candidate:
            _issue(
                issues,
                "error",
                "mass_input_candidate_mismatch",
                f"Mass inputs for {run_id} use a different phase candidate.",
                expected=expected_candidate,
                actual=candidate_id,
            )
        times = _strict_times(row.get("target_times_hours") or [])
        if times is None:
            _issue(
                issues,
                "error",
                "mass_input_times_invalid",
                f"Target times for {run_id or 'unknown run'} are invalid.",
            )
            times = []
        else:
            time_grids.append(times)
            if len(times) > maximum_images_per_run and not allow_time_subset:
                _issue(
                    issues,
                    "error",
                    "mass_input_frame_limit_exceeded",
                    f"{run_id} contains more frames than the mass plan permits.",
                    frame_count=len(times),
                    maximum=maximum_images_per_run,
                )

        geometry = dict(row.get("geometry") or {})
        width_m = _number(geometry.get("width_m"))
        height_m = _number(geometry.get("height_m"))
        if width_m is None or width_m <= 0.0 or height_m is None or height_m <= 0.0:
            _issue(
                issues,
                "error",
                "mass_input_geometry_invalid",
                f"Physical rig dimensions are invalid for {run_id or 'unknown run'}.",
            )

        count = len(times)
        area_frames = geometry.get("pixel_area_frames")
        if area_frames is not None and (
                not isinstance(area_frames, list) or len(area_frames) != count
                or any(not isinstance(item, Mapping) or item.get("frame_index") != index
                       or not (item.get("pixel_area") or item.get("rig_domain_revision"))
                       for index, item in enumerate(area_frames))):
            _issue(issues, "error", "mass_input_pixel_area_binding_invalid",
                   f"{run_id}: per-frame physical pixel-area bindings are incomplete or reordered.")
        if _finite_series(
            row.get("injected_mass_g") or [], count, nonnegative=True
        ) is None:
            _issue(
                issues,
                "error",
                "mass_input_injected_mass_invalid",
                f"Injected mass for {run_id or 'unknown run'} must match target times.",
            )
        state = dict(row.get("thermodynamic_state") or {})
        for key, nonnegative in (
            ("pressure_bar", True),
            ("temperature_c", False),
            ("pressure_gradient_bar_per_m", False),
            ("temperature_gradient_c_per_m", False),
        ):
            values = _finite_series(state.get(key) or [], count)
            if values is None or (nonnegative and any(value <= 0.0 for value in values)):
                _issue(
                    issues,
                    "error",
                    f"mass_input_{key}_invalid",
                    f"{key} for {run_id or 'unknown run'} must match target times.",
                )

        products = row.get("products") or {}
        if not isinstance(products, Mapping):
            products = {}
        loaded_fields: dict[str, IndexedFieldProduct] = {}
        for product in MASS_INPUT_PRODUCTS:
            reference = products.get(product)
            if not isinstance(reference, Mapping):
                _issue(
                    issues,
                    "error",
                    "mass_input_product_missing",
                    f"{run_id or 'Unknown run'} lacks {product}.",
                )
                continue
            raw_product_path = str(reference.get("path") or "").strip()
            expected_sha256 = str(reference.get("sha256") or "")
            if not raw_product_path or not expected_sha256:
                _issue(
                    issues,
                    "error",
                    "mass_input_product_reference_invalid",
                    f"{run_id}/{product} has an incomplete index reference.",
                )
                continue
            product_path = _resolved_path(raw_product_path, relative_to=path.parent)
            if not product_path.is_file():
                unavailable_artifact = True
                _issue(
                    issues,
                    "warning",
                    "mass_input_product_unavailable",
                    f"{run_id}/{product} is currently unavailable.",
                    path=str(product_path),
                )
                continue
            try:
                field = load_field_index(
                    product_path,
                    expected_sha256,
                    run_id=run_id,
                    product=product,
                    expected_times_hours=(
                        times if product in MASS_INPUT_DYNAMIC_PRODUCTS else None
                    ),
                )
                if product in MASS_INPUT_STATIC_PRODUCTS and len(field.frames) != 1:
                    raise ValueError("static geometry inputs require exactly one frame")
                if not _unit_valid(product, field.unit):
                    raise ValueError(f"unsupported unit {field.unit!r}")
                loaded_fields[product] = field
                product_artifacts.append(
                    {
                        "run_id": run_id,
                        "product": product,
                        "path": str(product_path),
                        "size_bytes": product_path.stat().st_size,
                        "sha256": sha256_file(product_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                _issue(
                    issues,
                    "error",
                    "mass_input_product_invalid",
                    f"Could not validate {run_id}/{product}: {type(exc).__name__}: {exc}",
                )
        shapes = {field.shape_rc for field in loaded_fields.values()}
        if len(shapes) > 1:
            _issue(
                issues,
                "error",
                "mass_input_shape_mismatch",
                f"Prepared fields for {run_id} do not share one registered grid.",
                shapes=[list(shape) for shape in sorted(shapes)],
            )
        elif len(shapes) == 1:
            shape = next(iter(shapes))
            run_shapes[run_id] = [int(shape[0]), int(shape[1])]

        preview_reference = row.get("preview_images")
        if preview_reference is None and require_preview_images:
            _issue(
                issues,
                "error",
                "mass_input_preview_missing",
                f"Corrected preview images are required for {run_id}.",
            )
        elif isinstance(preview_reference, Mapping):
            raw_preview_path = str(preview_reference.get("path") or "").strip()
            expected_sha256 = str(preview_reference.get("sha256") or "")
            preview_path = _resolved_path(raw_preview_path, relative_to=path.parent)
            if not raw_preview_path or not expected_sha256:
                _issue(
                    issues,
                    "error",
                    "mass_input_preview_reference_invalid",
                    f"Corrected preview reference for {run_id} is incomplete.",
                )
            elif not preview_path.is_file():
                unavailable_artifact = True
                _issue(
                    issues,
                    "warning",
                    "mass_input_preview_unavailable",
                    f"Corrected preview images for {run_id} are unavailable.",
                    path=str(preview_path),
                )
            else:
                try:
                    load_preview_index(
                        preview_path,
                        expected_sha256,
                        run_id=run_id,
                        expected_times_hours=times,
                    )
                    preview_artifacts.append(
                        {
                            "run_id": run_id,
                            "product": "preview_images",
                            "path": str(preview_path),
                            "size_bytes": preview_path.stat().st_size,
                            "sha256": sha256_file(preview_path),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    _issue(
                        issues,
                        "error",
                        "mass_input_preview_invalid",
                        f"Could not validate {run_id} previews: {type(exc).__name__}: {exc}",
                    )

    shared_times: list[float] = []
    if time_grids:
        shared_times = time_grids[0]
        if any(times != shared_times for times in time_grids[1:]):
            _issue(
                issues,
                "error",
                "mass_input_time_grids_differ",
                "Every run must expose the same synchronized target-time grid.",
            )

    errors = sum(issue["severity"] == "error" for issue in issues)
    return {
        "available": True,
        "valid": errors == 0 and not unavailable_artifact,
        "reference": {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "format": MASS_INPUT_MANIFEST_FORMAT,
            "product_artifacts": product_artifacts,
            "preview_artifacts": preview_artifacts,
        },
        "manifest": payload,
        "shared_target_times_hours": shared_times,
        "run_shapes": run_shapes,
        "issues": issues,
    }


def _resolved_parameter_row(
    shared: Mapping[str, Any],
    specific: Mapping[str, Any],
) -> dict[str, Any]:
    output = deepcopy(dict(shared))
    output.update(deepcopy(dict(specific)))
    return output


def validate_mass_parameter_artifact(
    payload: Mapping[str, Any],
    *,
    run_ids: Sequence[str],
    selected_model: str,
) -> dict[str, Any]:
    """Resolve and validate the calibrated optical-to-physical transfer."""

    candidate = deepcopy(dict(payload))
    issues: list[dict[str, Any]] = []
    if candidate.get("native_contract") == "native-round2-fixed-phase-v1":
        from darsia.fluidflower.mass_saved_metadata import FrozenEvidence
        artifact_model=str(candidate.get("selected_model") or "")
        if candidate.get("format")!=MASS_PARAMETER_ARTIFACT_FORMAT or artifact_model!=selected_model:
            _issue(issues,"error","native_round2_parameter_model","The saved round-2 response must match the selected mass model.")
        if candidate.get("parameter_scope")!="shared_across_series" or candidate.get("run_parameters"):
            _issue(issues,"error","native_round2_parameter_scope","Round 2 has only the frozen common aqueous response and dilution, with no per-run fit.")
        if candidate.get("phase_support_source")!="production_labels" or candidate.get("map_mass_rescaling") is not False:
            _issue(issues,"error","native_round2_parameter_support","Round 2 requires unchanged production labels and no map rescaling.")
        resolved={};production_allowed=False
        try:
            descriptor=dict(candidate.get("native_frozen_calibration") or {})
            freeze=FrozenEvidence.verify(descriptor.get("path",""))
            if descriptor.get("sha256")!=freeze.sha256:
                raise ValueError("The saved round-2 freeze SHA differs")
            parameters=dict(candidate.get("shared_parameters") or {})
            if parameters!=freeze.record.get("parameters"):
                raise ValueError("Shared parameters differ from the frozen round-2 response")
            expected_model={"N1":"titration","N2":"simple_flash","N3":"titration"}.get(parameters.get("candidate"))
            if artifact_model!=expected_model:
                raise ValueError("Native round-2 response and selected workflow model differ")
            for run in run_ids:
                freeze.check_frame_run(run)
            resolved={run:dict(parameters) for run in run_ids}
            final=dict(candidate.get("final_check") or {})
            if final:
                from pathlib import Path
                import json
                if sha256_file(final["path"])!=final.get("sha256"):
                    raise ValueError("The frozen final-check report changed")
                check=json.loads(Path(final["path"]).read_text(encoding="utf-8-sig"))
                if check.get("freeze",{}).get("sha256")!=freeze.sha256:
                    raise ValueError("Final check refers to a different freeze")
                production_allowed=check.get("full_production_allowed") is True
        except (OSError,ValueError,TypeError,KeyError) as exc:
            _issue(issues,"error","native_round2_parameter_freeze",f"Round-2 freeze validation failed: {exc}")
        return dict(valid=not any(row["severity"]=="error" for row in issues),selected_model=artifact_model,
            resolved_parameters=resolved,issues=issues,native_contract="native-round2-fixed-phase-v1",
            requires_native_fixed_executor=True,legacy_probability_executor_allowed=False,
            full_production_allowed=production_allowed)
    if candidate.get("native_contract") == "native-darsia-fixed-phase-mass-v1":
        from darsia.fluidflower.mass_saved_metadata import verify_frozen_calibration, EXCLUDED_RUNS
        artifact_model = str(candidate.get("selected_model") or "")
        if candidate.get("format") != MASS_PARAMETER_ARTIFACT_FORMAT or artifact_model != selected_model or selected_model != "titration":
            _issue(issues, "error", "native_parameter_model", "The common native main result requires a titration mass-parameter artifact.")
        if candidate.get("parameter_scope") != "shared_across_series" or candidate.get("run_parameters"):
            _issue(issues, "error", "native_parameter_scope", "Native parameters must be one shared series pair, with no fitted per-run overrides.")
        if candidate.get("phase_support_source") != "production_labels" or candidate.get("map_mass_rescaling") is not False:
            _issue(issues, "error", "native_parameter_support", "The native artifact must use locked labels and forbid injected-mass map rescaling.")
        freeze = dict(candidate.get("native_frozen_calibration") or {})
        resolved = {}
        try:
            digest = str(freeze.get("sha256") or "")
            if len(digest) != 64:
                raise ValueError("A frozen-calibration SHA-256 is required")
            frozen, _ = verify_frozen_calibration(freeze.get("path", ""), expected_sha256=digest)
            pair = dict(candidate.get("shared_parameters") or {})
            if pair != frozen["parameters"]:
                raise ValueError("The shared response scales differ from the frozen native pair")
            allowed = set(frozen["calibration_runs"]) | set(frozen["heldout_runs"])
            if any(run not in allowed or run in EXCLUDED_RUNS for run in run_ids):
                raise ValueError("Review includes a run outside the pre-registered native cohort")
            resolved = {run: dict(pair) for run in run_ids}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            _issue(issues, "error", "native_parameter_freeze", f"Native freeze validation failed: {exc}")
        return dict(valid=not any(row["severity"] == "error" for row in issues),
            selected_model=artifact_model, resolved_parameters=resolved, issues=issues,
            native_contract="native-darsia-fixed-phase-mass-v1",
            requires_native_fixed_executor=True, legacy_probability_executor_allowed=False)
    if candidate.get("format") != MASS_PARAMETER_ARTIFACT_FORMAT:
        _issue(
            issues,
            "error",
            "mass_parameter_format",
            "Unsupported mass-parameter artifact format.",
        )
    artifact_model = str(
        candidate.get("selected_model") or candidate.get("model") or ""
    )
    if artifact_model != selected_model or artifact_model not in {
        "simple_flash",
        "titration",
    }:
        _issue(
            issues,
            "error",
            "mass_parameter_model_mismatch",
            "Mass parameters do not match the selected calibration model.",
        )

    shared = dict(candidate.get("shared_parameters") or {})
    per_run = candidate.get("run_parameters") or {}
    if not isinstance(per_run, Mapping):
        _issue(
            issues,
            "error",
            "mass_parameter_runs_invalid",
            "run_parameters must be an object keyed by run ID.",
        )
        per_run = {}
    expected = [str(run_id) for run_id in run_ids]
    unknown = sorted({str(key) for key in per_run} - set(expected))
    if unknown:
        _issue(
            issues,
            "warning",
            "mass_parameter_unknown_runs",
            "The parameter artifact contains runs outside this mass plan.",
            runs=unknown,
        )

    resolved: dict[str, dict[str, Any]] = {}
    for run_id in expected:
        specific = per_run.get(run_id) or {}
        if not isinstance(specific, Mapping):
            _issue(
                issues,
                "error",
                "mass_parameter_run_invalid",
                f"Parameters for {run_id} must be an object.",
            )
            continue
        row = _resolved_parameter_row(shared, specific)
        transfer_shape = str(row.get("transfer_shape") or "")
        if transfer_shape not in {"linear", "smoothstep"}:
            _issue(
                issues,
                "error",
                "mass_parameter_transfer_shape",
                f"{run_id} must select linear or smoothstep transfer.",
            )
        numeric: dict[str, float] = {}
        for key in (
            "acid_low",
            "acid_high",
            "co2_onset",
            "co2_high",
            "gas_onset",
            "gas_high",
            "max_gas_saturation",
        ):
            value = _number(row.get(key))
            if value is None:
                _issue(
                    issues,
                    "error",
                    "mass_parameter_missing",
                    f"{run_id} lacks a finite {key} parameter.",
                )
            else:
                numeric[key] = value
        for low, high, label in (
            ("acid_low", "acid_high", "acidification"),
            ("co2_onset", "co2_high", "CO2 support"),
            ("gas_onset", "gas_high", "gas saturation"),
        ):
            if low in numeric and high in numeric and numeric[high] <= numeric[low]:
                _issue(
                    issues,
                    "error",
                    "mass_parameter_interval_invalid",
                    f"{run_id} has an invalid {label} interval.",
                )
        saturation = numeric.get("max_gas_saturation")
        if saturation is not None and not 0.0 <= saturation <= 1.0:
            _issue(
                issues,
                "error",
                "mass_parameter_saturation_invalid",
                f"{run_id} maximum gas saturation must be in [0, 1].",
            )
        beta = _number(row.get("diffusion_beta", 0.0))
        if beta is None or beta < 0.0:
            _issue(
                issues,
                "error",
                "mass_parameter_diffusion_invalid",
                f"{run_id} diffusion_beta must be finite and non-negative.",
            )
        else:
            row["diffusion_beta"] = beta
        if artifact_model == "titration":
            initial_ph = _number(row.get("initial_pH"))
            alkalinity = _number(row.get("alkalinity_M"))
            if initial_ph is None and alkalinity is None:
                _issue(
                    issues,
                    "error",
                    "mass_parameter_titration_state_missing",
                    f"{run_id} needs initial_pH or alkalinity_M for titration.",
                )
            if initial_ph is not None and not 0.5 < initial_ph < 13.5:
                _issue(
                    issues,
                    "error",
                    "mass_parameter_ph_invalid",
                    f"{run_id} initial_pH must lie between 0.5 and 13.5.",
                )
            blend = _number(row.get("aqueous_blend", 0.0))
            if blend is None or not 0.0 <= blend <= 1.0:
                _issue(
                    issues,
                    "error",
                    "mass_parameter_blend_invalid",
                    f"{run_id} aqueous_blend must be in [0, 1].",
                )
            else:
                row["aqueous_blend"] = blend
        resolved[run_id] = row

    errors = sum(issue["severity"] == "error" for issue in issues)
    return {
        "valid": errors == 0,
        "selected_model": artifact_model,
        "resolved_parameters": resolved,
        "issues": issues,
    }


def load_mass_input_products(
    manifest_path: Path,
    row: Mapping[str, Any],
) -> tuple[dict[str, IndexedFieldProduct], PreviewProduct | None]:
    """Load hash-verified field indexes for one mass-input run."""

    run_id = str(row.get("run_id") or "")
    times = [float(value) for value in row.get("target_times_hours") or []]
    products = dict(row.get("products") or {})
    fields: dict[str, IndexedFieldProduct] = {}
    for product in MASS_INPUT_PRODUCTS:
        reference = dict(products.get(product) or {})
        path = _resolved_path(
            str(reference.get("path") or ""), relative_to=manifest_path.parent
        )
        fields[product] = load_field_index(
            path,
            str(reference.get("sha256") or ""),
            run_id=run_id,
            product=product,
            expected_times_hours=(
                times if product in MASS_INPUT_DYNAMIC_PRODUCTS else None
            ),
        )
    preview: PreviewProduct | None = None
    preview_reference = row.get("preview_images")
    if isinstance(preview_reference, Mapping):
        path = _resolved_path(
            str(preview_reference.get("path") or ""),
            relative_to=manifest_path.parent,
        )
        preview = load_preview_index(
            path,
            str(preview_reference.get("sha256") or ""),
            run_id=run_id,
            expected_times_hours=times,
        )
    return fields, preview


__all__ = [
    "MASS_INPUT_MANIFEST_FORMAT",
    "MASS_INPUT_PRODUCTS",
    "MASS_PARAMETER_ARTIFACT_FORMAT",
    "inspect_mass_input_manifest",
    "load_mass_input_products",
    "validate_mass_parameter_artifact",
]
