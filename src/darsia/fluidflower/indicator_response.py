"""Measured indicator-color response models for FluidFlower workflows.

The model in this module is deliberately separate from radiometric correction.
Known-pH images are first put on the approved geometry and image-correction
contracts. Their colors are then used as chemical observations, never as light
targets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from darsia.fluidflower.geometry_facies import (
    GeometryRevisionStore,
    load_registered_geometry_image,
)
from darsia.fluidflower.image_correction import (
    COLORCHECKER_ALGORITHMS,
    DARSIA_BASELINE_COLOR_ALGORITHM,
    IDENTITY_ALGORITHM,
    UNIFORM24_ALGORITHM,
    CorrectionRevisionStore,
    apply_darsia_baseline_array_correction,
    apply_uniform24_array_transform,
    fit_uniform24_array_transform,
    srgb_to_linear,
)
from darsia.fluidflower.indicator_references import (
    INDICATOR_REFERENCE_FORMAT,
    REFERENCE_KIND_PH_BUFFER,
)
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.multiphase.flash import carbonate_dic_from_ph

if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject


INDICATOR_RESPONSE_CONFIG_FORMAT = "darsia-fluidflower-indicator-response-config-v1"
INDICATOR_RESPONSE_MODEL_FORMAT = "darsia-fluidflower-indicator-response-model-v1"
INDICATOR_RESPONSE_REVISION_FORMAT = (
    "darsia-fluidflower-indicator-response-revision-v1"
)
INDICATOR_RESPONSE_REFERENCE_FORMAT = (
    "darsia-fluidflower-indicator-response-model-reference-v1"
)
FEATURE_NAMES = (
    "linear_r",
    "linear_g",
    "linear_b",
    "lab_l",
    "lab_a",
    "lab_b",
    "chromaticity_r",
    "chromaticity_g",
    "chromaticity_b",
    "log_r_over_g",
    "log_b_over_g",
    "log_r_over_b",
)
OPTICAL_PARAMETERS_REPLACED_BY_RESPONSE_MODEL = {
    "aqueous_transition_low_fraction",
    "aqueous_transition_width_fraction",
    "aqueous_titration_blend",
}
CHEMICAL_PARAMETER_KEYS = {
    "initial_ph": "initial_ph",
    "alkalinity_M": "alkalinity_M",
    "indicator_concentration_M": "btb_M",
    "co2_saturation_M": "co2_saturation_M",
}


def _number(value: Any) -> float | None:
    from darsia.fluidflower.mass_inputs import _number as _canonical_finite_number
    return _canonical_finite_number(value)


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
                    f"Invalid indicator-response revision at line {line_number}"
                )
            rows.append(payload)
    return rows


def _json_digest(payload: Any) -> str:
    from .analysis_domains import _json_digest as shared_json_digest
    return shared_json_digest(payload)


def _load_field(path: Path) -> np.ndarray:
    selected = Path(path).expanduser().resolve()
    if selected.suffix.casefold() == ".npy":
        return np.asarray(np.load(selected, allow_pickle=False))
    if selected.suffix.casefold() == ".npz":
        with np.load(selected, allow_pickle=False) as archive:
            key = "array" if "array" in archive.files else archive.files[0]
            return np.asarray(archive[key])
    raise ValueError(f"Unsupported geometry field: {selected}")


def _default_bound(value: float) -> list[float]:
    return [float(value), float(value)]


def default_indicator_response_config(
    project: FluidFlowerProject,
) -> dict[str, Any]:
    """Return a reviewable fitting contract without claiming defaults are measured."""

    chemistry = dict(project.manifest.get("chemistry") or {})
    references = deepcopy(dict(chemistry.get("indicator_response_reference") or {}))
    initial_ph = _number(chemistry.get("initial_ph"))
    alkalinity = _number(chemistry.get("alkalinity_M"))
    btb = _number(chemistry.get("btb_M"))
    saturation = _number(chemistry.get("co2_saturation_M"))
    alkalinity = 1.25e-3 if alkalinity is None else alkalinity
    btb = 0.726e-3 if btb is None else btb
    saturation = 0.034 if saturation is None else saturation
    ph_bounds = (
        [max(0.5, initial_ph - 0.25), min(13.5, initial_ph + 0.25)]
        if initial_ph is not None
        else [6.0, 10.0]
    )
    source = (
        "project_manifest"
        if all(
            _number(chemistry.get(key)) is not None
            for key in ("alkalinity_M", "btb_M", "co2_saturation_M")
        )
        else "unreviewed_titration_defaults"
    )
    return {
        "schema": 1,
        "format": INDICATOR_RESPONSE_CONFIG_FORMAT,
        "indicator_system": str(
            chemistry.get("indicator_system") or "unspecified"
        ),
        "reference_contract": references,
        "sampling": {
            "per_facies": True,
            "max_pixels_per_image_and_facies": 30_000,
            "erode_facies_boundary_px": 2,
            "random_seed": 20_260_903,
        },
        "chemistry": {
            "adapter": "btb_carbonate_charge_balance_v1",
            "alkalinity_M": float(alkalinity),
            "btb_M": float(btb),
            "pKa1": 6.35,
            "pKa2": 10.33,
            "pKaB": 7.10,
            "co2_saturation_M": float(saturation),
            "initial_ph_bounds": ph_bounds,
            "measured_bounds": {
                "initial_ph": ph_bounds,
                "alkalinity_M": _default_bound(alkalinity),
                "btb_M": _default_bound(btb),
                "co2_saturation_M": _default_bound(saturation),
            },
            "bounds_reviewed": False,
            "parameter_source": source,
        },
        "holdout": {
            "strategy": "leave_one_reference_image_out",
            "minimum_training_ph_levels": 2,
            "interval_percentile": 90.0,
        },
    }


def validate_indicator_response_config(config: Mapping[str, Any]) -> list[str]:
    """Return blocking configuration issues for response fitting."""

    issues: list[str] = []
    if config.get("format") != INDICATOR_RESPONSE_CONFIG_FORMAT:
        issues.append("Unsupported indicator-response configuration format.")
    reference = dict(config.get("reference_contract") or {})
    if reference.get("format") != INDICATOR_REFERENCE_FORMAT:
        issues.append("The indicator reference contract is missing or unsupported.")
    if str(reference.get("status") or "") != "ready":
        issues.append(
            "At least two distinct, measured pH-buffer levels are required."
        )
    chemistry = dict(config.get("chemistry") or {})
    for key in (
        "alkalinity_M",
        "btb_M",
        "pKa1",
        "pKa2",
        "pKaB",
        "co2_saturation_M",
    ):
        value = _number(chemistry.get(key))
        if value is None or (key.endswith("_M") and value < 0.0):
            issues.append(f"Chemistry value {key} must be finite and non-negative.")
    bounds = dict(chemistry.get("measured_bounds") or {})
    for key in ("initial_ph", "alkalinity_M", "btb_M", "co2_saturation_M"):
        values = list(bounds.get(key) or [])
        if len(values) != 2:
            issues.append(f"Measured bounds are missing for {key}.")
            continue
        low, high = (_number(values[0]), _number(values[1]))
        if low is None or high is None or low > high:
            issues.append(f"Measured bounds for {key} require lower <= upper.")
        if key == "initial_ph" and (
            low is None or high is None or low <= 0.5 or high >= 13.5
        ):
            issues.append("Initial-pH bounds must lie strictly between 0.5 and 13.5.")
    sampling = dict(config.get("sampling") or {})
    if int(sampling.get("max_pixels_per_image_and_facies") or 0) < 100:
        issues.append("At least 100 pixels per image and facies are required.")
    return issues


def _geometry_masks(
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    labels = np.squeeze(_load_field(Path(str(config.get("labels_path") or ""))))
    if labels.ndim != 2:
        raise ValueError("The approved geometry labels must be a two-dimensional field")
    grouping_mode = str(config.get("grouping_mode") or "shared_facies")
    facies = np.full(labels.shape, -1, dtype=np.int32)
    active = np.zeros(labels.shape, dtype=bool)
    names: dict[str, str] = {"global": "All active porous regions"}
    for raw in config.get("regions") or []:
        row = dict(raw)
        if not bool(row.get("enabled", True)):
            continue
        if str(row.get("analysis_role") or "") not in {"porous_medium", "seal"}:
            continue
        label_id = int(row["label_id"])
        facies_id = (
            label_id if grouping_mode == "per_layer" else int(row["facies_id"])
        )
        mask = labels == label_id
        facies[mask] = facies_id
        active[mask] = True
        names.setdefault(str(facies_id), str(row.get("name") or f"Facies {facies_id}"))
    if not np.any(active):
        raise ValueError(
            "The geometry has no enabled porous-medium or seal regions for fitting"
        )
    return facies, active, names


def color_response_features(rgb: np.ndarray) -> np.ndarray:
    """Return full-color features without assuming a blue-yellow direction."""

    values = np.asarray(rgb, dtype=np.float32)
    if values.shape[-1] != 3:
        raise ValueError("Color-response features require RGB values")
    original_shape = values.shape[:-1]
    flat = np.clip(values.reshape(-1, 3), 0.0, 1.0)
    linear = srgb_to_linear(flat, dtype=np.float32)
    lab = cv2.cvtColor(flat.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab[:, 0] /= 100.0
    lab[:, 1:] /= 128.0
    total = np.maximum(np.sum(linear, axis=1, keepdims=True), 1.0e-6)
    chromaticity = linear / total
    epsilon = 1.0e-4
    ratios = np.column_stack(
        (
            np.log((linear[:, 0] + epsilon) / (linear[:, 1] + epsilon)),
            np.log((linear[:, 2] + epsilon) / (linear[:, 1] + epsilon)),
            np.log((linear[:, 0] + epsilon) / (linear[:, 2] + epsilon)),
        )
    )
    output = np.column_stack((linear, lab, chromaticity, ratios))
    return output.reshape(*original_shape, len(FEATURE_NAMES)).astype(np.float32)


def _sample_center(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    maximum: int,
    erosion_px: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    selected_mask = np.asarray(mask, dtype=np.uint8)
    if erosion_px > 0:
        kernel_size = 2 * int(erosion_px) + 1
        eroded = cv2.erode(
            selected_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        ).astype(bool)
        if int(np.count_nonzero(eroded)) >= 100:
            selected_mask = eroded.astype(np.uint8)
    indices = np.flatnonzero(selected_mask)
    if not indices.size:
        raise ValueError("A requested facies contains no usable pixels")
    if indices.size > maximum:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, size=maximum, replace=False))
    pixels = np.asarray(image, dtype=np.float32).reshape(-1, 3)[indices]
    features = color_response_features(pixels)
    center = np.median(features, axis=0)
    mad = 1.4826 * np.median(np.abs(features - center), axis=0)
    return center.astype(np.float64), mad.astype(np.float64), int(indices.size)


def _project_features(
    features: np.ndarray,
    model: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))
    location = np.asarray(model["feature_location"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    normalized = (values - location) / scale
    nodes = np.asarray(model["path_nodes"], dtype=np.float64)
    ph_nodes = np.asarray(model["ph_nodes"], dtype=np.float64)
    best_distance = np.full(values.shape[0], np.inf, dtype=np.float64)
    best_ph = np.full(values.shape[0], np.nan, dtype=np.float64)
    for index in range(nodes.shape[0] - 1):
        start = nodes[index]
        delta = nodes[index + 1] - start
        denominator = max(float(np.dot(delta, delta)), 1.0e-15)
        fraction = np.clip(((normalized - start) @ delta) / denominator, 0.0, 1.0)
        projection = start + fraction[:, None] * delta
        distance = np.linalg.norm(normalized - projection, axis=1)
        improved = distance < best_distance
        best_distance[improved] = distance[improved]
        interpolated = ph_nodes[index] + fraction * (
            ph_nodes[index + 1] - ph_nodes[index]
        )
        best_ph[improved] = interpolated[improved]
    return best_ph, best_distance


def _fit_path(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    centers = np.asarray([row["center"] for row in records], dtype=np.float64)
    spreads = np.asarray([row["mad"] for row in records], dtype=np.float64)
    location = np.median(centers, axis=0)
    between = 1.4826 * np.median(np.abs(centers - location), axis=0)
    within = np.median(spreads, axis=0)
    scale = np.maximum.reduce(
        (between, within, np.full(len(FEATURE_NAMES), 1.0e-4))
    )
    normalized = (centers - location) / scale
    ph_values = sorted({float(row["ph"]) for row in records})
    if len(ph_values) < 2:
        raise ValueError("A response path requires at least two distinct pH levels")
    nodes = np.vstack(
        [
            np.median(
                normalized[
                    np.asarray(
                        [float(row["ph"]) == ph for row in records], dtype=bool
                    )
                ],
                axis=0,
            )
            for ph in ph_values
        ]
    )
    steps = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    if not np.any(steps > 1.0e-6):
        raise ValueError("The measured pH levels are optically indistinguishable")
    model = {
        "feature_location": location.tolist(),
        "feature_scale": scale.tolist(),
        "path_nodes": nodes.tolist(),
        "ph_nodes": ph_values,
        "path_step_lengths": steps.tolist(),
        "path_length": float(np.sum(steps)),
        "minimum_path_step": float(np.min(steps)),
        "monotonic_coordinate": "ordered_measured_ph_polyline_projection",
    }
    predictions, distances = _project_features(centers, model)
    truth = np.asarray([float(row["ph"]) for row in records], dtype=np.float64)
    residuals = np.abs(predictions - truth)
    model["training"] = {
        "record_count": len(records),
        "distinct_ph_count": len(ph_values),
        "mae_ph": float(np.mean(residuals)),
        "maximum_error_ph": float(np.max(residuals)),
        "path_distance_p95": float(np.percentile(distances, 95.0)),
    }
    return model


def _fit_groups(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["scope"]), str(row["facies_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (scope, facies_id), rows in sorted(grouped.items()):
        if len({float(row["ph"]) for row in rows}) < 2:
            continue
        model = _fit_path(rows)
        holdout_truth: list[float] = []
        holdout_prediction: list[float] = []
        image_ids = sorted({str(row["image_id"]) for row in rows})
        for image_id in image_ids:
            training = [row for row in rows if str(row["image_id"]) != image_id]
            held_out = [row for row in rows if str(row["image_id"]) == image_id]
            if len({float(row["ph"]) for row in training}) < 2:
                continue
            fold_model = _fit_path(training)
            prediction, _distance = _project_features(
                np.asarray([row["center"] for row in held_out]), fold_model
            )
            holdout_truth.extend(float(row["ph"]) for row in held_out)
            holdout_prediction.extend(float(value) for value in prediction)
        if holdout_truth:
            errors = np.abs(
                np.asarray(holdout_prediction) - np.asarray(holdout_truth)
            )
            interval = float(np.percentile(errors, 90.0))
            holdout = {
                "status": "evaluated",
                "record_count": len(errors),
                "mae_ph": float(np.mean(errors)),
                "rmse_ph": float(np.sqrt(np.mean(errors * errors))),
                "maximum_error_ph": float(np.max(errors)),
                "absolute_error_p90_ph": interval,
                "interval_coverage": float(np.mean(errors <= interval)),
            }
        else:
            interval = max(
                0.05,
                0.5 * float(np.min(np.diff(np.asarray(model["ph_nodes"])))),
            )
            holdout = {
                "status": "not_evaluable",
                "record_count": 0,
                "reason": (
                    "Leave-one-image-out fitting leaves fewer than two pH levels."
                ),
            }
        model["uncertainty"] = {
            "base_half_width_ph": max(0.05, interval),
            "distance_reference": float(model["training"]["path_distance_p95"]),
            "distance_to_ph_scale": (
                float(max(model["ph_nodes"]) - min(model["ph_nodes"]))
                / max(float(model["path_length"]), 1.0e-6)
            ),
            "interpretation": "empirical holdout error plus out-of-path distance",
        }
        output.append(
            {
                "scope": scope,
                "facies_id": facies_id,
                "facies_name": str(rows[0].get("facies_name") or facies_id),
                "model": model,
                "holdout": holdout,
            }
        )
    if not output:
        raise ValueError("No scope/facies group contains two distinct pH levels")
    return output


def fit_indicator_response_from_arrays(
    observations: Sequence[Mapping[str, Any]],
    *,
    facies: np.ndarray,
    active_mask: np.ndarray,
    facies_names: Mapping[str, str] | None = None,
    indicator_system: str = "BTB",
    sampling: Mapping[str, Any] | None = None,
    chemistry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a monotone full-color pH path from corrected, registered arrays."""

    settings = {
        "per_facies": True,
        "max_pixels_per_image_and_facies": 30_000,
        "erode_facies_boundary_px": 2,
        "random_seed": 20_260_903,
    }
    settings.update(dict(sampling or {}))
    facies_array = np.asarray(facies, dtype=np.int32)
    active = np.asarray(active_mask, dtype=bool)
    if facies_array.shape != active.shape:
        raise ValueError("Facies and active masks must share one grid")
    names = {str(key): str(value) for key, value in dict(facies_names or {}).items()}
    records: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for image_index, raw in enumerate(observations):
        image = np.asarray(raw["image"], dtype=np.float32)
        if image.shape[:2] != active.shape or image.shape[-1] != 3:
            raise ValueError("Every reference image must match the approved geometry grid")
        ph = _number(raw.get("ph"))
        if ph is None or not 0.0 <= ph <= 14.0:
            raise ValueError("Every response observation requires a measured pH")
        image_id = str(raw.get("image_id") or f"reference-{image_index + 1}")
        scope = str(raw.get("scope") or "series").strip().casefold() or "series"
        masks: list[tuple[str, np.ndarray]] = [("global", active)]
        if bool(settings["per_facies"]):
            masks.extend(
                (str(value), active & (facies_array == value))
                for value in sorted(np.unique(facies_array[active]))
                if int(value) >= 0
            )
        sampled = 0
        for facies_id, mask in masks:
            if int(np.count_nonzero(mask)) < 100:
                continue
            seed_material = f"{settings['random_seed']}|{image_id}|{facies_id}"
            seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
            center, mad, sample_count = _sample_center(
                image,
                mask,
                maximum=int(settings["max_pixels_per_image_and_facies"]),
                erosion_px=int(settings["erode_facies_boundary_px"]),
                seed=seed,
            )
            records.append(
                {
                    "image_id": image_id,
                    "ph": float(ph),
                    "scope": scope,
                    "facies_id": facies_id,
                    "facies_name": names.get(facies_id, f"Facies {facies_id}"),
                    "center": center.tolist(),
                    "mad": mad.tolist(),
                    "sample_count": sample_count,
                }
            )
            sampled += sample_count
        image_rows.append(
            {
                "image_id": image_id,
                "ph": float(ph),
                "scope": scope,
                "sample_count": sampled,
            }
        )
    groups = _fit_groups(records)
    chemistry_config = deepcopy(dict(chemistry or {}))
    adapter = str(chemistry_config.get("adapter") or "")
    chemistry_supported = adapter == "btb_carbonate_charge_balance_v1" and (
        "btb" in str(indicator_system).casefold()
    )
    if chemistry_supported:
        for group in groups:
            model = group["model"]
            model["dic_nodes_M"] = carbonate_dic_from_ph(
                np.asarray(model["ph_nodes"]),
                alkalinity_M=float(chemistry_config["alkalinity_M"]),
                btb_M=float(chemistry_config["btb_M"]),
                pKa1=float(chemistry_config["pKa1"]),
                pKa2=float(chemistry_config["pKa2"]),
                pKaB=float(chemistry_config["pKaB"]),
            ).tolist()
    holdout_groups = [
        row for row in groups if row["holdout"].get("status") == "evaluated"
    ]
    bounds_reviewed = bool(chemistry_config.get("bounds_reviewed"))
    if not chemistry_supported:
        status = "optical_only_unsupported_chemistry_adapter"
    elif not bounds_reviewed:
        status = "provisional_chemistry"
    elif holdout_groups:
        status = "ready"
    else:
        status = "ready_limited_holdout"
    return {
        "schema": 1,
        "format": INDICATOR_RESPONSE_MODEL_FORMAT,
        "status": status,
        "indicator_system": str(indicator_system),
        "feature_contract": {
            "version": "full-color-response-features-v1",
            "features": list(FEATURE_NAMES),
            "assumed_color_direction": None,
            "input": "registered_radiometrically_corrected_srgb",
        },
        "sampling": deepcopy(settings),
        "images": image_rows,
        "records": records,
        "groups": groups,
        "chemistry": chemistry_config,
        "validation": {
            "strategy": "leave_one_reference_image_out",
            "evaluated_group_count": len(holdout_groups),
            "group_count": len(groups),
            "holdout_record_count": sum(
                int(row["holdout"].get("record_count") or 0)
                for row in holdout_groups
            ),
            "mean_group_mae_ph": (
                float(
                    np.mean(
                        [float(row["holdout"]["mae_ph"]) for row in holdout_groups]
                    )
                )
                if holdout_groups
                else None
            ),
        },
        "summary": {
            "reference_image_count": len(image_rows),
            "distinct_ph_count": len({row["ph"] for row in image_rows}),
            "facies_group_count": len(groups),
            "bounds_reviewed": bounds_reviewed,
            "chemistry_supported": chemistry_supported,
        },
    }


def _select_group(
    model: Mapping[str, Any], *, scope: str, facies_id: str
) -> Mapping[str, Any]:
    groups = [dict(row) for row in model.get("groups") or []]
    candidates = (
        (scope, facies_id),
        ("series", facies_id),
        (scope, "global"),
        ("series", "global"),
    )
    for wanted_scope, wanted_facies in candidates:
        for row in groups:
            if (
                str(row.get("scope")) == wanted_scope
                and str(row.get("facies_id")) == wanted_facies
            ):
                return dict(row["model"])
    raise KeyError(f"No response model exists for scope={scope}, facies={facies_id}")


def predict_indicator_ph(
    model: Mapping[str, Any],
    rgb: np.ndarray,
    *,
    scope: str = "series",
    facies_id: str | int = "global",
) -> dict[str, np.ndarray]:
    """Predict pH and an empirical interval from one response-model group."""

    values = np.asarray(rgb)
    shape = values.shape[:-1]
    features = color_response_features(values).reshape(-1, len(FEATURE_NAMES))
    selected = _select_group(
        model,
        scope=str(scope).strip().casefold() or "series",
        facies_id=str(facies_id),
    )
    ph, distance = _project_features(features, selected)
    uncertainty = dict(selected.get("uncertainty") or {})
    reference = float(uncertainty.get("distance_reference") or 0.0)
    distance_scale = float(uncertainty.get("distance_to_ph_scale") or 0.0)
    half_width = float(uncertainty.get("base_half_width_ph") or 0.0)
    half_width = half_width + np.maximum(distance - reference, 0.0) * distance_scale
    ph_min = float(min(selected["ph_nodes"]))
    ph_max = float(max(selected["ph_nodes"]))
    return {
        "ph": ph.reshape(shape),
        "ph_lower": np.clip(ph - half_width, ph_min, ph_max).reshape(shape),
        "ph_upper": np.clip(ph + half_width, ph_min, ph_max).reshape(shape),
        "path_distance": distance.reshape(shape),
    }


def predict_indicator_dic(
    model: Mapping[str, Any],
    rgb: np.ndarray,
    *,
    scope: str = "series",
    facies_id: str | int = "global",
) -> dict[str, np.ndarray]:
    """Map optical pH predictions to DIC through the reviewed carbonate contract."""

    chemistry = dict(model.get("chemistry") or {})
    if str(chemistry.get("adapter") or "") != "btb_carbonate_charge_balance_v1":
        raise ValueError("The response model has no supported carbonate adapter")
    prediction = predict_indicator_ph(
        model, rgb, scope=scope, facies_id=facies_id
    )
    kwargs = {
        "alkalinity_M": float(chemistry["alkalinity_M"]),
        "btb_M": float(chemistry["btb_M"]),
        "pKa1": float(chemistry["pKa1"]),
        "pKa2": float(chemistry["pKa2"]),
        "pKaB": float(chemistry["pKaB"]),
    }
    prediction["dic_M"] = carbonate_dic_from_ph(prediction["ph"], **kwargs)
    prediction["dic_lower_M"] = carbonate_dic_from_ph(
        prediction["ph_upper"], **kwargs
    )
    prediction["dic_upper_M"] = carbonate_dic_from_ph(
        prediction["ph_lower"], **kwargs
    )
    saturation = float(chemistry.get("co2_saturation_M") or 1.0)
    prediction["normalized_dic"] = np.clip(
        prediction["dic_M"] / max(saturation, 1.0e-15), 0.0, 1.0
    )
    return prediction


def inspect_indicator_response_model_reference(raw: Any) -> dict[str, Any]:
    """Verify a response-model reference and return its deployable contract."""

    reference = dict(raw) if isinstance(raw, Mapping) else {}
    path_text = str(reference.get("path") or "").strip()
    if not path_text:
        return {"available": False, "valid": False, "issues": ["No model attached."]}
    path = Path(path_text).expanduser().resolve()
    issues: list[str] = []
    if not path.is_file():
        return {
            "available": False,
            "valid": False,
            "path": str(path),
            "issues": [f"Indicator-response model is unavailable: {path}"],
        }
    file_sha256 = sha256_file(path)
    expected = str(reference.get("sha256") or "")
    if not expected or expected != file_sha256:
        issues.append("Indicator-response model failed its SHA-256 contract.")
    try:
        model = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"Could not read indicator-response model: {exc}")
        model = {}
    if not isinstance(model, Mapping) or model.get("format") != INDICATOR_RESPONSE_MODEL_FORMAT:
        issues.append("Unsupported indicator-response model format.")
        model = {}
    status = str(model.get("status") or "")
    if status not in {"ready", "ready_limited_holdout"}:
        issues.append(f"Indicator-response model is not deployable ({status or 'unknown'}).")
    return {
        "available": True,
        "valid": not issues,
        "path": str(path),
        "sha256": file_sha256,
        "model_id": str(model.get("model_id") or reference.get("model_id") or ""),
        "status": status,
        "indicator_system": str(model.get("indicator_system") or ""),
        "chemistry": deepcopy(dict(model.get("chemistry") or {})),
        "validation": deepcopy(dict(model.get("validation") or {})),
        "summary": deepcopy(dict(model.get("summary") or {})),
        "issues": issues,
        "model": deepcopy(dict(model)),
    }


class IndicatorResponseModelStore:
    """Append-only fitted response models with one replaceable current pointer."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "indicator_response"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        from .series_paths import series_folder_path
        self.artifact_root = series_folder_path(project.root, "indicator_response_models")

    def revisions(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.revisions_path)

    def current(self) -> dict[str, Any]:
        from .analysis_domains import AnalysisDomainStore
        return AnalysisDomainStore.current(self)

    def current_reference(self) -> dict[str, Any]:
        revision = self.current()
        if not revision:
            return {}
        return {
            "schema": 1,
            "format": INDICATOR_RESPONSE_REFERENCE_FORMAT,
            "model_id": str(revision.get("model_id") or ""),
            "path": str(revision.get("model_path") or ""),
            "sha256": str(revision.get("model_sha256") or ""),
            "status": str(revision.get("status") or ""),
        }

    def fit_and_save(
        self,
        config: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        """Fit from immutable source references and save one versioned model."""

        candidate = deepcopy(dict(config))
        issues = validate_indicator_response_config(candidate)
        if issues:
            raise ValueError(" ".join(issues))
        geometry_revision = GeometryRevisionStore(self.project).current()
        geometry_config = GeometryRevisionStore(self.project).current_config()
        facies, active, names = _geometry_masks(geometry_config)
        correction = CorrectionRevisionStore(self.project).current()
        algorithm = str(correction.get("algorithm") or "")
        if algorithm not in {IDENTITY_ALGORITHM, *COLORCHECKER_ALGORITHMS}:
            raise ValueError("Save an image-correction revision before fitting buffers")

        registered_reference: np.ndarray | None = None
        reference_provenance: dict[str, Any] = {}
        if algorithm in COLORCHECKER_ALGORITHMS:
            reference_path = Path(str(correction.get("reference_image") or ""))
            registered_reference, reference_provenance = (
                load_registered_geometry_image(geometry_config, reference_path)
            )

        observations: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        for index, entry in enumerate(
            (candidate.get("reference_contract") or {}).get("entries") or []
        ):
            row = dict(entry)
            if row.get("kind") != REFERENCE_KIND_PH_BUFFER:
                continue
            source = Path(str(row.get("image_path") or "")).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            registered, geometry_provenance = load_registered_geometry_image(
                geometry_config, source
            )
            correction_diagnostics: dict[str, Any]
            if algorithm == UNIFORM24_ALGORITHM:
                if registered_reference is None:
                    raise RuntimeError("Uniform24 reference image was not loaded")
                fitted = fit_uniform24_array_transform(
                    registered_reference,
                    registered,
                    corner=str(correction.get("colorchecker_corner") or "upper_right"),
                    roi_rc=correction.get("colorchecker_roi_rc"),
                    settings=dict(correction.get("settings") or {}),
                )
                corrected = apply_uniform24_array_transform(
                    registered, fitted["transform"]
                )
                correction_diagnostics = deepcopy(dict(fitted["diagnostics"]))
            elif algorithm == DARSIA_BASELINE_COLOR_ALGORITHM:
                if registered_reference is None:
                    raise RuntimeError("DarSIA ColorCorrection reference was not loaded")
                fitted = apply_darsia_baseline_array_correction(
                    registered_reference,
                    registered,
                    corner=str(correction.get("colorchecker_corner") or "upper_right"),
                    roi_rc=correction.get("colorchecker_roi_rc"),
                    settings=dict(correction.get("settings") or {}),
                )
                corrected = np.asarray(fitted["corrected"], dtype=np.float32)
                correction_diagnostics = deepcopy(dict(fitted["diagnostics"]))
            else:
                corrected = np.asarray(registered, dtype=np.float32)
                correction_diagnostics = {
                    "algorithm": IDENTITY_ALGORITHM,
                    "note": "No external radiometric target was configured.",
                }
            source_sha256 = sha256_file(source)
            image_id = f"buffer-{index + 1}-{source_sha256[:12]}"
            observations.append(
                {
                    "image_id": image_id,
                    "ph": float(row["ph"]),
                    "scope": str(row.get("scope") or "series"),
                    "image": corrected,
                }
            )
            source_rows.append(
                {
                    "image_id": image_id,
                    "path": str(source),
                    "sha256": source_sha256,
                    "ph": float(row["ph"]),
                    "scope": str(row.get("scope") or "series"),
                    "geometry": geometry_provenance,
                    "radiometric_qc": correction_diagnostics,
                }
            )

        model = fit_indicator_response_from_arrays(
            observations,
            facies=facies,
            active_mask=active,
            facies_names=names,
            indicator_system=str(candidate.get("indicator_system") or "unspecified"),
            sampling=dict(candidate.get("sampling") or {}),
            chemistry=dict(candidate.get("chemistry") or {}),
        )
        model_id = uuid.uuid4().hex
        model.update(
            {
                "model_id": model_id,
                "created_utc": utc_now(),
                "created_by": str(actor).strip(),
                "reason": str(reason).strip(),
                "source_dependencies": {
                    "reference_images": source_rows,
                    "geometry_revision_id": str(
                        geometry_revision.get("revision_id") or ""
                    ),
                    "geometry_revision_sha256": str(
                        geometry_revision.get("content_sha256") or ""
                    ),
                    "image_correction_revision_id": str(
                        correction.get("revision_id") or ""
                    ),
                    "image_correction_revision_sha256": str(
                        correction.get("content_sha256") or ""
                    ),
                    "radiometric_algorithm": algorithm,
                    "radiometric_reference": reference_provenance,
                },
            }
        )
        model["content_sha256"] = _json_digest(model)
        destination = self.artifact_root / model_id
        staging = self.artifact_root / f".{model_id}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            staged_model = staging / "model.json"
            _atomic_json(staged_model, model)
            model_sha256 = sha256_file(staged_model)
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        model_path = destination / "model.json"
        revision = {
            "schema": 1,
            "format": INDICATOR_RESPONSE_REVISION_FORMAT,
            "revision_id": uuid.uuid4().hex,
            "model_id": model_id,
            "recorded_utc": utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "status": str(model["status"]),
            "configuration": candidate,
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "summary": deepcopy(dict(model["summary"])),
            "validation": deepcopy(dict(model["validation"])),
            "upstream": deepcopy(dict(model["source_dependencies"])),
        }
        revision["content_sha256"] = _json_digest(revision)
        _append_jsonl(self.revisions_path, revision)
        _atomic_json(self.current_path, revision)
        self.project._audit(
            "indicator_response_model_fitted",
            actor,
            {
                "model_id": model_id,
                "status": model["status"],
                "reference_image_count": model["summary"]["reference_image_count"],
                "distinct_ph_count": model["summary"]["distinct_ph_count"],
            },
        )
        return deepcopy(revision)


__all__ = [
    "CHEMICAL_PARAMETER_KEYS",
    "FEATURE_NAMES",
    "INDICATOR_RESPONSE_CONFIG_FORMAT",
    "INDICATOR_RESPONSE_MODEL_FORMAT",
    "INDICATOR_RESPONSE_REFERENCE_FORMAT",
    "INDICATOR_RESPONSE_REVISION_FORMAT",
    "OPTICAL_PARAMETERS_REPLACED_BY_RESPONSE_MODEL",
    "IndicatorResponseModelStore",
    "color_response_features",
    "default_indicator_response_config",
    "fit_indicator_response_from_arrays",
    "inspect_indicator_response_model_reference",
    "predict_indicator_dic",
    "predict_indicator_ph",
    "validate_indicator_response_config",
]
