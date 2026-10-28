"""Portable, cross-series optical phase-model contracts.

This module owns the boundary between learned 2D optical classification and
the experiment-specific physical conversion performed later by DarSIA.  Model
bundles may combine annotations from several FluidFlower series, but they do
not contain depth, concentration, saturation, or mass calibration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np

from darsia.fluidflower.phase_features import (
    DARSIA_CHANGE_DETECTOR,
    DARSIA_CHANGE_FEATURE_NAMES,
    NO_CHANGE_DETECTOR,
    OpticalFeatureContract,
    optical_feature_matrix,
    prepare_optical_feature_workspace,
)
from darsia.fluidflower.project import FluidFlowerProject, sha256_file, utc_now


PHASE_LEARNING_PLAN_FORMAT = "darsia-fluidflower-phase-learning-plan-v1"
PHASE_LEARNING_REVISION_FORMAT = "darsia-fluidflower-phase-learning-revision-v1"
PHASE_MODEL_BUNDLE_FORMAT = "darsia-fluidflower-phase-model-bundle-view-v1"
PHASE_MODEL_HEADS = ("co2_presence", "gas_within_co2")
PHASE_ACTIVE_LEARNING_BATCH_FORMAT = "darsia-fluidflower-phase-active-learning-batch-v1"
PHASE_ACTIVE_LEARNING_RESPONSE_FORMAT = (
    "darsia-fluidflower-phase-active-learning-response-v1"
)
PHASE_REVIEW_CLASSES = (
    "untouched_water",
    "aqueous_co2_only",
    "aqueous_co2_dominant",
    "mixed_free_aqueous_co2",
    "free_co2_gas_dominant",
    "free_co2_gas_only",
    "mixed_aqueous_co2_and_water",
    "mixed_free_aqueous_co2_and_water",
    "co2_present_phase_uncertain",
    "artifact",
    "not_assessable",
    "ignore",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _append_jsonl as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_reference(path: Path) -> dict[str, Any]:
    from darsia.fluidflower.phase_interpretation import _file_reference as _canonical_file_reference
    return _canonical_file_reference(path)


def _read_json(path: Path) -> dict[str, Any]:
    from .phase_saved_review_metadata import read_model_json
    return read_model_json(path)


def phase_uncertainty_map(predictions: Mapping[str, Any]) -> np.ndarray:
    """Return normalized three-class entropy for one optical phase prediction.

    This is predictive uncertainty, not a calibrated physical uncertainty.  It is
    used only to choose informative regions for human review.
    """

    names = (
        "untouched_water_probability",
        "aqueous_co2_probability",
        "free_co2_gas_probability",
    )
    fields = [np.asarray(predictions[name], dtype=np.float64) for name in names]
    if any(field.ndim != 2 for field in fields) or any(
        field.shape != fields[0].shape for field in fields[1:]
    ):
        raise ValueError("Phase probabilities must be equally shaped 2D arrays")
    probabilities = np.stack(fields, axis=-1)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    normalizer = np.sum(probabilities, axis=-1, keepdims=True)
    probabilities = np.divide(
        probabilities,
        normalizer,
        out=np.full_like(probabilities, 1.0 / 3.0),
        where=normalizer > np.finfo(np.float64).eps,
    )
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, np.finfo(np.float64).tiny)),
        axis=-1,
    )
    return np.asarray(entropy / np.log(3.0), dtype=np.float32)


def _normalized_box_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    ax0 = float(first.get("x") or 0.0)
    ay0 = float(first.get("y") or 0.0)
    ax1 = ax0 + float(first.get("width") or 0.0)
    ay1 = ay0 + float(first.get("height") or 0.0)
    bx0 = float(second.get("x") or 0.0)
    by0 = float(second.get("y") or 0.0)
    bx1 = bx0 + float(second.get("width") or 0.0)
    by1 = by0 + float(second.get("height") or 0.0)
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    union = (
        max(0.0, (ax1 - ax0) * (ay1 - ay0))
        + max(0.0, (bx1 - bx0) * (by1 - by0))
        - intersection
    )
    return intersection / union if union > 0.0 else 0.0


def propose_uncertain_phase_regions(
    predictions: Mapping[str, Any],
    *,
    run_id: str,
    frame_index: int,
    elapsed_h: float | None,
    model_id: str,
    count: int,
    eligible_mask: Any | None = None,
    facies_labels: Any | None = None,
    box_size_px: int = 96,
    existing_regions: Sequence[Mapping[str, Any]] = (),
    minimum_domain_fraction: float = 0.95,
    maximum_iou: float = 0.18,
) -> list[dict[str, Any]]:
    """Select diverse high-uncertainty boxes from a visible-2D phase map.

    Candidate centers are sampled on a half-box grid.  Greedy selection balances
    model class, facies, and coarse position while rejecting boxes outside the
    supplied porous-medium mask and boxes already reviewed.
    """

    requested = max(0, int(count))
    if requested == 0:
        return []
    names = (
        "untouched_water_probability",
        "aqueous_co2_probability",
        "free_co2_gas_probability",
    )
    labels = ("water", "aqueous", "gas")
    arrays = [np.asarray(predictions[name], dtype=np.float32) for name in names]
    shape = arrays[0].shape
    if len(shape) != 2 or any(array.shape != shape for array in arrays[1:]):
        raise ValueError("Phase probabilities must be equally shaped 2D arrays")
    height, width = shape
    if min(height, width) < 2:
        raise ValueError("Phase probability image is too small")
    domain = (
        np.ones(shape, dtype=bool)
        if eligible_mask is None
        else np.asarray(eligible_mask, dtype=bool)
    )
    if domain.shape != shape:
        raise ValueError("Eligible mask must match phase probabilities")
    if not np.any(domain):
        return []
    facies = (
        np.zeros(shape, dtype=np.int32)
        if facies_labels is None
        else np.asarray(facies_labels)
    )
    if facies.shape != shape:
        raise ValueError("Facies labels must match phase probabilities")

    uncertainty = phase_uncertainty_map(predictions)
    probabilities = np.stack(arrays, axis=-1)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    size = max(4, min(int(box_size_px), height, width))
    half = size // 2
    stride = max(2, size // 2)
    candidates: list[dict[str, Any]] = []
    for center_y in range(half, max(half + 1, height - (size - half) + 1), stride):
        for center_x in range(half, max(half + 1, width - (size - half) + 1), stride):
            y0 = max(0, min(height - size, center_y - half))
            x0 = max(0, min(width - size, center_x - half))
            y1 = y0 + size
            x1 = x0 + size
            patch_domain = domain[y0:y1, x0:x1]
            domain_fraction = float(np.mean(patch_domain))
            if domain_fraction < float(minimum_domain_fraction):
                continue
            patch_uncertainty = uncertainty[y0:y1, x0:x1][patch_domain]
            patch_probabilities = probabilities[y0:y1, x0:x1][patch_domain]
            mean_probabilities = np.mean(patch_probabilities, axis=0)
            dominant_index = int(np.argmax(mean_probabilities))
            patch_facies = facies[y0:y1, x0:x1][patch_domain]
            unique, counts = np.unique(patch_facies, return_counts=True)
            facies_id = int(unique[int(np.argmax(counts))]) if unique.size else 0
            normalized = {
                "x": x0 / width,
                "y": y0 / height,
                "width": size / width,
                "height": size / height,
            }
            if any(
                _normalized_box_iou(normalized, previous) > float(maximum_iou)
                for previous in existing_regions
            ):
                continue
            candidates.append(
                {
                    **normalized,
                    "bbox_xywh_px": [int(x0), int(y0), int(size), int(size)],
                    "uncertainty_mean": float(np.mean(patch_uncertainty)),
                    "uncertainty_p90": float(np.quantile(patch_uncertainty, 0.9)),
                    "mean_probabilities": {
                        labels[index]: float(mean_probabilities[index])
                        for index in range(3)
                    },
                    "suggested_phase": labels[dominant_index],
                    "facies_id": facies_id,
                    "spatial_bin": [
                        min(2, int(3 * (y0 + size / 2.0) / height)),
                        min(3, int(4 * (x0 + size / 2.0) / width)),
                    ],
                    "domain_fraction": domain_fraction,
                }
            )

    selected: list[dict[str, Any]] = []
    phase_counts: dict[str, int] = {}
    facies_counts: dict[int, int] = {}
    spatial_counts: dict[tuple[int, int], int] = {}
    # Candidate records never change during greedy selection. Preserve the
    # exact digest tie-break while computing each canonical JSON/hash only
    # once, rather than once per remaining candidate on every selection.
    candidate_digests = {id(row): _json_digest(row) for row in candidates}
    remaining = list(candidates)
    while remaining and len(selected) < requested:

        def rank(row: Mapping[str, Any]) -> tuple[float, float, str]:
            phase = str(row["suggested_phase"])
            facies_id = int(row["facies_id"])
            spatial = tuple(int(value) for value in row["spatial_bin"])
            diversity = (
                0.12 / (1.0 + phase_counts.get(phase, 0))
                + 0.08 / (1.0 + facies_counts.get(facies_id, 0))
                + 0.08 / (1.0 + spatial_counts.get(spatial, 0))
            )
            return (
                float(row["uncertainty_mean"]) + diversity,
                float(row["uncertainty_p90"]),
                candidate_digests[id(row)],
            )

        chosen = max(remaining, key=rank)
        remaining.remove(chosen)
        if any(
            _normalized_box_iou(chosen, previous) > float(maximum_iou)
            for previous in selected
        ):
            continue
        phase = str(chosen["suggested_phase"])
        facies_id = int(chosen["facies_id"])
        spatial = tuple(int(value) for value in chosen["spatial_bin"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        facies_counts[facies_id] = facies_counts.get(facies_id, 0) + 1
        spatial_counts[spatial] = spatial_counts.get(spatial, 0) + 1
        seed = {
            "run_id": str(run_id).strip().casefold(),
            "frame_index": int(frame_index),
            "model_id": str(model_id),
            "bbox_xywh_px": chosen["bbox_xywh_px"],
        }
        selected.append(
            {
                "proposal_id": "phase-active-" + _json_digest(seed)[:20],
                "run_id": seed["run_id"],
                "frame_index": seed["frame_index"],
                "elapsed_h": None if elapsed_h is None else float(elapsed_h),
                "model_id": str(model_id),
                "shape": "box",
                "coordinate_space": "normalized_corrected_visible_2d",
                "analysis_domain": "porous_medium_only",
                "selection_basis": "predictive_entropy_with_phase_facies_spatial_balance",
                "model_suggestion_hidden_until_answer": True,
                **chosen,
            }
        )
    return selected


class PhaseActiveLearningStore:
    """Immutable proposal batches and append-only reviewer responses."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.root = project.root / "annotations" / "phase" / "active_learning"
        self.batch_root = self.root / "proposal_batches"
        self.responses_path = self.root / "responses.jsonl"
        self.current_path = self.root / "current_responses.json"

    def response_history(self) -> list[dict[str, Any]]:
        if not self.responses_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.responses_path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
        return rows

    def current_responses(self) -> dict[str, dict[str, Any]]:
        if self.current_path.is_file():
            payload = _read_json(self.current_path)
            return {
                str(key): dict(value)
                for key, value in dict(payload.get("responses") or {}).items()
            }
        current: dict[str, dict[str, Any]] = {}
        for row in self.response_history():
            current[str(row.get("proposal_id") or "")] = row
        return current

    def save_proposal_batch(
        self,
        proposals: Sequence[Mapping[str, Any]],
        *,
        actor: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        rows = [deepcopy(dict(row)) for row in proposals]
        identifiers = [str(row.get("proposal_id") or "") for row in rows]
        if not rows or any(not value for value in identifiers):
            raise ValueError("An active-learning batch requires identified proposals")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Active-learning proposal IDs must be unique")
        batch_id = uuid.uuid4().hex
        payload = {
            "schema": 1,
            "format": PHASE_ACTIVE_LEARNING_BATCH_FORMAT,
            "batch_id": batch_id,
            "created_utc": utc_now(),
            "created_by": str(actor).strip(),
            "metadata": deepcopy(dict(metadata or {})),
            "proposal_count": len(rows),
            "proposals": rows,
        }
        payload["content_sha256"] = _json_digest(payload)
        destination = self.batch_root / batch_id
        destination.mkdir(parents=True, exist_ok=False)
        _atomic_json(destination / "proposals.json", payload)
        return deepcopy(payload)

    def record_response(
        self,
        proposal: Mapping[str, Any],
        *,
        phase_class: str,
        reviewer: str,
        confidence: str = "medium",
        comment: str = "",
    ) -> dict[str, Any]:
        proposal_id = str(proposal.get("proposal_id") or "").strip()
        label = str(phase_class).strip()
        actor = str(reviewer).strip()
        if not proposal_id:
            raise ValueError("Proposal ID is required")
        if label not in PHASE_REVIEW_CLASSES:
            raise ValueError(f"Unsupported visible-phase class: {label!r}")
        if not actor:
            raise ValueError("Reviewer name is required")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError("Confidence must be high, medium, or low")
        current = self.current_responses()
        previous = current.get(proposal_id)
        response = {
            "schema": 1,
            "format": PHASE_ACTIVE_LEARNING_RESPONSE_FORMAT,
            "response_id": uuid.uuid4().hex,
            "proposal_id": proposal_id,
            "proposal": deepcopy(dict(proposal)),
            "phase_class": label,
            "confidence": confidence,
            "comment": str(comment).strip(),
            "reviewer": actor,
            "recorded_utc": utc_now(),
            "supersedes_response_id": (
                str(previous.get("response_id") or "") if previous else ""
            ),
            "interpretation_scope": "registered_visible_camera_plane_2d",
            "eligible_for_training": label not in {"ignore", "not_assessable"},
        }
        response["content_sha256"] = _json_digest(response)
        _append_jsonl(self.responses_path, response)
        current[proposal_id] = response
        _atomic_json(
            self.current_path,
            {
                "schema": 1,
                "format": PHASE_ACTIVE_LEARNING_RESPONSE_FORMAT,
                "updated_utc": response["recorded_utc"],
                "responses": current,
            },
        )
        self.project._audit(
            "phase_active_learning_response_saved",
            actor,
            {
                "proposal_id": proposal_id,
                "response_id": response["response_id"],
                "phase_class": label,
            },
        )
        return deepcopy(response)


def _model_files(root: Path) -> list[Path]:
    models = root / "models"
    if not models.is_dir():
        return []
    return sorted(
        path
        for path in models.rglob("*.json")
        if path.is_file()
        and (
            path.stem in PHASE_MODEL_HEADS
            or path.stem.removesuffix("_signal") in PHASE_MODEL_HEADS
        )
    )


def inspect_phase_model_bundle(root: Path) -> dict[str, Any]:
    """Inspect a serialized model bundle without changing it."""

    bundle_root = Path(root).expanduser().resolve()
    if not bundle_root.is_dir():
        raise NotADirectoryError(bundle_root)
    summary_path = bundle_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Model summary is unavailable: {summary_path}")
    summary = _read_json(summary_path)
    inventory = dict(summary.get("inventory") or {})
    datasets = sorted(str(value) for value in inventory.get("datasets") or [])
    runs = sorted(str(value) for value in inventory.get("runs") or [])

    rows: list[dict[str, Any]] = []
    heads_by_adapter: dict[str, set[str]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    for path in _model_files(bundle_root):
        payload = _read_json(path)
        adapter = path.parent.name
        head = path.stem.removesuffix("_signal")
        feature_contract = OpticalFeatureContract.from_dict(
            dict(payload.get("feature_contract") or {})
        )
        declared_features = tuple(
            str(value) for value in payload.get("feature_names") or []
        )
        if declared_features != feature_contract.feature_names:
            raise ValueError(f"Model feature contract differs in {path}")
        missing_change = (
            []
            if feature_contract.change_detector == NO_CHANGE_DETECTOR
            else [
                name
                for name in DARSIA_CHANGE_FEATURE_NAMES
                if name not in declared_features
                and not (
                    name.startswith("dynamic_") and not feature_contract.include_dynamic
                )
            ]
        )
        if missing_change:
            raise ValueError(
                f"Model {path} omits DarSIA change features: {missing_change}"
            )
        coefficients = np.asarray(payload.get("coefficients") or [], dtype=float)
        mean = np.asarray(payload.get("feature_mean") or [], dtype=float)
        scale = np.asarray(payload.get("feature_scale") or [], dtype=float)
        expected = len(declared_features)
        if not (coefficients.size == mean.size == scale.size == expected):
            raise ValueError(f"Serialized coefficient dimensions differ in {path}")
        heads_by_adapter.setdefault(adapter, set()).add(head)
        contracts.setdefault(adapter, feature_contract.to_dict())
        if contracts[adapter] != feature_contract.to_dict():
            raise ValueError(
                f"Model heads use different feature contracts in {adapter}"
            )
        rows.append(
            {
                "adapter": adapter,
                "head": head,
                "algorithm": str(payload.get("algorithm") or ""),
                "feature_count": expected,
                "threshold": float(
                    (
                        (payload.get("training_provenance") or {}).get(
                            "cross_validation"
                        )
                        or {}
                    ).get("oof_threshold", 0.5)
                ),
                "feature_contract": feature_contract.to_dict(),
                "file": _file_reference(path),
            }
        )
    if not rows:
        raise ValueError(
            f"No serialized phase heads found below {bundle_root / 'models'}"
        )
    incomplete = {
        adapter: sorted(set(PHASE_MODEL_HEADS) - heads)
        for adapter, heads in heads_by_adapter.items()
        if set(PHASE_MODEL_HEADS) - heads
    }
    if incomplete:
        raise ValueError(f"Incomplete hierarchical model adapters: {incomplete}")
    change_detectors = sorted(
        {str(contract["change_detector"]) for contract in contracts.values()}
    )
    return {
        "schema": 1,
        "format": PHASE_MODEL_BUNDLE_FORMAT,
        "root": str(bundle_root),
        "campaign_id": str(summary.get("campaign_id") or bundle_root.name),
        "summary": _file_reference(summary_path),
        "models": rows,
        "adapters": sorted(heads_by_adapter),
        "feature_contracts": contracts,
        "datasets": datasets,
        "runs": runs,
        "cross_series": len(datasets) > 1,
        "training_inventory": inventory,
        "change_detector": (
            change_detectors[0] if len(change_detectors) == 1 else "mixed"
        ),
        "change_detectors": change_detectors,
        "scientific_scope": {
            "domain": "registered_visible_camera_plane_2d",
            "outputs": [
                "co2_presence_probability",
                "gas_probability_conditional_on_co2",
                "water_probability",
                "aqueous_co2_probability",
                "free_co2_gas_probability",
            ],
            "contains_depth_or_3d_extrapolation": False,
            "contains_concentration_or_mass": False,
        },
    }


def default_phase_learning_plan() -> dict[str, Any]:
    """Return a portable plan for a hierarchical cross-series phase model."""

    return {
        "schema": 1,
        "format": PHASE_LEARNING_PLAN_FORMAT,
        "model_bundle_path": "",
        "datasets": [],
        "feature_contract": OpticalFeatureContract().to_dict(),
        "heads": list(PHASE_MODEL_HEADS),
        "training_policy": {
            "cross_series": True,
            "holdout_unit": "whole_run",
            "group_key": "dataset_and_run",
            "phase_partition": "co2_presence_then_gas_conditional_on_co2",
            "mass_or_depth_target": False,
        },
        "notes": "",
    }


def phase_learning_plan_from_bundle(
    bundle_root: Path,
    *,
    annotation_files: Sequence[Path] = (),
) -> dict[str, Any]:
    """Create a plan from a verified existing cross-series model bundle."""

    bundle = inspect_phase_model_bundle(bundle_root)
    annotations = [str(Path(path).expanduser().resolve()) for path in annotation_files]
    plan = default_phase_learning_plan()
    plan["model_bundle_path"] = str(Path(bundle_root).expanduser().resolve())
    for dataset in bundle["datasets"]:
        if dataset.casefold() == "ffac":
            runs = [run for run in bundle["runs"] if run.casefold().startswith("ac")]
        elif dataset.casefold() == "ffum":
            runs = [run for run in bundle["runs"] if run.casefold().startswith("run")]
        else:
            runs = list(bundle["runs"])
        plan["datasets"].append(
            {
                "dataset_id": dataset,
                "run_ids": runs,
                "annotation_files": annotations,
                "role": "training_evidence_from_imported_bundle",
            }
        )
    generic_contract = dict(bundle["feature_contracts"].get("generic") or {})
    if generic_contract:
        plan["feature_contract"] = generic_contract
    return plan


def validate_phase_learning_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate model, dataset, and scientific-scope contracts."""

    candidate = deepcopy(dict(plan))
    issues: list[dict[str, Any]] = []

    def issue(severity: str, code: str, message: str, **details: Any) -> None:
        issues.append(
            {
                "severity": severity,
                "code": code,
                "message": message,
                "details": details,
            }
        )

    if candidate.get("format") != PHASE_LEARNING_PLAN_FORMAT:
        issue("error", "unsupported_format", "Unsupported phase-learning plan")
    try:
        feature_contract = OpticalFeatureContract.from_dict(
            dict(candidate.get("feature_contract") or {})
        )
    except Exception as exc:  # noqa: BLE001
        feature_contract = None
        issue("error", "invalid_feature_contract", str(exc))
    heads = tuple(str(value) for value in candidate.get("heads") or [])
    if heads != PHASE_MODEL_HEADS:
        issue(
            "error",
            "invalid_hierarchical_heads",
            "Phase learning requires CO2-presence followed by gas-within-CO2.",
        )

    datasets: list[dict[str, Any]] = []
    seen_datasets: set[str] = set()
    all_annotation_refs: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(candidate.get("datasets") or [], start=1):
        row = dict(raw)
        dataset_id = str(row.get("dataset_id") or "").strip().casefold()
        if not dataset_id:
            issue("error", "dataset_id_missing", f"Dataset row {index} has no ID")
            continue
        if dataset_id in seen_datasets:
            issue("error", "duplicate_dataset", f"Dataset {dataset_id} is repeated")
            continue
        seen_datasets.add(dataset_id)
        run_ids = sorted(
            {
                str(value).strip().casefold()
                for value in row.get("run_ids") or []
                if str(value).strip()
            }
        )
        references: list[dict[str, Any]] = []
        for raw_path in row.get("annotation_files") or []:
            path = Path(str(raw_path)).expanduser().resolve()
            key = str(path).casefold()
            try:
                reference = all_annotation_refs.setdefault(key, _file_reference(path))
            except Exception as exc:  # noqa: BLE001
                issue(
                    "error",
                    "annotation_unavailable",
                    f"Annotation evidence is unavailable: {exc}",
                    dataset_id=dataset_id,
                )
                continue
            references.append(reference)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "run_ids": run_ids,
                "role": str(row.get("role") or "training"),
                "annotation_files": references,
            }
        )
    cross_series_requested = bool(
        (candidate.get("training_policy") or {}).get("cross_series", True)
    )
    if cross_series_requested and len(datasets) < 2:
        issue(
            "error",
            "cross_series_evidence_missing",
            "Cross-series learning requires at least two named datasets.",
        )
    if (
        str((candidate.get("training_policy") or {}).get("holdout_unit") or "")
        != "whole_run"
    ):
        issue(
            "error",
            "invalid_holdout_unit",
            "Scientific validation must hold out complete runs.",
        )
    if bool((candidate.get("training_policy") or {}).get("mass_or_depth_target")):
        issue(
            "error",
            "physical_target_in_optical_model",
            "Depth and mass targets belong to experiment-specific conversion, not phase learning.",
        )

    bundle: dict[str, Any] = {}
    bundle_path = str(candidate.get("model_bundle_path") or "").strip()
    if bundle_path:
        try:
            bundle = inspect_phase_model_bundle(Path(bundle_path))
        except Exception as exc:  # noqa: BLE001
            issue("error", "model_bundle_unavailable", str(exc))
        else:
            missing = sorted(set(seen_datasets) - set(bundle.get("datasets") or []))
            if missing:
                issue(
                    "warning",
                    "bundle_does_not_cover_dataset",
                    "Imported model provenance does not list every configured dataset.",
                    datasets=missing,
                )
            if feature_contract is not None:
                generic = dict(
                    (bundle.get("feature_contracts") or {}).get("generic") or {}
                )
                if (
                    generic
                    and OpticalFeatureContract.from_dict(generic) != feature_contract
                ):
                    issue(
                        "error",
                        "bundle_feature_contract_changed",
                        "Configured features differ from the imported generic model.",
                    )
    else:
        issue(
            "warning",
            "model_bundle_not_selected",
            "No existing model bundle is selected; the plan represents future training only.",
        )
    errors = sum(row["severity"] == "error" for row in issues)
    warnings = sum(row["severity"] == "warning" for row in issues)
    return {
        "schema": 1,
        "valid": errors == 0,
        "plan_sha256": _json_digest(candidate),
        "summary": {
            "error_count": errors,
            "warning_count": warnings,
            "dataset_count": len(datasets),
            "run_count": len({run for row in datasets for run in row["run_ids"]}),
            "annotation_file_count": len(all_annotation_refs),
            "cross_series": len(datasets) > 1,
        },
        "datasets": datasets,
        "model_bundle": bundle,
        "issues": issues,
    }


class PhaseLearningPlanStore:
    """Append-only cross-series phase-learning plans for one project."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.root = project.root / "overrides" / "phase_learning"
        self.revisions_path = self.root / "revisions.jsonl"
        self.current_path = self.root / "current.json"
        self.artifact_root = project.root / "results" / "phase_learning"

    def revisions(self) -> list[dict[str, Any]]:
        if not self.revisions_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.revisions_path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
        return rows

    def current(self) -> dict[str, Any]:
        if self.current_path.is_file():
            return _read_json(self.current_path)
        rows = self.revisions()
        return deepcopy(rows[-1]) if rows else {}

    def save_revision(
        self,
        plan: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
        validation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate = deepcopy(dict(plan))
        report = deepcopy(dict(validation or {}))
        if report.get("plan_sha256") != _json_digest(candidate):
            report = validate_phase_learning_plan(candidate)
        if int((report.get("summary") or {}).get("error_count") or 0):
            raise ValueError("Phase-learning plan has blocking validation errors")
        revision_id = uuid.uuid4().hex
        revision = {
            "schema": 1,
            "format": PHASE_LEARNING_REVISION_FORMAT,
            "revision_id": revision_id,
            "recorded_utc": utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "plan": candidate,
            "validation": report,
            "scientific_scope": {
                "domain": "registered_visible_camera_plane_2d",
                "cross_series_phase_learning": True,
                "contains_depth_or_3d_extrapolation": False,
                "contains_concentration_or_mass": False,
            },
        }
        revision["content_sha256"] = _json_digest(
            {"plan": candidate, "validation": report}
        )
        destination = self.artifact_root / revision_id
        destination.mkdir(parents=True, exist_ok=False)
        _atomic_json(destination / "revision.json", revision)
        _append_jsonl(self.revisions_path, revision)
        _atomic_json(self.current_path, revision)
        self.project._audit(
            "phase_learning_plan_saved",
            actor,
            {
                "revision_id": revision_id,
                "content_sha256": revision["content_sha256"],
                "datasets": [row["dataset_id"] for row in report["datasets"]],
                "campaign_id": (report.get("model_bundle") or {}).get("campaign_id"),
            },
        )
        return deepcopy(revision)


@dataclass(frozen=True)
class SerializedPhaseHead:
    """One standardized logistic phase head."""

    name: str
    feature_contract: OpticalFeatureContract
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    threshold: float
    radiometry_profile: Mapping[str, Any] | None = None

    @classmethod
    def load(cls, path: Path) -> "SerializedPhaseHead":
        payload = _read_json(Path(path))
        contract = OpticalFeatureContract.from_dict(
            dict(payload.get("feature_contract") or {})
        )
        mean = np.asarray(payload.get("feature_mean") or [], dtype=np.float64)
        scale = np.asarray(payload.get("feature_scale") or [], dtype=np.float64)
        coefficients = np.asarray(payload.get("coefficients") or [], dtype=np.float64)
        if not (
            mean.size == scale.size == coefficients.size == len(contract.feature_names)
        ):
            raise ValueError(f"Invalid serialized phase-head dimensions: {path}")
        threshold = float(
            (
                (payload.get("training_provenance") or {}).get("cross_validation") or {}
            ).get("oof_threshold", 0.5)
        )
        return cls(
            name=Path(path).stem,
            feature_contract=contract,
            mean=mean,
            scale=np.maximum(scale, np.finfo(np.float64).eps),
            coefficients=coefficients,
            intercept=float(payload.get("intercept") or 0.0),
            threshold=threshold,
            radiometry_profile=(payload.get("training_provenance") or {}).get("radiometry_profile"),
        )

    def probability(self, features: np.ndarray) -> np.ndarray:
        from .phase.heads import SerializedPhaseHead as CanonicalHead
        return CanonicalHead.probability(self, features)


@dataclass(frozen=True)
class HierarchicalPhaseModel:
    """Two optical heads with an explicit mutually exclusive phase partition."""

    co2_presence: SerializedPhaseHead
    gas_within_co2: SerializedPhaseHead
    adapter: str
    bundle_root: Path

    @classmethod
    def load(cls, root: Path, *, adapter: str = "generic") -> "HierarchicalPhaseModel":
        bundle_root = Path(root).expanduser().resolve()
        inspect_phase_model_bundle(bundle_root)
        model_root = bundle_root / "models" / str(adapter)
        co2 = SerializedPhaseHead.load(model_root / "co2_presence_signal.json")
        gas = SerializedPhaseHead.load(model_root / "gas_within_co2_signal.json")
        if co2.feature_contract != gas.feature_contract:
            raise ValueError("Hierarchical model heads use different feature contracts")
        return cls(co2, gas, str(adapter), bundle_root)

    @property
    def feature_contract(self) -> OpticalFeatureContract:
        return self.co2_presence.feature_contract

    def predict_features(self, features: np.ndarray) -> dict[str, np.ndarray]:
        from .phase.heads import HierarchicalPhaseModel as CanonicalModel
        return CanonicalModel.predict_features(self, features)

    def predict_image(
        self,
        image_rgb: Any,
        baseline_rgb: Any,
        *,
        previous_rgb: Any | None = None,
        clearwater_rgb: Any | None = None,
        texture_sigma: float = 2.0,
        spatial_context_sigma: float | None = None,
        radiometry_profile: Mapping[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        from darsia.fluidflower.phase_training.radiometry import model_profile
        if radiometry_profile != model_profile(self):
            raise ValueError("Model requires its bound radiometric profile before image inference")
        workspace = prepare_optical_feature_workspace(
            image_rgb,
            baseline_rgb,
            previous_rgb=previous_rgb,
            clearwater_rgb=clearwater_rgb,
            texture_sigma=texture_sigma,
            spatial_context_sigma=spatial_context_sigma,
            contract=self.feature_contract,
        )
        shape = workspace.current.shape[:2]
        features = optical_feature_matrix(workspace, (...,)).reshape(
            -1, len(self.feature_contract.feature_names)
        )
        return {
            key: value.reshape(shape)
            for key, value in self.predict_features(features).items()
        }


__all__ = [
    "HierarchicalPhaseModel",
    "PHASE_ACTIVE_LEARNING_BATCH_FORMAT",
    "PHASE_ACTIVE_LEARNING_RESPONSE_FORMAT",
    "PHASE_LEARNING_PLAN_FORMAT",
    "PHASE_LEARNING_REVISION_FORMAT",
    "PHASE_MODEL_BUNDLE_FORMAT",
    "PHASE_MODEL_HEADS",
    "PHASE_REVIEW_CLASSES",
    "PhaseActiveLearningStore",
    "PhaseLearningPlanStore",
    "SerializedPhaseHead",
    "default_phase_learning_plan",
    "inspect_phase_model_bundle",
    "phase_uncertainty_map",
    "phase_learning_plan_from_bundle",
    "propose_uncertain_phase_regions",
    "validate_phase_learning_plan",
]
