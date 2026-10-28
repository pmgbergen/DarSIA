"""Reproducible supervised baselines for FluidFlower facies interpretation.

The baseline deliberately separates universal spatial roles from named material
classes. It operates in the registered two-dimensional camera plane; depth and
three-dimensional extrapolation remain explicit downstream DarSIA concerns.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold

from darsia.fluidflower.facies_learning import (
    FACIES_COLORS,
    HYDRAULIC_CLASS_DEFINITIONS,
    ROLE_IDS,
    collect_facies_annotation_bundles,
)
from darsia.fluidflower.project import sha256_file, utc_now

FACIES_MODEL_FORMAT = "darsia-fluidflower-facies-model-v1"
FACIES_MODEL_INDEX_FORMAT = "darsia-fluidflower-facies-model-index-v1"
FACIES_SPLIT_PLAN_FORMAT = "darsia-fluidflower-facies-split-plan-v1"
UNKNOWN_CLASS = np.uint16(65535)


@dataclass(frozen=True)
class FaciesModelConfig:
    """Configuration for the first transferable, non-neural baseline."""

    n_estimators: int = 128
    max_pixels_per_bundle: int = 120_000
    boundary_fraction: float = 0.35
    min_samples_leaf: int = 2
    cpu_workers: int = 0
    random_state: int = 20_260_901
    calibrate_probabilities: bool = True
    include_normalized_position: bool = False
    prediction_tile_size: int = 512
    prediction_halo: int = 18
    store_probabilities: bool = False

    def validated(self) -> FaciesModelConfig:
        if self.n_estimators < 8:
            raise ValueError("At least eight trees are required")
        if self.max_pixels_per_bundle < 100:
            raise ValueError("At least 100 pixels per bundle are required")
        if not 0.0 <= self.boundary_fraction <= 1.0:
            raise ValueError("boundary_fraction must be between zero and one")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive")
        if self.prediction_tile_size < 64:
            raise ValueError("prediction_tile_size must be at least 64 pixels")
        if self.prediction_halo < 14:
            raise ValueError("prediction_halo must be at least 14 pixels")
        return self


def facies_feature_names(*, include_normalized_position: bool = False) -> list[str]:
    """Return the stable, chemistry-agnostic feature schema."""

    names = [
        "rgb_r",
        "rgb_g",
        "rgb_b",
        "lab_l",
        "lab_a",
        "lab_b",
        "chromaticity_r",
        "chromaticity_g",
        "chromaticity_b",
        "log_ratio_r_g",
        "log_ratio_r_b",
        "log_ratio_g_b",
        "luminance",
    ]
    for radius in (2, 6, 14):
        names.extend((f"luminance_mean_r{radius}", f"luminance_std_r{radius}"))
    names.extend(
        (
            "rgb_residual_r_r6",
            "rgb_residual_g_r6",
            "rgb_residual_b_r6",
            "gradient_x",
            "gradient_y",
            "gradient_magnitude",
            "laplacian_abs",
        )
    )
    if include_normalized_position:
        names.extend(("normalized_x", "normalized_y"))
    return names


def _rgb_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError("Feature images must have at least three channels")
    rgb = np.asarray(array[..., :3], dtype=np.float32)
    if rgb.size and float(np.nanmax(rgb)) > 1.5:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def extract_facies_features(
    image: np.ndarray,
    *,
    flat_indices: np.ndarray | None = None,
    include_normalized_position: bool = False,
    origin_yx: tuple[int, int] = (0, 0),
    full_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Extract full-colour and texture features at selected image pixels.

    Filter channels are sampled one at a time. This avoids materializing a
    multi-gigabyte feature cube for full-resolution camera images.
    """

    rgb = _rgb_float(image)
    height, width = rgb.shape[:2]
    if flat_indices is None:
        indices = np.arange(height * width, dtype=np.int64)
    else:
        indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= height * width
        ):
            raise IndexError("Feature index lies outside the image")
    sampled_rgb = rgb.reshape(-1, 3)[indices]
    columns: list[np.ndarray] = [sampled_rgb[:, channel] for channel in range(3)]

    lab = cv2.cvtColor(sampled_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    columns.extend((lab[:, 0] / 100.0, lab[:, 1] / 128.0, lab[:, 2] / 128.0))

    total = np.maximum(sampled_rgb.sum(axis=1), 1.0e-6)
    columns.extend(sampled_rgb[:, channel] / total for channel in range(3))
    epsilon = 1.0e-4
    columns.extend(
        (
            np.log((sampled_rgb[:, 0] + epsilon) / (sampled_rgb[:, 1] + epsilon)),
            np.log((sampled_rgb[:, 0] + epsilon) / (sampled_rgb[:, 2] + epsilon)),
            np.log((sampled_rgb[:, 1] + epsilon) / (sampled_rgb[:, 2] + epsilon)),
        )
    )

    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    columns.append(luminance.reshape(-1)[indices])
    for radius in (2, 6, 14):
        kernel = 2 * radius + 1
        mean = cv2.boxFilter(
            luminance,
            cv2.CV_32F,
            (kernel, kernel),
            normalize=True,
            borderType=cv2.BORDER_REFLECT101,
        )
        mean_square = cv2.boxFilter(
            luminance * luminance,
            cv2.CV_32F,
            (kernel, kernel),
            normalize=True,
            borderType=cv2.BORDER_REFLECT101,
        )
        variance = np.maximum(mean_square - mean * mean, 0.0)
        columns.extend(
            (mean.reshape(-1)[indices], np.sqrt(variance).reshape(-1)[indices])
        )

    for channel in range(3):
        local = cv2.boxFilter(
            rgb[..., channel],
            cv2.CV_32F,
            (13, 13),
            normalize=True,
            borderType=cv2.BORDER_REFLECT101,
        )
        residual = rgb[..., channel] - local
        columns.append(residual.reshape(-1)[indices])

    gradient_x = cv2.Sobel(
        luminance, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101
    )
    gradient_y = cv2.Sobel(
        luminance, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101
    )
    columns.extend(
        (
            gradient_x.reshape(-1)[indices],
            gradient_y.reshape(-1)[indices],
            np.hypot(gradient_x, gradient_y).reshape(-1)[indices],
            np.abs(
                cv2.Laplacian(
                    luminance,
                    cv2.CV_32F,
                    ksize=3,
                    borderType=cv2.BORDER_REFLECT101,
                )
            ).reshape(-1)[indices],
        )
    )

    if include_normalized_position:
        rows, columns_index = np.divmod(indices, width)
        full_height, full_width = full_shape or (height, width)
        global_rows = rows + int(origin_yx[0])
        global_columns = columns_index + int(origin_yx[1])
        columns.extend(
            (
                global_columns / max(full_width - 1, 1),
                global_rows / max(full_height - 1, 1),
            )
        )

    features = np.column_stack(columns).astype(np.float32, copy=False)
    expected = len(
        facies_feature_names(include_normalized_position=include_normalized_position)
    )
    if features.shape[1] != expected:
        raise RuntimeError(
            f"Feature contract mismatch: {features.shape[1]} columns, expected {expected}"
        )
    return features


def _boundaries(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(target.shape, dtype=bool)
    boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
    boundary[:, :-1] |= target[:, 1:] != target[:, :-1]
    boundary[1:, :] |= target[1:, :] != target[:-1, :]
    boundary[:-1, :] |= target[1:, :] != target[:-1, :]
    return boundary & mask


def _sample_pixels(
    target: np.ndarray,
    mask: np.ndarray,
    *,
    maximum: int,
    boundary_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    classes = np.unique(target[mask])
    classes = classes[classes > 0]
    if classes.size == 0:
        return np.empty(0, dtype=np.int64)
    boundary = _boundaries(target, mask)
    per_class = max(1, maximum // int(classes.size))
    selected: list[np.ndarray] = []
    flat_target = target.reshape(-1)
    flat_mask = mask.reshape(-1)
    flat_boundary = boundary.reshape(-1)
    for class_id in classes:
        candidates = np.flatnonzero(flat_mask & (flat_target == class_id))
        if candidates.size <= per_class:
            selected.append(candidates)
            continue
        edge_candidates = candidates[flat_boundary[candidates]]
        edge_count = min(
            edge_candidates.size,
            round(per_class * boundary_fraction),
        )
        edges = (
            rng.choice(edge_candidates, edge_count, replace=False)
            if edge_count
            else np.empty(0, dtype=np.int64)
        )
        remaining_mask = np.ones(candidates.size, dtype=bool)
        if edges.size:
            remaining_mask[np.isin(candidates, edges, assume_unique=False)] = False
        remaining = candidates[remaining_mask]
        interior_count = min(per_class - edge_count, remaining.size)
        interior = (
            rng.choice(remaining, interior_count, replace=False)
            if interior_count
            else np.empty(0, dtype=np.int64)
        )
        selected.append(np.concatenate((edges, interior)))
    output = np.concatenate(selected)
    rng.shuffle(output)
    return output.astype(np.int64, copy=False)


def _load_npz_array(path: Path, key: str = "array") -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        selected = key if key in archive.files else archive.files[0]
        return np.asarray(archive[selected])


def _bundle_paths(record: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = Path(str(record["manifest_path"])).resolve()
    manifest = dict(record.get("manifest") or {})
    if not manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    root = manifest_path.parent
    baseline_record = next(
        frame
        for frame in manifest.get("frames") or []
        if frame.get("name") == "baseline"
    )
    return root / str(baseline_record["relative_path"]), root / "targets.npz", manifest


def _material_local_to_global(
    manifest: Mapping[str, Any], global_ids: Mapping[str, int]
) -> tuple[dict[int, int], set[int]]:
    mapping: dict[int, int] = {}
    eligible_local: set[int] = set()
    for row in manifest.get("material_classes") or []:
        if not bool(row.get("eligible_for_cross_series_training")):
            continue
        local_id = int(row["class_id"])
        eligible_local.add(local_id)
        key = str(row.get("canonical_key") or "")
        if key in global_ids:
            mapping[local_id] = int(global_ids[key])
    return mapping, eligible_local


def _remap_materials(
    local: np.ndarray,
    manifest: Mapping[str, Any],
    global_ids: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapping, eligible_local = _material_local_to_global(manifest, global_ids)
    target = np.zeros(local.shape, dtype=np.uint16)
    eligible = np.zeros(local.shape, dtype=bool)
    seen = np.zeros(local.shape, dtype=bool)
    for local_id in eligible_local:
        selected = local == local_id
        eligible[selected] = True
        if local_id in mapping:
            target[selected] = mapping[local_id]
            seen[selected] = True
        else:
            target[selected] = UNKNOWN_CLASS
    return target, eligible, seen


def _global_material_catalog(
    records: Sequence[Mapping[str, Any]], *, split: str = "train"
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    entries: dict[str, dict[str, Any]] = {}
    for record in records:
        if str(record.get("split") or "") != split:
            continue
        manifest = dict(record.get("manifest") or {})
        for raw in manifest.get("material_classes") or []:
            row = dict(raw)
            if not bool(row.get("eligible_for_cross_series_training")):
                continue
            key = str(row.get("canonical_key") or "")
            if not key:
                continue
            entries.setdefault(
                key,
                {
                    "canonical_key": key,
                    "name": str(row.get("name") or key),
                    "analysis_role": str(row.get("analysis_role") or ""),
                },
            )
    keys = sorted(entries)
    ids = {key: index for index, key in enumerate(keys, start=1)}
    catalog = [{**entries[key], "class_id": ids[key]} for key in keys]
    return ids, catalog


def _deduplicate_bundle_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    assignments: dict[str, tuple[str, str]] = {}
    skipped = 0
    for raw_record in records:
        record = dict(raw_record)
        manifest = dict(record.get("manifest") or {})
        artifacts = dict(manifest.get("artifacts") or {})
        identity = _json_digest(
            {
                "baseline_sha256": str(
                    (artifacts.get("baseline") or {}).get("sha256") or ""
                ),
                "targets_sha256": str(
                    (artifacts.get("targets") or {}).get("sha256") or ""
                ),
                "image_correction_revision_id": str(
                    (manifest.get("image_correction") or {}).get("revision_id") or ""
                ),
            }
        )
        assignment = (
            str(record.get("group_id") or ""),
            str(record.get("split") or ""),
        )
        previous = assignments.get(identity)
        if previous is not None and previous != assignment:
            raise ValueError(
                "Identical facies sample content occurs in conflicting leakage "
                f"assignments: {previous} and {assignment}"
            )
        if previous is not None:
            skipped += 1
            continue
        assignments[identity] = assignment
        record["sample_identity_sha256"] = identity
        unique.append(record)
    return unique, skipped


def _apply_split_plan(
    records: Sequence[Mapping[str, Any]],
    split_plan: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Assign complete leakage groups without duplicating source bundles."""

    copied = [dict(record) for record in records]
    if split_plan is None:
        return copied, None
    plan = dict(split_plan)
    if plan.get("format") != FACIES_SPLIT_PLAN_FORMAT:
        raise ValueError("Unsupported facies split-plan format")
    raw_assignments = plan.get("assignments")
    if not isinstance(raw_assignments, Mapping) or not raw_assignments:
        raise ValueError("A facies split plan requires non-empty group assignments")

    allowed = {"train", "validation", "holdout"}
    assignments: dict[str, str] = {}
    display_names: dict[str, str] = {}
    for raw_group, raw_split in raw_assignments.items():
        group = str(raw_group).strip()
        normalized_group = group.casefold()
        split = str(raw_split).strip().casefold()
        if not normalized_group:
            raise ValueError("Facies split-plan group IDs cannot be empty")
        if split not in allowed:
            raise ValueError(f"Unsupported split {raw_split!r} for group {raw_group!r}")
        if normalized_group in assignments:
            raise ValueError(f"Duplicate facies split-plan group: {group!r}")
        assignments[normalized_group] = split
        display_names[normalized_group] = group

    available_groups = {
        str(record.get("group_id") or "").strip().casefold() for record in copied
    }
    unknown = sorted(set(assignments) - available_groups)
    if unknown:
        raise ValueError(
            "Facies split plan references unavailable groups: "
            + ", ".join(display_names[group] for group in unknown)
        )

    selected: list[dict[str, Any]] = []
    for record in copied:
        group = str(record.get("group_id") or "").strip().casefold()
        if group not in assignments:
            continue
        record["source_split"] = str(record.get("split") or "")
        record["split"] = assignments[group]
        selected.append(record)
    canonical = {
        "format": FACIES_SPLIT_PLAN_FORMAT,
        "name": str(plan.get("name") or "unnamed-split-plan").strip(),
        "purpose": str(plan.get("purpose") or "").strip(),
        "assignments": {
            display_names[group]: assignments[group] for group in sorted(assignments)
        },
    }
    canonical["content_sha256"] = _json_digest(canonical)
    return selected, canonical


def _base_estimator(config: FaciesModelConfig) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=config.cpu_workers or -1,
        random_state=config.random_state,
    )


def _calibration_splits(
    target: np.ndarray, groups: np.ndarray, random_state: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, str]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        return None, "fewer than three independent training groups"
    n_splits = min(3, int(unique_groups.size))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    classes = set(np.unique(target).tolist())
    splits = list(splitter.split(np.zeros(target.shape), target, groups))
    for train_index, test_index in splits:
        if set(np.unique(target[train_index]).tolist()) != classes:
            return None, "a calibration training fold does not contain every class"
        if set(np.unique(target[test_index]).tolist()) != classes:
            return None, "a calibration test fold does not contain every class"
    return splits, "group-isolated sigmoid calibration"


def _fit_head(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    config: FaciesModelConfig,
) -> tuple[Any, dict[str, Any]]:
    classes = np.unique(target)
    if classes.size < 2:
        raise ValueError("A supervised head requires at least two training classes")
    estimator: Any = _base_estimator(config)
    calibration = {
        "requested": bool(config.calibrate_probabilities),
        "applied": False,
        "method": None,
        "reason": "disabled by configuration",
    }
    if config.calibrate_probabilities:
        splits, reason = _calibration_splits(target, groups, config.random_state)
        if splits:
            estimator = CalibratedClassifierCV(
                estimator=estimator,
                method="sigmoid",
                cv=splits,
                n_jobs=config.cpu_workers or None,
                ensemble="auto",
            )
            calibration.update({"applied": True, "method": "sigmoid", "reason": reason})
        else:
            calibration["reason"] = reason
    estimator.fit(features, target)
    calibration["class_ids"] = [int(value) for value in estimator.classes_]
    return estimator, calibration


def _predict_proba(estimator: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(features), dtype=np.float32)
    prediction = np.asarray(estimator.predict(features))
    classes = np.asarray(estimator.classes_)
    return (prediction[:, None] == classes[None, :]).astype(np.float32)


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    count = probabilities.shape[1]
    if count <= 1:
        return np.zeros(probabilities.shape[0], dtype=np.float32)
    clipped = np.clip(probabilities, 1.0e-7, 1.0)
    return (-np.sum(clipped * np.log(clipped), axis=1) / np.log(float(count))).astype(
        np.float32
    )


def _predict_image(
    image: np.ndarray,
    estimators: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    config: FaciesModelConfig,
) -> dict[str, np.ndarray]:
    rgb = _rgb_float(image)
    height, width = rgb.shape[:2]
    output: dict[str, np.ndarray] = {}
    for head, estimator in estimators.items():
        mask = np.asarray(masks[head], dtype=bool)
        prediction = np.zeros((height, width), dtype=np.uint16)
        confidence = np.zeros((height, width), dtype=np.float16)
        uncertainty = np.zeros((height, width), dtype=np.float16)
        probabilities_output: np.ndarray | None = None
        classes = np.asarray(estimator.classes_, dtype=np.uint16)
        if config.store_probabilities:
            probabilities_output = np.zeros(
                (height, width, classes.size), dtype=np.float16
            )
        tile_size = config.prediction_tile_size
        halo = config.prediction_halo
        for y0 in range(0, height, tile_size):
            y1 = min(y0 + tile_size, height)
            for x0 in range(0, width, tile_size):
                x1 = min(x0 + tile_size, width)
                tile_mask = mask[y0:y1, x0:x1]
                if not np.any(tile_mask):
                    continue
                ey0 = max(0, y0 - halo)
                ey1 = min(height, y1 + halo)
                ex0 = max(0, x0 - halo)
                ex1 = min(width, x1 + halo)
                local_rows, local_columns = np.nonzero(tile_mask)
                expanded_rows = local_rows + (y0 - ey0)
                expanded_columns = local_columns + (x0 - ex0)
                expanded_width = ex1 - ex0
                indices = expanded_rows * expanded_width + expanded_columns
                features = extract_facies_features(
                    rgb[ey0:ey1, ex0:ex1],
                    flat_indices=indices,
                    include_normalized_position=config.include_normalized_position,
                    origin_yx=(ey0, ex0),
                    full_shape=(height, width),
                )
                probabilities = _predict_proba(estimator, features)
                winner = np.argmax(probabilities, axis=1)
                global_rows = local_rows + y0
                global_columns = local_columns + x0
                prediction[global_rows, global_columns] = classes[winner]
                confidence[global_rows, global_columns] = np.max(
                    probabilities, axis=1
                ).astype(np.float16)
                uncertainty[global_rows, global_columns] = _normalized_entropy(
                    probabilities
                ).astype(np.float16)
                if probabilities_output is not None:
                    probabilities_output[global_rows, global_columns, :] = (
                        probabilities.astype(np.float16)
                    )
        output[f"{head}_prediction"] = prediction
        output[f"{head}_confidence"] = confidence
        output[f"{head}_uncertainty"] = uncertainty
        output[f"{head}_class_ids"] = classes
        if probabilities_output is not None:
            output[f"{head}_probabilities"] = probabilities_output
    return output


def predict_facies_image(
    image: np.ndarray,
    estimators: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    *,
    config: FaciesModelConfig | None = None,
) -> dict[str, np.ndarray]:
    """Predict facies heads through the shared tiled inference path."""

    settings = (config or FaciesModelConfig()).validated()
    if set(estimators) != set(masks):
        raise ValueError("Estimator and class-mask heads must match.")
    return _predict_image(image, estimators, masks, settings)


def _confusion(
    truth: np.ndarray, prediction: np.ndarray, class_ids: Sequence[int]
) -> np.ndarray:
    ids = np.asarray([int(value) for value in class_ids], dtype=np.int64)
    flat_truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    flat_prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    truth_index = np.searchsorted(ids, flat_truth)
    prediction_index = np.searchsorted(ids, flat_prediction)
    valid = (
        (truth_index < ids.size)
        & (prediction_index < ids.size)
        & (ids[np.minimum(truth_index, ids.size - 1)] == flat_truth)
        & (ids[np.minimum(prediction_index, ids.size - 1)] == flat_prediction)
    )
    encoded = truth_index[valid] * ids.size + prediction_index[valid]
    return np.bincount(encoded, minlength=ids.size**2).reshape(ids.size, ids.size)


def _metrics_from_confusion(
    matrix: np.ndarray,
    class_ids: Sequence[int],
    names: Mapping[int, str],
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    recall = np.divide(
        diagonal,
        true_counts,
        out=np.full(diagonal.shape, np.nan),
        where=true_counts > 0,
    )
    precision = np.divide(
        diagonal,
        predicted_counts,
        out=np.full(diagonal.shape, np.nan),
        where=predicted_counts > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.full(diagonal.shape, np.nan),
        where=(precision + recall) > 0,
    )
    union = true_counts + predicted_counts - diagonal
    iou = np.divide(
        diagonal,
        union,
        out=np.full(diagonal.shape, np.nan),
        where=union > 0,
    )
    total = float(matrix.sum())
    per_class = []
    for index, class_id in enumerate(class_ids):
        per_class.append(
            {
                "class_id": int(class_id),
                "name": str(names.get(int(class_id), class_id)),
                "support": int(true_counts[index]),
                "precision": (
                    None if np.isnan(precision[index]) else float(precision[index])
                ),
                "recall": None if np.isnan(recall[index]) else float(recall[index]),
                "f1": None if np.isnan(f1[index]) else float(f1[index]),
                "iou": None if np.isnan(iou[index]) else float(iou[index]),
            }
        )

    def finite_mean(values: np.ndarray) -> float | None:
        finite = np.asarray(values)[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else None

    return {
        "evaluated_pixels": int(total),
        "accuracy": float(diagonal.sum() / total) if total else None,
        "balanced_accuracy": finite_mean(recall),
        "macro_f1": finite_mean(f1),
        "mean_iou": finite_mean(iou),
        "confusion_matrix": matrix.astype(np.int64).tolist(),
        "per_class": per_class,
    }


def _boundary_f1(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    tolerance_pixels: int = 2,
) -> dict[str, Any]:
    true_boundary = _boundaries(truth, mask)
    predicted_boundary = _boundaries(prediction, mask)
    kernel = np.ones((2 * tolerance_pixels + 1,) * 2, dtype=np.uint8)
    true_dilated = cv2.dilate(true_boundary.astype(np.uint8), kernel) > 0
    predicted_dilated = cv2.dilate(predicted_boundary.astype(np.uint8), kernel) > 0
    true_count = int(true_boundary.sum())
    predicted_count = int(predicted_boundary.sum())
    precision = (
        float((predicted_boundary & true_dilated).sum() / predicted_count)
        if predicted_count
        else None
    )
    recall = (
        float((true_boundary & predicted_dilated).sum() / true_count)
        if true_count
        else None
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "tolerance_pixels": tolerance_pixels,
        "true_boundary_pixels": true_count,
        "predicted_boundary_pixels": predicted_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _head_evaluation(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    class_ids: Sequence[int],
    names: Mapping[int, str],
    uncertainty: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    selected_truth = truth[mask]
    selected_prediction = prediction[mask]
    matrix = _confusion(selected_truth, selected_prediction, class_ids)
    metrics = _metrics_from_confusion(matrix, class_ids, names)
    metrics["boundary"] = _boundary_f1(truth, prediction, mask)
    selected_uncertainty = np.asarray(uncertainty[mask], dtype=np.float32)
    metrics["uncertainty"] = {
        "mean_normalized_entropy": (
            float(np.mean(selected_uncertainty)) if selected_uncertainty.size else None
        ),
        "p95_normalized_entropy": (
            float(np.quantile(selected_uncertainty, 0.95))
            if selected_uncertainty.size
            else None
        ),
    }
    return metrics, matrix


def _prediction_preview(
    image: np.ndarray,
    role_prediction: np.ndarray,
    material_prediction: np.ndarray | None,
    hydraulic_prediction: np.ndarray | None,
    role_uncertainty: np.ndarray,
    hydraulic_uncertainty: np.ndarray | None,
) -> np.ndarray:
    base = np.rint(255.0 * _rgb_float(image)).astype(np.uint8)

    def overlay(target: np.ndarray) -> np.ndarray:
        colors = FACIES_COLORS[np.mod(target, len(FACIES_COLORS))]
        result = base.copy()
        valid = target > 0
        result[valid] = np.rint(0.45 * base[valid] + 0.55 * colors[valid]).astype(
            np.uint8
        )
        return result

    role = overlay(role_prediction)
    material = overlay(
        material_prediction if material_prediction is not None else role_prediction
    )
    hydraulic = overlay(
        hydraulic_prediction if hydraulic_prediction is not None else role_prediction
    )
    role_heat = cv2.applyColorMap(
        np.rint(255.0 * np.asarray(role_uncertainty, dtype=np.float32)).astype(
            np.uint8
        ),
        cv2.COLORMAP_INFERNO,
    )[..., ::-1]
    hydraulic_heat = cv2.applyColorMap(
        np.rint(
            255.0
            * np.asarray(
                (
                    hydraulic_uncertainty
                    if hydraulic_uncertainty is not None
                    else role_uncertainty
                ),
                dtype=np.float32,
            )
        ).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )[..., ::-1]
    height, width = base.shape[:2]
    panels = [base, role, material, hydraulic, role_heat, hydraulic_heat]
    scale = min(1.0, 550.0 / width, 400.0 / height)
    if scale < 1.0:
        panels = [
            cv2.resize(
                panel,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            for panel in panels
        ]
    canvas = Image.new("RGB", (panels[0].shape[1] * 3, panels[0].shape[0] * 2), "black")
    labels = (
        "Bundle baseline",
        "Role prediction",
        "Named material prediction",
        "Hydraulic-class prediction",
        "Role uncertainty",
        "Hydraulic uncertainty",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = (index % 3) * panels[0].shape[1]
        y = (index // 3) * panels[0].shape[0]
        canvas.paste(Image.fromarray(panel), (x, y))
        draw.rectangle((x, y, x + 150, y + 20), fill=(20, 23, 26))
        draw.text((x + 5, y + 4), label, fill=(245, 245, 245))
    return np.asarray(canvas)


def _json_digest(payload: Mapping[str, Any]) -> str:
    from darsia.fluidflower.facies_learning import _json_digest as _canonical_mapping_digest
    return _canonical_mapping_digest(payload)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _software_versions() -> dict[str, str]:
    versions = {"python": __import__("platform").python_version()}
    for package in ("darsia", "numpy", "opencv-python", "scikit-learn", "joblib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "source-tree"
    return versions


def _model_card_markdown(manifest: Mapping[str, Any]) -> str:
    metrics = dict(manifest.get("metrics_summary") or {})
    lines = [
        "# FluidFlower facies baseline",
        "",
        f"Model ID: `{manifest.get('model_id')}`  ",
        f"Created: `{manifest.get('created_utc')}`",
        "",
        "## Intended use",
        "",
        (
            "Supervised interpretation of corrected FluidFlower baseline images in "
            "the registered two-dimensional camera plane."
        ),
        "",
        "## Scientific scope",
        "",
        "- Universal role and canonical material heads are trained separately.",
        "- Whole leakage groups are assigned to exactly one dataset split.",
        "- Holdout groups are never used for fitting or calibration.",
        (
            "- Depth, 2D-to-3D conversion, phase interpretation, concentration, and "
            "mass remain outside this model."
        ),
        (
            "- This ExtraTrees baseline is a research baseline, not a validated "
            "production model."
        ),
        "",
        "## Evaluation summary",
        "",
        f"- Validation bundles: {metrics.get('validation_bundle_count', 0)}",
        f"- Holdout bundles: {metrics.get('holdout_bundle_count', 0)}",
        f"- Role holdout mean IoU: {metrics.get('holdout_role_mean_iou')}",
        f"- Material holdout mean IoU: {metrics.get('holdout_material_mean_iou')}",
        f"- Hydraulic holdout mean IoU: {metrics.get('holdout_hydraulic_mean_iou')}",
        "",
        (
            "See `manifest.json` and `metrics.json` for complete provenance and "
            "per-class results."
        ),
    ]
    return "\n".join(lines) + "\n"


class FaciesModelStore:
    """Read and verify immutable supervised-model artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.index_path = self.root / "index.jsonl"

    def records(self) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"Facies model record at line {line_number} is not an object"
                    )
                rows.append(payload)
        return rows

    def verify(self, record: Mapping[str, Any]) -> dict[str, Any]:
        manifest_path = self.root / str(record.get("manifest_relpath") or "")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("format") != FACIES_MODEL_FORMAT:
            raise ValueError(f"Unsupported facies model: {manifest_path}")
        for name in ("model", "metrics"):
            artifact = dict((manifest.get("artifacts") or {}).get(name) or {})
            path = manifest_path.parent / str(artifact.get("relative_path") or "")
            if not path.is_file() or sha256_file(path) != str(artifact.get("sha256")):
                raise ValueError(f"Facies model artifact failed verification: {path}")
        return manifest


def train_facies_baseline(
    dataset_roots: Sequence[Path],
    output_root: Path,
    *,
    config: FaciesModelConfig | None = None,
    split_plan: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and persist a transferable facies baseline."""

    settings = (config or FaciesModelConfig()).validated()

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    report("Collecting and verifying annotation bundles")
    collected_records = collect_facies_annotation_bundles(dataset_roots, verify=True)
    planned_records, canonical_split_plan = _apply_split_plan(
        collected_records, split_plan
    )
    records, duplicate_record_count = _deduplicate_bundle_records(planned_records)
    if not records:
        raise ValueError("No approved facies annotation bundles were found")
    split_groups: dict[str, set[str]] = {
        split: {
            str(record.get("group_id") or "")
            for record in records
            if str(record.get("split") or "") == split
        }
        for split in ("train", "validation", "holdout")
    }
    if not split_groups["train"]:
        raise ValueError("At least one approved training group is required")
    if not split_groups["validation"] and not split_groups["holdout"]:
        raise ValueError(
            "At least one independent validation or holdout group is required"
        )

    material_ids, material_catalog = _global_material_catalog(records)
    role_names = {value: key for key, value in ROLE_IDS.items() if value > 0}
    material_names = {
        int(row["class_id"]): str(row["canonical_key"]) for row in material_catalog
    }
    hydraulic_catalog = [dict(row) for row in HYDRAULIC_CLASS_DEFINITIONS]
    hydraulic_names = {
        int(row["class_id"]): str(row["canonical_key"]) for row in hydraulic_catalog
    }
    rng = np.random.default_rng(settings.random_state)
    samples: dict[str, list[np.ndarray]] = {
        "features": [],
        "roles": [],
        "role_groups": [],
        "materials": [],
        "material_features": [],
        "material_groups": [],
        "hydraulic": [],
        "hydraulic_features": [],
        "hydraulic_groups": [],
    }
    training_rows = [
        record for record in records if str(record.get("split") or "") == "train"
    ]
    for index, record in enumerate(training_rows, start=1):
        report(f"Sampling training bundle {index}/{len(training_rows)}")
        baseline_path, targets_path, manifest = _bundle_paths(record)
        image = _load_npz_array(baseline_path)
        with np.load(targets_path, allow_pickle=False) as archive:
            roles = np.asarray(archive["roles"], dtype=np.uint16)
            context = np.asarray(archive["context_mask"], dtype=bool)
            local_materials = np.asarray(archive["material_classes"], dtype=np.uint16)
            hydraulic = np.asarray(archive["hydraulic_classes"], dtype=np.uint16)
        role_indices = _sample_pixels(
            roles,
            context & (roles > 0),
            maximum=settings.max_pixels_per_bundle,
            boundary_fraction=settings.boundary_fraction,
            rng=rng,
        )
        remapped, eligible, seen = _remap_materials(
            local_materials, manifest, material_ids
        )
        material_indices = _sample_pixels(
            remapped,
            eligible & seen,
            maximum=settings.max_pixels_per_bundle,
            boundary_fraction=settings.boundary_fraction,
            rng=rng,
        )
        hydraulic_indices = _sample_pixels(
            hydraulic,
            hydraulic > 0,
            maximum=settings.max_pixels_per_bundle,
            boundary_fraction=settings.boundary_fraction,
            rng=rng,
        )
        selected_sets = [
            indices
            for indices in (role_indices, material_indices, hydraulic_indices)
            if indices.size
        ]
        if not selected_sets:
            continue
        all_indices = np.unique(np.concatenate(selected_sets))
        all_features = extract_facies_features(
            image,
            flat_indices=all_indices,
            include_normalized_position=settings.include_normalized_position,
        )

        if role_indices.size:
            samples["features"].append(
                all_features[np.searchsorted(all_indices, role_indices)]
            )
            samples["roles"].append(roles.reshape(-1)[role_indices])
            samples["role_groups"].append(
                np.full(role_indices.size, str(record["group_id"]), dtype=object)
            )
        if material_indices.size:
            samples["material_features"].append(
                all_features[np.searchsorted(all_indices, material_indices)]
            )
            samples["materials"].append(remapped.reshape(-1)[material_indices])
            samples["material_groups"].append(
                np.full(material_indices.size, str(record["group_id"]), dtype=object)
            )
        if hydraulic_indices.size:
            samples["hydraulic_features"].append(
                all_features[np.searchsorted(all_indices, hydraulic_indices)]
            )
            samples["hydraulic"].append(hydraulic.reshape(-1)[hydraulic_indices])
            samples["hydraulic_groups"].append(
                np.full(hydraulic_indices.size, str(record["group_id"]), dtype=object)
            )

    if not samples["features"]:
        raise ValueError("Training bundles contain no eligible universal role pixels")
    role_features = np.concatenate(samples["features"])
    role_target = np.concatenate(samples["roles"])
    role_groups = np.concatenate(samples["role_groups"])
    report(f"Fitting universal role head on {role_target.size:,} sampled pixels")
    role_model, role_calibration = _fit_head(
        role_features, role_target, role_groups, settings
    )
    role_sample_count = int(role_target.size)

    material_model: Any | None = None
    material_calibration: dict[str, Any] = {
        "requested": bool(settings.calibrate_probabilities),
        "applied": False,
        "reason": "no cross-series-eligible named material samples",
        "class_ids": [],
    }
    material_sample_count = 0
    if samples["material_features"]:
        material_features = np.concatenate(samples["material_features"])
        material_target = np.concatenate(samples["materials"])
        material_groups = np.concatenate(samples["material_groups"])
        material_sample_count = int(material_target.size)
        if np.unique(material_target).size >= 2:
            report(
                f"Fitting canonical material head on {material_target.size:,} sampled pixels"
            )
            material_model, material_calibration = _fit_head(
                material_features, material_target, material_groups, settings
            )
        else:
            material_calibration["reason"] = (
                "fewer than two cross-series-eligible material classes"
            )

    hydraulic_model: Any | None = None
    hydraulic_calibration: dict[str, Any] = {
        "requested": bool(settings.calibrate_probabilities),
        "applied": False,
        "reason": "no permeability classes with an explicit convertible unit",
        "class_ids": [],
    }
    hydraulic_sample_count = 0
    if samples["hydraulic_features"]:
        hydraulic_features = np.concatenate(samples["hydraulic_features"])
        hydraulic_target = np.concatenate(samples["hydraulic"])
        hydraulic_groups = np.concatenate(samples["hydraulic_groups"])
        hydraulic_sample_count = int(hydraulic_target.size)
        if np.unique(hydraulic_target).size >= 2:
            report(
                "Fitting transferable hydraulic-class head on "
                f"{hydraulic_target.size:,} sampled pixels"
            )
            hydraulic_model, hydraulic_calibration = _fit_head(
                hydraulic_features,
                hydraulic_target,
                hydraulic_groups,
                settings,
            )
        else:
            hydraulic_calibration["reason"] = (
                "fewer than two represented permeability classes"
            )

    del samples, role_features, role_target, role_groups
    estimators = {"role": role_model}
    if material_model is not None:
        estimators["material"] = material_model
    if hydraulic_model is not None:
        estimators["hydraulic"] = hydraulic_model

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_id = uuid.uuid4().hex
    staging = output_root / f".{model_id}.{uuid.uuid4().hex}.tmp"
    destination = output_root / model_id
    staging.mkdir(parents=True, exist_ok=False)
    all_metrics: list[dict[str, Any]] = []
    aggregate_confusion: dict[tuple[str, str], np.ndarray] = {}
    try:
        model_payload = {
            "format": FACIES_MODEL_FORMAT,
            "feature_names": facies_feature_names(
                include_normalized_position=settings.include_normalized_position
            ),
            "config": asdict(settings),
            "role_model": role_model,
            "material_model": material_model,
            "hydraulic_model": hydraulic_model,
            "role_names": role_names,
            "material_catalog": material_catalog,
            "hydraulic_catalog": hydraulic_catalog,
            "split_plan": canonical_split_plan,
        }
        joblib.dump(model_payload, staging / "model.joblib", compress=3)
        predictions_root = staging / "predictions"
        predictions_root.mkdir()
        evaluation_rows = [
            record
            for record in records
            if str(record.get("split") or "") in {"validation", "holdout"}
        ]
        for index, record in enumerate(evaluation_rows, start=1):
            report(f"Evaluating bundle {index}/{len(evaluation_rows)}")
            baseline_path, targets_path, manifest = _bundle_paths(record)
            image = _load_npz_array(baseline_path)
            with np.load(targets_path, allow_pickle=False) as archive:
                roles = np.asarray(archive["roles"], dtype=np.uint16)
                context = np.asarray(archive["context_mask"], dtype=bool)
                local_materials = np.asarray(
                    archive["material_classes"], dtype=np.uint16
                )
                hydraulic = np.asarray(archive["hydraulic_classes"], dtype=np.uint16)
            remapped, material_eligible, material_seen = _remap_materials(
                local_materials, manifest, material_ids
            )
            masks = {"role": context & (roles > 0)}
            if material_model is not None:
                masks["material"] = material_eligible & material_seen
            if hydraulic_model is not None:
                masks["hydraulic"] = hydraulic > 0
            prediction = _predict_image(image, estimators, masks, settings)
            bundle_metrics: dict[str, Any] = {
                "bundle_id": str(record["bundle_id"]),
                "project_id": str(record.get("project_id") or ""),
                "series_id": str(record.get("series_id") or ""),
                "group_id": str(record.get("group_id") or ""),
                "split": str(record.get("split") or ""),
            }
            role_class_ids = [int(value) for value in role_model.classes_]
            role_mask = masks["role"] & np.isin(roles, role_class_ids)
            role_metrics, role_matrix = _head_evaluation(
                roles,
                prediction["role_prediction"],
                role_mask,
                role_class_ids,
                role_names,
                prediction["role_uncertainty"],
            )
            role_metrics["unseen_truth_fraction"] = float(
                1.0 - role_mask.sum() / max(masks["role"].sum(), 1)
            )
            bundle_metrics["role"] = role_metrics
            key = (bundle_metrics["split"], "role")
            aggregate_confusion[key] = (
                aggregate_confusion.get(key, np.zeros_like(role_matrix)) + role_matrix
            )

            if material_model is not None:
                material_class_ids = [int(value) for value in material_model.classes_]
                material_metrics, material_matrix = _head_evaluation(
                    remapped,
                    prediction["material_prediction"],
                    masks["material"],
                    material_class_ids,
                    material_names,
                    prediction["material_uncertainty"],
                )
                material_metrics["eligible_pixels"] = int(material_eligible.sum())
                material_metrics["seen_material_coverage"] = float(
                    material_seen.sum() / max(material_eligible.sum(), 1)
                )
                material_metrics["unseen_truth_fraction"] = float(
                    1.0 - material_seen.sum() / max(material_eligible.sum(), 1)
                )
                bundle_metrics["material"] = material_metrics
                key = (bundle_metrics["split"], "material")
                aggregate_confusion[key] = (
                    aggregate_confusion.get(key, np.zeros_like(material_matrix))
                    + material_matrix
                )
            if hydraulic_model is not None:
                hydraulic_class_ids = [int(value) for value in hydraulic_model.classes_]
                hydraulic_mask = masks["hydraulic"] & np.isin(
                    hydraulic, hydraulic_class_ids
                )
                hydraulic_metrics, hydraulic_matrix = _head_evaluation(
                    hydraulic,
                    prediction["hydraulic_prediction"],
                    hydraulic_mask,
                    hydraulic_class_ids,
                    hydraulic_names,
                    prediction["hydraulic_uncertainty"],
                )
                hydraulic_metrics["eligible_pixels"] = int(masks["hydraulic"].sum())
                hydraulic_metrics["seen_class_coverage"] = float(
                    hydraulic_mask.sum() / max(masks["hydraulic"].sum(), 1)
                )
                hydraulic_metrics["unseen_truth_fraction"] = float(
                    1.0 - hydraulic_mask.sum() / max(masks["hydraulic"].sum(), 1)
                )
                bundle_metrics["hydraulic"] = hydraulic_metrics
                key = (bundle_metrics["split"], "hydraulic")
                aggregate_confusion[key] = (
                    aggregate_confusion.get(key, np.zeros_like(hydraulic_matrix))
                    + hydraulic_matrix
                )
            all_metrics.append(bundle_metrics)

            bundle_root = predictions_root / str(record["bundle_id"])
            bundle_root.mkdir()
            np.savez_compressed(bundle_root / "prediction.npz", **prediction)
            _atomic_json(bundle_root / "metrics.json", bundle_metrics)
            preview = _prediction_preview(
                image,
                prediction["role_prediction"],
                prediction.get("material_prediction"),
                prediction.get("hydraulic_prediction"),
                prediction["role_uncertainty"],
                prediction.get("hydraulic_uncertainty"),
            )
            Image.fromarray(preview).save(bundle_root / "preview.png")

        aggregate: dict[str, Any] = {}
        for (split, head), matrix in aggregate_confusion.items():
            if head == "role":
                class_ids = [int(value) for value in role_model.classes_]
                names = role_names
            elif head == "material":
                class_ids = [int(value) for value in material_model.classes_]
                names = material_names
            else:
                class_ids = [int(value) for value in hydraulic_model.classes_]
                names = hydraulic_names
            aggregate.setdefault(split, {})[head] = _metrics_from_confusion(
                matrix, class_ids, names
            )
        metrics_payload = {"bundles": all_metrics, "aggregate": aggregate}
        _atomic_json(staging / "metrics.json", metrics_payload)

        metrics_summary = {
            "validation_bundle_count": sum(
                row["split"] == "validation" for row in all_metrics
            ),
            "holdout_bundle_count": sum(
                row["split"] == "holdout" for row in all_metrics
            ),
            "holdout_role_mean_iou": (
                (aggregate.get("holdout") or {}).get("role") or {}
            ).get("mean_iou"),
            "holdout_material_mean_iou": (
                (aggregate.get("holdout") or {}).get("material") or {}
            ).get("mean_iou"),
            "holdout_hydraulic_mean_iou": (
                (aggregate.get("holdout") or {}).get("hydraulic") or {}
            ).get("mean_iou"),
        }
        model_reference = {
            "relative_path": "model.joblib",
            "sha256": sha256_file(staging / "model.joblib"),
            "size_bytes": int((staging / "model.joblib").stat().st_size),
        }
        content_payload = {
            "format": FACIES_MODEL_FORMAT,
            "bundle_content_sha256": sorted(
                str(record.get("content_sha256") or "") for record in records
            ),
            "config": asdict(settings),
            "feature_names": model_payload["feature_names"],
            "split_plan_sha256": (
                None
                if canonical_split_plan is None
                else canonical_split_plan["content_sha256"]
            ),
        }
        manifest = {
            "schema": 1,
            "format": FACIES_MODEL_FORMAT,
            "model_id": model_id,
            "created_utc": utc_now(),
            "content_sha256": _json_digest(content_payload),
            "status": "research_baseline",
            "estimator": "ExtraTreesClassifier",
            "hardware": {
                "execution": "cpu",
                "gpu_acceleration": False,
                "cpu_workers": settings.cpu_workers or "all_available",
            },
            "configuration": asdict(settings),
            "feature_contract": {
                "names": model_payload["feature_names"],
                "dtype": "float32",
                "colour_assumption": "none",
                "uses_btb_specific_direction": False,
                "uses_absolute_position": settings.include_normalized_position,
            },
            "heads": {
                "role": {
                    "classes": [
                        {"class_id": int(value), "name": role_names[int(value)]}
                        for value in role_model.classes_
                    ],
                    "calibration": role_calibration,
                    "sample_count": role_sample_count,
                },
                "material": {
                    "enabled": material_model is not None,
                    "classes": material_catalog,
                    "calibration": material_calibration,
                    "sample_count": material_sample_count,
                },
                "hydraulic": {
                    "enabled": hydraulic_model is not None,
                    "classes": hydraulic_catalog,
                    "target_definition": (
                        "Expert permeability is converted to mD and binned into fixed "
                        "physical intervals; permeability is not an input feature."
                    ),
                    "calibration": hydraulic_calibration,
                    "sample_count": hydraulic_sample_count,
                },
            },
            "dataset": {
                "roots": [
                    str(Path(root).expanduser().resolve()) for root in dataset_roots
                ],
                "bundles": [
                    {
                        "bundle_id": str(record.get("bundle_id") or ""),
                        "content_sha256": str(record.get("content_sha256") or ""),
                        "series_id": str(record.get("series_id") or ""),
                        "project_id": str(record.get("project_id") or ""),
                        "group_id": str(record.get("group_id") or ""),
                        "split": str(record.get("split") or ""),
                        "source_split": str(
                            record.get("source_split") or record.get("split") or ""
                        ),
                    }
                    for record in records
                ],
                "duplicate_records_skipped": duplicate_record_count,
                "group_assignment": {
                    split: sorted(groups) for split, groups in split_groups.items()
                },
                "split_plan": canonical_split_plan,
                "holdout_used_for_fitting": False,
            },
            "scientific_scope": {
                "domain": "registered_camera_plane_2d",
                "depth_used": False,
                "contains_3d_extrapolation": False,
                "contains_phase_interpretation": False,
                "contains_concentration_or_mass_model": False,
                "transfer_claim": (
                    "Only evaluated whole-group results support transfer. Unseen named "
                    "materials and unseen hydraulic classes are reported separately."
                ),
            },
            "metrics_summary": metrics_summary,
            "software": _software_versions(),
            "artifacts": {
                "model": model_reference,
                "metrics": {
                    "relative_path": "metrics.json",
                    "sha256": sha256_file(staging / "metrics.json"),
                },
                "predictions": "predictions/",
            },
            "limitations": [
                "Research baseline; independent scientific validation is required.",
                "Training labels describe 2D camera-plane regions, not 3D volume.",
                "Canonical material transfer requires stable expert material names.",
                "Hydraulic classes are coarse bins and do not replace measured permeability.",
                "Probability calibration is omitted when independent groups are insufficient.",
                "Absolute image position is disabled by default to reduce geometry leakage.",
            ],
        }
        _atomic_json(staging / "manifest.json", manifest)
        (staging / "model_card.md").write_text(
            _model_card_markdown(manifest), encoding="utf-8"
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    record = {
        "schema": 1,
        "format": FACIES_MODEL_INDEX_FORMAT,
        "model_id": model_id,
        "created_utc": manifest["created_utc"],
        "status": manifest["status"],
        "model_relpath": model_id,
        "manifest_relpath": f"{model_id}/manifest.json",
        "content_sha256": manifest["content_sha256"],
        "validation_role_mean_iou": (
            (aggregate.get("validation") or {}).get("role") or {}
        ).get("mean_iou"),
        "holdout_role_mean_iou": metrics_summary["holdout_role_mean_iou"],
        "validation_hydraulic_mean_iou": (
            (aggregate.get("validation") or {}).get("hydraulic") or {}
        ).get("mean_iou"),
        "holdout_hydraulic_mean_iou": metrics_summary["holdout_hydraulic_mean_iou"],
    }
    index_path = output_root / "index.jsonl"
    with index_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    report("Facies baseline trained and evaluated")
    return {
        **record,
        "model_path": str(destination),
        "manifest_path": str(destination / "manifest.json"),
        "metrics_path": str(destination / "metrics.json"),
    }
