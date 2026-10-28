"""Few-shot material examples and conservative facies proposals.

This module bridges the gap between an unsupervised optical draft and a fully
reviewed facies map.  A researcher defines the material vocabulary, marks pure
interior examples, and obtains an explicitly unreviewed 2D proposal.  Once a
compatible transferable model exists, the same workflow can start from a blind
model proposal before local examples are added.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier

from darsia.fluidflower.facies_model import (
    FACIES_MODEL_FORMAT,
    FaciesModelConfig,
    extract_facies_features,
    predict_facies_image,
)
from darsia.fluidflower.project import FluidFlowerProject, sha256_file, utc_now


FACIES_EXAMPLE_SESSION_FORMAT = "darsia-fluidflower-facies-examples-v1"
FACIES_EXAMPLE_REVISION_FORMAT = "darsia-fluidflower-facies-example-revision-v1"
FACIES_EXAMPLE_PREDICTION_FORMAT = "darsia-fluidflower-facies-example-prediction-v1"
FACIES_START_STRATEGIES = ("guided_examples", "blind_model_first")
FACIES_MATERIAL_ROLES = ("porous_medium", "seal")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _append_jsonl as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    selected = Path(path).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(selected)
    if selected.suffix.casefold() == ".npz":
        with np.load(selected, allow_pickle=False) as archive:
            key = "array" if "array" in archive.files else archive.files[0]
            array = np.asarray(archive[key])
    elif selected.suffix.casefold() == ".npy":
        array = np.asarray(np.load(selected, allow_pickle=False))
    else:
        with Image.open(selected) as image:
            array = np.asarray(image.convert("RGB"))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError("Facies examples require an RGB image")
    rgb = np.asarray(array[..., :3], dtype=np.float32)
    if rgb.size and float(np.nanmax(rgb)) > 1.5:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def load_facies_example_image(path: Path) -> np.ndarray:
    """Load one immutable RGB source for GUI preview or model fitting."""

    return _load_rgb(path)


def default_facies_example_session(
    project: FluidFlowerProject | None = None,
) -> dict[str, Any]:
    """Return an empty, portable few-shot facies session."""

    return {
        "schema": 1,
        "format": FACIES_EXAMPLE_SESSION_FORMAT,
        "project_id": "" if project is None else project.project_id,
        "series_id": (
            ""
            if project is None
            else str(project.manifest.get("series_id") or "")
        ),
        "start_strategy": "guided_examples",
        "source_image_path": "",
        "pretrained_model_manifest": "",
        "minimum_examples_per_material": 8,
        "materials": [],
        "examples": [],
        "latest_prediction_manifest": "",
    }


def _normalized_material(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    material_id = str(row.get("material_id") or "").strip() or uuid.uuid4().hex
    name = str(row.get("name") or "").strip()
    if not name:
        raise ValueError(f"Sand type {index} requires a name")
    role = str(row.get("analysis_role") or "porous_medium").strip().casefold()
    if role not in FACIES_MATERIAL_ROLES:
        raise ValueError(f"Unsupported facies material role: {role!r}")
    return {
        "material_id": material_id,
        "name": name,
        "canonical_key": str(row.get("canonical_key") or name).strip().casefold(),
        "analysis_role": role,
        "porosity": row.get("porosity"),
        "permeability": row.get("permeability"),
    }


def _normalized_box(row: Mapping[str, Any], *, material_ids: set[str]) -> dict[str, Any]:
    material_id = str(row.get("material_id") or "").strip()
    if material_id not in material_ids:
        raise ValueError("Every example box must reference a defined sand type")
    values = {
        key: float(row.get(key) or 0.0)
        for key in ("x", "y", "width", "height")
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Example-box coordinates must be finite")
    if (
        values["width"] <= 0.0
        or values["height"] <= 0.0
        or values["x"] < 0.0
        or values["y"] < 0.0
        or values["x"] + values["width"] > 1.0 + 1.0e-9
        or values["y"] + values["height"] > 1.0 + 1.0e-9
    ):
        raise ValueError("Example boxes must lie inside the normalized image")
    source = str(row.get("source_image_path") or "").strip()
    if not source:
        raise ValueError("Every example box requires its source image")
    return {
        "example_id": str(row.get("example_id") or "").strip() or uuid.uuid4().hex,
        "material_id": material_id,
        "source_image_path": str(Path(source).expanduser().resolve()),
        "coordinate_space": "normalized_registered_2d",
        **{key: float(np.clip(value, 0.0, 1.0)) for key, value in values.items()},
        "created_utc": str(row.get("created_utc") or utc_now()),
        "created_by": str(row.get("created_by") or "").strip(),
        "note": str(row.get("note") or "").strip(),
    }


def normalize_facies_example_session(
    raw: Mapping[str, Any] | None,
    *,
    project: FluidFlowerProject | None = None,
) -> dict[str, Any]:
    """Normalize a saved session without mutating its source mapping."""

    candidate = default_facies_example_session(project)
    candidate.update(deepcopy(dict(raw or {})))
    strategy = str(candidate.get("start_strategy") or "guided_examples")
    if strategy not in FACIES_START_STRATEGIES:
        raise ValueError(f"Unsupported facies start strategy: {strategy!r}")
    minimum = int(candidate.get("minimum_examples_per_material") or 8)
    if not 1 <= minimum <= 1000:
        raise ValueError("Minimum examples per material must lie between 1 and 1000")
    materials = [
        _normalized_material(row, index=index)
        for index, row in enumerate(candidate.get("materials") or [], start=1)
        if isinstance(row, Mapping)
    ]
    identifiers = [str(row["material_id"]) for row in materials]
    names = [str(row["name"]).casefold() for row in materials]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Sand-type identifiers must be unique")
    if len(set(names)) != len(names):
        raise ValueError("Sand-type names must be unique")
    material_ids = set(identifiers)
    examples = [
        _normalized_box(row, material_ids=material_ids)
        for row in candidate.get("examples") or []
        if isinstance(row, Mapping)
    ]
    example_ids = [str(row["example_id"]) for row in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("Example-box identifiers must be unique")
    return {
        "schema": 1,
        "format": FACIES_EXAMPLE_SESSION_FORMAT,
        "project_id": str(candidate.get("project_id") or ""),
        "series_id": str(candidate.get("series_id") or ""),
        "start_strategy": strategy,
        "source_image_path": str(candidate.get("source_image_path") or "").strip(),
        "pretrained_model_manifest": str(
            candidate.get("pretrained_model_manifest") or ""
        ).strip(),
        "minimum_examples_per_material": minimum,
        "materials": materials,
        "examples": examples,
        "latest_prediction_manifest": str(
            candidate.get("latest_prediction_manifest") or ""
        ).strip(),
    }


def facies_example_quality(session: Mapping[str, Any]) -> dict[str, Any]:
    """Report whether examples are numerous and spatially diverse enough."""

    normalized = normalize_facies_example_session(session)
    minimum = int(normalized["minimum_examples_per_material"])
    by_material: list[dict[str, Any]] = []
    ready = len(normalized["materials"]) >= 2
    for material in normalized["materials"]:
        material_id = str(material["material_id"])
        boxes = [
            row
            for row in normalized["examples"]
            if str(row["material_id"]) == material_id
        ]
        bins = {
            (
                min(2, int(3.0 * (float(row["y"]) + 0.5 * float(row["height"])))),
                min(3, int(4.0 * (float(row["x"]) + 0.5 * float(row["width"])))),
            )
            for row in boxes
        }
        sources = {str(row["source_image_path"]) for row in boxes}
        required_bins = min(3, minimum)
        material_ready = len(boxes) >= minimum and len(bins) >= required_bins
        ready = ready and material_ready
        by_material.append(
            {
                "material_id": material_id,
                "name": str(material["name"]),
                "example_count": len(boxes),
                "minimum_count": minimum,
                "spatial_bin_count": len(bins),
                "minimum_spatial_bins": required_bins,
                "source_image_count": len(sources),
                "normalized_area_sum": float(
                    sum(float(row["width"]) * float(row["height"]) for row in boxes)
                ),
                "ready": material_ready,
            }
        )
    return {
        "ready_for_guided_prediction": ready,
        "material_count": len(normalized["materials"]),
        "example_count": len(normalized["examples"]),
        "minimum_examples_per_material": minimum,
        "materials": by_material,
    }


class FaciesExampleStore:
    """Atomic current state plus immutable revisions for every user edit."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.root = project.root / "annotations" / "facies" / "examples"
        self.current_path = self.root / "current.json"
        self.revisions_root = self.root / "revisions"
        self.history_path = self.root / "history.jsonl"

    def current(self) -> dict[str, Any]:
        if not self.current_path.is_file():
            return default_facies_example_session(self.project)
        payload = json.loads(self.current_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping):
            raise TypeError("Facies example state must contain one JSON object")
        return normalize_facies_example_session(payload, project=self.project)

    def save(
        self,
        session: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        reviewer = str(actor).strip()
        if not reviewer:
            raise ValueError("A reviewer name is required before examples can be saved")
        normalized = normalize_facies_example_session(session, project=self.project)
        revision_id = uuid.uuid4().hex
        saved = {
            **normalized,
            "revision_id": revision_id,
            "updated_utc": utc_now(),
            "updated_by": reviewer,
            "reason": str(reason).strip(),
        }
        saved["content_sha256"] = _json_digest(normalized)
        revision_path = self.revisions_root / f"{revision_id}.json"
        _atomic_json(revision_path, saved)
        _atomic_json(self.current_path, saved)
        _append_jsonl(
            self.history_path,
            {
                "schema": 1,
                "format": FACIES_EXAMPLE_REVISION_FORMAT,
                "revision_id": revision_id,
                "updated_utc": saved["updated_utc"],
                "updated_by": reviewer,
                "reason": saved["reason"],
                "content_sha256": saved["content_sha256"],
                "revision_path": str(revision_path),
                "material_count": len(normalized["materials"]),
                "example_count": len(normalized["examples"]),
            },
        )
        return deepcopy(saved)


def _analysis_image(image: np.ndarray, maximum_width: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, float(maximum_width) / max(width, 1))
    if scale >= 1.0:
        return image, 1.0
    resized = cv2.resize(
        image,
        (max(8, int(round(width * scale))), max(8, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return np.asarray(resized, dtype=np.float32), scale


def _training_samples(
    session: Mapping[str, Any],
    *,
    maximum_width: int,
    pixels_per_box: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    normalized = normalize_facies_example_session(session)
    class_catalog = [
        {**material, "class_id": index}
        for index, material in enumerate(normalized["materials"], start=1)
    ]
    class_by_material = {
        str(row["material_id"]): int(row["class_id"]) for row in class_catalog
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in normalized["examples"]:
        grouped.setdefault(str(row["source_image_path"]), []).append(row)
    rng = np.random.default_rng(int(random_seed))
    feature_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for path_text, boxes in grouped.items():
        image, _scale = _analysis_image(_load_rgb(Path(path_text)), maximum_width)
        height, width = image.shape[:2]
        indices: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for box in boxes:
            x0 = int(np.floor(float(box["x"]) * width))
            y0 = int(np.floor(float(box["y"]) * height))
            x1 = int(np.ceil((float(box["x"]) + float(box["width"])) * width))
            y1 = int(np.ceil((float(box["y"]) + float(box["height"])) * height))
            x0, x1 = np.clip((x0, x1), 0, width)
            y0, y1 = np.clip((y0, y1), 0, height)
            inset_x = max(0, int(round(0.05 * (x1 - x0))))
            inset_y = max(0, int(round(0.05 * (y1 - y0))))
            x0, x1 = x0 + inset_x, x1 - inset_x
            y0, y1 = y0 + inset_y, y1 - inset_y
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            selected = (yy * width + xx).reshape(-1)
            if selected.size > int(pixels_per_box):
                selected = rng.choice(
                    selected, int(pixels_per_box), replace=False
                )
            indices.append(np.asarray(selected, dtype=np.int64))
            labels.append(
                np.full(
                    selected.size,
                    class_by_material[str(box["material_id"])],
                    dtype=np.uint16,
                )
            )
        if not indices:
            continue
        flat_indices = np.concatenate(indices)
        feature_rows.append(extract_facies_features(image, flat_indices=flat_indices))
        targets.append(np.concatenate(labels))
    if not feature_rows:
        raise ValueError("No usable pixels were found in the material example boxes")
    return np.concatenate(feature_rows), np.concatenate(targets), class_catalog


def _regularize_prediction(
    image: np.ndarray,
    prediction: Mapping[str, np.ndarray],
    *,
    head: str,
) -> tuple[np.ndarray, np.ndarray]:
    from skimage.segmentation import slic

    probabilities = np.asarray(prediction[f"{head}_probabilities"], dtype=np.float32)
    classes = np.asarray(prediction[f"{head}_class_ids"], dtype=np.uint16)
    height, width = image.shape[:2]
    segment_count = int(np.clip(height * width / 700.0, 300, 3000))
    segments = slic(
        image,
        n_segments=segment_count,
        compactness=9.0,
        sigma=0.8,
        start_label=0,
        channel_axis=-1,
    ).astype(np.int32)
    flat_segments = segments.reshape(-1)
    counts = np.bincount(flat_segments).astype(np.float64)
    means = np.column_stack(
        [
            np.bincount(
                flat_segments,
                weights=probabilities[..., column].reshape(-1),
                minlength=counts.size,
            )
            / np.maximum(counts, 1.0)
            for column in range(probabilities.shape[-1])
        ]
    )
    winner = np.argmax(means, axis=1)
    facies = classes[winner][segments]
    clipped = np.clip(means, 1.0e-7, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    if means.shape[1] > 1:
        entropy /= np.log(float(means.shape[1]))
    else:
        entropy[:] = 0.0
    return facies.astype(np.uint16), entropy[segments].astype(np.float32)


def _physical_layers(
    facies: np.ndarray,
    class_catalog: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from scipy import ndimage as ndi
    from skimage.measure import label as connected_labels

    values = np.asarray(facies, dtype=np.uint16).copy()
    minimum = max(24, int(round(values.size * 0.00002)))
    stable = np.zeros(values.shape, dtype=bool)
    for class_id in np.unique(values):
        components = connected_labels(values == class_id, connectivity=1)
        counts = np.bincount(components.reshape(-1))
        valid = np.flatnonzero(counts >= minimum)
        valid = valid[valid != 0]
        if valid.size == 0 and counts.size > 1:
            valid = np.asarray([1 + int(np.argmax(counts[1:]))])
        stable |= np.isin(components, valid)
    missing = ~stable
    if np.any(missing) and np.any(stable):
        nearest = ndi.distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        values[missing] = values[tuple(nearest[:, missing])]

    catalog = {int(row["class_id"]): dict(row) for row in class_catalog}
    labels = np.zeros(values.shape, dtype=np.int32)
    regions: list[dict[str, Any]] = []
    layer_id = 0
    for class_id in sorted(int(value) for value in np.unique(values) if value > 0):
        components = connected_labels(values == class_id, connectivity=1)
        material = catalog.get(class_id, {})
        component_rows = sorted(
            range(1, int(components.max()) + 1),
            key=lambda value: int(np.count_nonzero(components == value)),
            reverse=True,
        )
        for material_layer_index, component_id in enumerate(component_rows, start=1):
            mask = components == component_id
            if not np.any(mask):
                continue
            layer_id += 1
            labels[mask] = layer_id
            regions.append(
                {
                    "label_id": layer_id,
                    "facies_id": class_id,
                    "material_id": str(material.get("material_id") or ""),
                    "name": (
                        f"{str(material.get('name') or f'Material {class_id}')} · "
                        f"proposed layer {material_layer_index}"
                    ),
                    "analysis_role": str(
                        material.get("analysis_role") or "porous_medium"
                    ),
                    "porosity": material.get("porosity"),
                    "permeability": material.get("permeability"),
                    "enabled": False,
                    "review_status": "unreviewed_model_proposal",
                }
            )
    return labels, regions


def _resolve_model_manifest(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    selected = Path(path).expanduser().resolve()
    manifest_path = selected / "manifest.json" if selected.is_dir() else selected
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("format") != FACIES_MODEL_FORMAT:
        raise ValueError("Selected file is not a DarSIA facies model manifest")
    model_record = dict((manifest.get("artifacts") or {}).get("model") or {})
    model_path = manifest_path.parent / str(model_record.get("relative_path") or "")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if str(model_record.get("sha256") or "") and sha256_file(model_path) != str(
        model_record["sha256"]
    ):
        raise ValueError("Pretrained facies model failed its SHA-256 verification")
    payload = joblib.load(model_path)
    if not isinstance(payload, Mapping):
        raise TypeError("Pretrained facies model payload is invalid")
    return manifest_path, manifest, dict(payload)


def create_facies_example_prediction(
    session: Mapping[str, Any],
    destination_root: Path,
    *,
    mode: str | None = None,
    maximum_width: int = 1200,
    pixels_per_box: int = 1800,
    n_estimators: int = 160,
    cpu_workers: int = 0,
    random_seed: int = 20_260_903,
) -> dict[str, Any]:
    """Create an immutable blind or guided material-and-layer proposal."""

    normalized = normalize_facies_example_session(session)
    selected_mode = str(mode or normalized["start_strategy"])
    if selected_mode not in FACIES_START_STRATEGIES:
        raise ValueError(f"Unsupported facies prediction mode: {selected_mode!r}")
    source_path = Path(str(normalized.get("source_image_path") or "")).expanduser()
    source = _load_rgb(source_path)
    analysis, _scale = _analysis_image(source, int(maximum_width))
    source_model: dict[str, Any] | None = None

    if selected_mode == "guided_examples":
        quality = facies_example_quality(normalized)
        if not quality["ready_for_guided_prediction"]:
            raise ValueError(
                "Each named sand type needs the configured number of examples and "
                "at least three spatially distinct locations"
            )
        features, target, class_catalog = _training_samples(
            normalized,
            maximum_width=int(maximum_width),
            pixels_per_box=int(pixels_per_box),
            random_seed=int(random_seed),
        )
        if np.unique(target).size < 2:
            raise ValueError("At least two represented sand types are required")
        estimator = ExtraTreesClassifier(
            n_estimators=max(8, int(n_estimators)),
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=int(cpu_workers) or -1,
            random_state=int(random_seed),
        )
        estimator.fit(features, target)
        estimators = {"material": estimator}
        model_payload: dict[str, Any] = {
            "format": FACIES_EXAMPLE_PREDICTION_FORMAT,
            "feature_contract": "facies_feature_names_v1",
            "material_model": estimator,
            "material_catalog": class_catalog,
        }
    else:
        model_text = str(normalized.get("pretrained_model_manifest") or "").strip()
        if not model_text:
            raise ValueError("Blind-first prediction requires a pretrained model")
        model_manifest_path, model_manifest, model_payload = _resolve_model_manifest(
            Path(model_text)
        )
        estimator = model_payload.get("material_model")
        if estimator is None:
            raise ValueError("The pretrained model has no named-material head")
        estimators = {"material": estimator}
        class_catalog = [
            {
                **dict(row),
                "material_id": str(
                    row.get("material_id")
                    or row.get("canonical_key")
                    or f"model-{row.get('class_id')}"
                ),
            }
            for row in model_payload.get("material_catalog") or []
        ]
        source_model = {
            "manifest_path": str(model_manifest_path),
            "model_id": str(model_manifest.get("model_id") or ""),
            "manifest_sha256": sha256_file(model_manifest_path),
        }

    raw_config = dict(model_payload.get("config") or {})
    allowed = set(FaciesModelConfig.__dataclass_fields__)
    model_config = FaciesModelConfig(
        **{key: value for key, value in raw_config.items() if key in allowed}
    ).validated()
    model_config = replace(
        model_config,
        cpu_workers=int(cpu_workers),
        store_probabilities=True,
    )
    prediction = predict_facies_image(
        analysis,
        estimators,
        {"material": np.ones(analysis.shape[:2], dtype=bool)},
        config=model_config,
    )
    facies_small, uncertainty_small = _regularize_prediction(
        analysis, prediction, head="material"
    )
    height, width = source.shape[:2]
    facies = cv2.resize(
        facies_small,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint16)
    uncertainty = cv2.resize(
        uncertainty_small,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    labels, regions = _physical_layers(facies, class_catalog)

    destination_root = Path(destination_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    prediction_id = uuid.uuid4().hex
    staging = destination_root / f".{prediction_id}.{uuid.uuid4().hex}.tmp"
    destination = destination_root / prediction_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        np.savez_compressed(staging / "facies.npz", array=facies)
        np.savez_compressed(staging / "labels.npz", array=labels)
        np.savez_compressed(staging / "uncertainty.npz", array=uncertainty)
        if selected_mode == "guided_examples":
            joblib.dump(model_payload, staging / "local_model.joblib", compress=3)
        preview = np.rint(255.0 * source).astype(np.uint8)
        palette = np.asarray(
            [
                (36, 41, 46),
                (77, 184, 166),
                (231, 178, 68),
                (89, 148, 205),
                (220, 105, 112),
                (151, 122, 201),
                (103, 178, 92),
                (225, 137, 67),
            ],
            dtype=np.uint8,
        )
        colors = palette[np.mod(facies, len(palette))]
        Image.fromarray(
            np.rint(0.52 * preview + 0.48 * colors).astype(np.uint8)
        ).save(staging / "preview.png")
        artifacts = {}
        for name in ("facies", "labels", "uncertainty", "preview"):
            filename = f"{name}.npz" if name != "preview" else "preview.png"
            artifacts[name] = {
                "relative_path": filename,
                "sha256": sha256_file(staging / filename),
            }
        if (staging / "local_model.joblib").is_file():
            artifacts["local_model"] = {
                "relative_path": "local_model.joblib",
                "sha256": sha256_file(staging / "local_model.joblib"),
            }
        manifest = {
            "schema": 1,
            "format": FACIES_EXAMPLE_PREDICTION_FORMAT,
            "prediction_id": prediction_id,
            "created_utc": utc_now(),
            "status": "unreviewed_model_proposal",
            "mode": selected_mode,
            "source_image": {
                "path": str(source_path.resolve()),
                "sha256": sha256_file(source_path.resolve()),
                "shape": [int(value) for value in source.shape],
            },
            "source_model": source_model,
            "example_session_sha256": _json_digest(normalized),
            "quality": facies_example_quality(normalized),
            "analysis": {
                "maximum_width": int(maximum_width),
                "analysis_shape": [int(value) for value in analysis.shape],
                "full_resolution_output_shape": [height, width],
                "regularization": "slic_probability_aggregation_v1",
                "layer_proposal": "connected_components_with_small_region_cleanup_v1",
            },
            "materials": class_catalog,
            "regions": regions,
            "artifacts": artifacts,
            "scientific_scope": {
                "domain": "registered_visible_camera_plane_2d",
                "review_required": True,
                "properties_inferred_from_image": False,
                "original_sources_modified": False,
            },
        }
        manifest["content_sha256"] = _json_digest(manifest)
        _atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        **manifest,
        "manifest_path": str(destination / "manifest.json"),
        "prediction_path": str(destination),
    }


def facies_config_from_example_prediction(
    config: Mapping[str, Any], prediction_manifest_path: Path
) -> dict[str, Any]:
    """Attach an unreviewed proposal to a working config without touching sources."""

    manifest_path = Path(prediction_manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("format") != FACIES_EXAMPLE_PREDICTION_FORMAT:
        raise ValueError("Unsupported facies example prediction")
    artifacts = dict(manifest.get("artifacts") or {})

    def artifact(name: str) -> str:
        record = dict(artifacts.get(name) or {})
        path = manifest_path.parent / str(record.get("relative_path") or "")
        if not path.is_file() or sha256_file(path) != str(record.get("sha256") or ""):
            raise ValueError(f"Facies proposal artifact failed verification: {name}")
        return str(path)

    updated = deepcopy(dict(config))
    updated["labels_path"] = artifact("labels")
    updated["facies_source_path"] = artifact("facies")
    updated["display_image_path"] = str(
        (manifest.get("source_image") or {}).get("path") or ""
    )
    updated["regions"] = deepcopy(list(manifest.get("regions") or []))
    updated["grouping_mode"] = "shared_facies"
    updated["facies_generation"] = {
        "method": "blind_pretrained_then_local_examples_v1"
        if manifest.get("mode") == "blind_model_first"
        else "named_material_examples_v1",
        "status": "unreviewed_model_proposal",
        "prediction_manifest_path": str(manifest_path),
        "prediction_content_sha256": str(manifest.get("content_sha256") or ""),
        "source_image_path": str(
            (manifest.get("source_image") or {}).get("path") or ""
        ),
        "original_sources_modified": False,
    }
    return updated


__all__ = [
    "FACIES_EXAMPLE_PREDICTION_FORMAT",
    "FACIES_EXAMPLE_REVISION_FORMAT",
    "FACIES_EXAMPLE_SESSION_FORMAT",
    "FACIES_MATERIAL_ROLES",
    "FACIES_START_STRATEGIES",
    "FaciesExampleStore",
    "create_facies_example_prediction",
    "default_facies_example_session",
    "facies_config_from_example_prediction",
    "facies_example_quality",
    "load_facies_example_image",
    "normalize_facies_example_session",
]
