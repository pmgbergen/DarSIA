"""Portable expert annotations for transferable FluidFlower facies models."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from darsia.fluidflower.geometry_facies import (
    GEOMETRY_REVISION_FORMAT,
    geometry_config_digest,
    load_registered_geometry_image,
    validate_geometry_config,
)
from darsia.fluidflower.image_correction import CorrectionRevisionStore
from darsia.fluidflower.project import sha256_file, utc_now

if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject


FACIES_ANNOTATION_FORMAT = "darsia-fluidflower-facies-annotation-v1"
FACIES_DATASET_INDEX_FORMAT = "darsia-fluidflower-facies-dataset-index-v1"
FACIES_SPLITS = ("train", "validation", "holdout", "pool")
ROLE_IDS = {
    "excluded": 0,
    "porous_medium": 1,
    "seal": 2,
    "free_water": 3,
    "frame": 4,
    "colorchecker": 5,
}
HYDRAULIC_CLASS_DEFINITIONS = (
    {
        "class_id": 1,
        "canonical_key": "permeability_lt_100_md",
        "name": "< 100 mD",
        "minimum_md": None,
        "maximum_md": 100.0,
    },
    {
        "class_id": 2,
        "canonical_key": "permeability_100_300_md",
        "name": "100-300 mD",
        "minimum_md": 100.0,
        "maximum_md": 300.0,
    },
    {
        "class_id": 3,
        "canonical_key": "permeability_300_1000_md",
        "name": "300-1000 mD",
        "minimum_md": 300.0,
        "maximum_md": 1000.0,
    },
    {
        "class_id": 4,
        "canonical_key": "permeability_1000_3000_md",
        "name": "1000-3000 mD",
        "minimum_md": 1000.0,
        "maximum_md": 3000.0,
    },
    {
        "class_id": 5,
        "canonical_key": "permeability_ge_3000_md",
        "name": ">= 3000 mD",
        "minimum_md": 3000.0,
        "maximum_md": None,
    },
)
FACIES_COLORS = np.asarray(
    [
        (47, 55, 61),
        (76, 189, 157),
        (229, 177, 68),
        (87, 148, 209),
        (218, 102, 112),
        (151, 121, 202),
        (105, 180, 94),
        (228, 137, 65),
        (70, 192, 205),
        (205, 103, 166),
        (169, 176, 82),
        (111, 135, 180),
        (237, 201, 118),
        (131, 195, 168),
        (197, 125, 94),
        (144, 159, 169),
    ],
    dtype=np.uint8,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
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
                    f"Facies dataset record at line {line_number} is not an object"
                )
            rows.append(payload)
    return rows


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _append_jsonl as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


@contextmanager
def _index_lock(root: Path, *, timeout_s: float = 30.0) -> Iterator[None]:
    lock_path = root / ".index.lock"
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    descriptor: int | None = None
    while descriptor is None:
        try:
            candidate = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            try:
                os.write(candidate, f"{os.getpid()} {utc_now()}\n".encode())
            except Exception:
                os.close(candidate)
                lock_path.unlink(missing_ok=True)
                raise
            descriptor = candidate
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 300.0
            except OSError:
                stale = False
            if stale:
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Facies dataset index is locked: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _load_array(path: Path) -> np.ndarray:
    selected = Path(path).expanduser()
    if selected.suffix.casefold() == ".npz":
        with np.load(selected, allow_pickle=False) as archive:
            key = "array" if "array" in archive.files else archive.files[0]
            return np.asarray(archive[key])
    if selected.suffix.casefold() == ".npy":
        return np.asarray(np.load(selected, allow_pickle=False))
    with Image.open(selected) as image:
        return np.asarray(image)


def _rgb_float(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("Facies annotation images must have at least three channels")
    output = np.asarray(image[..., :3], dtype=np.float32)
    if output.size and float(np.nanmax(output)) > 1.5:
        output /= 255.0
    return np.clip(output, 0.0, 1.0)


def _file_reference(path: Path) -> dict[str, Any]:
    from darsia.fluidflower.phase_interpretation import _file_reference as _canonical_file_reference
    return _canonical_file_reference(path)


def _artifact_reference(path: Path, relative_path: Path) -> dict[str, Any]:
    selected = Path(path)
    return {
        "relative_path": relative_path.as_posix(),
        "size_bytes": int(selected.stat().st_size),
        "sha256": sha256_file(selected),
    }


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _target_maps(
    config: Mapping[str, Any], labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    facies = np.zeros(labels.shape, dtype=np.int32)
    roles = np.zeros(labels.shape, dtype=np.uint8)
    context = np.zeros(labels.shape, dtype=np.uint8)
    training = np.zeros(labels.shape, dtype=np.uint8)
    per_layer = str(config.get("grouping_mode") or "shared_facies") == "per_layer"
    for region in config.get("regions") or []:
        label_id = int(region["label_id"])
        facies_id = label_id if per_layer else int(region["facies_id"])
        role = str(region.get("analysis_role") or "excluded")
        role_id = ROLE_IDS.get(role, ROLE_IDS["excluded"])
        selected = labels == label_id
        facies[selected] = facies_id
        roles[selected] = role_id
        if bool(region.get("enabled", True)) and role != "excluded":
            context[selected] = 1
        if bool(region.get("enabled", True)) and role in {
            "porous_medium",
            "seal",
        }:
            training[selected] = 1
    return facies, roles, context, training


def _canonical_material_targets(
    config: Mapping[str, Any], labels: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    entries: dict[str, dict[str, Any]] = {}
    label_keys: dict[int, str] = {}
    for raw_region in config.get("regions") or []:
        region = dict(raw_region)
        role = str(region.get("analysis_role") or "excluded")
        if not bool(region.get("enabled", True)) or role not in {
            "porous_medium",
            "seal",
        }:
            continue
        name = str(region.get("name") or "").strip()
        slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
        if not slug:
            continue
        key = f"{role}:{slug}"
        provisional = bool(re.fullmatch(r"(?:layer|region|optical-region)-?\d+", slug))
        label_id = int(region["label_id"])
        label_keys[label_id] = key
        entry = entries.setdefault(
            key,
            {
                "canonical_key": key,
                "name": name,
                "analysis_role": role,
                "provisional": provisional,
                "eligible_for_cross_series_training": not provisional,
                "label_ids": [],
                "facies_ids": [],
                "porosity_values": [],
                "permeability_values": [],
                "permeability_unit": str(
                    config.get("permeability_unit") or "source_unit"
                ),
            },
        )
        entry["label_ids"].append(label_id)
        entry["facies_ids"].append(int(region["facies_id"]))
        for source_key, output_key in (
            ("porosity", "porosity_values"),
            ("permeability", "permeability_values"),
        ):
            value = region.get(source_key)
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed):
                entry[output_key].append(parsed)
    keys = sorted(entries)
    ids = {key: index for index, key in enumerate(keys, start=1)}
    target = np.zeros(labels.shape, dtype=np.uint16)
    for label_id, key in label_keys.items():
        target[labels == label_id] = ids[key]
    definitions: list[dict[str, Any]] = []
    for key in keys:
        entry = entries[key]
        definitions.append(
            {
                **entry,
                "class_id": ids[key],
                "label_ids": sorted(set(entry["label_ids"])),
                "facies_ids": sorted(set(entry["facies_ids"])),
                "porosity_values": sorted(set(entry["porosity_values"])),
                "permeability_values": sorted(set(entry["permeability_values"])),
            }
        )
    return target, definitions


def _permeability_md(value: Any, unit: str) -> float | None:
    try:
        permeability = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(permeability) or permeability <= 0.0:
        return None
    normalized = str(unit).strip().casefold().replace("²", "2")
    if normalized in {"md", "millidarcy", "millidarcies"}:
        return permeability
    if normalized in {"d", "darcy", "darcies"}:
        return 1000.0 * permeability
    if normalized in {"m2", "m^2"}:
        return permeability / 9.869233e-16
    return None


def _hydraulic_class_targets(
    config: Mapping[str, Any], labels: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    target = np.zeros(labels.shape, dtype=np.uint8)
    unit = str(config.get("permeability_unit") or "source_unit")
    for raw_region in config.get("regions") or []:
        region = dict(raw_region)
        role = str(region.get("analysis_role") or "excluded")
        if not bool(region.get("enabled", True)) or role not in {
            "porous_medium",
            "seal",
        }:
            continue
        permeability = _permeability_md(region.get("permeability"), unit)
        if permeability is None:
            continue
        class_id = 5
        for definition in HYDRAULIC_CLASS_DEFINITIONS:
            maximum = definition["maximum_md"]
            if maximum is None or permeability < float(maximum):
                class_id = int(definition["class_id"])
                break
        target[labels == int(region["label_id"])] = class_id
    return target, [dict(definition) for definition in HYDRAULIC_CLASS_DEFINITIONS]


def _boundaries(field: np.ndarray) -> np.ndarray:
    boundary = np.zeros(field.shape, dtype=bool)
    boundary[:, 1:] |= field[:, 1:] != field[:, :-1]
    boundary[:, :-1] |= field[:, 1:] != field[:, :-1]
    boundary[1:, :] |= field[1:, :] != field[:-1, :]
    boundary[:-1, :] |= field[1:, :] != field[:-1, :]
    return boundary


def _preview(
    image: np.ndarray,
    labels: np.ndarray,
    facies: np.ndarray,
    context_mask: np.ndarray,
) -> np.ndarray:
    import cv2

    base = np.rint(255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    colors = FACIES_COLORS[np.mod(facies, len(FACIES_COLORS))]
    display = base.copy()
    valid = np.asarray(context_mask, dtype=bool)
    blended = np.rint(0.62 * base + 0.38 * colors).astype(np.uint8)
    display[valid] = blended[valid]
    label_boundary = _boundaries(labels)
    facies_boundary = _boundaries(facies)
    display[label_boundary] = (35, 38, 42)
    display[facies_boundary] = (245, 248, 250)
    height, width = display.shape[:2]
    scale = min(1.0, 1600.0 / width, 1000.0 / height)
    if scale < 1.0:
        display = cv2.resize(
            display,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return display


class FaciesAnnotationStore:
    """Immutable facies bundles plus a portable, append-only dataset index."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.bundles_root = self.root / "bundles"
        self.index_path = self.root / "index.jsonl"

    def records(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.index_path)

    def summary(self) -> dict[str, Any]:
        rows = self.records()
        split_counts = {split: 0 for split in FACIES_SPLITS}
        status_counts: dict[str, int] = {}
        group_splits: dict[str, set[str]] = {}
        for row in rows:
            split = str(row.get("split") or "")
            if split in split_counts:
                split_counts[split] += 1
            status = str(row.get("review_status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            group = str(row.get("group_id") or "")
            if group:
                group_splits.setdefault(group, set()).add(split)
        conflicts = {
            group: sorted(splits)
            for group, splits in group_splits.items()
            if len(splits) > 1
        }
        return {
            "bundle_count": len(rows),
            "split_counts": split_counts,
            "status_counts": status_counts,
            "group_count": len(group_splits),
            "split_conflicts": conflicts,
        }

    def _existing_record(
        self,
        *,
        content_sha256: str,
        group_id: str,
        split: str,
    ) -> dict[str, Any] | None:
        for row in self.records():
            if (
                str(row.get("group_id") or "") == group_id
                and str(row.get("split") or "") != split
            ):
                raise ValueError(
                    f"Group {group_id!r} is already assigned to the "
                    f"{row.get('split')!r} split"
                )
            if str(row.get("content_sha256") or "") == content_sha256:
                manifest = self.root / str(row.get("manifest_relpath") or "")
                if manifest.is_file():
                    return dict(row)
        return None

    def export(
        self,
        project: FluidFlowerProject,
        geometry_revision: Mapping[str, Any],
        *,
        split: str,
        group_id: str,
        actor: str,
        include_registered_frame: bool = False,
    ) -> dict[str, Any]:
        split = str(split).strip().casefold()
        if split not in FACIES_SPLITS:
            raise ValueError(f"Unsupported facies dataset split: {split!r}")
        group_id = str(group_id).strip().casefold()
        if not group_id:
            raise ValueError("A non-empty leakage group is required")
        if geometry_revision.get("format") != GEOMETRY_REVISION_FORMAT:
            raise ValueError("Save a geometry/facies revision before export")
        config = dict(geometry_revision.get("configuration") or {})
        validation = dict(geometry_revision.get("validation") or {})
        if validation.get("config_sha256") != geometry_config_digest(config):
            validation = validate_geometry_config(config)
        if int((validation.get("summary") or {}).get("error_count") or 0) > 0:
            raise ValueError("The geometry revision has blocking validation findings")
        edit_reference = dict(config.get("label_edit_session") or {})
        if int(edit_reference.get("active_operation_count") or 0) != int(
            edit_reference.get("materialized_active_operation_count") or 0
        ):
            raise ValueError("Materialize all active boundary edits before export")

        baseline_path = Path(str(config["baseline_path"])).expanduser().resolve()
        labels_path = Path(str(config["labels_path"])).expanduser().resolve()
        depth_path = Path(str(config["depth_path"])).expanduser().resolve()
        baseline = _rgb_float(_load_array(baseline_path))
        labels = np.asarray(np.squeeze(_load_array(labels_path)), dtype=np.int32)
        depth = np.asarray(np.squeeze(_load_array(depth_path)), dtype=np.float32)
        if baseline.shape[:2] != labels.shape or depth.shape != labels.shape:
            raise ValueError("Image, labels, and depth do not share one spatial grid")
        facies, roles, context_mask, training_mask = _target_maps(config, labels)
        material_classes, material_definitions = _canonical_material_targets(
            config, labels
        )
        hydraulic_classes, hydraulic_definitions = _hydraulic_class_targets(
            config, labels
        )

        sources = {
            "baseline": _file_reference(baseline_path),
            "labels": _file_reference(labels_path),
            "depth": _file_reference(depth_path),
        }
        review_image: np.ndarray | None = None
        review_provenance: dict[str, Any] | None = None
        display_path = str(config.get("display_image_path") or "")
        if include_registered_frame and not display_path:
            raise ValueError(
                "No registered experiment frame is selected in the geometry revision"
            )
        if include_registered_frame and display_path:
            review_path = Path(display_path).expanduser().resolve()
            review_image, review_provenance = load_registered_geometry_image(
                config, review_path
            )
            if review_image.shape[:2] != labels.shape:
                raise ValueError(
                    "Registered review frame does not match the label grid"
                )
            sources["registered_review_frame"] = _file_reference(review_path)

        module_status = str(
            (project.module_states.get("geometry_facies") or {}).get("status") or ""
        )
        review_status = (
            "approved" if module_status in {"approved", "complete"} else "draft"
        )
        correction = CorrectionRevisionStore(project).current()
        correction_reference = {
            "revision_id": str(correction.get("revision_id") or ""),
            "algorithm": str(correction.get("algorithm") or ""),
            "pipeline_order": list(correction.get("pipeline_order") or []),
        }
        content_payload = {
            "format": FACIES_ANNOTATION_FORMAT,
            "project_id": project.project_id,
            "geometry_revision_id": str(geometry_revision.get("revision_id") or ""),
            "geometry_config_sha256": geometry_config_digest(config),
            "source_sha256": {key: value["sha256"] for key, value in sources.items()},
            "split": split,
            "group_id": group_id,
            "review_status": review_status,
            "include_registered_frame": bool(review_image is not None),
            "image_correction_revision_id": correction_reference["revision_id"],
        }
        content_sha256 = _json_digest(content_payload)
        with _index_lock(self.root):
            existing = self._existing_record(
                content_sha256=content_sha256,
                group_id=group_id,
                split=split,
            )
        if existing is not None:
            return {**existing, "reused": True}

        bundle_id = uuid.uuid4().hex
        relative_root = Path("bundles") / bundle_id
        destination = self.root / relative_root
        staging = self.root / f".{bundle_id}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            images = staging / "images"
            images.mkdir()
            np.savez_compressed(images / "baseline.npz", array=baseline)
            frame_records = [
                {
                    "name": "baseline",
                    "relative_path": "images/baseline.npz",
                    "recommended_for_facies_training": True,
                    "contains_experiment_state": False,
                }
            ]
            if review_image is not None:
                np.savez_compressed(
                    images / "registered_review_frame.npz",
                    array=np.asarray(review_image, dtype=np.float32),
                )
                frame_records.append(
                    {
                        "name": "registered_review_frame",
                        "relative_path": "images/registered_review_frame.npz",
                        "recommended_for_facies_training": False,
                        "contains_experiment_state": True,
                        "registration": review_provenance,
                    }
                )
            np.savez_compressed(
                staging / "targets.npz",
                labels=labels,
                facies=facies,
                roles=roles,
                context_mask=context_mask,
                training_mask=training_mask,
                material_classes=material_classes,
                hydraulic_classes=hydraulic_classes,
                depth_m=depth,
            )
            with (staging / "regions.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                fieldnames = [
                    "label_id",
                    "facies_id",
                    "name",
                    "analysis_role",
                    "enabled",
                    "porosity",
                    "permeability",
                    "permeability_unit",
                ]
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for raw_region in config.get("regions") or []:
                    region = dict(raw_region)
                    writer.writerow(
                        {
                            **{key: region.get(key) for key in fieldnames},
                            "permeability_unit": str(
                                config.get("permeability_unit") or "source_unit"
                            ),
                        }
                    )
            Image.fromarray(_preview(baseline, labels, facies, context_mask)).save(
                staging / "preview.png"
            )

            artifact_paths = {
                "baseline": Path("images/baseline.npz"),
                "targets": Path("targets.npz"),
                "regions": Path("regions.csv"),
                "preview": Path("preview.png"),
            }
            if review_image is not None:
                artifact_paths["registered_review_frame"] = Path(
                    "images/registered_review_frame.npz"
                )
            manifest = {
                "schema": 1,
                "format": FACIES_ANNOTATION_FORMAT,
                "bundle_id": bundle_id,
                "created_utc": utc_now(),
                "created_by": str(actor).strip(),
                "content_sha256": content_sha256,
                "split": split,
                "group_id": group_id,
                "review_status": review_status,
                "eligible_for_training": review_status == "approved",
                "project": {
                    "project_id": project.project_id,
                    "series_id": str(project.manifest.get("series_id") or ""),
                    "name": project.name,
                    "template_id": str(
                        (project.manifest.get("template") or {}).get("id") or ""
                    ),
                    "rig_type": str(
                        (project.manifest.get("rig") or {}).get("type") or ""
                    ),
                    "indicator_system": str(
                        (project.manifest.get("chemistry") or {}).get(
                            "indicator_system"
                        )
                        or ""
                    ),
                },
                "annotation_scope": {
                    "domain": "registered_camera_plane_2d",
                    "labels": "detailed_physical_regions",
                    "facies": "shared_or_layer_specific_material_classes",
                    "depth_is_separate": True,
                    "contains_3d_extrapolation": False,
                },
                "geometry": {
                    "revision_id": str(geometry_revision.get("revision_id") or ""),
                    "config_sha256": geometry_config_digest(config),
                    "width_m": config.get("width_m"),
                    "height_m": config.get("height_m"),
                    "coordinate_convention": config.get("coordinate_convention"),
                    "grouping_mode": config.get("grouping_mode"),
                    "injectors": list(config.get("injectors") or []),
                    "label_edit_session": edit_reference,
                },
                "image_correction": correction_reference,
                "roles": ROLE_IDS,
                "material_classes": material_definitions,
                "hydraulic_classes": {
                    "unit": "mD",
                    "source_unit": str(
                        config.get("permeability_unit") or "source_unit"
                    ),
                    "eligible": str(
                        config.get("permeability_unit") or "source_unit"
                    ).casefold()
                    not in {"", "source_unit"},
                    "definitions": hydraulic_definitions,
                },
                "frames": frame_records,
                "spatial_shape": list(labels.shape),
                "sources": sources,
                "artifacts": {
                    key: _artifact_reference(staging / path, path)
                    for key, path in artifact_paths.items()
                },
                "validation_summary": dict(validation.get("summary") or {}),
            }
            _atomic_json(staging / "manifest.json", manifest)
            self.bundles_root.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        record = {
            "schema": 1,
            "format": FACIES_DATASET_INDEX_FORMAT,
            "bundle_id": bundle_id,
            "bundle_relpath": relative_root.as_posix(),
            "manifest_relpath": (relative_root / "manifest.json").as_posix(),
            "created_utc": manifest["created_utc"],
            "series_id": manifest["project"]["series_id"],
            "project_id": project.project_id,
            "geometry_revision_id": manifest["geometry"]["revision_id"],
            "split": split,
            "group_id": group_id,
            "review_status": review_status,
            "eligible_for_training": manifest["eligible_for_training"],
            "content_sha256": content_sha256,
        }
        try:
            with _index_lock(self.root):
                existing = self._existing_record(
                    content_sha256=content_sha256,
                    group_id=group_id,
                    split=split,
                )
                if existing is None:
                    _append_jsonl(self.index_path, record)
                else:
                    shutil.rmtree(destination, ignore_errors=True)
                    return {**existing, "reused": True}
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        project._audit(
            "facies_annotation_exported",
            actor,
            {
                "bundle_id": bundle_id,
                "dataset_root": str(self.root),
                "split": split,
                "group_id": group_id,
                "review_status": review_status,
            },
        )
        return {**record, "reused": False}

    def verify(self, record: Mapping[str, Any]) -> dict[str, Any]:
        manifest_path = self.root / str(record.get("manifest_relpath") or "")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("format") != FACIES_ANNOTATION_FORMAT:
            raise ValueError(f"Unsupported facies bundle: {manifest_path}")
        bundle_root = manifest_path.parent
        for artifact in (manifest.get("artifacts") or {}).values():
            path = bundle_root / str(artifact["relative_path"])
            if not path.is_file() or sha256_file(path) != str(artifact["sha256"]):
                raise ValueError(f"Facies bundle artifact failed verification: {path}")
        with np.load(bundle_root / "targets.npz", allow_pickle=False) as archive:
            shapes = {
                tuple(np.asarray(archive[key]).shape)
                for key in (
                    "labels",
                    "facies",
                    "roles",
                    "context_mask",
                    "training_mask",
                    "material_classes",
                    "hydraulic_classes",
                    "depth_m",
                )
            }
        if len(shapes) != 1:
            raise ValueError("Facies bundle target maps have inconsistent shapes")
        return dict(manifest)


def collect_facies_annotation_bundles(
    roots: Sequence[Path],
    *,
    include_drafts: bool = False,
    verify: bool = True,
) -> list[dict[str, Any]]:
    """Collect multiple portable stores while enforcing group-level split isolation."""

    collected: list[dict[str, Any]] = []
    group_splits: dict[str, set[str]] = {}
    for root in roots:
        store = FaciesAnnotationStore(Path(root))
        for record in store.records():
            if not include_drafts and not bool(record.get("eligible_for_training")):
                continue
            group_id = str(record.get("group_id") or "")
            split = str(record.get("split") or "")
            group_splits.setdefault(group_id, set()).add(split)
            manifest = store.verify(record) if verify else {}
            collected.append(
                {
                    **dict(record),
                    "dataset_root": str(store.root),
                    "manifest_path": str(
                        store.root / str(record.get("manifest_relpath") or "")
                    ),
                    "manifest": manifest,
                }
            )
    conflicts = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if group and len(splits) > 1
    }
    if conflicts:
        raise ValueError(f"Facies dataset split leakage detected: {conflicts}")
    return sorted(
        collected,
        key=lambda row: (
            str(row.get("split") or ""),
            str(row.get("group_id") or ""),
            str(row.get("created_utc") or ""),
        ),
    )
