"""Versioned geometry, facies, and injector definitions for FluidFlower projects."""

from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

import csv
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Mapping, TYPE_CHECKING
import uuid

import numpy as np
import pandas as pd
from PIL import Image

if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject


GEOMETRY_REVISION_FORMAT = "darsia-fluidflower-geometry-facies-v1"
RUN_GEOMETRY_FORMAT = "darsia-fluidflower-run-geometry-v1"
RUN_GEOMETRY_ACQUISITION_STATES = ("clearwater", "indicator", "unknown")
RUN_GEOMETRY_POLICIES = (
    "series_reference_only",
    "allow_reviewed_run_overrides",
    "require_reviewed_run_overrides",
)
RUN_GEOMETRY_CHOICES = ("series_reference", "run_override")
GEOMETRY_CONFIG_FORMAT = "darsia-fluidflower-geometry-config-v1"
LABEL_EDIT_FORMAT = "darsia-fluidflower-label-edits-v1"
INJECTION_POINT_CONFLICT_TOLERANCE_M = 1e-4
ANALYSIS_ROLES = {
    "porous_medium",
    "seal",
    "free_water",
    "frame",
    "colorchecker",
    "excluded",
}


def _utc_now() -> str:
    from darsia.fluidflower.project import utc_now

    return utc_now()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _append_jsonl as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _bounded_revision_text(revision: Mapping[str, Any]) -> str:
    """Count the unchanged writer's actual UTF-8/platform-newline bytes."""
    from io import BytesIO, TextIOWrapper
    from .geometry_workspace_frame import METADATA_LIMIT

    text = json.dumps(revision, indent=2, ensure_ascii=False) + "\n"
    with BytesIO() as buffer:
        with TextIOWrapper(buffer, encoding="utf-8", newline=None) as stream:
            stream.write(text)
            stream.flush()
            size = buffer.tell()
    if size > METADATA_LIMIT:
        raise ValueError("Geometry revision exceeds the original 1 MiB metadata limit; materialized arrays were preserved and no revision metadata was published.")
    return text


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Geometry revision at line {line_number} is not an object"
                )
            records.append(payload)
    return records


def _number(value: Any) -> float | None:
    if value is None or str(value).strip().casefold() in {"", "nan", "none"}:
        return None
    parsed = float(str(value).strip().replace(",", "."))
    return parsed if math.isfinite(parsed) else None


def _int(value: Any) -> int | None:
    number = _number(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _load_array(path: Path) -> np.ndarray:
    selected = Path(path).expanduser()
    suffix = selected.suffix.casefold()
    if suffix == ".npz":
        with np.load(_preset_io.source(selected), allow_pickle=False) as archive:
            key = "array" if "array" in archive.files else archive.files[0]
            return np.asarray(archive[key])
    if suffix == ".npy":
        return np.asarray(np.load(_preset_io.source(selected), allow_pickle=False))
    with Image.open(_preset_io.source(selected)) as image:
        return np.asarray(image)


def _darsia_metadata(path: Path) -> dict[str, Any]:
    selected = Path(path).expanduser()
    if selected.suffix.casefold() != ".npz":
        return {}
    try:
        with np.load(_preset_io.source(selected), allow_pickle=True) as archive:
            if "metadata" not in archive.files:
                return {}
            raw = archive["metadata"]
            value = raw.item() if raw.shape == () else raw.tolist()
            return dict(value) if isinstance(value, Mapping) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _shape_and_dtype(path: Path) -> tuple[tuple[int, ...], str]:
    selected = Path(path).expanduser()
    if selected.suffix.casefold() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }:
        with Image.open(selected) as image:
            width, height = image.size
            channels = len(image.getbands())
            shape = (height, width) if channels == 1 else (height, width, channels)
            return shape, str(image.mode)
    array = _load_array(selected)
    return tuple(int(value) for value in array.shape), str(array.dtype)


def _file_reference(
    path: str | Path, *, include_full_sha256: bool = False
) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    stat = selected.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with selected.open("rb") as stream:
        digest.update(stream.read(65536))
        if stat.st_size > 65536:
            stream.seek(max(0, stat.st_size - 65536))
            digest.update(stream.read(65536))
    reference = {
        "path": str(selected),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": digest.hexdigest(),
    }
    if include_full_sha256:
        full_digest = hashlib.sha256()
        with selected.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                full_digest.update(chunk)
        reference["sha256"] = full_digest.hexdigest()
    return reference


def _same_path(left: str | Path, right: str | Path) -> bool:
    if not str(left).strip() or not str(right).strip():
        return False
    left_text = os.path.normcase(str(Path(left).expanduser().resolve()))
    right_text = os.path.normcase(str(Path(right).expanduser().resolve()))
    return left_text == right_text


def geometry_config_digest(config: Mapping[str, Any]) -> str:
    canonical = deepcopy(dict(config))
    canonical["run_geometry"] = normalize_run_geometry_config(
        canonical.get("run_geometry")
    )
    for key in ("display_image_path", "draft_source_image_path"):
        if not str(canonical.get(key) or ""):
            canonical.pop(key, None)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def default_run_geometry_config() -> dict[str, Any]:
    """Return the non-destructive per-run layer-geometry contract."""

    return {
        "schema": 1,
        "format": RUN_GEOMETRY_FORMAT,
        "mode": "series_reference_with_per_run_layer_overrides",
        "geometry_policy": "allow_reviewed_run_overrides",
        "property_policy": "inherit_approved_series_properties",
        "automatic_property_adjustment": False,
        "change_detection": {
            "enabled": True,
            "warning_displacement_fraction_of_height": 0.005,
            "minimum_affected_boundary_fraction": 0.10,
        },
        "runs": [],
    }


def _effective_run_geometry_choice(row: Mapping[str, Any]) -> str:
    choice = str(row.get("geometry_choice") or "").strip()
    if choice in RUN_GEOMETRY_CHOICES:
        return choice
    if (
        str(row.get("labels_path") or "").strip()
        or str(row.get("facies_path") or "").strip()
    ):
        return "run_override"
    return "series_reference"


def normalize_run_geometry_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge older per-run contracts with current explicit policy defaults."""

    normalized = default_run_geometry_config()
    incoming = dict(config or {})
    for key, value in incoming.items():
        if key not in {"change_detection", "runs"}:
            normalized[key] = deepcopy(value)
    detection = dict(normalized["change_detection"])
    detection.update(dict(incoming.get("change_detection") or {}))
    normalized["change_detection"] = detection
    normalized["runs"] = []
    for raw in incoming.get("runs") or []:
        if not isinstance(raw, Mapping):
            continue
        row = deepcopy(dict(raw))
        row["geometry_choice"] = _effective_run_geometry_choice(row)
        normalized["runs"].append(row)
    return normalized


def _layer_boundary_mask(labels: np.ndarray, layer_label_ids: set[int]) -> np.ndarray:
    array = np.asarray(labels)
    active = np.isin(array, list(layer_label_ids))
    boundaries = np.zeros(array.shape, dtype=bool)
    vertical = (array[1:, :] != array[:-1, :]) & (active[1:, :] | active[:-1, :])
    horizontal = (array[:, 1:] != array[:, :-1]) & (active[:, 1:] | active[:, :-1])
    boundaries[1:, :] |= vertical
    boundaries[:-1, :] |= vertical
    boundaries[:, 1:] |= horizontal
    boundaries[:, :-1] |= horizontal
    return boundaries


def screen_layer_boundary_change(
    reference_labels: np.ndarray,
    run_labels: np.ndarray,
    *,
    width_m: float,
    height_m: float,
    layer_label_ids: set[int],
    warning_displacement_fraction_of_height: float = 0.005,
    minimum_affected_boundary_fraction: float = 0.10,
) -> dict[str, Any]:
    """Measure registered layer-boundary changes without applying an override."""

    from scipy import ndimage as ndi

    reference = np.squeeze(np.asarray(reference_labels))
    candidate = np.squeeze(np.asarray(run_labels))
    result: dict[str, Any] = {
        "status": "not_comparable",
        "substantial_change": False,
        "warning_displacement_fraction_of_height": float(
            warning_displacement_fraction_of_height
        ),
        "minimum_affected_boundary_fraction": float(minimum_affected_boundary_fraction),
    }
    if reference.ndim != 2 or candidate.ndim != 2 or reference.shape != candidate.shape:
        result["reason"] = "Layer maps must be two-dimensional on the same grid"
        return result
    if not layer_label_ids:
        result["reason"] = "No porous-medium or seal labels are available"
        return result
    if width_m <= 0.0 or height_m <= 0.0:
        result["reason"] = "Positive physical rig dimensions are required"
        return result
    threshold_fraction = float(warning_displacement_fraction_of_height)
    affected_threshold = float(minimum_affected_boundary_fraction)
    if not 0.0 < threshold_fraction < 1.0 or not 0.0 <= affected_threshold <= 1.0:
        result["reason"] = "Change-screen thresholds are outside valid ranges"
        return result

    reference_boundary = _layer_boundary_mask(reference, layer_label_ids)
    candidate_boundary = _layer_boundary_mask(candidate, layer_label_ids)
    result["reference_boundary_pixel_count"] = int(np.count_nonzero(reference_boundary))
    result["run_boundary_pixel_count"] = int(np.count_nonzero(candidate_boundary))
    if not np.any(reference_boundary) or not np.any(candidate_boundary):
        result["reason"] = "One or both maps contain no comparable layer boundaries"
        return result

    rows, columns = reference.shape
    sampling = (float(height_m) / rows, float(width_m) / columns)
    distance_to_reference = ndi.distance_transform_edt(
        ~reference_boundary, sampling=sampling
    )[candidate_boundary]
    distance_to_candidate = ndi.distance_transform_edt(
        ~candidate_boundary, sampling=sampling
    )[reference_boundary]
    symmetric_distances = np.concatenate(
        (distance_to_reference, distance_to_candidate)
    ).astype(np.float64, copy=False)
    threshold_m = threshold_fraction * float(height_m)
    affected_fraction = float(np.mean(symmetric_distances > threshold_m))
    p95_m = float(np.percentile(symmetric_distances, 95.0))
    active_union = np.isin(reference, list(layer_label_ids)) | np.isin(
        candidate, list(layer_label_ids)
    )
    mismatch_fraction = (
        float(np.mean(reference[active_union] != candidate[active_union]))
        if np.any(active_union)
        else 0.0
    )
    substantial = p95_m > threshold_m and affected_fraction >= affected_threshold
    result.update(
        {
            "status": "substantial_change" if substantial else "within_threshold",
            "substantial_change": substantial,
            "threshold_m": threshold_m,
            "median_displacement_m": float(np.median(symmetric_distances)),
            "p95_displacement_m": p95_m,
            "maximum_displacement_m": float(np.max(symmetric_distances)),
            "affected_boundary_fraction": affected_fraction,
            "label_mismatch_fraction": mismatch_fraction,
        }
    )
    return result


def default_geometry_config(
    project: "FluidFlowerProject" | None = None,
) -> dict[str, Any]:
    rig_type = "custom"
    if project is not None:
        rig_type = str((project.manifest.get("rig") or {}).get("type") or "custom")
    defaults = {
        "medium": (0.9, 0.49),
    }
    width_m, height_m = defaults.get(rig_type, (None, None))
    return {
        "schema": 1,
        "format": GEOMETRY_CONFIG_FORMAT,
        "frame": "raw",
        "rig_type": rig_type,
        "source_rig_path": "",
        "baseline_path": "",
        "depth_path": "",
        "labels_path": "",
        "facies_source_path": "",
        "facies_properties_source_path": "",
        "protocol_source_path": "",
        "display_image_path": "",
        "draft_source_image_path": "",
        "width_m": width_m,
        "height_m": height_m,
        "coordinate_convention": "origin_lower_left_x_right_y_up",
        "grouping_mode": "shared_facies",
        "permeability_unit": "source_unit",
        "regions": [],
        "injectors": [],
        "run_geometry": default_run_geometry_config(),
    }


def _load_properties(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not str(path) or not path.is_file():
        return {}
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Facies properties must use CSV, XLSX, or XLS")
    normalized = {str(column).strip().casefold(): column for column in frame.columns}
    id_column = normalized.get("id") or normalized.get("facies_id")
    if id_column is None:
        raise ValueError("Facies properties need an 'id' or 'facies_id' column")
    records: dict[int, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        facies_id = _int(row[id_column])
        if facies_id is None:
            continue
        records[facies_id] = {
            "facies_id": facies_id,
            "name": str(row.get(normalized.get("name"), f"Facies {facies_id}")),
            "porosity": _number(row.get(normalized.get("porosity"))),
            "permeability": _number(row.get(normalized.get("permeability"))),
        }
    return records


def _injectors_from_protocol(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not str(path):
        return []
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return []

    files = (
        [source]
        if source.is_file()
        else [
            candidate
            for candidate in sorted(source.iterdir())
            if candidate.suffix.casefold() in {".csv", ".xlsx", ".xls"}
        ]
    )
    tables: list[tuple[Path, str, pd.DataFrame]] = []
    for selected in files:
        suffix = selected.suffix.casefold()
        if suffix == ".csv":
            tables.append((selected, "", pd.read_csv(selected)))
            continue
        if suffix not in {".xlsx", ".xls"}:
            continue
        sheets = pd.read_excel(selected, sheet_name=None)
        injection_sheets = [
            (name, frame)
            for name, frame in sheets.items()
            if str(name).strip().casefold() in {"injection points", "injectors"}
        ]
        for sheet_name, frame in injection_sheets or list(sheets.items()):
            tables.append((selected, str(sheet_name), frame))

    records: dict[str, dict[str, Any]] = {}
    for selected, sheet_name, frame in tables:
        columns = {str(column).strip().casefold(): column for column in frame.columns}

        def column(*names: str) -> Any:
            return next((columns[name] for name in names if name in columns), None)

        location_column = column(
            "location_id", "injector_id", "injection_point", "location"
        )
        label_column = column("label", "name", "injector_name")
        run_column = column("run_id", "run")
        x_column = column("location_x", "x_m", "injector_x")
        y_column = column("location_y", "y_m", "injector_y")
        if location_column is None and (x_column is None or y_column is None):
            continue
        for _, row in frame.iterrows():
            location_id = (
                str(row.get(location_column) or "").strip()
                if location_column is not None
                else ""
            )
            if location_id.casefold() in {"nan", "none"}:
                location_id = ""
            x_m = _number(row.get(x_column)) if x_column is not None else None
            y_m = _number(row.get(y_column)) if y_column is not None else None
            if not location_id and (x_m is None or y_m is None):
                continue
            key = (
                f"id:{location_id.casefold()}"
                if location_id
                else f"point:{x_m:.12g}:{y_m:.12g}"
            )
            label = (
                str(row.get(label_column) or "").strip()
                if label_column is not None
                else ""
            )
            if label.casefold() in {"nan", "none"}:
                label = ""
            run_id = (
                str(row.get(run_column) or "").strip().casefold()
                if run_column is not None
                else ""
            )
            candidate = records.setdefault(
                key,
                {
                    "injector_id": location_id,
                    "protocol_location_id": location_id,
                    "name": label
                    or _injection_point_display_name(
                        location_id, fallback_number=len(records) + 1
                    ),
                    "x_m": None,
                    "y_m": None,
                    "uncertainty_m": None,
                    "coordinate_source": (
                        f"protocol:{selected.name}"
                        if x_m is not None and y_m is not None
                        else f"protocol identity:{selected.name}; position pending GUI"
                    ),
                    "protocol_x_m": None,
                    "protocol_y_m": None,
                    "protocol_coordinate_candidates": [],
                },
            )
            if not candidate.get("protocol_location_id") and location_id:
                candidate["protocol_location_id"] = location_id
            if x_m is None or y_m is None:
                continue
            protocol_position = {
                "run_id": run_id,
                "x_m": x_m,
                "y_m": y_m,
                "source": str(selected),
                "sheet": sheet_name,
            }
            candidates = candidate["protocol_coordinate_candidates"]
            if not any(
                math.isclose(float(item["x_m"]), x_m, abs_tol=1e-12)
                and math.isclose(float(item["y_m"]), y_m, abs_tol=1e-12)
                and str(item.get("run_id") or "") == run_id
                for item in candidates
            ):
                candidates.append(protocol_position)
            if candidate.get("protocol_x_m") is None:
                candidate["x_m"] = x_m
                candidate["y_m"] = y_m
                candidate["protocol_x_m"] = x_m
                candidate["protocol_y_m"] = y_m
                candidate["coordinate_source"] = f"protocol:{selected.name}"
    output = list(records.values())
    used_ids = {
        str(row.get("injector_id") or "").strip().casefold()
        for row in output
        if str(row.get("injector_id") or "").strip()
    }
    next_number = 1
    for row in output:
        if str(row.get("injector_id") or "").strip():
            continue
        while f"ip{next_number}" in used_ids:
            next_number += 1
        row["injector_id"] = f"ip{next_number}"
        if not str(row.get("name") or "").strip():
            row["name"] = f"Injection Point {next_number}"
        used_ids.add(f"ip{next_number}")
        next_number += 1
    return output


def load_protocol_injection_points(path: Path) -> list[dict[str, Any]]:
    """Load protocol identities and optional x/y positions for GUI review."""

    return _injectors_from_protocol(path)


def _injection_point_display_name(location_id: str, *, fallback_number: int) -> str:
    """Return a readable GUI label while preserving the protocol ID."""

    normalized = str(location_id or "").strip().casefold()
    if normalized.startswith("ip") and normalized[2:].isdigit():
        return f"Injection Point {int(normalized[2:])}"
    if location_id:
        return str(location_id).replace("_", " ").replace("-", " ").title()
    return f"Injection Point {fallback_number}"


def _rgb_float(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected an RGB image, received shape {image.shape}")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    else:
        image = np.asarray(image, dtype=np.float32)
        finite = image[np.isfinite(image)]
        if finite.size and float(np.max(finite)) > 1.5:
            image /= 255.0
    return np.clip(np.nan_to_num(image), 0.0, 1.0)


def load_registered_geometry_image(
    config: Mapping[str, Any], path: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one review frame in the same coordinate system as the label map."""

    selected = Path(path).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(selected)
    correction_paths: list[Path] = []
    rig_root = Path(str(config.get("source_rig_path") or "")).expanduser()
    optical_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    if selected.suffix.casefold() in optical_suffixes and rig_root.is_dir():
        shape_paths = sorted(rig_root.glob("shape_correction_*.npz"))
        color_paths = sorted(rig_root.glob("color_correction_*.npz"))
        correction_paths = shape_paths + color_paths
        if not correction_paths:
            correction_paths = sorted(rig_root.glob("correction_*.npz"))

    if correction_paths:
        import darsia

        corrections = [darsia.read_correction(item) for item in correction_paths]
        image = darsia.imread(
            selected,
            transformations=corrections,
            dtype=np.float32,
        ).img
        mode = "saved_rig_corrections"
    else:
        image = _load_array(selected)
        mode = "already_registered"

    rgb = _rgb_float(np.asarray(image))
    expected_shape, _dtype = _shape_and_dtype(Path(str(config["baseline_path"])))
    if tuple(rgb.shape[:2]) != tuple(expected_shape[:2]):
        raise ValueError(
            "The selected frame does not match the geometry grid after correction: "
            f"{rgb.shape[:2]} != {expected_shape[:2]}"
        )
    return rgb, {
        "path": str(selected),
        "mode": mode,
        "corrections": [item.name for item in correction_paths],
    }


def _save_darsia_field(
    path: Path,
    array: np.ndarray,
    *,
    width_m: float,
    height_m: float,
) -> None:
    metadata = {
        "space_dim": 2,
        "indexing": "ij",
        "dimensions": [float(height_m), float(width_m)],
        "origin": f"[0.0, {float(height_m)}]",
    }
    np.savez_compressed(path, array=array, metadata=metadata)


def create_optical_facies_draft(
    source_image_path: Path,
    destination: Path,
    *,
    width_m: float,
    height_m: float,
    depth_m: float,
    optical_classes: int = 6,
    rig_type: str = "custom",
    max_analysis_width: int = 1000,
    random_seed: int = 1729,
) -> dict[str, Any]:
    """Create a conservative, unsupervised facies draft from one baseline image.

    The draft only groups optically similar regions. It deliberately assigns no
    physical role or material property; those require expert review.
    """

    import cv2
    from scipy import ndimage as ndi
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from skimage.measure import label as connected_labels
    from skimage.segmentation import slic

    selected = Path(source_image_path).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(selected)
    if width_m <= 0.0 or height_m <= 0.0 or depth_m <= 0.0:
        raise ValueError("Width, height, and depth must be positive")
    if not 2 <= int(optical_classes) <= 24:
        raise ValueError("The optical draft supports 2 to 24 classes")

    source = _rgb_float(_load_array(selected))
    source_height, source_width = source.shape[:2]
    scale = min(1.0, float(max_analysis_width) / float(source_width))
    target = (
        max(8, int(round(source_width * scale))),
        max(8, int(round(source_height * scale))),
    )
    reduced = cv2.resize(source, target, interpolation=cv2.INTER_AREA)
    reduced8 = np.rint(255.0 * reduced).astype(np.uint8)
    lab = cv2.cvtColor(reduced8, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(reduced8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(gray, (0, 0), 3.0)
    local_square = cv2.GaussianBlur(gray * gray, (0, 0), 3.0)
    texture = np.sqrt(np.maximum(local_square - local_mean * local_mean, 0.0))
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    row_coordinate = np.linspace(0.0, 1.0, reduced.shape[0], dtype=np.float32)
    row_coordinate = np.broadcast_to(row_coordinate[:, None], gray.shape)

    requested_segments = int(
        np.clip(reduced.shape[0] * reduced.shape[1] / 900.0, 250, 1800)
    )
    superpixels = slic(
        reduced,
        n_segments=requested_segments,
        compactness=12.0,
        sigma=1.0,
        start_label=0,
        channel_axis=-1,
    ).astype(np.int32)
    count = int(superpixels.max()) + 1
    flat = superpixels.ravel()
    weights = np.bincount(flat, minlength=count).astype(np.float64)
    feature_images = [
        lab[..., 0],
        lab[..., 1],
        lab[..., 2],
        texture,
        gradient,
        row_coordinate,
    ]
    features = np.column_stack(
        [
            np.bincount(flat, weights=np.asarray(field).ravel(), minlength=count)
            / np.maximum(weights, 1.0)
            for field in feature_images
        ]
    )
    features[:, -1] *= 0.45
    normalized = StandardScaler().fit_transform(features)
    class_count = min(int(optical_classes), count)
    model = KMeans(
        n_clusters=class_count,
        random_state=int(random_seed),
        n_init=10,
    )
    cluster = model.fit_predict(normalized)
    cluster_rows = [
        float(np.median(features[cluster == value, -1])) for value in range(class_count)
    ]
    order = np.argsort(cluster_rows)
    remap = np.empty(class_count, dtype=np.int32)
    remap[order] = np.arange(1, class_count + 1, dtype=np.int32)
    facies_reduced = remap[cluster][superpixels]
    facies = cv2.resize(
        facies_reduced,
        (source_width, source_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    minimum_region = max(64, int(round(source_height * source_width * 0.0002)))
    cleaned = np.full(facies.shape, -1, dtype=np.int32)
    for facies_id in range(1, class_count + 1):
        components = connected_labels(facies == facies_id, connectivity=1)
        component_sizes = np.bincount(components.ravel())
        stable_ids = np.flatnonzero(component_sizes >= minimum_region)
        stable_ids = stable_ids[stable_ids != 0]
        keep = np.isin(components, stable_ids)
        cleaned[keep] = facies_id
    missing = cleaned < 0
    if np.all(missing):
        raise RuntimeError("Optical clustering produced no stable regions")
    if np.any(missing):
        nearest = ndi.distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        cleaned[missing] = cleaned[tuple(nearest[:, missing])]
    facies = cleaned

    labels = np.zeros(facies.shape, dtype=np.int32)
    regions: list[dict[str, Any]] = []
    label_id = 0
    for facies_id in range(1, class_count + 1):
        components = connected_labels(facies == facies_id, connectivity=1)
        for component_id in range(1, int(components.max()) + 1):
            mask = components == component_id
            if not np.any(mask):
                continue
            label_id += 1
            labels[mask] = label_id
            regions.append(
                {
                    "label_id": label_id,
                    "facies_id": facies_id,
                    "name": f"Optical class {facies_id} · region {component_id}",
                    "analysis_role": "excluded",
                    "porosity": None,
                    "permeability": None,
                    "enabled": False,
                }
            )

    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_path = output / "baseline.npz"
    depth_path = output / "depth.npz"
    labels_path = output / "labels.npz"
    facies_path = output / "facies.npz"
    _save_darsia_field(
        baseline_path,
        source.astype(np.float32),
        width_m=width_m,
        height_m=height_m,
    )
    _save_darsia_field(
        depth_path,
        np.full(labels.shape, depth_m, dtype=np.float32),
        width_m=width_m,
        height_m=height_m,
    )
    _save_darsia_field(
        labels_path,
        labels,
        width_m=width_m,
        height_m=height_m,
    )
    _save_darsia_field(
        facies_path,
        facies,
        width_m=width_m,
        height_m=height_m,
    )
    return {
        "schema": 1,
        "format": GEOMETRY_CONFIG_FORMAT,
        "rig_type": str(rig_type),
        "source_rig_path": str(output),
        "baseline_path": str(baseline_path),
        "depth_path": str(depth_path),
        "labels_path": str(labels_path),
        "facies_source_path": str(facies_path),
        "facies_properties_source_path": "",
        "protocol_source_path": "",
        "display_image_path": str(selected),
        "draft_source_image_path": str(selected),
        "width_m": float(width_m),
        "height_m": float(height_m),
        "coordinate_convention": "origin_lower_left_x_right_y_up",
        "grouping_mode": "shared_facies",
        "permeability_unit": "source_unit",
        "regions": regions,
        "injectors": [],
        "facies_generation": {
            "method": "slic_lab_texture_kmeans_v1",
            "status": "unreviewed_optical_draft",
            "source_image_path": str(selected),
            "optical_classes": class_count,
            "superpixel_count": count,
            "physical_region_count": len(regions),
            "analysis_width": target[0],
            "analysis_height": target[1],
            "random_seed": int(random_seed),
        },
    }


class GeometryLabelEditSession:
    """Append-only, full-resolution edits applied over an immutable label map."""

    def __init__(
        self,
        *,
        base_labels_path: Path,
        session_dir: Path,
        width_m: float,
        height_m: float,
        session_id: str | None = None,
    ) -> None:
        self.base_labels_path = Path(base_labels_path).expanduser().resolve()
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id or self.session_dir.name)
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        if self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("Label editing requires positive physical dimensions")
        base = np.squeeze(_load_array(self.base_labels_path))
        if base.ndim != 2:
            raise ValueError("Label editing requires one two-dimensional label map")
        self.base_labels = np.asarray(base, dtype=np.int32)
        self.labels = self.base_labels.copy()
        self.events_path = self.session_dir / "events.jsonl"
        self.manifest_path = self.session_dir / "session.json"
        if not self.manifest_path.is_file():
            _atomic_json(
                self.manifest_path,
                {
                    "schema": 1,
                    "format": LABEL_EDIT_FORMAT,
                    "session_id": self.session_id,
                    "created_utc": _utc_now(),
                    "base_labels": _file_reference(self.base_labels_path),
                    "width_m": self.width_m,
                    "height_m": self.height_m,
                },
            )
        self.events = _read_jsonl(self.events_path)
        self._rebuild()

    @classmethod
    def create(
        cls,
        config: Mapping[str, Any],
        sessions_root: Path,
    ) -> "GeometryLabelEditSession":
        root = Path(sessions_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        previous = dict(config.get("label_edit_session") or {})
        previous_events = Path(str(previous.get("events_path") or "")).expanduser()
        if previous and (
            previous_events.is_file()
            or (previous_events.parent / "session.json").is_file()
        ):
            return cls(
                base_labels_path=Path(str(previous["base_labels_path"])),
                session_dir=previous_events.parent,
                width_m=float(config["width_m"]),
                height_m=float(config["height_m"]),
                session_id=str(
                    previous.get("session_id") or previous_events.parent.name
                ),
            )
        active_path = root / "active.json"
        if active_path.is_file():
            try:
                active = json.loads(active_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError, TypeError):
                active = {}
            active_events = Path(str(active.get("events_path") or "")).expanduser()
            active_base = Path(str(active.get("base_labels_path") or "")).expanduser()
            requested_base = Path(str(config["labels_path"])).expanduser().resolve()
            if (
                active_base.is_file()
                and active_base.resolve() == requested_base
                and (active_events.parent / "session.json").is_file()
                and math.isclose(
                    float(active.get("width_m") or 0.0), float(config["width_m"])
                )
                and math.isclose(
                    float(active.get("height_m") or 0.0), float(config["height_m"])
                )
            ):
                return cls(
                    base_labels_path=active_base,
                    session_dir=active_events.parent,
                    width_m=float(config["width_m"]),
                    height_m=float(config["height_m"]),
                    session_id=str(
                        active.get("session_id") or active_events.parent.name
                    ),
                )
        session_id = uuid.uuid4().hex
        session = cls(
            base_labels_path=Path(str(config["labels_path"])),
            session_dir=root / session_id,
            width_m=float(config["width_m"]),
            height_m=float(config["height_m"]),
            session_id=session_id,
        )
        _atomic_json(
            active_path,
            {
                **session.config_reference(),
                "width_m": session.width_m,
                "height_m": session.height_m,
            },
        )
        return session

    def _state(self) -> tuple[set[str], list[str]]:
        operations: dict[str, dict[str, Any]] = {}
        active: list[str] = []
        redo: list[str] = []
        for event in self.events:
            kind = str(event.get("kind") or "")
            event_id = str(event.get("event_id") or "")
            if kind in {"polygon", "brush"}:
                operations[event_id] = event
                active.append(event_id)
                redo.clear()
            elif kind == "undo":
                target = str(event.get("target_event_id") or "")
                if target in active:
                    active.remove(target)
                    redo.append(target)
            elif kind == "redo":
                target = str(event.get("target_event_id") or "")
                if target in operations and target not in active:
                    active.append(target)
                if target in redo:
                    redo.remove(target)
        return set(active), redo

    def _points_to_pixels(self, points_m: list[list[float]]) -> np.ndarray:
        height, width = self.base_labels.shape
        return np.asarray(
            [
                [
                    int(round(np.clip(x_m / self.width_m, 0.0, 1.0) * (width - 1))),
                    int(
                        round(
                            (1.0 - np.clip(y_m / self.height_m, 0.0, 1.0))
                            * (height - 1)
                        )
                    ),
                ]
                for x_m, y_m in points_m
            ],
            dtype=np.int32,
        )

    def _event_mask(self, event: Mapping[str, Any]) -> np.ndarray:
        import cv2

        points = self._points_to_pixels(list(event.get("points_m") or []))
        mask = np.zeros(self.base_labels.shape, dtype=np.uint8)
        kind = str(event.get("kind") or "")
        if kind == "polygon" and len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
        elif kind == "brush" and len(points) >= 1:
            radius_m = float(event.get("brush_radius_m") or 0.0)
            radius_x = max(
                1,
                int(
                    round(
                        radius_m / self.width_m * max(1, self.base_labels.shape[1] - 1)
                    )
                ),
            )
            radius_y = max(
                1,
                int(
                    round(
                        radius_m / self.height_m * max(1, self.base_labels.shape[0] - 1)
                    )
                ),
            )
            centers: set[tuple[int, int]] = set()
            for start, end in zip(points[:-1], points[1:]):
                delta = end - start
                steps = max(
                    1,
                    int(
                        math.ceil(
                            2.0
                            * max(
                                abs(float(delta[0])) / radius_x,
                                abs(float(delta[1])) / radius_y,
                            )
                        )
                    ),
                )
                for index in range(steps + 1):
                    location = np.rint(start + delta * (index / steps)).astype(int)
                    centers.add((int(location[0]), int(location[1])))
            if len(points) == 1:
                centers.add((int(points[0, 0]), int(points[0, 1])))
            for center in centers:
                cv2.ellipse(
                    mask,
                    center,
                    (radius_x, radius_y),
                    0.0,
                    0.0,
                    360.0,
                    1,
                    -1,
                    cv2.LINE_8,
                )
        return mask.astype(bool)

    def _rebuild(self) -> None:
        active, _redo = self._state()
        self.labels = self.base_labels.copy()
        for event in self.events:
            if str(event.get("event_id") or "") not in active:
                continue
            if str(event.get("kind") or "") not in {"polygon", "brush"}:
                continue
            self.labels[self._event_mask(event)] = int(event["target_label"])

    def apply(
        self,
        *,
        kind: str,
        points_m: list[list[float]],
        target_label: int,
        actor: str,
        brush_radius_m: float = 0.0,
    ) -> dict[str, Any]:
        if kind not in {"polygon", "brush"}:
            raise ValueError("Label edit kind must be polygon or brush")
        if kind == "polygon" and len(points_m) < 3:
            raise ValueError("A polygon needs at least three points")
        if kind == "brush" and len(points_m) < 1:
            raise ValueError("A brush stroke needs at least one point")
        if int(target_label) < 0:
            raise ValueError("Target labels must be non-negative integers")
        event = {
            "schema": 1,
            "format": LABEL_EDIT_FORMAT,
            "event_id": uuid.uuid4().hex,
            "recorded_utc": _utc_now(),
            "recorded_by": str(actor).strip(),
            "kind": kind,
            "target_label": int(target_label),
            "points_m": [[float(x), float(y)] for x, y in points_m],
            "brush_radius_m": float(brush_radius_m),
        }
        mask = self._event_mask(event)
        event["affected_pixel_count"] = int(
            np.count_nonzero(self.labels[mask] != target_label)
        )
        _append_jsonl(self.events_path, event)
        self.events.append(event)
        self.labels[mask] = int(target_label)
        return deepcopy(event)

    def undo(self, *, actor: str) -> dict[str, Any] | None:
        active, _redo = self._state()
        target = next(
            (
                str(event.get("event_id") or "")
                for event in reversed(self.events)
                if str(event.get("kind") or "") in {"polygon", "brush"}
                and str(event.get("event_id") or "") in active
            ),
            "",
        )
        if not target:
            return None
        event = {
            "schema": 1,
            "format": LABEL_EDIT_FORMAT,
            "event_id": uuid.uuid4().hex,
            "recorded_utc": _utc_now(),
            "recorded_by": str(actor).strip(),
            "kind": "undo",
            "target_event_id": target,
        }
        _append_jsonl(self.events_path, event)
        self.events.append(event)
        self._rebuild()
        return deepcopy(event)

    def redo(self, *, actor: str) -> dict[str, Any] | None:
        _active, redo = self._state()
        if not redo:
            return None
        target = redo[-1]
        event = {
            "schema": 1,
            "format": LABEL_EDIT_FORMAT,
            "event_id": uuid.uuid4().hex,
            "recorded_utc": _utc_now(),
            "recorded_by": str(actor).strip(),
            "kind": "redo",
            "target_event_id": target,
        }
        _append_jsonl(self.events_path, event)
        self.events.append(event)
        self._rebuild()
        return deepcopy(event)

    def active_operation_count(self) -> int:
        active, _redo = self._state()
        return len(active)

    def can_undo(self) -> bool:
        return self.active_operation_count() > 0

    def can_redo(self) -> bool:
        _active, redo = self._state()
        return bool(redo)

    def preview(self, shape: tuple[int, int]) -> np.ndarray:
        import cv2

        return cv2.resize(
            self.labels,
            (int(shape[1]), int(shape[0])),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

    def materialize(self) -> dict[str, Any]:
        materialization_id = uuid.uuid4().hex
        destination = self.session_dir / "materializations" / materialization_id
        staging = destination.with_name(f".{materialization_id}.{uuid.uuid4().hex}.tmp")
        staging.mkdir(parents=True, exist_ok=False)
        try:
            metadata = _darsia_metadata(self.base_labels_path)
            np.savez_compressed(
                staging / "labels.npz", array=self.labels, metadata=metadata
            )
            manifest = {
                "schema": 1,
                "format": "darsia-fluidflower-label-materialization-v1",
                "materialization_id": materialization_id,
                "recorded_utc": _utc_now(),
                "session_id": self.session_id,
                "base_labels_path": str(self.base_labels_path),
                "events_path": str(self.events_path),
                "active_operation_count": self.active_operation_count(),
                "shape": list(self.labels.shape),
            }
            _atomic_json(staging / "materialization.json", manifest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return {
            **manifest,
            "labels_path": str(destination / "labels.npz"),
            "manifest_path": str(destination / "materialization.json"),
        }

    def config_reference(self) -> dict[str, Any]:
        return {
            "format": LABEL_EDIT_FORMAT,
            "session_id": self.session_id,
            "base_labels_path": str(self.base_labels_path),
            "events_path": str(self.events_path),
            "active_operation_count": self.active_operation_count(),
        }


def import_darsia_rig(
    folder: Path,
    *,
    facies_properties_path: Path | None = None,
    protocol_path: Path | None = None,
    rig_type: str = "custom",
) -> dict[str, Any]:
    """Import one saved DarSIA rig without modifying its source artifacts."""

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = {
        "baseline": root / "baseline.npz",
        "depth": root / "depth.npz",
        "labels": root / "labels.npz",
        "facies": root / "facies.npz",
        "porosity": root / "porosity.npz",
        "permeability": root / "permeability.npz",
    }
    for key in ("baseline", "depth", "labels"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"DarSIA rig is missing {paths[key].name}: {root}")

    metadata = _darsia_metadata(paths["baseline"]) or _darsia_metadata(paths["depth"])
    dimensions = list(metadata.get("dimensions") or [])
    height_m = _number(dimensions[0]) if len(dimensions) >= 2 else None
    width_m = _number(dimensions[1]) if len(dimensions) >= 2 else None
    labels = np.asarray(_load_array(paths["labels"]))
    facies = (
        np.asarray(_load_array(paths["facies"]))
        if paths["facies"].is_file()
        else labels
    )
    if labels.shape[:2] != facies.shape[:2]:
        raise ValueError("Imported labels and facies maps have different shapes")
    properties = _load_properties(facies_properties_path)
    porosity_map = (
        np.asarray(_load_array(paths["porosity"]), dtype=np.float64)
        if paths["porosity"].is_file() and not properties
        else None
    )
    permeability_map = (
        np.asarray(_load_array(paths["permeability"]), dtype=np.float64)
        if paths["permeability"].is_file() and not properties
        else None
    )

    regions: list[dict[str, Any]] = []
    for raw_label in sorted(np.unique(labels).tolist()):
        label_id = int(raw_label)
        mask = labels == raw_label
        facies_values, counts = np.unique(facies[mask], return_counts=True)
        facies_id = int(facies_values[int(np.argmax(counts))])
        prop = dict(properties.get(facies_id) or {})
        name = str(prop.get("name") or f"Layer {label_id}")
        porosity = prop.get("porosity")
        permeability = prop.get("permeability")
        if porosity is None and porosity_map is not None:
            values = porosity_map[mask]
            finite = values[np.isfinite(values)]
            porosity = float(np.median(finite)) if finite.size else None
        if permeability is None and permeability_map is not None:
            values = permeability_map[mask]
            finite = values[np.isfinite(values)]
            permeability = float(np.median(finite)) if finite.size else None
        lowered = name.casefold()
        if "water" in lowered:
            role = "free_water"
        elif porosity is None or permeability is None:
            role = "excluded"
        else:
            role = "porous_medium"
        regions.append(
            {
                "label_id": label_id,
                "facies_id": facies_id,
                "name": name,
                "analysis_role": role,
                "porosity": porosity,
                "permeability": permeability,
                "enabled": role not in {"frame", "colorchecker", "excluded"},
            }
        )

    return {
        "schema": 1,
        "format": GEOMETRY_CONFIG_FORMAT,
        "rig_type": str(rig_type),
        "source_rig_path": str(root),
        "baseline_path": str(paths["baseline"]),
        "depth_path": str(paths["depth"]),
        "labels_path": str(paths["labels"]),
        "facies_source_path": str(paths["facies"]) if paths["facies"].is_file() else "",
        "facies_properties_source_path": (
            str(Path(facies_properties_path).expanduser().resolve())
            if facies_properties_path is not None and str(facies_properties_path)
            else ""
        ),
        "protocol_source_path": (
            str(Path(protocol_path).expanduser().resolve())
            if protocol_path is not None and str(protocol_path)
            else ""
        ),
        "display_image_path": "",
        "draft_source_image_path": "",
        "width_m": width_m,
        "height_m": height_m,
        "coordinate_convention": "origin_lower_left_x_right_y_up",
        "grouping_mode": "shared_facies",
        "permeability_unit": "source_unit",
        "regions": regions,
        "injectors": _injectors_from_protocol(protocol_path),
        "run_geometry": default_run_geometry_config(),
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


def depth_and_injector_findings(depth, injector_rows, *, width_m, height_m) -> dict[str, Any]:
    """Return the existing depth and injector findings for supplied inputs."""

    issues: list[dict[str, Any]] = []
    if depth is not None:
        finite_depth = np.asarray(depth, dtype=np.float64)
        finite_depth = finite_depth[np.isfinite(finite_depth)]
        if not finite_depth.size:
            _issue(issues, "error", "empty_depth_map", "Depth map has no finite values")
            depth_stats: dict[str, Any] = {}
        else:
            depth_stats = {
                "minimum_m": float(np.min(finite_depth)),
                "median_m": float(np.median(finite_depth)),
                "maximum_m": float(np.max(finite_depth)),
            }
            if float(np.min(finite_depth)) <= 0.0:
                _issue(
                    issues,
                    "error",
                    "nonpositive_depth",
                    "Depth map contains non-positive values",
                )
            if float(np.max(finite_depth)) > 5.0 * max(
                float(np.median(finite_depth)), 1e-12
            ):
                _issue(
                    issues,
                    "warning",
                    "depth_outliers",
                    "Depth maximum exceeds five times the median",
                )
    else:
        depth_stats = {}

    injectors = [dict(row) for row in injector_rows or []]
    if not injectors:
        _issue(
            issues,
            "warning",
            "no_injectors",
            "No injection-point positions are defined",
        )
    injector_names: set[str] = set()
    injector_ids: set[str] = set()
    for index, injector in enumerate(injectors, start=1):
        name = str(injector.get("name") or f"Injection Point {index}").strip()
        injector_id = str(injector.get("injector_id") or name).strip()
        protocol_location_id = str(injector.get("protocol_location_id") or "").strip()
        normalized_id = injector_id.casefold()
        if normalized_id in injector_ids:
            _issue(
                issues,
                "warning",
                "duplicate_injector_id",
                f"Injection-point identity {injector_id!r} is repeated",
            )
        injector_ids.add(normalized_id)
        if (
            protocol_location_id
            and injector_id.casefold() != protocol_location_id.casefold()
        ):
            _issue(
                issues,
                "warning",
                "injector_identity_differs_from_protocol",
                f"{name} uses location ID {injector_id!r}, but Excel uses "
                f"{protocol_location_id!r}",
            )
        if name in injector_names:
            _issue(
                issues,
                "warning",
                "duplicate_injector_name",
                f"Injection-point name {name!r} is repeated",
            )
        injector_names.add(name)
        x_m = _number(injector.get("x_m"))
        y_m = _number(injector.get("y_m"))
        position_tolerance_m = max(
            INJECTION_POINT_CONFLICT_TOLERANCE_M,
            _number(injector.get("uncertainty_m")) or 0.0,
        )
        protocol_x_m = _number(injector.get("protocol_x_m"))
        protocol_y_m = _number(injector.get("protocol_y_m"))
        protocol_positions = [
            (
                _number(row.get("x_m")),
                _number(row.get("y_m")),
            )
            for row in injector.get("protocol_coordinate_candidates") or []
            if isinstance(row, Mapping)
        ]
        distinct_protocol_positions: list[tuple[float, float]] = []
        for protocol_x, protocol_y in protocol_positions:
            if protocol_x is None or protocol_y is None:
                continue
            if not any(
                math.hypot(protocol_x - known_x, protocol_y - known_y)
                <= position_tolerance_m
                for known_x, known_y in distinct_protocol_positions
            ):
                distinct_protocol_positions.append((protocol_x, protocol_y))
        if len(distinct_protocol_positions) > 1:
            _issue(
                issues,
                "warning",
                "protocol_injector_position_conflict",
                f"{name} has conflicting x/y positions in the protocol source",
                injector_id=injector_id,
                positions=distinct_protocol_positions,
                tolerance_m=position_tolerance_m,
            )
        if x_m is None or y_m is None:
            _issue(
                issues,
                "error",
                "invalid_injector_position",
                f"{name} has no finite position; place it in the GUI or enter x/y",
            )
        elif (
            width_m is not None
            and height_m is not None
            and not (0.0 <= x_m <= width_m and 0.0 <= y_m <= height_m)
        ):
            _issue(
                issues, "error", "injector_outside_rig", f"{name} lies outside the rig"
            )
        if (
            x_m is not None
            and y_m is not None
            and protocol_x_m is not None
            and protocol_y_m is not None
            and math.hypot(x_m - protocol_x_m, y_m - protocol_y_m)
            > position_tolerance_m
        ):
            _issue(
                issues,
                "warning",
                "injector_position_differs_from_protocol",
                f"{name} differs from its Excel protocol position",
                injector_id=injector_id,
                gui_position_m=[x_m, y_m],
                protocol_position_m=[protocol_x_m, protocol_y_m],
                difference_m=math.hypot(x_m - protocol_x_m, y_m - protocol_y_m),
                tolerance_m=position_tolerance_m,
            )

    return {"issues": issues, "depth": depth_stats, "injector_count": len(injectors)}


@dataclass(frozen=True)
class _GeometryDiagnosticInputs:
    """Owned diagnostic DATA from original producers; never source authority."""
    fields: Mapping[str, np.ndarray] = field(default_factory=dict)
    baseline_shape: tuple[int, ...] | None = None
    run_fields: Mapping[str, Mapping[str, np.ndarray]] = field(default_factory=dict)
    analysis_exclusion: np.ndarray | None = None
    review_findings: Any = None

    def __post_init__(self):
        def array(value, name):
            if not isinstance(value, np.ndarray) or value.ndim != 2:
                raise ValueError(f"Diagnostic {name} needs its actual two-dimensional array")
            numeric = np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)
            if not numeric or (name != "depth_path" and not np.issubdtype(value.dtype, np.integer)):
                raise ValueError(f"Diagnostic {name} has an unsupported array dtype")
            owned = np.array(value, copy=True)
            owned.setflags(write=False)
            return owned

        if not isinstance(self.fields, Mapping) or set(self.fields) - {"depth_path", "labels_path", "facies_source_path"}:
            raise ValueError("Diagnostic fields must name original depth, labels or facies inputs")
        object.__setattr__(self, "fields", MappingProxyType({key: array(value, key) for key, value in self.fields.items()}))
        if self.baseline_shape is not None:
            shape = tuple(self.baseline_shape)
            if len(shape) not in (2, 3) or any(type(value) is not int or value <= 0 for value in shape):
                raise ValueError("Diagnostic baseline shape must be the actual positive image shape")
            object.__setattr__(self, "baseline_shape", shape)
        if not isinstance(self.run_fields, Mapping):
            raise ValueError("Diagnostic run fields need explicit run keys")
        runs = {}
        for run, values in self.run_fields.items():
            if (not isinstance(run, str) or not run or run != run.strip().casefold()
                    or any(not (value.isalnum() or value in "-_") for value in run)
                    or not isinstance(values, Mapping) or set(values) - {"labels", "facies"}):
                raise ValueError("Diagnostic run fields need exact normalized runs and labels/facies keys")
            runs[run] = MappingProxyType({key: array(value, key) for key, value in values.items()})
        object.__setattr__(self, "run_fields", MappingProxyType(runs))
        if self.analysis_exclusion is not None:
            value = self.analysis_exclusion
            if not isinstance(value, np.ndarray) or value.ndim != 2 or value.dtype != np.bool_:
                raise ValueError("Diagnostic analysis exclusion needs its actual two-dimensional boolean mask")
            owned = np.array(value, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, "analysis_exclusion", owned)
        if self.review_findings is not None:
            from .painted_analysis_exclusions import DisplayFindingsChanged
            if isinstance(self.review_findings, DisplayFindingsChanged):
                value = DisplayFindingsChanged(self.review_findings.original_issues, str(self.review_findings))
            elif isinstance(self.review_findings, Mapping) and isinstance(self.review_findings.get("issues"), list):
                value = deepcopy(dict(self.review_findings))
            else:
                raise ValueError("Diagnostic review findings need the original result or failure")
            object.__setattr__(self, "review_findings", value)


def _geometry_input_report(candidate, *, display_attachments=None, inputs: _GeometryDiagnosticInputs | None = None):
    """Collect original input diagnostics before source-mode admission."""
    from .adoption_check_progress import report_check_progress
    report_check_progress("Validating the geometry input arrays and physical properties…")
    if inputs is not None:
        if not isinstance(inputs, _GeometryDiagnosticInputs):
            raise TypeError("Diagnostic inputs must be original producer DATA")
        run_ids = {str(row.get("run_id") or "").strip().casefold()
                   for row in (candidate.get("run_geometry") or {}).get("runs", []) if isinstance(row, Mapping)}
        if set(inputs.run_fields) - run_ids:
            raise ValueError("Diagnostic arrays include a run outside the actual geometry inputs")
    issues: list[dict[str, Any]] = []
    if candidate.get("format") != GEOMETRY_CONFIG_FORMAT:
        _issue(
            issues, "error", "unsupported_format", "Unsupported geometry configuration"
        )
    if candidate.get("frame", "raw") not in {"raw", "rig-corrected"}:
        _issue(issues, "error", "invalid_workspace_frame", "Unknown geometry workspace coordinate frame")
    width_m = _number(candidate.get("width_m"))
    height_m = _number(candidate.get("height_m"))
    if width_m is None or width_m <= 0.0:
        _issue(issues, "error", "invalid_width", "Rig width must be positive")
    if height_m is None or height_m <= 0.0:
        _issue(issues, "error", "invalid_height", "Rig height must be positive")

    arrays: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for key, required in (
        ("baseline_path", True),
        ("depth_path", True),
        ("labels_path", True),
        ("facies_source_path", False),
    ):
        if inputs is not None:
            if key == "baseline_path" and inputs.baseline_shape is not None:
                shapes[key] = inputs.baseline_shape
                continue
            if key in inputs.fields:
                array = np.asarray(inputs.fields[key])
                arrays[key] = array
                shapes[key] = tuple(int(value) for value in array.shape)
                continue
        raw = str(candidate.get(key) or "")
        if not raw:
            if required:
                _issue(issues, "error", f"missing_{key}", f"{key} is required")
            continue
        path = Path(raw).expanduser()
        if not path.is_file():
            _issue(
                issues, "error", f"unavailable_{key}", f"File is unavailable: {path}"
            )
            continue
        try:
            if key == "baseline_path":
                shape, _dtype = _shape_and_dtype(path)
                shapes[key] = shape
            else:
                array = np.asarray(_load_array(path))
                arrays[key] = array
                shapes[key] = tuple(int(value) for value in array.shape)
        except Exception as exc:  # noqa: BLE001
            _issue(
                issues,
                "error",
                f"unreadable_{key}",
                f"Could not read {path.name}: {type(exc).__name__}: {exc}",
            )

    spatial_shapes = {
        key: shape[:2] for key, shape in shapes.items() if len(shape) >= 2
    }
    if len(set(spatial_shapes.values())) > 1:
        _issue(
            issues,
            "error",
            "shape_mismatch",
            "Baseline, depth, labels, and facies must share the same spatial shape",
            shapes={key: list(value) for key, value in spatial_shapes.items()},
        )

    labels = arrays.get("labels_path")
    depth = arrays.get("depth_path")
    if labels is not None and labels.ndim > 2:
        labels = np.squeeze(labels)
    if depth is not None and depth.ndim > 2:
        depth = np.squeeze(depth)
    label_ids = (
        sorted(int(value) for value in np.unique(labels)) if labels is not None else []
    )
    regions = [dict(row) for row in candidate.get("regions") or []]
    region_by_label: dict[int, dict[str, Any]] = {}
    for row in regions:
        label_id = _int(row.get("label_id"))
        if label_id is None:
            _issue(
                issues,
                "error",
                "invalid_label_id",
                "Every layer needs an integer label ID",
            )
            continue
        if label_id in region_by_label:
            _issue(
                issues,
                "error",
                "duplicate_label_id",
                f"Label {label_id} is defined more than once",
            )
        region_by_label[label_id] = row
        role = str(row.get("analysis_role") or "")
        if role not in ANALYSIS_ROLES:
            _issue(
                issues,
                "error",
                "invalid_analysis_role",
                f"Label {label_id} has invalid role {role!r}",
            )
        if _int(row.get("facies_id")) is None:
            _issue(
                issues,
                "error",
                "invalid_facies_id",
                f"Label {label_id} needs an integer facies ID",
            )
        if bool(row.get("enabled", True)) and role in {"porous_medium", "seal"}:
            porosity = _number(row.get("porosity"))
            permeability = _number(row.get("permeability"))
            if porosity is None or not 0.0 < porosity <= 1.0:
                _issue(
                    issues,
                    "error",
                    "invalid_porosity",
                    f"Label {label_id} needs porosity in (0, 1]",
                )
            if permeability is None or permeability <= 0.0:
                _issue(
                    issues,
                    "error",
                    "invalid_permeability",
                    f"Label {label_id} needs positive permeability",
                )
    from .painted_analysis_exclusions import configuration_exclusion
    analysis_exclusion = None
    if "analysis_exclusions" in candidate or (inputs is not None and inputs.analysis_exclusion is not None):
        try:
            if inputs is not None:
                analysis_exclusion = inputs.analysis_exclusion
                if analysis_exclusion is not None and labels is not None and (
                        analysis_exclusion.shape != labels.shape or np.any(analysis_exclusion & (labels != 0))):
                    analysis_exclusion = None
                    raise ValueError("Approved analysis gaps cannot mask mapped labels or another native grid")
            else:
                analysis_exclusion = configuration_exclusion(candidate, labels=labels)
        except (ValueError, OSError, KeyError, TypeError) as error:
            _issue(issues, "error", "painted_analysis_exclusions_changed", str(error))
    required_label_ids = label_ids
    if analysis_exclusion is not None and labels is not None:
        required_label_ids = [int(value) for value in np.unique(labels[~analysis_exclusion])]
    missing_labels = sorted(set(required_label_ids) - set(region_by_label))
    unknown_labels = sorted(set(region_by_label) - set(label_ids))
    if missing_labels:
        _issue(
            issues,
            "error",
            "unmapped_labels",
            "Some image labels have no layer definition",
            labels=missing_labels,
        )
    if unknown_labels:
        _issue(
            issues,
            "warning",
            "unused_layer_definitions",
            "Some layer definitions are absent from the label map",
            labels=unknown_labels,
        )

    grouping_mode = str(candidate.get("grouping_mode") or "shared_facies")
    if grouping_mode not in {"shared_facies", "per_layer"}:
        _issue(
            issues,
            "error",
            "invalid_grouping_mode",
            "Grouping mode must be shared_facies or per_layer",
        )
    if grouping_mode == "shared_facies":
        by_facies: dict[int, list[dict[str, Any]]] = {}
        for row in region_by_label.values():
            facies_id = _int(row.get("facies_id"))
            if facies_id is not None:
                by_facies.setdefault(facies_id, []).append(row)
        for facies_id, rows in by_facies.items():
            signatures = {
                (_number(row.get("porosity")), _number(row.get("permeability")))
                for row in rows
            }
            if len(signatures) > 1:
                _issue(
                    issues,
                    "error",
                    "inconsistent_shared_facies",
                    f"Facies {facies_id} has conflicting properties across layers",
                )

    run_geometry = normalize_run_geometry_config(candidate.get("run_geometry"))
    run_rows = [
        dict(row) for row in run_geometry.get("runs") or [] if isinstance(row, Mapping)
    ]
    if run_geometry.get("format") != RUN_GEOMETRY_FORMAT:
        _issue(
            issues,
            "error",
            "run_geometry_format",
            "Unsupported per-run geometry format",
        )
    if run_geometry.get("mode") != "series_reference_with_per_run_layer_overrides":
        _issue(
            issues,
            "error",
            "run_geometry_mode",
            "Per-run geometry must inherit from one approved series reference",
        )
    geometry_policy = str(run_geometry.get("geometry_policy") or "")
    if geometry_policy not in RUN_GEOMETRY_POLICIES:
        _issue(
            issues,
            "error",
            "run_geometry_policy",
            "Choose whether all runs use the series geometry or reviewed per-run maps",
        )
    if (
        run_geometry.get("property_policy") != "inherit_approved_series_properties"
        or run_geometry.get("automatic_property_adjustment") is not False
    ):
        _issue(
            issues,
            "error",
            "automatic_property_adjustment_forbidden",
            "Layer compression cannot silently change porosity or permeability",
        )
    change_detection = dict(run_geometry.get("change_detection") or {})
    detection_enabled = change_detection.get("enabled")
    if not isinstance(detection_enabled, bool):
        _issue(
            issues,
            "error",
            "run_geometry_change_detection_enabled",
            "Layer-change detection must be explicitly enabled or disabled",
        )
        detection_enabled = False
    displacement_fraction = _number(
        change_detection.get("warning_displacement_fraction_of_height")
    )
    if displacement_fraction is None or not 0.0 < displacement_fraction < 1.0:
        _issue(
            issues,
            "error",
            "run_geometry_displacement_threshold",
            "Layer-change warning displacement must be a fraction in (0, 1)",
        )
    minimum_affected_fraction = _number(
        change_detection.get("minimum_affected_boundary_fraction")
    )
    if minimum_affected_fraction is None or not 0.0 <= minimum_affected_fraction <= 1.0:
        _issue(
            issues,
            "error",
            "run_geometry_affected_threshold",
            "Affected-boundary threshold must be a fraction in [0, 1]",
        )
    run_ids: set[str] = set()
    indicator_run_count = 0
    active_run_count = 0
    selected_run_override_count = 0
    screened_run_count = 0
    substantial_change_count = 0
    run_change_diagnostics: list[dict[str, Any]] = []
    manual_boundary_authorities = {}
    if candidate.get('painted_adoption_table'):
        try:
            from .manual_map_approval import approved_boundary_authorities
            manual_boundary_authorities = approved_boundary_authorities(candidate)
        except (ValueError, OSError, KeyError, TypeError) as error:
            _issue(issues, 'error', 'manual_map_approval_changed', str(error))
    baseline_shape = spatial_shapes.get("baseline_path")
    layer_label_ids = {
        int(label_id)
        for label_id, row in region_by_label.items()
        if bool(row.get("enabled", True))
        and str(row.get("analysis_role") or "") in {"porous_medium", "seal"}
    }
    for number, run_row in enumerate(run_rows, 1):
        run_id = str(run_row.get("run_id") or "").strip().casefold()
        report_check_progress(f"Validating geometry inputs for {run_id.upper()} ({number} of {len(run_rows)} runs)…")
        safe_run_id = bool(run_id) and all(
            character.isalnum() or character in "-_" for character in run_id
        )
        if not safe_run_id or run_id in run_ids:
            _issue(
                issues,
                "error",
                "run_geometry_identifier",
                "Per-run geometry identifiers must be non-empty and unique",
            )
            continue
        run_ids.add(run_id)
        if not bool(run_row.get("enabled", True)):
            continue
        active_run_count += 1
        raw_choice = str(run_row.get("geometry_choice") or "").strip()
        choice = _effective_run_geometry_choice(run_row)
        if raw_choice and raw_choice not in RUN_GEOMETRY_CHOICES:
            _issue(
                issues,
                "error",
                "run_geometry_choice",
                f"{run_id} has an unsupported geometry choice",
                run_id=run_id,
            )
        state = str(run_row.get("acquisition_state") or "unknown")
        if state not in RUN_GEOMETRY_ACQUISITION_STATES:
            _issue(
                issues,
                "error",
                "run_geometry_acquisition_state",
                f"{run_id} has an unsupported acquisition state",
            )
        elif state == "indicator":
            indicator_run_count += 1
            _issue(
                issues,
                "warning",
                "indicator_facies_carryover_risk",
                f"{run_id} uses an indicator-filled image; residual indicator "
                "colour can be mistaken for a layer boundary",
                run_id=run_id,
            )
        elif state == "unknown":
            _issue(
                issues,
                "warning",
                "run_geometry_acquisition_unknown",
                f"Record whether the facies source for {run_id} is clearwater or "
                "indicator-filled",
                run_id=run_id,
            )
        source_text = str(run_row.get("source_image_path") or "").strip()
        if not source_text:
            _issue(
                issues,
                "warning",
                "run_geometry_source_missing",
                f"{run_id} has no source image for visual review",
                run_id=run_id,
            )
        elif not Path(source_text).expanduser().is_file():
            _issue(
                issues,
                "error",
                "run_geometry_source_unavailable",
                f"Per-run source image is unavailable for {run_id}",
                path=source_text,
            )
        labels_text = str(run_row.get("labels_path") or "").strip()
        facies_text = str(run_row.get("facies_path") or "").strip()
        if geometry_policy == "series_reference_only" and choice == "run_override":
            _issue(
                issues,
                "info",
                "run_geometry_override_not_selected",
                f"{run_id} has a run map attached for review, but the series geometry "
                "is selected and will remain authoritative",
                run_id=run_id,
            )
        elif choice == "run_override":
            selected_run_override_count += 1
            if not labels_text and not facies_text and not (inputs is not None and inputs.run_fields.get(run_id)):
                _issue(
                    issues,
                    "error",
                    "run_geometry_override_missing",
                    f"{run_id} selects a run override but provides no layer or facies map",
                    run_id=run_id,
                )
        if (
            geometry_policy == "require_reviewed_run_overrides"
            and choice != "run_override"
        ):
            _issue(
                issues,
                "error",
                "run_geometry_override_required",
                f"{run_id} must use a reviewed per-run geometry under the selected policy",
                run_id=run_id,
            )
        loaded_run_fields: dict[str, np.ndarray] = {}
        for field, raw_path in (
            ("labels", labels_text),
            ("facies", facies_text),
        ):
            supplied = None if inputs is None else inputs.run_fields.get(run_id, {}).get(field)
            if supplied is not None:
                run_array = np.squeeze(supplied)
                shape = tuple(int(value) for value in run_array.shape)
                loaded_run_fields[field] = np.asarray(run_array)
            else:
                if not raw_path:
                    continue
                path = Path(raw_path).expanduser()
                if not path.is_file():
                    _issue(
                        issues,
                        "error",
                        f"run_geometry_{field}_unavailable",
                        f"Per-run {field} map is unavailable for {run_id}",
                        path=str(path),
                    )
                    continue
                try:
                    run_array = np.squeeze(_load_array(path))
                    shape = tuple(int(value) for value in run_array.shape)
                    loaded_run_fields[field] = np.asarray(run_array)
                except Exception as exc:  # noqa: BLE001
                    _issue(
                        issues,
                        "error",
                        f"run_geometry_{field}_unreadable",
                        f"Could not read {run_id} {field}: {type(exc).__name__}: {exc}",
                    )
                    continue
            if baseline_shape is not None and shape[:2] != baseline_shape:
                _issue(
                    issues,
                    "error",
                    f"run_geometry_{field}_shape",
                    f"{run_id} {field} does not share the registered series grid",
                    expected=list(baseline_shape),
                    actual=list(shape[:2]),
                )
        run_labels = loaded_run_fields.get("labels")
        if detection_enabled and run_id in manual_boundary_authorities:
            # The user already inspected this exact map on this run. Avoid
            # recomputing the expensive advisory distance transforms; retain
            # the immutable decision and review rather than inventing scores.
            run_change_diagnostics.append(dict(
                run_id=run_id, status='manually_approved',
                boundary_comparison='not_repeated_after_exact_manual_approval',
                geometry_choice=choice, geometry_policy=geometry_policy,
                authority=manual_boundary_authorities[run_id]))
        if (
            detection_enabled
            and run_id not in manual_boundary_authorities
            and run_labels is not None
            and labels is not None
            and width_m is not None
            and height_m is not None
            and displacement_fraction is not None
            and 0.0 < displacement_fraction < 1.0
            and minimum_affected_fraction is not None
            and 0.0 <= minimum_affected_fraction <= 1.0
        ):
            diagnostic = screen_layer_boundary_change(
                labels,
                run_labels,
                width_m=width_m,
                height_m=height_m,
                layer_label_ids=layer_label_ids,
                warning_displacement_fraction_of_height=displacement_fraction,
                minimum_affected_boundary_fraction=minimum_affected_fraction,
            )
            diagnostic.update(
                {
                    "run_id": run_id,
                    "geometry_choice": choice,
                    "geometry_policy": geometry_policy,
                }
            )
            run_change_diagnostics.append(diagnostic)
            if diagnostic.get("status") != "not_comparable":
                screened_run_count += 1
            if bool(diagnostic.get("substantial_change")):
                substantial_change_count += 1
                _issue(
                    issues,
                    "warning",
                    "substantial_layer_boundary_change",
                    f"{run_id} has a substantial registered layer-boundary change "
                    f"(95th percentile {1000.0 * float(diagnostic['p95_displacement_m']):.2f} mm; "
                    f"{100.0 * float(diagnostic['affected_boundary_fraction']):.1f}% affected). "
                    "Review the overlay and explicitly choose the authoritative geometry.",
                    **diagnostic,
                )
        proposals = list(run_row.get("property_change_proposals") or [])
        if proposals:
            _issue(
                issues,
                "warning",
                "run_property_changes_not_applied",
                f"{run_id} has {len(proposals)} property proposal(s); they remain "
                "inactive until a separate reviewed revision approves them",
                run_id=run_id,
            )

    depth_injector_findings = depth_and_injector_findings(
        depth, candidate.get("injectors") or [], width_m=width_m, height_m=height_m
    )
    issues.extend(depth_injector_findings["issues"])
    depth_stats = depth_injector_findings["depth"]
    injector_count = depth_injector_findings["injector_count"]

    pore_volume_m3: float | None = None
    active_area_fraction: float | None = None
    if (
        labels is not None
        and depth is not None
        and labels.shape == depth.shape
        and width_m is not None
        and height_m is not None
        and not missing_labels
    ):
        porosity_map = np.zeros(labels.shape, dtype=np.float64)
        active = np.zeros(labels.shape, dtype=bool)
        for label_id, row in region_by_label.items():
            role = str(row.get("analysis_role") or "")
            enabled = bool(row.get("enabled", True))
            if not enabled or role not in {"porous_medium", "seal"}:
                continue
            porosity = _number(row.get("porosity"))
            if porosity is None:
                continue
            mask = labels == label_id
            porosity_map[mask] = porosity
            active[mask] = True
        pixel_area_m2 = width_m * height_m / float(labels.shape[0] * labels.shape[1])
        pore_volume_m3 = float(
            np.sum(np.asarray(depth, dtype=np.float64) * porosity_map, dtype=np.float64)
            * pixel_area_m2
        )
        active_area_fraction = float(np.mean(active))
        if not np.any(active):
            _issue(
                issues,
                "warning",
                "no_active_porous_regions",
                "No reviewed porous-medium or seal regions are enabled",
            )

    display_findings = None
    if (candidate.get("painted_adoption_table") or display_attachments is not None
            or (inputs is not None and inputs.review_findings is not None)):
        from .painted_analysis_exclusions import check_review_findings, DisplayFindingsChanged
        report_check_progress("Validating painted-map review findings and display bindings…")
        try:
            if inputs is not None and inputs.review_findings is not None:
                if isinstance(inputs.review_findings, DisplayFindingsChanged):
                    raise inputs.review_findings
                display_findings = deepcopy(inputs.review_findings)
            else:
                display_findings = check_review_findings(candidate, display_attachments=display_attachments)
            issues.extend(display_findings["issues"])
        except DisplayFindingsChanged as error:
            issues.extend(error.original_issues)
            _issue(issues, "error", "painted_findings_changed", str(error))
        except (ValueError, OSError, KeyError, TypeError) as error:
            _issue(issues, "error", "painted_findings_changed", str(error))
    severity_counts = {
        severity: sum(issue["severity"] == severity for issue in issues)
        for severity in ("error", "warning", "info")
    }
    report = {
        "schema": 1,
        "format": "darsia-fluidflower-geometry-validation-v1",
        "validated_utc": _utc_now(),
        "config_sha256": geometry_config_digest(candidate),
        "summary": {
            "error_count": severity_counts["error"],
            "warning_count": severity_counts["warning"],
            "label_count": len(label_ids),
            "facies_count": len(
                {
                    _int(row.get("facies_id"))
                    for row in regions
                    if _int(row.get("facies_id")) is not None
                }
            ),
            "run_geometry_count": active_run_count,
            "selected_run_override_count": selected_run_override_count,
            "screened_run_geometry_count": screened_run_count,
            "substantial_layer_change_count": substantial_change_count,
            "run_geometry_change_diagnostics": run_change_diagnostics,
            "indicator_run_geometry_count": indicator_run_count,
            "injector_count": injector_count,
            "spatial_shape": list(next(iter(spatial_shapes.values()), ())),
            "depth": depth_stats,
            "active_area_fraction": active_area_fraction,
            "estimated_pore_volume_m3": pore_volume_m3,
        },
        "issues": issues,
    }
    if display_findings is not None and "display_findings_sha256" in display_findings:
        report["display_findings_sha256"] = display_findings["display_findings_sha256"]
        report["display_attachments"] = deepcopy(display_findings["display_attachments"])
    if analysis_exclusion is not None:
        report["summary"].update(accounting_scope="quantified_subtotal_with_unquantified_remainder",
            approved_analysis_gap_pixels=int(np.count_nonzero(analysis_exclusion)),
            quantified_estimated_pore_volume_m3=pore_volume_m3,
            estimated_pore_volume_m3=None, unquantified_pore_volume_m3=None)
    return report


def validate_geometry_config(config: Mapping[str, Any], *, display_attachments=None) -> dict[str, Any]:
    """Validate geometry inputs and calculate physical summary diagnostics."""

    candidate = deepcopy(dict(config))
    report = _geometry_input_report(candidate, display_attachments=display_attachments)
    if candidate.get("source_mode") in {"imported-rasters", "canonical-from-import"}:
        from .geometry_raster_sources import validate_source_mode
        from .adoption_check_progress import report_check_progress
        report_check_progress("Validating the geometry source and registration bindings…")
        return validate_source_mode(candidate, report)
    return report


def _canonical_facies(
    config: Mapping[str, Any],
    *, run_id: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = np.asarray(_load_array(Path(str(config["labels_path"]))))
    labels = np.squeeze(labels)
    regions = [dict(row) for row in config.get("regions") or []]
    mode = str(config.get("grouping_mode") or "shared_facies")
    output = np.zeros(labels.shape, dtype=np.int32)
    exclusion = None
    if "analysis_exclusions" in config:
        from .painted_analysis_exclusions import configuration_exclusion
        exclusion = configuration_exclusion(config, run_id, labels=labels)
    properties: dict[int, dict[str, Any]] = {}
    for row in regions:
        label_id = int(row["label_id"])
        analysis_id = label_id if mode == "per_layer" else int(row["facies_id"])
        selected = labels == label_id
        if exclusion is not None:
            selected &= ~exclusion
        output[selected] = analysis_id
        properties.setdefault(
            analysis_id,
            {
                "id": analysis_id,
                "name": str(row.get("name") or f"Region {analysis_id}"),
                "porosity": _number(row.get("porosity")),
                "permeability": _number(row.get("permeability")),
                "analysis_role": str(row.get("analysis_role") or "excluded"),
                "enabled": bool(row.get("enabled", True)),
                "permeability_unit": str(
                    config.get("permeability_unit") or "source_unit"
                ),
            },
        )
    return output, [properties[key] for key in sorted(properties)]


class GeometryRevisionStore:
    """Append-only geometry revisions with compact canonical DarSIA artifacts."""

    def __init__(self, project: "FluidFlowerProject") -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "geometry_facies"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        self.artifact_root = project.root / "results" / "geometry"

    def revisions(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.revisions_path)

    def current(self) -> dict[str, Any]:
        if self.current_path.is_file():
            payload = json.loads(self.current_path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                return payload
        revisions = self.revisions()
        return (
            deepcopy(revisions[-1])
            if revisions
            else default_geometry_config(self.project)
        )

    def current_config(self) -> dict[str, Any]:
        current = self.current()
        if current.get("format") == GEOMETRY_REVISION_FORMAT:
            config = deepcopy(dict(current.get("configuration") or {}))
            depth_artifact = str(
                (current.get("artifacts") or {}).get("depth_path") or ""
            )
            if depth_artifact:
                config["depth_path"] = depth_artifact
                config["depth_artifact_path"] = depth_artifact
            return config
        return deepcopy(current)

    def save_revision(
        self,
        config: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
        validation: Mapping[str, Any] | None = None,
        checked_candidate: Any = None,
        preserve_failed_artifacts: bool = False,
        review_resolutions: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate = deepcopy(dict(config))
        from .geometry_raster_sources import require_source_mode_transition
        require_source_mode_transition(self, candidate)
        display_attachments = (deepcopy(review_resolutions.get("display_attachments"))
            if isinstance(review_resolutions, Mapping) else None)
        if checked_candidate is not None:
            from .local_geometry_check import CheckedGeometryCandidate
            if not isinstance(checked_candidate, CheckedGeometryCandidate):
                raise TypeError("Save needs an actual local Check object, not a report flag")
            checked_candidate.report_for_save(candidate)
        def current_validation(value):
            # The local single-machine path has already run this exact Check.
            # Changed input versions refuse reuse; legacy callers are unchanged.
            if checked_candidate is not None:
                return checked_candidate.report_for(value)
            return (validate_geometry_config(value) if display_attachments is None
                    else validate_geometry_config(value, display_attachments=display_attachments))
        report = (current_validation(candidate) if checked_candidate is not None
                  else deepcopy(dict(validation or {})))
        if report.get("config_sha256") != geometry_config_digest(candidate):
            report = current_validation(candidate)
        input_resolutions = None
        from .painted_analysis_exclusions import findings_resolutions_required
        if review_resolutions is not None:
            from .painted_analysis_exclusions import verify_findings_resolutions
            # A resolution cannot remove a genuine current warning by supplying
            # an edited report with the same configuration digest.
            report = current_validation(candidate)
            input_resolutions = verify_findings_resolutions(candidate, report, review_resolutions)
        elif findings_resolutions_required(candidate):
            raise ValueError("The reviewed preparation requires the complete explicit findings resolutions from Use the map in the series")
        if int((report.get("summary") or {}).get("error_count") or 0) > 0:
            raise ValueError("Geometry configuration has blocking validation errors")

        depth_input_path = Path(str(candidate["depth_path"])).expanduser().resolve()
        depth_input_reference = _file_reference(
            depth_input_path, include_full_sha256=True
        )
        inherited_depth_source = candidate.get("depth_source_reference")
        if (
            _same_path(depth_input_path, candidate.get("depth_artifact_path") or "")
            and isinstance(inherited_depth_source, Mapping)
            and str(inherited_depth_source.get("path") or "")
            and str(inherited_depth_source.get("sha256") or "")
        ):
            depth_source_reference = deepcopy(dict(inherited_depth_source))
        else:
            depth_source_reference = deepcopy(depth_input_reference)

        revision_id = uuid.uuid4().hex
        destination = self.artifact_root / revision_id
        staging = self.artifact_root / f".{revision_id}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        run_artifacts: dict[str, dict[str, Any]] = {}
        depth_artifact_name = f"depth{depth_input_path.suffix.casefold()}"
        try:
            shutil.copy2(depth_input_path, staging / depth_artifact_name)
            source_mode_artifacts = {}
            if candidate.get("source_mode") in {"imported-rasters", "canonical-from-import"}:
                from .geometry_raster_sources import materialize_source_mode
                source_mode_artifacts, properties = materialize_source_mode(candidate, staging, destination)
            else:
                facies, properties = _canonical_facies(candidate)
                metadata = _darsia_metadata(Path(str(candidate["labels_path"])))
                np.savez_compressed(staging / "facies.npz", array=facies, metadata=metadata)
                with (staging / "facies_properties.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    fieldnames = [
                        "id",
                        "name",
                        "porosity",
                        "permeability",
                        "permeability_unit",
                        "analysis_role",
                        "enabled",
                    ]
                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(properties)
            run_geometry = normalize_run_geometry_config(candidate.get("run_geometry"))
            geometry_policy = str(run_geometry.get("geometry_policy") or "")
            diagnostics_by_run = {
                str(row.get("run_id") or ""): dict(row)
                for row in (report.get("summary") or {}).get(
                    "run_geometry_change_diagnostics", []
                )
            }
            for raw in run_geometry.get("runs") or []:
                row = dict(raw)
                if not bool(row.get("enabled", True)):
                    continue
                run_id = str(row.get("run_id") or "").strip().casefold()
                choice = _effective_run_geometry_choice(row)
                if (
                    geometry_policy == "series_reference_only"
                    or choice != "run_override"
                ):
                    continue
                labels_text = str(row.get("labels_path") or "").strip()
                facies_text = str(row.get("facies_path") or "").strip()
                if not run_id or (not labels_text and not facies_text):
                    continue
                run_root = staging / "runs" / run_id
                run_root.mkdir(parents=True, exist_ok=False)
                run_config = deepcopy(candidate)
                if labels_text:
                    run_labels = np.squeeze(_load_array(Path(labels_text)))
                    run_config["labels_path"] = labels_text
                    run_facies, run_properties = _canonical_facies(run_config, run_id=run_id)
                    if (row.get("registration") or {}).get("kind") == "painted-map-placement-v2":
                        # The positive V2 admission has replayed every mask.
                        # Keep its full NPZ (including effective coverage) as an
                        # exact Store copy, rather than silently dropping flags.
                        shutil.copyfile(labels_text, run_root / "labels.npz")
                        if (run_root / "labels.npz").read_bytes() != Path(labels_text).read_bytes():
                            raise ValueError("The V2 Store copy lost its continuation provenance.")
                    else:
                        np.savez_compressed(
                            run_root / "labels.npz",
                            array=np.asarray(run_labels, dtype=np.int32),
                            metadata=_darsia_metadata(Path(labels_text)),
                        )
                else:
                    run_facies = np.squeeze(_load_array(Path(facies_text)))
                    run_properties = properties
                if facies_text:
                    run_facies = np.squeeze(_load_array(Path(facies_text)))
                np.savez_compressed(
                    run_root / "facies.npz",
                    array=np.asarray(run_facies, dtype=np.int32),
                    metadata=_darsia_metadata(Path(labels_text or facies_text)),
                )
                if candidate.get("source_mode") == "canonical-from-import":
                    # The parent FULL CSV is already verified and materialized by
                    # materialize_source_mode. Keep unused IDs and extra columns.
                    source_csv = staging / "facies_properties.csv"
                    target_csv = run_root / "facies_properties.csv"
                    shutil.copyfile(source_csv, target_csv)
                    if target_csv.read_bytes() != source_csv.read_bytes():
                        raise ValueError("Registered run lost its FULL original properties CSV.")
                else:
                    with (run_root / "facies_properties.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as stream:
                        writer = csv.DictWriter(
                            stream,
                            fieldnames=[
                                "id",
                                "name",
                                "porosity",
                                "permeability",
                                "permeability_unit",
                                "analysis_role",
                                "enabled",
                            ],
                        )
                        writer.writeheader()
                        writer.writerows(run_properties)
                run_artifacts[run_id] = {
                    "facies_path": str(destination / "runs" / run_id / "facies.npz"),
                    "facies_properties_path": str(
                        destination / "runs" / run_id / "facies_properties.csv"
                    ),
                    **(
                        {
                            "labels_path": str(
                                destination / "runs" / run_id / "labels.npz"
                            )
                        }
                        if labels_text
                        else {}
                    ),
                    "property_policy": "inherit_approved_series_properties",
                    "geometry_choice": choice,
                    "change_screen": diagnostics_by_run.get(run_id, {}),
                }
            os.replace(staging, destination)
        except Exception:
            if not preserve_failed_artifacts:
                shutil.rmtree(staging, ignore_errors=True)
            raise

        depth_artifact_path = destination / depth_artifact_name
        approved_candidate = deepcopy(candidate)
        approved_candidate.update(
            {
                "depth_path": str(depth_artifact_path),
                "depth_artifact_path": str(depth_artifact_path),
                "depth_source_reference": depth_source_reference,
            }
        )
        reused_depth_reference = None
        if checked_candidate is not None:
            approved_report, reused_depth_reference = checked_candidate.materialized_report(
                approved_candidate, depth_input_reference, depth_source_reference)
        else:
            approved_report = current_validation(approved_candidate)
        if int((approved_report.get("summary") or {}).get("error_count") or 0) > 0:
            if not preserve_failed_artifacts:
                shutil.rmtree(destination, ignore_errors=True)
            raise ValueError("Materialized geometry revision failed validation")
        candidate = approved_candidate
        report = approved_report
        materialized_resolutions = None
        if input_resolutions is not None:
            from .painted_analysis_exclusions import seal_findings_resolutions
            # The Store relocates only its original depth artifact. Retain both
            # exact candidate identities and require the actual final issues to
            # match every original human action before publishing a revision.
            materialized_resolutions = seal_findings_resolutions(candidate, report, input_resolutions["records"],
                display_attachments=input_resolutions.get("display_attachments"))

        source_paths = {
            key: str(candidate.get(key) or "")
            for key in (
                "baseline_path",
                "labels_path",
                "facies_source_path",
                "facies_properties_source_path",
                "protocol_source_path",
                "display_image_path",
                "draft_source_image_path",
            )
            if str(candidate.get(key) or "")
        }
        edit_reference = dict(candidate.get("label_edit_session") or {})
        for output_key, reference_key in (
            ("label_edit_base_labels_path", "base_labels_path"),
            ("label_edit_events_path", "events_path"),
            (
                "label_edit_materialization_manifest_path",
                "materialization_manifest_path",
            ),
        ):
            path = str(edit_reference.get(reference_key) or "")
            if path and Path(path).is_file():
                source_paths[output_key] = path
        for raw in (candidate.get("run_geometry") or {}).get("runs") or []:
            row = dict(raw)
            run_id = str(row.get("run_id") or "").strip().casefold()
            for field in ("source_image_path", "labels_path", "facies_path"):
                path = str(row.get(field) or "").strip()
                if run_id and path and Path(path).is_file():
                    source_paths[f"run_{run_id}_{field}"] = path
        source_files = {
            key: _file_reference(path) for key, path in source_paths.items()
        }
        source_files["depth_path"] = depth_source_reference
        source_files["depth_materialization_input_path"] = depth_input_reference
        depth_artifact_reference = (reused_depth_reference if reused_depth_reference is not None
            else _file_reference(depth_artifact_path, include_full_sha256=True))
        revision = {
            "schema": 1,
            "format": GEOMETRY_REVISION_FORMAT,
            "frame": candidate.get("frame", "raw"),
            "revision_id": revision_id,
            "recorded_utc": _utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "configuration": candidate,
            "source_files": source_files,
            "artifacts": {
                "depth_path": str(depth_artifact_path),
                "depth_file": depth_artifact_reference,
                "facies_path": str(destination / "facies.npz"),
                "facies_properties_path": str(destination / "facies_properties.csv"),
                "run_geometry": run_artifacts,
                **source_mode_artifacts,
            },
            "validation": report,
        }
        if materialized_resolutions is not None:
            revision["review_resolutions"] = dict(input=input_resolutions, materialized=materialized_resolutions)
        (destination / "revision.json").write_text(
            _bounded_revision_text(revision),
            encoding="utf-8",
        )
        _append_jsonl(self.revisions_path, revision)
        _atomic_json(self.current_path, revision)
        self.project.update_metadata(
            "rig",
            {
                "geometry_revision_id": revision_id,
                "geometry": {
                    "width_m": candidate.get("width_m"),
                    "height_m": candidate.get("height_m"),
                    "coordinate_convention": candidate.get("coordinate_convention"),
                },
                "depth": {
                    "path": candidate.get("depth_path"),
                    "source": depth_source_reference,
                    "artifact": depth_artifact_reference,
                    **dict((report.get("summary") or {}).get("depth") or {}),
                },
                "injectors": deepcopy(candidate.get("injectors") or []),
                "facies": deepcopy(candidate.get("regions") or []),
                "run_geometry": deepcopy(
                    (candidate.get("run_geometry") or {}).get("runs") or []
                ),
                "run_geometry_policy": str(
                    (candidate.get("run_geometry") or {}).get(
                        "geometry_policy", "allow_reviewed_run_overrides"
                    )
                ),
            },
            updated_by=actor,
            reason=f"Geometry revision {revision_id}: {reason}".strip(),
        )
        self.project._audit(
            "geometry_facies_revision_saved",
            actor,
            {
                "revision_id": revision_id,
                "validation_summary": report.get("summary"),
                "artifact_path": str(destination),
            },
        )
        return deepcopy(revision)


def resolve_run_geometry_artifacts(
    revision: Mapping[str, Any], run_id: str
) -> dict[str, Any]:
    """Resolve a run override, falling back explicitly to the series artifacts."""

    artifacts = dict(revision.get("artifacts") or {})
    key = str(run_id).strip().casefold()
    run_artifacts = dict((artifacts.get("run_geometry") or {}).get(key) or {})
    if run_artifacts:
        return {
            "scope": "run_override",
            "run_id": key,
            **run_artifacts,
        }
    return {
        "scope": "series_reference",
        "run_id": key,
        "labels_path": str(
            (revision.get("configuration") or {}).get("labels_path") or ""
        ),
        "facies_path": str(artifacts.get("facies_path") or ""),
        "facies_properties_path": str(artifacts.get("facies_properties_path") or ""),
        "property_policy": "inherit_approved_series_properties",
    }


def load_geometry_preview(
    config: Mapping[str, Any],
    *,
    max_width: int = 1500,
    max_height: int = 900,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Load and downsample the rig fields needed by the Qt preview."""

    import cv2

    effective = deepcopy(dict(config))
    run_row: dict[str, Any] = {}
    if run_id:
        key = str(run_id).strip().casefold()
        matches = [
            dict(row)
            for row in (effective.get("run_geometry") or {}).get("runs") or []
            if str(row.get("run_id") or "").strip().casefold() == key
            and bool(row.get("enabled", True))
        ]
        if not matches:
            raise ValueError(f"No enabled per-run geometry exists for {key}")
        run_row = matches[0]
        if str(run_row.get("source_image_path") or ""):
            effective["baseline_path"] = str(run_row["source_image_path"])
        if str(run_row.get("labels_path") or ""):
            effective["labels_path"] = str(run_row["labels_path"])
    baseline = _rgb_float(_load_array(Path(str(effective["baseline_path"]))))
    labels = np.squeeze(_load_array(Path(str(effective["labels_path"]))))
    depth = np.squeeze(_load_array(Path(str(config["depth_path"]))))
    facies_path = str(run_row.get("facies_path") or "")
    if effective.get("source_mode") == "imported-rasters":
        from .geometry_raster_sources import imported_payload
        if run_row and (str(run_row.get("labels_path") or "") or facies_path):
            raise ValueError("Imported rasters cannot silently use a different run override.")
        facies = imported_payload(effective)["facies"]
    elif facies_path:
        facies = np.squeeze(_load_array(Path(facies_path)))
    else:
        facies, _properties = _canonical_facies(effective, run_id=run_id)
    height, width = baseline.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    baseline_small = cv2.resize(baseline, target, interpolation=interpolation)
    labels_small = cv2.resize(
        labels.astype(np.int32), target, interpolation=cv2.INTER_NEAREST
    )
    facies_small = cv2.resize(
        facies.astype(np.int32), target, interpolation=cv2.INTER_NEAREST
    )
    depth_small = cv2.resize(
        depth.astype(np.float32), target, interpolation=cv2.INTER_AREA
    )
    preview = {
        "baseline": np.asarray(baseline_small, dtype=np.float32),
        "labels": np.asarray(labels_small, dtype=np.int32),
        "facies": np.asarray(facies_small, dtype=np.int32),
        "depth": np.asarray(depth_small, dtype=np.float32),
        "source_shape": [height, width],
        "preview_shape": [target[1], target[0]],
        "run_id": str(run_id or ""),
    }
    display_path = str(config.get("display_image_path") or "") if not run_id else ""
    if display_path:
        review, provenance = load_registered_geometry_image(config, Path(display_path))
        review_small = cv2.resize(review, target, interpolation=interpolation)
        preview["review_frame"] = np.asarray(review_small, dtype=np.float32)
        preview["review_frame_provenance"] = provenance
    return preview
