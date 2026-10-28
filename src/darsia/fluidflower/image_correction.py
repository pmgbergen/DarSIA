"""Versioned radiometric correction for FluidFlower image workflows."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from darsia.fluidflower.image_formats import (
    ALL_IMAGE_SUFFIXES,
    CAMERA_RAW_SUFFIXES,
    select_preferred_images,
)

if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject


CORRECTION_REVISION_FORMAT = "darsia-fluidflower-image-correction-v1"
CORRECTION_TASK_CONTRACT_FORMAT = "darsia-fluidflower-image-correction-task-v1"
UNIFORM24_ALGORITHM = "uniform24-neutral-tone-affine-v1"
UNIFORM24_VALIDITY_CONTRACT = "uniform24-measured-swatch-censoring-v2"
DARSIA_BASELINE_COLOR_ALGORITHM = "darsia-baseline-color-correction-v1"
IDENTITY_ALGORITHM = "identity-no-external-color-target-v1"
COLORCHECKER_ALGORITHMS = frozenset(
    {DARSIA_BASELINE_COLOR_ALGORITHM, UNIFORM24_ALGORITHM}
)
SUPPORTED_CORRECTION_ALGORITHMS = frozenset(
    {IDENTITY_ALGORITHM, *COLORCHECKER_ALGORITHMS}
)
SUPPORTED_IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff")
SWATCH_NAMES = (
    "dark skin",
    "light skin",
    "blue sky",
    "foliage",
    "blue flower",
    "bluish green",
    "orange",
    "purplish blue",
    "moderate red",
    "purple",
    "yellow green",
    "orange yellow",
    "blue",
    "green",
    "red",
    "yellow",
    "magenta",
    "cyan",
    "white",
    "neutral 8",
    "neutral 6.5",
    "neutral 5",
    "neutral 3.5",
    "black",
)


def default_correction_config() -> dict[str, Any]:
    return {
        "application_scope": "series",
        "representative_run_id": "",
        "representative_preview_role": "",
        "algorithm": UNIFORM24_ALGORITHM,
        "pipeline_order": [
            "geometric_registration",
            "spatial_illumination",
            "colorchecker_uniform24",
        ],
        "reference_image": "",
        "target_image": "",
        "colorchecker_corner": "upper_right",
        "colorchecker_roi_rc": None,
        "settings": {
            "neutral_tone": True,
            "robust_affine": True,
            "ridge": 0.01,
            "huber_delta": 0.025,
            "clip": True,
            "residual_warning_threshold": 0.015,
            "max_swatch_warning_threshold": 0.05,
        },
        "batch": {
            "inventory_scope": "registered_run_mappings",
            "source_root": "",
            "recursive": True,
            "run_mapping": "parent_folder",
            "flat_run_id": "run",
            "output_encoding": "png16",
            "frames": [],
        },
    }


def identity_correction_config() -> dict[str, Any]:
    """Return an explicit no-external-target correction contract."""

    config = default_correction_config()
    config.update(
        {
            "algorithm": IDENTITY_ALGORITHM,
            "pipeline_order": [
                "geometric_registration",
                "spatial_illumination",
                "identity_radiometric",
            ],
            "reference_image": "",
            "colorchecker_roi_rc": None,
        }
    )
    return config


def darsia_baseline_correction_config() -> dict[str, Any]:
    """Return the original DarSIA baseline-relative ColorCorrection contract."""

    config = default_correction_config()
    config.update(
        {
            "algorithm": DARSIA_BASELINE_COLOR_ALGORITHM,
            "pipeline_order": [
                "geometric_registration",
                "spatial_illumination",
                "darsia_baseline_color_correction",
            ],
        }
    )
    config["settings"].update(
        {
            "whitebalancing": True,
            "colorbalancing": "affine",
            "balancing": "darsia",
            "clip": False,
        }
    )
    return config


def _utc_now() -> str:
    from darsia.fluidflower.project import utc_now

    return utc_now()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _json_digest(payload: Any) -> str:
    from darsia.fluidflower.analysis_domains import _json_digest as _canonical_ascii_digest
    return _canonical_ascii_digest(payload)


def _sha256_file(path: Path) -> str:
    from .project import _sha256_file_uncached
    return _sha256_file_uncached(path)


def discover_correction_frames(
    source_root: Path,
    *,
    recursive: bool = True,
    run_mapping: str = "parent_folder",
    flat_run_id: str = "run",
) -> list[dict[str, Any]]:
    """Build an immutable full-series inventory for batch correction."""

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if run_mapping not in {"parent_folder", "flat"}:
        raise ValueError("Run mapping must be parent_folder or flat")
    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root)
        if run_mapping == "flat":
            run_id = str(flat_run_id).strip().casefold() or "run"
        else:
            run_id = (
                relative.parts[0].strip().casefold()
                if len(relative.parts) > 1
                else str(flat_run_id).strip().casefold() or "run"
            )
        stat = path.stat()
        identity = {
            "run_id": run_id,
            "relative_path": relative.as_posix(),
            "size_bytes": int(stat.st_size),
            "sha256": _sha256_file(path),
        }
        records.append(
            {
                **identity,
                "frame_id": _json_digest(identity)[:24],
                "source_path": str(path),
                "source_mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return records


def discover_project_correction_frames(
    project: FluidFlowerProject,
) -> list[dict[str, Any]]:
    """Inventory only CO2 folders mapped to currently included experiments."""

    policy = project.image_file_policy()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run in project.manifest.get("runs") or []:
        if not isinstance(run, Mapping):
            continue
        run_id = str(run.get("run_id") or "run").strip().casefold() or "run"
        mapping = dict(run.get("source_mapping") or {})
        run_root = Path(str(mapping.get("experiment_folder") or "")).expanduser()
        sequence_rows = [
            dict(row)
            for row in mapping.get("co2_sequences") or []
            if isinstance(row, Mapping) and str(row.get("folder") or "")
        ]
        if sequence_rows:
            sequence_rows.sort(key=lambda row: int(row.get("order") or 10**9))
            folders = [Path(str(row["folder"])).expanduser() for row in sequence_rows]
        else:
            folders = [
                Path(str(value)).expanduser()
                for value in (
                    mapping.get("experiment_folders")
                    or mapping.get("co2_image_folders")
                    or []
                )
                if str(value)
            ]
        for sequence_index, folder in enumerate(folders, start=1):
            if not folder.is_dir():
                continue
            candidates = sorted(
                (
                    path.resolve()
                    for path in folder.rglob("*")
                    if path.is_file()
                    and path.suffix.casefold() in ALL_IMAGE_SUFFIXES
                ),
                key=lambda path: str(path).casefold(),
            )
            by_parent: dict[Path, list[Path]] = {}
            for path in candidates:
                by_parent.setdefault(path.parent, []).append(path)
            selected = [
                path
                for parent in sorted(by_parent, key=lambda path: str(path).casefold())
                for path in select_preferred_images(by_parent[parent], policy)
            ]
            raw_files = [
                path for path in selected if path.suffix.casefold() in CAMERA_RAW_SUFFIXES
            ]
            if raw_files:
                raise ValueError(
                    "The project selects camera RAW files, but this DarSIA installation "
                    "does not have a validated RAW decoder for image correction. Select "
                    "processed JPEG/TIFF/PNG files, or install and validate a RAW adapter."
                )
            for path in selected:
                path_key = os.path.normcase(str(path))
                if path_key in seen:
                    continue
                seen.add(path_key)
                stat = path.stat()
                try:
                    relative = path.relative_to(run_root).as_posix()
                except ValueError:
                    relative = path.name
                identity = {
                    "run_id": run_id,
                    "relative_path": relative,
                    "sequence_index": sequence_index,
                    "size_bytes": int(stat.st_size),
                    "sha256": _sha256_file(path),
                }
                records.append(
                    {
                        **identity,
                        "frame_id": _json_digest(identity)[:24],
                        "source_path": str(path),
                        "source_mtime_ns": int(stat.st_mtime_ns),
                    }
                )
    return records


def correction_task_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    batch = deepcopy(dict(config.get("batch") or {}))
    frames = [
        deepcopy(dict(row))
        for row in batch.get("frames") or []
        if isinstance(row, Mapping)
    ]
    reference_value = str(config.get("reference_image") or "").strip()
    reference = (
        Path(reference_value).expanduser().resolve() if reference_value else None
    )
    contract = {
        "schema": 1,
        "format": CORRECTION_TASK_CONTRACT_FORMAT,
        "application_scope": str(config.get("application_scope") or "series"),
        "representative_run_id": str(
            config.get("representative_run_id") or ""
        ).strip().casefold(),
        "representative_preview_role": str(
            config.get("representative_preview_role") or ""
        ),
        "algorithm": str(config.get("algorithm") or ""),
        "reference": {
            "path": str(reference) if reference is not None else "",
            "sha256": (
                _sha256_file(reference)
                if reference is not None and reference.is_file()
                else ""
            ),
        },
        "colorchecker_corner": str(config.get("colorchecker_corner") or "upper_right"),
        "colorchecker_roi_rc": deepcopy(config.get("colorchecker_roi_rc")),
        "settings": deepcopy(dict(config.get("settings") or {})),
        "batch": {
            "inventory_scope": str(
                batch.get("inventory_scope") or "manual_source_root"
            ),
            "source_root": str(batch.get("source_root") or ""),
            "recursive": bool(batch.get("recursive", True)),
            "run_mapping": str(batch.get("run_mapping") or "parent_folder"),
            "flat_run_id": str(batch.get("flat_run_id") or "run"),
            "output_encoding": str(batch.get("output_encoding") or "png16"),
            "frames": frames,
        },
    }
    contract["content_sha256"] = _json_digest(contract)
    return contract


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _append_jsonl as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


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
                raise ValueError(f"Invalid correction revision at line {line_number}")
            records.append(payload)
    return records


class CorrectionRevisionStore:
    """Append-only image-correction revisions plus a replaceable current pointer."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.revisions_path = (
            project.root / "overrides" / "image_correction" / "revisions.jsonl"
        )
        self.current_path = (
            project.root / "overrides" / "image_correction" / "current.json"
        )
        self.artifact_root = project.root / "results" / "image_correction" / "plans"

    def revisions(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.revisions_path)

    def current(self) -> dict[str, Any]:
        if self.current_path.is_file():
            payload = json.loads(self.current_path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                return payload
        revisions = self.revisions()
        return deepcopy(revisions[-1]) if revisions else default_correction_config()

    @staticmethod
    def dependency_sha256(revision: Mapping[str, Any] | None) -> str:
        """Return the stable scientific identity consumed by downstream stages."""

        row = dict(revision or {})
        stored = str(row.get("dependency_sha256") or "")
        if stored:
            return stored
        contract = dict(row.get("task_contract") or {})
        stored = str(contract.get("content_sha256") or "")
        if stored:
            return stored
        if row.get("algorithm") in SUPPORTED_CORRECTION_ALGORITHMS:
            return str(correction_task_contract(row)["content_sha256"])
        return ""

    def _record_activation(
        self,
        revision: Mapping[str, Any],
        *,
        previous: Mapping[str, Any] | None,
        actor: str,
        reason: str,
    ) -> tuple[str, ...]:
        previous_sha = self.dependency_sha256(previous)
        selected_sha = self.dependency_sha256(revision)
        changed = bool(previous_sha and previous_sha != selected_sha)
        revision_changed = changed or not previous_sha
        current_status = str(
            self.project.module_states.get("image_correction", {}).get("status") or ""
        )
        invalidated: tuple[str, ...] = ()
        if (
            changed
            or not previous_sha
            or current_status not in {"approved", "complete", "review_required"}
        ):
            invalidated = self.project.set_module_status(
                "image_correction",
                "review_required",
                updated_by=str(actor).strip(),
                reason=(
                    str(reason).strip()
                    or "A different image-correction revision was selected."
                ),
            )
        if revision_changed:
            run_ids = [
                str(row.get("run_id") or "")
                for row in self.project.manifest.get("runs") or []
                if str(row.get("run_id") or "")
            ]
            if run_ids:
                self.project.set_run_module_status(
                    run_ids,
                    "image_correction",
                    "review_required",
                    updated_by=str(actor).strip(),
                    reason=(
                        str(reason).strip()
                        or "A different image-correction revision was selected."
                    ),
                )
        return invalidated

    def activate_revision(
        self,
        revision_id: str,
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        """Select an immutable correction revision and invalidate its descendants."""

        selected = next(
            (
                row
                for row in self.revisions()
                if str(row.get("revision_id") or "") == str(revision_id).strip()
            ),
            None,
        )
        if selected is None:
            raise KeyError(f"Unknown image-correction revision: {revision_id!r}")
        previous = self.current()
        if str(previous.get("revision_id") or "") == str(
            selected.get("revision_id") or ""
        ):
            return deepcopy(selected)
        _atomic_json(self.current_path, selected)
        invalidated = self._record_activation(
            selected,
            previous=previous,
            actor=actor,
            reason=reason,
        )
        self.project._audit(
            "image_correction_revision_activated",
            actor,
            {
                "revision_id": selected["revision_id"],
                "dependency_sha256": self.dependency_sha256(selected),
                "reason": reason,
                "invalidated_modules": list(invalidated),
            },
        )
        return deepcopy(selected)

    def save_revision(
        self,
        config: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing_revisions = self.revisions()
        previous = self.current() if existing_revisions else {}
        candidate = deepcopy(dict(config))
        algorithm = str(candidate.get("algorithm") or "")
        if algorithm not in SUPPORTED_CORRECTION_ALGORITHMS:
            raise ValueError(
                f"Unsupported image-correction algorithm: {candidate.get('algorithm')!r}"
            )
        reference_value = str(candidate.get("reference_image") or "").strip()
        reference = Path(reference_value).expanduser() if reference_value else None
        if algorithm in COLORCHECKER_ALGORITHMS and (
            reference is None or not reference.is_file()
        ):
            raise FileNotFoundError(
                f"Reference image is unavailable: {reference_value or 'not selected'}"
            )
        roi = candidate.get("colorchecker_roi_rc")
        if roi is not None and np.asarray(roi).shape != (4, 2):
            raise ValueError("ColorChecker ROI must contain four row/column points")
        revision_id = uuid.uuid4().hex
        contract = correction_task_contract(candidate)
        revision = {
            "schema": 1,
            "format": CORRECTION_REVISION_FORMAT,
            "revision_id": revision_id,
            "plan_id": revision_id,
            "recorded_utc": _utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            **candidate,
            "diagnostics": deepcopy(dict(diagnostics or {})),
            "task_contract": contract,
            "dependency_sha256": contract["content_sha256"],
            "validation": {
                "execution_ready": bool(
                    contract["batch"]["frames"]
                    and (
                        algorithm == IDENTITY_ALGORITHM
                        or contract["reference"]["sha256"]
                    )
                ),
                "frame_count": len(contract["batch"]["frames"]),
            },
        }
        revision["content_sha256"] = _json_digest(revision)
        _append_jsonl(self.revisions_path, revision)
        _atomic_json(self.current_path, revision)
        _atomic_json(self.artifact_root / revision_id / "plan.json", revision)
        invalidated = self._record_activation(
            revision,
            previous=previous,
            actor=actor,
            reason=(
                f"Image-correction revision {revision_id[:8]} was saved. "
                "Dependent color paths and analyses require review or regeneration."
            ),
        )
        self.project._audit(
            "image_correction_revision_saved",
            actor,
            {
                "revision_id": revision["revision_id"],
                "algorithm": revision["algorithm"],
                "dependency_sha256": revision["dependency_sha256"],
                "reason": reason,
                "invalidated_modules": list(invalidated),
            },
        )
        return deepcopy(revision)


def srgb_to_linear(values: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    return np.asarray(
        np.where(
            array <= 0.04045,
            array / 12.92,
            ((array + 0.055) / 1.055) ** 2.4,
        ),
        dtype=dtype,
    )


def linear_to_srgb(values: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=dtype), 0.0, None)
    return np.asarray(
        np.where(
            array <= 0.0031308,
            array * 12.92,
            1.055 * array ** (1.0 / 2.4) - 0.055,
        ),
        dtype=dtype,
    )


def residual_summary(swatches: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    residuals = np.mean(
        np.abs(
            np.asarray(swatches, dtype=np.float64)
            - np.asarray(reference, dtype=np.float64)
        ),
        axis=-1,
    ).reshape(-1)
    if residuals.size != 24:
        raise ValueError("Expected all 24 ColorChecker fields")
    worst = int(np.argmax(residuals))
    return {
        "mean": float(np.mean(residuals)),
        "p95": float(np.percentile(residuals, 95.0)),
        "max": float(residuals[worst]),
        "worst_swatch_index": worst,
        "worst_swatch_name": SWATCH_NAMES[worst],
        "swatch_residuals": [float(value) for value in residuals],
    }


class NeutralToneFitError(ValueError):
    """Rejected neutral response, with the exact small inputs used by the fit."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)

    def __reduce__(self):
        return type(self), (str(self), self.diagnostics)


def _measured_validity(value, shape, *, role):
    if value is None:
        return np.ones(shape, dtype=bool)
    valid = np.asarray(value)
    if valid.dtype != np.bool_ or valid.shape != shape:
        raise ValueError(f"{role} validity must be an explicit boolean array of shape {shape}")
    return valid


def fit_neutral_tone_model(
    source_swatches: np.ndarray,
    destination_swatches: np.ndarray,
    *,
    source_validity: np.ndarray | None = None,
    reference_validity: np.ndarray | None = None,
) -> dict[str, Any]:
    source = srgb_to_linear(source_swatches)[-1]
    destination = srgb_to_linear(destination_swatches)[-1]
    censored = source_validity is not None or reference_validity is not None
    valid = (_measured_validity(source_validity, np.shape(source_swatches), role="Source")[-1]
             & _measured_validity(reference_validity, np.shape(destination_swatches), role="Reference")[-1])
    channels: list[dict[str, Any]] = []
    for channel in range(3):
        selected = np.flatnonzero(valid[:, channel])
        if censored and (len(selected) < 4
                or np.count_nonzero(np.diff(np.sort(source[selected, channel])) >= 1e-3) < 3):
            raise NeutralToneFitError(
                f"Insufficient valid measured neutrals for channel {channel}",
                dict(channel=channel, source_neutrals_linear=source.tolist(),
                     reference_neutrals_linear=destination.tolist(),
                     valid_neutrals=valid.tolist(), minimum_valid_neutrals=4),
            )
        order = selected[np.argsort(source[selected, channel])]
        source_sorted = source[order, channel]
        destination_sorted = destination[order, channel]
        anchors_x = source_sorted.copy()
        anchors_y = destination_sorted.copy()
        anchor_origins = [f"neutral_{int(index)}" for index in order]
        if anchors_x[0] > 1e-6:
            anchors_x = np.concatenate(([0.0], anchors_x))
            anchors_y = np.concatenate(([0.0], anchors_y))
            anchor_origins.insert(0, "black_endpoint")
        if anchors_x[-1] < 1.0 - 1e-6:
            white = 1.0
            extrapolated = censored and not valid[0, channel]
            if extrapolated:
                # Sensor clipping does not establish the reference's unit white.
                # Extend the last measured response instead of inventing that observation.
                measured_y = np.maximum.accumulate(anchors_y)
                last_interval = float(anchors_x[-1] - anchors_x[-2])
                if last_interval < 1e-3:
                    raise NeutralToneFitError(
                        f"Insufficient resolved highlight neutrals for channel {channel}",
                        dict(channel=channel, source_neutrals_linear=source.tolist(),
                             reference_neutrals_linear=destination.tolist(), valid_neutrals=valid.tolist()),
                    )
                slope = float((measured_y[-1] - measured_y[-2]) / last_interval)
                white = float(measured_y[-1] + slope * (1.0 - anchors_x[-1]))
                if not np.isfinite(white) or not 0.0 <= white <= 1.0:
                    raise NeutralToneFitError(
                        f"Measured highlight extrapolation is outside physical bounds for channel {channel}",
                        dict(channel=channel, source_neutrals_linear=source.tolist(),
                             reference_neutrals_linear=destination.tolist(),
                             valid_neutrals=valid.tolist(), highlight_slope=slope, highlight_at_one=white),
                    )
            anchors_x = np.concatenate((anchors_x, [1.0]))
            anchors_y = np.concatenate((anchors_y, [white]))
            anchor_origins.append("measured_highlight_extrapolation" if extrapolated else "white_endpoint")
        unique_x: list[float] = []
        unique_y: list[float] = []
        unique_origins: list[str] = []
        for x_value, y_value, origin in zip(anchors_x, anchors_y, anchor_origins, strict=True):
            x_float = float(np.clip(x_value, 0.0, 1.0))
            y_float = float(np.clip(y_value, 0.0, 1.0))
            if unique_x and abs(x_float - unique_x[-1]) <= 1e-6:
                unique_y[-1] = y_float
                unique_origins[-1] = origin
            else:
                unique_x.append(x_float)
                unique_y.append(y_float)
                unique_origins.append(origin)
        x_array = np.asarray(unique_x, dtype=np.float64)
        y_array = np.maximum.accumulate(np.asarray(unique_y, dtype=np.float64))
        intervals = np.diff(x_array)
        slopes = np.diff(y_array) / np.maximum(intervals, 1e-12)
        checked = slopes[intervals >= 1e-3]
        if not checked.size or np.min(checked) < 0.05 or np.max(checked) > 8.0:
            raise NeutralToneFitError(
                f"Neutral tone curve is outside physical bounds for channel {channel}",
                dict(
                    channel=channel,
                    source_neutrals_linear=source.tolist(),
                    reference_neutrals_linear=destination.tolist(),
                    source_sort_order=order.tolist(),
                    valid_neutrals=valid.tolist(),
                    anchors_x=x_array.tolist(),
                    anchors_y=y_array.tolist(),
                    anchor_origins=unique_origins,
                    slope_bounds=[0.05, 8.0],
                    minimum_checked_interval=1e-3,
                    intervals=[dict(
                        index=index, delta_x=float(interval), slope=float(slope),
                        checked=bool(interval >= 1e-3),
                        outside_bounds=bool(interval >= 1e-3 and (slope < 0.05 or slope > 8.0)),
                        endpoints=unique_origins[index:index + 2],
                    ) for index, (interval, slope) in enumerate(zip(intervals, slopes, strict=True))],
                ),
            )
        channels.append(
            {
                "anchors_x": [float(value) for value in x_array],
                "anchors_y": [float(value) for value in y_array],
                "slope_min": float(np.min(checked)),
                "slope_max": float(np.max(checked)),
            }
        )
        if censored:
            channels[-1].update(valid_neutral_indices=selected.tolist(),
                excluded_neutral_indices=np.flatnonzero(~valid[:,channel]).tolist(),
                endpoint_policy="last-valid-measured-slope" if not valid[0,channel] else "unit-white",
                anchor_origins=unique_origins)
    return {"mode": "neutral-tone", "channels": channels}


def apply_neutral_tone(image: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    source = np.asarray(image)
    if source.dtype == np.uint8:
        source = source.astype(np.float32) / 255.0
    elif source.dtype == np.uint16:
        source = source.astype(np.float32) / 65535.0
    else:
        source = source.astype(np.float32)
    linear = srgb_to_linear(source, dtype=np.float32)
    corrected = np.empty_like(linear)
    channels = list(model.get("channels") or [])
    if len(channels) != 3:
        raise ValueError("Neutral tone model requires three channels")
    for channel, settings in enumerate(channels):
        corrected[..., channel] = np.interp(
            np.clip(linear[..., channel], 0.0, 1.0),
            np.asarray(settings["anchors_x"], dtype=np.float32),
            np.asarray(settings["anchors_y"], dtype=np.float32),
        )
    return linear_to_srgb(corrected, dtype=np.float32)


def fit_regularized_affine(
    source_swatches: np.ndarray,
    destination_swatches: np.ndarray,
    *,
    ridge: float,
    huber_delta: float,
    valid_swatches: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    source = srgb_to_linear(source_swatches).reshape(-1, 3)
    destination = srgb_to_linear(destination_swatches).reshape(-1, 3)
    if valid_swatches is not None:
        valid = _measured_validity(valid_swatches, np.shape(source_swatches)[:-1], role="Complete patch").reshape(-1)
        source, destination = source[valid], destination[valid]
        if (len(source) < 4 or np.linalg.matrix_rank(np.column_stack((source,np.ones(len(source))))) < 4
                or np.linalg.matrix_rank(np.column_stack((destination,np.ones(len(destination))))) < 4):
            raise ValueError("Insufficient independent complete ColorChecker patches for affine fitting")
    design = np.column_stack((source, np.ones(source.shape[0], dtype=np.float64)))
    prior = np.zeros((4, 3), dtype=np.float64)
    prior[:3] = np.eye(3)
    penalty = np.diag([ridge, ridge, ridge, 4.0 * ridge])
    weights = np.ones(source.shape[0], dtype=np.float64)
    coefficients = prior.copy()
    for _ in range(8):
        weighted = design * weights[:, None]
        coefficients = np.linalg.solve(
            design.T @ weighted + penalty,
            design.T @ (destination * weights[:, None]) + penalty @ prior,
        )
        errors = np.linalg.norm(design @ coefficients - destination, axis=1)
        updated = np.minimum(1.0, huber_delta / np.maximum(errors, 1e-12))
        if np.max(np.abs(updated - weights)) < 1e-5:
            weights = updated
            break
        weights = updated
    scaling = coefficients[:3]
    translation = coefficients[3]
    diagnostics = {
        "condition": float(np.linalg.cond(scaling)),
        "diagonal_min": float(np.min(np.diag(scaling))),
        "diagonal_max": float(np.max(np.diag(scaling))),
        "off_diagonal_max_abs": float(
            np.max(np.abs(scaling - np.diag(np.diag(scaling))))
        ),
        "translation_max_abs": float(np.max(np.abs(translation))),
    }
    return scaling, translation, diagnostics


def _valid_residual_summary(swatches, reference, valid_swatches=None):
    summary = residual_summary(swatches, reference)
    if valid_swatches is None:
        return summary
    selected = np.flatnonzero(np.asarray(valid_swatches).reshape(-1))
    residuals = np.asarray(summary["swatch_residuals"])
    values = residuals[selected]
    worst = int(selected[np.argmax(values)])
    summary.update(mean=float(np.mean(values)), p95=float(np.percentile(values,95.)),
        max=float(residuals[worst]), worst_swatch_index=worst, worst_swatch_name=SWATCH_NAMES[worst],
        fit_swatch_indices=selected.tolist(), excluded_swatch_indices=np.flatnonzero(
            ~np.asarray(valid_swatches).reshape(-1)).tolist())
    return summary


def affine_is_bounded(diagnostics: Mapping[str, float]) -> bool:
    return bool(
        diagnostics["condition"] <= 4.0
        and diagnostics["diagonal_min"] >= 0.75
        and diagnostics["diagonal_max"] <= 1.35
        and diagnostics["off_diagonal_max_abs"] <= 0.20
        and diagnostics["translation_max_abs"] <= 0.06
    )


def apply_affine(
    image: np.ndarray,
    scaling: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    linear = srgb_to_linear(image, dtype=np.float32)
    transformed = linear @ np.asarray(scaling, dtype=np.float32)
    transformed += np.asarray(translation, dtype=np.float32)
    return linear_to_srgb(transformed, dtype=np.float32)


def _float_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    return np.asarray(array, dtype=np.float32)


def _preview_image(image: np.ndarray, maximum: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, float(maximum) / float(max(height, width)))
    if scale >= 1.0:
        return _float_rgb(image)
    return cv2.resize(
        _float_rgb(image),
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _checker_orientation_score(swatches: np.ndarray) -> float:
    neutral = np.asarray(swatches, dtype=np.float64)[-1]
    luminance = neutral @ np.asarray([0.2126, 0.7152, 0.0722])
    chroma = float(np.mean(np.std(neutral, axis=1)))
    descending_violation = float(np.sum(np.maximum(np.diff(luminance), 0.0)))
    dynamic_range_penalty = max(0.0, 0.30 - float(np.ptp(luminance)))
    return chroma + 2.0 * descending_violation + 2.0 * dynamic_range_penalty


def _detect_corner_colorchecker(
    reference_array: np.ndarray,
    corner: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and orient one Classic chart inside a corner search window."""

    from colour_checker_detection import detect_colour_checkers_segmentation

    import darsia

    height, width = reference_array.shape[:2]
    upper = corner.startswith("upper")
    right = corner.endswith("right")
    row_start, row_stop = (0, max(1, round(0.45 * height))) if upper else (
        min(height - 1, round(0.55 * height)),
        height,
    )
    col_start, col_stop = (min(width - 1, round(0.65 * width)), width) if right else (
        0,
        max(1, round(0.35 * width)),
    )
    crop = reference_array[row_start:row_stop, col_start:col_stop]
    working_width = min(1440, max(500, crop.shape[1]))
    detections = detect_colour_checkers_segmentation(
        crop,
        additional_data=True,
        working_width=working_width,
    )
    if not detections:
        raise ValueError(f"No ColorChecker was detected in the {corner.replace('_', ' ')}")
    scale = float(crop.shape[1]) / float(working_width)
    center = np.asarray(
        [0.0 if upper else height, width if right else 0.0], dtype=np.float64
    )
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for detection in detections:
        local_rc = np.fliplr(
            np.asarray(detection.quadrilateral, dtype=np.float64) * scale
        )
        global_rc = local_rc + np.asarray([row_start, col_start], dtype=np.float64)
        ordered = np.asarray(darsia.sort_quad(darsia.make_voxel(global_rc)), dtype=int)
        for reverse in (False, True):
            base = ordered[::-1] if reverse else ordered
            for shift in range(4):
                roi = np.roll(base, -shift, axis=0)
                try:
                    extractor = darsia.ColorCorrection(
                        config={"roi": roi.tolist(), "active": True, "clip": False}
                    )
                    swatches = np.asarray(
                        darsia.CustomColorChecker(
                            image=extractor._restrict_to_roi(reference_array)
                        ).swatches_rgb,
                        dtype=np.float32,
                    )
                except (AssertionError, ValueError):
                    continue
                center_distance = float(
                    np.linalg.norm(np.mean(roi, axis=0) - center)
                    / max(float(np.hypot(height, width)), 1.0)
                )
                candidates.append(
                    (_checker_orientation_score(swatches) + 0.02 * center_distance, roi, swatches)
                )
    if not candidates:
        raise ValueError("The detected ColorChecker could not be oriented")
    _, roi, swatches = min(candidates, key=lambda item: item[0])
    return np.asarray(swatches, dtype=np.float32), np.asarray(roi, dtype=int)


def _reference_colorchecker(
    reference_array: np.ndarray,
    *,
    corner: str,
    roi_rc: list[list[int]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    import darsia

    reference = darsia.Image(img=reference_array)
    if roi_rc is None:
        try:
            return _detect_corner_colorchecker(reference_array, corner)
        except (AssertionError, ValueError):
            pass
        try:
            reference_checker, detected_roi = darsia.find_colorchecker(reference, corner)
            swatches = np.asarray(reference_checker.swatches_rgb, dtype=np.float32)
            if _checker_orientation_score(swatches) <= 0.20:
                return swatches, np.asarray(detected_roi, dtype=int)
        except (AssertionError, ValueError):
            pass
        raise ValueError(f"No valid ColorChecker was detected near {corner}")
    roi = np.asarray(roi_rc, dtype=int)
    if roi.shape != (4, 2):
        raise ValueError("ColorChecker ROI must contain four row/column points")
    extractor = darsia.ColorCorrection(
        config={"roi": roi.tolist(), "active": True, "clip": False}
    )
    swatches = np.asarray(
        darsia.CustomColorChecker(
            image=extractor._restrict_to_roi(reference.img)
        ).swatches_rgb,
        dtype=np.float32,
    )
    return swatches, roi


def fit_uniform24_array_transform(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    *,
    corner: str = "upper_right",
    roi_rc: list[list[int]] | None = None,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit and serialize one reusable full-resolution Uniform24 transform."""

    reference_array = _float_rgb(np.asarray(reference_image))
    target_array = _float_rgb(np.asarray(target_image))
    if reference_array.shape[:2] != target_array.shape[:2]:
        raise ValueError("Reference and target images must share one registered grid")
    reference_swatches, roi = _reference_colorchecker(
        reference_array, corner=corner, roi_rc=roi_rc
    )
    return fit_uniform24_to_reference_swatches(
        target_array,
        reference_swatches,
        corner=corner,
        roi_rc=roi.tolist(),
        settings=settings,
    )


def fit_uniform24_to_reference_swatches(
    target_image: np.ndarray,
    reference_swatches: np.ndarray,
    *,
    corner: str = "upper_right",
    roi_rc: list[list[int]] | None = None,
    settings: Mapping[str, Any] | None = None,
    source_validity: np.ndarray | None = None,
    reference_validity: np.ndarray | None = None,
    validity_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one registered frame to an immutable 24-patch RGB target.

    The caller must bind the patch values and their provenance to its job
    receipt. Passing the same frozen target to every run gives a genuine common
    ColorChecker target without using a different run's image geometry.
    """

    import darsia

    reference_swatches = np.asarray(reference_swatches, dtype=np.float32)
    if reference_swatches.shape != (4, 6, 3):
        raise ValueError("Reference must contain four rows of six RGB swatches")
    if not np.all(np.isfinite(reference_swatches)) or np.any(
        (reference_swatches < 0.0) | (reference_swatches > 1.0)
    ):
        raise ValueError("Reference swatches must be finite sRGB values in [0, 1]")
    if _checker_orientation_score(reference_swatches) > 0.20:
        raise ValueError("Reference ColorChecker orientation or neutrals are invalid")
    censored = source_validity is not None or reference_validity is not None
    source_valid = _measured_validity(source_validity, reference_swatches.shape, role="Source")
    reference_valid = _measured_validity(reference_validity, reference_swatches.shape, role="Reference")
    complete_valid = np.all(source_valid & reference_valid, axis=2) if censored else None
    if censored and np.count_nonzero(complete_valid) < 4:
        raise ValueError("Insufficient complete valid ColorChecker patches")
    measured = None if not censored else dict(
        contract=UNIFORM24_VALIDITY_CONTRACT, source_validity=source_valid.tolist(),
        reference_validity=reference_valid.tolist(),
        complete_fit_swatch_indices=np.flatnonzero(complete_valid.reshape(-1)).tolist(),
        excluded_swatch_channels=np.argwhere(~(source_valid & reference_valid)).tolist(),
        evidence=deepcopy(dict(validity_evidence or {})),
        invalid_measurement_policy="exclude any explicitly invalid source/reference channel; affine requires complete RGB patches",
        highlight_policy="last-valid-measured-slope only when brightest neutral is invalid",
        minimum_valid_neutrals_per_channel=4, minimum_affine_design_rank=4)
    options = deepcopy(default_correction_config()["settings"])
    options.update(dict(settings or {}))
    target_array = _float_rgb(np.asarray(target_image))
    target = darsia.Image(img=target_array)
    _, roi = _reference_colorchecker(
        target_array, corner=corner, roi_rc=roi_rc
    )
    extractor = darsia.ColorCorrection(
        base=darsia.CustomColorChecker(reference_colors=reference_swatches),
        config={"roi": roi.tolist(), "active": True, "clip": False},
    )
    target_swatches = np.asarray(
        darsia.CustomColorChecker(
            image=extractor._restrict_to_roi(target.img)
        ).swatches_rgb,
        dtype=np.float32,
    )
    before = _valid_residual_summary(target_swatches, reference_swatches, complete_valid)
    try:
        tone_model = fit_neutral_tone_model(target_swatches, reference_swatches,
            source_validity=source_valid if censored else None,
            reference_validity=reference_valid if censored else None)
    except NeutralToneFitError as error:
        error.diagnostics.update(
            algorithm=UNIFORM24_ALGORITHM,
            source_shape=list(target.img.shape),
            roi_rc=roi.tolist(),
            source_swatches_srgb=target_swatches.tolist(),
            reference_swatches_srgb=reference_swatches.tolist(),
            source_orientation_score=float(_checker_orientation_score(target_swatches)),
        )
        if censored:
            error.diagnostics["measurement_validity"] = measured
        raise
    tone_swatches = (
        apply_neutral_tone(target_swatches, tone_model)
        if options.get("neutral_tone", True)
        else target_swatches.copy()
    )
    tone_summary = _valid_residual_summary(tone_swatches, reference_swatches, complete_valid)
    scaling, translation, affine_diagnostics = fit_regularized_affine(
        tone_swatches,
        reference_swatches,
        ridge=float(options["ridge"]),
        huber_delta=float(options["huber_delta"]),
        valid_swatches=complete_valid,
    )
    affine_bounds_ok = affine_is_bounded(affine_diagnostics)
    affine_candidate = apply_affine(tone_swatches, scaling, translation)
    affine_summary = _valid_residual_summary(affine_candidate, reference_swatches, complete_valid)
    affine_accepted = bool(options.get("robust_affine", True)) and bool(
        affine_bounds_ok and affine_summary["mean"] < tone_summary["mean"]
    )
    corrected_swatches = affine_candidate if affine_accepted else tone_swatches
    after = _valid_residual_summary(corrected_swatches, reference_swatches, complete_valid)
    warning_threshold = float(options["residual_warning_threshold"])
    max_threshold = float(options["max_swatch_warning_threshold"])
    flagged = bool(after["mean"] > warning_threshold or after["max"] > max_threshold)
    transform = {
        "algorithm": UNIFORM24_ALGORITHM,
        "roi_rc": roi.tolist(),
        "neutral_tone_enabled": bool(options.get("neutral_tone", True)),
        "tone_model": tone_model,
        "affine_enabled": bool(affine_accepted),
        "affine_scaling": np.asarray(scaling, dtype=np.float64).tolist(),
        "affine_translation": np.asarray(translation, dtype=np.float64).tolist(),
        "clip": bool(options.get("clip", True)),
    }
    if censored:
        transform["measurement_validity"] = measured
    return {
        "transform": transform,
        "roi_rc": roi.tolist(),
        "reference_swatches": reference_swatches,
        "target_swatches": target_swatches,
        "corrected_swatches": corrected_swatches,
        "diagnostics": {
            "algorithm": UNIFORM24_ALGORITHM,
            **({"measurement_validity": measured} if censored else {}),
            "source_shape": list(target.img.shape),
            "before": before,
            "tone": tone_summary,
            "after": after,
            "affine": affine_diagnostics,
            "affine_candidate": affine_summary,
            "affine_bounds_ok": affine_bounds_ok,
            "affine_accepted": affine_accepted,
            "flagged": flagged,
            "flag_reasons": [
                reason
                for condition, reason in (
                    (after["mean"] > warning_threshold, "mean ColorChecker residual"),
                    (after["max"] > max_threshold, "maximum ColorChecker residual"),
                    (
                        bool(options.get("robust_affine", True))
                        and not affine_bounds_ok,
                        "robust affine transform rejected by bounds",
                    ),
                )
                if condition
            ],
        },
    }


def apply_uniform24_array_transform(
    image: np.ndarray,
    transform: Mapping[str, Any],
) -> np.ndarray:
    """Apply a serialized Uniform24 transform without changing image resolution."""

    corrected = _float_rgb(np.asarray(image))
    if bool(transform.get("neutral_tone_enabled", True)):
        corrected = apply_neutral_tone(
            corrected,
            dict(transform.get("tone_model") or {}),
        )
    if bool(transform.get("affine_enabled")):
        corrected = apply_affine(
            corrected,
            np.asarray(transform.get("affine_scaling"), dtype=np.float64),
            np.asarray(transform.get("affine_translation"), dtype=np.float64),
        )
    if bool(transform.get("clip", True)):
        corrected = np.clip(corrected, 0.0, 1.0)
    return np.asarray(corrected, dtype=np.float32)


def apply_darsia_baseline_array_correction(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    *,
    corner: str = "upper_right",
    roi_rc: list[list[int]] | None = None,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply DarSIA's original baseline-relative ColorCorrection implementation.

    This deliberately preserves the historical DarSIA balancing choices so it can
    serve as a reproducible legacy comparator to Uniform24.
    """

    import darsia

    options = deepcopy(darsia_baseline_correction_config()["settings"])
    options.update(dict(settings or {}))
    reference_array = _float_rgb(np.asarray(reference_image))
    target_array = _float_rgb(np.asarray(target_image))
    if reference_array.shape[:2] != target_array.shape[:2]:
        raise ValueError("Reference and target images must share one registered grid")
    reference_swatches, roi = _reference_colorchecker(
        reference_array, corner=corner, roi_rc=roi_rc
    )
    correction = darsia.ColorCorrection(
        base=darsia.CustomColorChecker(reference_colors=reference_swatches),
        config={
            "roi": roi.tolist(),
            "active": True,
            "whitebalancing": bool(options.get("whitebalancing", True)),
            "colorbalancing": str(options.get("colorbalancing") or "affine"),
            "balancing": str(options.get("balancing") or "darsia"),
            "clip": bool(options.get("clip", False)),
        },
    )
    target_swatches = np.asarray(
        darsia.CustomColorChecker(
            image=correction._restrict_to_roi(target_array)
        ).swatches_rgb,
        dtype=np.float32,
    )
    corrected = np.asarray(correction.correct_array(target_array), dtype=np.float32)
    corrected_swatches = np.asarray(
        darsia.CustomColorChecker(
            image=correction._restrict_to_roi(corrected)
        ).swatches_rgb,
        dtype=np.float32,
    )
    before = residual_summary(target_swatches, reference_swatches)
    after = residual_summary(corrected_swatches, reference_swatches)
    warning_threshold = float(options["residual_warning_threshold"])
    max_threshold = float(options["max_swatch_warning_threshold"])
    flagged = bool(after["mean"] > warning_threshold or after["max"] > max_threshold)
    transform = {
        "algorithm": DARSIA_BASELINE_COLOR_ALGORITHM,
        "roi_rc": roi.tolist(),
        "whitebalancing": bool(options.get("whitebalancing", True)),
        "colorbalancing": str(options.get("colorbalancing") or "affine"),
        "balancing": str(options.get("balancing") or "darsia"),
        "clip": bool(options.get("clip", False)),
        "implementation": "darsia.ColorCorrection",
    }
    return {
        "corrected": corrected,
        "transform": transform,
        "roi_rc": roi.tolist(),
        "reference_swatches": reference_swatches,
        "target_swatches": target_swatches,
        "corrected_swatches": corrected_swatches,
        "diagnostics": {
            "algorithm": DARSIA_BASELINE_COLOR_ALGORITHM,
            "source_shape": list(target_array.shape),
            "before": before,
            "after": after,
            "flagged": flagged,
            "flag_reasons": [
                reason
                for condition, reason in (
                    (after["mean"] > warning_threshold, "mean ColorChecker residual"),
                    (after["max"] > max_threshold, "maximum ColorChecker residual"),
                )
                if condition
            ],
            "legacy_contract": (
                "Original DarSIA baseline-relative ColorCorrection with adaptive "
                "white and affine color balancing."
            ),
        },
    }


def calculate_darsia_baseline_array_preview(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    *,
    corner: str = "upper_right",
    roi_rc: list[list[int]] | None = None,
    settings: Mapping[str, Any] | None = None,
    preview_maximum: int = 1800,
) -> dict[str, Any]:
    """Return a display preview for the original DarSIA correction."""

    fitted = apply_darsia_baseline_array_correction(
        reference_image,
        target_image,
        corner=corner,
        roi_rc=roi_rc,
        settings=settings,
    )
    original_preview = _preview_image(np.asarray(target_image), preview_maximum)
    corrected_preview = _preview_image(fitted["corrected"], preview_maximum)
    difference = np.clip(np.abs(corrected_preview - original_preview) * 4.0, 0.0, 1.0)
    diagnostics = deepcopy(dict(fitted["diagnostics"]))
    diagnostics["preview_shape"] = list(corrected_preview.shape)
    return {
        "original": original_preview,
        "corrected": np.clip(corrected_preview, 0.0, 1.0),
        "difference": difference,
        "transform": fitted["transform"],
        "roi_rc": fitted["roi_rc"],
        "reference_swatches": fitted["reference_swatches"],
        "target_swatches": fitted["target_swatches"],
        "corrected_swatches": fitted["corrected_swatches"],
        "diagnostics": diagnostics,
    }


def calculate_uniform24_array_preview(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    *,
    corner: str = "upper_right",
    roi_rc: list[list[int]] | None = None,
    settings: Mapping[str, Any] | None = None,
    preview_maximum: int = 1800,
) -> dict[str, Any]:
    """Fit Uniform24 to two already registered arrays on the same image grid."""
    fitted = fit_uniform24_array_transform(
        reference_image,
        target_image,
        corner=corner,
        roi_rc=roi_rc,
        settings=settings,
    )
    original_preview = _preview_image(np.asarray(target_image), preview_maximum)
    corrected_preview = apply_uniform24_array_transform(
        original_preview,
        fitted["transform"],
    )
    difference = np.clip(np.abs(corrected_preview - original_preview) * 4.0, 0.0, 1.0)
    diagnostics = deepcopy(dict(fitted["diagnostics"]))
    diagnostics["preview_shape"] = list(corrected_preview.shape)
    return {
        "original": original_preview,
        "corrected": corrected_preview,
        "difference": difference,
        "transform": fitted["transform"],
        "roi_rc": fitted["roi_rc"],
        "reference_swatches": fitted["reference_swatches"],
        "target_swatches": fitted["target_swatches"],
        "corrected_swatches": fitted["corrected_swatches"],
        "diagnostics": diagnostics,
    }


def calculate_uniform24_preview(
    reference_path: Path,
    target_path: Path,
    *,
    corner: str = "upper_right",
    roi_rc: list[list[int]] | None = None,
    settings: Mapping[str, Any] | None = None,
    preview_maximum: int = 1800,
) -> dict[str, Any]:
    """Fit a frame-to-reference transform and return images plus field-wise QC."""

    import darsia

    options = deepcopy(default_correction_config()["settings"])
    options.update(dict(settings or {}))
    reference_path = Path(reference_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()
    reference = darsia.imread(reference_path)
    target = darsia.imread(target_path)
    result = calculate_uniform24_array_preview(
        reference.img,
        target.img,
        corner=corner,
        roi_rc=roi_rc,
        settings=options,
        preview_maximum=preview_maximum,
    )
    result["diagnostics"].update(
        {
            "reference_image": str(reference_path),
            "target_image": str(target_path),
        }
    )
    return result


def calculate_color_correction_comparison_preview(
    reference_path: Path,
    target_path: Path,
    *,
    corner: str = "upper_right",
    roi_rc: list[list[int]] | None = None,
    settings: Mapping[str, Any] | None = None,
    preview_maximum: int = 1800,
) -> dict[str, dict[str, Any]]:
    """Calculate raw, original-DarSIA, and Uniform24 variants from one image read."""

    import darsia

    reference_path = Path(reference_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()
    reference = np.asarray(darsia.imread(reference_path).img)
    target = np.asarray(darsia.imread(target_path).img)
    improved = calculate_uniform24_array_preview(
        reference,
        target,
        corner=corner,
        roi_rc=roi_rc,
        settings=settings,
        preview_maximum=preview_maximum,
    )
    legacy = calculate_darsia_baseline_array_preview(
        reference,
        target,
        corner=corner,
        roi_rc=improved["roi_rc"],
        settings=settings,
        preview_maximum=preview_maximum,
    )
    raw_preview = np.asarray(improved["original"], dtype=np.float32)
    raw = {
        "original": raw_preview,
        "corrected": raw_preview.copy(),
        "difference": np.zeros_like(raw_preview, dtype=np.float32),
        "transform": {"algorithm": IDENTITY_ALGORITHM},
        "roi_rc": improved["roi_rc"],
        "reference_swatches": improved["reference_swatches"],
        "target_swatches": improved["target_swatches"],
        "corrected_swatches": improved["target_swatches"],
        "diagnostics": {
            "algorithm": IDENTITY_ALGORITHM,
            "source_shape": list(target.shape),
            "flagged": False,
            "flag_reasons": [],
            "note": "Raw source frame; no radiometric transform was applied.",
        },
    }
    for result in (raw, legacy, improved):
        result["diagnostics"].update(
            {
                "reference_image": str(reference_path),
                "target_image": str(target_path),
            }
        )
    return {
        IDENTITY_ALGORITHM: raw,
        DARSIA_BASELINE_COLOR_ALGORITHM: legacy,
        UNIFORM24_ALGORITHM: improved,
    }


def calculate_identity_preview(
    target_path: Path,
    *,
    preview_maximum: int = 1800,
) -> dict[str, Any]:
    """Preview an explicit no-external-color-target radiometric stage."""

    import darsia

    resolved = Path(target_path).expanduser().resolve()
    target = np.asarray(darsia.imread(resolved).img)
    original = _preview_image(target, preview_maximum)
    return {
        "original": original,
        "corrected": original.copy(),
        "difference": np.zeros_like(original, dtype=np.float32),
        "transform": {"algorithm": IDENTITY_ALGORITHM},
        "roi_rc": None,
        "reference_swatches": np.empty((0, 3), dtype=np.float32),
        "target_swatches": np.empty((0, 3), dtype=np.float32),
        "corrected_swatches": np.empty((0, 3), dtype=np.float32),
        "diagnostics": {
            "algorithm": IDENTITY_ALGORITHM,
            "target_image": str(resolved),
            "flagged": False,
            "flag_reasons": [],
            "note": (
                "No external radiometric target was applied. Registration and spatial "
                "illumination remain separate upstream stages."
            ),
        },
    }
