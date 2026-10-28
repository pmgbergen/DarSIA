"""Versioned physical analysis domains on the corrected FluidFlower grid."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from darsia.fluidflower.geometry_facies import (
    GEOMETRY_REVISION_FORMAT,
    GeometryRevisionStore,
    load_registered_geometry_image,
)
from darsia.fluidflower.image_correction import (
    CORRECTION_REVISION_FORMAT,
    DARSIA_BASELINE_COLOR_ALGORITHM,
    IDENTITY_ALGORITHM,
    UNIFORM24_ALGORITHM,
    CorrectionRevisionStore,
    calculate_darsia_baseline_array_preview,
    calculate_uniform24_array_preview,
)
from darsia.fluidflower.project import utc_now

ANALYSIS_DOMAIN_CONFIG_FORMAT = "darsia-fluidflower-analysis-domains-v1"
ANALYSIS_DOMAIN_REVISION_FORMAT = "darsia-fluidflower-analysis-domain-revision-v1"


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
                    f"Invalid analysis-domain revision at line {line_number}"
                )
            rows.append(payload)
    return rows


def _json_digest(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _number(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    """Return the first non-empty value while retaining numerical zero."""

    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


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


def _preview(image: np.ndarray, maximum: int) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[:2]
    scale = min(1.0, float(maximum) / float(max(height, width)))
    if scale >= 1.0:
        return np.clip(array, 0.0, 1.0)
    return cv2.resize(
        np.clip(array, 0.0, 1.0),
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _source_path(
    geometry: Mapping[str, Any],
    correction: Mapping[str, Any],
    requested: str,
) -> Path:
    candidates = (
        requested,
        correction.get("target_image"),
        geometry.get("display_image_path"),
        correction.get("reference_image"),
        geometry.get("baseline_path"),
    )
    for value in candidates:
        text = str(value or "").strip()
        if text and Path(text).expanduser().is_file():
            return Path(text).expanduser().resolve()
    raise FileNotFoundError(
        "No registered analysis-domain reference image is available"
    )


def load_analysis_domain_reference(
    project: FluidFlowerProject,
    *,
    source_path: str = "",
    preview_maximum: int = 2400,
) -> dict[str, Any]:
    """Load one corrected, cropped, and unwrapped image for domain editing."""

    geometry_store = GeometryRevisionStore(project)
    geometry_revision = geometry_store.current()
    geometry = geometry_store.current_config()
    if geometry_revision.get("format") != GEOMETRY_REVISION_FORMAT:
        raise ValueError("Save a geometry revision before editing analysis domains")
    correction = CorrectionRevisionStore(project).current()
    selected = _source_path(geometry, correction, source_path)
    registered, registration = load_registered_geometry_image(geometry, selected)
    corrected = registered
    uniform24: dict[str, Any] = {
        "applied": False,
        "status": "no_saved_uniform24_revision",
        "revision_id": "",
    }

    if correction.get("format") == CORRECTION_REVISION_FORMAT:
        algorithm = str(correction.get("algorithm") or IDENTITY_ALGORITHM)
        if algorithm in {UNIFORM24_ALGORITHM, DARSIA_BASELINE_COLOR_ALGORITHM}:
            reference_path = Path(
                str(correction.get("reference_image") or "")
            ).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(
                    "The saved ColorChecker reference is unavailable: "
                    f"{reference_path}"
                )
            reference, reference_registration = load_registered_geometry_image(
                geometry, reference_path
            )
            preview = (
                calculate_uniform24_array_preview
                if algorithm == UNIFORM24_ALGORITHM
                else calculate_darsia_baseline_array_preview
            )
            result = preview(
                reference,
                registered,
                corner=str(correction.get("colorchecker_corner") or "upper_right"),
                roi_rc=None,
                settings=dict(correction.get("settings") or {}),
                preview_maximum=preview_maximum,
            )
            corrected = np.asarray(result["corrected"], dtype=np.float32)
            uniform24 = {
                "applied": True,
                "status": "registered_colorchecker_correction",
                "algorithm": algorithm,
                "revision_id": str(correction.get("revision_id") or ""),
                "reference_path": str(reference_path.resolve()),
                "reference_registration": reference_registration,
                "roi_rc_on_registered_grid": deepcopy(result["roi_rc"]),
                "diagnostics": deepcopy(result["diagnostics"]),
            }
        elif algorithm == IDENTITY_ALGORITHM:
            corrected = _preview(corrected, preview_maximum)
            uniform24 = {
                "applied": False,
                "status": "identity_control",
                "algorithm": algorithm,
                "revision_id": str(correction.get("revision_id") or ""),
            }
        else:
            raise ValueError(f"Unsupported image-correction algorithm: {algorithm!r}")
    else:
        corrected = _preview(corrected, preview_maximum)

    source_shape = list(registered.shape[:2])
    preview_shape = list(corrected.shape[:2])
    return {
        "image": np.clip(corrected, 0.0, 1.0),
        "provenance": {
            "schema": 1,
            "coordinate_system": "corrected_registered_rig_grid",
            "coordinate_convention": "origin_lower_left_x_right_y_up",
            "source_path": str(selected),
            "source_shape_rc": source_shape,
            "preview_shape_rc": preview_shape,
            "width_m": float(geometry.get("width_m") or 0.0),
            "height_m": float(geometry.get("height_m") or 0.0),
            "geometry_revision_id": str(geometry_revision.get("revision_id") or ""),
            "geometry_revision_sha256": str(
                geometry_revision.get("content_sha256") or ""
            ),
            "registration": registration,
            "uniform24": uniform24,
            "fully_corrected": bool(uniform24["applied"]),
        },
    }


def default_analysis_domain_config(project: FluidFlowerProject) -> dict[str, Any]:
    geometry_revision = GeometryRevisionStore(project).current()
    correction = CorrectionRevisionStore(project).current()
    geometry = GeometryRevisionStore(project).current_config()
    return {
        "schema": 1,
        "format": ANALYSIS_DOMAIN_CONFIG_FORMAT,
        "geometry_revision_id": str(geometry_revision.get("revision_id") or ""),
        "geometry_revision_sha256": str(geometry_revision.get("content_sha256") or ""),
        "image_correction_revision_id": str(correction.get("revision_id") or ""),
        "reference_image_path": "",
        "reference_provenance": {},
        "width_m": float(geometry.get("width_m") or 0.0),
        "height_m": float(geometry.get("height_m") or 0.0),
        "source_shape_rc": [],
        "domains": [],
    }


def domain_from_normalized_rectangle(
    *,
    name: str,
    normalized_xyxy_top_left: list[float],
    source_shape_rc: list[int],
    width_m: float,
    height_m: float,
    domain_id: str | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Create one physical rectangle from top-left normalized image bounds."""

    values = [float(value) for value in normalized_xyxy_top_left]
    if len(values) != 4:
        raise ValueError("A rectangle requires x0, y0, x1, y1")
    x0, x1 = sorted((values[0], values[2]))
    y0, y1 = sorted((values[1], values[3]))
    if x0 < 0.0 or y0 < 0.0 or x1 > 1.0 or y1 > 1.0 or x1 <= x0 or y1 <= y0:
        raise ValueError("Normalized rectangle must be non-empty and inside the image")
    rows, columns = [int(value) for value in source_shape_rc]
    return {
        "domain_id": str(domain_id or uuid.uuid4().hex),
        "name": str(name).strip(),
        "kind": "rectangle",
        "enabled": bool(enabled),
        "coordinate_system": "corrected_registered_rig_grid",
        "normalized_xyxy_top_left": [x0, y0, x1, y1],
        "pixel_bounds_rc": [
            round(y0 * rows),
            round(x0 * columns),
            round(y1 * rows),
            round(x1 * columns),
        ],
        "physical_bounds_m": {
            "x_min": x0 * float(width_m),
            "x_max": x1 * float(width_m),
            "y_min": (1.0 - y1) * float(height_m),
            "y_max": (1.0 - y0) * float(height_m),
        },
    }


def canonicalize_analysis_domain(
    raw: Mapping[str, Any],
    *,
    source_shape_rc: list[int],
    width_m: float,
    height_m: float,
) -> dict[str, Any]:
    """Normalize imported physical, pixel, or normalized rectangle coordinates."""

    row = dict(raw)
    normalized = row.get("normalized_xyxy_top_left")
    if isinstance(normalized, str):
        normalized = [part.strip() for part in normalized.split(",")]
    if not isinstance(normalized, (list, tuple)) or len(normalized) != 4:
        physical = row.get("physical_bounds_m")
        if not isinstance(physical, Mapping):
            physical = row
        x_min = _number(_first_present(physical, "x_min", "x_min_m"))
        x_max = _number(_first_present(physical, "x_max", "x_max_m"))
        y_min = _number(_first_present(physical, "y_min", "y_min_m"))
        y_max = _number(_first_present(physical, "y_max", "y_max_m"))
        if None not in {x_min, x_max, y_min, y_max}:
            normalized = [
                float(x_min) / width_m,
                1.0 - float(y_max) / height_m,
                float(x_max) / width_m,
                1.0 - float(y_min) / height_m,
            ]
        else:
            columns = int(source_shape_rc[1])
            rows = int(source_shape_rc[0])
            left = _number(row.get("left_px"))
            top = _number(row.get("top_px"))
            right = _number(row.get("right_px"))
            bottom = _number(row.get("bottom_px"))
            if None in {left, top, right, bottom}:
                raise ValueError(
                    "Imported domains need physical, corrected-grid pixel, or "
                    "normalized rectangle coordinates"
                )
            normalized = [
                float(left) / columns,
                float(top) / rows,
                float(right) / columns,
                float(bottom) / rows,
            ]
    return domain_from_normalized_rectangle(
        name=str(row.get("name") or row.get("domain_name") or "Analysis domain"),
        normalized_xyxy_top_left=[float(value) for value in normalized],
        source_shape_rc=source_shape_rc,
        width_m=width_m,
        height_m=height_m,
        domain_id=str(row.get("domain_id") or uuid.uuid4().hex),
        enabled=str(row.get("enabled", "true")).strip().casefold()
        not in {"0", "false", "no", "off"},
    )


def import_analysis_domains(
    path: Path,
    *,
    source_shape_rc: list[int],
    width_m: float,
    height_m: float,
) -> list[dict[str, Any]]:
    """Import rectangle coordinates from JSON, CSV, or TSV."""

    selected = Path(path).expanduser().resolve()
    suffix = selected.suffix.casefold()
    if suffix == ".json":
        payload = json.loads(selected.read_text(encoding="utf-8-sig"))
        rows = payload.get("domains") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON must contain a domain list or a domains field")
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with selected.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter=delimiter))
    else:
        raise ValueError("Analysis-domain import supports JSON, CSV, or TSV")
    return [
        canonicalize_analysis_domain(
            row,
            source_shape_rc=source_shape_rc,
            width_m=width_m,
            height_m=height_m,
        )
        for row in rows
        if isinstance(row, Mapping)
    ]


def validate_analysis_domain_config(
    project: FluidFlowerProject,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = deepcopy(dict(config))
    issues: list[dict[str, Any]] = []
    if candidate.get("format") != ANALYSIS_DOMAIN_CONFIG_FORMAT:
        _issue(issues, "error", "unsupported_format", "Unsupported domain format")

    geometry = GeometryRevisionStore(project).current()
    if geometry.get("format") != GEOMETRY_REVISION_FORMAT:
        _issue(
            issues,
            "error",
            "geometry_revision_missing",
            "A saved geometry revision is required.",
        )
    elif str(candidate.get("geometry_revision_id") or "") != str(
        geometry.get("revision_id") or ""
    ) or str(candidate.get("geometry_revision_sha256") or "") != str(
        geometry.get("content_sha256") or ""
    ):
        _issue(
            issues,
            "error",
            "geometry_revision_changed",
            "Analysis domains do not reference the current corrected geometry grid.",
        )

    correction = CorrectionRevisionStore(project).current()
    expected_correction = str(correction.get("revision_id") or "")
    if str(candidate.get("image_correction_revision_id") or "") != expected_correction:
        _issue(
            issues,
            "error",
            "image_correction_changed",
            "Analysis domains do not reference the current image correction.",
        )
    provenance = dict(candidate.get("reference_provenance") or {})
    if provenance.get("coordinate_system") != "corrected_registered_rig_grid":
        _issue(
            issues,
            "error",
            "reference_grid_invalid",
            "Domain reference must use the corrected and registered rig grid.",
        )
    correction_algorithm = str(correction.get("algorithm") or IDENTITY_ALGORITHM)
    if (
        expected_correction
        and correction_algorithm != IDENTITY_ALGORITHM
        and provenance.get("fully_corrected") is not True
    ):
        _issue(
            issues,
            "error",
            "radiometric_correction_not_applied",
            "The selected radiometric correction was not applied to the domain image.",
        )

    width = _number(candidate.get("width_m"))
    height = _number(candidate.get("height_m"))
    shape = candidate.get("source_shape_rc") or []
    if width is None or width <= 0.0 or height is None or height <= 0.0:
        _issue(
            issues,
            "error",
            "physical_dimensions_invalid",
            "Positive physical rig dimensions are required.",
        )
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(value, int) or value <= 0 for value in shape)
    ):
        _issue(
            issues,
            "error",
            "source_shape_invalid",
            "Corrected source image shape is required.",
        )
    geometry_config = GeometryRevisionStore(project).current_config()
    geometry_width = _number(geometry_config.get("width_m"))
    geometry_height = _number(geometry_config.get("height_m"))
    expected_shape = list(
        ((geometry.get("validation") or {}).get("summary") or {}).get("spatial_shape")
        or []
    )
    if (
        width is not None
        and height is not None
        and geometry_width is not None
        and geometry_height is not None
        and (
            not math.isclose(width, geometry_width, rel_tol=0.0, abs_tol=1.0e-12)
            or not math.isclose(height, geometry_height, rel_tol=0.0, abs_tol=1.0e-12)
        )
    ):
        _issue(
            issues,
            "error",
            "physical_dimensions_changed",
            "Analysis-domain dimensions differ from the current rig geometry.",
        )
    if expected_shape and list(shape) != expected_shape[:2]:
        _issue(
            issues,
            "error",
            "geometry_grid_shape_changed",
            "Analysis-domain pixels differ from the current corrected geometry grid.",
        )
    if list(provenance.get("source_shape_rc") or []) != list(shape):
        _issue(
            issues,
            "error",
            "reference_shape_changed",
            "Reference-image provenance and stored domain grid have different shapes.",
        )
    if str(provenance.get("geometry_revision_id") or "") != str(
        candidate.get("geometry_revision_id") or ""
    ) or str(provenance.get("geometry_revision_sha256") or "") != str(
        candidate.get("geometry_revision_sha256") or ""
    ):
        _issue(
            issues,
            "error",
            "reference_geometry_changed",
            "Reference-image provenance belongs to another geometry revision.",
        )
    uniform24 = dict(provenance.get("uniform24") or {})
    if (
        expected_correction
        and str(uniform24.get("revision_id") or "") != expected_correction
    ):
        _issue(
            issues,
            "error",
            "reference_correction_changed",
            "Reference-image provenance belongs to another image-correction revision.",
        )

    domains = candidate.get("domains") or []
    if not isinstance(domains, list):
        _issue(issues, "error", "domains_invalid", "Domains must be a list")
        domains = []
    ids: list[str] = []
    names: list[str] = []
    for domain in domains:
        if not isinstance(domain, dict):
            _issue(issues, "error", "domain_invalid", "Every domain must be an object")
            continue
        domain_id = str(domain.get("domain_id") or "").strip()
        name = str(domain.get("name") or "").strip()
        ids.append(domain_id)
        names.append(name.casefold())
        if not domain_id or not name or domain.get("kind") != "rectangle":
            _issue(
                issues,
                "error",
                "domain_identity_invalid",
                "Every rectangle needs a stable identifier and name.",
            )
        bounds = domain.get("normalized_xyxy_top_left") or []
        numbers = [_number(value) for value in bounds]
        if (
            len(numbers) != 4
            or any(value is None for value in numbers)
            or not (
                0.0 <= float(numbers[0]) < float(numbers[2]) <= 1.0
                and 0.0 <= float(numbers[1]) < float(numbers[3]) <= 1.0
            )
        ):
            _issue(
                issues,
                "error",
                "domain_bounds_invalid",
                f"Domain {name or domain_id or '?'} is outside the corrected image.",
            )
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        _issue(
            issues,
            "error",
            "duplicate_domains",
            "Domain identifiers and names must be unique.",
        )

    errors = sum(issue["severity"] == "error" for issue in issues)
    return {
        "schema": 1,
        "config_sha256": _json_digest(candidate),
        "valid": errors == 0,
        "summary": {
            "error_count": errors,
            "warning_count": sum(issue["severity"] == "warning" for issue in issues),
            "domain_count": len(domains),
            "enabled_domain_count": sum(
                bool(domain.get("enabled", True))
                for domain in domains
                if isinstance(domain, dict)
            ),
        },
        "issues": issues,
    }


class AnalysisDomainStore:
    """Append-only physical-domain revisions on the canonical corrected grid."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.override_root = project.root / "overrides" / "analysis_domains"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        self.artifact_root = project.root / "results" / "analysis_domains"

    def revisions(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.revisions_path)

    def current(self) -> dict[str, Any]:
        if self.current_path.is_file():
            payload = json.loads(self.current_path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                return payload
        rows = self.revisions()
        return deepcopy(rows[-1]) if rows else {}

    def save_revision(
        self,
        config: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        candidate = deepcopy(dict(config))
        report = self.validate_candidate(candidate)
        if not report["valid"]:
            raise ValueError("Analysis-domain revision has blocking errors")
        revision_id = uuid.uuid4().hex
        revision = {
            "schema": 1,
            "format": ANALYSIS_DOMAIN_REVISION_FORMAT,
            "revision_id": revision_id,
            "recorded_utc": utc_now(),
            "recorded_by": str(actor).strip(),
            "reason": str(reason).strip(),
            "configuration": candidate,
            "validation": report,
        }
        revision["content_sha256"] = _json_digest(revision)
        destination = self.artifact_root / revision_id
        destination.mkdir(parents=True, exist_ok=False)
        _atomic_json(destination / "revision.json", revision)
        _append_jsonl(self.revisions_path, revision)
        _atomic_json(self.current_path, revision)
        self.project._audit(
            "analysis_domains_saved",
            actor,
            {
                "revision_id": revision_id,
                "domain_count": report["summary"]["domain_count"],
                "geometry_revision_id": candidate["geometry_revision_id"],
                "image_correction_revision_id": candidate[
                    "image_correction_revision_id"
                ],
            },
        )
        return deepcopy(revision)

    def validate_candidate(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Validate the original corrected-grid format by default.

        Editors for another explicit coordinate frame can reuse the append-only
        revision writer with a separate namespace and a strict frame validator.
        """

        return validate_analysis_domain_config(self.project, config)


if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject
