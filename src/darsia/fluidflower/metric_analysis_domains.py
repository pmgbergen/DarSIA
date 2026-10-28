"""Named analysis boxes in the measured rig frame, independent of any raster.

Coordinates are millimetres, x right and y up from the inner left/bottom
intersection. The nominal legacy 900 by 490 mm grid is only an editor viewport;
it is never used to scale the measured frame or imported coordinates. This
module prepares DATA and display overlays; it starts no scientific work.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import threading
import uuid

import numpy as np

from .analysis_domains import AnalysisDomainStore
from .geometry_facies import GEOMETRY_REVISION_FORMAT, GeometryRevisionStore
from .image_correction import CorrectionRevisionStore
from .map_placement import read_run_geometry

FORMAT = "darsia-measured-analysis-domains-v1"
VIEWPORT_MM = (0.0, 900.0, 0.0, 490.0)
ROLES = ("", "box1", "box2")
_SAVE_LOCK = threading.Lock()


def descriptor(path: Path | str) -> dict[str, Any]:
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size)


def _verify(bound: Mapping[str, Any]) -> Path:
    actual = descriptor(bound["path"])
    if not same_file_binding(actual, bound):
        raise ValueError("The selected geometry or revision file has changed")
    return Path(actual["path"])


def _binding_location(value):
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        return value
    from .storage_paths import resolve_storage_path
    return dict(value, path=os.path.normcase(os.path.abspath(resolve_storage_path(value["path"]))))


def same_file_binding(first, second):
    """Compare exact saved descriptors through explicit storage relocations."""
    return _binding_location(first) == _binding_location(second)


def same_configuration_binding(first, second):
    """Match box authority while preserving its original coordinates and metadata."""
    def located(value):
        return {key: _binding_location(item) if key in {"geometry_index", "geometry_revision"} else item
                for key, item in value.items() if key != "domains"}
    return located(first) == located(second)


def box(name: str, bounds_mm, *, role="", domain_id=None, enabled=True, legacy=None):
    values = list(bounds_mm)
    if (len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) for v in values)
            or values[0] >= values[1] or values[2] >= values[3]):
        raise ValueError("Box bounds must be finite increasing x/y pairs in millimetres")
    if not str(name).strip() or role not in ROLES or type(enabled) is not bool:
        raise ValueError("A named box with an explicit valid mass role is required")
    result = dict(domain_id=str(domain_id or uuid.uuid4().hex), name=str(name).strip(),
                  enabled=enabled, role=role, bounds_mm=[float(v) for v in values])
    if legacy is not None:
        result["legacy_import"] = deepcopy(legacy)
    return result


def ui_domain(value, viewport=VIEWPORT_MM):
    """Adapt physical coordinates to the existing rectangle editor, for display."""
    x0, x1, y0, y1 = value["bounds_mm"]
    left, right, bottom, top = viewport
    return dict(deepcopy(value), physical_bounds_m=dict(x_min=x0/1000, x_max=x1/1000,
                y_min=y0/1000, y_max=y1/1000), normalized_xyxy_top_left=[
                    (x0-left)/(right-left), (top-y1)/(top-bottom),
                    (x1-left)/(right-left), (top-y0)/(top-bottom)])


def from_normalized(values, *, viewport=VIEWPORT_MM, **kwargs):
    x0, y0, x1, y1 = values
    left, right, bottom, top = viewport
    return box(bounds_mm=[left+x0*(right-left), left+x1*(right-left),
                          top-y1*(top-bottom), top-y0*(top-bottom)], **kwargs)


def import_legacy_boxes(values: Mapping[str, Any], *, source: Mapping[str, Any]):
    """Keep the exact original metre-coordinate ROI and its legacy-frame meaning.

    Import is a draft for review. It does not declare the old nominal frame to
    be measured geometry. Export refuses an unacknowledged legacy-frame choice.
    """
    if set(values) != {"box1", "box2"}:
        raise ValueError("The original box1 and box2 roles are both required")
    result = []
    for role in ("box1", "box2"):
        raw = deepcopy(values[role])
        roi = raw.get("roi", raw) if isinstance(raw, dict) else raw
        array = np.asarray(roi, dtype=np.float64)
        if array.shape != (2, 2) or not np.isfinite(array).all():
            raise ValueError("Legacy boxes require two finite x/y corners in metres")
        lo, hi = array.min(axis=0)*1000, array.max(axis=0)*1000
        result.append(box(role, [lo[0], hi[0], lo[1], hi[1]], role=role,
            legacy=dict(source=dict(source), original_value=raw,
                        frame="legacy_nominal_0.9x0.49_m", decision="pending")))
    return result


def load_legacy_registry(path):
    """Use the original ROI reader; no second TOML interpretation."""
    from darsia.presets.workflows.config.roi_registry import RoiRegistry
    bound = descriptor(path)
    rows = RoiRegistry().load(Path(path)).resolve_rois(["box1", "box2"])
    values = {name: np.asarray(row.roi).tolist() for name, row in rows.items()}
    if descriptor(path) != bound:
        raise ValueError("The imported registry changed during reading")
    return import_legacy_boxes(values, source=bound)


def new_configuration(project, geometry_index):
    bound = descriptor(geometry_index)
    models = read_run_geometry(geometry_index)
    for model in models.values():
        frame = model.model.get("metric_frame", {})
        if frame.get("unit") != "mm" or frame.get("x_axis") != "right" or frame.get("y_axis") != "up":
            raise ValueError("The measured models must declare millimetres, x right, y up")
        if float(frame.get("measured_width_mm", 0)) <= 0:
            raise ValueError("Each run needs its measured width")
    if descriptor(geometry_index) != bound:
        raise ValueError("The geometry index changed during reading")
    geometry_store = GeometryRevisionStore(project)
    geometry = geometry_store.current()
    correction = CorrectionRevisionStore(project).current()
    if (geometry.get("format") != GEOMETRY_REVISION_FORMAT or not geometry.get("revision_id")
            or not geometry_store.current_path.is_file()):
        raise ValueError("Save the geometry/facies revision before defining measured boxes")
    geometry_bound = descriptor(geometry_store.current_path)
    if geometry_store.current() != geometry:
        raise ValueError("The saved geometry revision changed during preparation")
    return dict(format=FORMAT, geometry_revision_id=geometry["revision_id"],
        geometry_revision_sha256=geometry_bound["sha256"], geometry_revision=geometry_bound,
        image_correction_revision_id=str(correction.get("revision_id") or ""),
        geometry_index=bound, run_order=list(models), units="mm", x_axis="right", y_axis="up",
        origin="inner left-wall and bottom intersection", viewport_mm=list(VIEWPORT_MM), domains=[])


def use_boxes_in_current_frame(project, configuration, geometry_index):
    """Explicitly retain the boxes in a newly selected compatible measured frame."""
    fresh = new_configuration(project, geometry_index)
    for key in ("format", "units", "x_axis", "y_axis", "origin", "viewport_mm",
                "image_correction_revision_id", "run_order"):
        if configuration.get(key) != fresh[key]:
            raise ValueError("Saved boxes cannot use the current frame: " + key + " differs.")
    fresh["domains"] = deepcopy(configuration["domains"])
    report = validate_configuration(project, fresh)
    if not report["valid"]:
        raise ValueError("Saved boxes cannot use the current frame: "
            + "; ".join(row["message"] for row in report["issues"]))
    return fresh


def project_box(value, geometry):
    """Return four camera-grid vertices; no axis-aligned approximation or refit."""
    x0, x1, y0, y1 = value["bounds_mm"]
    points = np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [1., 1., 1., 1.]])
    result = np.asarray(geometry.metric_to_grid) @ points
    if not np.isfinite(result).all() or np.any(np.abs(result[2]) < 1e-12):
        raise ValueError("The box crosses the model's projective horizon")
    return (result[:2] / result[2]).T


def mass_boxes(config):
    """The existing N1 inline box1/box2 ABI, with exactly one mm-to-m conversion.

    A copied legacy ROI stays byte-value identical until edited. Its physical
    interpretation must first have been acknowledged in the new editor.
    """
    result = {}
    for value in config["domains"]:
        role = value["role"]
        if not value["enabled"] or not role:
            continue
        if role in result:
            raise ValueError("Each mass role must be assigned exactly once")
        legacy = value.get("legacy_import")
        if legacy and legacy.get("decision") != "use_coordinates_in_measured_frame":
            raise ValueError("Review imported legacy-frame coordinates before mass use")
        x0, x1, y0, y1 = value["bounds_mm"]
        if legacy and legacy.get("unmodified", True):
            raw = legacy["original_value"]
            roi = raw.get("roi", raw) if isinstance(raw, dict) else raw
            array = np.asarray(roi, dtype=np.float64)
            if array.shape != (2, 2) or not np.isfinite(array).all():
                raise ValueError("Invalid preserved legacy ROI")
            lo, hi = array.min(axis=0)*1000, array.max(axis=0)*1000
            if [x0, x1, y0, y1] != [lo[0], hi[0], lo[1], hi[1]]:
                raise ValueError("The legacy ROI was edited without clearing its preserved value")
            result[role] = deepcopy(legacy["original_value"])
        else:
            result[role] = [[x0/1000, y0/1000], [x1/1000, y1/1000]]
    if set(result) != {"box1", "box2"}:
        raise ValueError("Assign one enabled box to each mass role: box1 and box2")
    return result


def validate_configuration(project, config):
    issues = []
    try:
        if config.get("format") != FORMAT:
            raise ValueError("Unsupported measured-box format")
        expected = new_configuration(project, _verify(config["geometry_index"]))
        for key in ("geometry_revision_id", "geometry_revision_sha256", "geometry_revision",
                    "image_correction_revision_id", "geometry_index", "run_order",
                    "units", "x_axis", "y_axis", "origin", "viewport_mm"):
            matches = (same_file_binding(config.get(key), expected[key])
                if key in {"geometry_revision", "geometry_index"} else config.get(key) == expected[key])
            if not matches:
                raise ValueError("The measured box binding changed: " + key)
        models = read_run_geometry(_verify(config["geometry_index"]))
        ids, names = set(), set()
        for value in config["domains"]:
            canonical = box(value["name"], value["bounds_mm"], role=value["role"],
                enabled=value["enabled"], domain_id=value["domain_id"], legacy=value.get("legacy_import"))
            if canonical != value:
                raise ValueError("A box contains unsupported or inconsistent fields")
            if value["domain_id"] in ids or value["name"].casefold() in names:
                raise ValueError("Box identifiers and names must be unique")
            ids.add(value["domain_id"]); names.add(value["name"].casefold())
            for model in models.values():
                project_box(value, model)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        issues.append(dict(level="error", message=str(exc)))
    return dict(valid=not issues, issues=issues,
                summary=dict(domain_count=len(config.get("domains", []))))


class MetricAnalysisDomainStore(AnalysisDomainStore):
    """The existing append-only writer, in a separate measured-frame namespace."""

    def __init__(self, project):
        super().__init__(project)
        self.override_root /= "measured_frame"
        self.revisions_path = self.override_root / "revisions.jsonl"
        self.current_path = self.override_root / "current.json"
        self.artifact_root /= "measured_frame"
        self.expected_current = self.current_path.read_bytes() if self.current_path.exists() else None

    def validate_candidate(self, config):
        return validate_configuration(self.project, config)

    def save_revision(self, config, *, actor, reason):
        if not self.validate_candidate(config)["valid"]:
            raise ValueError("Analysis-domain revision has blocking errors")
        # Reuse the existing OS byte-lock primitives. The small lock file stays
        # in place; ownership is released by the OS, not by deleting a file.
        from .device_choice import _try_lock, _unlock
        self.override_root.mkdir(parents=True, exist_ok=True)
        with _SAVE_LOCK, (self.override_root / ".save.lock").open("a+b") as handle:
            if not _try_lock(handle):
                raise ValueError("Another measured-box save is in progress")
            try:
                actual = self.current_path.read_bytes() if self.current_path.exists() else None
                if actual != self.expected_current:
                    raise ValueError("Another measured-box revision was saved; reopen before saving")
                result = super().save_revision(config, actor=actor, reason=reason)
                self.expected_current = self.current_path.read_bytes()
                return result
            finally:
                _unlock(handle)


def saved_mass_boxes(revision_file):
    """Resolve an immutable revision into the worker's existing input format."""
    bound = descriptor(revision_file)
    revision = json.loads(Path(bound["path"]).read_bytes())
    from .analysis_domains import _json_digest
    expected = revision.pop("content_sha256")
    if _json_digest(revision) != expected:
        raise ValueError("The saved box revision content digest differs")
    if revision["configuration"].get("format") != FORMAT:
        raise ValueError("Choose a measured-frame box revision")
    _verify(revision["configuration"]["geometry_revision"])
    read_run_geometry(_verify(revision["configuration"]["geometry_index"]))
    boxes = mass_boxes(revision["configuration"])
    if descriptor(revision_file) != bound:
        raise ValueError("The box revision changed during reading")
    return boxes, bound


def metric_preview(photo, geometry, *, labels=None):
    """Rectified display only; saved science rasters and transforms stay unchanged."""
    import cv2
    image = np.asarray(photo)
    if image.shape[:2] != geometry.shape_hw:
        raise ValueError("The photograph is not on the bound camera grid")
    # One display pixel per mm; y down in the preview, y up in physical space.
    mm_to_view = np.array([[1., 0., 0.], [0., -1., 490.], [0., 0., 1.]])
    grid_to_view = mm_to_view @ np.linalg.inv(geometry.metric_to_grid)
    rgb = cv2.warpPerspective(image, grid_to_view, (900, 490), flags=cv2.INTER_AREA)
    if labels is not None:
        values = np.asarray(labels)
        if values.shape != image.shape[:2]:
            raise ValueError("The saved sand map is not on the photograph grid")
        placed = cv2.warpPerspective(values, grid_to_view, (900, 490), flags=cv2.INTER_NEAREST)
        from .map_placement import boundary_overlay
        rgb = boundary_overlay(rgb, placed)
    return np.ascontiguousarray(rgb, dtype=np.float32) / 255.


def load_saved_sandmap_camera_grid(project, run, geometry):
    """Read this run's bound saved labels and display them on its camera grid.

    No map from another run is substituted. Nearest-neighbour resampling is
    solely for the display; the stored label array is never rewritten.
    """
    import cv2
    from . import geometry_workspace_frame as frame
    from .geometry_facies import _load_array
    config = frame.current_configuration(project)
    fields = frame.selected_fields(config, run)
    bound = fields["labels"]
    if config.get("source_mode") == "canonical-from-import":
        from .geometry_facies import _effective_run_geometry_choice
        selected = any(str(row.get("run_id") or "").strip().casefold() == str(run).casefold()
            and _effective_run_geometry_choice(row) == "run_override"
            for row in (config.get("run_geometry") or {}).get("runs") or [])
        if selected:
            from .geometry_raster_sources import registered_run_sources, verify_registered_sources
            current = GeometryRevisionStore(project).current()
            if current.get("configuration") != config:
                raise ValueError("The saved sand-map revision changed while loading")
            registered = registered_run_sources(project.root, current, run)
            verify_registered_sources(registered)
            bound = registered["files"]["labels"]
    if descriptor(bound["path"]) != {key: bound[key] for key in ("path", "sha256", "size_bytes")}:
        raise ValueError("The saved label source differs from its revision binding")
    labels = _load_array(Path(bound["path"]))
    native = geometry.model.get("baseline_native_grid", {})
    if list(labels.shape) != native.get("shape_hw"):
        raise ValueError("Saved labels and the model's native grid differ")
    matrix = np.asarray(native["H_rig_corrected_native_to_baseline_camera_u"], dtype=np.float64)
    return cv2.warpPerspective(labels, matrix,
        (geometry.shape_hw[1], geometry.shape_hw[0]), flags=cv2.INTER_NEAREST)


def box_overlay(photo, geometry, config):
    """Outlines for review/result displays using the same physical box definition."""
    import cv2
    result = np.ascontiguousarray(photo, dtype=np.uint8).copy()
    if result.shape[:2] != geometry.shape_hw or result.ndim != 3 or result.shape[2] != 3:
        raise ValueError("The overlay photograph is not on the bound RGB camera grid")
    for value in config["domains"]:
        if not value["enabled"]:
            continue
        points = project_box(value, geometry)
        if np.max(np.abs(points)) > 1e7:
            raise ValueError("A projected box exceeds the display coordinate range")
        points = np.rint(points).astype(np.int32)
        color = (255, 210, 0) if value["role"] == "box1" else (0, 255, 255)
        cv2.polylines(result, [points], True, color, 2, cv2.LINE_8)
        caption = value["name"] + (" / " + value["role"] if value["role"] else "")
        cv2.putText(result, caption, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    return result
