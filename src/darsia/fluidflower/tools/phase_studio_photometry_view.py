"""Inspect saved photometric variants on one fixed, registered photograph.

This is a read-only display of RGB arrays already produced for concentration
comparisons. It does not repeat phase inference, fit a calibration, or assert
that an alternate concentration correction is valid for production.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from darsia.fluidflower import saved_mass_rendering as backend
from darsia.fluidflower.tools.phase_studio_mass_comparison import matching_frame_index
from darsia.fluidflower.tools.phase_studio_photometry_canvas import SynchronizedImagePair
from darsia.fluidflower.tools.phase_studio_photometry_geometry import (
    load_registered_shape_rig, serialized_rig_fingerprint,
)
from darsia.fluidflower.tools import phase_studio_photometry_preview_cache as preview_cache
from darsia.fluidflower.tools.phase_studio_photometry_policy import variant_status
from darsia.fluidflower.photometry_consistency import evaluate_photometry_stage_contracts


_VARIANT_LABELS = {
    "none_cc_off": "No illumination · ColorChecker off",
    "none_cc_on": "No illumination · ColorChecker on",
    "historical_cc_off": "Historical illumination · ColorChecker off",
    "historical_cc_on": "Historical illumination · ColorChecker on",
    "facies_cc_off": "Facies illumination · ColorChecker off",
    "facies_cc_on": "Facies illumination · ColorChecker on",
}
_RAW_KEY = "geometry_only"
_RAW_LABEL = "Before · geometry only (no light/color correction)"
_IMAGE_CATEGORIES = {
    "frame": "Experiment photograph",
    "baseline": "Pre-injection reference",
    "clearwater": "Clear-water reference (saved mean)",
}


def variant_choices(entries, selected_result, selected_path):
    """Find saved, named correction branches for the selected phase candidate."""
    current_key = selected_result.get("method", {}).get("settings", {}).get("photometry_variant_id")
    if current_key not in _VARIANT_LABELS:
        return []
    selected = selected_result["selected_inputs"]["candidate"]
    eligible = []
    for entry in entries:
        method = entry.get("method") or {}
        key = method.get("photometry_variant_id")
        if (key not in _VARIANT_LABELS or entry.get("dataset") != selected.get("dataset")
                or entry.get("run") != selected.get("run")
                or entry.get("model") != selected.get("candidate_id")):
            continue
        eligible.append(entry)
    # A shared and an individual fit can use the same corrected RGB. Prefer
    # the selected result for its branch, then the newest saved alternative.
    eligible.sort(key=lambda e: (str(e.get("created_utc") or ""), str(e["path"])), reverse=True)
    chosen = {}
    for entry in eligible:
        chosen.setdefault(entry["method"]["photometry_variant_id"], str(entry["path"]))
    if current_key in chosen:
        chosen[current_key] = str(selected_path)
    return [(key, _VARIANT_LABELS[key] +
             (' · excluded for FF_UM (historical only)'
              if variant_status(selected.get('dataset'), {'photometry_variant_id': key}) == 'excluded'
              else ''), chosen[key])
            for key in _VARIANT_LABELS if key in chosen]


def _frame_for_same_source(anchor, anchor_index, other):
    """Require the same phase candidate, registered image, time and phase map."""
    a = anchor.get("selected_inputs", {}).get("candidate", {})
    b = other.get("selected_inputs", {}).get("candidate", {})
    for key in ("dataset", "run", "candidate_id", "params_sha256", "optical_model_sha256"):
        if not a.get(key) or a[key] != b.get(key):
            raise ValueError("Photometry variants use different phase candidates or registration")
    index = matching_frame_index(anchor, anchor_index, other)
    if index is None:
        raise ValueError("Photometry variants do not contain the same photograph and time")
    ar = anchor["frames"][anchor_index].get("source_reference") or {}
    br = other["frames"][index].get("source_reference") or {}
    for key in ("native_manifest_sha256", "photo_sha256", "phase_sha256", "raw_source_sha256"):
        if not ar.get(key) or ar[key] != br.get(key):
            raise ValueError("Photometry variants do not share the exact registered source and phase map: " + key)
    return index


def _photo_rgb(result_path, row):
    rgb = None
    # Local-dilution derivatives keep the same corrected photo in a separate
    # immutable photograph archive rather than duplicating it in each field.
    for name in ("fields", "photograph_arrays"):
        if not row.get(name):
            continue
        path = Path(row[name])
        if not path.is_absolute():
            path = Path(result_path).resolve().parent / path
        with np.load(path, allow_pickle=False) as archive:
            key = "photograph_rgb" if "photograph_rgb" in archive else "rgb" if "rgb" in archive else None
            if key is not None:
                rgb = np.asarray(archive[key], dtype=np.float32)
                break
    if rgb is None:
        raise ValueError("Saved concentration fields do not include corrected RGB")
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Saved corrected photograph is not a finite RGB image")
    if np.any((rgb < -1e-5) | (rgb > 1 + 1e-5)):
        raise ValueError("Saved corrected photograph is outside the display RGB range")
    return np.ascontiguousarray(np.clip(rgb * 255 + .5, 0, 255).astype(np.uint8))


def _display_rgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Reference photograph is not a finite RGB image")
    if np.any((rgb < -1e-5) | (rgb > 1 + 1e-5)):
        raise ValueError("Reference photograph is outside the saved display RGB range")
    return np.ascontiguousarray(np.clip(rgb * 255 + .5, 0, 255).astype(np.uint8))


def _reference_inputs(result):
    optical = result.get("method", {}).get("optical_provenance") or {}
    rows = optical.get("reference_inputs") or []
    if not isinstance(rows, list) or not rows or any(
            not isinstance(row, dict) or not row.get("sha256") for row in rows):
        raise ValueError("Saved result does not identify its clear-water source images")
    return rows


def _reference_identity(anchor, other, category):
    """Reject a visually plausible but scientifically different reference pair."""
    if category == "clearwater":
        a, b = _reference_inputs(anchor), _reference_inputs(other)
        if [row["sha256"] for row in a] != [row["sha256"] for row in b]:
            raise ValueError("Photometry variants use different clear-water reference images")
        return
    if category == "baseline":
        def source(result):
            optical = result.get("method", {}).get("optical_provenance") or {}
            return (optical.get("optical_binding") or {}).get("raw_baseline") or {}
        a, b = source(anchor), source(other)
        if not a.get("source_sha256") or a["source_sha256"] != b.get("source_sha256"):
            raise ValueError("Photometry variants use different pre-injection reference images")


def _reference_rgb(result, category):
    from darsia.fluidflower.saved_mass_rendering import sha
    from darsia.fluidflower.review_storage import resolve_review_path

    if category not in {"baseline", "clearwater"}:
        raise ValueError("Unknown reference image category")
    optical = result.get("method", {}).get("optical_provenance") or {}
    location, expected = optical.get("references"), optical.get("references_sha256")
    if not location or not expected:
        raise ValueError("Saved optical reference arrays are unavailable")
    path = resolve_review_path(Path(location))
    if sha(path) != expected:
        raise ValueError("Saved optical reference array checksum changed")
    with np.load(path, allow_pickle=False) as archive:
        if category not in archive:
            raise ValueError("Saved optical references do not contain " + category)
        return _display_rgb(archive[category])


def _reference_source_record(anchor, anchor_index, category):
    """Identify the registered raw source used to make the saved RGB."""
    if category == "frame":
        row = anchor["frames"][anchor_index]
        source = row.get("raw_photo_source") or {}
        expected = (row.get("source_reference") or {}).get("raw_source_sha256")
        if not source or not expected or source.get("source_sha256") != expected:
            raise ValueError("Saved frame lacks a verified raw-source binding")
        return [source]
    optical = anchor.get("method", {}).get("optical_provenance") or {}
    if category == "baseline":
        source = (optical.get("optical_binding") or {}).get("raw_baseline") or {}
        if not source.get("source_sha256"):
            raise ValueError("Saved pre-injection reference lacks a raw-source binding")
        return [source]
    return _reference_inputs(anchor)


class _PhotometryReadCache:
    """Small, per-dialog cache for immutable source images and their geometry.

    Only one comparison worker accesses this object at a time. The loaded rig
    is a snapshot for the dialog, so every browsed frame uses identical shape
    corrections even when the serialized files are on a slow NAS.
    """

    def __init__(self, *, persistent=False):
        self.persistent = persistent
        self.native_key = None
        self.native = None
        self.rig = None
        self.results = OrderedDict()
        self.corrected = OrderedDict()
        self.raw = OrderedDict()

    @staticmethod
    def _remember(entries, key, value, limit):
        entries[key] = value
        entries.move_to_end(key)
        while len(entries) > limit:
            entries.popitem(last=False)

    def remember_raw(self, key, image):
        self._remember(self.raw, key, image, 12)

    def remember_corrected(self, key, image):
        self._remember(self.corrected, key, image, 12)

    def read_result(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.results:
            self._remember(self.results, key, backend.read_result(path), 12)
        else:
            self.results.move_to_end(key)
        return self.results[key]

    def corrected_rgb(self, result_path, result, index, category):
        from darsia.fluidflower.review_storage import resolve_review_path

        if category == "frame":
            row = result["frames"][index]
            paths = []
            for name in ("fields", "photograph_arrays"):
                if row.get(name):
                    path = Path(row[name])
                    if not path.is_absolute():
                        path = Path(result_path).resolve().parent / path
                    paths.append(path)
        else:
            optical = result.get("method", {}).get("optical_provenance") or {}
            paths = [resolve_review_path(Path(optical["references"]))]
        states = []
        for path in paths:
            stat = path.stat()
            states.append((str(path), stat.st_size, stat.st_mtime_ns))
        key = (str(result_path), index if category == "frame" else None,
               category, tuple(states))
        if key not in self.corrected:
            rgb = (_photo_rgb(result_path, result["frames"][index]) if category == "frame"
                   else _reference_rgb(result, category))
            self.remember_corrected(key, rgb)
        else:
            self.corrected.move_to_end(key)
        return self.corrected[key]


def _registered_geometry_rig(anchor, native, dataset, run, anchor_index):
    """Load the frozen geometry once, without unrelated mass or color fields."""
    return load_registered_shape_rig(anchor, native, dataset, run, anchor_index)


def _preview_identity(anchor, native, dataset, run, category, anchor_index, shape,
                      native_path, expected_manifest, source_states):
    """Name a regenerable display cache using frozen and current source identity."""
    from darsia.fluidflower.review_storage import resolve_review_path
    from darsia.fluidflower.series_paths import config_files
    from darsia.fluidflower.tools.phase_studio_photometry_geometry import _frozen_ffac_rig_path

    contract = native.get("correction_contract") or {}
    if dataset == "ffac":
        rig_path = _frozen_ffac_rig_path(contract)
    elif dataset == "ffum":
        declared = contract.get("rig_path") or contract.get("legacy_rig")
        if not declared:
            raise ValueError("Frozen FF_UM geometry rig is not recorded")
        rig_path = resolve_review_path(Path(declared))
    else:
        raise ValueError("Geometry-only preview supports registered FF_AC and FF_UM results")
    optical = anchor.get("method", {}).get("optical_provenance") or {}
    artifacts = ((optical.get("optical_binding") or {}).get("recorded_rig") or {}).get(
        "selected_optical_artifacts") or []
    return {
        "dataset": dataset,
        "run": run,
        "category": category,
        "frame_index": anchor["frames"][anchor_index]["frame_index"] if category == "frame" else None,
        "display_shape_hw": tuple(shape),
        "native_manifest_path": str(native_path),
        "native_manifest_sha256": expected_manifest,
        "raw_sources": [{"path": path, "sha256": expected}
                        for path, expected, _size, _mtime in source_states],
        "rig_path": str(rig_path),
        "rig_artifacts": [{"path": str(resolve_review_path(Path(row["path"]))),
                           "sha256": row["sha256"]} for row in artifacts],
        "contract_sha256": hashlib.sha256(json.dumps(
            contract, sort_keys=True, default=str).encode()).hexdigest(),
        "config_files": [str(path) for path in config_files(dataset, run)],
    }


def _geometry_only_rgb(anchor, anchor_index, category, shape, cache=None):
    """Decode one byte-verified raw source with the registered geometry only.

    This is display-only and runs in the comparison worker. It never writes a
    concentration or phase-map product and never changes the frozen phase map.
    """
    import darsia
    from darsia.fluidflower.review_storage import resolve_review_path
    from darsia.fluidflower.saved_mass_rendering import read, sha

    selected = anchor["selected_inputs"]
    candidate = selected["candidate"]
    run = str(candidate["run"]).casefold()
    dataset = str(candidate["dataset"]).casefold()
    native_path = resolve_review_path(Path(selected["review_root"]) / "native_frames" / run / "manifest.json")
    expected_manifest = (anchor["frames"][anchor_index].get("source_reference") or {}).get(
        "native_manifest_sha256")
    if not expected_manifest:
        raise ValueError("Registered raw-image manifest identity is missing")
    manifest_stat = native_path.stat()
    native_key = (str(native_path), expected_manifest, manifest_stat.st_size, manifest_stat.st_mtime_ns)
    records = _reference_source_record(anchor, anchor_index, category)
    source_states = []
    for record in records:
        path = resolve_review_path(Path(record.get("source_path") or record.get("path") or ""))
        expected = record.get("source_sha256") or record.get("sha256")
        stat = path.stat()
        source_states.append((str(path), expected, stat.st_size, stat.st_mtime_ns))
    raw_key = (native_key, category, anchor_index if category == "frame" else None,
               tuple(shape), tuple(source_states))
    if cache is not None and cache.rig is not None and hasattr(cache.rig, "_photometry_rig_folder"):
        current_rig = serialized_rig_fingerprint(cache.rig._photometry_rig_folder)
        if current_rig != cache.rig._photometry_rig_fingerprint:
            cache.native_key, cache.native, cache.rig = None, None, None
            cache.raw.clear()
    if cache is not None and raw_key in cache.raw:
        cache.raw.move_to_end(raw_key)
        return cache.raw[raw_key]
    if cache is not None and cache.native_key == native_key and cache.native is not None:
        native = cache.native
        rig = cache.rig
    else:
        if sha(native_path) != expected_manifest:
            raise ValueError("Registered raw-image manifest changed")
        native = read(native_path)
        rig = None
    if str(native.get("run", "")).casefold() != run or str(native.get("dataset", "")).casefold() != dataset:
        raise ValueError("Registered raw-image manifest belongs to another experiment")
    if category == "frame":
        native_row = native["frame_sources"][anchor["frames"][anchor_index]["frame_index"]]
        if records[0]["source_sha256"] != native_row.get("source_sha256"):
            raise ValueError("Saved raw frame differs from the frozen review source")
    elif category == "baseline":
        if records[0]["source_sha256"] != (native.get("reference") or {}).get("source_sha256"):
            raise ValueError("Saved baseline differs from the frozen review source")
    preview_info = None
    if cache is not None and cache.persistent:
        preview_info = _preview_identity(anchor, native, dataset, run, category, anchor_index,
                                         shape, native_path, expected_manifest, source_states)
        preview = preview_cache.load(preview_info, shape)
        if preview is not None:
            cache.native_key, cache.native, cache.rig = native_key, native, rig
            cache.remember_raw(raw_key, preview)
            return preview
    if rig is None:
        rig = _registered_geometry_rig(anchor, native, dataset, run, anchor_index)
        if cache is not None:
            cache.native_key, cache.native, cache.rig = native_key, native, rig
    total = None
    for path_string, expected, _size, _mtime in source_states:
        path = Path(path_string)
        if not expected or sha(path) != expected:
            raise ValueError("Registered raw photograph changed: " + str(path))
        rgb = np.asarray(darsia.imread(path, transformations=rig.shape_corrections,
                          reference_date=rig.reference_date,
                          dtype=np.float32).img, dtype=np.float32)[..., :3]
        if tuple(rgb.shape[:2]) != tuple(native["analysis_image_shape_hw"]):
            raise ValueError("Raw geometry differs from the registered review grid")
        rgb = cv2.resize(np.clip(rgb, 0, 1), tuple(shape[::-1]), interpolation=cv2.INTER_AREA)
        total = rgb / len(records) if total is None else total + rgb / len(records)
        del rgb
    displayed = _display_rgb(total)
    if cache is not None:
        cache.remember_raw(raw_key, displayed)
    if preview_info is not None:
        try:
            preview_cache.store(preview_info, displayed)
        except OSError:
            pass  # A missing local cache never blocks a verified comparison.
    return displayed


def _raw_contract_text(anchor, anchor_index, category):
    row = anchor["frames"][anchor_index]
    source = _reference_source_record(anchor, anchor_index, category)
    optical = anchor.get("method", {}).get("optical_provenance") or {}
    artifacts = ((optical.get("optical_binding") or {}).get("recorded_rig") or {}).get(
        "selected_optical_artifacts") or []
    rig_note = ("serialized rig files SHA-256 verified" if artifacts else
                "serialized rig path/grid matched; historical rig file bytes not pinned")
    identity = [dict(path=str(item.get("source_path") or item.get("path")),
                     sha256=item.get("source_sha256") or item.get("sha256")) for item in source]
    label = (_IMAGE_CATEGORIES[category] if category != "clearwater"
             else f"Clear-water mean of {len(source)} registered image(s)")
    short = (f"{label} · geometry corrected; no illumination or ColorChecker\n"
             f"Raw source SHA {identity[0]['sha256'][:12]}"
             + (f" · {len(identity)} sources averaged" if len(identity) > 1 else "")
             + " · raw SHA verified; " + rig_note)
    full = json.dumps({"raw_source_images": identity,
                       "native_manifest_sha256": (row.get("source_reference") or {}).get("native_manifest_sha256"),
                       "photometry": "none; registered geometry only",
                       "display_source": "byte-verified raw decode; no saved variant substituted",
                       "geometry_verification": rig_note,
                       "fixed_phase_map": True}, indent=2, ensure_ascii=False)
    return "BEFORE LIGHT / COLOR CORRECTION", short, full


def display_comparison(anchor, anchor_index, left, right, category="frame", *, raw_override=None,
                       cache=None):
    """Return two comparable displays from saved branches or verified raw RGB.

    ``left`` and ``right`` are ``(variant_id, saved_result_path)``. The raw
    pseudo-variant always reconstructs the raw image off the UI thread using
    the recorded geometry rig. CC-off branches remain separate saved variants.
    """
    if category not in _IMAGE_CATEGORIES:
        raise ValueError("Unknown comparison image category")
    if left[0] == right[0]:
        raise ValueError("Choose two different correction displays")
    results = []
    for key, path in (left, right):
        if key == _RAW_KEY:
            if path is not None:
                raise ValueError("The before view must decode registered raw pixels, not a saved variant")
            results.append(None)
            continue
        result = cache.read_result(path) if cache is not None else backend.read_result(path)
        if result.get("complete") is not True:
            raise ValueError("A selected photometry result is incomplete")
        settings = result.get("method", {}).get("settings", {})
        expected_key = "none_cc_off" if key == _RAW_KEY else key
        if settings.get("photometry_variant_id") != expected_key:
            raise ValueError("Saved result has a different photometric variant")
        _frame_for_same_source(anchor, anchor_index, result)
        if category != "frame":
            _reference_identity(anchor, result, category)
        results.append(result)
    nonraw = next((result for key, result in zip((left[0], right[0]), results)
                   if key != _RAW_KEY and result is not None), None)
    if nonraw is None:
        raise ValueError("Select at least one saved corrected branch")
    def corrected_rgb(path, result, index):
        if cache is not None:
            return cache.corrected_rgb(path, result, index, category)
        if category == "frame":
            return _photo_rgb(path, result["frames"][index])
        return _reference_rgb(result, category)

    nonraw_path = left[1] if left[0] != _RAW_KEY else right[1]
    nonraw_index = _frame_for_same_source(anchor, anchor_index, nonraw)
    first_rgb = corrected_rgb(nonraw_path, nonraw, nonraw_index)
    shape = first_rgb.shape[:2]
    displays = []
    for (key, path), result in zip((left, right), results):
        if key == _RAW_KEY:
            if raw_override is not None:
                rgb = np.asarray(raw_override)
            elif cache is not None:
                rgb = _geometry_only_rgb(anchor, anchor_index, category, shape, cache=cache)
            else:
                rgb = _geometry_only_rgb(anchor, anchor_index, category, shape)
            metadata = _raw_contract_text(anchor, anchor_index, category)
        else:
            index = _frame_for_same_source(anchor, anchor_index, result)
            rgb = first_rgb if result is nonraw else corrected_rgb(path, result, index)
            metadata = correction_contract_text(result)
            diagnostics = (result["frames"][index].get("photometric_diagnostics") or {}
                           if category == "frame" else {})
            clipped = diagnostics.get("final_adapter_clipped_pixel_fraction")
            if clipped is not None:
                badge, short, full = metadata
                metadata = (badge, short + f"\nClipped display pixels: {float(clipped):.1%}", full)
        if rgb.dtype != np.uint8 or rgb.shape != (*shape, 3):
            raise ValueError("Photometry displays have different analysis grids")
        displays.append((rgb, metadata))
    return displays


def comparison_pair(anchor, anchor_index, left_path, right_path):
    """Load only the two chosen saved image arrays, without mass computation."""
    if Path(left_path).resolve() == Path(right_path).resolve():
        raise ValueError("Choose two different photometric corrections")
    left = backend.read_result(left_path)
    right = backend.read_result(right_path)
    images = []
    for path, result in ((left_path, left), (right_path, right)):
        if result.get("complete") is not True:
            raise ValueError("A selected photometry result is incomplete")
        settings = result.get("method", {}).get("settings", {})
        if settings.get("photometry_variant_id") not in _VARIANT_LABELS:
            raise ValueError("A selected result does not record a photometric variant")
        index = _frame_for_same_source(anchor, anchor_index, result)
        images.append((_photo_rgb(path, result["frames"][index]), result))
    if images[0][0].shape != images[1][0].shape:
        raise ValueError("Photometric variants have different image grids")
    return images


def correction_contract_text(result):
    """Show the two actual recorded contracts, without claiming equivalence."""
    optical = result.get("method", {}).get("optical_provenance") or {}
    current = optical.get("current_correction_contract") or {}
    phase = current.get("registered_phase_input_contract") or optical.get("source_phase_correction_contract") or {}
    variant = current.get("concentration_photometry") or {
        key: result.get("method", {}).get("settings", {}).get(key)
        for key in ("photometry_variant_id", "illumination", "colorchecker_enabled")
    }
    phase_name = str(phase.get("profile") or "not recorded")
    source = str(phase.get("source") or "")
    identity = lambda value: (hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]
                              if value else "unknown")
    checks = evaluate_photometry_stage_contracts(result.get("photometry_stage_contracts"))
    short = (f"Phase input: {phase_name} · display SHA {identity(phase)}" +
             (f" · {source}" if source else "") + "\n"
             f"Concentration: {variant.get('illumination', 'unknown')} illumination; "
             f"ColorChecker {'on' if variant.get('colorchecker_enabled') is True else 'off' if variant.get('colorchecker_enabled') is False else 'unknown'}"
             f" · display SHA {identity(current)}\n"
             f"Internal chains: phase {checks['phase_chain']['status']}, "
             f"concentration {checks['concentration_chain']['status']}; "
             f"cross-chain {checks['cross_chain']['status']} (informational)")
    full = json.dumps({"phase_input_contract": phase, "concentration_correction_contract": current,
                       "photometry_consistency": checks},
                      indent=2, ensure_ascii=False, default=str)
    if result.get("phase_inference_repeated") is False:
        badge = ("FIXED-PHASE ABLATION · internal correction consistency "
                 + checks["fixed_phase_ablation"]["status"].upper())
    else:
        badge = ("END-TO-END PHOTOMETRY · internal correction consistency "
                 + checks["end_to_end_photometry"]["status"].upper())
    return badge, short, full


class _PairLoader(QThread):
    ready = Signal(object, str)
    prepared = Signal(int, int)

    def __init__(self, anchor, index, left, right, category, cache, parent):
        super().__init__(parent)
        self.anchor, self.index = anchor, index
        self.left, self.right, self.category = left, right, category
        self.cache = cache

    def run(self):
        try:
            pairs = display_comparison(self.anchor, self.index, self.left, self.right,
                                       self.category, cache=self.cache)
            if self.isInterruptionRequested():
                return
            payload = []
            for rgb, metadata in pairs:
                height, width = rgb.shape[:2]
                image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
                payload.append((image, metadata))
            self.ready.emit(payload, "")
            # Prepare nearby frames while this one is on screen. A single
            # worker owns the mutable geometry corrections; any navigation
            # interrupts the preparation after its current photograph.
            if self.category == "frame":
                shape = pairs[0][0].shape[:2]
                has_raw = self.left[0] == _RAW_KEY or self.right[0] == _RAW_KEY
                total = len(self.anchor["frames"])
                completed = 1
                if has_raw:
                    self.prepared.emit(completed, total)
                other_indices = sorted(
                    (index for index in range(len(self.anchor["frames"])) if index != self.index),
                    key=lambda index: (abs(index - self.index), index < self.index),
                )
                for index in other_indices:
                    if self.isInterruptionRequested():
                        break
                    try:
                        if has_raw:
                            _geometry_only_rgb(self.anchor, index, "frame", shape, cache=self.cache)
                            completed += 1
                            self.prepared.emit(completed, total)
                        else:
                            display_comparison(self.anchor, index, self.left, self.right,
                                               "frame", cache=self.cache)
                    except Exception:
                        pass  # Optional preparation never hides the current image.
        except Exception as error:
            self.ready.emit(None, str(error))


class PhotometryComparisonDialog(QDialog):
    """On-demand, full-screen comparison of saved RGB on one fixed phase map."""

    def __init__(self, anchor, anchor_path, frame_index, entries, parent=None):
        super().__init__(parent)
        self.anchor, self.anchor_path = anchor, str(anchor_path)
        variants = variant_choices(entries, anchor, anchor_path)
        self.choices = [(_RAW_KEY, _RAW_LABEL, None), *variants]
        self._worker = None
        self._pending = False
        self._close_when_ready = False
        self._request_serial = 0
        self._worker_serial = 0
        self._read_cache = _PhotometryReadCache(persistent=True)
        self._loaded_display = None
        self.setWindowTitle("Photometric corrections · same image and fixed phase map")
        layout = QVBoxLayout(self)
        notice = QLabel("Same photograph and analysis grid, with the phase map fixed. "
                        "Before = geometry only; after = a saved light/ColorChecker correction. "
                        "Choose clear water or an experiment image to inspect how each correction changes its colors.")
        notice.setWordWrap(True)
        notice.setToolTip("The raw source is byte-verified and decoded with the registered geometry rig. "
                          "Concentration references, fitting, and applied images must use the same correction "
                          "within each variant. This comparison does not test phase-model changes; "
                          "full correction contracts and consistency checks are available below.")
        layout.addWidget(notice)
        bar = QHBoxLayout()
        layout.addLayout(bar)
        bar.addWidget(QLabel("Image"))
        self.image_category = QComboBox(self)
        for key, label in _IMAGE_CATEGORIES.items():
            self.image_category.addItem(label, key)
        bar.addWidget(self.image_category, 1)
        self.selectors = []
        self.badges = []
        self.summaries = []
        self.contracts = []
        self.image_pair = SynchronizedImagePair(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        for view in self.image_pair.views:
            splitter.addWidget(view)
        layout.addWidget(splitter, 1)
        captions = QHBoxLayout()
        layout.addLayout(captions)
        for side in ("Left", "Right"):
            selector = QComboBox(self)
            for key, label, path in self.choices:
                selector.addItem(label, (key, path))
            bar.addWidget(QLabel(side))
            bar.addWidget(selector, 1)
            self.selectors.append(selector)
            pane = QWidget(self)
            pane_layout = QVBoxLayout(pane)
            pane_layout.setContentsMargins(0, 0, 0, 0)
            badge = QLabel("Fixed-phase comparison")
            badge.setStyleSheet("color: #ffe08a; font-weight: bold;")
            pane_layout.addWidget(badge)
            summary = QLabel("Loading correction contracts…")
            summary.setWordWrap(True)
            pane_layout.addWidget(summary)
            contracts = QTextEdit(pane)
            contracts.setReadOnly(True)
            contracts.setVisible(False)
            contracts.setMaximumHeight(170)
            pane_layout.addWidget(contracts)
            self.badges.append(badge)
            self.summaries.append(summary)
            self.contracts.append(contracts)
            captions.addWidget(pane, 1)
        swap_button = QPushButton("⇄ Swap left / right")
        swap_button.setToolTip("Exchange the two displayed corrections without changing zoom or image position.")
        swap_button.clicked.connect(self.swap_sides)
        bar.addWidget(swap_button)
        if self.choices:
            self.selectors[0].setCurrentIndex(0)
            selected_key = anchor.get("method", {}).get("settings", {}).get("photometry_variant_id")
            second = next((i for i, (key, _, _) in enumerate(self.choices) if key == selected_key), 1)
            self.selectors[1].setCurrentIndex(second)
        navigation = QHBoxLayout()
        layout.addLayout(navigation)
        self.previous = QPushButton("◀ Previous")
        self.next_image = QPushButton("Next ▶")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, len(anchor["frames"]) - 1)
        self.frame_slider.setValue(frame_index)
        self.frame_label = QLabel()
        self.prepared_label = QLabel()
        navigation.addWidget(self.previous)
        navigation.addWidget(self.frame_slider, 1)
        navigation.addWidget(self.next_image)
        navigation.addWidget(self.frame_label)
        navigation.addWidget(self.prepared_label)
        self.previous.clicked.connect(lambda: self.frame_slider.setValue(max(0, self.frame_slider.value() - 1)))
        self.next_image.clicked.connect(lambda: self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1)))
        self.status = QLabel("")
        layout.addWidget(self.status)
        footer = QHBoxLayout()
        layout.addLayout(footer)
        fit_button = QPushButton("Fit both")
        fit_button.clicked.connect(self.image_pair.fit)
        footer.addWidget(fit_button)
        actual_button = QPushButton("1:1 pixels")
        actual_button.clicked.connect(self.image_pair.actual_size)
        footer.addWidget(actual_button)
        contracts_button = QPushButton("Show full correction contracts")
        contracts_button.setCheckable(True)
        contracts_button.toggled.connect(lambda checked: [w.setVisible(checked) for w in self.contracts])
        footer.addWidget(contracts_button)
        footer.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        footer.addWidget(close)
        for selector in self.selectors:
            selector.currentIndexChanged.connect(self._request)
        self.image_category.currentIndexChanged.connect(self._request)
        self.frame_slider.valueChanged.connect(self._request)
        self._request()

    def _request(self, *_):
        self._request_serial += 1
        index = self.frame_slider.value()
        row = self.anchor["frames"][index]
        category = self.image_category.currentData()
        is_frame = category == "frame"
        self.frame_slider.setEnabled(is_frame and self.frame_slider.maximum() > 0)
        self.previous.setEnabled(is_frame and index > 0)
        self.next_image.setEnabled(is_frame and index < self.frame_slider.maximum())
        self.frame_label.setText(
            f"{index + 1}/{len(self.anchor['frames'])} · {row.get('source_name', '')} · {float(row['elapsed_h']):.3f} h"
            if is_frame else _IMAGE_CATEGORIES[category])
        if len(self.choices) < 2 or any(selector.currentData() is None for selector in self.selectors):
            self._pending = False
            self._clear_photos()
            self.status.setText("A saved photometric variant is required for this phase model and run.")
            return
        if self.selectors[0].currentData()[0] == self.selectors[1].currentData()[0]:
            self._pending = False
            self._clear_photos()
            self.status.setText("Choose 'Before' and a saved correction, or two different saved corrections.")
            return
        self._pending = True
        self.prepared_label.clear()
        if self._worker is None:
            self._start_pending()
        else:
            self._worker.requestInterruption()

    def _clear_photos(self):
        self.image_pair.clear()
        self._loaded_display = None
        for badge, summary, contract in zip(
                self.badges, self.summaries, self.contracts):
            badge.clear()
            summary.clear()
            contract.clear()

    def swap_sides(self):
        """Exchange the current pair instantly while preserving shared zoom/pan."""
        left, right = (selector.currentIndex() for selector in self.selectors)
        for selector in self.selectors:
            selector.blockSignals(True)
        self.selectors[0].setCurrentIndex(right)
        self.selectors[1].setCurrentIndex(left)
        for selector in self.selectors:
            selector.blockSignals(False)
        category, index = self.image_category.currentData(), self.frame_slider.value()
        displayed = self._loaded_display
        if (displayed is not None and displayed[:2] == (category, index)
                and displayed[2:] == (self.selectors[1].currentData(), self.selectors[0].currentData())
                and self.image_pair.has_images and not self._pending):
            self.image_pair.swap_images()
            for widgets in (self.badges, self.summaries, self.contracts):
                if isinstance(widgets[0], QTextEdit):
                    contents = [widget.toPlainText() for widget in widgets]
                    for widget, value in zip(widgets, reversed(contents)):
                        widget.setPlainText(value)
                else:
                    contents = [widget.text() for widget in widgets]
                    for widget, value in zip(widgets, reversed(contents)):
                        widget.setText(value)
            self._loaded_display = (category, index,
                                    self.selectors[0].currentData(), self.selectors[1].currentData())
            return
        self._request()

    def _start_pending(self):
        if not self._pending or self._close_when_ready:
            return
        self._pending = False
        self._worker_serial = self._request_serial
        category = self.image_category.currentData()
        index = self.frame_slider.value()
        self.status.setText("Loading verified sources and saved correction arrays…")
        self._worker = _PairLoader(self.anchor, index,
                                   self.selectors[0].currentData(),
                                   self.selectors[1].currentData(),
                                   category, self._read_cache, self)
        self._worker.ready.connect(self._loaded)
        self._worker.prepared.connect(self._prepared)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _loaded(self, payload, error):
        if self._worker_serial != self._request_serial or self._pending or self._close_when_ready:
            return
        if error:
            self._clear_photos()
            self.status.setText("Comparison unavailable: " + error)
            return
        images = payload
        self.image_pair.set_images(images[0][0], images[1][0])
        for position, (_image, (badge, short, full)) in enumerate(images):
            self.badges[position].setText(badge)
            self.summaries[position].setText(short)
            self.contracts[position].setPlainText(full)
        self._loaded_display = (self.image_category.currentData(), self.frame_slider.value(),
                                self.selectors[0].currentData(), self.selectors[1].currentData())
        self.status.setText("Same registered source and analysis grid; both displays use the same 0–1 RGB scale. "
                            "Visual inspection does not validate the mass fit.")

    def _prepared(self, completed, total):
        if self._worker_serial != self._request_serial or self._pending or self._close_when_ready:
            return
        self.prepared_label.setText(
            "All raw previews ready" if completed == total else
            f"Preparing raw previews {completed}/{total}…"
        )

    def _finished(self):
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        if (not self._pending and self._worker_serial == self._request_serial
                and self.prepared_label.text().startswith("Preparing raw previews")):
            self.prepared_label.setText(self.prepared_label.text().replace(
                "Preparing raw previews", "Raw previews ready", 1).rstrip("…"))
        if self._close_when_ready:
            self.close()
        else:
            self._start_pending()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._close_when_ready = True
            self._worker.requestInterruption()
            self.status.setText("Closing when the saved image read finishes…")
            event.ignore()
            return
        super().closeEvent(event)
