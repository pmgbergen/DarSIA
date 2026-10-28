"""Read-only comparison of saved Studio phase maps from one or two experiments.

No inference, annotation migration, calibration or model activation occurs here.
Each pane uses its own saved corrected photograph and final phase overlay.
"""
from __future__ import annotations

from darsia.fluidflower.series_native_context import saved_reader_ui_entrypoint

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from darsia.fluidflower.review_storage import resolve_review_path
from threading import Thread
from contextvars import copy_context
from types import SimpleNamespace

from PySide6.QtCore import QEvent, QPointF, Qt, Signal, QSettings, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSplitter, QVBoxLayout, QInputDialog, QTabWidget, QWidget,
)

from darsia.fluidflower.tools.phase_studio_photometry_canvas import SynchronizedImagePair
from darsia.fluidflower.tools.phase_view_overlay import display_phase_overlay, overlay_asset_layer
from darsia.fluidflower.model_display import (
    created_utc_text, friendly_model_name, recorded_photometry_provenance,
)
from darsia.fluidflower.tools.phase_review_area_sort import (
    peak_phase_area_percent, sorted_phase_area_runs,
)
from darsia.fluidflower.raw_photographic_review import RAW_REVIEW_FORMAT
from darsia.fluidflower.phase_photographic_assets import V5_SOURCE_FORMAT, raw_detail_available, resolve_raw_detail, BASE_DISPLAY_MODES, DISPLAY_MODES, change_display_available, display_modes_available, resolve_display_mode, resolve_photo_sand_domain, original_display_available, load_original_display_rgb
from darsia.fluidflower.phase_rig_raw_display import (
    rig_raw_display_available, load_rig_raw_display_image,
)


PANEL_PHOTO_MODES = (
    ("Original", "original"), ("Natural", "natural"),
    ("Lift ×4.3", "lift1"), ("Lift ×7.1", "lift2"),
    ("Change (flat field)", "change"), ("RAW detail", "raw_detail"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def model_title(entry):
    """Show a provenance-backed name when available, leaving identity in details."""
    root = str(entry["root"])
    model_id = str(entry.get("model_id") or Path(root).name)
    created = str(entry.get("created_utc") or "").strip()
    if friendly := friendly_model_name(entry):
        try:
            date = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone().strftime("%d.%m.%Y")
        except ValueError:
            date = "date unknown"
        return f"{friendly} · {model_id} · {date}"
    return f"{model_id} · Created {created or 'time not recorded'}"


def model_details(entry):
    """Exact, unshortened identity for a selector tooltip and review caption."""
    root = str(entry["root"])
    return (f"{model_title(entry)}\n"
            f"Model ID: {entry.get('model_id') or Path(root).name}\n"
            f"Created UTC: {created_utc_text(entry)}\n"
            f"Folder: {root}")


def model_catalog(root):
    """Read saved maps under their recorded binding, without rerunning inference."""
    from darsia.fluidflower.tools.phase_studio_mass_inputs import _prepared_jobs
    from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity

    model_id, dataset, jobs = _prepared_jobs(root, saved_comparison=True)
    root = resolve_review_path(root)
    summary_path = root / "summary.json"
    request_path = root / "training_request.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    request = read_json(request_path) if request_path.is_file() else {}
    created_utc = str(summary.get("created_utc") or request.get("created_utc") or "")
    photometry_arm, photometry_stage = recorded_photometry_provenance(root, request, summary)
    binding_path = resolve_review_path(root / "validation") / "validation_binding.json"
    binding = read_json(binding_path) if binding_path.is_file() else {}
    recorded_physics = binding.get("physics_runtime")
    profile = load_physics_profile(root)
    current_physics = physics_runtime_identity(profile) if profile is not None else None
    receipt_path = Path(root) / "publication_receipt.json"
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    from darsia.fluidflower.phase_saved_registration import registration_warning
    from darsia.fluidflower.phase_model_history import registered_old_domain_seam_review, registered_combined_l1_review, registered_t3_review, registered_l1_neighbourhood_review, registered_p1_final_review
    track1_warning = registration_warning(root)
    if registered_p1_final_review(root, track1_warning):
        label = "Manual validation only · P1 final · No floor fill · Top-surface rule · not for mass analysis"
    elif registered_l1_neighbourhood_review(root, track1_warning):
        label = "Manual validation only · G3 gas · L1 neighbourhood rule · Floor contact · not for mass analysis"
    elif registered_t3_review(root, track1_warning):
        label = "Manual validation only · G3-T3 gas · Floor contact · not for mass analysis"
    elif registered_old_domain_seam_review(root, track1_warning):
        label = "Manual validation only · Old domain seam corrected · not for mass analysis"
    elif registered_combined_l1_review(root, track1_warning):
        label = "Manual validation only · T2 at L1 · Floor contact · not for mass analysis"
    elif track1_warning.startswith("Studio review candidate: approved G3"):
        top_revision = ((summary.get("approved_g3_production") or {}).get("domain_revision") or {})
        barrier_rule = ((summary.get("approved_g3_production") or {}).get("barrier_rule") or {})
        label = ("Manual validation only · Barrier rule B · not for mass analysis"
                 if barrier_rule.get("format") == "ffac-gas-barrier-rule-b-v1"
                 else "Manual validation only · Top-sand boundary corrected · not for mass analysis"
                 if top_revision.get("format") == "ffac-photo-sand-top-revision-v1"
                 else "Manual validation only · G3 production parity passed · not for mass analysis")
    elif track1_warning:
        label = "Manual validation only · QA failed/incomplete · not for mass analysis"
    elif receipt.get("publication_scope") == "studio_validation_only":
        label = "Manual validation only"
        if receipt.get("qa_status") == "fail":
            label += " · QA failed"
    elif (summary.get("evaluation") or {}).get("kind") == "training-only":
        label = "Unpublished review candidate · not approved for scientific use"
    else:
        label = "Saved model · review status unchanged"
    if recorded_physics != current_physics:
        label += " · Saved maps use an earlier runtime; inspection only"
    return dict(model_id=model_id, dataset=dataset, jobs={job["run"]: job for job in jobs},
                root=str(root), status=label, created_utc=created_utc,
                photometry_arm=photometry_arm, photometry_stage=photometry_stage,
                approved_g3_production=summary.get("approved_g3_production"),
                display_name=summary.get("display_name"),
                recorded_physics=recorded_physics, current_physics=current_physics)


def model_run_areas(catalog):
    """Read model-bound saved area reports; missing reports remain unknown."""
    areas = {}
    for run, job in catalog["jobs"].items():
        candidate = job.get("candidate") if isinstance(job, dict) else None
        areas[run] = (peak_phase_area_percent(SimpleNamespace(**candidate))
                      if isinstance(candidate, dict) else None)
    return areas


def runtime_comparison_note(catalogs):
    """Visible qualification for read-only comparisons of historical maps."""
    notes = []
    if any(row.get("recorded_physics") != row.get("current_physics")
           for row in catalogs if "recorded_physics" in row):
        notes.append("Saved maps were produced with a different runtime version")
    identities = {row.get("recorded_physics") for row in catalogs if row.get("recorded_physics")}
    if len(identities) > 1:
        notes.append("The models use different recorded physical rules")
    return " · ".join(notes)


def _metric_rig_transform(row):
    """One frame's recorded native-to-metric rig homography and canvas origin, or None."""
    import numpy as np
    revision = (row or {}).get("rig_domain_revision") or {}
    parameters = revision.get("parameters") or {}
    try:
        matrix = np.asarray(parameters["H_frame_native_to_metric_pixel"], dtype=float)
        origin = np.asarray(revision["native_origin_xy"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if (matrix.shape != (3, 3) or origin.shape != (2,)
            or not (np.isfinite(matrix).all() and np.isfinite(origin).all())
            or abs(float(np.linalg.det(matrix))) < 1e-12):
        return None
    return matrix, origin


def metric_rig_alignment(left_row, right_row):
    """Map a left display-canvas pixel to the right canvas pixel at the same rig point.

    Display only. Canvas pixel c is native pixel c + native_origin_xy, and each
    frame's recorded homography maps native pixels into the shared metric rig
    frame. Returns None when either frame lacks that record.
    """
    import numpy as np
    left, right = _metric_rig_transform(left_row), _metric_rig_transform(right_row)
    if left is None or right is None:
        return None
    (h_left, o_left), (h_right, o_right) = left, right
    canvas_to_native = np.array([[1.0, 0.0, o_left[0]], [0.0, 1.0, o_left[1]], [0.0, 0.0, 1.0]])
    native_to_canvas = np.array([[1.0, 0.0, -o_right[0]], [0.0, 1.0, -o_right[1]], [0.0, 0.0, 1.0]])
    matrix = native_to_canvas @ np.linalg.inv(h_right) @ h_left @ canvas_to_native
    return matrix / matrix[2, 2]


def _warp_display_mask(mask, matrix):
    import cv2
    import numpy as np
    if mask is None:
        return None
    value = np.asarray(mask)
    warped = cv2.warpPerspective(np.ascontiguousarray(value.astype(np.uint8)), matrix,
                                 (value.shape[1], value.shape[0]),
                                 flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped.astype(bool)


def _warp_display_image(image, matrix, *, overlay):
    """Resample a right-pane image onto the left pane's rig geometry (display only)."""
    import cv2
    import numpy as np
    if image is None or image.isNull():
        return image
    fmt = QImage.Format.Format_ARGB32 if overlay else QImage.Format.Format_RGB32
    source = image.convertToFormat(fmt)
    width, height = source.width(), source.height()
    pixels = np.frombuffer(source.constBits(), np.uint8).reshape(
        height, source.bytesPerLine())[:, :width * 4].reshape(height, width, 4)
    # Outside the right photograph: transparent overlay, rig-canvas background photo (BGRA).
    border = (0, 0, 0, 0) if overlay else (0x26, 0x23, 0x20, 0xFF)
    warped = cv2.warpPerspective(np.ascontiguousarray(pixels), matrix, (width, height),
                                 flags=(cv2.INTER_NEAREST if overlay else cv2.INTER_LINEAR)
                                 | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    result = QImage(warped.data, width, height, warped.strides[0], fmt).copy()
    for name in ("_photo_sand_domain", "_filled_phase_mask", "_floor_contact_fill_mask"):
        value = getattr(image, name, None)
        if value is not None:
            setattr(result, name, _warp_display_mask(value, matrix))
    return result


def _frame_rows(manifest):
    """Saved timeline rows: a RAW review's Studio rows, or the historical frame sources (none when absent)."""
    from darsia.fluidflower.raw_photographic_review import RAW_REVIEW_FORMAT, studio_rows
    if manifest.get("format") == RAW_REVIEW_FORMAT:
        return studio_rows(manifest)
    rows = manifest.get("frame_sources")
    return rows or []


def _content_shape(manifest):
    """The native grid (height, width) of a saved source, or None when it records none."""
    from darsia.fluidflower.raw_photographic_review import RAW_REVIEW_FORMAT, review_content_shape
    if manifest.get("format") == RAW_REVIEW_FORMAT:
        return list(review_content_shape(manifest))
    return manifest.get("analysis_image_shape_hw")


def run_manifest(job):
    path = Path(job["review_root"]) / "native_frames" / job["run"] / "manifest.json"
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def match_frames(left, right, *, left_digest, right_digest):
    """Match by raw photograph identity and time, never merely by frame number."""
    shape = _content_shape(left)
    if not shape or list(shape) != list(_content_shape(right) or []):
        raise ValueError("The saved analysis grids differ. Automatic resizing is not supported.")
    if left.get("dataset") != right.get("dataset") or left.get("run") != right.get("run"):
        raise ValueError("Choose the same experiment and series on both sides.")
    same_manifest = left_digest == right_digest

    def index(manifest):
        result = {}
        for position, row in enumerate(_frame_rows(manifest)):
            frame = int(row.get("frame_index", position))
            digest = row.get("source_sha256")
            if not digest and not same_manifest:
                raise ValueError("These source versions lack matching photograph hashes.")
            elapsed = float(row["elapsed_h"])
            if not math.isfinite(elapsed):
                raise ValueError("The saved frame time is not finite.")
            key = (digest or (left_digest, frame), round(elapsed, 9))
            if key in result:
                raise ValueError("Ambiguous duplicate photograph/time in the saved timeline.")
            result[key] = (frame, row)
        return result

    a, b = index(left), index(right)
    pairs = [dict(left=frame, right=b[key][0], elapsed_h=float(row["elapsed_h"]),
                  left_elapsed_h=float(row["elapsed_h"]),
                  right_elapsed_h=float(b[key][1]["elapsed_h"]),
                  source_name=row.get("source_name", ""), source_sha256=row.get("source_sha256", ""))
             for key, (frame, row) in a.items() if key in b]
    if not pairs:
        raise ValueError("The models have no byte-identified source photographs in common.")
    return dict(frames=pairs, left_total=len(a), right_total=len(b),
                identical_native_source=same_manifest,
                time_note="Same source photograph and recorded elapsed time",
                geometry_note=("Identical native source" if same_manifest else
                    "Same raw photograph and grid dimensions; each model's own saved correction"))


def align_cross_run_frames(left, right):
    """Use each left frame and the nearest recorded elapsed time on the right."""
    if left.get("dataset") != right.get("dataset") or left.get("run") == right.get("run"):
        raise ValueError("Cross-experiment alignment requires two runs in one series.")
    shape = _content_shape(left)
    if not shape or list(shape) != list(_content_shape(right) or []):
        raise ValueError("The saved analysis grids differ. Automatic resizing is not supported.")

    def timeline(manifest):
        rows = []
        seen = set()
        for position, source in enumerate(_frame_rows(manifest)):
            frame = int(source.get("frame_index", position))
            elapsed = float(source["elapsed_h"])
            if frame in seen or not math.isfinite(elapsed):
                raise ValueError("Ambiguous frame index or invalid time in the saved timeline.")
            seen.add(frame)
            rows.append((frame, elapsed, source))
        if not rows:
            raise ValueError("Both experiments need saved frames for comparison.")
        return rows

    a, b = timeline(left), timeline(right)
    frames = []
    for left_frame, elapsed, source in a:
        right_frame, right_elapsed, _ = min(
            b, key=lambda row: (abs(row[1] - elapsed), row[1], row[0]))
        frames.append(dict(left=left_frame, right=right_frame,
                           elapsed_h=elapsed, left_elapsed_h=elapsed,
                           right_elapsed_h=right_elapsed,
                           source_name=source.get("source_name", ""),
                           source_sha256=source.get("source_sha256", "")))
    # Display only: when every frame records its metric rig frame, the right pane
    # is resampled so both panes show the same physical rig position.
    physical = all(_metric_rig_transform(source) is not None for _frame, _elapsed, source in a + b)
    return dict(frames=frames, left_total=len(a), right_total=len(b),
                identical_native_source=False, physical_alignment=physical,
                time_note="Right frame nearest in recorded elapsed hours to each left frame; right frames may repeat",
                geometry_note=("Different experiments at the same physical rig position; the right pane is "
                               "resampled through each frame's recorded metric rig frame (display only)"
                               if physical else
                               "Different experiments on the same saved grid; each side shows its own photograph and correction"))


def load_run_pair(left, right, left_run, right_run=None):
    right_run = left_run if right_run is None else right_run
    a, b = left["jobs"][left_run], right["jobs"][right_run]
    ma, da = run_manifest(a)
    mb, db = run_manifest(b)
    alignment = (match_frames(ma, mb, left_digest=da, right_digest=db)
                 if left_run == right_run else align_cross_run_frames(ma, mb))
    return dict(**alignment, runs=(left_run, right_run),
                jobs=(a, b), manifests=(ma, mb))


def available_brightness_modes(pair) -> tuple[str, ...]:
    """Offer the change derivative only when both panes have its addendum."""
    if not pair or not all(display_modes_available(m) for m in pair["manifests"]):
        return BASE_DISPLAY_MODES
    if all(change_display_available(job["review_root"], job["run"],
                                    frame_count=len(_frame_rows(manifest)))
           for job, manifest in zip(pair["jobs"], pair["manifests"])):
        return DISPLAY_MODES
    return BASE_DISPLAY_MODES


def available_panel_photo_modes(pair, side) -> tuple[str, ...]:
    """Offer each pane only photographs bound to its own saved source."""
    if not pair:
        return ("natural",)
    manifest = pair["manifests"][side]
    job = pair["jobs"][side]
    available = {"natural"}
    if original_display_available(manifest):
        available.add("original")
    if display_modes_available(manifest):
        available.update(BASE_DISPLAY_MODES)
        if change_display_available(job["review_root"], job["run"],
                                    frame_count=len(_frame_rows(manifest))):
            available.add("change")
    if raw_detail_available(manifest):
        available.add("raw_detail")
    return tuple(key for _label, key in PANEL_PHOTO_MODES if key in available)


def _crop(image, manifest, *, overlay=False, frame_index=None, common_rig_revision=None):
    if image.isNull():
        raise ValueError("The saved image could not be decoded.")
    h, w = map(int, _content_shape(manifest))
    canvas = manifest.get("canvas_wh")
    box = manifest.get("content_box_xywh")
    revision = (_frame_rows(manifest)[frame_index].get("rig_domain_revision")
                if frame_index is not None else None)
    if revision is None and frame_index is None:
        revision = next((row["rig_domain_revision"] for row in _frame_rows(manifest)
                         if "phase_shape_hw" in (row.get("rig_domain_revision") or {})), None)
    from darsia.fluidflower.phase_rig_display_canvas import (
        on_common_rig_canvas, pad_rig_photograph, validate_rig_display_overlay)
    if revision and "phase_shape_hw" in revision:
        return (validate_rig_display_overlay(image, revision) if overlay
                else pad_rig_photograph(image, revision))
    if common_rig_revision is not None:
        return on_common_rig_canvas(image, manifest, common_rig_revision, overlay=overlay)
    if any(row.get("rig_domain_revision") for row in _frame_rows(manifest)):
        if not canvas or [image.width(), image.height()] != list(canvas):
            raise ValueError("Rig-domain photographs and maps require their full recorded review canvas.")
        return image
    if canvas and [image.width(), image.height()] == list(canvas) and box:
        x, y, bw, bh = map(int, box)
        if x < 0 or y < 0 or x + bw > image.width() or y + bh > image.height():
            raise ValueError("The recorded native content box is outside the image.")
        image = image.copy(x, y, bw, bh)
    if (image.width(), image.height()) != (w, h):
        raise ValueError("Saved photograph/overlay dimensions do not match the native grid.")
    return image


def load_images(pair, frame_position, phase="all", *, raw_detail=False, display_mode="natural"):
    """Read each pane's selected photograph on its unchanged native grid.

    Scalar arguments retain the historical common-view API; two-element tuples
    select independent left/right views. None of these modes alters evidence.
    """
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    frame = pair["frames"][frame_position]
    payload = []
    revisions = [_frame_rows(manifest)[frame[side]].get("rig_domain_revision")
                 for side, manifest in zip(("left", "right"), pair["manifests"])]
    padded = [revision for revision in revisions if revision and "phase_shape_hw" in revision]
    common_rig_revision = padded[0] if padded else None
    if padded and any(revision["phase_shape_hw"] != common_rig_revision["phase_shape_hw"]
                      or revision["canvas_wh"] != common_rig_revision["canvas_wh"] for revision in padded):
        raise ValueError("Compared rig candidates have different complete display grids")
    for position, (side, job, manifest) in enumerate(zip(
            ("left", "right"), pair["jobs"], pair["manifests"])):
        # Check that a live NAS update cannot silently relabel a loaded timeline.
        current, _ = run_manifest(job)
        if current != manifest:
            raise ValueError("The saved source manifest changed. Reload the selected models.")
        candidate = review.Candidate(**job["candidate"])
        root = Path(job["review_root"])
        withheld = review._candidate_prediction_withheld(root, candidate)
        photograph = review._native_frame_path(root, job["run"], frame[side])
        mode = display_mode[position] if isinstance(display_mode, (tuple, list)) else display_mode
        raw = raw_detail[position] if isinstance(raw_detail, (tuple, list)) else raw_detail
        original_image = None
        if mode == "original":
            rgb = load_original_display_rgb(root, job["run"], frame[side])
            height, width = rgb.shape[:2]
            original_image = QImage(rgb.data, width, height, rgb.strides[0],
                                    QImage.Format.Format_RGB888).copy()
        elif ((raw or mode in ("raw_detail", "change"))
              and rig_raw_display_available(manifest, frame_index=frame[side])):
            original_image = load_rig_raw_display_image(
                root, job["run"], frame[side], mode="raw_detail" if raw else mode)
        elif (raw or mode == "raw_detail") and ((manifest.get("v5_photometry") or {}).get("format") == V5_SOURCE_FORMAT
                                                or manifest.get("format") == RAW_REVIEW_FORMAT):
            photograph = resolve_raw_detail(root, job["run"], frame[side])
        elif display_modes_available(manifest):
            photograph = resolve_display_mode(root, job["run"], frame[side], mode)
        overlay = review._native_phase_map_path(root, candidate, "A", frame[side])
        overlay = review._native_phase_map_variant_path(overlay, overlay_asset_layer(phase), fallback=False)
        if (original_image is None and not photograph.is_file()) or (not withheld and not overlay.is_file()):
            raise FileNotFoundError(f"Saved {phase} map or photograph is unavailable: {overlay}")
        rig_canvas = bool(_frame_rows(manifest)[frame[side]].get("rig_domain_revision"))
        domain = (resolve_photo_sand_domain(root, job["run"], frame[side], canvas=rig_canvas or common_rig_revision is not None)
                  if display_modes_available(manifest) else None)
        from darsia.fluidflower.phase_rig_display_canvas import pad_rig_display_mask
        domain = pad_rig_display_mask(domain, common_rig_revision, manifest=manifest)
        if withheld:
            rendered = QImage()
            rendered._bottom_blind_withheld = True
        else:
            rendered = display_phase_overlay(_crop(QImage(str(overlay)), manifest, overlay=True,
                frame_index=frame[side], common_rig_revision=common_rig_revision), phase, photo_sand=domain)
        from darsia.fluidflower.phase_filled_display import resolve_filled_display_mask
        filled = (resolve_filled_display_mask(root, job["run"], frame[side], canvas=True)
                  if rig_canvas and not withheld else None)
        rendered._filled_phase_mask = pad_rig_display_mask(filled, common_rig_revision, manifest=manifest)
        from darsia.fluidflower.phase_floor_contact_display import resolve_floor_contact_display_mask
        contact = (resolve_floor_contact_display_mask(root, job["run"], frame[side])
                   if rig_canvas and not withheld and _frame_rows(manifest)[frame[side]].get("floor_contact_fill") else None)
        rendered._floor_contact_fill_mask = pad_rig_display_mask(contact, common_rig_revision, manifest=manifest)
        payload.append((_crop(original_image if original_image is not None else QImage(str(photograph)), manifest,
            frame_index=frame[side], common_rig_revision=common_rig_revision), rendered))
    if pair.get("physical_alignment") and len(payload) == 2:
        rows = [_frame_rows(manifest)[frame[side]]
                for side, manifest in zip(("left", "right"), pair["manifests"])]
        matrix = metric_rig_alignment(*rows)
        if matrix is not None:
            payload[1] = tuple(_warp_display_image(image, matrix, overlay=index == 1)
                               for index, image in enumerate(payload[1]))
    return payload


def compose(photograph, overlay, opacity, *, enabled=True, black=False, filled=True, floor_contact=True,
            hatched=None):
    """The photograph with its phase overlay at ``opacity``; display only.

    ``hatched`` (a dict) keeps each overlay's hatched copy for its latest
    switch state, so a change of opacity alone blends again without hatching
    the full-resolution overlay again.
    """
    if getattr(overlay, "_bottom_blind_withheld", False):
        enabled, black, filled, floor_contact = False, False, False, False
    result = QImage(photograph.size(), QImage.Format.Format_RGB32)
    result.fill(QColor("black"))
    painter = QPainter(result)
    if not black:
        painter.drawImage(0, 0, photograph)
    from darsia.fluidflower.phase_filled_display import hatch_filled_overlay
    from darsia.fluidflower.phase_floor_contact_display import hatch_floor_contact_overlay
    mask = getattr(overlay, "_filled_phase_mask", None)
    contact = getattr(overlay, "_floor_contact_fill_mask", None)
    # One hatched copy per overlay, for its latest switch state only.
    switches = (bool(enabled), bool(filled), bool(floor_contact))
    kept = hatched.get(id(overlay)) if hatched is not None else None
    if kept is not None and kept[0] is overlay and kept[1] == switches:
        overlay = kept[2]
    else:
        source = overlay
        overlay = hatch_filled_overlay(overlay, mask, enabled=filled, phase_visible=enabled)
        overlay = hatch_floor_contact_overlay(overlay, contact, enabled=floor_contact, phase_visible=enabled)
        if hatched is not None:
            hatched[id(source)] = (source, switches, overlay)
    if enabled or (filled and mask is not None):
        painter.setOpacity(max(0., min(1., float(opacity))))
        painter.drawImage(0, 0, overlay)
    painter.end()
    return result


class _ComparisonPointerOverlay(QWidget):
    """Paint a duplicate cursor without changing either saved image."""

    def __init__(self, pair, side):
        super().__init__(pair.views[side])
        self.pair, self.side = pair, side
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setGeometry(pair.views[side].rect())

    def paintEvent(self, event):
        point = self.pair._comparison_pointer
        if point is None or not self.pair.has_images or self.pair.center is None:
            return
        view = self.pair.views[self.side]
        position = QPointF(view.width() / 2, view.height() / 2) + (
            point - self.pair.center) * self.pair.zoom
        painter = QPainter(self)
        for width, color in ((3, "#101820"), (1, "#ffffff")):
            painter.setPen(QPen(QColor(color), width))
            painter.drawLine(position + QPointF(-9, 0), position + QPointF(9, 0))
            painter.drawLine(position + QPointF(0, -9), position + QPointF(0, 9))
        painter.end()


class _ComparisonImagePair(SynchronizedImagePair):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._comparison_pointer = None
        self.pointer_overlays = [_ComparisonPointerOverlay(self, side) for side in (0, 1)]
        for view in self.views:
            view.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched in self.views:
            side = self.views.index(watched)
            if event.type() == QEvent.Type.MouseMove and self.has_images:
                point = self.image_coordinate(side, event.position())
                size = self.grid_size
                self._comparison_pointer = (point if size and 0 <= point.x() < size.width()
                                            and 0 <= point.y() < size.height() else None)
                for overlay in self.pointer_overlays:
                    overlay.update()
            elif event.type() == QEvent.Type.Leave:
                self._comparison_pointer = None
                for overlay in self.pointer_overlays:
                    overlay.update()
            elif event.type() == QEvent.Type.Resize:
                self.pointer_overlays[side].setGeometry(watched.rect())
        return super().eventFilter(watched, event)

    def _update_views(self):
        super()._update_views()
        for overlay in getattr(self, "pointer_overlays", ()):
            overlay.update()


class PhaseModelComparisonDialog(QDialog):
    """One shared timeline and pixel viewport; all NAS I/O runs off the UI thread."""
    _completed = Signal(object)
    review_requested = Signal(str)

    def __init__(self, entries, selected_root="", parent=None, *, embedded=False):
        super().__init__(parent)
        from darsia.fluidflower.ui.interaction_safety import install_interaction_safety
        install_interaction_safety()
        self.embedded = embedded
        if embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
        self.setWindowTitle("Compare phase models")
        self.resize(1500, 900)
        self._generation = 0
        self._busy = False
        self._pending = None
        self._closed = False
        self._manual_entries = []
        self._entries = {str(entry["root"]): dict(entry) for entry in entries}
        self.catalogs = None
        self.pair = None
        self.images = None
        # Hatched overlays of the shown images, and a short pause before an
        # opacity change is blended (both images are full resolution).
        self._hatched = {}
        self._opacity_timer = QTimer(self)
        self._opacity_timer.setSingleShot(True)
        self._opacity_timer.setInterval(60)
        self._opacity_timer.timeout.connect(self._paint)
        self._space_swap = False
        self._badge_texts = ["", ""]
        self._completed.connect(self._finished)
        layout = QVBoxLayout(self)
        explanation = QLabel("Compare saved phase maps. Green = aqueous CO₂; pink = free gas. "
                             "Use Open for correction to continue annotating either exact model.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        models = QHBoxLayout()
        layout.addLayout(models)
        models.addWidget(QLabel("View"))
        self.view_mode = QComboBox()
        self.view_mode.addItem("Side by side", "pair")
        self.view_mode.addItem("Single model", "single")
        models.addWidget(self.view_mode)
        self.selectors, self.badges, self.review_buttons, self.model_labels = [], [], [], []
        for side in ("Left model", "Right model"):
            label = QLabel(side)
            self.model_labels.append(label)
            models.addWidget(label)
            combo = QComboBox()
            combo.setMinimumContentsLength(36)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            for entry in entries:
                root = str(entry["root"])
                combo.addItem(model_title(entry), root)
                combo.setItemData(combo.count() - 1, model_details(entry), Qt.ItemDataRole.ToolTipRole)
            self.selectors.append(combo)
            models.addWidget(combo, 1)
        add = QPushButton("Add model folder…")
        add.clicked.connect(self._add_model)
        models.addWidget(add)
        swap = QPushButton("Swap sides")
        self.swap_button = swap
        swap.clicked.connect(self.swap_sides)
        models.addWidget(swap)
        run_controls = QHBoxLayout()
        layout.addLayout(run_controls)
        self.left_run_label = QLabel("Left experiment")
        run_controls.addWidget(self.left_run_label)
        self.run = QComboBox()
        run_controls.addWidget(self.run, 1)
        run_controls.addWidget(QLabel("Sort left"))
        self.run_sort = QComboBox()
        for label, key in (
            ("Experiment number", "number"),
            ("Free gas area · lowest", "gas_asc"),
            ("Free gas area · highest", "gas_desc"),
            ("Aqueous CO₂ area · lowest", "aqueous_asc"),
            ("Aqueous CO₂ area · highest", "aqueous_desc"),
        ):
            self.run_sort.addItem(label, key)
        run_controls.addWidget(self.run_sort)
        self.right_run_label = QLabel("Right experiment")
        run_controls.addWidget(self.right_run_label)
        self.right_run = QComboBox()
        run_controls.addWidget(self.right_run, 1)
        self.right_sort_label = QLabel("Sort right")
        run_controls.addWidget(self.right_sort_label)
        self.right_run_sort = QComboBox()
        for index in range(self.run_sort.count()):
            self.right_run_sort.addItem(self.run_sort.itemText(index),
                                        self.run_sort.itemData(index))
        run_controls.addWidget(self.right_run_sort)
        self.runs = (self.run, self.right_run)
        self.run_sorts = (self.run_sort, self.right_run_sort)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.phase = QComboBox()
        for name, key in (("All predicted phases", "all"), ("Free gas", "gas"), ("Aqueous CO₂", "aqueous"), ("Unresolved source support (cyan)", "source_support")):
            self.phase.addItem(name, key)
        controls.addWidget(self.phase)
        self.black = QCheckBox("Black background")
        controls.addWidget(self.black)
        controls.addWidget(QLabel("Hold Space to swap the two views temporarily"))
        controls.addStretch(1)
        display_controls = QHBoxLayout()
        layout.addLayout(display_controls)
        self.panel_controls = []
        for side, title in enumerate(("Left view", "Right view")):
            group = QWidget(self)
            group_layout = QHBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.addWidget(QLabel(title))
            photo = QComboBox(group)
            for label, key in PANEL_PHOTO_MODES:
                photo.addItem(label, key)
            photo.setCurrentIndex(photo.findData("natural"))
            photo.setToolTip("Choose this panel's photograph. Saved phase maps and scientific inputs stay unchanged.")
            group_layout.addWidget(photo, 1)
            overlay = QCheckBox("Phase maps", group)
            overlay.setChecked(True)
            group_layout.addWidget(overlay)
            filled = QCheckBox("Extrapolated hatching", group)
            filled.setChecked(True)
            filled.setToolTip("Display filled rig-domain pixels without changing saved phase labels.")
            group_layout.addWidget(filled)
            contact = QCheckBox("Floor-contact hatching", group)
            contact.setChecked(True)
            contact.setToolTip("Display the flagged aqueous floor interpretation; no change to maps or concentration.")
            group_layout.addWidget(contact)
            group_layout.addWidget(QLabel("Opacity"))
            opacity = QSlider(Qt.Orientation.Horizontal, group)
            opacity.setRange(0, 100)
            opacity.setValue(50)
            opacity.setMaximumWidth(170)
            group_layout.addWidget(opacity, 1)
            self.panel_controls.append(dict(photo=photo, overlay=overlay,
                                            opacity=opacity, filled=filled, floor_contact=contact, widget=group))
            photo.currentIndexChanged.connect(self._frame_changed)
            overlay.toggled.connect(self._paint)
            opacity.valueChanged.connect(lambda _value: self._opacity_timer.start())
            filled.toggled.connect(self._paint)
            contact.toggled.connect(self._paint)
            display_controls.addWidget(group, 1)
        # Retain left-control aliases for callers of the earlier common-view UI.
        self.overlay = self.panel_controls[0]["overlay"]
        self.opacity = self.panel_controls[0]["opacity"]
        self.image_pair = _ComparisonImagePair(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.split = split
        for view in self.image_pair.views:
            split.addWidget(view)
        layout.addWidget(split, 1)
        captions = QHBoxLayout()
        layout.addLayout(captions)
        for index in (0, 1):
            badge = QLabel("")
            badge.setWordWrap(True)
            captions.addWidget(badge, 1)
            self.badges.append(badge)
            button = QPushButton("Open for correction")
            button.setEnabled(False)
            self.review_buttons.append(button)
            button.clicked.connect(lambda _checked=False, side=index:
                                   self.review_requested.emit(str(self.selectors[side].currentData() or "")))
            captions.addWidget(button)
        navigation = QHBoxLayout()
        layout.addLayout(navigation)
        self.previous = QPushButton("Previous")
        self.next = QPushButton("Next")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.frame_label = QLabel("")
        navigation.addWidget(self.previous)
        navigation.addWidget(self.timeline, 1)
        navigation.addWidget(self.next)
        navigation.addWidget(self.frame_label)
        self.previous.clicked.connect(lambda: self.timeline.setValue(self.timeline.value() - 1))
        self.next.clicked.connect(lambda: self.timeline.setValue(self.timeline.value() + 1))
        footer = QHBoxLayout()
        layout.addLayout(footer)
        self.status = QLabel("Loading saved models…")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        actions = [("Fit", self.image_pair.fit), ("1:1 pixels", self.image_pair.actual_size),
                   ("Reload", self._models_changed)]
        if not embedded:
            actions.append(("Close", self.close))
        for title, callback in actions:
            button = QPushButton(title)
            button.clicked.connect(callback)
            footer.addWidget(button)
        if entries:
            first = max(0, self.selectors[0].findData(str(selected_root)))
            self.selectors[0].setCurrentIndex(first)
            self.selectors[1].setCurrentIndex(next((i for i in range(len(entries)) if i != first), first))
        for selector in self.selectors:
            selector.currentIndexChanged.connect(self._models_changed)
        self.run.currentIndexChanged.connect(self._run_changed)
        self.right_run.currentIndexChanged.connect(self._run_changed)
        self.run_sort.currentIndexChanged.connect(lambda: self._sort_changed(0))
        self.right_run_sort.currentIndexChanged.connect(lambda: self._sort_changed(1))
        self.timeline.valueChanged.connect(self._frame_changed)
        self.phase.currentIndexChanged.connect(self._frame_changed)
        self.black.toggled.connect(self._paint)
        self.view_mode.currentIndexChanged.connect(self._mode_changed)
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)
        self._models_changed()

    def set_entries(self, entries, selected_root="", *, reset=False):
        """Update discovery results without discarding the user's model choices."""
        before = tuple(selector.currentData() for selector in self.selectors)
        if reset:
            self._manual_entries = []
        entries = list(entries) + self._manual_entries
        self._entries = {str(entry["root"]): dict(entry) for entry in entries}
        for side, selector in enumerate(self.selectors):
            selector.blockSignals(True)
            selector.clear()
            for entry in entries:
                root = str(entry["root"])
                title = model_title(entry)
                index = selector.findData(root)
                if index < 0:
                    selector.addItem(title, root)
                    index = selector.count() - 1
                else:
                    selector.setItemText(index, title)
                selector.setItemData(index, model_details(entry), Qt.ItemDataRole.ToolTipRole)
            preferred = selected_root if side == 0 and selected_root else (None if reset else before[side])
            index = selector.findData(preferred)
            if index >= 0:
                selector.setCurrentIndex(index)
            selector.blockSignals(False)
        if before != tuple(selector.currentData() for selector in self.selectors):
            self._models_changed()

    def _mode_changed(self, *_):
        single = self.view_mode.currentData() == "single"
        self._set_space_swap(False)
        if single:
            self._paired_right_run = self.right_run.currentData()
        self.model_labels[0].setText("Model" if single else "Left model")
        for control in (self.model_labels[1], self.selectors[1], self.swap_button,
                        self.right_run_label, self.right_run, self.right_sort_label,
                        self.right_run_sort, self.image_pair.views[1], self.badges[1],
                        self.review_buttons[1], self.panel_controls[1]["widget"]):
            control.setVisible(not single)
        self.left_run_label.setText("Experiment" if single else "Left experiment")
        if not getattr(self, "_allow_correction", True):
            for button in self.review_buttons:
                button.hide()
        self._models_changed(preserve_selection=True)
        if not single and self._restore_selection and getattr(self, "_paired_right_run", None):
            self._restore_selection["runs"] = (
                self._restore_selection["runs"][0], self._paired_right_run)
            self._paired_right_run = None

    def discover_entries(self, dataset, target_model_id=""):
        """Reuse Studio discovery off the GUI thread, including local saved models."""
        def discover():
            from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage, _models_for_series
            from darsia.fluidflower.series_paths import series_path
            settings = QSettings("DarSIA", "DarSIA Phase Model Studio")
            saved = json.loads(str(settings.value("modelHistory/completedModelRoots", "{}")))
            entries = SharedPhaseModelTrainingPage._discover_model_entries(
                str(series_path(dataset, "phase_models")), tuple(saved.values()) if isinstance(saved, dict) else ())
            entries = _models_for_series(entries, dataset)
            selected = next((str(row["root"]) for row in entries
                             if str(row.get("model_id")) == target_model_id), "")
            return entries, selected
        self.status.setText("Discovering saved phase models…")
        self._submit("entries", discover)

    def _submit(self, kind, function):
        self._generation += 1
        self._pending = (self._generation, kind, function)
        self._start_pending()

    def _start_pending(self):
        if self._busy or self._closed or self._pending is None:
            return
        generation, kind, function = self._pending
        self._pending = None
        self._busy = True

        def work():
            try:
                result, error = function(), ""
            except Exception as exc:
                result, error = None, f"{type(exc).__name__}: {exc}"
            try:
                self._completed.emit((generation, kind, result, error))
            except RuntimeError:
                pass  # Closing never waits for an unavailable network share.
        saved_reader_context = copy_context()
        Thread(target=lambda: saved_reader_context.run(work), name="phase-model-comparison-read", daemon=True).start()

    def _clear(self, *, preserve_view=False):
        self.images = None
        self._hatched = {}
        self.image_pair.clear(preserve_view=preserve_view)

    def _models_changed(self, *_, preserve_selection=False, swap_runs=False):
        selected_runs = tuple(combo.currentData() for combo in self.runs)
        if preserve_selection and self.pair:
            frame = self.pair["frames"][self.timeline.value()]
            self._restore_selection = dict(
                runs=selected_runs[::-1] if swap_runs else selected_runs,
                frame=frame["right"] if swap_runs else frame["left"],
                source_sha256=None if swap_runs else frame.get("source_sha256"),
                elapsed_h=(frame.get("right_elapsed_h", frame.get("elapsed_h"))
                           if swap_runs else frame.get("left_elapsed_h", frame.get("elapsed_h"))),
            )
        elif preserve_selection and any(selected_runs) and not self._restore_selection:
            self._restore_selection = dict(runs=selected_runs[::-1] if swap_runs else selected_runs,
                                           frame=None, source_sha256=None, elapsed_h=None)
        elif not preserve_selection:
            self._restore_selection = None
        self._clear(preserve_view=bool(self._restore_selection))
        self.catalogs, self.pair = None, None
        roots = tuple(selector.currentData() for selector in self.selectors)
        for selector, root in zip(self.selectors, roots):
            entry = self._entries.get(str(root or ""))
            selector.setToolTip(model_details(entry) if entry else str(root or ""))
        for button in self.review_buttons:
            button.setEnabled(False)
        self.frame_label.clear()
        for combo in self.runs:
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
        self.timeline.setEnabled(False)
        single = self.view_mode.currentData() == "single"
        if not roots[0] or (not single and not roots[1]):
            self._generation += 1
            self._pending = None
            self.status.setText("Choose a saved model." if single else "Choose a saved model for each side.")
            return
        self.status.setText("Reading the selected models and their saved map bindings…")
        def load():
            left = model_catalog(roots[0])
            left["area_peaks"] = model_run_areas(left)
            if single or roots[0] == roots[1]:
                return left, left
            right = model_catalog(roots[1])
            right["area_peaks"] = model_run_areas(right)
            return left, right
        self._submit("models", load)

    def _ordered_runs(self, side, available):
        mode = str(self.run_sorts[side].currentData() or "number")
        if mode == "number":
            def natural(run):
                digits = "".join(character for character in run if character.isdigit())
                return (int(digits) if digits else 10**9, run)
            return sorted(available, key=natural)
        phase, direction = mode.split("_", 1)
        areas = self.catalogs[side].get("area_peaks", {}) if self.catalogs else {}
        rows = [(run, run.upper(), "", 0) for run in available]
        values = {run: {"selected": areas.get(run)} for run in available}
        return [row[0] for row in sorted_phase_area_runs(
            rows, values, side="selected", phase=phase, largest_first=direction == "desc")]

    def _fill_runs(self, side, available, preferred=None):
        combo = self.runs[side]
        combo.blockSignals(True)
        combo.clear()
        areas = self.catalogs[side].get("area_peaks", {}) if self.catalogs else {}
        for run in self._ordered_runs(side, available):
            peak = (areas.get(run) or {}).get("gas")
            label = f"{run.upper()} · peak free CO₂ {peak:.2f}%" if peak is not None else run.upper()
            combo.addItem(label, run)
            combo.setItemData(combo.count() - 1,
                              (f"Peak detected free CO₂: {peak:.2f}% of porous domain"
                               if peak is not None else "Peak free CO₂ area unavailable"),
                              Qt.ItemDataRole.ToolTipRole)
        if preferred:
            index = combo.findData(preferred)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _sort_changed(self, side):
        if not self.catalogs:
            return
        current = self.runs[side].currentData()
        self._fill_runs(side, self.catalogs[side]["jobs"], current)
        if self.runs[side].currentData() != current:
            self._run_changed()

    def _run_changed(self, *_):
        self._clear(preserve_view=bool(self._restore_selection))
        self.pair = None
        runs = tuple(combo.currentData() for combo in self.runs)
        if self.view_mode.currentData() == "single" and runs[0]:
            index = self.right_run.findData(runs[0])
            if index >= 0:
                self.right_run.blockSignals(True)
                self.right_run.setCurrentIndex(index)
                self.right_run.blockSignals(False)
            runs = runs[0], runs[0]
        if not all(runs) or not self.catalogs:
            return
        catalogs = self.catalogs
        self.status.setText("Matching saved source photographs…" if runs[0] == runs[1]
                            else "Aligning experiments by nearest recorded elapsed time…")
        self._submit("run", lambda: load_run_pair(*catalogs, *runs))

    def _frame_changed(self, *_):
        if not self.pair:
            return
        self._clear(preserve_view=True)
        position, pair, phase = self.timeline.value(), self.pair, self.phase.currentData()
        self._update_frame_label()
        self.previous.setEnabled(position > 0)
        self.next.setEnabled(position < len(pair["frames"]) - 1)
        self.status.setText("Loading saved native photograph and phase maps…")
        modes = tuple(str(panel["photo"].currentData() or "natural")
                      for panel in self.panel_controls)
        raw_modes = tuple(mode == "raw_detail" for mode in modes)
        photograph_modes = tuple("natural" if raw else mode
                                 for mode, raw in zip(modes, raw_modes))
        # Keep scalar defaults compatible with the saved-map loader's earlier API.
        raw = raw_modes[0] if raw_modes[0] == raw_modes[1] else raw_modes
        mode = (photograph_modes[0] if photograph_modes[0] == photograph_modes[1]
                else photograph_modes)
        self._submit("images", lambda: load_images(
            pair, position, phase, raw_detail=raw, display_mode=mode))

    def _cycle_brightness(self, side=0):
        modes = tuple(mode for mode in available_panel_photo_modes(self.pair, side)
                      if mode in DISPLAY_MODES)
        combo = self.panel_controls[side]["photo"]
        current = str(combo.currentData() or "natural")
        position = modes.index(current) if current in modes else -1
        if modes:
            combo.setCurrentIndex(combo.findData(modes[(position + 1) % len(modes)]))

    def _refresh_panel_modes(self):
        for side, panel in enumerate(self.panel_controls):
            available = available_panel_photo_modes(self.pair, side)
            combo = panel["photo"]
            combo.blockSignals(True)
            for index in range(combo.count()):
                combo.model().item(index).setEnabled(combo.itemData(index) in available)
            if combo.currentData() not in available:
                combo.setCurrentIndex(combo.findData("natural"))
            combo.blockSignals(False)

    def _update_frame_label(self):
        position = self.timeline.value()
        row = self.pair["frames"][position]
        self.frame_label.setText(f"{position + 1}/{len(self.pair['frames'])} · "
                                f"L {row['left'] + 1} at {row.get('left_elapsed_h', row['elapsed_h']):.3f} h / "
                                f"R {row['right'] + 1} at {row.get('right_elapsed_h', row['elapsed_h']):.3f} h")

    def _finished(self, payload):
        self._busy = False
        generation, kind, result, error = payload
        if not self._closed and generation == self._generation:
            if error:
                self._clear()
                if kind == "models":
                    roots = tuple(str(selector.currentData() or "") for selector in self.selectors)
                    self.status.setText(f"{error}\nSelected model folders: {roots[0]} | {roots[1]}")
                else:
                    self.status.setText(error)
            elif kind == "entries":
                self.set_entries(*result)
                if not result[0]:
                    self.status.setText("No saved models found. Use Add model folder to select one.")
            elif kind == "models":
                self.catalogs = result
                if result[0]["dataset"] != result[1]["dataset"]:
                    self.catalogs = None
                    self.status.setText("Choose two models from the same experimental series.")
                else:
                    self._runtime_comparison_note = runtime_comparison_note(result)
                    preferred = (self._restore_selection["runs"] if self._restore_selection
                                 else (getattr(self, "_preferred_run", None), None))
                    self._fill_runs(0, result[0]["jobs"], preferred[0])
                    right_preferred = (self.run.currentData() if self.view_mode.currentData() == "single"
                                       else preferred[1] or self.run.currentData())
                    self._fill_runs(1, result[1]["jobs"], right_preferred)
                    self._preferred_run = None
                    for side, (badge, catalog) in enumerate(zip(self.badges, result)):
                        root = str(catalog["root"])
                        entry = dict(self._entries.get(root) or {"root": root})
                        entry["model_id"] = catalog["model_id"]
                        if catalog.get("created_utc"):
                            entry["created_utc"] = catalog["created_utc"]
                        if catalog.get("photometry_arm"):
                            entry["photometry_arm"] = catalog["photometry_arm"]
                        if catalog.get("photometry_stage"):
                            entry["photometry_stage"] = catalog["photometry_stage"]
                        self._entries[root] = entry
                        for selector in self.selectors:
                            index = selector.findData(root)
                            if index >= 0:
                                selector.setItemText(index, model_title(entry))
                                selector.setItemData(index, model_details(entry),
                                                     Qt.ItemDataRole.ToolTipRole)
                                if selector.currentData() == root:
                                    selector.setToolTip(model_details(entry))
                        self._badge_texts[side] = (f"{model_title(entry)} · {catalog['status']}\n"
                            f"ID: {entry['model_id']} · Created UTC: {created_utc_text(entry)}")
                        badge.setText(self._badge_texts[side])
                    for button in self.review_buttons:
                        button.setEnabled(True)
                    if self.run.count() and self.right_run.count():
                        self._run_changed()
                    else:
                        self.status.setText("One side has no experiments with saved phase maps.")
            elif kind == "run":
                self.pair = result
                self._refresh_panel_modes()
                self.timeline.blockSignals(True)
                self.timeline.setRange(0, len(result["frames"]) - 1)
                position = next((i for i, row in enumerate(result["frames"])
                    if self._restore_selection and
                    ((self._restore_selection["source_sha256"] and
                      row.get("source_sha256") == self._restore_selection["source_sha256"] and
                      row.get("left_elapsed_h", row.get("elapsed_h")) ==
                      self._restore_selection["elapsed_h"]) or
                     (not self._restore_selection["source_sha256"] and
                      row["left"] == self._restore_selection["frame"]))), 0)
                self.timeline.setValue(position)
                self.timeline.blockSignals(False)
                self.timeline.setEnabled(True)
                self._frame_changed()
            elif kind == "images":
                self.images = result
                self._hatched = {}
                self._paint()
                self._restore_selection = None
                pair = self.pair
                self.status.setText(f"{pair['geometry_note']} · "
                                    f"{pair.get('time_note', 'Saved timeline')} · "
                                    f"{len(pair['frames'])} displayed left frames "
                                    f"from {pair['left_total']} left / {pair['right_total']} right."
                                    + (" · " + self._runtime_comparison_note
                                       if getattr(self, "_runtime_comparison_note", "") else ""))
        self._start_pending()

    def _paint(self, *_):
        badge_texts = self._badge_texts[::-1] if self._space_swap else self._badge_texts
        for badge, text in zip(self.badges, badge_texts):
            badge.setText(text)
        self._opacity_timer.stop()
        if self.images:
            images = [compose(photo, overlay, controls["opacity"].value() / 100.,
                              enabled=controls["overlay"].isChecked(), black=self.black.isChecked(),
                              filled=controls["filled"].isChecked(),
                              floor_contact=controls["floor_contact"].isChecked(), hatched=self._hatched)
                      for (photo, overlay), controls in zip(self.images, self.panel_controls)]
            if self._space_swap and self.view_mode.currentData() != "single":
                images.reverse()
            self.image_pair.set_images(*images)

    def _set_space_swap(self, enabled):
        enabled = bool(enabled and self.view_mode.currentData() != "single")
        if enabled == self._space_swap:
            return
        self._space_swap = enabled
        badge_texts = self._badge_texts[::-1] if enabled else self._badge_texts
        for badge, text in zip(self.badges, badge_texts):
            badge.setText(text)
        if self.images and self.image_pair.has_images:
            # Space changes presentation only. Reuse the composed pixmaps so
            # a full-resolution photograph is neither loaded nor blended again.
            self.image_pair.swap_images()

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.key() == Qt.Key.Key_Space:
                if not event.isAutoRepeat():
                    self._set_space_swap(event.type() == QEvent.Type.KeyPress)
                return True
        elif event.type() in (QEvent.Type.WindowDeactivate, QEvent.Type.FocusOut,
                              QEvent.Type.Hide):
            self._set_space_swap(False)
        return super().eventFilter(watched, event)

    def _add_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a saved phase model folder")
        if folder:
            entry = dict(root=folder)
            self._manual_entries.append(entry)
            self._entries[folder] = entry
            for selector in self.selectors:
                if selector.findData(folder) < 0:
                    selector.addItem(model_title(entry), folder)
                    selector.setItemData(selector.count() - 1, model_details(entry),
                                         Qt.ItemDataRole.ToolTipRole)
            self.selectors[1].setCurrentIndex(self.selectors[1].findData(folder))

    def swap_sides(self):
        if self.view_mode.currentData() == "single":
            return
        a, b = (selector.currentData() for selector in self.selectors)
        if not a or not b:
            return
        self._set_space_swap(False)
        for selector in self.selectors:
            selector.blockSignals(True)
        self.selectors[0].setCurrentIndex(self.selectors[0].findData(b))
        self.selectors[1].setCurrentIndex(self.selectors[1].findData(a))
        for selector in self.selectors:
            selector.blockSignals(False)
        # A fresh generation invalidates any pending image/model read. Reusing
        # the old pane objects made a late worker repaint the original order.
        self._models_changed(preserve_selection=True, swap_runs=True)

    def closeEvent(self, event):
        if self.embedded:
            event.ignore()
            return
        self._closed = True
        self._pending = None
        self._generation += 1
        super().closeEvent(event)

    def reject(self):
        # Escape must not dismiss a persistent Studio page.
        if not self.embedded:
            super().reject()


def install_review_comparison_tabs(window, correction_widget):
    """Keep the annotation workspace intact while inspecting other saved models."""
    tabs = QTabWidget(window)
    tabs.addTab(correction_widget, "Single model · corrections")
    panel = PhaseModelComparisonDialog([], parent=tabs, embedded=True)
    # This separate read-only view never changes the correction store's model.
    for button in panel.review_buttons:
        button.hide()
    panel._allow_correction = False
    tabs.addTab(panel, "Inspect / compare phase maps")
    window.phase_inspection_tabs = tabs
    window.phase_comparison_panel = panel
    window.setCentralWidget(tabs)
    retry = QTimer(panel)
    retry.setInterval(100)
    def selected(index):
        retry.stop()
        if index != 1:
            return
        pair = window.current_pair or {}
        dataset = str(pair.get("dataset") or "")
        if not dataset:
            panel.status.setText("Waiting for the selected correction experiment…")
            retry.start()
            return
        key = (dataset, window.workspace.target_model_id)
        run = str(window.current_run or "")
        if getattr(panel, "_correction_run", None) != run:
            panel._correction_run = run
            panel._preferred_run = run
            for combo in panel.runs:
                run_index = combo.findData(run)
                if run_index >= 0:
                    combo.setCurrentIndex(run_index)
        if getattr(panel, "_discovery_key", None) != key:
            panel._discovery_key = key
            panel.set_entries([], reset=True)
            panel.discover_entries(*key)
    retry.timeout.connect(lambda: selected(tabs.currentIndex()))
    tabs.currentChanged.connect(selected)
    return tabs


@saved_reader_ui_entrypoint
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--reviewer", default="")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    window = PhaseModelComparisonDialog([dict(root=str(p.resolve())) for p in args.model])
    # Standalone inspection leaves all existing Studio/review processes untouched.
    def review(root):
        from darsia.fluidflower.phase_evidence_audit import verified_review_binding
        from darsia.fluidflower.phase_workflow import build_phase_review_command
        from darsia.fluidflower.series_paths import local_cache_root, series_path
        import subprocess
        try:
            actor = args.reviewer
            if not actor:
                actor, accepted = QInputDialog.getText(window, "Reviewer", "Name for saved corrections")
                if not accepted or not actor.strip():
                    return
                args.reviewer = actor.strip()
            # Open the existing map/annotation geometry; this action performs
            # no inference. New predictions and quiz replay retain strict gates.
            binding = verified_review_binding(Path(root), saved_comparison=True)
            selected = next(item for item in window.catalogs if item["root"] == root)
            command = build_phase_review_command(mode="open_correction", reviewer=actor,
                save_root=series_path(selected["dataset"], "reviews"), cache_root=local_cache_root(),
                campaign_root=Path(binding["combined_root"]),
                selected_candidates=binding["selected_candidates"], target_model_id=Path(root).name)
            subprocess.Popen(command)
        except Exception as error:
            window.status.setText(str(error))
    window.review_requested.connect(review)
    window.showMaximized()
    app.exec()


if __name__ == "__main__":
    main()
