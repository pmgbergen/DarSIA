#!/usr/bin/env python3
"""GPU-assisted desktop client for intensive FluidFlower phase validation."""

from __future__ import annotations

from darsia.fluidflower.series_native_context import saved_reader_ui_entrypoint
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.storage_paths import configured_tool_path
from darsia.fluidflower.review_snapshot_journal import (
    RECOVERY_JOURNAL_MAX_BYTES, append_bounded_snapshot_journal,
)

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from contextvars import copy_context
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

try:
    from PySide6.QtCore import (
        QObject,
        QEvent,
        QPointF,
        QRect,
        QRectF,
        QRunnable,
        QSize,
        Qt,
        QThreadPool,
        QTimer,
        QUrl,
        Signal,
    )
    from PySide6.QtGui import (
        QAction,
        QColor,
        QDesktopServices,
        QImage,
        QKeySequence,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QPolygonF,
        QSurfaceFormat,
    )
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractSpinBox,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGraphicsPathItem,
        QGraphicsEllipseItem,
        QGraphicsItem,
        QGraphicsLineItem,
        QGraphicsPixmapItem,
        QGraphicsPolygonItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QLayout,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QTabWidget,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised by launcher preflight
    raise SystemExit(
        "PySide6 is required. Run `uv sync --extra review-ui` in the ff_ac repository."
    ) from exc

from darsia.fluidflower.tools import calibration_pairwise_review as review
from darsia.fluidflower.tools.phase_view_overlay import (
    UNRESOLVED_RGB, display_phase_overlay, overlay_asset_layer,
)
from darsia.fluidflower.phase_photographic_assets import BASE_DISPLAY_MODES, DISPLAY_MODES, DISPLAY_MODE_LABELS, change_display_available, display_modes_available, resolve_display_mode, resolve_photo_sand_domain, load_original_display_rgb
from darsia.fluidflower.tools.phase_review_area_sort import (
    peak_phase_area_percent,
    sorted_phase_area_runs,
)
from darsia.fluidflower.tools import schema2_combined_review as combined
from darsia.fluidflower.tools import phase_injection_references as injection_refs
from darsia.fluidflower.phase_labels import (
    MANUAL_CATEGORY_TO_PHASE_CHOICE,
    PHASE_CHOICE_TO_MANUAL_CATEGORY,
    VISIBLE_PHASE_CHOICES,
    VISIBLE_PHASE_LABELS,
    assess_model_phase,
    visible_phase_payload,
)
from darsia.fluidflower.phase_physics_layer import DIFFERENCE_LEGEND


DEFAULT_FFAC_ROOT = (
    configured_tool_path("ffac", "calibration_logs")
    / "schema5_crossseries_reviewed_physics_blind_20260831"
)
DEFAULT_FFUM_ROOT = (
    configured_tool_path("ffum", "analysis_v2")
    / "schema5_crossseries_reviewed_physics_blind_20260831"
)
DEFAULT_COMBINED_ROOT = configured_tool_path("ffac", "review_campaign")
DEFAULT_VALIDATION_FFAC_ROOT = (
    configured_tool_path("ffac", "calibration_logs")
    / "schema2_hierarchical_blind_v1_20260829"
)
DEFAULT_VALIDATION_FFUM_ROOT = (
    configured_tool_path("ffum", "analysis_v2")
    / "schema2_hierarchical_blind_v1_20260830_injection_anchor_corrected"
)
DEFAULT_VALIDATION_COMBINED_ROOT = (
    configured_tool_path("ffac", "combined_reviews")
    / "schema2_hierarchical_blind_v1_20260829_combined_review"
)
DEFAULT_COMPARISON_FFAC_ROOT = (
    configured_tool_path("ffac", "calibration_logs")
    / "schema4_crossseries_mil_blind_v1_20260830"
)
DEFAULT_COMPARISON_FFUM_ROOT = (
    configured_tool_path("ffum", "analysis_v2")
    / "schema4_crossseries_mil_blind_v1_20260830"
)
DEFAULT_COMPARISON_COMBINED_ROOT = (
    configured_tool_path("ffac", "combined_reviews")
    / "schema4_crossseries_mil_blind_v1_20260830_combined_review"
)
DEFAULT_CACHE_ROOT = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    / "ff_ac"
    / "calibration_review_cache"
)
PHYSICS_LAYER_CANDIDATE = "phase-model-20260930-track1-natural-lifts-invalidfilled-ffac"
PHYSICS_LAYER_EXPORT_ROOT = configured_tool_path("ffac", "physics_layer_exports")


def _is_track1_physics_candidate(candidate_id: str, run: str) -> bool:
    """Studio appends run/revision identity to the selected model folder name."""
    value = str(candidate_id)
    return value == PHYSICS_LAYER_CANDIDATE or value.startswith(
        f"{PHYSICS_LAYER_CANDIDATE}__ffac__{run.casefold()}__"
    )
DEFAULT_USER_DATA_ROOT = (
    Path.home() / "Library" / "Application Support" / "FluidFlower Review"
    if sys.platform == "darwin"
    else Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    / "FluidFlower Review"
)
OFFLINE_BUNDLE_FORMAT = "fluidflower-review-desktop-offline-v1"
DESKTOP_UI_VERSION = 50

ISOLATED_COMPONENT_MAX_AREA_FRACTION = (
    review.ISOLATED_COMPONENT_MAX_AREA_FRACTION
)
ISOLATED_COMPONENT_COLOR_TOLERANCE = (
    review.ISOLATED_COMPONENT_COLOR_TOLERANCE
)
ISOLATED_COMPONENT_MIN_SEPARATION_FRACTION = (
    review.ISOLATED_COMPONENT_MIN_SEPARATION_FRACTION
)


P1_FINAL2_OVERLAYS = frozenset(("below_sand_floor", "removed_floor_connectivity", "floor_contact_fill_v2", "removed_gas_envelope"))
P1_FINAL3_OVERLAYS = P1_FINAL2_OVERLAYS | {"removed_contact_fill_b"}
SAVED_PROVENANCE_LAYERS = P1_FINAL3_OVERLAYS | {"removed_top", "removed_floor"}


def phase_diagnostic_labels(candidates: Sequence[review.Candidate]) -> dict[str, tuple[str, str]]:
    """Describe archived diagnostic masks using metadata loaded with candidates."""
    labels = {
        "raw": ("Raw optical prediction",
                "Raw optical prediction, before physical support: green = aqueous CO2; pink = free CO2."),
        "context_added": ("Context: added aqueous (green)",
                          "Green = aqueous CO2 added by spatial and temporal support; other pixels are transparent."),
        "context_removed": ("Context: suppressed presence (red)",
                            "Red = CO2 detections suppressed by spatial and temporal support; not a phase class."),
        "context_uncertain": ("Context: unresolved evidence (cyan)",
                              "Cyan = evidence without a resolved source/history connection; not a phase class."),
        "below_sand_floor": ("Removed below sand floor",
                             "Pixels outside the photo-verified pooled sand floor. Provenance only; not a phase class."),
        "removed_floor_connectivity": ("Removed by floor connectivity",
                                       "Disconnected floor CO2 removed by the fixed connectivity rule. Provenance only."),
        "floor_contact_fill_v2": ("Floor contact fill",
                                  "Aqueous CO2 contact added under the short-gap or pre-injection carryover rule. Bottom-strip mass remains blocked."),
        "removed_contact_fill_b": ("Contact fill (b) removed",
                                   "Carryover-based contact-fill pixels reverted to no CO2. Provenance only; bottom-strip mass remains blocked."),
        "removed_gas_envelope": ("Gas removed by injection envelope",
                                 "Free gas changed to aqueous CO2 outside the fixed injection-time envelope. CO2 union is preserved by this step."),
        "removed_top": ("Removed by top-surface rule",
                        "Orange shows P1 CO2 pixels removed by the fixed 60 mm top-surface rule. Provenance only; these pixels are no longer CO2 in P1-final."),
        "removed_floor": ("Floor fill (removed)",
                          "Blue shows P1 floor-contact fill removed from P1-final. Provenance only; bottom-strip mass remains blocked."),
    }
    labels["source_support"] = ("Unresolved source support (cyan)",
                                "Cyan = unresolved source support, including water, rig and chart outside the sand. This diagnostic is not a CO2 prediction.")
    if candidates and all((getattr(candidate, "quality_flags", None) or {}).get("p1_final3_revision") for candidate in candidates):
        labels["floor_contact_fill_v2"] = ("Floor contact fill ≤ 1.5 mm",
                                          "Aqueous CO2 added only across gaps of at most 1.5 mm; reason (b) is removed. Bottom-strip mass remains blocked.")
    native = [getattr(candidate, "phase_support_format", "") == "native-continuous-reference-gate-v1"
              for candidate in candidates]
    if native and all(native):
        labels.update({
            "context_added": ("Boundary: added vs V12-C4 (green)",
                              "Green = aqueous CO2 added relative to V12-C4 on the same frame; other pixels are transparent."),
            "context_removed": ("Boundary: removed vs V12-C4 (red)",
                                "Red = CO2 removed relative to V12-C4 on the same frame; not a phase class."),
            "context_uncertain": ("Context: unresolved evidence (cyan)",
                                  "Cyan = unresolved inherited or reference evidence; not a phase class."),
        })
    elif any(native):
        for key, color, action in (("context_added", "Green", "added aqueous CO2"),
                                    ("context_removed", "Red", "removed CO2")):
            labels[key] = (labels[key][0], f"{color} = {action}. For the native edge candidate, changes are "
                           "relative to V12-C4 on the same frame; the other candidate shows changes from spatial "
                           "and temporal support. Diagnostic layer, not a phase class.")
    return labels


def normalize_phase_overlay_alpha(image: QImage) -> QImage:
    """Make the filled phase class opaque while preserving antialiased edges.

    Older review assets encode the main fill opacity in the PNG and the desktop
    viewer applies a second opacity. Normalizing the dominant non-zero alpha to
    255 leaves the UI slider as the single opacity control.
    """

    if image.isNull() or not image.hasAlphaChannel():
        return image
    normalized = image.convertToFormat(QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(normalized.bits(), dtype=np.uint8).reshape(
        normalized.height(),
        normalized.bytesPerLine(),
    )
    alpha = pixels[:, 3 : normalized.width() * 4 : 4]
    rgb = pixels[:, : normalized.width() * 4].reshape(normalized.height(), normalized.width(), 4)[..., :3]
    phase_pixels = ~np.all(rgb == UNRESOLVED_RGB, axis=-1)
    counts = np.bincount(alpha[phase_pixels].reshape(-1), minlength=256)
    counts[0] = 0
    dominant_alpha = int(np.argmax(counts))
    if dominant_alpha <= 0 or dominant_alpha >= 255:
        return normalized
    scaled = np.minimum(
        255,
        (alpha.astype(np.uint16) * 255 + dominant_alpha // 2)
        // dominant_alpha,
    )
    # Source-support diagnostics retain their encoded opacity and do not set
    # the phase-fill normalization, including unresolved-only start frames.
    alpha[phase_pixels] = scaled.astype(np.uint8)[phase_pixels]
    return normalized

TRAINING_CAMPAIGN_ID = "schema5_crossseries_reviewed_physics_blind_20260831"
VALIDATION_CAMPAIGN_ID = "schema2_hierarchical_blind_v1_20260829"
COMPARISON_CAMPAIGN_ID = "schema4_crossseries_mil_blind_v1_20260830"
DEFAULT_TRAINING_FFAC_RUNS = ("ac32",)
DEFAULT_TRAINING_FFUM_RUNS = ("run18",)


@dataclass(frozen=True)
class ReviewWorkspace:
    """One isolated review purpose with its own inputs and decision store."""

    key: str
    title: str
    short_label: str
    description: str
    campaign_id: str
    ffac_root: Path
    ffum_root: Path
    combined_root: Path
    ffac_runs: tuple[str, ...] | None = None
    ffum_runs: tuple[str, ...] | None = None
    blind: bool = False
    decision_dir: Path | None = None
    seed_review_dir: Path | None = None
    offline: bool = False
    approval_only: bool = False
    review_surface: str = "default"
    single_model: bool = False
    review_session_id: str = ""
    target_model_id: str = ""
    evidence_audit_path: str = ""
    selected_candidates: Mapping[str, str] | None = None
    read_only: bool = False

    @property
    def output_dir(self) -> Path:
        return resolve_review_path(self.decision_dir or self.combined_root / "manual_pairwise_review")

    def run_filters(self) -> dict[str, set[str]]:
        return {
            dataset: {str(run).casefold() for run in runs}
            for dataset, runs in (
                ("ffac", self.ffac_runs),
                ("ffum", self.ffum_runs),
            )
            if runs is not None
        }

PHASE_CHOICES: tuple[tuple[str, str], ...] = tuple(
    (choice.key, choice.label) for choice in VISIBLE_PHASE_CHOICES
)
PHASE_LABELS = dict(VISIBLE_PHASE_LABELS) | {"ignore_region": "Ignored region"}

ANNOTATION_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("confirmed_only_aqueous_co2", "Only aqueous CO2", "aqueous"),
    (
        "confirmed_predominantly_aqueous_co2",
        "Mostly aqueous CO2, some free CO2",
        "total",
    ),
    ("confirmed_mixed_free_aqueous_co2", "Mixture: free + aqueous CO2", "total"),
    (
        "confirmed_predominantly_free_gas",
        "Mostly free CO2, some aqueous CO2",
        "total",
    ),
    ("confirmed_only_free_gas", "Only free CO2", "gas"),
    ("confirmed_mixed_aqueous_water", "Mixture: aqueous CO2 + untouched water", "aqueous"),
    (
        "confirmed_mixed_free_aqueous_water",
        "Mixture: free + aqueous CO2 + untouched water",
        "total",
    ),
    ("confirmed_no_co2", "Confirmed no CO2", "total"),
    ("confirmed_no_free_gas_in_selected_area", "No free gas anywhere in selected area", "gas"),
    ("confirmed_no_aqueous_co2_in_selected_area", "No aqueous CO2 anywhere in selected area", "aqueous"),
    ("unsupported_free_gas_classification", "Incorrect free-gas prediction in this model", "gas"),
    ("unsupported_aqueous_co2_classification", "Incorrect aqueous-CO2 prediction in this model", "aqueous"),
    (
        "likely_dense_aqueous_co2_misclassified_as_gas",
        "Likely dense aqueous CO2 misclassified as gas",
        "gas",
    ),
    ("implausible_post_injection_downward_gas_migration", "Implausible downward gas migration after injection", "gas"),
    ("implausible_late_free_gas_persistence", "Implausible late free-gas persistence", "gas"),
    ("implausible_gas_location", "Implausible gas location", "gas"),
    ("implausible_aqueous_location", "Implausible aqueous location", "aqueous"),
    ("missed_visible_co2", "Missed visible CO2", "total"),
    ("missed_visible_gas", "Missed visible gas", "gas"),
    ("disconnected_noise", "Disconnected noise", "total"),
    ("facies_edge_artifact", "Facies-edge artifact", "total"),
    (
        "incomplete_wash_indicator_carryover",
        "Indicator carryover from incomplete washing",
        "unspecified",
    ),
    ("baseline_or_flush_artifact", "Baseline or incomplete-flush artifact", "total"),
    ("lighting_or_color_correction_artifact", "Lighting or colour-correction artifact", "total"),
    (
        "phase_ambiguous_strong_color_response",
        "CO2 visible; phase split uncertain",
        "unspecified",
    ),
    ("phase_not_assessable", "Uncertain / cannot assess", "unspecified"),
    ("phase_signal_artifact", "Artifact / not a valid phase signal", "unspecified"),
    ("other", "Other", "unspecified"),
)
ANNOTATION_LABELS = {key: label for key, label, _phase in ANNOTATION_CATEGORIES}
ANNOTATION_PHASES = {key: phase for key, _label, phase in ANNOTATION_CATEGORIES}
# Legacy categories remain readable and trainable, but are not offered for new
# labels because they combined pure and dominant regions.
ANNOTATION_LABELS |= {
    "confirmed_aqueous_co2": "Legacy: mostly aqueous CO2",
    "confirmed_free_gas": "Legacy: mostly or entirely free CO2",
}
ANNOTATION_PHASES |= {
    "confirmed_aqueous_co2": "aqueous",
    "confirmed_free_gas": "gas",
}
PHASE_CHOICE_TO_ANNOTATION = {
    **PHASE_CHOICE_TO_MANUAL_CATEGORY,
}

BLIND_GAS_SESSION = "blind_gas_validation_v1"
BLIND_GAS_ARTIFACT_LABELS = (
    ("baseline_or_flush_artifact", "Baseline or incomplete-flush artifact"),
    ("lighting_or_color_correction_artifact", "Lighting or colour-correction artifact"),
)
BLIND_GAS_CATEGORIES = frozenset(
    key for key, _label in tuple(
        (PHASE_CHOICE_TO_ANNOTATION[choice.key], choice.label)
        for choice in VISIBLE_PHASE_CHOICES
    )
) | frozenset({
    "confirmed_no_free_gas_in_selected_area",
    "confirmed_no_aqueous_co2_in_selected_area",
    "incomplete_wash_indicator_carryover",
    "baseline_or_flush_artifact",
    "lighting_or_color_correction_artifact",
    "phase_signal_artifact",
})


def annotation_phase_payload(category: str) -> dict[str, Any]:
    """Return shared phase fields for a phase-label annotation category."""

    choice = MANUAL_CATEGORY_TO_PHASE_CHOICE.get(str(category))
    return visible_phase_payload(choice) if choice else {}


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def timestamp_utc(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def reviewer_identifier(name: str) -> str:
    from .calibration_pairwise_review import _reviewer_identifier
    return _reviewer_identifier(name)


def reviewer_output_dir(save_root: Path, reviewer_name: str, workspace: ReviewWorkspace) -> Path:
    root = (
        Path(save_root).expanduser().resolve()
        / reviewer_identifier(reviewer_name)
        / workspace.campaign_id
        / workspace.key
    )
    if workspace.review_session_id:
        root = root / "sessions" / workspace.review_session_id
    return resolve_review_path(root / "manual_pairwise_review")


def local_reviewer_workspace(
    workspace: ReviewWorkspace,
    save_root: Path,
    reviewer_name: str,
) -> ReviewWorkspace:
    target = reviewer_output_dir(save_root, reviewer_name, workspace)
    source_output = workspace.output_dir.resolve()
    seed = workspace.seed_review_dir
    if (
        not workspace.review_session_id
        and source_output != target
        and source_output.is_dir()
    ):
        seed = source_output
    elif workspace.review_session_id:
        seed = None
    return replace(
        workspace,
        decision_dir=target,
        seed_review_dir=seed,
        offline=True,
    )


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reviewer_profile_path(cache_root: Path) -> Path:
    return Path(cache_root).resolve() / "reviewer_profile.json"


def load_reviewer_profile(cache_root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(reviewer_profile_path(cache_root).read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, Mapping) else {}
    except (OSError, ValueError, TypeError):
        return {}


def save_reviewer_profile(
    cache_root: Path,
    *,
    reviewer_name: str | None = None,
    save_root: Path | None = None,
) -> dict[str, Any]:
    payload = load_reviewer_profile(cache_root)
    if reviewer_name is not None:
        payload["reviewer"] = str(reviewer_name).strip()
        payload["reviewer_id"] = reviewer_identifier(str(reviewer_name))
    if save_root is not None:
        payload["save_root"] = str(Path(save_root).expanduser().resolve())
    payload["updated_utc"] = utc_now()
    atomic_json(reviewer_profile_path(cache_root), payload)
    return payload


def effective_reviewer_settings(args: argparse.Namespace) -> tuple[str, Path]:
    """Resolve command-line settings while preserving an explicit local choice."""

    profile = load_reviewer_profile(args.cache_root)
    reviewer_name = str(args.reviewer or profile.get("reviewer") or "").strip()
    configured_root = Path(args.local_review_root).expanduser().resolve()
    default_root = (DEFAULT_USER_DATA_ROOT / "reviews").expanduser().resolve()
    remembered_root = str(profile.get("save_root") or "").strip()
    save_root = (
        Path(remembered_root).expanduser().resolve()
        if remembered_root and configured_root == default_root
        else configured_root
    )
    return reviewer_name, save_root


def ensure_writable_directory(path: Path) -> Path:
    """Create a directory and prove that a durable local write succeeds."""

    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    probe = resolved / f".fluidflower-write-test-{uuid.uuid4().hex}.tmp"
    try:
        with probe.open("x", encoding="ascii", newline="") as stream:
            stream.write("ok\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        probe.unlink(missing_ok=True)
    return resolved


def _answer_key(item: Mapping[str, Any]) -> str:
    candidate = str(item.get("candidate") or "A").upper()
    proposal_id = str(item.get("proposal_id") or "").strip()
    return f"{candidate}:{proposal_id}" if proposal_id else ""


def _annotation_key(item: Mapping[str, Any], index: int) -> str:
    annotation_id = str(item.get("id") or "").strip()
    if annotation_id:
        return annotation_id
    encoded = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
    return f"legacy-{index}-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:20]}"


def merge_review_recovery(
    durable: Mapping[str, Any] | None,
    recovery: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge a newer local recovery record without dropping durable labels."""

    durable = dict(durable or {})
    recovery = dict(recovery or {})
    merged = deepcopy(durable)
    for key, value in recovery.items():
        if key not in {
            "annotations",
            "phase_quiz_answers",
            "phase_quiz_selected_ids",
            "phase_quiz_rounds",
        }:
            merged[key] = deepcopy(value)

    answer_by_key: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for source in (
        durable.get("phase_quiz_answers") or [],
        recovery.get("phase_quiz_answers") or [],
    ):
        for raw in source:
            if not isinstance(raw, Mapping):
                continue
            key = _answer_key(raw)
            if key:
                answer_by_key[key] = deepcopy(dict(raw))
    deleted_answers = {
        str(value) for value in recovery.get("deleted_phase_quiz_answer_keys") or []
    }
    merged["phase_quiz_answers"] = [
        value for key, value in answer_by_key.items() if key not in deleted_answers
    ]

    annotation_by_key: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for source in (
        durable.get("annotations") or [],
        recovery.get("annotations") or [],
    ):
        for index, raw in enumerate(source):
            if not isinstance(raw, Mapping):
                continue
            key = _annotation_key(raw, index)
            annotation_by_key[key] = deepcopy(dict(raw))
    deleted_annotations = {
        str(value) for value in recovery.get("deleted_annotation_ids") or []
    }
    merged["annotations"] = [
        value
        for key, value in annotation_by_key.items()
        if key not in deleted_annotations
    ]

    selected_ids: list[str] = []
    for source in (
        durable.get("phase_quiz_selected_ids") or [],
        recovery.get("phase_quiz_selected_ids") or [],
    ):
        for raw in source:
            proposal_id = str(raw or "")
            if proposal_id and proposal_id not in selected_ids:
                selected_ids.append(proposal_id)
    for answer in merged["phase_quiz_answers"]:
        proposal_id = str(answer.get("proposal_id") or "")
        if proposal_id and proposal_id not in selected_ids:
            selected_ids.append(proposal_id)
    merged["phase_quiz_selected_ids"] = selected_ids
    merged["phase_quiz_rounds"] = [
        deepcopy(dict(item))
        for source in (
            durable.get("phase_quiz_rounds") or [],
            recovery.get("phase_quiz_rounds") or [],
        )
        for item in source
        if isinstance(item, Mapping)
    ]
    merged["phase_quiz_per_candidate_limit"] = len(selected_ids)
    merged["deleted_phase_quiz_answer_keys"] = sorted(deleted_answers)
    merged["deleted_annotation_ids"] = sorted(deleted_annotations)
    return merged


def normalized_point(raw: Any) -> QPointF | None:
    if isinstance(raw, str):
        fields = raw.replace(",", " ").split()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        fields = list(raw)
    else:
        return None
    if len(fields) != 2:
        return None
    try:
        return QPointF(float(fields[0]), float(fields[1]))
    except (TypeError, ValueError):
        return None


def clean_selected_ids(
    pool: Sequence[Mapping[str, Any]],
    raw_ids: Sequence[str] | None,
    answers: Sequence[Mapping[str, Any]],
    initial_count: int,
) -> list[str]:
    valid = {str(item.get("proposal_id") or "") for item in pool}
    selected: list[str] = []
    for raw_id in raw_ids or ():
        proposal_id = str(raw_id or "")
        if proposal_id in valid and proposal_id not in selected:
            selected.append(proposal_id)
    answered = [
        str(item.get("proposal_id") or "")
        for item in answers
        if str(item.get("candidate") or "").upper() == "A"
        and str(item.get("proposal_id") or "") in valid
    ]
    if not selected:
        selected.extend(dict.fromkeys(answered))
    if not selected:
        selected.extend(
            str(item.get("proposal_id") or "")
            for item in pool[: max(0, int(initial_count))]
        )
    for proposal_id in answered:
        if proposal_id not in selected:
            selected.append(proposal_id)
    return selected


def clean_quiz_rounds(
    raw_rounds: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Remove stale proposal ids while preserving the current quiz ordering."""

    valid = set(map(str, selected_ids))
    seen: set[str] = set()
    rounds: list[dict[str, Any]] = []
    for index, raw_round in enumerate(raw_rounds):
        ids: list[str] = []
        for raw_id in raw_round.get("proposal_ids") or ():
            proposal_id = str(raw_id or "")
            if proposal_id in valid and proposal_id not in seen:
                ids.append(proposal_id)
                seen.add(proposal_id)
        if ids:
            rounds.append(
                {
                    "round": int(raw_round.get("round") or index + 1),
                    "selection_mode": str(
                        raw_round.get("selection_mode")
                        or "adaptive_phase_facies_spatial_balance"
                    ),
                    "proposal_ids": ids,
                    "created_utc": str(raw_round.get("created_utc") or ""),
                }
            )
    missing = [proposal_id for proposal_id in selected_ids if proposal_id not in seen]
    if missing:
        rounds.insert(
            0,
            {
                "round": 1,
                "selection_mode": "legacy_or_initial_batch",
                "proposal_ids": missing,
                "created_utc": "",
            },
        )
    return rounds


def reconcile_phase_quiz_answers(
    pool: Sequence[Mapping[str, Any]],
    raw_answers: Sequence[Mapping[str, Any]],
    raw_annotations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Preserve labels from retired quiz pools as versioned manual regions."""

    valid_ids = {str(item.get("proposal_id") or "") for item in pool}
    answers: list[dict[str, Any]] = []
    annotations = [deepcopy(dict(item)) for item in raw_annotations]
    annotation_ids = {str(item.get("id") or "") for item in annotations}
    migrated = 0
    for raw_answer in raw_answers:
        answer = deepcopy(dict(raw_answer))
        proposal_id = str(answer.get("proposal_id") or "")
        candidate = str(answer.get("candidate") or "A").upper()
        if candidate == "A" and proposal_id in valid_ids:
            answers.append(answer)
            continue

        human_choice = str(answer.get("human_choice") or "not_assessable")
        category = PHASE_CHOICE_TO_ANNOTATION.get(human_choice, "other")
        annotation_id = "legacy-quiz-" + hashlib.sha256(
            f"{candidate}:{proposal_id}".encode("utf-8")
        ).hexdigest()[:20]
        if annotation_id in annotation_ids:
            continue
        annotation_ids.add(annotation_id)
        original_comment = str(answer.get("comment") or "").strip()
        migration_note = (
            f"Migrated from retired phase-quiz proposal {proposal_id}; "
            f"original class: {human_choice}."
        )
        annotations.append(
            {
                "id": annotation_id,
                "candidate": candidate if candidate in {"A", "B"} else "A",
                "frame_index": int(answer.get("frame_index") or 0),
                "elapsed_h": float(answer.get("elapsed_h") or 0.0),
                "category": category,
                "phase": ANNOTATION_PHASES.get(category, "unspecified"),
                "severity": "minor",
                "confidence": str(answer.get("confidence") or "unspecified"),
                "contours_visible": False,
                "shape": str(answer.get("shape") or "freehand"),
                "points": deepcopy(answer.get("points") or []),
                "x": float(answer.get("x") or 0.0),
                "y": float(answer.get("y") or 0.0),
                "width": float(answer.get("width") or 0.0),
                "height": float(answer.get("height") or 0.0),
                "comment": (
                    f"{migration_note} {original_comment}".strip()[:500]
                ),
            }
        )
        migrated += 1
    return answers, annotations, migrated


def contour_overlay_variant_path(combined_path: Path, mode: str) -> Path:
    """Resolve one phase-specific overlay, falling back to the combined asset."""

    normalized = str(mode or "all").casefold()
    if normalized not in {"gas", "aqueous", "mixed"}:
        return combined_path
    variant = combined_path.parent / normalized / combined_path.name
    return variant if variant.is_file() else combined_path


def partition_quiz_region_indices(
    proposals: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int]]:
    answered_ids = {
        str(item.get("proposal_id") or "")
        for item in answers
        if str(item.get("candidate") or "").upper() == "A"
    }
    new_indices: list[int] = []
    validated_indices: list[int] = []
    for index, proposal in enumerate(proposals):
        proposal_id = str(proposal.get("proposal_id") or "")
        target = validated_indices if proposal_id in answered_ids else new_indices
        target.append(index)
    return new_indices, validated_indices


def quiz_region_view_indices(
    proposals: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
    *,
    validated: bool,
    sort_mode: str = "time_asc",
    class_filter: str = "all",
) -> list[int]:
    """Return stable proposal indices for one filtered quiz list."""

    answer_by_id = {
        str(item.get("proposal_id") or ""): item
        for item in answers
        if str(item.get("candidate") or "").upper() == "A"
    }
    new_indices, validated_indices = partition_quiz_region_indices(
        proposals,
        answers,
    )
    indices = list(validated_indices if validated else new_indices)
    if validated and class_filter != "all":
        indices = [
            index
            for index in indices
            if str(
                answer_by_id.get(
                    str(proposals[index].get("proposal_id") or ""),
                    {},
                ).get("human_choice")
                or ""
            )
            == class_filter
        ]

    def time_key(index: int) -> tuple[float, int, int]:
        proposal = proposals[index]
        return (
            float(proposal.get("elapsed_h") or 0.0),
            int(proposal.get("frame_index") or 0),
            index,
        )

    if sort_mode == "time_desc":
        return sorted(
            indices,
            key=lambda index: (
                -time_key(index)[0],
                -time_key(index)[1],
                index,
            ),
        )
    if sort_mode == "class_asc" and validated:
        return sorted(
            indices,
            key=lambda index: (
                PHASE_LABELS.get(
                    str(
                        answer_by_id.get(
                            str(proposals[index].get("proposal_id") or ""),
                            {},
                        ).get("human_choice")
                        or ""
                    ),
                    "Unknown",
                ).casefold(),
                *time_key(index),
            ),
        )
    if sort_mode == "added":
        return indices
    return sorted(indices, key=time_key)


def annotation_view_indices(
    annotations: Sequence[Mapping[str, Any]],
    *,
    sort_mode: str = "time_asc",
    class_filter: str = "all",
) -> list[int]:
    """Return stable annotation indices for filtering, display and editing."""

    indices = [
        index
        for index, item in enumerate(annotations)
        if class_filter == "all"
        or str(item.get("category") or "other") == class_filter
    ]

    def time_key(index: int) -> tuple[float, int, int]:
        item = annotations[index]
        frame = int(item.get("frame_index") or 0)
        raw_elapsed = item.get("elapsed_h")
        elapsed = float(raw_elapsed) if raw_elapsed is not None else float(frame)
        return elapsed, frame, index

    if sort_mode == "time_desc":
        return sorted(
            indices,
            key=lambda index: (
                -time_key(index)[0],
                -time_key(index)[1],
                index,
            ),
        )
    if sort_mode == "class_asc":
        return sorted(
            indices,
            key=lambda index: (
                ANNOTATION_LABELS.get(
                    str(annotations[index].get("category") or "other"),
                    "Other",
                ).casefold(),
                *time_key(index),
            ),
        )
    if sort_mode == "created":
        return indices
    return sorted(indices, key=time_key)


@dataclass(frozen=True)
class DeferredImage:
    """Stable in-memory identity with filesystem work deferred to an image worker."""

    identity: str
    resolve: Callable[[], Path | QImage]
    normalize_overlay: bool = False
    reference_phase: bool = False
    prefetch_resolve: Callable[[], Path | None] | None = None
    display_domain: Callable[[], np.ndarray] | None = None
    display_filled: Callable[[], np.ndarray | None] | None = None
    display_floor_contact: Callable[[], np.ndarray | None] | None = None
    display_l1_neighbourhood: Callable[[], np.ndarray | None] | None = None
    display_masks: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None
    display_layer: str | None = None
    display_binding: str = ""
    display_dependencies: tuple[str, ...] = ()
    photo_binding: str = ""
    photo_dependencies: tuple[str, ...] = ()
    photo_revision: Mapping[str, Any] | None = None


def _presentation_assets(window):
    """Keep light descriptors only for the current loaded, source-bound pair."""
    context = (str(window.current_review_root), window.current_run,
               id(window.current_pair), id(getattr(window, "current_pair_object", None)))
    if getattr(window, "_presentation_asset_context", None) != context:
        window._presentation_asset_context = context
        window._presentation_assets = {}
    return window._presentation_assets


class ReviewImageAssets:
    """Shared, I/O-free asset requests for both native review surfaces."""

    def _selected_photo_asset(self,index,mode,*,reference=False):
        if "ffac_selected_policy_a_sources" not in (self.current_pair or {}):
            return None
        from darsia.fluidflower.phase_selected_policy_review import resolve_photograph
        root,run=self.current_review_root,self.current_run
        return ReviewImageAssets._bound_photo_asset(self,DeferredImage(
            f"selected-file-photo:{root}:{run}:{index}:{reference}:{mode}",
            lambda: resolve_photograph(root,run,index,mode=mode,reference=reference)),index,mode,reference=reference)

    def _annotation_display_frame(self,annotation):
        from darsia.fluidflower.phase_annotation_binding import presentation_frame
        return presentation_frame(annotation,getattr(self,"_selected_annotation_binding",None))

    def _annotation_display_payload(self,annotation):
        from darsia.fluidflower.phase_annotation_binding import presentation_annotation
        return presentation_annotation(annotation,getattr(self,"_selected_annotation_binding",None))

    def _add_brightness_control(self, layout) -> None:
        self.display_mode = "natural"
        self.brightness = QPushButton(DISPLAY_MODE_LABELS[self.display_mode])
        self.brightness.setToolTip("Cycle natural -> lift x4.3 -> lift x7.1 -> change (flat field). Photographic display only; model input and RAW detail stay unchanged.")
        self.brightness.setEnabled(False)
        self.brightness.clicked.connect(self._cycle_brightness)
        layout.addWidget(self.brightness)

    def _brightness_modes(self) -> tuple[str, ...]:
        loaded_modes = getattr(self, "_loaded_display_modes", None)
        if loaded_modes is not None:
            return loaded_modes
        available = display_modes_available({"frame_sources": (self.current_pair or {}).get("frame_sources") or []})
        if not available:
            return ("natural",)
        from darsia.fluidflower.phase_rig_raw_display import rig_raw_display_available
        if rig_raw_display_available(self.current_pair or {}, frame_index=getattr(self, "frame_index", 0)):
            return DISPLAY_MODES
        root, run = getattr(self, "current_review_root", None), getattr(self, "current_run", None)
        count = len((self.current_pair or {}).get("frame_sources") or [])
        if change_display_available(root, run, frame_count=count):
            return DISPLAY_MODES
        return BASE_DISPLAY_MODES

    def _update_brightness_control(self) -> None:
        modes = self._brightness_modes()
        if self.display_mode not in modes:
            self.display_mode = "natural"
        self.brightness.setText(DISPLAY_MODE_LABELS[self.display_mode])
        self.brightness.setEnabled(len(modes) > 1 and not self.raw_detail.isChecked()
                                   and (getattr(self, "active_injection_reference", None) is None
                                        or getattr(self, "_saved_rig_reference_index", lambda: None)() is not None))
        if hasattr(self, "quick_photo_controls"):
            self._update_quick_photo_controls()

    def _cycle_brightness(self) -> None:
        if hasattr(self, "original_photo"):
            self.original_photo.setChecked(False)
        modes = self._brightness_modes()
        if self.display_mode not in modes:
            self.display_mode = "natural"
        self.display_mode = modes[(modes.index(self.display_mode) + 1) % len(modes)]
        self._update_brightness_control()
        self._request_frame()

    def _display_settings_changed(self) -> None:
        self._update_brightness_control()
        self._request_frame()

    def _photo_sand_loader(self, index=None, *, reference=False):
        from darsia.fluidflower.raw_photographic_review import raw_review_row
        rows = (self.current_pair or {}).get("frame_sources") or []
        row_index = 0 if reference else index
        if row_index is None or not 0 <= row_index < len(rows):
            return None
        # A RAW review's photo support is a canvas PNG beside its photographs.
        raw = raw_review_row(rows[row_index])
        if not (rows[row_index].get("v5_display_modes") or raw):
            return None
        root, run = self.current_review_root, self.current_run
        full_photo = bool(rows[row_index].get("rig_domain_revision")) or raw
        return lambda: resolve_photo_sand_domain(root, run, index, reference=reference, canvas=full_photo)

    def _filled_display_loader(self, index):
        rows = (self.current_pair or {}).get("frame_sources") or []
        if index is None or not 0 <= index < len(rows) or not rows[index].get("rig_domain_revision"):
            return None
        if "ffac_selected_policy_a_sources" in (self.current_pair or {}):
            revision=rows[index]["rig_domain_revision"]
            def selected_filled():
                from darsia.fluidflower.phase_saved_review_metadata import load_rig_classes
                fields,_=load_rig_classes(revision,base_dir=self.current_review_root)
                return fields["filled_observed"] | fields["filled_unobserved"]
            return selected_filled
        from darsia.fluidflower.phase_filled_display import resolve_filled_display_mask
        root, run = self.current_review_root, self.current_run
        return lambda: resolve_filled_display_mask(root, run, index, canvas=True)

    def _frame_asset(self, index: int) -> DeferredImage:
        mode=("original" if hasattr(self,"original_photo") and self.original_photo.isChecked() else
              "raw_detail" if getattr(self,"raw_detail",None) is not None and self.raw_detail.isChecked() else
              getattr(self,"display_mode","natural"))
        if asset := ReviewImageAssets._selected_photo_asset(self,index,mode,reference=False):
            return asset
        root, run = self.current_review_root, self.current_run
        if hasattr(self, "original_photo") and self.original_photo.isChecked():
            return self._original_display_asset(index)
        raw_detail = bool(getattr(self, "raw_detail", None) is not None
                          and self.raw_detail.isChecked())
        if raw_detail:
            if asset := ReviewImageAssets._rig_raw_display_asset(self, index, "raw_detail"):
                return asset
            from darsia.fluidflower.phase_photographic_assets import resolve_raw_detail
            return ReviewImageAssets._bound_photo_asset(self, DeferredImage(
                f"v5-raw-detail:frame:{root}:{run}:{index}",
                lambda: resolve_raw_detail(root, run, index),
            ), index, "raw_detail")
        from darsia.fluidflower.raw_photographic_review import raw_review_row
        rows = (self.current_pair or {}).get("frame_sources") or []
        if 0 <= index < len(rows) and (rows[index].get("v5_display_modes") or raw_review_row(rows[index])):
            mode = getattr(self, "display_mode", "natural")
            if mode == "change" and (asset := ReviewImageAssets._rig_raw_display_asset(self, index, mode)):
                return asset
            return ReviewImageAssets._bound_photo_asset(self,
                DeferredImage(f"v5-display:{mode}:frame:{root}:{run}:{index}",
                              lambda: resolve_display_mode(root, run, index, mode)), index, mode)
        return ReviewImageAssets._bound_photo_asset(self, DeferredImage(
            f"frame:{root}:{run}:{index}",
            lambda: review._native_frame_path(root, run, index),
        ), index, "natural")

    def _reference_asset(self) -> DeferredImage:
        mode=("original" if hasattr(self,"original_photo") and self.original_photo.isChecked() else
              "raw_detail" if getattr(self,"raw_detail",None) is not None and self.raw_detail.isChecked() else
              getattr(self,"display_mode","natural"))
        if asset := ReviewImageAssets._selected_photo_asset(self,None,mode,reference=True):
            return asset
        root, run = self.current_review_root, self.current_run
        if hasattr(self, "original_photo") and self.original_photo.isChecked():
            return self._original_display_asset(None, reference=True)
        raw_detail = bool(getattr(self, "raw_detail", None) is not None
                          and self.raw_detail.isChecked())
        if raw_detail:
            if asset := ReviewImageAssets._rig_raw_display_asset(self, None, "raw_detail", reference=True):
                return asset
            from darsia.fluidflower.phase_photographic_assets import resolve_raw_detail
            return DeferredImage(
                f"v5-raw-detail:reference:{root}:{run}",
                lambda: resolve_raw_detail(root, run, None, reference=True),
            )
        if display_modes_available({"frame_sources": (self.current_pair or {}).get("frame_sources") or []}):
            mode = getattr(self, "display_mode", "natural")
            if mode == "change" and (asset := ReviewImageAssets._rig_raw_display_asset(self, None, mode, reference=True)):
                return asset
            return DeferredImage(f"v5-display:{mode}:reference:{root}:{run}",
                                 lambda: resolve_display_mode(root, run, None, mode, reference=True))
        return DeferredImage(
            f"reference:{root}:{run}",
            lambda: review._native_reference_path(root, run),
        )

    def _rig_raw_display_asset(self, index, mode, *, reference=False):
        if asset := ReviewImageAssets._selected_photo_asset(self,index,mode,reference=reference):
            return asset
        from darsia.fluidflower.phase_rig_raw_display import rig_raw_display_available, load_rig_raw_display_image
        from .phase_studio_display_cache import binding_token
        requests = _presentation_assets(self)
        request_key = ("rig-photo", index, mode, reference)
        if request_key in requests:
            return requests[request_key]
        pair = self.current_pair or {}
        manifest = dict(pair, reference=pair.get("reference") or pair.get("original_reference_source") or {})
        if not rig_raw_display_available(manifest, frame_index=index, reference=reference):
            return None
        row = manifest["reference"] if reference else manifest["frame_sources"][index]
        root, run = self.current_review_root, self.current_run
        loaded = getattr(self, "_loaded_photo_rows", {}).get("reference" if reference else index)
        binding = binding_token(dict(row_sha256=loaded["binding"], mode=mode)) if loaded else binding_token(dict(row=row, mode=mode))
        requests[request_key] = ReviewImageAssets._bound_photo_asset(self,
            DeferredImage(f"rig-raw-display:v1:{mode}:{root}:{run}:{index}:{reference}:{binding}",
                lambda: load_rig_raw_display_image(root, run, index, mode=mode, reference=reference)),
            index, mode, reference=reference)
        return requests[request_key]

    def _bound_photo_asset(self, asset, index, mode, *, reference=False):
        """Bind local decoded pixels to the exact recorded source and view."""
        from .phase_studio_display_cache import binding_token
        pair = self.current_pair or {}
        rows = pair.get("frame_sources") or []
        row = (pair.get("reference") or pair.get("original_reference_source") or {}) if reference else (
            rows[index] if type(index) is int and 0 <= index < len(rows) else {})
        if not row.get("source_sha256"):
            return asset
        if "ffac_selected_policy_a_sources" in pair:
            descriptor=row.get("photographs",{}).get(mode) or {}
            binding=binding_token(dict(camera=row.get("source_sha256"),photograph=descriptor,mode=mode))
            return replace(asset,photo_binding=binding,photo_dependencies=(str(descriptor.get("path") or ""),),photo_revision=None)
        requests = _presentation_assets(self)
        key = ("photo", asset.identity, mode, reference)
        if key in requests:
            return requests[key]
        if mode in BASE_DISPLAY_MODES:
            from darsia.fluidflower.phase_rig_raw_display import rig_raw_display_available
            if rig_raw_display_available(pair, frame_index=index, reference=reference):
                root, run = self.current_review_root, self.current_run
                def resolve_photo():
                    from darsia.fluidflower.phase_rig_raw_display import (
                        _native_frame_assets_cached, _verified_display_path,
                    )
                    state, _record, binding = _native_frame_assets_cached(root, run, index, reference=reference)
                    return _verified_display_path(binding["photographs"][mode], base=state["input_path"].parent)
                asset = replace(asset, resolve=resolve_photo)
        dependencies = [str(Path(self.current_review_root) / "native_frames" / self.current_run / "manifest.json")]
        loaded = getattr(self, "_loaded_photo_rows", {}).get("reference" if reference else index)
        if mode in {"natural", "lift1", "lift2"}:
            path = ((row.get("v5_display_modes") or {}).get(mode) or {}).get("jpeg_path") or row.get("frame_path")
            if path:
                dependencies.append(str(path))
        elif mode == "original":
            if row.get("source_path"):
                dependencies.append(str(row["source_path"]))
        else:
            def paths(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if isinstance(item, str) and (key == "path" or key.endswith("_path")):
                            dependencies.append(item)
                        elif isinstance(item, (dict, list)):
                            paths(item)
                elif isinstance(value, list):
                    for item in value:
                        paths(item)
            if loaded:
                dependencies.extend(loaded["dependencies"])
            else:
                paths(row)
            if mode == "change":
                baseline = getattr(self, "_loaded_photo_rows", {}).get("reference")
                if baseline:
                    dependencies.extend(baseline["dependencies"])
                else:
                    paths(pair.get("reference") or pair.get("original_reference_source") or {})
        binding = binding_token(dict(recipe="padded-photo-v2", row=row if not loaded else {"row_sha256": loaded["binding"]}, mode=mode,
            baseline=({name: pair.get(name) for name in ("reference", "original_reference_source")} if not loaded else
                      {"row_sha256": getattr(self, "_loaded_reference_binding", "")}) if mode == "change" else None,
            canvas=pair.get("canvas_wh"),
            native_box=pair.get("content_box_xywh"), analysis_shape=pair.get("analysis_image_shape_hw")))
        requests[key] = replace(asset, photo_binding=binding, photo_dependencies=tuple(dict.fromkeys(dependencies)),
                                photo_revision=row.get("rig_domain_revision"))
        return requests[key]

    def _original_display_asset(self, index: int | None, *, reference: bool = False) -> DeferredImage:
        if asset := ReviewImageAssets._selected_photo_asset(self,index,"original",reference=reference):
            return asset
        root, run = self.current_review_root, self.current_run
        row = ((self.current_pair or {}).get("original_reference_source") or {}
               if reference else ((self.current_pair or {}).get("frame_sources") or [])[index])
        digest = str(row.get("source_sha256") or "")
        receipt = str(row.get("v5_baseline_receipt_sha256" if reference
                              else "v5_frame_receipt_sha256") or "")
        def resolve() -> QImage:
            rgb = load_original_display_rgb(root, run, index, reference=reference)
            height, width = rgb.shape[:2]
            return QImage(rgb.data, width, height, rgb.strides[0],
                          QImage.Format.Format_RGB888).copy()
        return ReviewImageAssets._bound_photo_asset(self, DeferredImage(
            f"v5-original:camera-jpeg-v1:{root}:{run}:{index}:{reference}:{digest}:{receipt}",
            resolve,
        ), index, "original", reference=reference)

    def _overlay_asset(self, side: str, index: int, layer: str | None = None) -> DeferredImage:
        root, run = self.current_review_root, self.current_run
        candidate = self.current_pair_object.candidate(side)
        details = getattr(self, "_loaded_candidate_assets", {}).get(side)
        if (details["prediction_withheld"] if details is not None
                else review._candidate_prediction_withheld(root, candidate)):
            return DeferredImage(f"bottom-blind-withheld:{root}:{run}:{side}:{index}",
                                 lambda: QImage(), normalize_overlay=True)
        cleanup = getattr(self, "optical_cleanup", None)
        phase = str(layer or self.contour_filter.currentData() or "all")
        if layer is None and phase != "source_support" and cleanup is not None and cleanup.active:
            return cleanup.asset(DeferredImage, index)
        if getattr(self, "local_reference_search", None) is not None and self.local_reference_search.isChecked() and phase in {"all", "gas", "aqueous"}:
            from darsia.fluidflower.tools.phase_local_support_preview import resolve
            fingerprint = review._candidate_fingerprint(candidate)
            map_root = review._candidate_phase_map_root(root, candidate)
            return DeferredImage(
                f"local-support:{root}:{map_root}:{run}:{fingerprint}:{index}:{phase}",
                lambda: resolve(root, run, candidate, index, phase), normalize_overlay=True)
        requests = _presentation_assets(self)
        request_key = ("overlay", side, id(candidate), index, phase)
        if request_key in requests:
            return requests[request_key]
        fingerprint = details["fingerprint"] if details is not None else review._candidate_fingerprint(candidate)
        map_root = details["map_root"] if details is not None else review._candidate_phase_map_root(root, candidate)
        from darsia.fluidflower.tools.phase_studio_display_cache import binding_token
        rows = (self.current_pair or {}).get("frame_sources") or []
        row = rows[index] if 0 <= index < len(rows) else {}
        revision = row.get("rig_domain_revision")
        masks = None
        dependencies = ()
        if revision is not None and (revision.get("classes") or {}).get("path"):
            def masks():
                from darsia.fluidflower.phase_saved_review_metadata import load_rig_classes
                fields, _ = load_rig_classes(revision, base_dir=root)
                return fields["rig_domain"], fields["filled_observed"] | fields["filled_unobserved"]
            dependencies = (str(revision["classes"]["path"]),)
        elif row.get("v5_photo_sand_mask_path"):
            dependencies = (str(row["v5_photo_sand_mask_path"]),)
        saved = (getattr(candidate, "quality_flags", None) or {}).get("approved_g3_downstream") is True
        asset_side = "saved" if saved else side
        binding = binding_token(dict(row=row, pipeline=getattr(candidate, "params_sha256", None),
            model=getattr(candidate, "optical_model_sha256", None), layer=phase)) if row else ""
        contact_loader = None
        if row.get("floor_contact_fill") or (getattr(candidate, "quality_flags", None) or {}).get("floor_contact_rule"):
            from darsia.fluidflower.phase_floor_contact_display import resolve_floor_contact_display_mask
            contact_root = details["review_root"] if details is not None else review._candidate_review_root(root, candidate)
            contact_loader = lambda: resolve_floor_contact_display_mask(contact_root, run, index)
            if (descriptor := row.get("floor_contact_fill")) and descriptor.get("path"):
                dependencies = (*dependencies, str(descriptor["path"]))
        l1_loader = None
        if row.get("l1_neighbourhood_gas") or (getattr(candidate, "quality_flags", None) or {}).get("l1_neighbourhood_rule"):
            from darsia.fluidflower.phase_l1_neighbourhood_display import resolve_l1_neighbourhood_display_mask
            l1_root = details["review_root"] if details is not None else review._candidate_review_root(root, candidate)
            l1_loader = lambda: resolve_l1_neighbourhood_display_mask(l1_root, run, index)
            if (descriptor := row.get("l1_neighbourhood_gas")) and descriptor.get("path"):
                dependencies = (*dependencies, str(descriptor["path"]))
        requests[request_key] = DeferredImage(
            f"overlay:optical-v2:{root}:{map_root}:{run}:{asset_side}:{fingerprint}:{index}:{phase}:{binding}",
            lambda: review._preferred_phase_overlay_path(root, candidate, side, index, overlay_asset_layer(phase))[0],
            normalize_overlay=True,
            display_domain=self._photo_sand_loader(index),
            display_filled=self._filled_display_loader(index),
            display_masks=masks, display_layer=phase,
            display_floor_contact=contact_loader,
            display_l1_neighbourhood=l1_loader,
            display_binding=binding, display_dependencies=dependencies,
        )
        return requests[request_key]


def _region_phase_image(window: Any, side: str, index: int) -> tuple[QImage, Path]:
    """Reuse worker-normalized overlays when scoring a manually drawn region.

    ImageTask normalizes DeferredImage overlays before inserting them into the
    cache. Repeating that operation here scans and copies the entire native
    image on the GUI thread, even for a tiny box. Raw fallback images retain the
    original whole-image alpha normalization so sampled classifications agree.
    """
    source = window._overlay_path(side, index, "all")
    asset = window._overlay_asset(side, index, "all")
    image = window.image_cache.get(asset)
    already_normalized = (
        image is not None
        and isinstance(asset, DeferredImage)
        and asset.normalize_overlay
    )
    if image is None:
        image = QImage(str(source))
    if not already_normalized and not image.isNull():
        image = normalize_phase_overlay_alpha(image)
    return image, source


class LocalAssetCache:
    """Versioned local copy of NAS assets with offline fallback."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.source_locks: dict[str, threading.Lock] = {}
        from .phase_studio_display_cache import PreparedOverlayStore, PreparedPhotoStore
        self.prepared_overlays = PreparedOverlayStore(self.root / "prepared_overlays")
        self.prepared_photos = PreparedPhotoStore(self.root / "prepared_photos")

    def _paths(self, source: Path) -> tuple[Path, Path]:
        digest = hashlib.sha256(str(source).casefold().encode("utf-8")).hexdigest()
        suffix = source.suffix.casefold() or ".bin"
        return self.root / f"{digest}{suffix}", self.root / f"{digest}.json"

    def stage(self, source: Path) -> Path:
        source = Path(source)
        cached, metadata_path = self._paths(source)
        with self.lock:
            source_lock = self.source_locks.setdefault(str(cached), threading.Lock())
        # Unrelated images can copy concurrently; a shared destination cannot.
        with source_lock:
            try:
                stat = source.stat()
            except OSError:
                if cached.is_file():
                    return cached
                raise
            metadata: dict[str, Any] = {}
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    metadata = {}
            if (
                cached.is_file()
                and int(metadata.get("size") or -1) == int(stat.st_size)
                and int(metadata.get("mtime_ns") or -1) == int(stat.st_mtime_ns)
            ):
                return cached
            error: Exception | None = None
            for attempt in range(3):
                temporary = cached.with_name(f".{cached.name}.{os.getpid()}.tmp")
                try:
                    shutil.copyfile(source, temporary)
                    if temporary.stat().st_size != stat.st_size:
                        raise OSError("Incomplete local asset copy")
                    os.replace(temporary, cached)
                    atomic_json(
                        metadata_path,
                        {
                            "source": str(source),
                            "size": int(stat.st_size),
                            "mtime_ns": int(stat.st_mtime_ns),
                            "cached_utc": utc_now(),
                        },
                    )
                    return cached
                except Exception as exc:  # noqa: BLE001 - NAS retry boundary
                    error = exc
                    temporary.unlink(missing_ok=True)
                    if cached.is_file():
                        return cached
                    time.sleep(0.15 * (attempt + 1))
            assert error is not None
            raise error


class ImageTaskSignals(QObject):
    loaded = Signal(str, object, str)
    persist = Signal(str, object)


class _SavedReaderContextTask(QRunnable):
    """Capture only a saved/native-reading UI task's construction context."""
    def __init__(self):
        super().__init__()
        self._saved_reader_context = copy_context()

    def run(self):
        return self._saved_reader_context.run(self._run_saved_read)


class ImageTask(_SavedReaderContextTask):
    def __init__(self, key: str, source: Path | DeferredImage, disk_cache: LocalAssetCache) -> None:
        super().__init__()
        self.key = key
        self.source = source
        self.disk_cache = disk_cache
        self.signals = ImageTaskSignals()
        self.foreground = True
        self.allow_compute = False
        self.defer_persistence = False

    def _run_saved_read(self) -> None:
        try:
            expected_stamps = []
            if isinstance(self.source, DeferredImage):
                if self.source.photo_binding:
                    cached = self.disk_cache.prepared_photos.load(self.source.identity, self.source.photo_binding)
                    if cached is not None:
                        self.signals.loaded.emit(self.key, cached, "")
                        return
                if self.source.display_binding and self.source.display_layer is not None:
                    cached = self.disk_cache.prepared_overlays.load(self.source.identity,
                        self.source.display_binding, self.source.display_layer)
                    if cached is not None:
                        self.signals.loaded.emit(self.key, cached, "")
                        return
                try:
                    expected_stamps = self.disk_cache.prepared_photos._source_stamps(
                        (*self.source.photo_dependencies, *self.source.display_dependencies))
                except OSError:
                    expected_stamps = None  # Offline reads may display an existing local copy.
                resolve = (self.source.prefetch_resolve
                           if not (self.foreground or self.allow_compute) and self.source.prefetch_resolve is not None
                           else self.source.resolve)
                source = resolve()
                if source is None:
                    self.signals.loaded.emit(self.key, QImage(), "")
                    return
            else:
                source = self.source
            if (not isinstance(source, QImage) and isinstance(self.source, DeferredImage)
                    and expected_stamps is not None):
                try:
                    stamp = self.disk_cache.prepared_photos._stamp(source)
                    if not any(item["path"] == stamp["path"] for item in expected_stamps):
                        expected_stamps.append(stamp)
                except OSError:
                    expected_stamps = None
            if isinstance(source, QImage):
                # A derived, display-only layer can be rendered in this worker
                # without writing another full-resolution image to the NAS.
                image = source
                source_description = self.source.identity
            else:
                local_path = self.disk_cache.stage(source)
                image = QImage(str(local_path))
                source_description = str(local_path)
            if image.isNull():
                raise OSError(f"Could not decode {source_description}")
            if isinstance(self.source, DeferredImage) and self.source.normalize_overlay:
                image = normalize_phase_overlay_alpha(image)
            if isinstance(self.source, DeferredImage) and self.source.photo_revision is not None:
                from darsia.fluidflower.phase_rig_display_canvas import pad_rig_photograph
                image = pad_rig_photograph(image, self.source.photo_revision)
            if isinstance(self.source, DeferredImage) and self.source.display_masks is not None:
                image._photo_sand_domain, image._filled_phase_mask = self.source.display_masks()
            elif isinstance(self.source, DeferredImage) and self.source.display_domain is not None:
                # Presentation context accompanies the full cached map; scoring
                # still receives the original pixels, not the scene filter.
                image._photo_sand_domain = self.source.display_domain()
            if isinstance(self.source, DeferredImage) and self.source.display_masks is None and self.source.display_filled is not None:
                image._filled_phase_mask = self.source.display_filled()
            if isinstance(self.source, DeferredImage) and self.source.display_floor_contact is not None:
                image._floor_contact_fill_mask = self.source.display_floor_contact()
            if isinstance(self.source, DeferredImage) and self.source.display_l1_neighbourhood is not None:
                image._l1_neighbourhood_gas_mask = self.source.display_l1_neighbourhood()
            if isinstance(self.source, DeferredImage) and self.source.display_layer is not None:
                from .phase_studio_display_cache import prepared_overlay
                prepared_overlay(image, self.source.display_layer)
            if self.defer_persistence:
                self.signals.persist.emit(self.key, lambda: self._persist_image(image, source, expected_stamps))
            else:
                self._persist_image(image, source, expected_stamps)
            self.signals.loaded.emit(self.key, image, "")
        except Exception as exc:  # noqa: BLE001 - background UI boundary
            self.signals.loaded.emit(self.key, QImage(), f"{type(exc).__name__}: {exc}")

    def _persist_image(self, image, source, expected_stamps):
        if not isinstance(self.source, DeferredImage) or expected_stamps is None:
            return
        if self.defer_persistence:
            snapshot = QImage(image)
            for name, value in vars(image).items():
                setattr(snapshot, name, dict(value) if name == "_display_variants" else value)
            image = snapshot
        if self.source.display_binding and self.source.display_layer is not None and not isinstance(source, QImage):
            try:
                self.disk_cache.prepared_overlays.save(self.source.identity,
                    self.source.display_binding, self.source.display_layer, image,
                    (str(source), *self.source.display_dependencies), expected_stamps=expected_stamps)
            except OSError:
                pass  # Optional display derivatives never replace source assets.
        if self.source.photo_binding:
            try:
                sources = self.source.photo_dependencies if isinstance(source, QImage) else (
                    str(source), *self.source.photo_dependencies)
                self.disk_cache.prepared_photos.save(self.source.identity, self.source.photo_binding, image, sources,
                    expected_stamps=expected_stamps)
            except OSError:
                pass


class PresentationCacheSaveTask(QRunnable):
    def __init__(self, key, save):
        super().__init__()
        self.key, self.save = key, save
        self.signals = ImageTaskSignals()

    def run(self):
        try:
            self.save()
        finally:
            self.signals.loaded.emit(self.key, None, "")


class DisplayVariantTask(QRunnable):
    """Filter and hatch complete maps without scanning pixels in Qt slots."""

    def __init__(self, key, image, options):
        super().__init__()
        self.key, self.image, self.options = key, image, options
        self.signals = ImageTaskSignals()

    def run(self):
        try:
            from .phase_studio_display_cache import prepared_overlay
            # QImage shares immutable source pixels. All variant dictionary
            # changes belong to this worker copy until Qt admits the result.
            image = QImage(self.image)
            for name, value in vars(self.image).items():
                setattr(image, name, dict(value) if name == "_display_variants" else value)
            prepared_overlay(image, **self.options)
            self.signals.loaded.emit(self.key, image._display_variants, "")
        except Exception as exc:
            self.signals.loaded.emit(self.key, {}, f"{type(exc).__name__}: {exc}")


class AsyncImageCache(QObject):
    image_ready = Signal(str, object, str)
    work_started = Signal(str)

    def __init__(self, root: Path, *, workers: int = 3, max_bytes: int | None = None) -> None:
        super().__init__()
        self.disk = LocalAssetCache(root)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max(1, int(workers)))
        # Native reference replay can take minutes. Keep it serial and separate
        # from photo/timeline decoding so navigating remains responsive.
        self.reference_pool = QThreadPool(self)
        self.reference_pool.setMaxThreadCount(1)
        self.reference_running: set[str] = set()
        from .phase_studio_display_cache import default_image_cache_bytes
        self.max_bytes = max(256 << 20, int(max_bytes)) if max_bytes is not None else default_image_cache_bytes()
        memory_workers = 1 if self.max_bytes < 512 << 20 else 2 if self.max_bytes < 1 << 30 else 3 if self.max_bytes < 2 << 30 else 4
        self.pool.setMaxThreadCount(min(self.pool.maxThreadCount(), memory_workers))
        self.images: OrderedDict[str, QImage] = OrderedDict()
        self.pending: set[str] = set()
        self.tasks: dict[str, ImageTask] = {}
        self.queued: OrderedDict[str, ImageTask] = OrderedDict()
        self.running: set[str] = set()
        self.total_bytes = 0
        self.image_sizes: dict[str, int] = {}
        self.foreground_keys = set()
        self.display_pool = QThreadPool(self)
        self.display_pool.setMaxThreadCount(1)
        self.display_tasks = {}
        self.display_queued = OrderedDict()
        self.display_running = ""
        self.persistence_pool = QThreadPool(self)
        self.persistence_pool.setMaxThreadCount(1)
        self.persistence_queued = OrderedDict()
        self.persistence_running = None

    def _queue_persistence(self, key, save):
        self.persistence_queued[key] = save
        self.persistence_queued.move_to_end(key, last=False)
        # These are optional lossless derivatives. Never retain a run's full
        # images merely to write cache files after the user has moved on.
        while len(self.persistence_queued) > 2:
            self.persistence_queued.popitem(last=True)
        self._dispatch_persistence()

    def _dispatch_persistence(self):
        if self.persistence_running is None and self.persistence_queued:
            key, save = self.persistence_queued.popitem(last=False)
            task = PresentationCacheSaveTask(key, save)
            task.signals.loaded.connect(self._persistence_loaded)
            self.persistence_running = task
            self.persistence_pool.start(task)

    def _persistence_loaded(self, _key, _image, _error):
        self.persistence_running = None
        self._dispatch_persistence()

    def request_display(self, key, image, **options):
        """Return exact ready pixels, or queue a bounded display-only variant."""
        from .phase_studio_display_cache import overlay_variant_key
        variant = overlay_variant_key(image, **options)
        ready = getattr(image, "_display_variants", {}).get(variant)
        if ready is not None:
            return ready
        task_key = repr((key, id(image), variant))
        if task_key not in self.display_tasks:
            task = DisplayVariantTask(task_key, image, options)
            self.display_tasks[task_key] = (key, task)
            task.signals.loaded.connect(self._display_loaded)
            self.display_queued[task_key] = task
            self.display_queued.move_to_end(task_key, last=False)
            while len(self.display_queued) > 4:
                stale, _task = self.display_queued.popitem(last=True)
                self.display_tasks.pop(stale, None)
        elif task_key in self.display_queued:
            self.display_queued.move_to_end(task_key, last=False)
        self._dispatch_display()
        return None

    def _dispatch_display(self):
        if not self.display_running and self.display_queued:
            task_key, task = self.display_queued.popitem(last=False)
            self.display_running = task_key
            self.display_pool.start(task)

    def _display_loaded(self, task_key, variants, error):
        entry = self.display_tasks.pop(task_key, None)
        self.display_running = ""
        if entry is not None:
            key, task = entry
            # A completed stale frame must never replace a newer cache entry.
            if key not in self.images or self.images.get(key) is task.image:
                if not error:
                    current = getattr(task.image, "_display_variants", {})
                    current.update(variants)
                    while len(current) > 6:
                        current.pop(next(iter(current)))
                    task.image._display_variants = current
                    self.note_image_modified(key)
                self.image_ready.emit(key, task.image, error)
        self._dispatch_display()

    @staticmethod
    def key(source: Path | DeferredImage) -> str:
        identity = source.identity if isinstance(source, DeferredImage) else str(Path(source))
        if isinstance(source, DeferredImage):
            # The decoded/pending cache must use the same source and recipe
            # binding as the persistent presentation derivatives.
            identity += "|" + repr((source.photo_binding, source.display_binding,
                                    source.display_layer, source.normalize_overlay))
        return hashlib.sha256(identity.casefold().encode("utf-8")).hexdigest()

    def get(self, source: Path | DeferredImage) -> QImage | None:
        key = self.key(source)
        image = self.images.get(key)
        if image is not None:
            self.images.move_to_end(key)
        return image

    def note_image_modified(self, key: str) -> None:
        """Account for an alternate display variant added by a control toggle."""
        from .phase_studio_display_cache import image_memory_bytes
        if key in self.images:
            size = image_memory_bytes(self.images[key])
            self.total_bytes += size - self.image_sizes.get(key, 0)
            self.image_sizes[key] = size
            self._trim()

    def _trim(self) -> None:
        while self.total_bytes > self.max_bytes and len(self.images) > 1:
            old_key = next((key for key in self.images if key not in self.foreground_keys), next(iter(self.images)))
            self.images.pop(old_key)
            self.total_bytes -= self.image_sizes.pop(old_key, 0)

    def request(self, source: Path | DeferredImage, *, foreground: bool = True,
                allow_compute: bool = False) -> str:
        key = self.key(source)
        image = self.images.get(key)
        if image is not None:
            self.images.move_to_end(key)
            if foreground:
                QTimer.singleShot(0, lambda: self.image_ready.emit(key, image, ""))
            return key
        if key in self.pending:
            self.tasks[key].allow_compute |= allow_compute
            if foreground:
                self.tasks[key].foreground = True
            if foreground and key in self.queued:
                self.queued.move_to_end(key, last=False)
                self._dispatch()
            return key
        self.pending.add(key)
        task = ImageTask(key, source, self.disk)
        task.defer_persistence = True
        task.foreground = foreground
        task.allow_compute = allow_compute
        self.tasks[key] = task
        task.signals.loaded.connect(self._loaded)
        task.signals.persist.connect(self._queue_persistence)
        self.queued[key] = task
        if foreground:
            self.queued.move_to_end(key, last=False)
        self._dispatch()
        return key

    def prefetch(self, sources: Sequence[Path | DeferredImage], *, replace_pending: bool = False) -> None:
        if replace_pending:
            keep = {self.key(source) for source in sources}
            for key, task in list(self.queued.items()):
                if (key not in keep and not task.foreground and not task.allow_compute
                        and not (isinstance(task.source, DeferredImage) and task.source.reference_phase)):
                    self.queued.pop(key)
                    self.tasks.pop(key)
                    self.pending.discard(key)
        seen: set[str] = set()
        for source in sources:
            key = self.key(source)
            if key not in seen:
                seen.add(key)
                self.request(source, foreground=False)

    def prepare(self, sources: Sequence[DeferredImage]) -> None:
        """Prepare missing references serially without taking foreground priority."""
        for source in sources:
            self.request(source, foreground=False, allow_compute=True)

    def prefetch_window(self, sources: Sequence[Path | DeferredImage]) -> None:
        """Replace stale background navigation work with the current window."""
        self.prefetch(sources, replace_pending=True)

    def focus(self, keys):
        """Drop obsolete unstarted frame selections; current panes keep priority."""
        wanted = set(filter(None, keys))
        self.foreground_keys = wanted
        for key, task in list(self.queued.items()):
            if task.foreground and key not in wanted and not task.allow_compute:
                self.queued.pop(key)
                self.tasks.pop(key, None)
                self.pending.discard(key)
        for task_key, task in list(self.display_queued.items()):
            image_key = self.display_tasks[task_key][0]
            if image_key not in wanted:
                self.display_queued.pop(task_key)
                self.display_tasks.pop(task_key, None)

    def cancel_queued_reference_preparations(self) -> None:
        for key, task in list(self.queued.items()):
            if (not task.foreground and isinstance(task.source, DeferredImage)
                    and task.source.reference_phase):
                self.queued.pop(key)
                self.tasks.pop(key)
                self.pending.discard(key)

    def _dispatch(self) -> None:
        for key, task in list(self.queued.items()):
            reference = isinstance(task.source, DeferredImage) and task.source.reference_phase
            running = self.reference_running if reference else self.running
            pool = self.reference_pool if reference else self.pool
            # Keep one decoder available for a newly selected photograph while
            # neighboring frames are being prepared. At least one background
            # worker is useful on small-RAM hosts with a single decoder.
            limit = pool.maxThreadCount()
            if not reference and not task.foreground:
                limit = max(1, limit - 1)
            if len(running) < limit:
                self.queued.pop(key)
                running.add(key)
                pool.start(task)
                self.work_started.emit(key)

    def _loaded(self, key: str, image: QImage, error: str) -> None:
        self.pending.discard(key)
        task = self.tasks.pop(key, None)
        self.running.discard(key)
        self.reference_running.discard(key)
        if not error and image.isNull():
            # Upgrade a concurrent cache-only check once if either selection or
            # explicit background preparation arrived while it was running.
            if task is not None and (task.foreground or task.allow_compute):
                self.request(task.source, foreground=task.foreground,
                             allow_compute=task.allow_compute)
            self._dispatch()
            return
        if not error and not image.isNull():
            from .phase_studio_display_cache import image_memory_bytes
            size = image_memory_bytes(image)
            previous = self.images.pop(key, None)
            if previous is not None:
                self.total_bytes -= self.image_sizes.pop(key, 0)
            self.images[key] = image
            self.image_sizes[key] = size
            self.total_bytes += size
            self._trim()
        self.image_ready.emit(key, image, error)
        self._dispatch()


def isolated_phase_components(
    image: QImage,
    *,
    max_area_fraction: float = ISOLATED_COMPONENT_MAX_AREA_FRACTION,
    color_tolerance: int = ISOLATED_COMPONENT_COLOR_TOLERANCE,
) -> dict[str, Any]:
    """Locate tiny CO2-support islands without changing the phase map."""

    if image.isNull():
        return {
            "components": [],
            "max_area_px": 0,
            "total_components": 0,
        }

    import numpy as np

    rgba_image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    height = int(rgba_image.height())
    width = int(rgba_image.width())
    rows = np.frombuffer(rgba_image.bits(), dtype=np.uint8).reshape(
        height,
        int(rgba_image.bytesPerLine()),
    )
    rgba = rows[:, : width * 4].reshape(height, width, 4)
    return review.isolated_phase_components_rgba(
        rgba,
        max_area_fraction=max_area_fraction,
        color_tolerance=color_tolerance,
    )


class ComponentTaskSignals(QObject):
    loaded = Signal(str, object, str)


class ComponentTask(QRunnable):
    def __init__(self, key: str, image: QImage) -> None:
        super().__init__()
        self.key = key
        self.image = QImage(image)
        self.signals = ComponentTaskSignals()

    def run(self) -> None:
        try:
            result = isolated_phase_components(self.image)
            self.signals.loaded.emit(self.key, result, "")
        except Exception as exc:  # noqa: BLE001 - background UI boundary
            self.signals.loaded.emit(
                self.key,
                {},
                f"{type(exc).__name__}: {exc}",
            )


class SmallComponentCache(QObject):
    result_ready = Signal(str, object, str)

    def __init__(self, *, workers: int = 2, max_entries: int = 32) -> None:
        super().__init__()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max(1, int(workers)))
        self.max_entries = max(4, int(max_entries))
        self.results: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.pending: set[str] = set()
        self.tasks: dict[str, ComponentTask] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        result = self.results.get(key)
        if result is not None:
            self.results.move_to_end(key)
        return result

    def request(self, key: str, image: QImage) -> None:
        result = self.results.get(key)
        if result is not None:
            self.results.move_to_end(key)
            QTimer.singleShot(0, lambda: self.result_ready.emit(key, result, ""))
            return
        if key in self.pending or image.isNull():
            return
        self.pending.add(key)
        task = ComponentTask(key, image)
        self.tasks[key] = task
        task.signals.loaded.connect(self._loaded)
        self.pool.start(task)

    def _loaded(self, key: str, result: object, error: str) -> None:
        self.pending.discard(key)
        self.tasks.pop(key, None)
        if not error and isinstance(result, dict):
            self.results[key] = result
            self.results.move_to_end(key)
            while len(self.results) > self.max_entries:
                self.results.popitem(last=False)
        self.result_ready.emit(key, result, error)


class WrappingControlLayout(QLayout):
    """Keep photographic options readable as the review pane gets narrower."""

    def __init__(self) -> None:
        """Create a compact layout that places controls at their requested sizes."""
        super().__init__()
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(6)

    def addItem(self, item) -> None:
        """Append a control and invalidate the cached layout geometry."""
        self._items.append(item)
        self.invalidate()

    def count(self) -> int:
        """Return the number of managed layout items."""
        return len(self._items)

    def itemAt(self, index):
        """Return the indexed item without removing it, or None outside the layout."""
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        """Remove and return the indexed item when it exists."""
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        """Keep control sizes independent of spare horizontal or vertical space."""
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        """Tell Qt that wrapping determines the layout height."""
        return True

    def heightForWidth(self, width: int) -> int:
        """Measure the required wrapped height without moving controls."""
        return self._arrange(QRect(0, 0, width, 0))

    def setGeometry(self, rectangle) -> None:
        """Apply the available rectangle and place each wrapped control."""
        super().setGeometry(rectangle)
        self._arrange(rectangle, place=True)

    def minimumSize(self):
        """Require enough space for the largest visible control and layout margins."""
        size = QSize(0, 0)
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def sizeHint(self):
        """Use the minimum control footprint as the layout's preferred size."""
        return self.minimumSize()

    def _arrange(self, rectangle, *, place=False) -> int:
        """Measure or place visible controls in rows without shrinking their labels."""
        margins = self.contentsMargins()
        left, top = rectangle.x() + margins.left(), rectangle.y() + margins.top()
        right = rectangle.right() - margins.right() + 1
        x, y, row_height = left, top, 0
        for item in self._items:
            if item.isEmpty():
                continue
            size = item.sizeHint().expandedTo(item.minimumSize())
            if x > left and x + size.width() > right:
                x, y, row_height = left, y + row_height + self.spacing(), 0
            if place:
                item.setGeometry(QRect(x, y, size.width(), size.height()))
            x += size.width() + self.spacing()
            row_height = max(row_height, size.height())
        return y + row_height + margins.bottom() - rectangle.y()


class FrameTimelineLabel(QLabel):
    """Keep playback controls stationary when the frame description changes."""

    def __init__(self) -> None:
        super().__init__("Frame --")
        self._frame_count = 0
        self._status_texts = ("Frame --",)
        self._reference_texts: tuple[str, ...] = ()
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

    @staticmethod
    def frame_text(index: int, count: int, elapsed: float | None = None,
                   *, reference: bool = False, baseline: bool = False) -> str:
        if reference:
            return "Pre-injection reference · before injection"
        if baseline:
            return f"Pre-injection baseline · Frame 1 / {count} · negative control"
        return (f"Frame {index + 1} / {count}"
                + (f" · {elapsed:.2f} h" if elapsed is not None else ""))

    def set_timeline(self, frame_count: int, elapsed_hours: Sequence[float]) -> None:
        self._frame_count = int(frame_count)
        self._status_texts = (
            self.frame_text(0, frame_count, reference=True),
            self.frame_text(0, frame_count, baseline=True),
            *(self.frame_text(index, frame_count,
                 float(elapsed_hours[index]) if index < len(elapsed_hours) else None)
              for index in range(frame_count)),
        )
        self.updateGeometry()

    def set_reference_texts(self, texts: Sequence[str]) -> None:
        self._reference_texts = tuple(texts)
        self.updateGeometry()

    def set_frame(self, index: int, elapsed: float | None, *, reference: bool,
                  baseline: bool, reference_text: str | None = None) -> None:
        self.setText(reference_text if reference and reference_text else
                     self.frame_text(index, self._frame_count, elapsed,
                                     reference=reference, baseline=baseline))

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        margins = self.contentsMargins()
        padding = margins.left() + margins.right() + 2 * self.margin() + 8
        width = max(self.fontMetrics().horizontalAdvance(text)
                    for text in (*self._status_texts, *self._reference_texts))
        # Reserve the whole run's widest description, including the baseline.
        # A minimum width alone lets QLabel steal slider space on long frames.
        return QSize(max(225, width + padding), hint.height())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


class FrameTimelineSlider(QSlider):
    """Frame slider with quiz and per-frame review markers."""

    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.marker_value: int | None = None
        self.choice_markers: dict[int, str] = {}
        self._scrubbing = False
        self.setMinimumHeight(30)
        self.setTracking(True)
        self.setSingleStep(1)
        self.setPageStep(1)

    def set_marker(self, value: int | None, tooltip: str = "") -> None:
        self.marker_value = None if value is None else int(value)
        self.setToolTip(
            (tooltip + " " if tooltip else "")
            + "Drag anywhere on the timeline to browse earlier or later frames."
        )
        self.update()

    def set_choice_markers(self, values: Mapping[int, str] | None) -> None:
        self.choice_markers = {
            int(index): str(choice)
            for index, choice in (values or {}).items()
        }
        self.update()

    def _value_x(self, value: int) -> float | None:
        if self.maximum() <= self.minimum():
            return None
        normalized = max(self.minimum(), min(self.maximum(), int(value)))
        ratio = (normalized - self.minimum()) / (
            self.maximum() - self.minimum()
        )
        margin = 9.0
        return margin + ratio * max(1.0, self.width() - 2.0 * margin)

    def _marker_x(self) -> float | None:
        if self.marker_value is None:
            return None
        return self._value_x(self.marker_value)

    def _value_from_x(self, x: float) -> int:
        if self.maximum() <= self.minimum():
            return self.minimum()
        margin = 9.0
        span = max(1.0, self.width() - 2.0 * margin)
        ratio = min(1.0, max(0.0, (float(x) - margin) / span))
        return int(
            round(
                self.minimum()
                + ratio * (self.maximum() - self.minimum())
            )
        )

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        colors = {
            "candidate_a": QColor("#4d91c4"),
            "candidate_b": QColor("#d7a934"),
            "equivalent": QColor("#3fa66b"),
            "acceptable": QColor("#3fa66b"),
            "needs_repair": QColor("#c64f55"),
            "unresolved": QColor("#a0a8ae"),
        }
        for value, choice in self.choice_markers.items():
            x = self._value_x(value)
            if x is None:
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colors.get(choice, QColor("#a0a8ae")))
            painter.drawEllipse(QPointF(x, self.height() - 5.0), 3.2, 3.2)
        marker_x = self._marker_x()
        if marker_x is None:
            return
        red = QColor("#ef3e36")
        painter.setPen(QPen(red, 2.0))
        painter.drawLine(QPointF(marker_x, 9.0), QPointF(marker_x, self.height() - 5.0))
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(red)
        painter.drawEllipse(QPointF(marker_x, 7.0), 5.0, 5.0)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        marker_x = self._marker_x()
        if marker_x is not None and abs(event.position().x() - marker_x) <= 10.0:
            assert self.marker_value is not None
            self._scrubbing = True
            self.setSliderDown(True)
            self.setValue(self.marker_value)
            event.accept()
            return
        self._scrubbing = True
        self.setSliderDown(True)
        self.setValue(self._value_from_x(event.position().x()))
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        if self._scrubbing and event.buttons() & Qt.MouseButton.LeftButton:
            self.setValue(self._value_from_x(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._scrubbing and event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self._value_from_x(event.position().x()))
            self._scrubbing = False
            self.setSliderDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ReviewMinimap(QWidget):
    """Small full-frame overview with viewport and active-region indicators."""

    center_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.thumbnail = QPixmap()
        self.viewport_rect = QRectF(0.0, 0.0, 1.0, 1.0)
        self.focus_rect: QRectF | None = None
        self.setMinimumSize(205, 130)
        self.setMaximumWidth(250)
        self.setToolTip("Overview of the full frame. Click to recenter the main view.")

    def sizeHint(self) -> QSize:
        return QSize(230, 155)

    def set_image(self, image: QImage) -> None:
        thumbnail = image.scaled(
            QSize(460, 300),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.thumbnail = QPixmap.fromImage(thumbnail)
        self.update()

    def set_viewport_rect(self, rect: QRectF) -> None:
        self.viewport_rect = QRectF(
            max(0.0, min(1.0, rect.x())),
            max(0.0, min(1.0, rect.y())),
            max(0.0, min(1.0, rect.width())),
            max(0.0, min(1.0, rect.height())),
        )
        self.update()

    def set_focus_rect(self, rect: QRectF | None) -> None:
        self.focus_rect = None if rect is None else QRectF(rect)
        self.update()

    def _target_rect(self) -> QRectF:
        if self.thumbnail.isNull():
            return QRectF()
        available = QRectF(self.rect()).adjusted(7.0, 7.0, -7.0, -7.0)
        ratio = self.thumbnail.width() / max(1.0, float(self.thumbnail.height()))
        width = available.width()
        height = width / ratio
        if height > available.height():
            height = available.height()
            width = height * ratio
        return QRectF(
            available.center().x() - 0.5 * width,
            available.center().y() - 0.5 * height,
            width,
            height,
        )

    @staticmethod
    def _map_rect(rect: QRectF, target: QRectF) -> QRectF:
        return QRectF(
            target.x() + rect.x() * target.width(),
            target.y() + rect.y() * target.height(),
            rect.width() * target.width(),
            rect.height() * target.height(),
        )

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101316"))
        target = self._target_rect()
        if target.isEmpty():
            painter.setPen(QColor("#75808a"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No overview")
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self.thumbnail, QRectF(self.thumbnail.rect()))
        painter.setPen(QPen(QColor("#66717a"), 1.0))
        painter.drawRect(target)
        if self.focus_rect is not None:
            focus = self._map_rect(self.focus_rect, target).intersected(target)
            if not focus.isEmpty():
                painter.fillRect(focus, QColor(239, 62, 54, 48))
                focus_pen = QPen(QColor("#ef3e36"), 2.0)
                focus_pen.setCosmetic(True)
                painter.setPen(focus_pen)
                painter.drawRect(focus)
        visible = self._map_rect(self.viewport_rect, target).intersected(target)
        if not visible.isEmpty():
            visible_pen = QPen(QColor("#ffd166"), 2.0)
            visible_pen.setCosmetic(True)
            painter.setPen(visible_pen)
            painter.drawRect(visible)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        target = self._target_rect()
        position = event.position()
        if target.contains(position):
            self.center_requested.emit(
                QPointF(
                    (position.x() - target.x()) / target.width(),
                    (position.y() - target.y()) / target.height(),
                )
            )
            event.accept()
            return
        super().mousePressEvent(event)


class IsolatedComponentMarkerItem(QGraphicsItem):
    """Draw screen-space markers around small disconnected phase-map islands."""

    def __init__(self) -> None:
        super().__init__()
        self.components: tuple[Mapping[str, Any], ...] = ()
        self.scene_bounds = QRectF()
        self.setZValue(2)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        return QRectF(self.scene_bounds)

    def set_scene_bounds(self, bounds: QRectF) -> None:
        normalized = QRectF(bounds)
        if normalized == self.scene_bounds:
            return
        self.prepareGeometryChange()
        self.scene_bounds = normalized
        self.update()

    def set_components(self, components: Sequence[Mapping[str, Any]]) -> None:
        self.components = tuple(components)
        self.update()

    def paint(self, painter: QPainter, _option: Any, _widget: Any = None) -> None:
        if not self.components or self.scene_bounds.isEmpty():
            return
        transform = painter.worldTransform()
        scale_x = max(1.0e-6, math.hypot(transform.m11(), transform.m12()))
        scale_y = max(1.0e-6, math.hypot(transform.m21(), transform.m22()))
        bounds = self.scene_bounds
        pen = QPen(QColor(255, 255, 255, 235), 1.5)
        pen.setCosmetic(True)
        painter.save()
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for component in self.components:
            rect = QRectF(
                bounds.x() + float(component["x"]) * bounds.width(),
                bounds.y() + float(component["y"]) * bounds.height(),
                max(1.0 / scale_x, float(component["width"]) * bounds.width()),
                max(1.0 / scale_y, float(component["height"]) * bounds.height()),
            )
            device_width = rect.width() * scale_x
            device_height = rect.height() * scale_y
            if max(device_width, device_height) < 9.0:
                center = QPointF(
                    bounds.x() + float(component["center_x"]) * bounds.width(),
                    bounds.y() + float(component["center_y"]) * bounds.height(),
                )
                painter.drawEllipse(center, 4.5 / scale_x, 4.5 / scale_y)
                continue
            rect.adjust(-2.0 / scale_x, -2.0 / scale_y, 2.0 / scale_x, 2.0 / scale_y)
            painter.drawRoundedRect(rect, 2.0 / scale_x, 2.0 / scale_y)
        painter.restore()


class ReviewImageView(QGraphicsView):
    rectangle_drawn = Signal(object)
    freehand_drawn = Signal(object)
    guide_placed = Signal(str, object)
    view_changed = Signal()
    native_pointer_moved = Signal(object)
    native_pointer_left = Signal()
    annotation_clicked = Signal(str)

    def __init__(self, *, use_opengl: bool = True) -> None:
        super().__init__()
        self.graphics_scene = QGraphicsScene(self)
        self.setScene(self.graphics_scene)
        self.base_item = QGraphicsPixmapItem()
        self.base_item.setZValue(0)
        self.overlay_item = QGraphicsPixmapItem()
        self.overlay_item.setZValue(1)
        self.small_component_markers = IsolatedComponentMarkerItem()
        self.small_component_markers.setVisible(False)
        self.graphics_scene.addItem(self.base_item)
        self.graphics_scene.addItem(self.overlay_item)
        self.graphics_scene.addItem(self.small_component_markers)
        self.peer_cursor_item = QGraphicsEllipseItem(QRectF(-4.0, -4.0, 8.0, 8.0))
        cursor_pen = QPen(QColor("#fff3a0"), 1.0)
        cursor_pen.setCosmetic(True)
        self.peer_cursor_item.setPen(cursor_pen)
        self.peer_cursor_item.setBrush(QColor(0, 0, 0, 0))
        self.peer_cursor_item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
        )
        self.peer_cursor_item.setZValue(20)
        self.peer_cursor_item.setVisible(False)
        self.graphics_scene.addItem(self.peer_cursor_item)
        self.vector_items: list[Any] = []
        self.annotation_hit_regions: list[tuple[str, QRectF, tuple[QPointF, ...]]] = []
        self._pending_annotation_click: tuple[str, QPointF] | None = None
        self._annotation_click_dragged = False
        self.draw_mode = "pan"
        self.start_point: QPointF | None = None
        self.freehand_points: list[QPointF] = []
        self.temporary_item: Any = None
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor("#111417"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.horizontalScrollBar().valueChanged.connect(
            lambda _value: self.view_changed.emit()
        )
        self.verticalScrollBar().valueChanged.connect(
            lambda _value: self.view_changed.emit()
        )
        if use_opengl:
            try:
                self.setViewport(QOpenGLWidget())
                self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
            except Exception:
                pass

    def sizeHint(self) -> QSize:
        return QSize(1100, 720)

    def set_draw_mode(self, mode: str) -> None:
        if mode != self.draw_mode:
            self.cancel_current_draw()
            self._pending_annotation_click = None
        self.draw_mode = mode
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if mode == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.OpenHandCursor
            if mode == "pan"
            else Qt.CursorShape.CrossCursor
        )

    def cancel_current_draw(self) -> None:
        if self.temporary_item is not None:
            self.graphics_scene.removeItem(self.temporary_item)
        self.temporary_item = None
        self.start_point = None
        self.freehand_points = []

    def set_image(self, image: QImage, *, fit: bool = False) -> None:
        if self.draw_mode == "polygon":
            self.cancel_current_draw()
        pixmap = self.base_item.pixmap()
        if pixmap.isNull() or getattr(self, "_base_image_cache_key", None) != image.cacheKey():
            pixmap = QPixmap.fromImage(image)
            self.base_item.setPixmap(pixmap)
            self._base_image_cache_key = image.cacheKey()
        self.graphics_scene.setSceneRect(QRectF(pixmap.rect()))
        self.small_component_markers.set_scene_bounds(QRectF(pixmap.rect()))
        if fit or self.transform().m11() <= 0.0:
            self.fit_image()
        self.view_changed.emit()

    def set_base_offset(self, x_fraction: float = 0.0, y_fraction: float = 0.0) -> None:
        pixmap = self.base_item.pixmap()
        if pixmap.isNull():
            self.base_item.setPos(0.0, 0.0)
            return
        self.base_item.setPos(
            float(x_fraction) * pixmap.width(),
            float(y_fraction) * pixmap.height(),
        )

    def set_overlay(self, image: QImage | None) -> None:
        if image is None:
            self.overlay_item.setPixmap(QPixmap())
            self._overlay_image_cache_key = None
        elif (self.overlay_item.pixmap().isNull()
              or getattr(self, "_overlay_image_cache_key", None) != image.cacheKey()):
            self.overlay_item.setPixmap(QPixmap.fromImage(image))
            self._overlay_image_cache_key = image.cacheKey()
        if image is None:
            self.small_component_markers.set_components(())

    def set_overlay_offset(self, x_fraction: float = 0.0, y_fraction: float = 0.0) -> None:
        """Keep an exact reference map and its markers aligned with its photo."""
        pixmap = self.base_item.pixmap()
        position = QPointF() if pixmap.isNull() else QPointF(
            float(x_fraction) * pixmap.width(), float(y_fraction) * pixmap.height())
        self.overlay_item.setPos(position)
        self.small_component_markers.setPos(position)

    def set_overlay_opacity(self, opacity: float) -> None:
        self.overlay_item.setOpacity(max(0.0, min(1.0, float(opacity))))

    def set_phase_map_only(self, enabled: bool) -> None:
        """Show the transparent phase map on black without altering its pixels."""

        active = bool(enabled)
        self.base_item.setVisible(not active)
        self.setBackgroundBrush(QColor("#000000" if active else "#111417"))

    def set_small_detection_highlight(self, enabled: bool) -> None:
        """Show markers for small disconnected islands without altering the map."""

        self.small_component_markers.setVisible(bool(enabled))

    def set_small_detection_components(
        self,
        components: Sequence[Mapping[str, Any]],
    ) -> None:
        self.small_component_markers.set_components(components)

    def fit_image(self) -> None:
        if self.base_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.base_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.view_changed.emit()

    def native_image_zoom(self) -> None:
        if self.base_item.pixmap().isNull():
            return
        self.resetTransform()
        self.scale(1.0, 1.0)
        self.view_changed.emit()

    def zoom(self, factor: float) -> None:
        current = self.transform().m11()
        if (factor > 1.0 and current < 80.0) or (factor < 1.0 and current > 0.01):
            self.scale(factor, factor)
            self.view_changed.emit()

    def wheelEvent(self, event: Any) -> None:
        self.zoom(1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2)
        event.accept()

    def clear_vectors(self) -> None:
        for item in self.vector_items:
            self.graphics_scene.removeItem(item)
        self.vector_items.clear()
        self.annotation_hit_regions.clear()
        self._pending_annotation_click = None

    def add_annotation_hit_region(
        self,
        annotation_id: str,
        normalized_rect: QRectF,
        normalized_points: Sequence[QPointF] = (),
    ) -> None:
        if annotation_id:
            self.annotation_hit_regions.append(
                (annotation_id, normalized_rect, tuple(normalized_points))
            )

    def _annotation_at_view_point(self, position: Any) -> str:
        scene = self.graphics_scene.sceneRect()
        if scene.isEmpty():
            return ""
        point = self.mapToScene(position)
        if not scene.contains(point):
            return ""
        native = QPointF(
            (point.x() - scene.x()) / scene.width(),
            (point.y() - scene.y()) / scene.height(),
        )
        for annotation_id, rect, polygon_points in reversed(self.annotation_hit_regions):
            if polygon_points:
                if QPolygonF(polygon_points).containsPoint(
                    native, Qt.FillRule.OddEvenFill
                ):
                    return annotation_id
            elif rect.contains(native):
                return annotation_id
        return ""

    def set_peer_cursor(self, pixel: tuple[int, int] | None) -> None:
        scene = self.graphics_scene.sceneRect()
        if pixel is None or scene.isEmpty():
            self.peer_cursor_item.setVisible(False)
            return
        x, y = pixel
        if not (0 <= x < int(scene.width()) and 0 <= y < int(scene.height())):
            self.peer_cursor_item.setVisible(False)
            return
        self.peer_cursor_item.setPos(scene.x() + x + 0.5, scene.y() + y + 0.5)
        self.peer_cursor_item.setVisible(True)

    def add_polygon(
        self,
        normalized_points: Sequence[QPointF],
        color: QColor,
        *,
        width: float = 2.0,
        fill: QColor | None = None,
        pen_style: Qt.PenStyle = Qt.PenStyle.SolidLine,
    ) -> None:
        rect = self.graphics_scene.sceneRect()
        if rect.isEmpty() or len(normalized_points) < 2:
            return
        polygon = QPolygonF(
            [QPointF(point.x() * rect.width(), point.y() * rect.height()) for point in normalized_points]
        )
        item = QGraphicsPolygonItem(polygon)
        pen = QPen(color, width, pen_style)
        pen.setCosmetic(True)
        item.setPen(pen)
        if fill is not None:
            item.setBrush(fill)
        item.setZValue(5)
        self.graphics_scene.addItem(item)
        self.vector_items.append(item)

    def add_label(
        self,
        normalized_position: QPointF,
        text: str,
        color: QColor,
    ) -> None:
        rect = self.graphics_scene.sceneRect()
        if rect.isEmpty():
            return
        item = QGraphicsSimpleTextItem(text)
        font = item.font()
        font.setBold(True)
        font.setPointSize(10)
        item.setFont(font)
        item.setBrush(color)
        item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        item.setPos(
            normalized_position.x() * rect.width(),
            normalized_position.y() * rect.height(),
        )
        item.setZValue(7)
        self.graphics_scene.addItem(item)
        self.vector_items.append(item)

    def add_rect(
        self,
        normalized_rect: QRectF,
        color: QColor,
        *,
        width: float = 2.0,
    ) -> None:
        rect = self.graphics_scene.sceneRect()
        item = QGraphicsRectItem(
            normalized_rect.x() * rect.width(),
            normalized_rect.y() * rect.height(),
            normalized_rect.width() * rect.width(),
            normalized_rect.height() * rect.height(),
        )
        pen = QPen(color, width)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(5)
        self.graphics_scene.addItem(item)
        self.vector_items.append(item)

    def add_line(
        self,
        normalized_start: QPointF,
        normalized_end: QPointF,
        color: QColor,
        *,
        width: float = 2.0,
        pen_style: Qt.PenStyle = Qt.PenStyle.DashLine,
    ) -> None:
        rect = self.graphics_scene.sceneRect()
        if rect.isEmpty():
            return
        item = QGraphicsLineItem(
            normalized_start.x() * rect.width(),
            normalized_start.y() * rect.height(),
            normalized_end.x() * rect.width(),
            normalized_end.y() * rect.height(),
        )
        pen = QPen(color, width, pen_style)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(6)
        self.graphics_scene.addItem(item)
        self.vector_items.append(item)

    def focus_normalized_rect(self, rect: QRectF) -> None:
        scene = self.graphics_scene.sceneRect()
        target = QRectF(
            rect.x() * scene.width(),
            rect.y() * scene.height(),
            max(1.0, rect.width() * scene.width()),
            max(1.0, rect.height() * scene.height()),
        )
        margin_x = target.width() * 0.25
        margin_y = target.height() * 0.25
        self.fitInView(
            target.adjusted(-margin_x, -margin_y, margin_x, margin_y),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.view_changed.emit()

    def visible_normalized_rect(self) -> QRectF:
        scene = self.graphics_scene.sceneRect()
        if scene.isEmpty():
            return QRectF(0.0, 0.0, 1.0, 1.0)
        visible = self.mapToScene(self.viewport().rect()).boundingRect().intersected(scene)
        if visible.isEmpty():
            return QRectF(0.0, 0.0, 1.0, 1.0)
        return QRectF(
            (visible.x() - scene.x()) / scene.width(),
            (visible.y() - scene.y()) / scene.height(),
            visible.width() / scene.width(),
            visible.height() / scene.height(),
        )

    def center_on_normalized(self, point: QPointF) -> None:
        scene = self.graphics_scene.sceneRect()
        if scene.isEmpty():
            return
        self.centerOn(
            scene.x() + point.x() * scene.width(),
            scene.y() + point.y() * scene.height(),
        )
        self.view_changed.emit()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self.view_changed.emit()

    def _bounded_scene_point(self, position: Any) -> QPointF:
        point = self.mapToScene(position)
        rect = self.graphics_scene.sceneRect()
        return QPointF(
            min(max(point.x(), rect.left()), rect.right()),
            min(max(point.y(), rect.top()), rect.bottom()),
        )

    def mousePressEvent(self, event: Any) -> None:
        if self.draw_mode == "pan" and event.button() == Qt.MouseButton.LeftButton:
            annotation_id = self._annotation_at_view_point(event.position().toPoint())
            self._pending_annotation_click = (
                (annotation_id, event.position()) if annotation_id else None
            )
            self._annotation_click_dragged = False
        if self.draw_mode == "polygon":
            scene = self.graphics_scene.sceneRect()
            if scene.isEmpty():
                event.accept()
                return
            if event.button() == Qt.MouseButton.RightButton:
                points = self.freehand_points
                area = abs(sum(a.x()*b.y()-b.x()*a.y()
                               for a, b in zip(points, points[1:]+points[:1]))) / 2
                if len(points) >= 3 and area >= 1.0:
                    normalized = [QPointF(p.x()/scene.width(), p.y()/scene.height()) for p in points]
                    # Clear before emission: saving/redrawing may replace the scene.
                    self.cancel_current_draw()
                    self.freehand_drawn.emit(normalized)
                event.accept()
                return
            if event.button() == Qt.MouseButton.LeftButton:
                point = self._bounded_scene_point(event.position().toPoint())
                if len(self.freehand_points) < 1000 and (
                        not self.freehand_points or point != self.freehand_points[-1]):
                    self.freehand_points.append(point)
                if self.temporary_item is None:
                    self.temporary_item = QGraphicsPathItem()
                    pen = QPen(QColor("#ffd166"), 2.0)
                    pen.setCosmetic(True)
                    self.temporary_item.setPen(pen)
                    self.temporary_item.setZValue(10)
                    self.graphics_scene.addItem(self.temporary_item)
                self._preview_polygon()
                event.accept()
                return
        if self.draw_mode == "pan" or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.start_point = self._bounded_scene_point(event.position().toPoint())
        if self.draw_mode == "box":
            self.temporary_item = QGraphicsRectItem(QRectF(self.start_point, self.start_point))
            pen = QPen(QColor("#ffd166"), 2.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self.temporary_item.setPen(pen)
        elif self.draw_mode in {"horizontal-guide", "vertical-guide"}:
            scene = self.graphics_scene.sceneRect()
            if self.draw_mode == "horizontal-guide":
                self.temporary_item = QGraphicsLineItem(
                    scene.left(),
                    self.start_point.y(),
                    scene.right(),
                    self.start_point.y(),
                )
                color = QColor("#ffd166")
            else:
                self.temporary_item = QGraphicsLineItem(
                    self.start_point.x(),
                    scene.top(),
                    self.start_point.x(),
                    scene.bottom(),
                )
                color = QColor("#66d9ef")
            pen = QPen(color, 2.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self.temporary_item.setPen(pen)
        else:
            self.freehand_points = [self.start_point]
            self.temporary_item = QGraphicsPathItem()
            pen = QPen(QColor("#ffd166"), 2.0)
            pen.setCosmetic(True)
            self.temporary_item.setPen(pen)
        self.temporary_item.setZValue(10)
        self.graphics_scene.addItem(self.temporary_item)
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        if self._pending_annotation_click is not None:
            _annotation_id, press_position = self._pending_annotation_click
            if (event.position() - press_position).manhattanLength() > 4.0:
                self._annotation_click_dragged = True
        scene = self.graphics_scene.sceneRect()
        point = self.mapToScene(event.position().toPoint())
        if (not scene.isEmpty() and scene.contains(point)):
            self.native_pointer_moved.emit((
                int(math.floor(point.x() - scene.x())),
                int(math.floor(point.y() - scene.y())),
            ))
        else:
            self.native_pointer_left.emit()
        if self.draw_mode == "polygon" and self.freehand_points:
            self._preview_polygon(self._bounded_scene_point(event.position().toPoint()))
            event.accept()
            return
        if self.start_point is None or self.temporary_item is None:
            super().mouseMoveEvent(event)
            return
        point = self._bounded_scene_point(event.position().toPoint())
        if self.draw_mode == "box":
            self.temporary_item.setRect(QRectF(self.start_point, point).normalized())
        elif self.draw_mode == "horizontal-guide":
            scene = self.graphics_scene.sceneRect()
            self.temporary_item.setLine(
                scene.left(), point.y(), scene.right(), point.y()
            )
        elif self.draw_mode == "vertical-guide":
            scene = self.graphics_scene.sceneRect()
            self.temporary_item.setLine(
                point.x(), scene.top(), point.x(), scene.bottom()
            )
        else:
            if not self.freehand_points or math.hypot(
                point.x() - self.freehand_points[-1].x(),
                point.y() - self.freehand_points[-1].y(),
            ) >= 2.0:
                self.freehand_points.append(point)
            path = QPainterPath(self.freehand_points[0])
            for value in self.freehand_points[1:]:
                path.lineTo(value)
            self.temporary_item.setPath(path)
        event.accept()

    def leaveEvent(self, event: Any) -> None:
        self.native_pointer_left.emit()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self.draw_mode == "pan" and self._pending_annotation_click is not None:
            annotation_id, press_position = self._pending_annotation_click
            self._pending_annotation_click = None
            clicked = (
                event.button() == Qt.MouseButton.LeftButton
                and not self._annotation_click_dragged
                and (event.position() - press_position).manhattanLength() <= 4.0
            )
            self._annotation_click_dragged = False
            super().mouseReleaseEvent(event)
            if clicked:
                self.annotation_clicked.emit(annotation_id)
            return
        if self.draw_mode == "polygon":
            event.accept()
            return
        if self.start_point is None or self.temporary_item is None:
            super().mouseReleaseEvent(event)
            return
        scene = self.graphics_scene.sceneRect()
        if self.draw_mode == "box":
            rect = self.temporary_item.rect().normalized()
            if rect.width() >= 3.0 and rect.height() >= 3.0:
                self.rectangle_drawn.emit(
                    QRectF(
                        rect.x() / scene.width(),
                        rect.y() / scene.height(),
                        rect.width() / scene.width(),
                        rect.height() / scene.height(),
                    )
                )
        elif self.draw_mode in {"horizontal-guide", "vertical-guide"}:
            point = self._bounded_scene_point(event.position().toPoint())
            orientation = (
                "horizontal"
                if self.draw_mode == "horizontal-guide"
                else "vertical"
            )
            self.guide_placed.emit(
                orientation,
                QPointF(point.x() / scene.width(), point.y() / scene.height()),
            )
        elif len(self.freehand_points) >= 3:
            self.freehand_drawn.emit(
                [QPointF(point.x() / scene.width(), point.y() / scene.height()) for point in self.freehand_points]
            )
        self.graphics_scene.removeItem(self.temporary_item)
        self.temporary_item = None
        self.start_point = None
        self.freehand_points = []
        event.accept()

    def _preview_polygon(self, cursor: QPointF | None = None) -> None:
        if not self.freehand_points or self.temporary_item is None:
            return
        path = QPainterPath(self.freehand_points[0])
        for point in self.freehand_points[1:]:
            path.lineTo(point)
        if cursor is not None:
            path.lineTo(cursor)
        radius = 3.0 / max(abs(self.transform().m11()), 1e-6)
        for point in self.freehand_points:
            path.addEllipse(point, radius, radius)
        self.temporary_item.setPath(path)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_Escape and self.draw_mode == "polygon" and self.freehand_points:
            self.cancel_current_draw()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if self.draw_mode == "polygon":
            self.mousePressEvent(event)
            return
        super().mouseDoubleClickEvent(event)


class ScaledImageLabel(QLabel):
    """Aspect-preserving image label used for candidate mass plots."""

    def __init__(self, placeholder: str = "Loading plot...") -> None:
        super().__init__(placeholder)
        self.source_pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(170)
        self.setStyleSheet(
            "background:#111417; border:1px solid #343c43; color:#aeb7bf;"
        )

    def set_image(self, image: QImage) -> None:
        self.source_pixmap = QPixmap.fromImage(image)
        self._update_scaled_pixmap()

    def clear_image(self, placeholder: str = "Loading plot...") -> None:
        self.source_pixmap = QPixmap()
        self.setPixmap(QPixmap())
        self.setText(placeholder)

    def _update_scaled_pixmap(self) -> None:
        if self.source_pixmap.isNull():
            return
        self.setText("")
        target = QSize(max(1, self.width() - 10), max(1, self.height() - 10))
        self.setPixmap(
            self.source_pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()


def _quiz_generation_campaign(candidate: Any) -> Path | None:
    """Resolve only the selected model's own map package, never a sibling run."""
    if not getattr(candidate, "phase_model_only", False):
        return None
    if getattr(candidate, "phase_quiz_proposals_path", None):
        return None  # Explicit historical/targeted pools retain their own binding.
    raw_root = getattr(candidate, "phase_map_output_dir", None)
    if not raw_root:
        return None
    root = Path(raw_root)
    for package in (root, root.parent):
        manifest = package / "postflight_manifest.json"
        if not manifest.is_file():
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        matches = [job for job in payload.get("jobs", [])
                   if str(job.get("run", "")).casefold() == candidate.run.casefold()
                   and job.get("candidate_id") == candidate.candidate_id
                   and job.get("params_sha256") == candidate.params_sha256]
        if len(matches) == 1:
            return package
    return None


class QuizPreparationSignals(QObject):
    progress = Signal(object, str)
    finished = Signal(object, object, str)


class QuizPreparationTask(_SavedReaderContextTask):
    """One requested experiment; the image/annotation UI remains responsive."""

    def __init__(self, key: tuple[str, str], campaign: Path) -> None:
        super().__init__()
        self.key, self.campaign = key, campaign
        self.cancelled = threading.Event()
        self.signals = QuizPreparationSignals()

    def _run_saved_read(self) -> None:
        try:
            from darsia.fluidflower.phase_saved_review_proposals import ensure_review_proposals
            from darsia.fluidflower.phase_saved_review_metadata import configured_parameters

            def progress(message):
                if self.cancelled.is_set():
                    raise RuntimeError("Quiz preparation cancelled because its window was closed")
                self.signals.progress.emit(self.key, str(message))

            progress("Reading saved model-selected regions")
            path = ensure_review_proposals(
                self.campaign, self.key[0], progress=progress,
                parameters=configured_parameters("quiz"),
            )
            payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            self.signals.finished.emit(self.key, payload, "")
        except Exception as exc:  # noqa: BLE001 - visible asynchronous boundary
            self.signals.finished.emit(self.key, {}, f"{type(exc).__name__}: {exc}")


class SaveSignals(QObject):
    finished = Signal(int, str, object, str)


class PresentationSignals(QObject):
    progress = Signal(object, str)
    finished = Signal(object, str, str)


class PresentationTask(_SavedReaderContextTask):
    """Render a requested diagnostic/video without blocking native playback."""

    def __init__(self, candidate, kind):
        super().__init__()
        self.candidate, self.kind = candidate, kind
        self.key = (str(candidate.summary_path), kind)
        self.cancelled = threading.Event()
        self.signals = PresentationSignals()

    def _run_saved_read(self):
        try:
            from darsia.fluidflower.phase_saved_presentation import ensure_candidate_presentation
            def progress(message):
                if self.cancelled.is_set():
                    raise RuntimeError("Presentation preparation cancelled")
                self.signals.progress.emit(self.key, str(message))
            path = ensure_candidate_presentation(self.candidate, self.kind, progress=progress)
            self.signals.finished.emit(self.key, str(path), "")
        except Exception as exc:  # noqa: BLE001 - visible asynchronous boundary
            self.signals.finished.emit(self.key, "", f"{type(exc).__name__}: {exc}")


class SaveTask(QRunnable):
    def __init__(self, generation: int, run: str, store: review.ReviewStore, request: Mapping[str, Any]) -> None:
        super().__init__()
        self.generation = generation
        self.run_name = run
        self.store = store
        self.request = deepcopy(dict(request))
        self.signals = SaveSignals()

    def run(self) -> None:
        try:
            result = self.store.save(self.request)
            self.signals.finished.emit(self.generation, self.run_name, result, "")
        except Exception as exc:  # noqa: BLE001 - asynchronous persistence boundary
            self.signals.finished.emit(
                self.generation,
                self.run_name,
                {},
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )


class CalibrationReviewWindow(ReviewImageAssets, QMainWindow):
    def __init__(
        self,
        app_model: review.ReviewApp,
        store: review.ReviewStore,
        cache_root: Path,
        workspace: ReviewWorkspace,
        *,
        reviewer_name: str,
        use_opengl: bool,
        on_back: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        from darsia.fluidflower.ui.interaction_safety import install_interaction_safety
        install_interaction_safety()
        self.app_model = app_model
        self.store = store
        self.workspace = workspace
        self.on_back = on_back
        self.cache_root = cache_root.resolve()
        self.image_cache = AsyncImageCache(self.cache_root / "assets")
        self.image_cache.image_ready.connect(self._image_loaded)
        self.save_pool = QThreadPool(self)
        self.save_pool.setMaxThreadCount(1)
        self.save_tasks: dict[int, SaveTask] = {}
        self.save_task_requests: dict[int, dict[str, Any]] = {}
        self.pending_save_requests: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.save_generation = 0
        self.saved_generation = 0
        self.save_inflight = False
        self.save_dirty = False
        self.save_status_state = "blocked"
        self.sync_error_pending = False
        self.sync_retry_ms = 30_000
        self.sync_retry_max_ms = 300_000
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._start_background_save)

        self.quiz_preparation_pool = QThreadPool(self)
        self.quiz_preparation_pool.setMaxThreadCount(1)
        self.quiz_preparation_task: QuizPreparationTask | None = None
        self.quiz_preparation_requested: set[tuple[str, str]] = set()
        self.quiz_preparation_errors: dict[tuple[str, str], str] = {}
        self.quiz_preparation_closed = False

        self.current_run = ""
        self.current_pair: dict[str, Any] | None = None
        self.current_pair_object: Any = None
        self.comparison_mode = "pair"
        self.current_review_root: Path | None = None
        self.frame_index = 0
        self.quiz_index = 0
        self.pool: list[dict[str, Any]] = []
        self.selected_ids: list[str] = []
        self.quiz_rounds: list[dict[str, Any]] = []
        self.answers: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.deleted_phase_quiz_answer_keys: set[str] = set()
        self.deleted_annotation_ids: set[str] = set()
        self.last_journal_hash_by_run: dict[str, str] = {}
        self.flags: dict[str, list[str]] = {"A": [], "B": []}
        self.base_key = ""
        self.overlay_key = ""
        self.reference_mode = False
        self.fit_next_image = True
        self.focus_after_load = False
        self.tool_buttons_by_mode: dict[str, list[QPushButton]] = {
            "pan": [],
            "box": [],
            "freehand": [],
            "polygon": [],
        }

        self.setWindowTitle(f"{workspace.title} - FluidFlower Review Desktop")
        self.resize(1760, 1040)
        self.setMinimumSize(1200, 760)
        remembered_reviewer = reviewer_name.strip() or self._remembered_reviewer()
        self._build_ui(remembered_reviewer, use_opengl)
        self._apply_review_surface_profile()
        self._populate_runs()
        if self.run_list.count():
            self.run_list.setCurrentRow(0)
        self._apply_read_only_profile()

    def _build_ui(self, reviewer_name: str, use_opengl: bool) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QHBoxLayout()
        if self.on_back is not None:
            back = QPushButton("< Start")
            back.setObjectName("backButton")
            back.setToolTip("Return to the workspace selector")
            back.clicked.connect(self._return_to_start)
            header.addWidget(back)
        title = QLabel(self.workspace.title)
        title.setObjectName("pageTitle")
        header.addWidget(title)
        mode = QLabel(self.workspace.short_label.upper())
        mode.setObjectName("workspaceBadge")
        mode.setProperty("blind", self.workspace.blind)
        mode.setToolTip(self.workspace.description)
        header.addWidget(mode)
        header.addStretch(1)
        header.addWidget(QLabel("Reviewer"))
        self.reviewer = QLineEdit(reviewer_name)
        self.reviewer.setMinimumWidth(220)
        self.reviewer.setPlaceholderText("Enter your name before reviewing")
        self.reviewer.setReadOnly(bool(reviewer_name.strip()))
        if reviewer_name.strip():
            self.reviewer.setToolTip("Reviewer identity is selected on the start screen.")
        self.reviewer.textChanged.connect(self._reviewer_text_changed)
        self.reviewer.editingFinished.connect(self._reviewer_committed)
        header.addWidget(self.reviewer)
        self.save_status_icon = QLabel()
        self.save_status_icon.setObjectName("saveStatusIcon")
        self.save_status_icon.setFixedSize(13, 13)
        header.addWidget(self.save_status_icon)
        self.sync_label = QLabel("Enter reviewer name to begin")
        self.sync_label.setMinimumWidth(285)
        header.addWidget(self.sync_label)
        open_saved = QPushButton("Open saved files")
        open_saved.setToolTip(
            "Open the local folder containing review JSON, CSV, and recovery history."
        )
        open_saved.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.workspace.output_dir.resolve()))
            )
        )
        header.addWidget(open_saved)
        root.addLayout(header)

        self.workspace_notice = QLabel(self.workspace.description)
        self.workspace_notice.setObjectName(
            "blindWorkspaceNotice" if self.workspace.blind else "trainingWorkspaceNotice"
        )
        self.workspace_notice.setWordWrap(True)
        self.workspace_notice.setToolTip(
            f"Decisions are stored separately in {self.workspace.output_dir}"
        )
        root.addWidget(self.workspace_notice)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Experiments"))
        self.run_list = QListWidget()
        self.run_list.setMinimumWidth(170)
        self.run_list.currentItemChanged.connect(self._run_changed)
        left_layout.addWidget(self.run_list, 1)
        refresh = QPushButton("Refresh experiments")
        refresh.clicked.connect(self._populate_runs)
        left_layout.addWidget(refresh)
        splitter.addWidget(left)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.previous_frame = QToolButton()
        self.previous_frame.setText("<")
        self.previous_frame.setToolTip("Previous frame")
        self.previous_frame.clicked.connect(lambda: self._set_frame(self.frame_index - 1))
        toolbar.addWidget(self.previous_frame)
        self.next_frame = QToolButton()
        self.next_frame.setText(">")
        self.next_frame.setToolTip("Next frame")
        self.next_frame.clicked.connect(lambda: self._set_frame(self.frame_index + 1))
        toolbar.addWidget(self.next_frame)
        self.frame_slider = FrameTimelineSlider()
        self.frame_slider.valueChanged.connect(self._set_frame)
        toolbar.addWidget(self.frame_slider, 1)
        self.frame_label = QLabel("Frame --")
        self.frame_label.setMinimumWidth(190)
        toolbar.addWidget(self.frame_label)
        self.candidate_side = QComboBox()
        self.candidate_side.addItem("Candidate A", "A")
        self.candidate_side.addItem("Candidate B", "B")
        self.candidate_side.currentIndexChanged.connect(self._reload_overlay_and_vectors)
        toolbar.addWidget(self.candidate_side)
        self.contours = QCheckBox("Phase map")
        self.contours.stateChanged.connect(self._reload_overlay_and_vectors)
        toolbar.addWidget(self.contours)
        self.contour_filter = QComboBox()
        self.contour_filter.addItem("Optical phases · context OFF", "all")
        self.contour_filter.addItem("Free CO2 only", "gas")
        self.contour_filter.addItem("Aqueous CO2 only", "aqueous")
        self.contour_filter.addItem("Unresolved source support (cyan)", "source_support")
        self.contour_filter.setToolTip(
            "Choose which filled model phase class is shown over the corrected image."
        )
        self.contour_filter.currentIndexChanged.connect(
            self._reload_overlay_and_vectors
        )
        toolbar.addWidget(self.contour_filter)
        self.injectors = QCheckBox("Injection points")
        self.injectors.stateChanged.connect(self._render_vectors)
        toolbar.addWidget(self.injectors)
        self.reference = QCheckBox("Pre-injection reference")
        self.reference.stateChanged.connect(self._reference_changed)
        toolbar.addWidget(self.reference)
        self.raw_detail = QCheckBox("RAW detail (display)")
        self.raw_detail.setToolTip(
            "Show the SHA-verified fixed-LUT RAW-detail photograph. The phase map, "
            "scientific float signal and all annotation coordinates are unchanged.")
        self.raw_detail.setEnabled(False)
        self.raw_detail.toggled.connect(lambda _checked: self._display_settings_changed())
        toolbar.addWidget(self.raw_detail)
        self._add_brightness_control(toolbar)
        center_layout.addLayout(toolbar)

        self.focus_frame_label = QLabel("No focus-region source frame")
        self.focus_frame_label.setObjectName("focusFrameLabel")
        center_layout.addWidget(self.focus_frame_label)

        self.viewer = ReviewImageView(use_opengl=use_opengl)
        self.viewer.rectangle_drawn.connect(self._add_rectangle_annotation)
        self.viewer.freehand_drawn.connect(self._add_freehand_annotation)
        viewer_row = QHBoxLayout()
        viewer_row.setSpacing(8)
        viewer_row.addWidget(self.viewer, 1)
        overview_panel = QFrame()
        overview_panel.setObjectName("overviewPanel")
        overview_panel.setMinimumWidth(220)
        overview_panel.setMaximumWidth(260)
        overview_layout = QVBoxLayout(overview_panel)
        overview_layout.setContentsMargins(6, 6, 6, 6)
        overview_title = QLabel("Location in full rig")
        overview_title.setObjectName("overviewTitle")
        overview_layout.addWidget(overview_title)
        self.minimap = ReviewMinimap()
        self.minimap.center_requested.connect(self.viewer.center_on_normalized)
        overview_layout.addWidget(self.minimap)
        self.minimap_caption = QLabel("Full frame")
        self.minimap_caption.setObjectName("overviewCaption")
        self.minimap_caption.setWordWrap(True)
        overview_layout.addWidget(self.minimap_caption)
        overview_layout.addStretch(1)
        viewer_row.addWidget(overview_panel)
        center_layout.addLayout(viewer_row, 1)
        self.viewer.view_changed.connect(self._update_minimap_viewport)

        view_controls = QHBoxLayout()
        for label, callback, tooltip in (
            ("-", lambda: self.viewer.zoom(1.0 / 1.25), "Zoom out"),
            ("Fit", self.viewer.fit_image, "Fit complete image"),
            ("1 px", self.viewer.native_image_zoom, "One image pixel per screen pixel"),
            ("+", lambda: self.viewer.zoom(1.25), "Zoom in"),
        ):
            button = QPushButton(label)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            view_controls.addWidget(button)
        view_controls.addSpacing(10)
        self.manual_draw_label = QLabel("Manual annotation:")
        self.manual_draw_label.setStyleSheet("font-weight:600; color:#d8dde2;")
        view_controls.addWidget(self.manual_draw_label)
        self.pan_tool = QPushButton("Pan")
        self.box_tool = QPushButton("Draw box")
        self.freehand_tool = QPushButton("Draw freehand area")
        self.polygon_tool = QPushButton("Draw polygon")
        self.polygon_tool.setToolTip("Left-click at least three corners. Right-click closes the area. Esc cancels.")
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for button, mode in (
            (self.pan_tool, "pan"),
            (self.box_tool, "box"),
            (self.freehand_tool, "freehand"),
            (self.polygon_tool, "polygon"),
        ):
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, value=mode: self._set_draw_mode(value))
            self.tool_group.addButton(button)
            self.tool_buttons_by_mode[mode].append(button)
            view_controls.addWidget(button)
        self.pan_tool.setChecked(True)
        view_controls.addStretch(1)
        self.asset_label = QLabel("No image loaded")
        view_controls.addWidget(self.asset_label)
        center_layout.addLayout(view_controls)
        splitter.addWidget(center)

        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setMinimumWidth(410)
        panel_scroll.setMaximumWidth(520)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 4, 8, 8)

        self.quiz_title = QLabel("Blind phase labeling · model-selected regions")
        self.quiz_title.setObjectName("sectionTitle")
        panel_layout.addWidget(self.quiz_title)
        self.quiz_progress = QLabel("No experiment selected")
        self.quiz_progress.setWordWrap(True)
        panel_layout.addWidget(self.quiz_progress)
        quiz_nav = QHBoxLayout()
        self.previous_region = QPushButton("Previous")
        self.previous_region.clicked.connect(lambda: self._move_in_quiz_view(-1))
        quiz_nav.addWidget(self.previous_region)
        self.next_region = QPushButton("Next")
        self.next_region.clicked.connect(lambda: self._move_in_quiz_view(1))
        quiz_nav.addWidget(self.next_region)
        self.next_unanswered = QPushButton("Next unanswered")
        self.next_unanswered.clicked.connect(self._open_next_new_region)
        quiz_nav.addWidget(self.next_unanswered)
        panel_layout.addLayout(quiz_nav)
        add_regions_frame = QFrame()
        add_regions_frame.setObjectName("addQuizRegions")
        add_regions_layout = QVBoxLayout(add_regions_frame)
        add_regions_layout.setContentsMargins(8, 7, 8, 7)
        add_regions_layout.setSpacing(5)
        add_regions_title = QLabel("Request more informative regions / frames")
        add_regions_title.setStyleSheet("font-weight:600;")
        add_regions_layout.addWidget(add_regions_title)
        add_regions_controls = QHBoxLayout()
        add_regions_controls.addWidget(QLabel("Number to add"))
        self.new_round_count = QSpinBox()
        self.new_round_count.setRange(1, 1_000_000)
        self.new_round_count.setValue(8)
        self.new_round_count.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.new_round_count.setKeyboardTracking(False)
        self.new_round_count.setMinimumWidth(88)
        self.new_round_count.setToolTip(
            "Type any positive number. If fewer unused regions remain, all available regions are added."
        )
        add_regions_controls.addWidget(self.new_round_count)
        self.new_round = QPushButton("Add regions")
        self.new_round.clicked.connect(self._start_new_round)
        self.new_round_count.valueChanged.connect(
            self._update_add_regions_button
        )
        add_regions_controls.addWidget(self.new_round, 1)
        add_regions_layout.addLayout(add_regions_controls)
        self.quiz_availability = QLabel("No unused regions available")
        self.quiz_availability.setWordWrap(True)
        self.quiz_availability.setStyleSheet("color:#aeb7bf;")
        add_regions_layout.addWidget(self.quiz_availability)
        self.retry_quiz_preparation = QPushButton("Retry preparing regions")
        self.retry_quiz_preparation.setVisible(False)
        self.retry_quiz_preparation.clicked.connect(self._retry_quiz_preparation)
        add_regions_layout.addWidget(self.retry_quiz_preparation)
        panel_layout.addWidget(add_regions_frame)

        choice_grid = QGridLayout()
        self.phase_buttons: dict[str, QPushButton] = {}
        for index, (key, label) in enumerate(PHASE_CHOICES):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, value=key: self._answer(value))
            self.phase_buttons[key] = button
            choice_grid.addWidget(button, index, 0)
        panel_layout.addLayout(choice_grid)
        answer_controls = QHBoxLayout()
        answer_controls.addWidget(QLabel("Confidence"))
        self.confidence = QComboBox()
        for value in ("high", "medium", "low"):
            self.confidence.addItem(value.title(), value)
        self.confidence.setCurrentIndex(1)
        self.confidence.currentIndexChanged.connect(self._update_existing_answer)
        answer_controls.addWidget(self.confidence)
        self.ignore_region = QPushButton("Ignore region")
        self.ignore_region.clicked.connect(lambda: self._answer("ignore_region"))
        answer_controls.addWidget(self.ignore_region)
        self.clear_answer = QPushButton("Delete label")
        self.clear_answer.setToolTip(
            "Delete the selected answer and return the region to New / unlabeled."
        )
        self.clear_answer.clicked.connect(self._clear_answer)
        answer_controls.addWidget(self.clear_answer)
        panel_layout.addLayout(answer_controls)
        self.answer_note = QLineEdit()
        self.answer_note.setPlaceholderText("Optional note about this region")
        self.answer_note.editingFinished.connect(self._update_existing_answer)
        panel_layout.addWidget(self.answer_note)
        self.auto_advance = QCheckBox("Advance automatically after answer")
        self.auto_advance.setChecked(True)
        panel_layout.addWidget(self.auto_advance)

        quiz_list_controls = QGridLayout()
        quiz_list_controls.addWidget(QLabel("Review order"), 0, 0)
        self.quiz_sort = QComboBox()
        for label, value in (
            ("Time: early to late", "time_asc"),
            ("Time: late to early", "time_desc"),
            ("Class: A to Z", "class_asc"),
            ("Order added", "added"),
        ):
            self.quiz_sort.addItem(label, value)
        self.quiz_sort.currentIndexChanged.connect(
            self._quiz_view_controls_changed
        )
        quiz_list_controls.addWidget(self.quiz_sort, 0, 1)
        quiz_list_controls.addWidget(QLabel("Labeled class"), 1, 0)
        self.quiz_class_filter = QComboBox()
        self.quiz_class_filter.addItem("All labeled classes", "all")
        for key, label in PHASE_CHOICES:
            self.quiz_class_filter.addItem(label, key)
        self.quiz_class_filter.addItem("Ignored region", "ignore_region")
        self.quiz_class_filter.currentIndexChanged.connect(
            self._quiz_view_controls_changed
        )
        self.quiz_class_filter.setToolTip(
            "Filters the Labeled tab. New regions have no class yet and remain visible in New / unlabeled."
        )
        quiz_list_controls.addWidget(self.quiz_class_filter, 1, 1)
        panel_layout.addLayout(quiz_list_controls)

        self.quiz_tabs = QTabWidget()
        self.new_region_list = QListWidget()
        self.new_region_list.setMinimumHeight(150)
        self.new_region_list.itemActivated.connect(self._quiz_item_opened)
        self.new_region_list.itemClicked.connect(self._quiz_item_opened)
        self.validated_list = QListWidget()
        self.validated_list.setMinimumHeight(150)
        self.validated_list.itemActivated.connect(self._quiz_item_opened)
        self.validated_list.itemClicked.connect(self._quiz_item_opened)
        self.quiz_tabs.addTab(self.new_region_list, "New / unlabeled (0)")
        self.quiz_tabs.addTab(self.validated_list, "Labeled (0)")
        self.quiz_tabs.currentChanged.connect(self._quiz_tab_changed)
        panel_layout.addWidget(self.quiz_tabs)
        self.quiz_entry_help = QLabel(
            "Select a labeled entry, then choose another phase class above to change it. Delete label returns it to New / unlabeled."
        )
        self.quiz_entry_help.setWordWrap(True)
        self.quiz_entry_help.setStyleSheet("color:#aeb7bf;")
        panel_layout.addWidget(self.quiz_entry_help)

        self.manual_evidence_panel = QFrame()
        manual_evidence_layout = QVBoxLayout(self.manual_evidence_panel)
        manual_evidence_layout.setContentsMargins(0, 0, 0, 0)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        manual_evidence_layout.addWidget(separator)
        annotation_title = QLabel("Manual image evidence")
        annotation_title.setObjectName("sectionTitle")
        manual_evidence_layout.addWidget(annotation_title)

        manual_tools = QHBoxLayout()
        self.manual_tool_group = QButtonGroup(self)
        self.manual_tool_group.setExclusive(True)
        for mode, label in (
            ("pan", "Pan"),
            ("box", "Draw box"),
            ("freehand", "Draw freehand area"),
            ("polygon", "Draw polygon"),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(
                lambda _checked=False, value=mode: self._set_draw_mode(value)
            )
            self.manual_tool_group.addButton(button)
            self.tool_buttons_by_mode[mode].append(button)
            manual_tools.addWidget(button)
        manual_evidence_layout.addLayout(manual_tools)

        self.annotation_category = QComboBox()
        for key, label, _phase in ANNOTATION_CATEGORIES:
            self.annotation_category.addItem(label, key)
            if key in review.CANDIDATE_ISSUE_HELP:
                self.annotation_category.setItemData(
                    self.annotation_category.count() - 1,
                    review.CANDIDATE_ISSUE_HELP[key],
                    Qt.ItemDataRole.ToolTipRole,
                )
        manual_evidence_layout.addWidget(self.annotation_category)
        annotation_actions = QHBoxLayout()
        self.update_annotation_class = QPushButton("Apply class to selected")
        self.update_annotation_class.setEnabled(False)
        self.update_annotation_class.clicked.connect(
            self._update_selected_annotation_class
        )
        annotation_actions.addWidget(self.update_annotation_class)
        self.delete_annotation_button = QPushButton("Delete selected")
        self.delete_annotation_button.setEnabled(False)
        self.delete_annotation_button.clicked.connect(self._delete_annotation)
        annotation_actions.addWidget(self.delete_annotation_button)
        manual_evidence_layout.addLayout(annotation_actions)
        annotation_help = QLabel(
            "Choose a phase or issue, select Draw box, Draw polygon or Draw freehand area, then mark the visible evidence directly in the image."
        )
        annotation_help.setWordWrap(True)
        manual_evidence_layout.addWidget(annotation_help)
        self.annotation_category.currentIndexChanged.connect(
            lambda _index: annotation_help.setText(
                review.CANDIDATE_ISSUE_HELP.get(
                    str(self.annotation_category.currentData() or ""),
                    "Choose a phase or issue, select Draw box, Draw polygon or Draw freehand area, then mark the visible evidence directly in the image.",
                )
            )
        )

        annotation_view_controls = QGridLayout()
        annotation_view_controls.addWidget(QLabel("Box order"), 0, 0)
        self.annotation_sort = QComboBox()
        for label, value in (
            ("Time: early to late", "time_asc"),
            ("Time: late to early", "time_desc"),
            ("Class: A to Z", "class_asc"),
            ("Order created", "created"),
        ):
            self.annotation_sort.addItem(label, value)
        self.annotation_sort.currentIndexChanged.connect(
            self._annotation_view_controls_changed
        )
        annotation_view_controls.addWidget(self.annotation_sort, 0, 1)
        annotation_view_controls.addWidget(QLabel("Show class"), 1, 0)
        self.annotation_class_filter = QComboBox()
        self.annotation_class_filter.addItem("All manual classes", "all")
        for key, label, _phase in ANNOTATION_CATEGORIES:
            self.annotation_class_filter.addItem(label, key)
        self.annotation_class_filter.currentIndexChanged.connect(
            self._annotation_view_controls_changed
        )
        self.annotation_class_filter.setToolTip(
            "Filters both this list and the manual boxes drawn over the image."
        )
        annotation_view_controls.addWidget(self.annotation_class_filter, 1, 1)
        manual_evidence_layout.addLayout(annotation_view_controls)
        self.annotation_filter_summary = QLabel("Showing 0 of 0 manual regions")
        self.annotation_filter_summary.setStyleSheet("color:#aeb7bf;")
        manual_evidence_layout.addWidget(self.annotation_filter_summary)
        self.annotation_list = QListWidget()
        self.annotation_list.setMinimumHeight(120)
        self.annotation_list.currentItemChanged.connect(
            self._annotation_selection_changed
        )
        self.annotation_list.itemActivated.connect(self._annotation_item_opened)
        manual_evidence_layout.addWidget(self.annotation_list)
        panel_layout.addWidget(self.manual_evidence_panel)

        self.experiment_verdict_panel = QFrame()
        experiment_verdict_layout = QVBoxLayout(self.experiment_verdict_panel)
        experiment_verdict_layout.setContentsMargins(0, 0, 0, 0)
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        experiment_verdict_layout.addWidget(separator2)
        verdict_form = QFormLayout()
        self.verdict = QComboBox()
        for label, value in (
            ("Unresolved", "unresolved"),
            ("Candidate A", "candidate_a"),
            ("Candidate B", "candidate_b"),
            ("Equivalent", "equivalent"),
            ("Acceptable", "acceptable"),
            ("Needs repair", "needs_repair"),
        ):
            self.verdict.addItem(label, value)
        self.verdict.currentIndexChanged.connect(self._queue_save)
        verdict_form.addRow("Experiment verdict", self.verdict)
        experiment_verdict_layout.addLayout(verdict_form)
        self.notes = QTextEdit()
        self.notes.setPlaceholderText("Experiment-level notes")
        self.notes.setMaximumHeight(100)
        self.notes.textChanged.connect(self._queue_save)
        experiment_verdict_layout.addWidget(self.notes)
        panel_layout.addWidget(self.experiment_verdict_panel)
        self.save_now = QPushButton("Save now")
        self.save_now.setToolTip(
            "Autosave is always active. This button immediately retries or flushes the current review."
        )
        self.save_now.clicked.connect(self._save_now)
        panel_layout.addWidget(self.save_now)
        panel_layout.addStretch(1)
        panel_scroll.setWidget(panel)
        splitter.addWidget(panel_scroll)
        splitter.setSizes([180, 1120, 440])
        self.validation_surface = splitter
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")
        self._set_draw_mode("pan")
        self._install_shortcuts()
        self._update_reviewer_gate()
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#171a1d; color:#e7eaed; font-size:13px; }
            #pageTitle { font-size:20px; font-weight:600; }
            #workspaceBadge { background:#25546c; color:#f2f8fb; border:1px solid #4d819b; border-radius:3px; padding:3px 7px; font-weight:600; }
            #blindWorkspaceNotice { background:#382f1d; color:#ffe3a0; border-left:4px solid #d7a934; padding:7px 9px; }
            #trainingWorkspaceNotice { background:#1e342d; color:#bfe5d7; border-left:4px solid #4a9f80; padding:7px 9px; }
            #sectionTitle { font-size:15px; font-weight:600; margin-top:4px; }
            #focusFrameLabel { color:#ef3e36; font-weight:600; padding:1px 3px; }
            #addQuizRegions { background:#20262b; border:1px solid #52616d; border-radius:4px; }
            #overviewPanel { background:#15191c; border:1px solid #343c43; border-radius:4px; }
            #overviewTitle { font-size:13px; font-weight:600; }
            #overviewCaption { color:#aeb7bf; font-size:12px; }
            QPushButton, QToolButton, QComboBox, QLineEdit, QTextEdit, QListWidget {
                background:#24292e; border:1px solid #3a4249; border-radius:4px; padding:5px;
            }
            QPushButton:hover, QToolButton:hover { border-color:#6da7d9; }
            QPushButton:checked { background:#185d86; border-color:#77bce8; }
            QPushButton:disabled { color:#747b82; }
            QListWidget::item { padding:4px; }
            QListWidget::item:selected { background:#185d86; }
            QSlider::groove:horizontal { height:5px; background:#343b42; }
            QSlider::handle:horizontal { width:13px; margin:-5px 0; background:#71b7df; }
            QScrollBar:vertical { background:#1b1e21; width:12px; }
            QScrollBar::handle:vertical { background:#454e56; min-height:24px; }
            #saveStatusIcon { border:1px solid #6b737a; border-radius:6px; background:#596168; }
            """
        )

    def _apply_review_surface_profile(self) -> None:
        """Enforce a genuinely blind generated-region labeling surface."""

        if self.workspace.review_surface != "blind_regions":
            return
        self.contours.blockSignals(True)
        self.contours.setChecked(False)
        self.contours.blockSignals(False)
        self.contours.setVisible(False)
        self.contour_filter.setVisible(False)
        self.candidate_side.setVisible(False)
        self.pan_tool.setText("Pan image")
        self.experiment_verdict_panel.setVisible(False)
        self.workspace_notice.setText(
            "Blind phase labeling: the selected model prioritizes uncertain regions, "
            "but its prediction and phase map remain hidden in this labeling session. "
            "You can also add manual boxes or freehand "
            "evidence. The corrected full-resolution image, timeline, pre-injection "
            "reference, and injection points remain available."
        )
        self._refresh_blind_reveal_notice()

    def _refresh_blind_reveal_notice(self) -> None:
        if (self.workspace.review_surface == "blind_regions"
                and (self.workspace.output_dir / "blind_reveal.json").is_file()):
            self.workspace_notice.setText(
                "Predictions have been revealed in the separate comparison. "
                "The original blind labels are preserved in a fixed snapshot. "
                "Further edits remain editable, but are not counted as blind evidence."
            )

    def _apply_read_only_profile(self) -> None:
        if not self.store.is_read_only:
            return
        self.setWindowTitle(f"{self.workspace.title} - Read-only evidence audit")
        self.workspace_notice.setText(
            "Read-only evidence audit: this session was frozen as input to an "
            "immutable model version. Navigate every experiment, frame, region, "
            "and saved correction below; no answer or annotation can be changed."
        )
        self.reviewer.setReadOnly(True)
        for control in (
            self.new_round_count,
            self.new_round,
            self.confidence,
            self.ignore_region,
            self.clear_answer,
            self.answer_note,
            self.annotation_category,
            self.update_annotation_class,
            self.delete_annotation_button,
            self.verdict,
            self.notes,
            self.save_now,
        ):
            control.setEnabled(False)
        for button in self.phase_buttons.values():
            button.setEnabled(False)
        for mode in ("box", "freehand", "polygon"):
            for button in self.tool_buttons_by_mode[mode]:
                button.setEnabled(False)

    def _reviewer_profile_path(self) -> Path:
        return reviewer_profile_path(self.cache_root)

    def _remembered_reviewer(self) -> str:
        return str(load_reviewer_profile(self.cache_root).get("reviewer") or "").strip()

    def _remember_reviewer(self, reviewer_name: str) -> None:
        reviewer_name = reviewer_name.strip()
        if reviewer_name:
            save_reviewer_profile(self.cache_root, reviewer_name=reviewer_name)

    def _set_save_status(self, state: str, text: str, tooltip: str = "") -> None:
        colors = {
            "blocked": ("#596168", "#8a939b"),
            "pending": ("#d6a53a", "#f0ca70"),
            "saving": ("#4d91c4", "#86c3ef"),
            "saved": ("#3fa66b", "#7bd69d"),
            "error": ("#c64f55", "#ef8c91"),
        }
        fill, border = colors.get(state, colors["blocked"])
        self.save_status_state = state
        self.save_status_icon.setStyleSheet(
            f"border:1px solid {border}; border-radius:6px; background:{fill};"
        )
        self.sync_label.setText(text)
        detail = tooltip or text
        self.save_status_icon.setToolTip(detail)
        self.sync_label.setToolTip(detail)

    def _update_reviewer_gate(self) -> None:
        ready = bool(self.reviewer.text().strip())
        self.validation_surface.setEnabled(ready)
        self.save_now.setEnabled(ready)
        if not ready:
            self._set_save_status(
                "blocked",
                "Enter reviewer name to begin",
                "Review choices and drawing are locked until a reviewer name is entered.",
            )
        elif not self.save_inflight and not self.pending_save_requests:
            self._set_save_status(
                "saved",
                "Autosave on · all changes saved",
                "Green means the durable review file is up to date. Every edit also gets an immediate recovery draft.",
            )

    def _reviewer_text_changed(self, _text: str) -> None:
        self._update_reviewer_gate()

    def _reviewer_committed(self) -> None:
        reviewer_name = self.reviewer.text().strip()
        if not reviewer_name:
            self._update_reviewer_gate()
            return
        self._remember_reviewer(reviewer_name)
        self._queue_save()

    def _return_to_start(self) -> None:
        if self.current_run:
            self._queue_save()
        self.hide()
        if self.on_back is not None:
            self.on_back()

    def _install_shortcuts(self) -> None:
        for shortcut, callback in (
            ("Right", lambda: self._set_frame(self.frame_index + 1)),
            ("Left", lambda: self._set_frame(self.frame_index - 1)),
            ("Ctrl+0", self.viewer.fit_image),
            ("Ctrl+S", self._start_background_save),
        ):
            action = QAction(self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.addAction(action)

    def _set_draw_mode(self, mode: str) -> None:
        self.viewer.set_draw_mode(mode)
        for value, buttons in self.tool_buttons_by_mode.items():
            for button in buttons:
                button.blockSignals(True)
                button.setChecked(value == mode)
                button.blockSignals(False)

    def _populate_runs(self) -> None:
        selected = self.current_run
        self.run_list.blockSignals(True)
        self.run_list.clear()
        state = self.app_model.state()
        for row in state.get("runs") or []:
            run = str(row.get("run") or "")
            dataset = str(row.get("dataset") or "").upper()
            status = "saved" if row.get("reviewed") else "ready"
            item = QListWidgetItem(f"{dataset} · {run.upper()}\n{status}")
            item.setData(Qt.ItemDataRole.UserRole, run)
            self.run_list.addItem(item)
            if run == selected:
                self.run_list.setCurrentItem(item)
        self.run_list.blockSignals(False)

    def _run_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        next_run = str(current.data(Qt.ItemDataRole.UserRole) or "").casefold()
        if self.current_run and next_run != self.current_run:
            # Persist the old run before replacing its in-memory state.
            self._queue_save()
        self._load_run(next_run)

    def _draft_scope_dir(self) -> Path:
        scope = hashlib.sha256(str(self.store.json_path).casefold().encode("utf-8")).hexdigest()[:16]
        return self.cache_root / "drafts" / scope

    def _local_record_scope_dir(self) -> Path:
        scope = hashlib.sha256(str(self.store.json_path).casefold().encode("utf-8")).hexdigest()[:16]
        return self.cache_root / "local_records" / scope

    def _journal_scope_dir(self) -> Path:
        scope = hashlib.sha256(str(self.store.json_path).casefold().encode("utf-8")).hexdigest()[:16]
        return self.cache_root / "journals" / scope

    def _draft_path(self, run: str, reviewer_name: str | None = None) -> Path:
        reviewer_name = (
            self.reviewer.text().strip()
            if reviewer_name is None
            else str(reviewer_name).strip()
        ) or "anonymous"
        reviewer_key = hashlib.sha256(reviewer_name.casefold().encode("utf-8")).hexdigest()[:16]
        return self._draft_scope_dir() / reviewer_key / f"{run}.json"

    def _local_record_path(self, run: str, reviewer_name: str | None = None) -> Path:
        reviewer_name = (
            self.reviewer.text().strip()
            if reviewer_name is None
            else str(reviewer_name).strip()
        ) or "anonymous"
        reviewer_key = hashlib.sha256(reviewer_name.casefold().encode("utf-8")).hexdigest()[:16]
        return self._local_record_scope_dir() / reviewer_key / f"{run}.json"

    def _journal_path(self, run: str, reviewer_name: str | None = None) -> Path:
        reviewer_name = (
            self.reviewer.text().strip()
            if reviewer_name is None
            else str(reviewer_name).strip()
        ) or "anonymous"
        reviewer_key = hashlib.sha256(reviewer_name.casefold().encode("utf-8")).hexdigest()[:16]
        return self._journal_scope_dir() / reviewer_key / f"{run}.jsonl"

    def _portable_recovery_dir(self) -> Path:
        return self.store.output_dir / "recovery"

    def _portable_recovery_path(self, run: str) -> Path:
        return self._portable_recovery_dir() / f"{run}.latest.json"

    def _portable_journal_path(self, run: str) -> Path:
        return self._portable_recovery_dir() / f"{run}.history.jsonl"

    @staticmethod
    def _append_journal(path: Path, payload: Mapping[str, Any]) -> None:
        append_bounded_snapshot_journal(path, payload)

    @staticmethod
    def _latest_journal_request(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as stream:
                lines = stream.readlines()
        except OSError:
            return None
        for line in reversed(lines):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            request = payload.get("request") if isinstance(payload, Mapping) else None
            if isinstance(request, Mapping):
                return dict(request)
        return None

    def _write_recovery_draft(self, request: Mapping[str, Any]) -> Path:
        run = str(request.get("run") or self.current_run).casefold()
        reviewer_name = str(request.get("reviewer") or "")
        path = self._draft_path(run, reviewer_name)
        atomic_json(path, request)
        atomic_json(
            self._local_record_path(run, reviewer_name),
            request,
        )
        portable_path = self._portable_recovery_path(run)
        atomic_json(portable_path, request)

        content = dict(request)
        content.pop("updated_utc", None)
        content_hash = hashlib.sha256(
            json.dumps(content, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        if self.last_journal_hash_by_run.get(run) != content_hash:
            journal_entry = {
                "schema": 1,
                "revision_id": uuid.uuid4().hex,
                "recorded_utc": utc_now(),
                "content_sha256": content_hash,
                "request": dict(request),
            }
            for journal_path in (
                self._journal_path(run, reviewer_name),
                self._portable_journal_path(run),
            ):
                self._append_journal(journal_path, journal_entry)
            self.last_journal_hash_by_run[run] = content_hash
        return portable_path

    def _draft_candidates(self, run: str) -> list[Path]:
        current = self._draft_path(run)
        local = self._local_record_path(run)
        candidates = [current, local, self._portable_recovery_path(run)]
        return list(dict.fromkeys(path.resolve() for path in candidates))

    def _quiz_preparation_key(self) -> tuple[str, str] | None:
        if self.current_pair_object is None:
            return None
        return (self.current_run, self.current_pair_object.candidate_a.candidate_id)

    def _quiz_preparation_allowed(self) -> bool:
        return (self.workspace.review_surface == "blind_regions"
                and not self.store.is_read_only and not self.quiz_preparation_closed)

    def _request_quiz_preparation(self) -> None:
        """Run only after the user opens this experiment in blind labeling."""
        if not self._quiz_preparation_allowed() or self.pool or self.quiz_preparation_task:
            return
        key = self._quiz_preparation_key()
        if key is None or key in self.quiz_preparation_requested:
            return
        try:
            campaign = _quiz_generation_campaign(self.current_pair_object.candidate_a)
            if campaign is None:
                return
            self.quiz_preparation_requested.add(key)
            task = QuizPreparationTask(key, campaign)
            task.signals.progress.connect(self._quiz_preparation_progress)
            task.signals.finished.connect(self._quiz_preparation_finished)
            self.quiz_preparation_task = task
            self._quiz_preparation_progress(key, "Preparing model-selected regions")
            self.quiz_preparation_pool.start(task)
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            self.quiz_preparation_requested.add(key)
            self._quiz_preparation_finished(key, {}, f"{type(exc).__name__}: {exc}")

    def _quiz_preparation_progress(self, key, message: str) -> None:
        if self.quiz_preparation_closed or key != self._quiz_preparation_key():
            return
        self.quiz_progress.setText(f"{key[0].upper()}: preparing model-selected regionsâ€¦")
        self.quiz_availability.setText(message)
        self.retry_quiz_preparation.setVisible(False)

    def _retry_quiz_preparation(self) -> None:
        key = self._quiz_preparation_key()
        self.quiz_preparation_errors.pop(key, None)
        self.quiz_preparation_requested.discard(key)
        self._request_quiz_preparation()

    def _quiz_preparation_finished(self, key, payload, error: str) -> None:
        self.quiz_preparation_task = None
        if self.quiz_preparation_closed:
            return
        if error:
            self.quiz_preparation_errors[key] = error
        if key != self._quiz_preparation_key():
            # The completed pool remains with its own package. Start only the
            # experiment the user is looking at now; do not queue visited runs.
            self._request_quiz_preparation()
            return
        if error:
            self.quiz_progress.setText("Could not prepare model-selected regions")
            self.quiz_availability.setText(error)
            self.retry_quiz_preparation.setVisible(True)
            return
        self.quiz_preparation_errors.pop(key, None)
        self.pool = [dict(item, candidate="A") for item in payload.get("proposals", [])]
        count = max(0, min(int(payload.get("initial_proposal_count", len(self.pool))), len(self.pool)))
        candidate = self.current_pair["candidate_a"]
        candidate.update(phase_quiz_proposals=self.pool, phase_quiz_initial_count=count,
                         phase_quiz_ready=bool(self.pool), phase_quiz_available_count=len(self.pool))
        # Merge into live state: manual labels and notes made while this worker
        # ran must not be replaced by a disk reload.
        self.answers, self.annotations, migrated = reconcile_phase_quiz_answers(
            self.pool, self.answers, self.annotations)
        self.selected_ids = clean_selected_ids(self.pool, self.selected_ids, self.answers, count)
        self.quiz_rounds = clean_quiz_rounds(self.quiz_rounds, self.selected_ids)
        self.quiz_index = 0
        self._refresh_quiz()
        self._refresh_annotations()
        self._prefetch_selected_frames()
        self._queue_save()
        self.statusBar().showMessage(
            f"{key[0].upper()}: {len(self.pool)} model-selected regions ready"
            + (f"; preserved {migrated} earlier labels as manual regions" if migrated else ""), 10000)

    def _load_run(self, run: str) -> None:
        if not run:
            return
        self.current_run = run.casefold()
        self.current_pair = self.app_model.pair_payload(self.current_run)
        self.current_pair_object = self.app_model._pair(self.current_run)
        single_model = (
            str(self.current_pair.get("comparison_mode") or "pair")
            == "single_candidate"
        )
        self.candidate_side.setVisible(not single_model)
        if single_model:
            self.candidate_side.setCurrentIndex(
                max(0, self.candidate_side.findData("A"))
            )
        self.current_review_root = self.app_model._review_root(self.current_pair_object)
        raw_available = bool(self.current_pair.get("v5_raw_detail_available"))
        self.raw_detail.setEnabled(raw_available)
        if not raw_available:
            self.raw_detail.blockSignals(True)
            self.raw_detail.setChecked(False)
            self.raw_detail.blockSignals(False)
        self._update_brightness_control()
        decision = deepcopy(self.current_pair.get("decision") or {})
        loaded_newer_draft = False
        drafts: list[dict[str, Any]] = []
        for draft_path in self._draft_candidates(self.current_run):
            if not draft_path.is_file():
                continue
            try:
                draft = json.loads(draft_path.read_text(encoding="utf-8"))
                if timestamp_utc(draft.get("updated_utc")) > timestamp_utc(
                    decision.get("updated_utc")
                ):
                    drafts.append(draft)
            except (OSError, ValueError):
                pass
        for journal_path in (
            self._journal_path(self.current_run),
            self._portable_journal_path(self.current_run),
        ):
            draft = self._latest_journal_request(journal_path)
            if draft is not None and timestamp_utc(draft.get("updated_utc")) > timestamp_utc(
                decision.get("updated_utc")
            ):
                drafts.append(draft)
        for draft in sorted(drafts, key=lambda item: timestamp_utc(item.get("updated_utc"))):
            decision = merge_review_recovery(decision, draft)
            loaded_newer_draft = True
        if not self.reviewer.text().strip() and decision.get("reviewer"):
            self.reviewer.blockSignals(True)
            self.reviewer.setText(str(decision["reviewer"]))
            self.reviewer.blockSignals(False)
            self._remember_reviewer(str(decision["reviewer"]))
        self._update_reviewer_gate()
        candidate = self.current_pair["candidate_a"]
        self.pool = [dict(item, candidate="A") for item in candidate.get("phase_quiz_proposals") or []]
        pending_quiz = not self.pool and self._quiz_preparation_allowed()
        if pending_quiz:
            # An absent pool is not a retired pool. Preserve saved quiz labels
            # verbatim until its requested replacement is available.
            self.answers = deepcopy(decision.get("phase_quiz_answers") or [])
            self.annotations = deepcopy(decision.get("annotations") or [])
            migrated_answers = 0
        else:
            self.answers, self.annotations, migrated_answers = reconcile_phase_quiz_answers(
                self.pool,
                decision.get("phase_quiz_answers") or [],
                decision.get("annotations") or [],
            )
        self.deleted_phase_quiz_answer_keys = {
            str(value) for value in decision.get("deleted_phase_quiz_answer_keys") or []
        }
        self.deleted_annotation_ids = {
            str(value) for value in decision.get("deleted_annotation_ids") or []
        }
        raw_selected_ids = decision.get("phase_quiz_selected_ids")
        initial_count = int(
            candidate.get("phase_quiz_initial_count") or len(self.pool)
        )
        if pending_quiz:
            self.selected_ids = list(raw_selected_ids or [])
        elif not raw_selected_ids and not self.answers:
            initial_batch = review.select_adaptive_phase_quiz_batch(
                self.pool,
                (),
                (),
                initial_count,
            )
            self.selected_ids = [
                str(item.get("proposal_id") or "") for item in initial_batch
            ]
        else:
            self.selected_ids = clean_selected_ids(
                self.pool,
                raw_selected_ids,
                self.answers,
                initial_count,
            )
        self.quiz_rounds = (deepcopy(decision.get("phase_quiz_rounds") or []) if pending_quiz
                           else clean_quiz_rounds(decision.get("phase_quiz_rounds") or [], self.selected_ids))
        self.flags = deepcopy(decision.get("flags") or {"A": [], "B": []})
        self.notes.blockSignals(True)
        self.notes.setPlainText(str(decision.get("notes") or ""))
        self.notes.blockSignals(False)
        choice = str(decision.get("choice") or "unresolved")
        choice_index = self.verdict.findData(choice)
        self.verdict.blockSignals(True)
        self.verdict.setCurrentIndex(max(0, choice_index))
        self.verdict.blockSignals(False)
        proposals = self._selected_proposals()
        has_unanswered = any(
            self._answer_for(proposal) is None for proposal in proposals
        )
        unanswered = next(
            (index for index, proposal in enumerate(proposals) if self._answer_for(proposal) is None),
            0,
        )
        self.quiz_index = unanswered
        self.quiz_tabs.blockSignals(True)
        self.quiz_tabs.setCurrentIndex(0 if has_unanswered else 1)
        self.quiz_tabs.blockSignals(False)
        self.reference.blockSignals(True)
        self.reference.setChecked(False)
        self.reference.blockSignals(False)
        self.reference_mode = False
        self.contours.setChecked(False)
        self.frame_slider.setRange(0, max(0, int(candidate.get("frame_count") or 1) - 1))
        self.fit_next_image = True
        if proposals:
            self.frame_index = int(proposals[self.quiz_index].get("frame_index") or 0)
            self.focus_after_load = True
        else:
            self.frame_index = 0
        self._request_frame()
        self._refresh_quiz()
        self._refresh_annotations()
        self._prefetch_selected_frames()
        if migrated_answers:
            self.statusBar().showMessage(
                f"Preserved {migrated_answers} labels from an earlier quiz pool as manual regions.",
                10000,
            )
        if migrated_answers or loaded_newer_draft:
            QTimer.singleShot(0, self._queue_save)
        QTimer.singleShot(0, self._request_quiz_preparation)

    def _selected_proposals(self) -> list[dict[str, Any]]:
        by_id = {str(item.get("proposal_id") or ""): item for item in self.pool}
        return [by_id[value] for value in self.selected_ids if value in by_id]

    def _current_proposal(self) -> dict[str, Any] | None:
        proposals = self._selected_proposals()
        if not proposals:
            return None
        self.quiz_index %= len(proposals)
        return proposals[self.quiz_index]

    def _answer_for(self, proposal: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if proposal is None:
            return None
        proposal_id = str(proposal.get("proposal_id") or "")
        return next(
            (
                item
                for item in self.answers
                if str(item.get("candidate") or "").upper() == "A"
                and str(item.get("proposal_id") or "") == proposal_id
            ),
            None,
        )

    def _frame_path(self, index: int) -> Path:
        assert self.current_review_root is not None
        return review._native_frame_path(self.current_review_root, self.current_run, index)

    def _reference_path(self) -> Path:
        assert self.current_review_root is not None
        return review._native_reference_path(self.current_review_root, self.current_run)

    def _overlay_path(self, side: str, index: int, layer: str | None = None) -> Path:
        assert self.current_review_root is not None
        candidate = self.current_pair_object.candidate(side)
        path, _rendering = review._preferred_phase_overlay_path(
            self.current_review_root,
            candidate,
            side,
            index,
            overlay_asset_layer(str(layer or self.contour_filter.currentData() or "all")),
        )
        return path

    def _model_phase_in_region(
        self,
        side: str,
        rect: QRectF,
        points: Sequence[QPointF] = (),
    ) -> dict[str, Any]:
        """Summarize the model phase map inside a human-drawn 2D region."""

        image, source = _region_phase_image(self, side, self.frame_index)
        if image.isNull() or image.width() <= 0 or image.height() <= 0:
            return assess_model_phase("unscored", "not_assessable") | {
                "model_phase_distribution": {},
                "model_phase_sample_count": 0,
            }

        phase_image = image
        width, height = phase_image.width(), phase_image.height()
        left = max(0, min(width - 1, int(math.floor(rect.left() * width))))
        right = max(left + 1, min(width, int(math.ceil(rect.right() * width))))
        top = max(0, min(height - 1, int(math.floor(rect.top() * height))))
        bottom = max(top + 1, min(height, int(math.ceil(rect.bottom() * height))))
        sample_step = max(1, int(math.sqrt(max(1, (right - left) * (bottom - top)) / 4096.0)))
        polygon = QPainterPath()
        if len(points) >= 3:
            polygon.moveTo(points[0])
            for point in points[1:]:
                polygon.lineTo(point)
            polygon.closeSubpath()

        palette = {
            "aqueous": review.PHASE_MAP_RGB[0],
            "mixed": review.PHASE_MAP_RGB[1],
            "gas": review.PHASE_MAP_RGB[2],
        }
        counts = {"water": 0, "aqueous": 0, "mixed": 0, "gas": 0}
        for y in range(top, bottom, sample_step):
            normalized_y = (y + 0.5) / height
            for x in range(left, right, sample_step):
                normalized_x = (x + 0.5) / width
                if len(points) >= 3 and not polygon.contains(
                    QPointF(normalized_x, normalized_y)
                ):
                    continue
                color = phase_image.pixelColor(x, y)
                if color.alpha() <= 8 or max(color.red(), color.green(), color.blue()) <= 8:
                    counts["water"] += 1
                    continue
                rgb = (color.red(), color.green(), color.blue())
                phase = min(
                    palette,
                    key=lambda name: sum(
                        (rgb[channel] - palette[name][channel]) ** 2
                        for channel in range(3)
                    ),
                )
                counts[phase] += 1
        total = sum(counts.values())
        if total <= 0:
            model_phase = "unscored"
            fractions: dict[str, float] = {}
        else:
            fractions = {name: value / total for name, value in counts.items()}
            if counts["mixed"] / total >= 0.2 or (
                counts["aqueous"] / total >= 0.15 and counts["gas"] / total >= 0.15
            ):
                model_phase = "mixed"
            else:
                model_phase = max(counts, key=counts.get)
        return {
            "model_phase_class": model_phase,
            "model_phase_distribution": fractions,
            "model_phase_sample_count": total,
            "model_phase_source": str(source),
        }

    def _request_frame(self) -> None:
        if self.current_pair is None:
            return
        frame_count = int(self.current_pair["candidate_a"].get("frame_count") or 1)
        self.frame_index = max(0, min(frame_count - 1, int(self.frame_index)))
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.frame_index)
        self.frame_slider.blockSignals(False)
        elapsed_values = self.current_pair["candidate_a"].get("elapsed_hours") or []
        elapsed = elapsed_values[self.frame_index] if self.frame_index < len(elapsed_values) else None
        self.frame_label.setText(
            f"Frame {self.frame_index + 1} / {frame_count}"
            + (f" · {float(elapsed):.2f} h" if elapsed is not None else "")
        )
        source = self._reference_asset() if self.reference_mode else self._frame_asset(self.frame_index)
        image = self.image_cache.get(source)
        self.base_key = self.image_cache.key(source)
        if image is not None:
            self.displayed_base_key = self.base_key
            self._set_base_image(image)
            self._after_base_image_loaded()
        else:
            self.displayed_base_key = ""
            self.viewer.base_item.setPixmap(QPixmap())
            self.viewer.set_overlay(None)
            self.viewer.clear_vectors()
            self.image_cache.request(source)
            self.asset_label.setText("Loading full-resolution image...")
        self._request_overlay()
        nearby = [self._frame_asset(value) for value in
                  (self.frame_index + 1, self.frame_index + 2, self.frame_index + 3,
                   self.frame_index - 1, self.frame_index - 2)
                  if 0 <= value < frame_count]
        self.image_cache.prefetch(nearby)

    def _phase_prediction_hidden(self) -> bool:
        return self.workspace.review_surface == "blind_regions"

    def _request_overlay(self) -> None:
        if self._phase_prediction_hidden():
            self.overlay_key = ""
            self.viewer.set_overlay(None)
            self._render_vectors()
            return
        self.contour_filter.setEnabled(self.contours.isChecked())
        if self.current_pair is None or self.reference_mode or not self.contours.isChecked():
            self.overlay_key = ""
            self.viewer.set_overlay(None)
            self._render_vectors()
            return
        side = str(self.candidate_side.currentData() or "A")
        source = self._overlay_asset(side, self.frame_index)
        image = self.image_cache.get(source)
        self.overlay_key = self.image_cache.key(source)
        if image is not None and self.displayed_base_key == self.base_key:
            self.viewer.set_overlay(display_phase_overlay(image, str(self.contour_filter.currentData() or "all"), photo_sand=getattr(image, "_photo_sand_domain", None)))
        else:
            self.image_cache.request(source)
        self._render_vectors()
        frame_count = int(self.current_pair["candidate_a"].get("frame_count") or 1)
        self.image_cache.prefetch([
            self._overlay_asset(side, value)
            for value in (self.frame_index + 1, self.frame_index + 2,
                          self.frame_index + 3, self.frame_index - 1, self.frame_index - 2)
            if 0 <= value < frame_count
        ])

    def _image_loaded(self, key: str, image: QImage, error: str) -> None:
        if error:
            if key in {self.base_key, self.overlay_key}:
                self.asset_label.setText("Image load failed")
                self.statusBar().showMessage(error, 8000)
            return
        if key == self.base_key:
            self.displayed_base_key = key
            self._set_base_image(image)
            self._after_base_image_loaded()
            self._request_overlay()
        if key == self.overlay_key and self.displayed_base_key == self.base_key:
            self.viewer.set_overlay(display_phase_overlay(image, str(self.contour_filter.currentData() or "all"), photo_sand=getattr(image, "_photo_sand_domain", None)))

    def _set_base_image(self, image: QImage) -> None:
        self.viewer.set_image(image, fit=self.fit_next_image)
        self.fit_next_image = False
        self.minimap.set_image(image)
        self._update_minimap_viewport()

    def _update_minimap_viewport(self) -> None:
        visible = self.viewer.visible_normalized_rect()
        self.minimap.set_viewport_rect(visible)
        area_fraction = min(1.0, max(0.0, visible.width() * visible.height()))
        if area_fraction >= 0.98:
            self.minimap_caption.setText(
                "Yellow: current view (full frame visible)\nRed: active labeling region"
            )
        else:
            self.minimap_caption.setText(
                f"Yellow: current view ({100.0 * area_fraction:.1f}% of frame)\n"
                "Red: active labeling region · click map to recenter"
            )

    def _after_base_image_loaded(self) -> None:
        self.asset_label.setText("Full resolution · local cache · GPU view")
        self._render_vectors()
        if self.focus_after_load:
            self.focus_after_load = False
            self._focus_current_proposal()

    def _set_frame(self, index: int) -> None:
        if self.current_pair is None:
            return
        frame_count = int(self.current_pair["candidate_a"].get("frame_count") or 1)
        target = max(0, min(frame_count - 1, int(index)))
        if target == self.frame_index and self.base_key:
            return
        self.frame_index = target
        self.fit_next_image = False
        self.focus_after_load = False
        self._request_frame()
        self._refresh_quiz()
        self._refresh_annotations()
        self.image_cache.prefetch(
            [
                self._frame_asset(value)
                for value in (target + 1, target - 1)
                if 0 <= value < frame_count
            ]
        )

    def _reference_changed(self) -> None:
        self.reference_mode = self.reference.isChecked()
        self._request_frame()

    def _reload_overlay_and_vectors(self) -> None:
        self._request_overlay()
        self._refresh_annotations()

    def _proposal_points(self, proposal: Mapping[str, Any]) -> list[QPointF]:
        points = [normalized_point(value) for value in proposal.get("points") or []]
        clean = [value for value in points if value is not None]
        if len(clean) >= 3:
            return clean
        x = float(proposal.get("x") or 0.0)
        y = float(proposal.get("y") or 0.0)
        width = float(proposal.get("width") or 0.0)
        height = float(proposal.get("height") or 0.0)
        return [
            QPointF(x, y),
            QPointF(x + width, y),
            QPointF(x + width, y + height),
            QPointF(x, y + height),
        ]

    def _render_vectors(self) -> None:
        self.viewer.clear_vectors()
        self.minimap.set_focus_rect(None)
        if self.current_pair is None or self.reference_mode:
            return
        proposal = self._current_proposal()
        if self.quiz_index not in self._quiz_view_indices():
            proposal = None
        if proposal is not None:
            source_frame = int(proposal.get("frame_index") or 0)
            on_source_frame = self.frame_index == source_frame
            color = QColor("#ef3e36" if on_source_frame else "#ffd166")
            points = self._proposal_points(proposal)
            self.minimap.set_focus_rect(
                QRectF(
                    min(point.x() for point in points),
                    min(point.y() for point in points),
                    max(point.x() for point in points)
                    - min(point.x() for point in points),
                    max(point.y() for point in points)
                    - min(point.y() for point in points),
                )
            )
            self.viewer.add_polygon(
                points,
                color,
                width=2.7 if on_source_frame else 2.1,
                fill=QColor(239, 62, 54, 20) if on_source_frame else None,
                pen_style=(
                    Qt.PenStyle.SolidLine
                    if on_source_frame
                    else Qt.PenStyle.DashLine
                ),
            )
            top_left = min(points, key=lambda point: (point.y(), point.x()))
            self.viewer.add_label(
                QPointF(top_left.x(), max(0.0, top_left.y() - 0.006)),
                (
                    f"VALIDATION FRAME {source_frame + 1}"
                    if on_source_frame
                    else f"BOX SOURCE: FRAME {source_frame + 1}"
                ),
                color,
            )
        if self.injectors.isChecked():
            for point in self.current_pair.get("injection_points") or []:
                x = float(point.get("x") or 0.0)
                y = float(point.get("y") or 0.0)
                self.viewer.add_rect(QRectF(x - 0.004, y - 0.007, 0.008, 0.014), QColor("#ff3b30"), width=3.0)
        side = str(self.candidate_side.currentData() or "A")
        for annotation in self.annotations:
            if str(annotation.get("candidate") or "").upper() != side:
                continue
            class_filter = str(
                self.annotation_class_filter.currentData() or "all"
            )
            if (
                class_filter != "all"
                and str(annotation.get("category") or "other") != class_filter
            ):
                continue
            if int(annotation.get("frame_index") or 0) != self.frame_index:
                continue
            color = QColor("#00d4ff")
            if "free_gas" in str(annotation.get("category") or ""):
                color = QColor("#ff6b6b")
            points = [
                normalized_point(value)
                for value in annotation.get("points") or []
            ]
            clean = [value for value in points if value is not None]
            if annotation.get("shape") == "freehand" and len(clean) >= 3:
                self.viewer.add_polygon(clean, color, width=2.0)
            else:
                self.viewer.add_rect(
                    QRectF(
                        float(annotation.get("x") or 0.0),
                        float(annotation.get("y") or 0.0),
                        float(annotation.get("width") or 0.0),
                        float(annotation.get("height") or 0.0),
                    ),
                    color,
                    width=2.0,
                )

    def _focus_current_proposal(self) -> None:
        proposal = (
            self._current_proposal()
            if self.quiz_index in self._quiz_view_indices()
            else None
        )
        if proposal is None:
            return
        focus = proposal.get("focus_box") or proposal
        self.viewer.focus_normalized_rect(
            QRectF(
                float(focus.get("x") or 0.0),
                float(focus.get("y") or 0.0),
                max(0.002, float(focus.get("width") or 0.0)),
                max(0.002, float(focus.get("height") or 0.0)),
            )
        )

    def _set_quiz_index(self, index: int) -> None:
        proposals = self._selected_proposals()
        if not proposals:
            return
        self.quiz_index = int(index) % len(proposals)
        proposal = proposals[self.quiz_index]
        self.frame_index = int(proposal.get("frame_index") or 0)
        self.reference.blockSignals(True)
        self.reference.setChecked(False)
        self.reference.blockSignals(False)
        self.reference_mode = False
        self.contours.setChecked(False)
        self.focus_after_load = True
        self._request_frame()
        self._refresh_quiz()

    def _refresh_quiz(self) -> None:
        proposals = self._selected_proposals()
        all_new_indices, all_validated_indices = partition_quiz_region_indices(
            proposals,
            self.answers,
        )
        sort_mode = str(self.quiz_sort.currentData() or "time_asc")
        class_filter = str(self.quiz_class_filter.currentData() or "all")
        new_indices = quiz_region_view_indices(
            proposals,
            self.answers,
            validated=False,
            sort_mode=sort_mode,
        )
        validated_indices = quiz_region_view_indices(
            proposals,
            self.answers,
            validated=True,
            sort_mode=sort_mode,
            class_filter=class_filter,
        )
        answered = len(all_validated_indices)
        reserve = max(0, len(self.pool) - len(proposals))
        self.quiz_progress.setText(
            f"{answered} / {len(proposals)} labeled · {reserve} unused model-selected regions"
            if proposals
            else "No blind-label regions prepared"
        )
        self.new_round.setEnabled(reserve > 0)
        self.new_round_count.setEnabled(reserve > 0)
        if reserve > 0:
            self._update_add_regions_button()
            self.quiz_availability.setText(
                f"{reserve} unused phase/facies-balanced regions remain. You can add more now; existing unlabeled regions do not need to be completed first."
            )
        else:
            self.new_round.setText("All prepared regions added")
            self.quiz_availability.setText(
                "No additional distinct phase/facies/position combinations remain for this experiment."
            )
        self.next_unanswered.setEnabled(bool(new_indices))
        self.previous_region.setEnabled(bool(self._quiz_view_indices()))
        self.next_region.setEnabled(bool(self._quiz_view_indices()))
        active_indices = (
            new_indices if self.quiz_tabs.currentIndex() == 0 else validated_indices
        )
        proposal = (
            self._current_proposal()
            if self.quiz_index in active_indices
            else None
        )
        answer = self._answer_for(proposal)
        if proposal is None:
            self.frame_slider.set_marker(None)
            self.focus_frame_label.setText("No focus-region source frame")
            self.focus_frame_label.setStyleSheet("color:#8d969f;")
        else:
            source_frame = int(proposal.get("frame_index") or 0)
            source_elapsed = float(proposal.get("elapsed_h") or 0.0)
            on_source_frame = self.frame_index == source_frame
            self.frame_slider.set_marker(
                source_frame,
                f"Focus box source: frame {source_frame + 1} at {source_elapsed:.2f} h. Click the red marker to open it.",
            )
            if on_source_frame:
                self.focus_frame_label.setText(
                    f"Focus box source: Frame {source_frame + 1} · {source_elapsed:.2f} h · current frame"
                )
                self.focus_frame_label.setStyleSheet("color:#ef3e36; font-weight:600;")
            else:
                self.focus_frame_label.setText(
                    f"Tracking box from Frame {source_frame + 1} · {source_elapsed:.2f} h · currently viewing Frame {self.frame_index + 1}"
                )
                self.focus_frame_label.setStyleSheet("color:#ffd166; font-weight:600;")
        for key, button in self.phase_buttons.items():
            button.blockSignals(True)
            button.setChecked(bool(answer and answer.get("human_choice") == key))
            button.setEnabled(proposal is not None and int(proposal.get("frame_index") or 0) == self.frame_index)
            button.blockSignals(False)
        self.ignore_region.setEnabled(proposal is not None)
        self.clear_answer.setEnabled(answer is not None)
        if answer is None:
            self.quiz_entry_help.setText(
                "This region is new and has not been labeled. Choose a phase class above to save it."
            )
            self.quiz_entry_help.setStyleSheet("color:#ffd166;")
        else:
            self.quiz_entry_help.setText(
                "This region is already labeled. Choose another phase class above to replace it, or use Delete label to return it to New / unlabeled."
            )
            self.quiz_entry_help.setStyleSheet("color:#9bd4a7;")
        self.confidence.blockSignals(True)
        confidence = str((answer or {}).get("confidence") or "medium")
        self.confidence.setCurrentIndex(max(0, self.confidence.findData(confidence)))
        self.confidence.blockSignals(False)
        self.answer_note.blockSignals(True)
        self.answer_note.setText(str((answer or {}).get("comment") or ""))
        self.answer_note.blockSignals(False)
        self.new_region_list.clear()
        self.validated_list.clear()
        for index in new_indices:
            item = proposals[index]
            elapsed = float(item.get("elapsed_h") or 0.0)
            row = QListWidgetItem(
                f"NEW · Frame {int(item.get('frame_index') or 0) + 1} · {elapsed:.2f} h\nNot yet labeled"
            )
            font = row.font()
            font.setBold(True)
            row.setFont(font)
            row.setForeground(QColor("#ffd166"))
            row.setData(Qt.ItemDataRole.UserRole, index)
            self.new_region_list.addItem(row)
            if index == self.quiz_index:
                self.new_region_list.setCurrentItem(row)
        for index in validated_indices:
            item = proposals[index]
            value = self._answer_for(item) or {}
            elapsed = float(item.get("elapsed_h") or 0.0)
            label = PHASE_LABELS.get(
                str(value.get("human_choice") or ""),
                "Unknown",
            )
            row = QListWidgetItem(
                f"LABELED · Frame {int(item.get('frame_index') or 0) + 1} · {elapsed:.2f} h\n{label}"
            )
            row.setForeground(QColor("#9bd4a7"))
            row.setData(Qt.ItemDataRole.UserRole, index)
            self.validated_list.addItem(row)
            if index == self.quiz_index:
                self.validated_list.setCurrentItem(row)
        self.quiz_tabs.setTabText(
            0,
            f"New / unlabeled ({len(all_new_indices)})",
        )
        validated_count = (
            f"{len(validated_indices)}"
            if len(validated_indices) == len(all_validated_indices)
            else f"{len(validated_indices)} of {len(all_validated_indices)}"
        )
        self.quiz_tabs.setTabText(
            1,
            f"Labeled ({validated_count})",
        )
        self._render_vectors()
        if self.store.is_read_only:
            self._apply_read_only_profile()
        key = self._quiz_preparation_key()
        error = self.quiz_preparation_errors.get(key)
        self.retry_quiz_preparation.setVisible(bool(error) and self._quiz_preparation_allowed())
        if error:
            self.quiz_progress.setText("Could not prepare model-selected regions")
            self.quiz_availability.setText(error)
        elif self.quiz_preparation_task is not None and self.quiz_preparation_task.key == key:
            self._quiz_preparation_progress(key, "You can inspect the image and add manual labels while regions are prepared.")

    def _quiz_item_opened(self, item: QListWidgetItem) -> None:
        self._set_quiz_index(int(item.data(Qt.ItemDataRole.UserRole) or 0))

    def _quiz_view_indices(self) -> list[int]:
        proposals = self._selected_proposals()
        return quiz_region_view_indices(
            proposals,
            self.answers,
            validated=self.quiz_tabs.currentIndex() == 1,
            sort_mode=str(self.quiz_sort.currentData() or "time_asc"),
            class_filter=str(self.quiz_class_filter.currentData() or "all"),
        )

    def _quiz_view_controls_changed(self, _index: int = -1) -> None:
        indices = self._quiz_view_indices()
        if indices and self.quiz_index not in indices:
            self._set_quiz_index(indices[0])
        else:
            self._refresh_quiz()

    def _quiz_tab_changed(self, _index: int) -> None:
        indices = self._quiz_view_indices()
        if indices and self.quiz_index not in indices:
            self._set_quiz_index(indices[0])
        else:
            self._refresh_quiz()

    def _move_in_quiz_view(self, offset: int) -> None:
        indices = self._quiz_view_indices()
        if not indices:
            return
        if self.quiz_index in indices:
            position = indices.index(self.quiz_index)
            target = indices[(position + int(offset)) % len(indices)]
        else:
            target = indices[0 if offset >= 0 else -1]
        self._set_quiz_index(target)

    def _open_next_new_region(self) -> None:
        self.quiz_tabs.blockSignals(True)
        self.quiz_tabs.setCurrentIndex(0)
        self.quiz_tabs.blockSignals(False)
        self._next_unanswered()

    def _next_unanswered(self) -> None:
        proposals = self._selected_proposals()
        if not proposals:
            return
        new_indices = quiz_region_view_indices(
            proposals,
            self.answers,
            validated=False,
            sort_mode=str(self.quiz_sort.currentData() or "time_asc"),
        )
        if not new_indices:
            self.quiz_tabs.setCurrentIndex(1)
            self._refresh_quiz()
            return
        self.quiz_tabs.blockSignals(True)
        self.quiz_tabs.setCurrentIndex(0)
        self.quiz_tabs.blockSignals(False)
        if self.quiz_index in new_indices:
            position = new_indices.index(self.quiz_index)
            target = new_indices[(position + 1) % len(new_indices)]
        else:
            target = new_indices[0]
        self._set_quiz_index(target)

    def _answer(self, human_choice: str) -> None:
        proposal = self._current_proposal()
        if proposal is None or int(proposal.get("frame_index") or 0) != self.frame_index:
            return
        proposal_id = str(proposal.get("proposal_id") or "")
        self.deleted_phase_quiz_answer_keys.discard(f"A:{proposal_id}")
        self.answers = [
            item
            for item in self.answers
            if not (
                str(item.get("candidate") or "").upper() == "A"
                and str(item.get("proposal_id") or "") == proposal_id
            )
        ]
        self.answers.append(
            {
                "proposal_id": proposal_id,
                "candidate": "A",
                "candidate_id": str(proposal.get("candidate_id") or ""),
                "params_sha256": str(proposal.get("params_sha256") or ""),
                "frame_index": int(proposal.get("frame_index") or 0),
                "elapsed_h": float(proposal.get("elapsed_h") or 0.0),
                "suggested_phase": str(proposal.get("suggested_phase") or ""),
                "suggestion_basis": str(proposal.get("suggestion_basis") or ""),
                "coordinate_space": str(proposal.get("coordinate_space") or ""),
                "shape": str(proposal.get("shape") or "freehand"),
                "points": deepcopy(proposal.get("points") or []),
                "x": float(proposal.get("x") or 0.0),
                "y": float(proposal.get("y") or 0.0),
                "width": float(proposal.get("width") or 0.0),
                "height": float(proposal.get("height") or 0.0),
                "analysis_polygon_px": deepcopy(
                    proposal.get("analysis_polygon_px") or []
                ),
                "analysis_bbox_xywh": deepcopy(
                    proposal.get("analysis_bbox_xywh") or []
                ),
                "focus_box": deepcopy(proposal.get("focus_box") or {}),
                "model_stats": deepcopy(proposal.get("model_stats") or {}),
                **visible_phase_payload(human_choice),
                **assess_model_phase(
                    str(proposal.get("suggested_phase") or "unscored"),
                    human_choice,
                ),
                "confidence": "low" if human_choice == "ignore_region" else str(self.confidence.currentData() or "medium"),
                "comment": self.answer_note.text().strip(),
                "answered_utc": utc_now(),
            }
        )
        self._refresh_quiz()
        self._queue_save()
        if self.auto_advance.isChecked():
            QTimer.singleShot(30, self._next_unanswered)

    def _clear_answer(self) -> None:
        proposal = self._current_proposal()
        if proposal is None:
            return
        proposal_id = str(proposal.get("proposal_id") or "")
        self.deleted_phase_quiz_answer_keys.add(f"A:{proposal_id}")
        self.answers = [
            item
            for item in self.answers
            if not (
                str(item.get("candidate") or "").upper() == "A"
                and str(item.get("proposal_id") or "") == proposal_id
            )
        ]
        self.quiz_tabs.blockSignals(True)
        self.quiz_tabs.setCurrentIndex(0)
        self.quiz_tabs.blockSignals(False)
        self._refresh_quiz()
        self._queue_save()

    def _update_existing_answer(self) -> None:
        answer = self._answer_for(self._current_proposal())
        if answer is None:
            return
        answer["confidence"] = str(self.confidence.currentData() or "medium")
        answer["comment"] = self.answer_note.text().strip()
        answer["answered_utc"] = utc_now()
        self._refresh_quiz()
        self._queue_save()

    def _update_add_regions_button(self) -> None:
        if self.new_round.isEnabled():
            self.new_round.setText(
                f"Add {self.new_round_count.value()} to New / unlabeled"
            )

    def _start_new_round(self) -> None:
        if not self.pool:
            self.statusBar().showMessage(
                "No prepared model-selected regions are available for this experiment.",
                5000,
            )
            return
        requested = int(self.new_round_count.value())
        batch = review.select_adaptive_phase_quiz_batch(
            self.pool,
            self.selected_ids,
            self.answers,
            requested,
        )
        if not batch:
            self.statusBar().showMessage(
                "No additional unused model-selected regions are available.",
                5000,
            )
            return
        first_index = len(self.selected_ids)
        ids = [str(item.get("proposal_id") or "") for item in batch]
        self.selected_ids.extend(ids)
        self.quiz_rounds.append(
            {
                "round": len(self.quiz_rounds) + 1,
                "selection_mode": (
                    "active_learning_entropy_phase_facies_spatial_balance"
                ),
                "proposal_ids": ids,
                "created_utc": utc_now(),
            }
        )
        self.quiz_index = first_index
        self.quiz_tabs.setCurrentIndex(0)
        self.statusBar().showMessage(f"Preparing {len(batch)} regions...", 5000)
        self.image_cache.prefetch(
            [self._frame_asset(int(item.get("frame_index") or 0)) for item in batch]
        )
        self._queue_save()
        self._set_quiz_index(first_index)
        self.statusBar().showMessage(
            (
                f"Added all {len(batch)} available regions to New / unlabeled "
                f"(requested {requested})."
                if len(batch) < requested
                else f"Added {len(batch)} regions to New / unlabeled."
            ),
            5000,
        )

    def _prefetch_selected_frames(self) -> None:
        proposals = self._selected_proposals()
        self.image_cache.prefetch(
            [self._frame_asset(int(item.get("frame_index") or 0)) for item in proposals]
        )

    def _annotation_payload(self, shape: str, coordinates: QRectF, points: Sequence[QPointF]) -> dict[str, Any]:
        category = str(self.annotation_category.currentData() or "other")
        side = str(self.candidate_side.currentData() or "A")
        elapsed_values = self.current_pair["candidate_a"].get("elapsed_hours") or []
        elapsed_h = (
            float(elapsed_values[self.frame_index])
            if self.frame_index < len(elapsed_values)
            else float(self.frame_index)
        )
        return {
            "id": uuid.uuid4().hex,
            "candidate": side,
            "candidate_id": self.current_pair["reveal"][side]["candidate_id"],
            "params_sha256": self.current_pair["reveal"][side]["params_sha256"],
            "frame_index": self.frame_index,
            "elapsed_h": elapsed_h,
            "category": category,
            "phase": ANNOTATION_PHASES.get(category, "unspecified"),
            "severity": "major",
            "confidence": "high",
            "contours_visible": self.contours.isChecked(),
            "coordinate_space": "normalized_native_review_canvas_16x9",
            "shape": shape,
            "points": [[point.x(), point.y()] for point in points],
            "x": coordinates.x(),
            "y": coordinates.y(),
            "width": coordinates.width(),
            "height": coordinates.height(),
            "comment": "",
            **annotation_phase_payload(category),
            **review.annotation_category_semantics(category),
        }

    def _add_rectangle_annotation(self, rect: QRectF) -> None:
        if self.current_pair is None or self.reference_mode:
            return
        self.annotations.append(self._annotation_payload("rectangle", rect, []))
        self._ensure_new_annotation_visible(self.annotations[-1])
        self._refresh_annotations(selected_index=len(self.annotations) - 1)
        self._queue_save()

    def _add_freehand_annotation(self, points: Sequence[QPointF]) -> None:
        if self.current_pair is None or self.reference_mode or len(points) < 3:
            return
        xs = [point.x() for point in points]
        ys = [point.y() for point in points]
        rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        self.annotations.append(self._annotation_payload("freehand", rect, points[:1000]))
        self._ensure_new_annotation_visible(self.annotations[-1])
        self._refresh_annotations(selected_index=len(self.annotations) - 1)
        self._queue_save()

    def _ensure_new_annotation_visible(self, annotation: Mapping[str, Any]) -> None:
        class_filter = str(self.annotation_class_filter.currentData() or "all")
        category = str(annotation.get("category") or "other")
        if class_filter in {"all", category}:
            return
        self.annotation_class_filter.blockSignals(True)
        self.annotation_class_filter.setCurrentIndex(0)
        self.annotation_class_filter.blockSignals(False)

    def _annotation_elapsed_h(self, annotation: Mapping[str, Any]) -> float:
        raw_elapsed = annotation.get("elapsed_h")
        if raw_elapsed is not None:
            return float(raw_elapsed)
        frame = int(annotation.get("frame_index") or 0)
        if self.current_pair is not None:
            values = self.current_pair["candidate_a"].get("elapsed_hours") or []
            if frame < len(values):
                return float(values[frame])
        return float(frame)

    def _annotation_view_indices(self) -> list[int]:
        return annotation_view_indices(
            self.annotations,
            sort_mode=str(self.annotation_sort.currentData() or "time_asc"),
            class_filter=str(
                self.annotation_class_filter.currentData() or "all"
            ),
        )

    def _selected_annotation_index(self) -> int | None:
        item = self.annotation_list.currentItem()
        if item is None:
            return None
        index = int(item.data(Qt.ItemDataRole.UserRole) or 0)
        return index if 0 <= index < len(self.annotations) else None

    def _annotation_view_controls_changed(self, _index: int = -1) -> None:
        self._refresh_annotations(selected_index=self._selected_annotation_index())

    def _refresh_annotations(self, *, selected_index: int | None = None) -> None:
        if selected_index is None:
            selected_index = self._selected_annotation_index()
        visible_indices = self._annotation_view_indices()
        self.annotation_list.blockSignals(True)
        self.annotation_list.clear()
        selected_item: QListWidgetItem | None = None
        for index in visible_indices:
            item = self.annotations[index]
            label = ANNOTATION_LABELS.get(str(item.get("category") or ""), "Other")
            row = QListWidgetItem(
                f"{str(item.get('candidate') or 'A')} · Frame {int(item.get('frame_index') or 0) + 1} · {self._annotation_elapsed_h(item):.2f} h\n{label}"
            )
            row.setData(Qt.ItemDataRole.UserRole, index)
            self.annotation_list.addItem(row)
            if index == selected_index:
                selected_item = row
        if selected_item is not None:
            self.annotation_list.setCurrentItem(selected_item)
        self.annotation_list.blockSignals(False)
        self.annotation_filter_summary.setText(
            f"Showing {len(visible_indices)} of {len(self.annotations)} manual regions"
        )
        self._annotation_selection_changed(self.annotation_list.currentItem(), None)
        self._render_vectors()

    def _annotation_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        has_selection = current is not None
        self.update_annotation_class.setEnabled(
            has_selection and not self.store.is_read_only
        )
        self.delete_annotation_button.setEnabled(
            has_selection and not self.store.is_read_only
        )
        if current is None:
            return
        index = int(current.data(Qt.ItemDataRole.UserRole) or 0)
        if not 0 <= index < len(self.annotations):
            return
        category = str(self.annotations[index].get("category") or "other")
        category_index = self.annotation_category.findData(category)
        if category_index >= 0:
            self.annotation_category.blockSignals(True)
            self.annotation_category.setCurrentIndex(category_index)
            self.annotation_category.blockSignals(False)

    def _annotation_item_opened(self, item: QListWidgetItem) -> None:
        index = int(item.data(Qt.ItemDataRole.UserRole) or 0)
        if not 0 <= index < len(self.annotations):
            return
        annotation = self.annotations[index]
        side_index = self.candidate_side.findData(str(annotation.get("candidate") or "A"))
        self.candidate_side.setCurrentIndex(max(0, side_index))
        self._set_frame(int(annotation.get("frame_index") or 0))
        self.viewer.focus_normalized_rect(
            QRectF(
                float(annotation.get("x") or 0.0),
                float(annotation.get("y") or 0.0),
                float(annotation.get("width") or 0.01),
                float(annotation.get("height") or 0.01),
            )
        )

    def _update_selected_annotation_class(self) -> None:
        index = self._selected_annotation_index()
        if index is None:
            return
        category = str(self.annotation_category.currentData() or "other")
        self.annotations[index]["category"] = category
        self.annotations[index]["phase"] = ANNOTATION_PHASES.get(
            category,
            "unspecified",
        )
        for key in (
            "human_choice",
            "phase_label_schema",
            "phase_class",
            "phase_dominance",
            "phase_components",
            "training_eligible",
            "interpretation_scope",
        ):
            self.annotations[index].pop(key, None)
        self.annotations[index].update(annotation_phase_payload(category))
        for key in review.ANNOTATION_SEMANTIC_FIELDS:
            self.annotations[index].pop(key, None)
        self.annotations[index].update(
            review.annotation_category_semantics(category)
        )
        self._refresh_annotations(selected_index=index)
        self._queue_save()
        self.statusBar().showMessage(
            f"Updated annotation class to {ANNOTATION_LABELS.get(category, 'Other')}",
            5000,
        )

    def _delete_annotation(self) -> None:
        index = self._selected_annotation_index()
        if index is None:
            return
        visible_before = self._annotation_view_indices()
        position = visible_before.index(index) if index in visible_before else 0
        annotation_id = str(self.annotations[index].get("id") or "").strip()
        if annotation_id:
            self.deleted_annotation_ids.add(annotation_id)
        del self.annotations[index]
        visible_after = self._annotation_view_indices()
        next_index = (
            visible_after[min(position, len(visible_after) - 1)]
            if visible_after
            else None
        )
        self._refresh_annotations(selected_index=next_index)
        self._queue_save()
        self.statusBar().showMessage("Deleted selected manual annotation", 5000)

    def _save_request(self) -> dict[str, Any]:
        if not self.current_run:
            return {}
        reviewer_name = self.reviewer.text().strip()
        reviewer_id = reviewer_identifier(reviewer_name)
        annotations = deepcopy(self.annotations)
        answers = deepcopy(self.answers)
        for item in (*annotations, *answers):
            item["reviewer"] = reviewer_name
            item["reviewer_id"] = reviewer_id
        return {
            "run": self.current_run,
            "reviewer": reviewer_name,
            "reviewer_id": reviewer_id,
            "choice": str(self.verdict.currentData() or "unresolved"),
            "flags": deepcopy(self.flags),
            "annotations": annotations,
            "phase_quiz_answers": answers,
            "phase_quiz_selected_ids": list(self.selected_ids),
            "phase_quiz_rounds": deepcopy(self.quiz_rounds),
            "phase_quiz_per_candidate_limit": len(self.selected_ids),
            "deleted_phase_quiz_answer_keys": sorted(
                self.deleted_phase_quiz_answer_keys
            ),
            "deleted_annotation_ids": sorted(self.deleted_annotation_ids),
            "notes": self.notes.toPlainText().strip(),
            "updated_utc": utc_now(),
        }

    def _queue_save(self) -> None:
        if self.store.is_read_only or not self.current_run:
            return
        request = self._save_request()
        try:
            draft_path = self._write_recovery_draft(request)
        except Exception as exc:  # noqa: BLE001 - visible persistence boundary
            run = str(request.get("run") or self.current_run).casefold()
            self.pending_save_requests[run] = deepcopy(request)
            self.pending_save_requests.move_to_end(run)
            self.save_dirty = True
            self.sync_error_pending = True
            emergency_path = self._draft_path(run, str(request.get("reviewer") or ""))
            self._set_save_status(
                "error",
                "Local save needs attention · emergency copy retained",
                f"{type(exc).__name__}: {exc}. Emergency recovery: {emergency_path}",
            )
            self.statusBar().showMessage(
                f"Local autosave failed: {type(exc).__name__}: {exc}",
                15000,
            )
            self.save_timer.start(5000)
            return
        if not request.get("reviewer"):
            self._set_save_status(
                "blocked",
                "Recovery draft saved · enter reviewer name",
                f"A recovery copy exists at {draft_path}, but final review choices remain locked until a reviewer name is entered.",
            )
            return
        run = str(request["run"]).casefold()
        self.pending_save_requests[run] = deepcopy(request)
        self.pending_save_requests.move_to_end(run)
        self.save_dirty = True
        self._set_save_status(
            "pending",
            "Saved locally · canonical update queued",
            f"The latest edit and revision history are safe in {draft_path}. The canonical review files will update automatically.",
        )
        if self.sync_error_pending and not self.workspace.offline:
            if not self.save_timer.isActive():
                self.save_timer.start(self.sync_retry_ms)
        else:
            self.save_timer.start(650)

    def _save_now(self) -> None:
        if not self.reviewer.text().strip():
            self._update_reviewer_gate()
            return
        self._queue_save()
        self.save_timer.stop()
        self._start_background_save()

    def _start_background_save(self) -> None:
        if self.save_inflight:
            return
        if not self.pending_save_requests:
            if not self.current_run:
                return
            request = self._save_request()
            self._write_recovery_draft(request)
            if request.get("reviewer"):
                self.pending_save_requests[self.current_run] = deepcopy(request)
        if not self.pending_save_requests:
            self.save_dirty = False
            self._update_reviewer_gate()
            return
        run, request = self.pending_save_requests.popitem(last=False)
        if not request.get("reviewer"):
            self.save_dirty = bool(self.pending_save_requests)
            self._update_reviewer_gate()
            return
        try:
            self._write_recovery_draft(request)
        except Exception as exc:  # noqa: BLE001 - visible persistence boundary
            self.pending_save_requests[run] = request
            self.pending_save_requests.move_to_end(run, last=False)
            self.save_dirty = True
            self.sync_error_pending = True
            self._set_save_status(
                "error",
                "Local recovery retrying",
                f"{type(exc).__name__}: {exc}",
            )
            self.save_timer.start(5000)
            return
        self.save_generation += 1
        generation = self.save_generation
        task = SaveTask(generation, run, self.store, request)
        self.save_tasks[generation] = task
        self.save_task_requests[generation] = deepcopy(request)
        task.signals.finished.connect(self._save_finished)
        self.save_inflight = True
        self.save_dirty = bool(self.pending_save_requests)
        self._set_save_status(
            "saving",
            "Saving review..." if self.workspace.offline else "Syncing review to NAS...",
            f"Writing the durable review record for {run.upper()}. A recovery draft already exists.",
        )
        self.save_pool.start(task)

    def _save_finished(self, generation: int, run: str, _record: Mapping[str, Any], error: str) -> None:
        self._refresh_blind_reveal_notice()
        self.save_tasks.pop(generation, None)
        failed_request = self.save_task_requests.pop(generation, None)
        self.save_inflight = False
        if error:
            if failed_request is not None and run not in self.pending_save_requests:
                self.pending_save_requests[run] = failed_request
            try:
                failure_dir = self.cache_root / "failed_saves" / datetime.now().strftime("%Y%m%d")
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_stamp = datetime.now().strftime("%H%M%S_%f")
                atomic_json(
                    failure_dir / f"{failure_stamp}_{run}.json",
                    {
                        "schema": 1,
                        "run": run,
                        "generation": generation,
                        "failed_utc": utc_now(),
                        "error": error,
                        "request": failed_request or {},
                        "target": str(self.store.json_path),
                    },
                )
                with (self.cache_root / "save_errors.log").open(
                    "a", encoding="utf-8", newline=""
                ) as stream:
                    stream.write(
                        f"[{utc_now()}] run={run} generation={generation} "
                        f"target={self.store.json_path}\n{error}\n\n"
                    )
            except OSError:
                pass
            summary = error.splitlines()[0] if error else "Unknown save error"
            self._set_save_status(
                "error",
                "Recovery saved · local canonical retrying",
                f"{summary}. Local recovery copies and revision history are intact; the canonical local file will retry.",
            )
            self.statusBar().showMessage(summary, 10000)
            self.save_dirty = bool(self.pending_save_requests)
            self.sync_error_pending = True
            self.save_timer.start(self.sync_retry_ms)
            self.sync_retry_ms = min(
                self.sync_retry_max_ms,
                self.sync_retry_ms * 2,
            )
            return
        self.saved_generation = max(self.saved_generation, generation)
        self.sync_error_pending = False
        self.sync_retry_ms = 30_000
        self.save_dirty = bool(self.pending_save_requests)
        if self.pending_save_requests:
            self._set_save_status(
                "pending",
                f"Saved {run.upper()} · more changes queued",
                "One durable record was saved; newer edits are already protected by recovery drafts and will be saved next.",
            )
            self.save_timer.start(50)
        else:
            self._set_save_status(
                "saved",
                "All changes saved locally"
                if self.workspace.offline
                else "All changes saved · NAS synced",
                "Green means both the immediate recovery draft and the durable review record are up to date.",
            )
        self._populate_runs()

    def closeEvent(self, event: Any) -> None:
        self.quiz_preparation_closed = True
        if self.quiz_preparation_task is not None:
            self.quiz_preparation_task.cancelled.set()
        if self.current_run and not self.store.is_read_only:
            request = self._save_request()
            self._write_recovery_draft(request)
            if self.workspace.offline and request.get("reviewer"):
                try:
                    self.store.save(request)
                except Exception as exc:  # noqa: BLE001 - recovery draft remains authoritative
                    self.statusBar().showMessage(
                        f"Recovery draft saved; final close save failed: {exc}",
                        10000,
                    )
        super().closeEvent(event)


class ReviewHistorySignals(QObject):
    loaded = Signal(object, str)


class ReviewHistoryTask(_SavedReaderContextTask):
    """Read history off the UI thread; never write to the review store."""

    def __init__(self, **inputs: Any) -> None:
        super().__init__()
        self.inputs = inputs
        self.signals = ReviewHistorySignals()

    def _run_saved_read(self) -> None:
        try:
            from darsia.fluidflower.tools.phase_review_history import load_review_history
            result = load_review_history(**self.inputs)
        except Exception as exc:  # noqa: BLE001 - background UI boundary
            self.signals.loaded.emit({}, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.loaded.emit(result, "")


class ReviewHistoryRetractionTask(QRunnable):
    """Append a withdrawal only after the user clicks its explicit action."""

    def __init__(self, entry: Mapping[str, Any], **inputs: Any) -> None:
        super().__init__()
        self.entry, self.inputs = deepcopy(dict(entry)), inputs
        self.signals = ReviewHistorySignals()

    def run(self) -> None:
        try:
            from darsia.fluidflower.phase_review_retractions import retract_history_entry
            event = retract_history_entry(self.entry, **self.inputs)
        except Exception as exc:  # noqa: BLE001 - background write boundary
            self.signals.loaded.emit({}, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.loaded.emit(event, "")


class ReviewHistoryReclassificationTask(QRunnable):
    """Publish the old withdrawal and replacement annotation as one event."""

    def __init__(self, entry: Mapping[str, Any], **inputs: Any) -> None:
        super().__init__()
        self.entry, self.inputs = deepcopy(dict(entry)), deepcopy(inputs)
        self.signals = ReviewHistorySignals()

    def run(self) -> None:
        try:
            from darsia.fluidflower.phase_review_retractions import reclassify_history_entry
            event = reclassify_history_entry(self.entry, **self.inputs)
        except Exception as exc:  # noqa: BLE001 - background write boundary
            self.signals.loaded.emit({}, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.loaded.emit(event, "")


class ComparisonSaveTask(QRunnable):
    """Keep portable recovery and canonical NAS writes off the Qt thread."""

    def __init__(self, generation, run, store, request, recovery_dir):
        super().__init__()
        self.generation, self.run_name = generation, run
        self.store, self.request = store, deepcopy(request)
        self.recovery_dir = Path(recovery_dir)
        self.signals = SaveSignals()

    def run(self):
        read_only = False
        try:
            read_only = bool(self.store.is_read_only)
            if read_only:
                raise RuntimeError("This review session is locked as training evidence")
            atomic_json(self.recovery_dir / f"{self.run_name}.json", self.request)
            append_bounded_snapshot_journal(
                self.recovery_dir / f"{self.run_name}.history.jsonl",
                {"recorded_utc": utc_now(), "request": self.request},
            )
            record = self.store.save(self.request)
        except Exception as exc:  # noqa: BLE001 - recoverable asynchronous boundary
            self.signals.finished.emit(self.generation, self.run_name,
                {"read_only": read_only}, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(self.generation, self.run_name,
                {"record": record, "read_only": read_only}, "")


class ComparisonRefreshTask(QRunnable):
    """Read run summaries and cross-session box counts only on explicit refresh."""

    def __init__(self, generation, app_model, store, count_boxes):
        super().__init__()
        self.generation, self.app_model, self.store = generation, app_model, store
        self.count_boxes = count_boxes
        self.signals = SaveSignals()

    def run(self):
        try:
            decisions = self.store.decisions()
            rows = self.app_model.state().get("runs") or []
            areas = {}
            for row in rows:
                run = str(row.get("run") or "").casefold()
                if not run:
                    continue
                try:
                    pair = self.app_model._pair(run)
                except (KeyError, AttributeError, ValueError):
                    # A watched inventory can change between state() and this
                    # read; area is optional and must not hide review runs.
                    continue
                candidate_a, candidate_b = pair.candidate("A"), pair.candidate("B")
                area_a = peak_phase_area_percent(candidate_a)
                areas[run] = {"A": area_a,
                    "B": area_a if candidate_b is candidate_a
                    else peak_phase_area_percent(candidate_b)}
            payload = dict(decisions=decisions, rows=rows,
                areas=areas,
                read_only=bool(self.store.is_read_only),
                elsewhere=self.count_boxes() if rows else {})
        except Exception as exc:  # noqa: BLE001 - background read boundary
            self.signals.finished.emit(self.generation, "", {}, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(self.generation, "", payload, "")


class ComparisonRunLoadTask(_SavedReaderContextTask):
    """Read and verify a selected run entirely outside the Qt event loop."""

    def __init__(self, generation, run, app_model, store, reviewer, recovery_dir,
                 acknowledged_generation=0):
        super().__init__()
        self.generation, self.run_name = generation, run
        self.app_model, self.store, self.reviewer = app_model, store, reviewer
        self.recovery_dir = Path(recovery_dir)
        self.acknowledged_generation = acknowledged_generation
        self.cancelled = threading.Event()
        self.signals = SaveSignals()

    def _run_saved_read(self):
        try:
            if self.cancelled.is_set():
                self.signals.finished.emit(self.generation, self.run_name, {}, "")
                return
            pair = self.app_model.pair_payload(self.run_name, self.reviewer)
            if self.cancelled.is_set():
                self.signals.finished.emit(self.generation, self.run_name, {}, "")
                return
            pair_object = self.app_model._pair(self.run_name)
            review_root = self.app_model._review_root(pair_object)
            candidate_assets = {}
            for side in ("A", "B"):
                candidate = pair_object.candidate(side)
                same = next((candidate_assets[other] for other in candidate_assets
                             if pair_object.candidate(other) is candidate), None)
                if same is not None:
                    candidate_assets[side] = same
                    continue
                candidate_assets[side] = dict(
                    prediction_withheld=review._candidate_prediction_withheld(review_root, candidate),
                    fingerprint=review._candidate_fingerprint(candidate),
                    map_root=review._candidate_phase_map_root(review_root, candidate),
                    review_root=review._candidate_review_root(review_root, candidate))
            rows = pair.get("frame_sources") or []
            from .phase_studio_display_cache import binding_token
            def photo_row(row):
                dependencies = []
                def paths(value):
                    if isinstance(value, dict):
                        for name, item in value.items():
                            if isinstance(item, str) and (name == "path" or name.endswith("_path")):
                                dependencies.append(item)
                            elif isinstance(item, (dict, list)):
                                paths(item)
                    elif isinstance(value, list):
                        for item in value:
                            paths(item)
                paths(row)
                return dict(binding=binding_token(row), dependencies=tuple(dict.fromkeys(dependencies)))
            photo_rows = {index: photo_row(row) for index, row in enumerate(rows)}
            photo_rows["reference"] = photo_row(pair.get("reference") or pair.get("original_reference_source") or {})
            reference_binding = binding_token({name: pair.get(name) for name in ("reference", "original_reference_source")})
            modes = ("natural",)
            if display_modes_available({"frame_sources": rows}):
                from darsia.fluidflower.phase_rig_raw_display import rig_raw_display_available
                modes = (DISPLAY_MODES if rig_raw_display_available(pair, frame_index=0)
                         or change_display_available(review_root, self.run_name, frame_count=len(rows))
                         else BASE_DISPLAY_MODES)
            annotation_binding=None
            if "ffac_selected_policy_a_sources" in pair:
                from darsia.fluidflower.phase_annotation_binding import load_annotation_binding
                annotation_binding=load_annotation_binding(review_root,self.run_name)
            decision = self.store.decisions().get(self.run_name) or {}
            try:
                recovery = json.loads((self.recovery_dir / f"{self.run_name}.json").read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                recovery = None
            payload = dict(pair=pair, pair_object=pair_object,
                review_root=review_root, candidate_assets=candidate_assets, display_modes=modes,
                photo_rows=photo_rows, reference_binding=reference_binding,annotation_binding=annotation_binding,
                decision=decision, recovery=recovery, read_only=bool(self.store.is_read_only),
                acknowledged_generation=self.acknowledged_generation)
        except Exception as exc:  # noqa: BLE001 - background navigation boundary
            self.signals.finished.emit(self.generation, self.run_name, {}, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(self.generation, self.run_name, payload, "")


class BlindModelComparisonWindow(ReviewImageAssets, QMainWindow):
    """Purpose-built synchronized A/B viewer for held-out model comparisons."""

    PHASE_CLASSIFICATION_LABELS = tuple(
        (PHASE_CHOICE_TO_ANNOTATION[choice.key], choice.label)
        for choice in VISIBLE_PHASE_CHOICES
    )
    PHYSICAL_ISSUE_LABELS = (
        ("confirmed_no_free_gas_in_selected_area", "No free gas anywhere in selected area"),
        ("confirmed_no_aqueous_co2_in_selected_area", "No aqueous CO2 anywhere in selected area"),
        ("missed_visible_plume", "Missed visible plume"),
        ("missed_visible_gas", "Missed visible free CO2"),
        ("missed_visible_aqueous_co2", "Missed visible aqueous CO2"),
        (
            "unsupported_free_gas_classification",
            "Incorrect free-gas prediction in this model",
        ),
        (
            "unsupported_aqueous_co2_classification",
            "Incorrect aqueous-CO2 prediction in this model",
        ),
        ("false_seal_detection", "False seal detection"),
        (
            "incomplete_wash_indicator_carryover",
            "Indicator carryover from incomplete washing",
        ),
        (
            "straight_facies_edges",
            "Unsupported phase boundary following a facies edge",
        ),
        ("isolated_noise", "Isolated noise"),
        ("implausible_phase_split", "Implausible phase split"),
        ("late_mass_drift", "Late mass drift"),
        ("poor_mass_fit", "Poor mass fit"),
    )
    FLAG_LABELS = PHASE_CLASSIFICATION_LABELS + PHYSICAL_ISSUE_LABELS
    SHARED_ARTIFACT_LABELS = (
        (
            "incomplete_wash_indicator_carryover",
            "Indicator carryover from incomplete washing",
        ),
    )
    ISSUE_CATEGORY_LABELS = FLAG_LABELS
    ISSUE_COLORS = {
        "confirmed_no_co2": "#8ecae6",
        "confirmed_no_free_gas_in_selected_area": "#72bed8",
        "confirmed_no_aqueous_co2_in_selected_area": "#72d6c9",
        "confirmed_only_aqueous_co2": "#31d9a0",
        "confirmed_predominantly_aqueous_co2": "#5ee1b4",
        "confirmed_mixed_free_aqueous_co2": "#ffbf47",
        "confirmed_predominantly_free_gas": "#f27aa5",
        "confirmed_only_free_gas": "#f24c9d",
        "confirmed_mixed_aqueous_water": "#72d6c9",
        "confirmed_mixed_free_aqueous_water": "#e4a853",
        "phase_ambiguous_strong_color_response": "#ffd166",
        "phase_not_assessable": "#adb5bd",
        "phase_signal_artifact": "#b08cff",
        "missed_visible_plume": "#43d9ad",
        "missed_visible_gas": "#f7f7f2",
        "missed_visible_aqueous_co2": "#55e0a3",
        "unsupported_free_gas_classification": "#ff668f",
        "unsupported_aqueous_co2_classification": "#53a9ff",
        "false_seal_detection": "#ff5c5c",
        "straight_facies_edges": "#ffb347",
        "isolated_noise": "#d47cff",
        "implausible_phase_split": "#ff79b0",
        "late_mass_drift": "#ffd166",
        "poor_mass_fit": "#8fc8ff",
        "incomplete_wash_indicator_carryover": "#ffb000",
    }

    def __init__(
        self,
        app_model: review.ReviewApp,
        store: review.ReviewStore,
        cache_root: Path,
        workspace: ReviewWorkspace,
        *,
        reviewer_name: str,
        use_opengl: bool,
        on_back: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        from darsia.fluidflower.ui.interaction_safety import install_interaction_safety
        install_interaction_safety()
        self.app_model = app_model
        self.store = store
        if workspace.review_surface == "blind_gas_validation":
            inventory = getattr(app_model, "inventory", None)
            if inventory is not None:
                workspace = replace(workspace, selected_candidates={
                    pair.run: pair.candidate_a.candidate_id for pair in inventory.pairs
                })
        self.workspace = workspace
        self.on_back = on_back
        self.cache_root = Path(cache_root).resolve()
        self._blind_gas_session = workspace.review_surface == "blind_gas_validation"
        self._blind_gas_marker = workspace.output_dir / "blind_reveal.json"
        self._blind_gas_locked = self._blind_gas_session and not self._blind_gas_marker.is_file()
        if self._blind_gas_session and not self._blind_gas_locked:
            self._verify_blind_gas_reveal()
        self._space_held = False
        self._flicker_active = False
        self._flicker_key = ""
        self._flicker_view = ""
        self.image_cache = AsyncImageCache(
            self.cache_root / "assets",
            workers=4,
        )
        self.image_cache.image_ready.connect(self._image_loaded)
        self.small_component_cache = SmallComponentCache(workers=2)
        self.image_cache.work_started.connect(self._reference_work_started)
        self.small_component_cache.result_ready.connect(
            self._small_components_loaded
        )
        self.current_run = ""
        self.current_pair: dict[str, Any] | None = None
        self.current_pair_object: Any = None
        self.current_review_root: Path | None = None
        self.current_decision: dict[str, Any] = {}
        self.frame_index = 0
        self.frame_count = 0
        self.frame_choices: dict[int, str] = {}
        self.frame_choice_updated: dict[int, str] = {}
        self.base_key = ""
        self.displayed_base_key = ""
        self.overlay_keys = {"A": "", "B": ""}
        self.base_keys = {"A": "", "B": ""}
        self.displayed_base_keys = {"A": "", "B": ""}
        self.panel_photo_views = {"A": "natural", "B": "change"}
        self.panel_layers = {"A": "none", "B": "none"}
        self.panel_phase_enabled = {"A": True, "B": True}
        self.panel_opacities = {"A": 50, "B": 50}
        self._two_view_swap_held = False
        self._two_view_snapshot: dict[str, tuple[Any, Any, float, str]] = {}
        self._two_view_native_size: tuple[int, int] | None = None
        self._physics_layer_available = False
        self.overlay_swap_held = False
        self.reference_mode = False
        self.injection_references: list[dict[str, Any]] = []
        self.injection_reference_buttons: list[QPushButton] = []
        self.active_injection_reference: int | None = None
        self.plot_keys = {"A": "", "B": ""}
        self.fit_next_image = True
        self.presentation_pool = QThreadPool(self)
        self.presentation_pool.setMaxThreadCount(1)
        self.presentation_tasks = {}
        self.presentation_errors = {}
        self.presentation_validated = set()
        self.syncing_views = False
        self.loading_decision = False
        self.save_dirty = False
        # UI handlers never query the NAS-backed read-only property. The worker
        # refreshes this hint; ReviewStore still enforces the lock on each write.
        self._editing_read_only = bool(getattr(store, "read_only", False))
        if "is_read_only" in getattr(store, "__dict__", {}):
            self._editing_read_only = bool(store.is_read_only)
        self._comparison_save_pool = QThreadPool(self)
        self._comparison_save_pool.setMaxThreadCount(1)
        self._comparison_save_task = None
        self._comparison_pending_saves = OrderedDict()
        self._comparison_unprotected_requests = OrderedDict()
        self._comparison_latest_requests = {}
        self._comparison_saved_records = {}
        self._comparison_saved_generations = {}
        self._comparison_generation = 0
        self._comparison_save_retry_ms = 5000
        self._comparison_local_error = False
        self._comparison_closing = False
        self._comparison_close_with_recovery = False
        scope = hashlib.sha256(str(workspace.output_dir).casefold().encode()).hexdigest()[:24]
        self._comparison_recovery_dir = self.cache_root / "comparison_recovery" / scope
        self._run_list_rows = []
        self._run_decisions = {}
        self._run_area_peaks = {}
        self._all_model_box_counts = {}
        self._run_refresh_task = None
        self._run_refresh_generation = 0
        self._run_refresh_again = False
        self._comparison_load_generation = 0
        self._comparison_load_tasks = {}
        self._comparison_load_pool = QThreadPool(self)
        self._comparison_load_pool.setMaxThreadCount(1)
        self._loaded_candidate_assets = {}
        self._loaded_display_modes = None
        self._loaded_photo_rows = {}
        self._loaded_reference_binding = ""
        self.annotations: list[dict[str, Any]] = []
        self._review_history_data: dict[str, Any] | None = None
        self._review_history_task: ReviewHistoryTask | None = None
        self._review_history_error = ""
        self._review_retraction_task: QRunnable | None = None
        self._review_retraction_notice = ""
        self.reference_guides: list[dict[str, Any]] = []
        self.issue_category_groups: dict[str, QButtonGroup] = {}
        self.issue_draw_buttons: dict[str, dict[str, QPushButton]] = {
            "A": {},
            "B": {},
        }
        self.active_issue_side: str | None = None
        self.active_issue_draw_mode = "pan"
        self.active_guide_orientation: str | None = None
        self.issue_draw_group = QButtonGroup(self)
        self.issue_draw_group.setExclusive(True)
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._flush_save)
        self.speed_loop_held = False
        self.speed_return_frame_index: int | None = None
        self.speed_return_reference_mode = False
        self.speed_return_injection_reference: int | None = None
        self.speed_timer = QTimer(self)
        self.speed_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.speed_timer.timeout.connect(self._advance_speed_loop)
        self.fullscreen_loop_active = False
        self.fullscreen_candidate_side: str | None = None
        self.fullscreen_was_maximized = False
        self.fullscreen_restore_visibility: list[tuple[QWidget, bool]] = []
        self.flag_boxes: dict[str, dict[str, QRadioButton]] = {
            "A": {},
            "B": {},
        }
        self.plot_cards: dict[str, QFrame] = {}
        self.plot_headings: dict[str, QLabel] = {}
        self.candidate_headings: dict[str, QLabel] = {}
        self.issue_headings: dict[str, QLabel] = {}
        self.issue_help_labels: dict[str, QLabel] = {}
        self.legacy_unlocalized_flags: dict[str, set[str]] = {
            "A": set(),
            "B": set(),
        }

        self.setWindowTitle(f"{workspace.title} - FluidFlower Review Desktop")
        self.resize(1880, 1080)
        self.setMinimumSize(1280, 780)
        self._build_ui(reviewer_name.strip(), use_opengl)
        if self._blind_gas_session:
            self._apply_blind_gas_profile()
        QApplication.instance().installEventFilter(self)
        self._refresh_runs()
        if self.run_list.count():
            self.run_list.setCurrentRow(0)
        self._apply_read_only_profile()
        self._install_shortcuts()
        self._apply_approval_profile()
        if self._blind_gas_locked:
            self._apply_blind_gas_profile()

    def _verify_blind_gas_reveal(self) -> dict[str, Any]:
        marker = json.loads(self._blind_gas_marker.read_text(encoding="utf-8"))
        snapshot = Path(str(marker["frozen_blind_snapshot"]))
        expected = str(marker["frozen_blind_snapshot_sha256"])
        if (not snapshot.is_file() or
                hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected):
            raise ValueError("The blind-gas snapshot is missing or its SHA-256 does not match")
        return marker

    def _apply_blind_gas_profile(self) -> None:
        """Lock every prediction route before the first image can be presented."""
        locked = self._blind_gas_locked
        if locked:
            for control in (self.contours, self.phase_map_only, self.show_plots,
                            self.show_review_history):
                control.blockSignals(True)
                control.setChecked(False)
                control.blockSignals(False)
            self.physics_layer.blockSignals(True)
            self.physics_layer.setCurrentIndex(0)
            self.physics_layer.blockSignals(False)
            self._set_overlay_swap_held(False)
            for viewer in self.viewers.values():
                viewer.set_overlay(None)
                viewer.set_phase_map_only(False)
                viewer.set_small_detection_highlight(False)
            self.overlay_keys = {"A": "", "B": ""}
            self.panel_layers = {"A": "none", "B": "none"}
        for widget in (
            self.contours, self.phase_map_only, self.layer_label,
            self.contour_filter, self.physics_layer, self.physics_header,
            self.contour_opacity, self.contour_opacity_label, self.highlight_small,
            self.show_plots, self.plot_container, self.show_review_history,
            self.history_panel, self.contour_legend, self.mass_analysis_button,
            self.local_reference_search, self.optical_cleanup, self.export_video,
            self.verdict, self.verdict_progress, self.next_unreviewed,
            self.decision_title, self.decision_help,
        ):
            widget.setVisible(not locked)
            widget.setEnabled(not locked)
        self.run_sort.setVisible(not locked)
        self.run_sort.setEnabled(not locked)
        for label in self.metric_labels.values():
            label.setVisible(not locked)
        for side, options in self.flag_boxes.items():
            for key, option in options.items():
                allowed = not locked or key in BLIND_GAS_CATEGORIES
                option.setVisible(allowed and (side == "A" or self._two_views_active()
                                               or getattr(self, "comparison_mode", "single_candidate") != "single_candidate"))
                option.setEnabled(allowed and not self._editing_read_only)
            if locked and self._selected_issue_category(side) not in BLIND_GAS_CATEGORIES:
                self._clear_issue_category_selection(side)
        self.end_blind_session.setVisible(locked)
        self.end_blind_session.setEnabled(locked and not self._editing_read_only)
        self._update_two_view_controls()
        if not locked:
            self.plot_container.setVisible(self.show_plots.isChecked())
            self.history_panel.setVisible(self.show_review_history.isChecked())
        if locked:
            self.candidate_headings["A"].setText("Photograph · blind gas validation")
            self.issue_headings["A"].setText("Image region interpretation")
            self.comparison_notice.setText(
                "Blind gas validation: use photographs only. Hold Space to compare "
                "the current view with Change (flat field); release to return. "
                "End the blind session to seal all labels before model layers appear."
            )
            self.annotation_help.setText(
                "Each area is an image-scoped observation with the photographic view "
                "and blind-session provenance saved alongside it."
            )
        else:
            self._apply_comparison_mode(getattr(self, "comparison_mode", "single_candidate"))
            if self.current_pair is not None:
                self._configure_physics_layer()
                self._request_overlays()
                self._update_metrics()

    def _end_blind_gas_session(self) -> None:
        if not self._blind_gas_locked or self._editing_read_only:
            return
        try:
            decisions = deepcopy(self.store.decisions())
            for run, record in self._run_decisions.items():
                if (run not in decisions or timestamp_utc(record.get("updated_utc"))
                        > timestamp_utc((decisions.get(run) or {}).get("updated_utc"))):
                    decisions[run] = deepcopy(record)
            for run, (_generation, request) in self._comparison_latest_requests.items():
                decisions[run] = deepcopy(request)
            if self.current_run:
                decisions[self.current_run] = self._save_request()
            labels_by_run = {
                str(run): [
                    deepcopy(dict(item)) for item in (record.get("annotations") or [])
                    if isinstance(item, Mapping)
                    and item.get("session") == BLIND_GAS_SESSION
                    and not item.get("post_reveal")
                ]
                for run, record in decisions.items()
                if isinstance(record, Mapping)
            }
            stamp = utc_now()
            snapshot_dir = (
                self.workspace.output_dir / "blind_snapshots"
                / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex}"
            )
            snapshot_dir.mkdir(parents=True, exist_ok=False)
            snapshot = snapshot_dir / "blind_labels.json"
            atomic_json(snapshot, {
                "format": BLIND_GAS_SESSION,
                "sealed_utc": stamp,
                "reviewer": self.reviewer.text().strip(),
                "runs": labels_by_run,
                "label_count": sum(map(len, labels_by_run.values())),
            })
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            with (snapshot_dir / "blind_labels.sha256").open("x", encoding="ascii") as stream:
                stream.write(f"{digest}  {snapshot.name}\n")
                stream.flush()
                os.fsync(stream.fileno())
            if self._blind_gas_marker.exists():
                raise FileExistsError("A blind reveal marker already exists")
            atomic_json(self._blind_gas_marker, {
                "format": "blind-gas-reveal-v1",
                "blindness_status": "revealed",
                "first_prediction_reveal_utc": stamp,
                "frozen_blind_snapshot": str(snapshot.resolve()),
                "frozen_blind_snapshot_sha256": digest,
                "post_reveal_edits_are_blind": False,
            })
            self._verify_blind_gas_reveal()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.statusBar().showMessage(f"Blind session remains locked: {exc}", 15000)
            return
        self._set_space_held(False)
        self._blind_gas_locked = False
        try:
            write_review_session_manifest(self.workspace, self.reviewer.text().strip())
        except (OSError, ValueError, TypeError) as exc:
            self.statusBar().showMessage(f"Blind labels sealed; session metadata update failed: {exc}", 15000)
        self._apply_blind_gas_profile()
        self.statusBar().showMessage("Blind labels sealed; model controls unlocked", 10000)

    def _shown_photo_view(self) -> str:
        if self._flicker_active:
            return self._flicker_view
        if self.original_photo.isChecked():
            return "original"
        return "raw_detail" if self.raw_detail.isChecked() else self.display_mode

    def _presented_photo_view(self, side: str) -> str:
        if self._split_views_active():
            source = ("B" if side == "A" else "A") if self._two_view_swap_held else side
            return self.panel_photo_views[source]
        return self._shown_photo_view()

    def _presented_layer(self, side: str) -> str:
        if self._blind_gas_locked:
            return "none"
        if self._split_views_active():
            source = ("B" if side == "A" else "A") if self._two_view_swap_held else side
            return self._effective_panel_layer(source)
        return str(self.contour_filter.currentData() or "all")

    def _effective_panel_layer(self, side: str) -> str:
        if (self._blind_gas_locked or not self.panel_phase_enabled.get(side, True)
                or (not self._split_views_active() and not self.contours.isChecked())):
            return "none"
        return self.panel_layers[side]

    def _flicker_alternate_view(self) -> str:
        return "natural" if self._shown_photo_view() == "change" else "change"

    def _photo_view_asset(self, index: int | None, view: str, *, reference: bool = False) -> DeferredImage:
        root, run = self.current_review_root, self.current_run
        if view == "original":
            if not self._original_view_available(reference=reference):
                raise ValueError("Source-bound Original JPEG is unavailable for this run")
            return self._original_display_asset(index, reference=reference)
        if view == "raw_detail":
            if asset := ReviewImageAssets._rig_raw_display_asset(self, index, view, reference=reference):
                return asset
            from darsia.fluidflower.phase_photographic_assets import resolve_raw_detail
            return ReviewImageAssets._bound_photo_asset(self, DeferredImage(
                f"v5-raw-detail:{'reference' if reference else 'frame'}:{root}:{run}:{index}",
                lambda: resolve_raw_detail(root, run, index, reference=reference),
            ), index, view, reference=reference)
        if view not in self._brightness_modes():
            raise ValueError(f"Photographic view {view!r} is unavailable")
        if view == "change" and (asset := ReviewImageAssets._rig_raw_display_asset(self, index, view, reference=reference)):
            return asset
        if display_modes_available({"frame_sources": (self.current_pair or {}).get("frame_sources") or []}):
            return ReviewImageAssets._bound_photo_asset(self, DeferredImage(
                f"v5-display:{view}:{'reference' if reference else 'frame'}:{root}:{run}:{index}",
                lambda: resolve_display_mode(root, run, index, view, reference=reference),
            ), index, view, reference=reference)
        return ReviewImageAssets._bound_photo_asset(self, DeferredImage(
            f"{'reference' if reference else 'frame'}:{root}:{run}:{index}",
            lambda: (review._native_reference_path(root, run)
                     if reference else review._native_frame_path(root, run, index)),
        ), index, view, reference=reference)

    def _original_view_available(self, *, reference: bool = False) -> bool:
        key = "v5_original_reference_available" if reference else "v5_original_available"
        return bool((self.current_pair or {}).get(key))

    def _add_quick_photo_controls(self, layout, *, side: str | None = None) -> None:
        controls = {}
        for view, label in (("original", "Original"), ("natural", "Natural")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("photoView", True)
            button.setToolTip(
                "Toggle the camera JPEG on the review grid. Press again to return to the previous photographic view."
                if view == "original" else
                "Toggle the fixed Natural rendering. Press again to return to the previous photographic view."
            )
            button.clicked.connect(
                lambda _checked, photo=view, panel=side: self._toggle_quick_photo(photo, panel)
            )
            layout.addWidget(button)
            controls[view] = button
        self.quick_photo_controls[side] = controls
        if side is None:
            self.original_photo = controls["original"]

    def _update_quick_photo_controls(self) -> None:
        reference = bool(self.reference_mode)
        selectable = self.current_pair is not None and (self.active_injection_reference is None
                                                       or self._saved_rig_reference_index() is not None)
        if self.original_photo.isChecked() and not self._original_view_available(reference=reference):
            self.original_photo.setChecked(False)
        for side, controls in self.quick_photo_controls.items():
            current = (self.panel_photo_views[side] if side is not None else
                       "original" if self.original_photo.isChecked() else
                       "raw_detail" if self.raw_detail.isChecked() else self.display_mode)
            previous = self._quick_photo_views.get(side, "natural")
            if (current != previous and current in {"original", "natural"}
                    and not getattr(self, "_restoring_quick_photo", False)):
                return_views = self._quick_photo_return_views[side]
                if not (getattr(self, "_quick_photo_transition", False)
                        and current == "natural" and previous == "original"
                        and current in return_views):
                    return_views[current] = previous
            self._quick_photo_views[side] = current
            for view, button in controls.items():
                button.setEnabled(selectable and (view != "original"
                                  or self._original_view_available(reference=reference)))
                button.blockSignals(True)
                button.setChecked(view == current)
                button.blockSignals(False)
                button.setProperty("photoViewActive", view == current)

    def _toggle_quick_photo(self, view: str, side: str | None = None) -> None:
        # Qt flips a checkable button before clicked() is delivered. Use the
        # last selected view, rather than that already-flipped button state.
        if self._quick_photo_views.get(side) == view:
            previous = self._quick_photo_return_views[side].get(view, "natural")
            self._set_photo_view(previous, side, remember=False)
        else:
            self._select_quick_photo(view, side)

    def _select_quick_photo(self, view: str, side: str | None = None) -> None:
        if view not in {"original", "natural"}:
            raise ValueError(f"Unknown quick photo view: {view}")
        self._set_photo_view(view, side)

    def _set_photo_view(self, view: str, side: str | None = None, *, remember: bool = True) -> None:
        if (self.current_pair is None or (self.active_injection_reference is not None
                                         and self._saved_rig_reference_index() is None)
                or (view == "original" and not self._original_view_available(reference=self.reference_mode))):
            self._update_quick_photo_controls()
            return
        if (view == "raw_detail" and not self.current_pair.get("v5_raw_detail_available")
                or view not in {"original", "raw_detail"} and view not in self._brightness_modes()):
            view = "natural"
        self._set_space_held(False)
        was_restoring = getattr(self, "_restoring_quick_photo", False)
        was_quick_transition = getattr(self, "_quick_photo_transition", False)
        self._restoring_quick_photo = not remember
        self._quick_photo_transition = True
        try:
            if side in {"A", "B"}:
                control = self.panel_photo_controls[side]
                control.setCurrentIndex(control.findData(view))
            else:
                self.original_photo.setChecked(view == "original")
                self.raw_detail.blockSignals(True)
                self.raw_detail.setChecked(view == "raw_detail")
                self.raw_detail.blockSignals(False)
                self.raw_detail.setText("RAW detail: ON" if view == "raw_detail" else "RAW detail: OFF")
                if view not in {"original", "raw_detail"}:
                    self.display_mode = view
                self._update_brightness_control()
                self._request_frame()
            self._update_quick_photo_controls()
        finally:
            self._restoring_quick_photo = was_restoring
            self._quick_photo_transition = was_quick_transition

    def _prefetch_flicker(self) -> None:
        self._flicker_key = ""
        if not self._blind_gas_locked or self.current_pair is None or self.reference_mode:
            return
        alternate = self._flicker_alternate_view()
        if alternate not in self._brightness_modes():
            return
        asset = self._photo_view_asset(self.frame_index, alternate)
        self._flicker_key = self.image_cache.key(asset)
        self.image_cache.prefetch([asset])

    def _set_space_held(self, held: bool) -> None:
        """Space presents cached photos, either as flicker or as two-view swap."""
        if self._split_views_active():
            self._space_held = bool(held)
            self._set_two_view_swap_held(held)
            return
        if not held:
            self._space_held = False
            if self._flicker_active:
                self._flicker_active = False
                self._flicker_view = ""
                primary = self.image_cache.images.get(self.base_key)
                if primary is not None:
                    self.viewers["A"].set_image(self._rig_photo_image(primary), fit=False)
                    self.asset_labels["A"].setText("Full resolution · corrected frame")
            return
        if (self._space_held or not self._blind_gas_locked
                or self.current_pair is None or self.reference_mode
                or self.displayed_base_key != self.base_key):
            return
        alternate = self._flicker_alternate_view()
        if alternate not in self._brightness_modes():
            return
        asset = self._photo_view_asset(self.frame_index, alternate)
        image = self.image_cache.get(asset)
        if image is None:
            self._prefetch_flicker()
            self.statusBar().showMessage("Preparing second photographic view", 3000)
            return
        self._space_held = True
        self._flicker_active = True
        self._flicker_view = alternate
        self.viewers["A"].set_image(self._rig_photo_image(image), fit=False)
        self.asset_labels["A"].setText(f"Full resolution · {DISPLAY_MODE_LABELS[alternate]}")

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self and event.type() == QEvent.Type.WindowDeactivate:
            self._set_space_held(False)
        if (self._space_held and isinstance(watched, QWidget)
                and (watched is self or self.isAncestorOf(watched))
                and event.type() == QEvent.Type.FocusOut):
            self._set_space_held(False)
        if ((self._blind_gas_locked or self._split_views_active())
                and isinstance(watched, QWidget)
                and (watched is self or self.isAncestorOf(watched))
                and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
                and event.key() == Qt.Key.Key_Space):
            parent = watched
            while parent is not None and parent is not self:
                if isinstance(parent, (QLineEdit, QTextEdit, QAbstractSpinBox, QComboBox)):
                    return super().eventFilter(watched, event)
                parent = parent.parentWidget()
            if not event.isAutoRepeat():
                self._set_space_held(event.type() == QEvent.Type.KeyPress)
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def focusOutEvent(self, event: Any) -> None:
        self._set_space_held(False)
        self._set_two_view_swap_held(False)
        super().focusOutEvent(event)

    def _build_ui(self, reviewer_name: str, use_opengl: bool) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.header_widget = QWidget()
        header = QHBoxLayout(self.header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        if self.on_back is not None:
            back = QPushButton("< Start")
            back.setToolTip("Return to the workspace selector")
            back.clicked.connect(self._return_to_start)
            header.addWidget(back)
        title = QLabel(self.workspace.title)
        title.setObjectName("comparisonPageTitle")
        header.addWidget(title)
        badge = QLabel(self.workspace.short_label.upper())
        badge.setObjectName("comparisonBadge")
        header.addWidget(badge)
        header.addStretch(1)
        header.addWidget(QLabel("Reviewer"))
        self.reviewer = QLineEdit(reviewer_name)
        self.reviewer.setReadOnly(True)
        self.reviewer.setMinimumWidth(220)
        header.addWidget(self.reviewer)
        self.save_status_icon = QLabel()
        self.save_status_icon.setFixedSize(13, 13)
        header.addWidget(self.save_status_icon)
        self.save_status = QLabel("Autosave ready")
        self.save_status.setMinimumWidth(220)
        header.addWidget(self.save_status)
        self.open_saved_files = QPushButton("Open saved files")
        self.open_saved_files.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.workspace.output_dir.resolve()))
            )
        )
        header.addWidget(self.open_saved_files)
        if self._blind_gas_session:
            self.end_blind_session = QPushButton("End blind session")
            self.end_blind_session.setToolTip(
                "Seal all blind image labels with SHA-256 before showing model controls."
            )
            self.end_blind_session.clicked.connect(self._end_blind_gas_session)
            header.addWidget(self.end_blind_session)
        root.addWidget(self.header_widget)

        notice = QLabel(
            "Candidate identities are hidden. The two panels always show the same "
            "corrected frame with each candidate's own filled phase map. No quiz "
            "regions or training labels are displayed in this workspace. Values "
            "shown here are 2D optical scores; no physical mass is inferred."
        )
        notice.setObjectName("comparisonNotice")
        notice.setWordWrap(True)
        self.comparison_notice = notice
        root.addWidget(notice)

        outer = QSplitter(Qt.Orientation.Horizontal)
        self.outer_splitter = outer
        run_panel = QWidget()
        self.run_panel = run_panel
        run_layout = QVBoxLayout(run_panel)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.addWidget(QLabel("Experiments to review"))
        self.run_sort = QComboBox()
        self.run_sort.addItem("Experiment number", "number")
        self.run_sort.addItem("Most validation boxes first", "boxes")
        self.run_sort.addItem("Fewest validation boxes first", "boxes_ascending")
        sides = (("A", ""),) if self.workspace.single_model else (
            ("A", " · left model"), ("B", " · right model"))
        for side, suffix in sides:
            for phase, label in (("gas", "free CO2"), ("aqueous", "aqueous CO2")):
                self.run_sort.addItem(
                    f"Lowest peak {label} area{suffix}", f"peak_{phase}_{side}_asc")
                self.run_sort.addItem(
                    f"Highest peak {label} area{suffix}", f"peak_{phase}_{side}_desc")
        self.run_sort.setToolTip(
            "Order of this list only; nothing is filtered out. Box counts are over every saved "
            "review session, so an experiment worked through under an earlier model does not "
            "look untouched here. Other-session counts update with Refresh experiments; "
            "current edits update immediately. Phase area means the maximum predicted visible "
            "porous-domain percentage on any native frame of the selected model. It is neither "
            "physical CO2 mass nor a manual validation. Runs without a complete saved area "
            "report appear last. Area figures update with Refresh experiments."
        )
        self.run_sort.currentIndexChanged.connect(lambda _index: self._populate_runs())
        run_layout.addWidget(self.run_sort)
        self.run_list = QListWidget()
        self.run_list.setMinimumWidth(180)
        self.run_list.setWordWrap(True)
        self.run_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.run_list.currentItemChanged.connect(self._run_changed)
        run_layout.addWidget(self.run_list, 1)
        refresh = QPushButton("Refresh experiments")
        refresh.clicked.connect(self._refresh_runs)
        run_layout.addWidget(refresh)
        outer.addWidget(run_panel)

        comparison = QWidget()
        comparison_layout = QVBoxLayout(comparison)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setSpacing(5)

        controls = QHBoxLayout()
        self.injectors = QCheckBox("Injection points")
        self.injectors.setChecked(True)
        self.injectors.stateChanged.connect(self._render_injection_points)
        controls.addWidget(self.injectors)
        self.sync_views = QCheckBox("Synchronize pan and zoom")
        self.sync_views.setChecked(True)
        self.sync_views.stateChanged.connect(lambda _state: self._sync_view("A"))
        controls.addWidget(self.sync_views)
        self.two_views = QPushButton("Two views")
        self.two_views.setCheckable(True)
        self.two_views.setToolTip("Show two photographic views of the same run and frame.")
        self.two_views.toggled.connect(self._two_views_toggled)
        controls.addWidget(self.two_views)
        self.show_plots = QCheckBox("Show diagnostics")
        self.show_plots.setChecked(False)
        controls.addWidget(self.show_plots)
        self.export_video = QPushButton("Export video")
        self.export_video.setToolTip("Generate a video for the selected experiment on request. Native playback does not require a video file.")
        self.export_video.clicked.connect(self._export_video)
        controls.addWidget(self.export_video)
        controls.addStretch(1)
        comparison_layout.addLayout(controls)

        vertical = QSplitter(Qt.Orientation.Vertical)
        self.plot_container = QFrame()
        self.plot_container.setObjectName("comparisonPlotContainer")
        plot_layout = QHBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(5, 5, 5, 5)
        plot_layout.setSpacing(8)
        self.plot_labels: dict[str, ScaledImageLabel] = {}
        for side in ("A", "B"):
            card = QFrame()
            card.setObjectName(f"candidate{side}Card")
            self.plot_cards[side] = card
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(5, 5, 5, 5)
            heading = QLabel(f"Candidate {side} · diagnostic curve")
            heading.setObjectName("candidateHeading")
            self.plot_headings[side] = heading
            card_layout.addWidget(heading)
            plot = ScaledImageLabel()
            self.plot_labels[side] = plot
            card_layout.addWidget(plot, 1)
            plot_layout.addWidget(card, 1)
        self.show_plots.toggled.connect(self.plot_container.setVisible)
        self.show_plots.toggled.connect(self._diagnostics_toggled)
        self.plot_container.setVisible(self.show_plots.isChecked())
        vertical.addWidget(self.plot_container)

        image_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.image_splitter = image_splitter
        self.viewers: dict[str, ReviewImageView] = {}
        self.candidate_cards: dict[str, QFrame] = {}
        self.metric_labels: dict[str, QLabel] = {}
        self.asset_labels: dict[str, QLabel] = {}
        self.panel_control_rows: dict[str, QWidget] = {}
        self.panel_filled_controls: dict[str, QCheckBox] = {}
        self.panel_floor_contact_controls: dict[str, QCheckBox] = {}
        self.panel_l1_neighbourhood_controls: dict[str, QCheckBox] = {}
        self.panel_photo_controls: dict[str, QComboBox] = {}
        self.panel_layer_controls: dict[str, QComboBox] = {}
        self.panel_phase_controls: dict[str, QPushButton] = {}
        self.panel_opacity_controls: dict[str, QSlider] = {}
        self.panel_opacity_labels: dict[str, QLabel] = {}
        self.quick_photo_controls: dict[str | None, dict[str, QPushButton]] = {}
        self._quick_photo_views = {None: "natural", "A": "natural", "B": "change"}
        self._quick_photo_return_views = {side: {} for side in (None, "A", "B")}
        for side in ("A", "B"):
            card = QFrame()
            card.setObjectName(f"candidate{side}Card")
            self.candidate_cards[side] = card
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(5, 5, 5, 5)
            card_heading = QHBoxLayout()
            heading = QLabel(f"Candidate {side}")
            heading.setObjectName("candidateImageHeading")
            self.candidate_headings[side] = heading
            card_heading.addWidget(heading)
            card_heading.addStretch(1)
            asset = QLabel("Loading full-resolution frame...")
            asset.setObjectName("candidateAssetStatus")
            self.asset_labels[side] = asset
            card_heading.addWidget(asset)
            card_layout.addLayout(card_heading)
            metrics = QLabel("No frame metrics")
            metrics.setObjectName("candidateMetrics")
            metrics.setWordWrap(True)
            self.metric_labels[side] = metrics
            card_layout.addWidget(metrics)
            panel_controls = QWidget()
            panel_layout = QVBoxLayout(panel_controls)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(3)
            panel_row = QHBoxLayout()
            panel_row.setContentsMargins(0, 0, 0, 0)
            panel_layout.addLayout(panel_row)
            panel_row.addWidget(QLabel("Photo"))
            photo = QComboBox()
            for mode in ("original", *DISPLAY_MODES, "raw_detail"):
                photo.addItem(
                    "Original" if mode == "original" else
                    "RAW detail" if mode == "raw_detail" else DISPLAY_MODE_LABELS[mode],
                    mode,
                )
            photo.setCurrentIndex(photo.findData(self.panel_photo_views[side]))
            photo.currentIndexChanged.connect(
                lambda _index, panel_side=side: self._panel_photo_changed(panel_side)
            )
            self.panel_photo_controls[side] = photo
            panel_row.addWidget(photo)
            self._add_quick_photo_controls(panel_row, side=side)
            phase = QPushButton("Phase map: ON")
            phase.setCheckable(True)
            phase.setChecked(self.panel_phase_enabled[side])
            phase.setToolTip("Show or hide this panel's phase map without changing the other panel.")
            phase.toggled.connect(
                lambda checked, panel_side=side: self._panel_phase_toggled(panel_side, checked)
            )
            self.panel_phase_controls[side] = phase
            panel_row.addWidget(phase)
            panel_row = QHBoxLayout()
            panel_layout.addLayout(panel_row)
            panel_row.addWidget(QLabel("Layer"))
            layer = QComboBox()
            for name, value in (("None", "none"), ("Optical", "optical"),
                                ("Physics-corrected", "physics"), ("Difference", "difference"),
                                ("Removed by top-surface rule", "removed_top"), ("Floor fill (removed)", "removed_floor"),
                                ("Removed below sand floor", "below_sand_floor"),
                                ("Removed by floor connectivity", "removed_floor_connectivity"),
                                ("Floor contact fill", "floor_contact_fill_v2"),
                                ("Gas removed by injection envelope", "removed_gas_envelope"),
                                ("Contact fill (b) removed", "removed_contact_fill_b")):
                layer.addItem(name, value)
            layer.setItemData(layer.findData("difference"), DIFFERENCE_LEGEND,
                              Qt.ItemDataRole.ToolTipRole)
            layer.currentIndexChanged.connect(
                lambda _index, panel_side=side: self._panel_layer_changed(panel_side)
            )
            self.panel_layer_controls[side] = layer
            panel_row.addWidget(layer)
            filled = QCheckBox("Extrapolated hatching")
            filled.setChecked(True)
            filled.setToolTip("Mark all filled rig-domain pixels, including water. Independent of this panel's phase-map toggle; display only.")
            filled.toggled.connect(lambda _checked, panel_side=side: self._filled_display_toggled(panel_side))
            self.panel_filled_controls[side] = filled
            panel_row.addWidget(filled)
            contact = QCheckBox("Floor-contact hatching")
            contact.setChecked(True)
            contact.setToolTip("Dark teal opposite-sloping stripes mark aqueous pixels interpreted by the floor-contact rule. Display only; concentration is not measured here.")
            contact.toggled.connect(lambda _checked, panel_side=side: self._filled_display_toggled(panel_side))
            self.panel_floor_contact_controls[side] = contact
            panel_row.addWidget(contact)
            neighbourhood = QCheckBox("L1-neighbourhood hatching")
            neighbourhood.setChecked(True)
            neighbourhood.setToolTip("Vertical purple stripes mark gas added by the L1 neighbourhood rule. Display only; turning this off leaves the saved gas map unchanged.")
            neighbourhood.toggled.connect(lambda _checked, panel_side=side: self._filled_display_toggled(panel_side))
            self.panel_l1_neighbourhood_controls[side] = neighbourhood
            panel_row.addWidget(neighbourhood)
            panel_row.addWidget(QLabel("Opacity"))
            opacity = QSlider(Qt.Orientation.Horizontal)
            opacity.setRange(0, 100)
            opacity.setValue(self.panel_opacities[side])
            opacity.setFixedWidth(85)
            opacity.valueChanged.connect(
                lambda value, panel_side=side: self._panel_opacity_changed(panel_side, value)
            )
            self.panel_opacity_controls[side] = opacity
            panel_row.addWidget(opacity)
            percent = QLabel(f"{self.panel_opacities[side]}%")
            self.panel_opacity_labels[side] = percent
            panel_row.addWidget(percent)
            panel_row.addStretch(1)
            self.panel_control_rows[side] = panel_controls
            card_layout.addWidget(panel_controls)
            viewer = ReviewImageView(use_opengl=use_opengl)
            viewer.set_draw_mode("pan")
            viewer.rectangle_drawn.connect(
                lambda rect, candidate_side=side: self._add_issue_box(
                    candidate_side,
                    rect,
                )
            )
            viewer.freehand_drawn.connect(
                lambda points, candidate_side=side: self._add_issue_freehand(
                    candidate_side,
                    points,
                )
            )
            viewer.guide_placed.connect(
                lambda orientation, point, candidate_side=side: self._add_reference_guide(
                    candidate_side,
                    orientation,
                    point,
                )
            )
            self.viewers[side] = viewer
            viewer.annotation_clicked.connect(
                lambda annotation_id, panel_side=side: self._select_issue_annotation_from_image(panel_side, annotation_id)
            )
            viewer.native_pointer_moved.connect(
                lambda pixel, panel_side=side: self._two_view_pointer_moved(panel_side, pixel)
            )
            viewer.native_pointer_left.connect(
                lambda panel_side=side: self._two_view_pointer_left(panel_side)
            )
            card_layout.addWidget(viewer, 1)
            image_splitter.addWidget(card)
        self.viewers["A"].view_changed.connect(lambda: self._sync_view("A"))
        self.viewers["B"].view_changed.connect(lambda: self._sync_view("B"))
        image_splitter.setSizes([700, 700])
        vertical.addWidget(image_splitter)
        vertical.setSizes([245, 720])
        comparison_layout.addWidget(vertical, 1)

        image_controls = WrappingControlLayout()
        self.image_controls_layout = image_controls
        self.contours = QPushButton("Phase map: ON")
        self.contours.setCheckable(True)
        self.contours.setChecked(True)
        self.contours.setToolTip(
            "Show or hide phase layers over all selected photographic views, including Original and Change (flat field)."
        )
        self.contours.toggled.connect(self._contours_toggled)
        image_controls.addWidget(self.contours)
        self.phase_map_only = QPushButton("Black background: OFF")
        self.phase_map_only.setCheckable(True)
        self.phase_map_only.setToolTip(
            "Hide the corrected photograph and show the selected phase map at full "
            "opacity over a pure black background."
        )
        self.phase_map_only.toggled.connect(self._phase_map_only_toggled)
        image_controls.addWidget(self.phase_map_only)
        self.raw_detail = QPushButton("RAW detail: OFF")
        self.raw_detail.setCheckable(True)
        self.raw_detail.setEnabled(False)
        self.raw_detail.setToolTip(
            "Show the SHA-verified fixed-LUT RAW-detail photograph for this V5 model. "
            "Phase maps, scientific float inputs and annotation coordinates remain unchanged.")
        self.raw_detail.toggled.connect(self._raw_detail_toggled)
        image_controls.addWidget(self.raw_detail)
        self._add_brightness_control(image_controls)
        self._add_quick_photo_controls(image_controls)
        self.highlight_small = QCheckBox("Mark isolated specks")
        self.highlight_small.setChecked(True)
        self.highlight_small.setEnabled(False)
        self.highlight_small.setToolTip(
            "Mark only small CO2-support islands that are disconnected from the "
            "main plume. Large connected plumes are never highlighted; the white "
            "markers are display assistance only."
        )
        self.highlight_small.stateChanged.connect(
            lambda _state: self._apply_phase_map_only_view()
        )
        image_controls.addWidget(self.highlight_small)
        self.show_filled = QCheckBox("Extrapolated hatching")
        self.show_filled.setChecked(True)
        self.show_filled.setToolTip("Mark every filled rig-domain pixel, including water, independently of phase visibility. Display only; saved phase labels stay unchanged.")
        self.show_filled.toggled.connect(lambda _checked: self._filled_display_toggled())
        image_controls.addWidget(self.show_filled)
        self.show_floor_contact = QCheckBox("Floor-contact hatching")
        self.show_floor_contact.setChecked(True)
        self.show_floor_contact.setToolTip("Mark interpreted aqueous floor-contact pixels with dark teal opposite-sloping stripes. Turning this off changes only the display; mass remains blocked.")
        self.show_floor_contact.toggled.connect(lambda _checked: self._filled_display_toggled())
        image_controls.addWidget(self.show_floor_contact)
        self.show_l1_neighbourhood = QCheckBox("L1-neighbourhood hatching")
        self.show_l1_neighbourhood.setChecked(True)
        self.show_l1_neighbourhood.setToolTip("Mark gas added by the L1 neighbourhood rule with vertical purple stripes. Turning this off changes only the display, not phase labels or mass restrictions.")
        self.show_l1_neighbourhood.toggled.connect(lambda _checked: self._filled_display_toggled())
        image_controls.addWidget(self.show_l1_neighbourhood)
        self.layer_label = QLabel("Layer")
        self.layer_label.setEnabled(False)
        layer_controls = QWidget()
        self.layer_controls_widget = layer_controls
        layer_controls_row = QHBoxLayout(layer_controls)
        layer_controls_row.setContentsMargins(0, 0, 0, 0)
        layer_controls_row.addWidget(self.layer_label)
        self.contour_filter = QComboBox()
        self.contour_filter.addItem("All predicted phases", "all")
        self.contour_filter.addItem("Free CO2 only", "gas")
        self.contour_filter.addItem("Aqueous CO2 only", "aqueous")
        self.contour_filter.addItem("Raw optical prediction", "raw")
        self.contour_filter.addItem("Unresolved source support (cyan)", "source_support")
        self.contour_filter.addItem("Removed by top-surface rule", "removed_top")
        self.contour_filter.addItem("Floor fill (removed)", "removed_floor")
        for value in ("below_sand_floor", "removed_floor_connectivity", "floor_contact_fill_v2", "removed_gas_envelope", "removed_contact_fill_b"):
            self.contour_filter.addItem(phase_diagnostic_labels([])[value][0], value)
        self.contour_filter.setToolTip(
            "Choose which model phase class is visible. This filter works both "
            "over the corrected image and on a black background without changing "
            "either candidate's prediction. Raw optical prediction precedes all physics rules; "
            "context diagnostics show changes relative to the inherited model. "
            "Unresolved evidence is a review flag, not an extra physical phase."
        )
        self.contour_filter.setEnabled(False)
        self.contour_filter.currentIndexChanged.connect(self._request_overlays)
        layer_controls_row.addWidget(self.contour_filter)
        image_controls.addWidget(layer_controls)
        self.physics_layer = QComboBox()
        self.physics_layer.addItem("Optical", "optical")
        self.physics_layer.addItem("Physics-corrected", "physics")
        self.physics_layer.addItem("Difference", "difference")
        self.physics_layer.setItemData(2, DIFFERENCE_LEGEND, Qt.ItemDataRole.ToolTipRole)
        self.physics_layer.setToolTip(
            "Switch the display-only free-gas physics layer. The stored phase maps, "
            "model, and annotation evidence remain optical and unchanged.\n" + DIFFERENCE_LEGEND
        )
        self.physics_layer.setEnabled(False)
        self.physics_layer.setVisible(False)
        self.physics_layer.currentIndexChanged.connect(self._request_overlays)
        image_controls.addWidget(self.physics_layer)
        self.mass_analysis_button = QPushButton("Concentration + mass")
        self.mass_analysis_button.setToolTip(
            "Calculate concentration and mass from this candidate's fixed phase maps "
            "and the same registered photographs. Method selection is explicit."
        )
        self.mass_analysis_button.clicked.connect(self._open_mass_analysis)
        image_controls.addWidget(self.mass_analysis_button)
        self.local_reference_search = QPushButton("Local baseline search: OFF")
        self.local_reference_search.setCheckable(True)
        self.local_reference_search.setToolTip(
            "Preview continuous weak color changes next to the plume using the pre-injection image. "
            "Retain weak detections and fill small gaps along the combined plume boundary. "
            "The same treatment applies to reference images. Earlier annotations remain evidence for review."
        )
        self.local_reference_search.toggled.connect(self._local_search_toggled)
        image_controls.addWidget(self.local_reference_search)
        self.hold_overlay_swap = QPushButton("Hold to swap A/B maps")
        self.hold_overlay_swap.setEnabled(False)
        self.hold_overlay_swap.setToolTip(
            "Hold the button to show Candidate B's phase map over image A and "
            "Candidate A's phase map over image B. Release to restore both maps."
        )
        self.hold_overlay_swap.pressed.connect(
            lambda: self._set_overlay_swap_held(True)
        )
        self.hold_overlay_swap.released.connect(
            lambda: self._set_overlay_swap_held(False)
        )
        image_controls.addWidget(self.hold_overlay_swap)
        self.show_annotation_names = QPushButton("Box names: ON")
        self.show_annotation_names.setCheckable(True)
        self.show_annotation_names.setChecked(True)
        self.show_annotation_names.setToolTip(
            "Show or hide the category name above every model-issue box."
        )
        self.show_annotation_names.toggled.connect(
            self._annotation_names_toggled
        )
        self.show_annotations = QPushButton("Boxes + names: ON")
        self.show_annotations.setCheckable(True)
        self.show_annotations.setChecked(True)
        self.show_annotations.setToolTip(
            "Show or hide all model-issue boxes and their names. Injection points "
            "and fixed reference guides are unaffected."
        )
        self.show_annotations.toggled.connect(self._annotations_toggled)
        self.show_review_history = QPushButton("History: OFF")
        self.show_review_history.setCheckable(True)
        self.show_review_history.setToolTip(
            "Show saved boxes, frame verdicts and notes from earlier review sessions "
            "and model training snapshots. Dashed boxes are read-only history."
        )
        self.show_review_history.toggled.connect(self._review_history_toggled)
        self.reference = QPushButton("Pre-injection reference")
        self.reference.setProperty("referenceImage", True)
        self.reference.setCheckable(True)
        self.reference.setEnabled(False)
        self.reference.setToolTip(
            "Show the corrected pre-injection image aligned to the currently selected frame."
        )
        self.reference.toggled.connect(self._reference_toggled)
        opacity_controls = QWidget()
        self.opacity_controls_widget = opacity_controls
        opacity_controls_row = QHBoxLayout(opacity_controls)
        opacity_controls_row.setContentsMargins(0, 0, 0, 0)
        opacity_controls_row.addWidget(QLabel("Opacity"))
        self.contour_opacity = QSlider(Qt.Orientation.Horizontal)
        self.contour_opacity.setRange(20, 100)
        self.contour_opacity.setValue(50)
        self.contour_opacity.setFixedWidth(140)
        self.contour_opacity.setToolTip(
            "Adjust phase-map opacity while keeping the corrected image visible beneath it."
        )
        self.contour_opacity.valueChanged.connect(self._contour_opacity_changed)
        opacity_controls_row.addWidget(self.contour_opacity)
        self.contour_opacity_label = QLabel("50%")
        self.contour_opacity_label.setMinimumWidth(38)
        opacity_controls_row.addWidget(self.contour_opacity_label)
        image_controls.addWidget(opacity_controls)
        self._contour_opacity_changed(self.contour_opacity.value())
        comparison_layout.addLayout(image_controls)
        from darsia.fluidflower.ui.optical_cleanup import OpticalCleanup
        self.optical_cleanup = OpticalCleanup(self)
        comparison_layout.addWidget(self.optical_cleanup)

        self.reference_controls_widget = QWidget()
        reference_controls = QHBoxLayout(self.reference_controls_widget)
        reference_controls.setContentsMargins(0, 0, 0, 0)
        reference_controls.addWidget(QLabel("Reference images"))
        reference_controls.addWidget(self.reference)
        self.injection_reference_layout = QHBoxLayout()
        reference_controls.addLayout(self.injection_reference_layout)
        reference_controls.addStretch(1)
        comparison_layout.addWidget(self.reference_controls_widget)

        self.annotation_visibility_widget = QWidget()
        annotation_visibility = QHBoxLayout(self.annotation_visibility_widget)
        annotation_visibility.setContentsMargins(0, 0, 0, 0)
        annotation_visibility.addWidget(QLabel("Review annotations"))
        annotation_visibility.addWidget(self.show_annotation_names)
        annotation_visibility.addWidget(self.show_annotations)
        annotation_visibility.addWidget(self.show_review_history)
        annotation_visibility.addStretch(1)
        comparison_layout.addWidget(self.annotation_visibility_widget)

        self.guide_controls_widget = QWidget()
        guide_controls = WrappingControlLayout()
        self.guide_controls_widget.setLayout(guide_controls)
        guide_controls.setContentsMargins(0, 0, 0, 0)
        guide_controls.addWidget(QLabel("Fixed reference guides"))
        self.reference_guide_name = QLineEdit()
        self.reference_guide_name.setPlaceholderText("Optional guide name")
        self.reference_guide_name.setMinimumWidth(
            self.reference_guide_name.fontMetrics().horizontalAdvance(
                self.reference_guide_name.placeholderText()) + 20)
        self.reference_guide_name.setMaximumWidth(210)
        self.reference_guide_name.setToolTip(
            "Name the next line. If left blank, the app assigns H1, V1, and so on."
        )
        guide_controls.addWidget(self.reference_guide_name)
        self.horizontal_guide = QPushButton("Place horizontal")
        self.horizontal_guide.setCheckable(True)
        self.horizontal_guide.setToolTip(
            "Keep active and click either image to add horizontal guides at fixed image coordinates."
        )
        self.horizontal_guide.clicked.connect(
            lambda _checked=False: self._set_reference_guide_mode("horizontal")
        )
        self.issue_draw_group.addButton(self.horizontal_guide)
        guide_controls.addWidget(self.horizontal_guide)
        self.vertical_guide = QPushButton("Place vertical")
        self.vertical_guide.setCheckable(True)
        self.vertical_guide.setToolTip(
            "Keep active and click either image to add vertical guides at fixed image coordinates."
        )
        self.vertical_guide.clicked.connect(
            lambda _checked=False: self._set_reference_guide_mode("vertical")
        )
        self.issue_draw_group.addButton(self.vertical_guide)
        guide_controls.addWidget(self.vertical_guide)
        saved_guides = QWidget()
        saved_guides_row = QHBoxLayout(saved_guides)
        saved_guides_row.setContentsMargins(0, 0, 0, 0)
        saved_guides_row.addWidget(QLabel("Saved"))
        self.reference_guide_list = QComboBox()
        self.reference_guide_list.setMinimumWidth(230)
        self.reference_guide_list.setToolTip(
            "Named guides stored for this experiment and shown at the same position on every frame."
        )
        saved_guides_row.addWidget(self.reference_guide_list)
        guide_controls.addWidget(saved_guides)
        self.delete_reference_guide = QPushButton("Delete selected")
        self.delete_reference_guide.setEnabled(False)
        self.delete_reference_guide.clicked.connect(
            self._delete_selected_reference_guide
        )
        guide_controls.addWidget(self.delete_reference_guide)
        comparison_layout.addWidget(self.guide_controls_widget)

        contour_legend = QLabel(
            "Green = aqueous CO2. Pink = free CO2. "
            "Cyan within sand = unresolved source support, not CO2. "
            "Transparent = no visible CO2 predicted on supported pixels. These "
            "are mutually exclusive model display classes, not ground truth. "
            "Optional white markers identify only small disconnected islands; "
            "they are display assistance, not predicted area."
        )
        contour_legend.setObjectName("comparisonLegend")
        self.physics_header = QLabel("Gas physics layer v6 (free gas only)")
        self.physics_header.setObjectName("physicsLayerHeader")
        self.physics_header.setVisible(False)
        comparison_layout.addWidget(self.physics_header)
        self.contour_legend = contour_legend
        self.default_contour_legend = contour_legend.text()
        contour_legend.setWordWrap(True)
        self.contour_legend = contour_legend
        comparison_layout.addWidget(contour_legend)

        timeline = QHBoxLayout()
        self.speed_loop = QPushButton("Hold: speed loop")
        self.speed_loop.setToolTip(
            "Hold to loop from the pre-injection reference through all calibration "
            "frames. Release to return to the exact frame shown before playback. "
            "Zoom, pan position, and phase-map visibility are preserved."
        )
        self.speed_loop.pressed.connect(self._start_speed_loop)
        self.speed_loop.released.connect(self._stop_speed_loop)
        timeline.addWidget(self.speed_loop)
        self.fullscreen_buttons: dict[str, QPushButton] = {}
        for side in ("A", "B"):
            button = QPushButton(f"Full-screen {side}")
            button.setToolTip(
                f"Play only Candidate {side} in full screen. Phase-map and layer "
                "controls remain available below the image. Press Esc to exit."
            )
            button.clicked.connect(
                lambda _checked=False, candidate_side=side: self._toggle_fullscreen_loop(
                    candidate_side
                )
            )
            self.fullscreen_buttons[side] = button
            timeline.addWidget(button)
        timeline.addWidget(QLabel("fps"))
        self.speed_fps = QSpinBox()
        self.speed_fps.setRange(1, 15)
        self.speed_fps.setValue(8)
        self.speed_fps.setFixedWidth(58)
        self.speed_fps.setToolTip(
            "Maximum playback rate. Playback waits for each full-resolution frame "
            "and any visible phase maps before advancing."
        )
        self.speed_fps.valueChanged.connect(self._speed_rate_changed)
        timeline.addWidget(self.speed_fps)
        self.previous_frame = QToolButton()
        self.previous_frame.setText("<")
        self.previous_frame.setToolTip("Previous frame")
        self.previous_frame.clicked.connect(lambda: self._step_frame(-1))
        timeline.addWidget(self.previous_frame)
        self.frame_slider = FrameTimelineSlider()
        self.frame_slider.set_marker(None)
        self.frame_slider.valueChanged.connect(self._set_frame)
        timeline.addWidget(self.frame_slider, 1)
        self.next_frame = QToolButton()
        self.next_frame.setText(">")
        self.next_frame.setToolTip("Next frame")
        self.next_frame.clicked.connect(lambda: self._step_frame(1))
        timeline.addWidget(self.next_frame)
        self.frame_label = FrameTimelineLabel()
        timeline.addWidget(self.frame_label)
        comparison_layout.addLayout(timeline)

        self.view_controls_widget = QWidget()
        view_controls = QHBoxLayout(self.view_controls_widget)
        view_controls.setContentsMargins(0, 0, 0, 0)
        for label, callback, tooltip in (
            ("-", lambda: self._zoom_views(1.0 / 1.25), "Zoom out both panels"),
            ("Fit", self._fit_views, "Fit the full frame in both panels"),
            ("1 px", self._native_views, "One image pixel per screen pixel"),
            ("+", lambda: self._zoom_views(1.25), "Zoom in both panels"),
        ):
            button = QPushButton(label)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            view_controls.addWidget(button)
        view_controls.addStretch(1)
        self.frame_source = QLabel("No frame loaded")
        self.frame_source.setObjectName("comparisonFrameSource")
        view_controls.addWidget(self.frame_source)
        comparison_layout.addWidget(self.view_controls_widget)
        outer.addWidget(comparison)

        decision_scroll = QScrollArea()
        self.decision_scroll = decision_scroll
        decision_scroll.setWidgetResizable(True)
        decision_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        decision_scroll.setMinimumWidth(330)
        decision_scroll.setMaximumWidth(420)
        decision_panel = QWidget()
        decision_layout = QVBoxLayout(decision_panel)
        decision_title = QLabel("Current-frame model decision")
        decision_title.setObjectName("comparisonDecisionTitle")
        self.decision_title = decision_title
        decision_layout.addWidget(decision_title)
        decision_help = QLabel(
            "Choose which candidate is more physically plausible in the image "
            "currently shown. The decision belongs only to this frame. Use "
            "Unresolved when the optical evidence is insufficient."
        )
        decision_help.setWordWrap(True)
        self.decision_help = decision_help
        decision_layout.addWidget(decision_help)
        self.verdict = QComboBox()
        self.verdict.setMinimumContentsLength(20)
        self.verdict.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._configure_verdict_options("pair")
        self.verdict.currentIndexChanged.connect(self._frame_verdict_changed)
        decision_layout.addWidget(self.verdict)
        verdict_status = QHBoxLayout()
        self.verdict_progress = QLabel("0 / 0 frames reviewed")
        self.verdict_progress.setObjectName("candidateMetrics")
        verdict_status.addWidget(self.verdict_progress, 1)
        self.next_unreviewed = QPushButton("Next unreviewed")
        self.next_unreviewed.clicked.connect(self._open_next_unreviewed_frame)
        verdict_status.addWidget(self.next_unreviewed)
        decision_layout.addLayout(verdict_status)

        self.pan_images = QPushButton("Pan / inspect images")
        self.pan_images.setCheckable(True)
        self.pan_images.setChecked(True)
        self.pan_images.clicked.connect(lambda: self._set_issue_draw_side(None))
        self.issue_draw_group.addButton(self.pan_images)
        decision_layout.addWidget(self.pan_images)

        for side in ("A", "B"):
            heading = QLabel(f"Candidate {side} region interpretation")
            heading.setObjectName("candidateFlagHeading")
            self.issue_headings[side] = heading
            decision_layout.addWidget(heading)
            category_help = QLabel(
                "Select one visible phase class or one physical issue for the next "
                "area. Phase classes use the same vocabulary as blind labeling; "
                "the model disagreement is derived automatically."
            )
            category_help.setWordWrap(True)
            category_help.setObjectName("comparisonLegend")
            self.issue_help_labels[side] = category_help
            decision_layout.addWidget(category_help)
            category_group = QButtonGroup(self)
            category_group.setExclusive(True)
            self.issue_category_groups[side] = category_group
            phase_keys = {key for key, _label in self.PHASE_CLASSIFICATION_LABELS}
            phase_section = QLabel("Visible 2D phase classification")
            phase_section.setObjectName("comparisonLegend")
            decision_layout.addWidget(phase_section)
            physical_heading_added = False
            offered_labels = (
                self.FLAG_LABELS + BLIND_GAS_ARTIFACT_LABELS
                if self._blind_gas_session else self.FLAG_LABELS
            )
            for key, label in offered_labels:
                if key not in phase_keys and not physical_heading_added:
                    issue_section = QLabel("Physical, temporal, or detection issue")
                    issue_section.setObjectName("comparisonLegend")
                    decision_layout.addWidget(issue_section)
                    physical_heading_added = True
                option = QRadioButton(label)
                # Leave room for pane margins, the radio indicator and vertical scrolling.
                text_width = decision_scroll.minimumWidth() - 70
                lines = [""]
                for word in label.split():
                    text = (lines[-1] + " " + word).strip()
                    if lines[-1] and option.fontMetrics().horizontalAdvance(text) > text_width:
                        lines.append(word)
                    else:
                        lines[-1] = text
                option.setText("\n".join(lines))
                option.setAccessibleName(label)
                if key in review.CANDIDATE_ISSUE_HELP:
                    option.setToolTip(review.CANDIDATE_ISSUE_HELP[key])
                option.setProperty("issue_key", key)
                option.toggled.connect(
                    lambda checked, candidate_side=side, issue_key=key: self._issue_category_changed(
                        candidate_side,
                        issue_key,
                        checked,
                    )
                )
                category_group.addButton(option)
                self.flag_boxes[side][key] = option
                decision_layout.addWidget(option)
            draw_actions = WrappingControlLayout()
            for mode, label in (
                ("box", f"Draw {side} box"),
                ("freehand", f"Draw {side} freehand"),
                ("polygon", f"Draw {side} polygon"),
            ):
                draw = QPushButton(label)
                draw.setCheckable(True)
                draw.setEnabled(False)
                draw.setToolTip(
                    "Left-click at least three corners. Right-click closes the area. Esc cancels. "
                    if mode == "polygon" else
                    f"Select one Candidate {side} issue category first. Keep this "
                    "active to mark several areas; choose Pan / inspect when finished."
                )
                draw.clicked.connect(
                    lambda _checked=False, candidate_side=side, draw_mode=mode: self._set_issue_draw_side(
                        candidate_side,
                        draw_mode,
                    )
                )
                self.issue_draw_buttons[side][mode] = draw
                self.issue_draw_group.addButton(draw)
                draw_actions.addWidget(draw)
            decision_layout.addLayout(draw_actions)

        annotation_title = QLabel("Frame-specific issue areas")
        annotation_title.setObjectName("candidateFlagHeading")
        decision_layout.addWidget(annotation_title)
        annotation_help = QLabel(
            "The human-visible phase is stored once using the shared phase "
            "vocabulary. For phase labels, DarSIA also records what the model "
            "predicted in the area and derives missing or unsupported phases. "
            "Physical and temporal issues remain separate labels. Every area "
            "stores exactly one human category."
        )
        self.annotation_help = annotation_help
        annotation_help.setWordWrap(True)
        decision_layout.addWidget(annotation_help)
        self.issue_annotation_list = QListWidget()
        self.issue_annotation_list.setMinimumHeight(145)
        self.issue_annotation_list.currentItemChanged.connect(
            self._issue_annotation_selection_changed
        )
        self.issue_annotation_list.itemActivated.connect(
            self._issue_annotation_opened
        )
        self.issue_annotation_list.itemClicked.connect(
            self._issue_annotation_opened
        )
        decision_layout.addWidget(self.issue_annotation_list)
        annotation_actions = QHBoxLayout()
        self.update_issue_box = QPushButton("Apply category")
        self.update_issue_box.setEnabled(False)
        self.update_issue_box.clicked.connect(self._update_selected_issue_box)
        annotation_actions.addWidget(self.update_issue_box)
        self.delete_issue_box = QPushButton("Delete area")
        self.delete_issue_box.setEnabled(False)
        self.delete_issue_box.clicked.connect(self._delete_selected_issue_box)
        annotation_actions.addWidget(self.delete_issue_box)
        decision_layout.addLayout(annotation_actions)

        self.history_panel = QWidget()
        history_layout = QVBoxLayout(self.history_panel)
        history_layout.setContentsMargins(0, 6, 0, 6)
        history_heading = QHBoxLayout()
        history_heading.addWidget(QLabel("Saved history · original records preserved"))
        self.refresh_history = QPushButton("Refresh")
        self.refresh_history.clicked.connect(self._start_review_history_load)
        history_heading.addWidget(self.refresh_history)
        history_layout.addLayout(history_heading)
        self.history_status = QLabel("Turn on History to load earlier reviews.")
        self.history_status.setWordWrap(True)
        self.history_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        history_layout.addWidget(self.history_status)
        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(120)
        self.history_list.setMaximumHeight(230)
        self.history_list.itemClicked.connect(self._review_history_opened)
        self.history_list.itemActivated.connect(self._review_history_opened)
        history_layout.addWidget(self.history_list)
        self.history_details = QLabel("Select a historical entry to see its source and date.")
        self.history_details.setWordWrap(True)
        self.history_details.setTextFormat(Qt.TextFormat.PlainText)
        self.history_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        history_layout.addWidget(self.history_details)
        self.withdraw_history = QPushButton("Withdraw from future training")
        self.withdraw_history.setEnabled(False)
        self.withdraw_history.setToolTip(
            "Exclude the selected historical validation and its saved copies from future training. "
            "Existing models and original records stay unchanged. After retraining, "
            "the withdrawn item disappears from the new model's ordinary history."
        )
        self.withdraw_history.clicked.connect(self._withdraw_review_history)
        history_layout.addWidget(self.withdraw_history)
        history_layout.addWidget(QLabel("New classification for the selected historical box"))
        self.history_classification = QComboBox()
        self.history_classification.setMinimumContentsLength(20)
        self.history_classification.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.history_classification.addItem("Choose a new classificationâ€¦", "")
        for key, label in self.ISSUE_CATEGORY_LABELS:
            self.history_classification.addItem(label, key)
            if key in review.CANDIDATE_ISSUE_HELP:
                self.history_classification.setItemData(
                    self.history_classification.count() - 1,
                    review.CANDIDATE_ISSUE_HELP[key],
                    Qt.ItemDataRole.ToolTipRole,
                )
        self.history_classification.currentIndexChanged.connect(self._history_classification_changed)
        self.history_classification.setEnabled(False)
        history_layout.addWidget(self.history_classification)
        self.reclassify_history = QPushButton("Save replacement classification")
        self.reclassify_history.setEnabled(False)
        self.reclassify_history.setToolTip(
            "Save a new human interpretation of this same source-photo area and withdraw "
            "the old classification in one action. Existing models and original records stay unchanged."
        )
        self.reclassify_history.clicked.connect(self._reclassify_review_history)
        history_layout.addWidget(self.reclassify_history)
        self.history_panel.setVisible(False)
        decision_layout.addWidget(self.history_panel)

        decision_layout.addWidget(QLabel("Experiment-level notes"))
        self.notes = QTextEdit()
        self.notes.setPlaceholderText(
            "Describe spatial, phase, temporal, or mass-balance differences."
        )
        self.notes.setMinimumHeight(120)
        self.notes.textChanged.connect(self._schedule_save)
        decision_layout.addWidget(self.notes)
        self.save_now = QPushButton("Save now")
        self.save_now.clicked.connect(self._save_now)
        decision_layout.addWidget(self.save_now)
        decision_layout.addStretch(1)
        for label in decision_panel.findChildren(QLabel):
            label.setWordWrap(True)
        decision_scroll.setWidget(decision_panel)
        outer.addWidget(decision_scroll)
        outer.setSizes([185, 1380, 350])
        root.addWidget(outer, 1)

        if self.workspace.single_model and not self.workspace.blind:
            from darsia.fluidflower.tools.phase_model_comparison_view import install_review_comparison_tabs
            install_review_comparison_tabs(self, central)
        else:
            self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#171a1d; color:#e7eaed; font-size:13px; }
            #comparisonPageTitle { font-size:20px; font-weight:650; }
            #comparisonBadge { background:#664f16; color:#fff0b8; border:1px solid #d7a934; border-radius:3px; padding:3px 7px; font-weight:650; }
            #comparisonNotice { background:#382f1d; color:#ffe3a0; border-left:4px solid #d7a934; padding:7px 9px; }
            #comparisonLegend { color:#b7c2ca; font-size:12px; }
            #comparisonPlotContainer { background:#15191c; border:1px solid #343c43; }
            #candidateACard, #candidateBCard { background:#20252a; border:1px solid #3e474f; border-radius:3px; }
            #candidateACard { border-top:3px solid #4d91c4; }
            #candidateBCard { border-top:3px solid #d7a934; }
            #candidateHeading, #candidateImageHeading, #comparisonDecisionTitle { font-size:15px; font-weight:650; }
            #candidateMetrics { color:#cad2d8; font-size:12px; }
            #candidateAssetStatus, #comparisonFrameSource { color:#9eabb4; font-size:11px; }
            #candidateFlagHeading { margin-top:8px; font-weight:650; color:#dfe5e9; }
            QPushButton, QToolButton, QComboBox, QLineEdit, QTextEdit, QListWidget {
                background:#24292e; border:1px solid #3a4249; border-radius:4px; padding:5px;
            }
            QPushButton:hover, QToolButton:hover { border-color:#6da7d9; }
            QPushButton[referenceImage="true"]:checked:enabled {
                background:#71b7df; color:#102631; border-color:#a5dcff;
            }
            QPushButton[photoView="true"]:checked:enabled {
                background:#294452; color:#eaf4fc; border-color:#75abcd;
            }
            QListWidget::item { padding:5px; }
            QListWidget::item:selected { background:#185d86; }
            QSlider::groove:horizontal { height:5px; background:#343b42; }
            QSlider::handle:horizontal { width:13px; margin:-5px 0; background:#71b7df; }
            """
        )

    def _apply_read_only_profile(self) -> None:
        if not self._editing_read_only:
            return
        self.setWindowTitle(f"{self.workspace.title} - Read-only evidence audit")
        self.comparison_notice.setText(
            "Read-only evidence audit: this session was frozen as model-training "
            "input. Images, phase maps, frames, saved areas, and prior decisions "
            "remain inspectable, but the evidence cannot be modified."
        )
        self.reviewer.setReadOnly(True)
        self.verdict.setEnabled(False)
        self.notes.setEnabled(False)
        self.save_now.setEnabled(False)
        self.reference_guide_name.setEnabled(False)
        self.horizontal_guide.setEnabled(False)
        self.vertical_guide.setEnabled(False)
        self.delete_reference_guide.setEnabled(False)
        self.update_issue_box.setEnabled(False)
        self.delete_issue_box.setEnabled(False)
        for group in self.issue_category_groups.values():
            for button in group.buttons():
                button.setEnabled(False)
        for controls in self.issue_draw_buttons.values():
            for button in controls.values():
                button.setEnabled(False)
        self._set_save_status("saved", "Autosave ready")

    def _apply_approval_profile(self) -> None:
        """Hide authoring controls in the ordinary per-run inspection surface."""

        if not self.workspace.approval_only:
            return
        self.decision_scroll.setVisible(False)
        self.annotation_visibility_widget.setVisible(True)
        # The saved-area list can be revealed by clicking a box. Inspection
        # still cannot author evidence through controls in that same panel.
        for name in ("verdict", "notes", "save_now", "update_issue_box", "delete_issue_box"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(False)
                widget.hide()
        for controls in getattr(self, "issue_draw_buttons", {}).values():
            for button in controls.values():
                button.setEnabled(False)
        for group in getattr(self, "issue_category_groups", {}).values():
            for button in group.buttons():
                button.setEnabled(False)
        self.guide_controls_widget.setVisible(False)
        self.save_status_icon.setVisible(False)
        self.save_status.setVisible(False)
        self.open_saved_files.setVisible(False)
        self.outer_splitter.setSizes([185, 1550, 0])
        self.comparison_notice.setText(
            "Inspect the selected model on the complete corrected 2D image "
            "sequence. Phase-map visibility, layer filters, the pre-injection "
            "reference, diagnostics, zoom, timeline, and playback remain "
            "available. Record approval or request local correction in the "
            "DarSIA project window; this inspection window does not collect "
            "training annotations."
        )

    def _two_views_active(self) -> bool:
        return (getattr(self, "comparison_mode", "pair") == "single_candidate"
                and self.two_views.isChecked())

    def _split_views_active(self) -> bool:
        """All two-panel displays share controls, without changing label scope."""
        return getattr(self, "comparison_mode", "single_candidate") != "single_candidate" or self._two_views_active()

    def _panel_phase_toggled(self, side: str, checked: bool) -> None:
        self._set_two_view_swap_held(False)
        self.panel_phase_enabled[side] = bool(checked and not self._blind_gas_locked)
        if self.panel_phase_enabled[side] and self.panel_layers[side] == "none":
            self.panel_layers[side] = "optical"
        control = self.panel_phase_controls[side]
        control.blockSignals(True)
        control.setChecked(self.panel_phase_enabled[side])
        control.setText("Phase map: ON" if self.panel_phase_enabled[side] else "Phase map: OFF")
        control.blockSignals(False)
        self._update_two_view_controls()
        if self._split_views_active() and self.current_pair is not None:
            self._request_two_view_overlays(side)

    def _update_two_view_controls(self) -> None:
        if not hasattr(self, "panel_photo_controls"):
            return
        modes = set(self._brightness_modes()) if self.current_pair is not None else set(DISPLAY_MODES)
        raw_available = bool((self.current_pair or {}).get("v5_raw_detail_available"))
        physics_available = bool(getattr(self, "_physics_layer_available", False))
        for side in ("A", "B"):
            photo = self.panel_photo_controls[side]
            layer = self.panel_layer_controls[side]
            for index in range(photo.count()):
                value = str(photo.itemData(index))
                photo.model().item(index).setEnabled(
                    self._original_view_available(reference=self.reference_mode) if value == "original" else
                    raw_available if value == "raw_detail" else value in modes
                )
            if self._blind_gas_locked:
                self.panel_layers[side] = "none"
            phase = self.panel_phase_controls[side]
            phase.blockSignals(True)
            phase.setChecked(self.panel_phase_enabled[side] and not self._blind_gas_locked)
            phase.setText("Phase map: ON" if phase.isChecked() else "Phase map: OFF")
            phase.blockSignals(False)
            phase.setEnabled(self._split_views_active() and not self._blind_gas_locked)
            phase.setVisible(not self._blind_gas_locked)
            layer.setVisible(not self._blind_gas_locked)
            if contact := (getattr(self, "panel_floor_contact_controls", {}) or {}).get(side):
                contact.setVisible(not self._blind_gas_locked)
            if neighbourhood := (getattr(self, "panel_l1_neighbourhood_controls", {}) or {}).get(side):
                neighbourhood.setVisible(not self._blind_gas_locked and self._l1_neighbourhood_display_available(side))
            self.panel_opacity_controls[side].setVisible(not self._blind_gas_locked)
            self.panel_opacity_labels[side].setVisible(not self._blind_gas_locked)
            for index in range(layer.count()):
                value = str(layer.itemData(index))
                layer.model().item(index).setEnabled(
                    self._p1_final3_overlay_available(side) if value == "removed_contact_fill_b"
                    else (self._p1_final2_overlay_available(side) or self._p1_final3_overlay_available(side)) if value in P1_FINAL2_OVERLAYS
                    else self._p1_final_overlay_available(side) if value in {"removed_top", "removed_floor"}
                    else value in {"none", "optical"} or physics_available
                )
            layer.blockSignals(True)
            layer.setCurrentIndex(max(0, layer.findData(self.panel_layers[side])))
            layer.blockSignals(False)
            layer.setEnabled(self._split_views_active() and not self._blind_gas_locked)
            self.panel_opacity_controls[side].setEnabled(
                self._split_views_active() and not self._blind_gas_locked
                and self._effective_panel_layer(side) != "none"
            )
        self._update_quick_photo_controls()

    def _two_views_toggled(self, checked: bool) -> None:
        if checked and getattr(self, "comparison_mode", "pair") != "single_candidate":
            self.two_views.setChecked(False)
            return
        self._set_two_view_swap_held(False)
        self._set_space_held(False)
        if checked and not getattr(self, "_two_views_initialized", False):
            primary = self._shown_photo_view()
            self.panel_photo_views = {
                "A": primary,
                "B": "natural" if primary == "change" else "change",
            }
            for side, view in self.panel_photo_views.items():
                control = self.panel_photo_controls[side]
                control.blockSignals(True)
                control.setCurrentIndex(max(0, control.findData(view)))
                control.blockSignals(False)
            self.panel_layers = {side: "none" if self._blind_gas_locked else "optical"
                                 for side in ("A", "B")}
            self._two_views_initialized = True
        self.base_keys = {"A": "", "B": ""}
        self.displayed_base_keys = {"A": "", "B": ""}
        self._two_view_native_size = None
        self._apply_comparison_mode(getattr(self, "comparison_mode", "single_candidate"))
        if self._blind_gas_locked:
            self._apply_blind_gas_profile()
        self._request_frame()

    def _panel_photo_changed(self, side: str) -> None:
        view = str(self.panel_photo_controls[side].currentData() or "natural")
        self.panel_photo_views[side] = view
        self._update_quick_photo_controls()
        if self._split_views_active() and self.current_pair is not None:
            self._set_two_view_swap_held(False)
            self._request_two_view_frame(side)

    def _panel_layer_changed(self, side: str) -> None:
        previous_layer = self.panel_layers[side]
        layer = str(self.panel_layer_controls[side].currentData() or "none")
        if self._blind_gas_locked and layer != "none":
            control = self.panel_layer_controls[side]
            control.blockSignals(True)
            control.setCurrentIndex(0)
            control.blockSignals(False)
            layer = "none"
        self.panel_layers[side] = layer
        if layer == "none":
            self.panel_phase_enabled[side] = False
        elif previous_layer == "none":
            self.panel_phase_enabled[side] = True
        self._update_two_view_controls()
        if self._split_views_active() and self.current_pair is not None:
            self._set_two_view_swap_held(False)
            self._request_two_view_overlays(side)

    def _panel_opacity_changed(self, side: str, value: int) -> None:
        self.panel_opacities[side] = int(value)
        self.panel_opacity_labels[side].setText(f"{int(value)}%")
        if self._split_views_active():
            self._set_two_view_swap_held(False)
            self.viewers[side].set_overlay_opacity(int(value) / 100.0)

    def _two_view_pointer_moved(self, source_side: str, pixel: tuple[int, int]) -> None:
        if not self._split_views_active():
            return
        target_side = "B" if source_side == "A" else "A"
        source = self.viewers[source_side].base_item.pixmap()
        target = self.viewers[target_side].base_item.pixmap()
        if (source.isNull() or target.isNull() or
                (source.width(), source.height()) != (target.width(), target.height())):
            self.viewers[target_side].set_peer_cursor(None)
            return
        self.viewers[source_side].set_peer_cursor(pixel)
        self.viewers[target_side].set_peer_cursor(pixel)

    def _two_view_pointer_left(self, source_side: str) -> None:
        if self._split_views_active():
            self.viewers[source_side].set_peer_cursor(None)
            self.viewers["B" if source_side == "A" else "A"].set_peer_cursor(None)

    def _apply_comparison_mode(self, comparison_mode: str) -> None:
        """Present one model once when a campaign contains a single candidate."""

        self.comparison_mode = str(comparison_mode or "pair")
        single = self.comparison_mode == "single_candidate"
        two = single and self.two_views.isChecked()
        split = not single or two
        if not single and not getattr(self, "_pair_views_initialized", False):
            self.panel_photo_views = {"A": self._shown_photo_view(), "B": self._shown_photo_view()}
            self.panel_layers = {side: "none" if self._blind_gas_locked else "optical" for side in ("A", "B")}
            for side in ("A", "B"):
                control = self.panel_photo_controls[side]
                control.blockSignals(True)
                control.setCurrentIndex(max(0, control.findData(self.panel_photo_views[side])))
                control.blockSignals(False)
            self._pair_views_initialized = True
        self.two_views.setVisible(single)
        self.two_views.setEnabled(single)
        history_allowed = (single or not self.workspace.blind) and not self._blind_gas_locked
        self.show_review_history.setEnabled(history_allowed)
        if not history_allowed:
            self.show_review_history.setChecked(False)
            self.show_review_history.setToolTip(
                "History reveals earlier labels and model identities. "
                "Open model correction to inspect it outside a blinded A/B comparison."
            )
        self.plot_cards["B"].setVisible(not single)
        self.candidate_cards["B"].setVisible(not single or two)
        self.sync_views.setVisible(not single or two)
        self.sync_views.setChecked(True if split else self.sync_views.isChecked())
        self.sync_views.setEnabled(not split)
        self.hold_overlay_swap.setVisible(not single or two)
        self.fullscreen_buttons["B"].setVisible(not single)
        self.issue_headings["B"].setVisible(not single or two)
        self.issue_help_labels["B"].setVisible(not single or two)
        for option in self.flag_boxes["B"].values():
            option.setVisible(not single or two)
        for button in self.issue_draw_buttons["B"].values():
            button.setVisible(not single or two)
        for side in ("A", "B"):
            self.panel_control_rows[side].setVisible(split)
        self.contours.setVisible(not split and not self._blind_gas_locked)
        for widget in (self.phase_map_only, self.layer_controls_widget, self.opacity_controls_widget,
                       self.layer_label, self.contour_filter,
                       self.physics_layer, self.contour_opacity,
                       self.contour_opacity_label, self.highlight_small):
            widget.setVisible(not split and not self._blind_gas_locked)
        for widget in (self.raw_detail, self.brightness, *self.quick_photo_controls[None].values()):
            widget.setVisible(not split)
        self.show_filled.setVisible(not split and not self._blind_gas_locked)
        self.show_floor_contact.setVisible(not split and not self._blind_gas_locked)
        self.show_l1_neighbourhood.setVisible(not split and not self._blind_gas_locked
                                            and self._l1_neighbourhood_display_available())
        self.hold_overlay_swap.setText(
            "Hold to swap views" if split else "Hold to swap A/B maps"
        )
        self.hold_overlay_swap.setToolTip(
            "Hold to swap both photographs and layers; release to restore."
            if split else
            "Hold to show each candidate's phase map over the other photograph."
        )
        self._update_two_view_controls()

        if single:
            self.plot_headings["A"].setText("Model diagnostic curve")
            self.candidate_headings["A"].setText(
                "Photograph A · blind gas validation" if self._blind_gas_locked
                else "Photograph A" if two else "Model prediction"
            )
            self.candidate_headings["B"].setText(
                "Photograph B · blind gas validation" if self._blind_gas_locked
                else "Photograph B"
            )
            self.issue_headings["A"].setText(
                "Image region interpretation · A" if two and self._blind_gas_locked
                else "Model-region validation"
            )
            self.issue_headings["B"].setText(
                "Image region interpretation · B" if two and self._blind_gas_locked
                else "Same image · view B labels"
            )
            self.fullscreen_buttons["A"].setText("Full-screen model")
            self.issue_draw_buttons["A"]["box"].setText("Draw issue box")
            self.issue_draw_buttons["A"]["freehand"].setText(
                "Draw freehand issue"
            )
            self.issue_draw_buttons["A"]["polygon"].setText("Draw polygon")
            for button in self.issue_draw_buttons["A"].values():
                button.setToolTip(
                    "Select one phase class or physical-issue category first. Keep drawing active "
                    "to mark several areas; choose Pan / inspect when finished."
                )
            self.issue_draw_buttons["A"]["polygon"].setToolTip(
                "Select a category, then left-click at least three corners. Right-click closes the area. Esc cancels.")
            self.annotation_help.setText(
                "Every area stores one human interpretation. Phase selections use "
                "the same vocabulary as blind labeling and are compared with the "
                "phase map automatically; physical and temporal issues stay "
                "separate."
            )
            self.comparison_notice.setText(
                "Inspect one model on the full corrected image. The filled phase "
                "map, full-resolution source, diagnostics, timeline, playback, "
                "reference guides, issue regions, and continuous autosave remain "
                "available. Values are visible-2D optical predictions; no physical "
                "mass is inferred here."
            )
            self.decision_help.setText(
                "Judge the model in the image currently shown. The decision belongs "
                "only to this frame; localize every repair with a box or freehand area."
            )
            self.image_splitter.setSizes([700, 700] if two else [1400, 0])
        else:
            self.plot_headings["A"].setText("Candidate A · diagnostic curve")
            self.plot_headings["B"].setText("Candidate B · diagnostic curve")
            self.candidate_headings["A"].setText("Candidate A")
            self.candidate_headings["B"].setText("Candidate B")
            self.issue_headings["A"].setText("Candidate A issues")
            self.issue_headings["B"].setText("Candidate B issues")
            self.fullscreen_buttons["A"].setText("Full-screen A")
            self.fullscreen_buttons["B"].setText("Full-screen B")
            self.issue_draw_buttons["A"]["box"].setText("Draw A box")
            self.issue_draw_buttons["A"]["freehand"].setText(
                "Draw A freehand"
            )
            self.issue_draw_buttons["B"]["box"].setText("Draw B box")
            self.issue_draw_buttons["B"]["freehand"].setText(
                "Draw B freehand"
            )
            for side in ("A", "B"):
                self.issue_draw_buttons[side]["polygon"].setText(f"Draw {side} polygon")
            self.annotation_help.setText(
                "The candidate records which model exposed the problem. "
                "Supported phase evidence in the source-image region is reusable "
                "when training all later models. Every box or freehand area stores "
                "exactly one category; drawing remains active for several areas."
            )
            self.comparison_notice.setText(
                "Candidate identities are hidden. The two panels always show the "
                "same corrected frame with each candidate's own filled phase map. "
                "No training labels are displayed here; values are visible-2D "
                "optical predictions and do not infer physical mass."
            )
            self.decision_help.setText(
                "Choose which candidate is more physically plausible in the image "
                "currently shown. The decision belongs only to this frame. Use "
                "Unresolved when the optical evidence is insufficient."
            )
            self.image_splitter.setSizes([700, 700])

    def _configure_verdict_options(self, comparison_mode: str) -> None:
        options = (
            (
                ("Not reviewed for this frame", ""),
                ("Acceptable in this frame", "acceptable"),
                ("Needs repair in this frame", "needs_repair"),
                ("Unresolved", "unresolved"),
            )
            if str(comparison_mode) == "single_candidate"
            else (
                ("Not reviewed for this frame", ""),
                ("Candidate A is better in this frame", "candidate_a"),
                ("Equivalent in this frame", "equivalent"),
                ("Unresolved", "unresolved"),
                ("Candidate B is better in this frame", "candidate_b"),
            )
        )
        self.verdict.blockSignals(True)
        try:
            self.verdict.clear()
            for label, value in options:
                self.verdict.addItem(label, value)
            self.verdict.setCurrentIndex(0)
        finally:
            self.verdict.blockSignals(False)

    def _frame_verdict_changed(self, _value: Any = None) -> None:
        if (self.loading_decision or self.current_pair is None
                or self.reference_mode or self._editing_read_only
                or self._blind_gas_locked):
            return
        value = str(self.verdict.currentData() or "")
        if value:
            self.frame_choices[self.frame_index] = value
            self.frame_choice_updated[self.frame_index] = utc_now()
        else:
            self.frame_choices.pop(self.frame_index, None)
            self.frame_choice_updated.pop(self.frame_index, None)
        self._refresh_frame_verdict()
        self._schedule_save()

    def _refresh_frame_verdict(self) -> None:
        value = self.frame_choices.get(self.frame_index, "")
        self.verdict.blockSignals(True)
        try:
            index = self.verdict.findData(value)
            self.verdict.setCurrentIndex(max(0, index))
        finally:
            self.verdict.blockSignals(False)
        reviewed = len(self.frame_choices)
        self.verdict_progress.setText(
            f"{reviewed} / {self.frame_count} frames reviewed"
        )
        self.next_unreviewed.setEnabled(
            self.frame_count > 0 and reviewed < self.frame_count
        )
        self.frame_slider.set_choice_markers(self.frame_choices)

    def _open_next_unreviewed_frame(self) -> None:
        if self.frame_count <= 0:
            return
        for offset in range(1, self.frame_count + 1):
            index = (self.frame_index + offset) % self.frame_count
            if index not in self.frame_choices:
                self._set_frame(index)
                return

    def _install_shortcuts(self) -> None:
        for shortcut, callback in (
            ("Right", lambda: self._step_frame(1)),
            ("Left", lambda: self._step_frame(-1)),
            ("Ctrl+0", self._fit_views),
            ("Ctrl+S", self._save_now),
            ("F11", self._toggle_fullscreen_loop),
        ):
            action = QAction(self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.addAction(action)

    def _set_save_status(self, state: str, text: str) -> None:
        colors = {
            "pending": ("#d6a53a", "#f0ca70"),
            "saving": ("#4d91c4", "#86c3ef"),
            "saved": ("#3fa66b", "#7bd69d"),
            "error": ("#c64f55", "#ef8c91"),
        }
        fill, border = colors.get(state, colors["saved"])
        self.save_status_icon.setStyleSheet(
            f"border:1px solid {border}; border-radius:6px; background:{fill};"
        )
        self.save_status.setText(text)

    def _return_to_start(self) -> None:
        if self.fullscreen_loop_active:
            self._exit_fullscreen_loop()
        else:
            self._stop_speed_loop()
        self._flush_save()
        if self._comparison_local_error:
            return
        self.hide()
        if self.on_back is not None:
            self.on_back()

    def _boxes_across_all_models(self) -> dict[str, int]:
        """Validation boxes saved for each experiment, over every review session on disk.

        The open session is left out here and added by the caller from memory: its own file is
        written on save, so counting it from disk would lag the boxes just drawn.
        """
        from darsia.fluidflower import phase_workflow

        try:
            current = (self.workspace.output_dir / "pairwise_review_decisions.json").resolve()
        except (OSError, ValueError, AttributeError):
            current = None
        try:
            sessions = phase_workflow.discover_phase_review_sessions()
        except (OSError, TypeError, ValueError):
            return {}
        totals: dict[str, int] = {}
        for session in sessions:
            path = Path(str(session.get("decision_path") or ""))
            try:
                if not path.is_file() or (current is not None and path.resolve() == current):
                    continue
                decisions = json.loads(path.read_text(encoding="utf-8-sig")).get("decisions")
            except (OSError, TypeError, ValueError):
                continue
            if not isinstance(decisions, dict):
                continue
            for run, saved in decisions.items():
                if isinstance(saved, dict):
                    key = str(run).casefold()
                    totals[key] = totals.get(key, 0) + len(saved.get("annotations") or [])
        return totals

    @staticmethod
    def _run_number(run: str) -> tuple[int, str]:
        from .phase_review_area_sort import _run_number
        return _run_number(run)

    def _refresh_runs(self, _checked: bool = False) -> None:
        """Explicitly refresh disk state; autosave only updates cached labels."""
        if self._run_refresh_task is not None:
            self._run_refresh_again = True
            return
        self._run_refresh_generation += 1
        task = ComparisonRefreshTask(self._run_refresh_generation,
            self.app_model, self.store, self._boxes_across_all_models)
        self._run_refresh_task = task
        task.signals.finished.connect(self._runs_refreshed)
        QThreadPool.globalInstance().start(task)

    def _runs_refreshed(self, generation, _run, payload, error) -> None:
        if generation != self._run_refresh_generation:
            return
        self._run_refresh_task = None
        if self._comparison_closing:
            return
        if error:
            self.statusBar().showMessage("Could not refresh saved review counts: " + error, 12000)
        else:
            self._run_list_rows = list(payload.get("rows") or [])
            self._run_area_peaks = dict(payload.get("areas") or {})
            self._run_decisions = deepcopy(payload.get("decisions") or {})
            # A refresh that started before a save must not replace its result.
            for run, record in self._comparison_saved_records.items():
                if timestamp_utc(record.get("updated_utc")) >= timestamp_utc(
                        (self._run_decisions.get(run) or {}).get("updated_utc")):
                    self._run_decisions[run] = deepcopy(record)
            self._all_model_box_counts = dict(payload.get("elsewhere") or {})
            self._editing_read_only = bool(payload.get("read_only"))
            self._populate_runs()
            self._apply_read_only_profile()
        if self._run_refresh_again and not self._comparison_closing:
            self._run_refresh_again = False
            self._refresh_runs()

    def _populate_runs(self) -> None:
        selected = self.current_run
        self.run_list.blockSignals(True)
        self.run_list.clear()
        decisions = dict(self._run_decisions)
        for run, (_generation, request) in self._comparison_latest_requests.items():
            decisions[run] = request
        if self.current_run:
            decisions[self.current_run] = dict(decisions.get(self.current_run) or {},
                comparison_mode=str((self.current_pair or {}).get("comparison_mode") or "pair"),
                annotations=self.annotations, frame_choices=[
                    {"frame_index": index, "choice": choice}
                    for index, choice in self.frame_choices.items()])
        history = getattr(self, "_review_history_data", None)
        elsewhere = (
            {run: len(value.get("annotations") or []) for run, value in (history.get("by_run") or {}).items()}
            if history is not None else self._all_model_box_counts
        )
        order = ("number" if self._blind_gas_locked else
                 str(self.run_sort.currentData() or "number") if hasattr(self, "run_sort") else "number")
        area_sort = re.fullmatch(r"peak_(gas|aqueous)_([AB])_(asc|desc)", order)
        entries: list[tuple[str, str, str, int]] = []
        for row in self._run_list_rows:
            run = str(row.get("run") or "").casefold()
            dataset = str(row.get("dataset") or "").upper()
            reviewed = int(row.get("frame_choices_reviewed") or 0)
            total = int(row.get("frame_choices_total") or 0)
            local = decisions.get(run) or {}
            if "frame_choices" in local:
                summary = review.summarize_frame_choices(local["frame_choices"],
                    frame_count=total, comparison_mode=str(local.get("comparison_mode") or "pair"))
                reviewed = int(summary.get("reviewed") or 0)
                complete = bool(summary.get("complete"))
            else:
                complete = bool(row.get("frame_choices_complete"))
            if total:
                status = f"{reviewed}/{total} frames"
                if complete:
                    status += " · complete"
                elif row.get("reviewed") and not reviewed:
                    status += " · legacy run verdict"
            else:
                status = "saved" if row.get("reviewed") else "ready"
            annotation_count = len((decisions.get(run) or {}).get("annotations") or [])
            every_model = elsewhere.get(run, 0) + annotation_count
            if self._blind_gas_locked:
                status = f"{annotation_count} blind image labels"
            elif getattr(self, "_review_history_data", None) is not None:
                historical_count = len((self._review_history_data.get("by_run", {}).get(run) or {}).get("annotations") or [])
                every_model = annotation_count + historical_count
                status += f" · {annotation_count} active boxes · {historical_count} historical"
            else:
                status += f" · {annotation_count} boxes · {every_model} across all models"
            if area_sort:
                phase, side, _direction = area_sort.groups()
                area = (self._run_area_peaks.get(run) or {}).get(side)
                value = None if area is None else area.get(phase)
                phase_name = "free CO2" if phase == "gas" else "aqueous CO2"
                status += (f" · peak {phase_name} {value:.1f}% of porous area"
                           if value is not None else " · peak area unavailable")
            entries.append((run, dataset, status, every_model))
        if order == "boxes":
            entries.sort(key=lambda entry: (-entry[3], self._run_number(entry[0])))
        elif order == "boxes_ascending":
            entries.sort(key=lambda entry: (entry[3], self._run_number(entry[0])))
        elif area_sort:
            phase, side, direction = area_sort.groups()
            entries = sorted_phase_area_runs(entries, self._run_area_peaks,
                side=side, phase=phase, largest_first=direction == "desc")
        else:
            entries.sort(key=lambda entry: self._run_number(entry[0]))
        for run, dataset, status, _every_model in entries:
            item = QListWidgetItem(f"{dataset} · {run.upper()}\n{status}")
            item.setData(Qt.ItemDataRole.UserRole, run)
            self.run_list.addItem(item)
            if run == selected:
                self.run_list.setCurrentItem(item)
        self.run_list.blockSignals(False)
        if not self.current_run and self.run_list.count():
            self.run_list.setCurrentRow(0)

    def _run_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        run = str(current.data(Qt.ItemDataRole.UserRole) or "").casefold()
        if self.current_run and run != self.current_run:
            self._flush_save()
        self._load_run(run)

    def _load_run(self, run: str, *, _loaded=None) -> None:
        if not run:
            return
        if _loaded is None:
            self._comparison_load_generation += 1
            for pending in self._comparison_load_tasks.values():
                pending.cancelled.set()
            task = ComparisonRunLoadTask(self._comparison_load_generation,
                run.casefold(), self.app_model, self.store, self.reviewer.text().strip(),
                (self.store.output_dir if hasattr(self.store, "output_dir") else self.workspace.output_dir) / "comparison_recovery",
                self._comparison_saved_generations.get(run.casefold(), 0))
            self._comparison_load_tasks[task.generation] = task
            task.signals.finished.connect(self._comparison_run_loaded)
            self.statusBar().showMessage(f"Loading {run.upper()} in the backgroundâ€¦")
            self._comparison_load_pool.start(task)
            return
        if self.fullscreen_loop_active:
            self._exit_fullscreen_loop()
        else:
            self._stop_speed_loop()
        self.current_run = run.casefold()
        self.current_pair = _loaded["pair"]
        self.current_pair_object = _loaded["pair_object"]
        self.current_review_root = _loaded["review_root"]
        self._loaded_candidate_assets = _loaded.get("candidate_assets") or {}
        self._loaded_display_modes = _loaded.get("display_modes")
        self._loaded_photo_rows = _loaded.get("photo_rows") or {}
        self._loaded_reference_binding = _loaded.get("reference_binding") or ""
        self._selected_annotation_binding = _loaded.get("annotation_binding")
        raw_available = bool(self.current_pair.get("v5_raw_detail_available"))
        self.raw_detail.setEnabled(raw_available)
        if not raw_available:
            self.raw_detail.blockSignals(True)
            self.raw_detail.setChecked(False)
            self.raw_detail.blockSignals(False)
            self.raw_detail.setText("RAW detail: OFF")
        self._update_brightness_control()
        self._configure_decision_method()
        self.current_decision = deepcopy(_loaded["decision"])
        acknowledged = self._comparison_saved_records.get(self.current_run)
        if acknowledged is not None and (
                (_loaded is not None and self._comparison_saved_generations.get(self.current_run, 0)
                 > _loaded.get("acknowledged_generation", 0))
                or timestamp_utc(acknowledged.get("updated_utc")) >= timestamp_utc(
                    self.current_decision.get("updated_utc"))):
            self.current_decision = deepcopy(acknowledged)
        self._editing_read_only = bool(_loaded.get("read_only"))
        loaded_recovery = False
        recovery = _loaded.get("recovery")
        if (
            isinstance(recovery, Mapping)
            and timestamp_utc(recovery.get("updated_utc"))
            > timestamp_utc(self.current_decision.get("updated_utc"))
        ):
            self.current_decision = merge_review_recovery(
                self.current_decision,
                recovery,
            )
            loaded_recovery = True
        # A local pending draft wins even if an older in-flight NAS write was
        # timestamped later. The acknowledgement marker, not wall time, decides.
        local_draft = self._comparison_local_draft(self.current_run)
        queued = self._comparison_latest_requests.get(self.current_run)
        local_request = queued[1] if queued is not None else local_draft
        if local_request is not None:
            # This is a full snapshot from this session, including deletions.
            # Union merging would resurrect boxes removed while NAS was busy.
            self.current_decision = dict(self.current_decision, **deepcopy(local_request))
            loaded_recovery = True
        candidate_a = self.current_pair["candidate_a"]
        candidate_b = self.current_pair["candidate_b"]
        self.frame_count = min(
            int(candidate_a.get("frame_count") or 0),
            int(candidate_b.get("frame_count") or 0),
        )
        comparison_mode = str(
            self.current_pair.get("comparison_mode") or "pair"
        )
        if comparison_mode == "single_candidate":
            self.frame_count = int(candidate_a.get("frame_count") or 0)
        self.frame_label.set_timeline(
            self.frame_count, candidate_a.get("elapsed_hours") or []
        )
        self._apply_comparison_mode(comparison_mode)
        self._configure_physics_layer()
        self._update_two_view_controls()
        self._configure_verdict_options(comparison_mode)
        raw_frame_choices, remapped_frame_choices = self._remap_frame_choice_indices(
            self.current_decision.get("frame_choices"),
            candidate_a.get("elapsed_hours") or [],
            self.current_pair.get("frame_sources") or [],
        )
        if getattr(self,"_selected_annotation_binding",None) is not None and isinstance(raw_frame_choices,list):
            raw_frame_choices=[dict(item,frame_index=target) for item in raw_frame_choices
                if isinstance(item,Mapping) and (target:=ReviewImageAssets._annotation_display_frame(self,item)) is not None]
        normalized_frame_choices = review.normalize_frame_choices(
            raw_frame_choices,
            frame_count=self.frame_count,
            elapsed_hours=candidate_a.get("elapsed_hours") or [],
            comparison_mode=comparison_mode,
        )
        self.frame_choices = {
            int(item["frame_index"]): str(item["choice"])
            for item in normalized_frame_choices
        }
        self.frame_choice_updated = {
            int(item["frame_index"]): str(item.get("updated_utc") or "")
            for item in normalized_frame_choices
        }
        self._set_issue_draw_side(None)
        loaded_guides = review.normalize_reference_guides(
            self.current_decision.get("reference_guides")
        )
        self.reference_guides = (
            [] if self.workspace.approval_only else loaded_guides
        )
        self._refresh_reference_guide_list()
        self.annotations = [
            deepcopy(dict(item))
            for item in self.current_decision.get("annotations") or []
            if isinstance(item, Mapping)
        ]
        self.loading_decision = True
        try:
            saved_flags = self.current_decision.get("flags") or {}
            annotation_flags = self._annotation_flags(self.annotations)
            valid_flags = {key for key, _label in self.FLAG_LABELS}
            self.legacy_unlocalized_flags = {
                side: (
                    {
                        str(value)
                        for value in saved_flags.get(side) or []
                        if str(value) in valid_flags
                    }
                    - annotation_flags[side]
                )
                for side in ("A", "B")
            }
            for side in ("A", "B"):
                self._clear_issue_category_selection(side)
            self.notes.setPlainText(str(self.current_decision.get("notes") or ""))
        finally:
            self.loading_decision = False
        remapped_annotations = self._remap_annotation_frames(
            self.annotations,
            candidate_a.get("elapsed_hours") or [],
            self.current_pair.get("frame_sources") or [],
        )
        if remapped_annotations:
            self.current_decision["annotations"] = deepcopy(self.annotations)
        self.frame_index = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, self.frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self._refresh_frame_verdict()
        self.fit_next_image = True
        self.base_key = ""
        self.displayed_base_key = ""
        self.base_keys = {"A": "", "B": ""}
        self.displayed_base_keys = {"A": "", "B": ""}
        self._two_view_native_size = None
        self._set_two_view_swap_held(False)
        self.overlay_keys = {"A": "", "B": ""}
        self.overlay_swap_held = False
        self.hold_overlay_swap.setText(
            "Hold to swap views" if self._split_views_active() else "Hold to swap A/B maps"
        )
        self.phase_map_only.blockSignals(True)
        self.phase_map_only.setChecked(False)
        self.phase_map_only.blockSignals(False)
        self.phase_map_only.setText("Black background: OFF")
        reference_ready = bool(
            self.current_pair.get("reference_ready")
            and candidate_a.get("reference_frames_ready")
            and candidate_b.get("reference_frames_ready")
        )
        self.reference.setEnabled(reference_ready)
        self.reference.setToolTip(
            "Show the exact saved pre-injection source and its own existing map, in its own native geometry. "
            "Click again to return to the selected timeline image; no reference inference is started."
            if reference_ready and self.current_pair.get("rig_saved_reference_index") is not None else
            "Optional alignment check: temporarily replace the selected timeline image "
            "with the corrected pre-injection reference. The model-evaluated baseline "
            "is already Frame 1 in the timeline."
            if reference_ready
            else "The aligned pre-injection reference is not available for this experiment."
        )
        self.reference.blockSignals(True)
        self.reference.setChecked(False)
        self.reference.blockSignals(False)
        self.reference_mode = False
        self.reference.setText("Pre-injection reference")
        self.active_injection_reference = None
        self._configure_injection_references()
        self.contours.setEnabled(True)
        self.verdict.setEnabled(not self._editing_read_only)
        if self._blind_gas_session:
            self._apply_blind_gas_profile()
        self.plot_keys = {"A": "", "B": ""}
        for viewer in self.viewers.values():
            viewer.set_overlay(None)
            viewer.clear_vectors()
        for plot in self.plot_labels.values():
            plot.clear_image()
        self._request_plots()
        self._request_frame()
        self._refresh_issue_annotations()
        self._refresh_review_history()
        self._set_save_status(
            "pending" if loaded_recovery else "saved",
            (
                "Newer recovery loaded · promoting to durable file..."
                if loaded_recovery
                else "Loaded · autosave ready"
            ),
        )
        if remapped_frame_choices:
            self.current_decision["frame_choices"] = raw_frame_choices
        if remapped_annotations or remapped_frame_choices or loaded_recovery:
            QTimer.singleShot(0, self._schedule_save)
        if self._editing_read_only:
            self._apply_read_only_profile()

    def _comparison_run_loaded(self, generation, run, payload, error) -> None:
        self._comparison_load_tasks.pop(generation, None)
        if generation != self._comparison_load_generation or self._comparison_closing:
            return
        if error:
            self.statusBar().showMessage("Could not load experiment: " + error, 12000)
            return
        self._flush_save()
        self._load_run(run, _loaded=payload)

    @staticmethod
    def _remap_frame_choice_indices(
        raw_choices: Any,
        elapsed_hours: Sequence[float],
        frame_sources: Sequence[Mapping[str, Any]],
        *,
        tolerance_h: float = 0.05,
    ) -> tuple[Any, int]:
        """Keep saved verdicts attached to time when a baseline frame is inserted."""

        if not isinstance(raw_choices, list):
            return raw_choices, 0
        if frame_sources and all(row.get("source_label_identity") for row in frame_sources):
            return raw_choices, 0
        times = [float(value) for value in elapsed_hours]
        if not times:
            return raw_choices, 0
        baseline_indices = {
            index
            for index, row in enumerate(frame_sources)
            if str(row.get("selection_status") or "")
            == "pre_injection_negative_control"
        }
        ordinary_indices = [
            index for index in range(len(times)) if index not in baseline_indices
        ]
        remapped = 0
        result: list[Any] = []
        for raw in raw_choices:
            if not isinstance(raw, Mapping):
                result.append(raw)
                continue
            item = deepcopy(dict(raw))
            try:
                elapsed_h = float(item["elapsed_h"])
                previous = int(item.get("frame_index", -1))
            except (KeyError, TypeError, ValueError):
                result.append(item)
                continue
            if not math.isfinite(elapsed_h):
                result.append(item)
                continue
            if (
                previous in baseline_indices
                and previous < len(times)
                and math.isclose(
                    times[previous], elapsed_h, rel_tol=0.0, abs_tol=1e-9
                )
            ):
                result.append(item)
                continue
            candidates = ordinary_indices or list(range(len(times)))
            nearest = min(candidates, key=lambda index: abs(times[index] - elapsed_h))
            if abs(times[nearest] - elapsed_h) > float(tolerance_h):
                result.append(item)
                continue
            if nearest != previous:
                item.setdefault("original_frame_index", previous)
                item["frame_index"] = nearest
                remapped += 1
            result.append(item)
        return result, remapped

    @staticmethod
    def _remap_annotation_frames(
        annotations: Sequence[dict[str, Any]],
        elapsed_hours: Sequence[float],
        frame_sources: Sequence[Mapping[str, Any]] = (),
        *,
        tolerance_h: float = 0.05,
    ) -> int:
        if frame_sources and all(row.get("source_label_identity") for row in frame_sources):
            return 0
        times = [float(value) for value in elapsed_hours]
        if not times:
            return 0
        baseline_indices = {
            index
            for index, row in enumerate(frame_sources)
            if str(row.get("selection_status") or "")
            == "pre_injection_negative_control"
        }
        ordinary_indices = [
            index for index in range(len(times)) if index not in baseline_indices
        ]
        remapped = 0
        for annotation in annotations:
            try:
                elapsed_h = float(annotation["elapsed_h"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(elapsed_h):
                continue
            previous = int(annotation.get("frame_index") or 0)
            if (
                previous in baseline_indices
                and previous < len(times)
                and math.isclose(
                    times[previous], elapsed_h, rel_tol=0.0, abs_tol=1e-9
                )
            ):
                continue
            candidates = ordinary_indices or list(range(len(times)))
            nearest = min(candidates, key=lambda index: abs(times[index] - elapsed_h))
            if abs(times[nearest] - elapsed_h) > float(tolerance_h):
                continue
            if previous == nearest:
                continue
            annotation.setdefault("original_frame_index", previous)
            annotation["frame_index"] = nearest
            remapped += 1
        return remapped

    def _frame_path(self, index: int) -> Path:
        assert self.current_review_root is not None
        return review._native_frame_path(
            self.current_review_root,
            self.current_run,
            index,
        )

    def _reference_path(self) -> Path:
        assert self.current_review_root is not None
        return review._native_reference_path(self.current_review_root, self.current_run)

    def _reference_offset(self, side: str) -> tuple[float, float]:
        if self.current_pair is None:
            return 0.0, 0.0
        candidate = self.current_pair[
            "candidate_a" if side == "A" else "candidate_b"
        ]
        offsets = candidate.get("reference_offsets") or []
        if self.frame_index >= len(offsets) or not isinstance(
            offsets[self.frame_index], Mapping
        ):
            return 0.0, 0.0
        row = offsets[self.frame_index]
        return float(row.get("x") or 0.0), float(row.get("y") or 0.0)

    def _active_reference_offset(self, side: str) -> tuple[float, float]:
        if self._saved_rig_reference_index() is not None:
            return 0.0, 0.0
        target = self._reference_offset(side)
        if self.active_injection_reference is None:
            return target
        row = self.injection_references[self.active_injection_reference]
        source = row.get("reference_offset")
        if source is None:
            candidate = (self.current_pair or {}).get(
                "candidate_a" if side == "A" else "candidate_b", {})
            offsets = candidate.get("reference_offsets") or []
            index = row.get("frame_index")
            source = offsets[index] if isinstance(index, int) and 0 <= index < len(offsets) else {}
        return target[0] - float(source.get("x") or 0), target[1] - float(source.get("y") or 0)

    def _bottom_prediction_withheld(self, side: str) -> bool:
        pair = getattr(self, "current_pair_object", None)
        if pair is None:
            return False
        details = getattr(self, "_loaded_candidate_assets", {}).get(side)
        if details is not None:
            return bool(details["prediction_withheld"])
        root = getattr(self, "current_review_root", None) or self.store.output_dir
        return review._candidate_prediction_withheld(root, pair.candidate(side))

    def _overlay_path(self, side: str, index: int, layer: str | None = None) -> Path:
        assert self.current_review_root is not None
        candidate = self.current_pair_object.candidate(side)
        path, _rendering = review._preferred_phase_overlay_path(
            self.current_review_root,
            candidate,
            side,
            index,
            overlay_asset_layer(str(layer or self.contour_filter.currentData() or "all")),
        )
        return path

    def _model_phase_in_region(
        self,
        side: str,
        rect: QRectF,
        points: Sequence[QPointF] = (),
    ) -> dict[str, Any]:
        """Summarize the candidate's filled phase map inside one review area."""

        if getattr(self, "_blind_gas_locked", False) or self._bottom_prediction_withheld(side):
            raise RuntimeError("Model scoring is locked during blind gas validation")

        image, source = _region_phase_image(self, side, self.frame_index)
        if image.isNull() or image.width() <= 0 or image.height() <= 0:
            return {
                "model_phase_class": "unscored",
                "model_phase_distribution": {},
                "model_phase_sample_count": 0,
                "model_phase_source": str(source),
            }

        phase_image = image
        width, height = phase_image.width(), phase_image.height()
        left = max(0, min(width - 1, int(math.floor(rect.left() * width))))
        right = max(left + 1, min(width, int(math.ceil(rect.right() * width))))
        top = max(0, min(height - 1, int(math.floor(rect.top() * height))))
        bottom = max(top + 1, min(height, int(math.ceil(rect.bottom() * height))))
        sample_step = max(
            1,
            int(math.sqrt(max(1, (right - left) * (bottom - top)) / 4096.0)),
        )
        polygon = QPainterPath()
        if len(points) >= 3:
            polygon.moveTo(points[0])
            for point in points[1:]:
                polygon.lineTo(point)
            polygon.closeSubpath()

        palette = {
            "aqueous": review.PHASE_MAP_RGB[0],
            "mixed": review.PHASE_MAP_RGB[1],
            "gas": review.PHASE_MAP_RGB[2],
        }
        counts = {"water": 0, "aqueous": 0, "mixed": 0, "gas": 0}
        for y in range(top, bottom, sample_step):
            normalized_y = (y + 0.5) / height
            for x in range(left, right, sample_step):
                normalized_x = (x + 0.5) / width
                if len(points) >= 3 and not polygon.contains(
                    QPointF(normalized_x, normalized_y)
                ):
                    continue
                color = phase_image.pixelColor(x, y)
                if color.alpha() <= 8 or max(color.red(), color.green(), color.blue()) <= 8:
                    counts["water"] += 1
                    continue
                rgb = (color.red(), color.green(), color.blue())
                phase = min(
                    palette,
                    key=lambda name: sum(
                        (rgb[channel] - palette[name][channel]) ** 2
                        for channel in range(3)
                    ),
                )
                counts[phase] += 1
        total = sum(counts.values())
        fractions = (
            {name: value / total for name, value in counts.items()} if total else {}
        )
        if not total:
            model_phase = "unscored"
        elif counts["mixed"] / total >= 0.2 or (
            counts["aqueous"] / total >= 0.15 and counts["gas"] / total >= 0.15
        ):
            model_phase = "mixed"
        else:
            model_phase = max(counts, key=counts.get)
        return {
            "model_phase_class": model_phase,
            "model_phase_distribution": fractions,
            "model_phase_sample_count": total,
            "model_phase_source": str(source),
        }

    def _display_overlay(self, key: str, image: QImage) -> QImage | None:
        # Filtering is exclusively for the scene. Cached full maps remain
        # available to region scoring and all saved assets are unchanged.
        image = BlindModelComparisonWindow._rig_photo_overlay(self, image)
        layer = (str(self.contour_filter.currentData() or "all")
                 if BlindModelComparisonWindow._physics_layer_mode(self) == "optical" else "all")
        from .phase_studio_display_cache import prepared_overlay
        control = getattr(self, "show_filled", None)
        phase_control = getattr(self, "contours", None)
        cache = getattr(self, "image_cache", None)
        options = dict(layer=layer, filled=control is None or control.isChecked(),
                                  phase_visible=phase_control is None or phase_control.isChecked(),
                                  floor_contact=BlindModelComparisonWindow._floor_contact_display_enabled(self),
                                  l1_neighbourhood=BlindModelComparisonWindow._l1_neighbourhood_display_enabled(self))
        if callable(getattr(cache, "request_display", None)):
            return cache.request_display(key, image, **options)
        result = prepared_overlay(image, **options)
        if callable(getattr(cache, "note_image_modified", None)):
            cache.note_image_modified(key)
        return result

    def _rig_photo_overlay(self, image: QImage) -> QImage:
        from darsia.fluidflower.phase_rig_display_canvas import validate_rig_display_overlay
        return validate_rig_display_overlay(image, BlindModelComparisonWindow._rig_display_revision(self))

    def _rig_display_revision(self):
        rows = (getattr(self, "current_pair", None) or {}).get("frame_sources") or []
        index = getattr(self, "frame_index", -1)
        return rows[index].get("rig_domain_revision") if 0 <= index < len(rows) else None

    def _rig_photo_image(self, image: QImage) -> QImage:
        from darsia.fluidflower.phase_rig_display_canvas import pad_rig_photograph
        return pad_rig_photograph(image, self._rig_display_revision())

    def _filled_display_enabled(self, side=None):
        control = ((getattr(self, "panel_filled_controls", {}) or {}).get(side)
                   if side is not None else getattr(self, "show_filled", None))
        return control is None or control.isChecked()

    def _filled_display_available(self):
        rows = (self.current_pair or {}).get("frame_sources") or []
        return (not self.reference_mode and 0 <= self.frame_index < len(rows)
                and bool(rows[self.frame_index].get("rig_domain_revision")))

    def _floor_contact_display_enabled(self, side=None):
        control = ((getattr(self, "panel_floor_contact_controls", {}) or {}).get(side)
                   if side is not None else getattr(self, "show_floor_contact", None))
        return control is None or control.isChecked()

    def _p1_final3_overlay_available(self, side=None):
        return self._saved_revision_overlay_available(side, "p1_final3_revision")

    def _p1_final2_overlay_available(self, side=None):
        return self._saved_revision_overlay_available(side, "p1_final2_revision")

    def _p1_final_overlay_available(self, side=None):
        return self._saved_revision_overlay_available(side, "p1_final_revision")

    def _saved_revision_overlay_available(self, side, revision_key):
        if getattr(self, "_blind_gas_locked", False):
            return False
        pair = getattr(self, "current_pair_object", None)
        if pair is None:
            return False
        source_side = "A" if side is None or self._two_views_active() else side
        candidate = pair.candidate(source_side)
        return bool((getattr(candidate, "quality_flags", None) or {}).get(revision_key))

    def _l1_neighbourhood_display_enabled(self, side=None):
        control = ((getattr(self, "panel_l1_neighbourhood_controls", {}) or {}).get(side)
                   if side is not None else getattr(self, "show_l1_neighbourhood", None))
        return control is None or control.isChecked()

    def _l1_neighbourhood_display_available(self, side=None):
        if getattr(self, "reference_mode", False) or getattr(self, "_blind_gas_locked", False):
            return False
        pair = getattr(self, "current_pair_object", None)
        if pair is not None:
            source_side = "A" if side is None or self._two_views_active() else side
            candidate = pair.candidate(source_side)
            if (getattr(candidate, "quality_flags", None) or {}).get("l1_neighbourhood_rule") is True:
                return not self._bottom_prediction_withheld(source_side)
            return False
        rows = (getattr(self, "current_pair", None) or {}).get("frame_sources") or []
        index = getattr(self, "frame_index", -1)
        return 0 <= index < len(rows) and bool(rows[index].get("l1_neighbourhood_gas"))

    def _filled_display_toggled(self, side=None):
        self._set_two_view_swap_held(False)
        if self._split_views_active():
            self._request_two_view_overlays(side)
        else:
            self._request_overlays()

    def _physics_layer_mode(self) -> str:
        if getattr(self, "_blind_gas_locked", False):
            return "none"
        control = getattr(self, "physics_layer", None)
        return str(control.currentData() or "optical") if control is not None else "optical"

    def _configure_physics_layer(self) -> None:
        """Offer v6 only for its exact source candidate in model review."""
        if self._blind_gas_locked:
            self.physics_layer.setEnabled(False)
            self.physics_layer.setVisible(False)
            self.physics_header.setVisible(False)
            self._physics_layer_available = False
            return
        visible = self._playback_candidate_sides()
        available = any(
            _is_track1_physics_candidate(
                self.current_pair_object.candidate(side).candidate_id, self.current_run,
            )
            for side in visible
        )
        self.physics_layer.setVisible(available and not self._split_views_active())
        self.physics_layer.setEnabled(available)
        self.physics_header.setVisible(available)
        if available:
            try:
                payload = json.loads((PHYSICS_LAYER_EXPORT_ROOT / "manifest.json").read_text(encoding="utf-8"))
                limitations = payload.get("known_limitations") or []
                tooltip = "Known limitations:\n" + "\n".join(f"â€¢ {item}" for item in limitations)
            except (OSError, ValueError, TypeError) as exc:
                tooltip = f"Physics export unavailable: {exc}"
            self.physics_header.setToolTip(tooltip)
            self.physics_layer.setToolTip(
                "Choose optical, physics-corrected, or difference display. "
                "No model or saved map is changed.\n" + DIFFERENCE_LEGEND + "\n" + tooltip
            )
        if available != self._physics_layer_available:
            self.physics_layer.blockSignals(True)
            self.physics_layer.setCurrentIndex(1 if available else 0)
            self.physics_layer.blockSignals(False)
        self._physics_layer_available = available

    def _physics_overlay_asset(self, side: str, index: int, mode: str, phase: str) -> DeferredImage:
        """Render a verified read-only mask beside the stored optical map."""
        if self._bottom_prediction_withheld(side):
            raise PermissionError("New physics layers are withheld for the bottom blind check")
        candidate = self.current_pair_object.candidate(side)
        review_root, run = self.current_review_root, self.current_run
        reference = self.reference_mode
        fingerprint = review._candidate_fingerprint(candidate)
        # A changed or removed export must not be concealed by a previously
        # cached QImage. A new timestamp/size signature gets a new cache key;
        # the worker then rechecks the complete SHA chain before displaying it.
        export_files = (
            PHYSICS_LAYER_EXPORT_ROOT / "manifest.json",
            PHYSICS_LAYER_EXPORT_ROOT / run / "manifest.json",
            PHYSICS_LAYER_EXPORT_ROOT / run / f"frame_{index:03d}.physics_gas_mask.npz",
        )
        export_signature = []
        for path in export_files:
            try:
                stat = path.stat()
                export_signature.append(f"{stat.st_size}:{stat.st_mtime_ns}")
            except OSError:
                export_signature.append("missing")

        def resolve() -> QImage:
            if reference:
                raise ValueError("Physics mask is unavailable for the reference image; choose Optical or a timeline frame")
            if not _is_track1_physics_candidate(candidate.candidate_id, run):
                raise ValueError(f"No gas physics v6 export for candidate {candidate.candidate_id}")
            from darsia.fluidflower.phase_physics_layer import (
                load_verified_physics_frame, render_physics_overlay,
            )
            map_root = review._candidate_phase_map_root(review_root, candidate)
            model_root = next(
                (parent for parent in (map_root, *map_root.parents)
                 if (parent / "inference_run_bindings" / f"{run}.json").is_file()),
                None,
            )
            if model_root is None:
                raise ValueError(f"Cannot locate the source-bound model result for {run.upper()}")
            frame = load_verified_physics_frame(
                PHYSICS_LAYER_EXPORT_ROOT, model_root, review_root, run, index,
            )
            optical_path, _ = review._preferred_phase_overlay_path(
                review_root, candidate, side, index, "all",
            )
            optical_image = QImage(str(self.image_cache.disk.stage(optical_path)))
            if optical_image.isNull():
                raise OSError(f"Could not decode stored optical phase map: {optical_path}")
            optical_image = normalize_phase_overlay_alpha(optical_image).convertToFormat(
                QImage.Format.Format_RGBA8888
            )
            rows = np.frombuffer(optical_image.bits(), dtype=np.uint8).reshape(
                optical_image.height(), optical_image.bytesPerLine()
            )
            optical_rgba = rows[:, : optical_image.width() * 4].reshape(
                optical_image.height(), optical_image.width(), 4
            )
            rgba = render_physics_overlay(optical_rgba, frame, layer=mode)
            if phase != "all" and mode == "physics":
                from darsia.fluidflower.phase_rendering import AQUEOUS_RGBA, GAS_RGBA
                rgb = (GAS_RGBA[:3] if phase == "gas" else
                       AQUEOUS_RGBA[:3] if phase == "aqueous" else UNRESOLVED_RGB)
                rgba = rgba.copy()
                rgba[..., 3][np.any(rgba[..., :3] != rgb, axis=-1)] = 0
            rgba = np.ascontiguousarray(rgba)
            return QImage(
                rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                QImage.Format.Format_RGBA8888,
            ).copy()

        return DeferredImage(
            f"physics-v6:{review_root}:{run}:{candidate.candidate_id}:{fingerprint}:{index}:{mode}:{phase}:{reference}:{':'.join(export_signature)}",
            resolve,
            display_domain=self._photo_sand_loader(index) if not reference else None,
        )

    def _playback_candidate_sides(self) -> tuple[str, ...]:
        side = self.fullscreen_candidate_side
        if self.fullscreen_loop_active and side in {"A", "B"}:
            return (side,)
        if getattr(self, "comparison_mode", "pair") == "single_candidate":
            return ("A",)
        return ("A", "B")

    def _open_mass_analysis(self) -> None:
        if self._blind_gas_locked:
            return
        from darsia.fluidflower.tools.phase_studio_mass_ui import open_for_review
        open_for_review(self)

    def _local_search_toggled(self, enabled: bool) -> None:
        self.local_reference_search.setText("Local baseline search: " + ("ON" if enabled else "OFF"))
        self._request_overlays()

    def _configure_decision_method(self) -> None:
        from darsia.fluidflower.phase_review_basis import uses_optical, is_regional
        if all(is_regional(self.current_pair_object.candidate(side)) for side in self._playback_candidate_sides()):
            control = self.local_reference_search
            control.blockSignals(True)
            control.setChecked(False)
            control.blockSignals(False)
            control.setEnabled(False)
            control.setText("Trained color regions")
            control.setToolTip("Saved trained prediction from baseline-relative color change and connected regions. Select Raw optical prediction to compare the inherited pixel model.")
            self.mass_analysis_button.setText("Concentration + mass")
            self.mass_analysis_button.setToolTip("Uses this candidate's saved phase maps; concentration and mass are separate from phase review.")
            return
        if any(uses_optical(self.current_pair_object.candidate(side)) for side in ("A", "B")):
            control = self.local_reference_search
            control.blockSignals(True)
            control.setChecked(False)
            control.blockSignals(False)
            control.setEnabled(False)
            control.setText("Context additions: OFF")
            control.setToolTip("Studio reviews the archived optical prediction. Context and local-search additions are excluded.")
            self.mass_analysis_button.setText("Concentration + mass · archived model")
            self.mass_analysis_button.setToolTip("This calculation uses the archived model's fixed supported phase maps, not the optical review layer or cleanup preview.")
            return
        unified = any(getattr(self.current_pair_object.candidate(side), "phase_support_format", "")
                      in {"unified-phase-decision-v1", "unified-phase-decision-local-age-v2"} for side in ("A", "B"))
        control = self.local_reference_search
        if unified:
            control.blockSignals(True)
            control.setChecked(False)
            control.blockSignals(False)
            control.setEnabled(False)
            control.setText("Unified decision · reference search included")
            control.setToolTip("The selected phase method includes reference evidence and contour regularization. No extra phase-changing postprocessing is applied.")
        else:
            control.setEnabled(True)
            control.setText("Local baseline search: " + ("ON" if control.isChecked() else "OFF"))
            control.setToolTip("Optional legacy reference-search comparison. The selected model version remains unchanged.")

    @staticmethod
    def _overlay_source_side(panel_side: str, swap_held: bool) -> str:
        if panel_side not in {"A", "B"}:
            raise ValueError(f"Unknown candidate side: {panel_side}")
        if not swap_held:
            return panel_side
        return "B" if panel_side == "A" else "A"

    def _request_plots(self) -> None:
        if self._blind_gas_locked or self.current_pair_object is None or not self.show_plots.isChecked():
            return
        for side in self._playback_candidate_sides():
            candidate = self.current_pair_object.candidate(side)
            if self._bottom_prediction_withheld(side):
                self.plot_keys[side] = ""
                self.plot_labels[side].setText("Bottom-strip blind check · diagnostics withheld")
                continue
            source = candidate.mass_plot
            key = (str(candidate.summary_path), "plot")
            if not source.is_file() or (getattr(candidate, "presentation_deferred", False)
                                       and key not in self.presentation_validated):
                self.plot_labels[side].setText(self.presentation_errors.get(key) or "Preparing area diagnosticsâ€¦")
                if key not in self.presentation_errors:
                    self._start_presentation(candidate, "plot")
                continue
            image = self.image_cache.get(source)
            key = self.image_cache.key(source)
            self.plot_keys[side] = key
            if image is not None:
                self.plot_labels[side].set_image(image)
            else:
                self.image_cache.request(source)

    def _diagnostics_toggled(self, visible):
        if self._blind_gas_locked:
            self.plot_container.hide()
            return
        if visible:
            # Explicitly reopening diagnostics retries a failed on-demand build.
            self.presentation_errors = {key: value for key, value in self.presentation_errors.items() if key[1] != "plot"}
            self._request_plots()

    def _start_presentation(self, candidate, kind):
        if review._candidate_prediction_withheld(self.current_review_root or self.store.output_dir, candidate):
            return
        key = (str(candidate.summary_path), kind)
        if key in self.presentation_tasks:
            return
        task = PresentationTask(candidate, kind)
        task.signals.progress.connect(self._presentation_progress)
        task.signals.finished.connect(self._presentation_finished)
        self.presentation_tasks[key] = task
        self.presentation_pool.start(task)

    def _presentation_progress(self, key, message):
        self.statusBar().showMessage(message)

    def _presentation_finished(self, key, path, error):
        self.presentation_tasks.pop(key, None)
        if error:
            self.presentation_errors[key] = error
            self.statusBar().showMessage(error, 15000)
        else:
            self.presentation_validated.add(key)
            if key[1] == "video":
                self.statusBar().showMessage("Video exported: " + path, 15000)
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))
            else:
                self.statusBar().showMessage("Area diagnostics ready", 4000)
        self._request_plots()

    def _export_video(self):
        if self._blind_gas_locked:
            return
        if self.current_pair_object is None:
            return
        for side in self._playback_candidate_sides():
            self._start_presentation(self.current_pair_object.candidate(side), "video")

    def _two_view_photo_asset(self, side: str, index: int | None = None) -> DeferredImage:
        if self.reference_mode and (self.active_injection_reference is not None
                                    or self._saved_rig_reference_index() is not None):
            return self._active_reference_asset(view=self.panel_photo_views[side])
        view = self.panel_photo_views[side]
        if view == "raw_detail" and not (self.current_pair or {}).get("v5_raw_detail_available"):
            raise ValueError("RAW detail is unavailable for this run")
        elif view not in {"raw_detail", "original"} and view not in self._brightness_modes():
            raise ValueError(f"Photographic view {view!r} is unavailable for this run")
        return self._photo_view_asset(None if self.reference_mode else
                                     self.frame_index if index is None else index, view,
                                     reference=self.reference_mode)

    def _two_view_ready(self) -> bool:
        if not self._split_views_active() or (self.reference_mode and self.active_injection_reference is not None
                                            and self._saved_rig_reference_index() is None):
            return False
        for side in ("A", "B"):
            if (not self.base_keys[side]
                    or self.displayed_base_keys[side] != self.base_keys[side]
                    or self.viewers[side].base_item.pixmap().isNull()):
                return False
            if self._effective_panel_layer(side) != "none" and (
                    not self.overlay_keys[side]
                    or self.viewers[side].overlay_item.pixmap().isNull()):
                return False
        a = self.viewers["A"].base_item.pixmap()
        b = self.viewers["B"].base_item.pixmap()
        return (a.width(), a.height()) == (b.width(), b.height())

    def _update_two_view_swap_button(self) -> None:
        if self._split_views_active():
            self.hold_overlay_swap.setEnabled(self._two_view_ready())

    def _set_two_view_swap_held(self, held: bool) -> None:
        active = bool(held and self._two_view_ready())
        if active == self._two_view_swap_held:
            return
        if active:
            self._two_view_snapshot = {
                side: (
                    self.viewers[side].base_item.pixmap(),
                    self.viewers[side].overlay_item.pixmap(),
                    self.viewers[side].overlay_item.opacity(),
                    self.asset_labels[side].text(),
                )
                for side in ("A", "B")
            }
        for side, source in (("A", "B"), ("B", "A")):
            state = self._two_view_snapshot.get(source if active else side)
            if state is None:
                continue
            base, overlay, opacity, text = state
            viewer = self.viewers[side]
            viewer.base_item.setPixmap(base)
            viewer.overlay_item.setPixmap(overlay)
            viewer.overlay_item.setOpacity(opacity)
            self.asset_labels[side].setText(text)
        self._two_view_swap_held = active
        self.hold_overlay_swap.setText(
            "Release to restore views" if active else "Hold to swap views"
        )
        if not active:
            self._two_view_snapshot = {}

    def _request_two_view_frame(self, only_side: str | None = None) -> None:
        self._set_two_view_swap_held(False)
        sides = (only_side,) if only_side in {"A", "B"} else ("A", "B")
        for side in sides:
            try:
                asset = self._two_view_photo_asset(side)
            except ValueError as exc:
                self.base_keys[side] = ""
                self.displayed_base_keys[side] = ""
                self.overlay_keys[side] = ""
                self.viewers[side].base_item.setPixmap(QPixmap())
                self.viewers[side].set_overlay(None)
                self.asset_labels[side].setText(str(exc))
                continue
            key = self.image_cache.key(asset)
            if key != self.base_keys[side]:
                self.viewers[side].cancel_current_draw()
            self.base_keys[side] = key
            if side == "A":
                self.base_key = key
            image = self.image_cache.get(asset)
            if image is None:
                self.displayed_base_keys[side] = ""
                if side == "A":
                    self.displayed_base_key = ""
                self.viewers[side].base_item.setPixmap(QPixmap())
                self.viewers[side].set_overlay(None)
                self.viewers[side].set_peer_cursor(None)
                self.asset_labels[side].setText("Loading full-resolution frame...")
                self.image_cache.request(asset)
            else:
                self.displayed_base_keys[side] = key
                if side == "A":
                    self.displayed_base_key = key
                self._set_two_view_base_image(side, image)
            self._request_two_view_overlays(side)
        self._update_two_view_swap_button()

    def _set_two_view_base_image(self, side: str, image: QImage) -> None:
        image = self._rig_photo_image(image)
        other_side = "B" if side == "A" else "A"
        other = self.viewers[other_side].base_item.pixmap()
        if not other.isNull() and (image.width(), image.height()) != (other.width(), other.height()):
            self.viewers[side].base_item.setPixmap(QPixmap())
            self.displayed_base_keys[side] = ""
            self.statusBar().showMessage(
                "Photographic views have different native pixel dimensions", 10000
            )
            return
        self.syncing_views = True
        try:
            viewer = self.viewers[side]
            viewer.set_image(image, fit=self.fit_next_image)
            source_side = "A" if self._two_views_active() else side
            offset = self._active_reference_offset(source_side) if self.reference_mode else (0.0, 0.0)
            viewer.set_base_offset(*offset)
            viewer.set_overlay_offset(*offset)
            viewer.set_phase_map_only(False)
        finally:
            self.syncing_views = False
        self.asset_labels[side].setText(
            f"Full resolution · {self.panel_photo_controls[side].currentText()}"
        )
        self._render_injection_points()
        if all(self.displayed_base_keys[value] == self.base_keys[value]
               and self.base_keys[value] for value in ("A", "B")):
            self.fit_next_image = False
            self._sync_view("A")
        self._update_two_view_swap_button()

    def _two_view_overlay_image(self, side: str, image: QImage) -> QImage | None:
        image = self._rig_photo_overlay(image)
        from .phase_studio_display_cache import prepared_overlay
        # Physics/difference layers already have their own display colours;
        # context_* preserves their historical unfiltered rendering.
        layer = "all" if self.panel_layers[side] == "optical" else "__unfiltered__"
        options = dict(layer=layer, filled=self._filled_display_enabled(side),
                                  phase_visible=self._effective_panel_layer(side) != "none",
                                  floor_contact=self._floor_contact_display_enabled(side),
                                  l1_neighbourhood=self._l1_neighbourhood_display_enabled(side))
        if callable(getattr(self.image_cache, "request_display", None)):
            return self.image_cache.request_display(self.overlay_keys[side], image, **options)
        result = prepared_overlay(image, **options)
        if callable(getattr(self.image_cache, "note_image_modified", None)):
            self.image_cache.note_image_modified(self.overlay_keys[side])
        return result

    def _request_two_view_overlays(self, only_side: str | None = None) -> None:
        sides = (only_side,) if only_side in {"A", "B"} else ("A", "B")
        for side in sides:
            layer = self._effective_panel_layer(side)
            viewer = self.viewers[side]
            source_side = "A" if self._two_views_active() else side
            if self._bottom_prediction_withheld(source_side):
                self.overlay_keys[side] = ""
                viewer.set_overlay(None)
                viewer.set_phase_map_only(False)
                self.asset_labels[side].setText("Photograph only · bottom-strip blind labels pending")
                continue
            viewer.set_overlay_opacity(self.panel_opacities[side] / 100.0)
            viewer.set_small_detection_highlight(False)
            viewer.set_small_detection_components(())
            filled_only = (layer == "none" and self._filled_display_enabled(side)
                           and self._filled_display_available())
            if (layer == "none" and not filled_only) or self.current_pair is None:
                self.overlay_keys[side] = ""
                viewer.set_overlay(None)
                continue
            if layer in {"physics", "difference"} and not self._physics_layer_available:
                self.overlay_keys[side] = ""
                viewer.set_overlay(None)
                self.asset_labels[side].setText("Physics layer unavailable for this run")
                continue
            source_side = "A" if self._two_views_active() else side
            asset_layer = layer if layer in SAVED_PROVENANCE_LAYERS else "all"
            source = (
                self._active_reference_overlay_asset(source_side, asset_layer) if self.reference_mode
                else self._overlay_asset(source_side, self.frame_index, asset_layer)
            ) if layer in (SAVED_PROVENANCE_LAYERS | {"optical"}) or filled_only else self._physics_overlay_asset(
                source_side, self.frame_index, layer, "all"
            )
            key = self.image_cache.key(source)
            self.overlay_keys[side] = key
            image = self.image_cache.get(source)
            if image is not None and self.displayed_base_keys[side] == self.base_keys[side]:
                viewer.set_overlay(self._two_view_overlay_image(side, image))
            else:
                viewer.set_overlay(None)
                self.image_cache.request(source)
        self._update_two_view_swap_button()

    def _request_frame(self) -> None:
        if self._blind_gas_session or self._split_views_active():
            self._set_space_held(False)
        if self.current_pair is None or self.frame_count <= 0:
            return
        self.frame_index = max(
            0,
            min(self.frame_count - 1, int(self.frame_index)),
        )
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.frame_index)
        self.frame_slider.blockSignals(False)
        elapsed = self._elapsed_for("A", self.frame_index)
        self.frame_label.set_frame(
            self.frame_index, elapsed, reference=self.reference_mode,
            baseline=self._is_pre_injection_timeline_frame(self.frame_index),
            reference_text=self._injection_reference_status(),
        )
        self._refresh_frame_verdict()
        sources = self.current_pair.get("frame_sources") or []
        source = sources[self.frame_index] if self.frame_index < len(sources) else {}
        source_name = str(
            source.get("source_name") or Path(str(source.get("source_path") or "")).name
        )
        self.frame_source.setText(
            self._reference_description()
            if self.reference_mode
            else (source_name or "Corrected calibration frame")
        )
        if self._split_views_active():
            self._request_two_view_frame()
            self._update_metrics()
            self._prefetch_nearby_frames()
            return
        frame_path = (
            self._active_reference_asset()
            if self.reference_mode
            else self._frame_asset(self.frame_index)
        )
        image = self.image_cache.get(frame_path)
        next_base_key = self.image_cache.key(frame_path)
        if next_base_key != self.base_key:
            for viewer in self.viewers.values():
                viewer.cancel_current_draw()
        self.base_key = next_base_key
        if image is not None:
            self.displayed_base_key = self.base_key
            self._set_base_image(image)
        else:
            self.displayed_base_key = ""
            for viewer in self.viewers.values():
                viewer.base_item.setPixmap(QPixmap())
                viewer.set_overlay(None)
                viewer.clear_vectors()
            self.image_cache.request(frame_path)
            for label in self.asset_labels.values():
                label.setText("Loading full-resolution frame...")
        self._request_overlays()
        self._update_metrics()
        self._prefetch_nearby_frames()
        if self._blind_gas_locked:
            self._prefetch_flicker()

    def _request_overlays(self, _value: int | None = None) -> None:
        if self._split_views_active():
            self._request_two_view_overlays()
            return
        if self._blind_gas_locked:
            self.overlay_keys = {"A": "", "B": ""}
            for viewer in self.viewers.values():
                viewer.set_overlay(None)
                viewer.set_phase_map_only(False)
                viewer.set_small_detection_highlight(False)
            return
        enabled = self.contours.isChecked()
        physics_mode = self._physics_layer_mode()
        layer_enabled = enabled
        self.layer_label.setEnabled(layer_enabled)
        self.contour_filter.setEnabled(layer_enabled)
        self.phase_map_only.setEnabled(self.current_pair is not None)
        self.hold_overlay_swap.setEnabled(enabled and self.current_pair is not None and not self.reference_mode)
        if self.current_pair is None:
            return
        selected_candidates = [self.current_pair_object.candidate(side) for side in self._playback_candidate_sides()]
        diagnostic_labels = phase_diagnostic_labels(selected_candidates)
        for index in range(self.contour_filter.count()):
            value = str(self.contour_filter.itemData(index))
            if value in diagnostic_labels:
                self.contour_filter.setItemText(index, diagnostic_labels[value][0])
            available = (all((getattr(c, "quality_flags", None) or {}).get("p1_final3_revision") for c in selected_candidates)
                         if value == "removed_contact_fill_b" else
                         all((getattr(c, "quality_flags", None) or {}).get("p1_final2_revision")
                             or (getattr(c, "quality_flags", None) or {}).get("p1_final3_revision") for c in selected_candidates)
                         if value in P1_FINAL2_OVERLAYS else
                         all((getattr(c, "quality_flags", None) or {}).get("p1_final_revision") for c in selected_candidates)
                         if value in {"removed_top", "removed_floor"} else
                         all(c.raw_phase_map_output_dir and c.optical_model_sha256 for c in selected_candidates)
                         if value == "raw" else all(c.presence_diagnostics_dir for c in selected_candidates)
                         if value.startswith("context_") else True)
            if physics_mode == "physics" and value not in {"all", "gas", "aqueous", "source_support"}:
                available = False
            if physics_mode == "difference" and value != "all":
                available = False
            self.contour_filter.model().item(index).setEnabled(bool(available))
            if not available and self.contour_filter.currentIndex() == index:
                self.contour_filter.blockSignals(True)
                self.contour_filter.setCurrentIndex(0)
                self.contour_filter.blockSignals(False)
        view = str(self.contour_filter.currentData() or 'all')
        if physics_mode == "physics":
            legend = ("Gas physics layer v6 (free gas only). Pink = physics-corrected free gas; "
                      "green = remaining optical CO2 support shown as aqueous; "
                      "cyan = unresolved optical source support inside photographed sand. "
                      "This display layer does not change labels or the stored model.")
        elif physics_mode == "difference":
            legend = DIFFERENCE_LEGEND
        else:
            legend = diagnostic_labels.get(view, ("", self.default_contour_legend))[1]
        self.contour_legend.setText(legend)
        self.highlight_small.setEnabled(physics_mode == "optical" and not view.startswith('context_')
                                        and view != "source_support")
        self.optical_cleanup.decorate()
        filled_only = (not enabled and self._filled_display_enabled()
                       and self._filled_display_available())
        if not enabled and not filled_only:
            self.overlay_swap_held = False
            self.hold_overlay_swap.setText("Hold to swap A/B maps")
            self.overlay_keys = {"A": "", "B": ""}
            for viewer in self.viewers.values():
                viewer.set_overlay(None)
                viewer.set_small_detection_components(())
            return
        visible_sides = set(self._playback_candidate_sides())
        for panel_side in ("A", "B"):
            if panel_side not in visible_sides:
                self.overlay_keys[panel_side] = ""
                self.viewers[panel_side].set_overlay(None)
                self.viewers[panel_side].set_small_detection_components(())
                continue
            source_side = self._overlay_source_side(
                panel_side,
                self.overlay_swap_held,
            )
            if self._bottom_prediction_withheld(source_side):
                self.overlay_keys[panel_side] = ""
                self.viewers[panel_side].set_overlay(None)
                self.viewers[panel_side].set_phase_map_only(False)
                self.asset_labels[panel_side].setText("Photograph only · bottom-strip blind labels pending")
                continue
            source = (
                self._active_reference_overlay_asset(source_side) if self.reference_mode
                else self._overlay_asset(source_side, self.frame_index)
            ) if physics_mode == "optical" or filled_only else self._physics_overlay_asset(
                source_side, self.frame_index, physics_mode, view,
            )
            image = self.image_cache.get(source)
            key = self.image_cache.key(source)
            self.overlay_keys[panel_side] = key
            self.viewers[panel_side].set_overlay(None)
            if image is not None and self.displayed_base_key == self.base_key:
                self.viewers[panel_side].set_overlay(
                    self._display_overlay(key, image)
                )
                self._request_small_components(panel_side, key, image)
            else:
                getattr(self, "_reference_overlay_errors", {}).pop(key, None)
                self.image_cache.request(source)
            if self.reference_mode:
                self.asset_labels[panel_side].setText(self._reference_phase_label(panel_side))

        self._prefetch_nearby_frames()

    def _request_small_components(
        self,
        panel_side: str,
        key: str,
        image: QImage,
    ) -> None:
        if self.optical_cleanup.active:
            return
        if self._physics_layer_mode() != "optical":
            self.viewers[panel_side].set_small_detection_components(())
            return
        if not (
            self.phase_map_only.isChecked()
            and self.highlight_small.isChecked()
            and not str(self.contour_filter.currentData() or '').startswith('context_')
            and self.contour_filter.currentData() != "source_support"
            and self.contours.isChecked()
        ):
            return
        result = self.small_component_cache.get(key)
        if result is not None:
            self.viewers[panel_side].set_small_detection_components(
                result.get("components") or ()
            )
            return
        self.viewers[panel_side].set_small_detection_components(())
        self.small_component_cache.request(key, image)

    def _small_components_loaded(
        self,
        key: str,
        result: object,
        error: str,
    ) -> None:
        if error:
            if key in self.overlay_keys.values():
                self.statusBar().showMessage(
                    f"Could not identify isolated phase-map islands: {error}",
                    10000,
                )
            return
        if not isinstance(result, Mapping):
            return
        components = result.get("components") or ()
        for panel_side in ("A", "B"):
            if key == self.overlay_keys[panel_side]:
                self.viewers[panel_side].set_small_detection_components(components)

    def _set_overlay_swap_held(self, held: bool) -> None:
        if self._split_views_active():
            self._set_two_view_swap_held(held)
            return
        if self._blind_gas_locked:
            self.overlay_swap_held = False
            return
        active = bool(
            held
            and self.current_pair is not None
            and self.contours.isChecked()
            and self.hold_overlay_swap.isEnabled()
        )
        if active == self.overlay_swap_held:
            return
        self.overlay_swap_held = active
        self.hold_overlay_swap.setText(
            "Release to restore A/B maps" if active else "Hold to swap A/B maps"
        )
        self._request_overlays()

    def _contours_toggled(self, checked: bool) -> None:
        if self._blind_gas_locked:
            self.contours.blockSignals(True)
            self.contours.setChecked(False)
            self.contours.blockSignals(False)
            self._request_overlays()
            return
        if not checked:
            self._set_overlay_swap_held(False)
            if self.phase_map_only.isChecked():
                self.phase_map_only.setChecked(False)
        self.contours.setText("Phase map: ON" if checked else "Phase map: OFF")
        if self._split_views_active():
            self._update_two_view_controls()
            self._request_two_view_overlays()
            return
        self._apply_phase_map_only_view()
        self._request_overlays()

    def _phase_map_only_toggled(self, checked: bool) -> None:
        if self._blind_gas_locked:
            self.phase_map_only.blockSignals(True)
            self.phase_map_only.setChecked(False)
            self.phase_map_only.blockSignals(False)
            self._apply_phase_map_only_view()
            return
        if checked and not self.contours.isChecked():
            self.contours.setChecked(True)
        self.phase_map_only.setText(
            "Black background: ON" if checked else "Black background: OFF"
        )
        self._apply_phase_map_only_view()

    def _raw_detail_toggled(self, checked: bool) -> None:
        if checked:
            self.original_photo.setChecked(False)
        self.raw_detail.setText("RAW detail: ON" if checked else "RAW detail: OFF")
        self._update_brightness_control()
        self._request_frame()
        self._request_overlays()

    def _review_history_toggled(self, checked: bool) -> None:
        if self._blind_gas_locked:
            self.show_review_history.blockSignals(True)
            self.show_review_history.setChecked(False)
            self.show_review_history.blockSignals(False)
            self.history_panel.hide()
            return
        self.show_review_history.setText("History: ON" if checked else "History: OFF")
        self.history_panel.setVisible(checked)
        if checked and self._review_history_data is None:
            self._start_review_history_load()
        self._refresh_review_history()
        self._render_injection_points()

    def _start_review_history_load(self, _checked: bool = False) -> None:
        if self._blind_gas_locked:
            return
        if self._review_history_task is not None:
            return
        native_paths = {}
        for row in self._run_list_rows:
            run = str(row.get("run") or "").casefold()
            if run:
                pair = self.app_model._pair(run)
                native_paths[run] = review._native_manifest_path(
                    self.app_model._review_root(pair), run
                )
        # Include the current run when a filtered inventory does not list it.
        if self.current_run and self.current_review_root is not None:
            native_paths[self.current_run] = review._native_manifest_path(
                self.current_review_root, self.current_run
            )
        self._review_history_error = ""
        self.history_status.setText("Loading saved review historyâ€¦")
        self.refresh_history.setEnabled(False)
        self.withdraw_history.setEnabled(False)
        self.reclassify_history.setEnabled(False)
        self.history_classification.setEnabled(False)
        task = ReviewHistoryTask(
            campaign_root=self.workspace.combined_root,
            current_native_paths=native_paths,
            review_roots=[self.workspace.output_dir],
            active_decision_path=self.workspace.output_dir / "pairwise_review_decisions.json",
        )
        self._review_history_task = task
        task.signals.loaded.connect(self._review_history_loaded)
        QThreadPool.globalInstance().start(task)

    def _review_history_loaded(self, payload: object, error: str) -> None:
        self._review_history_task = None
        self._review_history_error = error
        self._review_history_data = (
            dict(payload) if not error and isinstance(payload, Mapping) else None
        )
        self.refresh_history.setEnabled(True)
        self._refresh_review_history()
        self._populate_runs()
        self._render_injection_points()

    def _history_run(self) -> Mapping[str, Any]:
        data = getattr(self, "_review_history_data", None) or {}
        return (data.get("by_run") or {}).get(self.current_run, {})

    @staticmethod
    def _history_provenance(entry: Mapping[str, Any]) -> str:
        original = (entry.get("original_annotation") or entry.get("original_decision")
                    or entry.get("decision") or entry.get("annotation") or {})
        stamp = original.get("updated_utc") or original.get("created_utc") or "Date not recorded"
        reviewer = original.get("reviewer") or ""
        lines = [f"{stamp} · {reviewer}".strip(" ·")]
        for key in ("candidate_id", "params_sha256", "candidate_a", "candidate_b", "candidate", "model_assessment"):
            if original.get(key):
                lines.append(f"Original {key.replace('_', ' ')}: {original[key]}")
        for origin in entry.get("origins") or []:
            lines.append(" · ".join(str(origin[key]) for key in
                         ("kind", "reviewer", "updated_utc", "model_id", "model_root", "decisions_path")
                         if origin.get(key)))
        if entry.get("mapping_error"):
            lines.append(str(entry["mapping_error"]))
        return "\n".join(lines)

    def _refresh_review_history(self) -> None:
        if not hasattr(self, "history_list"):
            return
        self.history_list.clear()
        self.withdraw_history.setEnabled(False)
        self.reclassify_history.setEnabled(False)
        self.history_classification.setEnabled(False)
        self.history_classification.setCurrentIndex(0)
        self.history_details.setText("Select a historical entry to see its source and date.")
        history = self._history_run()
        labels = dict(self.ISSUE_CATEGORY_LABELS)
        boxes = history.get("annotations") or []
        verdicts = history.get("frame_decisions") or []
        run_decisions = history.get("run_decisions") or []
        for kind, entries in (("annotation", boxes), ("decision", verdicts), ("run_decision", run_decisions)):
            for number, entry in enumerate(entries, 1):
                value = entry.get("annotation") if kind == "annotation" else entry.get("decision")
                value = value or {}
                frame = value.get("frame_index")
                elapsed = value.get("elapsed_h")
                time_text = f" · {float(elapsed):.2f} h" if elapsed is not None else ""
                where = f"Frame {int(frame) + 1}{time_text}" if frame is not None else "Experiment"
                if kind == "annotation":
                    category = str(value.get("issue_key") or value.get("category") or "Box")
                    title = f"H{number} · {where} · {labels.get(category, category)}"
                else:
                    title = f"{where} · original review: {value.get('choice') or value.get('verdict') or 'Notes'}"
                if not entry.get("displayable", False) and kind != "run_decision":
                    title += " · cannot align"
                if entry.get("replacement_status") == "pending":
                    title += " · new human classification · awaiting retraining"
                elif entry.get("replacement_status") == "applied":
                    title += " · replacement human classification"
                if entry.get("retraction_status") == "pending":
                    title += " · withdrawn · awaiting retraining"
                elif entry.get("retraction_status") == "applied":
                    title += " · withdrawn from this model"
                item = QListWidgetItem(title)
                item.setData(Qt.ItemDataRole.UserRole, (kind, entry))
                item.setToolTip(self._history_provenance(entry))
                self.history_list.addItem(item)
        unavailable = sum(not entry.get("displayable", False) for entry in [*boxes, *verdicts])
        warnings = (self._review_history_data or {}).get("warnings") or []
        if self._review_history_task is not None:
            text = "Loading saved review historyâ€¦"
        elif self._review_history_error:
            text = f"Could not load history: {self._review_history_error}. Use Refresh to retry."
        elif self._review_history_data is None:
            text = "Turn on History to load earlier reviews."
        else:
            text = f"{len(boxes)} historical boxes · {len(verdicts)} frame verdicts · {len(run_decisions)} run reviews"
            if unavailable:
                text += f" · {unavailable} cannot be aligned"
            if warnings:
                text += f" · {len(warnings)} source warnings (hover for details)"
        hidden = ((self._review_history_data or {}).get("withdrawn_by_run") or {}).get(self.current_run, {})
        hidden_count = sum(len(items) for items in hidden.values())
        if hidden_count:
            text += f" · {hidden_count} withdrawn entries excluded from this model"
        if self._review_retraction_notice:
            text += "\n" + self._review_retraction_notice
        self.history_status.setText(text)
        self.history_status.setToolTip("\n".join(map(str, warnings)))

    def _review_history_opened(
        self, item: QListWidgetItem, *, focus_image: bool = True,
    ) -> None:
        kind, entry = item.data(Qt.ItemDataRole.UserRole)
        self.history_classification.setCurrentIndex(0)
        self._sync_history_withdraw_button(entry)
        value = entry.get("annotation") if kind == "annotation" else entry.get("decision")
        value = value or {}
        origins = entry.get("origins") or []
        origin = origins[0] if origins else {}
        original = entry.get("original_annotation") or entry.get("original_decision") or value
        stamp = original.get("updated_utc") or original.get("created_utc") or origin.get("updated_utc") or "Date not recorded"
        reviewer = original.get("reviewer") or origin.get("reviewer") or "Reviewer not recorded"
        lines = [f"{reviewer} · {stamp}"]
        for key in ("candidate_id", "candidate", "candidate_a", "candidate_b", "model_assessment"):
            if original.get(key):
                candidate = original[key]
                if isinstance(candidate, Mapping):
                    candidate = candidate.get("candidate_id") or candidate.get("model_id") or "Recorded in source"
                lines.append(f"Original {key.replace('_', ' ')}: {candidate}")
        model = origin.get("model_id") or Path(str(origin.get("model_root") or "")).name
        if model:
            lines.append(f"Stored with model: {model}")
        if origin.get("decisions_path"):
            lines.append(f"Source: {Path(str(origin['decisions_path'])).name}")
        if len(origins) > 1:
            lines.append(f"{len(origins)} recorded copies · full source paths on hover")
        if entry.get("mapping_error"):
            lines.append(str(entry["mapping_error"]))
        if entry.get("replacement_status") == "pending":
            lines.insert(0, "New human classification · awaiting retraining")
        elif entry.get("replacement_status") == "applied":
            lines.insert(0, "Replacement human classification · included in this model's evidence selection")
        if entry.get("retraction_status") == "pending":
            lines.insert(0, "Withdrawn from future training · existing model unchanged")
        elif entry.get("retraction_status") == "applied":
            lines.insert(0, "Withdrawn · excluded when this model was trained")
        detail = "\n".join(lines)
        if value.get("notes"):
            detail += "\n" + str(value["notes"])
        self.history_details.setText(detail)
        self.history_details.setToolTip(json.dumps(entry, ensure_ascii=False, indent=2))
        self.issue_annotation_list.setCurrentRow(-1)
        self._set_issue_draw_side(None)
        if not entry.get("displayable", False) or value.get("frame_index") is None:
            return
        frame = int(value["frame_index"])
        if focus_image and 0 <= frame < self.frame_count:
            self._set_frame(frame)
            if kind == "annotation":
                rect = QRectF(float(value.get("x") or 0), float(value.get("y") or 0),
                              max(.002, float(value.get("width") or 0)),
                              max(.002, float(value.get("height") or 0)))
                for viewer in self.viewers.values():
                    viewer.focus_normalized_rect(rect)
        self._render_injection_points()

    def _sync_history_withdraw_button(self, entry: Mapping[str, Any] | None = None) -> None:
        context = ((self._review_history_data or {}).get("retraction_context") or {}).get(self.current_run, {})
        allowed = bool(entry and context.get("project_root") and not context.get("error")
            and not entry.get("retraction_status") and self.reviewer.text().strip()
            and not self.workspace.read_only and not self.workspace.approval_only
            and not self._editing_read_only
            and (self.comparison_mode == "single_candidate" or not self.workspace.blind)
            and self._review_retraction_task is None and self._review_history_task is None)
        self.withdraw_history.setEnabled(allowed)
        annotation = (entry or {}).get("annotation") or {}
        aligned = bool(allowed and annotation and entry.get("displayable")
            and self.comparison_mode == "single_candidate" and not self.reference_mode
            and annotation.get("frame_index") == self.frame_index
            and self.displayed_base_key == self.base_key)
        self.history_classification.setEnabled(aligned)
        selected = str(self.history_classification.currentData() or "")
        previous = str(annotation.get("issue_key") or annotation.get("category") or "")
        self.reclassify_history.setEnabled(aligned and bool(selected) and selected != previous)

    def _history_classification_changed(self, _index: int = 0) -> None:
        if not hasattr(self, "reclassify_history"):
            return
        item = self.history_list.currentItem()
        self._sync_history_withdraw_button(item.data(Qt.ItemDataRole.UserRole)[1] if item else None)

    def _reclassify_review_history(self) -> None:
        item = self.history_list.currentItem()
        if item is None:
            return
        _kind, entry = item.data(Qt.ItemDataRole.UserRole)
        self._sync_history_withdraw_button(entry)
        if not self.reclassify_history.isEnabled():
            return
        value = entry["annotation"]
        rect = QRectF(float(value["x"]), float(value["y"]),
                      float(value["width"]), float(value["height"]))
        points = [point for point in (normalized_point(p) for p in value.get("points") or [])
                  if point is not None]
        shape = "freehand" if value.get("shape") == "freehand" else "rectangle"
        try:
            replacement = self._issue_annotation_payload("A", shape, rect, points,
                category_override=str(self.history_classification.currentData()))
            # Reclassification changes the label only. Historical polygons may
            # predate the drawing tool's 1000-point cap and must remain exact.
            replacement["points"] = [[float(point.x()), float(point.y())] for point in points]
            replacement["reviewer"] = self.reviewer.text().strip()
            frame = next(row for index, row in enumerate(self.current_pair.get("frame_sources") or [])
                         if int(row.get("frame_index", index)) == self.frame_index)
            replacement["source_name"] = str(frame.get("source_name") or Path(str(frame.get("source_path") or "")).name)
            if frame.get("source_sha256"):
                replacement["source_sha256"] = frame["source_sha256"]
            context = self._review_history_data["retraction_context"][self.current_run]
            # Public pair payloads omit identities until reveal; the verified
            # selected candidate still owns the current model/job binding.
            candidate = self.current_pair_object.candidate("A")
            replacement.update(candidate_id=str(candidate.candidate_id),
                               params_sha256=str(candidate.params_sha256))
            job = {"candidate_id": str(candidate.candidate_id),
                   "run": self.current_run, "dataset": self.current_pair.get("dataset")}
            task = ReviewHistoryReclassificationTask(entry, series=context["series"], run=self.current_run,
                actor=self.reviewer.text().strip(), replacement_annotation=replacement,
                current_manifest_path=self.workspace.combined_root / "postflight_manifest.json",
                current_review_job=job, project_root=Path(context["project_root"]),
                reason="Historical classification corrected by the reviewer")
        except Exception as exc:  # noqa: BLE001 - visible UI boundary
            self.history_status.setText(f"Cannot prepare replacement classification: {exc}")
            return
        self._review_retraction_task = task
        self._review_retraction_notice = "Saving the replacement classification to the shared projectâ€¦"
        self.withdraw_history.setEnabled(False)
        self.reclassify_history.setEnabled(False)
        self.history_classification.setEnabled(False)
        self.refresh_history.setEnabled(False)
        self.history_status.setText(self._review_retraction_notice)
        task.signals.loaded.connect(self._review_history_reclassified)
        QThreadPool.globalInstance().start(task)

    def _review_history_reclassified(self, event: object, error: str) -> None:
        self._review_history_retracted(event, error, reclassification=True)

    def _withdraw_review_history(self) -> None:
        item = self.history_list.currentItem()
        if item is None:
            return
        _kind, entry = item.data(Qt.ItemDataRole.UserRole)
        self._sync_history_withdraw_button(entry)
        if not self.withdraw_history.isEnabled():
            return
        context = self._review_history_data["retraction_context"][self.current_run]
        task = ReviewHistoryRetractionTask(entry, series=context["series"], run=self.current_run,
            actor=self.reviewer.text().strip(), project_root=Path(context["project_root"]),
            reason="Withdrawn by the reviewer from saved history")
        self._review_retraction_task = task
        self._review_retraction_notice = "Saving withdrawal to the shared projectâ€¦"
        self.withdraw_history.setEnabled(False)
        self.reclassify_history.setEnabled(False)
        self.history_classification.setEnabled(False)
        self.refresh_history.setEnabled(False)
        self.history_status.setText(self._review_retraction_notice)
        task.signals.loaded.connect(self._review_history_retracted)
        QThreadPool.globalInstance().start(task)

    def _review_history_retracted(self, event: object, error: str, *, reclassification: bool = False) -> None:
        self._review_retraction_task = None
        self.refresh_history.setEnabled(True)
        if error:
            action = "Classification change" if reclassification else "Withdrawal"
            self._review_retraction_notice = f"{action} was not confirmed: {error}. You can retry."
            self.history_status.setText(self._review_retraction_notice)
            item = self.history_list.currentItem()
            self._sync_history_withdraw_button(item.data(Qt.ItemDataRole.UserRole)[1] if item else None)
            return
        self._review_retraction_notice = (
            "Replacement classification saved for future training. Existing model unchanged."
            if reclassification else "Withdrawn from future training. Existing model unchanged."
        )
        from darsia.fluidflower.tools.phase_review_history import apply_saved_retraction
        try:
            self._review_history_data = apply_saved_retraction(self._review_history_data or {}, event)
        except Exception as exc:  # noqa: BLE001 - append succeeded; keep its acknowledgement
            self._review_history_data = None
            self._review_retraction_notice += f" History refresh needed: {exc}"
        self._refresh_review_history()
        self._render_injection_points()
        self.withdraw_history.setEnabled(False)
        self._start_review_history_load()

    def _render_review_history(self) -> None:
        if hasattr(self, "reclassify_history"):
            self._history_classification_changed()
        if (self.reference_mode or not hasattr(self, "show_review_history")
                or not self.show_review_history.isChecked()):
            return
        labels = dict(self.ISSUE_CATEGORY_LABELS)
        selected_item = self.history_list.currentItem()
        selected_data = selected_item.data(Qt.ItemDataRole.UserRole) if selected_item else None
        selected = selected_data[1] if selected_data else {}
        for number, entry in enumerate(self._history_run().get("annotations") or [], 1):
            value = entry.get("annotation") or {}
            if not entry.get("displayable", False) or value.get("frame_index") != self.frame_index:
                continue
            rect = QRectF(float(value.get("x") or 0), float(value.get("y") or 0),
                          float(value.get("width") or 0), float(value.get("height") or 0))
            points = [point for point in (normalized_point(p) for p in value.get("points") or [])
                      if point is not None]
            is_freehand = str(value.get("shape") or "") == "freehand" and len(points) >= 3
            if not is_freehand:
                points = [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
            category = str(value.get("issue_key") or value.get("category") or "Box")
            color = QColor("#9ea6ad" if entry.get("retraction_status") else
                           "#75ddd3" if entry.get("replacement_status") else "#b8a4ff")
            is_selected = (entry.get("source_id") == selected.get("source_id")
                           and entry.get("payload_sha256") == selected.get("payload_sha256"))
            label = f"H{number}"
            if is_selected:
                label += f" · {labels.get(category, category)} · history"
                if entry.get("retraction_status"):
                    label += " · withdrawn"
                elif entry.get("replacement_status") == "pending":
                    label += " · pending replacement"
            for viewer in self.viewers.values():
                viewer.add_polygon(points, color, width=2.0, pen_style=Qt.PenStyle.DashLine)
                viewer.add_annotation_hit_region(
                    f"history:{number - 1}", rect, points if is_freehand else (),
                )
                if self.show_annotation_names.isChecked():
                    viewer.add_label(QPointF(rect.x(), max(.01, rect.y() - .018)),
                                     label, color)

    def _annotation_names_toggled(self, checked: bool) -> None:
        self.show_annotation_names.setText(
            "Box names: ON" if checked else "Box names: OFF"
        )
        self._render_injection_points()

    def _annotations_toggled(self, checked: bool) -> None:
        self.show_annotations.setText(
            "Boxes + names: ON" if checked else "Boxes + names: OFF"
        )
        self.show_annotation_names.setEnabled(checked)
        self._render_injection_points()

    def _configure_injection_references(self) -> None:
        self._reference_overlay_errors: dict[str, str] = {}
        self.image_cache.cancel_queued_reference_preparations()
        if not hasattr(self, "reference_prepare_timer"):
            self.reference_prepare_timer = QTimer(self)
            self.reference_prepare_timer.setInterval(1500)
            self.reference_prepare_timer.timeout.connect(self._prepare_reference_maps)
        # Optional reference overlays are requested by the reference buttons.
        # Opening an experiment must not start a separate inference/replay job.
        self.reference_prepare_timer.stop()
        for button in self.injection_reference_buttons:
            self.injection_reference_layout.removeWidget(button)
            button.deleteLater()
        self.injection_reference_buttons = []
        self.active_injection_reference = None
        try:
            self.injection_references = injection_refs.injection_end_references(
                self.current_pair or {}, self.current_review_root)
        except (OSError, ValueError, KeyError) as exc:
            self.injection_references = []
            self.statusBar().showMessage(f"Injection references unavailable: {exc}", 10000)
        for index, row in enumerate(self.injection_references):
            button = QPushButton(str(row["label"]))
            button.setProperty("referenceImage", True)
            button.setCheckable(True)
            button.setEnabled(bool(row.get("available")))
            button.setToolTip(self._injection_reference_tooltip(row))
            button.toggled.connect(
                lambda checked, index=index: self._injection_reference_toggled(index, checked))
            self.injection_reference_buttons.append(button)
            self.injection_reference_layout.addWidget(button)
        self.frame_label.set_reference_texts([
            self._injection_reference_status(index) or ""
            for index in range(len(self.injection_references))
        ])
        self._sync_reference_buttons()

    def _injection_reference_status(self, index: int | None = None) -> str | None:
        selected = self.active_injection_reference if index is None else index
        if selected is None or not 0 <= selected < len(self.injection_references):
            return None
        row = self.injection_references[selected]
        elapsed = row.get("elapsed_h")
        return (str(row["label"]) + " reference"
                + (f" · {float(elapsed):.3f} h" if elapsed is not None else ""))

    @staticmethod
    def _injection_reference_tooltip(row: Mapping[str, Any]) -> str:
        parts = [f"Injection ends at {float(row['end_h']):.4f} h."]
        if row.get("elapsed_h") is not None:
            selection = str(row.get("description") or "First available image at/after the end")
            parts.append(f"{selection}: {row.get('source_name', '')} "
                         f"at {float(row['elapsed_h']):.4f} h "
                         f"({float(row.get('gap_seconds') or 0):.1f} seconds after the end).")
        if row.get("selection_scope"):
            parts.append(str(row["selection_scope"]))
        if row.get("reason"):
            parts.append(str(row["reason"]))
        parts.append("Click again to return to the selected timeline image and zoom.")
        return "\n".join(parts)

    def _reference_description(self) -> str:
        if self.active_injection_reference is None:
            if (self.current_pair or {}).get("rig_saved_reference_index") is not None:
                row = self.current_pair.get("original_reference_source") or {}
                return (f"Pre-injection reference · {row.get('source_name', '')} · "
                        f"{float(row.get('elapsed_h', 0.)):.4f} h · own source geometry")
            return "Aligned pre-injection reference"
        row = self.injection_references[self.active_injection_reference]
        description = f"{self._injection_reference_status()} · {row.get('source_name', '')}"
        if row.get("selection_policy") == injection_refs.RIG_SAVED_POLICY:
            description += f" · saved frame {float(row['gap_seconds']):.1f} s after end"
        return description

    def _saved_rig_reference_index(self, row=None) -> int | None:
        if row is None and self.active_injection_reference is not None:
            row = self.injection_references[self.active_injection_reference]
        if row is not None:
            return (row.get("frame_index") if row.get("selection_policy") == injection_refs.RIG_SAVED_POLICY
                    and row.get("available") else None)
        return (self.current_pair or {}).get("rig_saved_reference_index")

    def _active_reference_asset(self, view: str | None = None) -> DeferredImage:
        index = self._saved_rig_reference_index()
        if index is not None:
            if view is None:
                view = ("original" if self.original_photo.isChecked() else
                        "raw_detail" if self.raw_detail.isChecked() else self.display_mode)
            # Camera identity alone does not prove equal rendering geometry:
            # use this saved frame's photograph AND its own saved phase map.
            return self._photo_view_asset(index, view)
        if self.active_injection_reference is None:
            return self._reference_asset()
        row = dict(self.injection_references[self.active_injection_reference])
        root, run = self.current_review_root, self.current_run
        return DeferredImage(
            f"injection-reference:{root}:{run}:{row['id']}:{row.get('source_name')}:"
            f"{row.get('native_manifest_sha256', '')}:{row.get('sha256', '')}",
            lambda: injection_refs.resolve_reference_asset(row, root, run),
        )

    def _active_reference_overlay_asset(self, side: str, layer: str | None = None) -> DeferredImage:
        row = (dict(self.injection_references[self.active_injection_reference])
               if self.active_injection_reference is not None else None)
        return self._reference_overlay_asset(side, row, layer)

    def _reference_overlay_asset(self, side: str, row: Mapping[str, Any] | None,
                                 layer: str | None = None) -> DeferredImage:
        root, run = self.current_review_root, self.current_run
        candidate = self.current_pair_object.candidate(side)
        if review._candidate_prediction_withheld(root, candidate):
            return DeferredImage(f"bottom-blind-reference-withheld:{root}:{run}:{side}",
                                 lambda: QImage(), normalize_overlay=True)
        row = None if row is None else dict(row)
        phase = str(layer or self.contour_filter.currentData() or "all")
        index = ((self.current_pair or {}).get("rig_saved_reference_index") if row is None
                 else self._saved_rig_reference_index(row))
        if index is not None:
            # The own-source existing map is the only overlay for a rig reference.
            # Never replay a legacy optical/reference model from this display action.
            return self._overlay_asset(side, index, phase)
        fingerprint = review._candidate_fingerprint(candidate)
        map_root = review._candidate_phase_map_root(root, candidate)
        reference_identity = "baseline" if row is None else json.dumps(
            {key: row.get(key) for key in ("id", "source_name", "elapsed_h", "native_manifest_sha256", "sha256")},
            sort_keys=True)
        if self.local_reference_search.isChecked() and phase in {"all", "gas", "aqueous"}:
            from darsia.fluidflower.tools.phase_local_support_preview import resolve as resolve_local
            return DeferredImage(
                f"reference-overlay:local-support:{root}:{run}:{fingerprint}:{reference_identity}:{phase}",
                lambda: resolve_local(root, run, candidate, 0, phase, reference_mode=True, reference=row),
                normalize_overlay=True, reference_phase=True)
        from darsia.fluidflower.phase_review_basis import reference_layer as selected_reference_layer
        reference_layer = selected_reference_layer(candidate, overlay_asset_layer(phase))
        optical = reference_layer == "raw" and phase != "raw"
        def display(path):
            if path is not None and optical and phase in {"gas", "aqueous"}:
                from darsia.fluidflower.phase_rendering import two_phase_overlay_path
                return two_phase_overlay_path(path, phase)
            return path
        def resolve() -> Path:
            from darsia.fluidflower.tools.phase_reference_maps import resolve_reference_overlay
            return display(resolve_reference_overlay(row, root, run, candidate, side, reference_layer))
        def prefetch() -> Path | None:
            from darsia.fluidflower.tools.phase_reference_maps import resolve_reference_overlay
            return display(resolve_reference_overlay(row, root, run, candidate, side, reference_layer, cached_only=True))
        return DeferredImage(
            f"reference-overlay:optical-v1:{root}:{map_root}:{run}:{side}:{fingerprint}:{reference_identity}:{phase}",
            resolve, normalize_overlay=True, reference_phase=True, prefetch_resolve=prefetch,
            display_domain=self._photo_sand_loader(reference=True) if row is None else None)

    def _prefetch_reference_maps(self) -> None:
        if self.current_pair is None or not self.contours.isChecked():
            return
        rows = ([None] if self.reference.isEnabled() else []) + [
            row for row in self.injection_references
            if row.get("available") and not row.get("needs_materialization")]
        self.image_cache.prefetch([
            self._reference_overlay_asset(side, row)
            for row in rows for side in self._playback_candidate_sides()])

    def _prepare_reference_maps(self) -> None:
        # Only the selected reference is eligible for explicit preparation.
        # Ordinary timeline viewing never warms other reference acquisitions.
        if (not self.reference_mode or self.current_pair is None or not self.contours.isChecked()
                or not self._speed_frame_assets_ready()):
            return
        self.image_cache.prepare([
            self._active_reference_overlay_asset(side)
            for side in self._playback_candidate_sides()])
        self.reference_prepare_timer.stop()

    def _reference_work_started(self, key: str) -> None:
        if self.reference_mode:
            for side in self._playback_candidate_sides():
                if key == self.overlay_keys.get(side):
                    self.asset_labels[side].setText(self._reference_phase_label(side))

    def _reference_phase_label(self, side: str) -> str:
        description = self._reference_description()
        if self._saved_rig_reference_index() is not None:
            return f"Full resolution · saved source photograph and map · {description}"
        if self.contours.isChecked():
            key = self.overlay_keys.get(side, "")
            error = getattr(self, "_reference_overlay_errors", {}).get(key)
            if error:
                return f"Phase map unavailable: {error}"
            if (self.displayed_base_key != self.base_key
                    or self.viewers[side].overlay_item.pixmap().isNull()):
                if key in self.image_cache.queued:
                    return f"Reference phase map queued for preparation · {description}"
                if key in self.image_cache.reference_running:
                    return ("Preparing exact reference phase map · first preparation can take a few minutes · "
                            + description)
                return f"Preparing exact reference phase map · {description}"
        return (f"Full resolution · {description}"
                + (" · phase map on black" if self.phase_map_only.isChecked() else ""))

    def _refresh_active_injection_reference(self) -> None:
        index = self.active_injection_reference
        if index is None or not self.injection_references[index].get("needs_materialization"):
            return
        previous = self.injection_references[index]
        rows = injection_refs.injection_end_references(self.current_pair, self.current_review_root)
        current = next((row for row in rows if row["id"] == previous["id"]), None)
        identity = ("source_name", "elapsed_h", "native_manifest_sha256")
        if current and all(current.get(key) == previous.get(key) for key in identity):
            # A worker may have just prepared this image. Its measured source
            # offset must be used before painting it against the selected frame.
            self.injection_references[index] = current
            self.injection_reference_buttons[index].setToolTip(
                self._injection_reference_tooltip(current))

    def _sync_reference_buttons(self) -> None:
        baseline = self.reference_mode and self.active_injection_reference is None
        self.reference.blockSignals(True)
        self.reference.setChecked(baseline)
        self.reference.blockSignals(False)
        self.reference.setText("Pre-injection reference")
        for index, button in enumerate(self.injection_reference_buttons):
            button.blockSignals(True)
            button.setChecked(self.reference_mode and index == self.active_injection_reference)
            button.blockSignals(False)
        for side, buttons in self.issue_draw_buttons.items():
            for button in buttons.values():
                button.setEnabled(not self.reference_mode and not self._editing_read_only
                                  and bool(self._selected_issue_category(side)))
        self._apply_issue_draw_mode()

    def _set_reference_view(self, active: bool, injection_index: int | None = None) -> None:
        self.reference_mode = bool(active)
        self.active_injection_reference = injection_index if active else None
        saved_rig_reference = self._saved_rig_reference_index() is not None
        if self.active_injection_reference is not None and not saved_rig_reference:
            # Injection-end references have no V5 RAW-detail display artifact.
            self.raw_detail.blockSignals(True)
            self.raw_detail.setChecked(False)
            self.raw_detail.blockSignals(False)
            self.raw_detail.setText("RAW detail: OFF")
        self.raw_detail.setEnabled(
            bool((self.current_pair or {}).get("v5_raw_detail_available"))
            and (self.active_injection_reference is None or saved_rig_reference))
        self._update_brightness_control()
        self.base_key = ""
        self.displayed_base_key = ""
        self.overlay_keys = {"A": "", "B": ""}
        for viewer in self.viewers.values():
            viewer.set_overlay(None)
        self._sync_reference_buttons()
        self.contours.setEnabled(True)
        self.verdict.setEnabled(not self.reference_mode and not self._editing_read_only)
        if self.reference_mode:
            self._set_overlay_swap_held(False)
        self._render_injection_points()
        self._request_frame()

    def _reference_toggled(self, checked: bool) -> None:
        if self.speed_loop_held:
            self._stop_speed_loop()
        self._set_reference_view(bool(checked and self.reference.isEnabled()))

    def _injection_reference_toggled(self, index: int, checked: bool) -> None:
        if not 0 <= index < len(self.injection_reference_buttons):
            return
        if checked and not self.injection_reference_buttons[index].isEnabled():
            return
        if self.speed_loop_held:
            self._stop_speed_loop()
        self._set_reference_view(checked, index if checked else None)

    def _contour_opacity_changed(self, value: int) -> None:
        self._apply_phase_map_only_view()

    def _apply_phase_map_only_view(self) -> None:
        if self._split_views_active():
            for side, viewer in self.viewers.items():
                viewer.set_phase_map_only(False)
                viewer.set_small_detection_highlight(False)
                viewer.set_overlay_opacity(self.panel_opacities[side] / 100.0)
            return
        active = bool(
            not self._blind_gas_locked
            and
            hasattr(self, "phase_map_only")
            and self.phase_map_only.isChecked()
            and self.contours.isChecked()
        )
        opacity = (
            1.0
            if active
            else max(20, min(100, int(self.contour_opacity.value()))) / 100.0
        )
        self.contour_opacity.setEnabled(not active)
        self.highlight_small.setEnabled(active and self._physics_layer_mode() == "optical")
        self.contour_opacity_label.setText(
            "100%" if active else f"{int(round(100 * opacity))}%"
        )
        for side, viewer in self.viewers.items():
            side_active = active and not self._bottom_prediction_withheld(
                self._overlay_source_side(side, self.overlay_swap_held))
            viewer.set_phase_map_only(side_active)
            viewer.set_small_detection_highlight(
                side_active and self.highlight_small.isChecked()
                and self._physics_layer_mode() == "optical"
            )
            viewer.set_overlay_opacity(opacity)
        if active and self.highlight_small.isChecked():
            for panel_side in ("A", "B"):
                key = self.overlay_keys.get(panel_side) or ""
                image = self.image_cache.images.get(key)
                if key and image is not None:
                    self._request_small_components(panel_side, key, image)
        if hasattr(self, "asset_labels"):
            for side, label in self.asset_labels.items():
                if self._bottom_prediction_withheld(self._overlay_source_side(side, self.overlay_swap_held)):
                    label.setText("Photograph only · bottom-strip blind labels pending")
                    continue
                label.setText(
                    self._reference_phase_label(side) if self.reference_mode else
                    "Full resolution · phase map on black" if active else
                    "Full resolution · corrected frame"
                )

    def _image_loaded(self, key: str, image: QImage, error: str) -> None:
        if self._split_views_active():
            photo_sides = [side for side in ("A", "B") if key == self.base_keys[side]]
            overlay_sides = [side for side in ("A", "B") if key == self.overlay_keys[side]]
            if photo_sides or overlay_sides:
                self._set_two_view_swap_held(False)
            if error and (photo_sides or overlay_sides):
                self.statusBar().showMessage(error, 10000)
                for side in photo_sides + overlay_sides:
                    self.asset_labels[side].setText(f"Image unavailable: {error}")
                return
            if not error:
                for side in photo_sides:
                    self.displayed_base_keys[side] = key
                    if side == "A":
                        self.displayed_base_key = key
                    self._set_two_view_base_image(side, image)
                    self._request_two_view_overlays(side)
                for side in overlay_sides:
                    if self.displayed_base_keys[side] == self.base_keys[side]:
                        self.viewers[side].set_overlay(
                            self._two_view_overlay_image(side, image)
                        )
            for side in ("A", "B"):
                if key == self.plot_keys[side]:
                    self.plot_labels[side].set_image(image)
            self._update_two_view_swap_button()
            return
        relevant = (
            key == self.base_key
            or key in self.overlay_keys.values()
            or key in self.plot_keys.values()
        )
        if error:
            if relevant:
                self.statusBar().showMessage(error, 10000)
                for side in ("A", "B"):
                    if key == self.overlay_keys[side]:
                        if self.reference_mode:
                            self._reference_overlay_errors[key] = error
                        self.asset_labels[side].setText(f"Phase map unavailable: {error}")
            return
        if key == self.base_key:
            self.displayed_base_key = key
            self._set_base_image(image)
            self._request_overlays()
        for side in ("A", "B"):
            if key == self.overlay_keys[side] and self.displayed_base_key == self.base_key:
                self.viewers[side].set_overlay(
                    self._display_overlay(key, image)
                )
                self._request_small_components(side, key, image)
                if self.reference_mode:
                    self._reference_overlay_errors.pop(key, None)
                    self.asset_labels[side].setText(self._reference_phase_label(side))
            if key == self.plot_keys[side]:
                self.plot_labels[side].set_image(image)

    def _set_base_image(self, image: QImage) -> None:
        self._refresh_active_injection_reference()
        image = self._rig_photo_image(image)
        self.syncing_views = True
        try:
            for side, viewer in self.viewers.items():
                if side not in self._playback_candidate_sides():
                    continue
                viewer.set_image(image, fit=self.fit_next_image)
                offset = self._active_reference_offset(side) if self.reference_mode else (0.0, 0.0)
                viewer.set_base_offset(*offset)
                viewer.set_overlay_offset(*offset)
        finally:
            self.syncing_views = False
        self.fit_next_image = False
        for label in self.asset_labels.values():
            label.setText(
                f"Full resolution · {self._reference_description()}"
                if self.reference_mode
                else "Full resolution · corrected frame"
            )
        self._apply_phase_map_only_view()
        self._render_injection_points()
        self._sync_view("A")

    def _render_injection_points(self, _value: int | None = None) -> None:
        for viewer in self.viewers.values():
            viewer.clear_vectors()
        if self.current_pair is None:
            return
        if self.injectors.isChecked():
            for row in self.current_pair.get("injection_points") or []:
                x = float(row.get("x") or 0.0)
                y = float(row.get("y") or 0.0)
                label = str(row.get("label") or "Injector")
                marker = QRectF(
                    max(0.0, x - 0.006),
                    max(0.0, y - 0.006),
                    0.012,
                    0.012,
                )
                for viewer in self.viewers.values():
                    viewer.add_rect(marker, QColor("#ef3e36"), width=3.0)
                    viewer.add_label(
                        QPointF(min(0.98, x + 0.008), max(0.01, y - 0.014)),
                        label,
                        QColor("#ef3e36"),
                    )
        for guide in self.reference_guides:
            orientation = str(guide.get("orientation") or "")
            position = float(guide.get("position") or 0.0)
            label = str(guide.get("label") or "Reference")
            if orientation == "horizontal":
                start = QPointF(0.0, position)
                end = QPointF(1.0, position)
                label_position = QPointF(0.008, max(0.004, min(0.97, position + 0.004)))
                color = QColor("#ffd166")
            elif orientation == "vertical":
                start = QPointF(position, 0.0)
                end = QPointF(position, 1.0)
                label_position = QPointF(max(0.004, min(0.97, position + 0.004)), 0.008)
                color = QColor("#66d9ef")
            else:
                continue
            for viewer in self.viewers.values():
                viewer.add_line(start, end, color, width=2.0)
                viewer.add_label(label_position, label, color)
        if getattr(self, "evidence_audit", None) is not None:
            self.evidence_audit.render()
        if not self.reference_mode:
            self._render_review_history()
        if self.reference_mode or not self.show_annotations.isChecked():
            return
        show_names = self.show_annotation_names.isChecked()
        labels = dict(self.ISSUE_CATEGORY_LABELS)
        for number, annotation in enumerate(self.annotations, start=1):
            annotation=ReviewImageAssets._annotation_display_payload(self,annotation)
            if annotation.get("frame_index") != self.frame_index:
                continue
            if self._blind_gas_locked and (
                annotation.get("session") != BLIND_GAS_SESSION
                or annotation.get("post_reveal")
            ):
                continue
            side = str(annotation.get("candidate") or "A").upper()
            if self._two_views_active() or self._is_shared_image_annotation(annotation):
                target_viewers = tuple(self.viewers.values())
            else:
                viewer = self.viewers.get(side)
                target_viewers = (viewer,) if viewer is not None else ()
            if not target_viewers:
                continue
            category = str(
                annotation.get("issue_key")
                or annotation.get("category")
                or "isolated_noise"
            )
            color = QColor(self.ISSUE_COLORS.get(category, "#ffd166"))
            rect = QRectF(
                float(annotation.get("x") or 0.0),
                float(annotation.get("y") or 0.0),
                float(annotation.get("width") or 0.0),
                float(annotation.get("height") or 0.0),
            )
            points = [
                point
                for point in (
                    normalized_point(value)
                    for value in annotation.get("points") or []
                )
                if point is not None
            ]
            is_freehand = (
                str(annotation.get("shape") or "") == "freehand"
                and len(points) >= 3
            )
            for viewer in target_viewers:
                if is_freehand:
                    viewer.add_polygon(points, color, width=3.0)
                else:
                    viewer.add_rect(rect, color, width=3.0)
                viewer.add_annotation_hit_region(
                    str(annotation.get("id") or f"index:{number - 1}"),
                    rect,
                    points if is_freehand else (),
                )
                if show_names:
                    viewer.add_label(
                        QPointF(
                            max(0.0, min(0.98, rect.x())),
                            max(0.01, min(0.98, rect.y() - 0.018)),
                        ),
                        f"{number}. {labels.get(category, category)}",
                        color,
                    )

    def _refresh_reference_guide_list(self, selected_id: str = "") -> None:
        if not hasattr(self, "reference_guide_list"):
            return
        current_id = selected_id or str(
            self.reference_guide_list.currentData() or ""
        )
        self.reference_guide_list.blockSignals(True)
        self.reference_guide_list.clear()
        if not self.reference_guides:
            self.reference_guide_list.addItem("No saved guides", "")
        else:
            for guide in self.reference_guides:
                orientation = str(guide.get("orientation") or "")
                prefix = "Horizontal" if orientation == "horizontal" else "Vertical"
                position = 100.0 * float(guide.get("position") or 0.0)
                self.reference_guide_list.addItem(
                    f"{prefix} · {guide.get('label') or 'Reference'} · {position:.1f}%",
                    str(guide.get("id") or ""),
                )
        index = self.reference_guide_list.findData(current_id)
        self.reference_guide_list.setCurrentIndex(max(0, index))
        self.reference_guide_list.blockSignals(False)
        self.delete_reference_guide.setEnabled(
            bool(self.reference_guides)
            and not self._editing_read_only and not self.workspace.approval_only
        )

    def _set_reference_guide_mode(self, orientation: str) -> None:
        if self._editing_read_only or self.workspace.approval_only:
            return
        normalized = str(orientation or "").casefold()
        if normalized not in {"horizontal", "vertical"}:
            self._set_issue_draw_side(None)
            return
        self.active_issue_side = None
        self.active_guide_orientation = normalized
        for viewer in self.viewers.values():
            viewer.set_draw_mode(f"{normalized}-guide")
        self.pan_images.blockSignals(True)
        self.pan_images.setChecked(False)
        self.pan_images.blockSignals(False)
        for buttons in self.issue_draw_buttons.values():
            for button in buttons.values():
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
        for value, button in (
            ("horizontal", self.horizontal_guide),
            ("vertical", self.vertical_guide),
        ):
            button.blockSignals(True)
            button.setChecked(value == normalized)
            button.blockSignals(False)
        self.statusBar().showMessage(
            f"Click either image to place named {normalized} reference guides",
            6000,
        )

    def _add_reference_guide(
        self,
        _side: str,
        orientation: str,
        point: QPointF,
    ) -> None:
        if self._editing_read_only or self.workspace.approval_only:
            return
        normalized = str(orientation or "").casefold()
        if normalized != self.active_guide_orientation:
            return
        position = point.y() if normalized == "horizontal" else point.x()
        prefix = "H" if normalized == "horizontal" else "V"
        count = sum(
            str(item.get("orientation") or "") == normalized
            for item in self.reference_guides
        )
        label = self.reference_guide_name.text().strip() or f"{prefix}{count + 1}"
        guide = {
            "id": uuid.uuid4().hex,
            "orientation": normalized,
            "position": max(0.0, min(1.0, float(position))),
            "label": label[:80],
            "coordinate_space": "normalized_native_review_canvas_16x9",
            "created_utc": utc_now(),
        }
        self.reference_guides.append(guide)
        self.reference_guide_name.clear()
        self._refresh_reference_guide_list(str(guide["id"]))
        self._render_injection_points()
        self._schedule_save()
        self.statusBar().showMessage(
            f"Added {normalized} guide {guide['label']}; placement remains active",
            5000,
        )

    def _delete_selected_reference_guide(self) -> None:
        if self._editing_read_only or self.workspace.approval_only:
            return
        guide_id = str(self.reference_guide_list.currentData() or "")
        if not guide_id:
            return
        before = len(self.reference_guides)
        self.reference_guides = [
            item
            for item in self.reference_guides
            if str(item.get("id") or "") != guide_id
        ]
        if len(self.reference_guides) == before:
            return
        self._refresh_reference_guide_list()
        self._render_injection_points()
        self._schedule_save()
        self.statusBar().showMessage("Reference guide deleted", 3500)

    @classmethod
    def _is_shared_image_annotation(cls, annotation: Mapping[str, Any]) -> bool:
        if annotation.get("source") == "same_run_two_view_image_area_v1":
            return True
        category = str(
            annotation.get("issue_key")
            or annotation.get("category")
            or ""
        )
        if category in {
            "unsupported_free_gas_classification",
            "unsupported_aqueous_co2_classification",
        }:
            return False
        shared_categories = {key for key, _label in cls.SHARED_ARTIFACT_LABELS}
        return bool(
            str(annotation.get("annotation_scope") or "")
            == "shared_source_image"
            or str(annotation.get("interpretation_scope") or "")
            == "source_image_shared_across_candidates"
            or category in shared_categories
        )

    @classmethod
    def _annotation_flags(
        cls,
        annotations: Sequence[Mapping[str, Any]],
    ) -> dict[str, set[str]]:
        valid = {key for key, _label in cls.FLAG_LABELS}
        result: dict[str, set[str]] = {"A": set(), "B": set()}
        for annotation in annotations:
            category = str(
                annotation.get("issue_key")
                or annotation.get("category")
                or ""
            )
            if category not in valid:
                continue
            if cls._is_shared_image_annotation(annotation):
                result["A"].add(category)
                result["B"].add(category)
                continue
            side = str(annotation.get("candidate") or "A").upper()
            if side in result:
                result[side].add(category)
        return result

    def _selected_issue_category(self, side: str) -> str:
        normalized = str(side or "").upper()
        group = self.issue_category_groups.get(normalized)
        button = group.checkedButton() if group is not None else None
        return str(button.property("issue_key") or "") if button is not None else ""

    def _clear_issue_category_selection(self, side: str) -> None:
        normalized = str(side or "").upper()
        group = self.issue_category_groups.get(normalized)
        if group is None:
            return
        group.setExclusive(False)
        for button in group.buttons():
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        group.setExclusive(True)
        for button in self.issue_draw_buttons.get(normalized, {}).values():
            button.setEnabled(False)

    def _set_selected_issue_category(self, side: str, key: str) -> bool:
        normalized = str(side or "").upper()
        button = self.flag_boxes.get(normalized, {}).get(str(key))
        if button is None:
            return False
        button.setChecked(True)
        return True

    def _issue_category_changed(self, side: str, _key: str, checked: bool) -> None:
        normalized = str(side or "").upper()
        if not checked:
            return
        if _key in review.CANDIDATE_ISSUE_HELP:
            self.issue_help_labels[normalized].setText(
                review.CANDIDATE_ISSUE_HELP[_key]
            )
        else:
            self.issue_help_labels[normalized].setText(
                "Select one visible phase class or one physical issue for the next "
                "area. Phase classes use the same vocabulary as blind labeling; "
                "the model disagreement is derived automatically."
            )
        for button in self.issue_draw_buttons.get(normalized, {}).values():
            button.setEnabled(
                not self.reference_mode and not self._editing_read_only
                and not self.workspace.approval_only
            )

    def _set_issue_draw_side(self, side: str | None, mode: str = "box") -> None:
        if self.reference_mode and side is not None:
            return
        normalized = (
            "" if self._editing_read_only or self.workspace.approval_only
            else str(side or "").upper()
        )
        draw_mode = str(mode or "box").casefold()
        if draw_mode not in {"box", "freehand", "polygon"}:
            draw_mode = "box"
        if normalized in self.viewers and not self._selected_issue_category(normalized):
            self.statusBar().showMessage(
                f"Select one Candidate {normalized} issue category before drawing",
                5000,
            )
            normalized = ""
        self.active_guide_orientation = None
        self.active_issue_side = (
            normalized
            if normalized in self.viewers
            else None
        )
        self.active_issue_draw_mode = (
            draw_mode if self.active_issue_side is not None else "pan"
        )
        for button in (self.horizontal_guide, self.vertical_guide):
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        self._apply_issue_draw_mode()

    def _apply_issue_draw_mode(self) -> None:
        """Keep the selected tool while temporarily inspecting reference images."""
        if self.active_guide_orientation is not None:
            return
        for candidate_side, viewer in self.viewers.items():
            viewer.set_draw_mode(
                self.active_issue_draw_mode
                if (candidate_side == self.active_issue_side
                    and not self.reference_mode and not self._editing_read_only
                    and not self.workspace.approval_only)
                else "pan"
            )
        self.pan_images.blockSignals(True)
        self.pan_images.setChecked(self.active_issue_side is None)
        self.pan_images.blockSignals(False)
        for candidate_side, buttons in self.issue_draw_buttons.items():
            for button_mode, button in buttons.items():
                button.blockSignals(True)
                button.setChecked(
                    candidate_side == self.active_issue_side
                    and button_mode == self.active_issue_draw_mode
                )
                button.blockSignals(False)
        if self.active_issue_side is None:
            self.statusBar().showMessage("Pan / inspect mode", 2500)
        else:
            shape_name = {"box": "boxes", "freehand": "freehand areas", "polygon":
                "polygons · left-click corners; right-click closes; Esc cancels"}[self.active_issue_draw_mode]
            self.statusBar().showMessage(
                "Drawing resumes when you return to the timeline image"
                if self.reference_mode else
                f"Draw one or more Candidate {self.active_issue_side} issue {shape_name}",
                5000,
            )

    def _issue_annotation_payload(
        self,
        side: str,
        shape: str,
        rect: QRectF,
        points: Sequence[QPointF] = (),
        *,
        category_override: str | None = None,
    ) -> dict[str, Any]:
        category = category_override or self._selected_issue_category(side)
        if not category:
            raise ValueError(f"Candidate {side} issue category is required")
        if self._blind_gas_locked and category not in BLIND_GAS_CATEGORIES:
            raise ValueError(f"Category {category!r} is unavailable in blind gas validation")
        semantics = review.annotation_category_semantics(category)
        phase_payload = annotation_phase_payload(category)
        model_summary: dict[str, Any] = {}
        model_assessment: dict[str, Any] = {}
        model_side = "A" if self._two_views_active() else side
        # A source-bound bottom check can hide predictions in an ordinary
        # correction session too. Human evidence must still be committed;
        # requesting a prediction here raises before append/autosave.
        prediction_locked = (
            self._blind_gas_locked or self._bottom_prediction_withheld(model_side)
        )
        if phase_payload and not prediction_locked:
            model_summary = self._model_phase_in_region(model_side, rect, points)
            model_assessment = assess_model_phase(
                str(model_summary.get("model_phase_class") or "unscored"),
                str(phase_payload.get("human_choice") or "not_assessable"),
            )
        shared = self._blind_gas_locked or self._two_views_active() or (
            semantics.get("interpretation_scope")
            == "source_image_shared_across_candidates"
        )
        elapsed_h = self._elapsed_for(model_side, self.frame_index)
        candidate_key = "candidate_a" if model_side == "A" else "candidate_b"
        candidate = self.current_pair.get(candidate_key) or {}
        reveal = (self.current_pair.get("reveal") or {}).get(model_side) or {}
        return {
            "id": uuid.uuid4().hex,
            "source": (
                "same_run_two_view_image_area_v1"
                if self._two_views_active()
                else
                "blind_gas_source_image_area_v1"
                if self._blind_gas_locked
                else
                ("shared_source_image_phase_area_v1"
                 if category in {
                     "confirmed_no_free_gas_in_selected_area",
                     "confirmed_no_aqueous_co2_in_selected_area",
                 }
                 else "shared_source_image_artifact_area_v2")
                if shared
                else "blind_candidate_comparison_issue_area_v2"
            ),
            "candidate": side,
            "panel_side": side if self._two_views_active() else None,
            "candidate_id": str(
                "" if self._blind_gas_locked else
                reveal.get("candidate_id") or candidate.get("candidate_id") or ""
            ),
            "params_sha256": str(
                "" if self._blind_gas_locked else
                reveal.get("params_sha256") or candidate.get("params_sha256") or ""
            ),
            "frame_index": int(self.frame_index),
            "elapsed_h": float(elapsed_h or 0.0),
            "category": category,
            "issue_key": category,
            "candidate_attribution": model_side,
            "annotation_scope": (
                "shared_source_image" if shared else "candidate_specific"
            ),
            "phase": "unspecified",
            "severity": "major",
            "confidence": "high",
            "contours_visible": bool(
                not prediction_locked and (
                    self._presented_layer(side) != "none" if self._split_views_active()
                    else self.contours.isChecked()
                )
            ),
            "display_mode": (
                "image_only" if prediction_locked else
                ("phase_map_overlay" if self._presented_layer(side) != "none"
                 else "image_only") if self._split_views_active() else
                "phase_map_on_black"
                if self.phase_map_only.isChecked()
                else ("phase_map_overlay" if self.contours.isChecked() else "image_only")
            ),
            "small_detection_highlight": bool(
                not prediction_locked and self.phase_map_only.isChecked()
                and self.highlight_small.isChecked()
            ),
            "small_detection_highlight_contract": "isolated_from_major_support_v2",
            "small_detection_max_area_fraction": ISOLATED_COMPONENT_MAX_AREA_FRACTION,
            "small_detection_min_separation_fraction": (
                ISOLATED_COMPONENT_MIN_SEPARATION_FRACTION
            ),
            "contour_layer": "none" if prediction_locked else self._presented_layer(side),
            "contour_opacity": (
                0.0 if prediction_locked else
                self.panel_opacities[("B" if side == "A" else "A") if self._two_view_swap_held else side] / 100.0
                if self._split_views_active() else float(self.contour_opacity.value()) / 100.0
            ),
            "local_baseline_search_preview": bool(self.local_reference_search.isChecked() and not prediction_locked),
            "photo_view": self._presented_photo_view(side),
            "photo_view_other": (
                self._presented_photo_view("B" if side == "A" else "A")
                if self._split_views_active() else None
            ),
            "session": BLIND_GAS_SESSION if self._blind_gas_session else None,
            "post_reveal": bool(self._blind_gas_session and not self._blind_gas_locked),
            "coordinate_space": "normalized_native_review_canvas_16x9",
            "shape": "freehand" if shape == "freehand" else "rectangle",
            "points": [[float(point.x()), float(point.y())] for point in points[:1000]],
            "x": float(rect.x()),
            "y": float(rect.y()),
            "width": float(rect.width()),
            "height": float(rect.height()),
            "comment": "",
            "category_cardinality": 1,
            "category_schema": (
                "shared_visible_phase_or_physical_issue_v3"
                if phase_payload
                else "single_issue_category_v1"
            ),
            "created_utc": utc_now(),
            **phase_payload,
            **model_summary,
            **model_assessment,
            **semantics,
        }

    def _append_issue_annotation(self, annotation: dict[str, Any]) -> None:
        if getattr(self,"_selected_annotation_binding",None) is not None:
            from darsia.fluidflower.phase_annotation_binding import stamp_camera_identity
            row=self.current_pair["frame_sources"][self.frame_index]
            annotation=stamp_camera_identity(annotation,row,self.current_run,
                dataset=self.current_pair.get("dataset"),selected_camera=True)
        side = str(annotation.get("candidate") or "A").upper()
        shared = self._is_shared_image_annotation(annotation)
        self.annotations.append(annotation)
        self._refresh_issue_annotations(selected_id=str(annotation["id"]))
        self._render_injection_points()
        self._schedule_save()
        shape_name = (
            "freehand area"
            if str(annotation.get("shape") or "") == "freehand"
            else "box"
        )
        self.statusBar().showMessage(
            (
                f"Added shared image artifact {shape_name} · drawing remains active"
                if shared
                else f"Added Candidate {side} issue {shape_name} · drawing remains active"
            ),
            4000,
        )

    def _add_issue_box(self, side: str, rect: QRectF) -> None:
        side = str(side).upper()
        if (
            self.current_pair is None
            or self.reference_mode
            or (self.displayed_base_keys[side] != self.base_keys[side]
                if self._split_views_active() else self.displayed_base_key != self.base_key)
            or self._editing_read_only
            or self.workspace.approval_only
            or side != self.active_issue_side
            or self.active_issue_draw_mode != "box"
            or rect.width() <= 0.0
            or rect.height() <= 0.0
        ):
            return
        self._append_issue_annotation(
            self._issue_annotation_payload(side, "rectangle", rect)
        )

    def _add_issue_freehand(
        self,
        side: str,
        points: Sequence[QPointF],
    ) -> None:
        side = str(side).upper()
        if (
            self.current_pair is None
            or self.reference_mode
            or (self.displayed_base_keys[side] != self.base_keys[side]
                if self._split_views_active() else self.displayed_base_key != self.base_key)
            or self._editing_read_only
            or self.workspace.approval_only
            or side != self.active_issue_side
            or self.active_issue_draw_mode not in {"freehand", "polygon"}
            or len(points) < 3
        ):
            return
        clean = list(points[:1000])
        xs = [float(point.x()) for point in clean]
        ys = [float(point.y()) for point in clean]
        rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        self._append_issue_annotation(
            self._issue_annotation_payload(side, "freehand", rect, clean)
        )

    def _selected_issue_annotation_index(self) -> int | None:
        item = self.issue_annotation_list.currentItem()
        if item is None:
            return None
        index = int(item.data(Qt.ItemDataRole.UserRole) or 0)
        return index if 0 <= index < len(self.annotations) else None

    def _select_issue_annotation_from_image(self, side: str, annotation_id: str) -> None:
        """Select a displayed region without changing frame, zoom, or draw mode."""

        if (
            side not in self.viewers
            or self.reference_mode
        ):
            return
        if annotation_id.startswith("history:"):
            self._select_history_annotation_from_image(annotation_id)
            return
        if not self.show_annotations.isChecked():
            return
        selected_index: int | None = None
        for index, annotation in enumerate(self.annotations):
            token = str(annotation.get("id") or f"index:{index}")
            if token == annotation_id:
                selected_index = index
                break
        if selected_index is None:
            return
        annotation = self.annotations[selected_index]
        if ReviewImageAssets._annotation_display_frame(self,annotation) != self.frame_index:
            return
        if self._blind_gas_locked and (
            annotation.get("session") != BLIND_GAS_SESSION
            or annotation.get("post_reveal")
        ):
            return
        if (
            not self._two_views_active()
            and not self._is_shared_image_annotation(annotation)
            and str(annotation.get("candidate") or "A").upper() != side
        ):
            return
        for row in range(self.issue_annotation_list.count()):
            item = self.issue_annotation_list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole) or 0) != selected_index:
                continue
            if self.workspace.approval_only:
                self.decision_scroll.setVisible(True)
                self.outer_splitter.setSizes([185, 1120, 430])
            self.issue_annotation_list.setCurrentItem(item)
            self.issue_annotation_list.scrollToItem(item)
            if self.workspace.approval_only:
                self.decision_scroll.ensureWidgetVisible(self.issue_annotation_list)
            self._issue_annotation_selection_changed(item, None)
            return

    def _select_history_annotation_from_image(self, annotation_id: str) -> None:
        if self._blind_gas_locked or not self.show_review_history.isChecked():
            return
        try:
            index = int(annotation_id.partition(":")[2])
        except ValueError:
            return
        entries = self._history_run().get("annotations") or []
        if not 0 <= index < len(entries):
            return
        entry = entries[index]
        value = entry.get("annotation") or {}
        if not entry.get("displayable") or value.get("frame_index") != self.frame_index:
            return
        for row in range(self.history_list.count()):
            item = self.history_list.item(row)
            data = item.data(Qt.ItemDataRole.UserRole)
            if not data or data[0] != "annotation" or data[1] != entry:
                continue
            self.history_list.setCurrentItem(item)
            self.history_list.scrollToItem(item)
            self._review_history_opened(item, focus_image=False)
            return

    def _refresh_issue_annotations(self, *, selected_id: str = "") -> None:
        labels = dict(self.ISSUE_CATEGORY_LABELS)
        self.issue_annotation_list.blockSignals(True)
        self.issue_annotation_list.clear()
        selected_item: QListWidgetItem | None = None
        ordered = sorted(
            range(len(self.annotations)),
            key=lambda index: (
                (ReviewImageAssets._annotation_display_frame(self,self.annotations[index]) if ReviewImageAssets._annotation_display_frame(self,self.annotations[index]) is not None else self.frame_count),
                str(self.annotations[index].get("candidate") or "A"),
                str(self.annotations[index].get("created_utc") or ""),
            ),
        )
        for index in ordered:
            annotation = self.annotations[index]
            if self._blind_gas_locked and (
                annotation.get("session") != BLIND_GAS_SESSION
                or annotation.get("post_reveal")
            ):
                continue
            side = str(annotation.get("candidate") or "A").upper()
            shared = self._is_shared_image_annotation(annotation)
            frame = ReviewImageAssets._annotation_display_frame(self,annotation)
            category = str(
                annotation.get("issue_key")
                or annotation.get("category")
                or "isolated_noise"
            )
            elapsed = annotation.get("elapsed_h")
            elapsed_text = (
                f" · {float(elapsed):.2f} h" if elapsed is not None else ""
            )
            reusable = category not in {
                "unsupported_free_gas_classification",
                "unsupported_aqueous_co2_classification",
            } and (
                str(annotation.get("training_scope") or "")
                == "source_image_all_candidate_models"
                or review.annotation_category_semantics(category).get(
                    "training_scope"
                )
                == "source_image_all_candidate_models"
            )
            scope_text = (
                "Shared image"
                if shared
                else (
                    f"Candidate {side} trigger · reusable image evidence"
                    if reusable
                    else f"Candidate {side}"
                )
            )
            assessment = str(annotation.get("model_assessment") or "")
            assessment_text = (
                f" · model: {assessment.replace('_', ' ')}"
                if annotation.get("human_choice") and assessment
                else ""
            )
            item = QListWidgetItem(
                f"{scope_text} · " +
                (f"Frame {frame + 1}{elapsed_text}\n" if frame is not None else f"Original Studio probe · outside selected timeline{elapsed_text}\n") +
                f"{labels.get(category, category)}{assessment_text}"
            )
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setForeground(QColor(self.ISSUE_COLORS.get(category, "#ffd166")))
            self.issue_annotation_list.addItem(item)
            if selected_id and str(annotation.get("id") or "") == selected_id:
                selected_item = item
        if selected_item is not None:
            self.issue_annotation_list.setCurrentItem(selected_item)
        self.issue_annotation_list.blockSignals(False)
        has_selection = self.issue_annotation_list.currentItem() is not None
        editable = not self._editing_read_only and not self.workspace.approval_only
        self.update_issue_box.setEnabled(has_selection and editable)
        self.delete_issue_box.setEnabled(has_selection and editable)

    def _issue_annotation_opened(self, item: QListWidgetItem) -> None:
        index = int(item.data(Qt.ItemDataRole.UserRole) or 0)
        if not 0 <= index < len(self.annotations):
            return
        annotation = self.annotations[index]
        side = str(annotation.get("candidate") or "A").upper()
        category = str(
            annotation.get("issue_key")
            or annotation.get("category")
            or "isolated_noise"
        )
        shared = self._is_shared_image_annotation(annotation)
        self._set_selected_issue_category(side, category)
        editable = not self._editing_read_only and not self.workspace.approval_only
        self.update_issue_box.setEnabled(editable)
        self.delete_issue_box.setEnabled(editable)
        target_frame=ReviewImageAssets._annotation_display_frame(self,annotation)
        if target_frame is None:
            self.update_issue_box.setEnabled(False)
            self.statusBar().showMessage("This original Studio probe is not in the selected timeline; its annotation is preserved",8000)
            return
        self._set_frame(target_frame)
        annotation=ReviewImageAssets._annotation_display_payload(self,annotation)
        target_viewers = (
            tuple(self.viewers.values())
            if self._two_views_active() or shared
            else tuple(
                viewer
                for viewer in (self.viewers.get(side),)
                if viewer is not None
            )
        )
        for viewer in target_viewers:
            viewer.focus_normalized_rect(
                QRectF(
                    float(annotation.get("x") or 0.0),
                    float(annotation.get("y") or 0.0),
                    max(0.002, float(annotation.get("width") or 0.0)),
                    max(0.002, float(annotation.get("height") or 0.0)),
                )
            )

    def _issue_annotation_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        has_selection = current is not None
        self.update_issue_box.setEnabled(
            has_selection and not self._editing_read_only and not self.workspace.approval_only
        )
        self.delete_issue_box.setEnabled(
            has_selection and not self._editing_read_only and not self.workspace.approval_only
        )
        if current is None:
            return
        index = int(current.data(Qt.ItemDataRole.UserRole) or 0)
        if not 0 <= index < len(self.annotations):
            return
        annotation = self.annotations[index]
        side = str(annotation.get("candidate") or "A").upper()
        category = str(
            annotation.get("issue_key")
            or annotation.get("category")
            or "isolated_noise"
        )
        if ReviewImageAssets._annotation_display_frame(self,annotation) is None:
            self.update_issue_box.setEnabled(False)
        self._set_selected_issue_category(side, category)

    def _update_selected_issue_box(self) -> None:
        if self._editing_read_only or self.workspace.approval_only:
            return
        index = self._selected_issue_annotation_index()
        if index is None:
            return
        annotation = self.annotations[index]
        if ReviewImageAssets._annotation_display_frame(self,annotation) is None:
            return
        if self._blind_gas_locked and (
            annotation.get("session") != BLIND_GAS_SESSION
            or annotation.get("post_reveal")
        ):
            return
        side = str(annotation.get("candidate") or "A").upper()
        category = self._selected_issue_category(side)
        if not category:
            self.statusBar().showMessage(
                f"Select one Candidate {side} category before applying",
                5000,
            )
            return
        if self._blind_gas_locked and category not in BLIND_GAS_CATEGORIES:
            return
        semantics = review.annotation_category_semantics(category)
        shared = self._blind_gas_locked or self._two_views_active() or (
            semantics.get("interpretation_scope")
            == "source_image_shared_across_candidates"
        )
        annotation["category"] = category
        annotation["issue_key"] = category
        annotation["candidate_attribution"] = "A" if self._two_views_active() else side
        annotation["annotation_scope"] = (
            "shared_source_image" if shared else "candidate_specific"
        )
        if self._two_views_active():
            annotation["source"] = "same_run_two_view_image_area_v1"
        phase_keys = {
            "human_choice",
            "phase_label_schema",
            "phase_class",
            "phase_dominance",
            "phase_components",
            "training_eligible",
            "model_phase_class",
            "model_phase_distribution",
            "model_phase_sample_count",
            "model_phase_source",
            "model_assessment",
            "missing_phase_components",
            "unsupported_phase_components",
        }
        for key in phase_keys:
            annotation.pop(key, None)
        phase_payload = annotation_phase_payload(category)
        # Updating a blind human choice needs the same no-scoring path as a
        # newly drawn area; retain its phase payload even while maps are hidden.
        annotation.update(phase_payload)
        model_side = "A" if self._two_views_active() else side
        prediction_locked = (
            self._blind_gas_locked or self._bottom_prediction_withheld(model_side)
        )
        if phase_payload and not prediction_locked:
            points = [
                point
                for point in (
                    normalized_point(value) for value in annotation.get("points") or []
                )
                if point is not None
            ]
            rect = QRectF(
                float(annotation.get("x") or 0.0),
                float(annotation.get("y") or 0.0),
                float(annotation.get("width") or 0.0),
                float(annotation.get("height") or 0.0),
            )
            model_summary = self._model_phase_in_region(
                model_side, rect, points
            )
            annotation.update(model_summary)
            annotation.update(
                assess_model_phase(
                    str(model_summary.get("model_phase_class") or "unscored"),
                    str(phase_payload["human_choice"]),
                )
            )
        semantic_keys = {
            key
            for values in review.ANNOTATION_CATEGORY_SEMANTICS.values()
            for key in values
        }
        for key in semantic_keys:
            annotation.pop(key, None)
        annotation.update(semantics)
        annotation["category_cardinality"] = 1
        annotation["category_schema"] = (
            "shared_visible_phase_or_physical_issue_v3"
            if phase_payload
            else "single_issue_category_v1"
        )
        annotation["updated_utc"] = utc_now()
        if prediction_locked:
            annotation["contours_visible"] = False
            annotation["display_mode"] = "image_only"
            annotation["contour_layer"] = "none"
            annotation["contour_opacity"] = 0.0
            annotation["small_detection_highlight"] = False
            annotation["local_baseline_search_preview"] = False
        if self._blind_gas_session or self._split_views_active():
            annotation["photo_view"] = self._presented_photo_view(side)
            annotation["photo_view_other"] = (
                self._presented_photo_view("B" if side == "A" else "A")
                if self._split_views_active() else None
            )
        if self._blind_gas_session:
            annotation["session"] = BLIND_GAS_SESSION
            annotation["post_reveal"] = not self._blind_gas_locked
            if self._blind_gas_locked:
                annotation["source"] = "blind_gas_source_image_area_v1"
                annotation["contours_visible"] = False
                annotation["display_mode"] = "image_only"
                annotation["contour_layer"] = "none"
                annotation["contour_opacity"] = 0.0
                for key in tuple(annotation):
                    if key.startswith("model_") or key in (
                        "missing_phase_components", "unsupported_phase_components"
                    ):
                        annotation.pop(key, None)
        selected_id = str(annotation.get("id") or "")
        self._refresh_issue_annotations(selected_id=selected_id)
        self._render_injection_points()
        self._schedule_save()

    def _delete_selected_issue_box(self) -> None:
        if self._editing_read_only or self.workspace.approval_only:
            return
        index = self._selected_issue_annotation_index()
        if index is None:
            return
        annotation = self.annotations[index]
        if self._blind_gas_locked and (
            annotation.get("session") != BLIND_GAS_SESSION
            or annotation.get("post_reveal")
        ):
            return
        del self.annotations[index]
        self._refresh_issue_annotations()
        self._render_injection_points()
        self._schedule_save()

    def _elapsed_for(self, side: str, index: int) -> float | None:
        if self.current_pair is None:
            return None
        candidate = self.current_pair[
            "candidate_a" if side == "A" else "candidate_b"
        ]
        values = candidate.get("elapsed_hours") or []
        return float(values[index]) if index < len(values) else None

    def _is_pre_injection_timeline_frame(self, index: int) -> bool:
        if self.current_pair is None:
            return False
        sources = self.current_pair.get("frame_sources") or []
        return bool(
            0 <= int(index) < len(sources)
            and str(sources[int(index)].get("selection_status") or "")
            == "pre_injection_negative_control"
        )

    def _metrics_for(self, side: str) -> Mapping[str, float]:
        if self.current_pair is None:
            return {}
        if self._bottom_prediction_withheld(side):
            return {}
        candidate = self.current_pair[
            "candidate_a" if side == "A" else "candidate_b"
        ]
        rows = candidate.get("frame_metrics") or []
        return rows[self.frame_index] if self.frame_index < len(rows) else {}

    @staticmethod
    def _format_mass(value: Any) -> str:
        try:
            return f"{1000.0 * float(value):.3f} g"
        except (TypeError, ValueError):
            return "--"

    def _update_metrics(self) -> None:
        if self._blind_gas_locked:
            for label in self.metric_labels.values():
                label.clear()
                label.hide()
            return
        if self.reference_mode:
            for label in self.metric_labels.values():
                label.setText(
                    f"{self._reference_description()} · no candidate score or phase verdict"
                )
            return
        for side in ("A", "B"):
            if self._bottom_prediction_withheld(side):
                self.metric_labels[side].setText("Bottom-strip blind check · predictions and metrics withheld")
                continue
            candidate = (self.current_pair or {}).get(
                "candidate_a" if side == "A" else "candidate_b", {})
            if candidate.get("phase_model_only"):
                self.metric_labels[side].setText(
                    "Visible 2D phase interpretation; concentration and mass are not evaluated here"
                )
                continue
            row = self._metrics_for(side)
            gas_score = row.get("gas_probability_mean")
            if gas_score is not None:
                values = [
                    f"2D mean free-CO2 score {float(gas_score):.4f}",
                ]
                support = row.get("support_fraction_p050")
                if support is not None:
                    values.append(
                        f"2D area with free-CO2 score >= 0.5: {100.0 * float(support):.2f}%"
                    )
                aqueous = row.get("aqueous_probability_mean")
                if aqueous is not None:
                    values.append(
                        f"2D mean aqueous-CO2 score {float(aqueous):.4f}"
                    )
                presence = row.get("co2_presence_probability_mean")
                if presence is not None:
                    values.append(
                        f"2D mean CO2-presence score {float(presence):.4f}"
                    )
                self.metric_labels[side].setText(" · ".join(values))
                continue
            injected = row.get("injected_mass_kg")
            total = row.get("detected_total_mass_kg")
            ratio = "--"
            if injected not in (None, 0) and total is not None:
                ratio = f"{float(total) / float(injected):.3f}"
            self.metric_labels[side].setText(
                "Protocol cumulative injected mass for the full rig "
                + self._format_mass(injected)
                + " · 3D-extrapolated detected mass "
                + self._format_mass(total)
                + " · Aqueous "
                + self._format_mass(row.get("detected_aqueous_mass_kg"))
                + " · Free CO2 "
                + self._format_mass(row.get("detected_gas_mass_kg"))
                + f" · detected/injected {ratio}"
            )

    def _set_frame(self, index: int) -> None:
        if self.current_pair is None or self.frame_count <= 0:
            return
        target = max(0, min(self.frame_count - 1, int(index)))
        leaving_reference = self.reference_mode
        if leaving_reference:
            self.reference_mode = False
            self.active_injection_reference = None
            self._sync_reference_buttons()
            self.contours.setEnabled(True)
            self.verdict.setEnabled(not self._editing_read_only)
        if target == self.frame_index and self.base_key:
            if leaving_reference:
                self.base_key = ""
                self._request_frame()
            return
        self.frame_index = target
        self.fit_next_image = False
        self._request_frame()

    def _prefetch_nearby_frames(self) -> None:
        if self.optical_cleanup.active:
            return
        focus = getattr(self.image_cache, "focus", None)
        if callable(focus):
            focus([self.base_key, *self.base_keys.values(), *self.overlay_keys.values(), *self.plot_keys.values()])
        quick = []
        if not self.speed_loop_held:
            quick = self._prefetch_quick_photos()
        if self._split_views_active():
            nearby: list[DeferredImage] = list(quick or [])
            offsets = tuple(range(1, self._playback_prefetch_count())) if self.speed_loop_held else (1, 2, -1)
            for offset in offsets:
                index = self.frame_index + offset
                if self.speed_loop_held and self.frame_count:
                    index %= self.frame_count
                if 0 <= index < self.frame_count:
                    nearby.extend(self._playback_frame_sources(index))
            prefetch = getattr(self.image_cache, "prefetch_window", self.image_cache.prefetch)
            prefetch(nearby)
            return
        nearby: list[DeferredImage] = list(quick or [])
        offsets = tuple(range(1, self._playback_prefetch_count())) if self.speed_loop_held else (1, 2, 3, -1, -2)
        for offset in offsets:
            value = self.frame_index + offset
            if self.speed_loop_held and self.frame_count:
                value %= self.frame_count
            elif not 0 <= value < self.frame_count:
                continue
            nearby.append(self._frame_asset(value))
            if self.contours.isChecked():
                nearby.extend(
                    self._overlay_asset(side, value)
                    for side in self._playback_candidate_sides()
                )
        prefetch = getattr(self.image_cache, "prefetch_window", self.image_cache.prefetch)
        prefetch(nearby)

    def _playback_prefetch_count(self) -> int:
        revision = self._rig_display_revision() or {}
        shape = revision.get("phase_shape_hw")
        budget = getattr(self.image_cache, "max_bytes", 3 << 30)
        if not shape:
            return 6
        pixels = int(shape[0]) * int(shape[1])
        sources = {self.image_cache.key(source): source for source in self._playback_frame_sources(self.frame_index)}
        # A prepared overlay keeps its scoring image, two display variants and
        # two masks. RAW photographs also retain three support masks.
        size = sum(pixels * (14 if source.display_layer is not None else
                            7 if source.identity.startswith("rig-raw-display:") else 4)
                   for source in sources.values())
        return max(2, min(6, int(budget) // max(1, size)))

    def _playback_frame_sources(self, index: int) -> list[DeferredImage]:
        """Request both independent panes, including their prepared overlays."""
        if not self._split_views_active():
            sources = [self._frame_asset(index)]
            if self.contours.isChecked() and not self.reference_mode:
                sources.extend(self._overlay_asset(side, index) for side in self._playback_candidate_sides())
            return sources
        sources = []
        sides = self._playback_candidate_sides() if self.fullscreen_loop_active else ("A", "B")
        for side in sides:
            try:
                sources.append(self._two_view_photo_asset(side, index))
            except ValueError:
                continue
            source_side = "A" if self._two_views_active() else side
            if self.reference_mode or self._bottom_prediction_withheld(source_side):
                continue
            layer = self._effective_panel_layer(side)
            filled_only = layer == "none" and self._filled_display_enabled(side) and self._filled_display_available()
            if layer in (SAVED_PROVENANCE_LAYERS | {"optical"}) or filled_only:
                sources.append(self._overlay_asset(source_side, index, layer if layer.startswith("removed_") else "all"))
            elif layer in {"physics", "difference"} and self._physics_layer_available:
                sources.append(self._physics_overlay_asset(source_side, index, layer, "all"))
        return sources

    def _prefetch_quick_photos(self) -> list[DeferredImage]:
        if self.active_injection_reference is not None and self._saved_rig_reference_index() is None:
            return []
        photos = []
        for view in ("natural", "original"):
            try:
                if self.reference_mode and self._saved_rig_reference_index() is not None:
                    photos.append(self._active_reference_asset(view=view))
                else:
                    photos.append(self._photo_view_asset(None if self.reference_mode else self.frame_index,
                                                        view, reference=self.reference_mode))
            except ValueError:
                pass
        self.image_cache.prefetch(photos)
        return photos

    def _show_pre_injection_reference(self) -> None:
        if self.current_pair is not None and self.reference.isEnabled():
            self._set_reference_view(True)

    def _step_frame(self, delta: int) -> None:
        if self.current_pair is None or self.frame_count <= 0 or delta == 0:
            return
        self._set_frame(self.frame_index + (1 if delta > 0 else -1))

    def _speed_rate_changed(self, fps: int) -> None:
        self.speed_timer.setInterval(max(1, int(round(1000.0 / max(1, fps)))))

    def _fullscreen_chrome_widgets(self) -> tuple[QWidget, ...]:
        return (
            self.header_widget,
            self.comparison_notice,
            self.run_panel,
            self.plot_container,
            self.guide_controls_widget,
            self.contour_legend,
            self.view_controls_widget,
            self.decision_scroll,
        )

    def _toggle_fullscreen_loop(self, side: str | None = None) -> None:
        requested_side = side if side in {"A", "B"} else "A"
        if (
            self.fullscreen_loop_active
            and requested_side == self.fullscreen_candidate_side
        ):
            self._exit_fullscreen_loop()
        elif self.fullscreen_loop_active:
            self._set_fullscreen_candidate(requested_side)
        else:
            self._enter_fullscreen_loop(requested_side)

    def _set_fullscreen_candidate(self, side: str) -> None:
        if side not in {"A", "B"}:
            raise ValueError(f"Unknown candidate side: {side}")
        self.fullscreen_candidate_side = side
        for candidate_side, card in self.candidate_cards.items():
            card.setVisible(candidate_side == side)
        for candidate_side, button in self.fullscreen_buttons.items():
            if candidate_side == side:
                if self.comparison_mode == "single_candidate":
                    button.setText("Exit full-screen model")
                    button.setToolTip(
                        "Exit full-screen model playback and restore validation controls."
                    )
                else:
                    button.setText(f"Exit full-screen {side}")
                    button.setToolTip(
                        f"Exit Candidate {side} full-screen playback and restore both panels."
                    )
            else:
                button.setText(f"Switch to full-screen {candidate_side}")
                button.setToolTip(
                    f"Switch full-screen playback directly to Candidate {candidate_side}."
                )
        self._request_overlays()

    def _enter_fullscreen_loop(self, side: str = "A") -> None:
        if self.current_pair is None or self.frame_count <= 0:
            return
        if self.speed_loop_held:
            self._stop_speed_loop()
        self.fullscreen_loop_active = True
        self.fullscreen_was_maximized = self.isMaximized()
        self.fullscreen_restore_visibility = [
            (widget, widget.isVisible())
            for widget in self._fullscreen_chrome_widgets()
        ]
        for widget, _visible in self.fullscreen_restore_visibility:
            widget.hide()
        self._set_fullscreen_candidate(side)
        self.speed_loop.setEnabled(False)
        self.showFullScreen()
        self._start_speed_loop()

    def _exit_fullscreen_loop(self) -> None:
        if not self.fullscreen_loop_active:
            return
        self._stop_speed_loop()
        self.fullscreen_loop_active = False
        self.fullscreen_candidate_side = None
        for card in self.candidate_cards.values():
            card.show()
        for widget, visible in self.fullscreen_restore_visibility:
            widget.setVisible(visible)
        self.fullscreen_restore_visibility = []
        for side, button in self.fullscreen_buttons.items():
            button.setText(f"Full-screen {side}")
            button.setToolTip(
                f"Play only Candidate {side} in full screen. Phase-map and layer "
                "controls remain available below the image. Press Esc to exit."
            )
        self.speed_loop.setEnabled(True)
        if self.fullscreen_was_maximized:
            self.showMaximized()
        else:
            self.showNormal()
        self._apply_comparison_mode(self.comparison_mode)
        self._request_overlays()

    def keyPressEvent(self, event: Any) -> None:
        if (
            self.fullscreen_loop_active
            and event.key() == Qt.Key.Key_Escape
        ):
            self._exit_fullscreen_loop()
            event.accept()
            return
        super().keyPressEvent(event)

    def _start_speed_loop(self) -> None:
        if self.current_pair is None or self.frame_count <= 0:
            return
        self.speed_return_frame_index = int(self.frame_index)
        self.speed_return_reference_mode = bool(self.reference_mode)
        self.speed_return_injection_reference = self.active_injection_reference
        self.speed_loop_held = True
        self.speed_loop.setText("Release: stop loop")
        self._speed_rate_changed(self.speed_fps.value())
        self._set_frame(0)
        self._prefetch_speed_window()
        self.speed_timer.start()

    def _stop_speed_loop(self) -> None:
        was_active = bool(self.speed_loop_held)
        return_index = self.speed_return_frame_index
        return_reference = bool(self.speed_return_reference_mode)
        return_injection = self.speed_return_injection_reference
        self.speed_return_injection_reference = None
        self.speed_loop_held = False
        self.speed_return_frame_index = None
        self.speed_return_reference_mode = False
        if hasattr(self, "speed_timer"):
            self.speed_timer.stop()
        if hasattr(self, "speed_loop"):
            self.speed_loop.setText("Hold: speed loop")
        if (
            was_active
            and return_index is not None
            and self.current_pair is not None
            and self.frame_count > 0
        ):
            self.frame_index = max(0, min(self.frame_count - 1, return_index))
            if return_reference and return_injection is not None:
                self._set_reference_view(True, return_injection)
            elif return_reference and self.reference.isEnabled():
                self._show_pre_injection_reference()
            else:
                self._set_reference_view(False)

    def _prefetch_speed_window(self) -> None:
        if self.optical_cleanup.active:
            return
        if self.current_pair is None or self.frame_count <= 0:
            return
        sources: list[DeferredImage] = []
        for offset in range(0, min(self._playback_prefetch_count(), self.frame_count)):
            index = (self.frame_index + offset) % self.frame_count
            sources.extend(self._playback_frame_sources(index))
        getattr(self.image_cache, "prefetch_window", self.image_cache.prefetch)(sources)

    def _speed_frame_assets_ready(self) -> bool:
        if self._split_views_active():
            return self._two_view_ready()
        if not self.base_key or self.displayed_base_key != self.base_key:
            return False
        if not self.contours.isChecked() or self.reference_mode:
            return True
        return all(
            bool(self.overlay_keys.get(side))
            and not self.viewers[side].overlay_item.pixmap().isNull()
            for side in self._playback_candidate_sides()
        )

    def _advance_speed_loop(self) -> None:
        if not self.speed_loop_held or self.frame_count <= 0:
            self._stop_speed_loop()
            return
        if not self._speed_frame_assets_ready():
            return
        if self.frame_index + 1 < self.frame_count:
            self._set_frame(self.frame_index + 1)
        else:
            self._set_frame(0)

    def _sync_view(self, source_side: str) -> None:
        if (
            self.syncing_views
            or not self.sync_views.isChecked()
            or self.viewers[source_side].base_item.pixmap().isNull()
        ):
            return
        target_side = "B" if source_side == "A" else "A"
        source = self.viewers[source_side]
        target = self.viewers[target_side]
        if target.base_item.pixmap().isNull():
            return
        visible = source.visible_normalized_rect()
        self.syncing_views = True
        try:
            target.setTransform(source.transform())
            target.center_on_normalized(visible.center())
        finally:
            self.syncing_views = False

    def _fit_views(self) -> None:
        self.syncing_views = True
        try:
            for viewer in self.viewers.values():
                viewer.fit_image()
        finally:
            self.syncing_views = False

    def _native_views(self) -> None:
        self.syncing_views = True
        try:
            for viewer in self.viewers.values():
                viewer.native_image_zoom()
        finally:
            self.syncing_views = False

    def _zoom_views(self, factor: float) -> None:
        self.syncing_views = True
        try:
            for viewer in self.viewers.values():
                viewer.zoom(factor)
        finally:
            self.syncing_views = False

    def _flags(self) -> dict[str, list[str]]:
        localized = self._annotation_flags(self.annotations)
        order = [key for key, _label in self.FLAG_LABELS]
        return {
            side: [
                key
                for key in order
                if key
                in (
                    localized[side]
                    | set(self.legacy_unlocalized_flags.get(side, set()))
                )
            ]
            for side in ("A", "B")
        }

    def _save_request(self) -> dict[str, Any]:
        current_pair = getattr(self, "current_pair", None) or {}
        comparison_mode = str(
            current_pair.get("comparison_mode") or "pair"
        )
        saved_frame_choices = getattr(self, "frame_choices", None)
        frame_count = int(getattr(self, "frame_count", 0) or 0)
        frame_choice_updated = getattr(self, "frame_choice_updated", {})
        frame_choices = [
            {
                "frame_index": index,
                "elapsed_h": (
                    self._elapsed_for("A", index)
                    if hasattr(self, "_elapsed_for")
                    else None
                ),
                "choice": saved_frame_choices[index],
                "updated_utc": frame_choice_updated.get(index) or utc_now(),
            }
            for index in sorted(saved_frame_choices or {})
        ]
        annotation_binding=getattr(self,"_selected_annotation_binding",None)
        if annotation_binding is not None:
            from darsia.fluidflower.phase_annotation_binding import stamp_camera_identity
            originals=(getattr(self,"current_decision",{}) or {}).get("frame_choices") or []
            kept=[]
            for item in originals:
                target=ReviewImageAssets._annotation_display_frame(self,item)
                if target is None or saved_frame_choices.get(target)==item.get("choice"):
                    kept.append(deepcopy(item))
            unchanged={ReviewImageAssets._annotation_display_frame(self,item) for item in kept}
            frame_choices=kept+[stamp_camera_identity(item,current_pair["frame_sources"][item["frame_index"]],self.current_run,
                dataset=current_pair.get("dataset"),selected_camera=True)
                for item in frame_choices if item["frame_index"] not in unchanged]
        aggregate_choice = (
            review.aggregate_frame_choice(
                frame_choices,
                frame_count=frame_count,
                comparison_mode=comparison_mode,
            )
            if saved_frame_choices is not None
            else str(self.verdict.currentData() or "unresolved")
        )
        current_decision = getattr(self, "current_decision", {}) or {}
        return {
            "run": self.current_run,
            "reviewer": self.reviewer.text().strip(),
            "choice": aggregate_choice,
            "comparison_mode": comparison_mode,
            "frame_choice_schema": review.FRAME_CHOICE_SCHEMA,
            "frame_choices": frame_choices,
            "legacy_run_choice": str(
                current_decision.get("legacy_run_choice")
                or (
                    current_decision.get("choice")
                    if not current_decision.get("frame_choices")
                    else ""
                )
                or ""
            ),
            "flags": self._flags(),
            "annotations": deepcopy(self.annotations),
            "reference_guides": deepcopy(
                getattr(self, "reference_guides", [])
            ),
            "phase_quiz_answers": [],
            "phase_quiz_selected_ids": [],
            "phase_quiz_rounds": [],
            "phase_quiz_per_candidate_limit": 0,
            "notes": self.notes.toPlainText().strip(),
            "updated_utc": utc_now(),
        }

    def _schedule_save(self, _value: Any = None) -> None:
        if self._editing_read_only or self.loading_decision or not self.current_run:
            return
        request = self._save_request()
        self._comparison_generation += 1
        revision = (self._comparison_generation, request)
        self._comparison_latest_requests[self.current_run] = revision
        self._comparison_unprotected_requests[self.current_run] = revision
        self.save_dirty = True
        try:
            self._protect_comparison_request(self.current_run, *revision)
        except (OSError, ValueError, TypeError) as exc:
            self._comparison_local_error = True
            self._set_save_status("error", "Local recovery failed · changes remain in this window")
            self.statusBar().showMessage(f"{type(exc).__name__}: {exc}", 12000)
            self.save_timer.start(self._comparison_save_retry_ms)
        else:
            self._comparison_local_error = bool(self._comparison_unprotected_requests)
            self._set_save_status("pending", "Protected locally · NAS save queued")
            self.save_timer.start(350)
        self._populate_runs()

    def _comparison_local_draft(self, run):
        try:
            payload = json.loads((self._comparison_recovery_dir / f"{run}.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return None
        if (payload.get("format") == "comparison-local-recovery-v1"
                and payload.get("pending") is True
                and isinstance(payload.get("request"), Mapping)):
            return dict(payload["request"])
        return None

    def _protect_comparison_request(self, run, generation, request):
        """Local durable evidence precedes queuing any canonical NAS mutation."""
        payload = dict(format="comparison-local-recovery-v1", pending=True,
            generation=generation, session_root=str(self.workspace.output_dir),
            recorded_utc=utc_now(), request=request)
        atomic_json(self._comparison_recovery_dir / f"{run}.json", payload)
        append_bounded_snapshot_journal(
            self._comparison_recovery_dir / f"{run}.history.jsonl", payload)
        self._comparison_unprotected_requests.pop(run, None)
        self._comparison_pending_saves[run] = (generation, deepcopy(request))
        self._comparison_pending_saves.move_to_end(run)

    def _save_now(self) -> None:
        self._schedule_save()
        self.save_timer.stop()
        self._flush_save()

    def _flush_save(self) -> None:
        # Retry any failed local writes before allowing their NAS snapshots to
        # start; older in-flight requests may finish but cannot acknowledge them.
        for run, (generation, request) in list(self._comparison_unprotected_requests.items()):
            try:
                self._protect_comparison_request(run, generation, request)
            except (OSError, TypeError, ValueError) as exc:
                self._comparison_local_error = True
                self._set_save_status("error", "Local recovery failed · keep this window open")
                self.statusBar().showMessage(f"{type(exc).__name__}: {exc}", 12000)
                self.save_timer.start(self._comparison_save_retry_ms)
                return
        self._comparison_local_error = False
        if self._comparison_save_task is not None or not self._comparison_pending_saves:
            return
        if self._editing_read_only:
            return
        run, (generation, request) = self._comparison_pending_saves.popitem(last=False)
        task = ComparisonSaveTask(generation, run, self.store, request,
            self.workspace.output_dir / "comparison_recovery")
        self._comparison_save_task = task
        task.signals.finished.connect(self._comparison_save_finished)
        self.save_dirty = True
        self._set_save_status("saving", f"Protected locally · saving {run.upper()} to NASâ€¦")
        self._comparison_save_pool.start(task)

    def _comparison_save_finished(self, generation, run, payload, error) -> None:
        task = self._comparison_save_task
        if task is None or task.generation != generation:
            return
        self._comparison_save_task = None
        latest = self._comparison_latest_requests.get(run)
        if error:
            # A later snapshot always wins; an old failed write is never put
            # ahead of newer edits to the same run.
            if run not in self._comparison_pending_saves and latest is not None:
                if run not in self._comparison_unprotected_requests:
                    self._comparison_pending_saves[run] = latest
            self._editing_read_only = bool(payload.get("read_only")) or self._editing_read_only
            self.save_dirty = True
            self._set_save_status("error", "Protected locally · NAS save failed; retrying")
            self.statusBar().showMessage(error, 12000)
            if self._editing_read_only:
                self._apply_read_only_profile()
            elif not self._comparison_closing:
                self.save_timer.start(self._comparison_save_retry_ms)
            if self._comparison_closing and not self._comparison_unprotected_requests:
                self._comparison_close_with_recovery = True
                QTimer.singleShot(0, self.close)
            return
        record = deepcopy(payload.get("record") or {})
        self._comparison_saved_records[run] = record
        self._comparison_saved_generations[run] = generation
        self._run_decisions[run] = record
        if latest is not None and latest[0] == generation:
            self._comparison_latest_requests.pop(run, None)
            if self.current_run == run:
                self.current_decision = deepcopy(record)
            try:
                atomic_json(self._comparison_recovery_dir / f"{run}.json",
                    dict(format="comparison-local-recovery-v1", pending=False,
                        generation=generation, session_root=str(self.workspace.output_dir),
                        recorded_utc=utc_now(), request=task.request))
            except (OSError, TypeError, ValueError) as exc:
                self.statusBar().showMessage("NAS saved; local recovery acknowledgement failed: " + str(exc), 12000)
        self.save_dirty = bool(self._comparison_pending_saves or self._comparison_unprotected_requests)
        self._populate_runs()
        if self.save_dirty:
            self._set_save_status("pending", "Protected locally · newer changes queued")
            self.save_timer.start(0 if self._comparison_closing else 50)
        else:
            self._set_save_status("saved", "All changes saved · NAS synced")
            if self._comparison_closing:
                QTimer.singleShot(0, self.close)

    def closeEvent(self, event: Any) -> None:
        self._stop_speed_loop()
        self.save_timer.stop()
        if not self._comparison_close_with_recovery:
            self._flush_save()
        if self._comparison_local_error:
            event.ignore()
            return
        if self._editing_read_only and self._comparison_save_task is None:
            self._comparison_close_with_recovery = True
        if not self._comparison_close_with_recovery and (
                self._comparison_save_task is not None or self._comparison_pending_saves):
            self._comparison_closing = True
            self.centralWidget().setEnabled(False)
            self._set_save_status("saving", "Protected locally · finishing NAS save before closingâ€¦")
            event.ignore()
            return
        self._comparison_closing = True
        self._comparison_load_generation += 1
        for pending in self._comparison_load_tasks.values():
            pending.cancelled.set()
        for task in self.presentation_tasks.values():
            task.cancelled.set()
        if hasattr(self, "reference_prepare_timer"):
            self.reference_prepare_timer.stop()
        self.image_cache.cancel_queued_reference_preparations()
        super().closeEvent(event)
        if self._blind_gas_session and event.isAccepted():
            QApplication.instance().removeEventFilter(self)


def workspace_window_class(
    workspace: ReviewWorkspace,
) -> type[CalibrationReviewWindow] | type[BlindModelComparisonWindow]:
    """Select the interaction surface without mixing review purposes."""

    return (
        BlindModelComparisonWindow
        if workspace.key in {"current_comparison", "comparison"}
        else CalibrationReviewWindow
    )


class ReviewHubWindow(QMainWindow):
    """Purpose selector that keeps training and evaluations visibly separate."""

    def __init__(
        self,
        args: argparse.Namespace,
        workspaces: Mapping[str, ReviewWorkspace],
        *,
        use_opengl: bool,
    ) -> None:
        super().__init__()
        from darsia.fluidflower.ui.interaction_safety import install_interaction_safety
        install_interaction_safety()
        self.args = args
        self.workspaces = dict(workspaces)
        self.use_opengl = use_opengl
        self.workspace_windows: dict[str, QMainWindow] = {}
        self.workspace_buttons: dict[str, QPushButton] = {}
        reviewer_name, save_root = effective_reviewer_settings(args)
        self.initial_reviewer_name = reviewer_name
        self.initial_save_root = save_root
        self.setWindowTitle(f"FluidFlower Review Desktop v{DESKTOP_UI_VERSION}")
        self.resize(1020, 900)
        self.setMinimumSize(760, 680)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(34, 28, 34, 28)
        outer.setSpacing(18)

        title = QLabel("FluidFlower Review Desktop")
        title.setObjectName("hubTitle")
        outer.addWidget(title)
        intro = QLabel(
            "Choose one purpose. Current and historical A/B comparisons, training "
            "annotations, and blind phase validation use separate workspaces and "
            "decision files."
        )
        intro.setObjectName("hubIntro")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        identity = QFrame()
        identity.setObjectName("localIdentity")
        identity_layout = QGridLayout(identity)
        identity_layout.setContentsMargins(18, 16, 18, 16)
        identity_layout.setHorizontalSpacing(12)
        identity_layout.setVerticalSpacing(9)
        identity_title = QLabel("Local review identity and storage")
        identity_title.setObjectName("identityTitle")
        identity_layout.addWidget(identity_title, 0, 0, 1, 3)
        identity_layout.addWidget(QLabel("Reviewer"), 1, 0)
        self.reviewer_name = QLineEdit(self.initial_reviewer_name)
        self.reviewer_name.setPlaceholderText("Reviewer full name")
        self.reviewer_name.textChanged.connect(self._update_open_gate)
        identity_layout.addWidget(self.reviewer_name, 1, 1, 1, 2)
        identity_layout.addWidget(QLabel("Local save folder"), 2, 0)
        self.save_root = QLineEdit(str(self.initial_save_root))
        self.save_root.setPlaceholderText("Choose a folder on this computer")
        self.save_root.textChanged.connect(self._update_open_gate)
        identity_layout.addWidget(self.save_root, 2, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_save_root)
        identity_layout.addWidget(browse, 2, 2)
        identity_note = QLabel(
            "Every edit is written locally immediately. Files are separated by "
            "reviewer, campaign, and workspace; NAS transfer is manual."
        )
        identity_note.setObjectName("identityNote")
        identity_note.setWordWrap(True)
        identity_layout.addWidget(identity_note, 3, 0, 1, 3)
        identity_layout.setColumnStretch(1, 1)
        outer.addWidget(identity)

        for workspace in self.workspaces.values():
            section = QFrame()
            section.setObjectName(
                "validationWorkspace" if workspace.blind else "trainingWorkspace"
            )
            row = QHBoxLayout(section)
            row.setContentsMargins(18, 16, 18, 16)
            row.setSpacing(18)
            text_column = QVBoxLayout()
            heading = QLabel(workspace.title)
            heading.setObjectName("workspaceTitle")
            text_column.addWidget(heading)
            label = QLabel(workspace.short_label.upper())
            label.setObjectName("hubWorkspaceBadge")
            text_column.addWidget(label, 0, Qt.AlignmentFlag.AlignLeft)
            description = QLabel(workspace.description)
            description.setWordWrap(True)
            description.setObjectName("workspaceDescription")
            text_column.addWidget(description)
            boundary_text = "Editable evidence; versioned training source"
            if workspace.key in {"current_comparison", "comparison"}:
                boundary_text = "Blinded A/B verdicts; isolated comparison log"
            elif workspace.blind:
                boundary_text = "Blind phase labels; isolated validation log"
            boundary = QLabel(boundary_text)
            boundary.setObjectName("workspaceBoundary")
            text_column.addWidget(boundary)
            row.addLayout(text_column, 1)
            button = QPushButton("Open")
            button.setObjectName("openWorkspaceButton")
            button.setMinimumSize(130, 46)
            button.clicked.connect(
                lambda _checked=False, key=workspace.key: self._open_workspace(key)
            )
            self.workspace_buttons[workspace.key] = button
            row.addWidget(button)
            outer.addWidget(section)

        outer.addStretch(1)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("hubStatus")
        outer.addWidget(self.status_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#171a1d; color:#e7eaed; font-size:14px; }
            #hubTitle { font-size:28px; font-weight:650; }
            #hubIntro { color:#b8c0c7; font-size:15px; max-width:820px; }
            #localIdentity { background:#20252a; border:1px solid #4c5963; border-left:5px solid #4d91c4; border-radius:4px; }
            #identityTitle { font-size:17px; font-weight:600; }
            #identityNote { color:#aeb7bf; font-size:12px; }
            #trainingWorkspace, #validationWorkspace { background:#20252a; border:1px solid #3e474f; border-radius:4px; }
            #trainingWorkspace { border-left:5px solid #4a9f80; }
            #validationWorkspace { border-left:5px solid #d7a934; }
            #workspaceTitle { font-size:20px; font-weight:600; }
            #hubWorkspaceBadge { color:#94c8df; font-size:12px; font-weight:650; }
            #workspaceDescription { color:#d4d9dd; }
            #workspaceBoundary { color:#9ca7af; font-size:12px; }
            #openWorkspaceButton { background:#185d86; border:1px solid #77bce8; border-radius:4px; padding:8px 18px; font-weight:600; }
            #openWorkspaceButton:hover { background:#216f9e; }
            #hubStatus { color:#aeb7bf; }
            """
        )
        self._update_open_gate()

    def _browse_save_root(self) -> None:
        initial = self.save_root.text().strip() or str(self.initial_save_root)
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose local review folder",
            initial,
        )
        if selected:
            self.save_root.setText(selected)

    def _update_open_gate(self, _text: str = "") -> None:
        reviewer_ready = bool(self.reviewer_name.text().strip())
        folder_ready = bool(self.save_root.text().strip())
        ready = reviewer_ready and folder_ready
        for button in self.workspace_buttons.values():
            button.setEnabled(ready)
        if not reviewer_ready:
            self.status_label.setText("Enter reviewer name before opening a workspace")
        elif not folder_ready:
            self.status_label.setText("Choose a local save folder before opening a workspace")
        else:
            self.status_label.setText("Ready · continuous local autosave enabled")

    def _open_workspace(self, key: str) -> None:
        reviewer_name = self.reviewer_name.text().strip()
        save_root_text = self.save_root.text().strip()
        if not reviewer_name or not save_root_text:
            self._update_open_gate()
            return
        base_workspace = self.workspaces[key]
        workspace = local_reviewer_workspace(
            base_workspace,
            Path(save_root_text),
            reviewer_name,
        )
        try:
            ensure_writable_directory(workspace.output_dir)
            save_reviewer_profile(
                self.args.cache_root,
                reviewer_name=reviewer_name,
                save_root=Path(save_root_text),
            )
        except OSError as exc:
            self.status_label.setText("Local save folder is not writable")
            QMessageBox.critical(
                self,
                "Could not use local save folder",
                f"{type(exc).__name__}: {exc}",
            )
            return
        existing = self.workspace_windows.get(key)
        if existing is not None:
            same_target = existing.workspace.output_dir.resolve() == workspace.output_dir.resolve()
            same_reviewer = existing.reviewer.text().strip().casefold() == reviewer_name.casefold()
            if same_target and same_reviewer:
                existing._populate_runs()
                existing.show()
                existing.raise_()
                existing.activateWindow()
                self.hide()
                return
            existing.close()
            existing.deleteLater()
            self.workspace_windows.pop(key, None)
        self.status_label.setText(f"Opening {workspace.title}...")
        for button in self.workspace_buttons.values():
            button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            app_model, store = build_workspace_model(
                self.args,
                workspace,
                reviewer_name=reviewer_name,
            )
            window_class = workspace_window_class(workspace)
            window = window_class(
                app_model,
                store,
                self.args.cache_root,
                workspace,
                reviewer_name=reviewer_name,
                use_opengl=self.use_opengl,
                on_back=lambda selected=key: self._return_from_workspace(selected),
            )
            self.workspace_windows[key] = window
            window.show()
            self.hide()
            self.status_label.setText("Ready")
        except Exception as exc:  # noqa: BLE001 - visible application boundary
            self.status_label.setText("Could not open workspace")
            QMessageBox.critical(
                self,
                f"Could not open {workspace.title}",
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            QApplication.restoreOverrideCursor()
            self._update_open_gate()

    def _return_from_workspace(self, key: str) -> None:
        window = self.workspace_windows.get(key)
        if window is not None:
            window.hide()
        self.show()
        self.raise_()
        self.activateWindow()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        choices=(
            "hub",
            "current_comparison",
            "training",
            "validation",
            "comparison",
        ),
        default="hub",
        help="Open the purpose selector or one isolated workspace directly.",
    )
    parser.add_argument("--ffac-root", type=Path, default=DEFAULT_FFAC_ROOT)
    parser.add_argument("--ffum-root", type=Path, default=DEFAULT_FFUM_ROOT)
    parser.add_argument("--combined-root", type=Path, default=DEFAULT_COMBINED_ROOT)
    parser.add_argument(
        "--validation-ffac-root",
        type=Path,
        default=DEFAULT_VALIDATION_FFAC_ROOT,
    )
    parser.add_argument(
        "--validation-ffum-root",
        type=Path,
        default=DEFAULT_VALIDATION_FFUM_ROOT,
    )
    parser.add_argument(
        "--validation-combined-root",
        type=Path,
        default=DEFAULT_VALIDATION_COMBINED_ROOT,
    )
    parser.add_argument(
        "--comparison-ffac-root",
        type=Path,
        default=DEFAULT_COMPARISON_FFAC_ROOT,
    )
    parser.add_argument(
        "--comparison-ffum-root",
        type=Path,
        default=DEFAULT_COMPARISON_FFUM_ROOT,
    )
    parser.add_argument(
        "--comparison-combined-root",
        type=Path,
        default=DEFAULT_COMPARISON_COMBINED_ROOT,
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--reviewer", default=os.environ.get("FFAC_REVIEWER", ""))
    parser.add_argument("--blind-seed", default=review.DEFAULT_BLIND_SEED)
    parser.add_argument("--ffac-runs", nargs="+")
    parser.add_argument("--ffum-runs", nargs="+")
    parser.add_argument("--validation-ffac-runs", nargs="+")
    parser.add_argument("--validation-ffum-runs", nargs="+")
    parser.add_argument("--comparison-ffac-runs", nargs="+")
    parser.add_argument("--comparison-ffum-runs", nargs="+")
    parser.add_argument(
        "--offline-bundle",
        type=Path,
        default=_bundled_offline_manifest(),
        help="Relocatable offline bundle manifest bundled with the desktop app.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Ignore any bundled offline manifest and use the explicit campaign roots.",
    )
    parser.add_argument(
        "--local-review-root",
        type=Path,
        default=DEFAULT_USER_DATA_ROOT / "reviews",
        help="Writable local decision store used by offline bundles.",
    )
    parser.add_argument("--no-sync-manifest", action="store_true")
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Show one selected model once instead of an A/B comparison.",
    )
    parser.add_argument(
        "--approval-only",
        action="store_true",
        help=(
            "Hide verdict and annotation controls for ordinary per-run 2D phase "
            "inspection. Approval is recorded by the calling project application."
        ),
    )
    parser.add_argument(
        "--review-surface",
        choices=("default", "blind_regions", "open_correction", "blind_gas_validation"),
        default="default",
        help=(
            "Choose a constrained interaction surface. blind_regions hides model "
            "predictions; blind_gas_validation collects image-only FF_AC labels; "
            "open_correction exposes one model and localized edits."
        ),
    )
    parser.add_argument(
        "--single-candidate",
        action="append",
        metavar="RUN=CANDIDATE_ID",
        help="Select the candidate used for one run in --single-model mode.",
    )
    parser.add_argument(
        "--review-session-id",
        default="",
        help="Unique immutable identity for a new model-bound review session.",
    )
    parser.add_argument(
        "--target-model-id",
        default="",
        help="Immutable phase-model version reviewed by this session.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Open an existing session for audit without permitting any edits.",
    )
    parser.add_argument("--review-current-model-root", type=Path,
                        help="Saved phase-model folder currently shown in Studio correction review")
    parser.add_argument("--review-model-choice", action="append", default=[],
                        help="Explicit Studio model choice as a JSON root/label object")
    parser.add_argument("--evidence-audit", type=Path, help="Immutable evidence-disagreement queue")
    parser.add_argument("--phase-workflow-role", choices=("label", "blind"),
                        help="Open the native shared phase correction session")
    parser.add_argument("--phase-workflow-series", default="",
                        help="Series owning the native shared phase session")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    return parser


def _bundled_offline_manifest() -> Path | None:
    configured = os.environ.get("FF_REVIEW_OFFLINE_BUNDLE")
    candidates = [Path(configured)] if configured else []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / "offline_bundle" / "offline_bundle.json")
    candidates.extend(
        (
            project_path('offline_bundle/offline_bundle.json'),
            REPO_ROOT.parent / "offline_bundle" / "offline_bundle.json",
        )
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _bundle_path(bundle_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else bundle_root / path).resolve()


def _offline_review_workspaces(args: argparse.Namespace) -> dict[str, ReviewWorkspace]:
    manifest_path = Path(args.offline_bundle).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if payload.get("format") != OFFLINE_BUNDLE_FORMAT:
        raise ValueError(f"Unsupported offline review bundle: {manifest_path}")
    bundle_id = str(payload.get("bundle_id") or "offline").strip()
    rows = payload.get("workspaces") or {}
    if not isinstance(rows, Mapping):
        raise ValueError(f"Invalid workspace map in {manifest_path}")
    decision_root = Path(args.local_review_root).resolve() / bundle_id
    workspaces: dict[str, ReviewWorkspace] = {}
    for key, row in rows.items():
        if not isinstance(row, Mapping):
            continue
        combined_root = _bundle_path(manifest_path.parent, row["combined_root"])
        seed_value = row.get("seed_review_dir")
        workspaces[key] = ReviewWorkspace(
            key=key,
            title=str(row.get("title") or key.title()),
            short_label=str(row.get("short_label") or "Offline"),
            description=str(
                row.get("description")
                or "Review images and labels stored entirely on this Mac."
            ),
            campaign_id=str(row.get("campaign_id") or payload.get("bundle_id")),
            ffac_root=combined_root,
            ffum_root=combined_root,
            combined_root=combined_root,
            blind=bool(row.get("blind")),
            decision_dir=decision_root / key / "manual_pairwise_review",
            seed_review_dir=(
                _bundle_path(manifest_path.parent, seed_value)
                if seed_value
                else None
            ),
            offline=True,
            review_surface=str(row.get("review_surface") or "default"),
            single_model=bool(row.get("single_model")),
        )
    if not workspaces:
        raise ValueError(f"Offline bundle contains no workspaces: {manifest_path}")
    return workspaces


def review_workspaces(args: argparse.Namespace) -> dict[str, ReviewWorkspace]:
    blind_gas_validation = args.review_surface == "blind_gas_validation"
    if args.offline_bundle is not None and not blind_gas_validation:
        return _offline_review_workspaces(args)
    if blind_gas_validation and not args.ffac_runs:
        raise ValueError("--review-surface blind_gas_validation requires --ffac-runs")
    if blind_gas_validation and args.ffum_runs:
        raise ValueError("Blind gas validation accepts FF_AC runs only")
    if blind_gas_validation and args.approval_only:
        raise ValueError("Blind gas validation requires annotation controls")
    single_model = bool(args.single_model or blind_gas_validation)
    open_correction = single_model and args.review_surface == "open_correction"
    blind_regions = single_model and args.review_surface == "blind_regions"
    workspaces = {
        "current_comparison": ReviewWorkspace(
            key="current_comparison",
            title=(
                "Blind free-gas validation"
                if blind_gas_validation
                else
                "Open model correction"
                if open_correction
                else "Single-model validation"
                if single_model
                else "Current V5 optical vs physics comparison"
            ),
            short_label=(
                "Photographs only"
                if blind_gas_validation
                else
                "Prediction visible; corrections enabled"
                if open_correction
                else "Validation only"
                if single_model
                else "Current A/B comparison"
            ),
            description=(
                "Mark visible free gas and image artifacts without seeing optical or "
                "physics phase maps. End the blind session to seal the labels before "
                "revealing model controls."
                if blind_gas_validation
                else
                "Inspect the selected prediction on complete corrected image sequences. "
                "Use boxes or freehand regions to localize missed phases, false "
                "detections, physical inconsistencies, and image artifacts."
                if open_correction
                else (
                    "Inspect one selected phase model on complete corrected images, "
                    "with all established validation tools except A/B comparison."
                )
                if single_model
                else (
                    "Compare the current reviewed V5 optical model with the same "
                    "model using generic physical constraints on FF_AC AC32 and "
                    "FF_UM run18. Candidate identities remain hidden while "
                    "per-frame verdicts and localized issue boxes are recorded."
                )
            ),
            campaign_id=TRAINING_CAMPAIGN_ID,
            ffac_root=Path(args.ffac_root),
            ffum_root=Path(args.ffum_root),
            combined_root=Path(args.combined_root),
            ffac_runs=(
                (None if single_model else DEFAULT_TRAINING_FFAC_RUNS)
                if args.ffac_runs is None
                else tuple(str(run).casefold() for run in args.ffac_runs)
            ),
            ffum_runs=(
                (() if blind_gas_validation else None if single_model else DEFAULT_TRAINING_FFUM_RUNS)
                if args.ffum_runs is None
                else tuple(str(run).casefold() for run in args.ffum_runs)
            ),
            blind=blind_gas_validation or not open_correction,
            approval_only=bool(args.approval_only),
            review_surface=str(args.review_surface),
            single_model=single_model,
            decision_dir=(
                Path(args.combined_root)
                / (
                    "blind_gas_validation_review"
                    if blind_gas_validation
                    else "single_model_validation_review"
                    if single_model
                    else "current_model_comparison_review"
                )
            ),
        ),
        "training": ReviewWorkspace(
            key="training",
            title=(
                "Blind phase labeling"
                if blind_regions
                else "Active learning review and annotations"
            ),
            short_label=(
                "Prediction hidden"
                if blind_regions
                else "Model training evidence"
            ),
            description=(
                "Review model-proposed uncertain regions from FF_AC and FF_UM, or "
                "draw targeted phase and artifact evidence manually. Suggestions "
                "hide the model answer until the reviewer responds, and all edits "
                "remain versioned evidence for the next model."
            ),
            campaign_id=TRAINING_CAMPAIGN_ID,
            ffac_root=Path(args.ffac_root),
            ffum_root=Path(args.ffum_root),
            combined_root=Path(args.combined_root),
            ffac_runs=(
                (None if single_model else DEFAULT_TRAINING_FFAC_RUNS)
                if args.ffac_runs is None
                else tuple(str(run).casefold() for run in args.ffac_runs)
            ),
            ffum_runs=(
                (None if single_model else DEFAULT_TRAINING_FFUM_RUNS)
                if args.ffum_runs is None
                else tuple(str(run).casefold() for run in args.ffum_runs)
            ),
            blind=blind_regions,
            review_surface=str(args.review_surface),
            single_model=single_model,
        ),
        "validation": ReviewWorkspace(
            key="validation",
            title="Blind model validation",
            short_label="Validation only",
            description=(
                "Evaluate disjoint experiments without seeing model identity. "
                "Answers remain isolated from training data until a new dataset "
                "version is explicitly approved."
            ),
            campaign_id=VALIDATION_CAMPAIGN_ID,
            ffac_root=Path(args.validation_ffac_root),
            ffum_root=Path(args.validation_ffum_root),
            combined_root=Path(args.validation_combined_root),
            ffac_runs=(
                None
                if args.validation_ffac_runs is None
                else tuple(
                    str(run).casefold() for run in args.validation_ffac_runs
                )
            ),
            ffum_runs=(
                None
                if args.validation_ffum_runs is None
                else tuple(
                    str(run).casefold() for run in args.validation_ffum_runs
                )
            ),
            blind=True,
        ),
        "comparison": ReviewWorkspace(
            key="comparison",
            title="Historical V3 vs V4 comparison",
            short_label="Historical A/B",
            description=(
                "Compare the frozen generic model with the mixed-region MIL model "
                "on unseen AC56 and FF_UM run10. Candidate identities stay hidden, "
                "and all verdicts remain separate from training and phase validation."
            ),
            campaign_id=COMPARISON_CAMPAIGN_ID,
            ffac_root=Path(args.comparison_ffac_root),
            ffum_root=Path(args.comparison_ffum_root),
            combined_root=Path(args.comparison_combined_root),
            ffac_runs=(
                ("ac56",)
                if args.comparison_ffac_runs is None
                else tuple(
                    str(run).casefold() for run in args.comparison_ffac_runs
                )
            ),
            ffum_runs=(
                ("run10",)
                if args.comparison_ffum_runs is None
                else tuple(
                    str(run).casefold() for run in args.comparison_ffum_runs
                )
            ),
            blind=True,
        ),
    }
    selected_candidates: dict[str, str] = {}
    if blind_gas_validation:
        workspaces = {"current_comparison": workspaces["current_comparison"]}
    for raw in args.single_candidate or []:
        run, separator, candidate_id = str(raw).partition("=")
        if separator and run.strip() and candidate_id.strip():
            selected_candidates[run.strip().casefold()] = candidate_id.strip()
    return {
        key: replace(
            workspace,
            review_session_id=str(args.review_session_id or "").strip(),
            target_model_id=str(args.target_model_id or "").strip(),
            evidence_audit_path=str(getattr(args, "evidence_audit", None) or ""),
            selected_candidates=selected_candidates,
            read_only=bool(args.read_only),
        )
        for key, workspace in workspaces.items()
    }


def _seed_reviewers(seed: Path) -> set[str]:
    canonical = seed / "pairwise_review_decisions.json"
    if not canonical.is_file():
        return set()
    try:
        payload = json.loads(canonical.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return set()
    return {
        str(item.get("reviewer") or "").strip().casefold()
        for item in (payload.get("decisions") or {}).values()
        if isinstance(item, Mapping) and str(item.get("reviewer") or "").strip()
    }


def _initialize_offline_review_dir(
    workspace: ReviewWorkspace,
    reviewer_name: str,
) -> None:
    target = workspace.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    seed = workspace.seed_review_dir
    if seed is None or not seed.is_dir():
        return
    canonical = target / "pairwise_review_decisions.json"
    if canonical.is_file():
        return
    seed_reviewers = _seed_reviewers(seed)
    if seed_reviewers and seed_reviewers != {reviewer_name.strip().casefold()}:
        return
    for source in seed.iterdir():
        if (
            source.is_file()
            and source.name.startswith("pairwise_review_")
            and source.suffix.casefold() in {".json", ".csv"}
        ):
            shutil.copy2(source, target / source.name)


def write_review_session_manifest(
    workspace: ReviewWorkspace,
    reviewer_name: str,
) -> Path:
    from darsia.fluidflower.storage_paths import machine_stored_text

    path = workspace.output_dir.resolve() / "review_session.json"
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, Mapping):
                existing = dict(loaded)
        except (OSError, ValueError, TypeError):
            pass
    now = utc_now()
    output_dir = workspace.output_dir.resolve()
    local_review_root = (
        output_dir.parents[5]
        if workspace.review_session_id
        else output_dir.parents[3]
    )
    payload = {
        "schema": 1,
        "format": "fluidflower-local-review-session-v1",
        "desktop_ui_version": DESKTOP_UI_VERSION,
        "reviewer": reviewer_name.strip(),
        "reviewer_id": reviewer_identifier(reviewer_name),
        "campaign_id": workspace.campaign_id,
        "workspace": workspace.key,
        "workspace_title": workspace.title,
        "decision_dir": machine_stored_text(workspace.output_dir.resolve()),
        "local_review_root": machine_stored_text(local_review_root),
        "combined_root": machine_stored_text(workspace.combined_root.resolve()),
        "ffac_root": machine_stored_text(workspace.ffac_root.resolve()),
        "ffum_root": machine_stored_text(workspace.ffum_root.resolve()),
        "seed_review_dir": (
            machine_stored_text(workspace.seed_review_dir.resolve())
            if workspace.seed_review_dir is not None
            else None
        ),
        "created_utc": str(existing.get("created_utc") or now),
        "updated_utc": now,
        "storage_policy": "continuous-local-autosave-with-append-only-history",
        "phase_display_policy": review.PHASE_DISPLAY_POLICY,
        "review_surface": workspace.review_surface,
        "single_model": workspace.single_model,
        "ffac_runs": (
            list(workspace.ffac_runs) if workspace.ffac_runs is not None
            else existing.get("ffac_runs")
        ),
        "ffum_runs": (
            list(workspace.ffum_runs) if workspace.ffum_runs is not None
            else existing.get("ffum_runs")
        ),
        "review_session_id": workspace.review_session_id,
        "target_model_id": workspace.target_model_id,
        "evidence_audit_path": workspace.evidence_audit_path,
        "selected_candidates": dict(
            workspace.selected_candidates or existing.get("selected_candidates") or {}
        ),
        "editing_locked": bool(existing.get("editing_locked", False)),
        "training_locks": list(existing.get("training_locks") or []),
    }
    if workspace.review_surface == "blind_gas_validation":
        payload.update(
            evidence_role="independent_blind_gas_validation",
            include_in_training=False,
            requires_explicit_training_selection=True,
            blindness_status="blind",
            post_reveal_edits_are_blind=False,
        )
    # Transfer evaluations stay excluded when resumed through the ordinary desktop
    # entry point; refreshing session metadata must not erase its evidence role.
    for key in ("evidence_role", "include_in_training", "requires_explicit_training_selection",
                "transfer_evaluation_root", "source_model_id", "source_model_sha256",
                "evaluation_model_sha256", "training_seen",
                "evaluation_review_mode", "optical_training_status", "calibration_evidence_status",
                "calibration_history", "independent_holdout", "blindness_status",
                "first_prediction_reveal_utc", "frozen_blind_snapshot",
                "frozen_blind_snapshot_sha256", "post_reveal_edits_are_blind",
                "review_lifecycle_status", "review_lifecycle_history"):
        if key in existing:
            payload[key] = existing[key]
    reveal_marker = output_dir / "blind_reveal.json"
    if reveal_marker.is_file():
        # A comparison can be opened while this window is still autosaving.
        # Retain the first reveal boundary on every subsequent session refresh.
        revealed = json.loads(reveal_marker.read_text(encoding="utf-8-sig"))
        for key in ("blindness_status", "first_prediction_reveal_utc", "frozen_blind_snapshot",
                    "frozen_blind_snapshot_sha256", "post_reveal_edits_are_blind"):
            payload[key] = revealed[key]
    atomic_json(path, payload)
    return path


def should_sync_source_manifest(args: argparse.Namespace) -> bool:
    """Refresh online campaign assets even when decisions are stored locally."""

    return not bool(args.no_sync_manifest) and args.offline_bundle is None


def build_workspace_model(
    args: argparse.Namespace,
    workspace: ReviewWorkspace,
    *,
    reviewer_name: str | None = None,
) -> tuple[review.ReviewApp, review.ReviewStore]:
    active_reviewer = str(reviewer_name or args.reviewer or "").strip()
    combined_root = workspace.combined_root.resolve()
    run_filters = workspace.run_filters()
    if should_sync_source_manifest(args):
        combined.sync_combined_manifest(
            combined_root,
            (
                ("ffac", workspace.ffac_root.resolve()),
                ("ffum", workspace.ffum_root.resolve()),
            ),
            run_filters,
            campaign_id=workspace.campaign_id,
        )
    inventory = review.discover_review_inventory(
        combined_root,
        blind_seed=args.blind_seed,
        allow_incomplete=True,
    )
    if workspace.review_surface == "blind_gas_validation":
        allowed_runs = {str(run).casefold() for run in workspace.ffac_runs or ()}
        if not allowed_runs or workspace.ffum_runs:
            raise ValueError("Blind gas validation needs an explicit FF_AC-only run list")
        pairs = tuple(pair for pair in inventory.pairs if pair.run.casefold() in allowed_runs)
        missing = allowed_runs - {pair.run.casefold() for pair in pairs}
        if missing:
            raise ValueError(f"Blind gas validation runs unavailable: {sorted(missing)}")
        if any(pair.candidate_a.dataset.casefold() != "ffac"
               or pair.candidate_b.dataset.casefold() != "ffac" for pair in pairs):
            raise ValueError("Blind gas validation inventory contains a non-FF_AC source")
        inventory = review.ReviewInventory(
            pairs=pairs,
            unpaired={run: value for run, value in inventory.unpaired.items()
                      if run.casefold() in allowed_runs},
            manifest_path=inventory.manifest_path,
            pending_runs=tuple(run for run in inventory.pending_runs
                               if run.casefold() in allowed_runs),
        )
    if workspace.single_model:
        requested: dict[str, str] = {}
        for raw in args.single_candidate or []:
            run, separator, candidate_id = str(raw).partition("=")
            if not separator or not run.strip() or not candidate_id.strip():
                raise ValueError("--single-candidate must use RUN=CANDIDATE_ID")
            requested[run.strip().casefold()] = candidate_id.strip()
        single_pairs: list[review.ReviewPair] = []
        for pair in inventory.pairs:
            available = {
                pair.candidate_a.candidate_id: pair.candidate_a,
                pair.candidate_b.candidate_id: pair.candidate_b,
            }
            requested_id = requested.get(pair.run.casefold())
            if requested_id:
                if requested_id not in available:
                    raise ValueError(
                        f"Candidate {requested_id!r} is unavailable for {pair.run}"
                    )
                candidate = available[requested_id]
            else:
                candidate = pair.candidate_a
            single_pairs.append(
                review.ReviewPair(
                    run=pair.run,
                    candidate_a=candidate,
                    candidate_b=candidate,
                    comparison_mode="single_candidate",
                )
            )
        inventory = review.ReviewInventory(
            pairs=tuple(single_pairs),
            unpaired=inventory.unpaired,
            manifest_path=inventory.manifest_path,
            pending_runs=inventory.pending_runs,
        )
        if workspace.review_surface == "blind_gas_validation":
            if requested and set(requested) != allowed_runs:
                raise ValueError("Blind gas validation candidate bindings must cover every selected run")
            workspace = replace(workspace, selected_candidates={
                pair.run: pair.candidate_a.candidate_id for pair in inventory.pairs
            })
    if workspace.offline:
        if not active_reviewer:
            raise ValueError("Reviewer name is required for local review storage")
        if not workspace.read_only:
            _initialize_offline_review_dir(workspace, active_reviewer)
            write_review_session_manifest(workspace, active_reviewer)
    store = review.ReviewStore(
        workspace.output_dir.resolve(),
        inventory.pairs,
        read_only=workspace.read_only,
    )
    if workspace.offline and not workspace.read_only:
        store.ensure_reviewer_provenance()
    app_model = review.ReviewApp(
        inventory,
        store,
        default_reviewer=active_reviewer,
        selection_dir=combined_root,
        blind_seed=args.blind_seed,
        watch=False,
        prepare_new=False,
    )
    return app_model, store


def build_review_model(
    args: argparse.Namespace,
) -> tuple[review.ReviewApp, review.ReviewStore]:
    """Backward-compatible builder for the training workspace."""

    return build_workspace_model(args, review_workspaces(args)["training"])


@saved_reader_ui_entrypoint
def main(argv: Sequence[str] | None = None) -> int:
    from darsia.fluidflower.machine_paths import install_machine_path_map
    from darsia.fluidflower.raw_production_review import install_readers
    install_machine_path_map()
    install_readers()
    args = build_parser().parse_args(argv)
    if args.online or args.review_surface == "blind_gas_validation":
        args.offline_bundle = None
    if args.phase_workflow_role:
        from darsia.fluidflower.ui.phase_model_correction import run_shared_main
        return run_shared_main(sys.modules[__name__], args)
    if (args.review_surface == "blind_gas_validation"
            and args.workspace not in {"hub", "current_comparison"}):
        raise SystemExit("Blind gas validation opens the current_comparison workspace")
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    surface = QSurfaceFormat()
    surface.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setSamples(4)
    QSurfaceFormat.setDefaultFormat(surface)
    qt_app = QApplication(sys.argv[:1])
    qt_app.setApplicationName("FluidFlower Calibration Review")
    qt_app.setStyle("Fusion")
    workspaces = review_workspaces(args)
    use_opengl = (
        not args.no_gpu
        and os.environ.get("QT_QPA_PLATFORM", "").casefold() != "offscreen"
    )
    if args.workspace == "hub":
        window: QMainWindow = ReviewHubWindow(
            args,
            workspaces,
            use_opengl=use_opengl,
        )
    else:
        reviewer_name, save_root = effective_reviewer_settings(args)
        if not reviewer_name:
            raise SystemExit(
                "Reviewer name is required. Start the hub or pass --reviewer."
            )
        workspace = local_reviewer_workspace(
            workspaces[args.workspace],
            save_root,
            reviewer_name,
        )
        ensure_writable_directory(workspace.output_dir)
        save_reviewer_profile(
            args.cache_root,
            reviewer_name=reviewer_name,
            save_root=save_root,
        )
        app_model, store = build_workspace_model(
            args,
            workspace,
            reviewer_name=reviewer_name,
        )
        window_class = workspace_window_class(workspace)
        window = window_class(
            app_model,
            store,
            args.cache_root,
            workspace,
            reviewer_name=reviewer_name,
            use_opengl=use_opengl,
        )
    if args.evidence_audit:
        if not isinstance(window, BlindModelComparisonWindow) or not args.single_model:
            raise ValueError("Evidence audit requires the single-model correction viewer")
        from darsia.fluidflower.ui.evidence_audit import attach_evidence_audit
        attach_evidence_audit(window, args.evidence_audit, sys.modules[__name__])
    from darsia.fluidflower.ui.review_version_controls import attach_review_version_controls
    attach_review_version_controls(window, args, sys.modules[__name__], use_opengl=use_opengl)
    window.show()

    if args.smoke_test or args.screenshot is not None:

        def finish_smoke_test() -> None:
            if args.screenshot is not None:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                screen = window.screen() or QApplication.primaryScreen()
                capture = (
                    screen.grabWindow(int(window.winId()))
                    if screen
                    else window.grab()
                )
                if not capture.save(str(args.screenshot)):
                    raise RuntimeError(f"Could not save {args.screenshot}")
            review_window = (
                window
                if isinstance(
                    window,
                    (CalibrationReviewWindow, BlindModelComparisonWindow),
                )
                else None
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "workspace": args.workspace,
                        "run_count": (
                            review_window.run_list.count()
                            if review_window is not None
                            else 0
                        ),
                        "current_run": (
                            review_window.current_run
                            if review_window is not None
                            else ""
                        ),
                        "gpu_view_requested": use_opengl,
                        "cache_root": str(args.cache_root.resolve()),
                    }
                ),
                flush=True,
            )
            qt_app.quit()

        QTimer.singleShot(3500 if args.workspace != "hub" else 600, finish_smoke_test)
    return int(qt_app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
