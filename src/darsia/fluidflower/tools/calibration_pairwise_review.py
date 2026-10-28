"""Blind pairwise review of FluidFlower calibration candidates.

The tool reads an existing postflight manifest and serves a local browser UI.
Each comparison contains two mass plots and two synchronized contour movies.
Candidate identity is hidden until a decision has been saved. Decisions are
written atomically to JSON and CSV with full candidate provenance.
"""

from __future__ import annotations

from contextvars import copy_context

from darsia.fluidflower.storage_paths import configured_tool_path
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.review_snapshot_journal import append_bounded_snapshot_journal

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from contextlib import contextmanager
import csv
import functools
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import shutil
import socket
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.phase_labels import (
    PHASE_CHOICE_TO_MANUAL_CATEGORY,
    VISIBLE_PHASE_CANONICAL_CLASSES,
    VISIBLE_PHASE_CHOICES,
    VISIBLE_PHASE_LABEL_SCHEMA,
    assess_model_phase,
    visible_phase_payload,
)
from darsia.fluidflower.phase_rendering import PHASE_DISPLAY_POLICY, two_phase_overlay_path
from darsia.fluidflower.tools.phase_injection_points import review_injection_points


DEFAULT_SELECTION_DIR = configured_tool_path('ffac', 'calibration_logs') / 'spatial_history_rerank_all_20260816_5000/winner_selection_all'
DEFAULT_BLIND_SEED = "ff-ac-pairwise-review-v1"
PORTABLE_BUNDLE_FORMAT = "ffac-pairwise-review-bundle"
PORTABLE_SESSION_FORMAT = "ffac-pairwise-review-session"
PORTABLE_BUNDLE_MARKER = "__FFAC_PORTABLE_BUNDLE__"
PORTABLE_UI_VERSION = 21
PHASE_MAP_RGB = (
    (49, 217, 160),
    (255, 191, 71),
    (242, 76, 157),
)
ISOLATED_COMPONENT_MAX_AREA_FRACTION = 2.0e-5
ISOLATED_COMPONENT_COLOR_TOLERANCE = 3
ISOLATED_COMPONENT_MIN_SEPARATION_FRACTION = 4.0e-3
PHASE_QUIZ_PROPOSAL_SCHEMA = 2
PHASE_QUIZ_INITIAL_PROPOSALS_PER_CANDIDATE = 8
PHASE_QUIZ_ROUND_SIZE = 8
PHASE_QUIZ_MAX_PROPOSALS_PER_CANDIDATE: int | None = None
PHASE_QUIZ_DOMAIN_MARGIN_FRACTION = 0.004
PHASE_QUIZ_DOMAIN_MARGIN_MIN_PX = 4
PHASE_QUIZ_DOMAIN_MARGIN_MAX_PX = 16
VALID_CHOICES = {
    "candidate_a",
    "candidate_b",
    "equivalent",
    "acceptable",
    "needs_repair",
    "unresolved",
}
FRAME_CHOICE_SCHEMA = "candidate-frame-preference-v1"
PAIR_FRAME_CHOICES = {
    "candidate_a",
    "candidate_b",
    "equivalent",
    "unresolved",
}
SINGLE_FRAME_CHOICES = {
    "acceptable",
    "needs_repair",
    "unresolved",
}


def frame_choice_values(comparison_mode: str) -> set[str]:
    """Return valid per-frame verdicts for the active comparison mode."""

    return (
        SINGLE_FRAME_CHOICES
        if str(comparison_mode) == "single_candidate"
        else PAIR_FRAME_CHOICES
    )


def normalize_frame_choices(
    raw_choices: Any,
    *,
    frame_count: int,
    elapsed_hours: Sequence[Any] = (),
    comparison_mode: str = "pair",
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Normalize reviewer verdicts without promoting legacy run-level choices."""

    if raw_choices in (None, ""):
        return []
    if isinstance(raw_choices, Mapping):
        source = [
            {"frame_index": key, "choice": value}
            for key, value in raw_choices.items()
        ]
    elif isinstance(raw_choices, list):
        source = raw_choices
    else:
        if strict:
            raise ValueError("frame_choices must be a list or object")
        return []

    allowed = frame_choice_values(comparison_mode)
    normalized: dict[int, dict[str, Any]] = {}
    for raw in source:
        if not isinstance(raw, Mapping):
            if strict:
                raise ValueError("Each frame choice must be an object")
            continue
        try:
            frame_index = int(raw.get("frame_index", -1))
        except (TypeError, ValueError):
            if strict:
                raise ValueError("Frame choice index must be an integer") from None
            continue
        choice = str(raw.get("choice") or "").casefold()
        if not 0 <= frame_index < max(0, int(frame_count)):
            if strict:
                raise ValueError(f"Invalid frame choice index: {frame_index}")
            continue
        if choice not in allowed:
            if strict:
                raise ValueError(
                    f"Invalid frame choice for {comparison_mode}: {choice!r}"
                )
            continue
        if frame_index in normalized and strict:
            raise ValueError(f"Duplicate frame choice index: {frame_index}")
        elapsed_h = None
        if frame_index < len(elapsed_hours):
            try:
                elapsed_h = float(elapsed_hours[frame_index])
            except (TypeError, ValueError):
                elapsed_h = None
        normalized[frame_index] = {
            "frame_index": frame_index,
            "elapsed_h": elapsed_h,
            "choice": choice,
            "updated_utc": str(raw.get("updated_utc") or "")[:64],
        }
    return [normalized[index] for index in sorted(normalized)]


def aggregate_frame_choice(
    frame_choices: Any,
    *,
    frame_count: int,
    comparison_mode: str = "pair",
) -> str:
    """Derive a legacy run-level summary from explicit per-frame verdicts."""

    if max(0, int(frame_count)) == 0:
        return "unresolved"
    records = normalize_frame_choices(
        frame_choices,
        frame_count=frame_count,
        comparison_mode=comparison_mode,
    )
    if len(records) < max(0, int(frame_count)) or any(
        row["choice"] == "unresolved" for row in records
    ):
        return "unresolved"
    counts = {
        value: sum(row["choice"] == value for row in records)
        for value in frame_choice_values(comparison_mode)
    }
    if str(comparison_mode) == "single_candidate":
        return "needs_repair" if counts.get("needs_repair", 0) else "acceptable"
    if counts.get("candidate_a", 0) > counts.get("candidate_b", 0):
        return "candidate_a"
    if counts.get("candidate_b", 0) > counts.get("candidate_a", 0):
        return "candidate_b"
    return "equivalent"


def summarize_frame_choices(
    frame_choices: Any,
    *,
    frame_count: int,
    comparison_mode: str = "pair",
) -> dict[str, Any]:
    """Build a compact progress and compatibility summary for exports."""

    records = normalize_frame_choices(
        frame_choices,
        frame_count=frame_count,
        comparison_mode=comparison_mode,
    )
    allowed = frame_choice_values(comparison_mode)
    return {
        "schema": FRAME_CHOICE_SCHEMA,
        "reviewed": len(records),
        "total": max(0, int(frame_count)),
        "complete": len(records) == max(0, int(frame_count)),
        "counts": {
            value: sum(row["choice"] == value for row in records)
            for value in sorted(allowed)
        },
        "aggregate_choice": aggregate_frame_choice(
            records,
            frame_count=frame_count,
            comparison_mode=comparison_mode,
        ),
    }
REVIEW_FLAG_LABELS = {
    "confirmed_no_free_gas_in_selected_area": "No free gas anywhere in selected area",
    "confirmed_no_aqueous_co2_in_selected_area": "No aqueous CO2 anywhere in selected area",
    "missed_visible_plume": "Missed visible plume",
    "missed_visible_gas": "Missed visible free CO2",
    "missed_visible_aqueous_co2": "Missed visible aqueous CO2",
    "unsupported_free_gas_classification": "Incorrect free-gas prediction in this model",
    "unsupported_aqueous_co2_classification": "Incorrect aqueous-CO2 prediction in this model",
    "false_seal_detection": "False seal detection",
    "incomplete_wash_indicator_carryover": "Indicator carryover from incomplete washing",
    "straight_facies_edges": "Unsupported phase boundary following a facies edge",
    "isolated_noise": "Isolated noise",
    "implausible_phase_split": "Implausible phase split",
    "late_mass_drift": "Late mass drift",
    "poor_mass_fit": "Poor mass fit",
    # Open correction summarizes localized phase labels as candidate flags too.
    **{
        PHASE_CHOICE_TO_MANUAL_CATEGORY[choice.key]: choice.label
        for choice in VISIBLE_PHASE_CHOICES
    },
}
VALID_FLAGS = set(REVIEW_FLAG_LABELS)
CANDIDATE_ISSUE_HELP = {
    "unsupported_free_gas_classification": (
        "Some free-gas predictions in the selected area are wrong for this model. "
        "Real free gas may also be present in the area. This box is evaluated only "
        "for this model and is not a whole-box negative training label."
    ),
    "unsupported_aqueous_co2_classification": (
        "Some aqueous-CO2 predictions in the selected area are wrong for this model. "
        "Real aqueous CO2 may also be present in the area. This box is evaluated "
        "only for this model and is not a whole-box negative training label."
    ),
}
VALID_ANNOTATION_CATEGORIES = {
    "confirmed_only_aqueous_co2",
    "confirmed_predominantly_aqueous_co2",
    "confirmed_predominantly_free_gas",
    "confirmed_only_free_gas",
    "confirmed_aqueous_co2",
    "confirmed_free_gas",
    "confirmed_mixed_free_aqueous_co2",
    "confirmed_mixed_aqueous_water",
    "confirmed_mixed_free_aqueous_water",
    "confirmed_no_co2",
    "confirmed_no_free_gas_in_selected_area",
    "confirmed_no_aqueous_co2_in_selected_area",
    "unsupported_free_gas_classification",
    "likely_dense_aqueous_co2_misclassified_as_gas",
    "phase_ambiguous_strong_color_response",
    "phase_not_assessable",
    "phase_signal_artifact",
    "implausible_post_injection_downward_gas_migration",
    "plausible_injection_forced_downward_gas",
    "plausible_dense_aqueous_co2_descent",
    "implausible_late_free_gas_persistence",
    "implausible_gas_location",
    "implausible_aqueous_location",
    "implausible_temporal_evolution",
    "missed_visible_co2",
    "missed_visible_gas",
    "disconnected_noise",
    "facies_edge_artifact",
    "incomplete_wash_indicator_carryover",
    "baseline_or_flush_artifact",
    "lighting_or_color_correction_artifact",
    "other",
} | VALID_FLAGS
VALID_ANNOTATION_PHASES = {"gas", "aqueous", "total", "unspecified"}
VALID_ANNOTATION_SEVERITIES = {"minor", "major", "critical"}
VALID_ANNOTATION_CONFIDENCES = {"high", "medium", "low", "unspecified"}
VALID_ANNOTATION_SHAPES = {"rectangle", "freehand", "point"}
VALID_REFERENCE_GUIDE_ORIENTATIONS = {"horizontal", "vertical"}

ANNOTATION_CATEGORY_SEMANTICS: Mapping[str, Mapping[str, str]] = {
    "missed_visible_plume": {
        "evidence_scope": "visible_source_image_region",
        "co2_evidence": "co2_present_phase_unresolved",
        "training_role": "regional_phase_evidence",
        "training_phase_class": "co2_present_region",
        "training_scope": "source_image_all_candidate_models",
        "evaluation_scope": "trigger_candidate_only",
    },
    "missed_visible_co2": {
        "evidence_scope": "visible_source_image_region",
        "co2_evidence": "co2_present_phase_unresolved",
        "training_role": "regional_phase_evidence",
        "training_phase_class": "co2_present_region",
        "training_scope": "source_image_all_candidate_models",
        "evaluation_scope": "trigger_candidate_only",
    },
    "missed_visible_gas": {
        "evidence_scope": "visible_source_image_region",
        "co2_evidence": "free_co2_present",
        "training_role": "regional_phase_evidence",
        "training_phase_class": "gas_present_region",
        "training_scope": "source_image_all_candidate_models",
        "evaluation_scope": "trigger_candidate_only",
    },
    "missed_visible_aqueous_co2": {
        "evidence_scope": "visible_source_image_region",
        "co2_evidence": "aqueous_co2_throughout_region",
        "training_role": "dense_phase_evidence",
        "training_phase_class": "aqueous",
        "training_scope": "source_image_all_candidate_models",
        "evaluation_scope": "trigger_candidate_only",
    },
    "isolated_noise": {
        "evidence_scope": "visible_source_image_region",
        "co2_evidence": "current_experiment_co2_not_supported",
        "training_role": "regional_phase_exclusion",
        "training_phase_class": "co2_absent_region",
        "training_scope": "source_image_all_candidate_models",
        "evaluation_scope": "trigger_candidate_only",
    },
    "disconnected_noise": {
        "evidence_scope": "visible_source_image_region",
        "co2_evidence": "current_experiment_co2_not_supported",
        "training_role": "regional_phase_exclusion",
        "training_phase_class": "co2_absent_region",
        "training_scope": "source_image_all_candidate_models",
        "evaluation_scope": "trigger_candidate_only",
    },
    "unsupported_free_gas_classification": {
        "evidence_scope": "candidate_phase_map_region",
        "co2_evidence": "candidate_free_gas_classification_unsupported",
        "training_role": "candidate_issue_only",
        "training_phase_class": "ignored",
        "training_scope": "none",
        "evaluation_scope": "trigger_candidate_only",
        "training_label_policy_version": "visible-phase-manual-v4",
    },
    "confirmed_no_free_gas_in_selected_area": {
        "evidence_scope": "visible_source_image_region",
        "co2_evidence": "free_co2_absent_throughout_selected_area",
        "training_role": "regional_phase_exclusion",
        "training_phase_class": "gas_absent_region",
        "training_scope": "source_image_all_candidate_models",
        "evaluation_scope": "source_image_shared_across_candidates",
        "interpretation_scope": "source_image_shared_across_candidates",
        "training_label_policy_version": "visible-phase-manual-v4",
    },
    "unsupported_aqueous_co2_classification": {
        "evidence_scope": "candidate_phase_map_region",
        "co2_evidence": "candidate_aqueous_co2_classification_unsupported",
        "training_role": "candidate_issue_only",
        "training_phase_class": "ignored",
        "training_scope": "none",
        "evaluation_scope": "trigger_candidate_only",
        "training_label_policy_version": "visible-phase-manual-v4",
    },
    "confirmed_no_aqueous_co2_in_selected_area": {
        "evidence_scope": "visible_source_image_region",
        "co2_evidence": "aqueous_co2_absent_throughout_selected_area",
        "training_role": "regional_phase_exclusion",
        "training_phase_class": "aqueous_absent_region",
        "training_scope": "source_image_all_candidate_models",
        "evaluation_scope": "source_image_shared_across_candidates",
        "interpretation_scope": "source_image_shared_across_candidates",
        "training_label_policy_version": "visible-phase-manual-v4",
    },
    "likely_dense_aqueous_co2_misclassified_as_gas": {
        "evidence_scope": "visible_source_image_region",
        "co2_evidence": "aqueous_co2_present_free_co2_not_supported",
        "training_role": "regional_phase_evidence",
        "training_phase_class": "aqueous_present_gas_absent_region",
        "training_scope": "source_image_all_candidate_models",
        "evaluation_scope": "trigger_candidate_only",
    },
    "incomplete_wash_indicator_carryover": {
        "evidence_scope": "current_experiment",
        "co2_evidence": "exclude_as_current_experiment_co2",
        "training_role": "artifact_exclusion",
        "training_phase_class": "current_experiment_co2_artifact",
        "training_scope": "source_image_all_candidate_models",
        "evaluation_scope": "source_image_shared_across_candidates",
        "artifact_origin": "previous_experiment_incomplete_washing",
        "interpretation_scope": "source_image_shared_across_candidates",
    },
}


def annotation_category_semantics(category: str) -> dict[str, str]:
    """Return review metadata; the versioned trainer resolves historical labels."""

    return dict(ANNOTATION_CATEGORY_SEMANTICS.get(str(category), {}))


ANNOTATION_SEMANTIC_FIELDS = frozenset(
    key for values in ANNOTATION_CATEGORY_SEMANTICS.values() for key in values)
VALID_PHASE_QUIZ_CHOICES = {
    *(choice.key for choice in VISIBLE_PHASE_CHOICES),
    "ignore_region",
}
VALID_PHASE_QUIZ_MODEL_PHASES = {
    "water",
    "aqueous",
    "gas",
    "mixed",
    "unscored",
}
PHASE_QUIZ_EXPECTED_ANSWERS = {
    "water": frozenset({"water_or_no_visible_co2"}),
    "aqueous": frozenset({"only_aqueous_co2", "predominantly_aqueous_co2"}),
    "gas": frozenset({"predominantly_free_gas", "free_gas_only"}),
    "mixed": frozenset(
        {
            "predominantly_aqueous_co2",
            "mixed_free_and_aqueous_co2",
            "predominantly_free_gas",
        }
    ),
}
PHASE_QUIZ_NONSCORING_CHOICES = {
    "mixed_free_aqueous_co2_and_water",
    "co2_present_phase_uncertain",
    "not_assessable",
    "phase_signal_artifact",
    "ignore_region",
}
TRANSFERABLE_PHASE_LABEL_SCHEMA = VISIBLE_PHASE_LABEL_SCHEMA
PHASE_QUIZ_CANONICAL_CLASSES = dict(VISIBLE_PHASE_CANONICAL_CLASSES) | {
    "ignore_region": "ignored"
}
MAX_ANNOTATION_POINTS = 1000
REVIEW_FRAME_WIDTH = 1600
REVIEW_FRAME_HEIGHT = 900
REVIEW_HEADER_HEIGHT = 126
REVIEW_FRAME_MARGIN = 18
NATIVE_REVIEW_JPEG_QUALITY = 98


def isolated_phase_components_rgba(
    rgba: Any,
    *,
    max_area_fraction: float = ISOLATED_COMPONENT_MAX_AREA_FRACTION,
    color_tolerance: int = ISOLATED_COMPONENT_COLOR_TOLERANCE,
    min_separation_fraction: float = ISOLATED_COMPONENT_MIN_SEPARATION_FRACTION,
) -> dict[str, Any]:
    """Return small connected islands in a filled phase map as normalized boxes."""

    import cv2
    import numpy as np

    pixels = np.asarray(rgba, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] not in {3, 4}:
        raise ValueError(f"Expected RGB(A) phase map, received shape {pixels.shape}")
    height, width = map(int, pixels.shape[:2])
    rgb = pixels[:, :, :3]
    alpha = (
        pixels[:, :, 3]
        if pixels.shape[2] == 4
        else np.full((height, width), 255, dtype=np.uint8)
    )
    support = np.zeros((height, width), dtype=np.uint8)
    tolerance = max(0, int(color_tolerance))
    for color in PHASE_MAP_RGB:
        lower = np.maximum(0, np.asarray(color, dtype=np.int16) - tolerance).astype(
            np.uint8
        )
        upper = np.minimum(255, np.asarray(color, dtype=np.int16) + tolerance).astype(
            np.uint8
        )
        support = cv2.bitwise_or(support, cv2.inRange(rgb, lower, upper))
    support[alpha <= 16] = 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        support,
        connectivity=8,
    )
    max_area_px = max(4, int(round(width * height * float(max_area_fraction))))
    min_separation_px = max(
        2,
        int(round(min(width, height) * float(min_separation_fraction))),
    )
    areas = stats[:, cv2.CC_STAT_AREA]
    large_labels = areas > max_area_px
    large_labels[0] = False
    labels_near_large: set[int] = set()
    if bool(np.any(large_labels)):
        large_support = (large_labels[labels].astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * min_separation_px + 1, 2 * min_separation_px + 1),
        )
        near_large = cv2.dilate(large_support, kernel)
        labels_near_large = {
            int(value)
            for value in np.unique(labels[(near_large > 0) & (labels > 0)])
        }
    components: list[dict[str, float | int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > max_area_px or label in labels_near_large:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        components.append(
            {
                "x": x / width,
                "y": y / height,
                "width": component_width / width,
                "height": component_height / height,
                "center_x": float(centroids[label, 0]) / width,
                "center_y": float(centroids[label, 1]) / height,
                "area_px": area,
            }
        )
    components.sort(key=lambda row: int(row["area_px"]))
    return {
        "components": components,
        "max_area_px": max_area_px,
        "max_area_fraction": float(max_area_fraction),
        "min_separation_px": min_separation_px,
        "min_separation_fraction": float(min_separation_fraction),
        "total_components": max(0, count - 1),
    }


@functools.lru_cache(maxsize=256)
def _isolated_phase_components_path_cached(
    path_text: str,
    size: int,
    mtime_ns: int,
) -> dict[str, Any]:
    del size, mtime_ns
    import cv2

    image = cv2.imread(path_text, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Could not decode phase map: {path_text}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return isolated_phase_components_rgba(image)


def isolated_phase_components_path(path: Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return _isolated_phase_components_path_cached(
        str(source.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _phase_quiz_feature_vector(proposal: Mapping[str, Any]) -> tuple[float, ...]:
    stats = proposal.get("model_stats") or {}
    delta = stats.get("mean_delta_rgb") or (0.0, 0.0, 0.0)
    current = stats.get("mean_current_chromaticity") or (0.0, 0.0, 0.0)
    reference = stats.get("mean_reference_chromaticity") or (0.0, 0.0, 0.0)
    return (
        float(stats.get("optical_change_quantile") or 0.0),
        math.log1p(10.0 * max(0.0, float(stats.get("mean_rgb_change_norm") or 0.0))),
        *(4.0 * float(value or 0.0) for value in delta),
        *(
            4.0 * (float(value or 0.0) - float(reference[index] or 0.0))
            for index, value in enumerate(current)
        ),
    )


def _phase_quiz_feature_distance(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    vector_left = _phase_quiz_feature_vector(left)
    vector_right = _phase_quiz_feature_vector(right)
    return math.sqrt(
        sum((value - vector_right[index]) ** 2 for index, value in enumerate(vector_left))
        / max(1, len(vector_left))
    )


def _phase_quiz_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if int(left.get("frame_index") or 0) != int(right.get("frame_index") or 0):
        return 0.0
    left_x = float(left.get("x") or 0.0)
    left_y = float(left.get("y") or 0.0)
    left_width = float(left.get("width") or 0.0)
    left_height = float(left.get("height") or 0.0)
    right_x = float(right.get("x") or 0.0)
    right_y = float(right.get("y") or 0.0)
    right_width = float(right.get("width") or 0.0)
    right_height = float(right.get("height") or 0.0)
    intersection = max(
        0.0,
        min(left_x + left_width, right_x + right_width) - max(left_x, right_x),
    ) * max(
        0.0,
        min(left_y + left_height, right_y + right_height) - max(left_y, right_y),
    )
    union = left_width * left_height + right_width * right_height - intersection
    return intersection / union if union > 0.0 else 0.0


def _phase_quiz_uncertainty(item: Mapping[str, Any]) -> float:
    """Read predictive entropy, retaining a conservative legacy fallback."""

    stats = item.get("model_stats") or {}
    for key in (
        "predictive_entropy_mean",
        "predictive_uncertainty_proxy",
        "uncertainty_mean",
    ):
        raw = stats.get(key, item.get(key))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return max(0.0, min(1.0, value))
    fallback: list[float] = []
    try:
        gas_fraction = float(stats.get("median_gas_mass_fraction"))
    except (TypeError, ValueError):
        pass
    else:
        if math.isfinite(gas_fraction):
            fallback.append(max(0.0, 1.0 - 2.0 * abs(gas_fraction - 0.5)))
    try:
        purity = float(stats.get("phase_purity"))
    except (TypeError, ValueError):
        pass
    else:
        if math.isfinite(purity):
            fallback.append(max(0.0, min(1.0, 1.0 - purity)))
    stratum = str(stats.get("sampling_stratum") or "")
    if "decision_boundary" in stratum or "uncertain" in stratum:
        fallback.append(0.9)
    return max(fallback, default=0.0)


def _strategic_phase_quiz_batch(
    pool: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    count: int,
) -> list[dict[str, Any]]:
    """Continue a phase/facies-balanced pool without repeating spatial regions."""

    phases = ("gas", "aqueous", "mixed", "water")
    by_id = {str(item.get("proposal_id") or ""): dict(item) for item in pool}
    selected = [by_id[value] for value in map(str, selected_ids) if value in by_id]
    selected_set = {str(item.get("proposal_id") or "") for item in selected}
    remaining = [
        dict(item)
        for item in pool
        if str(item.get("proposal_id") or "") not in selected_set
        and str(item.get("suggested_phase") or "") in phases
        and (item.get("model_stats") or {}).get("facies_id") is not None
    ]
    phase_counts = {phase: 0 for phase in phases}
    facies_counts: dict[int, int] = {}
    combination_counts: dict[tuple[str, int], int] = {}
    frame_counts: dict[int, int] = {}
    for item in selected:
        stats = item.get("model_stats") or {}
        phase = str(item.get("suggested_phase") or "")
        if phase not in phases or stats.get("facies_id") is None:
            continue
        facies_id = int(stats["facies_id"])
        frame = int(item.get("frame_index") or 0)
        phase_counts[phase] += 1
        facies_counts[facies_id] = facies_counts.get(facies_id, 0) + 1
        combination_counts[(phase, facies_id)] = (
            combination_counts.get((phase, facies_id), 0) + 1
        )
        frame_counts[frame] = frame_counts.get(frame, 0) + 1

    phase_rank = {phase: index for index, phase in enumerate(phases)}
    batch: list[dict[str, Any]] = []
    while remaining and len(batch) < max(0, int(count)):
        active = selected + batch

        def spatial_distance(item: Mapping[str, Any]) -> float:
            stats = item.get("model_stats") or {}
            facies_id = int(stats.get("facies_id"))
            xy = stats.get("normalized_center_xy") or (0.5, 0.5)
            same_facies = [
                other
                for other in active
                if int((other.get("model_stats") or {}).get("facies_id") or -1)
                == facies_id
            ]
            if not same_facies:
                return 1.0
            return min(
                math.hypot(
                    float(xy[0])
                    - float(
                        ((other.get("model_stats") or {}).get("normalized_center_xy") or (0.5, 0.5))[0]
                    ),
                    float(xy[1])
                    - float(
                        ((other.get("model_stats") or {}).get("normalized_center_xy") or (0.5, 0.5))[1]
                    ),
                )
                for other in same_facies
            )

        def rank(item: Mapping[str, Any]) -> tuple[Any, ...]:
            stats = item.get("model_stats") or {}
            phase = str(item.get("suggested_phase") or "")
            facies_id = int(stats.get("facies_id"))
            frame = int(item.get("frame_index") or 0)
            return (
                phase_counts.get(phase, 0),
                facies_counts.get(facies_id, 0),
                combination_counts.get((phase, facies_id), 0),
                frame_counts.get(frame, 0),
                -_phase_quiz_uncertainty(item),
                -spatial_distance(item),
                -float(item.get("priority") or 0.0),
                phase_rank.get(phase, 99),
                facies_id,
                str(item.get("proposal_id") or ""),
            )

        chosen = min(remaining, key=rank)
        remaining.remove(chosen)
        batch.append(chosen)
        stats = chosen.get("model_stats") or {}
        phase = str(chosen.get("suggested_phase") or "")
        facies_id = int(stats["facies_id"])
        frame = int(chosen.get("frame_index") or 0)
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        facies_counts[facies_id] = facies_counts.get(facies_id, 0) + 1
        combination_counts[(phase, facies_id)] = (
            combination_counts.get((phase, facies_id), 0) + 1
        )
        frame_counts[frame] = frame_counts.get(frame, 0) + 1
    return batch


def select_adaptive_phase_quiz_batch(
    pool: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    answers: Sequence[Mapping[str, Any]],
    count: int = PHASE_QUIZ_ROUND_SIZE,
) -> list[dict[str, Any]]:
    """Select the next blind, optically informative phase-quiz regions."""

    selected_set = {str(value) for value in selected_ids}
    if any(
        str(item.get("suggested_phase") or "") in {"gas", "aqueous", "mixed", "water"}
        and (item.get("model_stats") or {}).get("facies_id") is not None
        for item in pool
    ):
        return _strategic_phase_quiz_batch(pool, selected_ids, count)
    by_id = {str(item.get("proposal_id") or ""): item for item in pool}
    selected = [by_id[value] for value in selected_ids if value in by_id]
    answer_by_id = {
        str(item.get("proposal_id") or ""): item
        for item in answers
        if str(item.get("candidate") or "").upper() == "A"
    }
    positive_choices = {
        "only_aqueous_co2",
        "predominantly_aqueous_co2",
        "predominantly_free_gas",
        "mixed_free_and_aqueous_co2",
        "mixed_aqueous_co2_and_water",
        "co2_present_phase_uncertain",
    }
    positive = [
        item
        for item in selected
        if str(answer_by_id.get(str(item.get("proposal_id") or ""), {}).get("human_choice") or "")
        in positive_choices
    ]
    water = [
        item
        for item in selected
        if answer_by_id.get(str(item.get("proposal_id") or ""), {}).get("human_choice")
        == "water_or_no_visible_co2"
    ]
    denominator = len(positive) + len(water)
    water_fraction = len(water) / denominator if denominator else 0.0

    def score(proposal: Mapping[str, Any]) -> float:
        stats = proposal.get("model_stats") or {}
        stratum = str(stats.get("sampling_stratum") or "")
        quantile = float(stats.get("optical_change_quantile") or 0.0)
        change = max(0.0, float(stats.get("mean_rgb_change_norm") or 0.0))
        value = 5.0 * quantile + math.log1p(10.0 * change)
        value += 8.0 * _phase_quiz_uncertainty(proposal)
        if stratum == "strong_corrected_optical_change":
            value += 7.0
        elif stratum == "optical_change_boundary":
            value += 4.0
        elif stratum == "moderate_corrected_optical_change":
            value += 3.0
        elif stratum == "weak_change_negative_control":
            value -= 5.0
        elif stratum == "domain_only_spatial_repair":
            value -= 12.0
        elif any(token in stratum for token in ("high_gas", "decision_boundary", "strong_optical")):
            value += 2.0
        if water_fraction > 0.55:
            value += 5.0 * quantile
        if positive:
            distance = min(_phase_quiz_feature_distance(proposal, item) for item in positive)
            value += 4.0 / (1.0 + 5.0 * distance)
        if water:
            distance = min(_phase_quiz_feature_distance(proposal, item) for item in water)
            value -= 3.0 / (1.0 + 5.0 * distance)
        if any(_phase_quiz_overlap(proposal, item) > 0.18 for item in selected):
            value -= 8.0
        return value

    ranked = sorted(
        (dict(item) for item in pool if str(item.get("proposal_id") or "") not in selected_set),
        key=score,
        reverse=True,
    )
    batch: list[dict[str, Any]] = []
    frame_counts: dict[int, int] = {}
    for maximum_per_frame in (1, 2, math.inf):
        for proposal in ranked:
            if len(batch) >= max(0, int(count)):
                break
            proposal_id = str(proposal.get("proposal_id") or "")
            if any(str(item.get("proposal_id") or "") == proposal_id for item in batch):
                continue
            frame = int(proposal.get("frame_index") or 0)
            used = frame_counts.get(frame, 0)
            if used >= maximum_per_frame:
                continue
            stratum = str((proposal.get("model_stats") or {}).get("sampling_stratum") or "")
            controls = sum(
                "negative_control"
                in str((item.get("model_stats") or {}).get("sampling_stratum") or "")
                or str((item.get("model_stats") or {}).get("sampling_stratum") or "")
                == "domain_only_spatial_repair"
                for item in batch
            )
            if (
                "negative_control" in stratum or stratum == "domain_only_spatial_repair"
            ) and controls >= 1:
                continue
            if any(_phase_quiz_overlap(proposal, item) > 0.18 for item in batch):
                continue
            batch.append(proposal)
            frame_counts[frame] = used + 1
        if len(batch) >= max(0, int(count)):
            break
    return batch[: max(0, int(count))]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_reference_guides(
    raw_guides: Any,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Validate persistent, normalized image-reference lines."""

    if raw_guides is None:
        return []
    if not isinstance(raw_guides, list):
        if strict:
            raise ValueError("reference_guides must be a list")
        return []
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_guides):
        if not isinstance(raw, Mapping):
            if strict:
                raise ValueError("Each reference guide must be an object")
            continue
        orientation = str(raw.get("orientation") or "").strip().casefold()
        try:
            position = float(raw.get("position"))
        except (TypeError, ValueError):
            position = math.nan
        if (
            orientation not in VALID_REFERENCE_GUIDE_ORIENTATIONS
            or not math.isfinite(position)
            or not 0.0 <= position <= 1.0
        ):
            if strict:
                raise ValueError(f"Invalid reference guide at index {index}")
            continue
        guide_id = str(raw.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", guide_id) or guide_id in seen:
            guide_id = f"guide-{index + 1}"
            suffix = 2
            while guide_id in seen:
                guide_id = f"guide-{index + 1}-{suffix}"
                suffix += 1
        seen.add(guide_id)
        label = str(raw.get("label") or "").strip()[:80]
        if not label:
            prefix = "H" if orientation == "horizontal" else "V"
            label = f"{prefix}{index + 1}"
        clean.append(
            {
                "id": guide_id,
                "orientation": orientation,
                "position": position,
                "label": label,
                "coordinate_space": "normalized_native_review_canvas_16x9",
                "created_utc": str(raw.get("created_utc") or "")[:64],
            }
        )
    return clean


def _clean_annotation_geometry(
    annotation: Mapping[str, Any],
) -> tuple[str, dict[str, float], list[list[float]]]:
    """Validate one normalized rectangle, freehand region, or pixel point."""

    shape = str(annotation.get("shape") or "rectangle").strip().lower()
    if shape not in VALID_ANNOTATION_SHAPES:
        raise ValueError(f"Invalid annotation shape: {shape!r}")

    if shape == "freehand":
        raw_points = annotation.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("Freehand annotation points must be a list")
        if len(raw_points) > MAX_ANNOTATION_POINTS:
            raise ValueError(
                f"Freehand annotations support at most {MAX_ANNOTATION_POINTS} points"
            )
        points: list[list[float]] = []
        for raw_point in raw_points:
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
                raise ValueError("Each freehand point must contain x and y")
            point = [float(raw_point[0]), float(raw_point[1])]
            if not all(math.isfinite(value) for value in point):
                raise ValueError("Freehand annotation coordinates must be finite")
            if not all(0.0 <= value <= 1.0 for value in point):
                raise ValueError("Freehand annotation must stay within the image")
            if not points or point != points[-1]:
                points.append(point)
        if len(points) < 3:
            raise ValueError("Freehand annotation requires at least three points")
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        coordinates = {
            "x": min(xs),
            "y": min(ys),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys),
        }
    else:
        points = []
        coordinates = {
            name: float(annotation.get(name, math.nan))
            for name in ("x", "y", "width", "height")
        }

    if not all(math.isfinite(value) for value in coordinates.values()):
        raise ValueError("Annotation coordinates must be finite")
    if (
        coordinates["x"] < 0.0
        or coordinates["y"] < 0.0
        or coordinates["width"] <= 0.0
        or coordinates["height"] <= 0.0
        or coordinates["x"] + coordinates["width"] > 1.000001
        or coordinates["y"] + coordinates["height"] > 1.000001
    ):
        raise ValueError("Annotation region must stay within the image")
    return shape, coordinates, points


def _run_number(run: str) -> int:
    """Return the terminal experiment number for AC and external run ids."""

    match = re.search(r"(\d+)$", run.casefold())
    if match is None:
        raise ValueError(f"Invalid run identifier: {run!r}")
    return int(match.group(1))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    # pathlib.Path.write_text gained the newline argument after Python 3.9.
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    with temporary.open("wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _latest_jsonl_object(path: Path) -> dict[str, Any] | None:
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
        except (TypeError, ValueError):
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return None


def _reviewer_identifier(name: str) -> str:
    from darsia.fluidflower.reviewer_preferences import reviewer_identifier
    return reviewer_identifier(name)


@contextmanager
def _exclusive_path_lock(
    path: Path,
    *,
    timeout_s: float = 30.0,
    stale_after_s: float = 180.0,
) -> Any:
    """Serialize review writes across browser and desktop processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.write(
                descriptor,
                (
                    f"host={socket.gethostname()} pid={os.getpid()} "
                    f"created={_utc_now()}\n"
                ).encode("utf-8"),
            )
        except FileExistsError:
            try:
                age_s = time.time() - path.stat().st_mtime
            except OSError:
                age_s = 0.0
            if age_s >= stale_after_s:
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for review lock: {path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_review_jpeg(path: Path, image: Any, *, quality: int = 94) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.jpg")
    if not cv2.imwrite(
        str(temporary),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    ):
        raise RuntimeError(f"Could not write {temporary}")
    os.replace(temporary, path)


def _write_review_png(path: Path, image: Any) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    values = image
    if getattr(values, "ndim", 0) == 3 and values.shape[2] == 4:
        values = cv2.cvtColor(values, cv2.COLOR_RGBA2BGRA)
    elif getattr(values, "ndim", 0) == 3 and values.shape[2] == 3:
        values = cv2.cvtColor(values, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(
        str(temporary),
        values,
        [int(cv2.IMWRITE_PNG_COMPRESSION), 6],
    ):
        raise RuntimeError(f"Could not write {temporary}")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    path = resolve_review_path(path)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    path = resolve_review_path(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


@functools.lru_cache(maxsize=256)
def _calibration_records(path_text: str) -> tuple[dict[str, str], ...]:
    path = Path(path_text)
    if path.suffix.casefold() == ".json":
        # Saved optical maps have a native image timeline, not calibrated mass
        # curves. Preserve its times without manufacturing CSV/mass evidence.
        if path.name != "manifest.json" or path.parent.parent.name != "native_frames":
            raise ValueError("Expected a native review frame manifest")
        manifest = _read_json(path)
        frames = manifest.get("frame_sources") or []
        count = manifest.get("frame_count")
        if type(count) is not int or count <= 0 or len(frames) != count:
            raise ValueError("Incomplete native review timeline")
        records = []
        for index, frame in enumerate(frames):
            time = float(frame["elapsed_h"])
            digest = frame.get("source_sha256")
            if (frame.get("frame_index") != index or not math.isfinite(time)
                    or not isinstance(digest, str) or len(digest) != 64):
                raise ValueError("Invalid native review frame/time/source binding")
            records.append(dict(frame_index=str(index), elapsed_h=str(time), source_sha256=digest,
                                **({"source_name": str(frame["source_name"])} if frame.get("source_name") else {})))
        return tuple(records)
    with Path(path_text).open("r", encoding="utf-8-sig", newline="") as stream:
        return tuple(dict(row) for row in csv.DictReader(stream))


@dataclass(frozen=True)
class Candidate:
    run: str
    role: str
    family: str
    variant: str
    candidate_id: str
    grid_order: int
    params_sha256: str
    normalized_mae: float | None
    mass_plot: Path
    contour_movie: Path
    per_frame_csv: Path
    summary_path: Path
    plot_seconds: float
    image_seconds: float
    fps: float
    frame_count: int
    plot_title: str = ""
    quality_flags: Mapping[str, Any] | None = None
    dataset: str = ""
    review_output_dir: Path | None = None
    phase_map_output_dir: Path | None = None
    phase_model_only: bool = False
    raw_phase_map_output_dir: Path | None = None
    optical_model_sha256: str = ""
    presence_diagnostics_dir: Path | None = None
    phase_support_format: str = ""
    phase_quiz_proposals_path: Path | None = None
    phase_quiz_proposals_sha256: str = ""
    presentation_deferred: bool = False

    def provenance(self) -> dict[str, Any]:
        return {
            **({"phase_review_basis": self.phase_support_format if self.phase_support_format == "baseline-relative-optical-regions-v1"
                else "studio-optical-without-context-v1"} if self.phase_model_only else {}),
            "run": self.run,
            "role": self.role,
            "family": self.family,
            "variant": self.variant,
            "candidate_id": self.candidate_id,
            "params_sha256": self.params_sha256,
            "normalized_mae": self.normalized_mae,
            "phase_model_only": self.phase_model_only,
            **({"phase_quiz_proposals_path": str(self.phase_quiz_proposals_path),
                "phase_quiz_proposals_sha256": self.phase_quiz_proposals_sha256}
               if self.phase_quiz_proposals_path else {}),
            "raw_phase_map_output_dir": str(self.raw_phase_map_output_dir) if self.raw_phase_map_output_dir else None,
            "optical_model_sha256": self.optical_model_sha256,
            "presence_diagnostics_dir": str(self.presence_diagnostics_dir) if self.presence_diagnostics_dir else None,
            **({"phase_support_format": self.phase_support_format} if self.phase_support_format else {}),
            "mass_plot": str(self.mass_plot),
            "contour_movie": str(self.contour_movie),
            "per_frame_csv": str(self.per_frame_csv),
            "summary": str(self.summary_path),
            "frame_count": self.frame_count,
            "plot_title": self.plot_title,
            "quality_flags": dict(self.quality_flags or {}),
            "dataset": self.dataset,
            "review_output_dir": (
                str(self.review_output_dir) if self.review_output_dir else None
            ),
            "phase_map_output_dir": (
                str(self.phase_map_output_dir) if self.phase_map_output_dir else None
            ),
        }


@dataclass(frozen=True)
class ReviewPair:
    run: str
    candidate_a: Candidate
    candidate_b: Candidate
    comparison_mode: str = "pair"

    def candidate(self, label: str) -> Candidate:
        if label == "A":
            return self.candidate_a
        if label == "B":
            return self.candidate_b
        raise KeyError(label)


@dataclass(frozen=True)
class ReviewInventory:
    pairs: tuple[ReviewPair, ...]
    unpaired: Mapping[str, tuple[Candidate, ...]]
    manifest_path: Path
    pending_runs: tuple[str, ...] = ()


def _manifest_relative_path(selection_dir: Path, value: Any) -> Path:
    """Resolve portable manifest paths relative to the manifest directory."""

    path = Path(str(value))
    if not path.is_absolute():
        path = selection_dir / path
    return resolve_review_path(path)


def _t4b_saved_review_candidate(selection_dir: Path, job: Mapping[str, Any], root: Path) -> Candidate:
    """Adapt admitted joint-head saved maps without the legacy G3 warning gate."""
    from darsia.fluidflower.phase_saved_revision_review import t4b_saved_review
    return _admitted_saved_review_candidate(selection_dir, job, root, t4b_saved_review(root),
        revision="T4b", kind="ffac_g3_t4b_floor_review")


def _t4c_saved_review_candidate(selection_dir: Path, job: Mapping[str, Any], root: Path) -> Candidate:
    """Adapt exactly admitted calibrated placements to their native timeline."""
    from darsia.fluidflower.phase_saved_floor_review import t4c_saved_review
    return _admitted_saved_review_candidate(selection_dir, job, root, t4c_saved_review(root),
        revision="T4c", kind="ffac_g3_t4c_floor_review")


def _p1_final3_saved_review_candidate(selection_dir, job, root):
    from darsia.fluidflower.phase_saved_contact_review import p1_final3_saved_review, KIND
    return _admitted_saved_review_candidate(selection_dir, job, root, p1_final3_saved_review(root),
        revision="P1-final3", kind=KIND)


def _p1_final2_saved_review_candidate(selection_dir, job, root):
    from darsia.fluidflower.phase_saved_categorical_review import p1_final2_saved_review, KIND
    return _admitted_saved_review_candidate(selection_dir, job, root, p1_final2_saved_review(root),
        revision="P1-final2", kind=KIND)


def _p1_final_saved_review_candidate(selection_dir, job, root):
    from darsia.fluidflower.phase_saved_boundary_review import p1_final_saved_review, KIND
    return _admitted_saved_review_candidate(selection_dir, job, root, p1_final_saved_review(root),
        revision="P1-final", kind=KIND)


def _admitted_saved_review_candidate(selection_dir, job, root, checked, *, revision, kind):
    """Shared native-map adapter, after the revision's strict admission."""
    validation = resolve_review_path(root / "validation").resolve()

    def require(condition, message):
        if not condition:
            raise ValueError(revision + " saved review " + message)

    def path(value):
        return _manifest_relative_path(selection_dir, value).resolve()

    def same(left, right):
        try:
            return Path(left).samefile(right)
        except OSError:
            return False

    registration = checked["registration"]
    identity = tuple(checked["model_identity"])
    pipeline = checked["pipeline_sha256"]
    require(registration.get("registration_kind") == kind
        and pipeline == job.get("params_sha256") and identity[1] == job.get("optical_model_sha256"),
        "job differs from its admitted joint model or pipeline")
    flags, run = job.get("quality_flags") or {}, str(job["run"]).casefold()
    require(flags.get("approved_g3_downstream") is True and flags.get("mass_analysis_allowed") is False
        and flags.get("scientific_acceptance") is False and job.get("output_mode") == "review"
        and job.get("phase_model_only") is True, "requires intact review-only mass restrictions")
    binding = _read_json(root / "validation/validation_binding.json")
    manifest = _read_json(root / "validation/postflight_manifest.json")
    require(all((value.get("model_id"), value.get("model_sha256")) == identity
        and value.get("pipeline_sha256") == pipeline for value in (binding, manifest))
        and (binding.get("selected_candidates") or {}).get(run) == job.get("candidate_id")
        and [value for value in manifest["jobs"] if str(value["run"]).casefold() == run] == [job],
        "job differs from its immutable selected campaign")
    selection_dir = resolve_review_path(selection_dir).resolve()
    selected_binding = _read_json(selection_dir / "validation_binding.json")
    selected_manifest = _read_json(selection_dir / "postflight_manifest.json")
    prepared = path(binding["combined_root"])
    if same(selection_dir, prepared) or same(selection_dir, validation):
        require(selected_binding == binding and selected_manifest == manifest
            and _read_json(prepared / "validation_binding.json") == binding
            and _read_json(prepared / "postflight_manifest.json") == manifest,
            "publication pointer differs from its prepared metadata")
    else:
        package = validation / "run_packages" / run
        source_sha = _sha256_file(resolve_review_path(root / "review_source" / run / "postflight_manifest.json"))
        require(same(selection_dir, package) and same(path(selected_binding["combined_root"]), package)
            and selected_manifest.get("jobs") == [job]
            and selected_binding.get("selected_candidates") == {run: job["candidate_id"]}
            and all((value.get("model_id"), value.get("model_sha256")) == identity
                and value.get("pipeline_sha256") == pipeline
                and value.get("physics_runtime") == binding.get("physics_runtime")
                and value.get("output_mode") == "review"
                and value.get("source_manifest_sha256") == source_sha
                for value in (selected_binding, selected_manifest)),
            "per-run package differs from its model, job, runtime or source")
    review, maps = path(job["review_output_dir"]), path(job["phase_map_output_dir"])
    require(review.is_dir() and maps.is_dir() and maps.is_relative_to(validation),
        "native source or phase map folder is absent or unbound")
    native = review / "native_frames" / run / "manifest.json"
    require(_sha256_file(native) == (registration.get("native_run_manifest_sha256") or {}).get(run),
        "native timeline differs from its registration")
    source = _read_json(native)
    count = source.get("frame_count")
    require(type(count) is int and count > 0 and len(source.get("frame_sources") or []) == count
        and source.get("dataset") == job.get("dataset") == "ffac" and source.get("run") == run,
        "native timeline has an invalid run, series or frame count")
    return Candidate(run=run, role=job["role"], family=job.get("family", "unknown"), variant=job["variant"],
        candidate_id=job["candidate_id"], grid_order=job.get("grid_order", 1), params_sha256=pipeline,
        normalized_mae=None, mass_plot=root / "calibration_mass_fit.png", contour_movie=root / "calibration_review.mp4",
        per_frame_csv=native, summary_path=root / "summary.json", plot_seconds=0., image_seconds=0., fps=0.,
        frame_count=count, quality_flags=flags, dataset=job["dataset"], review_output_dir=review,
        phase_map_output_dir=maps, phase_model_only=True, optical_model_sha256=identity[1],
        raw_phase_map_output_dir=None, presence_diagnostics_dir=None,
        phase_support_format=job.get("phase_support_format", ""))


def _selected_policy_a_review_candidate(selection_dir,job,root,admission=None):
    """Use saved selected maps, with no legacy mass/movie assets or inference."""
    from darsia.fluidflower.phase_saved_selected_registration import selected_saved_review
    from darsia.fluidflower.phase_selected_policy_review import load_native_source,load_profile,runtime_identity
    admission=admission or selected_saved_review(root)
    profile=load_profile(root)
    run=str(job["run"]).casefold()
    source_root=_manifest_relative_path(selection_dir,job["review_output_dir"])
    state=load_native_source(source_root,run,verify_geometry=False)
    registration=admission["registration"]
    native=state["native_path"]
    if (root != _manifest_relative_path(selection_dir,job["candidate_folder"]).resolve()
            or job.get("params_sha256") != admission["pipeline_sha256"]
            or job.get("physics_runtime") != runtime_identity(profile)
            or job.get("native_manifest_sha256") != registration["native_run_manifest_sha256"].get(run)
            or _sha256_file(native) != job["native_manifest_sha256"]
            or job.get("frame_count") != state["native"]["frame_count"]
            or job.get("dataset") != "ffac" or source_root.resolve() != resolve_review_path(root/"source_alias").resolve()):
        raise ValueError("Selected Studio job differs from its registered camera timeline")
    return Candidate(run=run,role=str(job.get("role") or "candidate"),family=str(job.get("family") or "phase"),
        variant=str(job.get("variant") or "selected_policy_a"),candidate_id=job["candidate_id"],grid_order=int(job.get("grid_order") or 1),
        params_sha256=admission["pipeline_sha256"],normalized_mae=None,mass_plot=root/"no_mass_plot",contour_movie=root/"no_movie",
        per_frame_csv=native,summary_path=root/"summary.json",plot_seconds=0.,image_seconds=0.,fps=8.,
        frame_count=state["native"]["frame_count"],quality_flags=dict(job.get("quality_flags") or {}),dataset="ffac",
        review_output_dir=source_root,phase_map_output_dir=_manifest_relative_path(selection_dir,job["phase_map_output_dir"]),
        phase_model_only=True,optical_model_sha256=admission["model_identity"][1],phase_support_format=profile["format"])


def _candidate_from_job(selection_dir: Path, job: Mapping[str, Any], *, selected_admission=None) -> Candidate:
    if (job.get("quality_flags") or {}).get("selected_camera_review") is True:
        root=_manifest_relative_path(selection_dir,job["candidate_folder"]).resolve()
        return _selected_policy_a_review_candidate(selection_dir,job,root,selected_admission)
    if (job.get("quality_flags") or {}).get("approved_g3_downstream") is True:
        root = _manifest_relative_path(selection_dir, job["candidate_folder"]).resolve()
        if (root / "p1_final3_plan.json").is_file():
            from darsia.fluidflower.phase_saved_registration import registration_path
            registration = _read_json(resolve_review_path(registration_path(root.name)))
            if registration.get("registration_kind") != "ffac_g3_t4c_p1_final3_review":
                raise ValueError("P1-final3 saved review has no exact reason-b removal registration")
            return _p1_final3_saved_review_candidate(selection_dir, job, root)
        if (root / "p1_final2_plan.json").is_file():
            from darsia.fluidflower.phase_saved_registration import registration_path
            registration = _read_json(resolve_review_path(registration_path(root.name)))
            if registration.get("registration_kind") != "ffac_g3_t4c_p1_final2_review":
                raise ValueError("P1-final2 saved review has no exact A–D postprocessing registration")
            return _p1_final2_saved_review_candidate(selection_dir, job, root)
        if (root / "p1_final_plan.json").is_file():
            from darsia.fluidflower.phase_saved_registration import registration_path
            registration = _read_json(resolve_review_path(registration_path(root.name)))
            if registration.get("registration_kind") != "ffac_g3_t4c_p1_final_review":
                raise ValueError("P1-final saved review has no exact post-processing registration")
            return _p1_final_saved_review_candidate(selection_dir, job, root)
        if (root / "t4c_floor_plan.json").is_file():
            from darsia.fluidflower.phase_saved_registration import registration_path
            registration = _read_json(resolve_review_path(registration_path(root.name)))
            if registration.get("registration_kind") != "ffac_g3_t4c_floor_review":
                raise ValueError("T4c saved review has no exact calibrated-placement registration")
            return _t4c_saved_review_candidate(selection_dir, job, root)
        if (root / "t4b_floor_plan.json").is_file():
            from darsia.fluidflower.phase_saved_registration import registration_path
            registration = _read_json(resolve_review_path(registration_path(root.name)))
            if registration.get("registration_kind") == "ffac_g3_t4b_floor_review":
                return _t4b_saved_review_candidate(selection_dir, job, root)
        from darsia.fluidflower.phase_saved_candidate_admission import saved_review_candidate
        return saved_review_candidate(selection_dir, job)
    run = str(job["run"]).casefold()
    variant = str(job["variant"])
    folder_value = job.get("candidate_folder")
    folder = (
        _manifest_relative_path(selection_dir, folder_value)
        if folder_value
        else selection_dir / "postflight" / variant / run
    )
    summary_path = folder / "summary.json"
    mass_plot = folder / "calibration_mass_fit.png"
    contour_movie = folder / "calibration_review.mp4"
    per_frame_csv = folder / "calibration_review_per_frame.csv"
    missing = [
        str(path)
        for path in (summary_path, per_frame_csv)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing postflight assets for {run}/{variant}: {', '.join(missing)}"
        )

    summary = _read_json(summary_path)
    movie = summary.get("calibration_review_movie") or {}
    candidate_id = str(
        job.get("candidate_id")
        or summary.get("selection_candidate_id")
        or movie.get("params_sha256")
        or f"{run}__{variant}"
    )
    digest = str(job.get("params_sha256") or movie.get("params_sha256") or "")
    phase_model_only = bool(job.get("optical_model_sha256"))
    if job.get("presentation_status") == "deferred":
        if not phase_model_only:
            raise ValueError("Deferred presentation requires a native phase model candidate")
        review_output_dir = _manifest_relative_path(selection_dir, job["review_output_dir"])
        _deferred_presentation_validator(review_output_dir, run)(summary, run=run, candidate_id=candidate_id,
            params_sha256=digest,
            review_output_dir=review_output_dir,
            phase_map_output_dir=_manifest_relative_path(selection_dir, job["phase_map_output_dir"]),
            per_frame_csv=per_frame_csv)
    else:
        missing = [str(path) for path in (mass_plot, contour_movie) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing postflight assets for {run}/{variant}: {', '.join(missing)}")
    nmae = None if phase_model_only else summary.get("normalized_mae")
    quality_flags = (
        dict(job.get("quality_flags") or {})
        if isinstance(job.get("quality_flags"), Mapping)
        else {}
    )
    review_output_value = job.get("review_output_dir")
    return Candidate(
        run=run,
        role=str(job.get("role") or "candidate"),
        family=str(job.get("family") or "unknown"),
        variant=variant,
        candidate_id=candidate_id,
        grid_order=int(job.get("grid_order") or 999),
        params_sha256=digest,
        normalized_mae=float(nmae) if nmae is not None else None,
        mass_plot=mass_plot.resolve(),
        contour_movie=contour_movie.resolve(),
        per_frame_csv=per_frame_csv.resolve(),
        summary_path=summary_path.resolve(),
        plot_seconds=float(movie.get("plot_seconds") or 0.0),
        image_seconds=float(movie.get("image_seconds_per_calibration_frame") or 1.0),
        fps=float(movie.get("fps") or 5.0),
        frame_count=int(movie.get("calibration_frame_count") or 0),
        phase_model_only=phase_model_only,
        presentation_deferred=job.get("presentation_status") == "deferred",
        raw_phase_map_output_dir=_manifest_relative_path(selection_dir, job["raw_phase_map_output_dir"])
            if job.get("raw_phase_map_output_dir") else None,
        optical_model_sha256=str(job.get("optical_model_sha256") or ""),
        presence_diagnostics_dir=_manifest_relative_path(selection_dir, job["presence_diagnostics_dir"])
            if job.get("presence_diagnostics_dir") else None,
        phase_support_format=str(job.get("phase_support_format") or ""),
        phase_quiz_proposals_path=_manifest_relative_path(selection_dir, job["phase_quiz_proposals_path"])
            if job.get("phase_quiz_proposals_path") else None,
        phase_quiz_proposals_sha256=str(job.get("phase_quiz_proposals_sha256") or ""),
        plot_title=str(
            summary.get("review_plot_title")
            or f"{run.upper()} calibration: injected and detected CO2 mass"
        ),
        quality_flags=quality_flags,
        dataset=str(job.get("dataset") or quality_flags.get("dataset") or ""),
        review_output_dir=(
            _manifest_relative_path(selection_dir, review_output_value)
            if review_output_value
            else None
        ),
        phase_map_output_dir=(
            _manifest_relative_path(selection_dir, job["phase_map_output_dir"])
            if job.get("phase_map_output_dir")
            else None
        ),
    )


@functools.lru_cache(maxsize=128)
def _sanitized_plot_bytes(path_text: str, run: str, title: str = "") -> bytes:
    import cv2

    image = cv2.imread(path_text, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode mass plot: {path_text}")
    # The source title contains the role/variant and objective. Hide both so
    # the visual review is driven by the curves rather than candidate labels.
    header_bottom = min(image.shape[0], 132)
    image[:header_bottom, :] = 255
    cv2.putText(
        image,
        title or f"{run.upper()} calibration: injected and detected CO2 mass",
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (35, 39, 42),
        2,
        cv2.LINE_AA,
    )
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Could not encode mass plot: {path_text}")
    return encoded.tobytes()


@functools.lru_cache(maxsize=256)
def _contour_frame_bytes(
    movie_text: str,
    csv_text: str,
    run: str,
    label: str,
    frame_index: int,
    frame_count: int,
    plot_seconds: float,
    image_seconds: float,
    fps: float,
) -> bytes:
    import cv2

    if not 0 <= frame_index < frame_count:
        raise IndexError(frame_index)
    records = _calibration_records(csv_text)
    elapsed_h = (
        float(records[frame_index]["elapsed_h"])
        if frame_index < len(records)
        else float("nan")
    )
    plot_frames = max(1, int(round(plot_seconds * fps)))
    repeated_frames = max(1, int(round(image_seconds * fps)))
    movie_frame = plot_frames + frame_index * repeated_frames + repeated_frames // 2

    capture = cv2.VideoCapture(movie_text)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open contour movie: {movie_text}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, movie_frame)
        success, image = capture.read()
    finally:
        capture.release()
    if not success or image is None:
        raise RuntimeError(
            f"Could not decode contour frame {frame_index} from {movie_text}"
        )

    # Preserve mass values and contour legend, but replace the source variant
    # in the first header line with the blinded A/B label.
    image[:48, :] = (19, 23, 27)
    elapsed_label = f"{elapsed_h:.3f} h" if elapsed_h == elapsed_h else "n/a"
    cv2.putText(
        image,
        (
            f"{run.upper()}  |  Candidate {label}  |  calibration frame "
            f"{frame_index + 1}/{frame_count}  |  t={elapsed_label}"
        ),
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 247, 249),
        2,
        cv2.LINE_AA,
    )
    success, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 91]
    )
    if not success:
        raise RuntimeError(f"Could not encode contour frame from {movie_text}")
    return encoded.tobytes()


def _blind_pair(
    run: str,
    candidates: Sequence[Candidate],
    *,
    blind_seed: str,
) -> ReviewPair:
    if len(candidates) != 2:
        raise ValueError(f"{run} requires exactly two candidates, got {len(candidates)}")
    ordered = sorted(candidates, key=lambda item: (item.grid_order, item.candidate_id))
    fingerprint = "|".join(
        (blind_seed, run, *(item.candidate_id for item in ordered))
    )
    swap = hashlib.sha256(fingerprint.encode("utf-8")).digest()[0] & 1
    if swap:
        ordered.reverse()
    return ReviewPair(run=run, candidate_a=ordered[0], candidate_b=ordered[1])


def discover_review_inventory(
    selection_dir: Path,
    *,
    blind_seed: str = DEFAULT_BLIND_SEED,
    allow_incomplete: bool = False,
) -> ReviewInventory:
    manifest_path = selection_dir / "postflight_manifest.json"
    if allow_incomplete and not manifest_path.is_file():
        return ReviewInventory(
            pairs=(),
            unpaired={},
            manifest_path=manifest_path.resolve(),
            pending_runs=(),
        )
    manifest = _read_json(manifest_path)
    physics_profile = manifest.get("physics_profile") or {}
    phase_support_format = ((("unified-phase-decision-local-age-v2"
                             if physics_profile["unified_phase_decision"].get("format") == "unified-phase-decision-local-age-runtime-v2"
                             else "unified-phase-decision-v1") if physics_profile.get("unified_phase_decision")
                             else str(physics_profile.get("format") or ""))
                            if isinstance(physics_profile, Mapping) else "")
    jobs = manifest.get("jobs") or []
    if not isinstance(jobs, list) or (not jobs and not allow_incomplete):
        raise RuntimeError(f"No postflight jobs in {manifest_path}")

    selected_admission=None
    if physics_profile.get("format") == "ffac-selected-policy-a-review-physics-v1":
        from darsia.fluidflower.phase_saved_selected_registration import selected_saved_review
        selected_admission=selected_saved_review(_manifest_relative_path(selection_dir,manifest["model_root"]))

    grouped: dict[str, dict[str, Candidate]] = {}
    expected_by_run: dict[str, int] = {}
    incomplete_runs: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            continue
        run = str(job.get("run") or "").casefold()
        if not run:
            continue
        expected_by_run[run] = expected_by_run.get(run, 0) + 1
        try:
            candidate = (_candidate_from_job(selection_dir,job,selected_admission=selected_admission)
                if selected_admission is not None else _candidate_from_job(selection_dir,job))
            if not candidate.phase_support_format and phase_support_format:
                candidate = replace(candidate, phase_support_format=phase_support_format)
        except FileNotFoundError:
            if not allow_incomplete:
                raise
            incomplete_runs.add(run)
            continue
        grouped.setdefault(candidate.run, {})[candidate.candidate_id] = candidate

    pairs: list[ReviewPair] = []
    unpaired: dict[str, tuple[Candidate, ...]] = {}
    for run, by_id in grouped.items():
        if len(by_id) < expected_by_run.get(run, len(by_id)):
            incomplete_runs.add(run)
            continue
        candidates = tuple(by_id.values())
        if len(candidates) == 2:
            pairs.append(_blind_pair(run, candidates, blind_seed=blind_seed))
        elif len(candidates) == 1:
            candidate = candidates[0]
            pairs.append(
                ReviewPair(
                    run=run,
                    candidate_a=candidate,
                    candidate_b=candidate,
                    comparison_mode="single_candidate",
                )
            )
        else:
            unpaired[run] = tuple(
                sorted(candidates, key=lambda item: (item.grid_order, item.candidate_id))
            )
    # Review the newest experiments first, while keeping one deterministic order
    # across the server UI, portable bundle, progress counters, and exports.
    pairs.sort(key=lambda item: _run_number(item.run), reverse=True)
    unpaired = dict(
        sorted(
            unpaired.items(),
            key=lambda item: _run_number(item[0]),
            reverse=True,
        )
    )
    return ReviewInventory(
        pairs=tuple(pairs),
        unpaired=unpaired,
        manifest_path=manifest_path.resolve(),
        pending_runs=tuple(sorted(incomplete_runs, key=_run_number, reverse=True)),
    )


def _candidate_fingerprint(candidate: Candidate) -> str:
    payload = (
        f"review-asset-v2|{candidate.candidate_id}|{candidate.params_sha256}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _legacy_candidate_fingerprint(candidate: Candidate) -> str:
    return candidate.params_sha256[:16] or hashlib.sha256(
        candidate.candidate_id.encode("utf-8")
    ).hexdigest()[:16]


def _candidate_review_root(output_dir: Path, candidate: Candidate) -> Path:
    return resolve_review_path(getattr(candidate, "review_output_dir", None) or output_dir)


def _candidate_phase_map_root(output_dir: Path, candidate: Candidate) -> Path:
    return resolve_review_path(getattr(candidate, "phase_map_output_dir", None) or _candidate_review_root(
        output_dir, candidate
    ))


def _candidate_prediction_withheld(output_dir: Path, candidate: Candidate) -> bool:
    from darsia.fluidflower.phase_blind_review_policy import predictions_withheld
    run = getattr(candidate, "run", "")
    return predictions_withheld(_candidate_phase_map_root(output_dir, candidate), run)


def _withheld_prediction_asset(output_dir: Path, candidate: Candidate) -> Path:
    from darsia.fluidflower.phase_blind_review_policy import withheld_asset_path
    return withheld_asset_path(_candidate_phase_map_root(output_dir, candidate), candidate.run)


def _clean_frame_folder(output_dir: Path, candidate: Candidate, label: str) -> Path:
    fingerprint = _candidate_fingerprint(candidate)
    root = _candidate_review_root(output_dir, candidate)
    return root / "clean_frames" / candidate.run / f"{label}_{fingerprint}"


def _clean_frame_path(
    output_dir: Path,
    candidate: Candidate,
    label: str,
    frame_index: int,
) -> Path:
    return _clean_frame_folder(output_dir, candidate, label) / f"frame_{frame_index:03d}.jpg"


def _line_contour_frame_folder(
    output_dir: Path,
    candidate: Candidate,
    label: str,
) -> Path:
    fingerprint = _candidate_fingerprint(candidate)
    root = _candidate_review_root(output_dir, candidate)
    return root / "line_contour_frames" / candidate.run / f"{label}_{fingerprint}"


def _line_contour_frame_path(
    output_dir: Path,
    candidate: Candidate,
    label: str,
    frame_index: int,
) -> Path:
    return _line_contour_frame_folder(
        output_dir, candidate, label
    ) / f"frame_{frame_index:03d}.jpg"


def _reference_frame_path(output_dir: Path, run: str) -> Path:
    output_dir = resolve_review_path(output_dir)
    path = output_dir / "reference_frames" / run.casefold() / "pre_injection.jpg"
    return path.with_suffix(".png") if path.with_suffix(".png").is_file() else path


def _native_frame_folder(output_dir: Path, run: str) -> Path:
    return resolve_review_path(output_dir) / "native_frames" / run.casefold()


def _raw_review_photograph(output_dir: Path, run: str, frame_index: int | None = None, *,
                           reference: bool = False) -> Path | None:
    """The SHA-verified natural PNG of a RAW photographic review; None for a historical root."""
    from darsia.fluidflower.raw_photographic_review import read_raw_review_manifest, resolve_raw_photograph

    root = resolve_review_path(output_dir)
    if read_raw_review_manifest(root, run) is None:
        return None
    return resolve_raw_photograph(root, run, frame_index, mode="natural", reference=reference)


def _deferred_presentation_validator(review_output_dir: Path, run: str):
    """The deferred-presentation check for a run's native review.

    A RAW photographic review is checked by the current presentation reader,
    which reads its timeline from the review's frames. Every historical root
    keeps the frozen, configuration-bound admission copy.
    """
    from darsia.fluidflower.raw_photographic_review import read_raw_review_manifest

    if read_raw_review_manifest(resolve_review_path(review_output_dir), run) is not None:
        from darsia.fluidflower.phase_saved_presentation import validate_deferred_presentation
    else:
        from darsia.fluidflower.phase_saved_candidate_admission import validate_deferred_presentation
    return validate_deferred_presentation


def _native_frame_path(output_dir: Path, run: str, frame_index: int) -> Path:
    raw = _raw_review_photograph(output_dir, run, frame_index)
    if raw is not None:
        return raw
    manifest = _native_frame_folder(output_dir, run) / "manifest.json"
    if manifest.is_file() and "ffac_selected_policy_a_sources" in _read_json(manifest):
        from darsia.fluidflower.phase_selected_policy_review import resolve_photograph
        return resolve_photograph(resolve_review_path(output_dir),run,frame_index,verify=False)
    if manifest.is_file() and "ffac_rig_camera_sources" in json.loads(manifest.read_text(encoding="utf-8-sig")):
        from darsia.fluidflower.phase_saved_review_metadata import resolve_photograph
        return resolve_photograph(resolve_review_path(output_dir),run,frame_index,verify=False)
    if manifest.is_file() and "v5_native_source_alias" in json.loads(manifest.read_text(encoding="utf-8-sig")):
        from darsia.fluidflower.phase_source_alias import resolve_native_photo_path
        return resolve_native_photo_path(resolve_review_path(output_dir), run, frame_index)
    path = _native_frame_folder(output_dir, run) / f"frame_{frame_index:03d}.jpg"
    return path.with_suffix(".png") if path.with_suffix(".png").is_file() else path


def _native_photograph_availability(
    output_dir: Path, run: str, manifest: Mapping[str, Any], frame_count: int,
) -> tuple[Path, bool]:
    """Validate a rig timeline once when checking its photographs for review.

    Resolving each photograph separately reconstructs and compares the same large
    source document for every frame. Keep the exact source and per-frame certificate
    checks, but perform that reconstruction only once for this availability request.
    Selected image reads still verify their own asset content in the image worker.
    """
    from darsia.fluidflower.raw_photographic_review import (
        RAW_REVIEW_FORMAT, raw_asset_descriptor, validate_raw_review_manifest)
    if manifest.get("format") == RAW_REVIEW_FORMAT:
        # Existence only: every served photograph verifies its own SHA-256.
        manifest = validate_raw_review_manifest(manifest, run)
        if len(manifest["frames"]) != frame_count:
            raise ValueError("RAW review timeline differs from the candidate frame count")
        folder = _native_frame_folder(output_dir, run)
        paths = [Path(raw_asset_descriptor(folder, row["assets"]["natural"])["path"])
                 if "natural" in (row.get("assets") or {}) else None
                 for row in (manifest["reference"], *manifest["frames"])]
        return (paths[0] or folder / "reference" / "natural.png",
                all(path is not None and path.is_file() for path in paths))
    if "ffac_selected_policy_a_sources" in manifest:
        from darsia.fluidflower.phase_selected_policy_review import load_native_source
        state=load_native_source(output_dir,run,manifest,verify_geometry=False)
        if len(state["native"]["frame_sources"]) != frame_count:
            raise ValueError("Selected photographic timeline differs from candidate count")
        photographs=[state["native"]["reference"],*state["native"]["frame_sources"]]
        return Path(photographs[0]["photographs"]["natural"]["path"]), all(
            Path(row["photographs"]["natural"]["path"]).is_file() for row in photographs)
    if "ffac_rig_camera_sources" not in manifest:
        return _native_reference_path(output_dir, run), all(
            _native_frame_path(output_dir, run, index).is_file()
            for index in range(frame_count)
        )

    from darsia.fluidflower.phase_saved_review_metadata import bound_path
    from darsia.fluidflower.phase_rig_raw_display import _native_frame_assets_cached

    # This is the same presentation-only session cache used for RAW/Change.
    # Native/input/parent/camera stamps and every selected frame certificate are
    # checked on access; the independent scientific decoder is not changed.
    state, _record, _binding = _native_frame_assets_cached(
        output_dir, run, reference=True,
    )
    if state["native"] != manifest:
        raise ValueError("Review metadata differs from its source timeline")
    rows = state["inputs"]["frames"]
    if len(rows) != frame_count:
        raise ValueError("Review photograph count differs from its source timeline")
    base = state["input_path"].parent
    reference = bound_path(
        state["inputs"]["reference"]["photographs"]["natural"],
        base=base, verify=False,
    )
    # bound_path checks that each named photograph exists. Missing source assets
    # remain errors rather than being silently treated as an incomplete old review.
    for index, row in enumerate(rows):
        _native_frame_assets_cached(output_dir, run, index)
        bound_path(row["photographs"]["natural"], base=base, verify=False)
    return reference, True


def _native_frame_source_records(
    context: Any,
    loaded: Sequence[tuple[Any, float, float]],
) -> list[dict[str, Any]]:
    """Build traceable frame metadata from the exact images used by calibration."""

    selection_records = [
        dict(record)
        for record in (getattr(context, "frame_selection_records", None) or [])
        if isinstance(record, Mapping) and record.get("selected")
    ]
    if len(selection_records) != len(loaded):
        selection_records = [
            dict(record)
            for record in (getattr(context, "frame_selection_records", None) or [])
            if isinstance(record, Mapping)
        ][: len(loaded)]

    rows: list[dict[str, Any]] = []
    for index, (image, _injected, elapsed_h) in enumerate(loaded):
        selection = selection_records[index] if index < len(selection_records) else {}
        selected_value = selection.get("selected") or selection.get("requested")
        image_name = getattr(image, "name", None)
        selected_text = str(selected_value) if selected_value else ""
        source_name = (
            Path(selected_text).name
            if selected_text
            else Path(str(image_name)).name if image_name else f"frame_{index + 1}"
        )
        requested_text = str(selection.get("requested") or "")
        row: dict[str, Any] = {
            "frame_index": int(index),
            "elapsed_h": float(elapsed_h),
            "source_name": source_name,
            "source_path": selected_text,
            "requested_name": Path(requested_text).name if requested_text else source_name,
            "requested_path": requested_text,
            "selection_status": str(selection.get("status") or "unknown"),
            "color_correction_active": bool(
                selection.get("color_correction_active", False)
            ),
            "color_correction_flagged": bool(
                selection.get("color_correction_flagged", False)
            ),
        }
        residual = selection.get("color_correction_residual")
        if residual is not None:
            row["color_correction_residual"] = float(residual)
        correction_rows = [
            item
            for item in (selection.get("color_corrections") or [])
            if isinstance(item, Mapping)
        ]
        residuals_before = [
            float(item["residual_before_refinement"])
            for item in correction_rows
            if item.get("residual_before_refinement") is not None
        ]
        if residuals_before:
            row["color_correction_residual_before_refinement"] = max(
                residuals_before
            )
        residuals_before_correction = [
            float(item["residual_before_correction"])
            for item in correction_rows
            if item.get("residual_before_correction") is not None
        ]
        if residuals_before_correction:
            row["color_correction_residual_before_correction"] = max(
                residuals_before_correction
            )
        max_swatch_residuals = [
            float(item["max_swatch_residual"])
            for item in correction_rows
            if item.get("max_swatch_residual") is not None
        ]
        if max_swatch_residuals:
            row["color_correction_max_swatch_residual"] = max(
                max_swatch_residuals
            )
        refinement_steps = [
            int(item.get("refinement_steps_applied") or 0)
            for item in correction_rows
        ]
        if refinement_steps:
            row["color_correction_refinement_steps"] = max(refinement_steps)
        strategies = sorted(
            {
                str(item["correction_strategy"])
                for item in correction_rows
                if item.get("correction_strategy")
            }
        )
        if strategies:
            row["color_correction_strategy"] = "+".join(strategies)
        if correction_rows:
            row["color_corrections"] = correction_rows
        thresholds = [
            float(item["threshold"])
            for item in correction_rows
            if isinstance(item, Mapping) and item.get("threshold") is not None
        ]
        if thresholds:
            row["color_correction_threshold"] = min(thresholds)
        acquired = getattr(image, "date", None)
        if acquired is not None:
            row["acquired_at"] = (
                acquired.isoformat() if hasattr(acquired, "isoformat") else str(acquired)
            )
        rows.append(row)
    return rows


def _native_overlay_folder(
    output_dir: Path,
    candidate: Candidate,
    label: str,
) -> Path:
    fingerprint = _candidate_fingerprint(candidate)
    root = _candidate_review_root(output_dir, candidate)
    run_root = root / "native_overlays" / candidate.run
    expected = run_root / f"{label}_{fingerprint}"
    if expected.exists():
        return expected
    # The blind A/B label may change when candidates are combined into a new
    # review. Reuse the unique candidate-specific overlay folder regardless of
    # the presentation label under which it was originally generated.
    matches = sorted(run_root.glob(f"*_{fingerprint}"))
    if len(matches) == 1:
        return matches[0]
    legacy = _legacy_candidate_fingerprint(candidate)
    legacy_expected = run_root / f"{label}_{legacy}"
    if legacy_expected.exists():
        return legacy_expected
    legacy_matches = sorted(run_root.glob(f"*_{legacy}"))
    return legacy_matches[0] if len(legacy_matches) == 1 else expected


def _asset_frame_with_legacy_fallback(
    folder: Path,
    candidate: Candidate,
    label: str,
    frame_index: int,
) -> Path:
    """Resolve one frame even when a candidate-specific cache stopped midway."""

    filename = f"frame_{frame_index:03d}.png"
    primary = folder / filename
    if primary.is_file():
        return primary

    run_root = folder.parent
    legacy = _legacy_candidate_fingerprint(candidate)
    generated_label = folder.name.partition("_")[0]
    labels = tuple(dict.fromkeys((generated_label, label)))
    for candidate_label in labels:
        fallback = run_root / f"{candidate_label}_{legacy}" / filename
        if fallback.is_file():
            return fallback

    matches = [
        path / filename
        for path in sorted(run_root.glob(f"*_{legacy}"))
        if (path / filename).is_file()
    ]
    return matches[0] if len(matches) == 1 else primary


def _native_overlay_path(
    output_dir: Path,
    candidate: Candidate,
    label: str,
    frame_index: int,
) -> Path:
    if _candidate_prediction_withheld(output_dir, candidate):
        return _withheld_prediction_asset(output_dir, candidate)
    folder = _native_overlay_folder(
        output_dir,
        candidate,
        label,
    )
    return _asset_frame_with_legacy_fallback(
        folder,
        candidate,
        label,
        frame_index,
    )


def _native_overlay_variant_path(path: Path, phase: str = "all") -> Path:
    """Resolve a phase-specific contour layer, falling back to the combined one."""

    normalized = str(phase or "all").casefold()
    if normalized not in {"gas", "aqueous"}:
        return path
    candidate = path.parent / normalized / path.name
    return candidate if candidate.is_file() else path


def _native_phase_map_folder(
    output_dir: Path,
    candidate: Candidate,
    label: str,
) -> Path:
    fingerprint = _candidate_fingerprint(candidate)
    root = _candidate_phase_map_root(output_dir, candidate)
    run_root = root / "native_phase_maps" / candidate.run
    expected = run_root / f"{label}_{fingerprint}"
    if expected.exists():
        return expected
    matches = sorted(run_root.glob(f"*_{fingerprint}"))
    if len(matches) == 1:
        return matches[0]
    legacy = _legacy_candidate_fingerprint(candidate)
    legacy_expected = run_root / f"{label}_{legacy}"
    if legacy_expected.exists():
        return legacy_expected
    legacy_matches = sorted(run_root.glob(f"*_{legacy}"))
    return legacy_matches[0] if len(legacy_matches) == 1 else expected


def _native_phase_map_path(
    output_dir: Path,
    candidate: Candidate,
    label: str,
    frame_index: int,
) -> Path:
    if _candidate_prediction_withheld(output_dir, candidate):
        return _withheld_prediction_asset(output_dir, candidate)
    if (getattr(candidate,"quality_flags",None) or {}).get("selected_camera_review"):
        from darsia.fluidflower.phase_selected_policy_review import resolve_phase_map
        return resolve_phase_map(_candidate_review_root(output_dir,candidate),candidate.run,frame_index)
    folder = _native_phase_map_folder(
        output_dir,
        candidate,
        label,
    )
    return _asset_frame_with_legacy_fallback(
        folder,
        candidate,
        label,
        frame_index,
    )


def _native_phase_map_variant_path(
    path: Path,
    phase: str = "all",
    *,
    fallback: bool = True,
) -> Path:
    """Resolve a class-specific filled phase map, falling back to the combined map."""

    normalized = str(phase or "all").casefold()
    if normalized not in {"gas", "aqueous", "mixed", "removed_top", "removed_floor",
            "below_sand_floor", "removed_floor_connectivity", "floor_contact_fill_v2", "removed_gas_envelope", "removed_contact_fill_b"}:
        return path
    candidate = path.parent / normalized / path.name
    return candidate if candidate.is_file() or not fallback else path


def _preferred_phase_overlay_path(
    output_dir: Path,
    candidate: Candidate,
    label: str,
    frame_index: int,
    phase: str = "all",
) -> tuple[Path, str]:
    """Prefer a filled phase map while retaining legacy contour compatibility."""

    if _candidate_prediction_withheld(output_dir, candidate):
        return _withheld_prediction_asset(output_dir, candidate), "bottom_blind_withheld"

    if (getattr(candidate,"quality_flags",None) or {}).get("selected_camera_review"):
        from darsia.fluidflower.phase_selected_policy_review import resolve_phase_map
        return resolve_phase_map(_candidate_review_root(output_dir,candidate),candidate.run,frame_index,phase=phase), "selected_saved_phase_fill"
    if phase in {"below_sand_floor", "removed_floor_connectivity", "floor_contact_fill_v2", "removed_gas_envelope", "removed_contact_fill_b"}:
        final3 = bool((candidate.quality_flags or {}).get("p1_final3_revision"))
        if not (final3 or (candidate.quality_flags or {}).get("p1_final2_revision")):
            raise ValueError("This candidate has no P1-final2 A–D provenance overlay")
        if phase == "removed_contact_fill_b" and not final3:
            raise ValueError("This candidate has no P1-final3 reason-b removal provenance overlay")
        path = _native_phase_map_path(output_dir, candidate, label, frame_index)
        return _native_phase_map_variant_path(path, phase, fallback=False), ("p1_final3_provenance" if final3 else "p1_final2_provenance")
    if phase in {"removed_top", "removed_floor"}:
        if not (candidate.quality_flags or {}).get("p1_final_revision"):
            raise ValueError("This candidate has no P1-final removal provenance overlay")
        path = _native_phase_map_path(output_dir, candidate, label, frame_index)
        return _native_phase_map_variant_path(path, phase, fallback=False), "p1_final_removed"

    from darsia.fluidflower.phase_review_basis import uses_optical, optical_path, is_regional
    if uses_optical(candidate) and phase in {"all", "gas", "aqueous", "mixed"}:
        path = optical_path(candidate, frame_index)
        return (two_phase_overlay_path(path, phase) if path.is_file() else path), ("trained_regional" if is_regional(candidate) else "raw_optical")
    if phase == "raw":
        if not candidate.raw_phase_map_output_dir or not candidate.optical_model_sha256:
            raise ValueError("This candidate has no archived raw optical prediction")
        path = candidate.raw_phase_map_output_dir / "native_phase_maps" / candidate.run / (
            "A_" + candidate.optical_model_sha256[:16]) / f"frame_{frame_index:03d}.png"
        return (two_phase_overlay_path(path, "all") if path.is_file() else path), "raw_optical"
    diagnostics = {"context_added": "added_aqueous", "context_removed": "suppressed_presence",
                   "context_uncertain": "unresolved"}
    if phase in diagnostics:
        if not candidate.presence_diagnostics_dir:
            raise ValueError("This candidate has no archived presence-context diagnostics")
        return (candidate.presence_diagnostics_dir / candidate.run / diagnostics[phase]
                / f"frame_{frame_index:03d}.png"), "context_diagnostic"

    phase_map = _native_phase_map_path(
        output_dir,
        candidate,
        label,
        frame_index,
    )
    if phase_map.is_file():
        return two_phase_overlay_path(phase_map, phase), "phase_fill"
    contour = _native_overlay_path(
        output_dir,
        candidate,
        label,
        frame_index,
    )
    return _native_overlay_variant_path(contour, phase), "legacy_contour"


def _phase_quiz_path(
    output_dir: Path,
    candidate: Candidate,
    label: str,
) -> Path:
    fingerprint = _candidate_fingerprint(candidate)
    root = _candidate_phase_map_root(output_dir, candidate)
    return (
        root
        / "phase_quiz"
        / candidate.run
        / f"{label}_{fingerprint}_q{PHASE_QUIZ_PROPOSAL_SCHEMA}"
        / "proposals.json"
    )


def _legacy_phase_quiz_path(
    output_dir: Path,
    candidate: Candidate,
    label: str,
) -> Path:
    fingerprint = _legacy_candidate_fingerprint(candidate)
    root = _candidate_phase_map_root(output_dir, candidate)
    return (
        root
        / "phase_quiz"
        / candidate.run
        / f"{label}_{fingerprint}"
        / "proposals.json"
    )


def _load_phase_quiz_payload(
    output_dir: Path,
    candidate: Candidate,
    label: str,
) -> dict[str, Any]:
    if _candidate_prediction_withheld(output_dir, candidate):
        return {"proposals": [], "initial_proposal_count": 0}
    if getattr(candidate, "phase_quiz_proposals_path", None):
        from darsia.fluidflower.tools.phase_targeted_quiz import load_proposals
        return load_proposals(candidate, label)
    from darsia.fluidflower.phase_review_proposal_generation import load_generated_pool
    generated = load_generated_pool(output_dir, candidate)
    if generated is not None:
        proposals = [dict(row, candidate=label) for row in generated["proposals"]]
        return {"proposals": proposals,
                "initial_proposal_count": max(0, min(int(generated["initial_proposal_count"]), len(proposals)))}
    path = _phase_quiz_path(output_dir, candidate, label)
    loaded_payload: dict[str, Any] | None = None
    legacy = False
    if not path.is_file():
        # A/B is a blind presentation label, not candidate identity. A newly
        # combined review can legitimately reverse the labels used when the
        # proposal pool was generated, so fall back to the unique pool with
        # the same parameter fingerprint.
        fingerprint = _candidate_fingerprint(candidate)
        root = _candidate_phase_map_root(output_dir, candidate)
        matches = sorted(
            (root / "phase_quiz" / candidate.run).glob(
                f"*_{fingerprint}_q{PHASE_QUIZ_PROPOSAL_SCHEMA}/proposals.json"
            )
        )
        if len(matches) == 1:
            path = matches[0]
    if not path.is_file():
        legacy_fingerprint = _legacy_candidate_fingerprint(candidate)
        root = _candidate_phase_map_root(output_dir, candidate)
        legacy_matches = sorted(
            (root / "phase_quiz" / candidate.run).glob(
                f"*_{legacy_fingerprint}_q{PHASE_QUIZ_PROPOSAL_SCHEMA}/proposals.json"
            )
        )
        if len(legacy_matches) == 1:
            path = legacy_matches[0]
    if not path.is_file():
        # Early schema-2 caches used only the parameter digest in the folder
        # name. Two scientifically distinct candidates can share those
        # parameters, making the filename ambiguous after blind A/B relabeling.
        # Resolve that case from the stable candidate id stored in the payload.
        root = _candidate_phase_map_root(output_dir, candidate)
        identity_matches: list[tuple[Path, dict[str, Any]]] = []
        for candidate_path in sorted(
            (root / "phase_quiz" / candidate.run).glob(
                f"*_*_q{PHASE_QUIZ_PROPOSAL_SCHEMA}/proposals.json"
            )
        ):
            try:
                candidate_payload = _read_json(candidate_path)
            except (OSError, TypeError, ValueError):
                continue
            candidate_ids = {
                str(candidate_payload.get("candidate_id") or "")
            }
            candidate_ids.update(
                str(item.get("candidate_id") or "")
                for item in (candidate_payload.get("proposals") or [])[:1]
                if isinstance(item, Mapping)
            )
            if candidate.candidate_id in candidate_ids:
                identity_matches.append((candidate_path, candidate_payload))
        if len(identity_matches) == 1:
            path, loaded_payload = identity_matches[0]
    if not path.is_file():
        path = _legacy_phase_quiz_path(output_dir, candidate, label)
        legacy = True
    if not path.is_file():
        return {"proposals": [], "initial_proposal_count": 0}
    payload = loaded_payload or _read_json(path)
    proposals = payload.get("proposals") or []
    if not isinstance(proposals, list):
        raise ValueError(f"Invalid phase quiz proposal list: {path}")
    clean = [
        {**dict(item), "candidate": label}
        for item in proposals
        if isinstance(item, Mapping)
    ]
    if legacy:
        # Legacy proposal pools predate the explicit porous-medium contract.
        # Loading an unverified pool can place quiz regions in free water,
        # ColorChecker patches, the frame, or other non-porous image areas.
        clean = [
            item
            for item in clean
            if item.get("analysis_domain") == "porous_medium_only"
        ]
    initial_count = int(payload.get("initial_proposal_count", len(clean)))
    return {
        "proposals": clean,
        "initial_proposal_count": max(0, min(initial_count, len(clean))),
    }


def _load_phase_quiz_proposals(
    output_dir: Path,
    candidate: Candidate,
    label: str,
) -> list[dict[str, Any]]:
    return _load_phase_quiz_payload(output_dir, candidate, label)["proposals"]


def _phase_quiz_focus_box(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    padding = max(24, int(round(0.18 * max(width, height))))
    target_width = max(width + 2 * padding, min(480, image_width))
    target_height = max(height + 2 * padding, min(360, image_height))
    target_width = min(target_width, image_width)
    target_height = min(target_height, image_height)
    center_x = x + 0.5 * width
    center_y = y + 0.5 * height
    left = int(round(center_x - 0.5 * target_width))
    top = int(round(center_y - 0.5 * target_height))
    left = max(0, min(image_width - target_width, left))
    top = max(0, min(image_height - target_height, top))
    return left, top, target_width, target_height


def _phase_quiz_proposals_from_fields(
    *,
    run: str,
    label: str,
    candidate_id: str,
    params_sha256: str,
    frame_index: int,
    elapsed_h: float,
    concentration_aq: Any,
    saturation_g: Any,
    mass_aq: Any,
    mass_g: Any,
    canvas_wh: Sequence[int],
    content_box_xywh: Sequence[int],
    analysis_domain_mask: Any | None = None,
) -> list[dict[str, Any]]:
    """Build blinded, traceable 2D phase regions from numerical model fields."""

    import cv2
    import numpy as np

    aqueous = np.asarray(concentration_aq, dtype=np.float32)
    gas = np.asarray(saturation_g, dtype=np.float32)
    aqueous_mass = np.asarray(mass_aq, dtype=np.float32)
    gas_mass = np.asarray(mass_g, dtype=np.float32)
    if not (
        aqueous.shape
        == gas.shape
        == aqueous_mass.shape
        == gas_mass.shape
    ) or aqueous.ndim != 2:
        raise ValueError("Phase quiz fields must be equally shaped 2D arrays")
    if analysis_domain_mask is None:
        analysis_domain = np.ones(aqueous.shape, dtype=bool)
    else:
        analysis_domain = np.asarray(analysis_domain_mask, dtype=bool)
        if analysis_domain.shape != aqueous.shape:
            raise ValueError(
                "Phase quiz analysis-domain mask must match the numerical fields"
            )
    if not np.any(analysis_domain):
        raise ValueError("Phase quiz analysis domain contains no pixels")
    if len(canvas_wh) != 2 or len(content_box_xywh) != 4:
        raise ValueError("Phase quiz requires canvas and content-box geometry")

    image_height, image_width = aqueous.shape
    canvas_width, canvas_height = map(int, canvas_wh)
    content_x, content_y, content_width, content_height = map(
        int, content_box_xywh
    )
    if (content_width, content_height) != (image_width, image_height):
        raise ValueError(
            "Phase quiz content box must preserve the numerical field resolution"
        )

    domain_margin = max(
        PHASE_QUIZ_DOMAIN_MARGIN_MIN_PX,
        min(
            PHASE_QUIZ_DOMAIN_MARGIN_MAX_PX,
            int(round(PHASE_QUIZ_DOMAIN_MARGIN_FRACTION * min(aqueous.shape))),
        ),
    )
    domain_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * domain_margin + 1, 2 * domain_margin + 1),
    )
    safe_analysis_domain = (
        cv2.erode(analysis_domain.astype(np.uint8), domain_kernel) > 0
    )
    if not np.any(safe_analysis_domain):
        raise ValueError("Phase quiz porous-domain safety margin removes all pixels")

    aqueous = np.nan_to_num(aqueous, nan=0.0, posinf=0.0, neginf=0.0)
    gas = np.nan_to_num(gas, nan=0.0, posinf=0.0, neginf=0.0)
    aqueous_mass = np.maximum(
        np.nan_to_num(aqueous_mass, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    gas_mass = np.maximum(
        np.nan_to_num(gas_mass, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    total_mass = aqueous_mass + gas_mass
    present = safe_analysis_domain & ((aqueous >= 0.01) | (gas >= 0.005))
    gas_fraction = np.divide(
        gas_mass,
        total_mass,
        out=np.zeros_like(total_mass),
        where=total_mass > 1e-12,
    )
    # Older spatial evaluators expose concentration and saturation rather than
    # calibrated class probabilities.  Convert those fields into a bounded
    # three-class entropy proxy so active-learning rounds prefer ambiguous
    # phase boundaries.  Newer model bundles may replace this metadata with
    # their directly calibrated predictive entropy.
    co2_probability_proxy = np.clip(
        np.maximum(aqueous / 0.02, gas / 0.01), 0.0, 1.0
    )
    phase_probability_proxy = np.stack(
        (
            1.0 - co2_probability_proxy,
            co2_probability_proxy * (1.0 - gas_fraction),
            co2_probability_proxy * gas_fraction,
        ),
        axis=-1,
    )
    predictive_entropy_proxy = -np.sum(
        phase_probability_proxy
        * np.log(np.maximum(phase_probability_proxy, np.finfo(np.float32).tiny)),
        axis=-1,
    ) / np.log(3.0)
    model_masks = {
        "aqueous": safe_analysis_domain & present & (gas_fraction <= 0.35),
        "gas": safe_analysis_domain & present & (gas_fraction >= 0.65),
        "mixed": (
            safe_analysis_domain
            & present
            & (gas_fraction > 0.35)
            & (gas_fraction < 0.65)
        ),
    }
    minimum_area = max(64.0, 0.000004 * image_width * image_height)

    def normalized_point(point: Sequence[float]) -> list[float]:
        return [
            (content_x + float(point[0])) / canvas_width,
            (content_y + float(point[1])) / canvas_height,
        ]

    proposals: list[dict[str, Any]] = []
    for model_phase, phase_mask in model_masks.items():
        contours, _ = cv2.findContours(
            np.asarray(phase_mask, dtype=np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        ranked: list[tuple[float, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < minimum_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            approximation = cv2.approxPolyDP(
                contour,
                max(1.0, 0.0025 * perimeter),
                True,
            )
            while len(approximation) > 160:
                perimeter_factor = 0.004 * (len(approximation) / 160.0)
                approximation = cv2.approxPolyDP(
                    contour,
                    max(1.0, perimeter_factor * perimeter),
                    True,
                )
            if len(approximation) < 3:
                continue
            ranked.append((area, approximation))
        ranked.sort(key=lambda item: item[0], reverse=True)
        for area, approximation in ranked[:2]:
            x, y, width, height = cv2.boundingRect(approximation)
            region_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            cv2.drawContours(region_mask, [approximation], -1, 1, thickness=-1)
            source_component_area = area
            source_component_bbox = [int(x), int(y), int(width), int(height)]
            region_type = "connected_component"
            requires_local_probe = (
                area > 0.02 * image_width * image_height
                or width > 0.22 * image_width
                or height > 0.25 * image_height
            )
            if requires_local_probe:
                distance = cv2.distanceTransform(
                    region_mask,
                    cv2.DIST_L2,
                    5,
                )
                _minimum, maximum, _min_location, max_location = cv2.minMaxLoc(
                    distance
                )
                center_x, center_y = max_location
                if maximum >= 24.0:
                    radius = int(max(16, min(160, round(0.58 * maximum))))
                    angles = np.linspace(0.0, 2.0 * np.pi, 65)[:-1]
                    circle = np.column_stack(
                        (
                            center_x + radius * np.cos(angles),
                            center_y + radius * np.sin(angles),
                        )
                    )
                    approximation = np.rint(circle).astype(np.int32).reshape(-1, 1, 2)
                    region_type = "local_probe_inside_component"
                else:
                    half_width = min(360, max(96, width // 4))
                    half_height = min(270, max(72, height // 4))
                    left = max(0, center_x - half_width)
                    top = max(0, center_y - half_height)
                    right = min(image_width, center_x + half_width)
                    bottom = min(image_height, center_y + half_height)
                    local_mask = np.zeros_like(region_mask)
                    local_mask[top:bottom, left:right] = region_mask[
                        top:bottom,
                        left:right,
                    ]
                    local_contours, _ = cv2.findContours(
                        local_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    if local_contours:
                        approximation = max(
                            local_contours,
                            key=cv2.contourArea,
                        )
                        region_type = "local_window_inside_component"
                area = float(cv2.contourArea(approximation))
                x, y, width, height = cv2.boundingRect(approximation)
                region_mask.fill(0)
                cv2.drawContours(
                    region_mask,
                    [approximation],
                    -1,
                    1,
                    thickness=-1,
                )
            polygon = approximation.reshape(-1, 2)
            region = region_mask.astype(bool)
            if not np.any(region):
                continue
            phase_purity = float(np.mean(phase_mask[region]))
            median_gas_fraction = float(np.median(gas_fraction[region]))
            predictive_entropy_mean = float(
                np.mean(predictive_entropy_proxy[region])
            )
            focus = _phase_quiz_focus_box(
                x,
                y,
                width,
                height,
                image_width=image_width,
                image_height=image_height,
            )
            points = [normalized_point(point) for point in polygon]
            focus_x, focus_y, focus_width, focus_height = focus
            proposal_seed = (
                f"{run}|{candidate_id}|{frame_index}|{model_phase}|"
                f"{x}|{y}|{width}|{height}"
            )
            proposals.append(
                {
                    "proposal_id": "phase-" + hashlib.sha256(
                        proposal_seed.encode("utf-8")
                    ).hexdigest()[:20],
                    "run": run,
                    "candidate": label,
                    "candidate_id": candidate_id,
                    "params_sha256": params_sha256,
                    "frame_index": int(frame_index),
                    "elapsed_h": float(elapsed_h),
                    "suggested_phase": model_phase,
                    "suggestion_basis": (
                        "dominant local modelled CO2 mass within a numerical "
                        "c/csat-Sg connected component"
                    ),
                    "interpretation_scope": "visible_2d_slice_only",
                    "analysis_domain": "porous_medium_only",
                    "analysis_domain_margin_px": int(domain_margin),
                    "coordinate_space": "normalized_native_review_canvas_16x9",
                    "shape": "freehand",
                    "points": points,
                    "x": min(point[0] for point in points),
                    "y": min(point[1] for point in points),
                    "width": max(point[0] for point in points)
                    - min(point[0] for point in points),
                    "height": max(point[1] for point in points)
                    - min(point[1] for point in points),
                    "analysis_polygon_px": polygon.astype(int).tolist(),
                    "analysis_bbox_xywh": [int(x), int(y), int(width), int(height)],
                    "focus_box": {
                        "x": (content_x + focus_x) / canvas_width,
                        "y": (content_y + focus_y) / canvas_height,
                        "width": focus_width / canvas_width,
                        "height": focus_height / canvas_height,
                    },
                    "model_stats": {
                        "area_px": area,
                        "source_component_area_px": source_component_area,
                        "source_component_bbox_xywh": source_component_bbox,
                        "region_type": region_type,
                        "phase_purity": phase_purity,
                        "median_c_over_csat": float(np.median(aqueous[region])),
                        "median_sg": float(np.median(gas[region])),
                        "median_gas_mass_fraction": median_gas_fraction,
                        "predictive_entropy_mean": predictive_entropy_mean,
                        "uncertainty_kind": (
                            "phase_field_entropy_proxy_unless_replaced_by_model_entropy"
                        ),
                    },
                    "priority": float(
                        math.sqrt(area)
                        * max(0.05, phase_purity)
                        * (1.0 + predictive_entropy_mean)
                    ),
                }
            )

    water = safe_analysis_domain & ~present
    if np.any(water):
        distance = cv2.distanceTransform(
            np.asarray(water, dtype=np.uint8),
            cv2.DIST_L2,
            5,
        )
        _minimum, maximum, _min_location, max_location = cv2.minMaxLoc(distance)
        if maximum >= 12.0:
            center_x, center_y = max_location
            half = int(max(12, min(72, round(0.35 * maximum))))
            x = max(0, center_x - half)
            y = max(0, center_y - half)
            right = min(image_width - 1, center_x + half)
            bottom = min(image_height - 1, center_y + half)
            polygon = [[x, y], [right, y], [right, bottom], [x, bottom]]
            points = [normalized_point(point) for point in polygon]
            focus = _phase_quiz_focus_box(
                x,
                y,
                right - x,
                bottom - y,
                image_width=image_width,
                image_height=image_height,
            )
            focus_x, focus_y, focus_width, focus_height = focus
            region = np.zeros_like(water)
            region[y : bottom + 1, x : right + 1] = True
            proposal_seed = f"{run}|{candidate_id}|{frame_index}|water|{x}|{y}"
            proposals.append(
                {
                    "proposal_id": "phase-" + hashlib.sha256(
                        proposal_seed.encode("utf-8")
                    ).hexdigest()[:20],
                    "run": run,
                    "candidate": label,
                    "candidate_id": candidate_id,
                    "params_sha256": params_sha256,
                    "frame_index": int(frame_index),
                    "elapsed_h": float(elapsed_h),
                    "suggested_phase": "water",
                    "suggestion_basis": "low numerical c/csat and Sg negative control",
                    "interpretation_scope": "visible_2d_slice_only",
                    "analysis_domain": "porous_medium_only",
                    "analysis_domain_margin_px": int(domain_margin),
                    "coordinate_space": "normalized_native_review_canvas_16x9",
                    "shape": "freehand",
                    "points": points,
                    "x": points[0][0],
                    "y": points[0][1],
                    "width": points[1][0] - points[0][0],
                    "height": points[3][1] - points[0][1],
                    "analysis_polygon_px": polygon,
                    "analysis_bbox_xywh": [x, y, right - x, bottom - y],
                    "focus_box": {
                        "x": (content_x + focus_x) / canvas_width,
                        "y": (content_y + focus_y) / canvas_height,
                        "width": focus_width / canvas_width,
                        "height": focus_height / canvas_height,
                    },
                    "model_stats": {
                        "area_px": float((right - x) * (bottom - y)),
                        "phase_purity": float(np.mean(water[region])),
                        "median_c_over_csat": float(np.median(aqueous[region])),
                        "median_sg": float(np.median(gas[region])),
                        "median_gas_mass_fraction": float(
                            np.median(gas_fraction[region])
                        ),
                        "predictive_entropy_mean": float(
                            np.mean(predictive_entropy_proxy[region])
                        ),
                        "uncertainty_kind": (
                            "phase_field_entropy_proxy_unless_replaced_by_model_entropy"
                        ),
                    },
                    "priority": float(maximum),
                }
            )
    return proposals


def _select_phase_quiz_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    max_total: int | None = 12,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {
        phase: [] for phase in ("gas", "aqueous", "mixed", "water")
    }
    for proposal in proposals:
        phase = str(proposal.get("suggested_phase") or "")
        if phase in groups:
            groups[phase].append(dict(proposal))
    for group in groups.values():
        group.sort(
            key=lambda item: (
                -_phase_quiz_uncertainty(item),
                float(item.get("elapsed_h") or 0.0),
                -float(item.get("priority") or 0.0),
                str(item.get("proposal_id") or ""),
            )
        )
    limit = sum(len(group) for group in groups.values()) if max_total is None else max(0, int(max_total))
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        added = False
        for phase in ("gas", "aqueous", "mixed", "water"):
            if groups[phase] and len(selected) < limit:
                selected.append(groups[phase].pop(0))
                added = True
        if not added:
            break
    return selected


def _native_reference_path(output_dir: Path, run: str) -> Path:
    raw = _raw_review_photograph(output_dir, run, reference=True)
    if raw is not None:
        return raw
    output_dir = resolve_review_path(output_dir)
    manifest = output_dir / "native_frames" / run.casefold() / "manifest.json"
    if manifest.is_file() and "ffac_selected_policy_a_sources" in _read_json(manifest):
        from darsia.fluidflower.phase_selected_policy_review import resolve_photograph
        return resolve_photograph(output_dir,run,reference=True,verify=False)
    if manifest.is_file() and "ffac_rig_camera_sources" in json.loads(manifest.read_text(encoding="utf-8-sig")):
        from darsia.fluidflower.phase_saved_review_metadata import resolve_photograph
        return resolve_photograph(output_dir,run,reference=True,verify=False)
    if manifest.is_file() and "v5_native_source_alias" in json.loads(manifest.read_text(encoding="utf-8-sig")):
        from darsia.fluidflower.phase_source_alias import resolve_native_photo_path
        return resolve_native_photo_path(output_dir, run, reference=True)
    path = output_dir / "native_references" / run.casefold() / "pre_injection.jpg"
    return path.with_suffix(".png") if path.with_suffix(".png").is_file() else path


def _native_manifest_path(output_dir: Path, run: str) -> Path:
    return _native_frame_folder(output_dir, run) / "manifest.json"


def _sidecar_reference_registration(review_root: Path, run: str,
                                    native_manifest: Path | None = None) -> list[dict]:
    """Read the display-only alignment kept beside the reference photograph.

    A target prepared by prepare_ffum_phase_target never measured the baseline-to-frame
    translation, so its hashed native manifest carries none and the aligned pre-injection
    reference stays switched off. The measurement only places an existing photograph on
    screen, so it is kept next to that photograph rather than inside the manifest, and an
    evaluation that already exists stays verifiable against the inputs it was produced
    from. A record measured against another manifest is ignored rather than painted over
    the wrong frames.
    """
    try:
        from darsia.fluidflower.tools.phase_reference_registration import read

        return read(review_root, run,
                    native_manifest or _native_manifest_path(review_root, run))
    except (ImportError, OSError, ValueError, KeyError, TypeError):
        return []


def _native_review_dimensions(image_shape: Sequence[int]) -> tuple[int, int]:
    """Smallest 16:9 review canvas that fits every corrected source pixel."""

    source_height, source_width = int(image_shape[0]), int(image_shape[1])
    if source_height <= 0 or source_width <= 0:
        raise ValueError("Image dimensions must be positive")
    estimate = max(
        source_width / (REVIEW_FRAME_WIDTH - 2 * REVIEW_FRAME_MARGIN),
        source_height
        / (REVIEW_FRAME_HEIGHT - REVIEW_HEADER_HEIGHT - REVIEW_FRAME_MARGIN),
        1.0,
    )
    # A height divisible by nine gives an exact 16:9 canvas without browser
    # stretching. Search a few steps around the continuous estimate because
    # scaled header and margin values are rounded independently.
    first_height = max(900, int(math.floor(REVIEW_FRAME_HEIGHT * estimate)) - 18)
    first_height -= first_height % 9
    for height in range(first_height, first_height + 90, 9):
        width = 16 * (height // 9)
        layout_scale = height / REVIEW_FRAME_HEIGHT
        header = max(1, int(round(REVIEW_HEADER_HEIGHT * layout_scale)))
        margin = max(1, int(round(REVIEW_FRAME_MARGIN * layout_scale)))
        if (
            width - 2 * margin >= source_width
            and height - header - margin >= source_height
        ):
            return width, height
    raise RuntimeError("Could not determine a native-resolution review canvas")


def _review_content_box(image_shape: Sequence[int]) -> tuple[int, int, int, int]:
    source_height, source_width = (int(image_shape[0]), int(image_shape[1]))
    available_width = REVIEW_FRAME_WIDTH - 2 * REVIEW_FRAME_MARGIN
    available_height = (
        REVIEW_FRAME_HEIGHT - REVIEW_HEADER_HEIGHT - REVIEW_FRAME_MARGIN
    )
    scale = min(
        available_width / source_width,
        available_height / source_height,
    )
    target_width = max(2, int(round(source_width * scale)))
    target_height = max(2, int(round(source_height * scale)))
    x0 = (REVIEW_FRAME_WIDTH - target_width) // 2
    y0 = REVIEW_HEADER_HEIGHT + (available_height - target_height) // 2
    return x0, y0, target_width, target_height


def _aligned_reference_frame_folder(
    output_dir: Path,
    candidate: Candidate,
    label: str,
) -> Path:
    fingerprint = _candidate_fingerprint(candidate)
    root = _candidate_review_root(output_dir, candidate)
    return (
        root
        / "reference_frames"
        / candidate.run
        / f"{label}_{fingerprint}"
    )


def _aligned_reference_frame_path(
    output_dir: Path,
    candidate: Candidate,
    label: str,
    frame_index: int,
) -> Path:
    return _aligned_reference_frame_folder(
        output_dir, candidate, label
    ) / f"frame_{frame_index:03d}.jpg"


def _estimate_review_translation(
    reference: Any,
    target: Any,
    *,
    header_height: int = REVIEW_HEADER_HEIGHT,
    content_box: Sequence[int] | None = None,
) -> tuple[float, float, float, str]:
    """Estimate baseline-to-frame translation from static upper-rig structure."""

    import cv2
    import numpy as np

    reference_array = np.asarray(reference)
    target_array = np.asarray(target)
    if reference_array.shape != target_array.shape or reference_array.ndim != 3:
        raise ValueError("Reference and target review frames must have equal RGB shapes")
    if content_box is not None:
        x0, y0, content_width, content_height = (
            int(value) for value in content_box
        )
        if (
            x0 < 0
            or y0 < 0
            or content_width <= 0
            or content_height <= 0
            or x0 + content_width > reference_array.shape[1]
            or y0 + content_height > reference_array.shape[0]
        ):
            raise ValueError("Review content box lies outside the frame")
        reference_array = reference_array[
            y0 : y0 + content_height,
            x0 : x0 + content_width,
        ]
        target_array = target_array[
            y0 : y0 + content_height,
            x0 : x0 + content_width,
        ]
        header_height = 0
    height, width = reference_array.shape[:2]
    if height <= header_height + 40 or width < 100:
        raise ValueError("Review frame is too small for translation registration")

    content_height = height - header_height
    regions = (
        (
            "colorchecker",
            slice(header_height, header_height + max(80, int(0.38 * content_height))),
            slice(int(0.68 * width), int(0.99 * width)),
        ),
        (
            "upper-rig",
            slice(header_height, header_height + max(100, int(0.42 * content_height))),
            slice(0, width),
        ),
    )
    candidates: list[tuple[float, float, float, str]] = []
    for name, rows, columns in regions:
        reference_crop = cv2.cvtColor(
            reference_array[rows, columns], cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        target_crop = cv2.cvtColor(
            target_array[rows, columns], cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        reference_edges = cv2.Laplacian(
            cv2.GaussianBlur(reference_crop, (0, 0), 1.0), cv2.CV_32F
        )
        target_edges = cv2.Laplacian(
            cv2.GaussianBlur(target_crop, (0, 0), 1.0), cv2.CV_32F
        )
        window = cv2.createHanningWindow(
            (reference_edges.shape[1], reference_edges.shape[0]), cv2.CV_32F
        )
        (dx, dy), response = cv2.phaseCorrelate(
            reference_edges, target_edges, window
        )
        if not all(math.isfinite(value) for value in (dx, dy, response)):
            continue
        if abs(dx) > 0.05 * width or abs(dy) > 0.08 * content_height:
            continue
        candidates.append((float(dx), float(dy), float(response), name))
    if not candidates:
        return 0.0, 0.0, 0.0, "fallback-zero"
    return max(candidates, key=lambda item: item[2])


def _prepare_aligned_reference_frames(
    output_dir: Path,
    pair: ReviewPair,
    *,
    force: bool,
) -> None:
    """Create display-only baseline frames translated onto each candidate frame."""

    import cv2
    import numpy as np

    reference_path = _reference_frame_path(output_dir, pair.run)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError(f"Could not decode pre-injection reference: {reference_path}")
    reference_manifest_path = reference_path.parent / "manifest.json"
    reference_manifest = (
        _read_json(reference_manifest_path)
        if reference_manifest_path.is_file()
        else {}
    )
    source_name = str(reference_manifest.get("source_name") or "configured baseline")
    raw_content_box = reference_manifest.get("content_box_xywh")
    if isinstance(raw_content_box, list) and len(raw_content_box) == 4:
        content_box = tuple(int(value) for value in raw_content_box)
    else:
        content_box = (
            0,
            REVIEW_HEADER_HEIGHT,
            reference.shape[1],
            reference.shape[0] - REVIEW_HEADER_HEIGHT,
        )
    content_x, content_y, content_width, content_height = content_box

    for label, candidate in (("A", pair.candidate_a), ("B", pair.candidate_b)):
        folder = _aligned_reference_frame_folder(output_dir, candidate, label)
        folder.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for index in range(candidate.frame_count):
            clean_path = _clean_frame_path(output_dir, candidate, label, index)
            target_path = _aligned_reference_frame_path(
                output_dir, candidate, label, index
            )
            target = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
            if target is None:
                raise RuntimeError(f"Could not decode contour-free frame: {clean_path}")
            dx, dy, response, region = _estimate_review_translation(
                reference,
                target,
                content_box=content_box,
            )
            records.append(
                {
                    "frame_index": index,
                    "dx_px": dx,
                    "dy_px": dy,
                    "response": response,
                    "region": region,
                }
            )
            if not force and target_path.is_file():
                continue

            aligned = reference.copy()
            transform = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
            reference_content = reference[
                content_y : content_y + content_height,
                content_x : content_x + content_width,
            ]
            aligned[
                content_y : content_y + content_height,
                content_x : content_x + content_width,
            ] = cv2.warpAffine(
                reference_content,
                transform,
                (content_width, content_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=tuple(int(value) for value in reference[0, 0]),
            )
            aligned[:REVIEW_HEADER_HEIGHT, :] = reference[0, 0]
            cv2.putText(
                aligned,
                (
                    f"{pair.run.upper()}  |  Pre-injection reference aligned to "
                    f"Candidate {label} frame {index + 1}/{candidate.frame_count}"
                ),
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (245, 247, 249),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                aligned,
                (
                    f"Baseline {source_name}  |  display translation "
                    f"dx={dx:+.1f}px dy={dy:+.1f}px  |  calibration data unchanged"
                ),
                (24, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (218, 223, 228),
                1,
                cv2.LINE_AA,
            )
            temporary = target_path.with_suffix(".tmp.jpg")
            if not cv2.imwrite(
                str(temporary),
                aligned,
                [int(cv2.IMWRITE_JPEG_QUALITY), 94],
            ):
                raise RuntimeError(f"Could not write {temporary}")
            os.replace(temporary, target_path)

        _atomic_write(
            folder / "manifest.json",
            json.dumps(
                {
                    "schema": 1,
                    "run": pair.run,
                    "label": label,
                    "candidate_id": candidate.candidate_id,
                    "params_sha256": candidate.params_sha256,
                    "kind": "display_only_translated_pre_injection_reference",
                    "source_reference": str(reference_path),
                    "frames": records,
                    "created_utc": _utc_now(),
                },
                indent=2,
            ),
        )


def _write_reference_frame(
    output_dir: Path,
    run: str,
    context: Any,
) -> None:
    """Render the configured corrected baseline in review-frame coordinates."""

    import cv2
    import numpy as np

    from darsia.fluidflower.tools.calibration_review_movie import render_contour_frame

    color_analysis = getattr(context.calibration, "color_analysis", None)
    baseline = getattr(color_analysis, "base", None)
    if baseline is None or getattr(baseline, "img", None) is None:
        raise RuntimeError(
            f"{run.upper()} has no calibration-preprocessed pre-injection baseline; "
            "refusing to use a rig-level fallback with different corrections"
        )
    baseline_array = np.asarray(baseline.img)
    content_box = _review_content_box(baseline_array.shape)
    zeros = np.zeros(baseline_array.shape[:2], dtype=np.float32)
    frame = render_contour_frame(
        image=baseline_array,
        concentration_aq=zeros,
        saturation_g=zeros,
        run=run,
        variant="Pre-injection reference",
        frame_index=0,
        frame_count=1,
        elapsed_h=0.0,
        injected_mass_kg=0.0,
        detected_total_mass_kg=0.0,
        detected_gas_mass_kg=0.0,
        detected_aqueous_mass_kg=0.0,
        gas_fill_alpha=0.0,
        show_aqueous=False,
        show_gas=False,
    )

    source_value = getattr(context.fluidflower, "baseline_path", None)
    if source_value is None:
        source_value = getattr(baseline, "name", None)
    source_name = Path(str(source_value)).name if source_value else "configured baseline"
    frame[:126, :] = (19, 23, 27)
    cv2.putText(
        frame,
        f"{run.upper()}  |  Pre-injection reference  |  corrected analysis input",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 247, 249),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Configured baseline: {source_name}  |  expected CO2 mass: 0 g",
        (24, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (218, 223, 228),
        1,
        cv2.LINE_AA,
    )

    target = _reference_frame_path(output_dir, run)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.jpg")
    if not cv2.imwrite(
        str(temporary),
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 94],
    ):
        raise RuntimeError(f"Could not write {temporary}")
    os.replace(temporary, target)
    _atomic_write(
        target.parent / "manifest.json",
        json.dumps(
            {
                "schema": 1,
                "run": run,
                "kind": "corrected_pre_injection_analysis_baseline",
                "processing_source": "context.calibration.color_analysis.base",
                "source_name": source_name,
                "content_box_xywh": list(content_box),
                "created_utc": _utc_now(),
            },
            indent=2,
        ),
    )


def _native_content_box(
    image_shape: Sequence[int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    source_height, source_width = int(image_shape[0]), int(image_shape[1])
    layout_scale = height / REVIEW_FRAME_HEIGHT
    header = max(1, int(round(REVIEW_HEADER_HEIGHT * layout_scale)))
    margin = max(1, int(round(REVIEW_FRAME_MARGIN * layout_scale)))
    available_height = height - header - margin
    return (
        (width - source_width) // 2,
        header + (available_height - source_height) // 2,
        source_width,
        source_height,
    )


def _write_native_reference_frame(
    output_dir: Path,
    run: str,
    context: Any,
    *,
    width: int,
    height: int,
    force: bool,
) -> dict[str, Any]:
    """Write the corrected baseline without reducing analysis resolution."""

    import numpy as np

    from darsia.fluidflower.tools.calibration_review_movie import render_contour_frame

    color_analysis = getattr(context.calibration, "color_analysis", None)
    baseline = getattr(color_analysis, "base", None)
    if baseline is None or getattr(baseline, "img", None) is None:
        raise RuntimeError(
            f"{run.upper()} has no calibration-preprocessed pre-injection baseline"
        )
    baseline_array = np.asarray(baseline.img)
    source_value = getattr(context.fluidflower, "baseline_path", None)
    if source_value is None:
        source_value = getattr(baseline, "name", None)
    source_name = Path(str(source_value)).name if source_value else "configured baseline"
    target = _native_reference_path(output_dir, run)
    if force or not target.is_file():
        zeros = np.zeros(baseline_array.shape[:2], dtype=np.float32)
        frame = render_contour_frame(
            image=baseline_array,
            concentration_aq=zeros,
            saturation_g=zeros,
            run=run,
            variant="Pre-injection reference",
            frame_index=0,
            frame_count=1,
            elapsed_h=0.0,
            injected_mass_kg=0.0,
            detected_total_mass_kg=0.0,
            detected_gas_mass_kg=0.0,
            detected_aqueous_mass_kg=0.0,
            width=width,
            height=height,
            gas_fill_alpha=0.0,
            show_aqueous=False,
            show_gas=False,
            preserve_native_resolution=True,
            header_title=(
                f"{run.upper()}  |  Pre-injection reference  |  "
                "corrected analysis input at native resolution"
            ),
            header_mass=f"Configured baseline: {source_name}  |  expected CO2 mass: 0 g",
            status_text="",
        )
        _write_review_jpeg(
            target,
            frame,
            quality=NATIVE_REVIEW_JPEG_QUALITY,
        )
    return {
        "source_name": source_name,
        "source_shape_hw": [int(value) for value in baseline_array.shape[:2]],
        "content_box_xywh": list(_native_content_box(baseline_array.shape, width, height)),
    }


def _existing_registration_records(
    output_dir: Path,
    pair: ReviewPair,
    *,
    source_shape: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """Scale the existing display registration back to source-image pixels."""

    manifest_path = (
        _aligned_reference_frame_folder(output_dir, pair.candidate_a, "A")
        / "manifest.json"
    )
    reference_manifest_path = _reference_frame_path(output_dir, pair.run).parent / "manifest.json"
    if not manifest_path.is_file() or not reference_manifest_path.is_file():
        return {}
    manifest = _read_json(manifest_path)
    reference_manifest = _read_json(reference_manifest_path)
    content_box = reference_manifest.get("content_box_xywh")
    if not isinstance(content_box, list) or len(content_box) != 4:
        return {}
    content_width, content_height = float(content_box[2]), float(content_box[3])
    if content_width <= 0 or content_height <= 0:
        return {}
    source_height, source_width = float(source_shape[0]), float(source_shape[1])
    result: dict[int, dict[str, Any]] = {}
    for row in manifest.get("frames") or []:
        if not isinstance(row, Mapping):
            continue
        index = int(row.get("frame_index", -1))
        if index < 0:
            continue
        result[index] = {
            "dx_px": float(row.get("dx_px") or 0.0) * source_width / content_width,
            "dy_px": float(row.get("dy_px") or 0.0) * source_height / content_height,
            "response": float(row.get("response") or 0.0),
            "region": str(row.get("region") or "scaled-review-registration"),
            "source": "scaled_existing_review_registration",
        }
    return result




def _selected_pairs(
    inventory: ReviewInventory,
    runs: Sequence[str] | None,
) -> list[ReviewPair]:
    requested = {str(run).casefold() for run in (runs or ())}
    pairs = [pair for pair in inventory.pairs if not requested or pair.run in requested]
    unknown = requested - {pair.run for pair in pairs}
    if unknown:
        raise ValueError(f"Unknown or unpaired runs: {sorted(unknown)}")
    return pairs


def _portable_bundle_id(
    inventory: ReviewInventory,
    pairs: Sequence[ReviewPair],
    *,
    blind_seed: str,
) -> tuple[str, str]:
    manifest_sha256 = _sha256_file(inventory.manifest_path)
    identity = {
        "schema": 1,
        "portable_ui_version": PORTABLE_UI_VERSION,
        "source_manifest_sha256": manifest_sha256,
        "blind_seed": blind_seed,
        "pairs": [
            {
                "run": pair.run,
                "comparison_mode": pair.comparison_mode,
                "A": [
                    pair.candidate_a.candidate_id,
                    pair.candidate_a.params_sha256,
                ],
                "B": [
                    pair.candidate_b.candidate_id,
                    pair.candidate_b.params_sha256,
                ],
            }
            for pair in pairs
        ],
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), manifest_sha256


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _export_contour_frames(
    candidate: Candidate,
    label: str,
    target_paths: Sequence[Path],
    *,
    force: bool,
) -> None:
    missing_indices = [
        index
        for index, path in enumerate(target_paths)
        if force or not path.is_file()
    ]
    if not missing_indices:
        return

    import cv2

    records = _calibration_records(str(candidate.per_frame_csv))
    plot_frames = max(1, int(round(candidate.plot_seconds * candidate.fps)))
    repeated_frames = max(1, int(round(candidate.image_seconds * candidate.fps)))
    capture = cv2.VideoCapture(str(candidate.contour_movie))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open contour movie: {candidate.contour_movie}")
        for frame_index in missing_indices:
            movie_frame = (
                plot_frames
                + frame_index * repeated_frames
                + repeated_frames // 2
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, movie_frame)
            success, image = capture.read()
            if not success or image is None:
                raise RuntimeError(
                    f"Could not decode contour frame {frame_index} from "
                    f"{candidate.contour_movie}"
                )
            elapsed_h = (
                float(records[frame_index]["elapsed_h"])
                if frame_index < len(records)
                else float("nan")
            )
            image[:48, :] = (19, 23, 27)
            elapsed_label = f"{elapsed_h:.3f} h" if math.isfinite(elapsed_h) else "n/a"
            cv2.putText(
                image,
                (
                    f"{candidate.run.upper()}  |  Candidate {label}  |  "
                    f"calibration frame {frame_index + 1}/{candidate.frame_count}  "
                    f"|  t={elapsed_label}"
                ),
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (245, 247, 249),
                2,
                cv2.LINE_AA,
            )
            encoded, body = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 91],
            )
            if not encoded:
                raise RuntimeError(
                    f"Could not encode contour frame from {candidate.contour_movie}"
                )
            _atomic_write_bytes(target_paths[frame_index], body.tobytes())
    finally:
        capture.release()


def _portable_candidate_payload(
    candidate: Candidate,
    label: str,
    portable_dir: Path,
    review_output_dir: Path,
    *,
    force: bool,
    native_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if _candidate_prediction_withheld(review_output_dir, candidate):
        raise PermissionError("New prediction exports are withheld for the bottom blind check")
    relative_root = Path("assets") / candidate.run / label
    target_root = portable_dir / relative_root
    plot_relative = relative_root / "mass_plot.png"
    plot_target = portable_dir / plot_relative
    if force or not plot_target.is_file():
        _atomic_write_bytes(
            plot_target,
            _sanitized_plot_bytes(
                str(candidate.mass_plot), candidate.run, candidate.plot_title
            ),
        )

    overlay_paths = [
        target_root / "contour_overlay" / f"frame_{index:03d}.png"
        for index in range(candidate.frame_count)
    ]
    overlay_sources = [
        _native_overlay_path(review_output_dir, candidate, label, index)
        for index in range(candidate.frame_count)
    ]
    for source, target in zip(overlay_sources, overlay_paths, strict=True):
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing native contour overlay: {source}. Saved native review assets are unavailable."
            )
        if force or not target.is_file():
            _atomic_copy(source, target)

    phase_map_modes = ("all", "gas", "aqueous")
    phase_map_sources = {
        mode: [
            _native_phase_map_variant_path(
                _native_phase_map_path(
                    review_output_dir,
                    candidate,
                    label,
                    index,
                ),
                mode,
                fallback=False,
            )
            for index in range(candidate.frame_count)
        ]
        for mode in phase_map_modes
    }
    phase_maps_ready = all(
        source.is_file()
        for sources in phase_map_sources.values()
        for source in sources
    )
    phase_map_paths: dict[str, list[Path]] = {}
    if phase_maps_ready:
        for mode, sources in phase_map_sources.items():
            relative_mode = f"{PHASE_DISPLAY_POLICY}/{mode}"
            targets = [
                target_root / relative_mode / f"frame_{index:03d}.png"
                for index in range(candidate.frame_count)
            ]
            for index, (source, target) in enumerate(zip(sources, targets, strict=True)):
                if force or not target.is_file():
                    combined = _native_phase_map_path(review_output_dir, candidate, label, index)
                    _atomic_copy(two_phase_overlay_path(combined, mode), target)
            phase_map_paths[mode] = targets

    records = _calibration_records(str(candidate.per_frame_csv))
    elapsed_hours = [
        float(record["elapsed_h"])
        for record in records[: candidate.frame_count]
    ]
    frame_metrics = []
    for record in records[: candidate.frame_count]:
        frame_metrics.append(
            {
                key: float(record[key])
                for key in (
                    "elapsed_h",
                    "injected_mass_kg",
                    "detected_total_mass_kg",
                    "detected_gas_mass_kg",
                    "detected_aqueous_mass_kg",
                    "gas_probability_mean",
                    "support_fraction_p050",
                    "aqueous_probability_mean",
                    "envelope_gas_probability_mean",
                )
                if record.get(key) not in (None, "")
            }
        )
    phase_quiz_payload = _load_phase_quiz_payload(
        review_output_dir,
        candidate,
        label,
    )
    phase_quiz_proposals = phase_quiz_payload["proposals"]
    return {
        "label": f"Candidate {label}",
        "plot_url": plot_relative.as_posix(),
        "plain_frames": list(native_payload["plain_frames"]),
        "plain_frames_ready": True,
        "contour_overlays": [
            path.relative_to(portable_dir).as_posix() for path in overlay_paths
        ],
        "contour_overlays_ready": True,
        "contour_rendering": "transparent_native_resolution_overlay",
        "phase_fill_overlays": {
            mode: [path.relative_to(portable_dir).as_posix() for path in paths]
            for mode, paths in phase_map_paths.items()
        },
        "phase_fill_overlays_ready": phase_maps_ready,
        "overlay_rendering": (
            "mutually_exclusive_phase_fill_v1"
            if phase_maps_ready
            else "legacy_contour_fallback"
        ),
        "reference_frames": list(native_payload["reference_frames"]),
        "reference_offsets": list(native_payload["reference_offsets"]),
        "reference_frames_ready": True,
        "frame_count": candidate.frame_count,
        "elapsed_hours": elapsed_hours,
        "frame_metrics": [] if candidate.phase_model_only else frame_metrics,
        "phase_model_only": candidate.phase_model_only,
        "native_resolution": True,
        "canvas_wh": list(native_payload["canvas_wh"]),
        "analysis_image_shape_hw": list(native_payload["analysis_image_shape_hw"]),
        "phase_quiz_ready": bool(phase_quiz_proposals),
        "phase_quiz_proposals": phase_quiz_proposals,
        "phase_quiz_initial_count": phase_quiz_payload[
            "initial_proposal_count"
        ],
        "phase_quiz_available_count": len(phase_quiz_proposals),
    }


def _portable_native_payload(
    run: str,
    frame_count: int,
    portable_dir: Path,
    review_output_dir: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    source_manifest_path = _native_manifest_path(review_output_dir, run)
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing native-resolution manifest: {source_manifest_path}. "
            "Saved native review assets are unavailable."
        )
    source_manifest = _read_json(source_manifest_path)
    if not source_manifest.get("full_analysis_resolution"):
        raise RuntimeError(f"Native-resolution cache is not full resolution: {run}")
    if int(source_manifest.get("frame_count") or -1) != frame_count:
        raise RuntimeError(f"Native-resolution frame count mismatch for {run}")

    relative_root = Path("assets") / run / "native"
    plain_paths = [
        portable_dir / relative_root / "plain" / f"frame_{index:03d}.jpg"
        for index in range(frame_count)
    ]
    for index, target in enumerate(plain_paths):
        source = _native_frame_path(review_output_dir, run, index)
        if not source.is_file():
            raise FileNotFoundError(f"Missing native-resolution frame: {source}")
        if force or not target.is_file():
            _atomic_copy(source, target)

    reference_source = _native_reference_path(review_output_dir, run)
    if not reference_source.is_file():
        raise FileNotFoundError(f"Missing native-resolution reference: {reference_source}")
    reference_target = portable_dir / relative_root / "pre_injection_reference.jpg"
    if force or not reference_target.is_file():
        _atomic_copy(reference_source, reference_target)

    registration = source_manifest.get("reference_registration") or []
    if len(registration) != frame_count:
        raise RuntimeError(f"Native reference registration mismatch for {run}")
    reference_url = reference_target.relative_to(portable_dir).as_posix()
    return {
        "plain_frames": [
            path.relative_to(portable_dir).as_posix() for path in plain_paths
        ],
        "reference_frames": [reference_url for _ in range(frame_count)],
        "reference_offsets": [
            {
                "x": float(row.get("dx_canvas_fraction") or 0.0),
                "y": float(row.get("dy_canvas_fraction") or 0.0),
            }
            for row in registration
        ],
        "reference_url": reference_url,
        "canvas_wh": list(source_manifest["canvas_wh"]),
        "analysis_image_shape_hw": list(source_manifest["analysis_image_shape_hw"]),
        "content_box_xywh": list(source_manifest["content_box_xywh"]),
        "injection_points": review_injection_points(source_manifest),
        "injection_anchor_contract": (
            dict(source_manifest.get("injection_anchor_contract") or {})
            if isinstance(source_manifest.get("injection_anchor_contract"), Mapping)
            else {}
        ),
        "frame_sources": [
            {
                key: row[key]
                for key in (
                    "frame_index",
                    "elapsed_h",
                    "source_name",
                    "requested_name",
                    "selection_status",
                    "color_correction_active",
                    "color_correction_residual_before_correction",
                    "color_correction_residual_before_refinement",
                    "color_correction_residual",
                    "color_correction_max_swatch_residual",
                    "color_correction_refinement_steps",
                    "color_correction_strategy",
                    "color_correction_threshold",
                    "color_correction_flagged",
                    "acquired_at",
                )
                if key in row
            }
            for row in (source_manifest.get("frame_sources") or [])
            if isinstance(row, Mapping)
        ],
    }


def _portable_reference_payload(
    run: str,
    portable_dir: Path,
    review_output_dir: Path,
    *,
    force: bool,
) -> str:
    source = _reference_frame_path(review_output_dir, run)
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing pre-injection reference cache: {source}. Saved native review assets are unavailable."
        )
    relative = Path("assets") / run / "pre_injection_reference.jpg"
    target = portable_dir / relative
    if force or not target.is_file():
        _atomic_copy(source, target)
    return relative.as_posix()


PORTABLE_SERVER = r'''#!/usr/bin/env python3
import os
import socket
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # The portable review is regenerated in place during method development.
        # Never let an open browser combine an old HTML/JS shell with new native
        # image assets; that previously kept the low-resolution CSS zoom path alive.
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def available_port(start=8766, stop=8866):
    for port in range(start, stop):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free local port found")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    os.chdir(root)
    port = available_port()
    url = f"http://127.0.0.1:{port}/"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"FluidFlower calibration review: {url}", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
'''


PORTABLE_BATCH = r'''@echo off
setlocal
pushd "%~dp0" || exit /b 1
start "" "%~dp0OPEN_REVIEW.html"
popd
'''


PORTABLE_COMMAND = r'''#!/bin/sh
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if command -v open >/dev/null 2>&1; then
  exec open "$root/OPEN_REVIEW.html"
fi
if command -v xdg-open >/dev/null 2>&1; then
  exec xdg-open "$root/OPEN_REVIEW.html"
fi
printf 'Open this file in a web browser:\n%s\n' "$root/OPEN_REVIEW.html"
'''


PORTABLE_README = r'''FF AC portable calibration review

No Python is required.

1. Extract the complete ZIP archive before opening the review.
2. Double-click OPEN_REVIEW.html in Windows, macOS, or Linux.
   Windows users can alternatively double-click start_review.bat.
3. Keep the complete folder together; the HTML file uses the included images.

Review workflow:
  1. Enter your full name and select "Open".
  2. Use the shared visible-2D phase quiz. It shows the common corrected optical
     image once (using Candidate A as the display copy), because phase evidence in
     the camera image is independent of which calibration candidate is reviewed.
     Focus each outlined region and classify only what the image supports. The
     model suggestion appears after answering.
     "Mostly or entirely" is a broad dominance class: use it from a clear phase
     majority through a visually pure phase. Use a mixture class when both phases
     are materially visible and neither phase clearly dominates.
     Complete the current round before selecting "Start next adaptive round".
     New regions are prioritised using corrected optical change and earlier
     answers; existing answers are retained.
  3. Review candidates A and B, add flags, notes, and annotation regions, then save.
  4. Select "Export review" and return the generated JSON file to the project owner.

Read REVIEW_GUIDE_NO.txt or REVIEW_GUIDE_EN.txt before starting. In particular,
annotation regions must be kept local to the observed problem. Use Box for a
rectangular region, Freehand for an irregular outline, or Pixel for one exact
analysis-image pixel. Do not enclose an
entire CO2 plume when only part of its interpretation is questionable.

Runs are presented from the highest AC number to the lowest. Every edit is
stored immediately for the named reviewer in both the browser's ordinary local
storage and an append-only browser database. A valid older revision is therefore
recoverable even if the newest browser record is interrupted while being
written. In browsers that support direct file access, use "Choose autosave file"
once to keep an additional JSON copy in a folder you select. Safari keeps the
two browser-local copies but cannot retain a direct file handle; use "Export
review" for a transferable file. Use "Open review" to continue from an exported
JSON file on another computer. Reviews exported by an earlier UI version are
migrated when their source-manifest hash and candidate identities match this
package; saved decisions and draft annotations are retained.

The exported file is named for the reviewer. It records the reviewer name,
progress, candidate choice and candidate identities/hashes, flags, notes, and all
annotation regions and phase-quiz answers. Each region includes its shape,
candidate, frame number,
elapsed time, category, phase, severity, confidence, comment, and normalized
image coordinates.
Unsaved drafts are included in an export, but only saved decisions count as
completed runs.

The corrected image content is included at its native analysis resolution.
Select "1 px" for one source pixel per screen pixel, or "8 px" for discrete
pixel inspection, then pan to inspect local colour and texture. Pixel becomes
available at native zoom and stores the exact analysis-image coordinate.
Filled phase maps are independent transparent overlays, so toggling them never
replaces or recompresses the underlying image. Green is aqueous-dominant,
pink is free CO2. No third phase color is shown.
The Marker line control changes the visible annotation outline from 0.50 to
2.00 screen pixels. Use 0.50 or 0.75 px for precise work at native zoom.

The image assets are read-only. DarSIA, CUDA, OpenCL, Python, and access to the
original data are not needed. If a locally managed browser blocks file access,
serve_review.py remains available as an optional Python fallback.
'''


PORTABLE_REVIEW_GUIDE_NO = r'''FF AC Objective V2 - vurderingsveiledning

FORMÅL
Pakken inneholder en blind sammenligning av to kalibreringskandidater for hvert
forsøk. Kandidatidentiteten vises først etter at vurderingen er lagret. Vurder A
og B uavhengig; marker samme feil på begge bare dersom feilen faktisk finnes i
begge.

SLIK STARTER DU
1. Pakk ut hele ZIP-filen.
2. Åpne OPEN_REVIEW.html. Python og tilgang til originaldata er ikke nødvendig.
3. Skriv fullt navn og velg Open. Nettleseren lagrer fremdriften lokalt per navn.
4. Velg Export review når du er ferdig, og returner JSON-filen.

SYNLIG 2D-FASEQUIZ
- Focus region åpner et lokalt område i full analyseoppløsning og skjuler de
  fargekodede modellkonturene før svaret. Den hvite linjen angir området som skal
  vurderes. Klassifiser bare det som er optisk synlig i den kameraeksponerte
  2D-skiven: vann/ingen synlig CO2, hovedsakelig eller bare innløst CO2,
  hovedsakelig eller bare fri
  CO2, blanding av fri og innløst CO2, blanding av innløst CO2 og urørt vann,
  synlig CO2 med usikker fase, eller ikke vurderbart.
- «Hovedsakelig eller bare» er en bred dominansklasse. Bruk den fra tydelig
  overvekt av én fase til en visuelt ren fase. Bruk blandingsklassen når begge
  fasene er tydelig representert og ingen av dem klart dominerer. Det kreves ikke
  at den andre fasen er synlig.
- Modellens forslag vises først etter svaret. Uenighet er et nyttig resultat og
  skal ikke korrigeres for å passe modellen. Bruk High, Medium eller Low for hvor
  sikkert bildet støtter svaret.
- Next unanswered går til neste ubesvarte område. Draw detail aktiverer Freehand,
  slik at et lokalt avvik kan forklares med eksisterende annotasjonskategorier.
- Når et fokusområde er valgt, blir omrisset stående på samme bildekoordinater
  mens tidslinjen spilles av eller flyttes. På andre bilder vises det som en
  temporal sammenligning; selve quizsvaret registreres fortsatt bare på den
  opprinnelige kilderammen, som åpnes igjen med Focus region.
- Det røde punktet over tidslinjen viser kilderammen for valgt område. Ignore
  region ekskluderer et lite egnet område fra trening, mens Clear answer gjør
  området ubesvart igjen. Fasekartvisningen endres aldri automatisk av et svar.
- Fasequizen bruker det felles korrigerte bildet én gang, med kandidat A som
  visningskopi. Klassifiseringen gjelder det synlige 2D-signalet og er ikke en
  separat vurdering av kandidat A og B.
- Fullfør gjeldende runde før du starter neste adaptive runde. Nye områder
  prioriteres etter korrigert optisk endring og tidligere svar; gamle svar beholdes.
- Quizkonturen er et lokalt prøveområde inne i modellens numeriske faseklasse.
  Den er ikke en påstand om at hele den fysiske 3D-fasen er synlig i kameraets
  2D-snitt.

BILDEOPPLØSNING
- Det korrigerte analysebildet er inkludert med alle opprinnelige analysepiksler.
- Velg 1 px for å vise én bildepiksel per skjermpiksel, eller 8 px for tydelig
  inspeksjon av enkeltpiksler, og flytt deretter bildet med Pan. Fit viser hele
  bildet.
- Phase map er et separat transparent fyllag. Av/på-knappen endrer derfor ikke
  fargen eller komprimeringen i grunnbildet. Grønt er innløst-dominert, magenta
  er gass-dominert og ravgult er blandet/usikkert.
- Marker line regulerer den synlige markeringsstreken fra 0,50 til 2,00
  skjermpiksler. Bruk 0,50 eller 0,75 px ved detaljarbeid i 1 px-visning.

HVA SOM SKAL VURDERES
- Masseplott: samsvar mellom injisert og detektert total CO2, samt om fri og
  innløst CO2 utvikler seg plausibelt.
- Kjemisk følsomhet: systematiske masseavvik kan skyldes forskjeller i
  start-pH, alkalinitet, ionestyrke eller indikatorblanding. BTB responderer på
  pH, mens omregningen fra farge/pH til CO2-masse også avhenger av
  karbonatsystemet. Samme fargesignal trenger derfor ikke bety samme CO2-masse
  i alle forsøk. Noter dersom fasekartet ser visuelt riktig ut selv om
  massekurven avviker. Dette forklarer ikke isolert støy eller synlig CO2 som
  mangler fra fasekartet.
- Overflødig støy: isolerte eller blinkende områder uten romlig eller tidslig
  sammenheng med den etablerte CO2-plumen.
- Manglende deteksjon: synlig fargeendring som ligger utenfor det fylte
  CO2-fasekartet. Bruk Phase map av/på og Pre-injection reference for kontroll.
- Fri CO2: vurder plassering, levetid og utvikling. Det forventes svært lite
  residuell fri gass, men gass kan bevege seg inn eller ut av den observerte
  2D-skiven.
  Mistenkelige eksempler er ny, frakoblet gass etter injeksjon, langvarig stor
  gassandel uten tydelig optisk støtte, eller en sammenhengende gassfront som
  trekker seg mot en inaktiv injektor på en lite troverdig måte.
- Nedadgående bevegelse: under aktiv injeksjon kan trykk nær injektoren presse
  en sammenhengende gassfase lokalt nedover. Etter at injeksjonen er avsluttet,
  er fortsatt nedadgående migrasjon av fri gass uten en synlig sammenhengende
  eller trykkdrevet forklaring lite troverdig. CO2-rikt vann kan derimot få
  høyere tetthet, synke og samtidig gi sterk indikatorrespons. Bruk derfor
  Implausible downward gas migration after injection når modellen flytter fri
  gass nedover, og Likely dense aqueous CO2 misclassified as gas eller Plausible
  downward motion of dense aqueous CO2 når bildene støtter denne forklaringen.
  Husk at bevegelse inn og ut av kameraets 2D-snitt kan gi tilsynelatende
  endringer uten at fasen faktisk har flyttet seg slik i 3D.
- Innløsningsmønster: ikke bruk en hard regel om at all gass må løses nedenfra
  og opp. Marker konkret den delen av utviklingen som fremstår fysisk urimelig.
- Sikker faseinformasjon: bruk Only aqueous CO2, Mostly aqueous CO2,
  Mostly or entirely free CO2 og
  Confirmed no CO2 bare der bildet gir sterk, lokal støtte. Bruk Unsupported
  free-gas classification når modellen kaller et område gass, men bildet støtter
  CO2-påvirket vann eller ingen CO2 bedre. Disse etikettene er spesielt viktige
  for å teste kjemiresponsen og faseskillet uavhengig av totalmassekurven.

ANNOTASJONSOMRÅDER
- En markering er frame- og kandidatspesifikk. Bruk Box for et rektangel,
  Freehand for en uregelmessig, lukket kontur eller Pixel for én eksakt piksel
  i analysebildet. Pixel aktiveres først ved minst 1 px-visning. Markeringene har
  ingen bildefyll og
  skal derfor ikke skjule fargesignalene mens du tegner.
- Tegn stramt rundt det observerte problemet, med bare litt margin.
- Ikke marker hele plumen dersom bare et lokalt område er feil. Et for stort
  område merker også plausibel CO2 som feil og kan la en senere modell
  lære generell underdeteksjon.
- Gjenta området på 2-3 representative frames dersom problemet vedvarer; det er
  normalt ikke nødvendig å markere hvert bilde.
- Bruk Minor for en begrenset/usikker svakhet, Major for en tydelig fysisk feil,
  og Critical bare når kandidaten ikke kan brukes uten vesentlig reparasjon.
- Angi High, Medium eller Low confidence. Positive faseetiketter bør normalt
  begrenses til små områder med High confidence; ikke gjett fase i diffuse
  overgangssoner.
- Det er ikke nødvendig å markere masseplottet når masseavviket allerede er
  tydelig. Bruk Notes; områdene skal beskrive lokale bildefunn.
- Phase map av/på bruker samme korrigerte grunnbilde. Kartet legger bare på et
  transparent, gjensidig utelukkende klassefyll.
- Bruk frame-spesifikke Box- eller Freehand-områder for lokale funn, og Notes
  for problemer som gjelder hele kandidaten eller den samlede begrunnelsen.

Prioriter gjerne omtrent 3, 6, 9,4, 12, 15,8 og 22-24 timer, men se hele
sekvensen når en overgang er uklar. Velg Equivalent eller Unresolved dersom det
ikke finnes en tydelig vinner.
'''


PORTABLE_REVIEW_GUIDE_EN = r'''FF AC Objective V2 - review guide

PURPOSE
This package contains a blinded comparison of two calibration candidates for
each experiment. Candidate identities are revealed only after a decision is
saved. Assess A and B independently; annotate both only when the issue is
actually present in both.

GETTING STARTED
1. Extract the complete ZIP archive.
2. Open OPEN_REVIEW.html. Python and access to the original data are not needed.
3. Enter your full name and select Open. Progress is stored locally per reviewer.
4. Select Export review when finished and return the generated JSON file.

VISIBLE 2D PHASE QUIZ
- The quiz uses the shared corrected optical image once, with Candidate A as the
  display copy. Visible phase evidence is therefore not classified separately for
  Candidate A and Candidate B.
- Focus region opens a local area at full analysis resolution and hides the
  colour-coded model contours before the answer. The white outline is the area
  to assess. Classify only optical evidence in the camera-facing 2D slice:
  water/no visible CO2, mostly or entirely aqueous CO2, mostly or entirely free
  CO2, mixed free and
  aqueous CO2, mixed aqueous CO2 and untouched water, visible CO2 with an
  uncertain phase, or not assessable.
- "Mostly or entirely" is a broad dominance class. Use it from a clear majority
  of one phase through a visually pure phase. Use the mixture class when both
  phases are materially visible and neither clearly dominates. The other phase
  does not have to be visible.
- The model suggestion is revealed only after the answer. Disagreement is useful
  evidence and must not be changed merely to match the model. Record High,
  Medium, or Low confidence in the visible evidence.
- Next unanswered opens the next unanswered region. Draw detail activates
  Freehand so a local discrepancy can be explained with the existing annotation
  categories.
- Once a focus region is selected, its outline remains at the same image
  coordinates while the timeline is played or scrubbed. Other frames are shown
  as temporal comparisons; the quiz answer is still recorded only on its source
  frame, which Focus region opens again.
- The red point above the timeline marks the selected region's source frame.
  Ignore region excludes an unsuitable region from training, while Clear answer
  makes it unanswered again. Answering never changes phase-map visibility.
- Complete the current round before starting the next adaptive round. New
  regions are prioritised from corrected optical change and earlier answers;
  existing answers are retained.
- The quiz outline is a local probe inside the model's numerical phase class. It
  is not a claim that the complete physical 3D phase is visible in the camera's
  2D slice.

IMAGE RESOLUTION
- The corrected analysis image contains every native analysis pixel.
- Select 1 px to show one image pixel per screen pixel, or 8 px for discrete
  pixel inspection, then use Pan. Fit shows the complete image.
- Phase maps are a separate transparent filled layer. Toggling them therefore
  never changes the colour or compression of the base image. Green is
  aqueous CO2 and pink is free CO2. No third phase color is shown.
- Marker line adjusts the visible annotation outline from 0.50 to 2.00 screen
  pixels. Use 0.50 or 0.75 px for precise work in the 1 px view.

WHAT TO ASSESS
- Mass plot: agreement between injected and detected total CO2, and whether the
  gaseous/aqueous split evolves plausibly.
- Chemical sensitivity: systematic mass deviations may reflect differences in
  initial pH, alkalinity, ionic strength, or indicator mixture. BTB responds to
  pH, whereas conversion from colour/pH to CO2 mass also depends on carbonate
  chemistry. The same colour signal therefore need not represent the same CO2
  mass in every experiment. Note when the phase map appears visually credible
  despite a mass-curve mismatch. Chemistry does not explain isolated noise or
  visible CO2 omitted by the phase map.
- Excess noise: isolated or blinking regions without spatial or temporal support
  from the established CO2 plume.
- Missed detection: visible colour changes outside the filled CO2 phase map.
  Use Phase map and Pre-injection reference to cross-check the image.
- Free CO2: assess location, lifetime, and temporal evolution. Very little
  residual free gas is expected, although gas may enter or leave the observed
  2D slice.
  Suspicious examples include new disconnected gas after injection, prolonged
  free gas without clear optical evidence, or a coherent gas front retreating
  towards an inactive injector in an implausible way.
- Downward motion: during active injection, pressure near the injector can
  force a connected gas phase locally downward. After injection has ended,
  continued downward migration of free gas without a visible connected or
  pressure-driven explanation is implausible. CO2-rich water can instead become
  denser, descend, and produce a strong indicator response. Use Implausible
  downward gas migration after injection when the model moves free gas down,
  and Likely dense aqueous CO2 misclassified as gas or Plausible downward motion
  of dense aqueous CO2 when the images support that explanation. Motion into or
  out of the camera-facing 2D slice can produce apparent changes that do not
  represent the same motion in the complete 3D volume.
- Dissolution pattern: do not enforce a universal bottom-to-top dissolution
  rule. Annotate the specific part of the evolution that appears implausible.
- Confident phase evidence: use Only aqueous CO2, Mostly aqueous CO2,
  Mostly or entirely free CO2,
  and Confirmed no CO2 only where the image provides strong local evidence. Use
  Incorrect free-gas prediction in this model when some gas predictions in the
  selected area are wrong; real gas may also be present there. These labels are intended
  to test chemistry response and phase separation independently of total mass.

ANNOTATION REGIONS
- An annotation is specific to one candidate and one frame. Use Box for a
  rectangle, Freehand for an irregular closed outline, or Pixel for one exact
  analysis-image pixel. Pixel is enabled at 1 px zoom or closer. Annotations
  have no image fill and
  therefore do not hide colour evidence while drawing.
- Draw tightly around the observed problem, with only a small margin.
- Do not enclose the entire plume when only a local interpretation is wrong. An
  oversized region labels plausible CO2 as erroneous and may train later models to
  under-detect CO2 generally.
- Repeat the region on 2-3 representative frames when the problem persists; every
  frame normally does not need an annotation.
- Use Minor for a limited or uncertain weakness, Major for a clear physical
  error, and Critical only when substantial repair is required.
- Record High, Medium, or Low confidence. Positive phase labels should normally
  be small High-confidence regions; do not guess in diffuse transition zones.
- Marking the mass plot is unnecessary when the mismatch is already clear. Use
  Notes; regions should describe local image evidence.
- Phase map on/off uses the same corrected base image. The map adds only a
  transparent, mutually exclusive class fill.
- Use frame-specific Box or Freehand regions for local findings and Notes for
  sequence-wide problems or the overall reasoning.

Suggested checkpoints are approximately 3, 6, 9.4, 12, 15.8, and 22-24 hours.
Choose Equivalent or Unresolved when no defensible winner can be identified.
'''


PORTABLE_EMAIL_NO = r'''Emne: Blind vurdering av FF AC-kalibreringer

Hei,

Jeg sender en nettleserbasert pakke for blind sammenligning av to
kalibreringskandidater for hvert av åtte FF AC-forsøk. Formålet er å vurdere om
godt samsvar med injisert totalmasse også gir en fysisk troverdig romlig
fordeling mellom innløst og fri CO2. Kandidatidentiteten vises ikke før
vurderingen er lagret.

Pakk ut hele ZIP-filen og åpne OPEN_REVIEW.html. Programmet krever ikke Python
eller tilgang til originaldata. Skriv fullt navn og velg Open. Fremdriften
lagres automatisk i nettleseren per navn. Når gjennomgangen er ferdig, velg
Export review og send den genererte JSON-filen tilbake til meg.

For hvert forsøk vises kandidat A og B side ved side med massekurve og opptil 25
kalibreringsbilder. Phase map kan slås av for å se det korrigerte bildet uten
tolkning, og Pre-injection reference viser referansebildet behandlet på samme
måte. Bildene ligger i full analyseoppløsning. Velg 1 px for piksel-for-piksel-
visning, og bruk Pan for å flytte bildet. Fasekartet er et separat transparent
fyllag og endrer derfor ikke grunnbildets farger. Vurder begge kandidatene
uavhengig og velg A, B, Equivalent eller Unresolved.

Jeg er spesielt interessert i:
- overflødig, frakoblet eller blinkende støy;
- synlig CO2 som ikke fanges av konturene;
- usannsynlig plassering eller levetid for fri CO2;
- faseforveksling der synkende, CO2-rikt vann med sterk fargerespons tolkes som
  nedadgående fri gass;
- fortsatt nedadgående bevegelse av fri gass etter at aktiv injeksjon er slutt,
  uten en synlig sammenhengende eller trykkdrevet forklaring.

Avvik mellom injisert og detektert masse skal også vurderes særskilt. Noen
forsøk kan ha forskjeller i start-pH, alkalinitet eller indikatorblanding. BTB
responderer på pH, mens koblingen fra farge/pH til CO2-masse avhenger av
karbonatsystemet. Det samme fargesignalet trenger derfor ikke representere den
samme CO2-mengden i alle forsøk. Noter gjerne dersom massekurven avviker, men
konturene likevel følger den synlige plumen godt. Kjemiske forskjeller skal
imidlertid ikke brukes til å bortforklare frakoblet støy eller manglende synlig
deteksjon.

Det skal ikke brukes en absolutt regel om at gass alltid må løses nedenfra og
opp. Under aktiv injeksjon kan trykk presse sammenhengende gass lokalt nedover,
restgass kan ligge ved injektoren, og en fase kan bevege seg inn eller ut av
kameraets 2D-snitt. Etter injeksjonen er likevel fortsatt nedadgående migrasjon
av fri gass uten en slik forklaring lite troverdig. Marker den konkrete
observasjonen og velg den mest presise fysiske forklaringen i annotasjonsmenyen.

Annotasjonsbokser skal tegnes stramt rundt det aktuelle problemet. Ikke legg en
stor boks rundt hele plumen dersom bare en liten del er feil: da merkes også
fysisk plausibel CO2 som feil, noe som kan la en senere kalibreringsmodell lære
å underdetektere generelt. Dersom et problem vedvarer, er 2-3 representative
frames normalt tilstrekkelig. Bruk Notes for svakheter som gjelder hele
tidsserien.

En kort norsk og engelsk veiledning ligger også i pakken.

Vennlig hilsen
[Avsenderens navn]
'''


def _render_review_html(bundle: Mapping[str, Any] | None = None) -> str:
    serialized = "null"
    if bundle is not None:
        serialized = json.dumps(
            bundle,
            ensure_ascii=False,
            separators=(",", ":"),
        ).replace("</", "<\\/")
    if PORTABLE_BUNDLE_MARKER not in HTML:
        raise RuntimeError("Portable bundle marker is missing from review HTML")
    flag_labels = json.dumps(REVIEW_FLAG_LABELS, ensure_ascii=True).replace("</", "<\\/")
    return HTML.replace(PORTABLE_BUNDLE_MARKER, serialized).replace(
        "__FFAC_REVIEW_FLAG_LABELS__", flag_labels
    )


def prepare_portable_review(
    portable_dir: Path,
    review_output_dir: Path,
    inventory: ReviewInventory,
    *,
    blind_seed: str = DEFAULT_BLIND_SEED,
    runs: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create a self-contained, browser-based review package."""

    pairs = _selected_pairs(inventory, runs)
    portable_dir = portable_dir.resolve()
    bundle_id, manifest_sha256 = _portable_bundle_id(
        inventory,
        pairs,
        blind_seed=blind_seed,
    )
    pair_payloads: dict[str, Any] = {}
    for pair in pairs:
        print(f"Preparing portable review assets for {pair.run.upper()}...", flush=True)
        native_payload = _portable_native_payload(
            pair.run,
            pair.candidate_a.frame_count,
            portable_dir,
            review_output_dir,
            force=force,
        )
        candidate_a = _portable_candidate_payload(
            pair.candidate_a,
            "A",
            portable_dir,
            review_output_dir,
            force=force,
            native_payload=native_payload,
        )
        candidate_b = _portable_candidate_payload(
            pair.candidate_b,
            "B",
            portable_dir,
            review_output_dir,
            force=force,
            native_payload=native_payload,
        )
        pair_payloads[pair.run] = {
            "schema": 1,
            "run": pair.run,
            "comparison_mode": pair.comparison_mode,
            "reference_url": native_payload["reference_url"],
            "reference_ready": True,
            "reference_label": "Pre-injection reference",
            "native_resolution": True,
            "canvas_wh": native_payload["canvas_wh"],
            "analysis_image_shape_hw": native_payload[
                "analysis_image_shape_hw"
            ],
            "content_box_xywh": native_payload["content_box_xywh"],
            "injection_points": native_payload["injection_points"],
            "injection_anchor_contract": native_payload[
                "injection_anchor_contract"
            ],
            "frame_sources": native_payload["frame_sources"],
            "contours_are_separate_overlays": True,
            "phase_display_policy": PHASE_DISPLAY_POLICY,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "reveal": {
                "A": pair.candidate_a.provenance(),
                "B": pair.candidate_b.provenance(),
            },
        }

    bundle = {
        "schema": 1,
        "ui_version": PORTABLE_UI_VERSION,
        "format": PORTABLE_BUNDLE_FORMAT,
        "bundle_id": bundle_id,
        "title": "FluidFlower Calibration Review",
        "created_utc": _utc_now(),
        "source_manifest": str(inventory.manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "runs": [pair.run for pair in pairs],
        "unpaired_runs": [
            run for run in inventory.unpaired if not runs or run in set(runs)
        ],
        "pairs": pair_payloads,
        "session_format": PORTABLE_SESSION_FORMAT,
        "launch": {
            "primary": "OPEN_REVIEW.html",
            "python_required": False,
            "optional_python_server": "serve_review.py",
        },
        "review_logging": {
            "reviewer_name_required": True,
            "reviewer_id_embedded": True,
            "browser_autosave_per_reviewer": True,
            "browser_revision_database": True,
            "direct_file_autosave_when_supported": True,
            "export_format": "json",
            "export_filename_includes_reviewer": True,
            "export_contains": [
                "reviewer_name",
                "progress",
                "candidate_choice",
                "candidate_identity_and_parameter_hash",
                "flags",
                "notes",
                "annotation_regions",
                "phase_quiz_answers",
            ],
        },
        "documentation": {
            "readme": "README.txt",
            "review_guide_no": "REVIEW_GUIDE_NO.txt",
            "review_guide_en": "REVIEW_GUIDE_EN.txt",
            "email_text_no": "EMAIL_TEXT_NO.txt",
        },
    }
    portable_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        portable_dir / "portable_manifest.json",
        json.dumps(bundle, indent=2, ensure_ascii=False),
    )
    rendered_html = _render_review_html(bundle)
    _atomic_write(portable_dir / "index.html", rendered_html)
    _atomic_write(portable_dir / "OPEN_REVIEW.html", rendered_html)
    _atomic_write(portable_dir / "serve_review.py", PORTABLE_SERVER)
    _atomic_write(portable_dir / "start_review.bat", PORTABLE_BATCH)
    _atomic_write(portable_dir / "start_review.command", PORTABLE_COMMAND)
    _atomic_write(portable_dir / "README.txt", PORTABLE_README)
    _atomic_write(
        portable_dir / "REVIEW_GUIDE_NO.txt",
        PORTABLE_REVIEW_GUIDE_NO,
    )
    _atomic_write(
        portable_dir / "REVIEW_GUIDE_EN.txt",
        PORTABLE_REVIEW_GUIDE_EN,
    )
    _atomic_write(portable_dir / "EMAIL_TEXT_NO.txt", PORTABLE_EMAIL_NO)
    return {
        "schema": 1,
        "format": PORTABLE_BUNDLE_FORMAT,
        "bundle_id": bundle_id,
        "portable_dir": str(portable_dir),
        "run_count": len(pairs),
        "runs": [pair.run for pair in pairs],
        "index": str(portable_dir / "index.html"),
        "launcher_html": str(portable_dir / "OPEN_REVIEW.html"),
        "launcher_windows": str(portable_dir / "start_review.bat"),
        "manifest": str(portable_dir / "portable_manifest.json"),
    }


class ReviewStore:
    def __init__(
        self,
        output_dir: Path,
        pairs: Sequence[ReviewPair],
        *,
        read_only: bool = False,
    ) -> None:
        self.output_dir = resolve_review_path(output_dir)
        self.json_path = self.output_dir / "pairwise_review_decisions.json"
        self.csv_path = self.output_dir / "pairwise_review_decisions.csv"
        self.annotation_csv_path = self.output_dir / "pairwise_review_annotations.csv"
        self.phase_quiz_csv_path = self.output_dir / "pairwise_review_phase_quiz.csv"
        self.browser_draft_dir = self.output_dir / "browser_review_drafts"
        self.pairs = {pair.run: pair for pair in pairs}
        self.lock = threading.RLock()
        self.read_only = bool(read_only)
        self.payload = self._load()

    @property
    def is_read_only(self) -> bool:
        if self.read_only:
            return True
        session_path = self.output_dir / "review_session.json"
        if not session_path.is_file():
            return False
        try:
            return bool(_read_json(session_path).get("editing_locked"))
        except (OSError, TypeError, ValueError):
            return False

    def _assert_writable(self) -> None:
        if self.is_read_only:
            raise RuntimeError(
                "This review session has been frozen as training evidence and is "
                "locked to an immutable model version. Start a new review session "
                "to add or revise evidence."
            )

    def update_pairs(self, pairs: Sequence[ReviewPair]) -> None:
        """Publish newly completed pairs without invalidating saved reviews."""

        replacement = {pair.run: pair for pair in pairs}
        with self.lock:
            decisions = self.payload.get("decisions") or {}
            for run, decision in decisions.items():
                if run not in replacement or run not in self.pairs:
                    continue
                old_pair = self.pairs[run]
                new_pair = replacement[run]
                old_ids = {
                    old_pair.candidate_a.candidate_id,
                    old_pair.candidate_b.candidate_id,
                }
                new_ids = {
                    new_pair.candidate_a.candidate_id,
                    new_pair.candidate_b.candidate_id,
                }
                if old_ids != new_ids:
                    raise RuntimeError(
                        f"Candidate identities changed after {run.upper()} was reviewed; "
                        "start a new review output directory for the revised campaign."
                    )
                saved_ids = {
                    str((decision.get("candidate_a") or {}).get("candidate_id") or ""),
                    str((decision.get("candidate_b") or {}).get("candidate_id") or ""),
                }
                if saved_ids != new_ids:
                    raise RuntimeError(
                        f"Saved provenance for {run.upper()} differs from the live campaign."
                    )
            self.pairs = replacement

    def _load(self) -> dict[str, Any]:
        if self.json_path.is_file():
            payload = _read_json(self.json_path)
            payload["schema"] = 2
            payload.setdefault("decisions", {})
            return payload
        return {
            "schema": 2,
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "decisions": {},
        }

    def decisions(self) -> dict[str, dict[str, Any]]:
        with self.lock:
            self.payload = self._load()
            return dict(self.payload.get("decisions") or {})

    def ensure_reviewer_provenance(self) -> bool:
        """Persist reviewer identity on legacy records and every child label."""

        changed = False
        with self.lock:
            lock_path = self.output_dir / ".pairwise_review_write.lock"
            with _exclusive_path_lock(lock_path):
                self.payload = self._load()
                for record in (self.payload.get("decisions") or {}).values():
                    if not isinstance(record, dict):
                        continue
                    reviewer = str(record.get("reviewer") or "").strip()
                    if not reviewer:
                        continue
                    reviewer_id = _reviewer_identifier(reviewer)
                    if record.get("reviewer_id") != reviewer_id:
                        record["reviewer_id"] = reviewer_id
                        changed = True
                    for field in (
                        "frame_choices",
                        "annotations",
                        "phase_quiz_answers",
                    ):
                        for item in record.get(field) or []:
                            if not isinstance(item, dict):
                                continue
                            if item.get("reviewer") != reviewer:
                                item["reviewer"] = reviewer
                                changed = True
                            if item.get("reviewer_id") != reviewer_id:
                                item["reviewer_id"] = reviewer_id
                                changed = True
                if changed:
                    self.payload["reviewer_provenance_migrated_utc"] = _utc_now()
                    self._write_locked()
        return changed

    def reclassify_bottom_edge_indicator_carryover(
        self,
        *,
        run: str | None = None,
        minimum_y: float = 0.90,
    ) -> dict[str, Any]:
        """Reclassify unambiguous lower-edge noise while preserving provenance."""

        threshold = float(minimum_y)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_y must be between 0 and 1")
        selected_run = str(run or "").casefold()
        changed_by_run: dict[str, int] = {}
        backup_path: Path | None = None
        stamp = _utc_now()
        with self.lock:
            lock_path = self.output_dir / ".pairwise_review_write.lock"
            with _exclusive_path_lock(lock_path):
                self.payload = self._load()
                decisions = self.payload.get("decisions") or {}
                for run_name, decision in decisions.items():
                    if selected_run and str(run_name).casefold() != selected_run:
                        continue
                    changed = 0
                    for annotation in decision.get("annotations") or []:
                        category = str(
                            annotation.get("category")
                            or annotation.get("issue_key")
                            or ""
                        )
                        try:
                            y = float(annotation.get("y"))
                        except (TypeError, ValueError):
                            continue
                        if category != "isolated_noise" or y < threshold:
                            continue
                        annotation["previous_category"] = category
                        annotation["category"] = (
                            "incomplete_wash_indicator_carryover"
                        )
                        if "issue_key" in annotation:
                            annotation["issue_key"] = (
                                "incomplete_wash_indicator_carryover"
                            )
                        annotation["phase"] = "unspecified"
                        annotation["reclassified_utc"] = stamp
                        annotation["reclassification_rule"] = (
                            f"isolated_noise_at_or_below_normalized_y_{threshold:.3f}"
                        )
                        for key in ANNOTATION_CATEGORY_SEMANTICS[
                            "incomplete_wash_indicator_carryover"
                        ]:
                            annotation.pop(key, None)
                        annotation.update(
                            annotation_category_semantics(
                                "incomplete_wash_indicator_carryover"
                            )
                        )
                        changed += 1
                    if changed:
                        changed_by_run[str(run_name)] = changed
                total = sum(changed_by_run.values())
                if total:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = self.output_dir / (
                        "pairwise_review_decisions.before_indicator_carryover_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    if self.json_path.is_file():
                        shutil.copy2(self.json_path, backup_path)
                    migrations = self.payload.setdefault("annotation_migrations", [])
                    migrations.append(
                        {
                            "schema": 1,
                            "kind": "bottom_edge_indicator_carryover",
                            "created_utc": stamp,
                            "source_category": "isolated_noise",
                            "target_category": (
                                "incomplete_wash_indicator_carryover"
                            ),
                            "minimum_normalized_y": threshold,
                            "changed_by_run": changed_by_run,
                            "changed_total": total,
                            "backup_path": str(backup_path) if backup_path else "",
                        }
                    )
                    self.payload["updated_utc"] = stamp
                    self._write_locked()
        return {
            "changed_total": sum(changed_by_run.values()),
            "changed_by_run": changed_by_run,
            "minimum_normalized_y": threshold,
            "backup_path": str(backup_path) if backup_path else "",
        }

    @staticmethod
    def _reviewer_draft_key(reviewer: str) -> str:
        return hashlib.sha256(reviewer.casefold().encode("utf-8")).hexdigest()[:20]

    def _browser_draft_path(self, run: str, reviewer: str) -> Path:
        return (
            self.browser_draft_dir
            / self._reviewer_draft_key(reviewer)
            / f"{run}.json"
        )

    def _browser_draft_history_path(self, run: str, reviewer: str) -> Path:
        return (
            self.browser_draft_dir
            / self._reviewer_draft_key(reviewer)
            / f"{run}.history.jsonl"
        )

    def browser_draft(self, run: str, reviewer: str) -> dict[str, Any] | None:
        run = str(run or "").casefold()
        reviewer = str(reviewer or "").strip()
        if run not in self.pairs or not reviewer:
            return None
        path = self._browser_draft_path(run, reviewer)
        payload: dict[str, Any] | None = None
        try:
            if path.is_file():
                payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            payload = None
        history = _latest_jsonl_object(
            self._browser_draft_history_path(run, reviewer)
        )
        if history is not None:
            if history.get("operation") == "delete":
                return None
            history_payload = history.get("payload")
            if isinstance(history_payload, Mapping) and (
                payload is None
                or str(history_payload.get("updated_utc") or "")
                >= str(payload.get("updated_utc") or "")
            ):
                payload = dict(history_payload)
        if payload is None:
            return None
        if (
            str(payload.get("run") or "").casefold() != run
            or str(payload.get("reviewer") or "").strip().casefold()
            != reviewer.casefold()
            or not isinstance(payload.get("draft"), Mapping)
        ):
            return None
        return payload

    def browser_drafts(self, reviewer: str) -> dict[str, dict[str, Any]]:
        reviewer = str(reviewer or "").strip()
        if not reviewer:
            return {}
        drafts: dict[str, dict[str, Any]] = {}
        for run in self.pairs:
            payload = self.browser_draft(run, reviewer)
            if payload is not None:
                drafts[run] = payload
        return drafts

    def save_browser_draft(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_writable()
        run = str(request.get("run") or "").casefold()
        if run not in self.pairs:
            raise ValueError(f"Unknown review run: {run!r}")
        reviewer = str(request.get("reviewer") or "").strip()
        if not reviewer:
            raise ValueError("Reviewer is required")
        reviewer_id = _reviewer_identifier(reviewer)
        path = self._browser_draft_path(run, reviewer)
        history_path = self._browser_draft_history_path(run, reviewer)
        lock_path = path.with_suffix(".lock")
        if bool(request.get("delete")):
            now = _utc_now()
            with _exclusive_path_lock(lock_path):
                append_bounded_snapshot_journal(
                    history_path,
                    {
                        "schema": 1,
                        "kind": "browser-review-recovery-revision",
                        "revision_id": hashlib.sha256(
                            f"{run}:{reviewer}:{now}:delete".encode("utf-8")
                        ).hexdigest(),
                        "operation": "delete",
                        "run": run,
                        "reviewer": reviewer,
                        "reviewer_id": reviewer_id,
                        "recorded_utc": now,
                    },
                )
                path.unlink(missing_ok=True)
            return {
                "schema": 1,
                "run": run,
                "reviewer": reviewer,
                "reviewer_id": reviewer_id,
                "deleted": True,
                "updated_utc": now,
            }
        draft = request.get("draft")
        if not isinstance(draft, Mapping):
            raise ValueError("draft must be an object")
        clean_draft = dict(draft)
        clean_draft["reviewer"] = reviewer
        clean_draft["reviewer_id"] = reviewer_id
        for field in ("frame_choices", "annotations", "phase_quiz_answers"):
            clean_items: list[Any] = []
            for raw_item in clean_draft.get(field) or []:
                if not isinstance(raw_item, Mapping):
                    clean_items.append(raw_item)
                    continue
                item = dict(raw_item)
                item["reviewer"] = reviewer
                item["reviewer_id"] = reviewer_id
                clean_items.append(item)
            if field in clean_draft:
                clean_draft[field] = clean_items
        payload = {
            "schema": 1,
            "kind": "browser-review-recovery-draft",
            "run": run,
            "reviewer": reviewer,
            "reviewer_id": reviewer_id,
            "updated_utc": _utc_now(),
            "draft": clean_draft,
        }
        encoded = json.dumps(payload, indent=2, ensure_ascii=False)
        if len(encoded.encode("utf-8")) > 20 * 1024 * 1024:
            raise ValueError("Browser recovery draft exceeds 20 MiB")
        with _exclusive_path_lock(lock_path):
            append_bounded_snapshot_journal(
                history_path,
                {
                    "schema": 1,
                    "kind": "browser-review-recovery-revision",
                    "revision_id": hashlib.sha256(
                        encoded.encode("utf-8")
                    ).hexdigest(),
                    "operation": "save",
                    "run": run,
                    "reviewer": reviewer,
                    "reviewer_id": reviewer_id,
                    "recorded_utc": payload["updated_utc"],
                    "payload": payload,
                },
            )
            _atomic_write(path, encoded)
        return payload

    def _clean_phase_quiz_answers(
        self,
        pair: ReviewPair,
        raw_answers: Any,
    ) -> list[dict[str, Any]]:
        if raw_answers in (None, ""):
            return []
        if not isinstance(raw_answers, list):
            raise ValueError("phase_quiz_answers must be a list")
        proposals: dict[tuple[str, str], dict[str, Any]] = {}
        for label, candidate in (
            ("A", pair.candidate_a),
            ("B", pair.candidate_b),
        ):
            for proposal in _load_phase_quiz_proposals(
                self.output_dir,
                candidate,
                label,
            ):
                proposal_id = str(proposal.get("proposal_id") or "")
                if proposal_id:
                    proposals[(label, proposal_id)] = proposal

        clean_answers: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for raw_answer in raw_answers:
            if not isinstance(raw_answer, Mapping):
                raise ValueError("Each phase quiz answer must be an object")
            label = str(raw_answer.get("candidate") or "").upper()
            proposal_id = str(raw_answer.get("proposal_id") or "").strip()
            key = (label, proposal_id)
            if label not in {"A", "B"} or key not in proposals:
                raise ValueError(f"Unknown phase quiz proposal: {key!r}")
            if key in seen:
                raise ValueError(f"Duplicate phase quiz answer: {key!r}")
            seen.add(key)

            human_choice = str(raw_answer.get("human_choice") or "")
            if human_choice not in VALID_PHASE_QUIZ_CHOICES:
                raise ValueError(f"Invalid phase quiz choice: {human_choice!r}")
            confidence = str(raw_answer.get("confidence") or "medium")
            if confidence not in {"high", "medium", "low"}:
                raise ValueError(
                    f"Invalid phase quiz confidence: {confidence!r}"
                )

            proposal = proposals[key]
            suggested_phase = str(proposal.get("suggested_phase") or "")
            if suggested_phase not in VALID_PHASE_QUIZ_MODEL_PHASES:
                raise ValueError(
                    f"Invalid model phase in phase quiz proposal: {suggested_phase!r}"
                )
            expected = PHASE_QUIZ_EXPECTED_ANSWERS.get(suggested_phase)
            agreement = (
                None
                if expected is None
                or human_choice in PHASE_QUIZ_NONSCORING_CHOICES
                else human_choice in expected
            )
            phase_payload = visible_phase_payload(human_choice)
            existing_schema = str(raw_answer.get("phase_label_schema") or "")
            if (
                existing_schema
                and existing_schema != TRANSFERABLE_PHASE_LABEL_SCHEMA
                and raw_answer.get("phase_class")
            ):
                phase_payload.update(
                    {
                        "phase_label_schema": existing_schema,
                        "phase_class": str(raw_answer["phase_class"]),
                        "phase_dominance": str(
                            raw_answer.get("phase_dominance") or "legacy_unspecified"
                        ),
                        "phase_components": list(
                            raw_answer.get("phase_components") or []
                        ),
                    }
                )
            clean_answers.append(
                {
                    "proposal_id": proposal_id,
                    "candidate": label,
                    "candidate_id": str(proposal.get("candidate_id") or ""),
                    "params_sha256": str(proposal.get("params_sha256") or ""),
                    "frame_index": int(proposal.get("frame_index") or 0),
                    "elapsed_h": float(proposal.get("elapsed_h") or 0.0),
                    "suggested_phase": suggested_phase,
                    "suggestion_basis": str(
                        proposal.get("suggestion_basis") or ""
                    ),
                    **phase_payload,
                    **assess_model_phase(suggested_phase, human_choice),
                    "confidence": confidence,
                    "agreement": agreement,
                    "interpretation_scope": "visible_2d_slice_only",
                    "coordinate_space": str(
                        proposal.get("coordinate_space") or ""
                    ),
                    "shape": str(proposal.get("shape") or "freehand"),
                    "points": proposal.get("points") or [],
                    "x": float(proposal.get("x") or 0.0),
                    "y": float(proposal.get("y") or 0.0),
                    "width": float(proposal.get("width") or 0.0),
                    "height": float(proposal.get("height") or 0.0),
                    "analysis_polygon_px": (
                        proposal.get("analysis_polygon_px") or []
                    ),
                    "analysis_bbox_xywh": (
                        proposal.get("analysis_bbox_xywh") or []
                    ),
                    "focus_box": proposal.get("focus_box") or {},
                    "model_stats": proposal.get("model_stats") or {},
                    "comment": str(raw_answer.get("comment") or "").strip()[:500],
                    "answered_utc": str(
                        raw_answer.get("answered_utc") or _utc_now()
                    )[:64],
                }
            )
        return clean_answers

    def save(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_writable()
        run = str(request.get("run") or "").casefold()
        if run not in self.pairs:
            raise ValueError(f"Unknown review run: {run!r}")
        reviewer = str(request.get("reviewer") or "").strip()
        if not reviewer:
            raise ValueError("Reviewer is required")
        reviewer_id = _reviewer_identifier(reviewer)
        pair = self.pairs[run]
        frame_count = min(
            int(pair.candidate_a.frame_count),
            int(pair.candidate_b.frame_count),
        )
        candidate_a_records = _calibration_records(
            str(pair.candidate_a.per_frame_csv)
        )
        elapsed_hours = [row.get("elapsed_h") for row in candidate_a_records]
        has_frame_choices = "frame_choices" in request
        clean_frame_choices = normalize_frame_choices(
            request.get("frame_choices"),
            frame_count=frame_count,
            elapsed_hours=elapsed_hours,
            comparison_mode=pair.comparison_mode,
            strict=has_frame_choices,
        )
        if (pair.candidate_a.quality_flags or {}).get("selected_camera_review"):
            from darsia.fluidflower.phase_annotation_binding import stamp_camera_identity
            for index,frame_choice in enumerate(clean_frame_choices):
                source=candidate_a_records[frame_choice["frame_index"]]
                clean_frame_choices[index]=stamp_camera_identity(frame_choice,source,run,
                    dataset=pair.candidate_a.dataset,selected_camera=True)
        for frame_choice in clean_frame_choices:
            frame_choice["reviewer"] = reviewer
            frame_choice["reviewer_id"] = reviewer_id
            if not frame_choice["updated_utc"]:
                frame_choice["updated_utc"] = _utc_now()
        frame_choice_summary = summarize_frame_choices(
            clean_frame_choices,
            frame_count=frame_count,
            comparison_mode=pair.comparison_mode,
        )
        legacy_run_choice = str(request.get("legacy_run_choice") or "")
        if has_frame_choices and not legacy_run_choice:
            existing = (self.payload.get("decisions") or {}).get(run) or {}
            if existing and not existing.get("frame_choices"):
                legacy_run_choice = str(existing.get("choice") or "")
        choice = (
            str(frame_choice_summary["aggregate_choice"])
            if has_frame_choices
            else str(request.get("choice") or "").casefold()
        )
        if choice not in VALID_CHOICES:
            raise ValueError(f"Invalid choice: {choice!r}")

        flags = request.get("flags") or {}
        if not isinstance(flags, dict):
            raise ValueError("flags must be an object")
        clean_flags: dict[str, list[str]] = {}
        for label in ("A", "B"):
            values = flags.get(label) or []
            if not isinstance(values, list):
                raise ValueError(f"flags.{label} must be a list")
            invalid = sorted(set(map(str, values)) - VALID_FLAGS)
            if invalid:
                raise ValueError(f"Invalid flags for candidate {label}: {invalid}")
            clean_flags[label] = sorted(set(map(str, values)))

        existing_payload = self.payload
        if "reference_guides" not in request:
            try:
                existing_payload = self._load()
            except (OSError, ValueError, TypeError):
                pass
        existing_record = (existing_payload.get("decisions") or {}).get(run) or {}
        existing_annotations_by_id = {
            str(item.get("id")): item for item in existing_record.get("annotations") or []
            if item.get("id")
        }
        reference_guides = normalize_reference_guides(
            (
                request.get("reference_guides")
                if "reference_guides" in request
                else existing_record.get("reference_guides")
            ),
            strict="reference_guides" in request,
        )

        phase_quiz_answers = self._clean_phase_quiz_answers(
            pair,
            request.get("phase_quiz_answers"),
        )
        for answer in phase_quiz_answers:
            answer["reviewer"] = reviewer
            answer["reviewer_id"] = reviewer_id
        quiz_payloads = [
            _load_phase_quiz_payload(self.output_dir, pair.candidate_a, "A"),
            _load_phase_quiz_payload(self.output_dir, pair.candidate_b, "B"),
        ]
        initial_quiz_limit = max(
            (int(payload["initial_proposal_count"]) for payload in quiz_payloads),
            default=0,
        )
        available_quiz_limit = max(
            (len(payload["proposals"]) for payload in quiz_payloads),
            default=0,
        )
        shared_proposals = quiz_payloads[0]["proposals"] if quiz_payloads else []
        shared_by_id = {
            str(item.get("proposal_id") or ""): item for item in shared_proposals
        }
        raw_selected_ids = request.get("phase_quiz_selected_ids")
        if raw_selected_ids is not None and not isinstance(raw_selected_ids, list):
            raise ValueError("phase_quiz_selected_ids must be a list")
        answered_shared_ids = [
            str(item.get("proposal_id") or "")
            for item in phase_quiz_answers
            if item.get("candidate") == "A"
            and str(item.get("proposal_id") or "") in shared_by_id
        ]
        if raw_selected_ids is None:
            if answered_shared_ids:
                selected_ids = list(dict.fromkeys(answered_shared_ids))
            else:
                requested_limit = int(
                    request.get(
                        "phase_quiz_per_candidate_limit", initial_quiz_limit
                    )
                )
                if not initial_quiz_limit <= requested_limit <= available_quiz_limit:
                    raise ValueError(
                        "phase_quiz_per_candidate_limit must be between "
                        f"{initial_quiz_limit} and {available_quiz_limit}"
                    )
                selected_ids = [
                    str(item.get("proposal_id") or "")
                    for item in shared_proposals[:requested_limit]
                ]
        else:
            selected_ids = []
            for raw_id in raw_selected_ids:
                proposal_id = str(raw_id or "")
                if proposal_id not in shared_by_id:
                    raise ValueError(
                        f"Unknown shared phase quiz proposal: {proposal_id!r}"
                    )
                if proposal_id not in selected_ids:
                    selected_ids.append(proposal_id)
        for proposal_id in answered_shared_ids:
            if proposal_id not in selected_ids:
                selected_ids.append(proposal_id)
        phase_quiz_per_candidate_limit = len(selected_ids)

        raw_rounds = request.get("phase_quiz_rounds") or []
        if not isinstance(raw_rounds, list):
            raise ValueError("phase_quiz_rounds must be a list")
        clean_rounds: list[dict[str, Any]] = []
        round_ids: set[str] = set()
        selected_id_set = set(selected_ids)
        for index, raw_round in enumerate(raw_rounds):
            if not isinstance(raw_round, Mapping):
                raise ValueError("Each phase quiz round must be an object")
            ids = []
            for raw_id in raw_round.get("proposal_ids") or []:
                proposal_id = str(raw_id or "")
                if proposal_id not in selected_id_set:
                    raise ValueError(
                        "Phase quiz round references an inactive proposal"
                    )
                if proposal_id not in round_ids:
                    ids.append(proposal_id)
                    round_ids.add(proposal_id)
            if ids:
                clean_rounds.append(
                    {
                        "round": int(raw_round.get("round") or index + 1),
                        "selection_mode": str(
                            raw_round.get("selection_mode")
                            or "adaptive_corrected_optical_change"
                        )[:80],
                        "proposal_ids": ids,
                        "created_utc": str(raw_round.get("created_utc") or "")[:64],
                    }
                )
        missing_round_ids = [
            proposal_id for proposal_id in selected_ids if proposal_id not in round_ids
        ]
        if missing_round_ids:
            clean_rounds.insert(
                0,
                {
                    "round": 1,
                    "selection_mode": "legacy_or_initial_batch",
                    "proposal_ids": missing_round_ids,
                    "created_utc": "",
                },
            )
        annotations = request.get("annotations") or []
        if not isinstance(annotations, list):
            raise ValueError("annotations must be a list")
        clean_annotations: list[dict[str, Any]] = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError("Each annotation must be an object")
            label = str(annotation.get("candidate") or "").upper()
            if label not in {"A", "B"}:
                raise ValueError(f"Invalid annotation candidate: {label!r}")
            candidate = pair.candidate(label)
            frame_index = int(annotation.get("frame_index", -1))
            if not 0 <= frame_index < candidate.frame_count:
                raise ValueError(
                    f"Invalid frame index for candidate {label}: {frame_index}"
                )
            category = str(annotation.get("category") or "")
            if category not in VALID_ANNOTATION_CATEGORIES:
                raise ValueError(f"Invalid annotation category: {category!r}")
            phase = str(annotation.get("phase") or "unspecified")
            if phase not in VALID_ANNOTATION_PHASES:
                raise ValueError(f"Invalid annotation phase: {phase!r}")
            severity = str(annotation.get("severity") or "major")
            if severity not in VALID_ANNOTATION_SEVERITIES:
                raise ValueError(f"Invalid annotation severity: {severity!r}")
            confidence = str(annotation.get("confidence") or "unspecified")
            if confidence not in VALID_ANNOTATION_CONFIDENCES:
                raise ValueError(f"Invalid annotation confidence: {confidence!r}")
            shape, coordinates, points = _clean_annotation_geometry(annotation)
            pixel_metadata: dict[str, int] = {}
            if shape == "point":
                pixel_fields = (
                    "analysis_pixel_x",
                    "analysis_pixel_y",
                    "analysis_image_width",
                    "analysis_image_height",
                    "canvas_pixel_x",
                    "canvas_pixel_y",
                    "review_canvas_width",
                    "review_canvas_height",
                )
                try:
                    pixel_metadata = {
                        name: int(annotation[name]) for name in pixel_fields
                    }
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "Point annotations require exact analysis and canvas pixels"
                    ) from error
                if (
                    pixel_metadata["analysis_image_width"] <= 0
                    or pixel_metadata["analysis_image_height"] <= 0
                    or pixel_metadata["review_canvas_width"] <= 0
                    or pixel_metadata["review_canvas_height"] <= 0
                    or not 0
                    <= pixel_metadata["analysis_pixel_x"]
                    < pixel_metadata["analysis_image_width"]
                    or not 0
                    <= pixel_metadata["analysis_pixel_y"]
                    < pixel_metadata["analysis_image_height"]
                    or not 0
                    <= pixel_metadata["canvas_pixel_x"]
                    < pixel_metadata["review_canvas_width"]
                    or not 0
                    <= pixel_metadata["canvas_pixel_y"]
                    < pixel_metadata["review_canvas_height"]
                ):
                    raise ValueError("Point annotation pixel is outside the image")
            records = _calibration_records(str(candidate.per_frame_csv))
            elapsed_h = (
                float(records[frame_index]["elapsed_h"])
                if frame_index < len(records)
                else None
            )
            annotation_id = str(annotation.get("id") or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", annotation_id):
                annotation_id = f"box-{len(clean_annotations) + 1}"
            semantics = annotation_category_semantics(category)
            stored = existing_annotations_by_id.get(annotation_id)
            historical_candidate_issue = (category in {
                        "unsupported_free_gas_classification",
                        "unsupported_aqueous_co2_classification",
                    } and stored
                    and stored.get("category") == category
                    and not stored.get("training_label_policy_version")
                    and not annotation.get("training_label_policy_version"))
            if historical_candidate_issue:
                # Saving another field in an old review must not rewrite the
                # original candidate-specific issue's archived metadata.
                semantics = {key: stored[key] for key in ANNOTATION_SEMANTIC_FIELDS
                             if key in stored}
            elif category in {"unsupported_free_gas_classification",
                              "unsupported_aqueous_co2_classification"}:
                for key, expected in (("candidate_id", candidate.candidate_id),
                                      ("params_sha256", candidate.params_sha256)):
                    declared = annotation.get(key)
                    if declared and str(declared) != str(expected):
                        raise ValueError(f"Candidate-specific phase issue has a different {key}")
            default_annotation_scope = (
                "shared_source_image"
                if semantics.get("interpretation_scope")
                == "source_image_shared_across_candidates"
                else "candidate_specific"
            )
            phase_fields = {
                key: annotation[key]
                for key in (
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
                    "category_schema",
                    "evidence_audit",
                )
                if key in annotation
            }
            if (getattr(candidate,"quality_flags",None) or {}).get("selected_camera_review") and annotation.get("source_binding_contract"):
                from darsia.fluidflower.phase_annotation_binding import CONTRACT as camera_contract,stamp_camera_identity
                expected=stamp_camera_identity({},records[frame_index],run,dataset=candidate.dataset,selected_camera=True)
                if annotation.get("source_binding_contract")!=camera_contract or any(
                        annotation.get(key)!=expected[key] for key in ("source_sha256","source_label_identity")):
                    raise ValueError("Selected annotation is not bound to this exact camera acquisition")
                phase_fields.update({key:expected[key] for key in ("source_sha256","source_label_identity","source_binding_contract")})
            clean_annotation = (
                {
                    "id": annotation_id,
                    "candidate": label,
                    "candidate_id": (str(stored.get("candidate_id") or "")
                                     if historical_candidate_issue else candidate.candidate_id),
                    "params_sha256": (str(stored.get("params_sha256") or "")
                                      if historical_candidate_issue else candidate.params_sha256),
                    "frame_index": frame_index,
                    "elapsed_h": elapsed_h,
                    "category": category,
                    "candidate_attribution": str(
                        stored.get("candidate_attribution") or ""
                        if historical_candidate_issue else
                        annotation.get("candidate_attribution") or label
                    )[:8],
                    "phase": phase,
                    "severity": severity,
                    "confidence": confidence,
                    "contours_visible": bool(
                        annotation.get("contours_visible", True)
                    ),
                    "coordinate_space": "normalized_native_review_canvas_16x9",
                    "shape": shape,
                    "points": points,
                    **pixel_metadata,
                    "x": coordinates["x"],
                    "y": coordinates["y"],
                    "width": coordinates["width"],
                    "height": coordinates["height"],
                    "comment": str(annotation.get("comment") or "").strip()[:500],
                    **phase_fields,
                    **semantics,
                    "annotation_scope": str(
                        annotation.get("annotation_scope")
                        or default_annotation_scope
                    )[:80],
                    "previous_category": str(
                        annotation.get("previous_category") or ""
                    )[:80],
                    "reclassified_utc": str(
                        annotation.get("reclassified_utc") or ""
                    )[:64],
                    "reclassification_rule": str(
                        annotation.get("reclassification_rule") or ""
                    )[:160],
                    "source": str(annotation.get("source") or "")[:80],
                    "issue_key": str(
                        annotation.get("issue_key") or category
                    )[:80],
                    "contour_layer": str(
                        annotation.get("contour_layer") or "all"
                    )[:24],
                    "contour_opacity": max(
                        0.0,
                        min(1.0, float(annotation.get("contour_opacity", 1.0))),
                    ),
                    "local_baseline_search_preview": bool(annotation.get("local_baseline_search_preview", False)),
                    "created_utc": str(annotation.get("created_utc") or "")[:64],
                    "updated_utc": str(annotation.get("updated_utc") or "")[:64],
                    "reviewer": reviewer,
                    "reviewer_id": reviewer_id,
                }
            )
            from darsia.fluidflower.phase_annotation_binding import PHOTO_IDENTITY_KEY, stamp_camera_identity
            # New boxes record their photograph; saved legacy boxes stay as they were.
            if (stored is None or PHOTO_IDENTITY_KEY in stored) and frame_index < len(records):
                clean_annotation = stamp_camera_identity(
                    clean_annotation, records[frame_index], run, dataset=candidate.dataset)
            clean_annotations.append(clean_annotation)
        winner = None
        if choice == "candidate_a":
            winner = pair.candidate_a.candidate_id
        elif choice == "candidate_b":
            winner = pair.candidate_b.candidate_id

        record = {
            "schema": 2,
            "run": run,
            "comparison_mode": pair.comparison_mode,
            "reviewer": reviewer,
            "reviewer_id": reviewer_id,
            "choice": choice,
            "winner_candidate_id": winner,
            "frame_choice_schema": FRAME_CHOICE_SCHEMA if has_frame_choices else None,
            "frame_choices": clean_frame_choices,
            "frame_choice_summary": frame_choice_summary,
            "legacy_run_choice": (
                legacy_run_choice if legacy_run_choice in VALID_CHOICES else None
            ),
            "flags": clean_flags,
            "annotations": clean_annotations,
            "reference_guides": reference_guides,
            "phase_quiz_answers": phase_quiz_answers,
            "phase_quiz_selected_ids": selected_ids,
            "phase_quiz_rounds": clean_rounds,
            "phase_quiz_per_candidate_limit": phase_quiz_per_candidate_limit,
            "notes": str(request.get("notes") or "").strip(),
            "updated_utc": _utc_now(),
            "candidate_a": pair.candidate_a.provenance(),
            "candidate_b": pair.candidate_b.provenance(),
        }
        with self.lock:
            lock_path = self.output_dir / ".pairwise_review_write.lock"
            with _exclusive_path_lock(lock_path):
                self.payload = self._load()
                decisions = self.payload.setdefault("decisions", {})
                decisions[run] = record
                self.payload["updated_utc"] = record["updated_utc"]
                self._write_locked()
        try:
            self.save_browser_draft(
                {"run": run, "reviewer": reviewer, "delete": True}
            )
        except OSError:
            # The final record is authoritative. A stale recovery draft is
            # ignored by timestamp if a transient NAS error prevents cleanup.
            pass
        return record

    def _write_locked(self) -> None:
        self._assert_writable()
        _atomic_write(
            self.json_path,
            json.dumps(self.payload, indent=2, ensure_ascii=False),
        )
        columns = (
            "run",
            "comparison_mode",
            "reviewer",
            "reviewer_id",
            "choice",
            "winner_candidate_id",
            "frame_choice_count",
            "frame_choice_total",
            "frame_choices_complete",
            "frame_choices_json",
            "candidate_a_id",
            "candidate_a_role",
            "candidate_a_variant",
            "candidate_a_sha256",
            "candidate_a_nmae",
            "candidate_a_flags",
            "candidate_b_id",
            "candidate_b_role",
            "candidate_b_variant",
            "candidate_b_sha256",
            "candidate_b_nmae",
            "candidate_b_flags",
            "annotation_count",
            "annotations_json",
            "reference_guide_count",
            "reference_guides_json",
            "phase_quiz_answer_count",
            "phase_quiz_answers_json",
            "notes",
            "updated_utc",
        )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        decisions = self.payload.get("decisions") or {}
        for run in sorted(decisions, key=_run_number):
            row = decisions[run]
            candidate_a = row["candidate_a"]
            candidate_b = row["candidate_b"]
            writer.writerow(
                {
                    "run": run,
                    "comparison_mode": row.get("comparison_mode", "pair"),
                    "reviewer": row["reviewer"],
                    "reviewer_id": row.get("reviewer_id")
                    or _reviewer_identifier(row["reviewer"]),
                    "choice": row["choice"],
                    "winner_candidate_id": row.get("winner_candidate_id") or "",
                    "frame_choice_count": int(
                        (row.get("frame_choice_summary") or {}).get(
                            "reviewed", len(row.get("frame_choices") or [])
                        )
                    ),
                    "frame_choice_total": int(
                        (row.get("frame_choice_summary") or {}).get(
                            "total",
                            min(
                                int(candidate_a.get("frame_count") or 0),
                                int(candidate_b.get("frame_count") or 0),
                            ),
                        )
                    ),
                    "frame_choices_complete": bool(
                        (row.get("frame_choice_summary") or {}).get(
                            "complete", False
                        )
                    ),
                    "frame_choices_json": json.dumps(
                        row.get("frame_choices") or [],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "candidate_a_id": candidate_a["candidate_id"],
                    "candidate_a_role": candidate_a["role"],
                    "candidate_a_variant": candidate_a["variant"],
                    "candidate_a_sha256": candidate_a["params_sha256"],
                    "candidate_a_nmae": candidate_a.get("normalized_mae"),
                    "candidate_a_flags": "|".join(row["flags"]["A"]),
                    "candidate_b_id": candidate_b["candidate_id"],
                    "candidate_b_role": candidate_b["role"],
                    "candidate_b_variant": candidate_b["variant"],
                    "candidate_b_sha256": candidate_b["params_sha256"],
                    "candidate_b_nmae": candidate_b.get("normalized_mae"),
                    "candidate_b_flags": "|".join(row["flags"]["B"]),
                    "annotation_count": len(row.get("annotations") or []),
                    "annotations_json": json.dumps(
                        row.get("annotations") or [],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "reference_guide_count": len(
                        row.get("reference_guides") or []
                    ),
                    "reference_guides_json": json.dumps(
                        row.get("reference_guides") or [],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "phase_quiz_answer_count": len(
                        row.get("phase_quiz_answers") or []
                    ),
                    "phase_quiz_answers_json": json.dumps(
                        row.get("phase_quiz_answers") or [],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "notes": row["notes"],
                    "updated_utc": row["updated_utc"],
                }
            )
        _atomic_write(self.csv_path, stream.getvalue())

        annotation_columns = (
            "run",
            "comparison_mode",
            "reviewer",
            "reviewer_id",
            "choice",
            "annotation_id",
            "candidate",
            "candidate_id",
            "params_sha256",
            "frame_index",
            "elapsed_h",
            "category",
            "candidate_attribution",
            "phase",
            "severity",
            "confidence",
            "contours_visible",
            "coordinate_space",
            "shape",
            "points_json",
            "analysis_pixel_x",
            "analysis_pixel_y",
            "analysis_image_width",
            "analysis_image_height",
            "canvas_pixel_x",
            "canvas_pixel_y",
            "review_canvas_width",
            "review_canvas_height",
            "x",
            "y",
            "width",
            "height",
            "comment",
            "evidence_scope",
            "co2_evidence",
            "training_role",
            "training_phase_class",
            "training_scope",
            "evaluation_scope",
            "training_label_policy_version",
            "artifact_origin",
            "interpretation_scope",
            "annotation_scope",
            "previous_category",
            "reclassified_utc",
            "reclassification_rule",
            "decision_updated_utc",
        )
        annotation_stream = io.StringIO(newline="")
        annotation_writer = csv.DictWriter(
            annotation_stream,
            fieldnames=annotation_columns,
        )
        annotation_writer.writeheader()
        for run in sorted(decisions, key=_run_number):
            row = decisions[run]
            for annotation in row.get("annotations") or []:
                annotation_writer.writerow(
                    {
                        "run": run,
                        "comparison_mode": row.get("comparison_mode", "pair"),
                        "reviewer": row["reviewer"],
                        "reviewer_id": annotation.get("reviewer_id")
                        or row.get("reviewer_id")
                        or _reviewer_identifier(row["reviewer"]),
                        "choice": row["choice"],
                        **{
                            name: annotation.get(name)
                            for name in annotation_columns
                            if name
                            not in {
                                "run",
                                "comparison_mode",
                                "reviewer",
                                "reviewer_id",
                                "choice",
                                "annotation_id",
                                "decision_updated_utc",
                            }
                        },
                        "annotation_id": annotation.get("id"),
                        "points_json": json.dumps(
                            annotation.get("points") or [],
                            separators=(",", ":"),
                        ),
                        "decision_updated_utc": row["updated_utc"],
                    }
                )
        _atomic_write(self.annotation_csv_path, annotation_stream.getvalue())

        phase_quiz_columns = (
            "run",
            "comparison_mode",
            "reviewer",
            "reviewer_id",
            "choice",
            "proposal_id",
            "candidate",
            "candidate_id",
            "params_sha256",
            "frame_index",
            "elapsed_h",
            "suggested_phase",
            "human_choice",
            "phase_label_schema",
            "phase_class",
            "confidence",
            "agreement",
            "interpretation_scope",
            "coordinate_space",
            "shape",
            "points_json",
            "analysis_polygon_px_json",
            "analysis_bbox_xywh_json",
            "model_stats_json",
            "comment",
            "answered_utc",
            "decision_updated_utc",
        )
        phase_quiz_stream = io.StringIO(newline="")
        phase_quiz_writer = csv.DictWriter(
            phase_quiz_stream,
            fieldnames=phase_quiz_columns,
        )
        phase_quiz_writer.writeheader()
        for run in sorted(decisions, key=_run_number):
            row = decisions[run]
            for answer in row.get("phase_quiz_answers") or []:
                phase_quiz_writer.writerow(
                    {
                        "run": run,
                        "comparison_mode": row.get("comparison_mode", "pair"),
                        "reviewer": row["reviewer"],
                        "reviewer_id": answer.get("reviewer_id")
                        or row.get("reviewer_id")
                        or _reviewer_identifier(row["reviewer"]),
                        "choice": row["choice"],
                        **{
                            name: answer.get(name)
                            for name in phase_quiz_columns
                            if name
                            not in {
                                "run",
                                "comparison_mode",
                                "reviewer",
                                "reviewer_id",
                                "choice",
                                "points_json",
                                "analysis_polygon_px_json",
                                "analysis_bbox_xywh_json",
                                "model_stats_json",
                                "decision_updated_utc",
                            }
                        },
                        "points_json": json.dumps(
                            answer.get("points") or [],
                            separators=(",", ":"),
                        ),
                        "analysis_polygon_px_json": json.dumps(
                            answer.get("analysis_polygon_px") or [],
                            separators=(",", ":"),
                        ),
                        "analysis_bbox_xywh_json": json.dumps(
                            answer.get("analysis_bbox_xywh") or [],
                            separators=(",", ":"),
                        ),
                        "model_stats_json": json.dumps(
                            answer.get("model_stats") or {},
                            separators=(",", ":"),
                        ),
                        "decision_updated_utc": row["updated_utc"],
                    }
                )
        _atomic_write(self.phase_quiz_csv_path, phase_quiz_stream.getvalue())


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if match is None:
        raise ValueError("Unsupported Range header")
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        raise ValueError("Empty Range header")
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    else:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        start = max(0, size - suffix)
        end = size - 1
    if start >= size or start < 0 or end < start:
        raise ValueError("Range outside file")
    return start, min(end, size - 1)


class ReviewApp:
    def __init__(
        self,
        inventory: ReviewInventory,
        store: ReviewStore,
        *,
        default_reviewer: str = "",
        selection_dir: Path | None = None,
        blind_seed: str = DEFAULT_BLIND_SEED,
        watch: bool = False,
        prepare_new: bool = False,
        prepare_backend: str = "auto",
        watch_interval: float = 10.0,
    ) -> None:
        self.inventory = inventory
        self.store = store
        self.default_reviewer = default_reviewer
        self.pairs = {pair.run: pair for pair in inventory.pairs}
        self.selection_dir = (
            None if selection_dir is None else Path(selection_dir).resolve()
        )
        self.blind_seed = blind_seed
        self.watch = bool(watch and self.selection_dir is not None)
        self.prepare_new = False
        self.prepare_backend = prepare_backend
        self.watch_interval = max(2.0, float(watch_interval))
        self.inventory_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.watcher: threading.Thread | None = None
        self.inventory_revision = 1
        self.live_status: dict[str, Any] = {
            "enabled": self.watch,
            "state": "idle" if self.watch else "disabled",
            "updated_utc": _utc_now(),
            "message": "",
        }

    def _pair(self, run: str) -> ReviewPair:
        with self.inventory_lock:
            pair = self.pairs.get(run.casefold())
        if pair is None:
            raise KeyError(run)
        return pair

    def _review_root(self, pair: ReviewPair) -> Path:
        roots = {
            candidate.review_output_dir
            for candidate in (pair.candidate_a, pair.candidate_b)
            if candidate.review_output_dir is not None
        }
        if len(roots) > 1:
            raise RuntimeError(
                f"Candidates for {pair.run} use different review asset roots"
            )
        return next(iter(roots), self.store.output_dir)

    @staticmethod
    def _dataset(pair: ReviewPair) -> str:
        values = {
            candidate.dataset.casefold()
            for candidate in (pair.candidate_a, pair.candidate_b)
            if candidate.dataset
        }
        return next(iter(values)) if len(values) == 1 else ""

    def _publish_inventory(self, inventory: ReviewInventory) -> None:
        with self.inventory_lock:
            old_signature = tuple(
                (
                    pair.run,
                    pair.candidate_a.candidate_id,
                    pair.candidate_b.candidate_id,
                )
                for pair in self.inventory.pairs
            )
            new_signature = tuple(
                (
                    pair.run,
                    pair.candidate_a.candidate_id,
                    pair.candidate_b.candidate_id,
                )
                for pair in inventory.pairs
            )
            self.store.update_pairs(inventory.pairs)
            self.inventory = inventory
            self.pairs = {pair.run: pair for pair in inventory.pairs}
            if old_signature != new_signature:
                self.inventory_revision += 1

    def refresh_inventory(self) -> None:
        """Discover and prepare pairs whose postflight assets just completed."""

        if not self.watch or self.selection_dir is None:
            return
        discovered = discover_review_inventory(
            self.selection_dir,
            blind_seed=self.blind_seed,
            allow_incomplete=True,
        )
        with self.inventory_lock:
            current_runs = set(self.pairs)
            existing_pairs = tuple(self.inventory.pairs)
        discovered_by_run = {pair.run: pair for pair in discovered.pairs}
        # A brief NAS outage must not remove an already published review item.
        for pair in existing_pairs:
            discovered_by_run.setdefault(pair.run, pair)
        merged_pairs = tuple(
            sorted(
                discovered_by_run.values(),
                key=lambda item: _run_number(item.run),
                reverse=True,
            )
        )
        discovered = replace(
            discovered,
            pairs=merged_pairs,
            pending_runs=tuple(
                run for run in discovered.pending_runs if run not in discovered_by_run
            ),
        )
        new_runs = [
            pair.run for pair in discovered.pairs if pair.run not in current_runs
        ]
        self._publish_inventory(discovered)
        self.live_status = {
            "enabled": True,
            "state": "idle",
            "updated_utc": _utc_now(),
            "message": (
                "Published " + ", ".join(run.upper() for run in new_runs)
                if new_runs
                else ""
            ),
        }

    def _watch_loop(self) -> None:
        while not self.stop_event.wait(self.watch_interval):
            try:
                self.refresh_inventory()
            except Exception as exc:  # pragma: no cover - runtime/NAS safety net
                self.live_status = {
                    "enabled": True,
                    "state": "error",
                    "updated_utc": _utc_now(),
                    "message": f"{type(exc).__name__}: {exc}",
                }

    def start_live_updates(self) -> None:
        if not self.watch or self.watcher is not None:
            return
        watcher_context = copy_context()
        self.watcher = threading.Thread(
            target=lambda: watcher_context.run(self._watch_loop),
            name="ffac-review-inventory-watcher",
            daemon=True,
        )
        self.watcher.start()

    def stop_live_updates(self) -> None:
        self.stop_event.set()
        if self.watcher is not None:
            self.watcher.join(timeout=max(2.0, self.watch_interval + 1.0))
            self.watcher = None

    def state(self, reviewer: str = "") -> dict[str, Any]:
        decisions = self.store.decisions()
        browser_drafts = self.store.browser_drafts(reviewer)
        with self.inventory_lock:
            inventory = self.inventory
            revision = self.inventory_revision
            live_status = dict(self.live_status)
        runs = []
        for pair in inventory.pairs:
            decision = decisions.get(pair.run)
            draft = browser_drafts.get(pair.run)
            frame_total = min(
                int(pair.candidate_a.frame_count),
                int(pair.candidate_b.frame_count),
            )
            frame_summary = (
                dict(decision.get("frame_choice_summary") or {})
                if decision
                else {}
            )
            decision_complete = bool(
                decision
                and (
                    not decision.get("frame_choice_schema")
                    or frame_summary.get("complete", False)
                )
            )
            runs.append(
                {
                    "run": pair.run,
                    "dataset": self._dataset(pair),
                    "reviewed": decision_complete,
                    "drafted": draft is not None,
                    "choice": decision.get("choice") if decision else None,
                    "frame_choices_reviewed": int(
                        frame_summary.get(
                            "reviewed",
                            len(decision.get("frame_choices") or [])
                            if decision
                            else 0,
                        )
                    ),
                    "frame_choices_total": int(
                        frame_summary.get("total", frame_total)
                    ),
                    "frame_choices_complete": bool(
                        frame_summary.get("complete", False)
                    ),
                    "updated_utc": (
                        draft.get("updated_utc")
                        if draft is not None
                        else decision.get("updated_utc")
                        if decision
                        else None
                    ),
                }
            )
        return {
            "schema": 1,
            "runs": runs,
            "reviewed": sum(item["reviewed"] for item in runs),
            "drafted": sum(item["drafted"] for item in runs),
            "total": len(runs),
            "unpaired_runs": list(inventory.unpaired),
            "pending_runs": list(inventory.pending_runs),
            "inventory_revision": revision,
            "live": {
                **live_status,
                "poll_seconds": self.watch_interval,
                "prepare_new": self.prepare_new,
            },
            "default_reviewer": self.default_reviewer,
            "decision_json": str(self.store.json_path),
            "decision_csv": str(self.store.csv_path),
            "annotation_csv": str(self.store.annotation_csv_path),
            "phase_quiz_csv": str(self.store.phase_quiz_csv_path),
        }

    def pair_payload(self, run: str, reviewer: str = "") -> dict[str, Any]:
        pair = self._pair(run)
        decision = self.store.decisions().get(pair.run)
        browser_draft = self.store.browser_draft(pair.run, reviewer)
        review_root = self._review_root(pair)
        native_manifest_path = _native_manifest_path(review_root, pair.run)
        native_manifest = (
            _read_json(native_manifest_path)
            if native_manifest_path.is_file()
            else {}
        )
        reference_path, native_frames_ready = _native_photograph_availability(
            review_root, pair.run, native_manifest, pair.candidate_a.frame_count,
        )
        reference_version = (
            reference_path.stat().st_mtime_ns if reference_path.is_file() else 0
        )
        from darsia.fluidflower.phase_photographic_assets import original_display_available, raw_detail_available
        v5_raw_detail_available = (
            raw_detail_available(native_manifest)
            and raw_detail_available(native_manifest, reference=True)
        )
        native_overlays_ready = {
            label: all(
                _native_overlay_path(
                    self.store.output_dir,
                    candidate,
                    label,
                    index,
                ).is_file()
                for index in range(candidate.frame_count)
            )
            for label, candidate in (
                ("A", pair.candidate_a),
                ("B", pair.candidate_b),
            )
        }
        native_phase_maps_ready = {
            label: all(
                _native_phase_map_path(
                    self.store.output_dir,
                    candidate,
                    label,
                    index,
                ).is_file()
                for index in range(candidate.frame_count)
            )
            for label, candidate in (
                ("A", pair.candidate_a),
                ("B", pair.candidate_b),
            )
        }
        registration = native_manifest.get("reference_registration") or []
        if not registration:
            registration = _sidecar_reference_registration(
                review_root, pair.run, native_manifest_path)
        reference_offsets = [
            {
                "x": float(row.get("dx_canvas_fraction") or 0.0),
                "y": float(row.get("dy_canvas_fraction") or 0.0),
            }
            for row in registration
            if isinstance(row, Mapping)
        ]
        reference_ready = (
            reference_path.is_file()
            and native_frames_ready
            and len(reference_offsets) == pair.candidate_a.frame_count
        )
        from darsia.fluidflower.tools.phase_injection_references import rig_reference_frame_index
        from darsia.fluidflower.raw_photographic_review import RAW_REVIEW_FORMAT
        raw_review = native_manifest.get("format") == RAW_REVIEW_FORMAT
        rig_baseline_index = None if raw_review else rig_reference_frame_index(native_manifest)
        if raw_review:
            # The RAW baseline is shown as its own photograph; no translation is measured.
            reference_ready = native_frames_ready
        if rig_baseline_index is not None:
            # Reference mode shows this exact source and its own existing map.
            # It is not translated onto the currently selected frame.
            reference_ready = native_frames_ready
        reveal = None
        if decision is not None:
            reveal = {
                "A": pair.candidate_a.provenance(),
                "B": pair.candidate_b.provenance(),
            }
        elapsed_a = [
            float(row["elapsed_h"])
            for row in _calibration_records(str(pair.candidate_a.per_frame_csv))
        ][: pair.candidate_a.frame_count]
        elapsed_b = [
            float(row["elapsed_h"])
            for row in _calibration_records(str(pair.candidate_b.per_frame_csv))
        ][: pair.candidate_b.frame_count]
        records_a = _calibration_records(str(pair.candidate_a.per_frame_csv))
        records_b = _calibration_records(str(pair.candidate_b.per_frame_csv))
        if "ffac_selected_policy_a_sources" in native_manifest:
            elapsed_a=[float(row["elapsed_h"]) for row in native_manifest["frame_sources"]]
            elapsed_b=list(elapsed_a)

        withheld_a = _candidate_prediction_withheld(self.store.output_dir, pair.candidate_a)
        withheld_b = _candidate_prediction_withheld(self.store.output_dir, pair.candidate_b)
        if withheld_a:
            records_a = []
        if withheld_b:
            records_b = []

        def metrics(records: Sequence[Mapping[str, str]], count: int) -> list[dict[str, float]]:
            return [
                {
                    key: float(record[key])
                    for key in (
                        "elapsed_h",
                        "injected_mass_kg",
                        "detected_total_mass_kg",
                        "detected_gas_mass_kg",
                        "detected_aqueous_mass_kg",
                        "gas_probability_mean",
                        "support_fraction_p050",
                        "aqueous_probability_mean",
                        "co2_presence_probability_mean",
                        "envelope_gas_probability_mean",
                    )
                    if record.get(key) not in (None, "")
                }
                for record in records[:count]
            ]

        def metric_semantics(records: Sequence[Mapping[str, str]]) -> str:
            return (
                "normalized_2d_optical_evidence"
                if any(record.get("gas_probability_mean") not in (None, "") for record in records)
                else "physical_mass_with_3d_geometry"
            )

        canvas_wh = list(native_manifest.get("canvas_wh") or [])
        analysis_shape = list(native_manifest.get("analysis_image_shape_hw") or [])
        frame_sources = [
            dict(row)
            for row in (native_manifest.get("frame_sources") or [])
            if isinstance(row, Mapping)
        ]
        if raw_review:
            analysis_shape = list(native_manifest["native_shape_hw"])
            from darsia.fluidflower.raw_photographic_review import studio_rows
            frame_sources = studio_rows(native_manifest)
        phase_quiz_payload_a = _load_phase_quiz_payload(
            self.store.output_dir,
            pair.candidate_a,
            "A",
        )
        phase_quiz_payload_b = _load_phase_quiz_payload(
            self.store.output_dir,
            pair.candidate_b,
            "B",
        )
        phase_quiz_a = phase_quiz_payload_a["proposals"]
        phase_quiz_b = phase_quiz_payload_b["proposals"]
        if withheld_a:
            phase_quiz_a = []
            phase_quiz_payload_a = {**phase_quiz_payload_a, "initial_proposal_count": 0}
        if withheld_b:
            phase_quiz_b = []
            phase_quiz_payload_b = {**phase_quiz_payload_b, "initial_proposal_count": 0}
        return {
            "schema": 1,
            "run": pair.run,
            "dataset": self._dataset(pair),
            "comparison_mode": pair.comparison_mode,
            "reference_url": f"/media/{pair.run}/reference",
            "reference_version": reference_version,
            "reference_ready": reference_ready,
            "rig_saved_reference_index": rig_baseline_index,
            "reference_label": "Pre-injection reference",
            "native_resolution": bool(native_manifest.get("full_analysis_resolution")),
            "canvas_wh": canvas_wh,
            "analysis_image_shape_hw": analysis_shape,
            "content_box_xywh": list(native_manifest.get("content_box_xywh") or []),
            "injection_points": review_injection_points(native_manifest),
            "injection_anchor_contract": (
                dict(native_manifest.get("injection_anchor_contract") or {})
                if isinstance(
                    native_manifest.get("injection_anchor_contract"), Mapping
                )
                else {}
            ),
            **({"ffac_selected_policy_a_sources": native_manifest["ffac_selected_policy_a_sources"]} if "ffac_selected_policy_a_sources" in native_manifest else {}),
            "frame_sources": frame_sources,
            "v5_raw_detail_available": v5_raw_detail_available,
            "v5_original_available": original_display_available(native_manifest),
            "v5_original_reference_available": original_display_available(native_manifest, reference=True),
            "original_reference_source": dict(native_manifest.get("reference") or {}),
            "contours_are_separate_overlays": True,
            "phase_maps_are_separate_overlays": True,
            "phase_display_policy": PHASE_DISPLAY_POLICY,
            "bottom_blind_prediction_withheld": {"A": withheld_a, "B": withheld_b},
            "candidate_a": {
                "label": "Candidate A",
                "plot_url": f"/media/{pair.run}/A/plot",
                "frame_url": f"/media/{pair.run}/A/frame",
                "plain_frame_url": f"/media/{pair.run}/A/plain",
                "contour_overlay_url": f"/media/{pair.run}/A/overlay",
                "reference_frame_url": f"/media/{pair.run}/A/reference",
                "reference_version": reference_version,
                "plain_frames_ready": native_frames_ready,
                "contour_overlays_ready": native_overlays_ready["A"],
                "phase_fill_overlays_ready": native_phase_maps_ready["A"],
                "overlay_rendering": (
                    "mutually_exclusive_phase_fill_v1"
                    if native_phase_maps_ready["A"]
                    else "legacy_contour_fallback"
                ),
                "frame_count": pair.candidate_a.frame_count,
                "elapsed_hours": elapsed_a,
                "frame_metrics": ([] if pair.candidate_a.phase_model_only else
                                  metrics(records_a, pair.candidate_a.frame_count)),
                "phase_model_only": pair.candidate_a.phase_model_only,
                "frame_metric_semantics": metric_semantics(records_a),
                "reference_offsets": reference_offsets,
                "reference_frames_ready": reference_ready,
                "native_resolution": True,
                "canvas_wh": canvas_wh,
                "analysis_image_shape_hw": analysis_shape,
                "phase_quiz_ready": bool(phase_quiz_a),
                "phase_quiz_proposals": phase_quiz_a,
                "phase_quiz_initial_count": phase_quiz_payload_a[
                    "initial_proposal_count"
                ],
                "phase_quiz_available_count": len(phase_quiz_a),
            },
            "candidate_b": {
                "label": "Candidate B",
                "plot_url": f"/media/{pair.run}/B/plot",
                "frame_url": f"/media/{pair.run}/B/frame",
                "plain_frame_url": f"/media/{pair.run}/B/plain",
                "contour_overlay_url": f"/media/{pair.run}/B/overlay",
                "reference_frame_url": f"/media/{pair.run}/B/reference",
                "reference_version": reference_version,
                "plain_frames_ready": native_frames_ready,
                "contour_overlays_ready": native_overlays_ready["B"],
                "phase_fill_overlays_ready": native_phase_maps_ready["B"],
                "overlay_rendering": (
                    "mutually_exclusive_phase_fill_v1"
                    if native_phase_maps_ready["B"]
                    else "legacy_contour_fallback"
                ),
                "frame_count": pair.candidate_b.frame_count,
                "elapsed_hours": elapsed_b,
                "frame_metrics": ([] if pair.candidate_b.phase_model_only else
                                  metrics(records_b, pair.candidate_b.frame_count)),
                "phase_model_only": pair.candidate_b.phase_model_only,
                "frame_metric_semantics": metric_semantics(records_b),
                "reference_offsets": reference_offsets,
                "reference_frames_ready": reference_ready,
                "native_resolution": True,
                "canvas_wh": canvas_wh,
                "analysis_image_shape_hw": analysis_shape,
                "phase_quiz_ready": bool(phase_quiz_b),
                "phase_quiz_proposals": phase_quiz_b,
                "phase_quiz_initial_count": phase_quiz_payload_b[
                    "initial_proposal_count"
                ],
                "phase_quiz_available_count": len(phase_quiz_b),
            },
            "decision": decision,
            "browser_draft": browser_draft,
            "reveal": reveal,
        }

    def media_path(self, run: str, label: str, kind: str) -> Path:
        pair = self._pair(run)
        candidate = pair.candidate(label.upper())
        if _candidate_prediction_withheld(self.store.output_dir, candidate):
            raise PermissionError("New prediction media is withheld for the bottom blind check")
        if kind == "plot":
            return candidate.mass_plot
        if kind == "movie":
            return candidate.contour_movie
        raise KeyError(kind)

    def plot_bytes(self, run: str, label: str) -> bytes:
        pair = self._pair(run)
        candidate = pair.candidate(label.upper())
        if _candidate_prediction_withheld(self.store.output_dir, candidate):
            raise PermissionError("New prediction plots are withheld for the bottom blind check")
        return _sanitized_plot_bytes(
            str(candidate.mass_plot), pair.run, candidate.plot_title
        )

    def contour_frame_bytes(self, run: str, label: str, index: int) -> bytes:
        pair = self._pair(run)
        candidate = pair.candidate(label.upper())
        if _candidate_prediction_withheld(self.store.output_dir, candidate):
            raise PermissionError("New prediction frames are withheld for the bottom blind check")
        line_only = _line_contour_frame_path(
            self.store.output_dir,
            candidate,
            label.upper(),
            index,
        )
        if line_only.is_file():
            return line_only.read_bytes()
        return _contour_frame_bytes(
            str(candidate.contour_movie),
            str(candidate.per_frame_csv),
            pair.run,
            label.upper(),
            index,
            candidate.frame_count,
            candidate.plot_seconds,
            candidate.image_seconds,
            candidate.fps,
        )

    def plain_frame_bytes(self, run: str, label: str, index: int) -> bytes:
        pair = self._pair(run)
        normalized_label = label.upper()
        candidate = pair.candidate(normalized_label)
        if not 0 <= index < candidate.frame_count:
            raise IndexError(index)
        return _native_frame_path(self._review_root(pair), pair.run, index).read_bytes()

    def contour_overlay_bytes(
        self,
        run: str,
        label: str,
        index: int,
        phase: str = "all",
    ) -> bytes:
        pair = self._pair(run)
        normalized_label = label.upper()
        candidate = pair.candidate(normalized_label)
        if not 0 <= index < candidate.frame_count:
            raise IndexError(index)
        path, _rendering = _preferred_phase_overlay_path(
            self.store.output_dir,
            candidate,
            normalized_label,
            index,
            phase,
        )
        return path.read_bytes()

    def isolated_component_payload(
        self,
        run: str,
        label: str,
        index: int,
        phase: str = "all",
    ) -> dict[str, Any]:
        pair = self._pair(run)
        normalized_label = label.upper()
        candidate = pair.candidate(normalized_label)
        if not 0 <= index < candidate.frame_count:
            raise IndexError(index)
        path, rendering = _preferred_phase_overlay_path(
            self.store.output_dir,
            candidate,
            normalized_label,
            index,
            phase,
        )
        payload = dict(isolated_phase_components_path(path))
        payload.update(
            {
                "run": pair.run,
                "candidate": normalized_label,
                "frame_index": int(index),
                "phase": str(phase or "all"),
                "rendering": rendering,
                "contract": "isolated_from_major_support_v2",
            }
        )
        return payload

    def reference_frame_bytes(self, run: str) -> bytes:
        pair = self._pair(run)
        return _native_reference_path(self._review_root(pair), pair.run).read_bytes()

    def aligned_reference_frame_bytes(
        self,
        run: str,
        label: str,
        index: int,
    ) -> bytes:
        pair = self._pair(run)
        normalized_label = label.upper()
        candidate = pair.candidate(normalized_label)
        if not 0 <= index < candidate.frame_count:
            raise IndexError(index)
        return _native_reference_path(self._review_root(pair), pair.run).read_bytes()


def _send_json(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _send_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    size = path.stat().st_size
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    try:
        byte_range = parse_byte_range(handler.headers.get("Range"), size)
    except ValueError:
        handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        handler.send_header("Content-Range", f"bytes */{size}")
        handler.end_headers()
        return

    if byte_range is None:
        start, end = 0, size - 1
        status = HTTPStatus.OK
    else:
        start, end = byte_range
        status = HTTPStatus.PARTIAL_CONTENT
    length = end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(length))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "private, max-age=3600")
    if status == HTTPStatus.PARTIAL_CONTENT:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = length
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                break
            handler.wfile.write(block)
            remaining -= len(block)


def _send_bytes(
    handler: BaseHTTPRequestHandler,
    body: bytes,
    content_type: str,
) -> None:
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "private, max-age=3600")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FFACPairwiseReview/1"

        def log_message(self, format_string: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    body = _render_review_html().encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/state":
                    reviewer = (parse_qs(parsed.query).get("reviewer") or [""])[0]
                    _send_json(self, app.state(reviewer))
                    return
                if parsed.path == "/api/pair":
                    query = parse_qs(parsed.query)
                    run = (query.get("run") or [""])[0]
                    reviewer = (query.get("reviewer") or [""])[0]
                    _send_json(self, app.pair_payload(run, reviewer))
                    return
                if parsed.path == "/api/isolated-components":
                    query = parse_qs(parsed.query)
                    run = (query.get("run") or [""])[0]
                    label = (query.get("candidate") or ["A"])[0]
                    index = int((query.get("index") or ["0"])[0])
                    phase = str((query.get("phase") or ["all"])[0])
                    _send_json(
                        self,
                        app.isolated_component_payload(
                            run,
                            label,
                            index,
                            phase,
                        ),
                    )
                    return
                reference_match = re.fullmatch(
                    r"/media/((?:ac|run)\d+)/reference", parsed.path
                )
                if reference_match:
                    _send_bytes(
                        self,
                        app.reference_frame_bytes(reference_match.group(1)),
                        "image/jpeg",
                    )
                    return
                match = re.fullmatch(
                    r"/media/((?:ac|run)\d+)/(A|B)/(plot|movie|frame|plain|overlay|reference)",
                    parsed.path,
                )
                if match:
                    run, label, kind = match.groups()
                    if kind == "plot":
                        _send_bytes(self, app.plot_bytes(run, label), "image/png")
                    elif kind == "frame":
                        index = int((parse_qs(parsed.query).get("index") or ["0"])[0])
                        _send_bytes(
                            self,
                            app.contour_frame_bytes(run, label, index),
                            "image/jpeg",
                        )
                    elif kind == "plain":
                        index = int((parse_qs(parsed.query).get("index") or ["0"])[0])
                        _send_bytes(
                            self,
                            app.plain_frame_bytes(run, label, index),
                            "image/jpeg",
                        )
                    elif kind == "overlay":
                        query = parse_qs(parsed.query)
                        index = int((query.get("index") or ["0"])[0])
                        phase = str((query.get("phase") or ["all"])[0])
                        _send_bytes(
                            self,
                            app.contour_overlay_bytes(run, label, index, phase),
                            "image/png",
                        )
                    elif kind == "reference":
                        index = int((parse_qs(parsed.query).get("index") or ["0"])[0])
                        _send_bytes(
                            self,
                            app.aligned_reference_frame_bytes(run, label, index),
                            "image/jpeg",
                        )
                    else:
                        _send_file(self, app.media_path(run, label, kind))
                    return
                _send_json(self, {"error": "not_found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, FileNotFoundError) as exc:
                _send_json(self, {"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return
            except Exception as exc:  # pragma: no cover - HTTP safety net
                try:
                    _send_json(
                        self,
                        {"error": f"{type(exc).__name__}: {exc}"},
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    return

        def do_POST(self) -> None:  # noqa: N802
            endpoint = urlparse(self.path).path
            if endpoint not in {"/api/decision", "/api/draft"}:
                _send_json(self, {"error": "not_found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > 24 * 1024 * 1024:
                    raise ValueError("Review request exceeds 24 MiB")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if endpoint == "/api/draft":
                    draft = app.store.save_browser_draft(request)
                    _send_json(
                        self,
                        {
                            "status": "deleted" if draft.get("deleted") else "saved",
                            "draft": draft,
                            "state": app.state(str(request.get("reviewer") or "")),
                        },
                    )
                    return
                record = app.store.save(request)
                _send_json(
                    self,
                    {
                        "status": "saved",
                        "record": record,
                        "state": app.state(record["reviewer"]),
                        "pair": app.pair_payload(record["run"], record["reviewer"]),
                    },
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                _send_json(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FluidFlower Calibration Review</title>
<style>
:root { --ink:#202426; --muted:#687177; --line:#cfd5d8; --panel:#f6f7f7; --a:#087f73; --b:#b55d00; --ok:#2f7447; --warn:#a34927; --annotation-line-width:1px; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:#fff; font:14px/1.4 "Segoe UI",Arial,sans-serif; letter-spacing:0; }
[hidden] { display:none !important; }
button,input,textarea,select { font:inherit; letter-spacing:0; }
header { height:58px; display:flex; align-items:center; gap:24px; padding:0 18px; border-bottom:1px solid var(--line); background:#202426; color:#fff; }
header h1 { margin:0; font-size:18px; font-weight:650; }
.progress { margin-left:auto; color:#dce2e4; font-variant-numeric:tabular-nums; }
.shell { display:grid; grid-template-columns:220px minmax(0,1fr); min-height:calc(100vh - 58px); }
aside { border-right:1px solid var(--line); background:#f4f5f5; padding:12px; overflow:auto; max-height:calc(100vh - 58px); }
.reviewer-row { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:6px; margin-bottom:8px; }
.reviewer { width:100%; min-width:0; padding:8px 9px; border:1px solid #aeb6ba; background:#fff; }
.session-controls { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:8px; }
.session-controls .command { padding:6px 7px; font-size:12px; }
.session-meta { margin:0 0 12px; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }
.run-list { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
.run-button { min-height:34px; border:1px solid #c3c9cc; background:#fff; cursor:pointer; padding:5px 3px; }
.run-button.active { color:#fff; background:#31383b; border-color:#31383b; }
.run-button.reviewed:not(.active) { color:var(--ok); border-color:#8ab39a; background:#f1f8f3; }
.run-button.drafted:not(.reviewed):not(.active) { color:#765500; border-color:#d4b45c; background:#fff9e8; }
.run-button.quiz-complete:not(.active) { box-shadow:inset 0 -3px #2c6541; }
.review-locked { opacity:.48; pointer-events:none; user-select:none; }
.unpaired { margin-top:15px; padding-top:12px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }
main { min-width:0; padding:14px 16px 24px; }
.toolbar { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
.toolbar h2 { margin:0 auto 0 0; font-size:18px; }
.comparison-mode { color:#6f4d00; background:#fff5d6; border:1px solid #d7bb68; padding:5px 8px; font-size:12px; }
.physics-warning { color:#7a2017; background:#fff0ed; border:1px solid #d48a80; padding:5px 8px; font-size:12px; max-width:520px; }
.command { border:1px solid #9fa8ac; background:#fff; padding:7px 12px; cursor:pointer; }
.command:disabled { opacity:.4; cursor:default; }
.hold-swap.active,.hold-contour-swap.active { color:#fff; background:#31383b; border-color:#31383b; }
.comparison { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; }
.comparison.shared-phase-quiz { grid-template-columns:minmax(0,1fr); }
.comparison.shared-phase-quiz .candidate.b { display:none; }
.candidate { border:1px solid var(--line); min-width:0; background:#fff; }
.candidate.a { order:0; border-top:4px solid var(--a); }
.candidate.b { order:1; border-top:4px solid var(--b); }
.comparison.swap-held .candidate.a { order:1; }
.comparison.swap-held .candidate.b { order:0; }
.candidate h3 { margin:0; padding:8px 10px; font-size:14px; border-bottom:1px solid var(--line); }
.asset-label { padding:7px 10px 4px; color:var(--muted); font-size:12px; font-weight:600; text-transform:uppercase; }
.plot-shell { position:relative; width:100%; aspect-ratio:16/9; background:#fff; overflow:hidden; }
.plot { position:absolute; inset:0; display:block; width:100%; height:100%; object-fit:contain; background:#fff; pointer-events:none; }
.plot-navigation { position:absolute; inset:0; cursor:crosshair; }
.plot-point { position:absolute; width:16px; height:16px; padding:0; transform:translate(-50%,-50%); border:1px solid transparent; border-radius:50%; background:transparent; cursor:pointer; }
.plot-point:hover, .plot-point:focus-visible { border:2px solid #1677b8; background:rgba(22,119,184,.14); outline:none; }
.plot-point.active { border:2px solid #d13d32; background:rgba(209,61,50,.12); }
.plot-guide { position:absolute; width:1px; transform:translateX(-.5px); background:rgba(209,61,50,.72); pointer-events:none; }
.frame-shell { position:relative; width:100%; aspect-ratio:16/9; overflow:hidden; background:#111; touch-action:none; cursor:grab; }
.frame-shell.drawing { cursor:crosshair; }
body.space-pan .frame-shell { cursor:grab; }
.frame-shell.dragging { cursor:grabbing; }
.frame-layer { position:absolute; top:0; left:0; width:100%; height:100%; transform-origin:0 0; will-change:transform; backface-visibility:hidden; contain:layout paint style; }
.movie { position:absolute; inset:0; display:block; width:100%; height:100%; object-fit:contain; user-select:none; -webkit-user-drag:none; }
.phase-overlay { position:absolute; inset:0; display:block; width:100%; height:100%; object-fit:contain; pointer-events:none; user-select:none; -webkit-user-drag:none; }
.frame-layer.phase-map-only { background:#000; }
.frame-layer.phase-map-only .movie { visibility:hidden; }
.isolated-marker-layer { position:absolute; inset:0; width:100%; height:100%; overflow:visible; pointer-events:none; }
.isolated-marker { fill:none; stroke:#fff; stroke-width:1.5px; vector-effect:non-scaling-stroke; }
.frame-layer.pixel-inspection .movie,.frame-layer.pixel-inspection .phase-overlay { image-rendering:pixelated; }
.injection-point { fill:#e82727; stroke:#fff; stroke-width:2px; vector-effect:non-scaling-stroke; pointer-events:none; }
.contour-source-badge { position:absolute; z-index:3; top:8px; left:8px; padding:5px 8px; border:2px solid #fff; color:#fff; background:rgba(22,25,27,.88); font-size:12px; font-weight:650; pointer-events:none; }
.contour-source-badge[data-source="A"] { border-color:var(--a); }
.contour-source-badge[data-source="B"] { border-color:var(--b); }
.annotation-layer { position:absolute; inset:0; width:100%; height:100%; overflow:visible; fill:none; background:transparent; }
.annotation-hit { fill:transparent; stroke:transparent; pointer-events:all; cursor:inherit; }
.annotation-box { fill:none; stroke:#ff4f45; stroke-width:var(--annotation-line-width); pointer-events:none; }
.annotation-box.minor { stroke:#ffd166; }
.annotation-box.critical { stroke:#ff2419; }
.annotation-box.confirmed-aqueous-co2 { stroke:#24d0e8; }
.annotation-box.confirmed-free-gas { stroke:#ff9f2e; }
.annotation-box.confirmed-no-co2 { stroke:#85e08d; }
.annotation-box.unsupported-free-gas-classification { stroke:#e85d75; }
.annotation-box.likely-dense-aqueous-co2-misclassified-as-gas { stroke:#d966a7; }
.annotation-box.implausible-post-injection-downward-gas-migration { stroke:#ff384c; }
.annotation-box.plausible-dense-aqueous-co2-descent { stroke:#2db9cc; }
.annotation-box.incomplete-wash-indicator-carryover { stroke:#ffb000; }
.annotation-box.selected { stroke:#fff; stroke-width:var(--annotation-line-width); stroke-dasharray:6 4; }
.annotation-box.draft { stroke:#fff; stroke-width:var(--annotation-line-width); stroke-dasharray:6 4; fill:none !important; pointer-events:none; }
.image-quick-controls { display:flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:8px; margin:5px 0 4px; padding:5px 8px; border:1px solid var(--line); background:var(--panel); }
.image-quick-controls select { height:30px; border:1px solid #aeb6ba; background:#fff; padding:0 7px; }
.phase-fill-legend { display:flex; justify-content:center; flex-wrap:wrap; gap:12px; margin:0 0 6px; color:var(--muted); font-size:11px; }
.phase-fill-legend span { display:inline-flex; align-items:center; gap:5px; }
.phase-fill-swatch { width:12px; height:12px; border:1px solid rgba(0,0,0,.24); }
.phase-fill-swatch.aqueous { background:rgb(49,217,160); }
.phase-fill-swatch.gas { background:rgb(242,76,157); }
.inspection-tools { display:flex; align-items:center; flex-wrap:wrap; gap:8px 14px; margin-top:12px; padding:9px 10px; border:1px solid var(--line); background:var(--panel); }
.tool-group { display:flex; align-items:center; flex-wrap:wrap; gap:6px; padding-right:12px; border-right:1px solid var(--line); }
.tool-group:last-child { border-right:0; padding-right:0; }
.icon-command { width:35px; height:34px; border:1px solid #9fa8ac; background:#fff; cursor:pointer; font-size:17px; }
.toggle { display:flex; align-items:center; gap:6px; white-space:nowrap; }
.line-width-control { display:flex; align-items:center; gap:6px; white-space:nowrap; color:var(--muted); font-size:12px; }
.line-width-control input { width:86px; }
.line-width-control output { min-width:46px; color:var(--ink); font-variant-numeric:tabular-nums; }
.zoom-readout { min-width:135px; color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
.inspection-tools select { height:34px; border:1px solid #aeb6ba; background:#fff; padding:0 7px; }
.mode.active { color:#fff; background:#31383b; border-color:#31383b; }
.mark.active { background:#8d2c25; border-color:#8d2c25; }
.annotation-summary { min-width:150px; color:var(--muted); font-variant-numeric:tabular-nums; }
.annotation-category-help { width:100%; padding:7px 9px; border-left:3px solid #657177; background:#fff; color:#3f484c; font-size:12px; }
.annotation-editor { display:none; grid-template-columns:minmax(230px,1.5fr) repeat(4,minmax(125px,.7fr)) minmax(220px,1.5fr) auto; align-items:center; gap:7px; width:100%; padding-top:8px; border-top:1px solid var(--line); }
.annotation-editor.visible { display:grid; }
.annotation-editor input,.annotation-editor select { width:100%; min-width:0; padding:7px 8px; border:1px solid #aeb6ba; background:#fff; }
.annotation-browser { margin-top:8px; border:1px solid var(--line); background:#fff; }
.annotation-browser h3 { margin:0; padding:7px 9px; border-bottom:1px solid var(--line); font-size:13px; }
.annotation-list { display:grid; grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); gap:6px; max-height:178px; overflow:auto; padding:7px; }
.annotation-item { display:grid; grid-template-columns:1fr auto; gap:5px; align-items:center; border:1px solid #d5dade; background:#fafbfb; padding:6px 7px; text-align:left; cursor:pointer; }
.annotation-item.selected { border-color:#31383b; box-shadow:inset 3px 0 #31383b; background:#f0f3f4; }
.annotation-item-main { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.annotation-item-meta { color:var(--muted); font-size:11px; }
.annotation-delete { width:28px; height:28px; border:1px solid #b7bec1; background:#fff; cursor:pointer; font-size:17px; line-height:1; }
.video-tools { display:grid; grid-template-columns:auto minmax(180px,1fr) auto minmax(260px,.8fr); align-items:center; gap:10px; margin:8px 0 12px; padding:10px; border:1px solid var(--line); background:var(--panel); }
.video-tools input[type=range] { width:100%; }
.timeline-wrap { position:relative; width:100%; padding-top:9px; }
.timeline-wrap input[type=range] { display:block; margin:0; }
.phase-quiz-timeline-marker { position:absolute; top:0; width:10px; height:10px; padding:0; border:2px solid #fff; border-radius:50%; background:#d1261c; box-shadow:0 0 0 1px #8b1711; transform:translateX(-50%); cursor:pointer; z-index:2; }
.phase-quiz-timeline-marker.current { box-shadow:0 0 0 2px #8b1711; }
.frame-readout { display:flex; align-items:baseline; gap:12px; min-width:220px; font-variant-numeric:tabular-nums; white-space:nowrap; }
.frame-readout strong { font-size:15px; }
.frame-elapsed { color:var(--ink); font-size:15px; font-weight:650; }
.time { min-width:0; color:var(--muted); font-size:11px; line-height:1.3; text-align:right; overflow-wrap:anywhere; }
.phase-quiz { margin:0 0 12px; border:1px solid var(--line); border-left:4px solid #4d5960; background:#fff; }
.phase-quiz.complete { border-left-color:var(--ok); }
.phase-quiz-header { display:flex; align-items:center; gap:12px; padding:9px 11px; border-bottom:1px solid var(--line); background:var(--panel); }
.phase-quiz-header h3 { margin:0; font-size:14px; }
.phase-quiz-progress { margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }
.phase-quiz-body { display:grid; gap:9px; padding:10px 11px; }
.phase-quiz-context { color:var(--muted); font-size:12px; }
.phase-quiz-scope { display:flex; align-items:center; flex-wrap:wrap; gap:8px; padding:7px 9px; border:1px solid #d5dade; background:#fafbfb; }
.phase-quiz-scope label { font-weight:650; }
.phase-quiz-add-count { width:88px; height:34px; box-sizing:border-box; border:1px solid #aeb6ba; background:#fff; padding:0 8px; font:inherit; font-variant-numeric:tabular-nums; }
.phase-quiz-add-count.invalid { border-color:var(--warn); outline:1px solid var(--warn); }
.phase-quiz-scope button { min-height:34px; }
.phase-quiz-scope span { color:var(--muted); font-size:12px; }
.phase-quiz-actions,.phase-quiz-options,.phase-quiz-meta { display:flex; align-items:center; flex-wrap:wrap; gap:7px; }
.phase-quiz-actions .command.active { color:#fff; background:#5f676b; border-color:#5f676b; }
.phase-quiz-options button { min-height:36px; border:1px solid #aeb6ba; background:#fff; padding:7px 10px; cursor:pointer; }
.phase-quiz-options button.selected { color:#fff; background:#31383b; border-color:#31383b; }
.phase-quiz-meta select,.phase-quiz-meta input { height:34px; border:1px solid #aeb6ba; background:#fff; padding:0 8px; }
.phase-quiz-meta input { min-width:260px; flex:1; }
.phase-quiz-reveal { min-height:20px; color:#374247; font-size:12px; }
.phase-quiz-reveal.agree { color:var(--ok); }
.phase-quiz-reveal.disagree { color:var(--warn); font-weight:650; }
.phase-quiz-complete { padding:8px 10px; border:1px solid #8ab39a; background:#f1f8f3; color:#245b39; font-weight:650; }
.phase-quiz-answer-browser { border-top:1px solid var(--line); padding-top:9px; }
.phase-quiz-answer-browser h4 { margin:0 0 6px; font-size:12px; }
.phase-quiz-answer-list { display:grid; grid-template-columns:repeat(auto-fill,minmax(285px,1fr)); gap:6px; max-height:190px; overflow:auto; }
.phase-quiz-answer-item { display:grid; gap:2px; border:1px solid #d5dade; background:#fafbfb; padding:7px 8px; text-align:left; cursor:pointer; }
.phase-quiz-answer-item:hover,.phase-quiz-answer-item:focus-visible { border-color:#727d82; outline:none; }
.phase-quiz-answer-item.active { border-color:#31383b; box-shadow:inset 3px 0 #31383b; background:#f0f3f4; }
.phase-quiz-answer-main { font-size:12px; font-weight:650; }
.phase-quiz-answer-meta { color:var(--muted); font-size:11px; }
.phase-quiz-region { fill:none; stroke:#fff; stroke-width:1.25px; stroke-dasharray:7 4; filter:drop-shadow(0 0 1px #111); pointer-events:none; }
.phase-quiz-region.answered { stroke:#ffe066; }
.phase-quiz-region.temporal { stroke-dasharray:2 4; opacity:.88; }
.decision { margin-top:12px; border-top:1px solid var(--line); padding-top:12px; }
.frame-verdict-heading { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:7px; }
.frame-verdict-progress { color:var(--muted); font-size:12px; }
.verdict { display:grid; grid-template-columns:repeat(4,minmax(110px,1fr)); gap:8px; }
.verdict button { min-height:42px; border:1px solid #aeb6ba; background:#fff; cursor:pointer; }
.verdict button.selected { color:#fff; background:#31383b; border-color:#31383b; }
.notes { width:100%; min-height:64px; margin-top:10px; padding:8px; border:1px solid #aeb6ba; resize:vertical; }
.save-row { display:flex; align-items:center; gap:10px; margin-top:8px; }
.save { color:#fff; background:#2c6541; border:1px solid #2c6541; padding:9px 17px; cursor:pointer; }
.save-status { display:flex; align-items:center; min-width:0; gap:7px; color:var(--muted); }
.save-indicator { width:12px; height:12px; flex:0 0 12px; border-radius:50%; background:#8b9499; box-shadow:0 0 0 1px rgba(0,0,0,.18); }
.save-indicator.pending { background:#d6a53a; }
.save-indicator.syncing { background:#3286bd; }
.save-indicator.saved { background:#3a8b55; }
.save-indicator.error { background:#bd443b; }
.status { min-width:0; overflow-wrap:anywhere; }
.reveal { margin-top:10px; padding:10px; border:1px solid #b8c0c3; background:#f7f8f8; display:none; white-space:pre-wrap; font:12px/1.45 Consolas,monospace; }
@media (max-width:1400px) { .annotation-editor{grid-template-columns:repeat(2,minmax(0,1fr))} }
@media (max-width:1000px) { .shell{grid-template-columns:1fr} aside{max-height:none;border-right:0;border-bottom:1px solid var(--line)} .run-list{grid-template-columns:repeat(8,1fr)} .comparison{grid-template-columns:1fr} .tool-group{width:100%;flex-wrap:wrap;border-right:0;padding-right:0}.inspection-tools select{max-width:100%}.annotation-summary{min-width:0}.video-tools{grid-template-columns:auto minmax(120px,1fr) auto}.time{grid-column:1/-1;text-align:left}.frame-readout{min-width:0} }
@media (max-width:620px) { .video-tools{grid-template-columns:auto 1fr}.frame-readout{grid-column:1/-1;justify-content:space-between}.time{grid-column:1/-1} }
</style>
</head>
<body>
<header><h1>FluidFlower Calibration Review</h1><div class="progress" id="progress">0 / 0 reviewed</div></header>
<div class="shell">
<aside>
  <div class="reviewer-row">
    <input id="reviewer" class="reviewer" placeholder="Reviewer name">
    <button id="openReviewer" class="command" type="button">Open</button>
  </div>
  <div id="sessionControls" class="session-controls portable-only" hidden>
    <button id="openReview" class="command" type="button">Open review</button>
    <button id="exportReview" class="command" type="button">Export review</button>
    <button id="chooseAutosaveFile" class="command" type="button">Choose autosave file</button>
    <button id="newReview" class="command" type="button">New review</button>
  </div>
  <input id="reviewFile" type="file" accept="application/json,.json" hidden>
  <div id="sessionMeta" class="session-meta portable-only" hidden></div>
  <div id="runs" class="run-list"></div>
  <div id="unpaired" class="unpaired"></div>
</aside>
<main>
  <div class="toolbar">
    <h2 id="runTitle">Select a run</h2>
    <span id="comparisonMode" class="comparison-mode" hidden></span>
    <span id="physicsWarning" class="physics-warning" hidden></span>
    <button id="holdSwap" class="command hold-swap" type="button" aria-pressed="false" title="Hold to temporarily swap Candidate A and Candidate B">Hold to swap A / B</button>
    <button id="previous" class="command" type="button">Previous</button>
    <button id="next" class="command" type="button">Next</button>
  </div>
  <div id="comparison" class="comparison">
    <section id="candidatePanelA" class="candidate a">
      <h3 id="candidateTitleA">Candidate A</h3>
      <div class="asset-label">Candidate diagnostic</div><div id="plotShellA" class="plot-shell" data-side="A" title="Click a calibration point to open its image"><img id="plotA" class="plot" alt="Candidate A diagnostic"><div id="plotNavigationA" class="plot-navigation" aria-label="Candidate A clickable calibration points"></div></div>
      <div id="assetLabelA" class="asset-label">Calibration image</div>
      <div id="viewportA" class="frame-shell" data-side="A"><div id="layerA" class="frame-layer"><img id="frameA" class="movie" draggable="false" alt="Candidate A calibration frame"><img id="overlayA" class="phase-overlay" draggable="false" alt=""><svg id="isolatedMarkersA" class="isolated-marker-layer" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Small disconnected phase-map islands"></svg><svg id="annotationsA" class="annotation-layer" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Candidate A annotations"></svg></div><span id="contourSourceA" class="contour-source-badge" hidden></span></div>
    </section>
    <section id="candidatePanelB" class="candidate b">
      <h3 id="candidateTitleB">Candidate B</h3>
      <div class="asset-label">Candidate diagnostic</div><div id="plotShellB" class="plot-shell" data-side="B" title="Click a calibration point to open its image"><img id="plotB" class="plot" alt="Candidate B diagnostic"><div id="plotNavigationB" class="plot-navigation" aria-label="Candidate B clickable calibration points"></div></div>
      <div id="assetLabelB" class="asset-label">Calibration image</div>
      <div id="viewportB" class="frame-shell" data-side="B"><div id="layerB" class="frame-layer"><img id="frameB" class="movie" draggable="false" alt="Candidate B calibration frame"><img id="overlayB" class="phase-overlay" draggable="false" alt=""><svg id="isolatedMarkersB" class="isolated-marker-layer" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Small disconnected phase-map islands"></svg><svg id="annotationsB" class="annotation-layer" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Candidate B annotations"></svg></div><span id="contourSourceB" class="contour-source-badge" hidden></span></div>
    </section>
  </div>
  <div class="image-quick-controls" aria-label="Image overlay controls">
    <label class="toggle"><input id="contours" type="checkbox" checked> Phase map</label>
    <label class="toggle"><input id="phaseMapOnly" type="checkbox"> Black background</label>
    <label class="toggle"><input id="highlightSmall" type="checkbox" checked disabled> Mark isolated specks</label>
    <label class="toggle">Displayed phase
      <select id="contourMode" aria-label="Displayed phase">
        <option value="all">All predicted phases</option>
        <option value="gas">Free CO2 only</option>
        <option value="aqueous">Aqueous CO2 only</option>
      </select>
    </label>
    <label class="toggle"><input id="injectionPoints" type="checkbox"> Injection points</label>
    <button id="holdContourSwap" class="command hold-contour-swap" type="button" aria-pressed="false" title="Hold to show each candidate with the other candidate's phase map">Hold to swap phase maps</button>
  </div>
  <div class="phase-fill-legend" aria-label="Phase-map legend">
    <span><i class="phase-fill-swatch aqueous"></i>Aqueous-dominant</span>
    <span><i class="phase-fill-swatch gas"></i>Free-gas-dominant</span>
    <span>Cyan tint, when present = unresolved source support; transparent = no visible CO2 predicted on supported pixels</span>
    <span>White markers = small disconnected islands only; display aid, not predicted area</span>
  </div>
  <div class="video-tools" aria-label="Calibration frame playback">
    <button id="play" class="command" type="button">Play</button>
    <div class="timeline-wrap">
      <input id="scrub" type="range" min="0" max="1" step="1" value="0" aria-label="Calibration frame timeline">
      <button id="phaseQuizTimelineMarker" class="phase-quiz-timeline-marker" type="button" aria-label="Open selected focus-region source frame" title="Selected focus-region source frame" hidden></button>
    </div>
    <div class="frame-readout" aria-live="polite">
      <strong id="frameNumber">Frame 0 / 0</strong>
      <span id="frameElapsed" class="frame-elapsed">t = -- h</span>
    </div>
    <div id="time" class="time">Select an experiment to inspect frame details.</div>
  </div>
  <section id="phaseQuiz" class="phase-quiz" aria-labelledby="phaseQuizTitle">
    <div class="phase-quiz-header">
      <h3 id="phaseQuizTitle">Visible 2D phase quiz</h3>
      <span id="phaseQuizProgress" class="phase-quiz-progress">No regions prepared</span>
    </div>
    <div class="phase-quiz-body">
      <div id="phaseQuizContext" class="phase-quiz-context">Classify only what is visibly supported inside the outlined region in this camera-facing 2D slice. "Mostly or entirely" spans a clear phase majority through a visually pure phase; use a mixture only when both phases are materially visible and neither clearly dominates. The model suggestion stays hidden until you answer.</div>
      <div class="phase-quiz-scope">
        <label for="phaseQuizAddCount">Number to add</label>
        <input id="phaseQuizAddCount" class="phase-quiz-add-count" type="text" inputmode="numeric" value="8" autocomplete="off" aria-label="Number of quiz regions to add">
        <button id="phaseQuizNewRound" class="command" type="button">Add strategic regions: 8</button>
        <span id="phaseQuizAvailability">No reserve prepared</span>
      </div>
      <div class="phase-quiz-actions">
        <button id="phaseQuizPrevious" class="command" type="button">Previous region</button>
        <button id="phaseQuizFocus" class="command" type="button">Focus region</button>
        <button id="phaseQuizClose" class="command" type="button">Close active region</button>
        <button id="phaseQuizNext" class="command" type="button">Next region</button>
        <button id="phaseQuizUnanswered" class="command" type="button">Next unanswered</button>
        <button id="phaseQuizDraw" class="command" type="button">Draw detail</button>
        <button id="phaseQuizIgnore" class="command" type="button">Ignore region</button>
        <button id="phaseQuizClear" class="command" type="button">Clear answer</button>
      </div>
      <div id="phaseQuizOptions" class="phase-quiz-options" aria-label="Visible phase classification">
        <button type="button" data-phase-choice="water_or_no_visible_co2">Water / no visible CO2</button>
        <button type="button" data-phase-choice="only_aqueous_co2" title="The visible CO2 response is aqueous, with no visible free-gas phase or untouched-water component">Only aqueous CO2</button>
        <button type="button" data-phase-choice="predominantly_aqueous_co2" title="Both CO2 phases are visible, with aqueous CO2 clearly dominant">Mostly aqueous CO2, some free CO2</button>
        <button type="button" data-phase-choice="mixed_free_and_aqueous_co2">Mixture: free + aqueous CO2</button>
        <button type="button" data-phase-choice="predominantly_free_gas" title="Both CO2 phases are visible, with free CO2 clearly dominant">Mostly free CO2, some aqueous CO2</button>
        <button type="button" data-phase-choice="free_gas_only">Only free CO2</button>
        <button type="button" data-phase-choice="mixed_aqueous_co2_and_water">Mixture: aqueous CO2 + untouched water</button>
        <button type="button" data-phase-choice="mixed_free_aqueous_co2_and_water">Mixture: free + aqueous CO2 + untouched water</button>
        <button type="button" data-phase-choice="co2_present_phase_uncertain">CO2 visible; phase split uncertain</button>
        <button type="button" data-phase-choice="not_assessable">Uncertain / cannot assess</button>
        <button type="button" data-phase-choice="phase_signal_artifact">Artifact / not a valid phase signal</button>
      </div>
      <div class="phase-quiz-meta">
        <label for="phaseQuizConfidence">Confidence</label>
        <select id="phaseQuizConfidence"><option value="high">High</option><option value="medium" selected>Medium</option><option value="low">Low</option></select>
        <input id="phaseQuizComment" placeholder="Optional note about this region">
      </div>
      <div id="phaseQuizReveal" class="phase-quiz-reveal" aria-live="polite"></div>
      <div id="phaseQuizComplete" class="phase-quiz-complete" hidden></div>
      <div class="phase-quiz-answer-browser">
        <h4>Validated regions</h4>
        <div id="phaseQuizAnswerList" class="phase-quiz-answer-list"></div>
      </div>
    </div>
  </section>
  <div class="inspection-tools">
    <div class="tool-group">
      <label class="toggle"><input id="reference" type="checkbox"> Pre-injection reference</label>
      <button id="zoomOut" class="icon-command" type="button" title="Zoom out" aria-label="Zoom out">−</button>
      <button id="zoomReset" class="command" type="button" title="Fit the complete image" aria-label="Fit the complete image">Fit</button>
      <button id="zoomNative" class="command" type="button" title="Show one image pixel per screen pixel" aria-label="Show one image pixel per screen pixel">1 px</button>
      <button id="zoomPixel" class="command" type="button" title="Show one image pixel as approximately eight screen pixels" aria-label="Magnify individual image pixels">8 px</button>
      <button id="zoomIn" class="icon-command" type="button" title="Zoom in" aria-label="Zoom in">+</button>
      <output id="zoomReadout" class="zoom-readout" aria-live="polite">Fit</output>
      <label class="toggle"><input id="linkViews" type="checkbox" checked> Link views</label>
      <button id="panView" class="command mode active" type="button" title="Pan (or hold Space while dragging)">Pan</button>
    </div>
    <div class="tool-group">
      <select id="annotationCategory" aria-label="Visible phase or physical issue category">
        <optgroup label="Confirmed evidence">
          <option value="confirmed_only_aqueous_co2">Only aqueous CO2</option>
          <option value="confirmed_predominantly_aqueous_co2">Mostly aqueous CO2, some free CO2</option>
          <option value="confirmed_mixed_free_aqueous_co2">Mixture: free + aqueous CO2</option>
          <option value="confirmed_predominantly_free_gas">Mostly free CO2, some aqueous CO2</option>
          <option value="confirmed_only_free_gas">Only free CO2</option>
          <option value="confirmed_mixed_aqueous_water">Mixture: aqueous CO2 + untouched water</option>
          <option value="confirmed_mixed_free_aqueous_water">Mixture: free + aqueous CO2 + untouched water</option>
          <option value="confirmed_no_co2">Confirmed no CO2</option>
          <option value="confirmed_no_free_gas_in_selected_area">No free gas anywhere in selected area</option>
          <option value="confirmed_no_aqueous_co2_in_selected_area">No aqueous CO2 anywhere in selected area</option>
          <option value="phase_not_assessable">Uncertain / cannot assess</option>
          <option value="phase_signal_artifact">Artifact / not a valid phase signal</option>
        </optgroup>
        <optgroup label="Phase interpretation">
          <option value="unsupported_free_gas_classification">Incorrect free-gas prediction in this model</option>
          <option value="unsupported_aqueous_co2_classification">Incorrect aqueous-CO2 prediction in this model</option>
          <option value="likely_dense_aqueous_co2_misclassified_as_gas">Likely dense aqueous CO2 misclassified as gas</option>
          <option value="phase_ambiguous_strong_color_response">Strong colour response; phase ambiguous</option>
        </optgroup>
        <optgroup label="Temporal physics">
          <option value="implausible_post_injection_downward_gas_migration">Implausible downward gas migration after injection</option>
          <option value="plausible_injection_forced_downward_gas">Plausible injection-forced downward gas</option>
          <option value="plausible_dense_aqueous_co2_descent">Plausible downward motion of dense aqueous CO2</option>
          <option value="implausible_late_free_gas_persistence">Implausible late free-gas persistence</option>
          <option value="implausible_temporal_evolution">Other implausible temporal evolution</option>
        </optgroup>
        <optgroup label="Location, detection, and artifacts">
          <option value="implausible_gas_location">Implausible gas location</option>
          <option value="implausible_aqueous_location">Implausible aqueous location</option>
          <option value="missed_visible_co2">Missed visible CO2</option>
          <option value="missed_visible_gas">Missed visible gas</option>
          <option value="disconnected_noise">Disconnected noise</option>
          <option value="facies_edge_artifact">Facies-edge artifact</option>
          <option value="incomplete_wash_indicator_carryover">Indicator carryover from incomplete washing</option>
          <option value="baseline_or_flush_artifact">Baseline or incomplete-flush artifact</option>
          <option value="lighting_or_color_correction_artifact">Lighting or colour-correction artifact</option>
          <option value="other">Other</option>
        </optgroup>
      </select>
      <select id="annotationPhase" aria-label="Affected phase">
        <option value="gas">Gas</option><option value="aqueous" selected>Aqueous</option><option value="total">Total CO2</option><option value="unspecified">Unspecified</option>
      </select>
      <select id="annotationSeverity" aria-label="Issue severity">
        <option value="major">Major</option><option value="minor">Minor</option><option value="critical">Critical</option>
      </select>
      <select id="annotationConfidence" aria-label="Annotation confidence">
        <option value="high">High confidence</option><option value="medium">Medium confidence</option><option value="low">Low confidence</option>
      </select>
      <button id="markBox" class="command mode mark" type="button">Box</button>
      <button id="markFreehand" class="command mode mark" type="button">Freehand</button>
      <button id="markPixel" class="command mode mark" type="button" title="Mark one exact analysis-image pixel; zoom to at least 1 px first">Pixel</button>
      <label class="line-width-control" for="annotationLineWidth">Marker line
        <input id="annotationLineWidth" type="range" min="0.5" max="2" step="0.25" value="1" aria-label="Annotation line width">
        <output id="annotationLineWidthValue" for="annotationLineWidth">1.00 px</output>
      </label>
      <button id="undoBox" class="command" type="button">Undo latest annotation</button>
      <span id="annotationSummary" class="annotation-summary">0 annotations</span>
    </div>
    <div id="annotationCategoryHelp" class="annotation-category-help" aria-live="polite"></div>
    <div id="annotationEditor" class="annotation-editor">
      <strong id="annotationLabel">Selected annotation</strong>
      <select id="selectedAnnotationCategory" aria-label="Selected annotation category"></select>
      <select id="selectedAnnotationPhase" aria-label="Selected annotation phase"></select>
      <select id="selectedAnnotationSeverity" aria-label="Selected annotation severity"></select>
      <select id="selectedAnnotationConfidence" aria-label="Selected annotation confidence"></select>
      <input id="annotationComment" placeholder="Optional annotation comment">
      <button id="deleteBox" class="command" type="button">Delete selected</button>
    </div>
  </div>
  <section class="annotation-browser">
    <h3>Annotations in this experiment</h3>
    <div id="annotationList" class="annotation-list"></div>
  </section>
  <section class="decision">
    <div class="frame-verdict-heading"><strong>Current-frame verdict</strong><span id="frameVerdictProgress" class="frame-verdict-progress">0 / 0 frames reviewed</span></div>
    <div class="verdict" id="verdict">
      <button class="pair-verdict" data-choice="candidate_a" type="button">A better in this frame</button>
      <button class="pair-verdict" data-choice="equivalent" type="button">Equivalent in this frame</button>
      <button class="single-verdict" data-choice="acceptable" type="button" hidden>Acceptable in this frame</button>
      <button class="single-verdict" data-choice="needs_repair" type="button" hidden>Needs repair in this frame</button>
      <button data-choice="unresolved" type="button">Unresolved</button>
      <button class="pair-verdict" data-choice="candidate_b" type="button">B better in this frame</button>
    </div>
    <button id="nextUnreviewedFrame" class="command" type="button">Next unreviewed frame</button>
    <textarea id="notes" class="notes" placeholder="Review notes"></textarea>
    <div class="save-row"><button id="save" class="save" type="button">Save and next</button><span class="save-status"><span id="saveIndicator" class="save-indicator"></span><span id="status" class="status">Enter reviewer name to begin</span></span></div>
    <pre id="reveal" class="reveal"></pre>
  </section>
</main>
</div>
<script>
const PORTABLE_BUNDLE = __FFAC_PORTABLE_BUNDLE__;
const IS_PORTABLE = PORTABLE_BUNDLE !== null;
const SESSION_FORMAT = 'ffac-pairwise-review-session';
const MEMORY_STORAGE = new Map();
let STORAGE_PERSISTENT = true;
let DATABASE_PERSISTENT = 'indexedDB' in window;
let portableDatabasePromise=null, portableDatabaseChain=Promise.resolve();
let portableFileHandle=null, portableFileChain=Promise.resolve();
try {
  const probe='ffac-storage-probe';
  localStorage.setItem(probe,'1');
  localStorage.removeItem(probe);
} catch (_error) {
  STORAGE_PERSISTENT = false;
}
function storageGet(key){
  if(STORAGE_PERSISTENT){try{return localStorage.getItem(key);}catch(_error){STORAGE_PERSISTENT=false;}}
  return MEMORY_STORAGE.has(key)?MEMORY_STORAGE.get(key):null;
}
function storageSet(key,value){
  if(STORAGE_PERSISTENT){try{localStorage.setItem(key,value);return;}catch(_error){STORAGE_PERSISTENT=false;}}
  MEMORY_STORAGE.set(key,String(value));
}
function openPortableDatabase(){
  if(!IS_PORTABLE||!('indexedDB' in window))return Promise.reject(new Error('IndexedDB is unavailable'));
  if(portableDatabasePromise)return portableDatabasePromise;
  portableDatabasePromise=new Promise((resolve,reject)=>{const request=indexedDB.open('FluidFlowerReviewAutosave',1);request.onupgradeneeded=()=>{const database=request.result;if(!database.objectStoreNames.contains('sessions'))database.createObjectStore('sessions');if(!database.objectStoreNames.contains('revisions'))database.createObjectStore('revisions',{keyPath:'revision_id'});};request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error||new Error('Could not open browser autosave database'));});
  return portableDatabasePromise;
}
async function databaseSession(key){
  if(!IS_PORTABLE)return null;
  try{const database=await openPortableDatabase();return await new Promise((resolve,reject)=>{const request=database.transaction('sessions','readonly').objectStore('sessions').get(key);request.onsuccess=()=>resolve(request.result||null);request.onerror=()=>reject(request.error);});}catch(_error){DATABASE_PERSISTENT=false;return null;}
}
function queueDatabaseSession(key,session){
  if(!IS_PORTABLE)return Promise.resolve(false);
  const snapshot=clone(session),recorded=utcNow(),revision={revision_id:`${key}:${recorded}:${Math.random().toString(16).slice(2)}`,session_key:key,reviewer:snapshot.reviewer,recorded_utc:recorded,snapshot};
  portableDatabaseChain=portableDatabaseChain.catch(()=>false).then(async()=>{try{const database=await openPortableDatabase();await new Promise((resolve,reject)=>{const transaction=database.transaction(['sessions','revisions'],'readwrite');transaction.objectStore('sessions').put(snapshot,key);transaction.objectStore('revisions').put(revision);transaction.oncomplete=()=>resolve();transaction.onerror=()=>reject(transaction.error);transaction.onabort=()=>reject(transaction.error||new Error('Browser autosave transaction aborted'));});DATABASE_PERSISTENT=true;return true;}catch(error){DATABASE_PERSISTENT=false;setSaveStatus('error','Browser database backup failed · primary copy retained',String(error&&error.message||error));return false;}});
  return portableDatabaseChain;
}
function queuePortableFileSnapshot(session){
  if(!portableFileHandle)return Promise.resolve(false);
  const encoded=JSON.stringify(session,null,2)+'\n';
  portableFileChain=portableFileChain.catch(()=>false).then(async()=>{try{const writable=await portableFileHandle.createWritable();await writable.write(encoded);await writable.close();return true;}catch(error){setSaveStatus('error','Autosave file failed · browser copies retained',String(error&&error.message||error));return false;}});
  return portableFileChain;
}
const ANNOTATION_LINE_WIDTH_KEY='ffac-review-annotation-line-width';
function cleanAnnotationLineWidth(value){const number=Number(value);return Number.isFinite(number)?Math.max(0.5,Math.min(2,number)):1;}
function setAnnotationLineWidth(value,{persist=true}={}){const width=cleanAnnotationLineWidth(value);document.documentElement.style.setProperty('--annotation-line-width',`${width}px`);byId('annotationLineWidth').value=String(width);byId('annotationLineWidthValue').textContent=`${width.toFixed(2)} px`;if(persist)storageSet(ANNOTATION_LINE_WIDTH_KEY,String(width));}
function storageRemove(key){
  if(STORAGE_PERSISTENT){try{localStorage.removeItem(key);return;}catch(_error){STORAGE_PERSISTENT=false;}}
  MEMORY_STORAGE.delete(key);
}
const FLAG_LABELS = __FFAC_REVIEW_FLAG_LABELS__;
const CATEGORY_LABELS = {
  confirmed_only_aqueous_co2:'Only aqueous CO2',
  confirmed_predominantly_aqueous_co2:'Mostly aqueous CO2, some free CO2',
  confirmed_aqueous_co2:'Legacy: mostly aqueous CO2',
  confirmed_predominantly_free_gas:'Mostly free CO2, some aqueous CO2',
  confirmed_only_free_gas:'Only free CO2', confirmed_free_gas:'Legacy: mostly or entirely free CO2',
  confirmed_mixed_free_aqueous_co2:'Mixture: free + aqueous CO2',
  confirmed_mixed_aqueous_water:'Mixture: aqueous CO2 + untouched water',
  confirmed_mixed_free_aqueous_water:'Mixture: free + aqueous CO2 + untouched water',
  confirmed_no_co2:'Confirmed no CO2', confirmed_no_free_gas_in_selected_area:'No free gas anywhere in selected area', confirmed_no_aqueous_co2_in_selected_area:'No aqueous CO2 anywhere in selected area', unsupported_free_gas_classification:'Incorrect free-gas prediction in this model', unsupported_aqueous_co2_classification:'Incorrect aqueous-CO2 prediction in this model',
  likely_dense_aqueous_co2_misclassified_as_gas:'Likely dense aqueous CO2 misclassified as gas',
  phase_ambiguous_strong_color_response:'Strong colour response; phase ambiguous',
  implausible_post_injection_downward_gas_migration:'Implausible downward gas migration after injection',
  plausible_injection_forced_downward_gas:'Plausible injection-forced downward gas',
  plausible_dense_aqueous_co2_descent:'Plausible downward motion of dense aqueous CO2',
  implausible_late_free_gas_persistence:'Implausible late free-gas persistence',
  implausible_gas_location:'Implausible gas location', implausible_aqueous_location:'Implausible aqueous location',
  implausible_temporal_evolution:'Implausible temporal evolution',
  missed_visible_co2:'Missed visible CO2', missed_visible_gas:'Missed visible gas',
  disconnected_noise:'Disconnected noise', facies_edge_artifact:'Facies-edge artifact',
  incomplete_wash_indicator_carryover:'Indicator carryover from incomplete washing',
  baseline_or_flush_artifact:'Baseline or incomplete-flush artifact',
  lighting_or_color_correction_artifact:'Lighting or colour-correction artifact',
  phase_not_assessable:'Uncertain / cannot assess', phase_signal_artifact:'Artifact / not a valid phase signal', other:'Other'
};
const CATEGORY_DESCRIPTIONS = {
  confirmed_only_aqueous_co2:'The visible CO2 response is aqueous, with no visible free-gas phase or untouched-water component.',
  confirmed_predominantly_aqueous_co2:'Both CO2 phases are visible, with aqueous CO2 clearly dominant.',
  confirmed_aqueous_co2:'Legacy combined label retained for existing reviews.',
  confirmed_predominantly_free_gas:'Both CO2 phases are visible, with free CO2 clearly dominant.',
  confirmed_only_free_gas:'Free CO2 is visible without aqueous CO2 or untouched water.',
  confirmed_free_gas:'Legacy combined label retained for existing reviews.',
  confirmed_mixed_free_aqueous_co2:'Both free and aqueous CO2 are materially visible and neither clearly dominates.',
  confirmed_mixed_aqueous_water:'Both aqueous CO2 and untouched water are materially visible.',
  confirmed_mixed_free_aqueous_water:'Free CO2, aqueous CO2, and untouched water are all materially visible in the outlined 2D region.',
  confirmed_no_co2:'The region is visually consistent with untouched water or the pre-injection state.',
  confirmed_no_free_gas_in_selected_area:'Every part of the selected area is free of visible free gas. Aqueous CO2 may still be present; use this only when the entire area supports that conclusion.',
  confirmed_no_aqueous_co2_in_selected_area:'Every part of the selected area is free of visible aqueous CO2. Free gas may still be present; use this only when the entire area supports that conclusion.',
  unsupported_free_gas_classification:'Some free-gas predictions in the selected area are wrong for this model. Real free gas may also be present in the area. This box is evaluated only for this model and is not a whole-box negative training label.',
  unsupported_aqueous_co2_classification:'Some aqueous-CO2 predictions in the selected area are wrong for this model. Real aqueous CO2 may also be present in the area. This box is evaluated only for this model and is not a whole-box negative training label.',
  likely_dense_aqueous_co2_misclassified_as_gas:'A strong, descending colour response is more consistent with dense CO2-rich water than with downward-moving free gas.',
  phase_ambiguous_strong_color_response:'CO2-related colour change is visible, but the image does not support a confident gas/aqueous split.',
  implausible_post_injection_downward_gas_migration:'After active injection has ended, the modelled free-gas region continues to move downward without a connected pressure-driven explanation.',
  plausible_injection_forced_downward_gas:'During active injection, pressure near an injector can plausibly force connected free gas locally downward.',
  plausible_dense_aqueous_co2_descent:'Downward motion is plausibly caused by denser CO2-rich aqueous water rather than free gas.',
  implausible_late_free_gas_persistence:'The model retains free gas later or longer than supported by the optical sequence and experiment history.',
  implausible_gas_location:'Free gas is assigned to a location that is not physically or optically supported.',
  implausible_aqueous_location:'Aqueous CO2 is assigned to a location that is not physically or optically supported.',
  implausible_temporal_evolution:'The phase or plume changes between frames in another physically implausible way.',
  missed_visible_co2:'Visible CO2-related change lies outside the filled total-CO2 phase map.',
  missed_visible_gas:'Visible evidence for free gas lies outside the filled gas phase map.',
  disconnected_noise:'An isolated or blinking detection has no spatial or temporal connection to the established plume.',
  facies_edge_artifact:'The detection follows a sand-layer boundary more closely than a plausible plume boundary.',
  incomplete_wash_indicator_carryover:'Pre-existing indicator colour change from insufficient washing after an earlier experiment, commonly along the frame edge or lower boundary. It can resemble a CO2 response, but is not evidence of CO2 introduced in the current experiment.',
  baseline_or_flush_artifact:'The signal was already present before injection or is consistent with incomplete flushing.',
  lighting_or_color_correction_artifact:'The apparent phase or CO2 change follows illumination or colour-correction variation.',
  phase_not_assessable:'The image does not support a defensible phase interpretation.',
  phase_signal_artifact:'The visible response is an image or experiment artifact rather than a valid phase label.',
  other:'Use the comment field to describe a local observation not covered above.'
};
const PHASE_LABELS={gas:'Gas',aqueous:'Aqueous',total:'Total CO2',unspecified:'Unspecified'};
const SEVERITY_LABELS={minor:'Minor',major:'Major',critical:'Critical'};
const CONFIDENCE_LABELS={high:'High',medium:'Medium',low:'Low',unspecified:'Unspecified'};
const CATEGORY_PHASE_DEFAULTS={confirmed_only_aqueous_co2:'aqueous',confirmed_predominantly_aqueous_co2:'total',confirmed_aqueous_co2:'aqueous',confirmed_predominantly_free_gas:'total',confirmed_only_free_gas:'gas',confirmed_free_gas:'gas',confirmed_mixed_free_aqueous_co2:'total',confirmed_mixed_aqueous_water:'aqueous',confirmed_mixed_free_aqueous_water:'total',confirmed_no_co2:'total',confirmed_no_free_gas_in_selected_area:'gas',confirmed_no_aqueous_co2_in_selected_area:'aqueous',phase_not_assessable:'unspecified',phase_signal_artifact:'unspecified',unsupported_free_gas_classification:'gas',unsupported_aqueous_co2_classification:'aqueous',likely_dense_aqueous_co2_misclassified_as_gas:'gas',implausible_post_injection_downward_gas_migration:'gas',plausible_injection_forced_downward_gas:'gas',plausible_dense_aqueous_co2_descent:'aqueous',implausible_late_free_gas_persistence:'gas',implausible_gas_location:'gas',implausible_aqueous_location:'aqueous',missed_visible_gas:'gas',incomplete_wash_indicator_carryover:'unspecified'};
const CATEGORY_SEMANTICS={
  missed_visible_plume:{evidence_scope:'visible_source_image_region',co2_evidence:'co2_present_phase_unresolved',training_role:'regional_phase_evidence',training_phase_class:'co2_present_region',training_scope:'source_image_all_candidate_models',evaluation_scope:'trigger_candidate_only'},
  missed_visible_co2:{evidence_scope:'visible_source_image_region',co2_evidence:'co2_present_phase_unresolved',training_role:'regional_phase_evidence',training_phase_class:'co2_present_region',training_scope:'source_image_all_candidate_models',evaluation_scope:'trigger_candidate_only'},
  missed_visible_gas:{evidence_scope:'visible_source_image_region',co2_evidence:'free_co2_present',training_role:'regional_phase_evidence',training_phase_class:'gas_present_region',training_scope:'source_image_all_candidate_models',evaluation_scope:'trigger_candidate_only'},
  missed_visible_aqueous_co2:{evidence_scope:'visible_source_image_region',co2_evidence:'aqueous_co2_throughout_region',training_role:'dense_phase_evidence',training_phase_class:'aqueous',training_scope:'source_image_all_candidate_models',evaluation_scope:'trigger_candidate_only'},
  isolated_noise:{evidence_scope:'visible_source_image_region',co2_evidence:'current_experiment_co2_not_supported',training_role:'regional_phase_exclusion',training_phase_class:'co2_absent_region',training_scope:'source_image_all_candidate_models',evaluation_scope:'trigger_candidate_only'},
  disconnected_noise:{evidence_scope:'visible_source_image_region',co2_evidence:'current_experiment_co2_not_supported',training_role:'regional_phase_exclusion',training_phase_class:'co2_absent_region',training_scope:'source_image_all_candidate_models',evaluation_scope:'trigger_candidate_only'},
  unsupported_free_gas_classification:{evidence_scope:'candidate_phase_map_region',co2_evidence:'candidate_free_gas_classification_unsupported',training_role:'candidate_issue_only',training_phase_class:'ignored',training_scope:'none',evaluation_scope:'trigger_candidate_only',training_label_policy_version:'visible-phase-manual-v4'},
  confirmed_no_free_gas_in_selected_area:{evidence_scope:'visible_source_image_region',co2_evidence:'free_co2_absent_throughout_selected_area',training_role:'regional_phase_exclusion',training_phase_class:'gas_absent_region',training_scope:'source_image_all_candidate_models',evaluation_scope:'source_image_shared_across_candidates',interpretation_scope:'source_image_shared_across_candidates',training_label_policy_version:'visible-phase-manual-v4'},
  unsupported_aqueous_co2_classification:{evidence_scope:'candidate_phase_map_region',co2_evidence:'candidate_aqueous_co2_classification_unsupported',training_role:'candidate_issue_only',training_phase_class:'ignored',training_scope:'none',evaluation_scope:'trigger_candidate_only',training_label_policy_version:'visible-phase-manual-v4'},
  confirmed_no_aqueous_co2_in_selected_area:{evidence_scope:'visible_source_image_region',co2_evidence:'aqueous_co2_absent_throughout_selected_area',training_role:'regional_phase_exclusion',training_phase_class:'aqueous_absent_region',training_scope:'source_image_all_candidate_models',evaluation_scope:'source_image_shared_across_candidates',interpretation_scope:'source_image_shared_across_candidates',training_label_policy_version:'visible-phase-manual-v4'},
  likely_dense_aqueous_co2_misclassified_as_gas:{evidence_scope:'visible_source_image_region',co2_evidence:'aqueous_co2_present_free_co2_not_supported',training_role:'regional_phase_evidence',training_phase_class:'aqueous_present_gas_absent_region',training_scope:'source_image_all_candidate_models',evaluation_scope:'trigger_candidate_only'},
  incomplete_wash_indicator_carryover:{evidence_scope:'current_experiment',co2_evidence:'exclude_as_current_experiment_co2',training_role:'artifact_exclusion',training_phase_class:'current_experiment_co2_artifact',training_scope:'source_image_all_candidate_models',evaluation_scope:'source_image_shared_across_candidates',artifact_origin:'previous_experiment_incomplete_washing',interpretation_scope:'source_image_shared_across_candidates'}
};
function annotationCategorySemantics(category){return CATEGORY_SEMANTICS[category]?{...CATEGORY_SEMANTICS[category]}:{};}
const PHASE_QUIZ_LABELS={water_or_no_visible_co2:'Water / no visible CO2',only_aqueous_co2:'Only aqueous CO2',predominantly_aqueous_co2:'Mostly aqueous CO2, some free CO2',mixed_free_and_aqueous_co2:'Mixture: free + aqueous CO2',predominantly_free_gas:'Mostly free CO2, some aqueous CO2',free_gas_only:'Only free CO2',mixed_aqueous_co2_and_water:'Mixture: aqueous CO2 + untouched water',mixed_free_aqueous_co2_and_water:'Mixture: free + aqueous CO2 + untouched water',co2_present_phase_uncertain:'CO2 visible; phase split uncertain',not_assessable:'Uncertain / cannot assess',phase_signal_artifact:'Artifact / not a valid phase signal',ignore_region:'Ignored region'};
const PHASE_QUIZ_MODEL_LABELS={water:'Water / no visible CO2',aqueous:'Mostly or entirely aqueous CO2',gas:'Mostly or entirely free CO2',mixed:'CO2 present; mixed model phase',unscored:'Blind sampling region; no five-class model label'};
const PHASE_QUIZ_EXPECTED={water:['water_or_no_visible_co2'],aqueous:['only_aqueous_co2','predominantly_aqueous_co2'],gas:['predominantly_free_gas','free_gas_only'],mixed:['predominantly_aqueous_co2','mixed_free_and_aqueous_co2','predominantly_free_gas']};
const PHASE_QUIZ_CLASSES={water_or_no_visible_co2:'water',only_aqueous_co2:'aqueous',predominantly_aqueous_co2:'mixed_gas_aqueous',mixed_free_and_aqueous_co2:'mixed_gas_aqueous',predominantly_free_gas:'mixed_gas_aqueous',free_gas_only:'gas',mixed_aqueous_co2_and_water:'mixed_aqueous_water',mixed_free_aqueous_co2_and_water:'mixed_gas_aqueous_water',co2_present_phase_uncertain:'co2_present_region',not_assessable:'ignored',phase_signal_artifact:'ignored',ignore_region:'ignored'};
const PHASE_QUIZ_NONSCORING=new Set(['mixed_free_aqueous_co2_and_water','co2_present_phase_uncertain','not_assessable','phase_signal_artifact','ignore_region']);
const PHASE_CATEGORY_TO_CHOICE={confirmed_no_co2:'water_or_no_visible_co2',confirmed_only_aqueous_co2:'only_aqueous_co2',confirmed_predominantly_aqueous_co2:'predominantly_aqueous_co2',confirmed_mixed_free_aqueous_co2:'mixed_free_and_aqueous_co2',confirmed_predominantly_free_gas:'predominantly_free_gas',confirmed_only_free_gas:'free_gas_only',confirmed_mixed_aqueous_water:'mixed_aqueous_co2_and_water',confirmed_mixed_free_aqueous_water:'mixed_free_aqueous_co2_and_water',phase_ambiguous_strong_color_response:'co2_present_phase_uncertain',phase_not_assessable:'not_assessable',phase_signal_artifact:'phase_signal_artifact'};
const PHASE_CHOICE_DOMINANCE={water_or_no_visible_co2:'water',only_aqueous_co2:'aqueous',predominantly_aqueous_co2:'aqueous',mixed_free_and_aqueous_co2:'none',predominantly_free_gas:'gas',free_gas_only:'gas',mixed_aqueous_co2_and_water:'aqueous',mixed_free_aqueous_co2_and_water:'none',co2_present_phase_uncertain:'none',not_assessable:'none',phase_signal_artifact:'none'};
const PHASE_CHOICE_COMPONENTS={water_or_no_visible_co2:['water'],only_aqueous_co2:['aqueous'],predominantly_aqueous_co2:['aqueous','gas'],mixed_free_and_aqueous_co2:['aqueous','gas'],predominantly_free_gas:['aqueous','gas'],free_gas_only:['gas'],mixed_aqueous_co2_and_water:['aqueous','water'],mixed_free_aqueous_co2_and_water:['aqueous','gas','water'],co2_present_phase_uncertain:['co2'],not_assessable:[],phase_signal_artifact:[]};
function visiblePhaseFields(category){const choice=PHASE_CATEGORY_TO_CHOICE[category];if(!choice)return{};return{human_choice:choice,phase_label_schema:'fluidflower-visible-phase-v3',phase_class:PHASE_QUIZ_CLASSES[choice],phase_dominance:PHASE_CHOICE_DOMINANCE[choice],phase_components:PHASE_CHOICE_COMPONENTS[choice],training_eligible:!['not_assessable','phase_signal_artifact'].includes(choice),interpretation_scope:'visible_2d_slice_only'};}
let state=null, currentRun=null, currentPair=null, choice=null, frameChoices=new Map(), playTimer=null, frameIndex=0, frameCount=0;
let annotations=[], selectedAnnotationId=null, drawTool='pan', interaction=null, activeSide='A';
let candidateFlags={A:[],B:[]};
let activeSession=null, activeReviewerName='', spacePan=false, swapHeld=false, contourSwapHeld=false;
let phaseQuizAnswers=[], phaseQuizIndex=0, phaseQuizActive=true, phaseQuizPerCandidateLimit=0;
let phaseQuizSelectedIds=[], phaseQuizRounds=[];
let livePollBusy=false, livePollTimer=null;
let serverDraftTimer=null, pendingServerDraft=null, serverDraftChain=Promise.resolve();
const framePreloadCache=new Map();
const isolatedComponentCache=new Map();
let framePreloadTimer=null;
let isolatedMarkerGeneration=0;
const transforms={A:{scale:1,x:0,y:0},B:{scale:1,x:0,y:0}};
const byId = id => document.getElementById(id);
function setSaveStatus(kind,text,detail=''){const indicator=byId('saveIndicator');indicator.className=`save-indicator ${kind||''}`.trim();indicator.title=detail||text;byId('status').textContent=text;byId('status').title=detail||text;}
function reviewerIsReady(){const value=byId('reviewer').value.trim();return Boolean(value&&value===reviewerName());}
function updateReviewerGate(){const ready=reviewerIsReady(),main=document.querySelector('main'),runs=byId('runs');main.classList.toggle('review-locked',!ready);runs.classList.toggle('review-locked',!ready);main.toggleAttribute('inert',!ready);runs.toggleAttribute('inert',!ready);main.setAttribute('aria-disabled',String(!ready));runs.setAttribute('aria-disabled',String(!ready));byId('save').disabled=!ready;if(!ready)setSaveStatus('blocked','Enter reviewer name and select Open','Review controls stay locked until the reviewer identity has been opened.');}
function blockLockedReviewEvent(event){if(reviewerIsReady())return;event.preventDefault();event.stopImmediatePropagation();updateReviewerGate();}
for(const target of [document.querySelector('main'),byId('runs')])for(const type of ['click','pointerdown','keydown'])target.addEventListener(type,blockLockedReviewEvent,true);
function setSwapHeld(held){const button=byId('holdSwap'),active=Boolean(held)&&!button.hidden;swapHeld=active;byId('comparison').classList.toggle('swap-held',active);button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));button.textContent=active?'Release to restore':'Hold to swap A / B';}
function setContourSwapHeld(held,render=true){const button=byId('holdContourSwap'),active=Boolean(held)&&!button.hidden&&!button.disabled;contourSwapHeld=active;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));button.textContent=active?'Release to restore phase maps':'Hold to swap phase maps';if(render&&currentPair)renderFrame({preserveSelection:true});}
function contourSideForPanel(side,swapped=contourSwapHeld){return swapped?(side==='A'?'B':'A'):side;}
byId('selectedAnnotationCategory').innerHTML=byId('annotationCategory').innerHTML;
byId('selectedAnnotationPhase').innerHTML=byId('annotationPhase').innerHTML;
byId('selectedAnnotationSeverity').innerHTML=byId('annotationSeverity').innerHTML;
byId('selectedAnnotationConfidence').innerHTML=byId('annotationConfidence').innerHTML+'<option value="unspecified">Unspecified</option>';
function updateAnnotationCategoryHelp(category=byId('annotationCategory').value){byId('annotationCategoryHelp').textContent=CATEGORY_DESCRIPTIONS[category]||CATEGORY_DESCRIPTIONS.other;}
updateAnnotationCategoryHelp();
async function jsonFetch(url, options={}) { const response=await fetch(url,options); const data=await response.json(); if(!response.ok) throw new Error(data.error||response.statusText); return data; }
function utcNow(){return new Date().toISOString();}
function clone(value){return JSON.parse(JSON.stringify(value));}
function reviewerSlug(name){const slug=name.trim().normalize('NFKD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');return slug||'reviewer';}
function reviewerName(session=activeSession){return session&&session.reviewer&&session.reviewer.name?session.reviewer.name:activeReviewerName;}
function sessionStorageKey(name){return `ffac-review-session:${PORTABLE_BUNDLE.bundle_id}:${encodeURIComponent(name.trim().toLocaleLowerCase())}`;}
function lastReviewerStorageKey(){return `ffac-review-last-reviewer:${PORTABLE_BUNDLE.bundle_id}`;}
function newPortableSession(name){const clean=name.trim(),now=utcNow();return{schema:1,format:SESSION_FORMAT,bundle_id:PORTABLE_BUNDLE.bundle_id,source_manifest_sha256:PORTABLE_BUNDLE.source_manifest_sha256,reviewer:{name:clean,id:reviewerSlug(clean)},created_utc:now,updated_utc:now,current_run:PORTABLE_BUNDLE.runs[0]||null,decisions:{},drafts:{}};}
function expectedCandidate(run,label){const pair=PORTABLE_BUNDLE.pairs[run];return pair&&pair.reveal[label];}
function validatePortableAnnotation(item,run,pair){if(!item||typeof item!=='object'||Array.isArray(item))throw new Error(`Invalid annotation in ${run.toUpperCase()}`);const candidate=item.candidate==='A'?pair.candidate_a:item.candidate==='B'?pair.candidate_b:null;if(!candidate||!Number.isInteger(item.frame_index)||item.frame_index<0||item.frame_index>=candidate.frame_count)throw new Error(`Invalid annotation frame in ${run.toUpperCase()}`);if(!(item.category in CATEGORY_LABELS))throw new Error(`Invalid annotation category in ${run.toUpperCase()}`);const phase=item.phase||CATEGORY_PHASE_DEFAULTS[item.category]||'unspecified',severity=item.severity||'major',confidence=item.confidence||'unspecified',shape=item.shape||'rectangle';if(!(phase in PHASE_LABELS)||!(severity in SEVERITY_LABELS)||!(confidence in CONFIDENCE_LABELS)||!['rectangle','freehand','point'].includes(shape))throw new Error(`Invalid annotation metadata in ${run.toUpperCase()}`);const x=Number(item.x),y=Number(item.y),width=Number(item.width),height=Number(item.height);if(![x,y,width,height].every(Number.isFinite)||x<0||y<0||width<=0||height<=0||x+width>1.000001||y+height>1.000001)throw new Error(`Invalid annotation coordinates in ${run.toUpperCase()}`);if(shape==='freehand'){if(!Array.isArray(item.points)||item.points.length<3||item.points.length>1000)throw new Error(`Invalid freehand annotation in ${run.toUpperCase()}`);for(const point of item.points)if(!Array.isArray(point)||point.length!==2||!point.every(value=>Number.isFinite(Number(value))&&Number(value)>=0&&Number(value)<=1))throw new Error(`Invalid freehand point in ${run.toUpperCase()}`);}if(shape==='point'){const fields=['analysis_pixel_x','analysis_pixel_y','analysis_image_width','analysis_image_height','canvas_pixel_x','canvas_pixel_y','review_canvas_width','review_canvas_height'];if(!fields.every(name=>Number.isInteger(Number(item[name]))))throw new Error(`Invalid pixel annotation in ${run.toUpperCase()}`);if(item.analysis_pixel_x<0||item.analysis_pixel_y<0||item.analysis_pixel_x>=item.analysis_image_width||item.analysis_pixel_y>=item.analysis_image_height||item.canvas_pixel_x<0||item.canvas_pixel_y<0||item.canvas_pixel_x>=item.review_canvas_width||item.canvas_pixel_y>=item.review_canvas_height)throw new Error(`Pixel annotation outside image in ${run.toUpperCase()}`);}}
function phaseQuizCandidate(pair,side){return !pair?null:side==='A'?pair.candidate_a:pair.candidate_b;}
function phaseQuizStoredProposals(pair){if(!pair)return[];return['A','B'].flatMap(side=>(phaseQuizCandidate(pair,side).phase_quiz_proposals||[]).map(proposal=>({...proposal,candidate:side})));}
function phaseQuizSharedProposals(pair){if(!pair)return[];const candidate=phaseQuizCandidate(pair,'A');return(candidate&&candidate.phase_quiz_proposals||[]).map(proposal=>({...proposal,candidate:'A'}));}
function phaseQuizAllProposals(pair){return phaseQuizSharedProposals(pair);}
function phaseQuizAnswersForPair(rawAnswers,pair){const keys=new Set(phaseQuizStoredProposals(pair).map(proposal=>`${proposal.candidate}:${proposal.proposal_id}`));return(rawAnswers||[]).filter(answer=>keys.has(`${String(answer.candidate||'').toUpperCase()}:${answer.proposal_id}`));}
function phaseQuizInitialLimit(pair){if(!pair)return 0;const candidate=phaseQuizCandidate(pair,'A'),available=(candidate&&candidate.phase_quiz_proposals||[]).length,value=Number(candidate&&candidate.phase_quiz_initial_count);return Math.max(0,Math.min(available,Number.isFinite(value)?Math.trunc(value):available));}
function phaseQuizAvailableLimit(pair){return phaseQuizSharedProposals(pair).length;}
function cleanPhaseQuizLimit(pair,value,answers=[]){const available=phaseQuizAvailableLimit(pair);if(!available)return 0;const number=Number(value),requested=Number.isFinite(number)?Math.trunc(number):phaseQuizInitialLimit(pair),answered=phaseQuizInitialSelectedIds(pair,answers).length;return Math.max(Math.min(available,phaseQuizInitialLimit(pair)),Math.min(available,requested),Math.min(available,answered));}
function phaseQuizInitialSelectedIds(pair,answers=[]){const proposals=phaseQuizSharedProposals(pair),valid=new Set(proposals.map(item=>item.proposal_id)),answered=(answers||[]).filter(item=>String(item.candidate||'').toUpperCase()==='A'&&valid.has(item.proposal_id)).map(item=>item.proposal_id);if(answered.length)return[...new Set(answered)];return proposals.slice(0,phaseQuizInitialLimit(pair)).map(item=>item.proposal_id);}
function cleanPhaseQuizSelectedIds(pair,rawIds,answers=[]){const proposals=phaseQuizSharedProposals(pair),valid=new Set(proposals.map(item=>item.proposal_id)),selected=[];for(const id of Array.isArray(rawIds)?rawIds:[]){const value=String(id||'');if(valid.has(value)&&!selected.includes(value))selected.push(value);}if(!selected.length)selected.push(...phaseQuizInitialSelectedIds(pair,answers));for(const answer of answers||[]){if(String(answer.candidate||'').toUpperCase()==='A'&&valid.has(answer.proposal_id)&&!selected.includes(answer.proposal_id))selected.push(answer.proposal_id);}return selected;}
function cleanPhaseQuizRounds(pair,rawRounds,selectedIds){const valid=new Set(selectedIds||[]),seen=new Set(),clean=[];for(const [index,source] of (Array.isArray(rawRounds)?rawRounds:[]).entries()){const ids=[];for(const id of Array.isArray(source&&source.proposal_ids)?source.proposal_ids:[]){const value=String(id||'');if(valid.has(value)&&!seen.has(value)){ids.push(value);seen.add(value);}}if(ids.length)clean.push({round:Number(source.round)||index+1,selection_mode:String(source.selection_mode||'adaptive_optical_pool'),proposal_ids:ids,created_utc:String(source.created_utc||'')});}const missing=(selectedIds||[]).filter(id=>!seen.has(id));if(missing.length)clean.unshift({round:1,selection_mode:'legacy_or_initial_batch',proposal_ids:missing,created_utc:''});return clean;}
function phaseQuizProposalsForIds(pair,ids){const byIdMap=new Map(phaseQuizSharedProposals(pair).map(item=>[item.proposal_id,item]));return(ids||[]).map(id=>byIdMap.get(id)).filter(Boolean);}
function phaseQuizProposalsForLimit(pair,limit){if(!pair)return[];const count=Math.max(0,Math.min(phaseQuizAvailableLimit(pair),Number(limit)||phaseQuizInitialLimit(pair)));return phaseQuizSharedProposals(pair).slice(0,count);}
function phaseQuizProposals(pair){return phaseQuizProposalsForIds(pair,phaseQuizSelectedIds);}
function phaseQuizAnsweredCount(proposals,answers=[]){return(proposals||[]).filter(proposal=>(answers||[]).some(answer=>String(answer.candidate||'').toUpperCase()===proposal.candidate&&answer.proposal_id===proposal.proposal_id)).length;}
function validatePortablePhaseQuizAnswer(item,run,pair){if(!item||typeof item!=='object'||Array.isArray(item))throw new Error(`Invalid phase quiz answer in ${run.toUpperCase()}`);const side=String(item.candidate||'').toUpperCase(),proposal=phaseQuizStoredProposals(pair).find(candidate=>candidate.candidate===side&&candidate.proposal_id===item.proposal_id);if(!proposal)throw new Error(`Unknown phase quiz proposal in ${run.toUpperCase()}`);if(!(item.human_choice in PHASE_QUIZ_LABELS)||!['high','medium','low'].includes(item.confidence||'medium'))throw new Error(`Invalid phase quiz answer in ${run.toUpperCase()}`);}
function phaseQuizFeatureVector(proposal){const stats=proposal&&proposal.model_stats||{},delta=Array.isArray(stats.mean_delta_rgb)?stats.mean_delta_rgb:[0,0,0],current=Array.isArray(stats.mean_current_chromaticity)?stats.mean_current_chromaticity:[0,0,0],reference=Array.isArray(stats.mean_reference_chromaticity)?stats.mean_reference_chromaticity:[0,0,0];return[Number(stats.optical_change_quantile)||0,Math.log1p(10*Math.max(0,Number(stats.mean_rgb_change_norm)||0)),...delta.map(value=>4*Number(value||0)),...current.map((value,index)=>4*(Number(value||0)-Number(reference[index]||0)))];}
function phaseQuizFeatureDistance(a,b){const left=phaseQuizFeatureVector(a),right=phaseQuizFeatureVector(b);return Math.sqrt(left.reduce((sum,value,index)=>sum+(value-right[index])**2,0)/Math.max(1,left.length));}
function phaseQuizOverlap(a,b){if(Number(a.frame_index)!==Number(b.frame_index))return 0;const ax=Number(a.x),ay=Number(a.y),ar=ax+Number(a.width),ab=ay+Number(a.height),bx=Number(b.x),by=Number(b.y),br=bx+Number(b.width),bb=by+Number(b.height),intersection=Math.max(0,Math.min(ar,br)-Math.max(ax,bx))*Math.max(0,Math.min(ab,bb)-Math.max(ay,by)),union=Number(a.width)*Number(a.height)+Number(b.width)*Number(b.height)-intersection;return union>0?intersection/union:0;}
function strategicPhaseQuizBatch(pair,selectedIds,count=8){
  const phases=['gas','aqueous','mixed','water'],pool=phaseQuizSharedProposals(pair);
  const strategic=pool.filter(item=>phases.includes(String(item.suggested_phase||''))&&(item.model_stats||{}).facies_id!=null);
  if(!strategic.length)return null;
  const selectedSet=new Set(selectedIds||[]),byIdMap=new Map(strategic.map(item=>[item.proposal_id,item]));
  const selected=(selectedIds||[]).map(id=>byIdMap.get(id)).filter(Boolean),remaining=strategic.filter(item=>!selectedSet.has(item.proposal_id));
  const phaseCounts=new Map(phases.map(phase=>[phase,0])),faciesCounts=new Map(),combinationCounts=new Map(),frameCounts=new Map();
  const increment=item=>{const stats=item.model_stats||{},phase=String(item.suggested_phase||''),facies=Number(stats.facies_id),frame=Number(item.frame_index)||0,key=`${phase}:${facies}`;phaseCounts.set(phase,(phaseCounts.get(phase)||0)+1);faciesCounts.set(facies,(faciesCounts.get(facies)||0)+1);combinationCounts.set(key,(combinationCounts.get(key)||0)+1);frameCounts.set(frame,(frameCounts.get(frame)||0)+1);};
  selected.forEach(increment);
  const compare=(left,right)=>{for(let index=0;index<left.length;index+=1){if(left[index]===right[index])continue;return left[index]<right[index]?-1:1;}return 0;},batch=[];
  while(remaining.length&&batch.length<Math.max(0,Number(count)||0)){
    const active=selected.concat(batch),rank=item=>{const stats=item.model_stats||{},phase=String(item.suggested_phase||''),facies=Number(stats.facies_id),frame=Number(item.frame_index)||0,xy=Array.isArray(stats.normalized_center_xy)?stats.normalized_center_xy:[0.5,0.5],same=active.filter(other=>Number((other.model_stats||{}).facies_id)===facies),distance=same.length?Math.min(...same.map(other=>{const otherXY=Array.isArray((other.model_stats||{}).normalized_center_xy)?other.model_stats.normalized_center_xy:[0.5,0.5];return Math.hypot(Number(xy[0])-Number(otherXY[0]),Number(xy[1])-Number(otherXY[1]));})):1;return[phaseCounts.get(phase)||0,faciesCounts.get(facies)||0,combinationCounts.get(`${phase}:${facies}`)||0,frameCounts.get(frame)||0,-distance,-Number(item.priority||0),phases.indexOf(phase),facies,String(item.proposal_id||'')];};
    let bestIndex=0,bestRank=rank(remaining[0]);for(let index=1;index<remaining.length;index+=1){const candidateRank=rank(remaining[index]);if(compare(candidateRank,bestRank)<0){bestIndex=index;bestRank=candidateRank;}}
    const [chosen]=remaining.splice(bestIndex,1);batch.push(chosen);increment(chosen);
  }
  return batch;
}
function adaptivePhaseQuizBatch(pair,selectedIds,answers,count=8){const pool=phaseQuizSharedProposals(pair),selectedSet=new Set(selectedIds||[]),selected=phaseQuizProposalsForIds(pair,selectedIds),answerById=new Map((answers||[]).filter(item=>String(item.candidate||'').toUpperCase()==='A').map(item=>[item.proposal_id,item])),positiveChoices=new Set(['only_aqueous_co2','predominantly_aqueous_co2','predominantly_free_gas','mixed_free_and_aqueous_co2','mixed_aqueous_co2_and_water','mixed_free_aqueous_co2_and_water','co2_present_phase_uncertain']),positive=selected.filter(item=>positiveChoices.has((answerById.get(item.proposal_id)||{}).human_choice)),water=selected.filter(item=>(answerById.get(item.proposal_id)||{}).human_choice==='water_or_no_visible_co2'),waterFraction=(positive.length+water.length)?water.length/(positive.length+water.length):0,candidates=pool.filter(item=>!selectedSet.has(item.proposal_id));const baseScore=proposal=>{const stats=proposal.model_stats||{},stratum=String(stats.sampling_stratum||''),quantile=Number(stats.optical_change_quantile)||0,change=Math.max(0,Number(stats.mean_rgb_change_norm)||0);let score=5*quantile+Math.log1p(10*change);if(stratum==='strong_corrected_optical_change')score+=7;else if(stratum==='optical_change_boundary')score+=4;else if(stratum==='moderate_corrected_optical_change')score+=3;else if(stratum==='weak_change_negative_control')score-=5;else if(stratum==='domain_only_spatial_repair')score-=12;else if(stratum.includes('high_gas')||stratum.includes('decision_boundary')||stratum.includes('strong_optical'))score+=2;if(waterFraction>0.55)score+=5*quantile;if(positive.length){const distance=Math.min(...positive.map(item=>phaseQuizFeatureDistance(proposal,item)));score+=4/(1+5*distance);}if(water.length){const distance=Math.min(...water.map(item=>phaseQuizFeatureDistance(proposal,item)));score-=3/(1+5*distance);}if(selected.some(item=>phaseQuizOverlap(proposal,item)>0.18))score-=8;return score;};const ranked=candidates.map(proposal=>({proposal,score:baseScore(proposal)})).sort((a,b)=>b.score-a.score),batch=[],frameCounts=new Map();for(const maximumPerFrame of [1,2,Infinity]){for(const item of ranked){if(batch.length>=count)break;if(batch.some(existing=>existing.proposal_id===item.proposal.proposal_id))continue;const frame=Number(item.proposal.frame_index),used=frameCounts.get(frame)||0;if(used>=maximumPerFrame)continue;const stratum=String((item.proposal.model_stats||{}).sampling_stratum||''),controls=batch.filter(proposal=>String((proposal.model_stats||{}).sampling_stratum||'').includes('negative_control')||String((proposal.model_stats||{}).sampling_stratum||'')==='domain_only_spatial_repair').length;if((stratum.includes('negative_control')||stratum==='domain_only_spatial_repair')&&controls>=1)continue;if(batch.some(proposal=>phaseQuizOverlap(proposal,item.proposal)>0.18))continue;batch.push(item.proposal);frameCounts.set(frame,used+1);}if(batch.length>=count)break;}return batch.slice(0,count);}
function validatePortableSession(payload){
  if(!payload||typeof payload!=='object'||Array.isArray(payload))throw new Error('Review file must contain a JSON object');
  if(payload.schema!==1||payload.format!==SESSION_FORMAT)throw new Error('Unsupported review file format');
  const sameBundle=payload.bundle_id===PORTABLE_BUNDLE.bundle_id;
  const sameSource=Boolean(payload.source_manifest_sha256&&payload.source_manifest_sha256===PORTABLE_BUNDLE.source_manifest_sha256);
  if(!sameBundle&&!sameSource)throw new Error('This review belongs to a different calibration package');
  if(payload.source_manifest_sha256&&!sameSource)throw new Error('Source manifest does not match this package');
  const name=payload.reviewer&&typeof payload.reviewer.name==='string'?payload.reviewer.name.trim():'';
  if(!name)throw new Error('Review file has no reviewer name');
  if(!payload.decisions||typeof payload.decisions!=='object'||Array.isArray(payload.decisions))throw new Error('Review file has invalid decisions');
  for(const [run,decision] of Object.entries(payload.decisions)){
    const pair=PORTABLE_BUNDLE.pairs[run];
    if(!pair)throw new Error(`Unknown run in review file: ${run}`);
    if(!decision||decision.run!==run||!['candidate_a','candidate_b','equivalent','acceptable','needs_repair','unresolved'].includes(decision.choice))throw new Error(`Invalid decision for ${run.toUpperCase()}`);
    cleanPortableFrameChoices(decision.frame_choices||[],pair);
    for(const [key,label] of [['candidate_a','A'],['candidate_b','B']]){
      const actual=decision[key],expected=expectedCandidate(run,label);
      if(!actual||actual.candidate_id!==expected.candidate_id||actual.params_sha256!==expected.params_sha256)throw new Error(`Candidate identity mismatch for ${run.toUpperCase()} ${label}`);
    }
    if(!decision.flags||!Array.isArray(decision.flags.A)||!Array.isArray(decision.flags.B))throw new Error(`Invalid flags for ${run.toUpperCase()}`);
    for(const side of ['A','B'])for(const flag of decision.flags[side])if(!(flag in FLAG_LABELS))throw new Error(`Unknown flag in ${run.toUpperCase()}: ${flag}`);
    if(!Array.isArray(decision.annotations||[]))throw new Error(`Invalid annotations for ${run.toUpperCase()}`);
    for(const item of decision.annotations||[])validatePortableAnnotation(item,run,pair);
    if(!Array.isArray(decision.phase_quiz_answers||[]))throw new Error(`Invalid phase quiz answers for ${run.toUpperCase()}`);
    for(const item of decision.phase_quiz_answers||[])validatePortablePhaseQuizAnswer(item,run,pair);
  }
  const drafts=payload.drafts&&typeof payload.drafts==='object'&&!Array.isArray(payload.drafts)?payload.drafts:{};for(const [run,draft] of Object.entries(drafts)){const pair=PORTABLE_BUNDLE.pairs[run];if(!pair||!draft||typeof draft!=='object'||Array.isArray(draft))throw new Error(`Invalid draft for ${run.toUpperCase()}`);if(draft.choice&&!['candidate_a','candidate_b','equivalent','acceptable','needs_repair','unresolved'].includes(draft.choice))throw new Error(`Invalid draft verdict for ${run.toUpperCase()}`);cleanPortableFrameChoices(draft.frame_choices||[],pair);if(!Array.isArray(draft.annotations||[]))throw new Error(`Invalid draft annotations for ${run.toUpperCase()}`);for(const item of draft.annotations||[])validatePortableAnnotation(item,run,pair);if(!Array.isArray(draft.phase_quiz_answers||[]))throw new Error(`Invalid draft phase quiz answers for ${run.toUpperCase()}`);for(const item of draft.phase_quiz_answers||[])validatePortablePhaseQuizAnswer(item,run,pair);}
  const reviewer_id=reviewerSlug(name),stamp=item=>({...item,reviewer:name,reviewer_id});const clean=clone(payload);if(!sameBundle){clean.migrated_from_bundle_id=payload.bundle_id;clean.migrated_utc=utcNow();}clean.bundle_id=PORTABLE_BUNDLE.bundle_id;clean.source_manifest_sha256=PORTABLE_BUNDLE.source_manifest_sha256;clean.ui_version=PORTABLE_BUNDLE.ui_version;clean.reviewer={name,id:reviewer_id};clean.drafts=clone(drafts);for(const [run,decision] of Object.entries(clean.decisions)){const pair=PORTABLE_BUNDLE.pairs[run];decision.reviewer=name;decision.reviewer_id=reviewer_id;decision.frame_choices=(decision.frame_choices||[]).map(stamp);decision.annotations=(decision.annotations||[]).map(item=>stamp(normalizeAnnotation(item)));decision.phase_quiz_answers=(decision.phase_quiz_answers||[]).map(stamp);decision.phase_quiz_selected_ids=cleanPhaseQuizSelectedIds(pair,decision.phase_quiz_selected_ids,decision.phase_quiz_answers);decision.phase_quiz_rounds=cleanPhaseQuizRounds(pair,decision.phase_quiz_rounds,decision.phase_quiz_selected_ids);decision.phase_quiz_per_candidate_limit=decision.phase_quiz_selected_ids.length;}for(const [run,draft] of Object.entries(clean.drafts)){const pair=PORTABLE_BUNDLE.pairs[run];draft.reviewer=name;draft.reviewer_id=reviewer_id;draft.frame_choices=(draft.frame_choices||[]).map(stamp);draft.annotations=(draft.annotations||[]).map(item=>stamp(normalizeAnnotation(item)));draft.phase_quiz_answers=(draft.phase_quiz_answers||[]).map(stamp);draft.phase_quiz_selected_ids=cleanPhaseQuizSelectedIds(pair,draft.phase_quiz_selected_ids,draft.phase_quiz_answers);draft.phase_quiz_rounds=cleanPhaseQuizRounds(pair,draft.phase_quiz_rounds,draft.phase_quiz_selected_ids);draft.phase_quiz_per_candidate_limit=draft.phase_quiz_selected_ids.length;}clean.current_run=PORTABLE_BUNDLE.pairs[clean.current_run]?clean.current_run:(PORTABLE_BUNDLE.runs[0]||null);return clean;
}
function portableState(){const decisions=activeSession?activeSession.decisions:{},drafts=activeSession?activeSession.drafts:{};const runs=PORTABLE_BUNDLE.runs.map(run=>{const decision=decisions[run],draft=drafts[run],pair=PORTABLE_BUNDLE.pairs[run],source=draft||decision||{},answers=Array.isArray(source.phase_quiz_answers)?source.phase_quiz_answers:[],selectedIds=cleanPhaseQuizSelectedIds(pair,source.phase_quiz_selected_ids,answers),quizProposals=phaseQuizProposalsForIds(pair,selectedIds),quizTotal=quizProposals.length,quizAnswered=phaseQuizAnsweredCount(quizProposals,answers),frameReviewed=Array.isArray(source.frame_choices)?source.frame_choices.length:0,frameTotal=Math.min(pair.candidate_a.frame_count,pair.candidate_b.frame_count);return{run,reviewed:Boolean(decision),drafted:Boolean(draft),choice:decision?decision.choice:null,frame_choices_reviewed:frameReviewed,frame_choices_total:frameTotal,frame_choices_complete:frameReviewed===frameTotal,updated_utc:draft?draft.updated_utc:decision?decision.updated_utc:null,quiz_answered:quizAnswered,quiz_total:quizTotal,quiz_complete:quizTotal>0&&quizAnswered>=quizTotal};});return{schema:1,runs,reviewed:runs.filter(item=>item.reviewed).length,drafted:runs.filter(item=>item.drafted).length,total:runs.length,unpaired_runs:PORTABLE_BUNDLE.unpaired_runs||[],default_reviewer:reviewerName(),decision_json:'Browser autosave',decision_csv:'Exported review JSON',annotation_csv:'Included in review JSON'};}
function portablePair(run){const source=PORTABLE_BUNDLE.pairs[String(run).toLowerCase()];if(!source)throw new Error(`Unknown run: ${run}`);const decision=activeSession&&activeSession.decisions[source.run]?activeSession.decisions[source.run]:null;return{...source,decision,reveal:decision?source.reveal:null};}
function persistPortableSession(){if(!IS_PORTABLE||!activeSession)return Promise.resolve(false);const name=reviewerName(),key=sessionStorageKey(name);activeSession.reviewer={name,id:reviewerSlug(name)};activeSession.updated_utc=utcNow();storageSet(key,JSON.stringify(activeSession));storageSet(lastReviewerStorageKey(),name);queueDatabaseSession(key,activeSession);queuePortableFileSnapshot(activeSession);updateSessionMeta();return portableDatabaseChain;}
function exportFilename(){const name=reviewerName()||byId('reviewer').value.trim()||'reviewer',now=new Date(),pad=value=>String(value).padStart(2,'0'),stamp=`${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;return`ffac_calibration_review_${reviewerSlug(name)}_${state?state.reviewed:0}-of-${state?state.total:PORTABLE_BUNDLE.runs.length}_${stamp}.json`;}
function updateSessionMeta(){if(!IS_PORTABLE)return;const reviewed=activeSession?Object.keys(activeSession.decisions).length:0,drafts=activeSession?Object.keys(activeSession.drafts).length:0,parts=[STORAGE_PERSISTENT?'browser autosave':'memory only',DATABASE_PERSISTENT?'revision database':'database unavailable',portableFileHandle?'autosave file connected':'no autosave file selected'];byId('sessionMeta').textContent=activeSession?`${reviewerName()} | ${parts.join(' + ')} | ${reviewed} saved, ${drafts} drafts / ${PORTABLE_BUNDLE.runs.length} | ${exportFilename()}`:'Enter a reviewer name to open or start a review';}
async function activateReviewer(name,{reset=false}={}){
  const clean=name.trim();if(!clean)throw new Error('Reviewer is required');
  if(currentRun&&currentPair){saveCurrentDraft();await flushServerDraft();}
  if(IS_PORTABLE){if(activeSession)persistPortableSession();let session=null;if(!reset){const key=sessionStorageKey(clean),saved=storageGet(key),databaseSaved=saved?null:await databaseSession(key);if(saved||databaseSaved)session=validatePortableSession(saved?JSON.parse(saved):databaseSaved);}activeSession=session||newPortableSession(clean);activeReviewerName='';byId('reviewer').value=reviewerName();currentRun=null;currentPair=null;persistPortableSession();state=portableState();renderState();updateReviewerGate();await loadPair(activeSession.current_run||state.runs[0].run);setSaveStatus(STORAGE_PERSISTENT||DATABASE_PERSISTENT?'saved':'error',STORAGE_PERSISTENT||DATABASE_PERSISTENT?`Opened review for ${reviewerName()}`:'Temporary session · export frequently');return;}
  activeReviewerName=clean;storageSet('ffac-reviewer',clean);byId('reviewer').value=clean;currentRun=null;currentPair=null;state=await jsonFetch(`/api/state?reviewer=${encodeURIComponent(clean)}`);renderState();updateReviewerGate();const first=state.runs.find(item=>!item.reviewed)||state.runs[0];if(first)await loadPair(first.run);setSaveStatus('saved','Reviewer opened · autosave ready','Every edit is saved immediately in this browser and synchronized as a separate recovery draft.');
}
function legacyAnnotationDraftKey(run=currentRun){return `ffac-annotation-draft-${run}`;}
function annotationDraftKey(run=currentRun){const scope=IS_PORTABLE?(PORTABLE_BUNDLE.bundle_id||'portable'):(state&&state.decision_json?state.decision_json:location.pathname),reviewer=encodeURIComponent((reviewerName()||'unassigned').trim().toLocaleLowerCase());return `ffac-review-draft:${encodeURIComponent(scope)}:${reviewer}:${run}`;}
function unassignedAnnotationDraftKey(run=currentRun){const scope=IS_PORTABLE?(PORTABLE_BUNDLE.bundle_id||'portable'):(state&&state.decision_json?state.decision_json:location.pathname);return `ffac-review-draft:${encodeURIComponent(scope)}:unassigned:${run}`;}
function previousAnnotationDraftKey(run=currentRun){const scope=IS_PORTABLE?(PORTABLE_BUNDLE.bundle_id||'portable'):(state&&state.decision_json?state.decision_json:location.pathname);return `ffac-review-draft:${encodeURIComponent(scope)}:${run}`;}
function draftStorageText(run=currentRun){return storageGet(annotationDraftKey(run))||storageGet(unassignedAnnotationDraftKey(run))||storageGet(previousAnnotationDraftKey(run))||storageGet(legacyAnnotationDraftKey(run));}
function currentDraft(serverRecord=null){if(IS_PORTABLE)return activeSession&&activeSession.drafts?activeSession.drafts[currentRun]||null:null;let local=null;const text=draftStorageText();if(text){try{const parsed=JSON.parse(text);local=Array.isArray(parsed)?{annotations:parsed}:parsed&&typeof parsed==='object'?parsed:null;}catch(_error){local=null;}}const remote=serverRecord&&serverRecord.draft&&typeof serverRecord.draft==='object'?{...serverRecord.draft,updated_utc:serverRecord.updated_utc||serverRecord.draft.updated_utc}:null,candidates=[local,remote].filter(Boolean).sort((a,b)=>String(b.updated_utc||'').localeCompare(String(a.updated_utc||''))),draft=candidates[0]||null,decision=currentPair&&currentPair.decision;if(draft&&decision&&draft.updated_utc&&decision.updated_utc&&String(draft.updated_utc)<=String(decision.updated_utc))return null;return draft;}
function reviewContent(value){return{choice:value&&value.choice||null,frame_choice_schema:'candidate-frame-preference-v1',frame_choices:value&&Array.isArray(value.frame_choices)?clone(value.frame_choices):[],flags:value&&value.flags||{A:[],B:[]},annotations:value&&Array.isArray(value.annotations)?value.annotations.map(normalizeAnnotation):[],phase_quiz_answers:value&&Array.isArray(value.phase_quiz_answers)?clone(value.phase_quiz_answers):[],phase_quiz_selected_ids:value&&Array.isArray(value.phase_quiz_selected_ids)?clone(value.phase_quiz_selected_ids):[],phase_quiz_rounds:value&&Array.isArray(value.phase_quiz_rounds)?clone(value.phase_quiz_rounds):[],phase_quiz_per_candidate_limit:Number(value&&value.phase_quiz_per_candidate_limit)||0,notes:String(value&&value.notes||'')};}
function queueServerDraftSync(request){if(IS_PORTABLE||!request||!request.reviewer)return;pendingServerDraft=request;if(serverDraftTimer!==null)clearTimeout(serverDraftTimer);setSaveStatus('pending','Browser recovery saved · server journal queued','The immediate browser copy is safe. The reviewer-scoped server recovery and revision journal will now be updated.');serverDraftTimer=setTimeout(()=>flushServerDraft(),350);}
function flushServerDraft(){if(IS_PORTABLE)return Promise.resolve(true);if(serverDraftTimer!==null){clearTimeout(serverDraftTimer);serverDraftTimer=null;}const request=pendingServerDraft;if(!request)return serverDraftChain;pendingServerDraft=null;serverDraftChain=serverDraftChain.catch(()=>false).then(async()=>{setSaveStatus('syncing','Writing recovery journal...','The immediate browser copy remains available while the server-side revision is written.');try{const result=await jsonFetch('/api/draft',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});state=result.state||state;renderState();if(pendingServerDraft)setSaveStatus('pending','Browser recovery saved · newer edit queued');else setSaveStatus('saved',request.delete?'All changes saved · active draft cleared':'All changes saved · server journal current','Green means both the browser copy and reviewer-scoped server revision history are current.');return true;}catch(error){setSaveStatus('error','Browser recovery safe · server journal failed',String(error.message||error));return false;}});return serverDraftChain;}
function saveCurrentDraft(){if(!currentRun||!currentPair)return;choice=aggregateFrameChoice();const content=reviewContent({choice,frame_choices:frameChoiceRecords(),flags:getFlags(),annotations:clone(annotations),phase_quiz_answers:clone(phaseQuizAnswers),phase_quiz_selected_ids:clone(phaseQuizSelectedIds),phase_quiz_rounds:clone(phaseQuizRounds),phase_quiz_per_candidate_limit:phaseQuizSelectedIds.length,notes:byId('notes').value}),expanded=phaseQuizSelectedIds.length>phaseQuizInitialLimit(currentPair),meaningful=Boolean(content.frame_choices.length||content.annotations.length||content.phase_quiz_answers.length||expanded||content.notes.trim()||content.flags.A.length||content.flags.B.length),reviewer=reviewerName(),reviewer_id=reviewerSlug(reviewer||'reviewer'),stamp=item=>({...item,reviewer,reviewer_id}),draft={...content,reviewer,reviewer_id,frame_choices:content.frame_choices.map(stamp),annotations:content.annotations.map(stamp),phase_quiz_answers:content.phase_quiz_answers.map(stamp),frame_index:frameIndex,phase_quiz_index:phaseQuizIndex,phase_quiz_active:phaseQuizActive,updated_utc:utcNow()};if(IS_PORTABLE){if(!activeSession)return;activeSession.current_run=currentRun;const baseline=reviewContent(activeSession.decisions[currentRun]||null);if(meaningful&&JSON.stringify(content)!==JSON.stringify(baseline))activeSession.drafts[currentRun]=draft;else delete activeSession.drafts[currentRun];persistPortableSession();state=portableState();renderState();const durable=STORAGE_PERSISTENT||DATABASE_PERSISTENT;setSaveStatus(durable?'saved':'error',durable?'All changes saved in this browser':'Temporary recovery only · export review','Each edit is copied to browser storage and an append-only browser revision database when available.');return;}const baseline=reviewContent(currentPair.decision||null),changed=meaningful&&JSON.stringify(content)!==JSON.stringify(baseline);if(changed){storageSet(annotationDraftKey(),JSON.stringify(draft));if(reviewer)storageRemove(unassignedAnnotationDraftKey());storageRemove(previousAnnotationDraftKey());storageRemove(legacyAnnotationDraftKey());}else{storageRemove(annotationDraftKey());storageRemove(unassignedAnnotationDraftKey());storageRemove(previousAnnotationDraftKey());storageRemove(legacyAnnotationDraftKey());}if(reviewer)queueServerDraftSync(changed?{run:currentRun,reviewer,draft}:{run:currentRun,reviewer,delete:true});renderState();}
function saveAnnotationDraft(){saveCurrentDraft();}
async function loadPairData(run){return IS_PORTABLE?portablePair(run):jsonFetch(`/api/pair?run=${encodeURIComponent(run)}&reviewer=${encodeURIComponent(reviewerName())}`);}
function cleanPortableFrameChoices(rawChoices,pair,reviewer=''){const frameTotal=Math.min(Number(pair.candidate_a.frame_count)||0,Number(pair.candidate_b.frame_count)||0),allowed=pair.comparison_mode==='single_candidate'?new Set(['acceptable','needs_repair','unresolved']):new Set(['candidate_a','candidate_b','equivalent','unresolved']),seen=new Set(),reviewer_id=reviewer?reviewerSlug(reviewer):'';if(!Array.isArray(rawChoices))throw new Error('Frame choices must be a list');return rawChoices.map(source=>{const index=Number(source&&source.frame_index),value=String(source&&source.choice||'');if(!Number.isInteger(index)||index<0||index>=frameTotal||!allowed.has(value)||seen.has(index))throw new Error(`Invalid frame verdict at index ${index}`);seen.add(index);const elapsed=Number(pair.candidate_a.elapsed_hours&&pair.candidate_a.elapsed_hours[index]);return{frame_index:index,elapsed_h:Number.isFinite(elapsed)?elapsed:null,choice:value,updated_utc:String(source.updated_utc||utcNow()),...(reviewer?{reviewer,reviewer_id}:{})};}).sort((a,b)=>a.frame_index-b.frame_index);}
function cleanPortablePhaseQuizAnswers(rawAnswers,pair){const proposals=phaseQuizStoredProposals(pair),seen=new Set();return(rawAnswers||[]).map(source=>{validatePortablePhaseQuizAnswer(source,pair.run,pair);const side=String(source.candidate).toUpperCase(),key=`${side}:${source.proposal_id}`;if(seen.has(key))throw new Error(`Duplicate phase quiz answer: ${source.proposal_id}`);seen.add(key);const proposal=proposals.find(item=>item.candidate===side&&item.proposal_id===source.proposal_id),expected=PHASE_QUIZ_EXPECTED[proposal.suggested_phase],choice=source.human_choice,legacy=source.phase_label_schema&&source.phase_label_schema!=='fluidflower-visible-phase-v3';return{...clone(proposal),candidate:side,human_choice:choice,phase_label_schema:legacy?source.phase_label_schema:'fluidflower-visible-phase-v3',phase_class:legacy&&source.phase_class?source.phase_class:PHASE_QUIZ_CLASSES[choice],phase_dominance:legacy?(source.phase_dominance||'legacy_unspecified'):PHASE_CHOICE_DOMINANCE[choice],phase_components:legacy?(source.phase_components||[]):PHASE_CHOICE_COMPONENTS[choice],training_eligible:!PHASE_QUIZ_NONSCORING.has(choice),confidence:source.confidence||'medium',agreement:!expected||PHASE_QUIZ_NONSCORING.has(choice)?null:expected.includes(choice),interpretation_scope:'visible_2d_slice_only',comment:String(source.comment||'').trim().slice(0,500),answered_utc:source.answered_utc||utcNow()};});}
function portableDecision(request){
  const run=request.run.toLowerCase(),pair=PORTABLE_BUNDLE.pairs[run],reviewer=request.reviewer.trim();if(!pair)throw new Error(`Unknown run: ${run}`);if(!reviewer)throw new Error('Reviewer is required');if(!['candidate_a','candidate_b','equivalent','acceptable','needs_repair','unresolved'].includes(request.choice))throw new Error('Select a verdict');
  const reviewer_id=reviewerSlug(reviewer);
  const winner=request.choice==='candidate_a'?pair.reveal.A.candidate_id:request.choice==='candidate_b'?pair.reveal.B.candidate_id:null;
  const cleanFrameChoices=cleanPortableFrameChoices(request.frame_choices||[],pair,reviewer),frameTotal=Math.min(pair.candidate_a.frame_count,pair.candidate_b.frame_count),frameCounts=Object.fromEntries([...new Set(cleanFrameChoices.map(item=>item.choice))].map(value=>[value,cleanFrameChoices.filter(item=>item.choice===value).length]));
  const cleanAnnotations=(request.annotations||[]).map(source=>{const item=normalizeAnnotation(source),side=item.candidate,candidate=side==='A'?pair.candidate_a:pair.candidate_b,provenance=pair.reveal[side],elapsed=candidate.elapsed_hours&&candidate.elapsed_hours[item.frame_index];return{...clone(item),reviewer,reviewer_id,candidate_id:provenance.candidate_id,params_sha256:provenance.params_sha256,elapsed_h:Number.isFinite(elapsed)?elapsed:null,coordinate_space:'normalized_native_review_canvas_16x9'};});
  const cleanPhaseQuizAnswers=cleanPortablePhaseQuizAnswers(request.phase_quiz_answers||[],pair).map(item=>({...item,reviewer,reviewer_id}));
  const selectedIds=cleanPhaseQuizSelectedIds(pair,request.phase_quiz_selected_ids,cleanPhaseQuizAnswers),rounds=cleanPhaseQuizRounds(pair,request.phase_quiz_rounds,selectedIds),record={schema:2,run,comparison_mode:pair.comparison_mode||'pair',reviewer,reviewer_id,choice:request.choice,winner_candidate_id:winner,frame_choice_schema:'candidate-frame-preference-v1',frame_choices:cleanFrameChoices,frame_choice_summary:{schema:'candidate-frame-preference-v1',reviewed:cleanFrameChoices.length,total:frameTotal,complete:cleanFrameChoices.length===frameTotal,counts:frameCounts,aggregate_choice:request.choice},flags:clone(request.flags),annotations:cleanAnnotations,phase_quiz_answers:cleanPhaseQuizAnswers,phase_quiz_selected_ids:selectedIds,phase_quiz_rounds:rounds,phase_quiz_per_candidate_limit:selectedIds.length,notes:String(request.notes||'').trim(),updated_utc:utcNow(),candidate_a:clone(pair.reveal.A),candidate_b:clone(pair.reveal.B)};
  activeSession.decisions[run]=record;delete activeSession.drafts[run];activeSession.current_run=run;persistPortableSession();state=portableState();return{status:'saved',record,state,pair:portablePair(run)};
}
async function saveDecisionData(request){return IS_PORTABLE?portableDecision(request):jsonFetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});}
async function initializeReview(){
  if(!IS_PORTABLE){const remembered=storageGet('ffac-reviewer')||'';state=await jsonFetch(`/api/state?reviewer=${encodeURIComponent(remembered)}`);activeReviewerName=remembered||state.default_reviewer||'';byId('reviewer').value=activeReviewerName;updateReviewerGate();if(activeReviewerName)setSaveStatus('saved','Reviewer restored · autosave ready');return;}
  document.querySelectorAll('.portable-only').forEach(element=>element.hidden=false);
  byId('chooseAutosaveFile').hidden=!IS_PORTABLE||typeof window.showSaveFilePicker!=='function';
  const last=storageGet(lastReviewerStorageKey())||'';byId('reviewer').value=last;
  if(last){const key=sessionStorageKey(last),saved=storageGet(key),databaseSaved=saved?null:await databaseSession(key);try{activeSession=saved||databaseSaved?validatePortableSession(saved?JSON.parse(saved):databaseSaved):newPortableSession(last);}catch(error){storageRemove(key);activeSession=newPortableSession(last);setSaveStatus('error',`Local autosave was reset: ${error.message}`);}}
  else activeSession=null;
  state=portableState();updateSessionMeta();updateReviewerGate();if(activeSession){const durable=STORAGE_PERSISTENT||DATABASE_PERSISTENT;setSaveStatus(durable?'saved':'error',durable?'Reviewer restored · continuous browser autosave ready':'Temporary session · export frequently');}
}
async function choosePortableAutosaveFile(){if(!IS_PORTABLE||typeof window.showSaveFilePicker!=='function')throw new Error('This browser does not support direct file autosave');const suggested=exportFilename();portableFileHandle=await window.showSaveFilePicker({suggestedName:suggested,types:[{description:'FluidFlower review JSON',accept:{'application/json':['.json']}}]});if(activeSession){persistPortableSession();await portableFileChain;}updateSessionMeta();setSaveStatus('saved',`Continuous file autosave enabled · ${portableFileHandle.name}`,'Every subsequent change is written to browser storage, the revision database, and this selected JSON file.');}
function exportPortableReview(){const name=byId('reviewer').value.trim();if(!name)throw new Error('Reviewer is required');if(!activeSession||reviewerName()!==name)throw new Error('Open this reviewer before exporting');const saved=Object.keys(activeSession.decisions).length,drafts=Object.keys(activeSession.drafts||{}).length,total=PORTABLE_BUNDLE.runs.length;if((saved<total||drafts>0)&&!confirm(`This review has ${saved}/${total} saved decisions and ${drafts} draft(s). Draft annotations are included, but drafts do not count as completed decisions. Export anyway?`))return;const filename=exportFilename();activeSession.last_exported_utc=utcNow();activeSession.last_export_filename=filename;persistPortableSession();const blob=new Blob([JSON.stringify(activeSession,null,2)],{type:'application/json'}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=filename;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(link.href),1000);setSaveStatus('saved',`Exported ${filename}`);}
async function importPortableReview(file){const payload=validatePortableSession(JSON.parse(await file.text())),name=reviewerName(payload),existing=storageGet(sessionStorageKey(name));if(existing&&!confirm(`Replace the local review for ${name} with the selected file?`))return;storageSet(sessionStorageKey(name),JSON.stringify(payload));storageSet(lastReviewerStorageKey(),name);activeSession=payload;byId('reviewer').value=name;currentRun=null;currentPair=null;persistPortableSession();state=portableState();renderState();await loadPair(activeSession.current_run||state.runs[0].run);setSaveStatus('saved',`Opened review for ${name}`);}
function renderState() {
  const pending=(state.pending_runs||[]).length,live=state.live&&state.live.enabled;
  const localDraftCount=IS_PORTABLE?(state.drafted||0):state.runs.filter(item=>Boolean(item.drafted||draftStorageText(item.run))).length;
  byId('progress').textContent=`${state.reviewed} saved | ${localDraftCount} drafts | ${state.total} ready${live?` | ${pending} processing`:''}`;
  byId('runs').innerHTML=state.runs.map(item => {const local=!IS_PORTABLE?(()=>{const text=draftStorageText(item.run);if(!text)return null;try{const parsed=JSON.parse(text);return Array.isArray(parsed)?{annotations:parsed}:parsed;}catch(_error){return null;}})():null,drafted=item.drafted||Boolean(local),quizAnswers=Number(item.quiz_answered||(local&&Array.isArray(local.phase_quiz_answers)?local.phase_quiz_answers.length:0)),quizTotal=Number(item.quiz_total||(item.run===currentRun&&currentPair?phaseQuizProposals(currentPair).length:0)),quizComplete=Boolean(item.quiz_complete||(quizTotal>0&&quizAnswers>=quizTotal)),frameReviewed=Number(local&&Array.isArray(local.frame_choices)?local.frame_choices.length:item.frame_choices_reviewed||0),frameTotal=Number(item.frame_choices_total||0),frameComplete=frameTotal>0&&frameReviewed>=frameTotal,title=[runDisplay(item.run,item.dataset),frameTotal?`frames ${frameReviewed}/${frameTotal}`:'',quizTotal?`quiz ${quizAnswers}/${quizTotal}`:'',quizComplete?'quiz complete':''].filter(Boolean).join(' | '),suffix=frameComplete?' ✓':quizComplete?' · quiz ✓':'';return `<button type="button" class="run-button ${item.reviewed?'reviewed':''} ${drafted?'drafted':''} ${quizComplete?'quiz-complete':''} ${item.run===currentRun?'active':''}" data-run="${item.run}" title="${title}">${runDisplay(item.run,item.dataset)}${suffix}</button>`;}).join('');
  const messages=[];
  if((state.unpaired_runs||[]).length)messages.push(`No comparison: ${state.unpaired_runs.map(x=>x.toUpperCase()).join(', ')}`);
  if(pending)messages.push(`Waiting for complete postflight assets: ${state.pending_runs.map(x=>x.toUpperCase()).join(', ')}`);
  if(state.live&&state.live.state==='error')messages.push(`Live update error: ${state.live.message}`);
  byId('unpaired').textContent=messages.join('\n');
  document.querySelectorAll('.run-button').forEach(button => button.onclick=()=>loadPair(button.dataset.run));
  updateSessionMeta();updateReviewerGate();
}
async function refreshLiveInventory(){
  if(IS_PORTABLE||livePollBusy||!state||!(state.live&&state.live.enabled))return;
  livePollBusy=true;
  try{
    const next=await jsonFetch(`/api/state?reviewer=${encodeURIComponent(reviewerName())}`),previousRevision=state.inventory_revision;
    state=next;
    if(previousRevision!==next.inventory_revision)renderState();
    else {const localDraftCount=state.runs.filter(item=>Boolean(item.drafted||draftStorageText(item.run))).length;byId('progress').textContent=`${state.reviewed} saved | ${localDraftCount} drafts | ${state.total} ready | ${(state.pending_runs||[]).length} processing`;}
    if(!currentRun&&state.runs.length)await loadPair(state.runs[0].run);
  }catch(error){setSaveStatus('error',`Live update failed; local recovery remains safe`,error.message);}
  finally{livePollBusy=false;}
}
function startLiveInventoryPolling(){
  if(IS_PORTABLE||!state||!(state.live&&state.live.enabled)||livePollTimer!==null)return;
  const milliseconds=Math.max(2000,Number(state.live.poll_seconds||10)*1000);
  livePollTimer=setInterval(refreshLiveInventory,milliseconds);
}
function datasetLabel(value){const key=String(value||'').toLowerCase();if(key==='ffac')return'FF_AC';if(key==='ffum')return'FF_UM';return key?key.toUpperCase():'';}
function runDisplay(run,dataset){const prefix=datasetLabel(dataset);return prefix?`${prefix} · ${String(run).toUpperCase()}`:String(run).toUpperCase();}
function currentIndex() { return state.runs.findIndex(item=>item.run===currentRun); }
function move(delta) { const index=currentIndex(); if(index<0)return; const target=Math.max(0,Math.min(state.runs.length-1,index+delta)); loadPair(state.runs[target].run); }
function frameChoiceValues(){return currentPair&&currentPair.comparison_mode==='single_candidate'?new Set(['acceptable','needs_repair','unresolved']):new Set(['candidate_a','candidate_b','equivalent','unresolved']);}
function setFrameChoices(values){frameChoices=new Map();for(const item of Array.isArray(values)?values:[]){const index=Number(item&&item.frame_index),value=String(item&&item.choice||'');if(Number.isInteger(index)&&index>=0&&index<frameCount&&frameChoiceValues().has(value))frameChoices.set(index,{choice:value,updated_utc:String(item.updated_utc||'')});}renderFrameChoice();}
function frameChoiceRecords(){return [...frameChoices.entries()].sort((a,b)=>a[0]-b[0]).map(([index,item])=>({frame_index:index,elapsed_h:Number(currentPair&&currentPair.candidate_a&&currentPair.candidate_a.elapsed_hours&&currentPair.candidate_a.elapsed_hours[index]),choice:item.choice,updated_utc:item.updated_utc||utcNow()}));}
function aggregateFrameChoice(){if(frameCount<1||frameChoices.size<frameCount||[...frameChoices.values()].some(item=>item.choice==='unresolved'))return'unresolved';const counts={};for(const item of frameChoices.values())counts[item.choice]=(counts[item.choice]||0)+1;if(currentPair&&currentPair.comparison_mode==='single_candidate')return counts.needs_repair?'needs_repair':'acceptable';if((counts.candidate_a||0)>(counts.candidate_b||0))return'candidate_a';if((counts.candidate_b||0)>(counts.candidate_a||0))return'candidate_b';return'equivalent';}
function renderFrameChoice(){const value=frameChoices.get(frameIndex),selected=value&&value.choice||null;choice=aggregateFrameChoice();document.querySelectorAll('#verdict button').forEach(button=>button.classList.toggle('selected',button.dataset.choice===selected));const progress=byId('frameVerdictProgress');if(progress)progress.textContent=`${frameChoices.size} / ${frameCount} frames reviewed${selected?` | current: ${selected.replace(/_/g,' ')}`:' | current: not reviewed'}`;const next=byId('nextUnreviewedFrame');if(next)next.disabled=frameChoices.size>=frameCount;}
function setChoice(value){if(!frameChoiceValues().has(value))return;frameChoices.set(frameIndex,{choice:value,updated_utc:utcNow()});renderFrameChoice();}
function openNextUnreviewedFrame(){if(!frameCount||frameChoices.size>=frameCount)return;for(let offset=1;offset<=frameCount;offset+=1){const index=(frameIndex+offset)%frameCount;if(!frameChoices.has(index)){jumpToFrame(index,'A');return;}}}
function setComparisonMode(mode){const single=mode==='single_candidate';setSwapHeld(false);setContourSwapHeld(false,false);byId('holdSwap').hidden=single;byId('holdContourSwap').hidden=single;byId('comparisonMode').hidden=!single;byId('comparisonMode').textContent=single?'Single available candidate':'';byId('candidateTitleA').textContent=single?'Available candidate':'Candidate A';byId('candidateTitleB').textContent=single?'Same candidate, second view':'Candidate B';document.querySelectorAll('.pair-verdict').forEach(button=>button.hidden=single);document.querySelectorAll('.single-verdict').forEach(button=>button.hidden=!single);byId('verdict').style.gridTemplateColumns=single?'repeat(3,minmax(110px,1fr))':'repeat(4,minmax(110px,1fr))';}
function setFlags(flags={}) { candidateFlags={A:Array.isArray(flags.A)?clone(flags.A):[],B:Array.isArray(flags.B)?clone(flags.B):[]}; }
function getFlags() { return clone(candidateFlags); }
function revealDetails(reveal) {
  const box=byId('reveal');
  if(!reveal){box.style.display='none';box.textContent='';return;}
  const line=(label,c)=>`${label}: ${c.role} | ${c.variant}\n  NMAE: ${c.normalized_mae===null?'n/a':(100*c.normalized_mae).toFixed(2)+'%'} | hash ${c.params_sha256.slice(0,16)}`;
  box.textContent=`Identity revealed after saved decision\n${line('Candidate A',reveal.A)}\n${line('Candidate B',reveal.B)}`;
  box.style.display='block';
}
function stopPlayback() { if(playTimer!==null){clearInterval(playTimer);playTimer=null;} byId('play').textContent='Play'; }
function candidateData(side){return side==='A'?currentPair.candidate_a:currentPair.candidate_b;}
function nativeZoomScale(side){const view=byId(`viewport${side}`).getBoundingClientRect(),image=byId(`frame${side}`),canvas=currentPair&&currentPair.canvas_wh||[],width=image.naturalWidth||Number(canvas[0])||view.width,height=image.naturalHeight||Number(canvas[1])||view.height;return Math.max(1,width/Math.max(1,view.width),height/Math.max(1,view.height));}
function maximumZoom(side){const native=nativeZoomScale(side);return Math.max(32,Math.min(96,native*12));}
function updateZoomReadout(side=activeSide){const t=transforms[side],native=nativeZoomScale(side),screenPixels=t.scale/native,readout=byId('zoomReadout'),pixel=byId('markPixel'),reference=byId('reference').checked;readout.textContent=`${t.scale.toFixed(1)}× | ${screenPixels.toFixed(2)} screen px / image px`;const exactEnough=screenPixels>=0.98;pixel.disabled=reference||!exactEnough;pixel.title=reference?'Pixel labels are unavailable on the reference image':exactEnough?'Mark one exact analysis-image pixel using the selected category':'Select 1 px or zoom further before marking an individual pixel';if(pixel.disabled&&drawTool==='point')setDrawTool('pan');}
function applyTransform(side,updateReadout=true){const t=transforms[side],layer=byId(`layer${side}`);layer.style.transform=`translate3d(${t.x}px,${t.y}px,0) scale(${t.scale})`;if(updateReadout||!layer.dataset.pixelState){const pixelated=t.scale>nativeZoomScale(side)*1.05;layer.classList.toggle('pixel-inspection',pixelated);layer.dataset.pixelState=pixelated?'1':'0';}if(updateReadout&&side===activeSide)updateZoomReadout(side);}
function clampTransform(side,knownView=null){const view=knownView||byId(`viewport${side}`).getBoundingClientRect(),t=transforms[side];t.scale=Math.max(1,Math.min(maximumZoom(side),t.scale));t.x=Math.min(0,Math.max(view.width*(1-t.scale),t.x));t.y=Math.min(0,Math.max(view.height*(1-t.scale),t.y));}
function syncTransform(source,updateReadout=true,knownViews=null){if(!byId('linkViews').checked)return;const other=source==='A'?'B':'A',a=knownViews&&knownViews[source]||byId(`viewport${source}`).getBoundingClientRect(),b=knownViews&&knownViews[other]||byId(`viewport${other}`).getBoundingClientRect(),s=transforms[source];transforms[other]={scale:s.scale,x:s.x*(b.width/a.width),y:s.y*(b.height/a.height)};clampTransform(other,b);applyTransform(other,updateReadout);}
function zoomAt(side,factor,px,py){const t=transforms[side],old=t.scale,next=Math.max(1,Math.min(maximumZoom(side),old*factor)),ratio=next/old;t.x=px-(px-t.x)*ratio;t.y=py-(py-t.y)*ratio;t.scale=next;clampTransform(side);applyTransform(side);syncTransform(side);}
function resetZoom(side=activeSide){transforms[side]={scale:1,x:0,y:0};applyTransform(side);syncTransform(side);}
function nativeZoom(side=activeSide){const view=byId(`viewport${side}`).getBoundingClientRect(),image=byId(`frame${side}`),scale=Math.max(1,Math.min(maximumZoom(side),Math.max(image.naturalWidth/Math.max(1,view.width),image.naturalHeight/Math.max(1,view.height))));transforms[side]={scale,x:view.width*(1-scale)/2,y:view.height*(1-scale)/2};clampTransform(side);applyTransform(side);syncTransform(side);}
function pixelZoom(side=activeSide){const view=byId(`viewport${side}`).getBoundingClientRect(),target=Math.min(maximumZoom(side),nativeZoomScale(side)*8),factor=target/transforms[side].scale;zoomAt(side,factor,view.width/2,view.height/2);}
function currentPhaseQuizProposals(){return phaseQuizProposals(currentPair);}
function phaseQuizIsComplete(proposals=currentPhaseQuizProposals()){return proposals.length>0&&proposals.every(proposal=>Boolean(phaseQuizAnswerFor(proposal)));}
function currentPhaseQuizProposal(){const proposals=currentPhaseQuizProposals();if(!phaseQuizActive||!proposals.length)return null;phaseQuizIndex=Math.max(0,Math.min(proposals.length-1,phaseQuizIndex));return proposals[phaseQuizIndex];}
function phaseQuizAnswerFor(proposal){return proposal?phaseQuizAnswers.find(answer=>answer.candidate===proposal.candidate&&answer.proposal_id===proposal.proposal_id)||null:null;}
function focusPhaseQuizRegion(){const proposal=currentPhaseQuizProposal();if(!proposal)return;stopPlayback();activeSide=proposal.candidate;frameIndex=Math.max(0,Math.min(frameCount-1,Number(proposal.frame_index)));byId('reference').checked=false;byId('contours').checked=false;setContourSwapHeld(false,false);renderFrame({preserveSelection:true});const focus=()=>{const box=proposal.focus_box||{x:proposal.x,y:proposal.y,width:proposal.width,height:proposal.height},view=byId(`viewport${activeSide}`).getBoundingClientRect(),width=Math.max(0.002,Number(box.width)),height=Math.max(0.002,Number(box.height)),scale=Math.max(1,Math.min(maximumZoom(activeSide),Math.min(0.78/width,0.72/height))),centerX=Number(box.x)+width/2,centerY=Number(box.y)+height/2;transforms[activeSide]={scale,x:view.width/2-centerX*view.width*scale,y:view.height/2-centerY*view.height*scale};clampTransform(activeSide);applyTransform(activeSide);syncTransform(activeSide);renderAnnotations();};const image=byId(`frame${activeSide}`);if(image.complete&&image.naturalWidth)focus();else image.addEventListener('load',focus,{once:true});saveCurrentDraft();}
function setPhaseQuizIndex(index,{focus=true}={}){const proposals=currentPhaseQuizProposals();if(!proposals.length)return;phaseQuizIndex=(Number(index)+proposals.length)%proposals.length;phaseQuizActive=true;renderPhaseQuiz();if(focus)focusPhaseQuizRegion();else renderAnnotations();}
function closePhaseQuizRegion(){phaseQuizActive=false;renderFrame({preserveSelection:true});saveCurrentDraft();}
function nextUnansweredPhaseQuiz(){const proposals=currentPhaseQuizProposals();if(!proposals.length)return;for(let offset=phaseQuizActive?1:0;offset<=proposals.length;offset+=1){const index=(phaseQuizIndex+offset)%proposals.length;if(!phaseQuizAnswerFor(proposals[index])){setPhaseQuizIndex(index);return;}}closePhaseQuizRegion();}
function renderPhaseQuizTimelineMarker(proposal){const marker=byId('phaseQuizTimelineMarker');if(!proposal||frameCount<1){marker.hidden=true;return;}const sourceFrame=Math.max(0,Math.min(frameCount-1,Number(proposal.frame_index))),fraction=frameCount>1?sourceFrame/(frameCount-1):0.5,elapsed=Number(proposal.elapsed_h);marker.hidden=false;marker.style.left=`${100*fraction}%`;marker.classList.toggle('current',sourceFrame===frameIndex);marker.title=`Shared phase-quiz region | frame ${sourceFrame+1}${Number.isFinite(elapsed)?` | t = ${elapsed.toFixed(2)} h`:''}`;marker.setAttribute('aria-label',`Open ${marker.title}`);}
function phaseQuizRegionMarkup(side,referenceMode){const proposal=currentPhaseQuizProposal();if(referenceMode||!proposal||proposal.candidate!==side)return'';const sourceFrame=Number(proposal.frame_index),onSourceFrame=sourceFrame===frameIndex,answered=Boolean(phaseQuizAnswerFor(proposal)),title=`Shared visible-2D phase quiz region | source frame ${sourceFrame+1} | viewing frame ${frameIndex+1}`;return annotationShapeMarkup(proposal,`phase-quiz-region ${answered?'answered':''} ${onSourceFrame?'source-frame':'temporal'}`,'',title);}
function injectionPointMarkup(){if(!currentPair||!byId('injectionPoints').checked)return'';const points=Array.isArray(currentPair.injection_points)?currentPair.injection_points:[];return points.map((point,index)=>{const windows=Array.isArray(point.active_windows_h)?point.active_windows_h.map(window=>`${Number(window[0]).toFixed(2)}-${Number(window[1]).toFixed(2)} h`).join(', '):'',position=Number.isFinite(Number(point.x_m))&&Number.isFinite(Number(point.y_m))?` | (${Number(point.x_m).toFixed(2)}, ${Number(point.y_m).toFixed(2)}) m`:'',source=point.coordinate_source?` | source: ${point.coordinate_source}`:'',title=`${point.label||`Injector ${index+1}`}${windows?` | active ${windows}`:''}${position}${source}`;return`<ellipse class="injection-point" cx="${Number(point.x)}" cy="${Number(point.y)}" rx="0.007" ry="0.012"><title>${title}</title></ellipse>`;}).join('');}
function renderPhaseQuizAnswerList(proposals){const root=byId('phaseQuizAnswerList');root.innerHTML='';const answered=proposals.map((proposal,index)=>({proposal,index,answer:phaseQuizAnswerFor(proposal)})).filter(item=>item.answer);if(!answered.length){const empty=document.createElement('div');empty.className='phase-quiz-answer-meta';empty.textContent='No validated regions yet.';root.appendChild(empty);return;}for(const item of answered){const proposal=item.proposal,answer=item.answer,elapsed=Number(proposal.elapsed_h),button=document.createElement('button');button.type='button';button.className=`phase-quiz-answer-item ${phaseQuizActive&&item.index===phaseQuizIndex?'active':''}`;const main=document.createElement('span');main.className='phase-quiz-answer-main';main.textContent=`Frame ${Number(proposal.frame_index)+1}${Number.isFinite(elapsed)?` | ${elapsed.toFixed(2)} h`:''}`;const meta=document.createElement('span');meta.className='phase-quiz-answer-meta';meta.textContent=`${PHASE_QUIZ_LABELS[answer.human_choice]||answer.human_choice} | confidence ${answer.confidence||'medium'}`;button.append(main,meta);button.title='Open this validated region and its source frame';button.onclick=()=>setPhaseQuizIndex(item.index);root.appendChild(button);}}
function requestedPhaseQuizCount(){const field=byId('phaseQuizAddCount'),raw=String(field&&field.value||'').trim(),valid=/^[1-9]\d*$/.test(raw),number=valid?Number(raw):NaN,requested=Number.isSafeInteger(number)?number:Number.MAX_SAFE_INTEGER;field.classList.toggle('invalid',!valid);field.setAttribute('aria-invalid',valid?'false':'true');return valid?requested:null;}
function renderPhaseQuizScope(proposals){const button=byId('phaseQuizNewRound'),field=byId('phaseQuizAddCount'),label=byId('phaseQuizAvailability'),available=phaseQuizAvailableLimit(currentPair),reserve=Math.max(0,available-proposals.length),requested=requestedPhaseQuizCount(),count=requested===null?0:Math.min(requested,reserve);field.disabled=reserve<=0;button.disabled=reserve<=0||requested===null;button.textContent=reserve<=0?'All prepared regions added':requested===null?'Enter a positive whole number':`Add strategic region${count===1?'':'s'}: ${count}`;label.textContent=!available?'No proposal pool prepared':reserve<=0?`${proposals.length} distinct regions are active; no unused prepared combinations remain`:requested>reserve?`${reserve} unused regions remain; all ${reserve} will be added from the requested ${requested}.`:`${reserve} unused phase/facies/time/position-balanced regions remain. More can be added before the active set is complete.`;}
function startAdaptivePhaseQuizRound(){if(!currentPair)return;const requested=requestedPhaseQuizCount();if(requested===null){renderPhaseQuizScope(currentPhaseQuizProposals());return;}const previous=currentPhaseQuizProposals(),reserve=Math.max(0,phaseQuizAvailableLimit(currentPair)-previous.length),count=Math.min(requested,reserve),strategic=strategicPhaseQuizBatch(currentPair,phaseQuizSelectedIds,count)||[],afterStrategic=phaseQuizSelectedIds.concat(strategic.map(item=>item.proposal_id)),fallback=strategic.length<count?adaptivePhaseQuizBatch(currentPair,afterStrategic,phaseQuizAnswers,count-strategic.length):[],batch=strategic.concat(fallback);if(!batch.length){byId('status').textContent='No additional distinct prepared quiz regions remain.';return;}const ids=batch.map(item=>item.proposal_id);phaseQuizSelectedIds.push(...ids);phaseQuizPerCandidateLimit=phaseQuizSelectedIds.length;phaseQuizRounds.push({round:phaseQuizRounds.length+1,selection_mode:strategic.length?'adaptive_phase_facies_time_position_with_optical_fill':'adaptive_corrected_optical_change',proposal_ids:ids,created_utc:utcNow()});phaseQuizIndex=previous.length;phaseQuizActive=true;focusPhaseQuizRegion();if(batch.length<requested)byId('status').textContent=`Added ${batch.length} quiz regions from the ${requested} requested; no more distinct eligible regions were available.`;}
function renderPhaseQuiz(){const proposals=currentPhaseQuizProposals(),allProposals=phaseQuizAllProposals(currentPair),reserve=Math.max(0,allProposals.length-proposals.length),section=byId('phaseQuiz'),progress=byId('phaseQuizProgress'),context=byId('phaseQuizContext'),reveal=byId('phaseQuizReveal'),completeBox=byId('phaseQuizComplete'),controls=['phaseQuizPrevious','phaseQuizFocus','phaseQuizClose','phaseQuizNext','phaseQuizUnanswered','phaseQuizDraw','phaseQuizIgnore','phaseQuizClear'];renderPhaseQuizAnswerList(proposals);renderPhaseQuizScope(proposals);if(!proposals.length){section.classList.remove('complete');progress.textContent='No regions prepared';context.textContent='Shared phase quiz regions have not been generated for this experiment.';reveal.textContent='';completeBox.hidden=true;renderPhaseQuizTimelineMarker(null);for(const id of controls)byId(id).disabled=true;byId('phaseQuizConfidence').disabled=true;byId('phaseQuizComment').disabled=true;document.querySelectorAll('[data-phase-choice]').forEach(button=>{button.disabled=true;button.classList.remove('selected');});return;}const answered=proposals.filter(item=>phaseQuizAnswerFor(item)).length,complete=answered===proposals.length,proposal=currentPhaseQuizProposal();section.classList.toggle('complete',complete);completeBox.hidden=!complete;completeBox.textContent=complete?`Quiz round complete for ${runDisplay(currentRun,currentPair&&currentPair.dataset)}: all ${proposals.length} active shared regions are validated.${reserve>0?' Choose how many additional strategic regions to add above; corrected optical change and your completed answers guide the selection.':''} Choose an experiment verdict and save when the remaining review is complete.`:'';progress.textContent=complete?`Complete | ${answered} / ${proposals.length} validated`:`${answered} / ${proposals.length} answered${proposal?` | region ${phaseQuizIndex+1}`:''}`;if(!proposal){context.textContent=complete?'The active outline is closed. Select any item in the validated-regions list to review it again.':'No region is active. Select Next unanswered or an item in the validated-regions list.';reveal.className='phase-quiz-reveal';reveal.textContent=complete?'No quiz box remains pinned while you browse the experiment.':'';renderPhaseQuizTimelineMarker(null);byId('phaseQuizPrevious').disabled=false;byId('phaseQuizNext').disabled=false;byId('phaseQuizUnanswered').disabled=complete;for(const id of ['phaseQuizFocus','phaseQuizClose','phaseQuizDraw','phaseQuizIgnore','phaseQuizClear'])byId(id).disabled=true;byId('phaseQuizConfidence').disabled=true;byId('phaseQuizComment').disabled=true;byId('phaseQuizComment').value='';document.querySelectorAll('[data-phase-choice]').forEach(button=>{button.disabled=true;button.classList.remove('selected');});return;}for(const id of controls)byId(id).disabled=false;const sourceFrame=Number(proposal.frame_index),onSourceFrame=sourceFrame===frameIndex,answer=phaseQuizAnswerFor(proposal),ignored=Boolean(answer&&answer.human_choice==='ignore_region'),elapsed=Number(proposal.elapsed_h),candidate=candidateData(proposal.candidate),viewElapsed=Number(candidate&&candidate.elapsed_hours&&candidate.elapsed_hours[frameIndex]);renderPhaseQuizTimelineMarker(proposal);context.textContent=onSourceFrame?`Shared optical quiz | source frame ${sourceFrame+1} | ${Number.isFinite(elapsed)?`t = ${elapsed.toFixed(2)} h | `:''}classify only the visible evidence inside the white outline in this 2D slice.`:`Shared optical quiz | source frame ${sourceFrame+1} | viewing frame ${frameIndex+1}${Number.isFinite(viewElapsed)?` at t = ${viewElapsed.toFixed(2)} h`:''} | temporal comparison only; Focus region returns to the source frame for classification.`;document.querySelectorAll('[data-phase-choice]').forEach(button=>{button.disabled=!onSourceFrame;button.classList.toggle('selected',Boolean(answer)&&button.dataset.phaseChoice===answer.human_choice);});byId('phaseQuizIgnore').disabled=!onSourceFrame;byId('phaseQuizIgnore').classList.toggle('active',ignored);byId('phaseQuizClear').disabled=!onSourceFrame||!answer;byId('phaseQuizConfidence').disabled=!onSourceFrame||ignored;byId('phaseQuizComment').disabled=!onSourceFrame;byId('phaseQuizConfidence').value=answer&&answer.confidence||'medium';byId('phaseQuizComment').value=answer&&answer.comment||'';reveal.className='phase-quiz-reveal';if(!answer){reveal.textContent=onSourceFrame?'The model phase is hidden. The outlined region remains visible while you inspect its temporal evolution.':'Temporal comparison only. The same focus region is pinned to its image coordinates; use Focus region to return and answer.';return;}if(ignored){reveal.textContent='This region is ignored and excluded from transferable phase training. Clear answer makes it available again.';return;}const expected=PHASE_QUIZ_EXPECTED[proposal.suggested_phase],agreement=!expected||PHASE_QUIZ_NONSCORING.has(answer.human_choice)?null:expected.includes(answer.human_choice),agreementClass=agreement===true?'agree':agreement===false?'disagree':null;if(agreementClass)reveal.classList.add(agreementClass);reveal.textContent=`Your answer: ${PHASE_QUIZ_LABELS[answer.human_choice]} | model information after answer: ${PHASE_QUIZ_MODEL_LABELS[proposal.suggested_phase]||proposal.suggested_phase}${agreement===null?'':agreement?' | agreement':' | disagreement'}`;}
function answerPhaseQuiz(humanChoice){const proposal=currentPhaseQuizProposal();if(!proposal||Number(proposal.frame_index)!==frameIndex||!(humanChoice in PHASE_QUIZ_LABELS))return;const answer={proposal_id:proposal.proposal_id,candidate:proposal.candidate,human_choice:humanChoice,confidence:humanChoice==='ignore_region'?'low':byId('phaseQuizConfidence').value,comment:byId('phaseQuizComment').value.trim(),answered_utc:utcNow()};phaseQuizAnswers=phaseQuizAnswers.filter(item=>!(item.candidate===answer.candidate&&item.proposal_id===answer.proposal_id));phaseQuizAnswers.push(answer);if(phaseQuizIsComplete())phaseQuizActive=false;renderFrame({preserveSelection:true});saveCurrentDraft();}
function clearPhaseQuizAnswer(){const proposal=currentPhaseQuizProposal();if(!proposal||Number(proposal.frame_index)!==frameIndex)return;phaseQuizAnswers=phaseQuizAnswers.filter(item=>!(item.candidate===proposal.candidate&&item.proposal_id===proposal.proposal_id));renderFrame({preserveSelection:true});saveCurrentDraft();}
function updateCurrentPhaseQuizAnswer(field,value){const proposal=currentPhaseQuizProposal(),answer=phaseQuizAnswerFor(proposal);if(!answer)return;answer[field]=value;answer.answered_utc=utcNow();renderPhaseQuiz();saveCurrentDraft();}
function drawPhaseQuizDetail(){const proposal=currentPhaseQuizProposal();if(!proposal)return;activeSide=proposal.candidate;const answer=phaseQuizAnswerFor(proposal),category=answer&&Object.keys(PHASE_CATEGORY_TO_CHOICE).find(key=>PHASE_CATEGORY_TO_CHOICE[key]===answer.human_choice);if(category){byId('annotationCategory').value=category;byId('annotationCategory').dispatchEvent(new Event('change'));}setDrawTool('freehand');}
function pointInImage(event,side,knownRect=null){const rect=knownRect||byId(`viewport${side}`).getBoundingClientRect(),t=transforms[side];return{x:Math.max(0,Math.min(1,(event.clientX-rect.left-t.x)/(rect.width*t.scale))),y:Math.max(0,Math.min(1,(event.clientY-rect.top-t.y)/(rect.height*t.scale)))};}
function annotationById(id){return annotations.find(item=>item.id===id)||null;}
function normalizeAnnotation(item){const clean={...item};const historicalCandidateIssue=((clean.category==='unsupported_free_gas_classification'&&clean.training_phase_class==='gas_absent_region')||(clean.category==='unsupported_aqueous_co2_classification'&&clean.training_phase_class==='aqueous_absent_region'))&&!clean.training_label_policy_version;if(!historicalCandidateIssue){for(const key of ['evidence_scope','co2_evidence','training_role','training_phase_class','training_scope','evaluation_scope','artifact_origin','interpretation_scope','training_label_policy_version'])delete clean[key];Object.assign(clean,visiblePhaseFields(clean.category),annotationCategorySemantics(clean.category));}clean.phase=clean.phase||CATEGORY_PHASE_DEFAULTS[clean.category]||'unspecified';clean.severity=clean.severity||'major';clean.confidence=clean.confidence||'unspecified';clean.comment=clean.comment||'';clean.shape=['freehand','point'].includes(clean.shape)?clean.shape:'rectangle';clean.points=clean.shape==='freehand'&&Array.isArray(clean.points)?clean.points.map(point=>[Number(point[0]),Number(point[1])]):[];return clean;}
function annotationElapsed(item){const candidate=currentPair?(item.candidate==='A'?currentPair.candidate_a:currentPair.candidate_b):null,value=candidate&&Array.isArray(candidate.elapsed_hours)?candidate.elapsed_hours[item.frame_index]:null;return Number.isFinite(value)?`${Number(value).toFixed(2)} h`:`frame ${item.frame_index+1}`;}
function categoryClass(category){return String(category||'other').replaceAll('_','-');}
function pointString(points){return(points||[]).map(point=>`${Number(point[0])},${Number(point[1])}`).join(' ');}
function annotationShapeMarkup(item,className,attributes,title){const common=`class="${className}" ${attributes} vector-effect="non-scaling-stroke"`;if(item.shape==='freehand')return`<polygon ${common} points="${pointString(item.points)}"><title>${title}</title></polygon>`;return`<rect ${common} x="${item.x}" y="${item.y}" width="${item.width}" height="${item.height}"><title>${title}</title></rect>`;}
function annotationShapeLabel(item){if(item.shape==='point')return`Pixel (${item.analysis_pixel_x}, ${item.analysis_pixel_y})`;return item.shape==='freehand'?'Freehand':'Box';}
function updateAnnotationEditor(){const item=annotationById(selectedAnnotationId),editor=byId('annotationEditor');editor.classList.toggle('visible',Boolean(item));if(!item)return;byId('annotationLabel').textContent=`Candidate ${item.candidate} | ${annotationElapsed(item)} | ${annotationShapeLabel(item)}`;byId('selectedAnnotationCategory').value=item.category;byId('selectedAnnotationPhase').value=item.phase||'unspecified';byId('selectedAnnotationSeverity').value=item.severity||'major';byId('selectedAnnotationConfidence').value=item.confidence||'unspecified';byId('annotationComment').value=item.comment||'';}
function renderAnnotationList(){const root=byId('annotationList');root.innerHTML='';if(!annotations.length){const empty=document.createElement('div');empty.className='annotation-item-meta';empty.textContent='No annotations';root.appendChild(empty);return;}const ordered=[...annotations].sort((left,right)=>left.frame_index-right.frame_index||left.candidate.localeCompare(right.candidate));for(const item of ordered){const row=document.createElement('div');row.className=`annotation-item ${item.id===selectedAnnotationId?'selected':''}`;row.tabIndex=0;row.setAttribute('role','button');const text=document.createElement('div');text.className='annotation-item-main';const title=document.createElement('div');title.textContent=`${item.candidate} | ${annotationElapsed(item)} | ${CATEGORY_LABELS[item.category]||item.category}`;const meta=document.createElement('div');meta.className='annotation-item-meta';meta.textContent=`${annotationShapeLabel(item)} | ${PHASE_LABELS[item.phase]||item.phase} | ${SEVERITY_LABELS[item.severity]||item.severity} | confidence ${CONFIDENCE_LABELS[item.confidence]||item.confidence}`;text.append(title,meta);const remove=document.createElement('button');remove.type='button';remove.className='annotation-delete';remove.title='Delete this annotation';remove.setAttribute('aria-label','Delete this annotation');remove.textContent='×';remove.onclick=event=>{event.stopPropagation();removeAnnotation(item.id);};row.onclick=()=>selectAnnotation(item.id);row.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectAnnotation(item.id);}};row.append(text,remove);root.appendChild(row);}}
function renderAnnotations(){
  const undo=byId('undoBox');
  undo.disabled=annotations.length===0;
  undo.title=undo.disabled?'No annotations to undo':'Remove the most recently added annotation';
  byId('comparison').classList.toggle('shared-phase-quiz',Boolean(currentPhaseQuizProposal()));
  const referenceMode=Boolean(currentPair&&byId('reference').checked&&currentPair.reference_ready);
  for(const side of ['A','B']){
    const svg=byId(`annotations${side}`),visible=referenceMode?[]:annotations.filter(item=>item.candidate===side&&item.frame_index===frameIndex);
    svg.innerHTML=injectionPointMarkup()+phaseQuizRegionMarkup(side,referenceMode)+visible.map(item=>{const selected=item.id===selectedAnnotationId,title=`${CATEGORY_LABELS[item.category]} | ${annotationShapeLabel(item)} | ${PHASE_LABELS[item.phase]||item.phase} | confidence ${CONFIDENCE_LABELS[item.confidence]||item.confidence}`,hit=annotationShapeMarkup(item,'annotation-hit',`data-id="${item.id}" style="stroke-width:14px"`,title),outline=annotationShapeMarkup(item,`annotation-box ${item.shape} ${categoryClass(item.category)} ${item.severity} ${selected?'selected':''}`,'',title);return`<g>${hit}${outline}</g>`;}).join('');
  }
  renderAnnotationList();
  renderPhaseQuiz();
  if(referenceMode){byId('annotationSummary').textContent=`${annotations.length} total | reference view`;byId('annotationEditor').classList.remove('visible');return;}
  const current=annotations.filter(item=>item.frame_index===frameIndex).length;
  byId('annotationSummary').textContent=`${annotations.length} total | ${current} on frame`;
  updateAnnotationEditor();
}
function selectAnnotation(id){const item=annotationById(id);if(!item)return;selectedAnnotationId=id;activeSide=item.candidate;if(byId('reference').checked)byId('reference').checked=false;if(frameIndex!==item.frame_index){frameIndex=item.frame_index;renderFrame({preserveSelection:true});}else renderAnnotations();}
function setDrawTool(tool){drawTool=['pan','rectangle','freehand','point'].includes(tool)?tool:'pan';byId('markBox').classList.toggle('active',drawTool==='rectangle');byId('markFreehand').classList.toggle('active',drawTool==='freehand');byId('markPixel').classList.toggle('active',drawTool==='point');byId('panView').classList.toggle('active',drawTool==='pan');document.querySelectorAll('.frame-shell').forEach(view=>view.classList.toggle('drawing',drawTool!=='pan'));}
function annotationMetadata(side){const category=byId('annotationCategory').value,phaseMapOnly=byId('phaseMapOnly').checked&&byId('contours').checked&&!byId('reference').checked;return{id:`${Date.now().toString(36)}-${Math.random().toString(36).slice(2,8)}`,candidate:side,frame_index:frameIndex,category,phase:CATEGORY_PHASE_DEFAULTS[category]||byId('annotationPhase').value,severity:byId('annotationSeverity').value,confidence:byId('annotationConfidence').value,contours_visible:byId('contours').checked,display_mode:phaseMapOnly?'phase_map_on_black':byId('contours').checked?'phase_map_overlay':'image_only',small_detection_highlight:phaseMapOnly&&byId('highlightSmall').checked,small_detection_highlight_contract:'isolated_from_major_support_v2',small_detection_max_area_fraction:2e-5,small_detection_min_separation_fraction:4e-3,comment:'',...annotationCategorySemantics(category)};}
function storeAnnotation(item){annotations.push(item);selectedAnnotationId=item.id;saveAnnotationDraft();renderAnnotations();}
function phaseAnnotationMetadata(side){const base=annotationMetadata(side);return{...base,...visiblePhaseFields(base.category)};}
function addRectangleAnnotation(side,start,end){if(byId('reference').checked)return;const x=Math.min(start.x,end.x),y=Math.min(start.y,end.y),width=Math.abs(end.x-start.x),height=Math.abs(end.y-start.y);if(width<0.004||height<0.004)return;storeAnnotation({...phaseAnnotationMetadata(side),shape:'rectangle',points:[],x,y,width,height});}
function simplifyFreehandPoints(points){const kept=[];for(const point of points){const prior=kept[kept.length-1];if(!prior||Math.hypot(point.x-prior.x,point.y-prior.y)>=0.002)kept.push(point);}return kept.slice(0,1000);}
function addFreehandAnnotation(side,rawPoints){if(byId('reference').checked)return;const clean=simplifyFreehandPoints(rawPoints);if(clean.length<3)return;const xs=clean.map(point=>point.x),ys=clean.map(point=>point.y),x=Math.min(...xs),y=Math.min(...ys),width=Math.max(...xs)-x,height=Math.max(...ys)-y;if(width<0.004||height<0.004)return;storeAnnotation({...phaseAnnotationMetadata(side),shape:'freehand',points:clean.map(point=>[point.x,point.y]),x,y,width,height});}
function nativePixelMetadata(point){const canvas=currentPair&&currentPair.canvas_wh||[],content=currentPair&&currentPair.content_box_xywh||[],shape=currentPair&&currentPair.analysis_image_shape_hw||[];if(canvas.length!==2||content.length!==4||shape.length!==2)return null;const reviewCanvasWidth=Number(canvas[0]),reviewCanvasHeight=Number(canvas[1]),boxX=Number(content[0]),boxY=Number(content[1]),boxWidth=Number(content[2]),boxHeight=Number(content[3]),analysisImageHeight=Number(shape[0]),analysisImageWidth=Number(shape[1]);if(![reviewCanvasWidth,reviewCanvasHeight,boxX,boxY,boxWidth,boxHeight,analysisImageHeight,analysisImageWidth].every(Number.isFinite))return null;const canvasPixelX=Math.max(0,Math.min(reviewCanvasWidth-1,Math.floor(point.x*reviewCanvasWidth))),canvasPixelY=Math.max(0,Math.min(reviewCanvasHeight-1,Math.floor(point.y*reviewCanvasHeight)));if(canvasPixelX<boxX||canvasPixelY<boxY||canvasPixelX>=boxX+boxWidth||canvasPixelY>=boxY+boxHeight)return null;const analysisPixelX=Math.max(0,Math.min(analysisImageWidth-1,Math.floor((canvasPixelX-boxX)*analysisImageWidth/boxWidth))),analysisPixelY=Math.max(0,Math.min(analysisImageHeight-1,Math.floor((canvasPixelY-boxY)*analysisImageHeight/boxHeight)));return{analysis_pixel_x:analysisPixelX,analysis_pixel_y:analysisPixelY,analysis_image_width:analysisImageWidth,analysis_image_height:analysisImageHeight,canvas_pixel_x:canvasPixelX,canvas_pixel_y:canvasPixelY,review_canvas_width:reviewCanvasWidth,review_canvas_height:reviewCanvasHeight,x:canvasPixelX/reviewCanvasWidth,y:canvasPixelY/reviewCanvasHeight,width:1/reviewCanvasWidth,height:1/reviewCanvasHeight};}
function addPointAnnotation(side,point){if(byId('reference').checked)return;const pixel=nativePixelMetadata(point);if(!pixel){byId('status').textContent='Select a pixel inside the corrected analysis image, not the header or margin.';return;}storeAnnotation({...phaseAnnotationMetadata(side),shape:'point',points:[],...pixel});byId('status').textContent=`Marked analysis pixel (${pixel.analysis_pixel_x}, ${pixel.analysis_pixel_y}).`;}
function removeAnnotation(id){if(!id)return;annotations=annotations.filter(item=>item.id!==id);if(selectedAnnotationId===id)selectedAnnotationId=null;saveAnnotationDraft();renderAnnotations();}
function removeSelected(){removeAnnotation(selectedAnnotationId);}
function updateSelectedAnnotation(field,value){const item=annotationById(selectedAnnotationId);if(!item)return;item[field]=value;if(field==='category'){for(const key of ['evidence_scope','co2_evidence','training_role','training_phase_class','training_scope','evaluation_scope','artifact_origin','interpretation_scope','training_label_policy_version','human_choice','phase_label_schema','phase_class','phase_dominance','phase_components','training_eligible'])delete item[key];Object.assign(item,visiblePhaseFields(value),annotationCategorySemantics(value));if(CATEGORY_PHASE_DEFAULTS[value])item.phase=CATEGORY_PHASE_DEFAULTS[value];}saveAnnotationDraft();renderAnnotations();}
function frameAsset(candidate,index){if(Array.isArray(candidate.plain_frames))return candidate.plain_frames[index];return`${candidate.plain_frame_url}?index=${index}`;}
function overlayAsset(candidate,index){const phase=byId('contourMode')?byId('contourMode').value:'all',fills=candidate.phase_fill_overlays;if(candidate.phase_fill_overlays_ready&&fills&&typeof fills==='object'){const selected=Array.isArray(fills[phase])?fills[phase]:fills.all;if(Array.isArray(selected))return selected[index];}if(Array.isArray(candidate.contour_overlays))return candidate.contour_overlays[index];return`${candidate.contour_overlay_url}?index=${index}&phase=${encodeURIComponent(phase)}`;}
function isolatedComponentUrl(candidateSide,index){const phase=byId('contourMode')?byId('contourMode').value:'all';return`/api/isolated-components?run=${encodeURIComponent(currentRun)}&candidate=${encodeURIComponent(candidateSide)}&index=${index}&phase=${encodeURIComponent(phase)}`;}
function isolatedMarkerMarkup(components){return(components||[]).map(component=>{const cx=Number(component.center_x),cy=Number(component.center_y),rawWidth=Number(component.width),rawHeight=Number(component.height),width=Math.max(0.008,rawWidth+0.004),height=Math.max(0.014,rawHeight+0.007),x=Math.max(0,Math.min(1-width,cx-width/2)),y=Math.max(0,Math.min(1-height,cy-height/2));return`<rect class="isolated-marker" x="${x}" y="${y}" width="${width}" height="${height}" rx="0.003"><title>Disconnected island: ${Number(component.area_px)||0} phase-map pixels</title></rect>`;}).join('');}
async function renderIsolatedMarkers(panelSide,candidateSide,enabled,generation){const root=byId(`isolatedMarkers${panelSide}`);root.innerHTML='';root.hidden=!enabled;if(!enabled||IS_PORTABLE)return;const url=isolatedComponentUrl(candidateSide,frameIndex);let request=isolatedComponentCache.get(url);if(!request){request=jsonFetch(url);isolatedComponentCache.set(url,request);while(isolatedComponentCache.size>64)isolatedComponentCache.delete(isolatedComponentCache.keys().next().value);}try{const payload=await request;if(generation!==isolatedMarkerGeneration||!currentPair||!byId('highlightSmall').checked||!byId('phaseMapOnly').checked)return;root.innerHTML=isolatedMarkerMarkup(payload.components);root.hidden=false;}catch(_error){isolatedComponentCache.delete(url);root.innerHTML='';root.hidden=true;}}
function assignImageAsset(image,url){if(!url){image.removeAttribute('src');image.dataset.assetUrl='';return;}if(image.dataset.assetUrl===url)return;image.dataset.assetUrl=url;image.decoding='async';image.src=url;}
function preloadImageAsset(url){if(!url||framePreloadCache.has(url))return;const image=new Image();image.decoding='async';image.src=url;framePreloadCache.set(url,image);if(typeof image.decode==='function')image.decode().catch(()=>{});while(framePreloadCache.size>8){const oldest=framePreloadCache.keys().next().value,entry=framePreloadCache.get(oldest);framePreloadCache.delete(oldest);if(entry)entry.src='';}}
function scheduleNearbyFramePreload(showContours){if(framePreloadTimer!==null)clearTimeout(framePreloadTimer);const run=currentRun,index=frameIndex;framePreloadTimer=setTimeout(()=>{framePreloadTimer=null;if(!currentPair||currentRun!==run||frameIndex!==index)return;for(const offset of [1,-1,2]){const next=index+offset;if(next<0||next>=frameCount)continue;preloadImageAsset(frameAsset(currentPair.candidate_a,next));if(showContours&&!currentPhaseQuizProposal()){preloadImageAsset(overlayAsset(currentPair.candidate_a,next));preloadImageAsset(overlayAsset(currentPair.candidate_b,next));}}},80);}
function versionedAsset(url,version){if(!version)return url;return`${url}${url.includes('?')?'&':'?'}v=${encodeURIComponent(version)}`;}
function referenceAsset(candidate,index){if(Array.isArray(candidate.reference_frames))return versionedAsset(candidate.reference_frames[index],candidate.reference_version);if(candidate.reference_frame_url)return versionedAsset(`${candidate.reference_frame_url}?index=${index}`,candidate.reference_version);return versionedAsset(currentPair.reference_url,currentPair.reference_version);}
function inspectionAvailability(){if(!currentPair)return;const referenceReady=Boolean(currentPair.reference_ready&&currentPair.candidate_a.reference_frames_ready&&currentPair.candidate_b.reference_frames_ready),referenceMode=referenceReady&&byId('reference').checked,cleanReady=currentPair.candidate_a.plain_frames_ready&&currentPair.candidate_b.plain_frames_ready,overlayAvailable=candidate=>Boolean(candidate.phase_fill_overlays_ready||candidate.contour_overlays_ready),overlayReady=overlayAvailable(currentPair.candidate_a)&&overlayAvailable(currentPair.candidate_b),contourSwapReady=overlayReady&&!referenceMode&&byId('contours').checked&&!byId('holdContourSwap').hidden,phaseMapOnlyReady=overlayReady&&!referenceMode&&byId('contours').checked&&byId('phaseMapOnly').checked,injectorsReady=Array.isArray(currentPair.injection_points)&&currentPair.injection_points.length>0;byId('reference').disabled=!referenceReady;byId('reference').title=referenceReady?'Show the corrected pre-injection image aligned to the current frame':'Aligned pre-injection reference cache is not ready';byId('contours').disabled=referenceMode||!overlayReady;byId('contours').title=referenceMode?'The phase map does not apply to the pre-injection reference':overlayReady?'Show or hide the filled model phase map':'Phase-map cache is not ready';byId('phaseMapOnly').disabled=referenceMode||!overlayReady||!byId('contours').checked;byId('phaseMapOnly').title=referenceMode?'The phase map does not apply to the pre-injection reference':overlayReady?'Hide the photograph and show the selected phase map over black':'Phase-map cache is not ready';byId('highlightSmall').disabled=!phaseMapOnlyReady||IS_PORTABLE;byId('highlightSmall').title=IS_PORTABLE?'Use the desktop app or live review server for disconnected-island diagnostics':phaseMapOnlyReady?'Mark only small CO2-support islands disconnected from the main plume; predicted area is unchanged':'Enable black-background phase-map mode first';byId('contourMode').disabled=referenceMode||!overlayReady||!byId('contours').checked;byId('injectionPoints').disabled=!injectorsReady;byId('injectionPoints').title=injectorsReady?'Show the injection-source anchors used by phase physics; hover a marker for provenance':'Injection-source coordinates are not available';byId('holdContourSwap').disabled=!contourSwapReady;byId('holdContourSwap').title=contourSwapReady?"Hold to show each candidate with the other candidate's phase map":'Enable phase maps to compare the candidate classifications';byId('play').disabled=referenceMode;byId('scrub').disabled=referenceMode;byId('markBox').disabled=referenceMode;byId('markFreehand').disabled=referenceMode;byId('zoomNative').disabled=!cleanReady;byId('zoomPixel').disabled=!cleanReady;updateZoomReadout(activeSide);}
function frameSourceMetadata(index){return currentPair&&Array.isArray(currentPair.frame_sources)?currentPair.frame_sources[index]||null:null;}
function frameSourceSummary(index){const source=frameSourceMetadata(index);if(!source)return'';const parts=[];if(source.selection_status==='pre_injection_negative_control')parts.push('Pre-injection negative control');if(source.source_name)parts.push(source.source_name);const raw=Number(source.color_correction_residual_before_correction),refinementBefore=Number(source.color_correction_residual_before_refinement),residual=Number(source.color_correction_residual),worst=Number(source.color_correction_max_swatch_residual),threshold=Number(source.color_correction_threshold),strategy=String(source.color_correction_strategy||'').trim(),before=Number.isFinite(raw)?raw:refinementBefore;if(strategy)parts.push(`CC ${strategy}`);if(Number.isFinite(residual)){const change=Number.isFinite(before)&&Math.abs(before-residual)>5e-5?`${before.toFixed(4)}→${residual.toFixed(4)}`:residual.toFixed(4);parts.push(`CC residual ${change}${source.color_correction_flagged||Number.isFinite(threshold)&&residual>threshold?' flagged':''}`);}if(Number.isFinite(worst))parts.push(`worst swatch ${worst.toFixed(4)}`);return parts.join(' | ');}
function frameMetricLabel(candidate,index,showOverlay){const metric=Array.isArray(candidate.frame_metrics)?candidate.frame_metrics[index]:null,source=frameSourceSummary(index),prefix=`Frame ${index+1}${source?` | ${source}`:''}`;if(!metric)return`${prefix} | ${showOverlay?'native-resolution image + filled model phase map':'native-resolution corrected image'}`;const gasProbability=Number(metric.gas_probability_mean);if(Number.isFinite(gasProbability)){const support=Number(metric.support_fraction_p050),aqueous=Number(metric.aqueous_probability_mean);return`${prefix} | mean P(gas) ${gasProbability.toFixed(4)}${Number.isFinite(support)?` | area P(gas) >= 0.5 ${(100*support).toFixed(2)}%`:''}${Number.isFinite(aqueous)?` | mean P(aqueous CO2) ${aqueous.toFixed(4)}`:''}`;}const g=value=>Number.isFinite(Number(value))?(1000*Number(value)).toFixed(4):'n/a',injected=Number(metric.injected_mass_kg),total=Number(metric.detected_total_mass_kg),ratio=Number.isFinite(injected)&&injected>0&&Number.isFinite(total)?(total/injected).toFixed(3):'n/a';return`${prefix} | detected ${g(metric.detected_total_mass_kg)} g | aqueous ${g(metric.detected_aqueous_mass_kg)} g | gas ${g(metric.detected_gas_mass_kg)} g | ratio ${ratio}`;}
const PLOT_AXES={left:0.085,right:0.97,top:0.16,bottom:0.82};
function plotScale(candidate){const metrics=Array.isArray(candidate.frame_metrics)?candidate.frame_metrics:[],elapsed=metrics.map(row=>Number(row.elapsed_h)).filter(Number.isFinite),masses=metrics.flatMap(row=>['injected_mass_kg','detected_total_mass_kg','detected_gas_mass_kg','detected_aqueous_mass_kg'].map(key=>1000*Number(row[key])).filter(Number.isFinite));if(!elapsed.length||!masses.length)return null;let x0=Math.min(...elapsed),x1=Math.max(...elapsed),y0=Math.min(...masses),y1=Math.max(...masses);const dx=Math.max(1e-9,x1-x0),dy=Math.max(1e-9,y1-y0);x0-=0.05*dx;x1+=0.05*dx;y0-=0.05*dy;y1+=0.05*dy;return{x0,x1,y0,y1};}
function jumpToFrame(index,side){if(!currentPair)return;stopPlayback();activeSide=side;frameIndex=Math.max(0,Math.min(frameCount-1,Number(index)));if(byId('reference').checked)byId('reference').checked=false;renderFrame();saveCurrentDraft();}
function renderPlotNavigation(side){const candidate=candidateData(side),root=byId(`plotNavigation${side}`),metrics=Array.isArray(candidate.frame_metrics)?candidate.frame_metrics:[],scale=plotScale(candidate);root.innerHTML='';if(!scale)return;const xFraction=value=>PLOT_AXES.left+(Number(value)-scale.x0)/(scale.x1-scale.x0)*(PLOT_AXES.right-PLOT_AXES.left),yFraction=value=>PLOT_AXES.top+(scale.y1-Number(value))/(scale.y1-scale.y0)*(PLOT_AXES.bottom-PLOT_AXES.top),sourceAt=index=>frameSourceMetadata(index);const selectedMetric=metrics[frameIndex];if(selectedMetric){const guide=document.createElement('div');guide.className='plot-guide';guide.style.left=`${100*xFraction(selectedMetric.elapsed_h)}%`;guide.style.top=`${100*PLOT_AXES.top}%`;guide.style.height=`${100*(PLOT_AXES.bottom-PLOT_AXES.top)}%`;root.appendChild(guide);}metrics.forEach((metric,index)=>{const elapsed=Number(metric.elapsed_h),total=1000*Number(metric.detected_total_mass_kg);if(!Number.isFinite(elapsed)||!Number.isFinite(total))return;const source=sourceAt(index),button=document.createElement('button');button.type='button';button.dataset.index=String(index);button.className=`plot-point ${index===frameIndex?'active':''}`;button.style.left=`${100*xFraction(elapsed)}%`;button.style.top=`${100*yFraction(total)}%`;button.title=`Frame ${index+1} | ${elapsed.toFixed(3)} h${source&&source.source_name?` | ${source.source_name}`:''}`;button.setAttribute('aria-label',`Open ${button.title}`);button.onclick=event=>{event.stopPropagation();jumpToFrame(index,side);};root.appendChild(button);});root.onclick=event=>{if(event.target.closest('.plot-point'))return;const rect=root.getBoundingClientRect(),fraction=(event.clientX-rect.left)/Math.max(1,rect.width);if(fraction<PLOT_AXES.left||fraction>PLOT_AXES.right)return;const elapsed=scale.x0+(fraction-PLOT_AXES.left)/(PLOT_AXES.right-PLOT_AXES.left)*(scale.x1-scale.x0),nearest=metrics.reduce((best,row,index)=>Math.abs(Number(row.elapsed_h)-elapsed)<best.distance?{index,distance:Math.abs(Number(row.elapsed_h)-elapsed)}:best,{index:0,distance:Infinity});jumpToFrame(nearest.index,side);};}
function applyReferenceOffset(image,candidate,index,enabled){const offset=enabled&&Array.isArray(candidate.reference_offsets)?candidate.reference_offsets[index]:null;image.style.transform=offset?`translate(${100*Number(offset.x||0)}%,${100*Number(offset.y||0)}%)`:'';}
function renderFrame({preserveSelection=false}={}){if(!currentPair||frameCount<1)return;frameIndex=Math.max(0,Math.min(frameCount-1,frameIndex));byId('scrub').value=String(frameIndex);const referenceMode=Boolean(byId('reference').checked&&currentPair.reference_ready&&currentPair.candidate_a.reference_frames_ready&&currentPair.candidate_b.reference_frames_ready),showOverlay=byId('contours').checked,phaseMapOnly=!referenceMode&&showOverlay&&byId('phaseMapOnly').checked,highlightSmall=phaseMapOnly&&!IS_PORTABLE&&byId('highlightSmall').checked,sharedQuiz=Boolean(currentPhaseQuizProposal()),sharedFrameUrl=!referenceMode?frameAsset(currentPair.candidate_a,frameIndex):'',markerGeneration=++isolatedMarkerGeneration;for(const side of ['A','B']){const candidate=candidateData(side),contourSide=contourSideForPanel(side),contourCandidate=candidateData(contourSide),image=byId(`frame${side}`),overlay=byId(`overlay${side}`),layer=byId(`layer${side}`),badge=byId(`contourSource${side}`),visible=side==='A'||!sharedQuiz;layer.classList.toggle('phase-map-only',phaseMapOnly);if(visible){assignImageAsset(image,referenceMode?referenceAsset(candidate,frameIndex):sharedFrameUrl);applyReferenceOffset(image,candidate,frameIndex,referenceMode);image.alt=referenceMode?`${currentRun.toUpperCase()} pre-injection reference aligned to Candidate ${side}`:`Candidate ${side} native-resolution calibration frame`;assignImageAsset(overlay,!referenceMode&&showOverlay?overlayAsset(contourCandidate,frameIndex):'');overlay.hidden=referenceMode||!showOverlay;byId(`assetLabel${side}`).textContent=referenceMode?'Native-resolution pre-injection reference aligned to current frame':phaseMapOnly?'Native-resolution filled phase map on black':frameMetricLabel(candidate,frameIndex,showOverlay);renderPlotNavigation(side);}renderIsolatedMarkers(side,contourSide,highlightSmall&&visible,markerGeneration);badge.hidden=!contourSwapHeld||referenceMode||!showOverlay;badge.dataset.source=contourSide;badge.textContent=`Candidate ${contourSide} phase map`;}const elapsed=currentPair.candidate_a.elapsed_hours&&currentPair.candidate_a.elapsed_hours[frameIndex],shape=currentPair.analysis_image_shape_hw||currentPair.candidate_a.analysis_image_shape_hw||[],resolution=shape.length===2?`${shape[1]} × ${shape[0]} analysis pixels`:'',source=frameSourceMetadata(frameIndex),sourceText=frameSourceSummary(frameIndex),detail=[referenceMode?'Pre-injection reference aligned to current frame':phaseMapOnly?'Filled phase map on black':'Corrected calibration image',highlightSmall?'isolated-speck markers':null,sourceText,resolution].filter(Boolean).join(' | ');byId('frameNumber').textContent=`Frame ${frameIndex+1} / ${frameCount}`;byId('frameElapsed').textContent=Number.isFinite(elapsed)?`t = ${Number(elapsed).toFixed(2)} h`:'t = -- h';byId('time').textContent=detail;byId('time').title=source&&source.source_path?source.source_path:source&&source.source_name?source.source_name:'';renderFrameChoice();if(!preserveSelection)selectedAnnotationId=null;renderAnnotations();inspectionAvailability();if(!referenceMode)scheduleNearbyFramePreload(showOverlay);}
async function loadPair(run) {
  stopPlayback();
  if(currentRun&&currentRun!==run&&currentPair){saveCurrentDraft();await flushServerDraft();}
  currentRun=run; currentPair=await loadPairData(run);
  setComparisonMode(currentPair.comparison_mode||'pair');
  byId('runTitle').textContent=runDisplay(run,currentPair.dataset);const anchorContract=currentPair.injection_anchor_contract||{},anchorStale=Boolean(anchorContract.candidate_outputs_require_regeneration),physicsWarning=byId('physicsWarning');physicsWarning.hidden=!anchorStale;physicsWarning.textContent=anchorStale?'Injector markers corrected; displayed candidate phase maps were generated with stale source coordinates and require regeneration.':'';byId('plotA').src=currentPair.candidate_a.plot_url; byId('plotB').src=currentPair.candidate_b.plot_url;
  const cleanReady=currentPair.candidate_a.plain_frames_ready&&currentPair.candidate_b.plain_frames_ready;byId('reference').checked=false;byId('phaseMapOnly').checked=false;if(!cleanReady)byId('contours').checked=true;
  for(const side of ['A','B']){transforms[side]={scale:1,x:0,y:0};applyTransform(side);}
  const decision=currentPair.decision,draft=currentDraft(currentPair.browser_draft),source={...(decision||{}),...(draft||{})};frameCount=Math.min(currentPair.candidate_a.frame_count,currentPair.candidate_b.frame_count);frameIndex=draft&&Number.isInteger(draft.frame_index)?draft.frame_index:0;byId('scrub').min='0';byId('scrub').max=String(Math.max(0,frameCount-1));byId('scrub').step='1';setFrameChoices(source.frame_choices||[]);setFlags(source.flags||{});byId('notes').value=source.notes||'';annotations=Array.isArray(source.annotations)?clone(source.annotations).map(normalizeAnnotation):[];phaseQuizAnswers=phaseQuizAnswersForPair(Array.isArray(source.phase_quiz_answers)?clone(source.phase_quiz_answers):[],currentPair);phaseQuizSelectedIds=cleanPhaseQuizSelectedIds(currentPair,source.phase_quiz_selected_ids,phaseQuizAnswers);phaseQuizRounds=cleanPhaseQuizRounds(currentPair,source.phase_quiz_rounds,phaseQuizSelectedIds);phaseQuizPerCandidateLimit=phaseQuizSelectedIds.length;const proposals=currentPhaseQuizProposals(),firstUnanswered=proposals.findIndex(proposal=>!phaseQuizAnswerFor(proposal));phaseQuizIndex=draft&&Number.isInteger(draft.phase_quiz_index)?Math.max(0,Math.min(proposals.length-1,draft.phase_quiz_index)):Math.max(0,firstUnanswered);phaseQuizActive=draft&&typeof draft.phase_quiz_active==='boolean'?draft.phase_quiz_active:firstUnanswered>=0;if(phaseQuizIsComplete(proposals))phaseQuizActive=false;renderFrame();revealDetails(currentPair.reveal);if(draft)setSaveStatus('saved','Recovered autosaved draft','The newest reviewer-scoped recovery draft has been restored.');else if(decision)setSaveStatus('saved','Final review saved');else if(reviewerIsReady())setSaveStatus('saved','Autosave ready');
  if(IS_PORTABLE&&activeSession){activeSession.current_run=run;persistPortableSession();}
  renderState(); const index=currentIndex(); byId('previous').disabled=index<=0; byId('next').disabled=index>=state.runs.length-1;
}
byId('play').onclick=()=>{ if(playTimer!==null){stopPlayback();return;} byId('play').textContent='Pause';playTimer=setInterval(()=>{if(frameIndex>=frameCount-1){stopPlayback();return;}frameIndex+=1;renderFrame();},900); };
byId('scrub').oninput=()=>{frameIndex=Number(byId('scrub').value);renderFrame();saveCurrentDraft();};
byId('phaseQuizPrevious').onclick=()=>setPhaseQuizIndex(phaseQuizIndex-1);
byId('phaseQuizFocus').onclick=focusPhaseQuizRegion;
byId('phaseQuizClose').onclick=closePhaseQuizRegion;
byId('phaseQuizNext').onclick=()=>setPhaseQuizIndex(phaseQuizIndex+1);
byId('phaseQuizUnanswered').onclick=nextUnansweredPhaseQuiz;
byId('phaseQuizDraw').onclick=drawPhaseQuizDetail;
byId('phaseQuizIgnore').onclick=()=>answerPhaseQuiz('ignore_region');
byId('phaseQuizClear').onclick=clearPhaseQuizAnswer;
byId('phaseQuizTimelineMarker').onclick=focusPhaseQuizRegion;
byId('phaseQuizNewRound').onclick=startAdaptivePhaseQuizRound;
byId('phaseQuizAddCount').oninput=()=>renderPhaseQuizScope(currentPhaseQuizProposals());
document.querySelectorAll('[data-phase-choice]').forEach(button=>button.onclick=()=>answerPhaseQuiz(button.dataset.phaseChoice));
byId('phaseQuizConfidence').onchange=()=>updateCurrentPhaseQuizAnswer('confidence',byId('phaseQuizConfidence').value);
byId('phaseQuizComment').oninput=()=>updateCurrentPhaseQuizAnswer('comment',byId('phaseQuizComment').value.trim());
byId('contours').onchange=()=>{setContourSwapHeld(false,false);if(!byId('contours').checked)byId('phaseMapOnly').checked=false;renderFrame();saveCurrentDraft();};
byId('phaseMapOnly').onchange=()=>{if(byId('phaseMapOnly').checked)byId('contours').checked=true;renderFrame({preserveSelection:true});saveCurrentDraft();};
byId('highlightSmall').onchange=()=>{renderFrame({preserveSelection:true});saveCurrentDraft();};
byId('contourMode').onchange=()=>{renderFrame({preserveSelection:true});};
byId('injectionPoints').onchange=renderAnnotations;
byId('reference').onchange=()=>{stopPlayback();setContourSwapHeld(false,false);setDrawTool('pan');selectedAnnotationId=null;renderFrame();};
byId('markBox').onclick=()=>setDrawTool('rectangle');
byId('markFreehand').onclick=()=>setDrawTool('freehand');
byId('markPixel').onclick=()=>{if(!byId('markPixel').disabled)setDrawTool('point');};
byId('panView').onclick=()=>setDrawTool('pan');
byId('undoBox').onclick=()=>{if(!annotations.length)return;const last=annotations[annotations.length-1];annotations.pop();if(selectedAnnotationId===last.id)selectedAnnotationId=null;saveAnnotationDraft();renderAnnotations();};
byId('deleteBox').onclick=removeSelected;
byId('annotationComment').oninput=()=>{const item=annotationById(selectedAnnotationId);if(item){item.comment=byId('annotationComment').value;saveAnnotationDraft();}};
byId('annotationCategory').onchange=()=>{const category=byId('annotationCategory').value;if(CATEGORY_PHASE_DEFAULTS[category])byId('annotationPhase').value=CATEGORY_PHASE_DEFAULTS[category];if(category.startsWith('confirmed_'))byId('annotationConfidence').value='high';updateAnnotationCategoryHelp(category);};
byId('selectedAnnotationCategory').onchange=()=>updateSelectedAnnotation('category',byId('selectedAnnotationCategory').value);
byId('selectedAnnotationPhase').onchange=()=>updateSelectedAnnotation('phase',byId('selectedAnnotationPhase').value);
byId('selectedAnnotationSeverity').onchange=()=>updateSelectedAnnotation('severity',byId('selectedAnnotationSeverity').value);
byId('selectedAnnotationConfidence').onchange=()=>updateSelectedAnnotation('confidence',byId('selectedAnnotationConfidence').value);
byId('annotationLineWidth').oninput=()=>setAnnotationLineWidth(byId('annotationLineWidth').value);
for(const side of ['A','B']){
  const viewport=byId(`viewport${side}`),frame=byId(`frame${side}`);frame.addEventListener('load',()=>applyTransform(side));viewport.addEventListener('pointerenter',()=>{activeSide=side;updateZoomReadout(side);});viewport.addEventListener('wheel',event=>{event.preventDefault();activeSide=side;const rect=viewport.getBoundingClientRect();zoomAt(side,event.deltaY<0?1.22:1/1.22,event.clientX-rect.left,event.clientY-rect.top);},{passive:false});
  viewport.addEventListener('dblclick',()=>resetZoom(side));
  viewport.addEventListener('pointerdown',event=>{
    if(!event.isPrimary||(event.button!==0&&event.button!==1))return;
    activeSide=side;stopPlayback();event.preventDefault();
    const hit=event.target instanceof Element?event.target.closest('.annotation-hit'):null;
    const forcePan=event.button===1||spacePan;
    const tool=forcePan?'pan':event.shiftKey?'rectangle':drawTool;
    if(tool==='pan'){
      if(transforms[side].scale<=1){if(hit&&event.button===0)selectAnnotation(hit.dataset.id);return;}
      const viewRects={A:byId('viewportA').getBoundingClientRect(),B:byId('viewportB').getBoundingClientRect()};
      interaction={type:'pan',side,pointerId:event.pointerId,startClient:{x:event.clientX,y:event.clientY},startTransform:{...transforms[side]},viewRects,moved:false,selectId:hit&&event.button===0?hit.dataset.id:null};
      viewport.classList.add('dragging');
    }else{
      const viewportRect=viewport.getBoundingClientRect(),start=pointInImage(event,side,viewportRect);
      interaction={type:'draw',tool,side,pointerId:event.pointerId,viewportRect,start,current:start,points:[start]};
    }
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener('pointermove',event=>{if(!interaction||interaction.side!==side||interaction.pointerId!==event.pointerId)return;event.preventDefault();if(interaction.type==='pan'){const dx=event.clientX-interaction.startClient.x,dy=event.clientY-interaction.startClient.y;if(Math.hypot(dx,dy)>=3)interaction.moved=true;const t=transforms[side],start=interaction.startTransform;t.x=start.x+dx;t.y=start.y+dy;clampTransform(side,interaction.viewRects[side]);applyTransform(side,false);syncTransform(side,false,interaction.viewRects);return;}interaction.current=pointInImage(event,side,interaction.viewportRect);if(interaction.tool==='point')return;if(interaction.tool==='freehand')interaction.points.push(interaction.current);renderAnnotations();const svg=byId(`annotations${side}`),draft=document.createElementNS('http://www.w3.org/2000/svg',interaction.tool==='freehand'?'polyline':'rect');draft.setAttribute('class','annotation-box draft');draft.setAttribute('vector-effect','non-scaling-stroke');draft.setAttribute('style','fill:none');if(interaction.tool==='freehand')draft.setAttribute('points',pointString(simplifyFreehandPoints(interaction.points).map(point=>[point.x,point.y])));else{const start=interaction.start,current=interaction.current,x=Math.min(start.x,current.x),y=Math.min(start.y,current.y),width=Math.abs(current.x-start.x),height=Math.abs(current.y-start.y);for(const [key,value] of Object.entries({x,y,width,height}))draft.setAttribute(key,String(value));}svg.appendChild(draft);});
  const finish=(event,cancelled=false)=>{if(!interaction||interaction.side!==side||interaction.pointerId!==event.pointerId)return;const finished=interaction;interaction=null;viewport.classList.remove('dragging');if(viewport.hasPointerCapture(event.pointerId))viewport.releasePointerCapture(event.pointerId);if(cancelled){renderAnnotations();return;}if(finished.type==='pan'){if(!finished.moved&&finished.selectId)selectAnnotation(finished.selectId);return;}if(finished.tool==='point')addPointAnnotation(side,finished.start);else if(finished.tool==='freehand')addFreehandAnnotation(side,finished.points);else addRectangleAnnotation(side,finished.start,finished.current);};
  viewport.addEventListener('pointerup',event=>finish(event));
  viewport.addEventListener('pointercancel',event=>finish(event,true));
  viewport.addEventListener('lostpointercapture',event=>{if(interaction&&interaction.side===side&&interaction.pointerId===event.pointerId){interaction=null;viewport.classList.remove('dragging');renderAnnotations();}});
  viewport.addEventListener('auxclick',event=>{if(event.button===1)event.preventDefault();});
}
byId('zoomIn').onclick=()=>{const view=byId(`viewport${activeSide}`).getBoundingClientRect();zoomAt(activeSide,1.3,view.width/2,view.height/2);};
byId('zoomOut').onclick=()=>{const view=byId(`viewport${activeSide}`).getBoundingClientRect();zoomAt(activeSide,1/1.3,view.width/2,view.height/2);};
byId('zoomReset').onclick=()=>resetZoom(activeSide);
byId('zoomNative').onclick=()=>nativeZoom(activeSide);
byId('zoomPixel').onclick=()=>pixelZoom(activeSide);
byId('linkViews').onchange=()=>{if(byId('linkViews').checked)syncTransform(activeSide);};
const holdSwap=byId('holdSwap');
holdSwap.addEventListener('pointerdown',event=>{if(!event.isPrimary||event.button!==0||holdSwap.hidden)return;event.preventDefault();setSwapHeld(true);holdSwap.setPointerCapture(event.pointerId);});
const releaseHeldSwap=event=>{setSwapHeld(false);if(Number.isInteger(event.pointerId)&&holdSwap.hasPointerCapture(event.pointerId))holdSwap.releasePointerCapture(event.pointerId);};
holdSwap.addEventListener('pointerup',releaseHeldSwap);
holdSwap.addEventListener('pointercancel',releaseHeldSwap);
holdSwap.addEventListener('lostpointercapture',()=>setSwapHeld(false));
holdSwap.addEventListener('keydown',event=>{if((event.code==='Space'||event.code==='Enter')&&!event.repeat){event.preventDefault();setSwapHeld(true);}});
holdSwap.addEventListener('keyup',event=>{if(event.code==='Space'||event.code==='Enter'){event.preventDefault();setSwapHeld(false);}});
holdSwap.addEventListener('blur',()=>setSwapHeld(false));
const holdContourSwap=byId('holdContourSwap');
holdContourSwap.addEventListener('pointerdown',event=>{if(!event.isPrimary||event.button!==0||holdContourSwap.hidden||holdContourSwap.disabled)return;event.preventDefault();setContourSwapHeld(true);holdContourSwap.setPointerCapture(event.pointerId);});
const releaseHeldContourSwap=event=>{setContourSwapHeld(false);if(Number.isInteger(event.pointerId)&&holdContourSwap.hasPointerCapture(event.pointerId))holdContourSwap.releasePointerCapture(event.pointerId);};
holdContourSwap.addEventListener('pointerup',releaseHeldContourSwap);
holdContourSwap.addEventListener('pointercancel',releaseHeldContourSwap);
holdContourSwap.addEventListener('lostpointercapture',()=>setContourSwapHeld(false));
holdContourSwap.addEventListener('keydown',event=>{if((event.code==='Space'||event.code==='Enter')&&!event.repeat&&!holdContourSwap.disabled){event.preventDefault();setContourSwapHeld(true);}});
holdContourSwap.addEventListener('keyup',event=>{if(event.code==='Space'||event.code==='Enter'){event.preventDefault();setContourSwapHeld(false);}});
holdContourSwap.addEventListener('blur',()=>setContourSwapHeld(false));
function isEditingControl(){return['INPUT','TEXTAREA','SELECT','BUTTON'].includes(document.activeElement.tagName);}
document.addEventListener('keydown',event=>{if(event.code==='Space'&&!isEditingControl()){spacePan=true;document.body.classList.add('space-pan');event.preventDefault();return;}if(event.key==='Escape'){selectedAnnotationId=null;setDrawTool('pan');renderAnnotations();return;}if((event.key==='Delete'||event.key==='Backspace')&&selectedAnnotationId&&!isEditingControl()){event.preventDefault();removeSelected();}});
document.addEventListener('keyup',event=>{if(event.code==='Space'){spacePan=false;document.body.classList.remove('space-pan');}});
window.addEventListener('blur',()=>{spacePan=false;document.body.classList.remove('space-pan');setSwapHeld(false);setContourSwapHeld(false);});
document.querySelectorAll('#verdict button').forEach(button=>button.onclick=()=>{setChoice(button.dataset.choice);saveCurrentDraft();});
byId('nextUnreviewedFrame').onclick=openNextUnreviewedFrame;
byId('notes').oninput=saveCurrentDraft;
byId('previous').onclick=()=>move(-1); byId('next').onclick=()=>move(1);
byId('save').onclick=async()=>{try{const reviewer=byId('reviewer').value.trim();if(!reviewer||!reviewerIsReady())throw new Error('Enter the reviewer name and select Open');if(frameChoices.size<frameCount)throw new Error(`Review every frame before final save (${frameChoices.size}/${frameCount}). Use Unresolved when the image is inconclusive.`);choice=aggregateFrameChoice();saveCurrentDraft();await flushServerDraft();setSaveStatus('syncing','Saving final review...');const result=await saveDecisionData({run:currentRun,reviewer,choice,frame_choice_schema:'candidate-frame-preference-v1',frame_choices:frameChoiceRecords(),flags:getFlags(),annotations,phase_quiz_answers:phaseQuizAnswers,phase_quiz_selected_ids:phaseQuizSelectedIds,phase_quiz_rounds:phaseQuizRounds,phase_quiz_per_candidate_limit:phaseQuizSelectedIds.length,notes:byId('notes').value});if(!IS_PORTABLE){storageRemove(annotationDraftKey());storageRemove(unassignedAnnotationDraftKey());storageRemove(previousAnnotationDraftKey());storageRemove(legacyAnnotationDraftKey());}state=result.state;currentPair=result.pair;revealDetails(currentPair.reveal);renderState();setSaveStatus('saved','Final per-frame review saved');const index=currentIndex();const next=state.runs.slice(index+1).find(item=>!item.reviewed);if(next)setTimeout(()=>loadPair(next.run),250);}catch(error){setSaveStatus('error','Save failed · recovery draft retained',error.message);}};
byId('openReviewer').onclick=async()=>{try{await activateReviewer(byId('reviewer').value);}catch(error){setSaveStatus('error','Could not open reviewer',error.message);}};
byId('reviewer').addEventListener('input',updateReviewerGate);
byId('reviewer').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();byId('openReviewer').click();}});
byId('exportReview').onclick=()=>{try{saveCurrentDraft();exportPortableReview();}catch(error){setSaveStatus('error','Export failed',error.message);}};
byId('chooseAutosaveFile').onclick=async()=>{try{await choosePortableAutosaveFile();}catch(error){setSaveStatus('error','Could not enable file autosave',error.message);}};
byId('openReview').onclick=()=>byId('reviewFile').click();
byId('reviewFile').onchange=async event=>{const file=event.target.files&&event.target.files[0];event.target.value='';if(!file)return;try{await importPortableReview(file);}catch(error){setSaveStatus('error','Could not open review file',error.message);}};
byId('newReview').onclick=async()=>{try{const name=byId('reviewer').value.trim();if(!name)throw new Error('Reviewer is required');const reviewed=activeSession?Object.keys(activeSession.decisions).length:0;if((reviewed||Object.keys(activeSession&&activeSession.drafts||{}).length)&&!confirm(`Start a new empty review for ${name}? The current local autosave will be replaced.`))return;await activateReviewer(name,{reset:true});setSaveStatus('saved',`Started new review for ${name}`);}catch(error){setSaveStatus('error','Could not start review',error.message);}};
function preserveDraftOnPageExit(){saveCurrentDraft();if(!IS_PORTABLE&&pendingServerDraft&&navigator.sendBeacon){try{navigator.sendBeacon('/api/draft',new Blob([JSON.stringify(pendingServerDraft)],{type:'application/json'}));pendingServerDraft=null;}catch(_error){}}}
window.addEventListener('pagehide',preserveDraftOnPageExit);
window.addEventListener('beforeunload',preserveDraftOnPageExit);
(async()=>{setAnnotationLineWidth(storageGet(ANNOTATION_LINE_WIDTH_KEY)||1,{persist:false});await initializeReview();renderState();const preferred=IS_PORTABLE&&activeSession&&activeSession.current_run?state.runs.find(item=>item.run===activeSession.current_run):null;const first=preferred||state.runs.find(item=>!item.reviewed)||state.runs[0];if(first)await loadPair(first.run);startLiveInventoryPolling();updateReviewerGate();})().catch(error=>{setSaveStatus('error','Review failed to initialize',error.message);});
</script>
</body>
</html>'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("inventory", "prepare-portable", "serve"),
        nargs="?",
        default="serve",
    )
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--portable-dir", type=Path)
    parser.add_argument("--blind-seed", default=DEFAULT_BLIND_SEED)
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--runs", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda", "opencl"),
        default="auto",
        help="Backend used when regenerating line-only contour frames.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--watch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For serve mode, discover completed postflight pairs while the "
            "calibration/postflight campaign is still running."
        ),
    )
    parser.add_argument("--watch-interval", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    selection_dir = args.selection_dir.resolve()
    inventory = discover_review_inventory(
        selection_dir,
        blind_seed=args.blind_seed,
        allow_incomplete=bool(args.action == "serve" and args.watch),
    )
    output_dir = (args.output_dir or selection_dir / "manual_pairwise_review").resolve()
    store = ReviewStore(output_dir, inventory.pairs)
    summary = {
        "selection_dir": str(selection_dir),
        "manifest": str(inventory.manifest_path),
        "pair_count": len(inventory.pairs),
        "pair_runs": [pair.run for pair in inventory.pairs],
        "unpaired_count": len(inventory.unpaired),
        "unpaired_runs": list(inventory.unpaired),
        "pending_runs": list(inventory.pending_runs),
        "reviewed": len(store.decisions()),
        "decision_json": str(store.json_path),
        "decision_csv": str(store.csv_path),
        "annotation_csv": str(store.annotation_csv_path),
    }
    if args.action == "inventory":
        print(json.dumps(summary, indent=2))
        return
    if args.action == "prepare-portable":
        portable_dir = (
            args.portable_dir or output_dir / "portable_review"
        ).resolve()
        result = prepare_portable_review(
            portable_dir,
            output_dir,
            inventory,
            blind_seed=args.blind_seed,
            runs=args.runs,
            force=args.force,
        )
        print(json.dumps(result, indent=2))
        return

    app = ReviewApp(
        inventory,
        store,
        default_reviewer=args.reviewer,
        selection_dir=selection_dir,
        blind_seed=args.blind_seed,
        watch=args.watch,
        prepare_backend=args.backend,
        watch_interval=args.watch_interval,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{args.host}:{args.port}/"
    print(json.dumps({**summary, "url": url}, indent=2), flush=True)
    if args.open_browser:
        webbrowser.open(url)
    app.start_live_updates()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop_live_updates()
        server.server_close()


if __name__ == "__main__":
    main()
