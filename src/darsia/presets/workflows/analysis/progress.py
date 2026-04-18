"""Utilities for emitting analysis progress events."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, TypedDict


class AnalysisProgressEvent(TypedDict, total=False):
    """Payload contract for analysis progress events."""

    event: Literal["step_start", "image_progress", "step_complete"]
    step: str
    image_path: str
    image_index: int
    image_total: int
    image_duration_s: float
    step_elapsed_s: float


def _safe_duration(value: float | int | None) -> float | None:
    """Normalize duration values to finite non-negative seconds."""
    if value is None:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration):
        return None
    return max(0.0, duration)


def _safe_nonnegative_int(value: int | None) -> int | None:
    """Normalize counters to non-negative ints."""
    if value is None:
        return None
    if not isinstance(value, int):
        return None
    return max(0, value)


def publish_analysis_progress(
    progress_callback: Callable[[AnalysisProgressEvent], None] | None,
    payload: AnalysisProgressEvent,
) -> None:
    """Publish progress payload while guarding callback failures."""
    if progress_callback is None:
        return
    try:
        progress_callback(payload)
    except Exception:
        pass


def publish_step_start(
    progress_callback: Callable[[AnalysisProgressEvent], None] | None,
    *,
    step: str,
    image_total: int,
) -> None:
    """Publish analysis step start event."""
    payload: AnalysisProgressEvent = {
        "event": "step_start",
        "step": step,
        "image_total": max(0, image_total),
    }
    publish_analysis_progress(progress_callback, payload)


def publish_image_progress(
    progress_callback: Callable[[AnalysisProgressEvent], None] | None,
    *,
    step: str,
    image_path: str,
    image_index: int,
    image_total: int,
    image_duration_s: float | int | None,
    step_elapsed_s: float | int | None,
) -> None:
    """Publish per-image analysis progress event."""
    payload: AnalysisProgressEvent = {
        "event": "image_progress",
        "step": step,
        "image_path": image_path,
        "image_index": max(0, image_index),
        "image_total": max(0, image_total),
    }
    safe_image_duration = _safe_duration(image_duration_s)
    if safe_image_duration is not None:
        payload["image_duration_s"] = safe_image_duration
    safe_step_elapsed = _safe_duration(step_elapsed_s)
    if safe_step_elapsed is not None:
        payload["step_elapsed_s"] = safe_step_elapsed
    publish_analysis_progress(progress_callback, payload)


def publish_step_complete(
    progress_callback: Callable[[AnalysisProgressEvent], None] | None,
    *,
    step: str,
    image_total: int,
    step_elapsed_s: float | int | None,
) -> None:
    """Publish analysis step completion event."""
    payload: AnalysisProgressEvent = {
        "event": "step_complete",
        "step": step,
        "image_total": max(0, image_total),
    }
    safe_step_elapsed = _safe_duration(step_elapsed_s)
    if safe_step_elapsed is not None:
        payload["step_elapsed_s"] = safe_step_elapsed
    publish_analysis_progress(progress_callback, payload)


def normalize_progress_event(payload: object) -> AnalysisProgressEvent | None:
    """Normalize arbitrary payload to known progress-event structure."""
    if not isinstance(payload, dict):
        return None
    event = payload.get("event")
    if event not in {"step_start", "image_progress", "step_complete"}:
        return None
    step = payload.get("step")
    if not isinstance(step, str) or not step.strip():
        return None

    normalized: AnalysisProgressEvent = {"event": event, "step": step.strip()}

    image_total = _safe_nonnegative_int(payload.get("image_total"))
    if image_total is not None:
        normalized["image_total"] = image_total

    image_index = _safe_nonnegative_int(payload.get("image_index"))
    if image_index is not None:
        normalized["image_index"] = image_index

    image_path = payload.get("image_path")
    if isinstance(image_path, str):
        normalized["image_path"] = image_path

    image_duration_s = _safe_duration(payload.get("image_duration_s"))
    if image_duration_s is not None:
        normalized["image_duration_s"] = image_duration_s

    step_elapsed_s = _safe_duration(payload.get("step_elapsed_s"))
    if step_elapsed_s is not None:
        normalized["step_elapsed_s"] = step_elapsed_s

    return normalized
