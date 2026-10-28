"""Versioned frame-quality screening for FluidFlower image series.

The automatic screen is deliberately advisory. A frame is excluded from the
active analysis blacklist only after an explicit human decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from darsia.fluidflower.image_correction import (
    COLORCHECKER_ALGORITHMS,
    DARSIA_BASELINE_COLOR_ALGORITHM,
    IDENTITY_ALGORITHM,
    UNIFORM24_ALGORITHM,
    CorrectionRevisionStore,
    apply_darsia_baseline_array_correction,
    apply_uniform24_array_transform,
    fit_uniform24_array_transform,
)
from darsia.fluidflower.project import utc_now

if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject


FRAME_QUALITY_ASSESSMENT_FORMAT = "darsia-fluidflower-frame-quality-v1"
FRAME_QUALITY_BLACKLIST_FORMAT = "darsia-fluidflower-frame-blacklist-v1"
FRAME_QUALITY_DECISION_FORMAT = "darsia-fluidflower-frame-quality-decision-v1"
FRAME_QUALITY_ALGORITHM = "temporal-radiometric-qc-v1"

QUALITY_STATUSES = (
    "ok",
    "corrected",
    "suspicious",
    "exclusion_suggested",
    "analysis_failed",
)
QUALITY_DECISIONS = ("keep", "exclude", "clear")


def classify_chart_pillar_light(dc, dp, *, clipped_delta=0):
    """The pre-declared 28 September RAW chart/pillar rule, without fitting.

    It is separate from the advisory quality blacklist: an explicitly selected
    production manifest may apply this frozen rule without modifying that
    blacklist or any prior decision.
    """
    if not np.isfinite(dc) or not np.isfinite(dp) or clipped_delta > 2:
        return "reject"
    dd = dc - dp
    if abs(dd) <= .02 and abs(dc) <= .05 and abs(dp) <= .05:
        return "ok"
    if abs(dd) <= .05 and abs(dc) <= .15 and abs(dp) <= .15:
        return "minor"
    return "reject"


def raw_chart_pillar_reference(cache_metadata, cache_arrays):
    """Reconstruct the unchanged frozen reference statistics, not an image.

    Cache inputs are the source-bound 13 pre-injection RAW measurements. The
    returned arrays only specify chart patches and a fixed left-pillar strip.
    """
    lattice = cache_metadata["lattice"]
    neutral = np.asarray(lattice["neutral"], bool)
    chart_g = .5*(cache_arrays["chart"][..., 1] + cache_arrays["chart"][..., 2])
    chart_ok = neutral & ~cache_arrays["chart_clip"].any(0)
    ref_chart = np.median(chart_g[:, chart_ok], 0)
    planes_g = cache_arrays["planes"][:, 1]
    clipped = cache_arrays["clipbins"].any(0)
    med = np.median(planes_g, 0)
    best, start = -1., None
    for x in range(22):
        region = clipped[50:120, x:x+4]
        if region.mean() > .2:
            continue
        value = float(np.median(med[50:120, x:x+4][~region]))
        if value > best:
            best, start = value, x
    if start is None or best <= 0 or not np.any(chart_ok):
        raise ValueError("Frozen RAW reference has no valid chart/pillar support")
    pillar_mask = np.zeros_like(clipped)
    pillar_mask[50:120, start:start+4] = True
    pillar_mask &= ~clipped
    return dict(points=[(int(round(y)), int(round(x))) for y,x in lattice["points"]],
                half=int(lattice["half"]), chart_ok=chart_ok,
                ref_chart=ref_chart, pillar_mask=pillar_mask,
                ref_p=float(np.median(med[pillar_mask])))


def measure_raw_chart_pillar(raw_bytes, reference):
    """Measure only RAW green chart/pillar patches using the frozen rule.

    No corrected image, labels, mass or model prediction is read. RAW byte
    hashing/binding and camera-time provenance are the caller's responsibility.
    """
    import io
    import rawpy
    with rawpy.imread(io.BytesIO(raw_bytes)) as raw:
        values = raw.raw_image_visible.astype(np.float32).copy()
    if values.shape[0] % 2 or values.shape[1] % 2:
        raise ValueError("Unexpected odd Bayer geometry")
    green = .5*(values[0::2,1::2] + values[1::2,0::2])
    half = reference["half"]
    levels, clipped = [], []
    for i, (y,x) in enumerate(reference["points"]):
        if not reference["chart_ok"][i]:
            continue
        if y-half < 0 or x-half < 0 or y+half >= green.shape[0] or x+half >= green.shape[1]:
            raise ValueError("Frozen chart patch falls outside RAW")
        square = green[y-half:y+half+1, x-half:x+half+1]
        clipped.append(bool((square >= 15900).any()))
        levels.append(float(np.median(square))-800)
    good = ~np.asarray(clipped, bool)
    c = float(np.median(np.log(np.maximum(np.asarray(levels)[good],1)/reference["ref_chart"][good]))) if good.any() else None
    if green.shape[0] < 1836 or green.shape[1] < 2748:
        raise ValueError("Unexpected RAW geometry for frozen 12x12 pillar bins")
    binned = (green-800)[:1836,:2748].reshape(153,12,229,12).mean((1,3))
    p = float(np.log(max(float(np.median(binned[reference["pillar_mask"]])),1e-3)/reference["ref_p"]))
    return dict(c=c, p=p, clipped=int((~good).sum()),
                class_measurement="28sep-frozen-raw-chart-pillar-v1")


def default_frame_quality_config() -> dict[str, Any]:
    """Return conservative defaults for advisory frame screening."""

    return {
        "algorithm": FRAME_QUALITY_ALGORITHM,
        "signature_grid": [6, 10],
        "thresholds": {
            "raw_exposure_log": 0.18,
            "raw_spatial_log": 0.17,
            "raw_color_distance": 0.065,
            "raw_left_right_log": 0.22,
            "corrected_exposure_log": 0.10,
            "corrected_spatial_log": 0.11,
            "corrected_color_distance": 0.045,
            "corrected_left_right_log": 0.13,
            "transient_spatial_log": 0.14,
            "severe_transient_spatial_log": 0.24,
            "minimum_sharpness_ratio": 0.22,
            "extreme_dark_fraction": 0.35,
            "extreme_bright_fraction": 0.35,
        },
        "policy": {
            "automatic_exclusion": False,
            "manual_confirmation_required": True,
            "preserve_source_images": True,
            "neighbor_replacement": False,
        },
    }


def _canonical_digest(payload: Any) -> str:
    from darsia.fluidflower.analysis_domains import _json_digest as _canonical_ascii_digest
    return _canonical_ascii_digest(payload)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _append_jsonl as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(f"Invalid frame-quality record at line {line_number}")
            rows.append(dict(payload))
    return rows


def _float_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Frame-quality analysis requires an RGB image")
    array = array[..., :3]
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    return np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)


def frame_quality_signature(
    image: np.ndarray,
    *,
    grid_shape: Sequence[int] = (6, 10),
) -> dict[str, Any]:
    """Compress one image into illumination, color, and sharpness evidence."""

    rgb = _float_rgb(image)
    grid_height, grid_width = (int(value) for value in grid_shape)
    if grid_height < 2 or grid_width < 2:
        raise ValueError("The frame-quality signature grid must be at least 2 by 2")
    grid = cv2.resize(
        rgb,
        (grid_width, grid_height),
        interpolation=cv2.INTER_AREA,
    )
    luminance = 0.2126 * grid[..., 0] + 0.7152 * grid[..., 1] + 0.0722 * grid[..., 2]
    log_luminance = np.log(np.clip(luminance, 1.0e-4, 1.0))
    spatial_profile = log_luminance - np.median(log_luminance)
    rgb_median = np.median(grid.reshape(-1, 3), axis=0)
    chromaticity = rgb_median / max(float(np.sum(rgb_median)), 1.0e-6)
    grayscale = cv2.resize(
        luminance.astype(np.float32),
        (max(24, grid_width * 4), max(16, grid_height * 4)),
        interpolation=cv2.INTER_CUBIC,
    )
    sharpness = float(cv2.Laplacian(grayscale, cv2.CV_32F).var())
    left = float(np.median(luminance[:, : max(1, grid_width // 2)]))
    right = float(np.median(luminance[:, grid_width // 2 :]))
    full_luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return {
        "grid_shape": [grid_height, grid_width],
        "luma_median": float(np.median(luminance)),
        "luma_p05": float(np.percentile(luminance, 5.0)),
        "luma_p95": float(np.percentile(luminance, 95.0)),
        "chromaticity": [float(value) for value in chromaticity],
        "spatial_log_luma": [float(value) for value in spatial_profile.ravel()],
        "left_right_log": float(np.log((left + 1.0e-4) / (right + 1.0e-4))),
        "sharpness": sharpness,
        "dark_fraction": float(np.mean(full_luminance < 0.025)),
        "bright_fraction": float(np.mean(full_luminance > 0.985)),
    }


def _signature_arrays(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, np.ndarray]:
    signatures = [dict(row.get(key) or {}) for row in rows]
    return {
        "luma": np.asarray(
            [float(value.get("luma_median") or 0.0) for value in signatures],
            dtype=np.float64,
        ),
        "spatial": np.asarray(
            [value.get("spatial_log_luma") or [] for value in signatures],
            dtype=np.float64,
        ),
        "color": np.asarray(
            [value.get("chromaticity") or [0.0, 0.0, 0.0] for value in signatures],
            dtype=np.float64,
        ),
        "balance": np.asarray(
            [float(value.get("left_right_log") or 0.0) for value in signatures],
            dtype=np.float64,
        ),
        "sharpness": np.asarray(
            [float(value.get("sharpness") or 0.0) for value in signatures],
            dtype=np.float64,
        ),
        "dark": np.asarray(
            [float(value.get("dark_fraction") or 0.0) for value in signatures],
            dtype=np.float64,
        ),
        "bright": np.asarray(
            [float(value.get("bright_fraction") or 0.0) for value in signatures],
            dtype=np.float64,
        ),
    }


def _temporal_metrics(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> list[dict[str, float]]:
    arrays = _signature_arrays(rows, key)
    count = len(rows)
    reference_luma = max(float(np.median(arrays["luma"])), 1.0e-6)
    reference_spatial = np.median(arrays["spatial"], axis=0)
    reference_color = np.median(arrays["color"], axis=0)
    reference_balance = float(np.median(arrays["balance"]))
    reference_sharpness = max(float(np.median(arrays["sharpness"])), 1.0e-9)
    metrics: list[dict[str, float]] = []
    for index in range(count):
        spatial_delta = np.abs(arrays["spatial"][index] - reference_spatial)
        transient = 0.0
        if 0 < index < count - 1:
            neighbor_prediction = 0.5 * (
                arrays["spatial"][index - 1] + arrays["spatial"][index + 1]
            )
            transient = float(
                np.percentile(
                    np.abs(arrays["spatial"][index] - neighbor_prediction),
                    90.0,
                )
            )
        metrics.append(
            {
                "exposure_log": float(
                    abs(
                        np.log(
                            (max(float(arrays["luma"][index]), 1.0e-6)) / reference_luma
                        )
                    )
                ),
                "spatial_log": float(np.percentile(spatial_delta, 90.0)),
                "spatial_changed_fraction": float(np.mean(spatial_delta > 0.12)),
                "color_distance": float(
                    np.linalg.norm(arrays["color"][index] - reference_color)
                ),
                "left_right_log": float(
                    abs(arrays["balance"][index] - reference_balance)
                ),
                "transient_spatial_log": transient,
                "sharpness_ratio": float(
                    arrays["sharpness"][index] / reference_sharpness
                ),
                "dark_fraction": float(arrays["dark"][index]),
                "bright_fraction": float(arrays["bright"][index]),
            }
        )
    return metrics


def _metric_flags(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    prefix: str,
) -> list[str]:
    flags: list[str] = []
    if metrics["exposure_log"] > thresholds[f"{prefix}_exposure_log"]:
        flags.append(f"{prefix}_global_illumination_shift")
    if metrics["spatial_log"] > thresholds[f"{prefix}_spatial_log"]:
        flags.append(f"{prefix}_spatial_illumination_shift")
    if metrics["color_distance"] > thresholds[f"{prefix}_color_distance"]:
        flags.append(f"{prefix}_color_shift")
    if metrics["left_right_log"] > thresholds[f"{prefix}_left_right_log"]:
        flags.append(f"{prefix}_left_right_imbalance")
    if (
        metrics["transient_spatial_log"] > thresholds["transient_spatial_log"]
        and metrics["spatial_changed_fraction"] > 0.15
    ):
        flags.append(f"{prefix}_transient_occlusion_or_obstruction")
    if metrics["sharpness_ratio"] < thresholds["minimum_sharpness_ratio"]:
        flags.append(f"{prefix}_blur_or_focus_change")
    if metrics["dark_fraction"] > thresholds["extreme_dark_fraction"]:
        flags.append(f"{prefix}_extreme_underexposure")
    if metrics["bright_fraction"] > thresholds["extreme_bright_fraction"]:
        flags.append(f"{prefix}_extreme_overexposure")
    return flags


def assess_frame_quality_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess precomputed raw/corrected signatures independently per run."""

    settings = deepcopy(dict(config or default_frame_quality_config()))
    thresholds = dict(settings.get("thresholds") or {})
    defaults = default_frame_quality_config()["thresholds"]
    thresholds = {**defaults, **thresholds}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = deepcopy(dict(raw))
        grouped[str(row.get("run_id") or "run")].append(row)

    assessed: list[dict[str, Any]] = []
    run_summaries: dict[str, dict[str, int]] = {}
    for run_id, run_rows in grouped.items():
        raw_valid = [row for row in run_rows if row.get("raw_signature")]
        corrected_valid = [row for row in run_rows if row.get("corrected_signature")]
        raw_metrics_by_id = {
            str(row.get("frame_id") or ""): metrics
            for row, metrics in zip(
                raw_valid,
                _temporal_metrics(raw_valid, "raw_signature"),
                strict=True,
            )
        }
        corrected_metrics_by_id = {
            str(row.get("frame_id") or ""): metrics
            for row, metrics in zip(
                corrected_valid,
                _temporal_metrics(corrected_valid, "corrected_signature"),
                strict=True,
            )
        }
        counts = {status: 0 for status in QUALITY_STATUSES}
        for sequence_index, row in enumerate(run_rows):
            frame_id = str(row.get("frame_id") or "")
            raw_metrics = raw_metrics_by_id.get(frame_id, {})
            corrected_metrics = corrected_metrics_by_id.get(frame_id, {})
            flags: list[str] = []
            if row.get("analysis_error") or not raw_metrics or not corrected_metrics:
                status = "analysis_failed"
                flags.append("frame_quality_analysis_failed")
            else:
                raw_flags = _metric_flags(
                    raw_metrics,
                    thresholds,
                    prefix="raw",
                )
                corrected_flags = _metric_flags(
                    corrected_metrics,
                    thresholds,
                    prefix="corrected",
                )
                correction_qc = dict(row.get("correction_qc") or {})
                if correction_qc.get("flagged"):
                    corrected_flags.append("radiometric_correction_qc_failed")
                flags.extend(raw_flags)
                flags.extend(corrected_flags)
                severe_transient = (
                    corrected_metrics["transient_spatial_log"]
                    > thresholds["severe_transient_spatial_log"]
                    and corrected_metrics["spatial_changed_fraction"] > 0.25
                )
                severe_exposure = any(
                    flag
                    in {
                        "corrected_extreme_underexposure",
                        "corrected_extreme_overexposure",
                    }
                    for flag in corrected_flags
                )
                if corrected_flags and (severe_transient or severe_exposure):
                    status = "exclusion_suggested"
                elif corrected_flags:
                    status = "suspicious"
                elif raw_flags:
                    status = "corrected"
                else:
                    status = "ok"
            counts[status] += 1
            binding = {
                "frame_id": frame_id,
                "source_sha256": str(row.get("source_sha256") or ""),
                "correction_dependency_sha256": str(
                    row.get("correction_dependency_sha256") or ""
                ),
            }
            assessed.append(
                {
                    **row,
                    "sequence_index": sequence_index,
                    "quality_binding_sha256": _canonical_digest(binding),
                    "automatic_status": status,
                    "automatic_flags": list(dict.fromkeys(flags)),
                    "raw_metrics": raw_metrics,
                    "corrected_metrics": corrected_metrics,
                    "blacklist_candidate": status
                    in {"exclusion_suggested", "analysis_failed"},
                    "automatically_excluded": False,
                }
            )
        run_summaries[run_id] = counts

    return {
        "schema": 1,
        "format": FRAME_QUALITY_ASSESSMENT_FORMAT,
        "algorithm": FRAME_QUALITY_ALGORITHM,
        "config": settings,
        "frame_count": len(assessed),
        "run_count": len(grouped),
        "runs": run_summaries,
        "frames": assessed,
    }


def scan_frame_quality(
    frames: Sequence[Mapping[str, Any]],
    correction: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    workers: int = 2,
    max_workers: int | None = None,
    worker_limit: Callable[[], int] | None = None,
    worker_status: Callable[[int, int], None] | None = None,
    cancelled: Event | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Read frames, apply correction, and build one quality assessment.

    ``worker_limit`` may change while the scan is running. New work is submitted
    up to the latest requested limit after every completed frame. Reducing the
    limit never interrupts an active image read.
    """

    import darsia

    settings = deepcopy(dict(config or default_frame_quality_config()))
    grid_shape = settings.get("signature_grid") or [6, 10]
    algorithm = str(correction.get("algorithm") or IDENTITY_ALGORITHM)
    correction_dependency = CorrectionRevisionStore.dependency_sha256(correction)
    reference_image: np.ndarray | None = None
    if algorithm in COLORCHECKER_ALGORITHMS:
        reference_path = Path(str(correction.get("reference_image") or ""))
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        reference_image = np.asarray(darsia.imread(reference_path).img)
    elif algorithm != IDENTITY_ALGORITHM:
        raise ValueError(f"Unsupported image-correction algorithm: {algorithm!r}")

    ordered = [deepcopy(dict(frame)) for frame in frames]
    total = len(ordered)

    def analyze(frame: Mapping[str, Any]) -> dict[str, Any]:
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("Frame-quality screening was cancelled")
        source = Path(str(frame.get("source_path") or "")).expanduser().resolve()
        result = {
            "frame_id": str(frame.get("frame_id") or ""),
            "run_id": str(frame.get("run_id") or "run"),
            "source_path": str(source),
            "source_relative_path": str(frame.get("relative_path") or ""),
            "source_sha256": str(frame.get("sha256") or ""),
            "correction_revision_id": str(correction.get("revision_id") or ""),
            "correction_dependency_sha256": correction_dependency,
        }
        try:
            original = np.asarray(darsia.imread(source).img)
            if algorithm == UNIFORM24_ALGORITHM:
                if reference_image is None:
                    raise RuntimeError("The ColorChecker reference was not loaded")
                fitted = fit_uniform24_array_transform(
                    reference_image,
                    original,
                    corner=str(correction.get("colorchecker_corner") or "upper_right"),
                    roi_rc=correction.get("colorchecker_roi_rc"),
                    settings=dict(correction.get("settings") or {}),
                )
                corrected = apply_uniform24_array_transform(
                    original,
                    fitted["transform"],
                )
                correction_qc = deepcopy(dict(fitted.get("diagnostics") or {}))
            elif algorithm == DARSIA_BASELINE_COLOR_ALGORITHM:
                if reference_image is None:
                    raise RuntimeError("The ColorChecker reference was not loaded")
                fitted = apply_darsia_baseline_array_correction(
                    reference_image,
                    original,
                    corner=str(correction.get("colorchecker_corner") or "upper_right"),
                    roi_rc=correction.get("colorchecker_roi_rc"),
                    settings=dict(correction.get("settings") or {}),
                )
                corrected = np.asarray(fitted["corrected"], dtype=np.float32)
                correction_qc = deepcopy(dict(fitted.get("diagnostics") or {}))
            else:
                corrected = original
                correction_qc = {
                    "algorithm": IDENTITY_ALGORITHM,
                    "flagged": False,
                    "flag_reasons": [],
                }
            result.update(
                {
                    "raw_signature": frame_quality_signature(
                        original,
                        grid_shape=grid_shape,
                    ),
                    "corrected_signature": frame_quality_signature(
                        corrected,
                        grid_shape=grid_shape,
                    ),
                    "correction_qc": correction_qc,
                    "analysis_error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-frame failure
            result["analysis_error"] = f"{type(exc).__name__}: {exc}"
        return result

    completed = 0
    results: list[dict[str, Any] | None] = [None] * total
    initial_worker_count = max(1, min(int(workers), max(1, total)))
    worker_capacity = max(
        initial_worker_count,
        min(int(max_workers or initial_worker_count), max(1, total)),
    )
    worker_history: list[dict[str, int]] = []
    last_requested = 0

    def requested_workers() -> int:
        requested = worker_limit() if worker_limit is not None else initial_worker_count
        return max(1, min(int(requested), worker_capacity, max(1, total)))

    next_index = 0
    futures: dict[Any, int] = {}
    with ThreadPoolExecutor(max_workers=worker_capacity) as pool:

        def fill_available_slots() -> None:
            nonlocal next_index, last_requested
            requested = requested_workers()
            if requested != last_requested:
                worker_history.append(
                    {
                        "completed_frames": completed,
                        "requested_workers": requested,
                    }
                )
                last_requested = requested
            while next_index < total and len(futures) < requested:
                futures[pool.submit(analyze, ordered[next_index])] = next_index
                next_index += 1
            if worker_status is not None:
                worker_status(len(futures), requested)

        fill_available_slots()
        while futures:
            done, _pending = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                index = futures.pop(future)
                results[index] = future.result()
                completed += 1
                if progress is not None:
                    progress(completed, total)
            if cancelled is not None and cancelled.is_set():
                for pending in futures:
                    pending.cancel()
                raise InterruptedError("Frame-quality screening was cancelled")
            fill_available_slots()

    assessment = assess_frame_quality_rows(
        [row for row in results if row is not None],
        config=settings,
    )
    assessment.update(
        {
            "generated_utc": utc_now(),
            "correction_revision_id": str(correction.get("revision_id") or ""),
            "correction_dependency_sha256": correction_dependency,
            "initial_worker_count": initial_worker_count,
            "maximum_worker_count": worker_capacity,
            "dynamic_worker_limit": worker_limit is not None,
            "worker_history": worker_history,
        }
    )
    return assessment


class FrameQualityStore:
    """Persist assessments, append-only decisions, and derived blacklists."""

    def __init__(self, project: FluidFlowerProject) -> None:
        self.project = project
        self.result_root = project.root / "results" / "image_correction" / "quality"
        self.assessment_root = self.result_root / "assessments"
        self.override_root = project.root / "overrides" / "image_correction" / "quality"
        self.current_path = self.override_root / "current_assessment.json"
        self.decisions_path = self.override_root / "decisions.jsonl"
        self.blacklist_path = self.override_root / "active_blacklist.json"
        self.blacklist_csv_path = self.override_root / "active_blacklist.csv"
        self.run_blacklist_root = self.override_root / "runs"

    def current(self) -> dict[str, Any]:
        return _read_json(self.current_path)

    def save_assessment(
        self,
        assessment: Mapping[str, Any],
        *,
        actor: str,
    ) -> dict[str, Any]:
        reviewer = str(actor).strip()
        if not reviewer:
            raise ValueError("A reviewer name is required for a quality assessment")
        row = deepcopy(dict(assessment))
        if row.get("format") != FRAME_QUALITY_ASSESSMENT_FORMAT:
            raise ValueError("Unsupported frame-quality assessment format")
        assessment_id = uuid.uuid4().hex
        row.update(
            {
                "assessment_id": assessment_id,
                "recorded_utc": utc_now(),
                "recorded_by": reviewer,
            }
        )
        row["content_sha256"] = _canonical_digest(row)
        destination = self.assessment_root / assessment_id / "assessment.json"
        _atomic_json(destination, row)
        pointer = {
            "schema": 1,
            "format": FRAME_QUALITY_ASSESSMENT_FORMAT,
            "assessment_id": assessment_id,
            "path": str(destination),
            "content_sha256": row["content_sha256"],
            "correction_revision_id": str(row.get("correction_revision_id") or ""),
            "correction_dependency_sha256": str(
                row.get("correction_dependency_sha256") or ""
            ),
            "recorded_utc": row["recorded_utc"],
            "recorded_by": row["recorded_by"],
        }
        _atomic_json(self.current_path, pointer)
        self._write_blacklist(row)
        self.project._audit(
            "frame_quality_assessment_saved",
            reviewer,
            {
                "assessment_id": assessment_id,
                "frame_count": int(row.get("frame_count") or 0),
                "run_count": int(row.get("run_count") or 0),
                "content_sha256": row["content_sha256"],
            },
        )
        return row

    def current_assessment(self) -> dict[str, Any]:
        pointer = self.current()
        path = Path(str(pointer.get("path") or ""))
        if not path.is_file():
            return {}
        payload = _read_json(path)
        expected = str(pointer.get("content_sha256") or "")
        stored = str(payload.get("content_sha256") or "")
        actual = _canonical_digest(
            {key: value for key, value in payload.items() if key != "content_sha256"}
        )
        if expected and (stored != expected or actual != expected):
            raise RuntimeError("The active frame-quality assessment has changed")
        return payload

    def decisions(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.decisions_path)

    def latest_decisions(
        self,
        assessment: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        selected = dict(assessment or self.current_assessment())
        bindings = {
            str(row.get("frame_id") or ""): str(row.get("quality_binding_sha256") or "")
            for row in selected.get("frames") or []
            if isinstance(row, Mapping)
        }
        latest: dict[str, dict[str, Any]] = {}
        for decision in self.decisions():
            frame_id = str(decision.get("frame_id") or "")
            if (
                frame_id in bindings
                and str(decision.get("quality_binding_sha256") or "")
                == bindings[frame_id]
            ):
                latest[frame_id] = decision
        return latest

    def effective_frames(
        self,
        assessment: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        selected = dict(assessment or self.current_assessment())
        decisions = self.latest_decisions(selected)
        rows: list[dict[str, Any]] = []
        for raw in selected.get("frames") or []:
            if not isinstance(raw, Mapping):
                continue
            row = deepcopy(dict(raw))
            decision = deepcopy(
                dict(decisions.get(str(row.get("frame_id") or "")) or {})
            )
            manual = str(decision.get("decision") or "")
            if manual == "clear":
                manual = ""
            row["manual_decision"] = manual
            row["manual_reason"] = str(decision.get("reason") or "")
            row["manual_reviewer"] = str(decision.get("recorded_by") or "")
            row["included_in_analysis"] = manual != "exclude"
            row["effective_status"] = (
                "excluded" if manual == "exclude" else row.get("automatic_status")
            )
            rows.append(row)
        return rows

    def set_decision(
        self,
        frame_id: str,
        decision: str,
        *,
        actor: str,
        reason: str = "",
    ) -> dict[str, Any]:
        normalized = str(decision).strip().casefold()
        if normalized not in QUALITY_DECISIONS:
            raise ValueError(f"Unsupported frame-quality decision: {decision!r}")
        reviewer = str(actor).strip()
        if not reviewer:
            raise ValueError("A reviewer name is required for a frame decision")
        note = str(reason).strip()
        if normalized == "exclude" and not note:
            raise ValueError("An exclusion requires a review note")
        assessment = self.current_assessment()
        frame = next(
            (
                dict(row)
                for row in assessment.get("frames") or []
                if isinstance(row, Mapping)
                and str(row.get("frame_id") or "") == str(frame_id)
            ),
            None,
        )
        if frame is None:
            raise KeyError(f"Unknown frame in active quality assessment: {frame_id}")
        previous_blacklist = {
            str(row.get("frame_id") or "")
            for row in self.active_blacklist().get("entries") or []
            if isinstance(row, Mapping)
        }
        record = {
            "schema": 1,
            "format": FRAME_QUALITY_DECISION_FORMAT,
            "decision_id": uuid.uuid4().hex,
            "assessment_id": str(assessment.get("assessment_id") or ""),
            "frame_id": str(frame_id),
            "run_id": str(frame.get("run_id") or ""),
            "source_path": str(frame.get("source_path") or ""),
            "source_sha256": str(frame.get("source_sha256") or ""),
            "quality_binding_sha256": str(frame.get("quality_binding_sha256") or ""),
            "decision": normalized,
            "reason": note,
            "recorded_utc": utc_now(),
            "recorded_by": reviewer,
        }
        _append_jsonl(self.decisions_path, record)
        blacklist = self._write_blacklist(assessment)
        current_blacklist = {
            str(row.get("frame_id") or "")
            for row in blacklist.get("entries") or []
            if isinstance(row, Mapping)
        }
        if current_blacklist != previous_blacklist:
            invalidation_reason = (
                "The confirmed frame blacklist changed. Dependent phase and mass "
                "products must be regenerated."
            )
            self.project.set_module_status(
                "image_correction",
                "review_required",
                updated_by=reviewer,
                reason=invalidation_reason,
            )
            run_id = str(frame.get("run_id") or "")
            registered_runs = {
                str(row.get("run_id") or "")
                for row in self.project.manifest.get("runs") or []
                if isinstance(row, Mapping)
            }
            if run_id in registered_runs:
                self.project.set_run_module_status(
                    [run_id],
                    "image_correction",
                    "review_required",
                    updated_by=reviewer,
                    reason=invalidation_reason,
                )
        self.project._audit(
            "frame_quality_decision_saved",
            reviewer,
            {
                "frame_id": str(frame_id),
                "run_id": str(frame.get("run_id") or ""),
                "decision": normalized,
                "reason": note,
                "blacklist_count": len(current_blacklist),
            },
        )
        return record

    def active_blacklist(self) -> dict[str, Any]:
        return _read_json(self.blacklist_path)

    def excluded_paths(self) -> set[str]:
        return {
            str(row.get("source_path") or "")
            for row in self.active_blacklist().get("entries") or []
            if isinstance(row, Mapping) and str(row.get("source_path") or "")
        }

    def dependency_sha256(self) -> str:
        return str(self.active_blacklist().get("content_sha256") or "")

    def dependency_reference(self) -> dict[str, Any]:
        """Return the active immutable input reference for downstream stages."""

        blacklist = self.active_blacklist()
        return {
            "assessment_id": str(blacklist.get("assessment_id") or ""),
            "assessment_sha256": str(blacklist.get("assessment_sha256") or ""),
            "blacklist_path": (
                str(self.blacklist_path) if self.blacklist_path.is_file() else ""
            ),
            "blacklist_sha256": str(blacklist.get("content_sha256") or ""),
            "excluded_frame_count": int(blacklist.get("entry_count") or 0),
            "automatic_exclusion": False,
        }

    def filter_frames(
        self,
        frames: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove only manually confirmed exclusions from a frame sequence."""

        def normalized_path(value: Any) -> str:
            raw = str(value or "").strip()
            if not raw:
                return ""
            return os.path.normcase(
                os.path.abspath(os.path.normpath(os.path.expanduser(raw)))
            )

        excluded = {normalized_path(path) for path in self.excluded_paths() if path}
        return [
            deepcopy(dict(row))
            for row in frames
            if normalized_path(
                row.get("source_path") or row.get("path") or row.get("image_path") or ""
            )
            not in excluded
        ]

    def _write_blacklist(self, assessment: Mapping[str, Any]) -> dict[str, Any]:
        entries = [
            {
                "run_id": str(row.get("run_id") or ""),
                "frame_id": str(row.get("frame_id") or ""),
                "source_path": str(row.get("source_path") or ""),
                "source_sha256": str(row.get("source_sha256") or ""),
                "reason": str(row.get("manual_reason") or ""),
                "reviewer": str(row.get("manual_reviewer") or ""),
                "automatic_status": str(row.get("automatic_status") or ""),
                "automatic_flags": list(row.get("automatic_flags") or []),
            }
            for row in self.effective_frames(assessment)
            if not bool(row.get("included_in_analysis", True))
        ]
        entries.sort(key=lambda row: (row["run_id"], row["source_path"]))
        payload = {
            "schema": 1,
            "format": FRAME_QUALITY_BLACKLIST_FORMAT,
            "assessment_id": str(assessment.get("assessment_id") or ""),
            "assessment_sha256": str(assessment.get("content_sha256") or ""),
            "correction_revision_id": str(
                assessment.get("correction_revision_id") or ""
            ),
            "correction_dependency_sha256": str(
                assessment.get("correction_dependency_sha256") or ""
            ),
            "generated_utc": utc_now(),
            "automatic_exclusion": False,
            "entry_count": len(entries),
            "entries": entries,
        }
        payload["content_sha256"] = _canonical_digest(
            {
                key: value
                for key, value in payload.items()
                if key not in {"generated_utc", "content_sha256"}
            }
        )
        _atomic_json(self.blacklist_path, payload)
        self._write_blacklist_csv(self.blacklist_csv_path, entries)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in entries:
            grouped[row["run_id"]].append(row)
        self.run_blacklist_root.mkdir(parents=True, exist_ok=True)
        run_ids = {
            str(row.get("run_id") or "")
            for row in assessment.get("frames") or []
            if isinstance(row, Mapping)
        }
        for run_id in run_ids:
            self._write_blacklist_csv(
                self.run_blacklist_root / f"{run_id}.csv",
                grouped.get(run_id, []),
            )
        return payload

    @staticmethod
    def _write_blacklist_csv(
        path: Path,
        entries: Sequence[Mapping[str, Any]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "run_id",
                    "frame_id",
                    "source_path",
                    "source_sha256",
                    "reason",
                    "reviewer",
                    "automatic_status",
                    "automatic_flags",
                ),
            )
            writer.writeheader()
            for raw in entries:
                row = dict(raw)
                row["automatic_flags"] = ";".join(row.get("automatic_flags") or [])
                writer.writerow(row)
        os.replace(temporary, path)


__all__ = [
    "FRAME_QUALITY_ALGORITHM",
    "FRAME_QUALITY_ASSESSMENT_FORMAT",
    "FRAME_QUALITY_BLACKLIST_FORMAT",
    "FRAME_QUALITY_DECISION_FORMAT",
    "QUALITY_DECISIONS",
    "QUALITY_STATUSES",
    "FrameQualityStore",
    "assess_frame_quality_rows",
    "default_frame_quality_config",
    "frame_quality_signature",
    "scan_frame_quality",
]
