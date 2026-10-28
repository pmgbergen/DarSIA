"""Materialize model-bound phase-review assets from corrected source frames.

The source review campaign is treated as immutable evidence.  Generated phase
maps, active-learning proposals, and a compact compatibility manifest are
written to a model-specific cache that can be opened by the established
FluidFlower review application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import uuid
from dataclasses import replace

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_features import SPATIAL_CONTEXT_FRACTION
from darsia.fluidflower.phase_learning import (
    HierarchicalPhaseModel,
    propose_uncertain_phase_regions,
)
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.phase_rendering import (
    PHASE_DISPLAY_POLICY, two_phase_masks,
)


PHASE_VALIDATION_CAMPAIGN_FORMAT = "darsia-phase-validation-campaign-v1"
PHASE_REVIEW_PROPOSAL_SCHEMA = 2
# Structured per-frame timing for callers that opt in with
# ``progress.accepts_frame_timing = True``; every other callback only sees text.
FRAME_TIMING_PREFIX = "PHASE_FRAME_TIMING "
# Output-only work of a finished frame (PNG encoding, review proposals, the production
# diagnostics archive) and the JPEG decode of the next frame run beside the ordered loop.
# Measured 2026-09-17: that work was 12-15 s of a 37 s loop period in review mode. The files
# are byte-identical to inline writing; set this variable to "0" to write inline as before.
BACKGROUND_OUTPUTS_ENV = "PHASE_BACKGROUND_OUTPUTS"
OUTPUT_EXECUTION_BACKGROUND = "background-thread-v1"
OUTPUT_EXECUTION_INLINE = "inline"
_PROPOSAL_FIELDS = ("untouched_water_probability", "aqueous_co2_probability", "free_co2_gas_probability")
# Review proposals are interpreter-bound. On a thread of this process they hold the interpreter
# lock and slow the physics (measured 2026-09-17 on real frames: +10-35 %), so they run in this
# many helper processes instead; "0" keeps them on the output thread. The proposals are identical.
PROPOSAL_PROCESSES_ENV = "PHASE_PROPOSAL_PROCESSES"
PHASE_FILL_COLORS = {
    "aqueous": (49, 217, 160, 255),
    "mixed": (255, 191, 71, 255),
    "gas": (242, 76, 157, 255),
    "unresolved": (30, 210, 255, 72),
}
_V5_METRIC_OUTPUT_GUARDS = frozenset({
    "bottom_gas_guard", "output_guards", "free_gas_cleanup",
    "region_edge_guard",
})


def _v5_physical_scale_warning(physics_profile: Mapping[str, Any] | None,
                               frame_bindings: Sequence[Mapping[str, Any]]) -> str | None:
    """Keep provisional metric geometry out of millimetre-based decisions.

    Optical review can proceed on independently checked native pixels while
    absolute rig/injector scale is still under audit. Position-dependent
    millimetre rules remain prohibited. A separate, explicitly nominal
    2.5 mm2 speckle filter is allowed only on V5 manual-review maps.
    """
    missing = [int(row.get("frame_index", index))
               for index, row in enumerate(frame_bindings)
               if row.get("physical_scale_qa_pass") is not True]
    if not missing:
        return None
    profile = physics_profile or {}
    requested = sorted(_V5_METRIC_OUTPUT_GUARDS & set(profile))
    nominal_review = profile.get("v5_nominal_review_gas_minimum")
    if requested == ["free_gas_cleanup"] and nominal_review == {
            "contract": "ffac-v5-nominal-review-gas-minimum-v1",
            "minimum_area_mm2": 2.5,
            "physical_scale_qa_pass": False,
            "mass_analysis_allowed": False,
            } and profile.get("free_gas_cleanup") == {
                "format": "native-free-gas-minimum-v1",
                "minimum_area_mm2": 2.5,
            }:
        return ("Physical rig/injector scale is unverified. The nominal 2.5 mm² "
                "free-gas speckle filter is for manual Studio review only; "
                "physical/mass interpretation is unavailable.")
    if requested:
        raise ValueError(
            "V5 physical scale is unverified for frames "
            f"{missing[:8]}; metric output guards require independent rig/"
            f"injector scale QA: {requested}")
    return ("Physical rig/injector scale is unverified; optical phase maps are "
            "for manual Studio review only and physical/mass interpretation "
            "is unavailable.")


def _read_json(path: Path) -> dict[str, Any]:
    from .phase_saved_review_metadata import read_model_json
    return read_model_json(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _resolve(root: Path, value: object) -> Path:
    from .review_storage import resolve_review_path
    path = Path(str(value or ""))
    return resolve_review_path(path if path.is_absolute() else root / path)


def _model_identity(model_root: Path) -> tuple[str, str]:
    summary = _read_json(model_root / "summary.json")
    model_id = str(summary.get("model_id") or model_root.name).strip()
    digest = hashlib.sha256()
    for path in (
        model_root / "summary.json",
        model_root / "models" / "generic" / "co2_presence_signal.json",
        model_root / "models" / "generic" / "gas_within_co2_signal.json",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return model_id, digest.hexdigest()


def _source_jobs(source_root: Path) -> list[dict[str, Any]]:
    manifest = _read_json(source_root / "postflight_manifest.json")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in manifest.get("jobs") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        run_id = str(row.get("run") or "").strip().casefold()
        if run_id:
            grouped.setdefault(run_id, []).append(row)
    selected: list[dict[str, Any]] = []
    for run_id, rows in sorted(grouped.items()):
        rows.sort(
            key=lambda row: (
                "physics" not in str(row.get("role") or "").casefold(),
                "reviewed" not in str(row.get("role") or "").casefold(),
                int(row.get("grid_order") or 999),
            )
        )
        selected.append({**rows[0], "run": run_id})
    if not selected:
        raise ValueError(f"No review candidates were found in {source_root}")
    return selected


def _candidate_paths(source_root: Path, job: Mapping[str, Any]) -> tuple[Path, Path]:
    run_id = str(job["run"]).casefold()
    variant = str(job.get("variant") or "")
    candidate = (
        _resolve(source_root, job["candidate_folder"])
        if job.get("candidate_folder")
        else (source_root / "postflight" / variant / run_id).resolve()
    )
    review_root = (
        _resolve(source_root, job["review_output_dir"])
        if job.get("review_output_dir")
        else (source_root / "manual_pairwise_review").resolve()
    )
    from .review_storage import resolve_review_path
    return resolve_review_path(candidate), resolve_review_path(review_root)


def _frame_timing_summary(run_id: str, frame_count: int, timings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Actual seconds of the ordered loop for the last frame and for the run so far.

    ``total_s`` spans loading the prepared photograph to the written phase map.
    Waiting (for a shared optical worker or CPU slot) is reported separately
    from work, and the optical worker's own seconds are not added because they
    overlap other frames.  Image preparation from the raw photograph happens
    before this loop and is reported per run by the series batch.
    """
    last = timings[-1]
    totals = sorted(float(row["total_s"]) for row in timings if row.get("total_s") is not None)
    median = totals[len(totals) // 2] if len(totals) % 2 else (totals[len(totals) // 2 - 1] + totals[len(totals) // 2]) / 2
    waited = float(last.get("optical_wait_s") or 0.0) + float(last.get("physics_slot_wait_s") or 0.0)
    physics = float(last.get("physics_s") or 0.0) - float(last.get("physics_slot_wait_s") or 0.0)
    decisions = [float(row["decision_s"]) for row in timings if row.get("decision_s") is not None]
    clock = last.get("optical_worker_clock") if isinstance(last.get("optical_worker_clock"), Mapping) else {}
    backend = clock.get("backend")
    done = len(timings)
    summary = {
        "run": run_id, "frame_index": int(last["frame_index"]), "frames_done": done, "frame_count": int(frame_count),
        "last_total_s": round(float(last["total_s"]), 2), "last_waiting_s": round(waited, 2),
        "last_physics_s": round(physics, 2), "last_writing_s": round(float(last.get("writing_s") or 0.0), 2),
        "last_optical_backend": backend, "last_optical_worker_s": last.get("optical_worker_total_s"),
        "median_total_s": round(median, 2), "elapsed_s": round(sum(totals), 1),
        "eta_s": round(median * max(0, int(frame_count) - done), 1),
    }
    optical = f"optical {backend} {float(summary['last_optical_worker_s']):.1f} s" if backend and summary[
        "last_optical_worker_s"] is not None else "optical"
    if last.get("decision_pipelined"):
        # physics_s is the chronological part; the final decision runs beside the next photograph.
        summary["last_decision_s"] = round(decisions[-1], 2) if decisions else None
        optical += ", final decision alongside" + (f" {decisions[-1]:.0f} s" if decisions else "")
    summary["text"] = (f"last image {summary['last_total_s']:.0f} s ({optical}, physics {physics:.0f} s, "
                       f"waited {waited:.0f} s) · median {median:.0f} s/image · "
                       f"{summary['elapsed_s'] / 60:.1f} min so far, about {summary['eta_s'] / 60:.0f} min left")
    return summary


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _select_native_current(prepared: np.ndarray | None, v5_input: Any | None,
                           frame_path: Path, frame_index: int,
                           frame_canvas: np.ndarray | None,
                           content_box: tuple[int, int, int, int]) -> np.ndarray:
    """Choose the exact prepared float frame or the source-bound V5 sidecar."""
    if prepared is not None:
        return prepared
    if v5_input is not None:
        return v5_input.load(frame_path, f"frame_{frame_index:03d}")
    if frame_canvas is None:
        raise ValueError("No native frame input is available")
    x, y, width, height = content_box
    return frame_canvas[y:y + height, x:x + width]


def predict_image_tiled(
    model: HierarchicalPhaseModel,
    image_rgb: np.ndarray,
    baseline_rgb: np.ndarray,
    *,
    previous_rgb: np.ndarray | None = None,
    clearwater_rgb: np.ndarray | None = None,
    tile_height: int = 384,
    texture_sigma: float = 2.0,
    radiometry_profile: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Apply one optical model in bounded-memory horizontal tiles."""
    from darsia._opencv_runtime import scientific_thread, require_ipp_off
    scientific_thread("predict_image_tiled:before")
    current = np.asarray(image_rgb)
    baseline = np.asarray(baseline_rgb)
    if current.shape != baseline.shape or current.ndim != 3:
        raise ValueError("Current and baseline images must be matching RGB arrays")
    height, width = current.shape[:2]
    spatial_sigma = max(
        float(texture_sigma), SPATIAL_CONTEXT_FRACTION * float(min(height, width))
    )
    overlap = max(16, int(math.ceil(6.0 * spatial_sigma)))
    outputs: dict[str, np.ndarray] = {}
    for start in range(0, height, max(32, int(tile_height))):
        stop = min(height, start + max(32, int(tile_height)))
        extended_start = max(0, start - overlap)
        extended_stop = min(height, stop + overlap)
        selection = slice(extended_start, extended_stop)
        predicted = model.predict_image(
            current[selection],
            baseline[selection],
            previous_rgb=(
                None if previous_rgb is None else np.asarray(previous_rgb)[selection]
            ),
            clearwater_rgb=(
                None if clearwater_rgb is None else np.asarray(clearwater_rgb)[selection]
            ),
            texture_sigma=texture_sigma,
            spatial_context_sigma=spatial_sigma,
            **({"radiometry_profile": radiometry_profile} if radiometry_profile is not None else {}),
        )
        local = slice(start - extended_start, stop - extended_start)
        for name, values in predicted.items():
            output = outputs.setdefault(name, np.empty((height, width), dtype=np.float32))
            output[start:stop] = np.asarray(values[local], dtype=np.float32)
    require_ipp_off("predict_image_tiled:after")
    return outputs


def _phase_masks(predictions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = two_phase_masks(predictions)
    # Keep the legacy empty layer path readable by older review clients.
    return {**masks, "mixed": np.zeros_like(masks["gas"])}


def _retain_v5_observation_only_scores(
    physical_predictions: Mapping[str, np.ndarray],
    optical_predictions: Mapping[str, np.ndarray],
    *,
    observable: np.ndarray,
    physics_supported: np.ndarray,
    certified_rig: np.ndarray | None = None,
    review_safe_support: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Keep native optical evidence where the physical context is unknown.

    The regional decision is valid only inside its source-bound facies and
    permeability support.  A missing physical context cannot erase an
    observed overflow label or invent a porous-sand mass interpretation.
    """
    observable = np.asarray(observable, dtype=bool)
    physical = np.asarray(physics_supported, dtype=bool)
    if (observable.shape != physical.shape or not observable.any()
            or np.any(physical & ~observable)):
        raise ValueError("V5 physical support must be a nonempty observation subset")
    # A photo-certified piece of the rig is not overflow.  Keep the uncertain
    # boundary eligible for observation so a source-backed post-injection
    # overflow signal is not erased by a hard sand-connectivity rule.
    if certified_rig is None:
        rig = np.zeros_like(observable)
    else:
        rig = np.asarray(certified_rig)
        if (rig.dtype != np.dtype("bool") or rig.shape != observable.shape
                or np.any(rig & physical)):
            raise ValueError("V5 certified rig conflicts with physical support")
    if review_safe_support is None:
        supported = observable
    else:
        safe = np.asarray(review_safe_support)
        if (safe.dtype != np.dtype("bool") or safe.shape != observable.shape
                or np.any(physical & ~safe)):
            raise ValueError("V5 review-safe source support conflicts with physical support")
        supported = observable & safe
    outside = supported & ~physical & ~rig
    scores: dict[str, np.ndarray] = {}
    for name in ("co2_presence_probability", "gas_probability_conditional_on_co2"):
        optical = np.asarray(optical_predictions[name], dtype=np.float32)
        decided = np.asarray(physical_predictions[name], dtype=np.float32)
        if (optical.shape != observable.shape or decided.shape != observable.shape
                or not np.isfinite(optical).all() or not np.isfinite(decided).all()
                or np.any((optical < 0) | (optical > 1))
                or np.any((decided < 0) | (decided > 1))):
            raise ValueError("V5 optical/physical scores need finite native unit grids")
        scores[name] = np.where(physical, decided,
                                np.where(outside, optical, 0)).astype(np.float32)
    co2 = scores["co2_presence_probability"]
    gas = scores["gas_probability_conditional_on_co2"]
    free = co2 * gas
    report = {
        "observation_only_pixels": int(outside.sum()),
        "observation_only_co2_pixels": int(np.count_nonzero(outside & (co2 >= .5))),
        "observation_only_gas_pixels": int(np.count_nonzero(outside & (co2 >= .5) & (gas >= .6))),
    }
    if review_safe_support is not None:
        report["unresolved_photo_or_feature_pixels"] = int(
            np.count_nonzero(observable & ~supported))
        report["review_safe_support_pixels"] = int(np.count_nonzero(supported))
    if certified_rig is not None:
        report["source_excluded_rig_pixels"] = int(np.count_nonzero(observable & rig))
    return {
        **physical_predictions, **scores,
        "untouched_water_probability": 1 - co2,
        "aqueous_co2_probability": co2 - free,
        "free_co2_gas_probability": free,
    }, report


def _write_phase_maps(
    root: Path,
    run_id: str,
    fingerprint: str,
    frame_index: int,
    canvas_shape: tuple[int, int],
    content_box: tuple[int, int, int, int],
    predictions: Mapping[str, np.ndarray],
) -> None:
    _write_mask_maps(root, run_id, fingerprint, frame_index, canvas_shape, content_box,
                     _phase_masks(predictions))


def _save_phase_png(image: Image.Image, path: Path) -> dict[str, float]:
    """Encode locally, then publish one complete PNG; keep NAS writes separate."""
    import shutil
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scratch = os.environ.get('PHASE_SCRATCH_DIR') or None
    if scratch:
        Path(scratch).mkdir(parents=True, exist_ok=True)
    temporary = None
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 ** 2, dir=scratch) as encoded:
        image.save(encoded, format='PNG', compress_level=1)
        size = encoded.tell()
        encoded.seek(0)
        encoded_at = time.perf_counter()
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                shutil.copyfileobj(encoded, stream, length=1024 ** 2)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dict(encode_s=encoded_at - started, write_s=time.perf_counter() - encoded_at,
                bytes_written=size)


def _write_mask_maps(
    root: Path,
    run_id: str,
    fingerprint: str,
    frame_index: int,
    canvas_shape: tuple[int, int],
    content_box: tuple[int, int, int, int],
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Encode the categorical layers of one frame from its boolean masks.

    The masks are new arrays that nothing else holds, so this may run after the
    ordered loop has moved on to the next frame.
    """
    canvas_height, canvas_width = canvas_shape
    x, y, width, height = content_box
    folder = root / "native_phase_maps" / run_id / f"A_{fingerprint[:16]}"
    metrics = dict(encode_s=0.0, write_s=0.0, bytes_written=0)
    def save(image, path):
        for key, value in _save_phase_png(image, path).items():
            metrics[key] += value
    combined = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    for phase, mask in masks.items():
        if mask.shape != (height, width):
            raise ValueError(
                f"Phase-map shape {mask.shape} differs from content box {(height, width)}"
            )
        local = np.zeros((height, width, 4), dtype=np.uint8)
        local[mask] = PHASE_FILL_COLORS[phase]
        combined[y : y + height, x : x + width][mask] = PHASE_FILL_COLORS[phase]
        if phase == "unresolved":
            # Review-only support tint lives in the combined map. It is not a
            # fourth CO2 phase or a scientific output layer.
            continue
        path = folder / phase / f"frame_{frame_index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        save(Image.fromarray(_place_local(local, canvas_shape, x, y), "RGBA"), path)
    folder.mkdir(parents=True, exist_ok=True)
    save(Image.fromarray(combined, "RGBA"), folder / f"frame_{frame_index:03d}.png")
    return {'_phase_output_metrics': metrics}


def _place_local(
    local: np.ndarray, canvas_shape: tuple[int, int], x: int, y: int
) -> np.ndarray:
    canvas = np.zeros((*canvas_shape, 4), dtype=np.uint8)
    height, width = local.shape[:2]
    canvas[y : y + height, x : x + width] = local
    return canvas


def _canvas_proposals(
    predictions: Mapping[str, np.ndarray],
    *,
    run_id: str,
    frame_index: int,
    elapsed_h: float,
    model_id: str,
    candidate_id: str,
    fingerprint: str,
    canvas_shape: tuple[int, int],
    content_box: tuple[int, int, int, int],
    eligible_mask: np.ndarray | None = None,
    facies_labels: np.ndarray | None = None,
    count: int = 16,
    existing_regions: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    raw = propose_uncertain_phase_regions(
        predictions,
        run_id=run_id,
        frame_index=frame_index,
        elapsed_h=elapsed_h,
        model_id=model_id,
        count=count,
        eligible_mask=eligible_mask,
        facies_labels=facies_labels,
        existing_regions=existing_regions,
    )
    canvas_height, canvas_width = canvas_shape
    content_x, content_y, content_width, content_height = content_box
    output: list[dict[str, Any]] = []
    for row in raw:
        local_x = float(row["x"])
        local_y = float(row["y"])
        local_width = float(row["width"])
        local_height = float(row["height"])
        canvas_x = (content_x + local_x * content_width) / canvas_width
        canvas_y = (content_y + local_y * content_height) / canvas_height
        canvas_w = local_width * content_width / canvas_width
        canvas_h = local_height * content_height / canvas_height
        output.append(
            {
                **row,
                "candidate": "A",
                "candidate_id": candidate_id,
                "params_sha256": fingerprint,
                "analysis_domain": ("registered_porous_medium_2d" if eligible_mask is not None
                                    else "registered_visible_camera_plane_2d"),
                "coordinate_space": "normalized_native_review_canvas_16x9",
                "shape": "box",
                "x": canvas_x,
                "y": canvas_y,
                "width": canvas_w,
                "height": canvas_h,
                "points": [
                    [canvas_x, canvas_y],
                    [canvas_x + canvas_w, canvas_y],
                    [canvas_x + canvas_w, canvas_y + canvas_h],
                    [canvas_x, canvas_y + canvas_h],
                ],
                "focus_box": {
                    "x": max(0.0, canvas_x - canvas_w),
                    "y": max(0.0, canvas_y - canvas_h),
                    "width": min(1.0, 3.0 * canvas_w),
                    "height": min(1.0, 3.0 * canvas_h),
                },
            }
        )
    return output


def _write_diagnostic_evidence(root: Path, run_id: str, frame_index: int, physics: Any) -> None:
    """Keep the numerical archive without invoking diagnostic display rendering.

    The scientific sequence modules are fingerprinted by frozen models, so this
    output-only policy deliberately lives outside their runtime source files.
    """
    _write_diagnostic_flags(root, run_id, frame_index, _diagnostic_flags(physics))


def _diagnostic_flags(physics: Any, *, copy: bool = False) -> dict[str, np.ndarray]:
    return {name: (values.copy() if copy else values)
            for name, values in physics.last_diagnostics.items() if values.dtype == bool}


def _write_diagnostic_flags(root: Path, run_id: str, frame_index: int,
                            flags: Mapping[str, np.ndarray]) -> None:
    output = root / run_id
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"frame_{frame_index:03d}.npz").open("xb") as stream:
        np.savez_compressed(stream, **flags)


def _background_outputs_enabled() -> bool:
    return os.environ.get(BACKGROUND_OUTPUTS_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _proposal_inputs(predictions: Mapping[str, np.ndarray], *, copy: bool) -> Mapping[str, np.ndarray]:
    """The three fields proposals read, detached from the sequence when they run later."""
    if not copy:
        return predictions
    return {name: np.array(predictions[name], copy=True) for name in _PROPOSAL_FIELDS}


def _proposal_processes() -> int:
    try:
        requested = max(0, int(os.environ.get(PROPOSAL_PROCESSES_ENV, "2")))
    except ValueError:
        requested = 2
    # The resource planner already budgets helpers beside phase/GPU work.
    # Enforce a portable hardware guard here as well; measured allocations
    # above four are allowed instead of silently truncating their worker count.
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    try:
        import psutil
    except ImportError:
        pass
    else:
        try:
            available = min(available, psutil.cpu_count(logical=True) or available)
            affinity = getattr(psutil.Process(), "cpu_affinity", None)
            if affinity is not None:
                available = min(available, len(affinity()))
        except (OSError, AttributeError, psutil.Error):
            pass
    if os.environ.get("PHASE_CPU_LIMIT") is not None:
        try:
            available = min(available, max(1, int(os.environ["PHASE_CPU_LIMIT"])))
        except ValueError:
            pass
    return min(requested, max(0, int(available) - 1))


_PROPOSAL_GEOMETRY: dict[str, Any] = {}






























__all__ = [
    "PHASE_VALIDATION_CAMPAIGN_FORMAT",
    "materialize_phase_validation_campaign",
    "ensure_review_proposals",
    "predict_image_tiled",
]

def ensure_review_proposals(campaign_root, run_id, *, progress=None):
    """Read saved proposals; historical on-demand inference has been retired."""
    from darsia.fluidflower.phase_saved_review_proposals import ensure_review_proposals as saved
    from darsia.fluidflower.phase_saved_review_metadata import configured_parameters
    return saved(campaign_root, run_id, progress=progress, parameters=configured_parameters("quiz"))
