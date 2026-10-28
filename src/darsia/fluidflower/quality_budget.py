"""Explicit numerical quality and deterministic time-selection budgets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

QUALITY_PROFILE_FORMAT = "darsia-fluidflower-quality-profile-v1"
TIME_SELECTION_FORMAT = "darsia-fluidflower-time-selection-v1"

QUALITY_PROFILE_NAMES = ("reference", "production", "exploratory")
NUMERICAL_RESOLUTION_POLICIES = (
    "native_registered_grid",
    "scale_factor",
    "maximum_dimension",
)


def numerical_dtype_for_backend(backend: str) -> str:
    """Return the explicit dtype encoded by a numerical backend name."""

    value = str(backend).strip().casefold()
    if value.endswith("-float32"):
        return "float32"
    if value.endswith("-float64"):
        return "float64"
    raise ValueError(f"Backend does not encode float32 or float64: {backend}")


def summarize_quality_contract(
    config: Mapping[str, Any],
    *,
    backend: str,
) -> dict[str, str]:
    """Return a compact, human-readable numerical working-basis contract."""

    numerical = dict(config.get("numerical_resolution") or {})
    policy = str(numerical.get("policy") or "native_registered_grid")
    if policy == "native_registered_grid":
        scientific_grid = "native registered grid"
    elif policy == "scale_factor":
        scale = float(numerical.get("scale_factor") or 0.0)
        scientific_grid = f"{100.0 * scale:g}% of registered grid"
    elif policy == "maximum_dimension":
        maximum = int(numerical.get("maximum_dimension_px") or 0)
        scientific_grid = f"maximum {maximum} px"
    else:
        scientific_grid = policy or "unspecified grid"
    preview = dict(config.get("preview") or {})
    preview_maximum = int(preview.get("maximum_dimension_px") or 0)
    preview_quality = int(preview.get("jpeg_quality") or 0)
    compute_dtype = numerical_dtype_for_backend(backend)
    preview_label = f"max {preview_maximum} px, JPEG {preview_quality}"
    label = (
        f"{backend} | {compute_dtype} | scientific grid: {scientific_grid} | "
        f"preview only: {preview_label}"
    )
    return {
        "backend": str(backend),
        "compute_dtype": compute_dtype,
        "scientific_grid": scientific_grid,
        "preview": preview_label,
        "cpu_reference_dtype": str(
            (config.get("precision") or {}).get("cpu_reference_dtype") or "float64"
        ),
        "label": label,
    }


def default_quality_profile(name: str = "production") -> dict[str, Any]:
    """Return an explicit quality contract; previews never define science grids."""

    key = str(name).strip().casefold()
    if key not in QUALITY_PROFILE_NAMES:
        raise ValueError(f"Unsupported quality profile: {name}")
    values = {
        "reference": {
            "preview_maximum_dimension_px": 2560,
            "preview_jpeg_quality": 95,
            "cpu_float64_parity_fraction": 1.0,
        },
        "production": {
            "preview_maximum_dimension_px": 1920,
            "preview_jpeg_quality": 92,
            "cpu_float64_parity_fraction": 0.05,
        },
        "exploratory": {
            "preview_maximum_dimension_px": 1280,
            "preview_jpeg_quality": 85,
            "cpu_float64_parity_fraction": 0.01,
            "analysis_policy": "scale_factor",
            "analysis_scale_factor": 0.5,
        },
    }[key]
    values.setdefault("analysis_policy", "native_registered_grid")
    values.setdefault("analysis_scale_factor", 1.0)
    return {
        "schema": 1,
        "format": QUALITY_PROFILE_FORMAT,
        "name": key,
        "numerical_resolution": {
            "policy": values["analysis_policy"],
            "scale_factor": values["analysis_scale_factor"],
            "maximum_dimension_px": None,
            "minimum_dimension_px": 64,
            "allow_upsampling": False,
            "preview_resolution_is_scientifically_authoritative": False,
            "resampling_contract": "area_average_intensive_nearest_material",
        },
        "precision": {
            "policy": "backend_explicit",
            "compute_dtype": "backend_selected",
            "gpu_dtype": "float32",
            "cpu_reference_dtype": "float64",
            "absolute_parity_tolerance": 1.0e-5,
            "relative_parity_tolerance": 1.0e-4,
            "cpu_float64_parity_fraction": values[
                "cpu_float64_parity_fraction"
            ],
        },
        "preview": {
            "maximum_dimension_px": values["preview_maximum_dimension_px"],
            "jpeg_quality": values["preview_jpeg_quality"],
        },
    }


def validate_quality_profile(
    config: Mapping[str, Any],
    *,
    backend: str,
) -> dict[str, Any]:
    """Validate quality semantics independently of a specific compute adapter."""

    candidate = deepcopy(dict(config))
    issues: list[dict[str, str]] = []
    name = str(candidate.get("name") or "")
    if candidate.get("format") != QUALITY_PROFILE_FORMAT:
        issues.append(
            {"severity": "error", "code": "quality_format", "message": "Unsupported quality-profile format."}
        )
    if name not in QUALITY_PROFILE_NAMES:
        issues.append(
            {"severity": "error", "code": "quality_name", "message": "Unknown quality profile."}
        )
    numerical = dict(candidate.get("numerical_resolution") or {})
    resolution_policy = str(numerical.get("policy") or "")
    if resolution_policy not in NUMERICAL_RESOLUTION_POLICIES:
        issues.append(
            {
                "severity": "error",
                "code": "numerical_resolution_policy",
                "message": "Unknown numerical analysis-resolution policy.",
            }
        )
    scale_factor = numerical.get("scale_factor")
    if resolution_policy == "scale_factor" and (
        not isinstance(scale_factor, (int, float))
        or isinstance(scale_factor, bool)
        or not 0.0 < float(scale_factor) <= 1.0
    ):
        issues.append(
            {
                "severity": "error",
                "code": "analysis_scale_factor",
                "message": "Analysis scale factor must be in (0, 1].",
            }
        )
    maximum_dimension = numerical.get("maximum_dimension_px")
    if resolution_policy == "maximum_dimension" and (
        not isinstance(maximum_dimension, int)
        or isinstance(maximum_dimension, bool)
        or maximum_dimension < 64
    ):
        issues.append(
            {
                "severity": "error",
                "code": "analysis_maximum_dimension",
                "message": "Maximum analysis dimension must be at least 64 px.",
            }
        )
    minimum_dimension = numerical.get("minimum_dimension_px", 64)
    if (
        not isinstance(minimum_dimension, int)
        or isinstance(minimum_dimension, bool)
        or minimum_dimension < 16
    ):
        issues.append(
            {
                "severity": "error",
                "code": "analysis_minimum_dimension",
                "message": "Minimum analysis dimension must be at least 16 px.",
            }
        )
    if numerical.get("allow_upsampling") is not False:
        issues.append(
            {"severity": "error", "code": "quality_upsampling", "message": "Scientific upsampling is not allowed."}
        )
    if numerical.get("preview_resolution_is_scientifically_authoritative") is not False:
        issues.append(
            {
                "severity": "error",
                "code": "preview_authority",
                "message": "Preview resolution cannot define the numerical analysis grid.",
            }
        )
    if resolution_policy in {"scale_factor", "maximum_dimension"}:
        issues.append(
            {
                "severity": "warning",
                "code": "reduced_scientific_resolution",
                "message": "The numerical analysis uses a reduced registered grid; this is a recorded scientific approximation, not a preview setting.",
            }
        )
    precision = dict(candidate.get("precision") or {})
    if precision.get("policy") != "backend_explicit":
        issues.append(
            {"severity": "error", "code": "precision_policy", "message": "Precision must be explicit per backend."}
        )
    try:
        expected_dtype = numerical_dtype_for_backend(backend)
    except ValueError:
        expected_dtype = ""
        issues.append(
            {
                "severity": "error",
                "code": "backend_dtype_unknown",
                "message": "The selected backend does not declare float32 or float64 precision.",
            }
        )
    compute_dtype = str(precision.get("compute_dtype") or "backend_selected")
    if compute_dtype not in {"backend_selected", expected_dtype}:
        issues.append(
            {
                "severity": "error",
                "code": "compute_precision_mismatch",
                "message": "Quality precision must match the dtype encoded by the selected backend.",
            }
        )
    if precision.get("cpu_reference_dtype") != "float64":
        issues.append(
            {
                "severity": "error",
                "code": "precision_contract",
                "message": "The authoritative CPU parity reference must remain float64.",
            }
        )
    fraction = precision.get("cpu_float64_parity_fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not (
        0.0 < float(fraction) <= 1.0
    ):
        issues.append(
            {
                "severity": "error",
                "code": "parity_fraction",
                "message": "CPU-float64 parity fraction must be in (0, 1].",
            }
        )
    for key in ("absolute_parity_tolerance", "relative_parity_tolerance"):
        tolerance = precision.get(key)
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not math.isfinite(float(tolerance))
            or float(tolerance) <= 0.0
        ):
            issues.append(
                {
                    "severity": "error",
                    "code": key,
                    "message": f"{key} must be finite and positive.",
                }
            )
    if (
        backend == "cpu-float64"
        and isinstance(fraction, (int, float))
        and not isinstance(fraction, bool)
        and 0.0 < float(fraction) < 1.0
    ):
        # Every CPU-float64 result is itself a reference result.
        candidate.setdefault("precision", {})[
            "effective_cpu_float64_parity_fraction"
        ] = 1.0
    preview = dict(candidate.get("preview") or {})
    maximum = preview.get("maximum_dimension_px")
    quality = preview.get("jpeg_quality")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 256:
        issues.append(
            {"severity": "error", "code": "preview_dimension", "message": "Preview maximum dimension must be at least 256 px."}
        )
    if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 100:
        issues.append(
            {"severity": "error", "code": "preview_quality", "message": "Preview JPEG quality must be in [1, 100]."}
        )
    return {
        "valid": not any(row["severity"] == "error" for row in issues),
        "configuration": candidate,
        "issues": issues,
    }


def resolve_analysis_shape(
    source_shape: Sequence[int], profile: Mapping[str, Any]
) -> tuple[int, int]:
    """Resolve the authoritative numerical grid without ever upsampling."""

    if len(source_shape) != 2:
        raise ValueError("Analysis source shape must contain rows and columns")
    rows, columns = int(source_shape[0]), int(source_shape[1])
    if rows < 1 or columns < 1:
        raise ValueError("Analysis source dimensions must be positive")
    numerical = dict(profile.get("numerical_resolution") or {})
    policy = str(numerical.get("policy") or "native_registered_grid")
    if policy == "native_registered_grid":
        return rows, columns
    if policy == "scale_factor":
        scale = float(numerical.get("scale_factor") or 0.0)
    elif policy == "maximum_dimension":
        maximum = int(numerical.get("maximum_dimension_px") or 0)
        scale = min(1.0, maximum / float(max(rows, columns)))
    else:
        raise ValueError(f"Unsupported numerical resolution policy: {policy}")
    if not 0.0 < scale <= 1.0:
        raise ValueError("Analysis scale must be in (0, 1]")
    minimum = max(16, int(numerical.get("minimum_dimension_px") or 64))
    if min(rows, columns) > minimum:
        scale = max(scale, minimum / float(min(rows, columns)))
    scale = min(1.0, scale)
    return max(1, round(rows * scale)), max(1, round(columns * scale))


def default_time_selection_budget(*, maximum_time_points: int | None) -> dict[str, Any]:
    """Return an editable budget that preserves endpoints and named landmarks."""

    return {
        "policy": "uniform_with_required_landmarks",
        "start_time_hours": None,
        "end_time_hours": None,
        "required_landmark_times_hours": [],
        "preserve_window_endpoints": True,
        "maximum_time_points": maximum_time_points,
        "maximum_solver_tasks": None,
        "maximum_publication_bytes": None,
    }


def _finite_optional(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_integer(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        return None
    return int(value)


def _nearest_index(
    times: Sequence[float],
    candidates: Sequence[int],
    target: float,
) -> int:
    return min(candidates, key=lambda index: (abs(times[index] - target), index))


def resolve_time_selection(
    times_hours: Sequence[float],
    budget: Mapping[str, Any],
    *,
    tasks_per_time: int = 1,
    fixed_publication_bytes: int = 0,
    publication_bytes_per_time: int = 0,
) -> dict[str, Any]:
    """Resolve a deterministic subset and record every active limiting factor."""

    times = [float(value) for value in times_hours]
    issues: list[dict[str, Any]] = []
    if not times or any(not math.isfinite(value) for value in times):
        issues.append(
            {"severity": "error", "code": "time_grid_invalid", "message": "A finite non-empty time grid is required."}
        )
    if any(next_value <= value for value, next_value in zip(times, times[1:])):
        issues.append(
            {"severity": "error", "code": "time_grid_order", "message": "Time grid must be strictly increasing."}
        )
    if budget.get("policy") != "uniform_with_required_landmarks":
        issues.append(
            {"severity": "error", "code": "time_selection_policy", "message": "Unsupported time-selection policy."}
        )
    start_raw = budget.get("start_time_hours")
    end_raw = budget.get("end_time_hours")
    start = _finite_optional(start_raw)
    end = _finite_optional(end_raw)
    if start_raw not in {None, ""} and start is None:
        issues.append(
            {"severity": "error", "code": "start_time_invalid", "message": "Start time must be finite."}
        )
    if end_raw not in {None, ""} and end is None:
        issues.append(
            {"severity": "error", "code": "end_time_invalid", "message": "End time must be finite."}
        )
    if start is not None and end is not None and end < start:
        issues.append(
            {"severity": "error", "code": "time_window_invalid", "message": "End time cannot precede start time."}
        )
    candidates = [
        index
        for index, value in enumerate(times)
        if (start is None or value >= start) and (end is None or value <= end)
    ]
    if not candidates:
        issues.append(
            {"severity": "error", "code": "time_window_empty", "message": "The selected time window contains no frames."}
        )

    limits: dict[str, int] = {}
    maximum_time_points = budget.get("maximum_time_points")
    if maximum_time_points not in {None, ""}:
        parsed = _positive_integer(maximum_time_points)
        if parsed is None:
            issues.append(
                {"severity": "error", "code": "maximum_time_points_invalid", "message": "Maximum time points must be a positive integer."}
            )
        else:
            limits["maximum_time_points"] = parsed
    maximum_tasks = budget.get("maximum_solver_tasks")
    if maximum_tasks not in {None, ""}:
        parsed = _positive_integer(maximum_tasks)
        if parsed is None or tasks_per_time < 1:
            issues.append(
                {"severity": "error", "code": "maximum_solver_tasks_invalid", "message": "Maximum solver tasks must be a positive integer."}
            )
        else:
            limits["maximum_solver_tasks"] = parsed // tasks_per_time
    maximum_bytes = budget.get("maximum_publication_bytes")
    if maximum_bytes not in {None, ""}:
        parsed = _positive_integer(maximum_bytes)
        if parsed is None:
            issues.append(
                {"severity": "error", "code": "maximum_publication_bytes_invalid", "message": "Maximum publication bytes must be a positive integer."}
            )
        elif publication_bytes_per_time <= 0:
            issues.append(
                {"severity": "error", "code": "publication_estimate_missing", "message": "A publication-size budget requires a per-time size estimate."}
            )
        else:
            available = max(0, parsed - max(0, int(fixed_publication_bytes)))
            limits["maximum_publication_bytes"] = available // int(
                publication_bytes_per_time
            )

    capacity = len(candidates)
    if limits:
        capacity = min(capacity, *limits.values())
    landmarks: list[float] = []
    for raw in budget.get("required_landmark_times_hours") or []:
        value = _finite_optional(raw)
        if value is None:
            issues.append(
                {"severity": "error", "code": "landmark_invalid", "message": "Every required landmark time must be finite."}
            )
        else:
            landmarks.append(value)

    required: set[int] = set()
    if candidates and budget.get("preserve_window_endpoints") is True:
        required.update((candidates[0], candidates[-1]))
    for value in landmarks:
        if candidates:
            required.add(_nearest_index(times, candidates, value))
    if capacity < len(required):
        issues.append(
            {
                "severity": "error",
                "code": "budget_smaller_than_required_landmarks",
                "message": "The budget is too small to retain all endpoints and required landmarks.",
                "required_time_points": len(required),
                "available_time_points": capacity,
            }
        )

    selected = set(required)
    if capacity >= len(required):
        while len(selected) < capacity:
            remaining = [index for index in candidates if index not in selected]
            if not remaining:
                break
            if not selected:
                selected.add(remaining[0])
                continue
            next_index = max(
                remaining,
                key=lambda index: (
                    min(abs(times[index] - times[current]) for current in selected),
                    -index,
                ),
            )
            selected.add(next_index)
    selected_indices = sorted(selected)
    estimated_bytes = max(0, int(fixed_publication_bytes)) + len(
        selected_indices
    ) * max(0, int(publication_bytes_per_time))
    active_limits = [
        key for key, value in limits.items() if value == capacity and capacity < len(candidates)
    ]
    return {
        "schema": 1,
        "format": TIME_SELECTION_FORMAT,
        "valid": not any(row["severity"] == "error" for row in issues),
        "source_time_count": len(times),
        "window_candidate_count": len(candidates),
        "selected_time_count": len(selected_indices),
        "selected_time_indices": selected_indices,
        "selected_times_hours": [times[index] for index in selected_indices],
        "tasks_per_time": int(tasks_per_time),
        "selected_solver_task_count": len(selected_indices) * int(tasks_per_time),
        "fixed_publication_bytes": max(0, int(fixed_publication_bytes)),
        "publication_bytes_per_time": max(0, int(publication_bytes_per_time)),
        "estimated_publication_bytes": estimated_bytes,
        "active_limits": active_limits,
        "issues": issues,
    }


def estimate_mass_publication_bytes(
    manifest: Mapping[str, Any],
    *,
    products: Sequence[str],
    numerical_dtype: str,
    preview_maximum_dimension_px: int,
    run_shapes: Mapping[str, Sequence[int]] | None = None,
    analysis_profile: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Return a conservative uncompressed output estimate for a shared time grid."""

    dynamic_fields = {
        "mass_total",
        "mass_aqueous",
        "mass_free_gas",
        "concentration_aqueous",
        "saturation_free_gas",
    }
    selected = set(str(value) for value in products)
    bytes_per_value = 8 if numerical_dtype == "float64" else 4
    fixed_bytes = 1_048_576
    per_time_bytes = 0
    known_shapes = dict(run_shapes or {})
    for raw in manifest.get("run_inputs") or []:
        if not isinstance(raw, Mapping):
            continue
        grid = dict(raw.get("grid") or raw.get("geometry") or {})
        run_id = str(raw.get("run_id") or "")
        shape = known_shapes.get(run_id) or grid.get("shape_rc") or []
        if len(shape) != 2:
            products_map = raw.get("products") or {}
            # Index inspection already validates shape; missing inline shape only
            # makes the estimate intentionally conservative rather than scientific.
            pixel_count = 4_000_000 if products_map else 0
            source_pixel_count = pixel_count
            rows = columns = 0
        else:
            rows, columns = int(shape[0]), int(shape[1])
            source_pixel_count = max(0, rows * columns)
            analysis_rows, analysis_columns = resolve_analysis_shape(
                (rows, columns), analysis_profile or default_quality_profile()
            )
            pixel_count = max(0, analysis_rows * analysis_columns)
        per_time_bytes += pixel_count * bytes_per_value * len(
            dynamic_fields & selected
        )
        if "preview_images" in selected:
            if rows > 0 and columns > 0:
                scale = min(
                    1.0,
                    float(preview_maximum_dimension_px) / max(rows, columns),
                )
                preview_pixels = max(1, round(source_pixel_count * scale * scale))
            else:
                preview_pixels = preview_maximum_dimension_px**2
            per_time_bytes += preview_pixels * 3
        # Static transport field plus indexes and metadata.
        if "transport_weight" in selected:
            fixed_bytes += pixel_count * bytes_per_value
        fixed_bytes += 131_072
    return {
        "fixed_bytes": int(fixed_bytes),
        "bytes_per_time": int(per_time_bytes),
    }


def estimate_wasserstein_publication_bytes(
    *,
    tasks_per_time: int,
    store_transport_diagnostics: bool,
) -> dict[str, int]:
    """Estimate table/checkpoint publication size with deliberate headroom."""

    bytes_per_task = 24_576 if store_transport_diagnostics else 12_288
    return {
        "fixed_bytes": 2_097_152,
        "bytes_per_time": max(0, int(tasks_per_time)) * bytes_per_task,
    }


def format_bytes(value: int | float) -> str:
    """Return a short IEC size label for GUI summaries."""

    size = float(max(0.0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


__all__ = [
    "QUALITY_PROFILE_FORMAT",
    "QUALITY_PROFILE_NAMES",
    "NUMERICAL_RESOLUTION_POLICIES",
    "TIME_SELECTION_FORMAT",
    "default_quality_profile",
    "default_time_selection_budget",
    "estimate_mass_publication_bytes",
    "estimate_wasserstein_publication_bytes",
    "format_bytes",
    "numerical_dtype_for_backend",
    "resolve_analysis_shape",
    "resolve_time_selection",
    "summarize_quality_contract",
    "validate_quality_profile",
]
