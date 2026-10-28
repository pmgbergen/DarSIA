"""Colour-independent source support and local scientific invalid-pixel repair."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from pathlib import Path
from hashlib import sha256

import cv2
import numpy as np

from .io import sha256_file


def photo_safe_core_policy_metadata(
    enclosed_hole_policy: str, *, parameters: Mapping
) -> dict[str, Any]:
    """Describe the configured erosion and complement-filling policy without using image colour."""
    if not isinstance(enclosed_hole_policy, str) or enclosed_hole_policy not in (
        parameters["keep_policy"],
        parameters["fill_policy"],
    ):
        raise ValueError("Unknown Native source-photo enclosed-hole policy")
    metadata = dict(
        photo_sand_support_contract=(
            parameters["keep_contract"]
            if enclosed_hole_policy == parameters["keep_policy"]
            else parameters["fill_contract"]
        ),
        enclosed_hole_policy=enclosed_hole_policy,
        enclosed_hole_max_area_px=parameters["hole_max_area_px"],
        enclosed_hole_connectivity=parameters["connectivity"],
        fixed_native_mask_erosion_px=parameters["guard_px"],
    )
    if enclosed_hole_policy == parameters["fill_policy"]:
        metadata.update(
            vertical_gap_max_length_px=parameters["vertical_gap_max_px"],
            vertical_gap_rule=parameters["vertical_gap_rule"],
        )
    return metadata


def apply_photo_safe_core_enclosed_holes(
    safe: np.ndarray,
    *,
    enclosed_hole_policy: str,
    return_rule_masks: bool = False,
    parameters: Mapping,
) -> (
    tuple[np.ndarray, dict[str, Any]]
    | tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]
):
    """Fill only enclosed complement components and then vertically bounded gaps. Existing
    support is retained; parameters supply the area cap, connectivity and gap length.
    """
    core = np.asarray(safe)
    if core.ndim != 2 or core.dtype != np.bool_ or min(core.shape) <= 0:
        raise ValueError("Native source-photo safe core must be nonempty HW bool")
    metadata = photo_safe_core_policy_metadata(
        enclosed_hole_policy, parameters=parameters
    )
    added_pixels = added_components = 0
    enclosed = np.zeros(core.shape, dtype=bool) if return_rule_masks else None
    vertical = np.zeros(core.shape, dtype=bool) if return_rule_masks else None
    vertical_pixels = vertical_runs = 0
    if enclosed_hole_policy == parameters["fill_policy"]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (~core).astype(np.uint8),
            connectivity=parameters["connectivity"],
            ltype=cv2.CV_32S,
        )
        selected = np.zeros(count, dtype=bool)
        component = stats[1:]
        if len(component):
            left = component[:, cv2.CC_STAT_LEFT]
            top = component[:, cv2.CC_STAT_TOP]
            right = left + component[:, cv2.CC_STAT_WIDTH]
            bottom = top + component[:, cv2.CC_STAT_HEIGHT]
            area = component[:, cv2.CC_STAT_AREA]
            selected[1:] = (
                (left > 0)
                & (top > 0)
                & (right < core.shape[1])
                & (bottom < core.shape[0])
                & (area <= parameters["hole_max_area_px"])
            )
            added_components = int(np.count_nonzero(selected))
            added_pixels = int(
                np.sum(stats[selected, cv2.CC_STAT_AREA], dtype=np.int64)
            )
        rule1 = selected[labels]
        if return_rule_masks:
            enclosed = rule1
        core = core | rule1
        pad = parameters["vertical_gap_max_px"]
        padded = np.pad(core.astype(np.uint8), ((pad, pad), (0, 0)))
        closed = cv2.morphologyEx(
            padded,
            cv2.MORPH_CLOSE,
            np.ones((pad + 1, 1), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )[pad:-pad].astype(bool)
        rule2 = closed & ~core
        vertical_pixels = int(np.count_nonzero(rule2))
        vertical_runs = int(np.count_nonzero(rule2[0])) + int(
            np.count_nonzero(rule2[1:] & ~rule2[:-1])
        )
        if return_rule_masks:
            vertical = rule2
        core = core | rule2
    metadata.update(
        enclosed_hole_added_pixel_count=added_pixels,
        enclosed_hole_added_component_count=added_components,
    )
    if enclosed_hole_policy == parameters["fill_policy"]:
        metadata.update(
            vertical_gap_added_pixel_count=vertical_pixels,
            vertical_gap_added_run_count=vertical_runs,
            photo_sand_support_added_pixel_count=added_pixels + vertical_pixels,
        )
    if return_rule_masks:
        return (
            core,
            metadata,
            {"enclosed_rule1": enclosed, "vertical_gap_rule2": vertical},
        )
    return (core, metadata)


def project_source_photo_safe_core(
    sand_grid: np.ndarray,
    h_reference_to_frame_native: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    guard_px: int,
    enclosed_hole_policy: str,
    return_metadata: bool = False,
    parameters: Mapping,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Project a reference-photo support grid to a native frame, erode its boundary and
    optionally fill configured enclosed holes. No colour or phase labels determine
    support.
    """
    grid = np.asarray(sand_grid)
    if grid.shape != tuple(parameters["grid_shape_hw"]) or grid.dtype != np.bool_:
        raise ValueError("Source-photo support has another grid or dtype")
    if not isinstance(guard_px, int) or guard_px != parameters["guard_px"]:
        raise ValueError("Native source-photo boundary guard changed")
    if len(shape_hw) != 2 or min(shape_hw) <= 0:
        raise ValueError("Native source-photo destination shape is invalid")
    photo_safe_core_policy_metadata(enclosed_hole_policy, parameters=parameters)
    h = _normal_h(h_reference_to_frame_native, parameters=parameters)
    cell_to_reference = np.array(
        parameters["cell_to_reference_native"], dtype=np.float64
    )
    projected = cv2.warpPerspective(
        grid.astype(np.uint8),
        h @ cell_to_reference,
        (int(shape_hw[1]), int(shape_hw[0])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    kernel = np.ones((2 * guard_px + 1, 2 * guard_px + 1), np.uint8)
    safe = cv2.erode(
        projected, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    safe, metadata = apply_photo_safe_core_enclosed_holes(
        safe, enclosed_hole_policy=enclosed_hole_policy, parameters=parameters
    )
    return (safe, metadata) if return_metadata else safe


def _normal_h(value: Any, *, parameters: Mapping) -> np.ndarray:
    """Normalize a finite nonsingular projective transform with configured singularity
    thresholds.
    """
    h = np.asarray(value, np.float64)
    if (
        h.shape != (3, 3)
        or not np.isfinite(h).all()
        or abs(float(np.linalg.det(h))) < parameters["homography_epsilon"]
        or (abs(float(h[2, 2])) < parameters["homography_epsilon"])
    ):
        raise ValueError("Native per-frame homography is invalid")
    return h / h[2, 2]


def compute_feature_support(
    *,
    current_valid: np.ndarray,
    baseline_valid: np.ndarray,
    h_baseline_to_current: np.ndarray,
    previous_valid: np.ndarray | None = None,
    h_baseline_to_previous: np.ndarray | None = None,
    baseline_origin_xy: tuple[int, int],
    previous_origin_xy: tuple[int, int],
    parameters: Mapping,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect current support with bilinearly transported reference and previous
    support, then erode for texture footprints. Origins are explicit native coordinates;
    tolerance and footprint are parameters.
    """
    current = np.asarray(current_valid)
    baseline = np.asarray(baseline_valid)
    if (
        current.ndim != 2
        or current.dtype != np.bool_
        or baseline.ndim != 2
        or (baseline.dtype != np.bool_)
    ):
        raise ValueError("Native feature source masks must be HW bool")
    if previous_valid is not None:
        previous = np.asarray(previous_valid)
        if (
            previous.ndim != 2
            or previous.dtype != np.bool_
            or h_baseline_to_previous is None
        ):
            raise ValueError("Native previous source needs its exact bool mask and H")
    else:
        previous = None
    if len(baseline_origin_xy) != 2 or len(previous_origin_xy) != 2:
        raise ValueError("Native feature source origin is not native XY")

    def project(
        valid: np.ndarray, origin: tuple[int, int], h_native_to_current: np.ndarray
    ) -> np.ndarray:
        source_to_native = np.array(
            [[1.0, 0.0, origin[0]], [0.0, 1.0, origin[1]], [0.0, 0.0, 1.0]], np.float64
        )
        h = _normal_h(
            _normal_h(h_native_to_current, parameters=parameters) @ source_to_native,
            parameters=parameters,
        )
        coverage = cv2.warpPerspective(
            valid.astype(np.float32),
            h,
            (current.shape[1], current.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return coverage >= np.float32(1 - parameters["coverage_tolerance"])

    b_to_current = _normal_h(h_baseline_to_current, parameters=parameters)
    raw = current & project(baseline, baseline_origin_xy, b_to_current)
    if previous is not None:
        previous_to_current = _normal_h(
            b_to_current
            @ np.linalg.inv(_normal_h(h_baseline_to_previous, parameters=parameters)),
            parameters=parameters,
        )
        raw &= project(previous, previous_origin_xy, previous_to_current)
    safe = cv2.erode(
        raw.astype(np.uint8),
        np.ones(tuple(parameters["texture_footprint_hw"]), np.uint8),
        iterations=1,
    ).astype(bool)
    return (safe, raw)


def validate_invalid_policy(policy: str, *, parameters: Mapping) -> str:
    """Reject unknown scientific invalid-pixel policies."""
    if policy not in (parameters["keep_policy"], parameters["policy"]):
        raise ValueError(f"Unknown isolated-invalid policy: {policy!r}")
    return policy


def invalid_policy_contract(*, parameters: Mapping) -> dict:
    """Return the configured scientific invalid-pixel repair contract."""
    return dict(parameters["contract_metadata"])


def fill_isolated_invalid(
    log_rgb: np.ndarray, safe_filled: np.ndarray, *, parameters: Mapping
):
    """Repair only finite-surrounded small invalid components inside photo support. Sweeps
    are simultaneous log-space medians; original finite channels retain their bits.
    """
    signal = np.asarray(log_rgb)
    safe = np.asarray(safe_filled)
    if signal.dtype != np.float32 or signal.ndim != 3 or signal.shape[0] != 3:
        raise TypeError("Isolated-invalid repair needs CHW float32 log RGB")
    if safe.dtype != np.dtype(bool) or safe.shape != signal.shape[1:]:
        raise TypeError(
            "Isolated-invalid repair needs a native boolean photo-sand mask"
        )
    if np.isinf(signal).any():
        raise ValueError("Infinite scientific values must not be repaired")
    result = signal.copy()
    valid = np.isfinite(signal).all(axis=0) & safe
    invalid = ~np.isfinite(signal).all(axis=0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        invalid.astype(np.uint8), connectivity=parameters["connectivity"]
    )
    filled = np.zeros(safe.shape, bool)
    filled_components = largest = maximum_sweeps = 0
    height, width = safe.shape
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if (
            area > parameters["max_area_px"]
            or x == 0
            or y == 0
            or (x + w == width)
            or (y + h == height)
        ):
            continue
        component = labels[y : y + h, x : x + w] == label
        if not safe[y : y + h, x : x + w][component].all():
            continue
        patch = labels[y - 1 : y + h + 1, x - 1 : x + w + 1] == label
        ring = (
            cv2.dilate(
                patch.astype(np.uint8),
                np.ones(tuple(parameters["enclosure_footprint_hw"]), np.uint8),
            ).astype(bool)
            & ~patch
        )
        if not valid[y - 1 : y + h + 1, x - 1 : x + w + 1][ring].all():
            continue
        rows, cols = np.nonzero(component)
        pending = list(zip(rows + y, cols + x))
        sweeps = 0
        while pending:
            updates = []
            for row, col in pending:
                y0, y1, x0, x1 = (
                    max(0, row - parameters["neighbour_radius_px"]),
                    min(height, row + parameters["neighbour_radius_px"] + 1),
                    max(0, col - parameters["neighbour_radius_px"]),
                    min(width, col + parameters["neighbour_radius_px"] + 1),
                )
                donors = valid[y0:y1, x0:x1]
                if donors.any():
                    median = np.median(result[:, y0:y1, x0:x1][:, donors], axis=1)
                    updates.append((row, col, median))
            if not updates:
                raise ValueError(
                    "Enclosed isolated-invalid component has no finite local donor"
                )
            updated = set()
            for row, col, median in updates:
                absent = ~np.isfinite(result[:, row, col])
                result[absent, row, col] = median[absent]
                valid[row, col] = True
                filled[row, col] = True
                updated.add((row, col))
            pending = [pixel for pixel in pending if pixel not in updated]
            sweeps += 1
        filled_components += 1
        largest = max(largest, area)
        maximum_sweeps = max(maximum_sweeps, sweeps)
    metadata = dict(
        invalid_policy_contract(parameters=parameters),
        filled_pixel_count=int(filled.sum()),
        filled_component_count=filled_components,
        largest_filled_component_px=largest,
        maximum_fill_sweeps=maximum_sweeps,
        filled_mask_packbits_sha256=sha256(
            np.packbits(filled, bitorder="little").tobytes()
        ).hexdigest(),
    )
    return (result, filled, metadata)


def source_photo_grid(
    path: str | Path, digest: str, *, parameters: Mapping
) -> np.ndarray:
    """Load an immutable, content-bound reference support grid without changing its dtype."""
    source = Path(path)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or sha256_file(source) != digest
    ):
        raise ValueError("Source-photo support grid changed or lacks a digest")
    with np.load(source, allow_pickle=False) as archive:
        key = parameters["grid_archive_key"]
        if key not in archive.files:
            raise ValueError("Source-photo support archive lacks its configured grid")
        sand = np.asarray(archive[key])
    if sand.shape != tuple(parameters["grid_shape_hw"]) or sand.dtype != np.bool_:
        raise ValueError("Source-photo support has another grid or dtype")
    sand.setflags(write=False)
    return sand
