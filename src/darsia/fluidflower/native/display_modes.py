"""Configured display-only colour transfers, highlight treatment and hole cleanup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256

import cv2
import numpy as np

import warnings

from .io import sha256_file


def validate_display_mode(mode: str, *, parameters: Mapping) -> str:
    """Reject a photographic presentation mode absent from the supplied contract."""
    if mode not in parameters["modes"]:
        raise ValueError(f"Unknown photographic display mode: {mode!r}")
    return mode


def photographic_contract(*, parameters: Mapping) -> dict:
    """Return the supplied photographic transfer metadata without deriving calibration."""
    return dict(parameters["contract_metadata"])


def display_contract(*, parameters: Mapping) -> dict:
    """Return the supplied mode contract with the actual renderer code identity."""
    result = dict(parameters["contract_metadata"])
    result["renderer_code_sha256"] = sha256_file(Path(__file__))
    return result


def photographic_display(
    log_camera_rgb: np.ndarray, *, tile_rows: int, parameters: Mapping
) -> np.ndarray:
    """Render float32 log-camera channels as a quantized photograph using the explicit
    float64 colour matrix and transfer. Invalid pixels are presentation-only grey; the
    scientific input is untouched.
    """
    signal = np.asarray(log_camera_rgb)
    if signal.dtype != np.float32 or signal.ndim != 3 or signal.shape[0] != 3:
        raise TypeError("Photographic display requires CHW float32 log-camera RGB")
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    _, height, width = signal.shape
    output = np.empty((height, width, 3), dtype=np.float32)
    for start in range(0, height, tile_rows):
        end = min(start + tile_rows, height)
        tile = np.moveaxis(signal[:, start:end], 0, -1)
        valid = np.isfinite(tile).all(axis=-1)
        linear = np.exp(np.where(valid[..., None], tile, 0).astype(np.float64))
        srgb_linear = np.clip(
            linear @ np.array(parameters["matrix"], dtype=np.float64), 0, 1
        )
        encoded = np.where(
            srgb_linear <= parameters["srgb"]["threshold"],
            parameters["srgb"]["linear_gain"] * srgb_linear,
            parameters["srgb"]["power_gain"]
            * np.power(srgb_linear, 1 / parameters["srgb"]["power_denominator"])
            - parameters["srgb"]["power_offset"],
        )
        encoded = np.rint(255 * encoded) / 255
        encoded[~valid] = parameters["invalid_rgb8"] / 255
        output[start:end] = encoded.astype(np.float32)
    return output


def _fill_small_invalid(
    linear: np.ndarray, valid: np.ndarray, *, parameters: Mapping
) -> tuple[np.ndarray, dict]:
    """Fill only small display holes; retain all valid pixels byte-for-byte.

    Sparse 5x5 neighbourhoods avoid three whole-frame inpainting passes. Each
    iteration uses the previous iteration's valid pixels, so filling is not
    dependent on pixel traversal order. Larger invalid regions are never used
    as interpolation neighbours.
    """
    invalid = ~valid
    invalid_count = int(np.count_nonzero(invalid))
    if not invalid_count:
        return (
            invalid,
            {
                "invalid_input_pixels": 0,
                "filled_invalid_pixels": 0,
                "largest_filled_component_px": 0,
                "unfilled_invalid_pixels": 0,
                "fill_iterations": 0,
            },
        )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(invalid, dtype=np.uint8),
        connectivity=parameters["fill_connectivity"],
    )
    small_ids = np.flatnonzero(
        (stats[:, cv2.CC_STAT_AREA] <= parameters["fill_max_area_px"])
        & (np.arange(count) != 0)
    )
    lookup = np.zeros(count, dtype=bool)
    lookup[small_ids] = True
    rows, cols = np.nonzero(lookup[labels])
    del lookup
    filled_mask = np.zeros(valid.shape, dtype=bool)
    height, width = valid.shape
    iterations = 0
    offsets = np.array(
        [
            (dy, dx)
            for dy in range(
                -parameters["fill_radius_px"], parameters["fill_radius_px"] + 1
            )
            for dx in range(
                -parameters["fill_radius_px"], parameters["fill_radius_px"] + 1
            )
        ],
        dtype=np.int64,
    )
    while rows.size and iterations < parameters["fill_max_iterations"]:
        nr = rows[:, None] + offsets[None, :, 0]
        nc = cols[:, None] + offsets[None, :, 1]
        inside = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
        nr = np.clip(nr, 0, height - 1)
        nc = np.clip(nc, 0, width - 1)
        neighbours = linear[nr, nc].copy()
        neighbours[~(inside & valid[nr, nc])] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(neighbours, axis=1)
        ready = np.isfinite(medians).all(axis=-1)
        if not np.any(ready):
            break
        linear[rows[ready], cols[ready]] = medians[ready]
        valid[rows[ready], cols[ready]] = True
        filled_mask[rows[ready], cols[ready]] = True
        rows, cols = (rows[~ready], cols[~ready])
        iterations += 1
    filled_ids = np.unique(labels[filled_mask])
    largest = int(stats[filled_ids, cv2.CC_STAT_AREA].max()) if filled_ids.size else 0
    return (
        ~valid,
        {
            "invalid_input_pixels": invalid_count,
            "filled_invalid_pixels": int(np.count_nonzero(filled_mask)),
            "largest_filled_component_px": largest,
            "unfilled_invalid_pixels": int(np.count_nonzero(~valid)),
            "fill_iterations": iterations,
        },
    )


def _neutralise_highlights(
    linear: np.ndarray,
    photo_sand: np.ndarray,
    valid: np.ndarray,
    raw_saturated_mask: np.ndarray | None,
    *,
    parameters: Mapping,
) -> dict:
    """Neutralise white clipping outside sand without whitening yellow/orange."""
    outside = ~photo_sand & valid
    if raw_saturated_mask is not None:
        selected = outside & raw_saturated_mask
        linear[selected] = np.max(linear[selected], axis=-1, keepdims=True)
        return {
            "policy": "native_raw_any_channel_saturation_mask",
            "pixels_affected": int(np.count_nonzero(selected)),
            "raw_saturation_mask_available": True,
        }
    weight = np.clip(
        (np.minimum(linear[..., 0], linear[..., 2]) - parameters["highlight_threshold"])
        / parameters["highlight_width"],
        0,
        1,
    )
    selected = outside & (weight > 0)
    if np.any(selected):
        rgb = linear[selected]
        w = weight[selected, None]
        linear[selected] = rgb * (1 - w) + np.max(rgb, axis=-1, keepdims=True) * w
    return {
        "policy": "allowed_min_linear_RB_fallback",
        "pixels_affected": int(np.count_nonzero(selected)),
        "raw_saturation_mask_available": False,
        "limitation": "Frozen RAW renderer exposes saturation counts, not a registered per-pixel saturation mask",
    }


def _srgb_encode(linear: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    """Apply the configured linear-to-display transfer without changing the scientific signal."""
    linear = np.clip(linear, 0, 1)
    return np.where(
        linear <= parameters["srgb"]["threshold"],
        parameters["srgb"]["linear_gain"] * linear,
        parameters["srgb"]["power_gain"]
        * np.power(linear, 1 / parameters["srgb"]["power_denominator"])
        - parameters["srgb"]["power_offset"],
    )


def _median_chroma_rgb8(rgb8: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    """Median only the two 8-bit chroma planes; keep Y completely unfiltered."""
    ycrcb = cv2.cvtColor(np.ascontiguousarray(rgb8), cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 1] = cv2.medianBlur(ycrcb[..., 1], parameters["chroma_median_size_px"])
    ycrcb[..., 2] = cv2.medianBlur(ycrcb[..., 2], parameters["chroma_median_size_px"])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def render_display_modes(
    log_camera_rgb: np.ndarray,
    photo_sand: np.ndarray,
    *,
    raw_saturated_mask: np.ndarray | None = None,
    tile_rows: int,
    parameters: Mapping,
    selected_modes: tuple[str, ...] | None = None,
    chroma_filter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Render configured photographic modes from full float32 log-camera channels. Protect
    supplied photo support during highlight treatment; small-hole and chroma cleanup
    affect display only. Literal matrices and gain ordering come from parameters.
    """
    signal = np.asarray(log_camera_rgb)
    sand = np.asarray(photo_sand)
    if signal.dtype != np.float32 or signal.ndim != 3 or signal.shape[0] != 3:
        raise TypeError("Display modes require CHW float32 log-camera RGB, never uint8")
    height, width = signal.shape[1:]
    if height < 1 or width < 1:
        raise ValueError("Display modes require a non-empty image")
    if sand.dtype != np.bool_ or sand.shape != (height, width):
        raise TypeError("Photo-sand mask must be native-shape boolean")
    if not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")
    if raw_saturated_mask is not None:
        raw_saturated_mask = np.asarray(raw_saturated_mask)
        if (
            raw_saturated_mask.dtype != np.bool_
            or raw_saturated_mask.shape != sand.shape
        ):
            raise TypeError("RAW saturation mask must be native-shape boolean")
    linear = np.empty((height, width, 3), dtype=np.float32)
    valid = np.empty((height, width), dtype=bool)
    input_nonfinite = 0
    transform_nonfinite = 0
    for start in range(0, height, tile_rows):
        end = min(start + tile_rows, height)
        tile = np.moveaxis(signal[:, start:end], 0, -1)
        tile_valid = np.isfinite(tile).all(axis=-1)
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = np.exp(
                np.where(tile_valid[..., None], tile, 0).astype(np.float64)
            ) @ np.array(parameters["matrix"], dtype=np.float64)
            transformed = transformed.astype(np.float32)
        transform_valid = np.isfinite(transformed).all(axis=-1)
        input_nonfinite += int(np.count_nonzero(~tile_valid))
        transform_nonfinite += int(np.count_nonzero(tile_valid & ~transform_valid))
        tile_valid &= transform_valid
        transformed[~tile_valid] = np.nan
        linear[start:end] = np.maximum(transformed, 0)
        valid[start:end] = tile_valid
    unfilled, fill_metadata = _fill_small_invalid(linear, valid, parameters=parameters)
    highlight_metadata = _neutralise_highlights(
        linear, sand, valid, raw_saturated_mask, parameters=parameters
    )
    if selected_modes is not None:
        for name in selected_modes:
            validate_display_mode(name, parameters=parameters)
    modes = {}
    for name, gain in parameters["gains"].items():
        if selected_modes is not None and name not in selected_modes:
            continue
        rgb8 = np.empty((height, width, 3), dtype=np.uint8)
        for start in range(0, height, tile_rows):
            end = min(start + tile_rows, height)
            rgb = np.where(valid[start:end, :, None], linear[start:end], 0).astype(
                np.float64
            )
            if name != parameters["base_mode"]:
                luminance = rgb @ np.array(
                    parameters["luminance_weights"], dtype=np.float64
                )
                gained = gain * luminance
                toned = (
                    gained
                    * (1 + gained / (parameters["white_multiplier"] * gain) ** 2)
                    / (1 + gained)
                )
                ratio = np.divide(
                    toned, luminance, out=np.zeros_like(toned), where=luminance > 0
                )
                rgb *= ratio[..., None]
                maximum = np.max(rgb, axis=-1, keepdims=True)
                rgb /= np.maximum(maximum, 1)
            rgb8[start:end] = np.rint(
                255 * _srgb_encode(rgb, parameters=parameters)
            ).astype(np.uint8)
        rgb8[unfilled] = parameters["unfilled_rgb8"]
        rgb8 = (
            _median_chroma_rgb8(rgb8, parameters=parameters)
            if chroma_filter is None else chroma_filter(rgb8)
        )
        rgb8[unfilled] = parameters["unfilled_rgb8"]
        modes[name] = rgb8.astype(np.float32) / np.float32(255)
    metadata = {
        "contract": display_contract(parameters=parameters),
        "display_only": True,
        "input_shape_chw": list(signal.shape),
        "input_dtype": str(signal.dtype),
        "input_nonfinite_pixels": input_nonfinite,
        "transform_nonfinite_pixels": transform_nonfinite,
        "invalid_cleanup": fill_metadata,
        "highlight_cleanup": highlight_metadata,
        "mode_keys": list(parameters["modes"]) if selected_modes is None else list(modes),
        "output_dtype": "float32",
        "output_representation": "HWC_sRGB8_quantized_div255",
    }
    return (modes, metadata)


@dataclass(frozen=True)
class DisplayFlatfieldReference:
    """Unsmoothed calibrated luminance of an exact pre-injection reference."""

    luminance: np.ndarray
    valid: np.ndarray
    invalid_cleanup: dict


@dataclass(frozen=True)
class RegisteredDisplayFlatfieldReference:
    """Reference luminance transported to one frame, then box-smoothed."""

    luminance_box5: np.ndarray
    valid: np.ndarray
    invalid_cleanup: dict
    alignment: dict


def _require_display_log_signal(value: np.ndarray, *, name: str) -> np.ndarray:
    signal = np.asarray(value)
    if signal.dtype != np.float32 or signal.ndim != 3 or signal.shape[0] != 3:
        raise TypeError(f"{name} requires CHW float32 log-camera RGB, never uint8")
    if min(signal.shape[1:]) < 1:
        raise ValueError(f"{name} requires a non-empty image")
    return signal


def _display_calibrated_linear(signal: np.ndarray, *, tile_rows: int, parameters: Mapping) -> tuple[np.ndarray, np.ndarray, dict]:
    """Match the configured display modes through its isolated-invalid cleanup stage.

    Highlight neutralisation is deliberately omitted here: it acts only
    outside photo sand, where the existing lift1 image is shown unchanged.
    """
    height, width = signal.shape[1:]
    linear = np.empty((height, width, 3), np.float32)
    valid = np.empty((height, width), bool)
    nonfinite_input = 0
    nonfinite_transform = 0
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        tile = np.moveaxis(signal[:, start:stop], 0, -1)
        tile_valid = np.isfinite(tile).all(axis=-1)
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.exp(np.where(tile_valid[..., None], tile, 0).astype(np.float64)) @ np.array(parameters["display_modes"]["matrix"], dtype=np.float64)
            converted = converted.astype(np.float32)
        converted_valid = np.isfinite(converted).all(axis=-1)
        nonfinite_input += int(np.count_nonzero(~tile_valid))
        nonfinite_transform += int(np.count_nonzero(tile_valid & ~converted_valid))
        tile_valid &= converted_valid
        converted[~tile_valid] = np.nan
        linear[start:stop] = np.maximum(converted, 0)
        valid[start:stop] = tile_valid
    unfilled, fill = _fill_small_invalid(linear, valid, parameters=parameters["display_modes"])
    # A large invalid region remains unavailable.  Keep its NaNs out of the
    # 5x5 box filter so they cannot contaminate otherwise valid neighbours.
    linear[unfilled] = 0
    return linear, ~unfilled, {
        "input_nonfinite_pixels": nonfinite_input,
        "transform_nonfinite_pixels": nonfinite_transform,
        "invalid_cleanup": fill,
    }


def display_box_fully_valid(valid: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    """Require all pixels touched by the configured box box to have source support."""
    return cv2.erode(
        valid.astype(np.uint8), np.ones((parameters["change_display"]["box_size"], parameters["change_display"]["box_size"]), np.uint8),
        borderType=cv2.BORDER_REFLECT_101).astype(bool)


def prepare_display_flatfield_reference(reference_log_camera_rgb: np.ndarray, *,
                                source_valid_mask: np.ndarray | None = None,
                                tile_rows: int, parameters: Mapping) -> DisplayFlatfieldReference:
    """Prepare the run's pre-injection reference before per-frame registration.

    The caller must supply the same source-bound pre-injection reference used
    by feature support.  It remains on its source grid here; never blur before
    transporting it with each frame's source-bound homography.
    """
    reference = _require_display_log_signal(reference_log_camera_rgb, name="Flatfield reference")
    if not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")
    linear, valid, metadata = _display_calibrated_linear(reference, tile_rows=tile_rows, parameters=parameters)
    if source_valid_mask is not None:
        source_valid = np.asarray(source_valid_mask)
        if source_valid.dtype != np.bool_ or source_valid.shape != valid.shape:
            raise TypeError("Flatfield source valid mask must be source HW bool")
        # Preserve the deliberate display repair of tiny, fully enclosed
        # nonfinite pixels even when the original scientific mask marks them
        # invalid.  Do not extend into any other source-invalid pixel.
        original_finite = np.isfinite(reference).all(axis=0)
        repaired_isolated = valid & ~original_finite
        valid &= source_valid | repaired_isolated
    luminance = linear @ np.array(parameters["display_modes"]["luminance_weights"], dtype=np.float64).astype(np.float32)
    luminance[~valid] = 0
    luminance.setflags(write=False)
    valid.setflags(write=False)
    return DisplayFlatfieldReference(luminance, valid, metadata)


def register_display_flatfield_reference(
    prepared: DisplayFlatfieldReference, h_baseline_to_current: np.ndarray, *,
    target_shape_hw: tuple[int, int], source_origin_xy: tuple[int, int],
    target_origin_xy: tuple[int, int], parameters: Mapping,
) -> RegisteredDisplayFlatfieldReference:
    """Warp baseline luminance to the exact current frame, then smooth it.

    The homography and all-four-neighbour validity policy match
    ``the configured temporal homography policy``.  Warping the
    linear luminance is equivalent to taking luminance after bilinear RGB
    warping, without the extra memory needed for a 3-channel baseline warp.
    """
    if not isinstance(prepared, DisplayFlatfieldReference):
        raise TypeError("Expected an unregistered FlatfieldReference")
    source = np.asarray(prepared.luminance)
    source_valid = np.asarray(prepared.valid)
    if (source.dtype != np.float32 or source.ndim != 2 or
            source_valid.dtype != np.bool_ or source_valid.shape != source.shape or
            not np.isfinite(source).all()):
        raise TypeError("Unregistered flatfield reference is malformed")
    if len(target_shape_hw) != 2 or min(target_shape_hw) < 1:
        raise ValueError("Flatfield reference needs a positive target shape")
    height, width = map(int, target_shape_hw)
    if tuple(target_shape_hw) != (height, width):
        raise ValueError("Flatfield target shape must be integer")
    if len(source_origin_xy) != 2 or len(target_origin_xy) != 2:
        raise ValueError("Flatfield source and target origins must contain x and y")
    origin = tuple(map(int, source_origin_xy))
    target_origin = tuple(map(int, target_origin_xy))
    if tuple(source_origin_xy) != origin or tuple(target_origin_xy) != target_origin:
        raise ValueError("Flatfield source and target origins must be integer")
    from ..phase.features import _h as normalize_homography

    def _h(value):
        try:
            return normalize_homography(value, parameters={"alignment": {
                "homography_floor": parameters["change_display"]["homography_floor"]}})
        except ValueError as error:
            raise ValueError(parameters["change_display"]["homography_error"]) from error

    transform = _h(h_baseline_to_current)
    source_to_native = np.array([[1., 0., origin[0]], [0., 1., origin[1]],
                                 [0., 0., 1.]], np.float64)
    native_to_target = np.array([[1., 0., -target_origin[0]],
                                 [0., 1., -target_origin[1]],
                                 [0., 0., 1.]], np.float64)
    source_to_current = _h(native_to_target @ transform @ source_to_native)
    aligned = cv2.warpPerspective(
        source, source_to_current, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    footprint = cv2.warpPerspective(
        source_valid.astype(np.float32), source_to_current, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0)
    valid = footprint >= np.float32(1 - parameters["change_display"]["validity_tolerance"])
    aligned[~valid] = 0
    smoothed = cv2.blur(aligned, (parameters["change_display"]["box_size"], parameters["change_display"]["box_size"]))
    smoothed.setflags(write=False)
    valid.setflags(write=False)
    return RegisteredDisplayFlatfieldReference(
        smoothed, valid, prepared.invalid_cleanup,
        {"policy": "warp-calibrated-linear-luminance-then-box5",
         "source_origin_xy": list(origin),
         "target_origin_xy": list(target_origin),
         "target_shape_hw": [height, width],
         "valid_pixels": int(np.count_nonzero(valid)),
         "H_baseline_to_current": transform.tolist()},
    )


def _decode_display_lift(value: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != (*shape_hw, 3):
        raise ValueError("Existing lift1 image must have the native analysis shape")
    if image.dtype == np.uint8:
        return image
    if image.dtype != np.float32 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise TypeError("Existing lift1 image must be uint8 RGB or finite float32 RGB in [0, 1]")
    return np.rint(image * 255).astype(np.uint8)


def render_change_flatfield(current_log_camera_rgb: np.ndarray,
                            reference: RegisteredDisplayFlatfieldReference,
                            photo_sand: np.ndarray, *,
                            lift1_rgb: np.ndarray | None = None,
                            tile_rows: int, parameters: Mapping) -> tuple[np.ndarray, dict]:
    """Render the fixed change view as HWC uint8 RGB.

    ``lift1_rgb`` should be the existing analysis-grid lift1 display; its
    pixels are copied exactly outside photo sand.  If omitted, the configured
    renderer supplies it.  The input arrays are never modified.
    """
    signal = _require_display_log_signal(current_log_camera_rgb, name="Flatfield current frame")
    if not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")
    height, width = signal.shape[1:]
    sand = np.asarray(photo_sand)
    if sand.dtype != np.bool_ or sand.shape != (height, width):
        raise TypeError("Photo-sand mask must be native-shape boolean")
    if not isinstance(reference, RegisteredDisplayFlatfieldReference):
        raise TypeError("reference must be a per-frame RegisteredFlatfieldReference")
    prepared = reference
    if prepared.luminance_box5.shape != (height, width) or prepared.valid.shape != (height, width):
        raise ValueError("Pre-injection reference does not match current native geometry")
    if (prepared.luminance_box5.dtype != np.float32 or prepared.valid.dtype != np.bool_
            or not np.isfinite(prepared.luminance_box5).all()):
        raise TypeError("Prepared pre-injection reference is malformed")

    if lift1_rgb is None:
        modes, _ = render_display_modes(signal, sand, tile_rows=tile_rows, parameters=parameters["display_modes"])
        lift1 = modes[parameters["change_display"]["lift_mode"]]
    else:
        lift1 = _decode_display_lift(lift1_rgb, (height, width))
    output = np.array(lift1, dtype=np.uint8, copy=True)
    linear, valid, input_meta = _display_calibrated_linear(signal, tile_rows=tile_rows, parameters=parameters)
    # A centre pixel with a large invalid region one or two pixels away has
    # a zero-contaminated box result.  Leave it on the already valid lift1
    # photograph instead of presenting a false colour change.  Reflect at
    # image edges, matching cv2.blur's border semantics.
    usable = sand & display_box_fully_valid(valid, parameters=parameters) & display_box_fully_valid(prepared.valid, parameters=parameters)
    gain = parameters["display_modes"]["gains"][parameters["change_display"]["lift_mode"]]
    white = parameters["display_modes"]["white_multiplier"] * gain
    # Haloed tiles reproduce a full-frame OpenCV box filter while keeping the
    # extra current-frame memory bounded independently of image height.
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        halo_start, halo_stop = max(0, start - parameters["change_display"]["box_radius"]), min(height, stop + parameters["change_display"]["box_radius"])
        # BORDER_DEFAULT is the same reflected boundary used by cv2.blur in
        # the read-only prototype.  At interior tile boundaries the halo
        # supplies the real neighbouring rows.
        blurred = cv2.blur(linear[halo_start:halo_stop], (parameters["change_display"]["box_size"], parameters["change_display"]["box_size"]))[
            start - halo_start:stop - halo_start]
        divisor = np.maximum(prepared.luminance_box5[start:stop],
                             np.float32(parameters["change_display"]["minimum_reference_luminance"]))
        flatfield = blurred * (np.float32(parameters["change_display"]["reference_luminance"]) / divisor)[..., None]
        luminance = flatfield @ np.array(parameters["display_modes"]["luminance_weights"], dtype=np.float64).astype(np.float32)
        gained = gain * luminance.astype(np.float64)
        toned = gained * (1 + gained / white ** 2) / (1 + gained)
        ratio = np.divide(toned, luminance, out=np.zeros_like(toned), where=luminance > 0)
        rgb = flatfield.astype(np.float64) * ratio[..., None]
        maximum = np.max(rgb, axis=-1, keepdims=True)
        rgb /= np.maximum(maximum, 1)
        rgb8 = np.rint(255 * _srgb_encode(rgb, parameters=parameters["display_modes"])).astype(np.uint8)
        # Unlike the three photographic source modes, the flatfield view
        # intentionally uses the box filter on linear colour itself.  Its
        # fixed transform contains no additional chroma filter.
        selected = usable[start:stop]
        output[start:stop][selected] = rgb8[selected]
    return output, {
        "contract": parameters["change_display"]["contract"],
        "display_only": True,
        "source_mode": parameters["display_modes"]["contract_metadata"]["contract"],
        "reference_luminance": parameters["change_display"]["reference_luminance"],
        "box_size_px": parameters["change_display"]["box_size"],
        "lift_gain": gain,
        "lift_white": white,
        "sand_pixels": int(np.count_nonzero(sand)),
        "flatfield_pixels": int(np.count_nonzero(usable)),
        "sand_pixels_without_valid_pair": int(np.count_nonzero(sand & ~usable)),
        "current_cleanup": input_meta,
        "reference_cleanup": prepared.invalid_cleanup,
        "reference_alignment": prepared.alignment,
        "outside_photo_sand": "existing_lift1_unchanged",
    }
