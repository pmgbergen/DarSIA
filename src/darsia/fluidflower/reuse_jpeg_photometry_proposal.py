"""Unwired proposal for measured JPEG response and a fixed series flat field.

This is an arithmetic core, not an approved FF_RE calibration or decoder. The
caller supplies measured per-channel encoded-to-linear knots in explicit units,
a fixed positive illumination field, its support and three normalization levels.
No sRGB response, chart fit, black/white endpoint, clipping, extrapolation,
illumination fit or normalization statistic is inferred here. The proposed
piecewise-linear interpolation, field normalization and float precision remain
open choices for the project owner. Nothing imports this module from an existing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _frozen_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _boolean_mask(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype(bool) or result.shape != shape:
        raise ValueError(f"{name} must be a boolean array of shape {shape}")
    return result


@dataclass(frozen=True)
class JpegPhotometryResult:
    """Linear values and separate support reasons; unsupported values are NaN."""

    linear_rgb: np.ndarray
    corrected_rgb: np.ndarray
    channel_valid: np.ndarray
    invalid_input: np.ndarray
    out_of_calibration: np.ndarray
    flatfield_invalid: np.ndarray
    invalid_result: np.ndarray

    @property
    def valid_pixels(self) -> np.ndarray:
        """True only when all three corrected channels are supported."""
        return np.all(self.channel_valid, axis=-1)


@dataclass(frozen=True)
class FixedSeriesJpegProposal:
    """Caller-defined, reusable response and field; every policy is required.

    Each of the three knot arrays may have its own length. Encoded knots must
    increase strictly and measured linear knots must be finite, nonnegative and
    monotone. Knots define the entire calibrated support, including endpoints.
    No unsupported gap inside those intervals is assumed; the caller's validity
    mask must exclude such gaps as well as clipped, dark or otherwise unusable
    decoded samples. Encoded units must match the knots; uint8 is not rescaled.

    ``flatfield_rgb`` is a fixed per-channel illumination level on the exact
    analysis grid. At supported channels it must be finite and positive. The
    correction is ``linear * normalization_rgb / flatfield_rgb``. The caller
    chooses the physical reference, fit, masks and normalization levels; no
    spatial resampling or recalibration occurs during apply.

    Arithmetic uses float64 before the explicitly requested output cast.
    ``outside_support='nan'`` preserves unsupported values as NaN with masks;
    ``'raise'`` rejects a caller-valid sample outside the measured interval.
    Neither policy clips or extrapolates. Selection of either is a proposal.
    """

    series_id: str
    encoded_knots: tuple[np.ndarray, np.ndarray, np.ndarray]
    linear_knots: tuple[np.ndarray, np.ndarray, np.ndarray]
    flatfield_rgb: np.ndarray
    flatfield_support: np.ndarray
    normalization_rgb: np.ndarray
    interpolation: Literal["piecewise_linear"]
    outside_support: Literal["nan", "raise"]
    output_dtype: Literal["float32", "float64"]

    def __post_init__(self) -> None:
        if not isinstance(self.series_id, str) or not self.series_id.strip():
            raise ValueError("series_id must be a nonempty string")
        if self.interpolation != "piecewise_linear":
            raise ValueError("Only explicitly selected piecewise_linear is implemented")
        if self.outside_support not in ("nan", "raise"):
            raise ValueError("outside_support must be 'nan' or 'raise'")
        if self.output_dtype not in ("float32", "float64"):
            raise ValueError("output_dtype must be 'float32' or 'float64'")
        if len(self.encoded_knots) != 3 or len(self.linear_knots) != 3:
            raise ValueError("Exactly three measured channel responses are required")
        encoded, linear = [], []
        for x, y in zip(self.encoded_knots, self.linear_knots, strict=True):
            x, y = _frozen_array(x, np.float64), _frozen_array(y, np.float64)
            if x.ndim != 1 or len(x) < 2 or y.shape != x.shape:
                raise ValueError("Each response needs at least two matching 1D knots")
            if not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
                raise ValueError("Encoded knots must be finite and strictly increasing")
            if (not np.all(np.isfinite(y)) or np.any(y < 0)
                    or np.any(np.diff(y) < 0)):
                raise ValueError("Linear knots must be finite, nonnegative and monotone")
            encoded.append(x)
            linear.append(y)
        flat = _frozen_array(self.flatfield_rgb, np.float64)
        if flat.ndim != 3 or flat.shape[-1] != 3 or 0 in flat.shape:
            raise ValueError("flatfield_rgb must have nonempty shape (H, W, 3)")
        support = _boolean_mask(self.flatfield_support, flat.shape, "flatfield_support")
        if np.any(support & (~np.isfinite(flat) | (flat <= 0))):
            raise ValueError("Supported flat-field levels must be finite and positive")
        norm = _frozen_array(self.normalization_rgb, np.float64)
        if norm.shape != (3,) or not np.all(np.isfinite(norm)) or np.any(norm <= 0):
            raise ValueError("normalization_rgb must contain three finite positive levels")
        object.__setattr__(self, "encoded_knots", tuple(encoded))
        object.__setattr__(self, "linear_knots", tuple(linear))
        object.__setattr__(self, "flatfield_rgb", flat)
        object.__setattr__(self, "flatfield_support", _frozen_array(support, bool))
        object.__setattr__(self, "normalization_rgb", norm)

    def apply(
        self, encoded_rgb: np.ndarray, *, series_id: str, input_valid: np.ndarray
    ) -> JpegPhotometryResult:
        """Apply the same fixed proposal to a baseline or current decoded array."""
        if series_id != self.series_id:
            raise ValueError("The fixed calibration belongs to a different series")
        source = np.asarray(encoded_rgb, dtype=np.float64)
        if source.shape != self.flatfield_rgb.shape:
            raise ValueError("Encoded samples must match the fixed flat-field grid")
        caller_valid = _boolean_mask(input_valid, source.shape, "input_valid")
        invalid_input = ~caller_valid | ~np.isfinite(source)
        outside = np.zeros(source.shape, dtype=bool)
        linear = np.full(source.shape, np.nan, dtype=np.float64)
        for channel, (x, y) in enumerate(zip(self.encoded_knots, self.linear_knots, strict=True)):
            values = source[..., channel]
            outside[..., channel] = np.isfinite(values) & ((values < x[0]) | (values > x[-1]))
            supported = ~invalid_input[..., channel] & ~outside[..., channel]
            linear[..., channel][supported] = np.interp(values[supported], x, y)
        if self.outside_support == "raise" and np.any(outside & caller_valid):
            raise ValueError("Caller-valid encoded value lies outside measured response support")
        flat_invalid = ~self.flatfield_support
        valid = ~invalid_input & ~outside & ~flat_invalid
        corrected = np.full(source.shape, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            np.divide(linear * self.normalization_rgb, self.flatfield_rgb,
                      out=corrected, where=valid)
            corrected = corrected.astype(self.output_dtype)
            linear = linear.astype(self.output_dtype)
        invalid_result = valid & (~np.isfinite(corrected) | ~np.isfinite(linear))
        valid &= ~invalid_result
        corrected[~valid] = np.nan
        # Linear response remains available where the illumination field is
        # unsupported; corrected_rgb and channel_valid name the usable product.
        linear[~np.isfinite(linear)] = np.nan
        return JpegPhotometryResult(linear, corrected, valid, invalid_input,
                                    outside, flat_invalid, invalid_result)
