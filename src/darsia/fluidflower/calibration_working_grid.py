"""Source-bound primitives for a future reduced calibration computing grid.

These are intentionally not wired into P1: its RAW, training and history
producers must all adopt this contract together before offering reduced grids.
Unlike storage reduction, linear RGB is pooled before feature extraction.
The full-resolution case preserves the original buffers and coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

FORMAT = "darsia-calibration-working-grid-v1"
FACTORS = (1, 2, 4)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _shape(value):
    if (len(value) != 2 or any(type(v) is not int or v < 1 for v in value)):
        raise ValueError("A positive integer height/width grid is required")
    return tuple(value)


def _h(value):
    matrix = np.asarray(value, np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < np.finfo(float).eps):
        raise ValueError("A finite nonsingular pixel transform is required")
    return matrix


def _axis_integral(values, count, axis):
    """Exact overlap of uniform pixel footprints, with float64 accumulation."""
    source = np.moveaxis(np.asarray(values, np.float64), axis, 0)
    length = source.shape[0]
    edges = np.linspace(0., float(length), count + 1)
    lower = np.floor(edges).astype(np.intp)
    prefix = np.concatenate((np.zeros_like(source[:1]), np.cumsum(source, axis=0)), axis=0)
    partial = source[np.minimum(lower, length - 1)]
    fraction = (edges - lower).reshape((-1,) + (1,) * (source.ndim - 1))
    integral = prefix[lower] + fraction * partial
    return np.moveaxis(np.diff(integral, axis=0), 0, axis)


@dataclass(frozen=True)
class CalibrationWorkingGrid:
    source_shape_hw: tuple[int, int]
    shape_hw: tuple[int, int]
    factor: int
    source_basis_json: str

    @property
    def is_full(self):
        return self.factor == 1

    @property
    def source_basis(self):
        return json.loads(self.source_basis_json)

    def binding(self):
        value = dict(format=FORMAT, factor=self.factor,
                     source_shape_hw=list(self.source_shape_hw), shape_hw=list(self.shape_hw),
                     source_basis=self.source_basis,
                     sampling="same pixel footprint; area average of linear RGB before features",
                     area="sum of measured native physical pixel areas")
        return dict(value, sha256=_digest(value))

    @property
    def native_to_working_h(self):
        """Pixel centres, not top-left corners; the physical crop is unchanged."""
        sy, sx = np.asarray(self.shape_hw) / np.asarray(self.source_shape_hw)
        ox, oy = self.source_basis["native_origin_xy"]
        return np.array([[sx, 0., (sx - 1.) / 2. - sx * ox],
                         [0., sy, (sy - 1.) / 2. - sy * oy], [0., 0., 1.]])

    @property
    def working_to_camera_h(self):
        return (_h(self.source_basis["native_to_camera_h"])
                @ np.linalg.inv(self.native_to_working_h))

    def transport_from(self, previous, native_previous_to_current_h):
        """Keep current/previous history alignment in their actual reduced grids."""
        if self.factor != previous.factor:
            raise ValueError("A phase history cannot change computing resolution")
        return (self.native_to_working_h @ _h(native_previous_to_current_h)
                @ np.linalg.inv(previous.native_to_working_h))

    def extensive(self, values):
        """Conserve sums of mm², pore volume, or another extensive pixel field."""
        source = np.asarray(values)
        if source.shape[:2] != self.source_shape_hw or not np.isfinite(source).all():
            raise ValueError("Finite extensive values must share the source grid")
        if self.is_full:
            return source
        # Integral is over source pixel footprints; convert the sum contribution
        # assigned to each source footprint to a density before integrating.
        result = _axis_integral(source, self.shape_hw[0], 0)
        return _axis_integral(result, self.shape_hw[1], 1)

    def intensive(self, values, weights):
        """Physical-area/pore-volume weighted field, preserving its integral."""
        source, weight = np.asarray(values), np.asarray(weights)
        if (source.shape[:2] != self.source_shape_hw or weight.shape != self.source_shape_hw
                or not np.isfinite(source).all() or not np.isfinite(weight).all()
                or np.any(weight <= 0)):
            raise ValueError("Finite values and positive weights must share the source grid")
        if self.is_full:
            return source
        expanded = weight[(...,) + (None,) * (source.ndim - 2)]
        reduced_weight = self.extensive(weight)
        return self.extensive(source * expanded) / reduced_weight[(...,) + (None,) * (source.ndim - 2)]

    def linear_rgb(self, values):
        """Average decoded linear RGB before logs or classifier features exist."""
        source = np.asarray(values)
        if (source.shape != (*self.source_shape_hw, 3) or source.dtype.kind != "f"
                or not np.isfinite(source).all() or np.any(source < 0)):
            raise ValueError("Finite nonnegative floating-point linear RGB is required")
        if self.is_full:
            return source
        footprint = math.prod(self.source_shape_hw) / math.prod(self.shape_hw)
        return self.extensive(source) / footprint

    def training_labels(self, values, *, excluded=255):
        """Transfer only pure labelled footprints; mixed annotations stay unlabelled."""
        source = np.asarray(values)
        if source.shape != self.source_shape_hw or source.dtype.kind not in "iu":
            raise ValueError("Integer training labels must share the source grid")
        if self.is_full:
            return source
        output = np.full(self.shape_hw, excluded, dtype=source.dtype)
        footprint = math.prod(self.source_shape_hw) / math.prod(self.shape_hw)
        for label in np.unique(source):
            if label != excluded:
                fraction = self.extensive((source == label).astype(float)) / footprint
                output[np.isclose(fraction, 1., rtol=0, atol=1e-12)] = label
        return output

    def require_trained_model(self, model):
        """Additional grid admission; original model/feature checks still apply."""
        if self.is_full:
            return
        if self.binding() not in model.get("calibration_working_grid_bases", ()):
            raise ValueError("Retrain the phase model on this source-bound calibration grid")


def calibration_working_grid(source_basis, *, factor=1):
    """Bind a computing grid to real Build/photometry/metric identities.

    Per-frame camera transforms and origins must be supplied before reducing
    images. A common factor alone does not establish a shared image basis.
    """
    if type(factor) is not int or factor not in FACTORS:
        raise ValueError("Calibration computing factor must be 1, 2 or 4")
    basis = dict(source_basis)
    source_shape = _shape(basis["shape_hw"])
    origin = basis["native_origin_xy"]
    if (len(origin) != 2 or any(isinstance(v, bool) or not math.isfinite(v) for v in origin)):
        raise ValueError("An explicit finite native origin is required")
    _h(basis["native_to_camera_h"])
    for key in ("build_sha256", "photometry_sha256", "metric_sha256"):
        value = basis[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("The grid requires exact Build, photometry and metric identities")
    shape = tuple(math.ceil(v / factor) for v in source_shape)
    return CalibrationWorkingGrid(source_shape, shape, factor,
        json.dumps(basis, sort_keys=True, separators=(",", ":"), allow_nan=False))
