"""Explicit accounting for a native canvas with separately declared unknown support.

The caller verifies the approval, source and native-grid bindings before wrapping a
geometry. This module retains that provenance; it does not verify those bindings or
infer physical roles from pixel values. The original geometry remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

import darsia


def _json(value) -> str:
    """Detach a finite JSON value without silently replacing unknown values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mask(value, shape) -> np.ndarray:
    """Copy an exact boolean canvas into immutable bytes-backed storage."""
    array = np.asarray(value)
    if array.dtype != np.dtype(bool) or array.shape != shape:
        raise ValueError("Analysis-domain masks require the exact boolean native canvas")
    return np.frombuffer(array.tobytes(order="C"), dtype=bool).reshape(shape)


def _value(value):
    """Keep an original scalar or detach an array result into immutable storage."""
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
    return value


def _mask_image(mask, geometry):
    """Use original Geometry ROI coordinates without inferring a source frame."""
    return darsia.Image(mask, space_dim=geometry.space_dim, dimensions=list(geometry.dimensions),
                        series=False, scalar=True)


@dataclass(frozen=True)
class QuantifiedIntegral:
    """An explicit quantified contribution, never an implicit full-domain scalar."""

    quantified_value: float | np.ndarray
    unquantified_pixels: int
    approved_gap_pixels: int
    unknown_properties_pixels: int
    complete: bool
    full_value: float | np.ndarray | None
    provenance_json: str
    scope_json: str
    unknown_mass: None = None
    unknown_volume: None = None

    @property
    def provenance(self):
        """Return a detached copy of the caller's original binding provenance."""
        return json.loads(self.provenance_json)

    @property
    def scope(self):
        """Return the canvas and original ROI history used for this accounting."""
        return json.loads(self.scope_json)


@dataclass(frozen=True, init=False)
class AnalysisDomainGeometry(darsia.Geometry):
    """Guard complete totals while retaining independent causes of unknown support.

    The union of the two masks is unquantified within this native canvas. Neither
    mask establishes a physical domain or approves the other. No-mask callers keep
    their original Geometry; this wrapper is only an explicitly selected path.
    """

    _geometry: darsia.Geometry
    _approved_gap_mask: np.ndarray
    _unknown_properties_mask: np.ndarray | None
    _unquantified_mask: np.ndarray
    _provenance_json: str
    _scope_json: str

    def __init__(self, geometry, approved_gap_mask, provenance, *, unknown_properties_mask=None):
        """Snapshot exact masks and provenance around one existing native geometry."""
        if not isinstance(geometry, darsia.Geometry) or isinstance(geometry, AnalysisDomainGeometry):
            raise TypeError("Analysis-domain accounting requires one original Geometry")
        shape = tuple(geometry.num_voxels)
        gap = _mask(approved_gap_mask, shape)
        unknown = None if unknown_properties_mask is None else _mask(unknown_properties_mask, shape)
        combined = gap if unknown is None else _mask(gap | unknown, shape)
        provenance_json = _json(provenance)
        native_canvas = dict(num_voxels=[int(value) for value in shape],
                             dimensions=[float(value) for value in geometry.dimensions],
                             origin=[float(value) for value in _mask_image(gap, geometry).origin])
        scope_json = _json(dict(original_native_canvas=native_canvas, **native_canvas, subregions=[]))
        for name, value in (("_geometry", geometry), ("_approved_gap_mask", gap),
                            ("_unknown_properties_mask", unknown), ("_unquantified_mask", combined),
                            ("_provenance_json", provenance_json), ("_scope_json", scope_json)):
            object.__setattr__(self, name, value)

    @property
    def space_dim(self):
        """Expose the original geometry's spatial dimension."""
        return self._geometry.space_dim

    @property
    def num_voxels(self):
        """Return a detached native canvas shape."""
        return list(self._geometry.num_voxels)

    @property
    def dimensions(self):
        """Return the original physical dimensions without modifying them."""
        return list(self._geometry.dimensions)

    @property
    def voxel_size(self):
        """Return the original voxel dimensions."""
        return list(self._geometry.voxel_size)

    @property
    def voxel_volume(self):
        """Expose original effective volumes, including unknown physical NaNs."""
        return self._geometry.voxel_volume

    @property
    def cached_voxel_volume(self):
        """Expose the original geometry's current integration weights."""
        return self._geometry.cached_voxel_volume

    @property
    def weight(self):
        """Expose an original weighted geometry's weight without changing it."""
        return self._geometry.weight

    @property
    def depth(self):
        """Expose an original extruded geometry's physical depth."""
        return self._geometry.depth

    @property
    def porosity(self):
        """Expose original porosity without equating a gap with zero porosity."""
        return self._geometry.porosity

    @property
    def approved_gap_mask(self):
        """Return immutable approved-gap pixels on the current canvas."""
        return self._approved_gap_mask

    @property
    def unknown_properties_mask(self):
        """Return the independent immutable property mask, or its declared absence."""
        return self._unknown_properties_mask

    @property
    def unquantified_mask(self):
        """Return the immutable union, with overlapping causes counted once."""
        return self._unquantified_mask

    @property
    def provenance(self):
        """Return a detached copy of the original caller-owned provenance."""
        return json.loads(self._provenance_json)

    @property
    def scope(self):
        """Return the native canvas and retained original ROI history."""
        return json.loads(self._scope_json)

    def _data(self, data):
        """Require this exact native canvas rather than implicitly resize a mask."""
        if not isinstance(data, (np.ndarray, darsia.Image)):
            raise ValueError("Data type not supported.")
        array = data if isinstance(data, np.ndarray) else data.img
        if (tuple(array.shape[:self.space_dim]) != tuple(self.num_voxels)
                or isinstance(data, darsia.Image) and data.space_dim != self.space_dim):
            raise ValueError("Analysis-domain data must retain the exact native canvas")
        return array

    def _require_complete(self):
        """Prevent an unknown contribution from becoming a complete scalar total."""
        if self._unquantified_mask.any():
            raise ValueError("The native domain is incomplete; use integrate_quantified for its known contribution")

    def integrate(self, data):
        """Use the original integral only when this complete native scope is known."""
        self._data(data)
        self._require_complete()
        return self._geometry.integrate(data)

    def normalize(self, img, img_ref, return_ratio=False):
        """Keep original normalization and dispatch through this guarded integral."""
        return darsia.Geometry.normalize(self, img, img_ref, return_ratio=return_ratio)

    def make_extensive(self, data):
        """Refuse a complete extensive product when native support is unknown."""
        self._data(data)
        self._require_complete()
        return self._geometry.make_extensive(data)

    def integrate_quantified(self, data) -> QuantifiedIntegral:
        """Sum original known contributions while keeping unknown mass unassigned.

        The original multiply and successive spatial sums are retained. Selection
        removes unknown contributions before summation, including NaN physical
        weights; it does not change public weights, porosity or source pixels.
        """
        array = self._data(data)
        count = int(self._unquantified_mask.sum())
        if count == 0:
            quantified = self._geometry.integrate(data)
        else:
            if isinstance(data, darsia.ExtensiveImage):
                selection_shape = tuple(self.num_voxels) + (1,) * (array.ndim - self.space_dim)
                quantified = np.where(self._unquantified_mask.reshape(selection_shape),
                                      np.zeros((), dtype=array.dtype), array)
            else:
                self._geometry._prepare_cached_voxel_volume(list(self.num_voxels))
                weights = self._geometry.cached_voxel_volume
                quantified = np.zeros(np.broadcast_shapes(np.shape(weights), array.shape),
                                      dtype=np.result_type(weights, array))
                selection_shape = tuple(self.num_voxels) + (1,) * (quantified.ndim - self.space_dim)
                np.multiply(weights, array, out=quantified,
                            where=~self._unquantified_mask.reshape(selection_shape))
            for _ in range(self.space_dim):
                quantified = np.sum(quantified, axis=0)
        quantified = _value(quantified)
        return QuantifiedIntegral(
            quantified, count, int(self._approved_gap_mask.sum()),
            0 if self._unknown_properties_mask is None else int(self._unknown_properties_mask.sum()),
            count == 0, quantified if count == 0 else None,
            self._provenance_json, self._scope_json)

    def subregion(self, roi):
        """Retain both masks using the original geometry and Image ROI operations."""
        def crop(mask):
            """Slice one mask with the original coordinate-to-voxel interpretation."""
            return _mask_image(mask, self._geometry).subregion(roi)

        gap = crop(self._approved_gap_mask)
        unknown = None if self._unknown_properties_mask is None else crop(self._unknown_properties_mask)
        geometry = self._geometry.subregion(roi)
        wrapped = AnalysisDomainGeometry(geometry, gap.img, self.provenance,
            unknown_properties_mask=None if unknown is None else unknown.img)
        scope = self.scope
        current = dict(origin=[float(value) for value in gap.origin],
                       dimensions=[float(value) for value in gap.dimensions],
                       num_voxels=[int(value) for value in gap.num_voxels])
        scope.update(current)
        scope["subregions"].append(current)
        object.__setattr__(wrapped, "_scope_json", _json(scope))
        return wrapped

    def update(self, *args, **kwargs):
        """Require explicit reconstruction after a physical-property update."""
        raise NotImplementedError("Rebuild analysis-domain accounting with its exact masks after updating geometry")
