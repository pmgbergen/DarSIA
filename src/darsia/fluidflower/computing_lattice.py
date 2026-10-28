"""The origin-zero metric lattice on which the Phase-map method can compute.

The lattice has the cells of the storage layer: storage_resolution validates
the extent and assigns every source pixel centre to its cell, so computed and
stored cells are the same cells. Computing uses 0.5 or 1 mm.

For computing, row 0 holds the highest y, so rows run downward like image rows
and the phase rules see the rig the right way up. :meth:`ComputingLattice.
to_storage` flips an array into storage orientation (row = y increasing).

:class:`CellAssignment` reduces fields of one source raster per cell: sums,
area-weighted means with their valid fraction, mask fractions and area
majorities. Fractions are relative to the source area that falls in the cell.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .storage_resolution import make_grid, metric_grid_shape

COMPUTING_RESOLUTIONS_MM = (0.5, 1.0)
#: A cell belongs to a mask, and a mean is valid, when at least this share of
#: the cell's source area does or is.
MASK_FRACTION = 0.5
LATTICE_FORMAT = "darsia-computing-lattice-v1"


def computing_resolution_mm(value):
    """Admit a computing resolution of 0.5 or 1 mm; native is not a lattice."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("The computing resolution must be 0.5 or 1 mm")
    if float(value) not in COMPUTING_RESOLUTIONS_MM:
        raise ValueError("The computing resolution must be 0.5 or 1 mm")
    return float(value)


@dataclass(frozen=True)
class ComputingLattice:
    """Extent x0, x1, y0, y1 in metres, cell size in millimetres and shape (rows, columns)."""

    extent_m: tuple
    resolution_mm: float
    shape: tuple

    @property
    def spacing_m(self):
        """Cell size in metres."""
        return self.resolution_mm / 1000.0

    def metric_to_cell(self):
        """Map metres (x, y, 1) to cell coordinates (column, row, 1).

        Integer coordinates are cell centres in computing orientation, as for
        image pixels, so cell (r, c) covers [c - 0.5, c + 0.5) x [r - 0.5, r + 0.5).
        """
        h = self.spacing_m
        x0, _, _, y1 = self.extent_m
        return np.array([[1.0 / h, 0.0, -x0 / h - 0.5],
                         [0.0, -1.0 / h, y1 / h - 0.5],
                         [0.0, 0.0, 1.0]])

    def cell_to_metric(self):
        """Map cell coordinates (column, row, 1) to metres (x, y, 1)."""
        h = self.spacing_m
        x0, _, _, y1 = self.extent_m
        return np.array([[h, 0.0, x0 + 0.5 * h],
                         [0.0, -h, y1 - 0.5 * h],
                         [0.0, 0.0, 1.0]])

    def cell_centres_m(self):
        """Metre coordinates of every cell centre, float64, in computing orientation."""
        rows, columns = self.shape
        h = self.spacing_m
        x0, _, _, y1 = self.extent_m
        x = x0 + (np.arange(columns, dtype=np.float64) + 0.5) * h
        y = y1 - (np.arange(rows, dtype=np.float64) + 0.5) * h
        return np.broadcast_to(x[None, :], self.shape).copy(), np.broadcast_to(y[:, None], self.shape).copy()

    def to_storage(self, array):
        """Flip the rows of an array whose first two axes are this lattice."""
        array = np.asarray(array)
        if tuple(array.shape[:2]) != tuple(self.shape):
            raise ValueError("Array does not lie on this computing lattice")
        return array[::-1].copy()

    def metadata(self):
        """The lattice identity recorded in every coarse product."""
        return dict(format=LATTICE_FORMAT, extent_m=list(self.extent_m),
                    resolution_mm=self.resolution_mm, shape=list(self.shape),
                    axes="computing: row 0 = highest y, rows run down; column = x increasing",
                    storage_orientation="rows flipped: row = y increasing",
                    cells="storage_resolution origin-zero half-open cells; pixel centres assigned, no clamping",
                    mask_fraction=MASK_FRACTION)


def computing_lattice(extent_m, resolution_mm):
    """Validate an extent on the origin-zero lattice and return the lattice."""
    step = computing_resolution_mm(resolution_mm)
    if len(extent_m) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                 or not math.isfinite(v) for v in extent_m):
        raise ValueError("An explicit finite metric extent x0, x1, y0, y1 is required")
    shape = metric_grid_shape(extent_m, step)
    return ComputingLattice(tuple(map(float, extent_m)), step, tuple(map(int, shape)))


class CellAssignment:
    """The cells of one source raster's pixel centres, with per-cell reductions.

    ``x_m`` and ``y_m`` are the pixel-centre metre coordinates and
    ``pixel_area`` the positive source pixel areas (any fixed unit). Every
    source pixel centre must lie inside the lattice extent, as for storage.
    """

    def __init__(self, lattice, x_m, y_m, pixel_area):
        """Assign every source pixel with storage_resolution's own rule."""
        self.lattice = lattice
        area = np.asarray(pixel_area, np.float64)
        if area.shape != np.shape(x_m) or not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError("Source pixel areas must be positive and finite on the source grid")
        grid = make_grid(x_m, y_m, area, include=np.ones(area.shape, bool),
                         extent_m=lattice.extent_m, storage_resolution_mm=lattice.resolution_mm,
                         precision="float64")
        if tuple(grid.shape) != tuple(lattice.shape):
            raise ValueError("Storage grid and computing lattice differ")
        self.source_shape = area.shape
        self.cells = grid.probability_cell_index
        self.area = area
        self.size = int(np.prod(lattice.shape))
        self.cell_area = self._bincount(area)
        self.pixel_count = np.bincount(self.cells.ravel(), minlength=self.size).reshape(lattice.shape)[::-1]

    def _bincount(self, weights):
        """Per-cell sums of source weights, in computing orientation."""
        flat = np.bincount(self.cells.ravel(), weights=np.asarray(weights, np.float64).ravel(),
                           minlength=self.size)
        return flat.reshape(self.lattice.shape)[::-1]

    def _source(self, values, name):
        """An array whose last two axes are the assigned source raster."""
        values = np.asarray(values)
        if values.shape[-2:] != self.source_shape:
            raise ValueError(name + " does not lie on the assigned source raster")
        return values

    def total(self, values):
        """Per-cell sum of a source field, float64."""
        values = self._source(values, "Summed field")
        if not np.isfinite(values).all():
            raise ValueError("Summed fields must be finite")
        return self._bincount(values)

    def fraction(self, mask):
        """Area fraction of each cell's source area inside a boolean mask."""
        mask = self._source(mask, "Mask")
        if mask.dtype != bool:
            raise ValueError("Mask fractions need a boolean mask")
        inside = self._bincount(np.where(mask, self.area, 0.0))
        result = np.zeros(self.lattice.shape, np.float64)
        np.divide(inside, self.cell_area, out=result, where=self.cell_area > 0)
        return result

    def mask(self, mask):
        """Cells with at least :data:`MASK_FRACTION` of their area in the mask."""
        return self.fraction(mask) >= MASK_FRACTION

    def mean(self, values, valid):
        """Area-weighted mean over valid pixels and each cell's valid fraction.

        A cell whose valid fraction is below :data:`MASK_FRACTION` gets NaN.
        Returns float64 arrays (mean, fraction).
        """
        values, valid = self._source(values, "Averaged field"), self._source(valid, "Validity")
        if valid.dtype != bool or values.shape != valid.shape:
            raise ValueError("A mean needs a boolean validity on the field's grid")
        if not np.isfinite(values[valid]).all():
            raise ValueError("Valid source values must be finite")
        weight = np.where(valid, self.area, 0.0)
        covered = self._bincount(weight)
        numerator = self._bincount(np.where(valid, values, 0.0) * weight)
        fraction = np.zeros(self.lattice.shape, np.float64)
        np.divide(covered, self.cell_area, out=fraction, where=self.cell_area > 0)
        mean = np.full(self.lattice.shape, np.nan, np.float64)
        usable = (fraction >= MASK_FRACTION) & (covered > 0)
        np.divide(numerator, covered, out=mean, where=usable)
        return mean, fraction

    def majority(self, categories, where=None):
        """Category with the largest source area in each cell; ties take the lowest id.

        ``where`` limits the pixels that vote. A cell without votes gets 0.
        """
        categories = self._source(categories, "Categories")
        if not np.issubdtype(categories.dtype, np.integer):
            raise ValueError("Majorities need integer categories")
        voting = np.ones(categories.shape, bool) if where is None else self._source(where, "Voting mask")
        if voting.dtype != bool:
            raise ValueError("The voting mask must be boolean")
        pixels = np.flatnonzero(voting.ravel())
        values = categories.ravel()[pixels]
        order = np.argsort(values, kind="stable")
        pixels, values = pixels[order], values[order]
        found, starts = np.unique(values, return_index=True)
        stops = np.r_[starts[1:], len(values)]
        cells, area = self.cells.ravel(), self.area.ravel()
        result = np.zeros(self.size, categories.dtype)
        largest = np.zeros(self.size, np.float64)
        # Ascending ids with a strict comparison: a tie keeps the lowest id.
        for category, start, stop in zip(found, starts, stops):
            chosen = pixels[start:stop]
            weight = np.bincount(cells[chosen], weights=area[chosen], minlength=self.size)
            changed = weight > largest
            result[changed] = category
            largest[changed] = weight[changed]
        return result.reshape(self.lattice.shape)[::-1].copy()

    def log_rgb(self, log, valid):
        """Coarsen CHW float32 log RGB through its linear values.

        Each channel is decoded with exp in float64, averaged with area weights
        over valid pixels, and logged again. A cell is valid when its valid
        fraction reaches :data:`MASK_FRACTION`; invalid cells are NaN, as the
        native representation requires. Returns (log, valid, fraction).
        """
        log = self._source(log, "Log RGB")
        if log.dtype != np.float32 or log.ndim != 3 or log.shape[0] != 3:
            raise ValueError("Scientific log RGB must be CHW float32")
        valid = np.asarray(valid)
        if valid.dtype != bool or valid.shape != log.shape[1:] or np.any(valid & ~np.isfinite(log).all(0)):
            raise ValueError("Log RGB validity must be a boolean mask of finite pixels")
        result = np.full((3, *self.lattice.shape), np.nan, np.float32)
        fraction = None
        good = None
        for channel in range(3):
            with np.errstate(over="ignore", invalid="ignore"):
                linear = np.exp(np.where(valid, log[channel], 0.0).astype(np.float64))
            mean, fraction = self.mean(linear, valid)
            usable = np.isfinite(mean) & (mean > 0)
            good = usable if good is None else good & usable
            result[channel][usable] = np.log(mean[usable]).astype(np.float32)
        result[:, ~good] = np.nan
        return result, good, fraction.astype(np.float32)


def canvas_pitch_m(matrix, shape_hw):
    """Mean pixel size in metres of a raster mapped by a projective matrix.

    ``matrix`` maps source pixel coordinates (column, row, 1) to metres. The
    pitch is the square root of the area of the mapped raster quadrilateral
    divided by its pixel count.
    """
    rows, columns = map(int, shape_hw)
    corners = np.array([[-0.5, -0.5], [columns - 0.5, -0.5], [columns - 0.5, rows - 0.5], [-0.5, rows - 0.5]])
    matrix = np.asarray(matrix, np.float64)
    mapped = (matrix @ np.column_stack([corners, np.ones(4)]).T).T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise ValueError("The raster maps to infinity")
    x, y = mapped[:, 0] / mapped[:, 2], mapped[:, 1] / mapped[:, 2]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if not math.isfinite(area) or area <= 0:
        raise ValueError("The mapped raster has no area")
    return math.sqrt(area / (rows * columns))
