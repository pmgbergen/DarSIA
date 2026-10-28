"""Display independently saved cell masses admitted by the storage readers.

The caller must use the existing committed native-phase or N1 saved reader.
These display-shape checks do not replace that reader's provenance, immutable
asset, precision and independent scalar-closure verification. No file is read
or written here, and no mass component or unknown remainder is reconstructed.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import math
from numbers import Real

import numpy as np

from .storage_saved_rendering import release_stored_figure


_COMPONENTS = {"aqueous": "Aqueous CO₂", "gas": "Free-gas CO₂", "total": "Total CO₂"}
_READER_FORMAT = "darsia-read-coarse-stored-frame-v1"
_PRODUCERS = {None: "darsia-storage-frame-v1", "n1-mass-only": "darsia-n1-storage-frame-v1"}


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _edges(start, stop, count, step_mm):
    # Preserve fixed internal edges and a possibly shorter last display cell.
    # Decimal comparison avoids granting an extra cell through binary rounding.
    first, last = Decimal(str(start)), Decimal(str(stop))
    step = Decimal(str(step_mm)) / Decimal("1000")
    if not first + (count - 1) * step < last <= first + count * step:
        raise ValueError("Saved mass grid shape, fixed cell width and extent differ.")
    edges = float(start) + np.arange(count + 1, dtype=np.float64) * (float(step_mm) / 1000.)
    edges[-1] = stop
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Saved mass cell edges must increase without stretching or clamping.")
    return edges


def _display_inputs(stored, component, limit):
    if not _number(limit) or limit <= 0:
        raise ValueError("Mass display needs an explicit fixed positive limit in g per stored cell.")
    if component not in _COMPONENTS:
        raise ValueError("Choose the saved aqueous, gas or total mass component.")
    if not isinstance(stored, Mapping) or stored.get("format") != _READER_FORMAT:
        raise ValueError("A verified saved-storage reader frame is required.")
    producer, document = stored.get("producer_kind"), stored.get("storage_document")
    if (producer is not None and not isinstance(producer, str)
            or producer not in _PRODUCERS or not isinstance(document, Mapping)
            or document.get("format") != _PRODUCERS[producer]
            or document.get("complete") is not True
            or document.get("existing_native_products_unchanged") is not True
            or producer is not None and document.get("producer_kind") != producer):
        raise ValueError("The saved mass frame has an unsupported original producer.")
    precision = stored.get("precision")
    if (precision not in {"float32", "float64"} or document.get("precision") != precision
            or stored.get("imshow_origin") != "lower"):
        raise ValueError("Saved mass precision or metric orientation differs.")
    grid, arrays = stored.get("grid"), stored.get("arrays")
    if not isinstance(grid, Mapping) or not isinstance(arrays, Mapping):
        raise ValueError("Saved mass grid and arrays are required.")
    shape = grid.get("shape")
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(size) is not int or size <= 0 for size in shape)
            or grid.get("format") != "darsia-metric-storage-grid-v1"
            or grid.get("axes") != "row=y increasing, column=x increasing"
            or grid.get("storage_precision") != precision):
        raise ValueError("Saved mass grid shape, orientation or precision differs.")
    field = arrays.get("cell_mass_" + component + "_g")
    coverage = arrays.get("static_cell_has_area")
    if (not isinstance(field, np.ndarray) or field.shape != tuple(shape)
            or field.dtype != np.dtype(precision) or not np.isfinite(field).all()
            or np.any(field < 0) or not isinstance(coverage, np.ndarray)
            or coverage.shape != field.shape or coverage.dtype != np.bool_):
        raise ValueError("Saved cell mass or coverage differs from its declared grid and precision.")
    extent, resolution = grid.get("extent_m"), grid.get("resolution_mm")
    if (not isinstance(extent, (list, tuple)) or len(extent) != 4
            or not all(_number(value) for value in extent)
            or not _number(resolution) or resolution <= 0):
        raise ValueError("Saved mass display requires its finite metric extent and cell resolution.")
    x_edges = _edges(extent[0], extent[1], shape[1], resolution)
    y_edges = _edges(extent[2], extent[3], shape[0], resolution)
    scope = stored.get("analysis_scope")
    unknown = arrays.get("mask_approved_analysis_gap_area_fraction")
    if unknown is not None:
        if (not isinstance(unknown, np.ndarray) or unknown.shape != field.shape
                or unknown.dtype != np.dtype(precision)
                or not np.array_equal(np.isnan(unknown), ~coverage)
                or np.any(coverage & (~np.isfinite(unknown) | (unknown < 0) | (unknown > 1)))):
            raise ValueError("Saved unknown-area fractions differ from the mass grid or coverage.")
    has_unknown = unknown is not None and bool(np.any(unknown > 0))
    if scope is not None:
        if (not isinstance(scope, Mapping)
                or scope.get("status") != "quantified_subtotal_with_unquantified_remainder"
                or scope.get("whole_rig_mass_complete") is not False
                or scope.get("unknown_mass_g") is not None
                or stored.get("unknown_mass_g") is not None
                or stored.get("mass_totals_g") != dict.fromkeys(_COMPONENTS)
                or not has_unknown):
            raise ValueError("Saved mass scope must retain its unknown remainder and mask.")
    elif has_unknown:
        raise ValueError("An unknown-area mass mask needs its explicit quantified-subtotal scope.")
    return field, coverage, unknown if has_unknown else None, resolution, x_edges, y_edges


def mass_figure(stored, component, limit_g_per_cell):
    """Render an independent saved mass component without changing any array.

    The explicit positive scale is display metadata and stays fixed for the
    caller's comparison. Hatched cells contain unknown area; their colour is
    only the saved quantified contribution, never an estimate of the remainder.
    The caller owns the returned Figure and must call release_stored_figure.
    """
    field, coverage, unknown, resolution, x_edges, y_edges = _display_inputs(
        stored, component, limit_g_per_cell)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    figure = Figure(figsize=(7, 3.8), layout="constrained")
    try:
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.set_facecolor("#e6edf2")
        displayed = np.ma.array(field, mask=~coverage, copy=True)
        image = axis.pcolormesh(x_edges, y_edges, displayed, shading="flat",
            edgecolors="none", antialiased=False, cmap="viridis",
            vmin=0, vmax=float(limit_g_per_cell))
        axis.set(xlabel="x (m)", ylabel="y (m)",
                 xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[0], y_edges[-1]), aspect="equal")
        title = f"{_COMPONENTS[component]} mass · {resolution:g} mm stored cells"
        if unknown is not None:
            rows, columns = np.nonzero(unknown > 0)
            polygons = np.empty((len(rows), 4, 2), dtype=np.float64)
            polygons[:, :, 0] = np.column_stack((x_edges[columns], x_edges[columns + 1],
                                                x_edges[columns + 1], x_edges[columns]))
            polygons[:, :, 1] = np.column_stack((y_edges[rows], y_edges[rows],
                                                y_edges[rows + 1], y_edges[rows + 1]))
            overlay = PathPatch(Path.make_compound_path_from_polys(polygons),
                facecolor="none", edgecolor="#343a40", linewidth=0, hatch="////", zorder=3)
            axis.add_patch(overlay)
            title += "\nQuantified contribution; unknown areas remain"
            axis.text(.5, -.22, "Hatched: unknown area present; colour shows quantified contribution.",
                      transform=axis.transAxes, ha="center", va="top", fontsize=8)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="g per stored cell · fixed linear scale",
                        extend="max" if np.any(field[coverage] > limit_g_per_cell) else "neither")
        return figure
    except Exception:
        release_stored_figure(figure)
        raise
