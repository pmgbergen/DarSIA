"""Prepare fixed storage-grid metadata from declared finite-centre bounds.

This does not read images, generate coordinates, or change the strict storage
writer. The caller must bind the original metadata sources and later adopt the
chosen grid explicitly. Native storage bypasses even the extent inspection.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import math

from .storage_resolution import metric_grid_shape, resolution_mm


UPPER_EDGE_RULE = (
    "origin-zero integer h lattice; lower=floor(min/h); upper=floor(max/h)+1; "
    "an exact maximum on the exclusive upper edge adds exactly one cell, on each axis"
)


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("Finite centre bounds must contain JSON numbers, excluding booleans.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Finite centre bounds must not contain NaN or infinity.")
    return Decimal(str(value))


def _edge_index(value, step):
    numerator, denominator = value.as_integer_ratio()
    step_n, step_d = step.as_integer_ratio()
    return divmod(numerator * step_d, denominator * step_n)


def _edge_decimal(index, step):
    # Multiplication stays exact even for large declared origins. The existing
    # validator below refuses any float serialization that loses the lattice.
    with localcontext() as context:
        context.prec = len(str(abs(index))) + len(step.as_tuple().digits) + 8
        return step * index


def build_origin_grid(finite_centre_extents_m, requested_resolution_mm):
    """Return metadata for one fixed union grid, or None for native storage.

    Each extent is [x_min, x_max, y_min, y_max] of finite native pixel centres,
    not a half-open output extent. A single-point extent is allowed. The writer
    still rejects any actual centre outside the explicitly adopted [min,max).
    """
    step_mm = resolution_mm(requested_resolution_mm)
    if step_mm is None:
        return None
    if not isinstance(finite_centre_extents_m, (list, tuple)) or not finite_centre_extents_m:
        raise ValueError("At least one explicit finite-centre extent is required.")
    bounds = []
    for extent in finite_centre_extents_m:
        if not isinstance(extent, (list, tuple)) or len(extent) != 4:
            raise ValueError("Each finite-centre extent must be x_min,x_max,y_min,y_max.")
        values = tuple(_number(value) for value in extent)
        if values[0] > values[1] or values[2] > values[3]:
            raise ValueError("Finite-centre extent bounds must be ordered.")
        bounds.append(values)
    union = (min(v[0] for v in bounds), max(v[1] for v in bounds),
             min(v[2] for v in bounds), max(v[3] for v in bounds))
    step = Decimal(str(step_mm)) / Decimal("1000")
    quotients = [_edge_index(value, step) for value in union]
    indices = [quotients[0][0], quotients[1][0] + 1,
               quotients[2][0], quotients[3][0] + 1]
    decimal_edges = [_edge_decimal(index, step) for index in indices]
    extent = [float(edge) for edge in decimal_edges]
    if not all(math.isfinite(edge) for edge in extent):
        raise ValueError("The declared storage extent cannot be serialized as finite JSON floats.")
    height, width = metric_grid_shape(extent, step_mm)
    if [height, width] != [indices[3] - indices[2], indices[1] - indices[0]]:
        raise ValueError("Float serialization changed the declared integer grid ranges.")
    return dict(
        format="darsia-storage-grid-metadata-builder-v1",
        resolution_mm=step_mm,
        step_m_exact_decimal=str(step),
        origin_m=[0, 0],
        source_extent_count=len(bounds),
        finite_centres_union_m=[float(value) for value in union],
        integer_edge_indices_xy=indices,
        extent_m_exact_decimal=[str(value) for value in decimal_edges],
        extent_m=extent,
        shape_hw=[height, width],
        cell_count=height * width,
        upper_edge_ties_xy=[quotients[1][1] == 0, quotients[3][1] == 0],
        upper_edge_rule=UPPER_EDGE_RULE,
        writer_policy="strict half-open [min,max); STOP outside; never resize, clamp or drop centres",
        computing_resolution="native",
    )
