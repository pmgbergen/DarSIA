"""Explicit manual-map priority without changing the polygon geometry."""
from __future__ import annotations

from . import layer_painting as lp


def layer_ids(session):
    """Marked layers in session order; no facies names or role assumptions."""
    return tuple(row.layer_id for row in session.layers if row.always_on_top)


def rasterize_polygons(session, shape, scale):
    """Keep the original raster unchanged unless a layer has explicit priority.

    Priority polygons win over ordinary polygons. Within either group the
    existing smaller-polygon rule and drawing-order tie break are retained.
    Every polygon and vertex stays unchanged.
    """
    classes = lp.rasterize_polygons(session.polygons, shape, scale)
    return apply_priority(session, classes, scale)


def apply_priority(session, classes, scale):
    """Overlay marked polygon interiors on a newly drawn marks array."""
    marked = frozenset(layer_ids(session))
    if marked:
        polygons = [polygon for polygon in session.polygons if polygon.layer_id in marked]
        priority = lp.rasterize_polygons(polygons, classes.shape, scale)
        covered = priority != lp.OUTSIDE
        classes[covered] = priority[covered]
    return classes
