"""Extra *drawn* photo coverage for the live painting GUI only.

This does not change ManualMap, any saved version, or a proposal. Pixels beyond
the photo cannot be drawn and are never continued from a neighbouring label.
"""
from __future__ import annotations

import numpy as np

from . import layer_painting as lp
from . import painting_layer_priority as priority

KEY = "current_drawing_photo_coverage"


def add_explicit_outside(session, frame, layers, *, drawing=False):
    """Copy a live GUI map, adding only explicit marks outside its rig frame.

    ``layers`` may use the native or display grid. The original in-frame map,
    including its existing edge-fill policy, is kept byte for byte. Rasterising
    into that finite photo grid clips geometry at the photo, never at the rig.
    Strokes belong only to the current drawing; the manual map uses polygons.
    """
    layers = np.asarray(layers)
    if layers.ndim != 2 or layers.dtype != np.uint8:
        raise ValueError("The current drawing has no categorical photograph grid.")
    size = tuple(session.image_size or ())
    if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
        raise ValueError("The current drawing has no recorded photograph size.")
    result = layers.copy()
    if frame is None:
        # The existing caller already treats an absent frame as all-photo.
        return result, 0
    from .painting_run_preview import _same_path
    from .painting_preview_frame_domain import _check_frame_polygon

    if (str(frame.run_id).casefold() != session.run_id.casefold()
            or not frame.image or not _same_path(frame.image, session.image_path)
            or (frame.image_size is not None and tuple(frame.image_size) != size)):
        raise ValueError("The drawing frame belongs to another photograph or native grid.")
    polygon = lp.frame_polygon(frame, (1., 1.))
    _check_frame_polygon(polygon)
    scale = (layers.shape[0] / size[1], layers.shape[1] / size[0])
    outside = ~lp.frame_mask(frame, layers.shape, scale)
    if not np.any(outside):
        return result, 0
    explicit = priority.rasterize_polygons(session, layers.shape, scale)
    if drawing:
        strokes = lp.rasterize_strokes(session.strokes, layers.shape, scale)
        explicit[strokes != lp.OUTSIDE] = strokes[strokes != lp.OUTSIDE]
    drawn = outside & (explicit != lp.OUTSIDE)
    result[drawn] = explicit[drawn]
    return result, int(np.count_nonzero(drawn))
