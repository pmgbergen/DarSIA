"""Display-only internal sand interfaces, without edge-raster contour caps."""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QPainterPath


def full_material_display(source, source_to_native_h, native_to_photo_h,
                          own_frame_native_xy, shape_hw):
    """Carry the verified source directly to the full photograph for display.

    The existing nearest-edge continuation reaches the measured frame even
    when that frame extends outside the old native image's cropped raster.
    No saved source, native product or analysis support is changed here.
    """
    import cv2

    labels = np.asarray(source)
    if labels.ndim != 2 or labels.dtype != np.uint8 or not all(labels.shape):
        raise ValueError('Full-map display requires the original source label grid')
    matrices = [np.asarray(value, np.float64) for value in (source_to_native_h, native_to_photo_h)]
    if any(value.shape != (3, 3) or not np.isfinite(value).all() for value in matrices):
        raise ValueError('Full-map display requires both measured photograph transforms')
    frame = np.asarray(own_frame_native_xy, np.float64)
    if frame.shape != (4, 2) or not np.isfinite(frame).all():
        raise ValueError('Full-map display requires the measured reference frame')
    height, width = shape_hw
    if not (0 < height and 0 < width and height * width <= 4 * 64_000_000):
        raise ValueError('Full-map display exceeds its bounded photograph grid')
    transform = matrices[1] @ matrices[0]
    material = cv2.warpPerspective(labels, transform, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    projected = np.column_stack((frame, np.ones(4))) @ matrices[1].T
    if np.any(np.abs(projected[:, 2]) < 1e-12) or not np.isfinite(projected).all():
        raise ValueError('The measured reference frame crosses its display horizon')
    polygon = projected[:, :2] / projected[:, 2:3]
    inside = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(inside, np.rint(polygon * 256).astype(np.int32), 1, shift=8)
    material[inside == 0] = 0
    return material


def _runs(mask):
    """Return row, start and exclusive stop for each horizontal true run."""
    padded = np.pad(mask, ((0, 0), (1, 1)))
    change = np.diff(padded.astype(np.int8), axis=1)
    rows, starts = np.nonzero(change == 1)
    _, stops = np.nonzero(change == -1)
    return rows, starts, stops


def sand_boundary_segments(material_labels, porous_ids):
    """Open native-xy segments separating two distinct porous materials.

    Pixel centres have integer coordinates, so interfaces lie on half pixels.
    Other roles and the outside of the array do not create white sand lines.
    Contiguous collinear interfaces are merged; no segment joins across a gap.
    The supplied labels are never modified.
    """
    material = np.asarray(material_labels)
    if material.ndim != 2 or material.dtype.kind not in "iu":
        raise ValueError("Sand overlays require a two-dimensional integer material map")
    porous = np.isin(material, list(porous_ids))
    down = ((material[1:] != material[:-1]) & porous[1:] & porous[:-1])
    right = ((material[:, 1:] != material[:, :-1]) & porous[:, 1:] & porous[:, :-1])
    segments = []
    for row, start, stop in zip(*_runs(down)):
        segments.append(np.array([[start - .5, row + .5],
                                  [stop - .5, row + .5]], dtype=np.float64))
    for col, start, stop in zip(*_runs(right.T)):
        segments.append(np.array([[col + .5, start - .5],
                                  [col + .5, stop - .5]], dtype=np.float64))
    return segments


def sand_boundary_path(material_labels, porous_ids, native_to_photo_h):
    """Project the open internal interfaces onto the bound photograph."""
    matrix = np.asarray(native_to_photo_h, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Sand overlays require the explicit native-to-photo transform")
    segments = sand_boundary_segments(material_labels, porous_ids)
    path = QPainterPath()
    if not segments:
        return path
    points = np.asarray(segments).reshape(-1, 2)
    carried = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if not np.all(np.isfinite(carried)) or np.any(carried[:, 2] == 0):
        raise ValueError("The sand-overlay photograph transform is undefined here")
    projected = (carried[:, :2] / carried[:, 2:3]).reshape(-1, 2, 2)
    for start, stop in projected:
        path.moveTo(*start)
        path.lineTo(*stop)
    return path
