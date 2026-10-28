"""Additive presentation canvas for the padded FF_AC rig revision.

Only the display image is extended. Source pixels, native boxes, saved maps,
annotations and scientific arrays keep their exact coordinates and bytes.
"""
from __future__ import annotations

import numpy as np

from .phase_saved_review_metadata import rig_phase_shape

BACKGROUND = "#202326"


def _padded_revision(revision):
    return bool(revision and "phase_shape_hw" in revision)


def _copy_flags(source, result, revision, manifest=None):
    for name in ("_photo_sand_domain", "_filled_phase_mask", "_floor_contact_fill_mask"):
        value = getattr(source, name, None)
        if value is not None:
            setattr(result, name, pad_rig_display_mask(value, revision, manifest=manifest))
    return result


def _draw_on_canvas(image, size_wh, offset_xy, *, overlay, background=BACKGROUND):
    from PySide6.QtGui import QColor, QImage, QPainter
    # Opaque photographs retain their pixel format; transparent padding for an
    # overlay needs an alpha format, without changing its original RGBA values.
    fmt = QImage.Format.Format_ARGB32 if overlay else image.format()
    result = QImage(*size_wh, fmt)
    result.fill(QColor(0, 0, 0, 0) if overlay else QColor(background))
    painter = QPainter(result)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
    painter.drawImage(*offset_xy, image)
    painter.end()
    return result


def pad_rig_photograph(image, revision, *, background=BACKGROUND):
    """Append only declared bottom rows, with no crop, scaling or shift."""
    if not _padded_revision(revision):
        return image
    if image.isNull():
        raise ValueError("The rig photograph could not be decoded")
    phase_wh = rig_phase_shape(revision)[::-1]
    source_wh = tuple(map(int, revision["canvas_wh"]))
    size = image.width(), image.height()
    if size == phase_wh:
        return image
    if size != source_wh:
        raise ValueError("Rig photograph differs from its declared photographic canvas")
    result = _draw_on_canvas(image, phase_wh, (0, 0), overlay=False, background=background)
    return _copy_flags(image, result, revision)


def validate_rig_display_overlay(image, revision):
    """Retain the complete declared phase grid, including its filled rows."""
    if not _padded_revision(revision):
        return image
    if image.isNull() or (image.width(), image.height()) != rig_phase_shape(revision)[::-1]:
        raise ValueError("Rig phase overlay differs from its declared complete phase grid")
    return image


def on_common_rig_canvas(image, manifest, revision, *, overlay=False):
    """Align a mixed legacy/new comparison without changing native coordinates.

    This is explicitly opted into by a padded rig partner. A legacy native
    photograph or map is embedded at its own unchanged content box; a full
    old photographic canvas is kept at (0,0). Legacy-only comparisons do not
    call this helper and preserve their existing display convention.
    """
    if not _padded_revision(revision):
        return image
    if image.isNull():
        raise ValueError("The comparison image could not be decoded")
    photo_wh = tuple(map(int, revision["canvas_wh"]))
    phase_wh = rig_phase_shape(revision)[::-1]
    if tuple(map(int, manifest["canvas_wh"])) != photo_wh:
        raise ValueError("Mixed rig comparison photographs use different recorded canvases")
    size = image.width(), image.height()
    if size == phase_wh:
        return image
    if size == photo_wh:
        offset = (0, 0)
    else:
        x, y, width, height = map(int, manifest["content_box_xywh"])
        if size != (width, height) or x < 0 or y < 0 or x + width > photo_wh[0] or y + height > photo_wh[1]:
            raise ValueError("Legacy native image differs from its unchanged recorded content box")
        offset = x, y
    result = _draw_on_canvas(image, phase_wh, offset, overlay=overlay)
    return _copy_flags(image, result, revision, manifest)


def pad_rig_display_mask(mask, revision, *, manifest=None):
    """Embed source display flags at their exact coordinates; padding is false."""
    if mask is None or not _padded_revision(revision):
        return mask
    value = np.asarray(mask)
    phase_shape = rig_phase_shape(revision)
    photo_shape = tuple(map(int, revision["canvas_wh"]))[::-1]
    if value.dtype != bool or value.ndim != 2:
        raise ValueError("Rig display flags must be a two-dimensional boolean mask")
    if value.shape == phase_shape:
        return value
    result = np.zeros(phase_shape, bool)
    if value.shape == photo_shape:
        result[:photo_shape[0], :photo_shape[1]] = value
        return result
    if manifest is not None:
        x, y, width, height = map(int, manifest["content_box_xywh"])
        if value.shape == (height, width) and min(x, y) >= 0 and x + width <= photo_shape[1] and y + height <= photo_shape[0]:
            result[y:y+height, x:x+width] = value
            return result
    raise ValueError("Rig display flags differ from their recorded source or complete phase grid")
