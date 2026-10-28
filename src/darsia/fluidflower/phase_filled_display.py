"""Display-only hatching for flagged rig-domain extrapolation.

No phase label, scientific image, annotation or saved map is written here.
The original overlay remains available for scoring and toggling back.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .phase_saved_review_metadata import load_rig_classes, rig_phase_shape
from .phase_rig_display_canvas import pad_rig_display_mask


def resolve_filled_display_mask(review_root, run, frame_index, *, canvas=True):
    path = Path(review_root) / "native_frames" / str(run).casefold() / "manifest.json"
    native = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = native.get("frame_sources") or []
    if type(frame_index) is not int or not 0 <= frame_index < len(rows) or rows[frame_index].get("frame_index") != frame_index:
        raise ValueError("Filled display flags require an exact native frame")
    revision = rows[frame_index].get("rig_domain_revision")
    if revision is None:
        return None
    if tuple(revision["canvas_wh"]) != tuple(native["canvas_wh"]):
        # A saved full-rig view may use the declared phase grid, whose extra
        # bottom rows extend the unchanged photographic canvas. Require both
        # explicit grid bindings; no arbitrary resize, crop or shift is allowed.
        phase_shape = rig_phase_shape(revision)
        if (tuple(native["canvas_wh"])[::-1] != phase_shape
                or tuple(native.get("phase_shape_hw", ())) != phase_shape):
            raise ValueError("Filled display flags belong to another review canvas")
    shape = None if canvas else tuple(native["analysis_image_shape_hw"])
    fields, _ = load_rig_classes(revision, base_dir=path.parent, shape_hw=shape)
    filled = fields["filled_observed"] | fields["filled_unobserved"]
    return pad_rig_display_mask(filled, revision, manifest=native) if canvas else filled


def hatch_filled_rgba(rgba, filled_mask, *, enabled=True, period=12, phase_visible=True):
    """Hatch all filled classes, preserving source bytes and phase colours.

    Phase pixels retain their alpha and get lighter RGB stripes. Transparent
    water gets a separate neutral display stripe, including when phases are
    switched off. No photograph or stored overlay is modified.
    """
    image = np.asarray(rgba)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Filled hatching requires a uint8 RGBA display image")
    result = image.copy() if phase_visible else np.zeros_like(image)
    if not enabled or filled_mask is None:
        return result
    mask = np.asarray(filled_mask)
    if mask.dtype != bool or mask.shape != image.shape[:2]:
        raise ValueError("Filled hatching flags belong to another display grid")
    if type(period) is not int or period < 4:
        raise ValueError("Filled hatching period must be an integer at least four")
    x = np.arange(image.shape[1], dtype=np.int32)[None, :]
    y = np.arange(image.shape[0], dtype=np.int32)[:, None]
    stripe = mask & (((x + y) % period) < 2)
    coloured = stripe & (result[..., 3] > 0)
    neutral = stripe & ~coloured
    values = result[coloured, :3].astype(np.uint16)
    result[coloured, :3] = ((values * 3 + 255 * 2) // 5).astype(np.uint8)
    result[neutral] = [200, 200, 200, 96]
    return result


def hatch_filled_overlay(image, filled_mask=None, *, enabled=True, phase_visible=True):
    """Qt adapter; imports Qt only for the display caller."""
    from PySide6.QtGui import QImage
    if (not enabled or filled_mask is None) and phase_visible:
        return image
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    array = np.frombuffer(converted.constBits(), dtype=np.uint8).reshape(
        converted.height(), converted.bytesPerLine())[:, :converted.width() * 4].reshape(
            converted.height(), converted.width(), 4)
    rgba = hatch_filled_rgba(array, filled_mask, enabled=enabled, phase_visible=phase_visible)
    return QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                  QImage.Format.Format_RGBA8888).copy()
