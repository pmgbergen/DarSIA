"""Source-bound, display-only hatching for interpreted floor-contact pixels."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .phase_saved_review_metadata import configured_parameters


def resolve_floor_contact_display_mask(review_root, run, frame_index):
    """Read only the exact per-frame flag declared by a new candidate manifest."""
    manifest_path = Path(review_root) / "native_frames" / str(run).casefold() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows = manifest.get("frame_sources") or []
    if type(frame_index) is not int or not 0 <= frame_index < len(rows):
        raise ValueError("Floor-contact display requires an exact native frame")
    row = rows[frame_index]
    if row.get("frame_index") != frame_index:
        raise ValueError("Floor-contact native frame index binding differs")
    descriptor = row.get("floor_contact_fill")
    if row.get("floor_fill_removed") is True:
        return None  # P1-final retains the old mask as provenance, never as current fill.
    if descriptor is None:
        return None  # Earlier candidates have no floor interpretation flag.
    revision = row.get("rig_domain_revision") or {}
    shape = tuple(map(int, revision.get("phase_shape_hw", ())))
    if (descriptor.get("contract") != configured_parameters("display_views")["floor_contact_contract"] or descriptor.get("frame_index") != frame_index
            or descriptor.get("source_sha256") != row.get("source_sha256")
            or len(str(descriptor.get("source_sha256", ""))) != 64
            or tuple(descriptor.get("shape_hw", ())) != shape or len(shape) != 2):
        raise ValueError("Floor-contact flag source, frame or canvas binding differs")
    path = Path(descriptor["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != descriptor.get("sha256"):
        raise ValueError("Floor-contact flag SHA256 differs")
    array = np.load(io.BytesIO(data), allow_pickle=False)
    if descriptor.get("storage") == "packbits-npy":
        if array.dtype != np.uint8 or array.ndim != 1 or array.size != (shape[0]*shape[1]+7)//8:
            raise ValueError("Packed floor-contact flag has invalid storage")
        array = np.unpackbits(array, count=shape[0]*shape[1]).reshape(shape).astype(bool)
    elif array.dtype != bool or array.shape != shape:
        raise ValueError("Floor-contact flag must be a bool array on its recorded canvas")
    return array


def hatch_floor_contact_rgba(rgba, mask, *, enabled=True, phase_visible=True, period=12):
    """Opposite-sloping dark teal stripes distinguish contact from extrapolation.

    Only already visible aqueous pixels are marked; this flag does not reveal a
    hidden phase layer or change the original overlay used for region scoring.
    """
    image = np.asarray(rgba)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Floor-contact hatch requires uint8 RGBA display pixels")
    result = image.copy()
    if not enabled or not phase_visible or mask is None:
        return result
    flag = np.asarray(mask)
    if flag.dtype != bool or flag.shape != image.shape[:2] or type(period) is not int or period < 4:
        raise ValueError("Floor-contact flags belong to another display grid")
    x = np.arange(image.shape[1], dtype=np.int32)[None]
    y = np.arange(image.shape[0], dtype=np.int32)[:, None]
    # Phase filters may remove aqueous pixels. Never recreate them in gas-only,
    # unresolved-only or phase-off views merely because the flag exists.
    aqueous = (result[..., 1] > result[..., 0]) & (result[..., 1] > result[..., 2]) & (result[..., 3] > 0)
    stripe = flag & aqueous & (((x-y) % period) < 2)
    rgb = result[stripe, :3].astype(np.uint16)
    result[stripe, :3] = ((rgb + np.asarray([0, 70, 100], np.uint16)) // 2).astype(np.uint8)
    return result


def hatch_floor_contact_overlay(image, mask=None, *, enabled=True, phase_visible=True):
    from PySide6.QtGui import QImage
    if not enabled or not phase_visible or mask is None:
        return image
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    array = np.frombuffer(converted.constBits(), np.uint8).reshape(converted.height(), converted.bytesPerLine())[:, :converted.width()*4].reshape(converted.height(), converted.width(), 4)
    rgba = hatch_floor_contact_rgba(array, mask, enabled=enabled, phase_visible=phase_visible)
    return QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()


def __getattr__(name):
    if name != "CONTRACT":
        raise AttributeError(name)
    return configured_parameters("display_views")["floor_contact_contract"]
