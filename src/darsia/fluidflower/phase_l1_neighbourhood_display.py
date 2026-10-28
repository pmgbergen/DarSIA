"""Source-bound display hatching for the stored L1 neighbourhood gas flag."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .review_storage import resolve_review_path

CONTRACT = "ffac-l1-neighbourhood-gas-v1"


def resolve_l1_neighbourhood_display_mask(review_root, run, frame_index):
    """Read the declared flag without inferring, changing or revealing labels."""
    from .phase_rig_bottom_review_policy import require_predictions_visible

    require_predictions_visible(review_root, run)
    manifest_path = resolve_review_path(review_root) / "native_frames" / str(run).casefold() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows = manifest.get("frame_sources") or []
    if type(frame_index) is not int or not 0 <= frame_index < len(rows):
        raise ValueError("L1-neighbourhood display requires an exact native frame")
    row = rows[frame_index]
    if row.get("frame_index") != frame_index:
        raise ValueError("L1-neighbourhood native frame index binding differs")
    descriptor = row.get("l1_neighbourhood_gas")
    if descriptor is None:
        return None
    shape = tuple(map(int, (row.get("rig_domain_revision") or {}).get("phase_shape_hw", ())))
    source = row.get("source_sha256")
    if (not isinstance(descriptor, dict) or descriptor.get("contract") != CONTRACT
            or descriptor.get("frame_index") != frame_index
            or descriptor.get("source_sha256") != source
            or not isinstance(source, str) or len(source) != 64
            or tuple(descriptor.get("shape_hw", ())) != shape or len(shape) != 2
            or min(shape, default=0) <= 0):
        raise ValueError("L1-neighbourhood flag source, frame or canvas binding differs")
    path = Path(descriptor["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = resolve_review_path(path, context=manifest_path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != descriptor.get("sha256"):
        raise ValueError("L1-neighbourhood flag SHA256 differs")
    array = np.load(io.BytesIO(data), allow_pickle=False)
    if descriptor.get("storage") == "packbits-npy":
        if array.dtype != np.uint8 or array.ndim != 1 or array.size != (shape[0]*shape[1]+7)//8:
            raise ValueError("Packed L1-neighbourhood flag has invalid storage")
        array = np.unpackbits(array, count=shape[0]*shape[1]).reshape(shape).astype(bool)
    elif array.dtype != bool or array.shape != shape:
        raise ValueError("L1-neighbourhood flag must be bool on its recorded canvas")
    return array


def hatch_l1_neighbourhood_rgba(rgba, mask, *, enabled=True, phase_visible=True, period=12):
    """Vertical purple stripes mark added gas only on an already visible gas layer.

    The source overlay and its alpha remain unchanged. The pattern differs from
    the diagonal extrapolation and opposite-sloping teal floor-contact stripes.
    """
    image = np.asarray(rgba)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("L1-neighbourhood hatch requires uint8 RGBA display pixels")
    result = image.copy()
    if not enabled or not phase_visible or mask is None:
        return result
    flag = np.asarray(mask)
    if flag.dtype != bool or flag.shape != image.shape[:2] or type(period) is not int or period < 4:
        raise ValueError("L1-neighbourhood flags belong to another display grid")
    x = np.arange(image.shape[1], dtype=np.int32)[None]
    gas = (image[..., 0] > image[..., 1]) & (image[..., 2] > image[..., 1]) & (image[..., 3] > 0)
    stripe = flag & gas & ((x % period) < 2)
    rgb = result[stripe, :3].astype(np.uint16)
    result[stripe, :3] = ((rgb + np.asarray([80, 30, 100], np.uint16)) // 2).astype(np.uint8)
    return result


def hatch_l1_neighbourhood_overlay(image, mask=None, *, enabled=True, phase_visible=True):
    from PySide6.QtGui import QImage

    if not enabled or not phase_visible or mask is None:
        return image
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    array = np.frombuffer(converted.constBits(), np.uint8).reshape(converted.height(), converted.bytesPerLine())[:, :converted.width()*4].reshape(converted.height(), converted.width(), 4)
    rgba = hatch_l1_neighbourhood_rgba(array, mask, enabled=enabled, phase_visible=phase_visible)
    return QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
