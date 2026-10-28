"""Render-only separation of predicted phases and unresolved source support."""

import numpy as np
from PySide6.QtGui import QImage


UNRESOLVED_RGB = (30, 210, 255)


def display_phase_overlay(image: QImage, layer: str = "all", *, photo_sand: np.ndarray | None = None) -> QImage:
    """Return a display copy; the full saved/cache map remains scoring evidence.

    Cyan is a source-support diagnostic. For a verified photographic display
    contract it is visible only inside that frame's photo-sand domain. Older
    maps retain the explicit source-support layer. Context diagnostic layers
    keep their own rendering unchanged.
    """
    if image.isNull() or (layer.startswith("context_") and photo_sand is None):
        return image
    result = image.convertToFormat(QImage.Format.Format_RGBA8888).copy()
    rows = np.frombuffer(result.bits(), dtype=np.uint8).reshape(
        result.height(), result.bytesPerLine()
    )
    rgba = rows[:, :result.width() * 4].reshape(result.height(), result.width(), 4)
    unresolved = np.all(rgba[..., :3] == UNRESOLVED_RGB, axis=-1)
    if photo_sand is not None:
        if photo_sand.dtype != np.bool_ or photo_sand.shape != rgba.shape[:2]:
            raise ValueError("Photo-sand display domain disagrees with the overlay grid")
        rgba[..., 3][unresolved & ~photo_sand] = 0
        if layer == "source_support":
            rgba[..., 3][~unresolved] = 0
    else:
        # Historical maps have no verified photo-sand rendering contract.
        rgba[..., 3][~unresolved if layer == "source_support" else unresolved] = 0
    return result


def overlay_asset_layer(layer: str) -> str:
    """The support display reads the full stored map without creating a variant."""
    return "all" if layer == "source_support" else layer
