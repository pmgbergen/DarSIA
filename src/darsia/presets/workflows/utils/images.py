"""Utilities for loading images with optional caching in preset workflows."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

import darsia
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.utils.standard_images import roi_to_mask

logger = logging.getLogger(__name__)


def load_images_with_cache(
    rig: Any,
    paths: list[Path],
    use_cache: bool,
    cache_dir: Path | None,
) -> list[darsia.Image]:
    """Load a list of images, using an `.npz` disk cache when requested.

    For each path *p* in *paths*:

    * If *use_cache* is ``True`` and *cache_dir* is not ``None``:

      - If ``cache_dir / f"{p.stem}.npz"`` exists, the image is loaded from
        that file via :func:`darsia.imread`.
      - Otherwise the image is read through *rig* and immediately saved to
        the cache path.

    * If *use_cache* is ``False`` (or *cache_dir* is ``None``), the image is
      always read directly through *rig*.

    Args:
        rig: Object exposing a ``read_image(path: Path) -> darsia.Image`` method
            (e.g. a FluidFlower rig instance).
        paths: Ordered list of image paths to load.
        use_cache: Whether to use on-disk caching.
        cache_dir: Directory that holds (or will hold) the cached ``.npz`` files.
            Ignored when *use_cache* is ``False``.

    Returns:
        Ordered list of loaded :class:`darsia.Image` objects.

    """
    images: list[darsia.Image] = []
    for p in paths:
        if use_cache and cache_dir is not None:
            cache_path = cache_dir / f"{p.stem}.npz"
            if cache_path.exists():
                image = darsia.imread(cache_path)
            else:
                image = rig.read_image(p)
                image.save(cache_path)
        else:
            image = rig.read_image(p)
        images.append(image)
    return images


def get_calibration_mask(
    mask: darsia.Image,
    roi_entries: dict[str, RoiConfig] | None = None,
) -> darsia.Image:
    """Build the calibration mask, optionally restricted to a union of ROIs.

    Starting from *mask* (typically ``fluidflower.boolean_porosity``), the
    function optionally intersects it with the union of the bounding boxes
    defined by *roi_entries*.  If the intersection is empty (no overlap
    between the ROI union and the base mask), the original *mask* is returned
    unchanged and a warning is logged.

    Args:
        mask: Base boolean mask (a copy is used internally so the original is
            not modified).
        roi_entries: Optional dict mapping ROI names to :class:`RoiConfig`
            objects.  When ``None`` or empty, *mask* is returned as-is (copy).

    Returns:
        A boolean :class:`darsia.Image` representing the calibration mask.

    """
    calibration_mask = mask.copy()

    if roi_entries:
        union_mask = darsia.zeros_like(calibration_mask, dtype=np.bool_)
        for roi_cfg in roi_entries.values():
            roi_mask = roi_to_mask(roi_cfg.roi, calibration_mask, mode="voxels")
            union_mask.img |= roi_mask.img
        blended = calibration_mask.copy()
        blended.img &= union_mask.img
        if not np.any(blended.img):
            logger.warning(
                "The union of the provided ROIs does not overlap with the "
                "porosity mask. Falling back to the full porosity mask for "
                "colour-path calibration."
            )
        else:
            calibration_mask = blended

    return calibration_mask
