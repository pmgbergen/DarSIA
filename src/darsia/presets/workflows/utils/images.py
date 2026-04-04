"""Utilities for loading images with optional caching in preset workflows."""

import logging
from pathlib import Path
from typing import Any

import darsia

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



