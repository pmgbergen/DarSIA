"""Utilities for loading images with optional caching in preset workflows."""

from __future__ import annotations

import inspect
import logging
from os.path import commonpath
from pathlib import Path
from typing import TYPE_CHECKING

import darsia

if TYPE_CHECKING:
    from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def read_native(rig: Rig, path: Path) -> darsia.Image:
    """Image read by `rig` in the type of the corrections (`Rig.read_image`, native).

    The 8-bit image of a photo takes an eighth of the memory of floats. Users of the
    image convert it where they need floats (`Image.as_float`). Rigs whose `read_image`
    has no `native` option read as usual.

    Parameters
    ----------
    rig : Rig
        Object with a `read_image(path)` method.
    path : Path
        Path to the image.

    Returns
    -------
    darsia.Image
        The corrected image.
    """
    if "native" in inspect.signature(rig.read_image).parameters:
        return rig.read_image(path, native=True)
    return rig.read_image(path)


def load_images_with_cache(
    rig: Rig,
    paths: list[Path],
    use_cache: bool,
    cache_dir: Path | None,
    native: bool = False,
) -> list[darsia.Image]:
    """Load a list of images, using an `.npz` disk cache when requested.

    For each path *p* in *paths*:

    * If *use_cache* is ``True`` and *cache_dir* is not ``None``:

      - If the cache file corresponding to the image path exists (preserving
        relative subfolder structure under ``cache_dir``), the image is loaded
        from that file via :func:`darsia.imread`.
      - Otherwise the image is read through *rig* and immediately saved to
        the cache path.

    * If *use_cache* is ``False`` (or *cache_dir* is ``None``), the image is
      always read directly through *rig*.

    Parameters
    ----------
    rig : Rig
        Object exposing a ``read_image(path: Path) -> darsia.Image`` method
        (e.g. Rig).
    paths : list[Path]
        Ordered list of image paths to load.
    use_cache : bool
        Whether to use on-disk caching.
    cache_dir : Path | None
        Directory that holds (or will hold) the cached ``.npz`` files.
        Ignored when *use_cache* is ``False``.
    native : bool
        Keep the data type resulting from the corrections instead of converting to
        float (see ``Rig.read_image``), to hold many images with less memory. Only
        without caching: cached images are float, as other workflows expect them.

    Returns
    -------
        Ordered list of loaded :class:`darsia.Image` objects.
    """
    images: list[darsia.Image] = []
    resolved_paths = [path.resolve() for path in paths]
    if resolved_paths:
        try:
            common_root = Path(commonpath([str(path) for path in resolved_paths]))
        except ValueError:
            common_root = None
    else:
        common_root = None
    for p in paths:
        if use_cache and cache_dir is not None:
            resolved = p.resolve()
            if common_root is not None:
                try:
                    relative = resolved.relative_to(common_root)
                except ValueError:
                    relative = Path(resolved.name)
            else:
                relative = Path(resolved.name)
            cache_path = cache_dir / relative.with_suffix(".npz")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                image = darsia.imread(cache_path)
            else:
                image = rig.read_image(p)
                image.save(cache_path)
        elif native:
            image = rig.read_image(p, native=True)
        else:
            image = rig.read_image(p)
        images.append(image)
    return images
