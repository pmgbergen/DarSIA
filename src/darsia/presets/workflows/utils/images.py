"""Utilities for loading images with optional caching in preset workflows."""

import logging
from os.path import commonpath
from pathlib import Path

import darsia
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def load_images_with_cache(
    rig: Rig,
    paths: list[Path],
    use_cache: bool,
    cache_dir: Path | None,
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

    Args:
        rig: Object exposing a ``read_image(path: Path) -> darsia.Image`` method
            (e.g. Rig).
        paths: Ordered list of image paths to load.
        use_cache: Whether to use on-disk caching.
        cache_dir: Directory that holds (or will hold) the cached ``.npz`` files.
            Ignored when *use_cache* is ``False``.

    Returns:
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
        else:
            image = rig.read_image(p)
        images.append(image)
    return images
