"""Colormaps of the format registry: options and resolution of `cmap` names.

Supported names:
- ``matplotlib.<name>``: any matplotlib colormap; a curated subset is offered as options.
- ``color_path.<embedding_id>[.<label>]``: the colors of a calibrated color path
  (default label: the embedding's reference label).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import darsia

logger = logging.getLogger(__name__)

CURATED_MATPLOTLIB_CMAPS = (
    # Sequential: non-negative quantities (mass, concentration, saturation)
    "viridis",
    "cividis",
    "plasma",
    "inferno",
    "magma",
    "turbo",
    "gray",
    "gray_r",
    # Diverging: signed quantities with a meaningful zero (e.g. differences)
    "coolwarm",
    "seismic",
    "berlin",
    "managua",
)
"""Matplotlib colormaps offered as options (any other matplotlib name still works)."""


def _color_path_embeddings(color_registry: Any) -> dict[str, Any]:
    """Colour-path embeddings of a color registry, by id."""
    from darsia.signals.color import ColorPathEmbedding

    embeddings = {} if color_registry is None else color_registry.resolve_all()
    return {
        name: emb
        for name, emb in embeddings.items()
        if isinstance(emb, ColorPathEmbedding)
    }


def matplotlib_cmap_options() -> list[str]:
    """Curated matplotlib colormaps available in the installed matplotlib."""
    import matplotlib

    return [
        f"matplotlib.{name}"
        for name in CURATED_MATPLOTLIB_CMAPS
        if name in matplotlib.colormaps
    ]


def color_path_cmap_options(color_registry: Any) -> list[str]:
    """Colormaps of calibrated color paths: color_path.<id> and color_path.<id>.<label>."""
    options: list[str] = []
    for key, embedding in _color_path_embeddings(color_registry).items():
        csv = Path(embedding.color_paths_csv_file)
        if not csv.exists():
            continue
        options.append(f"color_path.{key}")
        labels = darsia.LabelColorPathMap.load_csv(csv)
        options.extend(f"color_path.{key}.{label}" for label in sorted(labels))
    return options


def cmap_options(color_registry: Any = None) -> list[str]:
    """All offered colormap names: curated matplotlib maps, then calibrated color paths."""
    return matplotlib_cmap_options() + color_path_cmap_options(color_registry)


def _load_color_path_cmap(cmap_name: str, color_registry: Any):
    """Colormap of a calibrated colour path: color_path.<embedding_id>[.<label>]."""
    parts = cmap_name.split(".")
    if len(parts) not in (2, 3):
        raise ValueError("expected color_path.<embedding_id>[.<label>]")
    key = parts[1]
    label = int(parts[2]) if len(parts) == 3 else None

    paths = _color_path_embeddings(color_registry)
    if key not in paths:
        raise KeyError(
            f"no colour-path embedding '{key}' (available: {sorted(paths) or 'none'})."
        )
    embedding = paths[key]
    label = embedding.reference_label if label is None else label
    csv = Path(embedding.color_paths_csv_file)
    if not csv.exists():
        raise FileNotFoundError(f"colour paths not calibrated yet ({csv}).")
    color_paths = darsia.LabelColorPathMap.load_csv(csv)
    if label not in color_paths:
        raise KeyError(
            f"no colour path for label {label} (labels: {sorted(color_paths)})."
        )
    return color_paths[label].get_color_map(name=cmap_name)


def resolve_cmap(cmap_name: str | None, color_registry: Any = None):
    """Colormap of a cmap name, or None (use the default) with a warning on failure.

    Parameters
    ----------
    cmap_name : str | None
        ``matplotlib.<name>`` or ``color_path.<embedding_id>[.<label>]``.
    color_registry : ColorEmbeddingRegistry | None
        Registry holding the color-path embeddings (``config.color``).

    Returns
    -------
        A matplotlib colormap, or None.
    """
    if cmap_name is None:
        return None
    try:
        if cmap_name.startswith("matplotlib."):
            import matplotlib

            return matplotlib.colormaps[cmap_name.split(".", 1)[1]]
        if cmap_name.startswith("color_path."):
            return _load_color_path_cmap(cmap_name, color_registry)
        raise ValueError("use matplotlib.<name> or color_path.<embedding_id>[.<label>]")
    except (ValueError, KeyError, FileNotFoundError) as error:
        logger.warning(
            "Colormap '%s' not available, using the default: %s", cmap_name, error
        )
        return None
