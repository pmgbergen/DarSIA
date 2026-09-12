"""Catalogues of reusable presets for DarSIA workflow configurations."""

from .base import ArrayOfTablesCatalogue, catalogue_path
from .corrections import CurvatureCatalogue
from .formats import FormatCatalogue
from .rig import RigCatalogue
from .series import PIECE_LABELS, SeriesCatalogue, SeriesPreset

CATALOGUE_FILES = {
    "curvature": (CurvatureCatalogue, "corrections.toml"),
    "format": (FormatCatalogue, "formats.toml"),
    "rig": (RigCatalogue, "rig.toml"),
    "series": (SeriesCatalogue, "series.toml"),
}
"""Known catalogue kinds: kind -> (catalogue class, bundled TOML filename)."""


def load_catalogue(kind: str) -> ArrayOfTablesCatalogue:
    """Load a bundled catalogue by kind (e.g. ``"rig"``, ``"curvature"``).

    Single place that knows where the bundled catalogue files live, so callers
    never hardcode paths.

    Raises
    ------
    KeyError
        If the kind is unknown.
    """
    if kind not in CATALOGUE_FILES:
        raise KeyError(
            f"Unknown catalogue kind '{kind}'. Known kinds: {sorted(CATALOGUE_FILES)}"
        )
    catalogue_class, filename = CATALOGUE_FILES[kind]
    return catalogue_class().load(catalogue_path(filename))


__all__ = [
    "ArrayOfTablesCatalogue",
    "CATALOGUE_FILES",
    "CurvatureCatalogue",
    "FormatCatalogue",
    "PIECE_LABELS",
    "RigCatalogue",
    "SeriesCatalogue",
    "SeriesPreset",
    "catalogue_path",
    "load_catalogue",
]
