"""Catalogues of reusable presets for DarSIA workflow configurations."""

from .corrections import CurvatureCatalogue
from .formats import FormatCatalogue
from .rig import RigCatalogue
from .series import SeriesCatalogue

__all__ = ["CurvatureCatalogue", "FormatCatalogue", "RigCatalogue", "SeriesCatalogue"]
