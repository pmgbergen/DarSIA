"""Catalogue of curvature correction presets loaded from TOML array-of-tables."""

import logging
from dataclasses import dataclass
from typing import ClassVar

from ..corrections import CurvatureCorrectionConfig
from .base import ArrayOfTablesCatalogue

logger = logging.getLogger(__name__)


@dataclass
class CurvatureCatalogue(ArrayOfTablesCatalogue):
    """A catalogue of named curvature correction presets.

    Entries are loaded from a top-level [[curvature_preset]] TOML array-of-tables,
    each with a required 'name' field and optional 'description', plus nested
    [curvature_preset.init/crop/bulge/stretch] sub-sections using the same field
    names as an actual [corrections.curvature] section (so presets can be
    copy-pasted directly from known-good configs).
    """

    array_key: ClassVar[str] = "curvature_preset"

    def _build_entry(self, entry: dict) -> CurvatureCorrectionConfig:
        return CurvatureCorrectionConfig().load(entry)

    def get(self, name: str) -> CurvatureCorrectionConfig:
        """Retrieve a curvature preset by name."""
        return super().get(name)
