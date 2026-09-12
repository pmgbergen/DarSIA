"""Catalogue of rig setup presets loaded from TOML array-of-tables."""

import logging
from dataclasses import dataclass
from typing import ClassVar

from ..rig import RigConfig
from .base import ArrayOfTablesCatalogue

logger = logging.getLogger(__name__)


@dataclass
class RigCatalogue(ArrayOfTablesCatalogue):
    """A catalogue of named rig setup presets.

    Entries are loaded from a top-level [[rig_preset]] TOML array-of-tables,
    each with a required 'name' field and width/height/dim fields (matching the
    [rig] section schema).
    """

    array_key: ClassVar[str] = "rig_preset"

    def _build_entry(self, entry: dict) -> RigConfig:
        spec = RigConfig()
        spec.width = float(entry.get("width", 0))
        spec.height = float(entry.get("height", 0))
        spec.dim = int(entry.get("dim", 2))
        return spec

    def get(self, name: str) -> RigConfig:
        """Retrieve a rig preset by name."""
        return super().get(name)
