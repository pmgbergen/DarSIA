"""Catalogue of image export format presets loaded from TOML array-of-tables."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from ..format_registry import FormatRegistry, ImageExportFormat
from .base import ArrayOfTablesCatalogue

logger = logging.getLogger(__name__)


@dataclass
class FormatCatalogue(ArrayOfTablesCatalogue):
    """A catalogue of named export format presets.

    Entries are loaded from a top-level [[format]] TOML array-of-tables, each with
    required 'type' and 'name' fields plus optional format-specific fields
    (resolution, dpi, cmap, ...). The schema is identical to FormatRegistry's, so
    presets can be copy-pasted directly from known-good configs.
    """

    array_key: ClassVar[str] = "format"

    def load(self, path: Path | list[Path] | str | list[str]) -> "FormatCatalogue":
        """Load all format presets, reusing FormatRegistry's own entry parsing.

        Overrides the generic loader because FormatRegistry validates and
        normalizes a whole file at once rather than entry by entry.
        """
        paths = (
            [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]
        )
        self.presets = {}

        for single_path in paths:
            if not single_path.exists():
                continue
            registry = FormatRegistry()
            registry.load(single_path)
            for name, spec in registry.formats.items():
                if name in self.presets:
                    raise ValueError(
                        f"Format name '{name}' is duplicated across catalogue files. "
                        "Format names must be globally unique."
                    )
                self.presets[name] = spec

        return self

    def get(self, name: str) -> ImageExportFormat:
        """Retrieve a format preset by name."""
        return super().get(name)
