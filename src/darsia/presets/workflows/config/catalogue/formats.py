"""Catalogue of image export format presets loaded from TOML array-of-tables."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..format_registry import FormatRegistry, ImageExportFormat

logger = logging.getLogger(__name__)


@dataclass
class FormatCatalogue:
    """A catalogue of named export format presets.

    Entries are loaded from a top-level [[format]] TOML array-of-tables,
    each with required 'type' and 'name' fields, plus optional format-specific
    fields (resolution, dpi, cmap, etc.). The schema is identical to
    FormatRegistry.load() — presets can be copy-pasted directly from known-good configs.
    """

    presets: dict[str, ImageExportFormat] = field(
        default_factory=dict,
        metadata={
            "name": "Format presets",
            "help": "Named export format presets",
        },
    )
    """Dict of named export format specifications."""

    def load(self, path: Path | list[Path]) -> "FormatCatalogue":
        """Load all format preset entries from the top-level [[format]]
        array-of-tables in TOML.

        Uses FormatRegistry's own parsing logic to avoid code duplication;
        merges across multiple files and enforces global name uniqueness.

        Args:
            path: Path or list of Paths to TOML catalogue file(s).

        Returns:
            self

        Raises:
            ValueError: If any format entry has a duplicate name (within or across files).
        """
        if isinstance(path, list):
            paths = [Path(p) for p in path]
        else:
            paths = [Path(path)]
        self.presets = {}
        seen_names: set[str] = set()

        for p in paths:
            if not p.exists():
                continue
            # Use FormatRegistry's hand-parsed load; it raises on invalid entries
            registry = FormatRegistry()
            registry.load(p)

            # Merge into self.presets, checking for duplicates across files
            for name, spec in registry.formats.items():
                if name in seen_names:
                    raise ValueError(
                        f"Format name '{name}' is duplicated across catalogue files. "
                        "Format names must be globally unique."
                    )
                seen_names.add(name)
                self.presets[name] = spec

        return self

    def names(self) -> list[str]:
        """Return all registered preset names, sorted alphabetically."""
        return sorted(self.presets.keys())

    def get(self, name: str) -> ImageExportFormat:
        """Retrieve a preset by name.

        Args:
            name: The preset name.

        Returns:
            The ImageExportFormat preset (already normalized via load()).

        Raises:
            KeyError: If the preset name is not found.
        """
        if name not in self.presets:
            raise KeyError(
                f"Preset '{name}' not found in catalogue. "
                f"Available presets: {self.names()}"
            )
        return self.presets[name]
