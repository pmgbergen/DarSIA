"""Catalogue of curvature correction presets loaded from TOML array-of-tables."""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..corrections import CurvatureCorrectionConfig

logger = logging.getLogger(__name__)


@dataclass
class CurvatureCatalogue:
    """A catalogue of named curvature correction presets.

    Entries are loaded from a top-level [[curvature_preset]] TOML array-of-tables,
    each with a required 'name' field and optional 'description', plus nested
    [curvature_preset.init/crop/bulge/stretch] sub-sections using the same field
    names as an actual [corrections.curvature] section (so presets can be
    copy-pasted directly from known-good configs).
    """

    presets: dict[str, CurvatureCorrectionConfig] = field(
        default_factory=dict,
        metadata={
            "name": "Curvature presets",
            "help": "Named curvature correction presets",
        },
    )
    """Dict of named curvature correction configurations."""

    def load(self, path: Path | list[Path]) -> "CurvatureCatalogue":
        """Load all curvature preset entries from the top-level [[curvature_preset]]
        array-of-tables in TOML.

        Hand-parses TOML (like RoiRegistry) since array-of-tables is not supported
        by the generic _get_section_from_toml helper.

        Args:
            path: Path or list of Paths to TOML catalogue file(s).

        Returns:
            self

        Raises:
            ValueError: If [curvature_preset] section is not an array-of-tables
                (strict format enforcement) or if any preset entry has a duplicate
                name.
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
            with open(p, "rb") as f:
                data = tomllib.load(f)

            if "curvature_preset" not in data:
                continue

            preset_data = data["curvature_preset"]
            # Strict format: [[curvature_preset]] is a list, not nested dicts like
            # [curvature_preset.*]
            if not isinstance(preset_data, list):
                raise ValueError(
                    "The [curvature_preset] section must be an array-of-tables format "
                    "(use [[curvature_preset]]), not nested tables."
                )

            for entry in preset_data:
                name = entry.get("name")
                if name is None:
                    raise ValueError(
                        "Each [[curvature_preset]] entry must have a required 'name' field."
                    )
                name = str(name).strip()

                if name in seen_names:
                    raise ValueError(
                        f"Preset name '{name}' is duplicated. Preset names must be "
                        "globally unique."
                    )
                seen_names.add(name)

                self.presets[name] = CurvatureCorrectionConfig().load(entry)

        return self

    def names(self) -> list[str]:
        """Return all registered preset names, sorted alphabetically."""
        return sorted(self.presets.keys())

    def get(self, name: str) -> CurvatureCorrectionConfig:
        """Retrieve a preset by name.

        Args:
            name: The preset name.

        Returns:
            A copy-like reference to the CurvatureCorrectionConfig preset
            (the dict is already normalized via load()).

        Raises:
            KeyError: If the preset name is not found.
        """
        if name not in self.presets:
            raise KeyError(
                f"Preset '{name}' not found in catalogue. "
                f"Available presets: {self.names()}"
            )
        return self.presets[name]
