"""Catalogue of rig setup presets loaded from TOML array-of-tables."""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..rig import RigConfig

logger = logging.getLogger(__name__)


@dataclass
class RigCatalogue:
    """A catalogue of named rig setup presets.

    Entries are loaded from a top-level [[rig_preset]] TOML array-of-tables,
    each with a required 'name' field and required width/height/dim fields
    (matching [rig] section schema).
    """

    presets: dict[str, RigConfig] = field(
        default_factory=dict,
        metadata={
            "name": "Rig presets",
            "help": "Named rig setup presets",
        },
    )
    """Dict of named rig configurations."""

    def load(self, path: Path | list[Path] | str | list[str]) -> "RigCatalogue":
        """Load all rig preset entries from the top-level [[rig_preset]]
        array-of-tables in TOML.

        Args:
            path: Path, str, or list of Paths/strs to TOML catalogue file(s).

        Returns:
            self

        Raises:
            ValueError: If [rig_preset] section is not an array-of-tables
                or if any preset entry has a duplicate name or missing fields.
        """
        if isinstance(path, (str, Path)):
            paths = [Path(path)]
        else:
            paths = [Path(p) for p in path]
        self.presets = {}
        seen_names: set[str] = set()

        for p in paths:
            if not p.exists():
                continue
            with open(p, "rb") as f:
                data = tomllib.load(f)

            if "rig_preset" not in data:
                continue

            preset_data = data["rig_preset"]
            if not isinstance(preset_data, list):
                raise ValueError(
                    "The [rig_preset] section must be an array-of-tables format "
                    "(use [[rig_preset]]), not nested tables."
                )

            for idx, entry in enumerate(preset_data):
                name = entry.get("name")
                if name is None:
                    raise ValueError(
                        f"[[rig_preset]] entry {idx} must have a required 'name' field."
                    )
                name = str(name).strip()

                if name in seen_names:
                    raise ValueError(
                        f"Preset name '{name}' is duplicated. Preset names must be "
                        "globally unique."
                    )
                seen_names.add(name)

                spec = RigConfig()
                spec.width = float(
                    entry.get("width", 0)
                )  # required, no default in real use
                spec.height = float(entry.get("height", 0))
                spec.dim = int(entry.get("dim", 2))
                self.presets[name] = spec

        return self

    def names(self) -> list[str]:
        """Return all registered preset names, sorted alphabetically."""
        return sorted(self.presets.keys())

    def get(self, name: str) -> RigConfig:
        """Retrieve a preset by name.

        Args:
            name: The preset name.

        Returns:
            The RigConfig preset.

        Raises:
            KeyError: If the preset name is not found.
        """
        if name not in self.presets:
            raise KeyError(
                f"Preset '{name}' not found in catalogue. "
                f"Available presets: {self.names()}"
            )
        return self.presets[name]
