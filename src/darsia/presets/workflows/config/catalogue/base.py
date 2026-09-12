"""Shared machinery for TOML array-of-tables preset catalogues.

Every catalogue is the same shape: a bundled TOML file holding a top-level
``[[<array_key>]]`` array-of-tables, each entry carrying a unique ``name`` plus
whatever fields that preset kind needs. Subclasses declare the array key and how
to turn one entry into a config object; parsing, merging across files and
name-uniqueness are handled here.
"""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

CATALOGUE_DIR = Path(__file__).parent
"""Directory holding the bundled catalogue TOML files."""


def catalogue_path(filename: str) -> Path:
    """Return the path of a bundled catalogue file (e.g. ``"rig.toml"``)."""
    return CATALOGUE_DIR / filename


@dataclass
class ArrayOfTablesCatalogue:
    """A catalogue of named presets loaded from a TOML array-of-tables.

    Subclasses set :attr:`array_key` and implement :meth:`_build_entry`.
    """

    array_key: ClassVar[str] = ""
    """Top-level TOML array-of-tables key holding the entries."""

    presets: dict[str, Any] = field(default_factory=dict)
    """Dict of named presets, in insertion order."""

    def _build_entry(self, entry: dict) -> Any:
        """Turn one raw TOML entry into this catalogue's preset object."""
        raise NotImplementedError

    def load(
        self, path: Path | list[Path] | str | list[str]
    ) -> "ArrayOfTablesCatalogue":
        """Load every preset entry from one or more catalogue TOML files.

        Hand-parses TOML, since array-of-tables is not supported by the generic
        ``_get_section_from_toml`` helper. Missing files are skipped, so an
        optional user-supplied catalogue can be listed alongside the bundled one.

        Parameters
        ----------
        path : Path | list[Path] | str | list[str]
            Catalogue file(s) to read.

        Returns
        -------
            Self.

        Raises
        ------
        ValueError
            If the entries are not an array-of-tables, or an entry is missing
            ``name``, or a name is duplicated within or across files.
        """
        paths = (
            [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]
        )
        self.presets = {}

        for single_path in paths:
            if not single_path.exists():
                continue
            with open(single_path, "rb") as stream:
                data = tomllib.load(stream)

            if self.array_key not in data:
                continue

            entries = data[self.array_key]
            if not isinstance(entries, list):
                raise ValueError(
                    f"The [{self.array_key}] section must be an array-of-tables "
                    f"(use [[{self.array_key}]]), not nested tables."
                )

            for index, entry in enumerate(entries):
                name = entry.get("name")
                if name is None:
                    raise ValueError(
                        f"[[{self.array_key}]] entry {index} in {single_path} must "
                        "have a required 'name' field."
                    )
                name = str(name).strip()
                if name in self.presets:
                    raise ValueError(
                        f"Preset name '{name}' is duplicated. Preset names must be "
                        "globally unique."
                    )
                self.presets[name] = self._build_entry(entry)

        return self

    def names(self) -> list[str]:
        """Return all registered preset names, sorted alphabetically."""
        return sorted(self.presets.keys())

    def get(self, name: str) -> Any:
        """Retrieve a preset by name.

        Raises
        ------
        KeyError
            If the preset name is not found.
        """
        if name not in self.presets:
            raise KeyError(
                f"Preset '{name}' not found in catalogue. "
                f"Available presets: {self.names()}"
            )
        return self.presets[name]
