"""Facies configuration for the setup."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_key, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class FaciesConfig:
    """Facies configuration for the setup."""

    id: list[int] = field(default_factory=list)
    """List of facies IDs."""
    props: Path = field(default_factory=Path)
    """Path to the facies properties file."""
    path: Path = field(default_factory=Path)
    """Path to the facies file."""
    groups: dict[int, str] = field(default_factory=dict)
    """Mapping from facies ID to facies."""

    def load(self, path: Path, results: Path | None = None) -> "FaciesConfig":
        """Load facies config from a toml file from [section]."""
        sec = _get_section_from_toml(path, "facies")
        self.id = _get_key(sec, "id", required=True, type_=list)
        self.props = _get_key(sec, "props", required=True, type_=Path)
        self.path = _get_key(sec, "path", required=False, type_=Path)
        self.id_label_map = {i: sec[str(i)]["labels"] for i in self.id}
        if not self.path:
            assert results is not None
            self.path = results / "setup" / "facies.npz"

        return self
