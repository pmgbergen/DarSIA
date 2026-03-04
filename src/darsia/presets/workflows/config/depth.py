"""Depth configuration for the setup."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_key, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class DepthConfig:
    """Depth configuration for the setup."""

    measurements: Path = field(default_factory=Path)
    """Path to the csv file containing the depth measurements."""
    depth_map: Path = field(default_factory=Path)
    """Path to the depth map file."""

    def load(self, path: Path, results: Path | None = None) -> "DepthConfig":
        """Load depth config from a toml file from [section]."""
        sec = _get_section_from_toml(path, "depth")
        self.measurements = _get_key(sec, "measurements", required=True, type_=Path)
        self.depth_map = _get_key(sec, "depth_map", required=False, type_=Path)
        if not self.depth_map:
            assert results is not None
            self.depth_map = results / "rig" / "depth_map.npz"
        return self

    def error(self):
        raise ValueError("Use [depth] in the config file to load depth.")
