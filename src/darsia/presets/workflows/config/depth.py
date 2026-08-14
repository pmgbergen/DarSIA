"""Depth configuration for the setup."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_key, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class DepthConfig:
    """Depth configuration for the setup."""

    measurements: Path = field(
        default_factory=Path,
        metadata={
            "name": "Measurements",
            "help": "Path to the csv file containing the depth measurements.",
        },
    )
    """Path to the csv file containing the depth measurements."""
    depth_map: Path | None = field(
        default=None,
        metadata={
            "name": "Depth map",
            "help": (
                "Path to the depth map file. If left empty, computed automatically "
                "as <results>/setup/depth/depth_map.npz."
            ),
            "hidden": True,
        },
    )
    """Path to the depth map file. Computed under `results` if not given."""
    target_resolution: tuple[int, int] = field(
        default=(500, 1000),
        metadata={
            "name": "Interpolation resolution",
            "help": (
                "Target pixel resolution (height, width) of the grid used when "
                "interpolating the depth measurements onto the rig."
            ),
        },
    )
    """Target grid resolution (height, width) for interpolating depth measurements."""

    def load(self, path: Path, results: Path | None = None) -> "DepthConfig":
        """Load depth config from a toml file from [section]."""
        sec = _get_section_from_toml(path, "depth")
        self.measurements = _get_key(sec, "measurements", required=True, type_=Path)
        default_depth_map = (
            results / "setup" / "depth" / "depth_map.npz" if results else None
        )
        self.depth_map = _get_key(
            sec, "depth_map", default=default_depth_map, required=False, type_=Path
        )
        assert self.depth_map is not None, "results is required if depth_map is not set"
        self.target_resolution = _get_key(
            sec, "resolution", default=(500, 1000), required=False, type_=tuple
        )
        return self

    def error(self):
        raise ValueError("Use [depth] in the config file to load depth.")
