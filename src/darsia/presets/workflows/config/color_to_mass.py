"""Configuration for color to mass calibration"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from warnings import warn

logger = logging.getLogger(__name__)

from .utils import (
    _get_key,
    _get_section_from_toml,
)
from .time_data import TimeData


@dataclass
class ColorToMassConfig:
    """Configuration for color to mass calibration"""

    mode: str = "manual"
    """Calibration mode (e.g., 'manual', 'automatic')."""
    fluid: str | None = "co2"
    """Fluid type for mass analysis (e.g., 'tracer', 'co2')."""
    data: TimeData | None = None
    """Calibration data configuration."""
    calibration_folder: Path = field(default_factory=Path)
    """Path to the calibration folder."""

    def load(
        self, path: Path, data: Path | None, results: Path | None = None
    ) -> "ColorToMassConfig":
        """Load color to mass config from a toml file from [section].

        Args:
            path: Path to the TOML file.
            data: Path to the data folder.
            results: Path to the results folder.

        """
        # Get section
        sec = _get_section_from_toml(path, "color_to_mass")

        # Mode and fluid
        self.mode = _get_key(sec, "mode", default="manual", required=False, type_=str)
        self.fluid = _get_key(sec, "fluid", default="co2", required=False, type_=str)

        # Calibration data
        try:
            self.data = TimeData().load(sec["data"], data)
        except KeyError:
            warn("No data found. Use [color_to_mass.data].")
            self.data = None

        # Where to store the calibration results
        self.calibration_folder = _get_key(
            sec, "calibration_folder", required=False, type_=Path
        )
        if not self.calibration_folder:
            assert results is not None
            self.calibration_folder = results / "calibration" / "color_to_mass"

        return self
