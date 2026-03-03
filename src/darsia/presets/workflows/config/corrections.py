"""Configuration for corrections."""

from dataclasses import dataclass
from pathlib import Path

from .utils import _get_section_from_toml


@dataclass
class CorrectionsConfig:
    """Configuration for image corrections.

    Attributes:
        type_conversion: Enable type conversion to float32 (default: True).
        resize: Enable resize correction to baseline shape (default: True).
        drift: Enable drift correction based on color checker (default: True).
        curvature: Enable curvature correction from laser grid (default: True).
        color: Enable color correction based on color checker (default: False).
        relative_color: Enable relative color correction based on color checker (default: False).
        illumination: Enable illumination correction based on color checker (default: False).

    """

    type: bool = True
    resize: bool = True
    drift: bool = True
    curvature: bool = True
    color: bool = False
    relative_color: bool = False
    illumination: bool = False

    def load(self, path: Path | list[Path]) -> "CorrectionsConfig":
        """Load correction configuration from TOML file.

        Args:
            path: Path to TOML config file
            results: Path to results folder

        Returns:
            self with loaded configuration
        """
        sec = _get_section_from_toml(path, "corrections")

        # Load individual correction settings
        self.type = sec.get("type", self.type)
        self.resize = sec.get("resize", self.resize)
        self.drift = sec.get("drift", self.drift)
        self.curvature = sec.get("curvature", self.curvature)
        self.color = sec.get("color", self.color)
        self.relative_color = sec.get("relative_color", self.relative_color)
        self.illumination = sec.get("illumination", self.illumination)

        return self
