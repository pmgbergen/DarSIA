"""Configuration for ROIs."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from darsia import CoordinateArray

logger = logging.getLogger(__name__)

from .utils import _get_key, _get_section, _get_section_from_toml


@dataclass
class RoiConfig:
    """Configuration for a single ROI."""

    roi: CoordinateArray = field(default_factory=CoordinateArray)
    """ROI coordinates."""
    name: str = ""
    """Name of the ROI."""

    def load(self, sec: dict) -> "RoiConfig":
        self.roi = CoordinateArray(
            [
                _get_key(sec, "corner_1", required=True, type_=list),
                _get_key(sec, "corner_2", required=True, type_=list),
            ]
        )

        name = _get_key(sec, "name", required=True, type_=str)
        self.name = name
        return self


class MultiRoiConfig:
    """Configuration for multiple ROIs."""

    roi: dict[str, RoiConfig] = field(default_factory=dict)
    """Dictionary of ROI configurations."""

    def load(self, path: Path) -> "MultiRoiConfig":
        # Load the entire TOML data to access events section
        roi_sec = _get_section_from_toml(path, "roi")
        self.roi = {}
        for key in roi_sec.keys():
            self.roi[key] = RoiConfig().load(_get_section(roi_sec, key))
        return self


@dataclass
class RoiAndLabelConfig:
    """Configuration for an ROI with an associated label."""

    roi: CoordinateArray = field(default_factory=CoordinateArray)
    """ROI coordinates."""
    name: str = ""
    """Name of the ROI."""
    label: int = -1
    """Label associated with the ROI."""

    def load(self, sec: dict) -> "RoiAndLabelConfig":
        self.roi = CoordinateArray(
            [
                _get_key(sec, "corner_1", required=True, type_=list),
                _get_key(sec, "corner_2", required=True, type_=list),
            ]
        )

        self.name = _get_key(sec, "name", required=True, type_=str)
        self.label = _get_key(sec, "label", required=True, type_=int)
        return self


@dataclass
class RoiAndSubroiConfig(RoiConfig):
    """Configuration for an ROI with a sub-ROI."""

    subroi_config: RoiConfig = field(default_factory=RoiConfig)
    """Sub-ROI configuration."""

    def load(self, sec: dict) -> "RoiAndSubroiConfig":
        super().load(sec)
        subroi_sec = _get_section(sec, "subroi")
        self.subroi_config = RoiConfig().load(subroi_sec)
        return self
