"""Configuration for ROIs."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from darsia import CoordinateArray

from .utils import _get_key, _get_section, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class RoiConfig:
    """Configuration for a single ROI."""

    roi: CoordinateArray = field(default_factory=CoordinateArray)
    """ROI coordinates."""
    name: str = ""
    """Unique registry key for this ROI (e.g., 'full', 'calibration', 'storage')."""
    label: int | None = None
    """Optional label restriction: None means applies to all labels, int means restricted to
    this label."""

    def load(self, sec: dict) -> "RoiConfig":
        self.roi = CoordinateArray(
            [
                _get_key(sec, "corner_1", required=True, type_=list),
                _get_key(sec, "corner_2", required=True, type_=list),
            ]
        )

        self.name = _get_key(sec, "name", required=True, type_=str)
        self.label = _get_key(sec, "label", required=False, type_=int, default=None)
        return self


@dataclass
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


RoiAndLabelConfig = RoiConfig
"""Deprecated alias for RoiConfig. Kept for backward compatibility."""


@dataclass
class RoiAndSubroiConfig(RoiConfig):
    """Configuration for an ROI with a sub-ROI (inherits optional label from RoiConfig)."""

    subroi_config: RoiConfig = field(default_factory=RoiConfig)
    """Sub-ROI configuration."""

    def load(self, sec: dict) -> "RoiAndSubroiConfig":
        super().load(sec)
        subroi_sec = _get_section(sec, "subroi")
        self.subroi_config = RoiConfig().load(subroi_sec)
        return self
