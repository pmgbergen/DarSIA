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

    name: str = field(
        default="",
        metadata={
            "name": "Name",
            "help": "Unique registry key for this ROI.",
        },
    )
    corner_1: list[float] = field(
        default_factory=list,
        metadata={
            "name": "Corner 1",
            "help": "First corner coordinate as [x, y].",
            "placeholder": "[x, y]",
        },
    )
    corner_2: list[float] = field(
        default_factory=list,
        metadata={
            "name": "Corner 2",
            "help": "Second corner coordinate as [x, y].",
            "placeholder": "[x, y]",
        },
    )
    label: int | None = field(
        default=None,
        metadata={
            "name": "Label",
            "help": "Optional label restriction. Leave blank to apply to all labels.",
        },
    )

    @property
    def roi(self) -> CoordinateArray:
        """Computed ROI coordinates from corner_1 and corner_2."""
        return CoordinateArray([self.corner_1, self.corner_2])

    def load(self, sec: dict) -> "RoiConfig":
        self.name = _get_key(sec, "name", required=True, type_=str)
        self.corner_1 = _get_key(sec, "corner_1", required=True, type_=list)
        self.corner_2 = _get_key(sec, "corner_2", required=True, type_=list)
        self.label = _get_key(sec, "label", required=False, type_=int, default=None)
        return self


@dataclass
class MultiRoiConfig:
    """Configuration for multiple ROIs."""

    roi: dict[str, RoiConfig] = field(default_factory=dict)
    """Dictionary of ROI configurations."""

    def load(self, path: Path) -> "MultiRoiConfig":
        raise NotImplementedError("TBC. Does not use the ROI style.")
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
