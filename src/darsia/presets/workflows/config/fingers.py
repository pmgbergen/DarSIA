"""Configuration for fingers, based on segmentation analysis."""

import logging
from dataclasses import dataclass

from .roi import RoiConfig
from .utils import _get_key, _get_section

logger = logging.getLogger(__name__)


@dataclass
class FingersConfig:
    """Configuration for fingers."""

    mode: str | None = None
    """Type for segmentation."""
    threshold: float = 0.0
    """Threshold for segmentation."""
    roi: dict[str, RoiConfig] | None = None
    """ROIs for analysis."""

    def load(self, sec: dict) -> "FingersConfig":
        self.mode = _get_key(sec, "mode", required=True, type_=str)
        self.threshold = _get_key(sec, "threshold", required=True, type_=float)

        # Load ROIs
        try:
            roi_sec = _get_section(sec, "roi")
            self.roi = {}
            for key in roi_sec.keys():
                self.roi[key] = RoiConfig().load(_get_section(roi_sec, key))
        except KeyError:
            self.roi = {}

        return self

    def error(self):
        raise ValueError(f"Use [analysis.fingers] in the config file to load fingers.")
