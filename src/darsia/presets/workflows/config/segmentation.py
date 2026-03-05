"""Configuration for segmentation."""

import logging
from dataclasses import dataclass, field

from .utils import _get_key

logger = logging.getLogger(__name__)


@dataclass
class SegmentationConfig:
    """Configuration for segmentation."""

    label: str | None = None
    """Label for segmentation."""
    mode: str | None = None
    """Type for segmentation."""
    thresholds: list[float] = field(default_factory=list)
    """List of thresholds."""
    color: list[int, int, int] = field(default_factory=list)
    """RGB color for contours."""
    alpha: list[float] = field(default_factory=list)
    """Alpha values for contours."""
    linewidth: int = 2
    """Line width for contour visualization."""

    def load(self, sec: dict) -> "SegmentationConfig":
        self.label = _get_key(sec, "label", required=True, type_=str)
        self.mode = _get_key(sec, "mode", required=True, type_=str)
        self.thresholds = _get_key(sec, "thresholds", required=True, type_=list)
        self.color = _get_key(sec, "color", required=True, type_=list)
        self.alpha = _get_key(sec, "alpha", required=False, type_=list)
        if not self.alpha:
            self.alpha = [1.0] * len(self.thresholds)
        self.linewidth = _get_key(
            sec, "linewidth", default=2, required=False, type_=int
        )
        return self

    def error(self):
        raise ValueError(
            f"Use [analysis.segmentation] in the config file to load segmentation."
        )
