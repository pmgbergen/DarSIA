"""Configuration for segmentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from darsia.presets.workflows.mode_resolution import validate_mode_syntax

from .utils import _get_key

if TYPE_CHECKING:
    from .color_embedding_registry import ColorEmbeddingRegistry

logger = logging.getLogger(__name__)


@dataclass
class SegmentationValueLabelsConfig:
    """Configuration for optional contour value labels."""

    show_values: bool = False
    """Whether to plot threshold values along contours."""
    value_color: list[int] = field(default_factory=list)
    """RGB color for contour value labels."""
    value_size: float = 0.5
    """Font scale for contour value labels."""
    value_alpha: float = 1.0
    """Alpha value for contour value labels."""
    value_density: float = 0.35
    """Label density along contours."""
    value_min_distance_px: float = 40.0
    """Minimum pixel distance between two labels."""
    value_max_per_contour: int = 3
    """Maximum number of labels per contour."""
    value_format: str = "{:.2f}"
    """Format string used for threshold values."""

    def load(
        self, sec: dict, default_color: list[int]
    ) -> "SegmentationValueLabelsConfig":
        self.show_values = _get_key(
            sec, "show_values", default=False, required=False, type_=bool
        )
        self.value_color = _get_key(
            sec, "value_color", default=default_color, required=False, type_=list
        )
        self.value_size = _get_key(
            sec, "value_size", default=0.5, required=False, type_=float
        )
        self.value_alpha = _get_key(
            sec, "value_alpha", default=1.0, required=False, type_=float
        )
        self.value_density = _get_key(
            sec, "value_density", default=0.35, required=False, type_=float
        )
        self.value_min_distance_px = _get_key(
            sec, "value_min_distance_px", default=40.0, required=False, type_=float
        )
        self.value_max_per_contour = _get_key(
            sec, "value_max_per_contour", default=3, required=False, type_=int
        )
        self.value_format = _get_key(
            sec, "value_format", default="{:.2f}", required=False, type_=str
        )
        return self


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
    values: SegmentationValueLabelsConfig = field(
        default_factory=SegmentationValueLabelsConfig
    )
    """Contour value labels configuration."""

    def load(
        self, sec: dict, color_embedding_registry: ColorEmbeddingRegistry | None = None
    ) -> "SegmentationConfig":
        self.label = _get_key(sec, "label", required=True, type_=str)
        self.mode = _get_key(sec, "mode", required=True, type_=str)
        if not validate_mode_syntax(
            self.mode, color_embedding_registry=color_embedding_registry
        ):
            raise ValueError(
                f"Unsupported analysis.segmentation.mode '{self.mode}'. Supported modes "
                "are legacy mass modes, rescaled modes, "
                "and 'color.<id>' (defined under [color.*.*])."
            )
        self.thresholds = _get_key(sec, "thresholds", required=True, type_=list)
        self.color = _get_key(sec, "color", required=True, type_=list)
        self.alpha = _get_key(sec, "alpha", required=False, type_=list)
        if not self.alpha:
            self.alpha = [1.0] * len(self.thresholds)
        self.linewidth = _get_key(
            sec, "linewidth", default=2, required=False, type_=int
        )
        # Optional nested section [analysis.segmentation.values] with precedence over
        # flat keys in [analysis.segmentation].
        values_sec = sec if not isinstance(sec.get("values"), dict) else sec["values"]
        self.values = SegmentationValueLabelsConfig().load(values_sec, self.color)
        return self

    def error(self):
        raise ValueError(
            f"Use [analysis.segmentation] in the config file to load segmentation."
        )
