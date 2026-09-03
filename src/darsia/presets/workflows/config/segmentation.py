"""Configuration for segmentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import darsia
from darsia.presets.workflows.mode_resolution import validate_mode_syntax

from .contour_smoother import ContourSmootherSelection
from .utils import _get_key

if TYPE_CHECKING:
    from .color_embedding_registry import ColorEmbeddingRegistry

logger = logging.getLogger(__name__)


@dataclass
class SegmentationValueLabelsConfig:
    """Configuration for optional contour value labels."""

    active: bool = field(
        default=False,
        metadata={
            "name": "Activate value labels",
            "help": (
                "When enabled, threshold values are plotted along contours. "
                "Other settings below are preserved even when disabled."
            ),
            "section_active": True,
            "hidden": True,
        },
    )
    """Whether to enable value labels on contours."""
    value_color: tuple[int, int, int] = field(
        default=(0, 0, 0),
        metadata={
            "name": "Color",
            "help": "RGB color for value labels.",
            "placeholder": "255, 255, 255",
        },
    )
    """RGB color for contour value labels."""
    value_size: float = field(
        default=0.5,
        metadata={
            "name": "Font size",
            "help": "Font scale factor for value labels.",
            "placeholder": "0.5",
        },
    )
    """Font scale for contour value labels."""
    value_alpha: float = field(
        default=1.0,
        metadata={
            "name": "Opacity",
            "help": "Alpha value (opacity) for value labels [0.0-1.0].",
            "placeholder": "1.0",
        },
    )
    """Alpha value for contour value labels."""
    value_density: float = field(
        default=0.35,
        metadata={
            "name": "Label density",
            "help": "Density of labels along contours [0.0-1.0].",
            "placeholder": "0.35",
        },
    )
    """Label density along contours."""
    value_min_distance_px: float = field(
        default=40.0,
        metadata={
            "name": "Min distance (px)",
            "help": "Minimum pixel distance between adjacent labels.",
            "placeholder": "40.0",
        },
    )
    """Minimum pixel distance between two labels."""
    value_max_per_contour: int = field(
        default=3,
        metadata={
            "name": "Max labels per contour",
            "help": "Maximum number of labels per contour.",
            "placeholder": "3",
        },
    )
    """Maximum number of labels per contour."""
    value_format: str = field(
        default="{:.2f}",
        metadata={
            "name": "Format string",
            "help": "Python format string for threshold values (e.g., '{:.2f}').",
            "placeholder": "{:.2f}",
        },
    )
    """Format string used for threshold values."""

    @property
    def show_values(self) -> bool:
        """Shallow wrapper for `active` to keep GUI and analysis code in sync.

        TODO: Consolidate `active` and `show_values` into a single field after
        refactoring downstream consumers to use `active` directly.
        """
        return self.active

    def load(
        self, sec: dict, default_color: list[int]
    ) -> "SegmentationValueLabelsConfig":
        self.active = _get_key(sec, "active", default=False, required=False, type_=bool)
        self.value_color = _get_key(
            sec, "value_color", default=default_color, required=False, type_=tuple
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

    name: str | None = field(
        default=None,
        metadata={
            "name": "Entry name",
            "help": "Unique identifier/key for this segmentation configuration.",
            "placeholder": "e.g., aqueous, gas",
        },
    )
    """Name/key for this segmentation configuration."""
    label: str | None = field(
        default=None,
        metadata={
            "name": "Label",
            "help": "Display label for segmentation (e.g., 'CO2(aq)', 'CO2(g)').",
            "placeholder": "e.g., CO2(aq)",
        },
    )
    """Label for segmentation."""
    mode: str | None = field(
        default=None,
        metadata={
            "name": "Mode",
            "help": "Segmentation mode (e.g., 'concentration_aq', 'saturation_g').",
            "placeholder": "e.g., concentration_aq",
        },
    )
    """Type for segmentation."""
    thresholds: list[float] = field(
        default_factory=list,
        metadata={
            "name": "Thresholds",
            "help": "List of threshold values for contour generation.",
            "placeholder": "Threshold values, e.g., 0.05, 0.1, 0.5, 0.9",
        },
    )
    """List of thresholds."""
    color: tuple[int, int, int] = field(
        default=(0, 0, 0),
        metadata={
            "name": "Color",
            "help": "RGB color for contours [0-255, 0-255, 0-255].",
            "placeholder": "RGB color, e.g., 255, 113, 107",
        },
    )
    """RGB color for contours."""
    alpha: list[float] = field(
        default_factory=list,
        metadata={
            "name": "Alpha",
            "help": "Opacity values for each threshold [0.0-1.0].",
            "placeholder": "Opacity values, e.g., 0.6, 0.7, 0.9, 1.0",
        },
    )
    """Alpha values for contours."""
    linewidth: int = field(
        default=2,
        metadata={
            "name": "Line width",
            "help": "Contour line thickness in pixels.",
            "placeholder": "e.g., 8",
        },
    )
    """Line width for contour visualization."""
    values: SegmentationValueLabelsConfig = field(
        default_factory=SegmentationValueLabelsConfig,
        metadata={
            "name": "Show labels",
            "help": "Optional contour value label configuration.",
            "active_list_key": "active",
        },
    )
    """Contour value labels configuration."""
    contour_smoother_selection: ContourSmootherSelection = field(
        default_factory=ContourSmootherSelection,
        metadata={
            "name": "Contour smoother",
            "help": "Contour smoothing algorithm.",
            "active_list_key": "active",
        },
    )
    """Contour smoother selection and options."""

    @property
    def contour_smoother(self) -> darsia.ContourSmoother | None:
        """Backward-compatible property: returns the built contour smoother."""
        return self.contour_smoother_selection.build()

    def load(
        self, sec: dict, color_embedding_registry: ColorEmbeddingRegistry | None = None
    ) -> "SegmentationConfig":
        self.name = _get_key(sec, "name", required=False, default=None, type_=str)
        self.label = _get_key(sec, "label", required=True, type_=str)
        self.mode = _get_key(sec, "mode", required=True, type_=str)
        validate_mode_syntax(
            self.mode, color_embedding_registry, "analysis.segmentation.mode"
        )
        self.thresholds = _get_key(sec, "thresholds", required=True, type_=list)
        self.color = _get_key(sec, "color", required=True, type_=tuple)
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

        # Load contour smoother
        self.contour_smoother_selection = ContourSmootherSelection().load(sec)

        return self

    def error(self):
        raise ValueError(
            "Use [analysis.segmentation] in the config file to load segmentation."
        )
