"""Configuration for fingers, based on segmentation analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import darsia
from darsia.presets.workflows.mode_resolution import validate_mode_syntax

from .contour_smoother import ContourSmootherSelection
from .roi_registry import _load_roi_key_list
from .utils import _get_key

if TYPE_CHECKING:
    from .color_embedding_registry import ColorEmbeddingRegistry
    from .roi_registry import RoiRegistry

logger = logging.getLogger(__name__)

SUPPORTED_ANALYSIS_FINGER_MODES = {
    "mass",
    "rescaled_mass",
    "extensive_mass",
    "extensive_rescaled_mass",
    "saturation_g",
    "rescaled_saturation_g",
    "concentration_aq",
    "rescaled_concentration_aq",
}


@dataclass
class FingersConfig:
    """Configuration for fingers."""

    name: str | None = field(
        default=None,
        metadata={
            "name": "Entry name",
            "help": "Unique identifier/key for this fingers configuration.",
            "placeholder": "e.g., primary, secondary",
        },
    )
    """Name/key for this fingers configuration."""
    mode: str | None = field(
        default=None,
        metadata={
            "name": "Mode",
            "help": "Analysis mode (e.g., 'mass', 'concentration_aq').",
            "placeholder": "e.g., mass",
            "options": list(SUPPORTED_ANALYSIS_FINGER_MODES),
            "group": "Contour",
        },
    )
    """Type for segmentation."""
    roi: str | None = field(
        default=None,
        metadata={
            "name": "ROI",
            "help": "ROI definition for finger analysis.",
            "widget": "roi_key_list",
            "max_rows": 1,
            "group": "Contour",
        },
    )
    """ROI for analysis."""
    threshold: float = field(
        default=0.0,
        metadata={
            "name": "Threshold",
            "help": "Threshold value for finger detection.",
            "placeholder": "0.0",
            "group": "Contour",
        },
    )
    """Threshold for segmentation."""
    contour_smoother_selection: ContourSmootherSelection = field(
        default_factory=ContourSmootherSelection,
        metadata={
            "name": "Contour smoother",
            "help": "Contour smoothing algorithm.",
            "active_list_key": "active",
        },
    )
    """Contour smoother selection and options."""
    reduce_to_main_contour: bool = field(
        default=True,
        metadata={
            "name": "Reduce to main contour",
            "help": "Whether to keep only the main contour (e.g., for mass mode).",
            "group": "Contour processing",
        },
    )
    """Whether to reduce to main contour (e.g. for mass mode)."""
    fill_holes: bool = field(
        default=False,
        metadata={
            "name": "Fill holes",
            "help": "Whether to fill holes in finger segmentation masks.",
            "group": "Contour processing",
        },
    )
    """Whether to fill holes in finger segmentation masks before contour extraction."""
    include_skeleton_analysis: bool = field(
        default=False,
        metadata={
            "name": "Include skeleton analysis",
            "help": "Whether to include skeleton analysis in the workflow.",
            "group": "Extra analysis",
        },
    )
    """Whether to include skeleton analysis in the fingers workflow."""
    save_result_plots: bool = field(
        default=True,
        metadata={
            "name": "Save result plots",
            "help": (
                "Whether to render and save tips/fjords/skeleton/path-evolution "
                "overlay PNGs to disk. Disable to speed up fingers analysis when "
                "these diagnostic images aren't needed."
            ),
            "group": "Extra analysis",
        },
    )
    """Whether to render and save fingers-analysis result overlay PNGs."""
    include_gradient_based_analysis: bool = field(
        default=False,
        metadata={
            "name": "Include gradient analysis",
            "help": (
                "Whether to include gradient-based analysis (requires gradient_mode)."
            ),
            "group": "Extra analysis",
        },
    )
    """Whether to include gradient-based analysis in the fingers workflow."""
    gradient_mode: str | None = field(
        default=None,
        metadata={
            "name": "Gradient mode",
            "help": "Mode for gradient-based analysis, if included.",
            "options": list(SUPPORTED_ANALYSIS_FINGER_MODES),
            "depends_on": {
                "field": "include_gradient_based_analysis",
                "value": True,
            },
            "group": "Extra analysis",
        },
    )
    """Mode for gradient-based analysis, if included."""

    @property
    def contour_smoother(self) -> darsia.ContourSmoother | None:
        """Backward-compatible property: returns the built contour smoother."""
        return self.contour_smoother_selection.build()

    def load(
        self,
        sec: dict,
        roi_registry: RoiRegistry | None = None,
        color_embedding_registry: ColorEmbeddingRegistry | None = None,
    ) -> "FingersConfig":
        self.name = _get_key(sec, "name", required=False, default=None, type_=str)
        self.mode = _get_key(sec, "mode", required=True, type_=str)
        validate_mode_syntax(
            self.mode,
            color_embedding_registry,
            "analysis.fingers.mode",
        )
        self.threshold = _get_key(sec, "threshold", required=True, type_=float)

        # Load ROI – single selection (max_rows=1 in GUI)
        roi_list = _load_roi_key_list(
            sec,
            "roi",
            context="analysis.fingers.roi",
            roi_registry=roi_registry,
            allow_str=True,
            none_if_absent=True,
        )
        # Extract single ROI name if list provided, or use directly if string
        self.roi = roi_list[0] if isinstance(roi_list, list) and roi_list else roi_list

        # Load contour smoother
        self.contour_smoother_selection = ContourSmootherSelection().load(sec)

        # Load reduce_to_main_contour
        self.reduce_to_main_contour = _get_key(
            sec, "reduce_to_main_contour", required=False, default=True, type_=bool
        )

        self.fill_holes = _get_key(
            sec, "fill_holes", required=False, default=self.fill_holes, type_=bool
        )

        self.include_skeleton_analysis = _get_key(
            sec,
            "include_skeleton_analysis",
            required=False,
            default=self.include_skeleton_analysis,
            type_=bool,
        )

        self.save_result_plots = _get_key(
            sec,
            "save_result_plots",
            required=False,
            default=self.save_result_plots,
            type_=bool,
        )

        self.include_gradient_based_analysis = _get_key(
            sec,
            "include_gradient_based_analysis",
            required=False,
            default=self.include_gradient_based_analysis,
            type_=bool,
        )

        if self.include_gradient_based_analysis:
            self.gradient_mode = _get_key(
                sec,
                "gradient_mode",
                required=True,
                type_=str,
            )
            validate_mode_syntax(
                self.gradient_mode,
                color_embedding_registry,
                "analysis.fingers.gradient_mode",
            )

        return self

    def error(self):
        raise ValueError("Use [analysis.fingers] in the config file to load fingers.")
