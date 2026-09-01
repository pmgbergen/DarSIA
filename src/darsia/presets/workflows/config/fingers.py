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


@dataclass
class FingersConfig:
    """Configuration for fingers."""

    mode: str | None = None
    """Type for segmentation."""
    threshold: float = 0.0
    """Threshold for segmentation."""
    roi: list[str] | None = field(
        default=None,
        metadata={
            "name": "ROIs",
            "help": "ROI definitions for finger analysis.",
            "widget": "roi_key_list",
        },
    )
    """ROIs for analysis."""
    contour_smoother_selection: ContourSmootherSelection = field(
        default_factory=ContourSmootherSelection,
        metadata={
            "name": "Contour smoother",
            "help": "Contour smoothing algorithm.",
            "active_list_key": "active",
        },
    )
    """Contour smoother selection and options."""
    reduce_to_main_contour: bool = True
    """Whether to reduce to main contour (e.g. for mass mode)."""
    fill_holes: bool = False
    """Whether to fill holes in finger segmentation masks before contour extraction."""
    include_skeleton_analysis: bool = False
    """Whether to include skeleton analysis in the fingers workflow."""
    include_gradient_based_analysis: bool = False
    """Whether to include gradient-based analysis in the fingers workflow."""
    gradient_mode: str | None = None
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
        self.mode = _get_key(sec, "mode", required=True, type_=str)
        validate_mode_syntax(
            self.mode,
            color_embedding_registry,
            "analysis.fingers.mode",
        )
        self.threshold = _get_key(sec, "threshold", required=True, type_=float)

        # Load ROIs – support registry-key references as list[str].
        self.roi = _load_roi_key_list(
            sec,
            "roi",
            context="analysis.fingers.roi",
            roi_registry=roi_registry,
            none_if_absent=True,
        )

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
        raise ValueError(f"Use [analysis.fingers] in the config file to load fingers.")
