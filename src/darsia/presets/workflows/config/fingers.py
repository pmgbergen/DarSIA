"""Configuration for fingers, based on segmentation analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import darsia
from darsia.presets.workflows.mode_resolution import validate_mode_syntax

from .contour_smoother import SavitzkyGolaySmootherConfig
from .roi import RoiConfig
from .utils import _get_key, _get_section

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
    roi: dict[str, RoiConfig] | None = None
    """ROIs for analysis."""
    contour_smoother: darsia.ContourSmoother | None = None
    """Optional contour smoother for finger contours."""
    reduce_to_main_contour: bool = True
    """Whether to reduce to main contour (e.g. for mass mode)."""

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

        # Load ROIs – support both registry-key references and inline definitions.
        roi_raw = sec.get("roi")
        if isinstance(roi_raw, list) and roi_registry is not None:
            # New format: roi = ["key1", "key2"] resolved via registry
            self.roi = roi_registry.resolve_rois(roi_raw)
        elif isinstance(roi_raw, dict):
            # Old inline format: [analysis.fingers.roi.*] sub-sections
            self.roi = {}
            for key in roi_raw.keys():
                self.roi[key] = RoiConfig().load(_get_section(roi_raw, key))
        else:
            try:
                roi_sec = _get_section(sec, "roi")
                self.roi = {}
                for key in roi_sec.keys():
                    self.roi[key] = RoiConfig().load(_get_section(roi_sec, key))
            except KeyError:
                self.roi = {}

        # Load contour smoother
        contour_smoother = _get_key(
            sec, "contour_smoother", required=False, default="none", type_=str
        ).lower()
        if contour_smoother == "none":
            self.contour_smoother = None
        else:
            smoother_options_sec = sec.get("contour_smoother_options", {})

            if contour_smoother == "savitzky_golay":
                smoother_options = SavitzkyGolaySmootherConfig().load(
                    smoother_options_sec
                )
                self.contour_smoother = darsia.SavitzkyGolaySmoother(
                    window_length=smoother_options.window_length,
                    polyorder=smoother_options.polyorder,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported contour smoother type: {contour_smoother}"
                )

        # Load reduce_to_main_contour
        self.reduce_to_main_contour = _get_key(
            sec, "reduce_to_main_contour", required=False, default=True, type_=bool
        )

        return self

    def error(self):
        raise ValueError(f"Use [analysis.fingers] in the config file to load fingers.")
