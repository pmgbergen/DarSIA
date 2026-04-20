"""Configuration for fingers, based on segmentation analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from darsia.presets.workflows.mode_resolution import validate_mode_syntax

from .roi import RoiConfig
from .utils import _get_key, _get_section

if TYPE_CHECKING:
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

    def load(
        self,
        sec: dict,
        roi_registry: RoiRegistry | None = None,
        color_embedding_registry=None,
    ) -> "FingersConfig":
        self.mode = _get_key(sec, "mode", required=True, type_=str)
        if not validate_mode_syntax(
            self.mode, color_embedding_registry=color_embedding_registry
        ):
            raise ValueError(
                f"Unsupported analysis.fingers.mode '{self.mode}'. Supported modes "
                "are legacy mass modes, rescaled modes, and 'color.<id>'."
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

        return self

    def error(self):
        raise ValueError(f"Use [analysis.fingers] in the config file to load fingers.")
