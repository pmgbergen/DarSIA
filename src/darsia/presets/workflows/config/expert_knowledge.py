"""Configuration for expert-knowledge ROI constraints on scalar fields.

Top-level (not nested under analysis/calibration), since expert knowledge
already constrains both analysis and calibration via the shared
AnalysisContext - see ExpertKnowledgeAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ._key_list_sources import ROI_SOURCES
from .utils import _get_key, _get_section_from_toml

if TYPE_CHECKING:
    from .roi_registry import RoiRegistry


@dataclass
class ExpertKnowledgeConfig:
    """Configuration for expert-knowledge ROI constraints on scalar fields."""

    saturation_g: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Saturation ROIs",
            "help": "ROI keys where saturation_g constraints apply.",
            "widget": "key_list",
            "key_list_sources": ROI_SOURCES,
        },
    )
    """ROI registry keys constraining where saturation_g may be non-zero."""
    concentration_aq: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Concentration ROIs",
            "help": "ROI keys where concentration_aq constraints apply.",
            "widget": "key_list",
            "key_list_sources": ROI_SOURCES,
        },
    )
    """ROI registry keys constraining where concentration_aq may be non-zero."""

    def load(
        self, path: Path | list[Path], roi_registry: "RoiRegistry | None" = None
    ) -> "ExpertKnowledgeConfig":
        sec = _get_section_from_toml(path, "expert_knowledge")

        self.saturation_g = _get_key(sec, "saturation_g", required=False, default=[])
        self.concentration_aq = _get_key(
            sec, "concentration_aq", required=False, default=[]
        )

        if not isinstance(self.saturation_g, list) or not all(
            isinstance(key, str) for key in self.saturation_g
        ):
            raise ValueError("expert_knowledge.saturation_g must be a list[str].")
        if not isinstance(self.concentration_aq, list) or not all(
            isinstance(key, str) for key in self.concentration_aq
        ):
            raise ValueError("expert_knowledge.concentration_aq must be a list[str].")

        # Validate registry references eagerly when provided.
        if roi_registry is not None:
            if len(self.saturation_g) > 0:
                roi_registry.resolve_rois(self.saturation_g)
            if len(self.concentration_aq) > 0:
                roi_registry.resolve_rois(self.concentration_aq)
        elif len(self.saturation_g) > 0 or len(self.concentration_aq) > 0:
            raise ValueError(
                "expert_knowledge requires a loaded ROI registry when "
                "saturation_g or concentration_aq keys are provided."
            )

        return self
