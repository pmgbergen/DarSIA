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
    no_saturation_g: list[str] = field(
        default_factory=list,
        metadata={
            "name": "No-Saturation ROIs",
            "help": "ROI keys where saturation_g is forced to zero.",
            "widget": "key_list",
            "key_list_sources": ROI_SOURCES,
        },
    )
    """ROI registry keys where saturation_g is forced to zero (exclusion)."""
    no_concentration_aq: list[str] = field(
        default_factory=list,
        metadata={
            "name": "No-Concentration ROIs",
            "help": "ROI keys where concentration_aq is forced to zero.",
            "widget": "key_list",
            "key_list_sources": ROI_SOURCES,
        },
    )
    """ROI registry keys where concentration_aq is forced to zero (exclusion)."""

    def load(
        self, path: Path | list[Path], roi_registry: "RoiRegistry | None" = None
    ) -> "ExpertKnowledgeConfig":
        sec = _get_section_from_toml(path, "expert_knowledge")

        self.saturation_g = _get_key(sec, "saturation_g", required=False, default=[])
        self.concentration_aq = _get_key(
            sec, "concentration_aq", required=False, default=[]
        )
        self.no_saturation_g = _get_key(
            sec, "no_saturation_g", required=False, default=[]
        )
        self.no_concentration_aq = _get_key(
            sec, "no_concentration_aq", required=False, default=[]
        )

        roi_key_fields = (
            ("saturation_g", self.saturation_g),
            ("concentration_aq", self.concentration_aq),
            ("no_saturation_g", self.no_saturation_g),
            ("no_concentration_aq", self.no_concentration_aq),
        )
        for name, value in roi_key_fields:
            if not isinstance(value, list) or not all(
                isinstance(key, str) for key in value
            ):
                raise ValueError(f"expert_knowledge.{name} must be a list[str].")

        # Validate registry references eagerly when provided.
        any_keys_provided = any(len(value) > 0 for _, value in roi_key_fields)
        if roi_registry is not None:
            for _, value in roi_key_fields:
                if len(value) > 0:
                    roi_registry.resolve_rois(value)
        elif any_keys_provided:
            raise ValueError(
                "expert_knowledge requires a loaded ROI registry when "
                "saturation_g, concentration_aq, no_saturation_g, or "
                "no_concentration_aq keys are provided."
            )

        return self
