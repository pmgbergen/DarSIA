"""Configuration for expert-knowledge constraints on scalar fields.

Constraints by ROIs (where a field may be non-zero) and by bounds (the range of its
values). Top-level (not nested under analysis/calibration), since expert knowledge
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

BOUNDED_FIELDS = ("concentration_aq", "saturation_g")
"""Fields which may be bounded."""


def _get_bound(sec: dict, key: str, default: float | None) -> float | None:
    """A bound: a number, or "none" for no bound (blank or missing: the default)."""
    value = sec.get(key)
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return _get_key(sec, key, default, required=False, type_=float)


@dataclass
class BoundsConfig:
    """Lower and upper bound of the values of a field (not both None)."""

    lower: float | None = field(
        default=0.0,
        metadata={
            "name": "Lower bound",
            "help": "Smaller values are raised to it. Default 0; none for no bound.",
            "placeholder": "e.g., 0.0 (empty: 0; none: no bound)",
        },
    )
    """Lower bound; None for no bound."""
    upper: float | None = field(
        default=None,
        metadata={
            "name": "Upper bound",
            "help": "Larger values are lowered to it. Default: no bound.",
            "placeholder": "e.g., 1.0 (empty or none: no bound)",
        },
    )
    """Upper bound; None for no bound."""

    def load(self, sec: dict, context: str) -> "BoundsConfig":
        self.lower = _get_bound(sec, "lower", self.lower)
        self.upper = _get_bound(sec, "upper", self.upper)
        if self.lower is None and self.upper is None:
            raise ValueError(f"{context}: needs a lower or an upper bound.")
        if self.lower is not None and self.upper is not None:
            if self.upper < self.lower:
                raise ValueError(f"{context}: upper must not be smaller than lower.")
        return self


@dataclass
class ExpertKnowledgeBoundsConfig:
    """Bounds of the fields, one table per bounded field.

    A field is bounded if its table is present (an empty table means the defaults
    [0, None]); an ``active`` list, as for the corrections, restricts this to the
    fields listed.

    """

    concentration_aq: BoundsConfig | None = field(
        default=None,
        metadata={
            "name": "Concentration bounds",
            "help": "Range of concentration_aq, applied after the flash.",
            "active_list_key": "active",
        },
    )
    """Bounds of concentration_aq."""
    saturation_g: BoundsConfig | None = field(
        default=None,
        metadata={
            "name": "Saturation bounds",
            "help": "Range of saturation_g, applied after the flash (CO2 only).",
            "active_list_key": "active",
            "fluids": ("co2",),
        },
    )
    """Bounds of saturation_g."""

    def load(self, sec: dict) -> "ExpertKnowledgeBoundsConfig":
        unknown = set(sec) - {"active", *BOUNDED_FIELDS}
        if len(unknown) > 0:
            raise ValueError(
                f"expert_knowledge.bounds: unknown fields {sorted(unknown)}; "
                f"bounded fields are {list(BOUNDED_FIELDS)}."
            )
        active = sec.get("active")  # None: all present are active
        for name in BOUNDED_FIELDS:
            if name in sec and (active is None or name in active):
                context = f"expert_knowledge.bounds.{name}"
                setattr(self, name, BoundsConfig().load(sec[name], context))
        return self


@dataclass
class ExpertKnowledgeConfig:
    """Configuration for expert-knowledge constraints on scalar fields."""

    saturation_g: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Saturation ROIs",
            "help": "ROI keys where saturation_g constraints apply (CO2 only).",
            "widget": "key_list",
            "key_list_sources": ROI_SOURCES,
            "fluids": ("co2",),
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
            "help": "ROI keys where saturation_g is forced to zero (CO2 only).",
            "widget": "key_list",
            "key_list_sources": ROI_SOURCES,
            "fluids": ("co2",),
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
    bounds: ExpertKnowledgeBoundsConfig = field(
        default_factory=ExpertKnowledgeBoundsConfig,
        metadata={
            "name": "Bounds",
            "help": (
                "Range of the values per field, applied before the ROI constraints."
            ),
        },
    )
    """Bounds of the fields."""

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
        self.bounds = ExpertKnowledgeBoundsConfig().load(sec.get("bounds", {}))

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
