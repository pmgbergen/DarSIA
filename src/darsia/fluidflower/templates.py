"""Versioned templates for starting FluidFlower experimental series."""

from __future__ import annotations

from dataclasses import dataclass


RIG_TYPES: dict[str, str] = {
    "a4": "A4 FluidFlower",
    "medium": "Medium FluidFlower (0.90 x 0.49 m)",
    "large": "Large FluidFlower (2.80 x 1.50 m)",
    "custom": "Custom FluidFlower apparatus",
}


@dataclass(frozen=True)
class RigClass:
    """A class of FluidFlower rigs and its nominal inner size.

    The size is a suggestion for the rig frame until a series has measured
    values; series of one class can differ (for example FF_UM is 2.745 m wide).
    """

    key: str
    label: str
    short: str
    width_m: float | None = None
    height_m: float | None = None
    source: str = ""

    def size_text(self) -> str:
        if self.width_m is None or self.height_m is None:
            return ""
        return f"{self.width_m:.2f} x {self.height_m:.2f} m"

    def title(self) -> str:
        """Label with the nominal inner size, when known."""

        size = self.size_text()
        return f"{self.label} ({size})" if size else self.label


RIG_CLASSES: dict[str, RigClass] = {
    "medium": RigClass(
        "medium",
        "Medium FluidFlower",
        "Medium",
        0.92,
        0.55,
        "DarSIA's SimpleFluidFlower preset (0.92 x 0.55 m, water height 0.529 m, "
        "depth 12 mm); FF_RE measured 0.9214 m wide. FF_AC's [rig] 0.90 x 0.49 m "
        "is the part of the rig visible in its photographs.",
    ),
    "large": RigClass(
        "large",
        "Large FluidFlower",
        "Large",
        2.80,
        1.50,
        "FluidFlower templates and FF_BM [rig] (2.80 x 1.50 m); FF_UM uses "
        "2.745 x 1.50 m.",
    ),
    "a4": RigClass("a4", "A4 FluidFlower", "A4"),
    "custom": RigClass("custom", "Custom apparatus", "Custom"),
}


def rig_class_rank(key: str) -> int:
    """Sort order of rig classes; unknown or unset classes come last."""

    order = list(RIG_CLASSES)
    return order.index(key) if key in RIG_CLASSES else len(order)


@dataclass(frozen=True)
class ExperimentalSeriesTemplate:
    """Editable defaults, never a substitute for measured series metadata."""

    template_id: str
    name: str
    description: str
    default_rig_type: str
    protocol_template: str
    overflow_mode: str
    indicator_system: str


EXPERIMENTAL_SERIES_TEMPLATES: tuple[ExperimentalSeriesTemplate, ...] = (
    ExperimentalSeriesTemplate(
        "ff_um",
        "FF_UM",
        "Published large-FluidFlower series using the universal event-table schema.",
        "large",
        "fluidflower_timeline_v2",
        "none",
        "BTB",
    ),
    ExperimentalSeriesTemplate(
        "ff_ac",
        "FF_AC",
        "Albus/medium-FluidFlower series with optional scale-derived overflow.",
        "medium",
        "fluidflower_timeline_v2",
        "scale_images",
        "BTB",
    ),
    ExperimentalSeriesTemplate(
        "ff_bm",
        "FF_BM",
        "Published FluidFlower benchmark series.",
        "large",
        "fluidflower_timeline_v2",
        "none",
        "BTB",
    ),
    ExperimentalSeriesTemplate(
        "custom",
        "New custom series",
        "Create a new series using the universal FluidFlower event-table schema.",
        "custom",
        "fluidflower_timeline_v2",
        "none",
        "unspecified",
    ),
)

TEMPLATE_BY_ID = {
    template.template_id: template for template in EXPERIMENTAL_SERIES_TEMPLATES
}


def template_by_id(template_id: str) -> ExperimentalSeriesTemplate:
    try:
        return TEMPLATE_BY_ID[str(template_id).casefold()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown FluidFlower series template: {template_id!r}"
        ) from exc
