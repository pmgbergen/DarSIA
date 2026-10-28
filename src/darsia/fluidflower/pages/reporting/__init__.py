"""Optional reporting-page package retained for downstream integrations."""

from darsia.fluidflower.page_registry import FluidFlowerSubpagePlugin

SUBPAGE_PLUGINS = (
    FluidFlowerSubpagePlugin(
        "reporting",
        "report_package",
        "Report package",
        "darsia.fluidflower.pages.reporting.subpages.report_package:build",
        10,
    ),
    FluidFlowerSubpagePlugin(
        "reporting",
        "scientific_review",
        "Scientific review",
        "darsia.fluidflower.pages.reporting.subpages.scientific_review:build",
        20,
    ),
)

__all__ = ["SUBPAGE_PLUGINS"]
