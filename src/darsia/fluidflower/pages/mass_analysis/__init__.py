"""Mass-analysis integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="mass_analysis",
    factory="darsia.fluidflower.pages.mass_analysis.page:MassAnalysisPage",
    service_modules=(
        "darsia.fluidflower.mass_analysis",
        "darsia.fluidflower.mass_inputs",
        "darsia.fluidflower.mass_materialization",
    ),
)

__all__ = ["PAGE_PLUGIN"]
