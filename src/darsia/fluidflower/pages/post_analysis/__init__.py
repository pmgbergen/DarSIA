"""Post-analysis integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="post_analysis",
    factory="darsia.fluidflower.pages.post_analysis.page:PostAnalysisPage",
    service_modules=(
        "darsia.fluidflower.post_analysis",
        "darsia.fluidflower.post_materialization",
    ),
)

__all__ = ["PAGE_PLUGIN"]
