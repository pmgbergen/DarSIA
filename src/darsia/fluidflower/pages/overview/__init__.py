"""Workflow-overview integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="overview",
    factory="darsia.fluidflower.pages.overview.page:OverviewPage",
    service_modules=("darsia.fluidflower.project", "darsia.fluidflower.modules"),
    scrollable=False,
)

__all__ = ["PAGE_PLUGIN"]
