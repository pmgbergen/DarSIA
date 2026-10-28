"""Wasserstein-analysis integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="wasserstein",
    factory="darsia.fluidflower.pages.wasserstein.page:WassersteinPage",
    service_modules=(
        "darsia.fluidflower.wasserstein",
        "darsia.fluidflower.wasserstein_materialization",
    ),
)

__all__ = ["PAGE_PLUGIN"]
