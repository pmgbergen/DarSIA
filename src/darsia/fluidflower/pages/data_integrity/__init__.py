"""Data-integrity integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="data_integrity",
    factory="darsia.fluidflower.pages.data_integrity.page:DataIntegrityPage",
    service_modules=("darsia.fluidflower.data_integrity",),
)

__all__ = ["PAGE_PLUGIN"]
