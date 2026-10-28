"""Overflow-measurement integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="overflow",
    factory="darsia.fluidflower.pages.overflow.page:OverflowPage",
    service_modules=(
        "darsia.fluidflower.overflow",
        "darsia.fluidflower.overflow_workflow",
    ),
)

__all__ = ["PAGE_PLUGIN"]
