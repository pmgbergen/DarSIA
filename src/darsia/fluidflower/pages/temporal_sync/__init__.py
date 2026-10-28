"""Temporal-synchronization integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="temporal_sync",
    factory="darsia.fluidflower.pages.temporal_sync.page:TimelinePage",
    service_modules=("darsia.fluidflower.timeline",),
)

__all__ = ["PAGE_PLUGIN"]
