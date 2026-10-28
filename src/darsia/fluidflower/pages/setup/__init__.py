"""Series-setup integration package."""

from darsia.fluidflower.page_registry import (
    FluidFlowerPagePlugin,
    FluidFlowerSubpagePlugin,
)

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="setup",
    factory="darsia.fluidflower.pages.setup.page:create_page",
    service_modules=(
        "darsia.fluidflower.project",
        "darsia.fluidflower.protocol_schema",
        "darsia.fluidflower.source_discovery",
    ),
    scrollable=False,
)

SUBPAGE_PLUGINS = (
    FluidFlowerSubpagePlugin(
        "setup_wizard",
        "identity",
        "Series identity",
        "darsia.fluidflower.pages.setup.steps.identity:build",
        10,
        "identity",
    ),
    FluidFlowerSubpagePlugin(
        "setup_wizard",
        "image_sources",
        "Apparatus, protocol and sources",
        "darsia.fluidflower.pages.setup.steps.image_sources:build",
        20,
        "images",
    ),
    FluidFlowerSubpagePlugin(
        "setup_wizard",
        "summary",
        "Create project",
        "darsia.fluidflower.pages.setup.steps.summary:build",
        30,
        "summary",
    ),
)

__all__ = ["PAGE_PLUGIN", "SUBPAGE_PLUGINS"]
