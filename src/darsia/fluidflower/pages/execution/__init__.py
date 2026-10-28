"""Execution-and-workers integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="execution",
    factory="darsia.fluidflower.pages.execution.page:ExecutionPage",
    service_modules=(
        "darsia.fluidflower.execution",
        "darsia.fluidflower.quality_budget",
    ),
)

__all__ = ["PAGE_PLUGIN"]
