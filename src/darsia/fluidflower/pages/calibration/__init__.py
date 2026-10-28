"""Calibration integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="calibration",
    factory="darsia.fluidflower.pages.calibration.page:CalibrationPage",
    service_modules=(
        "darsia.fluidflower.calibration",
        "darsia.fluidflower.indicator_response",
    ),
)

__all__ = ["PAGE_PLUGIN"]
