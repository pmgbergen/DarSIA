"""Optical- and chemical-response calibration integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="optical_calibration",
    factory=(
        "darsia.fluidflower.pages.optical_calibration.page:OpticalCalibrationPage"
    ),
    service_modules=(
        "darsia.fluidflower.indicator_response",
        "darsia.fluidflower.pages.calibration.indicator_response_panel",
    ),
)

__all__ = ["PAGE_PLUGIN"]
