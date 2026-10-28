"""Image-correction integration package."""

from darsia.fluidflower.page_registry import FluidFlowerPagePlugin

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="image_correction",
    factory="darsia.fluidflower.pages.image_correction.page:ImageCorrectionPage",
    service_modules=(
        "darsia.fluidflower.image_correction",
        "darsia.fluidflower.image_correction_materialization",
        "darsia.fluidflower.frame_quality",
        "darsia.fluidflower.pages.image_correction.frame_quality_panel",
    ),
)

__all__ = ["PAGE_PLUGIN"]
