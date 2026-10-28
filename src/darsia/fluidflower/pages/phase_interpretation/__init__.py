"""Phase-interpretation integration package."""

from darsia.fluidflower.page_registry import (
    FluidFlowerPagePlugin,
    FluidFlowerSubpagePlugin,
)

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="phase_interpretation",
    factory="darsia.fluidflower.pages.phase_interpretation.page:PhaseInterpretationPage",
    service_modules=(
        "darsia.fluidflower.phase_features",
        "darsia.fluidflower.phase_interpretation",
        "darsia.fluidflower.phase_learning",
        "darsia.fluidflower.phase_workflow",
    ),
)

SUBPAGE_PLUGINS = (
    FluidFlowerSubpagePlugin(
        "phase_interpretation",
        "active_learning",
        "Active learning review",
        "darsia.fluidflower.pages.phase_interpretation.subpages.active_learning:build",
        10,
    ),
    FluidFlowerSubpagePlugin(
        "phase_interpretation",
        "model_validation",
        "Model validation",
        "darsia.fluidflower.pages.phase_interpretation.subpages.model_validation:build",
        20,
    ),
    FluidFlowerSubpagePlugin(
        "phase_interpretation",
        "studio",
        "Phase Model Studio",
        "darsia.fluidflower.pages.phase_interpretation.subpages.train_model:build",
        30,
    ),
)

__all__ = ["PAGE_PLUGIN", "SUBPAGE_PLUGINS"]
