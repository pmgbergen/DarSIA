"""Geometry-and-facies integration package."""

from darsia.fluidflower.page_registry import (
    FluidFlowerPagePlugin,
    FluidFlowerSubpagePlugin,
)

PAGE_PLUGIN = FluidFlowerPagePlugin(
    key="geometry_facies",
    factory="darsia.fluidflower.pages.geometry_facies.page:GeometryFaciesPage",
    service_modules=(
        "darsia.fluidflower.geometry_facies",
        "darsia.fluidflower.facies_examples",
        "darsia.fluidflower.facies_learning",
        "darsia.fluidflower.facies_model",
    ),
)

SUBPAGE_PLUGINS = (
    FluidFlowerSubpagePlugin(
        "geometry_facies",
        "inputs",
        "Inputs",
        "darsia.fluidflower.pages.geometry_facies.subpages.inputs:build",
        10,
    ),
    FluidFlowerSubpagePlugin(
        "geometry_facies",
        "layers",
        "Layers",
        "darsia.fluidflower.pages.geometry_facies.subpages.layers:build",
        20,
    ),
    FluidFlowerSubpagePlugin(
        "geometry_facies",
        "injection_points",
        "Injection points",
        "darsia.fluidflower.pages.geometry_facies.subpages.injection_points:build",
        30,
    ),
    FluidFlowerSubpagePlugin(
        "geometry_facies",
        "run_geometry",
        "Run geometry",
        "darsia.fluidflower.pages.geometry_facies.subpages.run_geometry:build",
        40,
        "run_geometry_tab",
    ),
    FluidFlowerSubpagePlugin(
        "geometry_facies",
        "boundary_edits",
        "Boundary edits",
        "darsia.fluidflower.pages.geometry_facies.subpages.boundary_edits:build",
        50,
        "boundary_tab",
    ),
    FluidFlowerSubpagePlugin(
        "geometry_facies",
        "material_examples",
        "Material examples",
        "darsia.fluidflower.pages.geometry_facies.subpages.material_examples:build",
        55,
        "material_examples_tab",
    ),
    FluidFlowerSubpagePlugin(
        "geometry_facies",
        "training_data",
        "Training data",
        "darsia.fluidflower.pages.geometry_facies.subpages.training_data:build",
        60,
        "training_data_tab",
    ),
    FluidFlowerSubpagePlugin(
        "geometry_facies",
        "qc_history",
        "QC and history",
        "darsia.fluidflower.pages.geometry_facies.subpages.qc_history:build",
        70,
        "qc_tab",
    ),
)

__all__ = ["PAGE_PLUGIN", "SUBPAGE_PLUGINS"]
