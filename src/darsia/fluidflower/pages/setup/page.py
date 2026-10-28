"""Interactive established-series setup page."""

from __future__ import annotations

from darsia.fluidflower.pages.setup.incremental_runs import (
    EstablishedSourceSetupEditor,
)
from darsia.fluidflower.project import FluidFlowerProject


def create_page(project: FluidFlowerProject) -> EstablishedSourceSetupEditor:
    """Create the complete established-series setup work surface."""

    return EstablishedSourceSetupEditor(project)


__all__ = ["create_page"]
