"""Template for segmentation analysis."""

from __future__ import annotations

import logging
from pathlib import Path

from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    prepare_analysis_context,
)
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.segmentation_contours import SegmentationContours

logger = logging.getLogger(__name__)


def analysis_segmentation_from_context(
    ctx: AnalysisContext,
    show: bool = False,
) -> None:
    """Segmentation analysis using pre-prepared context.

    Args:
        ctx: Pre-prepared analysis context with color_to_mass_analysis initialized.
        show: Whether to show the images.

    """
    assert ctx.config.analysis is not None
    assert ctx.config.analysis.segmentation is not None
    assert ctx.color_to_mass_analysis is not None

    fluidflower = ctx.fluidflower
    image_paths = ctx.image_paths
    color_to_mass_analysis = ctx.color_to_mass_analysis

    # Extract segmentation config (asserted not None above)
    segmentation_config = ctx.config.analysis.segmentation

    # ! ---- CONTOUR PLOTTING ----
    segmentation_contours = SegmentationContours(segmentation_config.config)

    # Loop over images and analyze
    for path in image_paths:
        # Extract color signal and assign mass
        img = fluidflower.read_image(path)
        mass_analysis_result = color_to_mass_analysis(img)

        # Produce contour images
        contour_image = segmentation_contours(
            img,
            saturation_g=mass_analysis_result.saturation_g,
            concentration_aq=mass_analysis_result.concentration_aq,
            mass=mass_analysis_result.mass,
        )

        if show:
            contour_image.show(
                title=f"Contours for {path.stem} | {img.time} seconds", delay=False
            )

        contour_path = segmentation_config.folder / f"{path.stem}.jpg"
        contour_image.write(contour_path, quality=80)


def analysis_segmentation(
    cls: type[Rig],
    path: Path | list[Path],
    show: bool = False,
    all: bool = False,
    use_facies: bool = True,
):
    """Segmentation analysis (standalone entry point).

    Args:
        cls: FluidFlower rig class.
        path: Path or list of paths to config files.
        show: Whether to show the images.
        all: Whether to use all images.
        use_facies: Whether to use facies as labels.

    """
    ctx = prepare_analysis_context(
        cls=cls,
        path=path,
        all=all,
        use_facies=use_facies,
        require_color_to_mass=True,
    )
    analysis_segmentation_from_context(ctx, show=show)
