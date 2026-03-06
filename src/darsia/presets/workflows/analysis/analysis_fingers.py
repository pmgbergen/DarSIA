"""Template for finger analysis."""

from __future__ import annotations

import logging
from pathlib import Path

from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    prepare_analysis_context,
)
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.segmentation_contours import SimpleSegmentation
from darsia.single_image_analysis.contouranalysis import ContourAnalysis

logger = logging.getLogger(__name__)


def analysis_fingers_from_context(
    ctx: AnalysisContext,
    show: bool = False,
) -> None:
    """Segmentation analysis using pre-prepared context.

    Args:
        ctx: Pre-prepared analysis context with color_to_mass_analysis initialized.
        show: Whether to show the images.

    """
    assert ctx.config.analysis is not None
    assert ctx.config.analysis.fingers is not None
    assert ctx.color_to_mass_analysis is not None

    fluidflower = ctx.fluidflower
    image_paths = ctx.image_paths
    color_to_mass_analysis = ctx.color_to_mass_analysis

    # Extract finger analysis config (asserted not None above)
    fingers_config = ctx.config.analysis.fingers.config
    segmentation_analysis = SimpleSegmentation(
        mode=fingers_config.mode, threshold=fingers_config.threshold
    )
    contour_analysis = ContourAnalysis(verbosity=True)

    # Loop over images and analyze
    for path in image_paths:
        # Extract color signal and assign mass
        img = fluidflower.read_image(path)
        mass_analysis_result = color_to_mass_analysis(img)

        # Produce contour images
        segmentation = segmentation_analysis(
            img,
            saturation_g=mass_analysis_result.saturation_g,
            concentration_aq=mass_analysis_result.concentration_aq,
            mass=mass_analysis_result.mass,
        )

        for key, roi in fingers_config.roi.items():
            # Perform finger analysis if configured
            contour_analysis.load(img, segmentation, roi=roi.roi, fill_holes=False)

            print(key, roi.name, roi.roi)

            # Determine various contour values.
            contour_length = contour_analysis.length()
            print(f"Contour length: {contour_length=}")

            fingers = contour_analysis.fingers()
            print(f"Fingers: {fingers=}")

            number_peaks = contour_analysis.number_peaks()
            print(f"Number of peaks: {number_peaks=}")


def analysis_fingers(
    cls: type[Rig],
    path: Path | list[Path],
    show: bool = False,
    all: bool = False,
    use_facies: bool = True,
):
    """Fingers analysis (standalone entry point).

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
    analysis_fingers_from_context(ctx, show=show)
