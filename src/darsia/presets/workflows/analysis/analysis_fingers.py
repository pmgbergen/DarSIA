"""Template for finger analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    prepare_analysis_context,
)
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.segmentation_contours import SimpleSegmentation
from darsia.single_image_analysis.contouranalysis import (
    ContourAnalysis,
    ContourEvolutionAnalysis,
)

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
    contour_analysis = ContourAnalysis()
    contour_evolution_analysis = ContourEvolutionAnalysis()

    # Data management.
    results_folder = ctx.config.analysis.fingers.folder
    results_folder.mkdir(parents=True, exist_ok=True)
    for key in fingers_config.roi:
        (results_folder / "tips" / key).mkdir(parents=True, exist_ok=True)
        (results_folder / "valleys" / key).mkdir(parents=True, exist_ok=True)
        (results_folder / "paths" / key).mkdir(parents=True, exist_ok=True)
        (results_folder / "valley_paths" / key).mkdir(parents=True, exist_ok=True)

    # DataFrame to store results.
    df = pd.DataFrame(
        columns=[
            "time",
            "key",
            "image",
            "contour_length",
            "number_tips",
            "number_peaks",
            "number_valleys",
        ]
    )

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

        for key, roi_config in fingers_config.roi.items():
            # TODO: allow to tune the threshold value, and mode in a interactive way.

            # Perform finger analysis if configured
            contour_analysis.load(
                img, segmentation, roi=roi_config.roi, fill_holes=False
            )

            # Extract contour
            contours = contour_analysis.contours()

            # Determine various contour values.
            contour_length = contour_analysis.length()
            peaks, valleys = contour_analysis.fingers()
            number_peaks = contour_analysis.number_peaks()
            number_valleys = len(valleys)
            contour_analysis.plot_finger_peaks(
                img,
                peaks,
                roi_config.roi,
                contours=contours,
                path=results_folder / "tips" / key / f"{path.stem}.png",
                show=show,
                **{
                    # TODO enable control from config.
                    "peak_color": "r",
                    "peak_size": 10,
                    "contour_color": "w",
                    "contour_linewidth": 1,
                    # "plot_boundary": True,
                    # "boundary_color": "y",
                    # "boundary_linewidth": 2,
                    # "highlight_roi": True,
                },
            )
            contour_analysis.plot_valleys(
                img,
                valleys,
                roi_config.roi,
                contours=contours,
                path=results_folder / "valleys" / key / f"{path.stem}.png",
                show=show,
                **{
                    "valley_color": "c",
                    "valley_linewidth": 1,
                    "plot_valley_dots": True,
                    "valley_dot_color": "r",
                    "valley_dot_size": 20,
                    "contour_color": "w",
                    "contour_linewidth": 1,
                },
            )

            # Update evolution analysis.
            contour_evolution_analysis.add(peaks=peaks, valleys=valleys, time=img.time)
            contour_evolution_analysis.find_paths()
            contour_evolution_analysis.find_valley_paths()
            # contour_evolution_analysis.plot(img, roi=roi_config.roi)

            contour_evolution_analysis.plot_paths(
                img,
                roi=roi_config.roi,
                path=results_folder / "paths" / key / f"{path.stem}.png",
                show=show,
            )
            contour_evolution_analysis.plot_valley_paths(
                img,
                roi=roi_config.roi,
                path=results_folder / "valley_paths" / key / f"{path.stem}.png",
                show=show,
                color=None,
            )
            # number_paths = contour_evolution_analysis.number_paths

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "time": img.time,
                            "key": key,
                            "image": path.name,
                            "contour_length": contour_length,
                            # number_tips historically represented detected peaks; keep it in
                            # sync with number_peaks for backward compatibility.
                            "number_tips": number_peaks,
                            "number_peaks": number_peaks,
                            "number_valleys": number_valleys,
                            # "number_merged_paths": ...,
                            # "number_new_paths": ...,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
            df.to_csv(results_folder / "results.csv", index=False)


def analysis_fingers(
    cls: type[Rig],
    path: Path | list[Path],
    show: bool = False,
    all: bool = False,
):
    """Fingers analysis (standalone entry point).

    Args:
        cls: Rig class.
        path: Path or list of paths to config files.
        show: Whether to show the images.
        all: Whether to use all images.

    """
    ctx = prepare_analysis_context(
        cls=cls,
        path=path,
        all=all,
        require_color_to_mass=True,
    )
    analysis_fingers_from_context(ctx, show=show)
