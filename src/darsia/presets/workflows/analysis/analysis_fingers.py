"""Template for finger analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
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
    assert fingers_config.roi is not None
    segmentation_analysis = SimpleSegmentation(
        mode=fingers_config.mode, threshold=fingers_config.threshold
    )
    contour_analysis = ContourAnalysis()
    contour_evolution_analysis = {
        key: ContourEvolutionAnalysis() for key in fingers_config.roi
    }
    contour_evolution_times = {key: [] for key in fingers_config.roi}

    # Data management.
    results_folder = ctx.config.analysis.fingers.folder
    results_folder.mkdir(parents=True, exist_ok=True)
    for key in fingers_config.roi:
        (results_folder / "tips" / key).mkdir(parents=True, exist_ok=True)
        (results_folder / "paths" / key).mkdir(parents=True, exist_ok=True)

    # DataFrame to store results.
    df = pd.DataFrame(columns=["time", "key", "image", "contour_length", "number_tips"])

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

            # Update evolution analysis.
            contour_evolution_analysis[key].add(peaks=peaks, valleys=valleys, time=img.time)
            contour_evolution_analysis[key].find_paths()
            contour_evolution_times[key].append(float(img.time))
            # contour_evolution_analysis[key].plot(img, roi=roi_config.roi)

            contour_evolution_analysis[key].plot_paths(
                img,
                roi=roi_config.roi,
                path=results_folder / "paths" / key / f"{path.stem}.png",
                show=show,
            )
            # number_paths = contour_evolution_analysis.number_paths

            roi_top_left_x = 0
            roi_top_left_y = 0
            if roi_config.roi is not None:
                roi_pixels = roi_config.roi.to_voxel(img.coordinatesystem)
                roi_top_left_x = int(np.min(roi_pixels[:, 1]))
                roi_top_left_y = int(np.min(roi_pixels[:, 0]))

            path_log = {}
            for finger_path in contour_evolution_analysis[key].paths:
                if len(finger_path) == 0:
                    continue

                start_unit = finger_path[0]
                path_id = f"path_t{int(start_unit.time)}_p{int(start_unit.peak)}"
                coordinates = []
                for unit in finger_path:
                    time_index = int(unit.time)
                    if 0 <= time_index < len(contour_evolution_times[key]):
                        unit_time = float(contour_evolution_times[key][time_index])
                    else:
                        continue

                    x = int(unit.position[0]) + roi_top_left_x
                    y = int(unit.position[1]) + roi_top_left_y
                    coordinates.append([x, y, unit_time])

                if len(coordinates) == 0:
                    continue

                path_log[path_id] = {
                    "start": coordinates[0][2],
                    "end": coordinates[-1][2],
                    "coordinates": coordinates,
                }

            with open(results_folder / "paths" / key / "paths.json", "w") as f:
                json.dump(path_log, f, indent=2)

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "time": img.time,
                            "key": key,
                            "image": path.name,
                            "contour_length": contour_length,
                            "number_peaks": number_peaks,
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
