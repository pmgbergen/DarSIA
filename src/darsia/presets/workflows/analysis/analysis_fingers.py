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
    # Keep evolution state per ROI to prevent mixing path histories across ROIs.
    contour_evolution_analysis = {
        key: ContourEvolutionAnalysis() for key in fingers_config.roi
    }
    contour_evolution_times = {key: [] for key in fingers_config.roi}

    # Data management.
    results_folder = ctx.config.analysis.fingers.folder
    results_folder.mkdir(parents=True, exist_ok=True)
    for key in fingers_config.roi:
        (results_folder / "tips" / key).mkdir(parents=True, exist_ok=True)
        (results_folder / "fjords" / key).mkdir(parents=True, exist_ok=True)
        (results_folder / "tips_paths" / key).mkdir(parents=True, exist_ok=True)
        (results_folder / "fjord_paths" / key).mkdir(parents=True, exist_ok=True)

    # DataFrame to store results.
    df = pd.DataFrame(
        columns=[
            "time",
            "key",
            "image",
            "contour_length",
            "number_tips",
            "number_fjords",
            "roi_width",
            "tip_frequency",
            "tip_wavelength",
        ]
    )

    # Config of plotting.
    # TODO enable control from config.
    plot_format = ".png"
    contour_color = "w"
    peak_color = "r"
    peak_size = 10
    contour_linewidth = 1
    plot_boundary = False
    boundary_color = "y"
    boundary_linewidth = 2
    highlight_roi = False

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
            number_tips = contour_analysis.number_peaks()
            number_fjords = contour_analysis.number_valleys()
            roi_width = float(np.abs(roi_config.roi[1, 0] - roi_config.roi[0, 0]))
            tip_frequency = np.nan
            tip_wavelength = np.nan
            if roi_width > 0:
                tip_frequency = float(number_tips) / roi_width
                if number_tips > 0:
                    tip_wavelength = roi_width / float(number_tips)
            else:
                logger.warning(
                    "Skip frequency/wavelength computation due to non-positive"
                    " ROI width for ROI '%s'.",
                    key,
                )

            # Plot finger peaks and contours.
            contour_analysis.plot_finger_peaks(
                img,
                peaks,
                roi_config.roi,
                contours=contours,
                path=(results_folder / "tips" / key / f"{path.stem}").with_suffix(
                    plot_format
                ),
                show=show,
                **{
                    "peak_color": peak_color,
                    "peak_size": peak_size,
                    "contour_color": contour_color,
                    "contour_linewidth": contour_linewidth,
                    "plot_boundary": plot_boundary,
                    "boundary_color": boundary_color,
                    "boundary_linewidth": boundary_linewidth,
                    "highlight_roi": highlight_roi,
                },
            )
            contour_analysis.plot_valleys(
                img,
                valleys,
                roi_config.roi,
                contours=contours,
                path=results_folder / "fjords" / key / f"{path.stem}.png",
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
            contour_evolution_analysis[key].add(
                peaks=peaks, valleys=valleys, time=img.time
            )
            contour_evolution_analysis[key].find_paths()
            contour_evolution_analysis[key].find_valley_paths()
            contour_evolution_times[key].append(float(img.time))
            # contour_evolution_analysis[key].plot(img, roi=roi_config.roi)

            contour_evolution_analysis[key].plot_paths(
                img,
                roi=roi_config.roi,
                path=(results_folder / "paths" / key / f"{path.stem}").with_suffix(
                    plot_format
                ),
                show=show,
            )
            contour_evolution_analysis[key].plot_valley_paths(
                img,
                roi=roi_config.roi,
                path=results_folder / "fjord_paths" / key / f"{path.stem}.png",
                show=show,
                color=None,
            )
            # number_paths = contour_evolution_analysis.number_paths

            roi_top_left_x = 0
            roi_top_left_y = 0
            if roi_config.roi is not None:
                roi_pixels = roi_config.roi.to_voxel(img.coordinatesystem)
                roi_top_left_x = int(np.min(roi_pixels[:, 1]))
                roi_top_left_y = int(np.min(roi_pixels[:, 0]))

            path_log = {}
            active_fingers_by_time = {}
            for finger_path in contour_evolution_analysis[key].paths:
                if len(finger_path) == 0:
                    continue

                start_unit = finger_path[0]
                path_id = f"path_t{int(start_unit.time)}_p{int(start_unit.peak)}"
                path_id_base = path_id
                suffix = 1
                while path_id in path_log:
                    path_id = f"{path_id_base}_{suffix}"
                    suffix += 1
                coordinates = []
                for unit in finger_path:
                    time_index = int(unit.time)
                    if 0 <= time_index < len(contour_evolution_times[key]):
                        unit_time = contour_evolution_times[key][time_index]
                    else:
                        logger.warning(
                            "Skip path unit with invalid time index %s for ROI '%s'.",
                            time_index,
                            key,
                        )
                        continue

                    x = int(unit.position[0]) + roi_top_left_x
                    y = int(unit.position[1]) + roi_top_left_y
                    coordinates.append([x, y, unit_time])

                if len(coordinates) == 0:
                    continue

                lengths = [[0.0, float(coordinates[0][2])]]
                cumulative_length = 0.0
                for index in range(1, len(coordinates)):
                    x_prev, y_prev, _ = coordinates[index - 1]
                    x_curr, y_curr, t_curr = coordinates[index]
                    cumulative_length += float(np.hypot(x_curr - x_prev, y_curr - y_prev))
                    lengths.append([cumulative_length, float(t_curr)])

                heights = []
                min_y = float(coordinates[0][1])
                max_y = float(coordinates[0][1])
                heights.append([0.0, float(coordinates[0][2])])
                for index in range(1, len(coordinates)):
                    y_curr = float(coordinates[index][1])
                    t_curr = float(coordinates[index][2])
                    min_y = min(min_y, y_curr)
                    max_y = max(max_y, y_curr)
                    heights.append([max_y - min_y, t_curr])

                speeds = []
                for index in range(1, len(coordinates)):
                    x_prev, y_prev, t_prev = coordinates[index - 1]
                    x_curr, y_curr, t_curr = coordinates[index]
                    dt = float(t_curr - t_prev)
                    if dt <= 0:
                        logger.warning(
                            "Skip speed computation with non-positive dt=%s for ROI '%s'.",
                            dt,
                            key,
                        )
                        continue
                    vx = float(x_curr - x_prev) / dt
                    vy = float(y_curr - y_prev) / dt
                    speed = float(np.hypot(vx, vy))
                    speeds.append([speed, float(t_curr)])

                path_log[path_id] = {
                    "start": coordinates[0][2],
                    "end": coordinates[-1][2],
                    "coordinates": coordinates,
                    "speed": speeds,
                    "length": lengths,
                    "height": heights,
                }

                length_by_time = {float(time): float(length) for length, time in lengths}
                for x, y, time in coordinates:
                    time = float(time)
                    active_fingers_by_time.setdefault(time, []).append(
                        {
                            "x": float(x),
                            "y": float(y),
                            "length": length_by_time.get(time, 0.0),
                        }
                    )

            statistics = {}
            for time in sorted(active_fingers_by_time):
                active_fingers = active_fingers_by_time[time]
                distances = []
                for i, finger_i in enumerate(active_fingers):
                    for finger_j in active_fingers[i + 1 :]:
                        distances.append(
                            float(
                                np.hypot(
                                    finger_j["x"] - finger_i["x"],
                                    finger_j["y"] - finger_i["y"],
                                )
                            )
                        )
                lengths_at_time = [float(finger["length"]) for finger in active_fingers]
                statistics[time] = {
                    "distances": distances,
                    "lengths": lengths_at_time,
                }

            path_log["statistics"] = statistics

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
                            "number_tips": number_tips,
                            "number_fjords": number_fjords,
                            "roi_width": roi_width,
                            "tip_frequency": tip_frequency,
                            "tip_wavelength": tip_wavelength,
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
