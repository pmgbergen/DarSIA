"""Template for finger analysis."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np
import pandas as pd

import darsia
from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    infer_require_color_to_mass_from_config,
    prepare_analysis_context,
)
from darsia.presets.workflows.analysis.progress import (
    AnalysisProgressEvent,
    publish_image_progress,
)
from darsia.presets.workflows.analysis.streaming import publish_stream_images
from darsia.presets.workflows.mode_resolution import mode_requires_color_to_mass
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
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
    progress_callback: Callable[[AnalysisProgressEvent], None] | None = None,
) -> None:
    """Segmentation analysis using pre-prepared context.

    Args:
        ctx: Pre-prepared analysis context with color_to_mass_analysis initialized.
        show: Whether to show the images.

    """
    assert ctx.config.analysis is not None
    assert ctx.config.analysis.fingers is not None

    fluidflower = ctx.fluidflower
    image_paths = ctx.image_paths

    # Extract finger analysis config (asserted not None above)
    fingers_config = ctx.config.analysis.fingers.config
    assert fingers_config.roi is not None
    requires_color_to_mass = mode_requires_color_to_mass(fingers_config.mode)
    if requires_color_to_mass and ctx.color_to_mass_analysis is None:
        raise ValueError(
            "Fingers config uses color-to-mass modes, but color-to-mass analysis "
            "is not initialized."
        )
    color_to_mass_analysis = ctx.color_to_mass_analysis
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
        (results_folder / "paths" / key).mkdir(parents=True, exist_ok=True)

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

    # Dictionary to collect path statistics for all ROIs.
    path_statistics = {}
    path_statistics["paths"] = {
        key: {"roi": roi_config.roi.tolist()}
        for key, roi_config in fingers_config.roi.items()
    }
    path_statistics["times"] = []
    path_statistics["images"] = []

    # Config of plotting.
    # TODO enable control from config.
    plot_format = ".png"
    contour_color = "w"
    peak_color = "r"
    peak_size = 5
    contour_linewidth = 0.5
    plot_boundary = False
    boundary_color = "y"
    boundary_linewidth = 2
    highlight_roi = False

    # Loop over images and analyze
    step_started_at = monotonic()
    image_total = len(image_paths)
    for image_index, path in enumerate(image_paths, start=1):
        image_started_at = monotonic()
        # Extract color signal and assign mass
        img = fluidflower.read_image(path)
        mass_analysis_result = (
            color_to_mass_analysis(img) if requires_color_to_mass else None
        )

        # Produce contour images
        segmentation = segmentation_analysis(
            img,
            saturation_g=(
                mass_analysis_result.saturation_g if mass_analysis_result else None
            ),
            concentration_aq=(
                mass_analysis_result.concentration_aq if mass_analysis_result else None
            ),
            mass=mass_analysis_result.mass if mass_analysis_result else None,
            mass_analysis_result=mass_analysis_result,
            colorrange_config=getattr(ctx.config, "colorrange", None),
        )

        stream_images: dict[str, Any] | None = None
        if stream_callback is not None:
            stream_images = {
                "fingers_source_image": img,
                "fingers_segmentation": segmentation,
            }

        # Extract time from image.
        image_time = img.time
        path_statistics["times"].append(float(image_time))
        path_statistics["images"].append(path.name)

        for key, roi_config in fingers_config.roi.items():
            # Perform finger analysis if configured
            contour_analysis.load(
                img, segmentation, roi=roi_config.roi, fill_holes=False
            )

            # Extract contour
            contours = contour_analysis.contours()

            # Determine various contour values.
            contour_length = contour_analysis.length()
            peaks, valleys = contour_analysis.local_extrema()
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
            tips_path = (results_folder / "tips" / key / f"{path.stem}").with_suffix(
                plot_format
            )
            contour_analysis.plot_peaks(
                img,
                peaks,
                roi_config.roi,
                contours=contours,
                path=tips_path,
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
            contour_evolution_times[key].append(float(img.time))

            # Plotting.
            paths_path = (results_folder / "paths" / key / f"{path.stem}").with_suffix(
                plot_format
            )
            contour_evolution_analysis[key].plot_paths(
                img,
                roi=roi_config.roi,
                path=paths_path,
                show=show,
            )

            # Fetch ROI top-left corner in pixel coordinates for path coordinate conversion.
            roi_top_left_row = 0
            roi_top_left_col = 0
            if roi_config.roi is not None:
                roi_pixels = roi_config.roi.to_voxel(img.coordinatesystem)
                roi_top_left_row = int(np.min(roi_pixels[:, 0]))
                roi_top_left_col = int(np.min(roi_pixels[:, 1]))

            # Process paths to extract statistics.
            path_log = {}
            active_fingers_by_time = {}
            for finger_path in contour_evolution_analysis[key].paths:
                if len(finger_path) == 0:
                    continue

                # Generate unique path ID based on start time and peak
                # with suffix if needed to avoid duplicates.
                start_unit = finger_path[0]
                path_id = f"path_t{int(start_unit.time)}_p{int(start_unit.peak)}"
                path_id_base = path_id
                suffix = 1
                while path_id in path_log:
                    path_id = f"{path_id_base}_{suffix}"
                    suffix += 1

                # Collect times.
                times = []
                for unit in finger_path:
                    time_index = int(unit.time)
                    if 0 <= time_index < len(contour_evolution_times[key]):
                        unit_time = float(contour_evolution_times[key][time_index])
                    times.append(unit_time)

                # Convert path coordinates from pixel to physical units and collect times.
                coordinates = []
                for unit in finger_path:
                    # NOTE/TODO: Flip in row/col (see ContourAnalysis...)
                    local_row = int(unit.position[1])
                    local_col = int(unit.position[0])
                    pixel_row = local_row + roi_top_left_row
                    pixel_col = local_col + roi_top_left_col
                    coordinate = img.coordinatesystem.coordinate(
                        darsia.Voxel([pixel_row, pixel_col])
                    )
                    coordinates.append(coordinate.tolist())

                if len(coordinates) == 0:
                    continue

                # Compute travel distances.
                travel_distances = [0.0]
                vertical_travel_distances = [0.0]
                cumulative_length = 0.0
                cummulative_vertical_length = 0.0
                for index in range(1, len(coordinates)):
                    x_prev, y_prev = coordinates[index - 1]
                    x_curr, y_curr = coordinates[index]
                    cumulative_length += float(
                        np.hypot(x_curr - x_prev, y_curr - y_prev)
                    )
                    cummulative_vertical_length += float(np.abs(y_curr - y_prev))
                    travel_distances.append(cumulative_length)
                    vertical_travel_distances.append(cummulative_vertical_length)

                # Compute speeds.
                velocities = []
                speeds = []
                vertical_speeds = []
                for index in range(1, len(coordinates)):
                    x_prev, y_prev = coordinates[index - 1]
                    x_curr, y_curr = coordinates[index]
                    t_prev = times[index - 1]
                    t_curr = times[index]
                    dt = float(t_curr - t_prev)
                    if dt <= 0:
                        logger.warning(
                            "Skip speed computation with non-positive dt=%s for ROI '%s'.",
                            dt,
                            key,
                        )
                        speeds.append(float("nan"))
                        continue
                    vx = float(x_curr - x_prev) / dt
                    vy = float(y_curr - y_prev) / dt

                    velocities.append([vx, vy])

                    speed = float(np.sqrt(vx**2 + vy**2))
                    speeds.append(speed)

                    vertical_speed = float(vy)
                    vertical_speeds.append(vertical_speed)

                # Sanity check on length of lists.
                assert (
                    len(times)
                    == len(coordinates)
                    == len(travel_distances)
                    == len(vertical_travel_distances)
                )
                assert (
                    len(velocities)
                    == len(speeds)
                    == len(vertical_speeds)
                    == len(coordinates) - 1
                )

                # Collect path log for this path.
                path_log[path_id] = {
                    "start": times[0],
                    "end": times[-1],
                    "time": times,
                    "coordinates": coordinates,
                    "velocities": velocities,
                    "speed": speeds,
                    "vertical_speed": vertical_speeds,
                    "travel_distance": travel_distances,
                    "vertical_travel_distance": vertical_travel_distances,
                }

                # Aggregate active fingers by time for statistics.
                for (x, _), time, travel_distance in zip(
                    coordinates, times, travel_distances
                ):
                    active_fingers_by_time.setdefault(time, []).append(
                        {
                            "x": float(x),
                            "travel_distance": float(travel_distance),
                            "speed": float(speeds[-1]) if speeds else float("nan"),
                            "vertical_speed": (
                                float(vertical_speeds[-1])
                                if vertical_speeds
                                else float("nan")
                            ),
                        }
                    )

            # Compute statistics for this ROI based on active fingers at each time.
            statistics = {}
            for time in sorted(active_fingers_by_time):
                active_fingers = active_fingers_by_time[time]

                # Compute horizontal distances between active fingers at this time.
                x_coords_sorted = sorted(
                    float(finger["x"]) for finger in active_fingers
                )
                dist_x = []
                if len(x_coords_sorted) > 1:
                    dist_x = np.diff(x_coords_sorted).tolist()

                # Compute lengths of active fingers at this time (using travel distance as a
                # proxy for length).
                travel_distances_at_time = [
                    float(finger["travel_distance"]) for finger in active_fingers
                ]

                # Compute the velocities of active fingers at this time
                # (using the last velocity before or at this time).
                speeds_at_time = [finger["speed"] for finger in active_fingers]
                speeds_y_at_time = [
                    finger["vertical_speed"] for finger in active_fingers
                ]

                # Count new fingers that have zero travel distance at this time
                # (i.e., just appeared).
                new_fingers = int(
                    sum(
                        np.isclose(length, 0.0, atol=1e-10)
                        for length in travel_distances_at_time
                    )
                )

                statistics[time] = {
                    "horizontal_distances": dist_x,
                    "travel_distances": travel_distances_at_time,
                    "finger_speeds": speeds_at_time,
                    "finger_vertical_speeds": speeds_y_at_time,
                    "new_fingers": new_fingers,
                    "number_fingers": len(active_fingers),
                    "contour_length": contour_length,
                    "roi_width": roi_width,
                    "tip_frequency": tip_frequency,
                    "tip_wavelength": tip_wavelength,
                }

            path_log["statistics"] = statistics

            # Collect path log for this ROI into the overall statistics dictionary.
            path_statistics["paths"][key].update(path_log)

            # Save overall path statistics for all ROIs to a single JSON file.
            with open(results_folder / "statistics.json", "w") as f:
                json.dump(path_statistics, f, indent=2)

            # Save tabular statistics to DataFrame and CSV.
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
            df.to_csv(results_folder / "statistics.csv", index=False)

            if stream_images is not None:
                tips_plot = cv2.cvtColor(
                    cv2.imread(str(tips_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
                )
                if tips_plot is not None:
                    stream_images[f"fingers_tips_{key}"] = tips_plot
                paths_plot = cv2.cvtColor(
                    cv2.imread(str(paths_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
                )
                if paths_plot is not None:
                    stream_images[f"fingers_paths_{key}"] = paths_plot

        if stream_images is not None:
            publish_stream_images(
                stream_callback=stream_callback,
                image_payload=stream_images,
                logger=logger,
                error_message=f"Failed to stream finger previews for image '{path}'.",
            )
        publish_image_progress(
            progress_callback,
            step="fingers",
            image_path=str(path.resolve()),
            image_index=image_index,
            image_total=image_total,
            image_duration_s=monotonic() - image_started_at,
            step_elapsed_s=monotonic() - step_started_at,
        )


def analysis_fingers(
    cls: type[Rig],
    path: Path | list[Path],
    show: bool = False,
    all: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
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
        require_color_to_mass=infer_require_color_to_mass_from_config(
            path,
            include_fingers=True,
        ),
    )
    analysis_fingers_from_context(ctx, show=show, stream_callback=stream_callback)
