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
from darsia.single_image_analysis.contouranalysis import ContourAnalysis
from darsia.single_image_analysis.path_evolution_analysis import PathEvolutionAnalysis
from darsia.single_image_analysis.skeleton_analysis import SkeletonAnalysis

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
    contour_analysis = ContourAnalysis(
        contour_smoother=fingers_config.contour_smoother,
        reduce_to_main_contour=fingers_config.reduce_to_main_contour,
    )
    skeleton_analysis = SkeletonAnalysis(
        contour_smoother=fingers_config.contour_smoother,
        reduce_to_main_contour=fingers_config.reduce_to_main_contour,
    )

    # Keep evolution state per ROI to prevent mixing path histories across ROIs.
    evolution_times = {key: [] for key in fingers_config.roi}
    peak_evolution_analysis = {
        key: PathEvolutionAnalysis() for key in fingers_config.roi
    }

    # Data management.
    results_folder = ctx.config.analysis.fingers.folder
    results_folder.mkdir(parents=True, exist_ok=True)
    for key in fingers_config.roi:
        (results_folder / "tips" / key).mkdir(parents=True, exist_ok=True)
        (results_folder / "fjords" / key).mkdir(parents=True, exist_ok=True)
        (results_folder / "paths" / key).mkdir(parents=True, exist_ok=True)
        (results_folder / "skeleton" / key).mkdir(parents=True, exist_ok=True)

    # DataFrame to store results.
    df = pd.DataFrame(
        columns=[
            "time",
            "key",
            "image",
            "contour_length",
            "number_tips",
            "number_fjords",
            "number_skeleton_leaves",
            "number_skeleton_junctions",
            "roi_width",
            "finger_frequency",
            "finger_wavelength",
        ]
    )

    # Dictionary to collect path statistics for all ROIs.
    peak_path_statistics = {}
    peak_path_statistics["paths"] = {
        key: {"roi": roi_config.roi.tolist()}
        for key, roi_config in fingers_config.roi.items()
    }
    peak_path_statistics["times"] = []
    peak_path_statistics["images"] = []

    # Config of plotting.
    # TODO enable control from config.
    contour_color = "w"
    peak_color = "r"
    peak_size = 5
    contour_linewidth = 0.5
    plot_boundary = False
    boundary_color = "y"
    boundary_linewidth = 2
    highlight_roi = False
    skeleton_color = "w"
    skeleton_linewidth = 1

    # Loop over images and analyze
    step_started_at = monotonic()
    image_total = len(image_paths)

    import random

    # random.shuffle(image_paths)

    for image_index, path in enumerate(image_paths, start=1):
        image_started_at = monotonic()
        # Extract color signal and assign mass
        try:
            img = fluidflower.read_image(path)
        except Exception as e:
            logger.error(f"Failed to read image '{path}': {e}")
            continue

        # Perform color-to-mass analysis if any layer requires it.
        mass_analysis_result = (
            color_to_mass_analysis(img) if requires_color_to_mass else None
        )

        # Produce contour images.
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
            color_embedding_registry=getattr(ctx.config, "color", None),
            color_embedding_runtime=getattr(ctx, "color_embedding_runtime", None),
        )

        stream_images: dict[str, Any] | None = None
        if stream_callback is not None:
            stream_images = {
                "fingers_source_image": img,
                "fingers_segmentation": segmentation,
            }

        # Extract time from image.
        image_time = img.time
        peak_path_statistics["times"].append(float(image_time))
        peak_path_statistics["images"].append(path.name)

        # Contour and skeleton analysis.
        for key, roi_config in fingers_config.roi.items():
            # Perform finger analysis if configured
            contour_analysis.load(
                img=img,
                mask=segmentation,
                roi=roi_config.roi,
                fill_holes=fingers_config.fill_holes,
            )

            # Perform skeleton analysis if configured
            skeleton_analysis.load(
                img=img,
                mask=segmentation,
                roi=roi_config.roi,
                fill_holes=fingers_config.fill_holes,
            )

            # Extract contour and skeleton data.
            contours = contour_analysis.contours()
            skeleton = skeleton_analysis.skeleton(contours)

            # Determine various contour values.
            contour_length = contour_analysis.length()
            peaks, valleys = contour_analysis.local_extrema()
            number_tips = contour_analysis.number_peaks()
            number_fjords = contour_analysis.number_valleys()
            roi_width = float(np.abs(roi_config.roi[1, 0] - roi_config.roi[0, 0]))
            peak_frequency = np.nan
            peak_wavelength = np.nan
            if roi_width > 0:
                peak_frequency = float(number_tips) / roi_width
                if number_tips > 0:
                    peak_wavelength = roi_width / float(number_tips)
            else:
                logger.warning(
                    "Skip frequency/wavelength computation due to non-positive"
                    " ROI width for ROI '%s'.",
                    key,
                )

            # Determine skeleton values.
            # TODO include in config.
            leaves, junctions, base_junctions = skeleton_analysis.leaves_and_junctions(
                skeleton, min_branch_length=0.05, max_group_distance=0.01
            )
            number_leaves = leaves.shape[0]
            number_junctions = junctions.shape[0]
            number_base_junctions = base_junctions.shape[0]
            leaf_frequency = np.nan
            leaf_wavelength = np.nan
            if roi_width > 0:
                leaf_frequency = float(number_leaves) / roi_width
                if number_leaves > 0:
                    leaf_wavelength = roi_width / float(number_leaves)
            else:
                logger.warning(
                    "Skip leaf frequency/wavelength computation due to non-positive"
                    " ROI width for ROI '%s'.",
                    key,
                )

            # Plot finger peaks and contours.
            tips_path = (results_folder / "tips" / key / f"{path.stem}").with_suffix(
                ".png"
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

            # Plot skeleton with leaves and junctions.
            skeleton_path = (
                results_folder / "skeleton" / key / f"{path.stem}"
            ).with_suffix(".png")

            skeleton_analysis.plot_skeleton(
                img=img,
                skeleton=skeleton,
                leaves=leaves,
                junctions=junctions,
                base_junctions=base_junctions,
                roi=roi_config.roi,
                path=skeleton_path,
                show=show,
                **{
                    "skeleton_color": skeleton_color,
                    "skeleton_linewidth": skeleton_linewidth,
                    "leaf_color": "g",
                    "leaf_size": 20,
                    "junction_color": "m",
                    "junction_size": 20,
                    "base_junction_color": "b",
                    "base_junction_size": 20,
                },
            )

            # Update evolution analysis.
            evolution_times[key].append(float(img.time))
            peak_evolution_analysis[key].add(points=peaks, time=img.time)

            # Plotting.
            peak_paths_path = (
                results_folder / "paths" / key / f"{path.stem}"
            ).with_suffix(".png")
            peak_evolution_analysis[key].plot_paths(
                img,
                roi=roi_config.roi,
                path=peak_paths_path,
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
            peak_path_log = {}
            active_peaks_by_time = {}
            for peak_path in peak_evolution_analysis[key].paths:
                if len(peak_path) == 0:
                    continue

                # Generate unique path ID based on start time and peak
                # with suffix if needed to avoid duplicates.
                start_unit = peak_path[0]
                path_id = f"path_t{int(start_unit.time)}_p{int(start_unit.id)}"
                path_id_base = path_id
                suffix = 1
                while path_id in peak_path_log:
                    path_id = f"{path_id_base}_{suffix}"
                    suffix += 1

                # Collect times.
                times = []
                for unit in peak_path:
                    time_index = int(unit.time)
                    if 0 <= time_index < len(evolution_times[key]):
                        unit_time = float(evolution_times[key][time_index])
                    times.append(unit_time)

                # Convert path coordinates from pixel to physical units and collect times.
                coordinates = []
                for unit in peak_path:
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
                        velocities.append([float("nan"), float("nan")])
                        vertical_speeds.append(float("nan"))
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
                peak_path_log[path_id] = {
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
                    active_peaks_by_time.setdefault(time, []).append(
                        {
                            "origin": (
                                float(coordinates[0][0]),
                                float(coordinates[0][1]),
                                float(times[0]),
                            ),
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
            times = set(active_peaks_by_time.keys())
            num_active_peaks = 0
            num_new_peaks = 0
            num_continuing_peaks = 0
            num_ending_peaks = 0
            for time_index, time in enumerate(sorted(times)):
                active_peaks = active_peaks_by_time[time]

                # Num active fingers at this time.
                num_active_peaks = len(active_peaks)

                # Compute horizontal distances between active fingers at this time.
                x_coords_sorted = sorted(float(finger["x"]) for finger in active_peaks)
                dist_x = []
                if len(x_coords_sorted) > 1:
                    dist_x = np.diff(x_coords_sorted).tolist()

                # Compute lengths of active fingers at this time (using travel distance as a
                # proxy for length).
                travel_distances_at_time = [
                    float(finger["travel_distance"]) for finger in active_peaks
                ]

                # Compute the velocities of active fingers at this time
                # (using the last velocity before or at this time).
                speeds_at_time = [finger["speed"] for finger in active_peaks]
                speeds_y_at_time = [finger["vertical_speed"] for finger in active_peaks]

                # Count new fingers that have zero travel distance at this time
                # (i.e., just appeared).
                new_peaks = int(
                    sum(
                        np.isclose(length, 0.0, atol=1e-10)
                        for length in travel_distances_at_time
                    )
                )

                # Continue new fingers.
                num_continuing_peaks = 0
                current_origin = [finger["origin"] for finger in active_peaks]
                if time_index > 0:
                    prev_active_peaks = active_peaks_by_time.get(
                        evolution_times[key][time_index - 1], []
                    )
                    prev_origin = [finger["origin"] for finger in prev_active_peaks]

                    # Use np.allclose on origins to determine if any active fingers at this time are continuations of fingers from the previous time point.
                    num_continuing_peaks = 0
                    for curr in current_origin:
                        if any(
                            np.allclose(curr, prev, atol=1e-10) for prev in prev_origin
                        ):
                            num_continuing_peaks += 1

                num_new_peaks = num_active_peaks - num_continuing_peaks

                # Continue ending fingers.
                num_ending_peaks = 0
                if time_index < len(times) - 1:
                    next_active_peaks = active_peaks_by_time.get(
                        evolution_times[key][time_index + 1], []
                    )

                    next_origin = [finger["origin"] for finger in next_active_peaks]
                    for curr in current_origin:
                        if not any(
                            np.allclose(curr, next, atol=1e-10) for next in next_origin
                        ):
                            num_ending_peaks += 1

                statistics[time] = {
                    "horizontal_distances": dist_x,
                    "travel_distances": travel_distances_at_time,
                    "speeds": speeds_at_time,
                    "vertical_speeds": speeds_y_at_time,
                    "number_new_units_based_on_travel_distance": new_peaks,
                    "number_new_units": num_new_peaks,
                    "number_continuing_units": num_continuing_peaks,
                    "number_ending_units": num_ending_peaks,
                    "number_active_units": num_active_peaks,
                    "roi_width": roi_width,
                    "frequency": peak_frequency,
                    "wavelength": peak_wavelength,
                    "contour_length": contour_length,
                }
            peak_path_log["statistics"] = statistics

            # Collect path log for this ROI into the overall statistics dictionary.
            peak_path_statistics["paths"][key].update(peak_path_log)

            # Save overall path statistics for all ROIs to a single JSON file.
            with open(results_folder / "statistics.json", "w") as f:
                json.dump(peak_path_statistics, f, indent=2)

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
                            "finger_frequency": peak_frequency,
                            "finger_wavelength": peak_wavelength,
                            "number_active_peaks": num_active_peaks,
                            "number_new_peaks": num_new_peaks,
                            "number_continuing_peaks": num_continuing_peaks,
                            "number_ending_peaks": num_ending_peaks,
                            # "number_splitting_fingers": num_new_junctions,
                            # "number_merging_fingers": num_ending_junctions,
                            # "number_active_split": num_active_junctions,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
            df.to_csv(results_folder / "statistics.csv", index=False)

            if stream_images is not None:
                tips_plot_raw = cv2.imread(str(tips_path), cv2.IMREAD_UNCHANGED)
                if tips_plot_raw is not None:
                    stream_images[f"fingers_tips_{key}"] = cv2.cvtColor(
                        tips_plot_raw, cv2.COLOR_BGR2RGB
                    )
                paths_plot_raw = cv2.imread(str(peak_paths_path), cv2.IMREAD_UNCHANGED)
                if paths_plot_raw is not None:
                    stream_images[f"fingers_paths_{key}"] = cv2.cvtColor(
                        paths_plot_raw, cv2.COLOR_BGR2RGB
                    )
                skeleton_plot_raw = cv2.imread(str(skeleton_path), cv2.IMREAD_UNCHANGED)
                if skeleton_plot_raw is not None:
                    stream_images[f"fingers_skeleton_{key}"] = cv2.cvtColor(
                        skeleton_plot_raw, cv2.COLOR_BGR2RGB
                    )

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
