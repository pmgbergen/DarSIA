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
from darsia.presets.workflows.segmentation_contours import (  # GradientBasedSegmentation,
    SimpleSegmentation,
)
from darsia.single_image_analysis.contouranalysis import (
    ContourAnalysis,
    _corners_of_roi,
)
from darsia.single_image_analysis.path_evolution_analysis import PathEvolutionAnalysis
from darsia.single_image_analysis.skeleton_analysis import SkeletonAnalysis

logger = logging.getLogger(__name__)


def _extract_lower_arc(contour: np.ndarray) -> np.ndarray:
    """
    Extract the lower arc of a circle-like contour.

    Finds the leftmost and rightmost points, then keeps only the arc
    connecting them via the bottom (higher row indices = lower y-values visually).
    Removes the upper arc.

    Args:
        contour: OpenCV contour format (N, 1, 2)

    Returns:
        Lower arc contour in same format (N, 1, 2)
    """
    # Convert from (N, 1, 2) to (N, 2) for easier indexing
    contour_2d = contour.squeeze()

    if len(contour_2d.shape) == 1:
        return contour  # Single point, return as-is

    cols = contour_2d[:, 0].astype(float)
    rows = contour_2d[:, 1].astype(float)

    # Find leftmost and rightmost points
    left_idx = np.argmin(cols)
    right_idx = np.argmax(cols)

    # Get the two possible arcs connecting left → right
    if left_idx < right_idx:
        # Direct path: left → right
        arc1_indices = np.arange(left_idx, right_idx + 1)
        # Wraparound path: right → end + start → left
        arc2_indices = np.concatenate(
            [np.arange(right_idx, len(cols)), np.arange(0, left_idx + 1)]
        )
    else:
        # Direct path: right → left
        arc1_indices = np.arange(right_idx, left_idx + 1)
        # Wraparound path: left → end + start → right
        arc2_indices = np.concatenate(
            [np.arange(left_idx, len(cols)), np.arange(0, right_idx + 1)]
        )

    # The "lower" arc has higher mean row value (lower visually = higher row indices)
    arc1_rows = rows[arc1_indices]
    arc2_rows = rows[arc2_indices]

    mean_row_arc1 = np.mean(arc1_rows)
    mean_row_arc2 = np.mean(arc2_rows)

    # Select the arc with higher mean row (visually lower = larger row indices)
    if mean_row_arc1 > mean_row_arc2:
        lower_indices = arc1_indices
    else:
        lower_indices = arc2_indices

    # Extract lower arc and sort by column for clean output
    lower_cols = cols[lower_indices]
    lower_rows = rows[lower_indices]

    sort_idx = np.argsort(lower_cols)
    lower_cols_sorted = lower_cols[sort_idx]
    lower_rows_sorted = lower_rows[sort_idx]

    # Convert back to (N, 1, 2) format
    lower_arc = np.column_stack([lower_cols_sorted, lower_rows_sorted]).astype(np.int32)
    lower_arc = lower_arc.reshape(-1, 1, 2)

    return lower_arc


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
    if fingers_config.include_gradient_based_analysis:
        # gradient_based_segmentation_analysis = GradientBasedSegmentation(
        gradient_based_segmentation_analysis = SimpleSegmentation(
            mode=fingers_config.gradient_mode,
            threshold=0.5,  # fingers_config.threshold
        )
        gradient_based_contour_analysis = ContourAnalysis(
            contour_smoother=fingers_config.contour_smoother,
            reduce_to_main_contour=fingers_config.reduce_to_main_contour,
        )

    categories = ["peak", "fjord", "leaf", "junction", "base_junction"]
    if fingers_config.include_gradient_based_analysis:
        categories.append("interface")
    path_categories = [
        "paths",
        "fjord_paths",
        "leaf_paths",
        "junction_paths",
        "base_junction_paths",
    ]
    if fingers_config.include_gradient_based_analysis:
        path_categories.append("interface_paths")

    # Keep evolution state per ROI to prevent mixing path histories across ROIs.
    evolution_times = {key: [] for key in fingers_config.roi}
    evolution_analysis = {}
    for key in categories:
        evolution_analysis[key] = {
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
        (results_folder / "skeleton-leaf-paths" / key).mkdir(
            parents=True, exist_ok=True
        )
        (results_folder / "skeleton-junction-paths" / key).mkdir(
            parents=True, exist_ok=True
        )
        (results_folder / "skeleton-base-junction-paths" / key).mkdir(
            parents=True, exist_ok=True
        )
        if fingers_config.include_gradient_based_analysis:
            (results_folder / "interface" / key).mkdir(parents=True, exist_ok=True)
            (results_folder / "interface-contour" / key).mkdir(
                parents=True, exist_ok=True
            )
            (results_folder / "interface-contour-npy" / key).mkdir(
                parents=True, exist_ok=True
            )
            (results_folder / "interface-paths" / key).mkdir(
                parents=True, exist_ok=True
            )

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
    path_statistics = {}
    for key in path_categories:
        path_statistics[key] = {
            key: {"roi": roi_config.roi.tolist()}
            for key, roi_config in fingers_config.roi.items()
        }
    path_statistics["times"] = []
    path_statistics["images"] = []

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

        # Extract gradient-based segmentation if configured.
        if fingers_config.include_gradient_based_analysis:
            gradient_based_segmentation = gradient_based_segmentation_analysis(
                img,
                saturation_g=(
                    mass_analysis_result.saturation_g if mass_analysis_result else None
                ),
                concentration_aq=(
                    mass_analysis_result.concentration_aq
                    if mass_analysis_result
                    else None
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
            if fingers_config.include_gradient_based_analysis:
                stream_images["fingers_gradient_based_segmentation"] = (
                    gradient_based_segmentation
                )

        # Extract time from image.
        image_time = img.time
        path_statistics["times"].append(float(image_time))
        path_statistics["images"].append(path.name)

        # Contour and skeleton analysis.
        for key, roi_config in fingers_config.roi.items():
            # Perform finger analysis if configured
            contour_analysis.load(
                img=img,
                mask=segmentation,
                roi=roi_config.roi,
                fill_holes=fingers_config.fill_holes,
            )

            # Perform contour analysis on gradient-based segmentation if configured.
            if fingers_config.include_gradient_based_analysis:
                gradient_based_contour_analysis.load(
                    img=img,
                    mask=gradient_based_segmentation,
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
            if fingers_config.include_gradient_based_analysis:
                gradient_based_contours = gradient_based_contour_analysis.contours()
                lower_gradient_based_contours = [
                    _extract_lower_arc(contour) for contour in gradient_based_contours
                ]

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
                skeleton, max_group_distance=0.01
            )
            number_leaves = leaves.shape[0]
            number_junctions = junctions.shape[0]
            number_base_junctions = base_junctions.shape[0]

            # Gradient based contour values if configured.
            if fingers_config.include_gradient_based_analysis:
                # Determine various contour values.
                # gradient_based_contour_length = gradient_based_contour_analysis.length()
                gradient_based_peaks, _ = gradient_based_contour_analysis.local_extrema(
                    contours=lower_gradient_based_contours
                )
                # gradient_based_number_tips = (
                #     gradient_based_contour_analysis.number_peaks()
                # )
                # gradient_based_number_fjords = (
                #     gradient_based_contour_analysis.number_valleys()
                # )

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

            if fingers_config.include_gradient_based_analysis:
                gradient_path = (
                    results_folder / "interface" / key / f"{path.stem}"
                ).with_suffix(".png")
                gradient_based_contour_analysis.plot_peaks(
                    img,
                    gradient_based_peaks,
                    roi_config.roi,
                    contours=lower_gradient_based_contours,
                    path=gradient_path,
                    show=show,
                    **{
                        "peak_color": peak_color,
                        "peak_size": peak_size,
                        "contour_color": contour_color,
                        "contour_alpha": 0.5,
                        "contour_linewidth": contour_linewidth,
                        "plot_boundary": plot_boundary,
                        "boundary_color": boundary_color,
                        "boundary_linewidth": boundary_linewidth,
                        "highlight_roi": highlight_roi,
                    },
                )
                gradient_path_contour = (
                    results_folder / "interface-contour" / key / f"{path.stem}"
                ).with_suffix(".png")
                gradient_based_contour_analysis.plot_peaks(
                    img,
                    gradient_based_peaks,
                    roi_config.roi,
                    contours=lower_gradient_based_contours,
                    path=gradient_path_contour,
                    show=show,
                    **{
                        "peak_color": peak_color,
                        "peak_size": 0,
                        "contour_color": contour_color,
                        "contour_linewidth": contour_linewidth,
                        "contour_alpha": 0.5,
                        "plot_boundary": plot_boundary,
                        "boundary_color": boundary_color,
                        "boundary_linewidth": boundary_linewidth,
                        "highlight_roi": highlight_roi,
                    },
                )

                # Export the lower_gradient_based_contours as .npy for potential further
                # analysis.
                gradient_contour_npy_path = (
                    results_folder / "interface-contour-npy" / key / f"{path.stem}"
                ).with_suffix(".npy")
                top_left_roi_pixel, bottom_right_roi_pixel = _corners_of_roi(
                    fluidflower.baseline, roi_config.roi
                )
                # Stack translated contour, use top_left_roi_pixel for translation.
                # NOTE: Need to flip order due to origin from cv2.
                correct_contour = [
                    np.stack(
                        [
                            contour[:, 0, 1] + top_left_roi_pixel[1],
                            contour[:, 0, 0] + top_left_roi_pixel[0],
                        ],
                        axis=-1,
                    )
                    for contour in lower_gradient_based_contours
                ]
                # Convert the voxel coordinates back to physical coordinates for export.
                lower_gradient_based_contours_physics = [
                    img.coordinatesystem.coordinate(contour.squeeze()).astype(
                        np.float32
                    )
                    for contour in correct_contour
                ]
                np.save(
                    gradient_contour_npy_path, lower_gradient_based_contours_physics
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
            evolution_analysis["peak"][key].add(points=peaks, time=img.time)
            evolution_analysis["fjord"][key].add(points=valleys, time=img.time)

            if fingers_config.include_gradient_based_analysis:
                evolution_analysis["interface"][key].add(
                    points=gradient_based_peaks, time=img.time
                )

            # Need to flip row/col for skeleton points to match contour/skeleton images.
            # Mainly for plotting.
            flipped_leaves = np.array([[point[0][1], point[0][0]] for point in leaves])
            flipped_junctions = np.array(
                [[point[0][1], point[0][0]] for point in junctions]
            )
            flipped_base_junctions = np.array(
                [[point[0][1], point[0][0]] for point in base_junctions]
            )
            evolution_analysis["leaf"][key].add(points=flipped_leaves, time=img.time)
            evolution_analysis["junction"][key].add(
                points=flipped_junctions, time=img.time
            )
            evolution_analysis["base_junction"][key].add(
                points=flipped_base_junctions, time=img.time
            )
            for category in categories:
                evolution_analysis[category][key].find_paths()

            # Plotting.
            peak_paths_path = (
                results_folder / "paths" / key / f"{path.stem}"
            ).with_suffix(".png")
            evolution_analysis["peak"][key].plot_paths(
                img,
                roi=roi_config.roi,
                path=peak_paths_path,
                show=show,
            )

            if fingers_config.include_gradient_based_analysis:
                gradient_paths_path = (
                    results_folder / "interface-paths" / key / f"{path.stem}"
                ).with_suffix(".png")
                evolution_analysis["interface"][key].plot_paths(
                    img,
                    roi=roi_config.roi,
                    path=gradient_paths_path,
                    show=show,
                )

            leaf_paths_path = (
                results_folder / "skeleton-leaf-paths" / key / f"{path.stem}"
            ).with_suffix(".png")
            evolution_analysis["leaf"][key].plot_paths(
                img,
                roi=roi_config.roi,
                path=leaf_paths_path,
                show=show,
            )
            junction_paths_path = (
                results_folder / "skeleton-junction-paths" / key / f"{path.stem}"
            ).with_suffix(".png")
            evolution_analysis["junction"][key].plot_paths(
                img,
                roi=roi_config.roi,
                path=junction_paths_path,
                show=show,
            )
            base_junction_paths_path = (
                results_folder / "skeleton-base-junction-paths" / key / f"{path.stem}"
            ).with_suffix(".png")
            evolution_analysis["base_junction"][key].plot_paths(
                img,
                roi=roi_config.roi,
                path=base_junction_paths_path,
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
            active_paths_by_time = {}
            for category in categories:
                path_log[category] = {}
                active_paths_by_time[category] = {}
                for _path in evolution_analysis[category][key].paths:
                    if len(_path) == 0:
                        continue

                    # Generate unique path ID based on start time and peak
                    # with suffix if needed to avoid duplicates.
                    start_unit = _path[0]
                    path_id = f"path_t{int(start_unit.time)}_p{int(start_unit.id)}"
                    path_id_base = path_id
                    suffix = 1
                    while path_id in path_log[category]:
                        path_id = f"{path_id_base}_{suffix}"
                        suffix += 1

                    # Collect times.
                    times = []
                    for unit in _path:
                        time_index = int(unit.time)
                        if 0 <= time_index < len(evolution_times[key]):
                            unit_time = float(evolution_times[key][time_index])
                        times.append(unit_time)

                    # Convert path coordinates from pixel to physical units and collect times.
                    coordinates = []
                    for unit in _path:
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
                    path_log[category][path_id] = {
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

                    # Aggregate active paths by time for statistics.
                    for (x, y), time, travel_distance in zip(
                        coordinates, times, travel_distances
                    ):
                        active_paths_by_time[category].setdefault(time, []).append(
                            {
                                "origin": (
                                    float(coordinates[0][0]),
                                    float(coordinates[0][1]),
                                    float(times[0]),
                                ),
                                "x": float(x),
                                "coordinate": [float(x), float(y)],
                                "travel_distance": float(travel_distance),
                                "speed": float(speeds[-1]) if speeds else float("nan"),
                                "vertical_speed": (
                                    float(vertical_speeds[-1])
                                    if vertical_speeds
                                    else float("nan")
                                ),
                            }
                        )

            # Compute statistics for this ROI based on active paths at each time.
            times = sorted(path_statistics["times"])
            statistics = {}
            num_paths = {}
            for category in categories:
                statistics[category] = {}
                num_paths[category] = {
                    "active": 0,
                    "new": 0,
                    "continuing": 0,
                    "ending": 0,
                }
            for time_index, time in enumerate(times):
                for category in categories:
                    if time not in active_paths_by_time[category]:
                        continue

                    # Fetch active paths at this time for this category.
                    active_paths = active_paths_by_time[category][time]

                    # Compute horizontal distances between active fingers at this time.
                    x_coords_sorted = sorted(
                        float(finger["x"]) for finger in active_paths
                    )
                    dist_x = []
                    if len(x_coords_sorted) > 1:
                        dist_x = np.diff(x_coords_sorted).tolist()

                    # Collect coordinates of active fingers at this time.
                    coordinates_at_time = [
                        finger["coordinate"] for finger in active_paths
                    ]

                    # Compute lengths of active fingers at this time (using travel distance as
                    # a proxy for length).
                    travel_distances_at_time = [
                        float(finger["travel_distance"]) for finger in active_paths
                    ]

                    # Compute the velocities of active fingers at this time
                    # (using the last velocity before or at this time).
                    speeds_at_time = [
                        finger["speed"]
                        for finger in active_paths
                        if not np.isnan(finger["speed"])
                    ]
                    speeds_y_at_time = [
                        finger["vertical_speed"]
                        for finger in active_paths
                        if not np.isnan(finger["vertical_speed"])
                    ]

                    # Count new paths that have zero travel distance at this time
                    # (i.e., just appeared).
                    num_new_paths_based_on_travel_distance = int(
                        sum(
                            np.isclose(length, 0.0, atol=1e-10)
                            for length in travel_distances_at_time
                        )
                    )

                    # Count active paths at this time.
                    num_paths[category]["active"] = len(active_paths)

                    # Count continuing paths.
                    current_origin = [finger["origin"] for finger in active_paths]
                    num_continuing_paths = 0
                    if time_index > 0:
                        prev_active_paths = active_paths_by_time[category].get(
                            evolution_times[key][time_index - 1], []
                        )
                        prev_origin = [finger["origin"] for finger in prev_active_paths]
                        # Use np.allclose on origins to determine if any active paths at
                        # this time are continuations of paths from the previous time point.
                        for curr in current_origin:
                            for prev in prev_origin:
                                if np.allclose(curr, prev, atol=1e-10):
                                    num_continuing_paths += 1
                                    break
                    num_paths[category]["continuing"] = num_continuing_paths

                    # Count ending paths.
                    num_ending_paths = 0
                    if time_index < len(times) - 1:
                        next_active_paths = active_paths_by_time[category].get(
                            evolution_times[key][time_index + 1], []
                        )

                        next_origin = [finger["origin"] for finger in next_active_paths]
                        # Use np.allclose on origins to determine if any active paths at
                        # this time are continuations of paths from the previous time point.
                        for curr in current_origin:
                            no_match = True
                            for next in next_origin:
                                if np.allclose(curr, next, atol=1e-10):
                                    no_match = False
                                    break
                            if no_match:
                                num_ending_paths += 1
                    num_paths[category]["ending"] = num_ending_paths

                    # Conclude on number of new paths.
                    num_paths[category]["new"] = (
                        num_paths[category]["active"]
                        - num_paths[category]["continuing"]
                    )

                    # NOTE: Use short name for style.
                    tmp_num = num_new_paths_based_on_travel_distance

                    statistics[category][time] = {
                        "horizontal_distances": dist_x,
                        "coordinates": coordinates_at_time,
                        "travel_distances": travel_distances_at_time,
                        "speeds": speeds_at_time,
                        "vertical_speeds": speeds_y_at_time,
                        "number_new_paths_based_on_travel_distance": tmp_num,
                        "number_new_paths": num_paths[category]["new"],
                        "number_continuing_paths": num_paths[category]["continuing"],
                        "number_ending_paths": num_paths[category]["ending"],
                        "number_active_paths": num_paths[category]["active"],
                        "roi_width": roi_width,
                        "frequency": peak_frequency,
                        "wavelength": peak_wavelength,
                        "contour_length": contour_length,
                    }

            # Include statistics in path log.
            for category in categories:
                path_log[category]["statistics"] = statistics[category]

            # Collect path log for this ROI into the overall statistics dictionary.
            path_statistics["paths"][key].update(path_log["peak"])
            path_statistics["fjord_paths"][key].update(path_log["fjord"])
            path_statistics["leaf_paths"][key].update(path_log["leaf"])
            path_statistics["junction_paths"][key].update(path_log["junction"])
            path_statistics["base_junction_paths"][key].update(
                path_log["base_junction"]
            )
            if fingers_config.include_gradient_based_analysis:
                path_statistics["interface_paths"][key].update(path_log["interface"])

            # Save overall path statistics for all ROIs to a single JSON file.
            with open(results_folder / "statistics.json", "w") as f:
                json.dump(path_statistics, f, indent=2)

            # Save tabular statistics to DataFrame and CSV.
            _data = {
                "time": img.time,
                "key": key,
                "image": path.name,
                "contour_length": contour_length,
                "number_tips": number_tips,
                "number_fjords": number_fjords,
                "number_leaves": number_leaves,
                "number_junctions": number_junctions,
                "number_base_junctions": number_base_junctions,
                "roi_width": roi_width,
                "finger_frequency": peak_frequency,
                "finger_wavelength": peak_wavelength,
                # Finger counting based on peaks/contours.
                "number_fingers": num_paths["peak"]["active"],
                "number_new_fingers": num_paths["peak"]["new"],
                "number_continuing_fingers": num_paths["peak"]["continuing"],
                "number_ending_fingers": num_paths["peak"]["ending"],
                # Finger counting based on skeleton leaves.
                "number_skeleton_leaves": num_paths["leaf"]["active"],
                "number_new_skeleton_leaves": num_paths["leaf"]["new"],
                "number_continuing_skeleton_leaves": num_paths["leaf"]["continuing"],
                "number_ending_skeleton_leaves": num_paths["leaf"]["ending"],
                # Number of base fingers based on skeleton base junctions.
                "number_base_fingers": num_paths["base_junction"]["active"],
                "number_new_base_fingers": num_paths["base_junction"]["new"],
                "number_continuing_base_fingers": num_paths["base_junction"][
                    "continuing"
                ],
                "number_ending_base_fingers": num_paths["base_junction"]["ending"],
                # Number of splitting fingers based on skeleton junctions.
                "number_splitting_fingers": num_paths["junction"]["active"],
                "number_new_splitting_fingers": num_paths["junction"]["new"],
                "number_continuing_splitting_fingers": num_paths["junction"][
                    "continuing"
                ],
                "number_ending_splitting_fingers": num_paths["junction"]["ending"],
            }
            if fingers_config.include_gradient_based_analysis:
                _data.update(
                    {
                        # Finger counting based on gradient-based peaks/contours if
                        # configured.
                        "number_interface_fingers": num_paths["interface"]["active"],
                        "number_new_interface_fingers": num_paths["interface"]["new"],
                        "number_continuing_interface_fingers": num_paths["interface"][
                            "continuing"
                        ],
                        "number_ending_interface_fingers": num_paths["interface"][
                            "ending"
                        ],
                    }
                )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        _data,
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
                leaf_paths_plot_raw = cv2.imread(
                    str(leaf_paths_path), cv2.IMREAD_UNCHANGED
                )
                if leaf_paths_plot_raw is not None:
                    stream_images[f"fingers_leaf_paths_{key}"] = cv2.cvtColor(
                        leaf_paths_plot_raw, cv2.COLOR_BGR2RGB
                    )
                junction_paths_plot_raw = cv2.imread(
                    str(junction_paths_path), cv2.IMREAD_UNCHANGED
                )
                if junction_paths_plot_raw is not None:
                    stream_images[f"fingers_junction_paths_{key}"] = cv2.cvtColor(
                        junction_paths_plot_raw, cv2.COLOR_BGR2RGB
                    )
                interface_tips_plot_raw = cv2.imread(
                    str(gradient_path), cv2.IMREAD_UNCHANGED
                )
                if interface_tips_plot_raw is not None:
                    stream_images[f"fingers_interface_tips_{key}"] = cv2.cvtColor(
                        interface_tips_plot_raw, cv2.COLOR_BGR2RGB
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
