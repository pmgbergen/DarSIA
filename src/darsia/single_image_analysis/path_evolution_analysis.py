"""General analysis of path evolution, e.g., for fingers."""

from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

import darsia

from .contouranalysis import _corners_of_roi

logger = logging.getLogger(__name__)
# In order to uniquely identify a location in the collection of paths, define
# a subunit, storing the position in terms of local and global characteristics.
PathUnit = namedtuple("PathUnit", ["time", "id", "position"])
# A connected path later will define a list of PathUnit. Collections of such
# will define connected paths.


class PathEvolutionAnalysis:
    def __init__(self, verbosity: bool = False) -> None:
        """
        Args:
            verbosity (bool): verbosity - mostly useful for debugging.

        """

        self.points: dict[int, np.ndarray] = {}
        """Dictionary storing points at different time points."""
        self.paths: list[list[PathUnit]] = []
        """List storing paths."""
        self.verbosity = verbosity
        """Verbosity flag."""
        self.times: list[float] = []
        """Sorted list of time points."""

    def add(self, points: np.ndarray, time: float | None = None) -> None:
        """Add points for a new time point.

        Args:
            points (np.ndarray): pixels of points at the new time point.
            time (float, optional): time corresponding to the new time point;
                if None, index is used as time.

        """
        # Add time to self.times and sort
        if time is None:
            raise ValueError("Time cannot be None when adding points.")
        self.times.append(time)
        self.times.sort()

        # Find the index of the new time point
        index = self.times.index(time)

        # Insert points at the correct index in self.points.
        # May need to move existing entries to the right if the new time point is inserted in
        # the middle of self.times.
        self.points = {i + (i >= index): p for i, p in self.points.items()}
        self.points[index] = points.copy()
        self.total_time = len(self.times)

    def plot_paths(
        self,
        img: darsia.Image | None = None,
        roi: darsia.CoordinateArray | None = None,
        path: Path | None = None,
        show: bool = False,
        dpi: int = 1000,
    ) -> None:
        if img is None:
            raise ValueError("img cannot be None when plotting paths.")

        top_left_roi_pixel, _ = _corners_of_roi(img, roi)

        # Draw provided image in the background
        plt.figure("Paths")
        plt.imshow(img.img)

        # Determine longest (finger) path
        max_path_length = 0
        for i, _path in enumerate(self.paths):
            # Assemble path by connecting positions
            path_pos = np.zeros((0, 2), dtype=int)
            for unit in _path:
                path_pos = np.vstack((path_pos, unit.position))
            max_path_length = max(max_path_length, path_pos.shape[0])

        # Add paths
        cmap = plt.cm.get_cmap("viridis")
        num_paths = len(self.paths)
        denominator = max(num_paths - 1, 1)
        for i, _path in enumerate(self.paths):
            # Assemble path by connecting positions
            path_pos = np.zeros((0, 2), dtype=int)
            for unit in _path:
                path_pos = np.vstack((path_pos, unit.position))
            path_length = path_pos.shape[0]
            plt.plot(
                path_pos[:, 0] + top_left_roi_pixel[0],
                path_pos[:, 1] + top_left_roi_pixel[1],
                color=cmap(i / denominator),
                linewidth=path_length / max_path_length * 2,
            )

        # Turn off axis
        plt.axis("off")

        if path is not None:
            suffix = path.suffix
            if suffix in [".png", ".jpg", ".jpeg", ".svg"]:
                format = suffix[1:]
                plt.savefig(
                    path, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0
                )
            else:
                plt.savefig(
                    path.with_suffix(".png"),
                    format="png",
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0,
                )

        # Finalize plot
        if show:
            plt.show()
        else:
            plt.close()

    def _find_paths(self, points: dict[int, np.ndarray]) -> list[list[PathUnit]]:
        paths = []

        def _reshape_points(array: np.ndarray) -> np.ndarray:
            squeezed_array = np.asarray(array)
            if squeezed_array.size == 0:
                return np.zeros((0, 2), dtype=int)
            return np.squeeze(squeezed_array).reshape(-1, 2)

        def _same_path_unit(path_unit_0, path_unit_1):
            return path_unit_0.time == path_unit_1.time and np.allclose(
                path_unit_0.position, path_unit_1.position
            )

        def _include_segments(time_prev, time_next, segments, pts_prev, pts_next):
            for segment in segments:
                path_unit_prev = PathUnit(
                    time_prev, segment[0], pts_prev[segment[0], :]
                )
                path_unit_next = PathUnit(
                    time_next, segment[1], pts_next[segment[1], :]
                )

                added_to_preexisting_path = False
                for path in paths:
                    last_unit = path[-1]
                    identical_units = _same_path_unit(last_unit, path_unit_prev)
                    if identical_units:
                        path.append(path_unit_next)
                        added_to_preexisting_path = True
                        break

                if not added_to_preexisting_path:
                    paths.append([path_unit_prev, path_unit_next])

        def _include_points(time_next, point_indices, pts_next):
            for point_index in point_indices:
                path_unit_next = PathUnit(
                    time_next, point_index, pts_next[point_index, :]
                )
                paths.append([path_unit_next])

        if self.total_time == 1:
            pts = _reshape_points(points.get(0, np.zeros((0, 1, 2), dtype=int)))
            _include_points(0, range(len(pts)), pts)
            return paths

        for time_index in range(self.total_time - 1):
            pts_prev = _reshape_points(
                points.get(time_index, np.zeros((0, 1, 2), dtype=int))
            )
            pts_next = _reshape_points(
                points.get(time_index + 1, np.zeros((0, 1, 2), dtype=int))
            )

            prev_next_pairs = []
            new_paths = []

            if len(pts_prev) == 0 and len(pts_next) > 0:
                for point_index in range(len(pts_next)):
                    new_paths.append(point_index)
                _include_points(time_index + 1, new_paths, pts_next)
                continue

            if len(pts_next) == 0 or len(pts_prev) == 0:
                continue

            paired_slices = [(slice(0, len(pts_prev)), slice(0, len(pts_next)))]
            dist = distance_matrix(pts_prev, pts_next)

            for _ in range(max(len(pts_prev), len(pts_next))):
                if len(paired_slices) == 0:
                    break

                ind_prev, ind_next = paired_slices.pop(0)
                local_distance_matrix = dist[ind_prev, ind_next]
                num_local_cols = local_distance_matrix.shape[1]
                local_argmin_1d = np.argmin(np.ravel(local_distance_matrix))
                local_argmin_2d = np.array(
                    [
                        int(local_argmin_1d / num_local_cols),
                        local_argmin_1d % num_local_cols,
                    ]
                )
                argmin_2d = local_argmin_2d + np.array([ind_prev.start, ind_next.start])
                prev_next_pairs.append(argmin_2d)

                pre_slice = (
                    slice(ind_prev.start, argmin_2d[0]),
                    slice(ind_next.start, argmin_2d[1]),
                )
                post_slice = (
                    slice(argmin_2d[0] + 1, ind_prev.stop),
                    slice(argmin_2d[1] + 1, ind_next.stop),
                )

                def _nonempty_slice(sl) -> bool:
                    return sl.stop - sl.start > 0

                if _nonempty_slice(post_slice[0]) and _nonempty_slice(post_slice[1]):
                    paired_slices.insert(0, post_slice)
                elif _nonempty_slice(post_slice[1]):
                    for point_index in range(post_slice[1].start, post_slice[1].stop):
                        new_paths.append(point_index)

                if _nonempty_slice(pre_slice[0]) and _nonempty_slice(pre_slice[1]):
                    paired_slices.insert(0, pre_slice)
                elif _nonempty_slice(pre_slice[1]):
                    for point_index in range(pre_slice[1].start, pre_slice[1].stop):
                        new_paths.append(point_index)

            prev_next_pairs = np.array(prev_next_pairs).reshape(-1, 2)
            if prev_next_pairs.shape[0] > 0:
                arg_sorted_prev_next_pairs = np.argsort(prev_next_pairs[:, 0])
                prev_next_pairs = prev_next_pairs[arg_sorted_prev_next_pairs]

            new_paths = np.array(new_paths)
            if new_paths.shape[0] > 0:
                arg_sorted_new_paths = np.argsort(new_paths)
                new_paths = new_paths[arg_sorted_new_paths]

            _include_segments(
                time_index, time_index + 1, prev_next_pairs, pts_prev, pts_next
            )
            _include_points(time_index + 1, new_paths, pts_next)

        return paths

    def find_paths(self, reset: bool = True) -> None:
        """
        Find paths in the points.
        """

        if reset:
            self.paths = []
        self.paths.extend(self._find_paths(self.points))
