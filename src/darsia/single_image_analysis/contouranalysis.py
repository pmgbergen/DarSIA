"""Module containing analysis tools for segmented images.

This includes measuring lengths of contours, weighted sums (generalized mass analysis).

"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional, Union, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage
from scipy.signal import find_peaks
from scipy.spatial import distance_matrix

import darsia


def contour_length(
    img: darsia.Image,
    roi: Optional[darsia.CoordinateArray] = None,
    values_of_interest: Optional[Union[int, list[int]]] = None,
    fill_holes: bool = True,
    verbosity: bool = False,
) -> float:
    """
    Calculation of the contour length of a segmented region.

    Args:
        img (darsia.Image): segmented image with boolean or integer values.
        roi (np.ndarray): set of points, for which a bounding box defines a ROI.
        values_of_interest (int or list of int): only active if integer-valued
            image provided; defines the values of interest, i.e., which part
            of the image is treated as active.
        fill_holes (bool): flag controlling whether holes in the determined mask
            are filled before the contour length is computed; if not, holes are
            treated as contour; default is True.
        verbosity (bool): flag controlling whether intermediate results are plotted;
            default is False.

    Returns:
        float: contour length in metric units based on the coordinate system of the
            input image.
    """
    # Make copy of image and restrict to region of interest
    img_roi = img.copy() if roi is None else cast(darsia.Image, img.subregion(roi))

    # Extract boolean mask covering pixels of interest.
    if img_roi.img.dtype == bool:
        if values_of_interest is None:
            mask: np.ndarray = img_roi.img
        else:
            mask = np.zeros_like(img_roi.img, dtype=bool)
            if isinstance(values_of_interest, int):
                mask[img_roi.img == values_of_interest]
            elif isinstance(values_of_interest, list):
                for value in values_of_interest:
                    mask[img_roi.img == value] = True
    elif img_roi.img.dtype in [np.uint8, np.int32, np.int64]:
        assert values_of_interest is not None
        mask = np.zeros(img_roi.img.shape[:2], dtype=bool)
        if isinstance(values_of_interest, int):
            mask[img_roi.img == values_of_interest] = True
        elif isinstance(values_of_interest, list):
            for value in values_of_interest:
                mask[img_roi.img == value] = True
    else:
        raise ValueError(f"Images with dtype {img_roi.img.dtype} not supported.")

    # Fill all holes
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    # Obtain connected and covered regions and region properties
    label_img, num_labels = skimage.measure.label(mask, return_num=True)
    props = skimage.measure.regionprops(label_img)

    # Determine the length of all contours and sum up
    contour_length = 0.0
    for counter in range(num_labels):
        # Add the perimeter for the region with label l+1 (0 is ignored in regionprops).
        contour_length += props[counter].perimeter

    # Convert contour length from pixel units to metric units
    metric_contour_length = cast(float, img_roi.coordinatesystem.length(contour_length))

    # Plot masks and print contour length if requested.
    if verbosity:
        plt.figure()
        plt.imshow(img.img)
        plt.figure()
        plt.imshow(img_roi.img)
        plt.figure()
        plt.imshow(mask)
        plt.show()
        print(f"The contour length is {metric_contour_length}.")

    return metric_contour_length


class ContourAnalysis:
    """Contour analysis object."""

    def __init__(self, verbosity: bool) -> None:
        """Constructor.

        Args:
            verbosity (bool): Verbosity flag.

        """

        self.verbosity = verbosity
        """Vebosity flag."""

    def load_labels(
        self,
        img: darsia.Image,
        roi: Optional[darsia.CoordinateArray] = None,
        values_of_interest: Optional[Union[int, list[int]]] = None,
        fill_holes: bool = True,
    ) -> None:
        """Read labeled image and restrict to values of interest.

        Args:
            img (Image): labeled image.
            roi (array, optional): set of points defining a box.
            values_of_interest (int, list of int, optional): label values of interest.
            fill_holes (bool): flag controlling whether holes in labels are filles.

        """

        # Make copy of image and restrict to region of interest
        img_roi = img.copy() if roi is None else cast(darsia.Image, img.subregion(roi))

        # Extract boolean mask covering values of interest.
        if img_roi.img.dtype == bool:
            mask: np.ndarray = img_roi.img

        elif img_roi.img.dtype in [np.uint8, np.int32, np.int64]:

            assert values_of_interest is not None

            # Check values of interest
            mask = np.zeros(img_roi.img.shape[:2], dtype=bool)
            if isinstance(values_of_interest, int):
                mask[img_roi.img == values_of_interest] = True

            elif isinstance(values_of_interest, list):
                for value in values_of_interest:
                    mask[img_roi.img == value] = True
        else:
            raise ValueError(f"Images with dtype {img_roi.img.dtype} not supported.")

        # Fill all holes
        if fill_holes:
            mask = ndi.binary_fill_holes(mask)

        self.coordinatesystem = img_roi.coordinatesystem
        """Coordinate system of subimage."""

        self.mask = mask
        """Mask."""

        self.roi = roi
        """Region of interest."""

        self.img = img_roi
        """Subimage."""

        # Plot masks and print contour length if requested.
        if self.verbosity:
            plt.figure("Restricted image")
            plt.imshow(img_roi.img)
            plt.figure("Mask for values of interest")
            plt.imshow(mask)
            plt.show()

    def length(self) -> float:
        """Determine length of loaded labeled image.

        Returns:
            float: length of the interface between values of interest and others.
                Output in metric units.

        """
        # Determine the actual contour length of the different regions in
        # the mask - includes boundary.
        contour_length_true = self._length_value(True)
        contour_length_false = self._length_value(False)

        # Determine the lenght of the entire boundary.
        perimeter = darsia.perimeter(self.roi)

        # Venn diagram concept to determine the internal contour of the CO2.
        # Cut values at 0, slightly negative values can occur e.g., when
        # contour_length_co2 is very small due to a different evaluation
        # routine for the contour length and the permiter of a box.
        interface_length = max(
            0.5 * (contour_length_true + contour_length_false - perimeter), 0.0
        )

        return interface_length

    def _length_value(self, value: bool) -> float:
        """Auxiliary function to determine the length of the contour
        of the regions with presribed value within self.mask.

        Args:
            value (bool): value of interest.

        Returns:
            float: contour length.

        """
        # Fix mask depending on the value of interest.
        mask = self.mask if value else np.logical_not(self.mask)

        # Obtain connected and covered regions and region properties
        label_img, num_labels = skimage.measure.label(mask, return_num=True)
        props = skimage.measure.regionprops(label_img)

        # Determine the length of all contours and sum up
        contour_length = 0.0
        for counter in range(num_labels):
            # Add the perimeter for the region with label l+1 (0 is ignored in regionprops).
            contour_length += props[counter].perimeter

        # Convert contour length from pixel units to metric units
        metric_contour_length = cast(
            float, self.coordinatesystem.length(contour_length, axis="x")
        )

        return metric_contour_length

    def fingers(self, direction=np.array([0.0, -1.0])) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine local extrema of the contour, where the extremality
        is defined by a direction.

        Args:
            direction (np.ndarray): direction vector with orientation

        Returns:
            array: pixels of peaks.
            array: pixels of valleys.

        """
        # TODO extend to directions other than [0., -1.]
        assert np.isclose(direction, np.array([0, -1])).all()

        # Extract interface - extract the boundary
        contours, _ = cv2.findContours(
            skimage.img_as_ubyte(self.mask),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE,
        )

        # Special case of no contour
        if len(contours) == 0:
            return np.zeros((0, 1, 2), dtype=int), np.zeros((0, 1, 2), dtype=int)

        # NOTE: Only possible to continue with one contour
        # assert len(contours) == 1
        peaks_pixels = np.zeros((0, 2), dtype=int)
        valleys_pixels = np.zeros((0, 2), dtype=int)
        for contour in contours:

            # Exclude pixels on the boundary
            rows, cols = self.mask.shape
            left_boundary = contour[:, :, 0] == 0
            right_boundary = contour[:, :, 0] == cols - 1
            top_boundary = contour[:, :, 1] == 0
            bottom_boundary = contour[:, :, 1] == rows - 1

            boundary_mask = np.logical_or(
                np.logical_or(
                    np.logical_or(
                        left_boundary,
                        right_boundary,
                    ),
                    top_boundary,
                ),
                bottom_boundary,
            )

            inner_mask = np.logical_not(boundary_mask)
            inner_contour = contour[inner_mask]

            # Extract the y-axis as signal
            # FIXME depends on direction - in general rotation needed.
            contour_1d = inner_contour[:, 1]

            # Apply light smoothing
            smooth_contour_1d = ndi.gaussian_filter1d(
                contour_1d.astype(np.float32), sigma=10
            )

            # Peaks and valleys
            peaks_ind, _ = find_peaks(smooth_contour_1d)
            valleys_ind, _ = find_peaks(-smooth_contour_1d)

            if self.verbosity:
                peaks_val = smooth_contour_1d[peaks_ind]
                valleys_val = smooth_contour_1d[valleys_ind]
                plt.plot(smooth_contour_1d)
                plt.plot(
                    peaks_ind,
                    peaks_val,
                    marker="o",
                    markersize=20,
                    markeredgecolor="red",
                    markerfacecolor="green",
                    linestyle="None",
                )
                plt.plot(
                    valleys_ind,
                    valleys_val,
                    marker="o",
                    markersize=20,
                    markeredgecolor="blue",
                    markerfacecolor="purple",
                    linestyle="None",
                )
                plt.show()

            # Fetch pixels accounting for valleys and peaks
            _peaks_pixels = inner_contour[peaks_ind, :]
            _valleys_pixels = inner_contour[valleys_ind, :]

            peaks_pixels = np.vstack((peaks_pixels, _peaks_pixels))
            valleys_pixels = np.vstack((valleys_pixels, _valleys_pixels))

        # Sort peaks and valleys - no reason why the contours have been traversed not ordered.
        arg_sorted_peaks_pixels = np.argsort(peaks_pixels[:, 0], axis=0)
        arg_sorted_valleys_pixels = np.argsort(valleys_pixels[:, 0], axis=0)
        sorted_peaks_pixels = peaks_pixels[arg_sorted_peaks_pixels]
        sorted_valleys_pixels = valleys_pixels[arg_sorted_valleys_pixels]

        reshaped_peaks_pixels = np.reshape(sorted_peaks_pixels, (-1, 1, 2))
        reshaped_valleys_pixels = np.reshape(sorted_valleys_pixels, (-1, 1, 2))

        if self.verbosity:
            plt.figure("Original image with peaks")
            plt.imshow(self.img.img)
            plt.scatter(
                reshaped_peaks_pixels[:, 0, 0],
                reshaped_peaks_pixels[:, 0, 1],
                c="r",
                s=20,
            )
            plt.show()

        # Return peaks and valley pixels
        return reshaped_peaks_pixels, reshaped_valleys_pixels

    def number_peaks(self) -> int:
        """Determine number of peaks.

        Returns:
            int: number of peaks.

        """
        peaks_pixels, _ = self.fingers()
        return len(peaks_pixels)


# In order to uniquely identify a location in the collection of paths, define
# a subunit, storing the position in terms of local and global characteristics.
PathUnit = namedtuple("PathUnit", ["time", "peak", "position"])
# A connected path later will define a list of PathUnit. Collections of such
# will define connected paths.


class ContourEvolutionAnalysis:
    # TODO try to merge with the above class.

    def __init__(self, verbosity: bool = False) -> None:
        """
        Args:
            verbosity (bool): verbosity - mostly useful for debugging.

        """

        self.index = 0
        self.peaks = {}
        self.valleys = {}
        self.paths = []

        self.verbosity = verbosity

    def add(
        self, peaks: np.ndarray, valleys: np.ndarray, time: Optional[float] = None
    ) -> None:

        # TODO use time
        self.peaks[self.index] = peaks.copy()
        self.valleys[self.index] = valleys.copy()
        self.index += 1

        self.total_time = self.index

    def plot(self, img: Optional[darsia.Image] = None) -> None:

        if img is None:
            raise False

        background = np.zeros(img.img.shape[:2], dtype=int)
        plt.figure("Tips and valleys - evolution")
        plt.imshow(background)
        for peaks in self.peaks.values():
            plt.scatter(peaks[:, 0, 0], peaks[:, 0, 1], c="y", s=20, label="peaks")
        for valleys in self.valleys.values():
            plt.scatter(
                valleys[:, 0, 0], valleys[:, 0, 1], c="r", s=10, label="valleys"
            )
        plt.show()

    def plot_paths(self, img: Optional[darsia.Image] = None) -> None:

        if img is None:
            raise False

        # Draw provided image in the background
        plt.figure("Paths")
        plt.imshow(img.img)

        # Determine longest path
        max_path_length = 0
        for i, path in enumerate(self.paths):
            # Assemble path by connecting positions
            path_pos = np.zeros((0, 2), dtype=int)
            for unit in path:
                path_pos = np.vstack((path_pos, unit.position))
            max_path_length = max(max_path_length, path_pos.shape[0])

        # Add paths
        cmap = plt.cm.get_cmap("viridis")
        num_paths = len(self.paths)
        for i, path in enumerate(self.paths):
            # Assemble path by connecting positions
            path_pos = np.zeros((0, 2), dtype=int)
            for unit in path:
                path_pos = np.vstack((path_pos, unit.position))
            path_length = path_pos.shape[0]
            plt.plot(
                path_pos[:, 0],
                path_pos[:, 1],
                color=cmap(i / (num_paths - 1)),
                linewidth=path_length / max_path_length * 2,
            )

        plt.savefig("paths.svg", format="svg", dpi=1000)

        # Finalize plot
        plt.show()

    def find_paths(self, reset: bool = True) -> None:
        """
        Find paths in the peaks and valleys.
        """

        # TODO decide whether to reset
        if reset:
            self.paths = []

        def _same_path_unit(path_unit_0, path_unit_1):
            """
            Check whether all required items are the same (position not needed
            if implemented without errors).
            """

            return (
                path_unit_0.time == path_unit_1.time
                and path_unit_0.peak == path_unit_1.peak
            )

        def _include_segments(time_prev, time_next, segments, pts_prev, pts_next):
            """
            Assemble paths.
            """

            # Consider each segment separately:
            for segment in segments:

                # Define a path unit for the start point - ignore
                path_unit_prev = PathUnit(
                    time_prev, segment[0], pts_prev[segment[0], :]
                )

                # Define a path unit for the end point
                path_unit_next = PathUnit(
                    time_next, segment[1], pts_next[segment[1], :]
                )

                # Traverse the paths to check whether a preexisting path ends with
                # the start of the segment.
                added_to_preexisting_path = False
                for path in self.paths:

                    # Check last element in the path
                    last_unit = path[-1]
                    identical_units = _same_path_unit(last_unit, path_unit_prev)

                    if identical_units:
                        path.append(path_unit_next)
                        added_to_preexisting_path = True
                        break

                # Add the entire segment if no suitable path exists for this
                if not added_to_preexisting_path:
                    # print("create new path")
                    self.paths.append([path_unit_prev, path_unit_next])

        def _include_points(time_next, points, pts_next):
            """
            Assemble paths.
            """

            # Consider each segment separately:
            for point in points:

                # Define a path unit for the end point
                path_unit_next = PathUnit(time_next, point, pts_next[point, :])
                self.paths.append([path_unit_next])

        # TODO replace time_index with something global?
        for time_index in range(self.total_time - 1):

            # Fetch peak indices
            peaks_prev = np.squeeze(self.peaks[time_index]).reshape(-1, 2)
            peaks_next = np.squeeze(self.peaks[time_index + 1]).reshape(-1, 2)

            # Initialize correpsonding peaks
            prev_next_pairs = []
            new_paths = []

            if len(peaks_prev) == 0:
                for peak_index in list(np.arange(len(peaks_prev))):
                    new_paths.append(peak_index)
                continue

            if len(peaks_next) == 0:
                continue

            # Initialize paired slices
            paired_slices = [(slice(0, len(peaks_prev)), slice(0, len(peaks_next)))]

            # Construct distance matrix
            dist = distance_matrix(peaks_prev, peaks_next)

            for iteration in range(max(len(peaks_prev), len(peaks_next))):

                # Stopping criterion.
                if len(paired_slices) == 0:
                    break

                # First assignments
                ind_prev, ind_next = paired_slices.pop(0)

                if self.verbosity:
                    print("slices", ind_prev, ind_next)

                # Find smallest entry in the restricted distance matrix
                # print(dist[ind_prev, ind_next])
                local_distance_matrix = dist[ind_prev, ind_next]
                num_local_cols = local_distance_matrix.shape[1]
                local_argmin_1d = np.argmin(np.ravel(local_distance_matrix))
                local_argmin_2d = np.array(
                    [
                        int(local_argmin_1d / num_local_cols),
                        local_argmin_1d % num_local_cols,
                    ]
                )
                if self.verbosity:
                    print("local argmin 2d", local_argmin_1d, local_argmin_2d)
                # Globalize argmin_2d
                argmin_2d = local_argmin_2d + np.array([ind_prev.start, ind_next.start])
                if self.verbosity:
                    print(
                        "argmin",
                        argmin_2d,
                        peaks_prev[argmin_2d[0]],
                        peaks_next[argmin_2d[1]],
                    )

                # Bookkeeping.
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

                # In order to keep the order, start with the later slice
                if _nonempty_slice(post_slice[0]) and _nonempty_slice(post_slice[1]):
                    paired_slices.insert(0, post_slice)
                elif _nonempty_slice(post_slice[0]):
                    if self.verbosity:
                        print(
                            f"Fingers according to slice {post_slice[0]} are terminated."
                        )
                elif _nonempty_slice(post_slice[1]):
                    if self.verbosity:
                        print(f"New paths according to slice {post_slice[1]}.")
                    for peak_index in list(
                        np.arange(post_slice[1].start, post_slice[1].stop)
                    ):
                        new_paths.append(peak_index)

                if _nonempty_slice(pre_slice[0]) and _nonempty_slice(pre_slice[1]):
                    paired_slices.insert(0, pre_slice)
                elif _nonempty_slice(pre_slice[0]):
                    if self.verbosity:
                        print(
                            f"Fingers according to slice {pre_slice[0]} are terminated."
                        )
                elif _nonempty_slice(pre_slice[1]):
                    if self.verbosity:
                        print(f"New paths according to slice {pre_slice[1]}.")
                    for peak_index in list(
                        np.arange(pre_slice[1].start, pre_slice[1].stop)
                    ):
                        new_paths.append(peak_index)

                if self.verbosity:
                    print()

            # For simpler handling, convert to array format (reshape only for
            # handling empty lists) and sort
            prev_next_pairs = np.array(prev_next_pairs).reshape(-1, 2)
            arg_sorted_prev_next_pairs = np.argsort(prev_next_pairs[:, 0])
            prev_next_pairs = prev_next_pairs[arg_sorted_prev_next_pairs]

            new_paths = np.array(new_paths)
            arg_sorted_new_paths = np.argsort(new_paths)
            new_paths = new_paths[arg_sorted_new_paths]

            if self.verbosity:
                print(
                    f"""final: {time_index} vs. {time_index + 1}. Found
                    {len(prev_next_pairs)} many pairs among {len(peaks_prev)}
                    and {len(peaks_next)} fingers, respectively."""
                )
                print(prev_next_pairs)
                print(new_paths)

            # Glue together segments where possible (sort in).
            _include_segments(
                time_index, time_index + 1, prev_next_pairs, peaks_prev, peaks_next
            )
            _include_points(time_index + 1, new_paths, peaks_next)
