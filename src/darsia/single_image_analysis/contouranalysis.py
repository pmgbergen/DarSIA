"""Module containing analysis tools for segmented images.

This includes measuring lengths of contours, weighted sums (generalized mass analysis).

"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
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


def _corners_of_roi(
    img: darsia.Image,
    roi: darsia.CoordinateArray,
):
    # Extract the top left pixel of the roi. NOTE: Need to swap for matplotlib,
    # which uses (x, y) convention for pixels, while the image uses (row, column)
    # convention.
    if roi is not None:
        roi_pixels = roi.to_voxel(img.coordinatesystem)
        top_left_roi_pixel = [np.min(roi_pixels[:, 1]), np.min(roi_pixels[:, 0])]
        bottom_right_roi_pixel = [
            np.max(roi_pixels[:, 1]),
            np.max(roi_pixels[:, 0]),
        ]
    else:
        top_left_roi_pixel = [0, 0]
        bottom_right_roi_pixel = [img.img.shape[1], img.img.shape[0]]

    return top_left_roi_pixel, bottom_right_roi_pixel


class ContourAnalysis:
    """Contour analysis object."""

    def __init__(self, verbosity: bool = False) -> None:
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

    def load(
        self,
        img: darsia.Image,
        mask: darsia.Image,
        roi: Optional[darsia.CoordinateArray] = None,
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
        mask_roi: darsia.Image = (
            mask.copy() if roi is None else cast(darsia.Image, mask.subregion(roi))
        )

        # Extract boolean mask covering values of interest.
        mask_roi_array = mask_roi.img

        # Fill all holes
        if fill_holes:
            mask = ndi.binary_fill_holes(mask)

        self.coordinatesystem = mask_roi.coordinatesystem
        """Coordinate system of subimage."""

        self.mask = mask_roi_array
        """Mask."""

        self.roi = roi
        """Region of interest."""

        self.img = mask_roi
        """Subimage."""

        # Plot masks and print contour length if requested.
        # if self.verbosity:
        #    mask_roi.show()
        #    # plt.figure("Restricted image")
        #    # plt.imshow(mask_roi.as.img)
        #    # plt.figure("Mask for values of interest")
        #    # plt.imshow(mask)
        #    # plt.show()

    def contours(self) -> list[np.ndarray]:
        """Determine contour of loaded labeled image.

        Returns:
            list[np.ndarray]: list of contours, where each contour is given as an
                array of pixels.

        """
        contours, _ = cv2.findContours(
            skimage.img_as_ubyte(self.mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        return contours

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
            contours (np.ndarray | None): contours to analyze. If None, contours are determined
                from the mask; default is None.
            direction (np.ndarray): direction vector with orientation

        Returns:
            array: pixels of peaks.
            array: pixels of valleys.

        """
        # Sanity check
        if not np.isclose(direction, np.array([0, -1])).all():
            # TODO extend to directions other than [0., -1.]
            raise NotImplementedError(
                """Currently only direction [0., -1.] supported, i.e., vertical direction """
                """with peaks pointing downwards."""
            )

        # Extract interface - extract the boundary
        contours = self.contours()

        # Special case of no contour
        if len(contours) == 0:
            return np.zeros((0, 1, 2), dtype=int), np.zeros((0, 1, 2), dtype=int)

        # Continue with each contour separately.
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
            self.plot_finger_peaks(self.img, reshaped_peaks_pixels)

        return reshaped_peaks_pixels, reshaped_valleys_pixels

    def plot_finger_peaks(
        self,
        img: darsia.Image,
        peaks_pixels: np.ndarray,
        roi: darsia.CoordinateArray | None = None,
        contours: list[np.ndarray] | None = None,
        path: Path | None = None,
        show: bool = True,
        **kwargs,
    ) -> None:
        """Plot peaks on top of the provided image.

        Args:
            img (darsia.Image): image to plot on.
            peaks_pixels (np.ndarray): pixels of peaks.
            contours (list[np.ndarray], optional): contours to plot; if None, no contours are
                plotted; default is None.
            roi (darsia.CoordinateArray | None): region of interest. If provided, peaks are
                translated to the top left corner of the ROI; default is None.
            path (Path, optional): path to save the plot; if None, no saving is performed.
            show (bool): flag controlling whether the plot is shown; default is True.
            **kwargs: additional keyword arguments for plotting.
                - color (str): color for the peaks; default is "r".
                - size (int): size for the peaks; default is 20.

        """

        # Extract the top left pixel of the roi. NOTE: Need to swap for matplotlib,
        # which uses (x, y) convention for pixels, while the image uses (row, column)
        # convention.
        top_left_roi_pixel, bottom_right_roi_pixel = _corners_of_roi(img, roi)

        plt.figure("Original image with peaks")
        plt.imshow(img.img)
        if contours is not None:
            for contour in contours:
                plt.plot(
                    contour[:, 0, 0] + top_left_roi_pixel[0],
                    contour[:, 0, 1] + top_left_roi_pixel[1],
                    c=kwargs.get("contour_color", "w"),
                    linewidth=kwargs.get("contour_linewidth", 1),
                )
        plt.scatter(
            # Translate pixels to the top left corner of the ROI
            peaks_pixels[:, 0, 0] + top_left_roi_pixel[0],
            peaks_pixels[:, 0, 1] + top_left_roi_pixel[1],
            c=kwargs.get("peak_color", "r"),
            s=kwargs.get("peak_size", 20),
        )
        if kwargs.get("plot_boundary", False):
            plt.gca().add_patch(
                plt.Rectangle(
                    (top_left_roi_pixel[0], top_left_roi_pixel[1]),
                    bottom_right_roi_pixel[0] - top_left_roi_pixel[0],
                    bottom_right_roi_pixel[1] - top_left_roi_pixel[1],
                    linewidth=kwargs.get("boundary_linewidth", 2),
                    edgecolor=kwargs.get("boundary_color", "y"),
                    facecolor="none",
                )
            )
        if kwargs.get("highlight_roi", False):
            # Add dark overlay to the area outside the ROI
            plt.gca().add_patch(
                plt.Rectangle(
                    (0, 0), img.img.shape[1], img.img.shape[0], color="black", alpha=0.5
                )
            )
            plt.gca().add_patch(
                plt.Rectangle(
                    (top_left_roi_pixel[0], top_left_roi_pixel[1]),
                    bottom_right_roi_pixel[0] - top_left_roi_pixel[0],
                    bottom_right_roi_pixel[1] - top_left_roi_pixel[1],
                    color="white",
                    alpha=0.5,
                )
            )
        if path is not None:
            plt.tight_layout()
            plt.savefig(path, format="png", dpi=1000)

        if show:
            plt.show()
        else:
            plt.close()

    def number_peaks(self) -> int:
        """Determine number of peaks.

        Returns:
            int: number of peaks.

        """
        peaks_pixels, _ = self.fingers()
        return len(peaks_pixels)

    def plot_valleys(
        self,
        img: darsia.Image,
        valleys_pixels: np.ndarray,
        roi: darsia.CoordinateArray | None = None,
        contours: list[np.ndarray] | None = None,
        path: Path | None = None,
        show: bool = True,
        **kwargs,
    ) -> None:
        """Plot valleys on top of the provided image.

        Args:
            img (darsia.Image): image to plot on.
            valleys_pixels (np.ndarray): pixels of valleys.
            contours (list[np.ndarray], optional): contours to plot; if None, no contours are
                plotted; default is None.
            roi (darsia.CoordinateArray | None): region of interest. If provided, valleys are
                translated to the top left corner of the ROI; default is None.
            path (Path, optional): path to save the plot; if None, no saving is performed.
            show (bool): flag controlling whether the plot is shown; default is True.
            **kwargs: additional keyword arguments for plotting.
                - valley_color (str): color for valley lines; default is "c".
                - valley_linewidth (float): line width for valley lines; default is 1.
                - y_min (float): lower y-limit for valley lines; default is top of ROI.
                - y_max (float): upper y-limit for valley lines; default is bottom of ROI.
                - plot_valley_dots (bool): if True, valley dots are added; default is False.
                - valley_dot_color (str): color for valley dots; default is valley_color.
                - valley_dot_size (float): dot size for valley dots; default is 20.

        """

        top_left_roi_pixel, bottom_right_roi_pixel = _corners_of_roi(img, roi)
        y_min = kwargs.get("y_min", top_left_roi_pixel[1])
        y_max = kwargs.get("y_max", bottom_right_roi_pixel[1])
        valley_color = kwargs.get("valley_color", "c")

        plt.figure("Original image with valleys")
        plt.imshow(img.img)
        if contours is not None:
            for contour in contours:
                plt.plot(
                    contour[:, 0, 0] + top_left_roi_pixel[0],
                    contour[:, 0, 1] + top_left_roi_pixel[1],
                    c=kwargs.get("contour_color", "w"),
                    linewidth=kwargs.get("contour_linewidth", 1),
                )

        valley_x = valleys_pixels[:, 0, 0] + top_left_roi_pixel[0]
        plt.vlines(
            valley_x,
            y_min,
            y_max,
            colors=valley_color,
            linewidth=kwargs.get("valley_linewidth", 1),
        )

        if kwargs.get("plot_valley_dots", False):
            plt.scatter(
                valley_x,
                valleys_pixels[:, 0, 1] + top_left_roi_pixel[1],
                c=kwargs.get("valley_dot_color", valley_color),
                s=kwargs.get("valley_dot_size", 20),
            )

        if kwargs.get("plot_boundary", False):
            plt.gca().add_patch(
                plt.Rectangle(
                    (top_left_roi_pixel[0], top_left_roi_pixel[1]),
                    bottom_right_roi_pixel[0] - top_left_roi_pixel[0],
                    bottom_right_roi_pixel[1] - top_left_roi_pixel[1],
                    linewidth=kwargs.get("boundary_linewidth", 2),
                    edgecolor=kwargs.get("boundary_color", "y"),
                    facecolor="none",
                )
            )
        if kwargs.get("highlight_roi", False):
            plt.gca().add_patch(
                plt.Rectangle(
                    (0, 0), img.img.shape[1], img.img.shape[0], color="black", alpha=0.5
                )
            )
            plt.gca().add_patch(
                plt.Rectangle(
                    (top_left_roi_pixel[0], top_left_roi_pixel[1]),
                    bottom_right_roi_pixel[0] - top_left_roi_pixel[0],
                    bottom_right_roi_pixel[1] - top_left_roi_pixel[1],
                    color="white",
                    alpha=0.5,
                )
            )

        if path is not None:
            plt.tight_layout()
            plt.savefig(path, format="png", dpi=1000)

        if show:
            plt.show()
        else:
            plt.close()


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
        self.valley_paths = []

        self.verbosity = verbosity

    def add(
        self, peaks: np.ndarray, valleys: np.ndarray, time: Optional[float] = None
    ) -> None:
        self.peaks[self.index] = peaks.copy()
        self.valleys[self.index] = valleys.copy()
        self.time = time
        self.index += 1

        self.total_time = self.index

    def plot(
        self,
        img: Optional[darsia.Image] = None,
        roi: Optional[darsia.CoordinateArray] = None,
    ) -> None:
        # TODO.
        if img is None:
            raise ValueError("img is required to plot contour evolution.")

        top_left_roi_pixel, bottom_right_roi_pixel = _corners_of_roi(img, roi)
        background = np.zeros(img.img.shape[:2], dtype=int)
        plt.figure("Tips and valleys - evolution")
        plt.imshow(background)
        for peaks in self.peaks.values():
            plt.scatter(
                peaks[:, 0, 0] + top_left_roi_pixel[0],
                peaks[:, 0, 1] + top_left_roi_pixel[1],
                c="y",
                s=20,
                label="peaks",
            )
        for valleys in self.valleys.values():
            plt.scatter(
                valleys[:, 0, 0] + top_left_roi_pixel[0],
                valleys[:, 0, 1] + top_left_roi_pixel[1],
                c="r",
                s=10,
                label="valleys",
            )
        plt.show()

    def plot_paths(
        self,
        img: Optional[darsia.Image] = None,
        roi: darsia.CoordinateArray | None = None,
        path: Path | None = None,
        show: bool = False,
    ) -> None:
        if img is None:
            raise ValueError("img is required to plot paths.")

        top_left_roi_pixel, bottom_right_roi_pixel = _corners_of_roi(img, roi)

        # Draw provided image in the background
        plt.figure("Paths")
        plt.imshow(img.img)

        # Determine longest (finger) path
        max_path_length = 0
        for i, finger_path in enumerate(self.paths):
            # Assemble path by connecting positions
            path_pos = np.zeros((0, 2), dtype=int)
            for unit in finger_path:
                path_pos = np.vstack((path_pos, unit.position))
            max_path_length = max(max_path_length, path_pos.shape[0])

        # Add paths
        cmap = plt.cm.get_cmap("viridis")
        num_paths = len(self.paths)
        denominator = max(num_paths - 1, 1)
        for i, finger_path in enumerate(self.paths):
            # Assemble path by connecting positions
            path_pos = np.zeros((0, 2), dtype=int)
            for unit in finger_path:
                path_pos = np.vstack((path_pos, unit.position))
            path_length = path_pos.shape[0]
            plt.plot(
                path_pos[:, 0] + top_left_roi_pixel[0],
                path_pos[:, 1] + top_left_roi_pixel[1],
                color=cmap(i / denominator),
                linewidth=path_length / max_path_length * 2,
            )

        if path is not None:
            plt.savefig(path.with_suffix(".svg"), format="svg", dpi=1000)

        # Finalize plot
        if show:
            plt.show()
        else:
            plt.close()

    def plot_valley_paths(
        self,
        img: Optional[darsia.Image] = None,
        roi: darsia.CoordinateArray | None = None,
        path: Path | None = None,
        show: bool = False,
        color: str | None = None,
    ) -> None:
        if img is None:
            raise ValueError("img is required to plot valley paths.")

        top_left_roi_pixel, bottom_right_roi_pixel = _corners_of_roi(img, roi)

        plt.figure("Valley paths")
        plt.imshow(img.img)

        max_path_length = 0
        for valley_path in self.valley_paths:
            path_pos = np.zeros((0, 2), dtype=int)
            for unit in valley_path:
                path_pos = np.vstack((path_pos, unit.position))
            max_path_length = max(max_path_length, path_pos.shape[0])

        if max_path_length == 0:
            max_path_length = 1

        y_min = top_left_roi_pixel[1]
        y_max = bottom_right_roi_pixel[1]

        cmap = plt.cm.get_cmap("viridis")
        num_paths = len(self.valley_paths)
        denominator = max(num_paths - 1, 1)
        for i, valley_path in enumerate(self.valley_paths):
            path_pos = np.zeros((0, 2), dtype=int)
            for unit in valley_path:
                path_pos = np.vstack((path_pos, unit.position))

            if path_pos.shape[0] == 0:
                continue

            path_length = path_pos.shape[0]
            path_color = cmap(i / denominator) if color is None else color
            line_width = path_length / max_path_length * 2

            x_global = path_pos[:, 0] + top_left_roi_pixel[0]
            y_global = path_pos[:, 1] + top_left_roi_pixel[1]
            plt.plot(
                x_global,
                y_global,
                color=path_color,
                linewidth=line_width,
                alpha=1.0,
            )

            first_x = x_global[0]
            first_y = y_global[0]
            if first_y != y_min:
                plt.plot(
                    [first_x, first_x],
                    [y_min, first_y],
                    color=path_color,
                    linewidth=line_width,
                    alpha=0.5,
                )

            last_x = x_global[-1]
            last_y = y_global[-1]
            if last_y != y_max:
                plt.plot(
                    [last_x, last_x],
                    [last_y, y_max],
                    color=path_color,
                    linewidth=line_width,
                    alpha=0.5,
                )

        if path is not None:
            plt.savefig(path.with_suffix(".svg"), format="svg", dpi=1000)

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
            return (
                path_unit_0.time == path_unit_1.time
                and path_unit_0.peak == path_unit_1.peak
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
                path_unit_next = PathUnit(time_next, point_index, pts_next[point_index, :])
                paths.append([path_unit_next])

        if self.total_time == 1:
            pts = _reshape_points(points.get(0, np.zeros((0, 1, 2), dtype=int)))
            _include_points(0, list(np.arange(len(pts))), pts)
            return paths

        for time_index in range(self.total_time - 1):
            pts_prev = _reshape_points(points.get(time_index, np.zeros((0, 1, 2), dtype=int)))
            pts_next = _reshape_points(
                points.get(time_index + 1, np.zeros((0, 1, 2), dtype=int))
            )

            prev_next_pairs = []
            new_paths = []

            if len(pts_prev) == 0 and len(pts_next) > 0:
                for point_index in list(np.arange(len(pts_next))):
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
                    for point_index in list(
                        np.arange(post_slice[1].start, post_slice[1].stop)
                    ):
                        new_paths.append(point_index)

                if _nonempty_slice(pre_slice[0]) and _nonempty_slice(pre_slice[1]):
                    paired_slices.insert(0, pre_slice)
                elif _nonempty_slice(pre_slice[1]):
                    for point_index in list(
                        np.arange(pre_slice[1].start, pre_slice[1].stop)
                    ):
                        new_paths.append(point_index)

            prev_next_pairs = np.array(prev_next_pairs).reshape(-1, 2)
            if prev_next_pairs.shape[0] > 0:
                arg_sorted_prev_next_pairs = np.argsort(prev_next_pairs[:, 0])
                prev_next_pairs = prev_next_pairs[arg_sorted_prev_next_pairs]

            new_paths = np.array(new_paths)
            if new_paths.shape[0] > 0:
                arg_sorted_new_paths = np.argsort(new_paths)
                new_paths = new_paths[arg_sorted_new_paths]

            _include_segments(time_index, time_index + 1, prev_next_pairs, pts_prev, pts_next)
            _include_points(time_index + 1, new_paths, pts_next)

        return paths

    def find_paths(self, reset: bool = True) -> None:
        """
        Find paths in the peaks and valleys.
        """

        if reset:
            self.paths = []
        self.paths.extend(self._find_paths(self.peaks))

    def find_valley_paths(self, reset: bool = True) -> None:
        """Find paths in the valleys."""
        if reset:
            self.valley_paths = []
        self.valley_paths.extend(self._find_paths(self.valleys))
