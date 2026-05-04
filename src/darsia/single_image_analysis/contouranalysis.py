"""Module containing analysis tools for segmented images.

This includes measuring lengths of contours, weighted sums (generalized mass analysis).

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage
from scipy.signal import find_peaks

import darsia

logger = logging.getLogger(__name__)


def contour_length(
    img: darsia.Image,
    roi: darsia.CoordinateArray | None = None,
    values_of_interest: int | list[int] | None = None,
    fill_holes: bool = False,
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
        plt.axis("off")
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

    def __init__(
        self,
        verbosity: bool = False,
        contour_smoother: darsia.ContourSmoother | None = None,
        reduce_to_main_contour: bool = False,
    ) -> None:
        """Constructor.

        Args:
            verbosity (bool): Verbosity flag.

        """

        self.verbosity = verbosity
        """Vebosity flag."""
        self.contour_smoother = contour_smoother
        """Optional contour smoother for the contours determined from the mask."""
        self.reduce_to_main_contour = reduce_to_main_contour
        """Whether to reduce to main contour."""

    def load(
        self,
        img: darsia.Image,
        mask: darsia.Image,
        roi: darsia.CoordinateArray | None = None,
        fill_holes: bool = False,
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
            mask_roi_array = ndi.binary_fill_holes(mask_roi_array)

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

    @darsia.timing_decorator
    def contours(self) -> list[np.ndarray]:
        """Determine contour of loaded labeled image.

        Returns:
            list[np.ndarray]: list of contours, where each contour is given as an
                array of pixels.

        """
        # Extract contours.
        contours, _ = cv2.findContours(
            skimage.img_as_ubyte(self.mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        # Determine the main contour as the one with the largest area.
        if self.reduce_to_main_contour and len(contours) > 1:
            contour_areas = [cv2.contourArea(contour) for contour in contours]
            main_contour_index = np.argmax(contour_areas)
            contours = [contours[main_contour_index]]

        # Smooth contours if smoother provided.
        if self.contour_smoother:
            contours = [self.contour_smoother(contour) for contour in contours]

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

    def local_extrema(
        self, direction=np.array([0.0, -1.0])
    ) -> tuple[np.ndarray, np.ndarray]:
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
            self.plot_peaks(self.img, reshaped_peaks_pixels)

        return reshaped_peaks_pixels, reshaped_valleys_pixels

    @darsia.timing_decorator
    def plot_peaks(
        self,
        img: darsia.Image,
        peaks_pixels: np.ndarray,
        roi: darsia.CoordinateArray | None = None,
        contours: list[np.ndarray] | None = None,
        path: Path | None = None,
        show: bool = True,
        dpi: int = 1000,
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
            dpi (int): dots per inch for the saved plot; default is 1000.
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
            plt.axis("off")
            plt.savefig(path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)

        if show:
            plt.show()
        else:
            plt.close()

    def number_peaks(self) -> int:
        """Determine number of peaks.

        Returns:
            int: number of peaks.

        """
        peaks_pixels, _ = self.local_extrema()
        return len(peaks_pixels)

    def plot_valleys(
        self,
        img: darsia.Image,
        valleys_pixels: np.ndarray,
        roi: darsia.CoordinateArray | None = None,
        contours: list[np.ndarray] | None = None,
        path: Path | None = None,
        show: bool = True,
        dpi: int = 1000,
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
            dpi (int): dots per inch for the saved plot; default is 1000.
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

        # Make sure the y_min/y_max are within the image bounds
        y_min = max(0, y_min)
        y_max = min(img.img.shape[0], y_max)

        plt.figure("Original image with valleys")
        plt.imshow(img.img)

        # Match image dimensions
        plt.xlim(0, img.img.shape[1])
        plt.ylim(img.img.shape[0], 0)  # Inverted because image y-axis is top-down

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
            plt.axis("off")
            plt.savefig(path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)

        if show:
            plt.show()
        else:
            plt.close()

    def number_valleys(self) -> int:
        """Determine number of valleys.

        Returns:
            int: number of valleys.

        """
        _, valleys_pixels = self.local_extrema()
        return len(valleys_pixels)
