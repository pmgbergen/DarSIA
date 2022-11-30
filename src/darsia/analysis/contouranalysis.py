"""
Module containing analysis tools for segmented images. This includes
measuring lengths of contours, weighted sums (generalized mass analysis).
"""
from __future__ import annotations

from typing import Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage
import cv2
from scipy.signal import find_peaks

import darsia


def contour_length(
    img: darsia.Image,
    roi: Optional[np.ndarray] = None,
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
    img_roi = (
        img.copy() if roi is None else cast(darsia.Image, darsia.extractROI(img, roi))
    )

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
    metric_contour_length = cast(
        float, img_roi.coordinatesystem.pixelsToLength(contour_length)
    )

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
    def __init__(self, verbosity: bool) -> None:
        self.verbosity = verbosity

    def load_labels(
        self,
        img: darsia.Image,
        roi: Optional[np.ndarray] = None,
        values_of_interest: Optional[Union[int, list[int]]] = None,
        fill_holes: bool = True,
    ) -> None:

        # Make copy of image and restrict to region of interest
        img_roi = (
            img.copy()
            if roi is None
            else cast(darsia.Image, darsia.extractROI(img, roi))
        )

        assert (
            roi is not None
        )  # TODO include the possibility to consider the entire image.

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

        # Cache coordinate system
        self.coordinatesystem = img_roi.coordinatesystem

        # Cache the mask
        self.mask = mask

        # Cache roi
        self.roi = roi

        # Cache image
        self.img = img_roi

        # Plot masks and print contour length if requested.
        if self.verbosity:
            plt.figure("Restricted image")
            plt.imshow(img_roi.img)
            plt.figure("Mask for values of interest")
            plt.imshow(mask)
            plt.show()

    def length(self) -> float:
        """
        Return length of the interface between values of interest and others.
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
        """
        Auxiliary function to determine the length of the contour
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
            float, self.coordinatesystem.pixelsToLength(contour_length)
        )

        return metric_contour_length

    def tips(self, direction=np.array([0.0, -1.0])) -> np.ndarray:
        """
        Determine local extrema of the contour, where the extremality
        is defined by a direction.

        Args:
            direction (np.ndarray): direction vector with orientation
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
            return np.array((0, 2), dtype=int), np.array((0, 2), dtype=int)

        # NOTE: Only possible to continue with one contour
        # assert len(contours) == 1
        peaks_pixels = np.array((0, 2), dtype=int)
        valleys_pixels = np.array((0, 2), dtype=int)
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

            # Extract the y-axis as signal FIXME depends on direction - in general rotation needed.
            contour_1d = inner_contour[:, 1]

            # Apply light smoothing
            smooth_contour_1d = ndi.gaussian_filter1d(
                contour_1d.astype(np.float32), sigma=10
            )

            # Peaks and valleys
            peaks_ind, _ = find_peaks(smooth_contour_1d)
            valleys_ind, _ = find_peaks(-smooth_contour_1d)

            # TODO include verbosity
            # if self.verbosity:
            #    peaks_val = smooth_contour_1d[peaks_ind]
            #    valleys_val = smooth_contour_1d[valleys_ind]
            #    plt.plot(smooth_contour_1d)
            #    plt.plot(peaks_ind, peaks_val, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green", linestyle="None")
            #    plt.plot(valleys_ind, valleys_val, marker="o", markersize=20, markeredgecolor="blue", markerfacecolor="purple", linestyle="None")
            #    plt.show()

            # Fetch pixels accounting for valleys and peaks
            _peaks_pixels = inner_contour[peaks_ind, :]
            _valleys_pixels = inner_contour[peaks_ind, :]

            peaks_pixels = np.vstack((peaks_pixels, _peaks_pixels))
            valleys_pixels = np.vstack((valleys_pixels, _valleys_pixels))

        peaks_pixels = np.reshape(peaks_pixels, (-1, 1, 2))
        valleys_pixels = np.reshape(valleys_pixels, (-1, 1, 2))

        if self.verbosity or True:
            plt.figure("Original image with peaks")
            plt.imshow(self.img.img)
            plt.scatter(peaks_pixels[:, 0, 0], peaks_pixels[:, 0, 1], c="r", s=20)
            plt.show()

        # Return peaks and valley pixels
        return peaks_pixels, valleys_pixels
