from __future__ import annotations

from typing import Union, cast

import numpy as np

import daria as da


class CoordinateSystem:
    """Class for coordinate system for images.

    A coordinate system has knowledge about the conversion of pixels (in standard format),
    i.e., (row,col) format with (0,0) being identified with the top left corner.
    Conversion maps from pixel to coordinates and vice-versa are provided.

    Attributes:

    """

    def __init__(self, img: da.Image, dim: int = 2):
        """Generate a coordinate system based on the metadata of an existing image."""

        # Copy metadata from image. Will be used for conversions.
        self._dx: float = img.dx
        self._dy: float = img.dy

        # Determine the coordinate, corresponding to the origin pixel (0,0), i.e.,
        # the top left corner.
        self._coordinate_of_origin_pixel: np.ndarray = img.origo + np.array(
            [0, img.height]
        )

        # Determine the pixel, corresponding to the physical origin (0,0)
        self._pixel_of_origin_coordinate: np.ndarray = self.coordinateToPixel([0, 0])

        # Determine the bounding box of the image, in physical dimensions,
        # also defining the effective boundaries of the coordinate system.
        # for this, address the lower left and upper right corners.
        xmin, ymin = self.pixelToCoordinate(img.corners["lowerleft"])
        xmax, ymax = self.pixelToCoordinate(img.corners["upperright"])
        self.domain: dict = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

    def pixelsToLength(
        self, num_pixels: Union[int, float, np.ndarray], axis: str = "x"
    ) -> Union[float, np.ndarray]:
        """
        Convert number of pixels to metric units, when interpreting the pixels
        in some given axis.

        Args:
            num_pixels (int or 1d or 2d array of ints): number(s) of pixels
            axis (str): either "x", or "y" determining the conversion rate

        Returns:
            float or 1d array of floats: length(s) in metric units
        """
        if axis == "x":
            return num_pixels * self._dx
        elif axis == "y":
            return num_pixels * self._dy
        else:
            raise ValueError("Axis type not supported.")

    def lengthToPixels(
        self, length: Union[float, np.ndarray], axis: str = "x"
    ) -> Union[int, np.ndarray]:
        """
        Convert length in metric units to number of pixels, when interpreting
        the length in some given axis.

        Args:
            length (float or 1d array of floats): length(s) in metric units
            axis (str): either "x" or "y" determining the conversion rate

        Returns:
            int or 1d array of ints: number(s) of pixels
        """
        # Include all touched pixels; use therefore ceil.
        if axis == "x":
            return np.ceil(length / self._dx).astype(int)
        elif axis == "y":
            return np.ceil(length / self._dy).astype(int)
        else:
            raise ValueError("Axis type not supported.")

    def pixelToCoordinate(
        self, pixel: Union[np.ndarray, list[int]], reverse: bool = False
    ) -> np.ndarray:
        """
        Conversion from pixel to Cartesian coordinate, i.e., from (row,col) to (x,y)
        format plus scaling.

        Handles both single and multiple pixels.

        Arguments:
            pixel (np.ndarray): pixel location in (row,col) format; one pixel per row
            reverse (bool): flag whether the input is using reverse matrix indexing;
                default is False

        Returns:
            np.ndarray: corresponding coordinate in (x,y) format
        """
        # Convert list to array
        if isinstance(pixel, list):
            pixel = np.array(pixel)
        assert isinstance(pixel, np.ndarray)

        # Fetch the top left corner.
        x0, y0 = self._coordinate_of_origin_pixel

        # Aim at handling both single pixels stored in a 1d array as well as
        # multiple pixels stored in a 2d array. Convert to the more general
        # case.
        original_shape = pixel.shape
        pixel = np.atleast_2d(pixel)

        # Initialize coordinates
        coordinate = np.empty_like(pixel, dtype=float)

        # Combine two operations. 1. Convert from (row,col) to (x,y) format
        # (if required) and scale correctly, to obtain the physical coordinates
        # with correct units.
        vertical_pixel_pos = 1 if reverse else 0
        horizontal_pixel_pos = 0 if reverse else 1

        coordinate[:, 0] = (
            x0 + cast(np.ndarray, pixel)[:, horizontal_pixel_pos] * self._dx
        )
        coordinate[:, 1] = (
            y0 - cast(np.ndarray, pixel)[:, vertical_pixel_pos] * self._dy
        )

        # Return in same format as the input
        return coordinate.reshape(original_shape)

    def coordinateToPixel(
        self, coordinate: Union[np.ndarray, list[float]], reverse: bool = False
    ) -> np.ndarray:
        """
        Conversion from Cartesian coordinate to pixel, from (x,y) to (row,col)
        format plus scaling.

        Handles both single and multiple coordinates.

        Arguments:
            coordinate (np.ndarray): coordinate in (x,y) format; one coordinate per row
            reverse (bool): flag whether the output is using reverse matrix indexing;
                default is False

        Returns:
            np.ndarray: corresponding pixels in (row,col) format
        """
        # Convert list to array
        if isinstance(coordinate, list):
            coordinate = np.array(coordinate)
        assert isinstance(coordinate, np.ndarray)

        # Fetch the top left corner.
        x0, y0 = self._coordinate_of_origin_pixel

        # Aim at handling both single coordinates stored in a 1d array as well as
        # multiple coordinates stored in a 2d array. Convert to the more general
        # case.
        original_shape = coordinate.shape
        coordinate = np.atleast_2d(coordinate)

        # Initialize coordinates
        pixel = np.empty_like(coordinate, dtype=int)

        # Invert pixelToCoordinate. Again combine two operations. 1. Convert from
        # (row,col) to (x,y) format (if required) and scale correctly, to obtain
        # the physical coordinates with correct units. Use floor to indicate which
        # pixel is marked by the coordinate.
        vertical_pixel_pos = 1 if reverse else 0
        horizontal_pixel_pos = 0 if reverse else 1

        pixel[:, vertical_pixel_pos] = np.floor(
            (y0 - cast(np.ndarray, coordinate)[:, 1]) / self._dy
        )
        pixel[:, horizontal_pixel_pos] = np.floor(
            (cast(np.ndarray, coordinate)[:, 0] - x0) / self._dx
        )

        # Return in same format as the input
        return pixel.reshape(original_shape).astype(int)

    def pixelToCoordinateVector(
        self, pixel_vector: np.ndarray, reverse: bool = False
    ) -> np.ndarray:
        """
        Conversion from (translation) vectors in terms of pixels to coordinates.

        Arguments:
            pixel_vector (np.ndarray): vector(s) in pixel plane (one vector per row)
            reverse (bool): flag whether the input is using reverse matrix indexing;
                default is False

        Returns:
            np.ndarray: vector(s) in metric coordinate system
        """
        # Vectors are relative, and hence do not involve any information
        # on any fixed points as the origin. Yet, the conversion requires
        # to conform with the switch from Cartesian to matrix indexing
        # and thereby also the orientation of the vertical axis. Additionally,
        # the scaling from pixels to metric units is required.
        coordinate_vector = (
            np.atleast_2d(pixel_vector)
            if reverse
            else np.fliplr(np.atleast_2d(pixel_vector))
        ) * np.array([self._dx, -self._dy])

        # Reshape needed if only a single vector has been used in the argument.
        return coordinate_vector.reshape(pixel_vector.shape)
