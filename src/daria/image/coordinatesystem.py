from __future__ import annotations

from typing import Union

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
        self._width: float = img.width
        self._height: float = img.height
        self._num_pixels_width: int = img.num_pixels_width
        self._num_pixels_height: int = img.num_pixels_height
        self._dx: float = img.dx
        self._dy: float = img.dy

        # Determine the coordinate, corresponding to the origin pixel (0,0), i.e.,
        # the top left corner.
        self._coordinate_of_origin_pixel: np.ndarray = img.origo + np.array(
            [0, self._height]
        )

        # Determine the pixel, corresponding to the physical origin (0,0)
        self._pixel_of_origin_coordinate: np.ndarray = self.coordinateToPixel(
            np.array([0, 0])
        )

        # Determine the bounding box of the image, in physical dimensions,
        # also defining the effective boundaries of the coordinate system.
        # for this, address the lower left and upper right corners.
        xmin, ymin = self.pixelToCoordinate(img.corners["lowerleft"])
        xmax, ymax = self.pixelToCoordinate(img.corners["upperright"])
        self.domain = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

    def pixelsToLength(
        self, num_pixels: Union[int, np.ndarray], axis: str = "x"
    ) -> Union[float, np.ndarray]:
        """
        Convert number of pixels to metric units, when interpreting the pixels
        in some given axis.

        Args:
            num_pixels (int or 1d array of ints): number(s) of pixels
            axis (str): either "x" or "y" determining the conversion rate

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
    ) -> Union[int, np.np.ndarray]:
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

    def pixelToCoordinate(self, pixel: Union[np.ndarray, list[int]]) -> np.ndarray:
        """
        Conversion from pixel to Cartesian coordinate, i.e., from (row,col) to (x,y)
        format plus scaling.

        Handles both single and multiple pixels.

        Arguments:
            pixel (np.ndarray): pixel location in (row,col) format; one pixel per row

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

        # Combine two operations. 1. Convert from (row,col) to (x,y) format and
        # scale correctly, to obtain the physical coordinates with correct units.
        coordinate[:, 0] = x0 + pixel[:, 1] / self._num_pixels_width * self._width
        coordinate[:, 1] = y0 - pixel[:, 0] / self._num_pixels_height * self._height

        # Return in same format as the input
        return coordinate.reshape(original_shape)

    def coordinateToPixel(
        self, coordinate: Union[np.ndarray, list[float]]
    ) -> np.ndarray:
        """
        Conversion from Cartesian coordinate to pixel, from (x,y) to (row,col)
        format plus scaling.

        Handles both single and multiple coordinates.

        Arguments:
            coordinate (np.ndarray): coordinate in (x,y) format; one coordinate per row

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
        # (row,col) to (x,y) format and scale correctly, to obtain the physical
        # coordinates with correct units. Use floor to indicate which pixel is
        # marked by the coordinate.
        pixel[:, 0] = np.floor(
            (y0 - coordinate[:, 1]) * self._num_pixels_height / self._height
        )
        pixel[:, 1] = np.floor(
            (coordinate[:, 0] - x0) * self._num_pixels_width / self._width
        )

        # Return in same format as the input
        return pixel.reshape(original_shape).astype(int)
