from __future__ import annotations

from typing import Union, cast

import numpy as np

import darsia


class CoordinateSystem:
    """Class for coordinate system for images.

    A coordinate system has knowledge about the conversion of pixels (in standard format),
    i.e., (row,col) format with (0,0) being identified with the top left corner.
    Conversion maps from pixel to coordinates and vice-versa are provided.

    Attributes:

    """

    def __init__(self, img: darsia.Image):
        """Generate a coordinate system based on the metadata of an existing image."""

        assert isinstance(img, darsia.Image)

        # Copy metadata from image. Will be used for conversions.
        self._dx: float = img.dx
        self._dy: float = img.dy

        # Determine the coordinate, corresponding to the origin pixel (0,0), i.e.,
        # the top left corner.
        self._coordinate_of_origin_pixel: np.ndarray = img.origin + np.array(
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


class GeneralCoordinateSystem:
    """Class for coordinate system for general space-time images.

    A coordinate system has knowledge about the conversion of voxels (in standard format),
    i.e., ijk format with (0,0,0) being identified with the top left front corner.
    This implicitly concludes that the coordinate system is directly related
    to a certain fixed orientation, provided at initialization. Conversion maps
    from voxels to coordinates and vice-versa are provided.

    Attributes:

    """

    # TODO replace CoordinateSystem with GeneralCoordinateSystem

    def __init__(self, img: darsia.GeneralImage):
        """Generate a coordinate system based on the metadata of an existing image."""

        # Cache orientation - NOTE: The c
        # TODO allow for other orientations than matrix indexing?
        assert img.orientation in ["ij", "ijk"]
        self.orientation = img.orientation

        # TODO generalize to darsia.Image.
        assert isinstance(img, darsia.GeneralImage)

        # Identify relevant Cartesian axes.
        dim = img.space_dim
        self.axes = ["x", "y"] if dim == 2 else ["x", "y", "z"]

        # Determine voxel size in x, y, z directions
        self.voxel_size = {}
        for axis in self.axes:
            pos, _ = darsia.interpret_orientation(axis, self.orientation)
            self.voxel_size[axis] = img.voxel_size[pos]

        # Determine the coordinate, corresponding to the origin of the matrix indexing.
        # Need to add the dimension of the image for axis with reverse orientation.
        self._coordinate_of_origin_voxel: np.ndarray = img.origin
        for axis in self.axes:
            pos, revert = darsia.interpret_orientation(axis, self.orientation)
            if revert:
                self._coordinate_of_origin_voxel[pos] += img.dimensions[pos]

        # Go towards the other end of the image.
        opposite_corner_voxel = img.img.shape[:dim]
        self._coordinate_of_opposite_voxel = self.coordinate(opposite_corner_voxel)

        # Determine the voxel, corresponding to the physical origin.
        origin_coordinate = img.origin
        self._voxel_of_origin_coordinate: np.ndarray = self.voxel(origin_coordinate)

        # Determine the bounding box of the image, in physical dimensions,
        # also defining the effective boundaries of the coordinate system.
        self.domain = {}
        for i, axis in enumerate(self.axes):
            pos, _ = darsia.interpret_orientation(axis, self.orientation)
            self.domain[axis + "min"] = self._coordinate_of_origin_voxel[i]
            self.domain[axis + "max"] = self._coordinate_of_opposite_voxel[i]

    def length(self, num: Union[int, np.ndarray], axis: str) -> float:
        """
        Convert number of voxels/voxels to metric units, when interpreting
        in some given axis.

        Args:
            num (int or array): number(s) of voxels/voxels
            axis (str): axis "x", "y", or "z"

        Returns:
            float or array: length in metric units

        """
        assert axis in self.axes
        return num * self.voxel_size[axis]

    def num_voxels(
        self, length: Union[float, np.ndarray], axis: str
    ) -> Union[int, np.ndarray]:
        """
        Convert length in metric units to number of voxels, when interpreting
        the length in some given axis.

        Args:
            length (float or 1d array of floats): length(s) in metric units
            axis (str): axis "x", "y", or "z"

        Returns:
            int or array: number(s) of voxels

        """
        # Include all touched voxels; use therefore ceil.
        assert axis in self.axes
        return np.ceil(length / self.voxel_size[axis]).astype(int)

    def coordinate(
        self, voxel: Union[np.ndarray, list[int], tuple[int]], reverse: bool = False
    ) -> np.ndarray:
        """
        Conversion from voxel to Cartesian coordinate, i.e., from (row,col) to (x,y)
        format plus scaling for a 2d image.

        Handles both single and multiple voxels.

        Arguments:
            voxel (np.ndarray): voxel location in the same format as the orientation
                of the underlying baseline image (see __init__); one voxel per row.

        Returns:
            np.ndarray: corresponding coordinate in (x,y) format
        """
        # Convert list to array
        if isinstance(voxel, tuple):
            voxel = list(voxel)
        if isinstance(voxel, list):
            voxel = np.array(voxel)
        assert isinstance(voxel, np.ndarray)

        # Aim at handling both single voxels stored in a 1d array as well as
        # multiple voxels stored in a 2d array. Convert to the more general
        # case. Cache the original size for the later output.
        voxel_array = np.atleast_2d(voxel)

        # Determine coordinates with the help of the origin
        coordinate = np.empty_like(voxel_array, dtype=float)
        for i, axis in enumerate(self.axes):
            pos, revert = darsia.interpret_orientation(axis, self.orientation)
            scaling = -1 if revert else 1
            coordinate[:, i] = (
                self._coordinate_of_origin_voxel[i]
                + scaling * voxel_array[:, pos] * self.voxel_size[axis]
            )

        # Return in same format as the input
        return coordinate.reshape(voxel.shape)

    def voxel(self, coordinate: Union[np.ndarray, list[float]]) -> np.ndarray:
        """
        Conversion from Cartesian coordinate to voxel in matrix indexing format.

        Handles both single and multiple coordinates.

        Arguments:
            coordinate (np.ndarray): coordinate in Cartesian format, i.e., [x,y,z];
                one coordinate per row.

        Returns:
            np.ndarray: corresponding pixels in "ij"/"ijk" format.

        """
        # Convert list to array
        if isinstance(coordinate, list):
            coordinate = np.array(coordinate)
        assert isinstance(coordinate, np.ndarray)

        # Aim at handling both single coordinates stored in a 1d array as well as
        # multiple coordinates stored in a 2d array. Convert to the more general
        # case.
        coordinate_array = np.atleast_2d(coordinate)

        # Inverse of self.coordinate().
        pixel = np.empty_like(coordinate_array, dtype=int)
        for i, axis in enumerate(self.axes):
            pos, revert = darsia.interpret_orientation(axis, self.orientation)
            scaling = -1 if revert else 1
            pixel[:, pos] = np.floor(
                scaling
                * (coordinate_array[:, i] - self._coordinate_of_origin_voxel[i])
                / self.voxel_size[axis]
            )

        # Return in same format as the input, and force int dtype.
        return pixel.reshape(coordinate.shape).astype(int)

    def coordinate_vector(self, pixel_vector: np.ndarray) -> np.ndarray:
        """
        Conversion from vectors (relative distances) in terms of pixels to coordinates.

        Arguments:
            pixel_vector (np.ndarray): vector(s) in pixel plane (one vector per row)

        Returns:
            np.ndarray: coordinate vector(s) in Cartesian format.

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

        # Aim at handling both single coordinates stored in a 1d array as well as
        # multiple coordinates stored in a 2d array. Convert to the more general
        # case.
        pixel_vector_array = np.atleast_2d(pixel_vector)

        coordinate_vector = np.empty_like(pixel_vector_array, dtype=float)
        for i, axis in enumerate(self.axes):
            pos, revert = darsia.interpret_orientation(axis, self.orientation)
            scaling = -1 if revert else 1
            coordinate_vector[:, i] = (
                scaling * pixel_vector_array[:, pos] * self.voxel_size[axis]
            )

        # Reshape needed if only a single vector has been used in the argument.
        return coordinate_vector.reshape(pixel_vector.shape)

    def pixel_vector(self, coordinate_vector: np.ndarray) -> np.ndarray:
        """
        Analogue to self.coorindate_vector().

        """
        raise NotImplementedError
