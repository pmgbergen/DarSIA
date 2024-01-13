"""Physical coordinate systems for Image objects."""

from __future__ import annotations

from typing import Union

import numpy as np

import darsia


class CoordinateSystem:
    """Class for coordinate system for general space-time images.

    A coordinate system has knowledge about the conversion of voxels (in standard format),
    i.e., ijk format with (0,0,0) being identified with the top left front corner.
    This implicitly concludes that the coordinate system is directly related
    to a certain fixed indexing, provided at initialization. Conversion maps
    from voxels to coordinates and vice-versa are provided.

    Attributes:

    """

    def __init__(self, img: darsia.Image):
        """Generate a coordinate system based on the metadata of an existing image.

        Args:
            img (darsia.Image): image for which a coordinate system shall be constructed.

        """

        assert img.indexing in ["i", "ij", "ijk"], f"Indexing not supported."
        self.indexing = img.indexing
        """Indexing of the underlying image."""

        self.dim = img.space_dim
        """Dimension of the underlying space."""

        self.shape = img.img.shape[: self.dim]
        """Dimensions of the underlying images in terms of voxels."""

        self.dimensions = img.dimensions
        """Physical dimensions in each index direction."""

        self.axes = "xyz"[: self.dim]
        """Axes of the underlying space."""

        self.voxel_size = {}
        """Voxel size in each dimension/axis."""
        for axis in self.axes:
            pos, _ = darsia.interpret_indexing(axis, self.indexing)
            self.voxel_size[axis] = img.voxel_size[pos]

        self._coordinate_of_origin_voxel: darsia.Coordinate = img.origin
        """Coordinate of origin voxel."""

        opposite_corner_voxel = img.img.shape[: self.dim]
        self._coordinate_of_opposite_voxel: darsia.Coordinate = self.coordinate(
            opposite_corner_voxel
        )
        """Coordinate of opposite voxel."""

        corners = np.vstack(
            (self._coordinate_of_origin_voxel, self._coordinate_of_opposite_voxel)
        )
        self.domain = {}
        """Extremal points/bounding box of the active coordinate system / the image."""
        for i, axis in enumerate(self.axes):
            self.domain[axis + "min"] = np.min(corners[:, i])
            self.domain[axis + "max"] = np.max(corners[:, i])

    @property
    def voxels(self) -> darsia.VoxelArray:
        """Voxel array of image, collecting all voxels.

        Returns:
            VoxelArray: voxel array of image

        """
        if not hasattr(self, "_voxels"):
            self._voxels = darsia.make_voxel(
                np.indices(self.shape, dtype=int).reshape((self.dim, -1), order="F").T
            )
        return self._voxels

    @property
    def coordinates(self) -> darsia.CoordinateArray:
        """Coordinate array of image, collecting all coordinates.

        Returns:
            CoordinateArray: coordinate array of image

        """
        if not hasattr(self, "_coordinates"):
            self._coordinates = self.coordinate(self.voxels)
        return self._coordinates

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
        self,
        voxel: Union[
            np.ndarray,
            list[int],
            tuple[int],
            darsia.Voxel,
            darsia.VoxelArray,
        ],
    ) -> Union[darsia.Voxel, darsia.VoxelArray]:
        """
        Conversion from voxel to Cartesian coordinate, i.e., from (row,col) to (x,y)
        format plus scaling for a 2d image.

        Handles both single and multiple voxels.

        Arguments:
            voxel (np.ndarray, list, tuple, Coordinate, or Voxel): voxel location in the
                same format as the indexing of the underlying baseline image (see __init__);
                one voxel per row.

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
            pos, revert = darsia.interpret_indexing(axis, self.indexing)
            scaling = -1 if revert else 1
            coordinate[:, i] = (
                self._coordinate_of_origin_voxel[i]
                + scaling * voxel_array[:, pos] * self.voxel_size[axis]
            )

        # Return in same format as the input
        return darsia.make_coordinate(coordinate.reshape(voxel.shape))

    def voxel(
        self,
        coordinate: Union[np.ndarray, list[float], darsia.Voxel, darsia.Coordinate],
    ) -> np.ndarray:
        """
        Conversion from Cartesian coordinate to voxel in matrix indexing format.

        Handles both single and multiple coordinates.

        Arguments:
            coordinate (np.ndarray, list, Coordinate or Voxel): coordinate in Cartesian
                format, i.e., [x,y,z]; one coordinate per row.

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
            pos, revert = darsia.interpret_indexing(axis, self.indexing)
            scaling = -1 if revert else 1
            pixel[:, pos] = np.floor(
                scaling
                * (coordinate_array[:, i] - self._coordinate_of_origin_voxel[i])
                / self.voxel_size[axis]
            )

        # Return in same format as the input, and force int dtype.
        return darsia.make_voxel(np.round(pixel.reshape(coordinate.shape)).astype(int))

    def coordinate_vector(self, pixel_vector: np.ndarray) -> np.ndarray:
        """
        Conversion from vectors (relative distances) in terms of pixels to coordinates.

        Arguments:
            pixel_vector (np.ndarray): vector(s) in pixel plane (one vector per row)

        Returns:
            np.ndarray: coordinate vector(s) in Cartesian format.

        """
        # Aim at handling both single coordinates stored in a 1d array as well as
        # multiple coordinates stored in a 2d array. Convert to the more general
        # case.
        pixel_vector_array = np.atleast_2d(pixel_vector)

        coordinate_vector = np.empty_like(pixel_vector_array, dtype=float)
        for i, axis in enumerate(self.axes):
            pos, revert = darsia.interpret_indexing(axis, self.indexing)
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


def check_equal_coordinatesystems(
    coordinatesystem1: CoordinateSystem,
    coordinatesystem2: CoordinateSystem,
    exclude_size: bool = False,
) -> tuple[bool, dict]:
    """Check whether two coordinate systems are equivalent, i.e., they share basic
    attributes.

    Args:
        coordinatesystem1 (CoordinateSystem): first coordinate system
        coordinatesystem2 (CoordinateSystem): second coordinate system
        exclude_size (bool): flag controlling whether the size quantities are exluded.

    Returns:
        bool: True iff the two coordinate systems are equivalent.
        dict: log of the failed checks.

    """
    success = True
    failure_log = []

    if not (coordinatesystem1.indexing == coordinatesystem2.indexing):
        failure_log.append("indexing")

    if not (coordinatesystem1.dim == coordinatesystem2.dim):
        failure_log.append("space_dim")

    if not exclude_size:
        if not (np.allclose(coordinatesystem1.shape, coordinatesystem2.shape)):
            failure_log.append("shape")

    if not (np.allclose(coordinatesystem1.dimensions, coordinatesystem2.dimensions)):
        failure_log.append("dimensions")

    if not (coordinatesystem1.axes == coordinatesystem2.axes):
        failure_log.append("axes")

    if not exclude_size:
        voxel_size_equal = True
        for axis in coordinatesystem1.axes:
            voxel_size_equal = (
                voxel_size_equal
                and coordinatesystem1.voxel_size[axis]
                == coordinatesystem2.voxel_size[axis]
            )
        if not voxel_size_equal:
            failure_log.append("voxel_size")

    if not (
        np.allclose(
            coordinatesystem1._coordinate_of_origin_voxel,
            coordinatesystem2._coordinate_of_origin_voxel,
        )
    ):
        failure_log.append("coordinate_of_origin_voxel")

    if not (
        np.allclose(
            coordinatesystem1._coordinate_of_opposite_voxel,
            coordinatesystem2._coordinate_of_opposite_voxel,
        )
    ):
        failure_log.append("coordinate_of_opposite_voxel")

    success = len(failure_log) == 0

    return success, failure_log
