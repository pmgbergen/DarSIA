"""Utility class defining different point formats.

Implemented are Cartesian coordinates and voxel coordinates allowing for different indexing.

"""
from __future__ import annotations

from typing import Any, Union, overload

import numpy as np

import darsia

# ! ---- Implementation for single points ----


class BasePoint(np.ndarray):
    """Base class for defining points."""

    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return


class Coordinate(BasePoint):
    """Cartesian coordinate."""

    ...


class Voxel(BasePoint):
    """Voxel coordinate."""

    def __new__(cls, input_array, matrix_indexing=True):
        obj = np.asarray(input_array).astype(int).view(cls)
        obj.matrix_indexing = True
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.matrix_indexing = getattr(obj, "matrix_indexing", True)


# ! ---- Implementation for collection of points ----


class VoxelArray(Voxel):
    """Container for collection of Voxel."""

    @overload  # type: ignore [override]
    def __getitem__(self, key: int) -> Voxel:
        """Specialized item-access, returning objects instead of simple np.ndarray.

        Args:
            key (int): identificator (for row in VoxelArray)

        Returns:
            Voxel: corresponding voxel

        """
        ...

    @overload
    def __getitem__(self, key: np.ndarray) -> VoxelArray:
        """Specialized item-access, returning objects instead of simple np.ndarray.

        Args:
            key (np.ndarray): identificator (for rows in VoxelArray)

        Returns:
            VoxelArray: corresponding voxels

        """
        ...

    def __getitem__(self, key: Any) -> Union[Voxel, VoxelArray, np.ndarray]:
        """Specialized item-access, returning objects instead of simple np.ndarray."""
        if isinstance(key, int):
            return Voxel(np.asarray(self)[key])
        elif isinstance(key, np.ndarray) and len(key.shape) == 1:
            return VoxelArray(np.asarray(self)[key])
        else:
            return np.asarray(self)[key]


class CoordinateArray(Coordinate):
    """Container for collection of Coordinate."""

    @overload  # type: ignore [override]
    def __getitem__(self, key: int) -> Coordinate:
        """Specialized item-access, returning objects instead of simple np.ndarray.

        Args:
            key (int): identificator (for row in CoordinateArray)

        Returns:
            Coordinate: corresponding coordinate

        """
        ...

    @overload
    def __getitem__(self, key: np.ndarray) -> CoordinateArray:
        """Specialized item-access, returning objects instead of simple np.ndarray.

        Args:
            key (np.ndarray): identificator (for rows in CoordinateArray)

        Returns:
            CoordinateArray: corresponding coordinates

        """
        ...

    def __getitem__(self, key: Any) -> Union[Coordinate, CoordinateArray, np.ndarray]:
        """Specialized item-access, returning objects instead of simple np.ndarray."""

        if isinstance(key, int):
            return Coordinate(np.asarray(self)[key])
        elif isinstance(key, np.ndarray) and len(key.shape) == 1:
            return CoordinateArray(np.asarray(self)[key])
        else:
            return np.asarray(self)[key]


# ! ---- Constructor routines


def make_coordinate(pts: Union[list, np.ndarray]) -> Union[Coordinate, CoordinateArray]:
    pts = np.array(pts)
    if len(pts.shape) == 1:
        return Coordinate(pts)
    else:
        assert len(pts.shape) == 2 and pts.shape[1] in [
            1,
            2,
            3,
        ], "only support 1d, 2d, 3d"
        return CoordinateArray(pts)


def make_voxel(pts: Union[list, np.ndarray]) -> Union[Voxel, VoxelArray]:
    pts = np.array(pts)
    if len(pts.shape) == 1:
        return Voxel(pts)
    else:
        assert len(pts.shape) == 2 and pts.shape[1] in [
            1,
            2,
            3,
        ], "only support 1d, 2d, 3d"
        return VoxelArray(pts)


# ! ---- Conversion routines

# The routines will eventualy be associated to the base class, but depending on the base
# type, the output differs.


def to_coordinate(
    self, coordinatesystem: darsia.CoordinateSystem
) -> Union[Coordinate, CoordinateArray]:
    """Conversion of point to Coordinate.

    Args:
        coordinatesystem (CoordinateSystem): coordinate system used for conversion

    Returns:
        Coordinate or CoordinateArray: Coordinate variant of point (type depends on input)

    """
    if isinstance(self, Coordinate) or isinstance(self, CoordinateArray):
        return self.copy()
    elif isinstance(self, Voxel):
        return make_coordinate(coordinatesystem.coordinate(self))
    else:
        raise NotImplementedError(f"{type(self)} not supported")


def to_voxel(
    self, coordinatesystem: darsia.CoordinateSystem
) -> Union[Voxel, VoxelArray]:
    """Conversion of point to Voxel.

    Args:
        coordinatesystem (CoordinateSystem): coordinate system used for conversion

    Returns:
        Voxel: Voxel variant of point

    """
    if isinstance(self, Voxel) or isinstance(self, VoxelArray):
        return self.copy()
    elif isinstance(self, Coordinate):
        return make_voxel(coordinatesystem.voxel(self))
    else:
        raise NotImplementedError(f"{type(self)} not supported")


# Assign method to base class
BasePoint.to_coordinate = to_coordinate  # type: ignore [attr-defined]
BasePoint.to_voxel = to_voxel  # type: ignore [attr-defined]
