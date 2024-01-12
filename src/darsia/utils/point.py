"""Utility class defining different point formats.

Implemented are Cartesian coordinates and voxel coordinates allowing for different indexing.

"""

from typing import Any, Union, overload

import numpy as np

import darsia

# ! ---- Predefine all classes


class BasePoint(np.ndarray):
    ...


class Coordinate(BasePoint):
    ...


class Voxel(BasePoint):
    ...


class VoxelArray(Voxel):
    ...


class CoordinateArray(Coordinate):
    ...


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

    def to_matrix_indexing() -> None:
        raise NotImplementedError("Currently merely matrix indexing supported.")

    def to_reverse_matrix_indexing() -> None:
        raise NotImplementedError("Currently merely matrix indexing supported.")


# Conversion routines - to be associated to base class
def to_coordinate(self, coordinatesystem: darsia.CoordinateSystem) -> Coordinate:
    """Conversion of point to Coordinate.

    Args:
        coordinatesystem (CoordinateSystem): coordinate system used for conversion

    Returns:
        Coordinate: Coordinate variant of point

    """
    if isinstance(self, Coordinate):
        return Coordinate(self.pt)
    elif isinstance(self, Voxel):
        return Coordinate(coordinatesystem.coordinate(self.pt))
    else:
        raise NotImplementedError(f"{type(self)} not supported")


def to_voxel(self, coordinatesystem: darsia.CoordinateSystem) -> Voxel:
    """Conversion of point to Voxel.

    Args:
        coordinatesystem (CoordinateSystem): coordinate system used for conversion

    Returns:
        Voxel: Voxel variant of point

    """
    if isinstance(self, Voxel):
        return Voxel(self.pt)
    elif isinstance(self, Coordinate):
        return Voxel(coordinatesystem.voxel(self.pt))
    else:
        raise NotImplementedError(f"{type(self)} not supported")


# Assign method to base class
BasePoint.to_coordinate = to_coordinate
BasePoint.to_voxel = to_voxel

# ! ---- Implementation for collection of points ----


class VoxelArray(Voxel):
    """Container for collection of Voxel."""

    @overload
    def __getitem__(self, arg: int) -> Voxel:
        """Specialized item-access, returning objects instead of simple np.ndarray.

        Args:
            arg (int): identificator (for row in VoxelArray)

        Returns:
            Voxel: corresponding voxel

        """
        ...

    @overload
    def __getitem__(self, arg: np.ndarray) -> VoxelArray:
        """Specialized item-access, returning objects instead of simple np.ndarray.

        Args:
            arg (np.ndarray): identificator (for rows in VoxelArray)

        Returns:
            VoxelArray: corresponding voxels

        """
        ...

    def __getitem__(
        self, arg: Union[int, np.ndarray, Any]
    ) -> Union[Voxel, VoxelArray, np.ndarray]:
        """Specialized item-access, returning objects instead of simple np.ndarray."""
        if isinstance(arg, int):
            return Voxel(np.asarray(self)[arg])
        elif isinstance(arg, np.ndarray) and len(arg.shape) == 1:
            return VoxelArray(np.asarray(self)[arg])
        else:
            return np.asarray(self)[arg]


class CoordinateArray(Coordinate):
    """Container for collection of Coordinate."""

    @overload
    def __getitem__(self, arg: int) -> Coordinate:
        """Specialized item-access, returning objects instead of simple np.ndarray.

        Args:
            arg (int): identificator (for row in CoordinateArray)

        Returns:
            Coordinate: corresponding coordinate

        """
        ...

    @overload
    def __getitem__(self, arg: np.ndarray) -> CoordinateArray:
        """Specialized item-access, returning objects instead of simple np.ndarray.

        Args:
            arg (np.ndarray): identificator (for rows in CoordinateArray)

        Returns:
            CoordinateArray: corresponding coordinates

        """
        ...

    def __getitem__(
        self, arg: Union[int, np.ndarray, Any]
    ) -> Union[Coordinate, CoordinateArray, np.ndarray]:
        """Specialized item-access, returning objects instead of simple np.ndarray."""
        if isinstance(arg, int):
            return Coordinate(np.asarray(self)[arg])
        elif isinstance(arg, np.ndarray) and len(arg.shape) == 1:
            return CoordinateArray(np.asarray(self)[arg])
        else:
            return np.asarray(self)[arg]


# ! ---- Constructor routines


def make_coordinate(
    pts: Union[list, np.ndarray]
) -> Union[Coordinate, list[Coordinate]]:
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


def make_voxel(pts: Union[list, np.ndarray]) -> Union[Voxel, list[Voxel]]:
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
