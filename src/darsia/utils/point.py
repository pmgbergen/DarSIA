"""Utility class defining different point formats.

Implemented are Cartesian coordinates and voxel coordinates allowing for different indexing.

"""

from __future__ import annotations

from typing import Any, Optional, Union, overload

import numpy as np

import darsia

# ! ---- Implementation for single points ----


class BasePoint(np.ndarray):
    """Base class for defining points."""

    def __new__(cls, input_array=None):
        if input_array is None:
            input_array = np.empty((0,))
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return


class Coordinate(BasePoint):
    """Cartesian coordinate."""

    ...


class Voxel(BasePoint):
    """Voxel coordinate.

    A Voxel by default uses matrix indexing, i.e. the first index corresponds to the
    row, the second to the column and the third to the depth (if applicable).

    """

    def __new__(cls, input_array, matrix_indexing=True):
        obj = np.asarray(input_array).astype(int).view(cls)
        if not matrix_indexing:
            obj = np.fliplr(np.atleast_2d(obj)).reshape(obj.shape).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return


class VoxelCenter(BasePoint):
    """Voxel center coordinate."""

    def __new__(cls, input_array, matrix_indexing=True):
        obj = np.asarray(input_array).astype(int)
        obj = obj + 0.5 * np.ones(obj.shape)
        obj = obj.view(cls)
        if not matrix_indexing:
            obj = np.fliplr(np.atleast_2d(obj)).reshape(obj.shape).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return


# ! ---- Implementation for collection of points ----


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


class VoxelCenterArray(VoxelCenter):
    """Container for collection of VoxelCenter."""

    @overload  # type: ignore [override]
    def __getitem__(self, key: int) -> VoxelCenter:
        """Specialized item-access, returning objects instead of simple np.ndarray.

        Args:
            key (int): identificator (for row in VoxelArray)

        Returns:
            VoxelCenter: corresponding voxel

        """
        ...

    @overload
    def __getitem__(self, key: np.ndarray) -> VoxelCenterArray:
        """Specialized item-access, returning objects instead of simple np.ndarray.

        Args:
            key (np.ndarray): identificator (for rows in VoxelArray)

        Returns:
            VoxelCenterArray: corresponding voxels

        """
        ...

    def __getitem__(self, key: Any) -> Union[VoxelCenter, VoxelCenterArray, np.ndarray]:
        """Specialized item-access, returning objects instead of simple np.ndarray."""
        if isinstance(key, int):
            return VoxelCenter(np.asarray(self)[key])
        elif isinstance(key, np.ndarray) and len(key.shape) == 1:
            return VoxelCenterArray(np.asarray(self)[key])
        else:
            return np.asarray(self)[key]


# ! ---- Constructor routines


def make_coordinate(pts: Union[list, np.ndarray]) -> Union[Coordinate, CoordinateArray]:
    """Quick-access constructor for Coordinate or CoordinateArray.

    Args:
        pts (Union[list, np.ndarray]): list of points or array of points

    Returns:
        Union[Coordinate, CoordinateArray]: Coordinate or CoordinateArray variant of
            point (type depends on input), i.e. if a single point is provided, a
            Coordinate is returned, if a list of points is provided, a CoordinateArray
            is returned.

    """
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


def make_voxel(
    pts: Union[list, np.ndarray], matrix_indexing: bool = True
) -> Union[Voxel, VoxelArray]:
    """Quick-access constructor for Voxel or VoxelArray.

    Args:
        pts (Union[list, np.ndarray]): list of points or array of points
        matrix_indexing (bool, optional): whether to use matrix indexing (first index
            corresponds to row, second to column, third to depth). Defaults to True.

    Returns:
        Union[Voxel, VoxelArray]: Voxel or VoxelArray variant of point (type depends on
            input), i.e. if a single point is provided, a Voxel is returned, if a list
            of points is provided, a VoxelArray is returned.

    """
    pts_array = np.array(pts)
    if len(pts_array.shape) == 1:
        return Voxel(pts_array, matrix_indexing=matrix_indexing)
    else:
        assert len(pts_array.shape) == 2 and pts_array.shape[1] in [
            1,
            2,
            3,
        ], "only support 1d, 2d, 3d"
        return VoxelArray(pts_array, matrix_indexing=matrix_indexing)


def make_voxel_center(
    pts: Union[list, np.ndarray], matrix_indexing: bool = True
) -> Union[VoxelCenter, VoxelArray]:
    """Quick-access constructor for VoxelCenter or VoxelCenterArray.

    Args:
        pts (Union[list, np.ndarray]): list of points or array of points
        matrix_indexing (bool, optional): whether to use matrix indexing (first index
            corresponds to row, second to column, third to depth). Defaults to True.

    Returns:
        Union[VoxelCenter, VoxelCenterArray]: VoxelCenter or VoxelCenterArray variant
            of point (type depends on input), i.e. if a single point is provided, a
            VoxelCenter is returned, if a list of points is provided, a VoxelCenterArray
            is returned.

    """
    pts_array = np.array(pts)
    if len(pts_array.shape) == 1:
        return VoxelCenter(pts_array, matrix_indexing=matrix_indexing)
    else:
        assert len(pts_array.shape) == 2 and pts_array.shape[1] in [
            1,
            2,
            3,
        ], "only support 1d, 2d, 3d"
        return VoxelCenterArray(pts_array, matrix_indexing=matrix_indexing)


# ! ---- Conversion routines

# The routines will eventualy be associated to the base class, but depending on the base
# type, the output differs.


def to_coordinate(
    self, coordinatesystem: Optional[darsia.CoordinateSystem] = None
) -> Union[Coordinate, CoordinateArray]:
    """Conversion of point to Coordinate.

    Args:
        coordinatesystem (CoordinateSystem, optional): coordinate system used for conversion

    Returns:
        Coordinate or CoordinateArray: Coordinate variant of point (type depends on input)

    """
    if isinstance(self, Coordinate):
        return self.copy()
    elif isinstance(self, Voxel) or isinstance(self, VoxelCenter):
        assert coordinatesystem is not None, "coordinatesystem must be provided"
        return make_coordinate(coordinatesystem.coordinate(self))
    else:
        raise NotImplementedError(f"{type(self)} not supported")


def to_voxel(
    self, coordinatesystem: Optional[darsia.CoordinateSystem] = None
) -> Union[Voxel, VoxelArray]:
    """Conversion of point to Voxel.

    Args:
        coordinatesystem (CoordinateSystem, optional): coordinate system used for conversion

    Returns:
        Voxel or VoxelArray: Voxel variant of point

    """
    if isinstance(self, Voxel):
        return self.copy()
    elif isinstance(self, VoxelCenter):
        return make_voxel(self)
    elif isinstance(self, Coordinate):
        assert coordinatesystem is not None, "coordinatesystem must be provided"
        return make_voxel(coordinatesystem.voxel(self))
    else:
        raise NotImplementedError(f"{type(self)} not supported")


def to_voxel_center(
    self, coordinatesystem: Optional[darsia.CoordinateSystem] = None
) -> Union[VoxelCenter, VoxelCenterArray]:
    """Conversion of point to VoxelCenter.

    Args:
        coordinatesystem (CoordinateSystem, optional): coordinate system used for conversion

    Returns:
        VoxelCenter or VoxelCenterArray: VoxelCenter variant of point

    """
    if isinstance(self, VoxelCenter):
        return self.copy()
    elif isinstance(self, Voxel):
        return make_voxel_center(self)
    elif isinstance(self, Coordinate):
        assert coordinatesystem is not None, "coordinatesystem must be provided"
        return make_voxel_center(coordinatesystem.voxel(self))
    else:
        raise NotImplementedError(f"{type(self)} not supported")


def to(self, cls, coordinatesystem: Optional[darsia.CoordinateSystem] = None):
    """Conversion of point to a different point type.

    Args:
        cls (type): class to convert to
        coordinatesystem (CoordinateSystem, optional): coordinate system used for conversion

    Returns:
        cls: cls variant of point

    """
    if cls in [Coordinate, CoordinateArray]:
        return self.to_coordinate(coordinatesystem)
    elif cls in [Voxel, VoxelArray]:
        return self.to_voxel(coordinatesystem)
    elif cls in [VoxelCenter, VoxelCenterArray]:
        return self.to_voxel_center(coordinatesystem)
    else:
        raise NotImplementedError(f"{cls} not supported")


# Assign method to base class
BasePoint.to_coordinate = to_coordinate  # type: ignore [attr-defined]
BasePoint.to_voxel = to_voxel  # type: ignore [attr-defined]
BasePoint.to_voxel_center = to_voxel_center  # type: ignore [attr-defined]
BasePoint.to = to  # type: ignore [attr-defined]
