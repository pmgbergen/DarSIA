"""Module for general shape transformations."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

import darsia


class BaseTransformation(ABC):
    """Base class for geometrical transformations of images."""

    def __init__(self) -> None:
        self.input_dtype = np.ndarray
        """Type of input for generalized perspective transformation"""
        self.output_dtype = np.ndarray
        """Type of output for generalized perspective transformation"""
        self.input_array_dtype = np.ndarray
        """Type of input arrays for generalized perspective transformation"""
        self.output_array_dtype = np.ndarray
        """Type of output arrays for generalized perspective transformation"""

    def set_dtype(
        self,
        pts_src: Union[darsia.Coordinate, darsia.Voxel, darsia.VoxelCenter],
        pts_dst: Union[darsia.Coordinate, darsia.Voxel, darsia.VoxelCenter],
    ) -> None:
        """Set array input and output type for generalized perspective transformation.

        Args:
            pts_src (Union[darsia.Coordinate, darsia.Voxel, darsia.VoxelCenter]): source
                points
            pts_dst (Union[darsia.Coordinate, darsia.Voxel, darsia.VoxelCenter]): target
                points

        """
        # Assert (implicitly) pts_src and pts_dst are lists of coordinates or voxels.
        assert pts_src.shape == pts_dst.shape, "source and target points must match"

        # Update input and output type
        self.input_dtype = type(pts_src[0])
        self.output_dtype = type(pts_dst[0])

        # Update array input and output type
        if self.input_dtype == darsia.Coordinate:
            self.input_array_dtype = darsia.CoordinateArray
        elif self.input_dtype == darsia.Voxel:
            self.input_array_dtype = darsia.VoxelArray
        elif self.input_dtype == darsia.VoxelCenter:
            self.input_array_dtype = darsia.VoxelCenterArray
        elif self.input_dtype == np.ndarray:
            self.input_array_dtype = np.ndarray
        else:
            raise ValueError("input type not supported")

        if self.output_dtype == darsia.Coordinate:
            self.output_array_dtype = darsia.CoordinateArray
        elif self.output_dtype == darsia.Voxel:
            self.output_array_dtype = darsia.VoxelArray
        elif self.output_dtype == darsia.VoxelCenter:
            self.output_array_dtype = darsia.VoxelCenterArray
        elif self.output_dtype == np.ndarray:
            self.output_array_dtype = np.ndarray
        else:
            raise ValueError("output type not supported")

    @abstractmethod
    def set_parameters_as_vector(self, parameters: np.ndarray) -> None:
        """Set parameters of transformation as vector.

        Required for parameter fitting through optimization.

        Args:
            parameters (np.ndarray): parameters of transformation

        """
        ...

    @abstractmethod
    def fit(
        self,
        pts_src: Union[
            darsia.CoordinateArray, darsia.VoxelArray, darsia.VoxelCenterArray
        ],
        pts_dst: Union[
            darsia.CoordinateArray, darsia.VoxelArray, darsia.VoxelCenterArray
        ],
        **kwargs
    ) -> None:
        """Fit parameters of transformation.

        Required for parameter fitting through optimization.

        """
        ...

    def __call__(
        self,
        x: Union[np.ndarray, darsia.Coordinate, darsia.Voxel, darsia.VoxelCenter],
    ) -> Union[np.ndarray, darsia.Coordinate, darsia.Voxel, darsia.VoxelCenter]:
        """Apply transformation to array.

        Args:
            x (np.ndarray): array to transform (type depends on input type
                of transformation)

        Returns:
            np.ndarray: transformed array

        """
        # For now, convert to plain numpy array
        x_arr = np.asarray(x)

        # For dimensionality reasons, collection of points require different treatment
        # than single points; the below code is written for arrays with columns as
        # points
        x_arr = np.atleast_2d(x_arr)
        array_input = x_arr.shape == x.shape

        # Evaluate inverse transformation in terms of arrays
        out_arr = self.call_array(x_arr)

        # Convert to right output type
        if array_input:
            return self.output_array_dtype(out_arr)
        else:
            return self.output_dtype(out_arr[0])

    def inverse(
        self,
        x: Union[np.ndarray, darsia.Coordinate, darsia.Voxel, darsia.VoxelCenter],
    ) -> Union[np.ndarray, darsia.Coordinate, darsia.Voxel, darsia.VoxelCenter]:
        """Apply inverse transformation to array.

        Args:
            x (np.ndarray): array to transform (type depends on input type
                of transformation)

        Returns:
            np.ndarray: transformed array

        """
        # For now, convert to plain numpy array
        x_arr = np.asarray(x)

        # For dimensionality reasons, collection of points require different treatment
        # than single points; the below code is written for arrays with columns as
        # points
        x_arr = np.atleast_2d(x_arr)
        array_input = x_arr.shape == x.shape

        # Evaluate inverse transformation in terms of arrays
        out_arr = self.inverse_array(x_arr)

        # Convert to right output type
        if array_input:
            return self.input_array_dtype(out_arr)
        else:
            return self.input_dtype(out_arr[0])

    @abstractmethod
    def call_array(self, x: np.ndarray) -> np.ndarray:
        """Apply transformation to array.

        Args:
            x (np.ndarray): array to transform (type depends on input type
                of transformation)

        Returns:
            np.ndarray: transformed array

        """
        ...

    @abstractmethod
    def inverse_array(self, x: np.ndarray) -> np.ndarray:
        """Apply inverse transformation to array.

        Args:
            x (np.ndarray): array to transform (type depends on input type
                of transformation)

        Returns:
            np.ndarray: transformed array

        """
        ...


class TransformationCorrection(darsia.BaseCorrection):
    """Class for applying a transformation to an image as correction."""

    def __init__(
        self,
        coordinatesystem_src: darsia.CoordinateSystem,
        coordinatesystem_dst: darsia.CoordinateSystem,
        transformation: BaseTransformation,
    ) -> None:
        self.coordinatesystem_src = coordinatesystem_src
        """Coordinate system of source image."""
        self.coordinatesystem_dst = coordinatesystem_dst
        """Coordinate system of destination image."""
        self.transformation = transformation
        """Transformation to apply as correction."""

    def correct_array(self, array_src: np.ndarray) -> np.ndarray:
        """Correction routine of array data.

        Args:
            image_src (np.ndarray): array corresponding to some source image

        Returns:
            np.ndarray: array corresponding to some destination image

        """
        # Strategy: Warp entire array by mapping target voxels to destination voxels by
        # applying the inverse mapping.
        dim = self.coordinatesystem_src.dim
        shape = *self.coordinatesystem_dst.shape, *list(array_src.shape)[dim:]
        array_dst = np.zeros(shape, dtype=array_src.dtype)

        # Collect all target voxels in num_voxels_dst x dim format, and shift to centers.
        voxels_dst = self.coordinatesystem_dst.voxels

        # Find corresponding voxels in the original image by applying the inverse map.
        # This depends on how the transformation is set up. Follow a 3-step strategy:

        # 1. Determine input of transformation - via voxel centers.
        transformation_input = voxels_dst.to_voxel_center().to(
            self.transformation.input_dtype, self.coordinatesystem_dst
        )

        # 2. Apply inverse transformation
        transformation_output = self.transformation.inverse(transformation_input)

        # 3. Convert output to voxel
        voxels_src = transformation_output.to_voxel(self.coordinatesystem_src)

        # Determine active voxels - have to lie within active coordinate system
        valid_voxels = np.all(
            np.logical_and(
                voxels_src >= np.zeros(dim, dtype=int),
                voxels_src < self.coordinatesystem_src.shape,
            ),
            axis=1,
        )

        # Warp. Assign voxel values (no interpolation)
        array_dst[tuple(voxels_dst[valid_voxels, j] for j in range(dim))] = array_src[
            tuple(voxels_src[valid_voxels, j] for j in range(dim))
        ]

        return array_dst
