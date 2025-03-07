"""Module containing coordinate transformation.

The transformatiosn are restricted to affine transformations (translation, rotation,
scaling). A coordinate transformation alters the voxel data and the metadata.

"""

from __future__ import annotations

import copy

# Check if on a windows machine
import platform
from typing import Union

if platform.system() == "Windows":
    lir = None
else:
    import largestinteriorrectangle as lir

import numpy as np

import darsia


class CoordinateTransformation:
    """
    General affine transformation (translation, scaling, rotation),
    applicable for general (up to 4d) images.

    NOTE: Inherit from base correction to make use of the plain array correction
    routines but complement with meta corrections.

    """

    def __init__(
        self,
        coordinatesystem_src: darsia.CoordinateSystem,
        coordinatesystem_dst: darsia.CoordinateSystem,
        pts_src: Union[
            darsia.CoordinateArray, darsia.VoxelArray, darsia.VoxelCenterArray
        ],
        pts_dst: Union[
            darsia.CoordinateArray, darsia.VoxelArray, darsia.VoxelCenterArray
        ],
        fit_options: dict = {},
    ) -> None:
        """Constructor.

        Args:
            coordinatesystem_src (CoordinateSystem): source coordinate system
            coordinatesystem_dst (CoordinateSystem): target coordinate system
            pts_src (CoordinateArray, VoxelArray, VoxelCenterArray): source points
            pts_dst (CoordinateArray, VoxelArray, VoxelCenterArray): target points
            fit_options (dict): options for the affine fit

        """
        # Cache coordinate systems
        self.coordinatesystem_src = coordinatesystem_src
        """Coordinate system corresponding to the input."""

        self.coordinatesystem_dst = coordinatesystem_dst
        """Coordinate system corresponding to the output/target."""

        # Fetch additional properties
        assert self.coordinatesystem_src.dim == self.coordinatesystem_dst.dim
        self.dim = self.coordinatesystem_src.dim
        """Dimension of the underlying Euclidean spaces."""

        self.affine_correction = darsia.AffineCorrection(
            coordinatesystem_src,
            coordinatesystem_dst,
            pts_src,
            pts_dst,
            fit_options,
        )
        """Correction object for the array data."""

    def find_intersection(self) -> tuple[slice, slice]:
        """Determine the active canvas in coordinatesystem_dst, covered by
        coordinatesystem_src after transformed onto the target canvas.

        NOTE: Only supported for 2d.
        NOTE: Requires extra dependency.

        Returns:
            tuple of slices: voxel intervals ready to be used to extract subregions.

        Raises:
            NotImplementedError: if dimension not 2
            ImportError: if Python package largestinteriorrectangle not installed.

        """

        if self.dim not in [2, 3]:
            raise NotImplementedError(
                "Intersection option only supported in 2d and 3d."
            )

        # Find the voxel locations of the corners in the source array - need them
        # sorted.
        shape_src = self.coordinatesystem_src.shape
        if self.dim == 2:
            corner_voxels_src: darsia.VoxelArray = darsia.make_voxel(
                [
                    [0, 0],
                    [shape_src[0], 0],
                    [shape_src[0], shape_src[1]],
                    [0, shape_src[1]],
                ]
            )
        elif self.dim == 3:
            corner_voxels_src: darsia.VoxelArray = darsia.make_voxel(
                [
                    [0, 0, 0],
                    [shape_src[0], 0, 0],
                    [shape_src[0], shape_src[1], 0],
                    [0, shape_src[1], 0],
                    [0, 0, shape_src[2]],
                    [shape_src[0], 0, shape_src[2]],
                    [shape_src[0], shape_src[1], shape_src[2]],
                    [0, shape_src[1], shape_src[2]],
                ]
            )

        # Map these to the target canvas
        assert False, "TODO: Use 3 stage recipe from BaseTransformationorrection"
        if self.affine_correction.use_cartesian:
            corner_coordinates_src = corner_voxels_src.to_coordinate(
                self.coordinatesystem_src
            )
            corner_coordinates_dst = self.affine_correction.transformation(
                corner_coordinates_src
            )
            corner_voxels_dst = corner_coordinates_dst.to_voxel(
                self.coordinatesystem_dst
            )
        else:
            corner_voxels_dst = self.affine_correction.transformation(corner_voxels_src)

        # Clip to active canvas
        num_corners = len(corner_voxels_src)
        shape_dst = self.coordinatesystem_dst.shape
        active_corner_voxels_dst = np.clip(
            corner_voxels_dst,
            0,
            np.outer(np.ones(num_corners), np.array(shape_dst) - 1),
        )

        # Determine the largest interior rectangle - require to transform to format
        # expected by lir
        if self.dim == 2:
            if lir is None:
                raise ImportError(
                    "Python package largestinteriorrectangle not installed."
                )
            else:
                lir_dst = lir.lir(np.array([active_corner_voxels_dst]).astype(np.int32))
                rectangle_mask_corners = [lir.pt1(lir_dst), lir.pt2(lir_dst)]
                return (
                    slice(rectangle_mask_corners[0][0], rectangle_mask_corners[1][0]),
                    slice(rectangle_mask_corners[0][1], rectangle_mask_corners[1][1]),
                )
        elif self.dim == 3:
            # NOTE: In 3d, not the largest interior but smallest exterior rectangle is
            # returned, which is not ideal, but is easier to access. The application of
            # lir requires the input to be ordered and oriented in a specific way. In
            # addition, it only works for 2d arrays. Therefore, we need to apply lir
            # for each plane separately. After all, it is not straightforward: FIXME.

            return (
                slice(
                    int(np.min(active_corner_voxels_dst[:, 0])),
                    int(np.max(active_corner_voxels_dst[:, 0])),
                ),
                slice(
                    int(np.min(active_corner_voxels_dst[:, 1])),
                    int(np.max(active_corner_voxels_dst[:, 1])),
                ),
                slice(
                    int(np.min(active_corner_voxels_dst[:, 2])),
                    int(np.max(active_corner_voxels_dst[:, 2])),
                ),
            )

    def correct_metadata(self, image: darsia.Image) -> dict:
        """Correction routine of metadata.

        Args:
            image (darsia.Image): image corresponding to some source image

        Returns:
            dict: metadata corresponding to a destination image

        """
        # Fetch src meta
        meta_src = image.metadata()

        # Start with copy
        meta_dst = copy.copy(meta_src)

        # Modify dimensions
        meta_dst["dimensions"] = self.coordinatesystem_dst.dimensions

        # Modify origin
        meta_dst["origin"] = self.coordinatesystem_dst._coordinate_of_origin_voxel

        return meta_dst

    def __call__(self, image: darsia.Image) -> darsia.Image:
        """Main routine, transforming an image and its meta data.

        Args:
            image (darsia.Image): input image

        Returns:
            darsia.Image: transformed image

        """
        # Transform the image data (without touching the meta)
        transformed_image_with_original_meta = self.affine_correction(
            image, overwrite=False
        )

        # Transform the meta
        transformed_meta = self.correct_metadata(image)

        # Define the transformed image with tranformed meta
        return type(image)(transformed_image_with_original_meta.img, **transformed_meta)
