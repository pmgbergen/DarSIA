"""Module containing geometrical transformations for Image objects,
which also change the metadata (opposing to correction routines
aiming at modifying arrays only).

"""
import copy
import itertools
from typing import Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation

import darsia


class AngularConservativeAffineMap:
    """Affine map, restricted to translation, scaling, rotation,
    resulting in conservation of angles.

    """

    def __init__(
        self,
        pts_src: Optional[list] = None,
        pts_dst: Optional[list] = None,
        shape: Optional[tuple] = None,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            pts_src (list): voxel coordinates corresponding to source data
            pts_dst (list): voxel coordinates corresponding to destination data

        NOTE: If no input is provided, an identity is constructed.

        Raises:
            ValueError: if not sufficient input data is provided.
            ValueError: if dimension not 2 or 3.

        """

        # ! ---- Dimension

        # Determine dimensionality from input - perform additional safety checks
        if pts_src is not None and pts_dst is not None:
            assert len(pts_src) == len(pts_dst)
            assert all([len(pts_src[0]) == len(p) for p in pts_src])
            assert all([len(pts_dst[0]) == len(p) for p in pts_dst])
            assert len(pts_src[0]) == len(pts_dst[0])
            dim = len(pts_src[0])

        elif pts_src is not None or pts_dst is not None:
            raise ValueError

        else:
            assert "dim" in kwargs
            dim = kwargs.get("dim")

        if dim not in [2, 3]:
            raise ValueError
        self.dim = dim
        """Dimension of the Euclidean space."""

        # ! ---- Target space / canvas

        self.shape = shape
        """Shape of target arrays, defining canvas."""

        # ! ---- Map

        # If data provided, fit the parameters accordingly
        if pts_src is not None and pts_dst is not None:

            self.fit(pts_src, pts_dst)

        else:

            # Define identity
            self.translation = np.zeros(self.dim, dtype=float)
            """Translation vector."""

            self.scaling = 1.0
            """Scaling factor."""

            self.rotation = np.eye(self.dim)
            """Rotation matrix."""

            self.rotation_inv = np.eye(self.dim)
            """Inverse of rotation matrix."""

    def set_parameters(
        self,
        translation: Optional[np.ndarray] = None,
        scaling: Optional[float] = None,
        rotations: Optional[list[Union[tuple[float, str], np.ndarray]]] = None,
    ) -> None:
        """Set-access of parameters of map.

        Args:
            translation (array, optional): translation vector
            scaling (float, optional): scaling value
            rotations (array or list of tuples, optional): rotation angles and axes

        """
        if translation is not None:
            self.translation = translation

        if scaling is not None:
            self.scaling = scaling

        if rotations is not None:

            if self.dim == 2:
                degree = rotations[0]
                vector = np.array([0, 0, 1])

                rotation = Rotation.from_rotvec(degree * vector)
                rotation_inv = Rotation.from_rotvec(-degree * vector)

                self.rotation = rotation.as_matrix()[:2, :2]
                self.rotation_inv = rotation_inv.as_matrix()[:2, :2]

            elif self.dim == 3:

                # TODO why initialization needed?
                self.rotation = np.eye(self.dim)
                self.rotation_inv = np.eye(self.dim)

                for degree, cartesian_axis in rotations:
                    matrix_axis, reverted = darsia.interpret_indexing(
                        cartesian_axis, "xyz"[: self.dim]
                    )
                    vector = np.eye(self.dim)[matrix_axis]
                    scaling = -1 if reverted else 1

                    rotation = Rotation.from_rotvec(scaling * degree * vector)
                    rotation_inv = Rotation.from_rotvec(-degree * vector)

                    self.rotation = np.matmul(self.rotation, rotation.as_matrix())
                    self.rotation_inv = np.matmul(
                        self.rotation_inv, rotation_inv.as_matrix()
                    )

    def set_parameters_as_vector(self, parameters: np.ndarray) -> None:
        """Wrapper for set_parameters.

        Args:
            parameters (array): all parameters concatenated as array.

        """
        rotations_dofs = 1 if self.dim == 2 else self.dim
        assert len(parameters) == self.dim + 1 + rotations_dofs
        translation = parameters[0 : self.dim]
        scaling = parameters[self.dim]
        rotations = parameters[-rotations_dofs]

        self.set_parameters(translation, scaling, rotations)

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """Application of map.

        Args:
            array (np.ndarray): (collection of) dim-dimensional Euclidean vector

        Returns:
            np.ndarray: function values of affine map

        """
        dim, num = np.atleast_2d(array).shape
        assert dim == self.dim
        return np.outer(
            self.translation, np.ones(num)
        ) + self.scaling * self.rotation.map(array)

    def inverse(self, array: np.ndarray) -> np.ndarray:
        """Application of inverse of the map.

        Args:
            array (np.ndarray): (collection of) dim-dimensional Euclidean vector

        Returns:
            np.ndarray: function values of affine inverse map

        """
        dim, num = np.atleast_2d(array).shape
        assert dim == self.dim

        return -np.outer(
            self.translation, np.ones(num)
        ) + 1.0 / self.scaling * self.rotation_inv.dot(array)

    def fit(self, pts_src, pts_dst) -> None:
        """Least-squares parameter fit based on source and target coordinates.

        Args:
            pts_src (list): coordinates corresponding to source data
            pts_dst (list): coordinates corresponding to destination data

        """
        pass


class CoordinateTransformation(darsia.BaseCorrection):
    """
    General affine transformation (translation, scaling, rotation),
    applicable for general (up to 4d) images.

    NOTE: Inherit from base correction to make use of the plain array correction
    routines but complement with meta corrections.

    """

    def __init__(self, pts_src: list, pts_dst: list, **kwargs) -> None:
        """Constructor.

        Args:
            pts_src (list): voxel coordinates corresponding to source data
            pts_dst (list): voxel coordinates corresponding to destination data

        """
        # Construct
        self.angular_conservative_map = AngularConservativeAffineMap(
            pts_src=None, pts_dst=None, dim=2
        )

    def correct_array(self, array_src: np.ndarray) -> np.ndarray:
        """Correction routine of array data.

        Args:
            image_src (np.ndarray): array corresponding to some source image

        Returns:
            np.ndarray: array corresponding to some destination image

        """
        # TODO consider resize before and after for better results (does not work in 3d)

        # Strategy: Warp entire array by mapping target voxels to destination voxels by
        # applying the inverse mapping.

        # Implicitly assume the mapped image is of same size as the input image
        shape = array_src.shape  # TODO use shape...
        num_voxels = np.prod(array_src.shape[: self.dim])

        # Collect all voxels in dim x num_voxels format
        if self.dim == 2:
            voxels_dst = list(itertools.product(*[range(shape[0]), range(shape[1])]))
        elif self.dim == 3:
            voxels_dst = list(
                itertools.product(*[range(shape[0]), range(shape[1]), range(shape[2])])
            )
        voxels_dst = np.transpose(np.array(voxels_dst))

        # Find corresponding voxels in the original image by applying the inverse map
        voxels_src = self.angular_conservative_map.inverse(voxels_dst)

        # Remove pixels outside the ROI
        voxels_src = np.clip(
            voxels_src.astype(int),
            0,
            np.outer(np.array(shape) - 1, np.ones(num_voxels)),
        ).astype(int)

        # Assign pixel values (no interpolation)
        array_dst = np.zeros(shape)
        array_dst[tuple(voxels_dst[j] for j in range(self.dim))] = array_src[
            tuple(voxels_src[j] for j in range(self.dim))
        ]
        return array_dst

    def correct_metadata(self, meta_src: dict) -> dict:
        """Correction routine of metadata.

        Args:
            meta_src (dict): metadata corresponding to some source image

        Returns:
            dict: metadata corresponding to a destination image

        """
        # Start with copy
        meta_dst = copy.copy(meta_src)

        # Modify dimensions
        meta_dst["dimensions"] = meta_src["dimensions"]

        # Modify origin
        meta_dst["origin"] = meta_src["origin"]

        return meta_dst

    def __call__(self, image: darsia.Image) -> darsia.Image:
        """Main routine, transforming an image and its meta.

        Args:
            image (darsia.Image): input image

        Returns:
            darsia.Image: transformed image

        """
        # Transform the image data
        self.dim = image.space_dim
        transformed_image = super().__call__(image, return_image=True)

        # Transform the meta
        meta = image.metadata()
        transformed_meta = self.correct_metadata(meta)

        return type(image)(transformed_image.img, **transformed_meta)
