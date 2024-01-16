"""Module encoding standard transformations used in computer graphics.

These combine translation, rotation, and scaling. A class encodes the map itself,
including automatic determining. In addition, the actual correction object is
contained.

"""
from __future__ import annotations

from typing import Optional, Union
from warnings import warn

import numpy as np
import scipy.optimize as optimize
from scipy.spatial.transform import Rotation

import darsia


class AffineTransformation(darsia.BaseTransformation):
    """Affine map, restricted to translation, scaling, rotation, resulting in
    conservation of angles.

    """

    # ! ---- Setter routines

    def __init__(
        self,
        dim: int,
    ) -> None:
        """Constructor for identity encoded as affine transformation.

        Args:
            dim (int): dimension of the Euclidean space

        Raises:
            ValueError: if not sufficient input data is provided.
            ValueError: if dimension not 2 or 3.

        """
        # Pre-define management of input/output types
        super().__init__()

        # Define identity
        self.dim = dim
        """Dimension of the Euclidean space."""
        self.translation = np.zeros(self.dim, dtype=float)
        """Translation vector."""
        self.scaling = 1.0
        """Scaling factor."""
        self.rotation = np.eye(self.dim)
        """Rotation matrix."""
        self.rotation_inv = np.eye(self.dim)
        """Inverse of rotation matrix."""
        self.isometry = False
        """Flag storing whether the underlying transformation is restricted to isometry."""

    def set_parameters(
        self,
        translation: Optional[np.ndarray] = None,
        scaling: Optional[float] = None,
        rotation: Optional[np.ndarray] = None,
    ) -> None:
        """Set-access of parameters of map.

        Args:
            translation (array, optional): translation vector.
            scaling (float, optional): scaling value.
            rotation (array, optional): rotation angles in radians for the rotation
                around the x, y, and z axis, respectively. In 2d, the length is 1. In
                3d, the length is 3.

        """
        if translation is not None:
            self.translation = translation

        if scaling is not None:
            self.scaling = scaling

        if rotation is not None:
            if self.dim == 2:
                assert len(rotation) == 1
                degree = rotation[0]
                vector = np.array([0, 0, 1])
                self.rotation = Rotation.from_rotvec(degree * vector).as_matrix()[
                    :2, :2
                ]
                self.rotation_inv = Rotation.from_rotvec(-degree * vector).as_matrix()[
                    :2, :2
                ]

            elif self.dim == 3:
                assert len(rotation) == 3

                # Initialize rotation matrices
                self.rotation = np.eye(self.dim)
                self.rotation_inv = np.eye(self.dim)

                for axis_index in range(3):
                    degree = rotation[axis_index]
                    cartesian_axis = "xyz"[axis_index]
                    matrix_axis, reverted = darsia.interpret_indexing(
                        cartesian_axis, "xyz"[: self.dim]
                    )
                    vector = np.eye(self.dim)[matrix_axis]
                    flip_factor = -1 if reverted else 1

                    rotation_matrix = Rotation.from_rotvec(
                        flip_factor * degree * vector
                    ).as_matrix()
                    rotation_matrix_inv = Rotation.from_rotvec(
                        -degree * vector
                    ).as_matrix()

                    self.rotation = np.matmul(self.rotation, rotation_matrix)
                    self.rotation_inv = np.matmul(
                        self.rotation_inv, rotation_matrix_inv
                    )

    def set_parameters_as_vector(self, parameters: np.ndarray) -> None:
        """Wrapper for set_parameters.

        Args:
            parameters (array): all parameters concatenated as array. In 2d, the length
                is either 3 (translation, rotation) or 4 (translation, scaling,
                rotation). In 3d, the length is either 6 (translation, rotation) or 7
                (translation, scaling, rotation).

        """
        num_rotations_dofs: int = 1 if self.dim == 2 else self.dim
        if self.isometry:
            assert len(parameters) == self.dim + num_rotations_dofs
        else:
            assert len(parameters) == self.dim + 1 + num_rotations_dofs
        translation = parameters[0 : self.dim]
        scaling = 1.0 if self.isometry else parameters[self.dim]
        rotation = parameters[-num_rotations_dofs:]

        self.set_parameters(translation, scaling, rotation)

    def fit(
        self,
        pts_src: Union[darsia.VoxelArray, darsia.CoordinateArray],
        pts_dst: Union[darsia.VoxelArray, darsia.CoordinateArray],
        fit_options: dict = {},
    ) -> bool:
        """Least-squares parameter fit based on source and target coordinates.

        Fits both forward and inverse map.

        Args:
            pts_src (VoxelArray or CoordinateArray): source points
            pts_dst (VoxelArray or CoordinateArray): target points
            fit_options (dict): options for the fit routine
                preconditioning (bool): Flag controlling whether the optimization
                    problem is preconditioned by estimating a better initial guess for
                    the translation.
                tol (float): tolerance for optimization
                maxiter (int): maximum number of iterations for optimization
                isometry (bool): Flag controlling whether the underlying transformation
                    is an isometry.

        Returns:
            bool: success of parameter fit

        """
        # Check input
        assert pts_src.shape == pts_dst.shape, "Shape mismatch."
        assert pts_src.shape[1] == self.dim, "Dimension mismatch."
        self.set_dtype(pts_src, pts_dst)

        # Fetch calibration options - update internal flag controlling the parameters
        # entering the optimization (isometry)
        preconditioning = fit_options.get("preconditioning", True)
        tol = fit_options.get("tol", 1e-2)
        maxiter = fit_options.get("maxiter", 100)
        self.isometry = fit_options.get("isometry", False)

        # For the initial guess, start with the identity
        if self.isometry:
            # Initial guess:
            # [*translation[0:2], rotation], if dim == 2
            # [*translation[0:3], *rotation[0:3]], if dim == 3
            identity_parameters = np.array(
                [0, 0, 0] if self.dim == 2 else [0, 0, 0, 0, 0, 0]
            )
        else:
            # Initial guess: same as for isometry, but including scaling at pos (dim+1)
            identity_parameters = np.array(
                [0, 0, 1, 0] if self.dim == 2 else [0, 0, 0, 1, 0, 0, 0]
            )

        # Allow for preconditioning of the problem, i.e., controlled modification of
        # the src points.
        if preconditioning:
            # Estimate better initial guess for the translation and correct the src
            # points. This will help the optimization to converge faster. Estimate the
            # translation by comparing the centers of mass of the src and dst points.
            center_src = np.mean(pts_src, axis=0)
            center_dst = np.mean(pts_dst, axis=0)
            preconditioning_translation = center_dst - center_src

            # Guess some better initial translation and update the src points
            # initial_translation = np.zeros(self.dim, dtype=float)
            pts_src = pts_src + preconditioning_translation

            # When preconditioning is enabled, the identity is used as initial guess
            # as the parameterswe use the initial guess as the
            initial_guess = identity_parameters
        else:
            # Allow the user to define a tailored initial guess, but use the identity
            # as default.
            initial_guess = fit_options.get("initial_guess", identity_parameters)

        # Define least squares objective function
        def objective_function(params: np.ndarray):
            self.set_parameters_as_vector(params)
            pts_mapped = self.call_array(pts_src)
            return np.sum((pts_dst - pts_mapped) ** 2)

        # Perform optimization step
        opt_result = optimize.minimize(
            objective_function,
            initial_guess,
            method="Powell",
            tol=tol,
            options={"maxiter": maxiter, "disp": True},
        )

        # Correct for preconditioning.
        if preconditioning:
            # Preliminary update of model parameters
            self.set_parameters_as_vector(opt_result.x)

            # Need to apply scaling and rotation to the translation vector
            scaled_rotated_preconditioning_translation = self.scaling * np.transpose(
                self.rotation.dot(
                    np.transpose(np.atleast_2d(preconditioning_translation))
                )
            )

            # Update overall translation.
            opt_result.x[0 : self.dim] = (
                opt_result.x[0 : self.dim] + scaled_rotated_preconditioning_translation
            )

        # Final update of model parameters
        self.set_parameters_as_vector(opt_result.x)

        # Check success
        if opt_result.success:
            print(
                f"Calibration successful with obtained model parameters {opt_result.x}."
            )
        else:
            warn(f"Calibration not successful. Obtained results: {opt_result.x}")

        return opt_result.success

    # ! ---- Application

    def call_array(self, x: np.ndarray) -> np.ndarray:
        """Application of map to arrays.

        Args:
            x (np.ndarray): (collection of) dim-dimensional Euclidean vector

        Returns:
            np.ndarray: function values of affine map

        """
        num, dim = x.shape
        assert dim == self.dim
        function_values = np.outer(
            np.ones(num), self.translation
        ) + self.scaling * np.transpose(self.rotation.dot(np.transpose(x)))

        return function_values

    def inverse_array(self, x: np.ndarray) -> np.ndarray:
        """Application of inverse of the map to arrays.

        Args:
            x (np.ndarray): (collection of) dim-dimensional Euclidean vector,

        Returns:
            np.ndarray: function values of affine inverse map

        """
        num, dim = x.shape
        assert dim == self.dim
        function_values = (
            1.0
            / self.scaling
            * np.transpose(
                self.rotation_inv.dot(
                    np.transpose(x - np.outer(np.ones(num), self.translation))
                )
            )
        )

        return function_values


class AffineCorrection(darsia.TransformationCorrection):
    """Affine correction based on affine transformation (translation, scaling,
    rotation).

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
            coordinatesystem_src (CoordinateSystem): coordinate system corresponding
                to voxels_src
            coordinatesystem_dst (CoordinateSystem): coordinate system corresponding
                to voxels_dst
            pts_src (CoordinateArray, VoxelArray, or VoxelCenterArray): source points
            pts_dst (CoordinateArray, VoxelArray, or VoxelCenterArray): target points
            fit_options (dict): options for the fit routine
                isometry (bool): Flag controlling whether the underlying transformation
                    is an isometry - in this case, the underlying transformation will
                    operate in the Coordinate space.

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

        # If isometry is turned on, make sure to use the coordinates of voxel centers
        isometry = fit_options.get("isometry", False)
        if isometry:
            pts_src = pts_src.to_voxel_center().to_coordinate(self.coordinatesystem_src)
            pts_dst = pts_dst.to_voxel_center().to_coordinate(self.coordinatesystem_dst)

        affine_transformation = AffineTransformation(self.dim)
        affine_transformation.fit(pts_src, pts_dst, fit_options)

        super().__init__(
            coordinatesystem_src=coordinatesystem_src,
            coordinatesystem_dst=coordinatesystem_dst,
            transformation=affine_transformation,
        )
