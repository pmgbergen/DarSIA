"""Module containing approximation spaces and global linear approximations."""

from abc import ABC
from typing import Optional, Union

import numpy as np

import darsia


class ApproximationSpace(ABC):
    """Abstract class for approximation spaces."""

    @property
    def size(self) -> int:
        """Dimension of the approximation space."""
        ...

    def basis(self, x: np.ndarray, i: int) -> np.ndarray:
        """Evaluation of the basis functions.

        Args:
            x (np.ndarray): Coordinates.
            i (int): Index of the basis function.

        Returns:
            np.ndarray: Value of the basis function.

        """
        ...

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the basis functions at a given point.

        Args:
            x (np.ndarray): Coordinates.

        Returns:
            np.ndarray: Values of the basis functions.

        """
        return [self.basis(x, i) for i in range(self.size)]


class PolynomialApproximationSpace(ApproximationSpace):
    """Global polynomial approximation space in 2d.

    Class that provides a polynomial approximation space in 2d
    with basis functions x^i * y^j for i + j <= degree.

    """

    def __init__(self, degree: int) -> None:
        """Initialize the polynomial approximation space.

        Args:
            degree (int): Degree of the polynomial.

        """
        self.degree = degree

    @property
    def size(self) -> int:
        """Dimension of the polynomial approximation space."""
        return (self.degree + 1) * (self.degree + 2) // 2

    def basis(self, x: np.ndarray, k: int) -> np.ndarray:
        """Evaluation of the basis functions.

        Args:
            x (np.ndarray): Coordinates.
            k (int): Index of the basis function.

        Returns:
            np.ndarray: Value of the basis function.

        """
        i, j = divmod(k, self.degree + 1)
        return x[..., 0] ** i * x[..., 1] ** j


class RadialPolynomialApproximationSpace(ApproximationSpace):
    """Global radial polynomial approximation space in 2d.

    Class that provides a polynomial approximation space in 2d
    for given center with basis functions |x-center|^i, i <= degree.

    """

    def __init__(self, degree: int, center: Optional[np.ndarray] = None) -> None:
        """Initialize the radial polynomial approximation space.

        Args:
            degree (int): Degree of the polynomial.
            center (Optional[np.ndarray]): Center of the radial polynomial.

        """
        self.degree = degree
        self.center = center if center is not None else np.zeros(2)

    def set_center(self, center: np.ndarray) -> None:
        """Set the center of the radial polynomial.

        Args:
            center (np.ndarray): Center of the radial polynomial.

        """
        self.center = center

    @property
    def size(self) -> int:
        """Dimension of the radial polynomial approximation space."""
        return self.degree + 1

    def basis(self, x: np.ndarray, i: int) -> np.ndarray:
        """Evaluation of the basis functions.

        Args:
            x (np.ndarray): Coordinates.
            i (int): Index of the basis function.

        """
        return np.linalg.norm(x - self.center, axis=0) ** i


# TODO
# class PiecewiseLinearApproximationSpace(ApproximationSpace):
#     # Piecewise linear approximation space based on hat functions for given patches
#
#     def __init__(self, patches: darsia.Patches) -> None:
#         # Cache the patches
#         self.patches = patches
#
#     @ property
#     def dimension(self) -> int:
#         # The total number of corners of the patches
#         return self.patches.num_corners
#
#     def hat(self, x_local: np.ndarray, i: int) -> float:
#         """Evaluation of the 2d hat function.
#
#         Args:
#             x_local (np.ndarray): Local coordinates.
#             i (int): Index of the hat function.
#
#         Returns:
#             float: Value of the hat function.
#
#         """
#         # Get the corner coordinates of the patch
#         corner = self.patches.corners[i]
#         # Compute the hat function
#         return np.prod([max(0, 1 - abs(x - c)) for x, c in zip(x_local, corner)])
#
#     def basis(self, x: np.ndarray, i: int) -> np.ndarray:
#         ...


class LinearApproximation:
    """Linear combination of any approximation space.

    Coefficients can have any dimensionality.

    """

    def __init__(
        self, space: ApproximationSpace, dim: Union[int, tuple[int, int]]
    ) -> None:
        """Initialize the linear approximation.

        Args:
            space (ApproximationSpace): Approximation space.
            dim (Union[int, tuple[int, int]]): Dimension of the coefficients.

        """
        self.space = space
        self.shape = (space.size, dim) if isinstance(dim, int) else (space.size, *dim)
        self.size = np.prod(self.shape)
        # Initialize the coefficients
        self.coefficients = np.zeros(self.shape, dtype=float)

    def _evaluate(self, voxels: darsia.VoxelArray) -> np.ndarray:
        """Evaluate the linear combination on a given coordinate system.

        Args:
            voxels (darsia.VoxelArray): Voxel array.

        Returns:
            np.ndarray: Flat result.

        """
        # Evaluate the linear combination on a given coordinate system
        flat_result = np.zeros((len(voxels), 9))

        # Evaluate the basis functions
        for i in range(self.space.size):
            flat_result += np.outer(
                self.space.basis(voxels, i), np.ravel(self.coefficients[i], "F")
            )

        return flat_result

    def evaluate(
        self, input: Union[darsia.CoordinateSystem, darsia.VoxelArray]
    ) -> np.ndarray:
        """Evaluate the linear combination on a given input.

        Args:
            input (Union[darsia.CoordinateSystem, darsia.VoxelArray]): Input.

        Returns:
            np.ndarray: Result.

        """
        if isinstance(input, darsia.CoordinateSystem):
            # Consider all accessible voxels, and anticipate that the result
            # shall be reshaped to the shape of the coordinate system
            flat_result = self._evaluate(input.voxels)
            return flat_result.reshape((*input.shape, 3, 3), order="F")

        elif isinstance(input, darsia.VoxelArray):
            return self._evaluate(input)

        else:
            raise ValueError("Invalid input type.")
