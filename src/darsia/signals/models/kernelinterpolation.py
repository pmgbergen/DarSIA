"""Kernel-based interpolation.

The kernel-based interpolation is a general interpolation method that can be used
for different interpolation tasks. The interpolation is based on a kernel matrix
that is computed from the input data. The input data consists of a set of data
points (supports) and a set of goal points (values). The kernel matrix is
then used to compute the interpolation weights. The interpolation weights are
computed by solving a linear system of equations. The interpolation weights are
then used to compute the interpolated image.

One example use case is the concentration analysis.

"""

from typing import Literal, Optional, Union
from warnings import warn

import numpy as np

import darsia


class KernelInterpolation(darsia.Model):
    """General kernel-based interpolation."""

    def __init__(
        self,
        kernel: darsia.BaseKernel,
        supports: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
    ):
        """Setup of the kernel-based interpolation.

        Args:
            kernel (darsia.BaseKernel): kernel
            supports (np.ndarray): support points
            values (np.ndarray): goal points

        Raises:
            AssertionError: if the input data is not compatible, i.e., the number of
                supports and values does not match.

        """
        self.kernel = kernel
        """Kernel."""
        self.supports = supports
        """Support points."""
        self.values = values
        """Goal points."""
        self.num_supports = 0
        """Number of support points."""
        self.interpolation_weights = None
        """Interpolation weights."""
        self.update(kernel=kernel, supports=supports, values=values)

    def update(
        self,
        kernel: Optional[darsia.BaseKernel] = None,
        supports: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
    ) -> None:
        """Update kernel and interpolation weights.

        Args:
            kernel (darsia.BaseKernel, optional): kernel
            supports (np.ndarray, optional): support points
            values (np.ndarray, optional): goal points

        """
        # Convert to arrays if necessary
        if isinstance(supports, list):
            supports = np.array(supports)
        if isinstance(values, list):
            values = np.array(values)

        # Update single components of the interpolation
        if kernel is not None:
            self.update_kernel(kernel)
        if supports is not None:
            self.supports = supports
            self.num_supports = self.supports.shape[0]
        if values is not None:
            self.values = values
        if self.supports is not None and self.values is not None:
            self.update_interpolation()
        else:
            warn("No input data given. Interpolation weights not updated.")

    def update_kernel(self, kernel: darsia.BaseKernel) -> None:
        """Update kernel.

        Args:
            kernel (darsia.BaseKernel): kernel

        """
        self.kernel = kernel

    def update_interpolation(self) -> None:
        """Update interpolation weights.

        Raises:
            AssertionError: if the input data is not compatible, i.e., the number of
                supports and values does not match.

        """
        assert len(self.values) == self.num_supports, "Input data not compatible."

        # Reduce to unique supports for unique solvability
        self.supports, indices, counts = np.unique(
            np.round(self.supports, decimals=5),
            return_index=True,
            return_counts=True,
            axis=0,
        )
        # Warn the user that some supports were removed
        if not np.allclose(counts, 1):
            warn(
                f"Supports are not unique. {np.sum(counts - 1)} supports were removed."
            )
        # Adapt the remaining values
        self.num_supports = self.supports.shape[0]
        self.values = self.values[indices]

        # Solve the interpolation problem, i.e., set up the kernel matrix, and solve
        # for the interpolation weights
        X = np.ones((self.num_supports, self.num_supports))  # kernel matrix
        for i in range(self.num_supports):
            for j in range(self.num_supports):
                X[i, j] = self.kernel(self.supports[i], self.supports[j])
        self.interpolation_weights = np.linalg.solve(X, self.values)

    def update_model_parameters(
        self,
        parameters: np.ndarray,
        dofs: Optional[Union[list[Literal["kernel", "values"]], Literal["all"]]] = None,
    ) -> None:
        if "supports" in dofs:
            raise ValueError("Supports cannot be updated. Use update method.")

        if dofs is None or dofs == "all" or set(dofs) == set(["kernel", "values"]):
            self.update(
                kernel=parameters[0],
                values=parameters[1 : 1 + self.num_supports],
            )
        elif set(dofs) == set(["kernel"]):
            self.update(kernel=parameters[0])
        elif set(dofs) == set(["values"]):
            self.update(values=parameters[: self.num_supports])
        else:
            raise ValueError("Invalid dofs.")

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply interpolation.

        Args:
            signal (np.ndarray): signal to be interpolated

        Returns:
            np.ndarray: interpolated signal

        """
        # NOTE: Shape is not clear at input as it may be used via advenced indexing
        output = self.interpolation_weights[0] * self.kernel(signal, self.supports[0])
        for n in range(1, self.num_supports):
            output += self.interpolation_weights[n] * self.kernel(
                signal, self.supports[n]
            )
        return output
