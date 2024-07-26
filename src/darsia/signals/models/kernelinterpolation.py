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
        supports: Optional[np.ndarray | list] = None,
        values: Optional[np.ndarray | list] = None,
        append: bool = False,
    ) -> None:
        """Update kernel and interpolation weights.

        Args:
            kernel (darsia.BaseKernel, optional): kernel
            supports (np.ndarray, optional): support points
            values (np.ndarray, optional): goal points
            append (bool): flag to append new data to existing data

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
            if self.supports is None or not append:
                self.supports = supports.astype(np.float32)
            else:
                self.supports = np.vstack((self.supports, supports)).astype(np.float32)
            self.num_supports = self.supports.shape[0]
            if hasattr(self, "Xinv"):
                del self.Xinv
        if values is not None:
            if self.values is None or not append:
                self.values = values
            else:
                self.values = np.hstack((self.values, values))
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

    def setup_kernel_problem(self) -> None:
        """Setup of linear kernel problem."""
        assert (
            len(self.values) == self.num_supports
        ), f"Input data not compatible: {len(self.values)} != {self.num_supports}."

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
        self.X = np.ones((self.num_supports, self.num_supports))  # kernel matrix
        for i in range(self.num_supports):
            for j in range(self.num_supports):
                self.X[i, j] = self.kernel(self.supports[i], self.supports[j])
        self.Xinv = np.linalg.inv(self.X)

    def update_interpolation(self) -> None:
        """Update interpolation weights.

        Raises:
            AssertionError: if the input data is not compatible, i.e., the number of
                supports and values does not match.

        """
        if not hasattr(self, "Xinv"):
            self.setup_kernel_problem()
        self.interpolation_weights = self.Xinv @ self.values

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
        if self.supports is None or self.interpolation_weights is None:
            # NOTE: Currently only scalar output supported - forced here.
            return np.zeros(signal.shape[:1], dtype=np.float32)
        else:
            return self.kernel.linear_combination(
                signal.astype(np.float32),
                self.supports,
                self.interpolation_weights.astype(np.float32),
            )


class AdvancedKernelInterpolation(KernelInterpolation):

    def __init__(self, kernel: darsia.BaseKernel) -> None:
        super().__init__(kernel=kernel)

        self.fixed_supports = None
        """Fixed support points."""
        self.fixed_values = None
        """Fixed goal points."""
        self.num_fixed_supports = 0
        """Number of fixed support points."""
        self.variable_supports = None
        """Variable support points."""
        self.variable_values = None
        """Variable goal points."""
        self.num_variable_supports = 0

    def update_advanced(
        self,
        fixed_supports: Optional[np.ndarray] = None,
        fixed_values: Optional[np.ndarray] = None,
        variable_supports: Optional[np.ndarray] = None,
        variable_values: Optional[np.ndarray] = None,
    ) -> None:
        """Update support points.

        Args:
            fixed_supports (np.ndarray, optional): fixed support points
            fixed_values (np.ndarray, optional): fixed goal points
            variable_supports (np.ndarray, optional): variable support points
            variable_values (np.ndarray, optional): variable goal points

        """
        # Convert to arrays if necessary
        if isinstance(fixed_supports, list):
            fixed_supports = np.array(fixed_supports)
        if isinstance(fixed_values, list):
            fixed_values = np.array(fixed_values)
        if isinstance(variable_supports, list):
            variable_supports = np.array(variable_supports)
        if isinstance(variable_values, list):
            variable_values = np.array(variable_values)

        # Update single components of the interpolation
        if fixed_supports is not None:
            self.fixed_supports = fixed_supports.astype(np.float32)
        if fixed_values is not None:
            self.fixed_values = fixed_values
        if variable_supports is not None:
            self.variable_supports = variable_supports.astype(np.float32)
        if variable_values is not None:
            self.variable_values = variable_values

        # Put the data into right shape
        self.fixed_supports = self.fixed_supports.reshape(-1, 3)
        self.variable_supports = self.variable_supports.reshape(-1, 3)

        # Make sure that the data is compatible
        assert (self.fixed_supports is None and self.fixed_values is None) or len(
            self.fixed_values
        ) == self.fixed_supports.shape[0], "Input data not compatible."
        assert (self.variable_supports is None and self.variable_values is None) or len(
            self.variable_values
        ) == self.variable_supports.shape[0], "Input data not compatible."
        self.num_fixed_supports = self.fixed_supports.shape[0]
        self.num_variable_supports = self.variable_supports.shape[0]

        # Update the interpolation weights for concatenated supports and values
        self.update(
            supports=np.vstack((self.fixed_supports, self.variable_supports)),
            values=np.hstack((self.fixed_values, self.variable_values)),
        )

    def update_variable_model_parameters(self, parameters: np.ndarray) -> None:
        assert (
            len(parameters) == self.num_variable_supports
        ), "Input data not compatible."
        self.update_advanced(
            variable_values=parameters,
        )
