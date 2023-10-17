"""Kernel-based interpolation.

The kernel-based interpolation is a general interpolation method that can be used
for different interpolation tasks. The interpolation is based on a kernel matrix
that is computed from the input data. The input data consists of a set of data
points (colours) and a set of goal points (concentrations). The kernel matrix is
then used to compute the interpolation weights. The interpolation weights are
computed by solving a linear system of equations. The interpolation weights are
then used to compute the interpolated image.

One example use case is the concentration analysis.

"""

import numpy as np

import darsia


class KernelInterpolation:
    """General kernel-based interpolation."""

    def __init__(
        self, kernel: darsia.BaseKernel, colours: np.ndarray, concentrations: np.ndarray
    ):
        """Setup of the kernel-based interpolation.

        Args:
            kernel (darsia.BaseKernel): kernel
            colours (np.ndarray): colours (support points)
            concentrations (np.ndarray): concentrations (goal points)

        Raises:
            AssertionError: if the input data is not compatible, i.e., the number of
                colours and concentrations does not match.

        """

        self.kernel = kernel
        """Kernel."""

        num_data = colours.shape[0]
        assert len(concentrations) == num_data, "Input data not compatible."

        x = np.array(colours)  # data points / control points / support points
        y = np.array(concentrations)  # goal points
        X = np.ones((num_data, num_data))  # kernel matrix
        for i in range(num_data):
            for j in range(num_data):
                X[i, j] = self.kernel(x[i], x[j])

        alpha = np.linalg.solve(X, y)

        # Cache
        self.x = x
        """Support points."""

        self.alpha = alpha
        """Interpolation weights."""

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply interpolation.

        Args:
            signal (np.ndarray): signal to be interpolated

        Returns:
            np.ndarray: interpolated signal

        """
        ph_image = np.zeros(signal.shape[:2])
        for n in range(self.alpha.shape[0]):
            ph_image += self.alpha[n] * self.kernel(signal, self.x[n])
        return ph_image
