from typing import Union

import numpy as np
import scipy as sp


class AndersonAcceleration:
    """Anderson acceleration as described by Walker and Ni in doi:10.2307/23074353."""

    def __init__(self, dimension: Union[int, tuple[int]], depth: int = 0) -> None:
        """Initialize Anderson acceleration."""

        if isinstance(dimension, np.integer):
            self._dimension = dimension
            self._tensor = False
        elif isinstance(dimension, tuple):
            self.tensor_shape = dimension
            self._dimension = int(np.prod(dimension))
            self._tensor = True
        else:
            raise ValueError("Dimension not recognized.")

        self._depth = depth

        # Initialize arrays for iterates.
        self.reset()
        self._fkm1: np.ndarray = self._Fk.copy()
        self._gkm1: np.ndarray = self._Gk.copy()

    def reset(self) -> None:
        """Reset Anderson acceleration."""
        self._Fk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in increments
        self._Gk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in fixed point applications

    def __call__(self, gk: np.ndarray, fk: np.ndarray, iteration: int) -> np.ndarray:
        """Apply Anderson acceleration.

        Args:
            gk (array): application of some fixed point iteration onto approximation xk,
                i.e., g(xk).
            fk (array): residual g(xk) - xk; in general some increment.
            iteration (int): current iteration count.

        Returns:
            array: next approximation.

        """

        if self._tensor:
            gk = np.ravel(gk)
            fk = np.ravel(fk)

        if iteration == 0:
            self._Fk = np.zeros((self._dimension, self._depth))  # changes in increments
            self._Gk = np.zeros(
                (self._dimension, self._depth)
            )  # changes in fixed point applications

        mk = min(iteration, self._depth)

        # Apply actual acceleration (not in the first iteration).
        if mk > 0:
            # Build matrices of changes
            col = (iteration - 1) % self._depth
            self._Fk[:, col] = fk - self._fkm1
            self._Gk[:, col] = gk - self._gkm1

            # Solve least squares problem
            lstsq_solution = sp.linalg.lstsq(self._Fk[:, 0:mk], fk)
            gamma_k = lstsq_solution[0]
            # Do the mixing
            xkp1 = gk - np.dot(self._Gk[:, 0:mk], gamma_k)
        else:
            xkp1 = gk

        # Store values for next iteration
        self._fkm1 = fk.copy()
        self._gkm1 = gk.copy()

        if self._tensor:
            xkp1 = np.reshape(xkp1, self.tensor_shape)

        return xkp1
