from typing import Optional, Union

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator

import darsia as da


class CG(da.Solver):
    """Conjugate gradient solver based on scipy's cg for the problem:
    mass_coeff * x - diffusion_coeff * laplace(x) = rhs

    """

    def __call__(
        self, x0: np.ndarray, rhs: np.ndarray, h: Optional[Union[float, list]] = None
    ) -> np.ndarray:
        """Solve the problem.

        Args:
            x0 (np.ndarray): initial guess
            rhs (np.ndarray): right hand side
            h (float, optional): grid spacing

        Returns:
            np.ndarray: solution

        """
        # Safety check
        assert x0.shape == rhs.shape

        # Determine the degrees of freedom
        _dof = np.prod(x0.shape)

        # Define the action of the problem on a flat array
        def _mv(x: np.ndarray) -> np.ndarray:
            # Reshape the input to nd array
            x = np.reshape(x, x0.shape)
            # Apply the operator and flatten the output
            return (
                self.mass_coeff * x
                - da.laplace(x, dim=self.dim, h=h, diffusion_coeff=self.diffusion_coeff)
            ).flatten()

        # Define the linear operator
        lhsoperator = LinearOperator((_dof, _dof), matvec=_mv)

        # Solve the problem using scipys conjugate gradient solver
        x, _ = sps.linalg.cg(
            lhsoperator,
            rhs.flatten(),
            x0.flatten(),
            tol=self.tol,
            maxiter=self.maxiter,
        )

        # Return the solution reshaped to the original shape
        return np.reshape(x, x0.shape[: self.dim])
