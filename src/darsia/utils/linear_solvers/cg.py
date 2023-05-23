from typing import Optional, Union

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator

import darsia as da


class CG(da.Solver):

    """Conjugate gradient solver based on scipy's cg.
    For the problem: mass_coeff * x - diffusion_coeff * laplace(x) = rhs
    """

    def __init__(
        self,
        mass_coeff: Union[float, np.ndarray],
        diffusion_coeff: Union[float, np.ndarray],
        dim: int = 2,
        maxiter: int = 100,
        tol: Optional[float] = None,
        verbose=False,
    ) -> None:
        """
        Initialize the solver.

        Args:
            mass_coeff [np.ndarray,float]: mass coefficient
            diffusion_coeff [np.ndarray,float]: diffusion coefficient
            dim [int]: dimension of the problem
            maxiter [int]: maximum number of iterations
            tol [Optional[float]]: tolerance
            verbose [bool]: print information
        """
        super().__init__(
            mass_coeff=mass_coeff,
            diffusion_coeff=diffusion_coeff,
            dim=dim,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
        )
        if self.tol is None:
            self.tol = 1e-2

    def __call__(
        self, x0: np.ndarray, rhs: np.ndarray, h: Optional[float] = None
    ) -> np.ndarray:
        """
        Solve the problem.

        Args:
            x0 [np.ndarray]: initial guess
            rhs [np.ndarray]: right hand side
            h [Optional[float]]: grid spacing

        Returns:
            solution [np.ndarray]
        """
        assert x0.shape == rhs.shape

        # Compute the degrees of freedom
        _dof = np.prod(x0.shape)

        # Create the linear operator
        def _mv(x: np.ndarray) -> np.ndarray:
            x = np.reshape(x, x0.shape)
            return (
                self.mass_coeff * x
                - da.laplace(x, dim=self.dim, h=h, diffusion_coeff=self.diffusion_coeff)
            ).flatten()

        lhsoperator = LinearOperator((_dof, _dof), matvec=_mv)

        # Solve the problem using scipys conjugate gradient solver
        out_im = np.reshape(
            sps.linalg.cg(
                lhsoperator,
                rhs.flatten(),
                x0.flatten(),
                tol=self.tol,
                maxiter=self.maxiter,
            )[0],
            x0.shape[: self.dim],
        )

        return out_im
