"""Module containing abstract base class for solvers for the problem:
mass_coeff * x - diffusion_coeff * laplace(x) = rhs

"""

import abc
from typing import Optional, Union

import numpy as np


class Solver:
    """Abstract base class for solvers for the problem:
    mass_coeff * x - diffusion_coeff * laplace(x) = rhs

    """

    def __init__(
        self,
        mass_coeff: Union[float, np.ndarray],
        diffusion_coeff: Union[float, np.ndarray],
        dim: int = 2,
        maxiter: int = 100,
        tol: Optional[float] = None,
        verbose=False,
        **kwargs,
    ) -> None:
        self.mass_coeff = mass_coeff
        """Mass coefficient."""
        self.diffusion_coeff = diffusion_coeff
        """Diffusion coefficient."""
        self.dim = dim
        """Spatial dimension of the problem."""
        self.maxiter = maxiter
        """Maximum number of iterations."""
        self.tol = tol
        """Tolerance for convergence."""
        self.verbose = verbose
        """Verbosity."""

    @abc.abstractmethod
    def __call__(self, x0: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Main method of the solver - run the solver.

        Args:
            x0 (np.ndarray): initial guess
            rhs (np.ndarray): right hand side

        Returns:
            np.ndarray: solution

        """
        pass
