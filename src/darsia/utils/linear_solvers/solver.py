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
        dim: int = 2,
        maxiter: int = 1,
        tol: Optional[float] = None,
        mass_coeff: Optional[Union[float, np.ndarray]] = None,
        diffusion_coeff: Optional[Union[float, np.ndarray]] = None,
        verbose=False,
        **kwargs,
    ) -> None:
        self.dim = dim
        """Spatial dimension of the problem."""
        self.maxiter = maxiter
        """Maximum number of iterations."""
        self.tol = tol
        """Tolerance for convergence."""
        self.mass_coeff = mass_coeff
        """Mass coefficient."""
        self.diffusion_coeff = diffusion_coeff
        """Diffusion coefficient."""
        self.verbose = verbose
        """Verbosity."""

    def update_params(
        self,
        dim: Optional[int] = None,
        mass_coeff: Optional[Union[float, np.ndarray]] = None,
        diffusion_coeff: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        """Update parameters of the solver.

        Args:
            dim (int, optional): spatial dimension of the problem
            mass_coeff (float or array, optional): mass coefficient
            diffusion_coeff (float or array, optional): diffusion coefficient

        """
        self.dim = dim if dim is not None else self.dim
        self.mass_coeff = mass_coeff if mass_coeff is not None else self.mass_coeff
        self.diffusion_coeff = (
            diffusion_coeff if diffusion_coeff is not None else self.diffusion_coeff
        )

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
