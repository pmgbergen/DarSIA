import abc
from typing import Optional, Union

import numpy as np


class Solver:

    """Abstract base class for solvers for the problem
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
    ) -> None:
        self.mass_coeff = mass_coeff
        self.diffusion_coeff = diffusion_coeff
        self.dim = dim
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose

    @abc.abstractmethod
    def __call__(self, x0: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        pass
