"""
Jacobi solver for linear systems. To be used either as a solver or as a smoother.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

import darsia as da


class Jacobi(da.Solver):
    """
    Jacobi solver for the problem: mass_coeff * x - diffusion_coeff * laplace(x) = rhs

    NOTE: Add possibility to actually have a heterogeneous diffusion coefficient. As of now it
    is assumed to be constant, or multiplied after Laplacian.
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

    def _average(self, im: np.ndarray) -> np.ndarray:
        """
        Averaging operator for the jacobi solver. Averages along all axes regardless of
        dimension. This is a helper function for the jacobi solver, that returns an
        array with the same shape as the input array, but where each input has the
        value of the average of its neighbors.

        input:
            im (np.ndarray): image to average

        output:
            np.ndarray: averaged image
        """
        im_av: np.ndarray = np.zeros_like(im)
        for ax in range(im.ndim):
            im_av += np.concatenate(
                (da.array_slice(im, ax, 0, 1), da.array_slice(im, ax, 0, -1)), axis=ax
            ) + np.concatenate(
                (da.array_slice(im, ax, 1, None), da.array_slice(im, ax, -1, None)),
                axis=ax,
            )
        return im_av

    def diag(self, h: float = 1) -> Union[float, np.ndarray]:
        """
        Compute diagonal of the Jacobi matrix with a given mesh diameter h

        args:
            h (float): mesh diameter

        Returns:
            Union[float, np.ndarray]: diagonal of the Jacobi matrix
        """
        return self.mass_coeff + self.diffusion_coeff * 2 * self.dim / h**2

    def __call__(
        self,
        x0: np.ndarray,
        rhs: np.ndarray,
        h: float = 1.0,
    ) -> np.ndarray:
        """
        Jacobi solver for linear systems. To be used either as a solver or as a smoother.

        input:
            x0 (np.ndarray): initial guess
            rhs (np.ndarray): right hand side of the linear system
            h (float): mesh diameter

        output:
            np.ndarray: solution to the linear system
        """
        x: np.ndarray = x0

        diag = self.diag(h=h)

        if self.verbose:
            # Split the tolerance based part to avoid unnecessary boolean evaluation
            if self.tol is None:
                for _ in range(self.maxiter):
                    x = (rhs + self.diffusion_coeff * self._average(x) / h**2) / diag
                    print(f"Jacobi iteration {_} of {self.maxiter} completed.")
            else:
                for _ in range(self.maxiter):
                    x_new = (
                        rhs + self.diffusion_coeff * self._average(x) / h**2
                    ) / diag
                    err = np.linalg.norm(x_new - x) / np.linalg.norm(x0)
                    if err < self.tol:
                        break
                    x = x_new
                    print(
                        f"Jacobi iteration {_} of {self.maxiter} completed with an "
                        f"increment of norm {err}."
                    )
        else:
            # Split the tolerance based part to avoid unnecessary boolean evaluation
            if self.tol is None:
                for _ in range(self.maxiter):
                    x = (rhs + self.diffusion_coeff * self._average(x) / h**2) / diag
            else:
                for _ in range(self.maxiter):
                    x_new = (
                        rhs + self.diffusion_coeff * self._average(x) / h**2
                    ) / diag
                    if np.linalg.norm(x_new - x) < self.tol:
                        break
                    x = x_new
        return x
