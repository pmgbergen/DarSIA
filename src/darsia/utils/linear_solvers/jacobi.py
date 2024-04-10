"""Jacobi solver for linear systems. To be used either as a solver or as a smoother.

"""

from __future__ import annotations

from typing import Union

import numpy as np

import darsia as da


class Jacobi(da.Solver):
    """Jacobi solver for the problem:
    mass_coeff * x - diffusion_coeff * laplace(x) = rhs

    Can be used either as a solver or as a smoother.

    TODO: Add possibility to actually have a heterogeneous diffusion coefficient.
    As of now it is assumed to be constant, or multiplied after Laplacian.

    """

    def _neighbor_accumulation(self, im: np.ndarray) -> np.ndarray:
        """Accumulation of neighbor pixels.

        Accumulates for each entry, the entries of neighbors, regardless of dimension.
        This is a helper function for the Jacobi solver, that returns an array with the
        same shape as the input array, but where each input has the value of the
        accumulation of its neighbors.

        Args:
            im (np.ndarray): image to accumulate

        Returns:
            np.ndarray: averaged image

        """
        im_av: np.ndarray = np.zeros_like(im)
        for ax in range(im.ndim):
            # Use ghost copy from outside
            im_av[da.array_slice_argument(im, ax, 0, 1)] += da.array_slice(im, ax, 0, 1)
            # 'Left' and 'right' copies in the interior
            im_av[da.array_slice_argument(im, ax, 1, None)] += da.array_slice(
                im, ax, 0, -1
            )
            im_av[da.array_slice_argument(im, ax, 0, -1)] += da.array_slice(
                im, ax, 1, None
            )
            # Use ghost copy from outside
            im_av[da.array_slice_argument(im, ax, -1, None)] += da.array_slice(
                im, ax, -1, None
            )
        return im_av

    def _diag(self, h: float = 1) -> Union[float, np.ndarray]:
        """
        Compute diagonal of the stiffness matrix.

        The stiffness matrix is understood in finite difference sense, with the
        possibility to identify pixel sizes with phyiscal mesh sizes. This is a helper
        function for the main Jacobi solver.

        Args:
            h (float): mesh diameter

        Returns:
            Union[float, np.ndarray]: diagonal of the stiffness matrix

        """
        return self.mass_coeff + self.diffusion_coeff * 2 * self.dim / h**2

    def __call__(
        self,
        x0: np.ndarray,
        rhs: np.ndarray,
        h: float = 1.0,
    ) -> np.ndarray:
        """One iteration of a Jacobi solver for linear systems.

        Args:
            x0 (np.ndarray): initial guess
            rhs (np.ndarray): right hand side of the linear system
            h (float): mesh diameter

        Returns:
            np.ndarray: approximation to the linear system

        """
        x: np.ndarray = x0

        # Main idea - Iteration is performed in non-residual form:
        #   x_new = np.divide(
        #       rhs + self.diffusion_coeff * self._neighbor_accumulation(x) / h**2,
        #       diag
        #   )
        # To reduce computations, the expression is mildly optimized.

        # Precompute constant expressions
        if not hasattr(self, "const_diag"):
            self.const_diag = self._diag(h)
        if not hasattr(self, "const_diag_scaled"):
            self.const_diag_scaled = np.divide(
                self.const_diag, self.diffusion_coeff / h**2
            )
        rhs_scaled = np.divide(rhs, self.const_diag)

        # Split the tolerance based part to avoid unnecessary boolean evaluation
        if self.tol is None:
            for _ in range(self.maxiter):
                x = rhs_scaled + np.divide(
                    self._neighbor_accumulation(x), self.const_diag_scaled
                )
                if self.verbose:
                    print(f"Jacobi iteration {_} of {self.maxiter} completed.")

        else:
            for _ in range(self.maxiter):
                x_new = rhs_scaled + np.divide(
                    self._neighbor_accumulation(x), self.const_diag_scaled
                )
                err = np.linalg.norm(x_new - x) / np.linalg.norm(x0)
                if err < self.tol:
                    break
                x = x_new
                if self.verbose:
                    print(
                        f"Jacobi iteration {_} of {self.maxiter} completed with an "
                        f"increment of norm {err}."
                    )

        return x
