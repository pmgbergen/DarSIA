"""Jacobi solver for linear systems. To be used either as a solver or as a smoother.

"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

import darsia as da


class Jacobi(da.Solver):
    """Jacobi solver for the problem:
    mass_coeff * x - diffusion_coeff * laplace(x) = rhs

    Can be used either as a solver or as a smoother.

    TODO: Add possibility to actually have a heterogeneous diffusion coefficient.
    As of now it is assumed to be constant, or multiplied after Laplacian.

    """

    def _neighbor_accumulation(
        self, im: np.ndarray, h: Optional[Union[float, list]] = None
    ) -> np.ndarray:
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
        if h is None:
            for ax in range(self.dim):
                im_av += np.concatenate(
                    (da.array_slice(im, ax, 0, 1), da.array_slice(im, ax, 0, -1)),
                    axis=ax,
                ) + np.concatenate(
                    (da.array_slice(im, ax, 1, None), da.array_slice(im, ax, -1, None)),
                    axis=ax,
                )
        else:
            if isinstance(h, float):
                h = [h] * self.dim
            for ax in range(self.dim):
                im_av += (
                    np.concatenate(
                        (da.array_slice(im, ax, 0, 1), da.array_slice(im, ax, 0, -1)),
                        axis=ax,
                    )
                    + np.concatenate(
                        (
                            da.array_slice(im, ax, 1, None),
                            da.array_slice(im, ax, -1, None),
                        ),
                        axis=ax,
                    )
                ) / h[ax] ** 2
        return im_av

    def _diag(self, h: Optional[Union[float, list]] = None) -> Union[float, np.ndarray]:
        """
        Compute diagonal of the stiffness matrix.

        The stiffness matrix is understood in finite difference sense, with the
        possibility to identify pixel sizes with phyiscal voxel sizes. This is a helper
        function for the main Jacobi solver.

        Args:
            h (float): voxel size

        Returns:
            Union[float, np.ndarray]: diagonal of the stiffness matrix

        """
        if h is None:
            voxel_weights = self.dim
        else:
            if isinstance(h, float):
                voxel_weights = self.dim / (h**2)
            else:
                assert len(h) == self.dim, "h must be a float or a list of length dim"
                voxel_weights = sum([1 / h[i] ** 2 for i in range(self.dim)])

        return self.mass_coeff + 2 * self.diffusion_coeff * voxel_weights

    def __call__(
        self,
        x0: np.ndarray,
        rhs: np.ndarray,
        h: Optional[Union[float, list]] = None,
    ) -> np.ndarray:
        """One iteration of a Jacobi solver for linear systems.

        Args:
            x0 (np.ndarray): initial guess
            rhs (np.ndarray): right hand side of the linear system
            h (optional[float, list]): voxel size

        Returns:
            np.ndarray: solution to the linear system

        """
        x: np.ndarray = x0

        diag = self._diag(h)

        # Split the tolerance based part to avoid unnecessary boolean evaluation
        if self.tol is None:
            for _ in range(self.maxiter):
                x = (
                    rhs + self.diffusion_coeff * self._neighbor_accumulation(x, h)
                ) / diag
                if self.verbose:
                    print(f"Jacobi iteration {_} of {self.maxiter} completed.")
        else:
            for _ in range(self.maxiter):
                x_new = (
                    rhs + self.diffusion_coeff * self._neighbor_accumulation(x, h)
                ) / diag
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
