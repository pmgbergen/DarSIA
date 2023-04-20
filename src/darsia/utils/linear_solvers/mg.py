"""
Multigrid solver for linear systems arising from discretizations of the
minimization problem. Could be used as a preconditioner or a solver in itself.
"""


from typing import Optional, Union

import numpy as np

import darsia as da


class MG(da.Solver):
    """
    Muiltilvel solver for the problem: mass_coeff * x - diffusion_coeff * laplace(x) = rhs.
    For arrays with odd number of elements the last element is dropped in the refinement
    stages.
    """

    def __init__(
        self,
        mass_coeff: Union[float, np.ndarray],
        diffusion_coeff: Union[float, np.ndarray],
        depth: int = 2,
        smoother_iterations: int = 5,
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
            depth [int]: depth of the multigrid hierarchy
            smoother_iterations [int]: number of iterations for the smoother
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
        self.smoother = da.Jacobi(
            mass_coeff=mass_coeff,
            diffusion_coeff=diffusion_coeff,
            dim=dim,
            maxiter=smoother_iterations,
        )
        self.smoother_iterations = smoother_iterations
        self.depth = depth
        self.heterogeneous = False
        if isinstance(mass_coeff, np.ndarray) or isinstance(
            diffusion_coeff, np.ndarray
        ):
            self.heterogeneous = True

    def operator(self, x: np.ndarray, h: float) -> np.ndarray:
        """The solution operator for the problem

        Args:
            x [np.ndarray]: input
            h [float]: grid spacing

        Returns:
            output [np.ndarray]

        """

        return self.mass_coeff * x - self.diffusion_coeff * da.laplace(
            x, dim=self.dim, h=h
        )

    def restriction(self, x: np.ndarray) -> np.ndarray:
        """Restrict x, i.e., coarsen it by a factor 2. Even and odd indices are averaged.
        Last index is dropped if odd.

        Args:
            x [np.ndarray]: input

        Returns:
            output [np.ndarray]
        """

        for ax in range(self.dim):
            x = (
                np.take(
                    x,
                    np.arange(start=0, stop=x.shape[ax] - 1, step=2),
                    axis=ax,
                )
                + np.take(x, np.arange(start=1, stop=x.shape[ax], step=2), axis=ax)
            ) / 2

        return x

    def restrict_parameters(self) -> None:
        """
        In case of heterogeneous parameters the parameters are restricted
        by averaging.
        """
        if isinstance(self.mass_coeff, np.ndarray):
            self.mass_coeff = self.restriction(self.mass_coeff)
        if isinstance(self.diffusion_coeff, np.ndarray):
            self.diffusion_coeff = self.restriction(self.diffusion_coeff)
        self.smoother = da.Jacobi(
            mass_coeff=self.mass_coeff,
            diffusion_coeff=self.diffusion_coeff,
            dim=self.dim,
            maxiter=self.smoother_iterations,
        )

    def prolongation(self, x: np.ndarray) -> np.ndarray:
        """Prolongate x, i.e., refine it by a factor 2. All values are repeated twice.


        Args:
            x [np.ndarray]: input

        Returns:
            output [np.ndarray]
        """

        for ax in range(self.dim):
            x = np.repeat(x, 2, axis=ax)

        return x

    def prolongate_parameters(self, pad_tuple) -> None:
        """
        In case of heterogeneous parameters the parameters are prolongated
        by averaging.

        Args:
            pad_tuple [tuple]: tuple of tuples, each tuple contains the number of
                elements to be padded before and after the corresponding axis.

        """
        if isinstance(self.mass_coeff, np.ndarray):
            self.mass_coeff = self.prolongation(self.mass_coeff)
            self.mass_coeff = np.pad(self.mass_coeff, pad_tuple, "edge")
        if isinstance(self.diffusion_coeff, np.ndarray):
            self.diffusion_coeff = self.prolongation(self.diffusion_coeff)
            self.diffusion_coeff = np.pad(self.diffusion_coeff, pad_tuple, "edge")
        self.smoother = da.Jacobi(
            mass_coeff=self.mass_coeff,
            diffusion_coeff=self.diffusion_coeff,
            dim=self.dim,
            maxiter=self.smoother_iterations,
        )

    def base_V_Cycle(
        self, x0: np.ndarray, rhs: np.ndarray, depth: int, h: float = 1
    ) -> np.ndarray:
        """Base V-Cycle (recursive function)

        Args:
            x0 [np.ndarray]: initial guess
            rhs [np.ndarray]: right hand side
            depth [int]: depth of the V-Cycle
            h [float]: grid spacing

        Returns:
            x [np.ndarray]: solution
        """
        # Presmooth
        x = self.smoother(x0, rhs, h=h)

        # Compute residual
        r = rhs - self.operator(x, h=h)

        # Restrict residual (and parameters in case of heterogeneities)
        r = self.restriction(r)
        if self.heterogeneous:
            self.restrict_parameters()

        # Solve/smooth coarse problem or further V-cycle
        if depth == 0:
            eps = self.smoother(x0=np.zeros_like(r), rhs=r, h=2 * h)
        else:
            eps = self.base_V_Cycle(
                x0=np.zeros_like(r), rhs=r, depth=depth - 1, h=2 * h
            )

        # Prolongate correction
        eps = self.prolongation(eps)

        # Pad correction if necessary (to account for odd number of grid points)
        pad_tuple = tuple((0, x.shape[i] - eps.shape[i]) for i in range(self.dim))
        if self.heterogeneous:
            self.prolongate_parameters(pad_tuple)
        eps = np.lib.pad(
            eps,
            pad_tuple,
            "edge",
        )
        x += eps

        # Postsmooth
        x = self.smoother(x0=x, rhs=rhs, h=h)
        return x

    def __call__(self, x0: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """
        V-Cycle multigrid solver for linear systems arising from discretizations
        of the minimization problem. Could be used as a preconditioner or a solver in itself.

        Args:
            x0 [np.ndarray]: initial guess
            rhs [np.ndarray]: right hand side

        Returns:
            x [np.ndarray]: solution
        """
        x = x0
        if self.tol is None:
            for i in range(self.maxiter):
                x = self.base_V_Cycle(x0=x, rhs=rhs, depth=self.depth)
        else:
            for i in range(self.maxiter):
                x_new = self.base_V_Cycle(x0=x, rhs=rhs, depth=self.depth)
                nrm = np.linalg.norm(x_new - x) / np.linalg.norm(x0)
                if self.verbose:
                    print(f"at iteration {i} the residual norm is {nrm}")
                if nrm < self.tol:
                    break
                x = x_new
        return x
