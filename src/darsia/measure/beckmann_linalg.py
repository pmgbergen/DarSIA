"""Module containing all linear solver options to solve the mixed form of Beckmann's problem."""

from __future__ import annotations
from enum import StrEnum

import time
import warnings
from abc import abstractmethod
from enum import Enum
from typing import Optional
from warnings import warn

import numpy as np
import pyamg
import scipy.sparse as sps
from scipy.stats import hmean

import darsia


# Define BeckmannLinearSolverType
class BeckmannLinearSolverType(StrEnum):
    DIRECT = "direct"
    AMG = "amg"
    CG = "cg"
    KSP = "ksp"


class BeckmannLinearSolver:
    """Class providing linear solver options for Beckmann's problem.

    Args:
        grid (darsia.Grid): underlying grid

    """

    @abstractmethod
    def __call__(
        self,
        A: sps.csr_matrix,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict]:
        """Solve linear system Ax = b.

        Args:
            A (sps.csr_matrix): system matrix
            b (np.ndarray): right hand side
            x0 (Optional[np.ndarray]): initial guess

        Returns:
            tuple[np.ndarray, dict]: solution and info dictionary

        """
        pass


class BeckmannDirectSolver(BeckmannLinearSolver):
    """Direct solver for Beckmann's problem.

    Args:
        grid (darsia.Grid): underlying grid

    """

    def __init__(self, options: dict) -> None:
        self.linear_solver: Optional[sps.linalg.splu] = None
        """Initialize direct solver."""

    def setup(self, matrix: sps.csc_matrix) -> None:
        """Setup direct solver.

        Args:
            matrix (sps.csr_matrix): system matrix

        """
        self.linear_solver = sps.linalg.splu(matrix)

    def __call__(
        self,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """Solve linear system Ax = b using a direct solver.

        Args:
            rhs (np.ndarray): right hand side

        Returns:
            np.ndarray: solution

        """
        return self.linear_solver.solve(rhs)


class BeckmannAMGSolver(BeckmannLinearSolver):
    def __init__(self, amg_options: dict, linear_solver_options: dict) -> None:
        # Setup AMG options
        self.amg_options = {
            "strength": "symmetric",  # change the strength of connection
            "aggregate": "standard",  # use a standard aggregation method
            "smooth": ("jacobi"),  # prolongation smoother
            "presmoother": (
                "block_gauss_seidel",
                {"sweep": "symmetric", "iterations": 1},
            ),
            "postsmoother": (
                "block_gauss_seidel",
                {"sweep": "symmetric", "iterations": 1},
            ),
            "coarse_solver": "pinv2",  # pseudo inverse via SVD
            "max_coarse": 100,  # maximum number on a coarse level
        }
        """dict: options for the AMG solver"""

        # Allow to overwrite default options.
        self.amg_options.update(amg_options)

        # Setup linear solver options
        atol = linear_solver_options.get("atol", 1e-6)
        rtol = linear_solver_options.get("rtol", None)
        if not rtol:
            warn("rtol not used for AMG solver.")
        maxiter = linear_solver_options.get("maxiter", 100)
        self.residual_history: list[float] = []
        """list: history of residuals for the AMG solver"""
        self.solver_options = {
            "tol": atol,
            "maxiter": maxiter,
            "residuals": self.residual_history,
        }
        """dict: options for the iterative linear solver"""

        self.linear_solver: Optional[pyamg.amg_core.solve] = None
        """Initialize AMG solver."""

    def setup(self, matrix: sps.csc_matrix) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Implicit conversion of A to CSR")
            self.linear_solver = pyamg.smoothed_aggregation_solver(
                matrix, **self.amg_options
            )

    def __call__(
        self,
        rhs: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve linear system Ax = b using an AMG solver.

        Args:
            rhs (np.ndarray): right hand side
            x0 (Optional[np.ndarray]): initial guess

        Returns:
            np.ndarray: solution

        """
        if x0 is None:
            x0 = np.zeros_like(rhs)

        # Solve system
        solution = self.linear_solver.solve(rhs, x0=x0, **self.solver_options)

        return solution


class BeckmannCGSolver(BeckmannLinearSolver):
    def __init__(self, amg_options: dict, linear_solver_options: dict) -> None:
        # Setup AMG options for preconditioner
        self.amg_options = {
            "strength": "symmetric",  # change the strength of connection
            "aggregate": "standard",  # use a standard aggregation method
            "smooth": ("jacobi"),  # prolongation smoother
            "presmoother": (
                "block_gauss_seidel",
                {"sweep": "symmetric", "iterations": 1},
            ),
            "postsmoother": (
                "block_gauss_seidel",
                {"sweep": "symmetric", "iterations": 1},
            ),
            "coarse_solver": "pinv2",  # pseudo inverse via SVD
            "max_coarse": 100,  # maximum number on a coarse level
        }
        """dict: options for the AMG solver"""

        # Allow to overwrite default options.
        self.amg_options.update(amg_options)

        # Define solver options
        linear_solver_options = linear_solver_options.get("linear_solver_options", {})
        rtol = linear_solver_options.get("rtol", 1e-6)
        atol = linear_solver_options.get("atol", 0)
        maxiter = linear_solver_options.get("maxiter", 100)
        self.solver_options = {
            "rtol": rtol,
            "atol": atol,
            "maxiter": maxiter,
            # "M": amg, # Set in setup
        }
        """dict: options for the iterative linear solver"""

        self.linear_solver: Optional[darsia.linalg.CG] = None
        """Initialize CG solver with AMG preconditioner."""

    def setup(self, matrix: sps.csc_matrix) -> None:
        """Setup an CG solver with AMG preconditioner for the given matrix.

        Args:
            matrix (sps.csc_matrix): matrix

        Defines:
            pyamg.amg_core.solve: AMG solver
            dict: options for the AMG solver

        """
        # Define CG solver
        self.linear_solver = darsia.linalg.CG(matrix)

        # Define AMG preconditioner
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Implicit conversion of A to CSR")
            amg = pyamg.smoothed_aggregation_solver(
                matrix, **self.amg_options
            ).aspreconditioner(cycle="V")
        self.solver_options["M"] = amg

    def __call__(
        self,
        rhs: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve linear system Ax = b using a CG solver with AMG preconditioner.

        Args:
            rhs (np.ndarray): right hand side
            x0 (Optional[np.ndarray]): initial guess

        Returns:
            np.ndarray: solution

        """
        if x0 is None:
            x0 = np.zeros_like(rhs)

        # Solve system
        solution = self.linear_solver.solve(rhs, x0=x0, **self.solver_options)

        return solution


class BeckmannKSPSolver(BeckmannLinearSolver):
    def __init__(
        self,
        linear_solver_options: dict,
    ) -> None:
        # Define solver options
        rtol = linear_solver_options.get("rtol", 1e-6)
        atol = linear_solver_options.get("atol", 0)
        maxiter = linear_solver_options.get("maxiter", 100)
        approach = linear_solver_options.get("approach", "direct")

        if approach == "direct":
            self.solver_options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        else:
            prec = linear_solver_options.get("pc_type", "hypre")
            self.solver_options = {
                "ksp_type": approach,
                # "ksp_monitor_true_residual": None,
                "ksp_rtol": rtol,
                "ksp_atol": atol,
                "ksp_max_it": maxiter,
                "pc_type": prec,
            }

        # self.field_ises = None
        # """list: list of (field name, indices) tuples to define block structure"""

        # self.nullspace = None
        # """ list of nullspace vectors of the matrix"""

    def _setup(
        self,
        matrix: sps.csc_matrix,
    ) -> None:
        """Setup an KSP solver from PETSc for the given matrix.

        Args:
            matrix (sps.csc_matrix): matrix

        Defines:
            PETSc.ksp: KSP solver
            dict: options for the KSP solver
        """
        # Define CG solver
        self.linear_solver = darsia.linalg.KSP(
            matrix
            # , field_ises=self.field_ises, nullspace=self.nullspace
        )
        self.linear_solver.setup(self.solver_options)
        """dict: options for the iterative linear solver"""

    def setup(self, matrix: sps.csc_matrix) -> None:
        """Setup KSP solver for the given matrix.

        Args:
            matrix (sps.csc_matrix): system matrix

        """
        if not hasattr(self, "linear_solver"):
            self._setup(matrix)

        # if reuse_solver:
        #    self.linear_solver.setup(self.solver_options)

        ## Free memory if solver needs to be re-setup
        # if hasattr(self, "linear_solver"):
        #    if not reuse_solver:
        #        self.linear_solver.kill()
        #        self._setup(self.fully_reduced_matrix)
        # else:
        #    self._setup(self.fully_reduced_matrix)

    def __call__(
        self,
        rhs: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve linear system Ax = b using an AMG solver.

        Args:
            rhs (np.ndarray): right hand side
            x0 (Optional[np.ndarray]): initial guess

        Returns:
            np.ndarray: solution

        """
        if x0 is None:
            x0 = np.zeros_like(rhs)

        # Solve system
        solution = self.linear_solver.solve(rhs, x0=x0, **self.solver_options)

        return solution


class BeckmannKSPFieldSplitSolver(BeckmannKSPSolver):
    def __init__(
        self,
        linear_solver_options: dict,
        field_ises: list[tuple[str, np.ndarray]],
        nullspace: list[np.ndarray],
    ) -> None:
        # Define solver options
        rtol = linear_solver_options.get("rtol", 1e-6)
        atol = linear_solver_options.get("atol", 0)
        maxiter = linear_solver_options.get("maxiter", 100)
        approach = linear_solver_options.get("approach", "direct")

        if approach == "direct":
            self.solver_options = {
                "ksp_type": "preonly",  # do not apply Krylov iterations
                "pc_type": "lu",
                "pc_factor_shift_type": "inblocks",  # for the zero entries
                "pc_factor_mat_solver_type": "mumps",
            }
        else:
            prec = linear_solver_options.get("pc_type", "hypre")
            # Block preconditioning approach
            # the the 0 and 1 in fieldsplit_0 and fieldsplit_1 are the strings
            # passed to the field_ises
            self.solver_options = {
                "ksp_type": approach,
                "ksp_rtol": rtol,
                "ksp_atol": atol,
                "ksp_max_it": maxiter,
                "ksp_monitor": None,
                # "ksp_monitor_true_residual": None, #this is for debugging
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                # use a full factorization of the Schur complement
                # other options are "diag","lower","upper"
                "pc_fieldsplit_schur_precondition": "selfp",
                # selfp -> form an approximate Schur complement
                # using S=-B diag(A)^{-1} B^T
                # which is the exact Schur complement in our case
                # https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/
                "fieldsplit_flux_ksp_type": "preonly",
                "fieldsplit_flux_pc_type": "jacobi",
                # use the diagonal of the flux (it is the inverse)
                "fieldsplit_pressure": {
                    "ksp_type": "preonly",
                    "pc_type": prec,
                },
                # an example of the nested dictionary
            }

        self.linear_solver.setup(self.solver_options)
        """dict: options for the iterative linear solver"""

        self.field_ises = field_ises
        """list: list of (field name, indices) tuples to define block structure"""

        self.nullspace: list[np.ndarray] = nullspace
        """ list of nullspace vectors of the matrix"""
