"""Wasserstein distance computed using variational methods."""

from __future__ import annotations

import time
import tracemalloc
import warnings
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import numpy as np
import pyamg
import scipy.sparse as sps
from scipy.stats import hmean

import darsia

# General TODO list
# - improve assembling of operators through partial assembling
# - allow to reuse setup.
# - bregman_update responding to flat_scaling approaching 1.
# - stopping criteria


class L1Mode(Enum):
    """Mode for computing the l1 dissipation."""

    RAVIART_THOMAS = "raviart_thomas"
    CONSTANT_SUBCELL_PROJECTION = "constant_subcell_projection"
    CONSTANT_CELL_PROJECTION = "constant_cell_projection"


class MobilityMode(Enum):
    """Mode for computing the mobility."""

    CELL_BASED = "cell_based"
    CELL_BASED_ARITHMETIC = "cell_based_arithmetic"
    CELL_BASED_HARMONIC = "cell_based_harmonic"
    SUBCELL_BASED = "subcell_based"
    FACE_BASED = "face_based"


class VariationalWassersteinDistance(darsia.EMD):
    """Base class for setting up the variational Wasserstein distance.

    The variational Wasserstein distance is defined as the solution to the following
    optimization problem (also called the Beckman problem):

        inf ||u||_{L^1} s.t. div u = m_2 - m_1, u in H(div).

    u is the flux, m_1 and m_2 are the mass distributions which are transported by u
    from m_1 to m_2. The problem is solved approximately, eploying an iterative
    tpfa-type finite volume method. A close connection to the lowest Raviart-Thomas
    mixed finite element method is exploited.

    There are two main solution strategies implemented in sepcialized classes:
    - Finite Volume Quasi-Newton's method (:class:`WassersteinDistanceNewton`)
    - Finite Volume Split Bregman method (:class:`WassersteinDistanceBregman`)

    """

    def __init__(
        self,
        grid: darsia.Grid,
        weight: Optional[darsia.Image] = None,
        options: dict = {},
    ) -> None:
        """Initialization of the variational Wasserstein distance.

        Args:

            grid (darsia.Grid): tensor grid associated with the images
            options (dict): options for the nonlinear and linear solver. The following
                options are supported:
                - l1_mode (darsia.L1Mode): mode for computing the l1 dissipation. Defaults to
                    "RAVIART_THOMAS". Supported modes are:
                    - "RAVIART_THOMAS": Apply exact integration of RT0 extensions into
                        cells. Underlying functional for mixed finite element method
                        (MFEM).
                    - "CONSTANT_SUBCELL_PROJECTION": Apply subcell_based projection onto
                        constant vectors and sum up. Equivalent to a mixed finite volume
                        method (FV).
                    - "CONSTANT_CELL_PROJECTION": Apply cell-based L2 projection onto
                        constant vectors and sum up. Simpler calculation than
                        subcell-projection, but not directly connected to any
                        discretization.
                - mobility_mode (MobilityMode): mode for computing the mobility. Defaults to
                    MobilityMode.CELL_BASED. Supported modes are:
                    - CELL_BASED: Cell-based mode determines the norm of the fluxes on
                        the faces via averaging of neighboring cells.
                    - CELL_BASED_ARITHMETIC: Cell-based mode determines the norm of
                        the fluxes on the faces via arithmetic averaging of neighboring
                        cells.
                    - CELL_BASED_HARMONIC: Cell-based mode determines the norm of the
                        fluxes on the faces via harmonic averaging of neighboring cells.
                    - SUBCELL_BASED: Subcell-based mode determines the norm of the
                        fluxes on the faces via averaging of neighboring subcells.
                    - FACE_BASED: Face-based mode determines the norm of the fluxes on
                        the faces via direct computation on the faces.
                - num_iter (int): maximum number of iterations. Defaults to 100.
                - tol_residual (float): tolerance for the residual. Defaults to
                    np.finfo(float).max.
                - tol_increment (float): tolerance for the increment. Defaults to
                    np.finfo(float).max.
                - tol_distance (float): tolerance for the distance. Defaults to
                    np.finfo(float).max.
                - L (float): regularization parameter for the Newton and Bregman method.
                    Represents an approximate flux norm (scalar or vector). Defaults to
                    1.0.
                - linear_solver (str): type of linear solver. Defaults to "direct".
                    Supported solvers are:
                    - "direct": direct solver
                    - "amg": algebraic multigrid solver
                    - "cg": conjugate gradient solver preconditioned with AMG
                    - "ksp": PETSc KSP solver
                - formulation (str): formulation of the linear system. Defaults to
                    "pressure". Supported formulations are:
                    - "full": full system
                    - "flux_reduced": reduced system with fluxes eliminated
                    - "pressure": reduced system with fluxes and lagrange multiplier
                        eliminated
                - linear_solver_options (dict): options for the linear solver. Defaults
                    to {}.
                - amg_options (dict): options for the AMG solver. Defaults to {}.
                - aa_depth (int): depth of the Anderson acceleration. Defaults to 0.
                - aa_restart (int): restart of the Anderson acceleration. Defaults to
                    None.
                - regularization (float): regularization parameter for avoiding division
                    by zero. Defaults to np.finfo(float).eps.
                - lumping (bool): lump the mass matrix. Defaults to True.

        """
        # Cache geometrical infos
        self.grid = grid
        """darsia.Grid: grid"""

        self.voxel_size = grid.voxel_size
        """np.ndarray: voxel size"""

        # Cache solver options
        self.options = options
        """dict: options for the solver"""

        self.regularization = self.options.get("regularization", np.finfo(float).eps)
        """float: regularization parameter"""

        self.verbose = self.options.get("verbose", False)
        """bool: verbosity"""

        self.l1_mode: L1Mode = self.options.get("l1_mode", L1Mode.RAVIART_THOMAS)
        """str: mode for computing the l1 dissipation"""

        self.mobility_mode: MobilityMode = self.options.get(
            "mobility_mode", MobilityMode.CELL_BASED
        )
        """str: mode for computing the mobility"""

        self.weight = weight
        """Weight defined on cells"""

        # Setup of method
        self._setup_dof_management()
        self._setup_face_weights()
        self._setup_discretization()
        self._setup_linear_solver()
        self._setup_acceleration()

    def _setup_dof_management(self) -> None:
        """Setup of Raviart-Thomas-type DOF management.

        The following degrees of freedom are considered (also in this order):
        - flat fluxes (normal fluxes on the faces)
        - flat pressures (pressures on the cells)
        - lagrange multiplier (scalar variable) - Idea: Fix the pressure in the
        center of the domain to zero via a constraint and a Lagrange multiplier.

        """
        # ! ---- Number of dofs ----
        num_flux_dofs = self.grid.num_faces
        num_pressure_dofs = self.grid.num_cells
        num_lagrange_multiplier_dofs = 1
        num_dofs = num_flux_dofs + num_pressure_dofs + num_lagrange_multiplier_dofs

        # ! ---- Indices in global system ----
        self.flux_indices = np.arange(num_flux_dofs, dtype=np.int32)
        """np.ndarray: indices of the fluxes"""

        self.pressure_indices = np.arange(
            num_flux_dofs, num_flux_dofs + num_pressure_dofs, dtype=np.int32
        )
        """np.ndarray: indices of the pressures"""

        self.lagrange_multiplier_indices = np.array(
            [num_flux_dofs + num_pressure_dofs], dtype=np.int32
        )
        """np.ndarray: indices of the lagrange multiplier"""

        # ! ---- Fast access to components through slices ----
        self.flux_slice = slice(0, num_flux_dofs)
        """slice: slice for the fluxes"""

        self.pressure_slice = slice(num_flux_dofs, num_flux_dofs + num_pressure_dofs)
        """slice: slice for the pressures"""

        self.lagrange_multiplier_slice = slice(
            num_flux_dofs + num_pressure_dofs,
            num_flux_dofs + num_pressure_dofs + num_lagrange_multiplier_dofs,
        )
        """slice: slice for the lagrange multiplier"""

        self.reduced_system_slice = slice(num_flux_dofs, None)
        """slice: slice for the reduced system (pressures and lagrange multiplier)"""

        # Embedding operators
        self.flux_embedding = sps.csc_matrix(
            (
                np.ones(num_flux_dofs, dtype=float),
                (self.flux_indices, self.flux_indices),
            ),
            shape=(num_dofs, num_flux_dofs),
        )
        """sps.csc_matrix: embedding operator for fluxes"""

    def _setup_face_weights(self) -> None:
        """Convert cell weights to face weights by harmonic averaging."""

        if self.weight is None:
            self.cell_weights = np.ones(self.grid.shape, dtype=float)
            """np.ndarray: cell weights"""
            self.face_weights = np.ones(self.grid.num_faces, dtype=float)
            """np.ndarray: face weights"""
        else:
            self.cell_weights = self.weight.img
            self.face_weights = self._harmonic_average(self.cell_weights)

    def _setup_discretization(self) -> None:
        """Setup of fixed discretization operators."""

        # ! ---- Constraint for the pressure correpsonding to Lagrange multiplier ----

        center_cell = np.array(self.grid.shape) // 2
        self.constrained_cell_flat_index = np.ravel_multi_index(
            center_cell, self.grid.shape
        )
        """int: flat index of the cell where the pressure is constrained to zero"""

        num_pressure_dofs = self.grid.num_cells
        self.pressure_constraint = sps.csc_matrix(
            (
                np.ones(1, dtype=float),
                (np.zeros(1, dtype=int), np.array([self.constrained_cell_flat_index])),
            ),
            shape=(1, num_pressure_dofs),
            dtype=float,
        )
        """sps.csc_matrix: effective constraint for the pressure"""

        # ! ---- Discretization operators ----

        self.div = darsia.FVDivergence(self.grid).mat
        """sps.csc_matrix: divergence operator: flat fluxes -> flat pressures"""

        self.mass_matrix_cells = darsia.FVMass(self.grid).mat
        """sps.csc_matrix: mass matrix on cells: flat pressures -> flat pressures"""

        lumping = self.options.get("lumping", True)
        self.weighted_mass_matrix_faces_init = sps.diags(
            self.face_weights, format="csc"
        ) @ (darsia.FVMass(self.grid, "faces", lumping).mat)
        """sps.csc_matrix: weighted mass matrix on faces: flat fluxes -> flat fluxes"""
        self.mass_matrix_faces = darsia.FVMass(self.grid, "faces", lumping).mat
        """sps.csc_matrix: mass matrix on faces: flat fluxes -> flat fluxes"""

        L_init = self.options.get("L_init", 1.0)
        self.darcy_init = sps.bmat(
            [
                [L_init * self.weighted_mass_matrix_faces_init, -self.div.T, None],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ],
            format="csc",
        )
        """sps.csc_matrix: initial Darcy operator"""

        self.broken_darcy = sps.bmat(
            [
                [None, -self.div.T, None],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ],
            format="csc",
        )
        """sps.csc_matrix: linear part of the Darcy operator with pressure constraint"""

    def _setup_face_reconstruction(self) -> None:
        """Setup of face reconstruction via RT0 basis functions and arithmetic avg.

        NOTE: Do not make part of self._setup_discretization() as not always required.

        """
        self.face_reconstruction = darsia.FVFullFaceReconstruction(self.grid)
        """sps.csc_matrix: full face reconstruction: flat fluxes -> vector fluxes"""

    def _setup_linear_solver(self) -> None:
        self.linear_solver_type = self.options.get("linear_solver", "direct")
        """str: type of linear solver"""

        self.formulation: str = self.options.get("formulation", "pressure")
        """str: formulation type"""

        # Safety checks
        assert self.linear_solver_type in [
            "direct",
            "amg",
            "cg",
            "ksp",
        ], f"Linear solver {self.linear_solver_type} not supported."
        assert self.formulation in [
            "full",
            "flux_reduced",
            "pressure",
        ], f"Formulation {self.formulation} not supported."

        # if self.linear_solver_type == "ksp":
        #     if self.formulation == "flux_reduced":
        #         raise ValueError(
        #             "KSP solver only supports for full and pressure formulation."
        #         )

        # Setup inrastructure for Schur complement reduction
        if self.formulation == "flux_reduced":
            self.setup_eliminate_flux()

        elif self.formulation == "pressure":
            self.setup_eliminate_flux()
            self.setup_eliminate_lagrange_multiplier()

    def setup_direct(self) -> None:
        """Setup the infrastructure for direct solvers."""

        self.solver_options = {}
        """dict: options for the direct solver"""

    def setup_direct_solver(self, matrix: sps.csc_matrix) -> sps.linalg.splu:
        """Setup a direct solver for the given matrix.

        Args:
            matrix (sps.csc_matrix): matrix

        Defines:
            sps.linalg.splu: direct solver
            dict: (empty) solver options

        """
        self.linear_solver = sps.linalg.splu(matrix)
        self.solver_options = {}

    def setup_amg_options(self) -> None:
        """Setup the infrastructure for multilevel solvers.

        Basic default setup based on jacobi and block Gauss-Seidel smoothers.
        User-defined options can be passed via the options dictionary, using the key
        "amg_options". The options follow the pyamg interface.

        """
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

        # Allow to overwrite default options - use pyamg interface.
        user_defined_amg_options = self.options.get("amg_options", {})
        self.amg_options.update(user_defined_amg_options)

    def setup_amg_solver(self, matrix: sps.csc_matrix) -> None:
        """Setup an AMG solver for the given matrix.

        Args:
            matrix (sps.csc_matrix): matrix

        Defines:
            pyamg.amg_core.solve: AMG solver
            dict: options for the AMG solver

        """
        # Define AMG solver
        self.setup_amg_options()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Implicit conversion of A to CSR")
            self.linear_solver = pyamg.smoothed_aggregation_solver(
                matrix, **self.amg_options
            )

        # Define solver options
        linear_solver_options = self.options.get("linear_solver_options", {})
        atol = linear_solver_options.get("atol", 1e-6)
        rtol = linear_solver_options.get("rtol", None)
        if not rtol:
            warn("rtol not used for AMG solver.")
        maxiter = linear_solver_options.get("maxiter", 100)
        self.amg_residual_history = []
        """list: history of residuals for the AMG solver"""
        self.solver_options = {
            "tol": atol,
            "maxiter": maxiter,
            "residuals": self.amg_residual_history,
        }
        """dict: options for the iterative linear solver"""

    def setup_cg_solver(self, matrix: sps.csc_matrix) -> None:
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
        self.setup_amg_options()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Implicit conversion of A to CSR")
            amg = pyamg.smoothed_aggregation_solver(
                matrix, **self.amg_options
            ).aspreconditioner(cycle="V")

        # Define solver options
        linear_solver_options = self.options.get("linear_solver_options", {})
        rtol = linear_solver_options.get("rtol", 1e-6)
        atol = linear_solver_options.get("atol", 0)
        maxiter = linear_solver_options.get("maxiter", 100)
        self.solver_options = {
            "rtol": rtol,
            "atol": atol,
            "maxiter": maxiter,
            "M": amg,
        }
        """dict: options for the iterative linear solver"""

    def setup_ksp_solver(
        self,
        matrix: sps.csc_matrix,
        field_ises: Optional[list[tuple[str, np.ndarray]]] = None,
        nullspace: Optional[list[np.ndarray]] = None,
    ) -> None:
        """Setup an KSP solver from PETSc for the given matrix.

        Args:
            matrix (sps.csc_matrix): matrix
            nullspace (list[np.ndarray]): list of nullspace vectors of the matrix

        Defines:
            PETSc.ksp: KSP solver
            dict: options for the KSP solver
        """
        # Define CG solver
        self.linear_solver = darsia.linalg.KSP(
            matrix,
            field_ises=field_ises,
            nullspace=nullspace,
        )

        # Define solver options
        linear_solver_options = self.options.get("linear_solver_options", {})
        rtol = linear_solver_options.get("rtol", 1e-6)
        atol = linear_solver_options.get("atol", 0)
        maxiter = linear_solver_options.get("maxiter", 100)
        approach = linear_solver_options.get("approach", "direct")

        if field_ises is None:
            if approach == "direct":
                self.solver_options = {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                }
            else:
                self.solver_options = {
                    "ksp_type": approach,
                    # "ksp_monitor_true_residual": None,
                    "ksp_rtol": rtol,
                    "ksp_atol": atol,
                    "ksp_max_it": maxiter,
                    "pc_type": "hypre",
                }
                if self.grid.dim == 3:
                    self.solver_options.update(
                        {
                            # tuning parameters for the multigrid
                            # https://mooseframework.inl.gov/releases/moose/2021-09-15/application_development/hypre.html
                            "pc_hypre_type": "boomeramg",
                            "pc_hypre_boomeramg_strong_threshold": 0.7,
                            "pc_hypre_boomeramg_max_iter": 1,
                            "pc_hypre_boomeramg_agg_nl": 2,
                            "pc_hypre_boomeramg_interp_type": "ext+i",  # "classic" or "ext+i"
                        }
                    )
                # Include other PETSc options passed from the user
                other_petsc_options = linear_solver_options.get("petsc_options", {})
                self.solver_options.update(other_petsc_options)
        else:
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

    def setup_eliminate_flux(self) -> None:
        """Setup the infrastructure for reduced systems through Gauss elimination.

        Provide internal data structures for the reduced system, still formulated in
        terms of pressures and Lagrange multiplier. Merely the flux is eliminated using
        a Schur complement approach.

        """
        #   ---- Preliminaries ----

        # Fixate some jacobian for copying the data structure
        jacobian = self.darcy_init.copy()

        # ! ---- Eliminate flux block ----

        # Build Schur complement wrt. flux-flux block
        J_inv = sps.diags(1.0 / jacobian.diagonal()[self.flux_slice])
        D = jacobian[self.reduced_system_slice, self.flux_slice].copy()
        schur_complement = D.dot(J_inv.dot(D.T))

        # Cache divergence matrix together with Lagrange multiplier
        self.D = D.copy()
        """sps.csc_matrix: divergence + lagrange multiplier matrix"""

        self.DT = self.D.T.copy()
        """sps.csc_matrix: transposed divergence + lagrange multiplier  matrix"""

        # Cache (constant) jacobian subblock
        self.jacobian_subblock = jacobian[
            self.reduced_system_slice, self.reduced_system_slice
        ].copy()
        """sps.csc_matrix: constant jacobian subblock of the reduced system"""

        # Add Schur complement - use this to identify sparsity structure
        # Cache the reduced jacobian
        self.reduced_jacobian = self.jacobian_subblock + schur_complement
        """sps.csc_matrix: reduced jacobian incl. Schur complement"""

    def setup_eliminate_lagrange_multiplier(self) -> None:
        """Additional setup of infrastructure for fully reduced systems.

        Here, the Lagrange multiplier is eliminated through Gauss elimination. It is
        implicitly assumed, that the flux block is eliminated already.

        """
        # Find row entries to be removed
        rm_row_entries = np.arange(
            self.reduced_jacobian.indptr[self.constrained_cell_flat_index],
            self.reduced_jacobian.indptr[self.constrained_cell_flat_index + 1],
        )

        # Find column entries to be removed
        rm_col_entries = np.where(
            self.reduced_jacobian.indices == self.constrained_cell_flat_index
        )[0]

        # Collect all entries to be removes
        rm_indices = np.unique(
            np.concatenate((rm_row_entries, rm_col_entries)).astype(int)
        )
        # Cache for later use in eliminate_lagrange_multiplier
        self.rm_indices = rm_indices
        """np.ndarray: indices to be removed in the reduced system"""

        # Identify rows to be reduced
        rm_rows = [
            np.max(np.where(self.reduced_jacobian.indptr <= index)[0])
            for index in rm_indices
        ]

        # Reduce data - simply remove
        fully_reduced_jacobian_data = np.delete(self.reduced_jacobian.data, rm_indices)

        # Reduce indices - remove and shift
        fully_reduced_jacobian_indices = np.delete(
            self.reduced_jacobian.indices, rm_indices
        )
        fully_reduced_jacobian_indices[
            fully_reduced_jacobian_indices > self.constrained_cell_flat_index
        ] -= 1

        # Reduce indptr - shift and remove
        # NOTE: As only a few entries should be removed, this is not too expensive
        # and a for loop is used
        fully_reduced_jacobian_indptr = self.reduced_jacobian.indptr.copy()
        for row in rm_rows:
            fully_reduced_jacobian_indptr[row + 1 :] -= 1
        fully_reduced_jacobian_indptr = np.unique(fully_reduced_jacobian_indptr)

        # Make sure two rows are removed and deduce shape of reduced jacobian
        assert (
            len(fully_reduced_jacobian_indptr) == len(self.reduced_jacobian.indptr) - 2
        ), "Two rows should be removed."
        fully_reduced_jacobian_shape = (
            len(fully_reduced_jacobian_indptr) - 1,
            len(fully_reduced_jacobian_indptr) - 1,
        )

        # Cache the fully reduced jacobian
        self.fully_reduced_jacobian = sps.csc_matrix(
            (
                fully_reduced_jacobian_data,
                fully_reduced_jacobian_indices,
                fully_reduced_jacobian_indptr,
            ),
            shape=fully_reduced_jacobian_shape,
        )
        """sps.csc_matrix: fully reduced jacobian"""

        # Cache the indices and indptr
        self.fully_reduced_jacobian_indices = fully_reduced_jacobian_indices.copy()
        """np.ndarray: indices of the fully reduced jacobian"""

        self.fully_reduced_jacobian_indptr = fully_reduced_jacobian_indptr.copy()
        """np.ndarray: indptr of the fully reduced jacobian"""

        self.fully_reduced_jacobian_shape = fully_reduced_jacobian_shape
        """tuple: shape of the fully reduced jacobian"""

        # ! ---- Identify inclusions (index arrays) ----

        # Define reduced system indices wrt full system
        reduced_system_indices = np.concatenate(
            [self.pressure_indices, self.lagrange_multiplier_indices]
        )

        # Define fully reduced system indices wrt reduced system - need to remove cell
        # (and implicitly lagrange multiplier)
        self.fully_reduced_system_indices = np.delete(
            np.arange(self.grid.num_cells), self.constrained_cell_flat_index
        )
        """np.ndarray: indices of the fully reduced system in terms of reduced system"""

        # Define fully reduced system indices wrt full system
        self.fully_reduced_system_indices_full = reduced_system_indices[
            self.fully_reduced_system_indices
        ]
        """np.ndarray: indices of the fully reduced system in terms of full system"""

    def _setup_acceleration(self) -> None:
        """Setup of acceleration methods."""

        # ! ---- Acceleration ----
        aa_depth = self.options.get("aa_depth", 0)
        aa_restart = self.options.get("aa_restart", None)
        self.anderson = (
            darsia.AndersonAcceleration(
                dimension=None, depth=aa_depth, restart=aa_restart
            )
            if aa_depth > 0
            else None
        )
        """darsia.AndersonAcceleration: Anderson acceleration"""

    # ! ---- Effective quantities ----

    def cell_weighted_flux(self, cell_flux: np.ndarray) -> np.ndarray:
        """Compute the cell-weighted flux.

        Args:
            cell_flux (np.ndarray): cell fluxes

        Returns:
            np.ndarray: cell-weighted flux

        """
        if self.weight is None:
            return cell_flux
        elif len(self.cell_weights.shape) == self.grid.dim:
            return cell_flux * self.cell_weights[..., np.newaxis]

        elif (
            len(self.cell_weights.shape) == self.grid.dim + 1
            and self.cell_weights.shape[-1] == 1
        ):
            raise NotImplementedError("Need to reduce the dimension")
            # Try: return cell_flux * self.cell_weights

        elif (
            len(self.cell_weights.shape) == self.grid.dim + 1
            and self.cell_weights.shape[-1] == self.grid.dim
        ):
            return cell_flux * self.cell_weights

        elif len(
            self.cell_weights.shape
        ) == self.grid.dim + 2 and self.cell_weights.shape[-2:] == (
            self.grid.dim,
            self.grid.dim,
        ):
            raise NotImplementedError("Need to apply matrix vector product.")

        else:
            raise NotImplementedError("Dimension not supported.")

    def transport_density(
        self, flat_flux: np.ndarray, weighted: bool = True, flatten: bool = True
    ) -> np.ndarray:
        """Compute the transport density from the solution.

        Args:
            flat_flux (np.ndarray): face fluxes
            weighted (bool): apply weighting. Defaults to True.
            flatten (bool): flatten the result. Defaults to True.

        Returns:
            np.ndarray: transport density, flattened if requested

        Notes:
        Type of integration depends on the selected mode, see self.l1_mode. Supported
        modes are:
        - 'raviart_thomas': Apply exact integration of RT0 extensions into cells.
            Underlying functional for mixed finite element method (MFEM).
        - 'constant_subcell_projection': Apply subcell_based projection onto constant
            vectors and sum up. Equivalent to a mixed finite volume method (FV).
        - 'constant_cell_projection': Apply cell-based L2 projection onto constant
            vectors and sum up. Simpler calculation than subcell-projection, but not
            directly connected to any discretization.

        """
        # The different modes merely differ in the integration rule.

        if self.l1_mode == L1Mode.RAVIART_THOMAS:
            # Apply numerical integration of RT0 extensions into cells.
            # Underlying functional for mixed finite element method (MFEM).
            quad_pts, quad_weights = darsia.quadrature.gauss_reference_cell(
                self.grid.dim, "max"
            )

        elif self.l1_mode == L1Mode.CONSTANT_SUBCELL_PROJECTION:
            # Apply subcell_based projection onto constant vectors and sum up.
            # Equivalent to a mixed finite volume method (FV). Identical to quadrature
            # over corners.
            quad_pts, quad_weights = darsia.quadrature.reference_cell_corners(
                self.grid.dim
            )

        elif self.l1_mode == L1Mode.CONSTANT_CELL_PROJECTION:
            # L2 projection onto constant vectors identical to quadrature of order 0.
            quad_pts, quad_weights = darsia.quadrature.gauss_reference_cell(
                self.grid.dim, 0
            )

        else:
            raise ValueError(f"Mode {self.l1_mode} not supported.")

        # Integrate over reference cell (normalization not required)
        transport_density = np.zeros(self.grid.shape, dtype=float)
        for quad_pt, quad_weight in zip(quad_pts, quad_weights):
            cell_flux = darsia.face_to_cell(self.grid, flat_flux, pt=quad_pt)
            if weighted:
                weighted_cell_flux = self.cell_weighted_flux(cell_flux)
                cell_flux_norm = np.linalg.norm(weighted_cell_flux, 2, axis=-1)
            else:
                cell_flux_norm = np.linalg.norm(cell_flux, 2, axis=-1)
            transport_density += quad_weight * cell_flux_norm

        if flatten:
            return np.ravel(transport_density, "F")
        else:
            return transport_density

    def l1_dissipation(self, flat_flux: np.ndarray) -> float:
        """Compute the l1 dissipation of the solution.

        Args:
            flat_flux (np.ndarray): flat fluxes

        Returns:
            float: l1 dissipation

        """
        # The L1 dissipation corresponds to the integral over the transport density
        transport_density = self.transport_density(flat_flux)
        return self.mass_matrix_cells.dot(transport_density).sum()

    # ! ---- Lumping of effective mobility

    def _product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the product of two arrays.

        Args:
            a (np.ndarray): array a
            b (np.ndarray): array b

        Returns:
            np.ndarray: product

        """
        if len(a.shape) == len(b.shape) + 1 and a.shape[:-1] == b.shape:
            return a * b[..., np.newaxis]
        elif len(a.shape) == len(b.shape) - 1 and a.shape == b.shape[:-1]:
            return a[..., np.newaxis] * b
        elif len(a.shape) == len(b.shape) and a.shape == b.shape:
            return a * b
        else:
            raise ValueError("Shapes not compatible.")

    def _harmonic_average(self, cell_values: np.ndarray) -> np.ndarray:
        """Compute the harmonic average of cell values on faces.

        NOTE: Averaging strategies originate from the FV literature and are adapted
        to the current setting. Usually, the effective mobility is computed cell-wise
        and then averaged to faces. When translated to flux computations, the effective
        mobility is inverted again, correlating to the face weights. Therefore,
        inversions are required.

        """
        return 1.0 / darsia.cell_to_face_average(
            self.grid, 1.0 / cell_values, mode="harmonic"
        )

    def _compute_face_weight(self, flat_flux: np.ndarray) -> np.ndarray:
        """FV-style face weighting, using harmonic averaging.

        The goal is to compute a face weight resembling the effective mobility
        which is essentially cell_weight ** 2 / |weight * flux|. Different choices
        how to compute the effective mobility are possible, combining different
        averaging strategies.

        NOTE: Averaging strategies originate from the FV literature and are adapted
        to the current setting. Usually, the effective mobility is computed cell-wise
        and then averaged to faces. When translated to flux computations, the effective
        mobility is inverted again, correlating to the face weights. Therefore,
        inversions are required.

        """
        if self.mobility_mode in [
            MobilityMode.CELL_BASED,
            MobilityMode.CELL_BASED_HARMONIC,
        ]:
            # Idea: Compute first weight ** 2 / |weight * flux| on cells
            # and reduce the values to faces via harmonic averaging.
            # |weight * flux| on cells
            weighted_cell_flux_norm = self.transport_density(
                flat_flux, weighted=True, flatten=False
            )
            regularized_weighted_cell_flux_norm = np.maximum(
                weighted_cell_flux_norm,
                self.regularization,
            )
            # weight  * (weight / |weight * flux|): cell -> face via harmonic averaging
            # TODO use _harmonic_average
            cell_weights_inv = self._product(
                1.0 / self.cell_weights**2, regularized_weighted_cell_flux_norm
            )
            face_weights_inv = darsia.cell_to_face_average(
                self.grid, cell_weights_inv, mode="harmonic"
            )
            face_weights = 1.0 / face_weights_inv
        elif self.mobility_mode == MobilityMode.CELL_BASED_ARITHMETIC:
            # Idea: Combine two factors: Reduce weight to faces via
            # harmonic averaging and compute weight / |weight * flux| on faces via
            # arithmetic averaging.
            # cell_weight: cell -> face via harmonic averaging
            harm_avg_face_weights = self._harmonic_average(self.cell_weights)
            # |weight * flux|
            weighted_cell_flux_norm = self.transport_density(
                flat_flux, weighted=True, flatten=False
            )
            regularized_weighted_cell_flux_norm = np.maximum(
                weighted_cell_flux_norm,
                self.regularization,
            )
            # |weight| / |weight * flux|: cell -> face via arithmetic averaging
            # of the inverse
            weight_ratio_inv = self._product(
                1.0 / self.cell_weights, regularized_weighted_cell_flux_norm
            )
            arithm_avg_weight_ratio_inv = darsia.cell_to_face_average(
                self.grid, weight_ratio_inv, mode="arithmetic"
            )
            face_weights = harm_avg_face_weights / arithm_avg_weight_ratio_inv
            face_weights_inv = 1.0 / face_weights
        elif self.mobility_mode == MobilityMode.SUBCELL_BASED:
            # Idea: Aply harmonic averaging of cell weights and weighted sub cell
            # reconstruction to compute |weight * flux| on faces.

            # cell_weight: cell -> face via harmonic averaging
            harm_avg_face_weights = self._harmonic_average(self.cell_weights)

            # Subcell-based mode determines the norm of the fluxes on the faces via
            # averaging of neighboring subcells.

            # Initialize the flux norm |weight * flux| on the faces
            num_subcells = 2**self.grid.dim
            subcell_flux_norm = np.zeros(
                (self.grid.num_faces, num_subcells), dtype=float
            )
            flat_weighted_flux_norm = np.zeros(self.grid.num_faces, dtype=float)

            # Fetch cell corners
            cell_corners = self.grid.cell_corners

            # Strategy: Follow the lead: 1. find all faces, 2. Visit their neighbouring
            # cells, 3. find the corresponding corners, 4. compute the flux in each

            # Iterate over all normal orientations
            for orientation in range(self.grid.dim):
                # Fetch all faces with this orientations
                faces = self.grid.faces[orientation]

                # Pick the neighbouring cells (use left and right just for synonyms)
                for i, side in enumerate(range(2)):
                    # Fetch cells and respective corners corresponding to the faces.
                    cells = self.grid.connectivity[faces, side]
                    # Due to the structured nature, all faces have the same connectivity
                    # and corner indices.
                    cell_corner_indices = self.grid.cell_corner_indices[faces[0], side]

                    # Pick the corresponding coordinates
                    coordinates = cell_corners[cell_corner_indices]

                    for j, pt in enumerate(coordinates):
                        # Evaluate the norm of the flux at the coordinates
                        subcell_flux = darsia.face_to_cell(self.grid, flat_flux, pt=pt)
                        # Apply cell-based weighting
                        weighted_subcell_flux = self.cell_weighted_flux(subcell_flux)

                        # Store the norm of the subcell flux from the cell associated to
                        # the flux
                        id = i * len(coordinates) + j
                        subcell_flux_norm[faces, id] = np.linalg.norm(
                            weighted_subcell_flux, 2, axis=-1
                        ).ravel("F")[cells]

            # Average over the subcells using harmonic averaging
            flat_weighted_flux_norm = hmean(subcell_flux_norm, axis=1)

            # Combine weights**2 / |weight * flux| on faces
            face_weights = harm_avg_face_weights**2 / flat_weighted_flux_norm
            face_weights_inv = 1.0 / face_weights

        elif self.mobility_mode == MobilityMode.FACE_BASED:
            # Idea: Use harmonic averaging of cell weights and face reconstruction
            # to compute |weight * flux| on faces.

            # weight: cell -> face
            harm_avg_face_weights = self._harmonic_average(self.cell_weights)

            # Define natural vector valued flux on faces (taking arithmetic averages
            # of continuous fluxes over cells evaluated at faces)
            if not hasattr(self, "face_reconstruction"):
                self._setup_face_reconstruction()
            full_face_flux = self.face_reconstruction(flat_flux)

            # Determine the l2 norm of the fluxes on the faces
            weighted_face_flux = self._product(harm_avg_face_weights, full_face_flux)
            norm_weighted_face_flux = np.maximum(
                np.linalg.norm(weighted_face_flux, 2, axis=1), self.regularization
            )

            # Combine weights**2 / |weight * flux| on faces
            face_weights = harm_avg_face_weights**2 / norm_weighted_face_flux
            face_weights_inv = norm_weighted_face_flux / harm_avg_face_weights**2
        else:
            raise ValueError(f"Mobility mode {self.mobility_mode} not supported.")

        return face_weights, face_weights_inv

    def optimality_conditions(
        self, rhs: np.ndarray, solution: np.ndarray
    ) -> np.ndarray:
        """Evaluate optimality conditions of the constrained minimization problem.

        This is identical to the residual of the Newton system.

        Args:
            rhs (np.ndarray): right hand side
            solution (np.ndarray): solution

        Returns:
            np.ndarray: residual

        """
        flat_flux = solution[self.flux_slice]
        face_weights, _ = self._compute_face_weight(flat_flux)
        weight = sps.diags(face_weights)

        return (
            rhs
            - self.broken_darcy.dot(solution)
            - self.flux_embedding.dot(weight.dot(self.mass_matrix_faces.dot(flat_flux)))
        )

    # ! ---- Solver methods ----
    def setup_petsc_variables(self, weight: np.ndarray = None):
        """
        Instantiate the Petsc variables for the Schur complement
        """
        # Setup Petsc operators
        self.div_petsc = darsia.linalg.numpy_to_petsc(self.div)
        n = self.grid.num_faces

        # THE FOLLOWING DOES NOT WORK in the matrix-matrix multiplication
        # PETSc.Mat().createDiagonal(self.inv_weight_laplacian_vec)
        if weight is None:
            ones = np.ones(n)
            scipy_sparse = sps.diags(ones)
            self.weight_laplacian_vec = darsia.linalg.numpy_to_petsc(ones)
        else:
            scipy_sparse = sps.diags(weight)
            self.weight_laplacian_vec = darsia.linalg.numpy_to_petsc(weight)
        self.weight_laplacian_matrix = darsia.linalg.numpy_to_petsc(scipy_sparse)

        # We need to create a matrix and set the diagonal

        self.weight_laplacian_matrix.setDiagonal(self.weight_laplacian_vec)

        # We store also the grad in order to use the matrix-matrix-matrix multiplication
        self.grad_petsc = self.div_petsc.copy()
        self.grad_petsc.transpose()

        # assign
        self.laplacian_matrix = self.div_petsc.matMatMult(
            self.weight_laplacian_matrix, self.grad_petsc
        )

    def assemble_schur_complement(self, weights: np.ndarray):
        """
        Assemble the Schur complement matrix = D^T * W * D
        in the Petsc matrix self.weight_laplacian_matrix
        """
        self.weight_laplacian_vec.setArray(weights)
        self.weight_laplacian_matrix.setDiagonal(self.weight_laplacian_vec)
        self.div_petsc.matMatMult(
            self.weight_laplacian_matrix, self.grad_petsc, result=self.laplacian_matrix
        )

    def linear_solve(
        self,
        matrix: sps.csc_matrix,
        rhs: np.ndarray,
        previous_solution: Optional[np.ndarray] = None,
        reuse_solver: bool = False,
    ) -> tuple:
        """Solve the linear system.

        Defines the Schur complement reduction and the pure pressure reduction, if
        selected. For reusing the setup, the resulting solver is cached as
        self.linear_solver.

        Args:
            matrix (sps.csc_matrix): matrix
            rhs (np.ndarray): right hand side
            previous_solution (np.ndarray): previous solution. Defaults to None.
            reuse_solver (bool): reuse the solver. Defaults to False.

        Returns:
            tuple: solution, stats

        """

        setup_linear_solver = not (reuse_solver) or not (hasattr(self, "linear_solver"))
        if self.formulation == "full":
            assert (
                self.linear_solver_type == "direct" or self.linear_solver_type == "ksp"
            ), "Only direct solver or ksp supported for full formulation."

            # Setup LU factorization for the full system
            tic = time.time()
            if setup_linear_solver:
                if self.linear_solver_type == "direct":
                    self.setup_direct_solver(matrix)
                elif self.linear_solver_type == "ksp":
                    if hasattr(self, "linear_solver"):
                        self.linear_solver.kill()
                    # Extract get the flux-pressure matrix
                    # TODO: Avoid all these conversions and memory allocations
                    diag = matrix.diagonal()
                    A_00 = sps.diags(diag[self.flux_slice])
                    flux_pressure_matrix = sps.bmat(
                        [
                            [A_00, -self.div.T],
                            [self.div, None],
                        ],
                        format="csc",
                    )
                    kernel = np.zeros(flux_pressure_matrix.shape[0])
                    kernel[self.pressure_slice] = 1.0
                    kernel /= np.linalg.norm(kernel)
                    self.setup_ksp_solver(
                        flux_pressure_matrix,
                        field_ises=[
                            ("flux", self.flux_indices),
                            ("pressure", self.pressure_indices),
                        ],
                        nullspace=[kernel],
                    )
            time_setup = time.time() - tic

            # Solve the full system
            tic = time.time()
            if self.linear_solver_type == "direct":
                solution = self.linear_solver.solve(rhs)
            elif self.linear_solver_type == "ksp":
                solution_flux_pressure = self.linear_solver.solve(rhs[0:-1])
                solution = np.zeros_like(rhs)
                solution[0:-1] = solution_flux_pressure
                solution[-1] = 0.0
            time_solve = time.time() - tic

        elif self.formulation == "flux_reduced":
            # Solve flux-reduced / pressure-multiplier problem, following:
            # 1. Eliminate flux block
            # 2. Build linear solver for reduced system
            # 3. Solve reduced system
            # 4. Compute flux update

            # Start timer to measure setup time
            tic = time.time()

            # Allocate memory for solution
            solution = np.zeros_like(rhs)

            if setup_linear_solver:
                # 1. Reduce flux block
                if self.linear_solver_type == "ksp":
                    # Update the Schur complement matrix B^T * A^{-1} * B
                    weight = 1.0 / matrix.diagonal()[self.flux_slice]
                    self.matrix_flux_inv = sps.diags(weight)
                    if not hasattr(self, "div_petsc"):
                        # setup for the first time
                        self.setup_petsc_variables(weight=weight)
                    else:
                        self.assemble_schur_complement(weights=weight)

                else:
                    # Compute the inverse of the flux block
                    self.matrix_flux_inv = self.compute_invA(matrix)
                    # form the schur complement
                    self.reduced_matrix = self.compute_Schur_complement(
                        self.matrix_flux_inv
                    )

            # create the reduced rhs
            if self.linear_solver_type == "ksp":
                self.reduced_rhs = rhs[self.pressure_slice].copy()
                self.reduced_rhs -= self.div.dot(
                    self.matrix_flux_inv.dot(rhs[self.flux_slice])
                )
            else:
                self.reduced_rhs = self.compute_reduced_rhs(self.matrix_flux_inv, rhs)

            if setup_linear_solver:
                # 2. Build linear solver for reduced system
                if self.linear_solver_type == "direct":
                    self.setup_direct_solver(self.reduced_matrix)
                elif self.linear_solver_type == "amg":
                    self.setup_amg_solver(self.reduced_matrix)
                elif self.linear_solver_type == "cg":
                    self.setup_cg_solver(self.reduced_matrix)
                elif self.linear_solver_type == "ksp":
                    # Setup KSP solver for the first time
                    if not hasattr(self, "linear_solver"):
                        kernel = np.ones(self.grid.num_cells)
                        kernel /= np.linalg.norm(kernel)
                        self.setup_ksp_solver(self.laplacian_matrix, nullspace=[kernel])
                    else:
                        # Just update the matrix
                        self.linear_solver.ksp.setOperators(self.laplacian_matrix)

            # Stop timer to measure setup time
            time_setup = time.time() - tic

            # 3. Solve for the pressure and lagrange multiplier
            tic = time.time()
            if self.linear_solver_type == "ksp":
                pot = self.linear_solver.solve(self.reduced_rhs, **self.solver_options)
                solution[self.pressure_slice] = pot
                solution[self.lagrange_multiplier_slice] = 0.0
            else:
                solution[self.reduced_system_slice] = self.linear_solver.solve(
                    self.reduced_rhs, **self.solver_options
                )

            # 4. Compute flux update
            solution[self.flux_slice] = self.compute_flux_update(solution, rhs)
            time_solve = time.time() - tic

        elif self.formulation == "pressure":
            # Solve pure pressure problem ,following:
            # 1. Eliminate flux block
            # 2. Eliminate lagrange multiplier
            # 3. Build linear solver for pure pressure system
            # 4. Solve pure pressure system
            # 5. Compute lagrange multiplier - not required, as it is zero
            # 6. Compute flux update

            # Start timer to measure setup time
            tic = time.time()

            # Allocate memory for solution
            solution = np.zeros_like(rhs)

            # NOTE: It is implicitly assumed that the lagrange multiplier is zero
            # in the constrained cell. This is not checked here. And no update is
            # performed.
            if previous_solution is not None and (
                abs(
                    previous_solution[
                        self.grid.num_faces + self.constrained_cell_flat_index
                    ]
                )
                > 1e-6
            ):
                raise NotImplementedError(
                    "Implementation requires solution satisfy the constraint."
                )

            if setup_linear_solver:
                # 1. Reduce flux block
                # Compute the inverse of the flux block
                self.matrix_flux_inv = self.compute_invA(matrix)
                # form the schur complement
                self.reduced_matrix = self.compute_Schur_complement(
                    self.matrix_flux_inv
                )

            self.reduced_rhs = self.compute_reduced_rhs(self.matrix_flux_inv, rhs)
            self.fully_reduced_rhs = self.eliminate_lagrange_multiplier_rhs(
                self.reduced_rhs
            )

            if setup_linear_solver:
                # 2. Reduce to pure pressure system
                self.fully_reduced_matrix = self.eliminate_lagrange_multiplier_matrix(
                    self.reduced_matrix
                )

                # 3. Build linear solver for pure pressure system
                if self.linear_solver_type == "direct":
                    self.setup_direct_solver(self.fully_reduced_matrix)

                elif self.linear_solver_type == "amg":
                    self.setup_amg_solver(self.fully_reduced_matrix)

                elif self.linear_solver_type == "cg":
                    self.setup_cg_solver(self.fully_reduced_matrix)

                elif self.linear_solver_type == "ksp":
                    if hasattr(self, "linear_solver"):
                        if not reuse_solver:
                            self.linear_solver.kill()
                            self.setup_ksp_solver(self.fully_reduced_matrix)
                    else:
                        self.setup_ksp_solver(self.fully_reduced_matrix)

                    self.linear_solver.setup(self.solver_options)

            # Stop timer to measure setup time
            time_setup = time.time() - tic

            # 4. Solve the pure pressure system
            tic = time.time()
            solution[self.fully_reduced_system_indices_full] = self.linear_solver.solve(
                self.fully_reduced_rhs, **self.solver_options
            )

            # 5. Compute lagrange multiplier - not required, as it is zero
            pass

            # 6. Compute flux update
            solution[self.flux_slice] = self.compute_flux_update(solution, rhs)
            time_solve = time.time() - tic

        # Define solver statistics
        stats = {
            "time_setup": time_setup,
            "time_solve": time_solve,
        }
        if self.linear_solver_type in ["amg_flux_reduced", "amg_pressure"]:
            stats["amg num iterations"] = len(self.res_history_amg)
            stats["amg residual"] = self.res_history_amg[-1]
            stats["amg residuals"] = self.res_history_amg

        return solution, stats

    def eliminate_flux(self, jacobian: sps.csc_matrix, residual: np.ndarray) -> tuple:
        """Eliminate the flux block from the jacobian and residual.

        Employ a Schur complement/block Gauss elimination approach.

        Args:
            jacobian (sps.csc_matrix): jacobian
            residual (np.ndarray): residual

        Returns:
            tuple: reduced jacobian, reduced residual, inverse of flux block

        """
        # Build Schur complement wrt flux-block
        J_inv = sps.diags(1.0 / jacobian.diagonal()[self.flux_slice])
        schur_complement = self.D.dot(J_inv.dot(self.DT))

        # Gauss eliminiation on matrices
        reduced_jacobian = self.jacobian_subblock + schur_complement

        # Gauss elimination on vectors
        reduced_residual = residual[self.reduced_system_slice].copy()
        reduced_residual -= self.D.dot(J_inv.dot(residual[self.flux_slice]))

        return reduced_jacobian, reduced_residual, J_inv

    def compute_reduced_rhs(
        self, J_inv: sps.csc_matrix, residual: np.ndarray
    ) -> np.ndarray:
        """
        Compute the reduced right hand side.
        """
        # Gauss elimination on vectors
        reduced_residual = residual[self.reduced_system_slice].copy()
        reduced_residual -= self.D.dot(J_inv.dot(residual[self.flux_slice]))

        return reduced_residual

    def compute_invA(self, jacobian: sps.csc_matrix) -> sps.csc_matrix:
        """
        Set the inverse of the flux block
        """
        return sps.diags(1.0 / jacobian.diagonal()[self.flux_slice])

    def compute_Schur_complement(self, invA: sps.csc_matrix) -> sps.csc_matrix:
        """
        Compute the Schur complement, including the lagrange multiplier block
        """
        # Build Schur complement wrt flux-block
        return self.jacobian_subblock + self.D.dot(invA.dot(self.DT))

    def eliminate_lagrange_multiplier(
        self, reduced_jacobian, reduced_residual
    ) -> tuple:
        """Eliminate the lagrange multiplier from the reduced system.

        Employ a Schur complement/block Gauss elimination approach.

        Args:
            reduced_jacobian (sps.csc_matrix): reduced jacobian
            reduced_residual (np.ndarray): reduced residual

        Returns:
            tuple: fully reduced jacobian, fully reduced residual

        """
        # Make sure the jacobian is a CSC matrix
        assert isinstance(
            reduced_jacobian, sps.csc_matrix
        ), "Jacobian should be a CSC matrix."

        # Effective Gauss-elimination for the particular case of the lagrange multiplier
        self.fully_reduced_jacobian.data[:] = np.delete(
            reduced_jacobian.data.copy(), self.rm_indices
        )
        # NOTE: The indices have to be restored if the LU factorization is to be used
        # FIXME omit if not required
        self.fully_reduced_jacobian.indices = self.fully_reduced_jacobian_indices.copy()

        # Rhs is not affected by Gauss elimination as it is assumed that the residual
        # is zero in the constrained cell, and the pressure is zero there as well.
        # If not, we need to do a proper Gauss elimination on the right hand side!
        if abs(reduced_residual[-1]) > 1e-6:
            raise NotImplementedError("Implementation requires residual to be zero.")
        fully_reduced_residual = reduced_residual[
            self.fully_reduced_system_indices
        ].copy()

        return self.fully_reduced_jacobian, fully_reduced_residual

    def eliminate_lagrange_multiplier_rhs(self, reduced_residual) -> tuple:
        """Eliminate the lagrange multiplier from the reduced system.

        Employ a Schur complement/block Gauss elimination approach.

        Args:
            reduced_residual (np.ndarray): reduced residual

        Returns:
            np.ndarray: fully reduced residual

        """
        # Rhs is not affected by Gauss elimination as it is assumed that the residual
        # is zero in the constrained cell, and the pressure is zero there as well.
        # If not, we need to do a proper Gauss elimination on the right hand side!
        if abs(reduced_residual[-1]) > 1e-6:
            raise NotImplementedError("Implementation requires residual to be zero.")
        fully_reduced_residual = reduced_residual[
            self.fully_reduced_system_indices
        ].copy()

        return fully_reduced_residual

    def eliminate_lagrange_multiplier_matrix(self, reduced_jacobian) -> tuple:
        """Eliminate the lagrange multiplier from the reduced system.

        Employ a Schur complement/block Gauss elimination approach.

        Args:
            reduced_jacobian (sps.csc_matrix): reduced jacobian
            reduced_residual (np.ndarray): reduced residual

        Returns:
            tuple: fully reduced jacobian, fully reduced residual

        """
        # Make sure the jacobian is a CSC matrix
        assert isinstance(
            reduced_jacobian, sps.csc_matrix
        ), "Jacobian should be a CSC matrix."

        # Effective Gauss-elimination for the particular case of the lagrange multiplier
        self.fully_reduced_jacobian.data[:] = np.delete(
            reduced_jacobian.data.copy(), self.rm_indices
        )
        # NOTE: The indices have to be restored if the LU factorization is to be used
        # FIXME omit if not required
        self.fully_reduced_jacobian.indices = self.fully_reduced_jacobian_indices.copy()

        return self.fully_reduced_jacobian

    def compute_flux_update(self, solution: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Compute the flux update from the solution.

        Args:
            solution (np.ndarray): solution
            rhs (np.ndarray): right hand side

        Returns:
            np.ndarray: flux update

        """
        rhs_flux = rhs[self.flux_slice]
        return self.matrix_flux_inv.dot(
            rhs_flux + self.DT.dot(solution[self.reduced_system_slice])
        )

    # ! ---- Main methods ----

    @abstractmethod
    def _solve(self, rhs: np.ndarray) -> tuple:
        """Solve for the Wasserstein distance.

        Args:
            rhs (np.ndarray): right hand side

        Returns:
            tuple: distance, solution, info

        """
        pass

    def __call__(
        self,
        img_1: darsia.Image,
        img_2: darsia.Image,
    ) -> float:
        """L1 Wasserstein distance for two images with same mass.

        NOTE: Images need to comply with the setup of the object.

        Args:
            img_1 (darsia.Image): image 1, source distribution
            img_2 (darsia.Image): image 2, destination distribution

        Returns:
            float: distance between img_1 and img_2.
            dict (optional): solution
            dict (optional): info

        """

        # Compatibilty check
        assert img_1.scalar and img_2.scalar
        self._compatibility_check(img_1, img_2)

        # Determine difference of distributions and define corresponding rhs
        mass_diff = img_2.img - img_1.img
        flat_mass_diff = np.ravel(mass_diff, "F")

        # Main method
        distance, solution, info = self._solve(flat_mass_diff)

        # Split the solution
        flat_flux = solution[self.flux_slice]
        flat_pressure = solution[self.pressure_slice]

        # Reshape the fluxes and pressure to grid format
        flux = darsia.face_to_cell(self.grid, flat_flux)
        pressure = flat_pressure.reshape(self.grid.shape, order="F")

        # Determine transport density
        transport_density = self.transport_density(flat_flux, flatten=False)

        # Cell-weighted flux
        weighted_flux = self.cell_weighted_flux(flux)

        # Return solution
        return_info = self.options.get("return_info", False)
        return_status = self.options.get("return_status", False)
        if return_info:
            info.update(
                {
                    "grid": self.grid,
                    "mass_diff": mass_diff,
                    "flux": flux,
                    "weight": self.cell_weights,
                    "weight_inv": 1.0 / self.cell_weights,
                    "weighted_flux": weighted_flux,
                    "pressure": pressure,
                    "transport_density": transport_density,
                    "src": img_1,
                    "dst": img_2,
                }
            )
            return distance, info
        elif return_status:
            return distance, info["converged"]
        else:
            return distance

    # ! ---- Utility methods ----

    def _analyze_timings(self, timings: dict) -> dict:
        """Analyze the timing of the current iteration.

        Utility function for self._solve().

        Args:
            timings (dict): timings

        Returns:
            dict: total time

        """
        total_timings = {
            "assemble": sum([t["time_assemble"] for t in timings]),
            "setup": sum([t["time_setup"] for t in timings]),
            "solve": sum([t["time_solve"] for t in timings]),
            "acceleration": sum([t["time_acceleration"] for t in timings]),
        }
        total_timings["total"] = (
            total_timings["assemble"]
            + total_timings["setup"]
            + total_timings["solve"]
            + total_timings["acceleration"]
        )

        return total_timings


class WassersteinDistanceNewton(VariationalWassersteinDistance):
    """Class to determine the L1 EMD/Wasserstein distance solved with Newton's method.

    Here, self.L has the interpretation of a lower cut-off value in the linearization
    only. With such relaxation, the Beckman problem itself is not regularized, but
    instead the solution trajectory is merely affected.

    """

    def __init__(self, grid, weight, options) -> None:
        super().__init__(grid, weight, options)

        self.L = self.options.get("L", np.finfo(float).max)
        """float: relaxation/cut-off parameter for mobility, deactivated by default"""

    def _setup_discretization(self) -> None:
        """Setup of fixed discretization operators.

        Add linear contribution of the optimality conditions of the Newton linearization.

        """
        super()._setup_discretization()

        self.broken_darcy = sps.bmat(
            [
                [None, -self.div.T, None],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ],
            format="csc",
        )
        """sps.csc_matrix: linear part of the Darcy operator with pressure constraint"""

    def residual(self, rhs: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """Compute the residual of the solution.

        Args:
            rhs (np.ndarray): right hand side
            solution (np.ndarray): solution

        Returns:
            np.ndarray: residual

        """
        return self.optimality_conditions(rhs, solution)

    def jacobian(self, solution: np.ndarray) -> sps.linalg.LinearOperator:
        """Compute the LU factorization of the Jacobian of the solution.

        Args:
            solution (np.ndarray): solution

        Returns:
            sps.linalg.splu: LU factorization of the jacobian

        """
        flat_flux = solution[self.flux_slice]
        face_weights, _ = self._compute_face_weight(flat_flux)
        weight = sps.diags(face_weights)
        approx_jacobian = sps.bmat(
            [
                [
                    weight @ self.mass_matrix_faces,
                    -self.div.T,
                    None,
                ],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ],
            format="csc",
        )
        return approx_jacobian

    def _solve(self, flat_mass_diff: np.ndarray) -> tuple[float, np.ndarray, dict]:
        """Solve the Beckman problem using Newton's method.

        Args:
            flat_mass_diff (np.ndarray): difference of mass distributions

        Returns:
            tuple: distance, solution, info

        """
        # Setup time and memory profiling
        tic = time.time()
        tracemalloc.start()

        # Solver parameters. By default tolerances for increment and distance are
        # set, such that they do not affect the convergence.
        num_iter = self.options.get("num_iter", 100)
        tol_residual = self.options.get("tol_residual", np.finfo(float).max)
        tol_increment = self.options.get("tol_increment", np.finfo(float).max)
        tol_distance = self.options.get("tol_distance", np.finfo(float).max)

        # Define right hand side
        rhs = np.concatenate(
            [
                np.zeros(self.grid.num_faces, dtype=float),
                self.mass_matrix_cells.dot(flat_mass_diff),
                np.zeros(1, dtype=float),
            ]
        )

        # Initialize Newton iteration with Darcy solution for unitary mobility
        solution_i = np.zeros_like(rhs, dtype=float)
        solution_i, _ = self.linear_solve(
            self.darcy_init.copy(), rhs.copy(), solution_i
        )

        # Initialize distance in case below iteration fails
        new_distance = 0

        # Initialize container for storing the convergence history
        convergence_history = {
            "distance": [],
            "residual": [],
            "flux_increment": [],
            "distance_increment": [],
            "timing": [],
            "run_time": [],
        }

        # Print  header for later printing performance to screen
        # - distance
        # - distance increment
        # - flux increment
        # - residual
        if self.verbose:
            print(
                "Newton iter. \t| W^1 \t\t| Δ W^1 / W^1 \t| Δ flux \t| residual",
                "\n",
                """---------------|---------------|---------------|---------------|"""
                """---------------""",
            )

        # Newton iteration
        for iter in range(num_iter):
            # It is possible that the linear solver fails. In this case, we simply
            # stop the iteration and return the current solution.
            try:
                # Keep track of old flux, and old distance
                old_solution_i = solution_i.copy()
                flux = solution_i[self.flux_slice]
                old_distance = self.l1_dissipation(flux)

                # Assemble linear problem in Newton step
                tic = time.time()
                residual_i = self.residual(rhs, solution_i)
                approx_jacobian = self.jacobian(solution_i)
                toc = time.time()
                time_assemble = toc - tic

                # Solve linear system for the update
                update_i, stats_i = self.linear_solve(
                    approx_jacobian, residual_i, solution_i
                )

                # Include assembly in statistics
                stats_i["time_assemble"] = time_assemble

                # Update the solution with the full Netwon step
                solution_i += update_i

                # Apply Anderson acceleration to flux contribution (the only nonlinear part).
                # Application to full solution, or just the pressure, lead to divergence,
                # while application to the flux, results in improved performance.
                tic = time.time()
                if self.anderson is not None:
                    solution_i[self.flux_slice] = self.anderson(
                        solution_i[self.flux_slice],
                        update_i[self.flux_slice],
                        iter,
                    )
                stats_i["time_acceleration"] = time.time() - tic

                # Update discrete W1 distance
                flux = solution_i[self.flux_slice]
                new_distance = self.l1_dissipation(flux)

                # Update increment
                increment = solution_i - old_solution_i

                # Compute the error and store as part of the convergence history:
                # 0 - full residual (Newton interpretation)
                # 1 - flux increment (fixed-point interpretation)
                # 2 - distance increment (Minimization interpretation)

                # Update convergence history
                convergence_history["distance"].append(new_distance)
                convergence_history["residual"].append(np.linalg.norm(residual_i, 2))
                convergence_history["flux_increment"].append(
                    np.linalg.norm(increment[self.flux_slice], 2)
                )
                convergence_history["distance_increment"].append(
                    abs(new_distance - old_distance)
                )
                convergence_history["timing"].append(stats_i)

                # Extract current total run time
                current_run_time = self._analyze_timings(convergence_history["timing"])[
                    "total"
                ]
                convergence_history["run_time"].append(current_run_time)

                # Print performance to screen
                # - distance
                # - distance increment
                # - flux increment
                # - residual
                if self.verbose:
                    distance_increment = (
                        convergence_history["distance_increment"][-1] / new_distance
                    )
                    flux_increment = (
                        convergence_history["flux_increment"][-1]
                        / convergence_history["flux_increment"][0]
                    )
                    residual = (
                        convergence_history["residual"][-1]
                        / convergence_history["residual"][0]
                    )
                    print(
                        f"""Iter. {iter} \t| {new_distance:.6e} \t| """
                        f"""{distance_increment:.6e} \t| {flux_increment:.6e} \t| """
                        f"""{residual:.6e}"""
                    )

                # Stopping criterion - force one iteration. BAse stopping criterion on
                # different interpretations of the Newton method:
                # - Newton interpretation: full residual
                # - Fixed-point interpretation: flux increment
                # - Minimization interpretation: distance increment
                # For default tolerances, the code is prone to overflow. Surpress the
                # warnings here.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="overflow encountered")
                    if iter > 1 and (
                        (
                            convergence_history["residual"][-1]
                            < tol_residual * convergence_history["residual"][0]
                            and convergence_history["flux_increment"][-1]
                            < tol_increment * convergence_history["flux_increment"][0]
                            and convergence_history["distance_increment"][-1]
                            < tol_distance
                        )
                    ):
                        break
            except Exception:
                warnings.warn("Newton iteration abruptly stopped due to some error.")
                break

        # Summarize profiling (time in seconds, memory in GB)
        total_timings = self._analyze_timings(convergence_history["timing"])
        peak_memory_consumption = tracemalloc.get_traced_memory()[1] / 10**9

        # Define performance metric
        info = {
            "converged": iter < num_iter - 1,
            "number_iterations": iter,
            "convergence_history": convergence_history,
            "timings": total_timings,
            "peak_memory_consumption": peak_memory_consumption,
        }

        return new_distance, solution_i, info


class WassersteinDistanceBregman(VariationalWassersteinDistance):
    """Class to determine the Wasserstein distance solved with the Bregman method."""

    def __init__(
        self,
        grid: darsia.Grid,
        weight: Optional[darsia.Image] = None,
        options: dict = {},
    ) -> None:
        """Initialize the Bregman method.

        Args:
            grid (darsia.Grid): grid
            options (dict, optional): options. Defaults to {}.

        """

        super().__init__(grid, weight, options)
        self.L = self.options.get("L", 1.0)
        """Penality parameter for the Bregman iteration, associated to face mobility."""

    def _setup_dof_management(self) -> None:
        """Bregman-specific setup of the dof management."""
        super()._setup_dof_management()

        self.force_slice = slice(self.grid.num_faces, None)
        """slice: slice for the force."""

    def _shrink(
        self,
        flat_flux: np.ndarray,
        shrink_factor: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Shrink operation in the split Bregman method, operating on fluxes.

        To comply with the RT0 setting, the shrinkage operation merely determines the
        scalar. We still aim at following along the direction provided by the vectorial
        fluxes.

        Args:
            flat_flux (np.ndarray): flux
            shrink_factor (float or np.ndarray): shrink factor

        Returns:
            np.ndarray: shrunk fluxes

        """
        _, face_weights_inv = self._compute_face_weight(flat_flux)
        flat_scaling = np.maximum(face_weights_inv - shrink_factor, 0) / (
            face_weights_inv + self.regularization
        )
        return flat_scaling * flat_flux

    def _update_regularization(
        self, flat_flux: np.ndarray, homogeneous: bool = False
    ) -> tuple:
        """Update the regularization based on the current approximation of the flux.

        Args:
            flat_flux (np.ndarray): flux
            homogeneous (bool, optional): homogeneous regularization. Defaults to False.

        Returns:
            tuple: l_scheme_mixed_darcy, weight, shrink_factor

        """

        # Assign the weight and shrink factor
        face_weights, face_weights_inv = self._compute_face_weight(flat_flux)
        weight = sps.diags(face_weights)
        shrink_factor = face_weights_inv

        # Update the Darcy system
        l_scheme_mixed_darcy = sps.bmat(
            [
                [weight @ self.mass_matrix_faces, -self.div.T, None],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ],
            format="csc",
        )

        return l_scheme_mixed_darcy, weight, shrink_factor

    def _solve(self, flat_mass_diff: np.ndarray) -> tuple[float, np.ndarray, dict]:
        """Solve the Beckman problem using the Bregman method.

        Args:
            flat_mass_diff (np.ndarray): difference of mass distributions

        Returns:
            tuple: distance, solution, info

        """
        # Setup time and memory profiling
        tic = time.time()
        tracemalloc.start()

        # Solver parameters
        num_iter = self.options.get("num_iter", 100)
        tol_residual = self.options.get("tol_residual", np.finfo(float).max)
        tol_increment = self.options.get("tol_increment", np.finfo(float).max)
        tol_distance = self.options.get("tol_distance", np.finfo(float).max)

        # Define right hand side
        rhs = np.concatenate(
            [
                np.zeros(self.grid.num_faces, dtype=float),
                self.mass_matrix_cells.dot(flat_mass_diff),
                np.zeros(1, dtype=float),
            ]
        )

        # Initialize Newton iteration with Darcy solution for unitary mobility
        solution_i = np.zeros_like(rhs, dtype=float)
        solution_i, _ = self.linear_solve(
            self.darcy_init.copy(), rhs.copy(), solution_i
        )

        # Initialize distance in case below iteration fails
        new_distance = 0

        # Initialize container for storing the convergence history
        convergence_history = {
            "distance": [],
            "mass_conservation_residual": [],
            "aux_force_increment": [],
            "distance_increment": [],
            "timing": [],
            "run_time": [],
        }

        # Print header
        if self.verbose:
            print(
                "Bregman iter. \t| W^1 \t\t| Δ W^1/W^1 \t| Δ aux/force \t| mass residual",
                "\n",
                """---------------|---------------|---------------|---------------|"""
                """---------------""",
            )

        # Relaxation parameter entering Bregman regularization
        weight = 1.0 / self.L * sps.diags(self.face_weights, format="csc")
        shrink_factor = self.L / self.face_weights

        # Initialize linear problem corresponding to Bregman regularization
        l_scheme_mixed_darcy = sps.bmat(
            [
                [weight @ self.mass_matrix_faces, -self.div.T, None],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ],
            format="csc",
        )

        # Initialize Bregman variables
        flux = solution_i[self.flux_slice]
        old_aux_flux = self._shrink(flux, shrink_factor)
        old_force = flux - old_aux_flux
        old_distance = self.l1_dissipation(flux)

        iter = 0

        # Control the update of the Bregman weight
        bregman_update = self.options.get("bregman_update", lambda iter: False)
        bregman_homogeneous = self.options.get("bregman_homogeneous", False)

        for iter in range(num_iter):
            # It is possible that the linear solver fails. In this case, we simply
            # stop the iteration and return the current solution.
            try:
                # (Possibly) update the regularization, based on the current approximation
                # of the flux - use the inverse of the norm of the flux
                update_solver = bregman_update(iter)
                if update_solver:
                    # 0. Update regularization
                    tic = time.time()
                    (
                        l_scheme_mixed_darcy,
                        weight,
                        shrink_factor,
                    ) = self._update_regularization(flux, bregman_homogeneous)
                    # 1. Make relaxation step (solve quadratic optimization problem)
                    # Here, re-initialize the aux flux and force with zero values again.
                    rhs_i = rhs.copy()
                    time_assemble = time.time() - tic
                    # Force to update the internally stored linear solver
                    tic = time.time()
                    solution_i, stats_i = self.linear_solve(
                        l_scheme_mixed_darcy,
                        rhs_i,
                        reuse_solver=False,
                    )
                    flux = solution_i[self.flux_slice]
                    stats_i["time_solve"] = time.time() - tic
                    stats_i["time_assemble"] = time_assemble

                    # 2. Shrink step for vectorial fluxes.
                    tic = time.time()
                    new_aux_flux = self._shrink(flux, shrink_factor)
                    stats_i["time_shrink"] = time.time() - tic

                    # 3. Update force
                    tic = time.time()
                    new_force = flux - new_aux_flux

                else:
                    # 1. Make relaxation step (solve quadratic optimization problem)
                    tic = time.time()
                    rhs_i = rhs.copy()
                    rhs_i[self.flux_slice] = weight @ self.mass_matrix_faces.dot(
                        old_aux_flux - old_force
                    )
                    time_assemble = time.time() - tic
                    # Force to update the internally stored linear solver
                    tic = time.time()
                    solution_i, stats_i = self.linear_solve(
                        l_scheme_mixed_darcy,
                        rhs_i,
                        reuse_solver=iter > 0,
                    )
                    flux = solution_i[self.flux_slice]
                    stats_i["time_solve"] = time.time() - tic
                    stats_i["time_assemble"] = time_assemble

                    # 2. Shrink step for vectorial fluxes.
                    tic = time.time()
                    new_aux_flux = self._shrink(flux + old_force, shrink_factor)
                    stats_i["time_shrink"] = time.time() - tic

                    # 3. Update force
                    tic = time.time()
                    new_force = old_force + flux - new_aux_flux

                # Apply Anderson acceleration to flux contribution (the only nonlinear part).
                tic = time.time()
                if self.anderson is not None:
                    aux_inc = new_aux_flux - old_aux_flux
                    force_inc = new_force - old_force
                    inc = np.concatenate([aux_inc, force_inc])
                    iteration = np.concatenate([new_aux_flux, new_force])
                    new_iteration = self.anderson(iteration, inc, iter)
                    new_aux_flux = new_iteration[self.flux_slice]
                    new_force = new_iteration[self.force_slice]
                stats_i["time_acceleration"] = time.time() - tic

                # Update distance
                new_distance = self.l1_dissipation(flux)

                # Catch nan values
                if np.isnan(new_distance):
                    info = {
                        "converged": False,
                        "number_iterations": iter,
                        "convergence_history": convergence_history,
                    }
                    return new_distance, solution_i, info

                # Determine the error in the mass conservation equation
                mass_conservation_residual = (
                    self.div.dot(flux) - rhs[self.pressure_slice]
                )

                # Reference values
                flux_ref = np.linalg.norm(flux, 2)
                mass_ref = np.linalg.norm(rhs[self.pressure_slice], 2)

                # Determine increments
                aux_increment = new_aux_flux - old_aux_flux
                force_increment = new_force - old_force
                distance_increment = new_distance - old_distance

                # Compute the error and store as part of the convergence history:
                # 0 - aux/force increments (fixed-point formulation)
                # 1 - distance increment (minimization formulation)
                # 2 - mass conservation residual (constraint in optimization formulation)

                # Update convergence history
                convergence_history["distance"].append(new_distance)
                convergence_history["distance_increment"].append(
                    abs(distance_increment)
                )
                convergence_history["aux_force_increment"].append(
                    np.linalg.norm(np.concatenate([aux_increment, force_increment]), 2)
                    / flux_ref
                )
                convergence_history["mass_conservation_residual"].append(
                    np.linalg.norm(mass_conservation_residual, 2) / mass_ref
                )
                convergence_history["timing"].append(stats_i)

                # Extract current total run time
                current_run_time = self._analyze_timings(convergence_history["timing"])[
                    "total"
                ]
                convergence_history["run_time"].append(current_run_time)

                # Print status
                if self.verbose:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message="overflow encountered"
                        )
                        distance_increment = (
                            convergence_history["distance_increment"][-1] / new_distance
                        )
                        aux_force_increment = (
                            convergence_history["aux_force_increment"][-1]
                            / convergence_history["aux_force_increment"][0]
                        )
                        mass_conservation_residual = convergence_history[
                            "mass_conservation_residual"
                        ][-1]
                        print(
                            f"Iter. {iter} \t| {new_distance:.6e} \t| "
                            ""
                            f"""{distance_increment:.6e} \t| {aux_force_increment:.6e} \t| """
                            f"""{mass_conservation_residual:.6e}"""
                        )

                # Base stopping citeria on the different interpretations of the split Bregman
                # method:
                # - fixed-point formulation: aux flux and force increment
                # - minimization formulation: distance increment
                # - constrained optimization formulation: mass conservation residual
                # For default tolerances, the code is prone to overflow. Surpress the
                # warnings here.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="overflow encountered")
                    if iter > 1 and (
                        (
                            convergence_history["aux_force_increment"][-1]
                            < tol_increment
                            * convergence_history["aux_force_increment"][0]
                            and convergence_history["distance_increment"][-1]
                            / new_distance
                            < tol_distance
                            and convergence_history["mass_conservation_residual"][-1]
                            < tol_residual
                        )
                    ):
                        break

                # Update Bregman variables
                old_aux_flux = new_aux_flux.copy()
                old_force = new_force.copy()
                old_distance = new_distance

            except Exception:
                warnings.warn("Bregman iteration abruptly stopped due to some error.")
                break

        # Solve for the pressure by solving a single Newton iteration
        newton_jacobian, _, _ = self._update_regularization(flux)
        solution_i = np.zeros_like(rhs)
        solution_i[self.flux_slice] = flux.copy()
        newton_residual = self.optimality_conditions(rhs, solution_i)
        newton_update, _ = self.linear_solve(
            newton_jacobian, newton_residual, solution_i
        )
        solution_i[self.pressure_slice] = newton_update[self.pressure_slice]

        # Summarize profiling (time in seconds, memory in GB)
        total_timings = self._analyze_timings(convergence_history["timing"])
        peak_memory_consumption = tracemalloc.get_traced_memory()[1] / 10**9

        # Define performance metric
        info = {
            "converged": iter < num_iter - 1,
            "number_iterations": iter,
            "convergence_history": convergence_history,
            "timings": total_timings,
            "peak_memory_consumption": peak_memory_consumption,
        }

        return new_distance, solution_i, info


# Unified access
def wasserstein_distance(
    mass_1: darsia.Image,
    mass_2: darsia.Image,
    method: str,
    weight: Optional[darsia.Image] = None,
    **kwargs,
):
    """Unified access to Wasserstein distance computation between images with same mass.

    Args:
        mass_1 (darsia.Image): image 1, source distribution
        mass_2 (darsia.Image): image 2, destination distribution
        method (str): method to use ("newton", "bregman", or "cv2.emd")
        **kwargs: additional arguments (only for "newton" and "bregman")
            - options (dict): options for the method.

    """
    # Define method for computing 1-Wasserstein distance

    if method.lower() in ["newton", "bregman"]:
        # Use Finite Volume Iterative Method (Newton or Bregman)

        # Extract grid - implicitly assume mass_2 to generate same grid
        grid: darsia.Grid = darsia.generate_grid(mass_1)

        # Fetch options and define Wasserstein method
        options = kwargs.get("options", {})

        # Define method
        if method.lower() == "newton":
            w1 = WassersteinDistanceNewton(grid, weight, options)
        elif method.lower() == "bregman":
            w1 = WassersteinDistanceBregman(grid, weight, options)

    elif method.lower() == "cv2.emd":
        # Use Earth Mover's Distance from CV2
        assert weight is None, "Weighted EMD not supported by cv2."
        preprocess = kwargs.get("preprocess")
        w1 = darsia.EMD(preprocess)

    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    # Compute and return Wasserstein distance
    return w1(mass_1, mass_2)


def wasserstein_distance_to_vtk(
    path: Union[str, Path],
    info: dict,
) -> None:
    """Write the output of the Wasserstein distance to a VTK file.

    Args:
        path (Union[str, Path]): path to the VTK file
        info (dict): information dictionary

    NOTE: Requires pyevtk to be installed.

    """
    data = [
        (key, info[key], format)
        for key, format in [
            ("src", darsia.Format.SCALAR),
            ("dst", darsia.Format.SCALAR),
            ("mass_diff", darsia.Format.SCALAR),
            ("flux", darsia.Format.VECTOR),
            ("weighted_flux", darsia.Format.VECTOR),
            ("pressure", darsia.Format.SCALAR),
            ("transport_density", darsia.Format.SCALAR),
            ("weight", darsia.Format.TENSOR),
            ("weight_inv", darsia.Format.TENSOR),
        ]
    ]
    darsia.plotting.to_vtk(path, data)
