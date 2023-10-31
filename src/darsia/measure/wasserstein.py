"""Wasserstein distance computed using variational methods.

"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pyamg
import scipy.sparse as sps

import darsia

# General TODO list
# - improve documentation, in particular with focus on keywords
# - remove plotting
# - improve assembling of operators through partial assembling
# - improve stopping criteria
# - use better quadrature for l1_dissipation?
# - allow to reuse setup.


class VariationalWassersteinDistance(darsia.EMD):
    """Base class for setting up the variational Wasserstein distance.

    The variational Wasserstein distance is defined as the solution to the following
    optimization problem (also called the Beckman problem):

        inf ||u||_{L^1} s.t. div u = m_1 - m_2, u in H(div).

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
        options: dict = {},
    ) -> None:
        """Initialization of the variational Wasserstein distance.

        Args:

            grid (darsia.Grid): tensor grid associated with the images
            options (dict): options for the solver
                - num_iter (int): maximum number of iterations. Defaults to 100.
                - tol (float): tolerance for the stopping criterion. Defaults to 1e-6.
                - L (float): parameter for the Bregman iteration. Defaults to 1.0.
                - regularization (float): regularization parameter for the Bregman
                    iteration. Defaults to 0.0.
                - aa_depth (int): depth of the Anderson acceleration. Defaults to 0.
                - aa_restart (int): restart of the Anderson acceleration. Defaults to None.
                - scaling (float): scaling of the fluxes in the plot. Defaults to 1.0.
                - lumping (bool): lump the mass matrix. Defaults to True.

        """
        # Cache geometrical infos
        self.grid = grid
        """darsia.Grid: grid"""

        self.voxel_size = grid.voxel_size
        """np.ndarray: voxel size"""

        self.options = options
        """dict: options for the solver"""

        self.regularization = self.options.get("regularization", np.finfo(np.float).eps)
        """float: regularization parameter"""

        self.verbose = self.options.get("verbose", False)
        """bool: verbosity"""

        # Setup of method
        self._setup_dof_management()
        self._setup_discretization()
        self._setup_linear_solver()
        self._setup_acceleration()

    def _setup_dof_management(self) -> None:
        """Setup of DOF management.

        The following degrees of freedom are considered (also in this order):
        - flat fluxes (normal fluxes on the faces)
        - flat potentials (potentials on the cells)
        - lagrange multiplier (scalar variable) - Idea: Fix the potential in the
        center of the domain to zero. This is done by adding a constraint to the
        potential via a Lagrange multiplier.

        """
        # ! ---- Number of dofs ----
        num_flux_dofs = self.grid.num_faces
        num_potential_dofs = self.grid.num_cells
        num_lagrange_multiplier_dofs = 1
        num_dofs = num_flux_dofs + num_potential_dofs + num_lagrange_multiplier_dofs

        # ! ---- Indices in global system ----
        self.flux_indices = np.arange(num_flux_dofs)
        """np.ndarray: indices of the fluxes"""

        self.potential_indices = np.arange(
            num_flux_dofs, num_flux_dofs + num_potential_dofs
        )
        """np.ndarray: indices of the potentials"""

        self.lagrange_multiplier_indices = np.array(
            [num_flux_dofs + num_potential_dofs], dtype=int
        )
        """np.ndarray: indices of the lagrange multiplier"""

        # ! ---- Fast access to components through slices ----
        self.flux_slice = slice(0, num_flux_dofs)
        """slice: slice for the fluxes"""

        self.potential_slice = slice(num_flux_dofs, num_flux_dofs + num_potential_dofs)
        """slice: slice for the potentials"""

        self.lagrange_multiplier_slice = slice(
            num_flux_dofs + num_potential_dofs,
            num_flux_dofs + num_potential_dofs + num_lagrange_multiplier_dofs,
        )
        """slice: slice for the lagrange multiplier"""

        self.reduced_system_slice = slice(num_flux_dofs, None)
        """slice: slice for the reduced system (potentials and lagrange multiplier)"""

        # Embedding operators
        self.flux_embedding = sps.csc_matrix(
            (
                np.ones(num_flux_dofs, dtype=float),
                (self.flux_indices, self.flux_indices),
            ),
            shape=(num_dofs, num_flux_dofs),
        )
        """sps.csc_matrix: embedding operator for fluxes"""

    def _setup_discretization(self) -> None:
        """Setup of fixed discretization operators."""

        # ! ---- Constraint for the potential correpsonding to Lagrange multiplier ----

        center_cell = np.array(self.grid.shape) // 2
        self.constrained_cell_flat_index = np.ravel_multi_index(
            center_cell, self.grid.shape
        )
        """int: flat index of the cell where the potential is constrained to zero"""

        num_potential_dofs = self.grid.num_cells
        self.potential_constraint = sps.csc_matrix(
            (
                np.ones(1, dtype=float),
                (np.zeros(1, dtype=int), np.array([self.constrained_cell_flat_index])),
            ),
            shape=(1, num_potential_dofs),
            dtype=float,
        )
        """sps.csc_matrix: effective constraint for the potential"""

        # ! ---- Discretization operators ----

        self.div = darsia.FVDivergence(self.grid).mat
        """sps.csc_matrix: divergence operator: flat fluxes -> flat potentials"""

        self.mass_matrix_cells = darsia.FVMass(self.grid).mat
        """sps.csc_matrix: mass matrix on cells: flat potentials -> flat potentials"""

        lumping = self.options.get("lumping", True)
        self.mass_matrix_faces = darsia.FVMass(self.grid, "faces", lumping).mat
        """sps.csc_matrix: mass matrix on faces: flat fluxes -> flat fluxes"""

        self.face_reconstruction = darsia.FVFullFaceReconstruction(self.grid)
        """sps.csc_matrix: full face reconstruction: flat fluxes -> vector fluxes"""

        # Linear part of the Darcy operator with potential constraint.
        self.broken_darcy = sps.bmat(
            [
                [None, -self.div.T, None],
                [self.div, None, -self.potential_constraint.T],
                [None, self.potential_constraint, None],
            ],
            format="csc",
        )
        """sps.csc_matrix: linear part of the Darcy operator"""

        L_init = self.options.get("L_init", 1.0)
        self.darcy_init = sps.bmat(
            [
                [L_init * self.mass_matrix_faces, -self.div.T, None],
                [self.div, None, -self.potential_constraint.T],
                [None, self.potential_constraint, None],
            ],
            format="csc",
        )
        """sps.csc_matrix: initial Darcy operator"""

    def _setup_linear_solver(self) -> None:
        self.linear_solver_type = self.options.get("linear_solver", "lu")
        assert self.linear_solver_type in [
            "lu",
            "lu-flux-reduced",
            "amg-flux-reduced",
            "lu-potential",
            "amg-potential",
        ], f"Linear solver {self.linear_solver_type} not supported."
        """str: type of linear solver"""

        if self.linear_solver_type in ["amg-flux-reduced", "amg-potential"]:
            # TODO add possibility for user control
            self.ml_options = {
                # B=X.reshape(
                #    n * n, 1
                # ),  # the representation of the near null space (this is a poor choice)
                # BH=None,  # the representation of the left near null space
                "symmetry": "hermitian",  # indicate that the matrix is Hermitian
                # strength="evolution",  # change the strength of connection
                "aggregate": "standard",  # use a standard aggregation method
                "smooth": (
                    "jacobi",
                    {"omega": 4.0 / 3.0, "degree": 2},
                ),  # prolongation smoothing
                "presmoother": ("block_gauss_seidel", {"sweep": "symmetric"}),
                "postsmoother": ("block_gauss_seidel", {"sweep": "symmetric"}),
                # improve_candidates=[
                #    ("block_gauss_seidel", {"sweep": "symmetric", "iterations": 4}),
                #    None,
                # ],
                "max_levels": 4,  # maximum number of levels
                "max_coarse": 1000,  # maximum number on a coarse level
                # keep=False,  # keep extra operators around in the hierarchy (memory)
            }
            """dict: options for the AMG solver"""

            self.tol_amg = self.options.get("linear_solver_tol", 1e-6)
            """float: tolerance for the AMG solver"""

            self.res_history_amg = []
            """list: history of residuals for the AMG solver"""

        # Setup inrastructure for Schur complement reduction
        if self.linear_solver_type in ["lu-flux-reduced", "amg-flux-reduced"]:
            self.setup_one_level_schur_reduction()

        elif self.linear_solver_type in ["lu-potential", "amg-potential"]:
            self.setup_two_level_schur_reduction()

    def setup_one_level_schur_reduction(self) -> None:
        """Setup the infrastructure for reduced systems through Gauss elimination.

        Provide internal data structures for the reduced system.

        """
        # Step 1: Compute the jacobian of the Darcy problem

        jacobian = self.darcy_init.copy()

        # Step 2: Remove flux blocks through Schur complement approach

        # Build Schur complement wrt. flux-flux block
        J_inv = sps.diags(1.0 / jacobian.diagonal()[self.flux_slice])
        D = jacobian[self.reduced_system_slice, self.flux_slice].copy()
        schur_complement = D.dot(J_inv.dot(D.T))

        # Cache divergence matrix
        self.D = D.copy()
        """sps.csc_matrix: divergence matrix"""

        self.DT = self.D.T.copy()
        """sps.csc_matrix: transposed divergence matrix"""

        # Cache (constant) jacobian subblock
        self.jacobian_subblock = jacobian[
            self.reduced_system_slice, self.reduced_system_slice
        ].copy()
        """sps.csc_matrix: constant jacobian subblock of the reduced system"""

        # Add Schur complement - use this to identify sparsity structure
        # Cache the reduced jacobian
        self.reduced_jacobian = self.jacobian_subblock + schur_complement
        """sps.csc_matrix: reduced jacobian incl. Schur complement"""

    def setup_two_level_schur_reduction(self) -> None:
        """Additional setup of infrastructure for fully reduced systems."""
        # Step 1 and 2:
        self.setup_one_level_schur_reduction()

        # Step 3: Remove Lagrange multiplier block through Gauss elimination

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
        # Cache for later use in remove_lagrange_multiplier
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

        # Step 4: Identify inclusions (index arrays)

        # Define reduced system indices wrt full system
        reduced_system_indices = np.concatenate(
            [self.potential_indices, self.lagrange_multiplier_indices]
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

    def compute_transport_density(self, solution: np.ndarray) -> np.ndarray:
        """Compute the transport density from the solution.

        Args:
            solution (np.ndarray): solution

        Returns:
            np.ndarray: transport density

        """
        # Convert (scalar) normal fluxes to vector-valued fluxes on cells
        flat_flux = solution[self.flux_slice]
        cell_flux = darsia.face_to_cell(self.grid, flat_flux)
        # Simply take the norm without any other integration
        norm = np.linalg.norm(cell_flux, 2, axis=-1)
        return norm

    def l1_dissipation(self, flat_flux: np.ndarray, mode: str) -> float:
        """Compute the l1 dissipation potential of the solution.

        Args:
            flat_flux (np.ndarray): flat fluxes

        Returns:
            float: l1 dissipation potential

        """
        if mode == "cell_arithmetic":
            cell_flux = darsia.face_to_cell(self.grid, flat_flux)
            cell_flux_norm = np.ravel(np.linalg.norm(cell_flux, 2, axis=-1))
            return self.mass_matrix_cells.dot(cell_flux_norm).sum()
        elif mode == "face_arithmetic":
            face_flux_norm = self.vector_face_flux_norm(flat_flux, "face_arithmetic")
            return self.mass_matrix_faces.dot(face_flux_norm).sum()

    # ! ---- Lumping of effective mobility

    def vector_face_flux_norm(self, flat_flux: np.ndarray, mode: str) -> np.ndarray:
        """Compute the norm of the vector-valued fluxes on the faces.

        Args:
            flat_flux (np.ndarray): flat fluxes (normal fluxes on the faces)
            mode (str): mode of the norm, either "cell_arithmetic", "cell_harmonic" or
                "face_arithmetic". In the cell-based modes, the fluxes are projected to
                the cells and the norm is computed there. In the face-based mode, the
                norm is computed directly on the faces.

        Returns:
            np.ndarray: norm of the vector-valued fluxes on the faces

        """

        # Determine the norm of the fluxes on the faces
        if mode in ["cell_arithmetic", "cell_harmonic"]:
            # Consider the piecewise constant projection of vector valued fluxes
            cell_flux = darsia.face_to_cell(self.grid, flat_flux)
            # Determine the norm of the fluxes on the cells
            cell_flux_norm = np.maximum(
                np.linalg.norm(cell_flux, 2, axis=-1), self.regularization
            )
            # Determine averaging mode from mode - either arithmetic or harmonic
            average_mode = mode.split("_")[1]
            flat_flux_norm = darsia.cell_to_face(
                self.grid, cell_flux_norm, mode=average_mode
            )

        elif mode == "face_arithmetic":
            # Define natural vector valued flux on faces (taking arithmetic averages
            # of continuous fluxes over cells evaluated at faces)
            full_face_flux = self.face_reconstruction(flat_flux)
            # Determine the l2 norm of the fluxes on the faces
            flat_flux_norm = np.linalg.norm(full_face_flux, 2, axis=1)

        else:
            raise ValueError(f"Mode {mode} not supported.")

        return flat_flux_norm

    # ! ---- Solver methods ----

    def linear_solve(
        self,
        matrix: sps.csc_matrix,
        rhs: np.ndarray,
        previous_solution: Optional[np.ndarray] = None,
        reuse_solver: bool = False,
    ) -> tuple:
        """Solve the linear system.

        For reusing the setup, the resulting solver is cached as self.linear_solver.

        Args:
            matrix (sps.csc_matrix): matrix
            rhs (np.ndarray): right hand side
            previous_solution (np.ndarray): previous solution. Defaults to None.

        Returns:
            tuple: solution, stats

        """

        setup_linear_solver = not (reuse_solver) or not (hasattr(self, "linear_solver"))

        if self.linear_solver_type == "lu":
            # Setup LU factorization for the full system
            tic = time.time()
            if setup_linear_solver:
                self.linear_solver = sps.linalg.splu(matrix)
            time_setup = time.time() - tic

            # Solve the full system
            tic = time.time()
            solution = self.linear_solver.solve(rhs)
            time_solve = time.time() - tic

        elif self.linear_solver_type in [
            "lu-flux-reduced",
            "amg-flux-reduced",
            "lu-potential",
            "amg-potential",
        ]:
            # Solve potential-multiplier problem

            # Allocate memory for solution
            solution = np.zeros_like(rhs)

            # Reduce flux block
            tic = time.time()
            (
                self.reduced_matrix,
                self.reduced_rhs,
                matrix_flux_inv,
            ) = self.remove_flux(matrix, rhs)

            if self.linear_solver_type == "lu-flux-reduced":
                # LU factorization for reduced system
                if setup_linear_solver:
                    self.linear_solver = sps.linalg.splu(self.reduced_matrix)
                time_setup = time.time() - tic

                # Solve for the potential and lagrange multiplier
                tic = time.time()
                solution[self.reduced_system_slice] = self.linear_solver.solve(
                    self.reduced_rhs
                )

            elif self.linear_solver_type == "amg-flux-reduced":
                # AMG solver for reduced system
                if setup_linear_solver:
                    self.linear_solver = pyamg.smoothed_aggregation_solver(
                        self.reduced_matrix, **self.ml_options
                    )
                time_setup = time.time() - tic

                # Solve for the potential and lagrange multiplier
                tic = time.time()
                solution[self.reduced_system_slice] = self.linear_solver.solve(
                    self.reduced_rhs,
                    tol=self.tol_amg,
                    residuals=self.res_history_amg,
                )
            else:
                # Solve pure potential problem

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

                # Reduce to pure potential system
                (
                    self.fully_reduced_matix,
                    self.fully_reduced_rhs,
                ) = self.remove_lagrange_multiplier(
                    self.reduced_matrix,
                    self.reduced_rhs,
                )

                if self.linear_solver_type == "lu-potential":
                    # Finish LU factorization of the pure potential system
                    if setup_linear_solver:
                        self.linear_solver = sps.linalg.splu(self.fully_reduced_matrix)
                    time_setup = time.time() - tic

                    # Solve the pure potential system
                    tic = time.time()
                    solution[
                        self.fully_reduced_system_indices_full
                    ] = self.linear_solver.solve(self.fully_reduced_rhs)

                elif self.linear_solver_type == "amg-potential":
                    # Finish AMG setup of th pure potential system
                    if setup_linear_solver:
                        self.linear_solver = pyamg.smoothed_aggregation_solver(
                            self.fully_reduced_jacobian, **self.ml_options
                        )
                    time_setup = time.time() - tic

                    # Solve the pure potential system
                    tic = time.time()
                    solution[
                        self.fully_reduced_system_indices_full
                    ] = self.linear_solver.solve(
                        self.fully_reduced_rhs,
                        tol=self.tol_amg,
                        residuals=self.res_history_amg,
                    )

            # Compute flux update
            solution[self.flux_slice] = matrix_flux_inv.dot(
                rhs[self.flux_slice] + self.DT.dot(solution[self.reduced_system_slice])
            )
            time_solve = time.time() - tic

        stats = {
            "time setup": time_setup,
            "time solve": time_solve,
        }
        if self.linear_solver_type in ["amg-flux-reduced", "amg-potential"]:
            stats["amg residuals"] = self.res_history_amg
            stats["amg num iterations"] = len(self.res_history_amg)
            stats["amg residual"] = self.res_history_amg[-1]

        return solution, stats

    def remove_flux(self, jacobian: sps.csc_matrix, residual: np.ndarray) -> tuple:
        """Remove the flux block from the jacobian and residual.

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

    def remove_lagrange_multiplier(self, reduced_jacobian, reduced_residual) -> tuple:
        """Shortcut for removing the lagrange multiplier from the reduced jacobian.

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

    # ! ---- Main methods ----

    def __call__(
        self,
        img_1: darsia.Image,
        img_2: darsia.Image,
    ) -> float:
        """L1 Wasserstein distance for two images with same mass.

        NOTE: Images need to comply with the setup of the object.

        Args:
            img_1 (darsia.Image): image 1
            img_2 (darsia.Image): image 2

        Returns:
            float or array: distance between img_1 and img_2.

        """

        # Start taking time
        tic = time.time()

        # Compatibilty check
        assert img_1.scalar and img_2.scalar
        self._compatibility_check(img_1, img_2)

        # Determine difference of distriutions and define corresponding rhs
        mass_diff = img_1.img - img_2.img
        flat_mass_diff = np.ravel(mass_diff)

        # Main method
        distance, solution, status = self._solve(flat_mass_diff)

        # Split the solution
        flat_flux = solution[self.flux_slice]
        flat_potential = solution[self.potential_slice]

        # Reshape the fluxes and potential to grid format
        flux = darsia.face_to_cell(self.grid, flat_flux)
        potential = flat_potential.reshape(self.grid.shape)

        # Determine transport density
        transport_density = self.compute_transport_density(solution)

        # Stop taking time
        toc = time.time()
        status["elapsed_time"] = toc - tic
        print("Elapsed time: ", toc - tic)

        # Plot solution
        plot_solution = self.options.get("plot_solution", False)
        if plot_solution:
            self._plot_solution(mass_diff, flux, potential, transport_density)

        # Return solution
        return_solution = self.options.get("return_solution", False)
        if return_solution:
            return distance, flux, potential, transport_density, status
        else:
            return distance

    # TODO rm.
    def _plot_solution(
        self,
        mass_diff: np.ndarray,
        flux: np.ndarray,
        potential: np.ndarray,
        transport_density: np.ndarray,
    ) -> None:
        """Plot the solution.

        Args:
            mass_diff (np.ndarray): difference of mass distributions
            flux (np.ndarray): fluxes
            potential (np.ndarray): potential
            transport_density (np.ndarray): transport density

        Raises:
            NotImplementedError: plotting only implemented for 2D

        """

        if self.grid.dim != 2:
            raise NotImplementedError("Plotting only implemented for 2D.")

        # Fetch options
        plot_options = self.options.get("plot_options", {})
        name = plot_options.get("name", None)

        # Store plot
        save_plot = plot_options.get("save", False)
        if save_plot:
            folder = plot_options.get("folder", ".")
            Path(folder).mkdir(parents=True, exist_ok=True)
        show_plot = plot_options.get("show", True)

        # Control of flux arrows
        scaling = plot_options.get("scaling", 1.0)
        resolution = plot_options.get("resolution", 1)

        # Meshgrid
        Y, X = np.meshgrid(
            self.voxel_size[0] * (0.5 + np.arange(self.grid.shape[0] - 1, -1, -1)),
            self.voxel_size[1] * (0.5 + np.arange(self.grid.shape[1])),
            indexing="ij",
        )

        # Plot the potential
        plt.figure("Beckman solution potential")
        plt.pcolormesh(X, Y, potential, cmap="turbo")
        plt.colorbar(label="potential")
        plt.quiver(
            X[::resolution, ::resolution],
            Y[::resolution, ::resolution],
            scaling * flux[::resolution, ::resolution, 0],
            -scaling * flux[::resolution, ::resolution, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.25,
            width=0.005,
        )
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        # plt.ylim(top=0.08)  # TODO rm?
        if save_plot:
            plt.savefig(
                folder + "/" + name + "_beckman_solution_potential.png",
                dpi=500,
                transparent=True,
            )

        # Plot the fluxes
        plt.figure("Beckman solution fluxes")
        plt.pcolormesh(X, Y, mass_diff, cmap="turbo")  # , vmin=-1, vmax=3.5)
        plt.colorbar(label="mass difference")
        plt.quiver(
            X[::resolution, ::resolution],
            Y[::resolution, ::resolution],
            scaling * flux[::resolution, ::resolution, 0],
            -scaling * flux[::resolution, ::resolution, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.25,
            width=0.005,
        )
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        # plt.ylim(top=0.08)
        plt.text(
            0.0025,
            0.075,
            name,
            color="white",
            alpha=0.9,
            rotation=0,
            fontsize=14,
        )  # TODO rm?
        if save_plot:
            plt.savefig(
                folder + "/" + name + "_beckman_solution_fluxes.png",
                dpi=500,
                transparent=True,
            )

        # Plot the transport density
        plt.figure("L1 optimal transport density")
        plt.pcolormesh(X, Y, transport_density, cmap="turbo")
        plt.colorbar(label="flux modulus")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        # plt.ylim(top=0.08)  # TODO rm?
        if save_plot:
            plt.savefig(
                folder + "/" + name + "_beckman_solution_transport_density.png",
                dpi=500,
                transparent=True,
            )

        if show_plot:
            plt.show()
        else:
            plt.close("all")


class WassersteinDistanceNewton(VariationalWassersteinDistance):
    """Class to determine the L1 EMD/Wasserstein distance solved with Newton's method.

    Here, self.L has the interpretation of a lower cut-off value in the linearization
    only. With such relaxation, the BEckman problem itself is not regularized, but
    instead the solution trajectory is merely affected.

    """

    def __init__(self, grid, options) -> None:
        super().__init__(grid, options)

        self.L = self.options.get("L", 1.0)
        """float: relaxation parameter, lower cut-off for the mobility"""

    def residual(self, rhs: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """Compute the residual of the solution.

        Args:
            rhs (np.ndarray): right hand side
            solution (np.ndarray): solution

        Returns:
            np.ndarray: residual

        """
        flat_flux = solution[self.flux_slice]
        mode = self.options.get("mode", "face_arithmetic")
        flat_flux_norm = np.maximum(
            self.vector_face_flux_norm(flat_flux, mode=mode), self.regularization
        )
        flat_flux_normed = flat_flux / flat_flux_norm

        return (
            rhs
            - self.broken_darcy.dot(solution)
            - self.flux_embedding.dot(self.mass_matrix_faces.dot(flat_flux_normed))
        )

    def jacobian(self, solution: np.ndarray) -> sps.linalg.LinearOperator:
        """Compute the LU factorization of the jacobian of the solution.

        Args:
            solution (np.ndarray): solution

        Returns:
            sps.linalg.splu: LU factorization of the jacobian

        """
        flat_flux = solution[self.flux_slice]
        mode = self.options.get("mode", "face_arithmetic")
        flat_flux_norm = np.maximum(
            self.vector_face_flux_norm(flat_flux, mode=mode), self.regularization
        )
        approx_jacobian = sps.bmat(
            [
                [
                    sps.diags(np.maximum(self.L, 1.0 / flat_flux_norm), dtype=float)
                    * self.mass_matrix_faces,
                    -self.div.T,
                    None,
                ],
                [self.div, None, -self.potential_constraint.T],
                [None, self.potential_constraint, None],
            ],
            format="csc",
        )
        return approx_jacobian

    def _solve(self, flat_mass_diff: np.ndarray) -> tuple[float, np.ndarray, dict]:
        """Solve the Beckman problem using Newton's method.

        Args:
            flat_mass_diff (np.ndarray): difference of mass distributions

        Returns:
            tuple: distance, solution, status

        """
        # TODO rm: Observation: AA can lead to less stagnation, more accurate results,
        # and therefore better solutions to mu and u. Higher depth is better, but more
        # expensive.

        # Setup
        tic = time.time()

        # Solver parameters
        num_iter = self.options.get("num_iter", 100)
        tol_residual = self.options.get("tol_residual", 1e-6)
        tol_increment = self.options.get("tol_increment", 1e-6)
        tol_distance = self.options.get("tol_distance", 1e-6)

        # Define right hand side
        rhs = np.concatenate(
            [
                np.zeros(self.grid.num_faces, dtype=float),
                self.mass_matrix_cells.dot(flat_mass_diff),
                np.zeros(1, dtype=float),
            ]
        )

        # Initialize solution
        solution_i = np.zeros_like(rhs)

        # Initialize container for storing the convergence history
        convergence_history = {
            "distance": [],
            "residual": [],
            "decomposed residual": [],
            "increment": [],
            "decomposed increment": [],
            "distance increment": [],
            "timing": [],
        }

        # Print header
        if self.verbose:
            print(
                "--- ; ",
                "Newton iteration",
                "distance",
                "residual",
                "mass conservation residual",
                "increment",
                "flux increment",
                "distance increment",
            )

        # Newton iteration
        for iter in range(num_iter):
            # Keep track of old flux, and old distance
            old_solution_i = solution_i.copy()
            old_flux = solution_i[self.flux_slice]
            old_distance = self.l1_dissipation(old_flux, "cell_arithmetic")

            # Assemble linear problem in Newton step
            tic = time.time()
            if iter == 0:
                # Determine residual and (full) Jacobian of a linear Darcy problem
                residual_i = rhs.copy()
                approx_jacobian = self.darcy_init.copy()
            else:
                # Determine residual and (full) Jacobian
                residual_i = self.residual(rhs, solution_i)
                approx_jacobian = self.jacobian(solution_i)
            toc = time.time()
            time_assemble = toc - tic

            # Solve linear system for the update
            update_i, stats_i = self.linear_solve(
                approx_jacobian, residual_i, solution_i
            )

            # Diagnostics
            # TODO move?
            if self.linear_solver_type in ["amg-flux-reduced", "amg-potential"]:
                if self.options.get("linear_solver_verbosity", False):
                    # print(ml) # TODO rm?
                    print(
                        f"""AMG iterations: {stats_i["amg num iterations"]}; """
                        f"""Residual after AMG step: {stats_i["amg residual"]}"""
                    )

            # Update the solution with the full Netwon step
            solution_i += update_i

            # Apply Anderson acceleration to flux contribution (the only nonlinear part).
            # Application to full solution, or just the potential, lead to divergence,
            # while application to the flux, results in improved performance.
            tic = time.time()
            if self.anderson is not None:
                solution_i[self.flux_slice] = self.anderson(
                    solution_i[self.flux_slice],
                    update_i[self.flux_slice],
                    iter,
                )
            toc = time.time()
            time_anderson = toc - tic

            # Update stats
            stats_i["time assemble"] = time_assemble
            stats_i["time acceleration"] = time_anderson

            # Update distance
            new_flux = solution_i[self.flux_slice]
            new_distance = self.l1_dissipation(new_flux, "cell_arithmetic")

            # Compute the error:
            # - full residual
            # - residual of the flux equation
            # - residual of mass conservation equation
            # - residual of the constraint equation
            # - full increment
            # - flux increment
            # - potential increment
            # - lagrange multiplier increment
            # - distance increment
            increment = solution_i - old_solution_i
            error = [
                np.linalg.norm(residual_i, 2),
                [
                    np.linalg.norm(residual_i[self.flux_slice], 2),
                    np.linalg.norm(residual_i[self.potential_slice], 2),
                    np.linalg.norm(residual_i[self.lagrange_multiplier_slice], 2),
                ],
                np.linalg.norm(increment, 2),
                [
                    np.linalg.norm(increment[self.flux_slice], 2),
                    np.linalg.norm(increment[self.potential_slice], 2),
                    np.linalg.norm(increment[self.lagrange_multiplier_slice], 2),
                ],
                abs(new_distance - old_distance),
            ]

            # Update convergence history
            convergence_history["distance"].append(new_distance)
            convergence_history["residual"].append(error[0])
            convergence_history["decomposed residual"].append(error[1])
            convergence_history["increment"].append(error[2])
            convergence_history["decomposed increment"].append(error[3])
            convergence_history["distance increment"].append(error[4])
            convergence_history["timing"].append(stats_i)

            new_distance_faces = self.l1_dissipation(new_flux, "face_arithmetic")
            if self.verbose:
                print(
                    "Newton iteration",
                    iter,
                    new_distance,
                    new_distance_faces,
                    error[0],  # residual
                    error[1],  # mass conservation residual
                    error[2],  # full increment
                    error[3],  # flux increment
                    error[4],  # distance increment
                    stats_i,  # timing
                )

            # Stopping criterion
            # TODO include criterion build on staganation of the solution
            if iter > 1 and (
                (error[0] < tol_residual and error[2] < tol_increment)
                or error[4] < tol_distance  # TODO rm the latter
            ):
                break

        # Define performance metric
        status = {
            "converged": iter < num_iter,
            "number iterations": iter,
            "distance": new_distance,
            "residual": error[0],
            "mass conservation residual": error[1],
            "increment": error[2],
            "flux increment": error[3],
            "distance increment": abs(new_distance - old_distance),
            "convergence history": convergence_history,
        }

        return new_distance, solution_i, status


class WassersteinDistanceBregman(VariationalWassersteinDistance):
    # TODO __init__ correct method?
    def __init__(
        self,
        grid: darsia.Grid,
        options: dict = {},
    ) -> None:
        super().__init__(grid, options)
        self.L = self.options.get("L", 1.0)
        """Penality parameter for the Bregman iteration."""

        self.force_slice = slice(self.grid.num_faces, None)
        """slice: slice for the force."""

    def _shrink(
        self,
        flat_flux: np.ndarray,
        shrink_factor: Union[float, np.ndarray],
        mode: str = "cell_arithmetic",
    ) -> np.ndarray:
        """Shrink operation in the split Bregman method.

        Operation on fluxes.

        Args:
            flat_flux (np.ndarray): flux
            shrink_factor (float or np.ndarray): shrink factor
            mode (str, optional): mode of the shrink operation. Defaults to "cell_arithmetic".

        Returns:
            np.ndarray: shrunk fluxes

        """
        if mode == "cell_arithmetic":
            # Idea: Determine the shrink factor based on the cell reconstructions of the
            # fluxes. Convert cell-based shrink factors to face-based shrink factors
            # through arithmetic averaging.
            cell_flux = darsia.face_to_cell(self.grid, flat_flux)
            norm = np.linalg.norm(cell_flux, 2, axis=-1)
            cell_scaling = np.maximum(norm - shrink_factor, 0) / (
                norm + self.regularization
            )
            flat_scaling = darsia.cell_to_face(
                self.grid, cell_scaling, mode="arithmetic"
            )

        elif mode == "face_arithmetic":
            # Define natural vector valued flux on faces (taking arithmetic averages
            # of continuous fluxes over cells evaluated at faces)
            full_face_flux = self.face_reconstruction(flat_flux)
            # Determine the l2 norm of the fluxes on the faces, add some regularization
            norm = np.linalg.norm(full_face_flux, 2, axis=1)
            flat_scaling = np.maximum(norm - shrink_factor, 0) / (
                norm + self.regularization
            )

        elif mode == "face_normal":
            # Only consider normal direction (does not take into account the full flux)
            # TODO rm.
            norm = np.linalg.norm(flat_flux, 2, axis=-1)
            flat_scaling = np.maximum(norm - shrink_factor, 0) / (
                norm + self.regularization
            )

        else:
            raise NotImplementedError(f"Mode {mode} not supported.")

        return flat_scaling * flat_flux

    def _solve(self, flat_mass_diff):
        # Solver parameters
        num_iter = self.options.get("num_iter", 100)
        tol_residual = self.options.get("tol_residual", 1e-6)
        tol_increment = self.options.get("tol_increment", 1e-6)
        tol_distance = self.options.get("tol_distance", 1e-6)

        # Relaxation parameter
        self.L = self.options.get("L", 1.0)
        rhs = np.concatenate(
            [
                np.zeros(self.grid.num_faces, dtype=float),
                self.mass_matrix_cells.dot(flat_mass_diff),
                np.zeros(1, dtype=float),
            ]
        )

        # Keep track of how often the distance increases.
        num_neg_diff = 0

        # Initialize container for storing the convergence history
        convergence_history = {
            "distance": [],
            "mass residual": [],
            "force": [],
            "flux increment": [],
            "aux increment": [],
            "force increment": [],
            "distance increment": [],
            "timing": [],
        }

        # Print header
        if self.verbose:
            print(
                "--- ; ",
                "Bregman iteration",
                "L",
                "distance",
                "mass conservation residual",
                "[flux, aux, force] increment",
                "distance increment",
            )

        # Initialize Bregman variables and flux with Darcy flow
        shrink_mode = "face_arithmetic"
        dissipation_mode = "cell_arithmetic"
        weight = self.L
        shrink_factor = 1.0 / self.L

        # Solve linear Darcy problem as initial guess
        l_scheme_mixed_darcy = sps.bmat(
            [
                [self.L * self.mass_matrix_faces, -self.div.T, None],
                [self.div, None, -self.potential_constraint.T],
                [None, self.potential_constraint, None],
            ],
            format="csc",
        )
        solution_i = np.zeros_like(rhs, dtype=float)
        solution_i, _ = self.linear_solve(l_scheme_mixed_darcy, rhs, solution_i)

        # Extract intial values
        old_flux = solution_i[self.flux_slice]
        old_aux_flux = self._shrink(old_flux, shrink_factor, shrink_mode)
        old_force = old_flux - old_aux_flux
        old_distance = self.l1_dissipation(old_flux, dissipation_mode)

        for iter in range(num_iter):
            bregman_mode = self.options.get("bregman_mode", "standard")
            if bregman_mode == "standard":
                # std split Bregman method

                # 1. Make relaxation step (solve quadratic optimization problem)
                tic = time.time()
                rhs_i = rhs.copy()
                rhs_i[self.flux_slice] = self.L * self.mass_matrix_faces.dot(
                    old_aux_flux - old_force
                )
                solution_i, _ = self.linear_solve(
                    l_scheme_mixed_darcy, rhs_i, reuse_solver=True
                )
                new_flux = solution_i[self.flux_slice]
                time_linearization = time.time() - tic

                # 2. Shrink step for vectorial fluxes. To comply with the RT0 setting, the
                # shrinkage operation merely determines the scalar. We still aim at
                # following along the direction provided by the vectorial fluxes.
                tic = time.time()
                new_aux_flux = self._shrink(
                    new_flux + old_force, shrink_factor, shrink_mode
                )
                time_shrink = time.time() - tic

                # 3. Update force
                new_force = old_force + new_flux - new_aux_flux

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

                toc = time.time()
                time_anderson = toc - tic

            elif bregman_mode == "reordered":
                # Reordered split Bregman method

                # 1. Shrink step for vectorial fluxes. To comply with the RT0 setting, the
                # shrinkage operation merely determines the scalar. We still aim at
                # following along the direction provided by the vectorial fluxes.
                tic = time.time()
                new_aux_flux = self._shrink(
                    old_flux + old_force, shrink_factor, shrink_mode
                )
                time_shrink = time.time() - tic

                # 2. Update force
                new_force = old_force + old_flux - new_aux_flux

                # 3. Solve linear system with trust in current flux.
                tic = time.time()
                rhs_i = rhs.copy()
                rhs_i[self.flux_slice] = self.L * self.mass_matrix_faces.dot(
                    new_aux_flux - new_force
                )
                solution_i, _ = self.linear_solve(
                    l_scheme_mixed_darcy, rhs_i, reuse_solver=True
                )
                new_flux = solution_i[self.flux_slice]
                time_linearization = time.time() - tic

                # Apply Anderson acceleration to flux contribution (the only nonlinear part).
                tic = time.time()
                if self.anderson is not None:
                    flux_inc = new_flux - old_flux
                    force_inc = new_force - old_force
                    inc = np.concatenate([flux_inc, force_inc])
                    iteration = np.concatenate([new_flux, new_force])
                    # new_flux = self.anderson(new_flux, flux_inc, iter)
                    new_iteration = self.anderson(iteration, inc, iter)
                    new_flux = new_iteration[self.flux_slice]
                    new_force = new_iteration[self.force_slice]
                toc = time.time()
                time_anderson = toc - tic

            elif bregman_mode == "adaptive":
                # Bregman split with updated weight
                update_cond = self.options.get(
                    "bregman_update_cond", lambda iter: False
                )
                update_solver = update_cond(iter)
                if update_solver:
                    # TODO: self._update_weight(old_flux)

                    # Update weight as the inverse of the norm of the flux
                    old_flux_norm = np.maximum(
                        self.vector_face_flux_norm(old_flux, "face_arithmetic"),
                        self.regularization,
                    )
                    old_flux_norm_inv = 1.0 / old_flux_norm
                    weight = sps.diags(old_flux_norm_inv)
                    shrink_factor = old_flux_norm

                    # Redefine Darcy system
                    l_scheme_mixed_darcy = sps.bmat(
                        [
                            [weight * self.mass_matrix_faces, -self.div.T, None],
                            [self.div, None, -self.potential_constraint.T],
                            [None, self.potential_constraint, None],
                        ],
                        format="csc",
                    )

                # 1. Make relaxation step (solve quadratic optimization problem)
                tic = time.time()
                rhs_i = rhs.copy()
                rhs_i[self.flux_slice] = weight * self.mass_matrix_faces.dot(
                    old_aux_flux - old_force
                )
                solution_i, _ = self.linear_solve(
                    l_scheme_mixed_darcy, rhs_i, reuse_solver=not (update_solver)
                )

                # Diagnostics
                if self.linear_solver_type in ["amg-flux-reduced", "amg-potential"]:
                    if self.options.get("linear_solver_verbosity", False):
                        num_amg_iter = len(self.res_history_amg)
                        res_amg = self.res_history_amg[-1]
                        print(self.l_scheme_mixed_darcy_solver)
                        print(
                            f"""#AMG iterations: {num_amg_iter}; Residual after """
                            f"""AMG step: {res_amg}"""
                        )

                new_flux = solution_i[self.flux_slice]
                time_linearization = time.time() - tic

                # 2. Shrink step for vectorial fluxes. To comply with the RT0 setting, the
                # shrinkage operation merely determines the scalar. We still aim at
                # following along the direction provided by the vectorial fluxes.
                tic = time.time()
                new_aux_flux = self._shrink(
                    new_flux + old_force, shrink_factor, shrink_mode
                )
                time_shrink = time.time() - tic

                # 3. Update force
                new_force = old_force + new_flux - new_aux_flux

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

                toc = time.time()
                time_anderson = toc - tic

            else:
                raise NotImplementedError(f"Bregman mode {bregman_mode} not supported.")

            # Collect stats
            stats_i = [time_linearization, time_shrink, time_anderson]

            # Update distance
            new_distance = self.l1_dissipation(new_flux, dissipation_mode)

            # Determine the error in the mass conservation equation
            mass_conservation_residual = np.linalg.norm(
                self.div.dot(new_flux) - rhs[self.potential_slice], 2
            )

            # Determine increments
            flux_increment = new_flux - old_flux
            aux_increment = new_aux_flux - old_aux_flux
            force_increment = new_force - old_force

            # Determine force
            force = np.linalg.norm(new_force, 2)

            # Compute the error:
            # - residual of mass conservation equation - zero only if exact solver used
            # - force
            # - flux increment
            # - aux increment
            # - force increment
            # - distance increment
            error = [
                mass_conservation_residual,
                force,
                np.linalg.norm(flux_increment),
                np.linalg.norm(aux_increment),
                np.linalg.norm(force_increment),
                abs(new_distance - old_distance),
            ]

            # Update convergence history
            convergence_history["distance"].append(new_distance)
            convergence_history["mass residual"].append(error[0])
            convergence_history["force"].append(error[1])
            convergence_history["flux increment"].append(error[2])
            convergence_history["aux increment"].append(error[3])
            convergence_history["force increment"].append(error[4])
            convergence_history["distance increment"].append(error[5])
            convergence_history["timing"].append(stats_i)

            # Print status
            if self.verbose:
                print(
                    "Bregman iteration",
                    iter,
                    new_distance,
                    self.L,
                    error[0],  # mass conservation residual
                    [
                        error[2],  # flux increment
                        error[3],  # aux increment
                        error[4],  # force increment
                    ],
                    error[5],  # distance increment
                    # stats_i,  # timings
                )

            # Keep track if the distance increases.
            if new_distance > old_distance:
                num_neg_diff += 1

            # TODO include criterion build on staganation of the solution
            if iter > 1 and (
                (error[0] < tol_residual and error[4] < tol_increment)
                or error[5] < tol_distance
            ):
                break

            # TODO rm?
            # update_l = self.options.get("update_l", False)
            # if update_l:
            #     tol_distance = self.options.get("tol_distance", 1e-12)
            #     max_iter_increase_diff = self.options.get(
            #        "max_iter_increase_diff",
            #        20
            #     )
            #     l_factor = self.options.get("l_factor", 2)
            #     if (
            #         abs(new_distance - old_distance) < tol_distance
            #         or num_neg_diff > max_iter_increase_diff
            #     ):
            #         # Update L
            #         self.L = self.L * l_factor
            #
            #         # Update linear system
            #         l_scheme_mixed_darcy = sps.bmat(
            #             [
            #                 [
            #                    self.L * self.mass_matrix_faces,
            #                    -self.div.T,
            #                    None
            #                 ],
            #                 [self.div, None, -self.potential_constraint.T],
            #                 [None, self.potential_constraint, None],
            #             ],
            #             format="csc",
            #         )
            #         self.l_scheme_mixed_darcy_lu = (
            #            sps.linalg.splu(l_scheme_mixed_darcy)
            #         )
            #
            #         # Reset stagnation counter
            #         num_neg_diff = 0
            #
            #     L_max = self.options.get("L_max", 1e8)
            #     if self.L > L_max:
            #         break

            # Update Bregman variables
            old_flux = new_flux.copy()
            old_aux_flux = new_aux_flux.copy()
            old_force = new_force.copy()
            old_distance = new_distance

        # TODO solve for potential and multiplier
        solution_i = np.zeros_like(rhs)
        solution_i[self.flux_slice] = new_flux.copy()
        # TODO continue

        # Define performance metric
        status = {
            "converged": iter < num_iter,
            "number iterations": iter,
            "distance": new_distance,
            "mass conservation residual": error[0],
            "flux increment": error[2],
            "distance increment": abs(new_distance - old_distance),
            "convergence history": convergence_history,
        }

        return new_distance, solution_i, status


# Unified access
def wasserstein_distance(
    mass_1: darsia.Image,
    mass_2: darsia.Image,
    method: str,
    **kwargs,
):
    """Unified access to Wasserstein distance computation between images with same mass.

    Args:
        mass_1 (darsia.Image): image 1
        mass_2 (darsia.Image): image 2
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
            w1 = WassersteinDistanceNewton(grid, options)
        elif method.lower() == "bregman":
            w1 = WassersteinDistanceBregman(grid, options)

    elif method.lower() == "cv2.emd":
        # Use Earth Mover's Distance from CV2
        preprocess = kwargs.get("preprocess")
        w1 = darsia.EMD(preprocess)
        return w1(mass_1, mass_2)

    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    # Compute and return Wasserstein distance
    return w1(mass_1, mass_2)
