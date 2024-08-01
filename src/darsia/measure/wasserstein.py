"""Wasserstein distance computed using variational methods."""
from __future__ import annotations

import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyamg
import scipy.sparse as sps
from scipy.stats import hmean
from petsc4py import PETSc
from ot import dist
from ot.bregman import sinkhorn2
from ot.bregman import empirical_sinkhorn2
import darsia
import copy

# General TODO list
# - improve assembling of operators through partial assembling
# - allow to reuse setup.



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
        options: dict = {},
    ) -> None:
        """Initialization of the variational Wasserstein distance.

        Args:

            grid (darsia.Grid): tensor grid associated with the images
            options (dict): options for the nonlinear and linear solver. The following
                options are supported:
                - l1_mode (str): mode for computing the l1 dissipation. Defaults to
                    "raviart_thomas". Supported modes are:
                    - "raviart_thomas": Apply exact integration of RT0 extensions into
                        cells. Underlying functional for mixed finite element method
                        (MFEM).
                    - "constant_subcell_projection": Apply subcell_based projection onto
                        constant vectors and sum up. Equivalent to a mixed finite volume
                        method (FV).
                    - "constant_cell_projection": Apply cell-based L2 projection onto
                        constant vectors and sum up. Simpler calculation than
                        subcell-projection, but not directly connected to any
                        discretization.
                - mobility_mode (str): mode for computing the mobility. Defaults to
                    "face_based". Supported modes are:
                    - "cell_based": Cell-based mode determines the norm of the fluxes on
                        the faces via averaging of neighboring cells.
                    - "cell_based_arithmetic": Cell-based mode determines the norm of
                        the fluxes on the faces via arithmetic averaging of neighboring
                        cells.
                    - "cell_based_harmonic": Cell-based mode determines the norm of the
                        fluxes on the faces via harmonic averaging of neighboring cells.
                    - "subcell_based": Subcell-based mode determines the norm of the
                        fluxes on the faces via averaging of neighboring subcells.
                    - "face_based": Face-based mode determines the norm of the fluxes on
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
                    - "ksp": Krylov subspace solver by PETSc
                - formulation (str): formulation of the linear system. Defaults to
                    "pressure". Supported formulations are:
                    - "full": full system
                    - "flux_reduced": reduced system with fluxes eliminated
                    - "pressure": reduced system with fluxes and lagrange multiplier
                        eliminated
                - linear_solver_options (dict): options for the linear solver. Defaults
                    to {}.
                - ksp_options (dict): options for the KSP solver. Defaults to {}.
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

        self.l1_mode = self.options.get("l1_mode", "raviart_thomas")
        """str: mode for computing the l1 dissipation"""

        self.mobility_mode = self.options.get("mobility_mode", "cell_based")
        """str: mode for computing the mobility"""

        
        # Setup of method
        self._setup_dof_management()
        self._setup_discretization()
        self._setup_linear_solver()
        self._setup_acceleration()

        self.kappa = self.options.get("kappa", np.ones(self.grid.shape,dtype=float))
        """np.ndarray: kappa"""
        self.kappa_faces = darsia.cell_to_face_average(self.grid, self.kappa, mode="harmonic")
        """np.ndarray: kappa on faces"""

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
        num_dofs = num_flux_dofs + num_pressure_dofs

        # ! ---- Indices in global system ----
        self.flux_indices = np.arange(num_flux_dofs, dtype='int32')
        """np.ndarray: indices of the fluxes"""

        self.pressure_indices = np.arange(
            num_flux_dofs, num_flux_dofs + num_pressure_dofs, dtype='int32'
        )
        """np.ndarray: indices of the pressures"""

        self.lagrange_multiplier_indices = np.array(
            [num_flux_dofs + num_pressure_dofs], dtype=int
        )
        """np.ndarray: indices of the lagrange multiplier"""

        # ! ---- Fast access to components through slices ----
        self.flux_slice = slice(0, num_flux_dofs)
        """slice: slice for the fluxes"""

        self.pressure_slice = slice(num_flux_dofs, num_flux_dofs + num_pressure_dofs)
        """slice: slice for the pressures"""

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

        self.flux_is = PETSc.IS()
        self.pressure_is = PETSc.IS()
        # TODO: choose between createGeneral and createStride
        #self.flux_is.createGeneral(self.flux_indices,comm=PETSc.COMM_WORLD)
        #self.pressure_is.createGeneral(self.pressure_indices,comm=PETSc.COMM_WORLD)
        
        self.flux_is.createStride(size=num_flux_dofs, first=0, step=1, comm=PETSc.COMM_WORLD) 
        self.pressure_is.createStride(size=num_pressure_dofs, first=num_flux_dofs , step=1, comm=PETSc.COMM_WORLD)

        self.field_ises = [('0',self.flux_is), ('1',self.pressure_is)]
        """ PETSc IS: index sets for the fields"""

    def _setup_discretization(self) -> None:
        """Setup of fixed discretization operators."""

        # ! ---- Constraint for the pressure correpsonding to Lagrange multiplier ----

        center_cell = np.array(self.grid.shape) // 2
        self.constrained_cell_flat_index = np.ravel_multi_index(
            center_cell, self.grid.shape
        )
        """int: flat index of the cell where the pressure is constrained to zero"""

        # ! ---- Discretization operators ----

        self.div = darsia.FVDivergence(self.grid).mat
        """sps.csc_matrix: divergence operator: flat fluxes -> flat pressures"""

        self.mass_matrix_cells = darsia.FVMass(self.grid).mat
        """sps.csc_matrix: mass matrix on cells: flat pressures -> flat pressures"""

        lumping = self.options.get("lumping", True)
        self.mass_matrix_faces = darsia.FVMass(self.grid, "faces", lumping).mat
        """sps.csc_matrix: mass matrix on faces: flat fluxes -> flat fluxes"""

        L_init = self.options.get("L_init", 1.0)
        self.darcy_init = sps.bmat(
            [
                [L_init * self.mass_matrix_faces, -self.div.T],
                [self.div, None],
            ],
            format="csc",
        )
        """sps.csc_matrix: initial Darcy operator"""

        self.broken_darcy = sps.bmat(
            [
                [None, -self.div.T],
                [self.div, None],
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
            "ksp",
        ], f"Linear solver {self.linear_solver_type} not supported."
        assert self.formulation in [
            "full",
            "flux-reduced",
            "pressure",
        ], f"Formulation {self.formulation} not supported."

    
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

    def transport_density(
        self, flat_flux: np.ndarray, flatten: bool = True
    ) -> np.ndarray:
        """Compute the transport density from the solution.

        Args:
            flat_flux (np.ndarray): face fluxes
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

        if self.l1_mode == "raviart_thomas":
            # Apply numerical integration of RT0 extensions into cells.
            # Underlying functional for mixed finite element method (MFEM).
            quad_pts, quad_weights = darsia.quadrature.gauss_reference_cell(
                self.grid.dim, "max"
            )

        elif self.l1_mode == "constant_subcell_projection":
            # Apply subcell_based projection onto constant vectors and sum up.
            # Equivalent to a mixed finite volume method (FV). Identical to quadrature
            # over corners.
            quad_pts, quad_weights = darsia.quadrature.reference_cell_corners(
                self.grid.dim
            )

        elif self.l1_mode == "constant_cell_projection":
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

    def vector_face_flux_norm(self, flat_flux: np.ndarray, mode: str) -> np.ndarray:
        """Compute the norm of the vector-valued fluxes on the faces.

        Args:
            flat_flux (np.ndarray): flat fluxes (normal fluxes on the faces)
            mode (str): mode of the norm, either "cell_based", "cell_based_harmonic" (same
                as "cell_based", "cell_based_arithmetic", "face_based", or "subcell_based".

        In the cell-based modes, the fluxes are projected to the cells and the norm across
        faces is computed via averaging. Similar for subcell_based definition. In the
        face-based mode, the norm is computed directly on the faces; for this fluxes are
        constructed with knowledge of RT0 basis functions.

        Returns:
            np.ndarray: flat norm of the vector-valued fluxes on the faces

        """

        # Determine the norm of the fluxes on the faces
        if mode in ["cell_based", "cell_based_arithmetic", "cell_based_harmonic"]:
            # Cell-based mode determines the norm of the fluxes on the faces via
            # averaging of neighboring cells.

            # Extract average mode from mode
            if mode == "cell_based":
                average_mode = "harmonic"
            else:
                average_mode = mode.split("_")[2]

            # The flux norm is identical to the transport density
            cell_flux_norm = self.transport_density(flat_flux, flatten=False)

            # Map to faces via averaging of neighboring cells
            flat_flux_norm = darsia.cell_to_face_average(
                self.grid, cell_flux_norm, mode=average_mode
            )

        elif mode == "subcell_based":
            # Subcell-based mode determines the norm of the fluxes on the faces via
            # averaging of neighboring subcells.

            # Initialize the flux norm
            num_subcells = 2**self.grid.dim
            subcell_flux_norm = np.zeros(
                (self.grid.num_faces, num_subcells), dtype=float
            )
            flat_flux_norm = np.zeros(self.grid.num_faces, dtype=float)

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
                        # Store the norm of the subcell flux from the cell associated to
                        # the flux
                        id = i * len(coordinates) + j
                        subcell_flux_norm[faces, id] = np.linalg.norm(
                            subcell_flux, 2, axis=-1
                        ).ravel("F")[cells]

            # Average over the subcells using harmonic averaging
            flat_flux_norm = hmean(subcell_flux_norm, axis=1)

        elif mode == "face_based":
            if not hasattr(self, "face_reconstruction"):
                self._setup_face_reconstruction()

            # Define natural vector valued flux on faces (taking arithmetic averages
            # of continuous fluxes over cells evaluated at faces)
            full_face_flux = self.face_reconstruction(flat_flux)

            # Determine the l2 norm of the fluxes on the faces
            flat_flux_norm = np.linalg.norm(full_face_flux, 2, axis=1)

        else:
            raise ValueError(f"Mode {mode} not supported.")

        # Scale by kappa
        flat_flux_norm /= self.kappa_faces


        return flat_flux_norm

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
        flat_flux_norm = np.maximum(
            self.vector_face_flux_norm(flat_flux, mode=self.mobility_mode),
            self.regularization,
        )
        flat_flux_normed = flat_flux / flat_flux_norm

        return (
            rhs
            - self.broken_darcy.dot(solution)
            - self.flux_embedding.dot(self.mass_matrix_faces.dot(flat_flux_normed))
        )

    # ! ---- Solver methods ----

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

        # Free memory if solver needs to be re-setup
        if not (reuse_solver) and hasattr(self, "linear_solver"):
            self.linear_solver.kill()
        
        tic = time.time() 
        if setup_linear_solver:
            # Define CG solver
            kernel = np.zeros(matrix.shape[0], dtype=float)
            kernel[self.pressure_indices] = 1.0
            # normalize the kernel
            kernel = kernel / np.linalg.norm(kernel)
            self.linear_solver = darsia.linalg.KSP(matrix, 
                                                   field_ises=self.field_ises,
                                                   nullspace=[kernel],
                                                    appctx={
                                                       "regularized_flat_flux_norm": self.regularized_flat_flux_norm,
                                                       "div": self.div,
                                                   })

            # Define solver options
            linear_solver_options = self.options.get("linear_solver_options", {})
            tol = linear_solver_options.get("tol", 1e-6)
            maxiter = linear_solver_options.get("maxiter", 100)
            if self.formulation == "full":
                if self.linear_solver_type=="direct":
                    self.solver_options = {
                        "ksp_type": "preonly", # do not apply Krylov iterations
                        "pc_type": "lu", 
                        "pc_factor_shift_type": "inblocks", # for the zero entries 
                        "pc_factor_mat_solver_type" : "mumps"
                    }
                else:
                    self.solver_options = {
                        "ksp_type": "gmres",
                        "ksp_rtol": tol,
                        "ksp_maxit": maxiter,
                        # "ksp_monitor_true_residual": None, #this is for debugging
                        "pc_type": "fieldsplit",
                        "pc_fieldsplit_type":"schur",
                        "pc_fieldsplit_schur_fact_type": "full",
                        # use a full factorization of the Schur complement
                        # other options are "diag","lower","upper"
                        "pc_fieldsplit_schur_precondition": "selfp", 
                        # selfp -> form an approximate Schur complement 
                        # using S=-B diag(A)^{-1} B^T
                        # which is what we want
                        # https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/
                        "fieldsplit_0_ksp_type":"preonly",
                        "fieldsplit_0_pc_type":"jacobi",
                        "fieldsplit_1_ksp_type":"preonly",
                        "fieldsplit_1_pc_type": "hypre",
                        # User defined preconditioner
                        #"fieldsplit_1_pc_type": "python",
                        #"fieldsplit_1_pc_python_type": __name__+".SchurComplementPC",
                    }
            if self.formulation == "flux_reduced" or self.formulation == "pressure":
                # example of nested dictionary
                # the underscore is used to separate the nested dictionary
                # see in linals. The flatten_parameters will transform
                # the nested dictionary into the commented
                if self.linear_solver_type=="direct":
                    ksp_schur = {
                        "ksp": "preonly",
                        "pc_type": "lu", 
                        "pc_factor_mat_solver_type" : "mumps",
                    }
                else:
                    ksp_schur = {
                        "ksp": {
                            "type": "gmres",
                            "rtol": tol,
                            "maxit": maxiter
                        },
                        "pc_type": "hypre",
                    }
                
                self.solver_options = {
                    "ksp_type": "preonly", # prec. only and solve at the schur complement
                    "pc_type": "fieldsplit",
                    "pc_fieldsplit_type": "schur",
                    "pc_fieldsplit_schur_fact_type": "full",
                    "pc_fieldsplit_schur_precondition": "selfp", 
                    # selfp -> form an approximate Schur complement 
                    # using S=-B diag(A)^{-1} B^T
                    # which is what we want
                    "fieldsplit_0_ksp_type": "preonly",
                    "fieldsplit_0_pc_type": "jacobi",
                    # schur solver
                    "fieldsplit_1" : ksp_schur,
                }
            #self.solver_options["ksp_monitor"] = None
            self.linear_solver.setup(self.solver_options)
            #self.linear_solver.ksp.view()
        time_setup = time.time() - tic           
        
        # Solve the full system
        tic = time.time()
        solution = self.linear_solver.solve(rhs)
        time_solve = time.time() - tic

        # Define solver statistics
        stats = {
            "time_setup": time_setup,
            "time_solve": time_solve,
        }
        
        return solution, stats

    
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

        # Return solution
        return_info = self.options.get("return_info", False)
        if return_info:
            info.update(
                {
                    "grid": self.grid,
                    "mass_diff": mass_diff,
                    "flux": flux,
                    "pressure": pressure,
                    "transport_density": transport_density,
                    "src": img_1,
                    "dst": img_2,
                }
            )
            return distance, info
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

    def __init__(self, grid, options) -> None:
        super().__init__(grid, options)

        self.L = self.options.get("L", np.finfo(float).max)
        """float: relaxation/cut-off parameter for mobility, deactivated by default"""

    def _setup_discretization(self) -> None:
        """Setup of fixed discretization operators.

        Add linear contribution of the optimality conditions of the Newton linearization.

        """
        super()._setup_discretization()

        self.broken_darcy = sps.bmat(
            [
                [None, -self.div.T],
                [self.div, None,],
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
        flat_flux_norm = self.vector_face_flux_norm(flat_flux, mode=self.mobility_mode)
        regularized_flat_flux_norm = np.clip(
            flat_flux_norm, self.regularization, self.L
        )
        self.regularized_flat_flux_norm = regularized_flat_flux_norm
        approx_jacobian = sps.bmat(
            [
                [
                    sps.diags(
                        1.0 / regularized_flat_flux_norm,
                        dtype=float,
                    )
                    * self.mass_matrix_faces,
                    -self.div.T,
                ],
                [self.div, None],
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
            ]
        )
        self.regularized_flat_flux_norm = np.ones(self.grid.num_faces, dtype=float)

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
                "Newton iter. \t| W^1 \t\t| Δ W^1 \t| Δ flux \t| residual",
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
                old_flux = solution_i[self.flux_slice]
                old_distance = self.l1_dissipation(old_flux)

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
                new_flux = solution_i[self.flux_slice]
                new_distance = self.l1_dissipation(new_flux)

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
                    distance_increment = convergence_history["distance_increment"][-1]
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
            "converged": iter < num_iter,
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
        options: dict = {},
    ) -> None:
        """Initialize the Bregman method.

        Args:
            grid (darsia.Grid): grid
            options (dict, optional): options. Defaults to {}.

        """
        super().__init__(grid, options)
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
        mode: str = "cell_based",
    ) -> np.ndarray:
        """Shrink operation in the split Bregman method, operating on fluxes.

        To comply with the RT0 setting, the shrinkage operation merely determines the
        scalar. We still aim at following along the direction provided by the vectorial
        fluxes.

        Args:
            flat_flux (np.ndarray): flux
            shrink_factor (float or np.ndarray): shrink factor
            mode (str, optional): mode of the shrink operation. Defaults to
                "cell_based".

        Returns:
            np.ndarray: shrunk fluxes

        """
        vector_face_flux_norm = self.vector_face_flux_norm(flat_flux, mode=mode)
        flat_scaling = np.maximum(vector_face_flux_norm - shrink_factor, 0) / (
            vector_face_flux_norm + self.regularization
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

        # Add regularization to the norm of the flux
        flux_norm = np.maximum(
            self.vector_face_flux_norm(flat_flux, self.mobility_mode),
            self.regularization,
        )
        # Pick the max value if homogeneous regularization is desired
        if homogeneous:
            flux_norm[:] = np.max(flux_norm)
        # Assign the weight and shrink factor
        weight = sps.diags(1.0 / flux_norm)
        shrink_factor = flux_norm

        # Update the Darcy system
        l_scheme_mixed_darcy = sps.bmat(
            [
                [weight * self.mass_matrix_faces, -self.div.T],
                [self.div, None,],
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
            ]
        )
        self.regularized_flat_flux_norm = np.ones(self.grid.num_faces, dtype=float)

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
                "Bregman iter. \t| W^1 \t\t| Δ W^1 \t| Δ aux/force \t| mass residual",
                "\n",
                """---------------|---------------|---------------|---------------|"""
                """---------------""",
            )

        # Relaxation parameter entering Bregman regularization
        self.L = self.options.get("L", 1.0)
        weight = 1.0 / self.L
        shrink_factor = self.L

        # Initialize linear problem corresponding to Bregman regularization
        l_scheme_mixed_darcy = sps.bmat(
            [
                [weight * self.mass_matrix_faces, -self.div.T],
                [self.div, None],
            ],
            format="csc",
        )

        # Initialize Bregman variables
        old_flux = solution_i[self.flux_slice]
        old_aux_flux = self._shrink(old_flux, shrink_factor, self.mobility_mode)
        old_force = old_flux - old_aux_flux
        old_distance = self.l1_dissipation(old_flux)

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
                    ) = self._update_regularization(old_flux, bregman_homogeneous)
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
                    new_flux = solution_i[self.flux_slice]
                    stats_i["time_solve"] = time.time() - tic
                    stats_i["time_assemble"] = time_assemble

                    # 2. Shrink step for vectorial fluxes.
                    tic = time.time()
                    new_aux_flux = self._shrink(
                        new_flux, shrink_factor, self.mobility_mode
                    )
                    stats_i["time_shrink"] = time.time() - tic

                    # 3. Update force
                    new_force = new_flux - new_aux_flux

                else:
                    # 1. Make relaxation step (solve quadratic optimization problem)
                    tic = time.time()
                    rhs_i = rhs.copy()
                    rhs_i[self.flux_slice] = weight * self.mass_matrix_faces.dot(
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
                    new_flux = solution_i[self.flux_slice]
                    stats_i["time_solve"] = time.time() - tic
                    stats_i["time_assemble"] = time_assemble

                    # 2. Shrink step for vectorial fluxes.
                    tic = time.time()
                    new_aux_flux = self._shrink(
                        new_flux + old_force, shrink_factor, self.mobility_mode
                    )
                    stats_i["time_shrink"] = time.time() - tic

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
                stats_i["time_acceleration"] = time.time() - tic

                # Update distance
                new_distance = self.l1_dissipation(new_flux)

                # Determine the error in the mass conservation equation
                mass_conservation_residual = (
                    self.div.dot(new_flux) - rhs[self.pressure_slice]
                )

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
                convergence_history["aux_force_increment"].append(
                    np.linalg.norm(np.concatenate([aux_increment, force_increment]), 2)
                )
                convergence_history["distance_increment"].append(
                    abs(distance_increment)
                )
                convergence_history["mass_conservation_residual"].append(
                    np.linalg.norm(mass_conservation_residual, 2)
                )
                convergence_history["timing"].append(stats_i)

                # Extract current total run time
                current_run_time = self._analyze_timings(convergence_history["timing"])[
                    "total"
                ]
                convergence_history["run_time"].append(current_run_time)

                # Print status
                if self.verbose:
                    distance_increment = convergence_history["distance_increment"][-1]
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
                            < tol_distance
                            and convergence_history["mass_conservation_residual"][-1]
                            < tol_residual
                        )
                    ):
                        break

                # Update Bregman variables
                old_flux = new_flux.copy()
                old_aux_flux = new_aux_flux.copy()
                old_force = new_force.copy()
                old_distance = new_distance

            except Exception:
                warnings.warn("Bregman iteration abruptly stopped due to some error.")
                break

        # Solve for the pressure by solving a single Newton iteration
        newton_jacobian, _, _ = self._update_regularization(new_flux)
        solution_i = np.zeros_like(rhs)
        solution_i[self.flux_slice] = new_flux.copy()
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
            "converged": iter < num_iter,
            "number_iterations": iter,
            "convergence_history": convergence_history,
            "timings": total_timings,
            "peak_memory_consumption": peak_memory_consumption,
        }

        return new_distance, solution_i, info
    
class SchurComplementPC(object):
    """
    This is a test for building my own preconditioner,
    getting the extra info from the dictionary appctx passed
    to the linear solver. 
    We are trying to replate what is done in firedrake.
    """
    def setUp(self,pc):
        # get info from the parent KSP object
        appctx = pc.getAttr("appctx")
        flux_norm = appctx["regularized_flat_flux_norm"]
        div = appctx["div"] # this should be also obtained by the matrix
        S = div * sps.diags(flux_norm, dtype=float) * div.T
        
        self.ksp = darsia.linalg.KSP(S)
        self.ksp.setup({"ksp_type":"preonly","pc_type":"hypre"})

    def apply(self, pc, x, y):
        self.ksp.ksp.solve(x,y)


class WassersteinDistanceSinkhorn(darsia.EMD):
    """
    Class based on the Sinkhorn algorithm to compute the Wasserstein distance
    """
    def __init__(self, 
                grid: darsia.Grid,
                options: dict = {}):
        
        # Cache geometrical infos
        self.grid = grid
        """darsia.Grid: grid"""

        self.voxel_size = grid.voxel_size
        """np.ndarray: voxel size"""

        # Cache solver options
        self.options = options
        """dict: options for the solver"""

        self.sinkhorn_regularization = self.options.get("sinkhorn_regularization", 1e-1)
        """float: regularization parameter"""

        self.sinkhorn_algorithm = options.get("sinkhorn_algorithm", "sinkhorn")
        """ str: algorithm to use for the Sinkhorn algorithm"""       
        """ Available algorithms are:
        sinkhorn,sinkhorn_log, greenkhorn, sinkhorn_stabilized, sinkhorn_epsilon_scaling,
        """

        self.num_iter = self.options.get("num_iter", 100)
        """ int: max number of iterations"""

        self.only_non_zeros = options.get("only_non_zeros", True)
        """ bool: consider only non-zero pixels"""
        
        self.verbose = options.get("verbose", True)
        """ bool: verbosity"""

        self.store_cost_matrix = options.get("store_cost_matrix", False)
        """ bool: store the cost matrix"""
        if self.store_cost_matrix:
            self.M = dist(self.grid.cell_centers, self.grid.cell_centers, metric='euclidean')


        self.geomloss_scaling = self.options.get("geomloss_scaling", 0.5)
        """float: scaling factor for eps"""

        # TODO: rewrite
        N = self.grid.shape
        x_i = []
        for i, N in enumerate(self.grid.shape):
            x_i.append(np.arange(N)*self.grid.voxel_size[i]+ self.grid.voxel_size[i]/2)

        self.cc_xyz = np.meshgrid(*x_i, indexing="ij")
        self.cc = np.vstack([c.ravel() for c in self.cc_xyz]).T


    def support(self, img: darsia.Image) -> np.ndarray:
        """
        Return the indices of the non-zero pixels in the image.

        Args:
            img (darsia.Image): image

        Returns: 
            np.ndarray: support

        """
        # flatten the image
        img_flat = img.img.ravel()

        # return the indices of the non-zero pixels
        return np.where(img_flat > 0)
    
    def interpolate_kantorovich_potentials(self, 
                            support_1, img1, pot_1,
                            support_2, img2, pot_2) -> (np.ndarray, np.ndarray): 
        """
        When we work only on the "support" of the images,
        we need to extend the kantorovich potentials to the whole domain.
        We can do it using the entropic interpolation 
        (see Eq 3.7 in https://hal.science/hal-02539799/document).
        It should given the same result of the Sinkhorn algorithm
        passing the whole images.
        """

        coord_support_1 = self.cc[support_1]
        coord_support_2 = self.cc[support_2]

        pot_1_whole = np.zeros(self.cc.shape[0])
        for i, coord_center in enumerate(self.cc):
            M_i = dist(coord_center, coord_support_2, metric='euclidean')
            pot_1_whole[i] = -self.sinkhorn_regularization * np.log(
                np.dot(
                    np.exp(pot_2[i]- M_i )/self.sinkhorn_regularization),
                    img2)
            
        pot_2_whole = np.zeros(self.cc.shape[0])
        for i, coord_center in enumerate(self.cc):
            M_i = dist(coord_center, coord_support_1, metric='euclidean')
            pot_2_whole[i] = -self.sinkhorn_regularization * np.log(
                np.dot(
                    np.exp(pot_1 - M_i )/self.sinkhorn_regularization),
                    img1)
            
        return pot_1_whole, pot_2_whole

    def img2pythorch(self, img: darsia.Image) -> torch.Tensor:
        """
        Convert a darsia image to a pythorch tensor suitable for geomloss.sinkhorn_image
        that that takes tensor with (nimages, nchannels, *spatial dimensions)
        """
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
         
        return torch.from_numpy(img.img).type(dtype).view(1,1,*img.img.shape)

        
    def __call__(self, img_1: darsia.Image, img_2: darsia.Image) -> float:
        """
        Earth mover's distance between images with same total sum.

        Args:
            img_1 (darsia.Image): image 1
            img_2 (darsia.Image): image 2

        Returns:
            float or array: distance between img_1 and img_2.

        """
        # FIXME investigation required regarding resize preprocessing...
        # Preprocess images
        preprocessed_img_1 = img_1
        preprocessed_img_2 = img_2

        # Compatibilty check
        self._compatibility_check(preprocessed_img_1, preprocessed_img_2)

        # Only non-zero pixels are considered
        if self.only_non_zeros:
            support_1 = np.where(preprocessed_img_1.img.flatten("F")>0)
            support_2 = np.where(preprocessed_img_2.img.flatten("F")>0)

            non_zero_img1 = preprocessed_img_1.img.flatten("F")[support_1]*np.prod(self.voxel_size)
            non_zero_img2 = preprocessed_img_2.img.flatten("F")[support_2]*np.prod(self.voxel_size)
            
            coord_support_1 = self.cc[support_1]
            coord_support_2 = self.cc[support_2]
        else:
            non_zero_img1 = preprocessed_img_1.img.flatten("F") * np.prod(self.voxel_size)
            non_zero_img2 = preprocessed_img_2.img.flatten("F") * np.prod(self.voxel_size)
            coord_support_1 = self.cc
            coord_support_2 = self.cc


        # Compute the distance
        if "empirical" in self.sinkhorn_algorithm:
            distance, log = empirical_sinkhorn2(
                coord_support_1, # coordinate of non-zero pixels in image 1
                coord_support_2, # coordinate of non-zero pixels in image 2
                reg=self.sinkhorn_regularization,
                a=non_zero_img1,
                b=non_zero_img2, 
                metric='euclidean', # distance metric 
                numIterMax=self.num_iter,
                isLazy=False,  #boolean, 
                # If True, then only calculate the cost matrix by block and return
                # the dual potentials only (to save memory). If False, calculate full
                #cost matrix and return outputs of sinkhorn function.
                verbose=self.verbose, 
                log=True, # return ierr and log
                )
            self.niter = log['niter']
            self.kantorovich_potential_source = log['u']
            self.kantorovich_potential_target = log['v']
        elif "geomloss_sinkhorn_images" == self.sinkhorn_algorithm:
            raise NotImplementedError("geomloss_sinkhorn_images not implemented yet")
            
            # package to be imported
            from geomloss.sinkhorn_images import sinkhorn_divergence
            import torch
            use_cuda = torch.cuda.is_available()
            
            # convert the images to pythorch tensors
            torch_img1 = self.img2pythorch(preprocessed_img_1)
            torch_img2 = self.img2pythorch(preprocessed_img_2)
            
            distance, log = sinkhorn_divergence(
                torch_img1,
                torch_img2,
                p=1,
                blur=None,
                reach=None,
                axes=self.grid.voxel_size*self.grid.shape,
                scaling=0.5,
                cost=None,
                debias=True,
                potentials=True,
                verbose=True,
                )
            # sinkhorn and sinkhorn_log work with the potentials    
            self.niter = log['niter']
            self.kantorovich_potential_source = log['u']
            self.kantorovich_potential_target = log['v']
        
        elif "geomloss_sinkhorn_samples" in self.sinkhorn_algorithm:
            #
            # Explicit sparsity of images 
            # 
            from geomloss.sinkhorn_divergence import epsilon_schedule
            from geomloss.sinkhorn_samples import sinkhorn_tensorized
            import torch
            
            use_cuda = torch.cuda.is_available()
            dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
            
            # first 1 in view() is for passing the batch size
            point_img1 = torch.from_numpy(non_zero_img1).type(dtype).view(1,len(non_zero_img1))
            point_img2 = torch.from_numpy(non_zero_img2).type(dtype).view(1,len(non_zero_img2))
            x = torch.from_numpy(coord_support_1).type(dtype).view(1,*coord_support_1.shape)
            y = torch.from_numpy(coord_support_2).type(dtype).view(1,*coord_support_2.shape)
            
            diameter = np.linalg.norm(self.grid.voxel_size*self.grid.shape)
            blur = self.sinkhorn_regularization
            scaling = self.geomloss_scaling
            p_exponent = 1
            f_torch, g_torch = sinkhorn_tensorized(
                point_img1,#a 
                x,#x
                point_img2, #b
                y,#y
                p=p_exponent,
                blur=blur, #blur
                reach=None,
                diameter=diameter, #diameter
                scaling=0.5, # reduction of the regularization
                cost=None,
                debias=False,
                potentials=True,
            )

            f = f_torch.detach().cpu().numpy().flatten()
            g = g_torch.detach().cpu().numpy().flatten()
            distance = np.dot(f, non_zero_img1) + np.dot(g, non_zero_img2) 
            print(f"distance: {distance=} {f.shape=} {g.shape=}")


            # it is not clear to me how to get the number of iterations
            # what about the residual of the marginals?
            self.niter = len(epsilon_schedule(p_exponent, diameter, blur, scaling)) 
            self.kantorovich_potential_source = f
            self.kantorovich_potential_target = g


        else:
            M = dist(coord_support_1, coord_support_2, metric='euclidean')
            distance, log = sinkhorn2(
                non_zero_img1, non_zero_img2, M, 
                self.sinkhorn_regularization,
                method=self.sinkhorn_algorithm,
                numItermax=self.options["num_iter"],
                stopThr=1e-8,#self.options["tol_residual"],
                verbose=self.verbose,
                log=True)

            
            if self.sinkhorn_algorithm == 'sinkhorn_stabilized':
                self.niter = log['n_iter']
                self.kantorovich_potential_source = np.exp(log['logu'])
                self.kantorovich_potential_target = np.exp(log['logv'])
            else:
                # sinkhorn and sinkhorn_log work with the potentials    
                self.niter = log['niter']
                self.kantorovich_potential_source = log['u']
                self.kantorovich_potential_target = log['v']
            

        info = {
            "converged" : True,
            "niter": self.niter,
            "kantorovich_potential_source": self.kantorovich_potential_source,
            "kantorovich_potential_target": self.kantorovich_potential_target,
        }

        return distance, info

class WassersteinDistanceDMK(darsia.EMD):
    """
    This contains the implementation of the GproxPDHG algorithm
    described in "SOLVING LARGE-SCALE OPTIMIZATION PROBLEMS WITH
    A CONVERGENCE RATE INDEPENDENT OF GRID SIZE"
    """

    def __init__(self, grid, options) -> None:
        # Cache geometrical infos
        self.grid = grid
        """darsia.Grid: grid"""

        self.voxel_size = grid.voxel_size
        """np.ndarray: voxel size"""

        # Cache solver options
        self.options = options
        """dict: options for the solver"""

        self.verbose = self.options.get("verbose", False)
        """bool: verbosity"""

        self.div_contraint_tol = self.options.get("",1e-6)


        # Allocate space for main and auxiliary variables
        self.transport_density = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: flux"""

        self.flux = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: flux"""


        self.pressure = np.zeros(self.grid.num_cells, dtype=float)
        """ np.ndarray: pressure of the poisson problem -div(p) = f = img1- img2"""

        self.flux_old = copy.deepcopy(self.flux)
        self.transport_density_old = copy.copy(self.transport_density)
        self.pressure_old = copy.deepcopy(self.pressure)



        self._setup_discretization()

    def _setup_discretization(self) -> None:
        """Setup of fixed discretization operators.

        Add linear contribution of the optimality conditions of the Newton linearization.

        """

        # ! ---- Discretization operators ----

        if self.grid.voxel_size[0] != self.grid.voxel_size[1]:
            raise ValueError("The grid must be isotropic")
        self.h = self.grid.voxel_size[0]


        self.div = darsia.FVDivergence(self.grid).mat / self.h
        """sps.csc_matrix: divergence operator: flat fluxes -> flat pressures"""
        
        self.grad = darsia.FVDivergence(self.grid).mat.T / self.h**2


        self.mass_matrix_cells = darsia.FVMass(self.grid).mat
        """sps.csc_matrix: mass matrix on cells: flat pressures -> flat pressures"""

                

        

    def setup_elliptic_solver(self, transport_density, rtol=1e-6):
        """
        
        """
        self.weighted_Laplacian_matrix = self.div * sps.diags(transport_density) / self.h * self.grad

        # Define CG solver
        kernel = np.ones(self.grid.num_cells, dtype=float) / np.sqrt(self.grid.num_cells)
        weighted_Poisson_solver = darsia.linalg.KSP(self.weighted_Laplacian_matrix, 
                                                nullspace=[kernel],
                                                appctx={})
        
        weighted_Poisson_ksp_ctrl = {
                        "ksp_type": "cg",
                        "ksp_rtol": rtol,
                        "ksp_maxit": 100,
                        "pc_type": "hypre",
                        "ksp_monitor": None,
        }
        weighted_Poisson_solver.setup(weighted_Poisson_ksp_ctrl)
        
        return weighted_Poisson_solver
        
    def _solve(self, 
               flat_mass_diff: np.ndarray,
               ) -> tuple[float, np.ndarray, dict]:
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
        num_iter = self.options.get("num_iter", 400)
        tol_residual = self.options.get("tol_residual", np.finfo(float).max)
        tol_increment = self.options.get("tol_increment", np.finfo(float).max)
        tol_distance = self.options.get("tol_distance", np.finfo(float).max)
            
        
        # Initialize Newton iteration with Darcy solution for unitary mobility
        self.transport_density[:] = 1.0 / self.kappa_faces
        Poisson_solver = self.setup_elliptic_solver(self.transport_density)
        pressure = Poisson_solver.solve(flat_mass_diff)
        gradient_pressure = self.grad.dot(pressure)
        distance = self.compute_dual(pressure, flat_mass_diff) 
        flux = self.transport_density * gradient_pressure

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
                "DMK iter. \t| W^1 \t\t| Δ W^1 \t| Δ flux \t| residual",
                "\n",
                """---------------|---------------|---------------|---------------|"""
                """---------------""",
            )


        self.iter = 0
        deltat = 0.01
        # DMK iterations
    
        while self.iter <= num_iter:
            self.flux_old[:] = flux[:]
            self.pressure_old[:] = pressure[:]
            self.transport_density_old[:] = self.transport_density[:]
            distance_old = distance

            # start update 
            start = time.time()

            # udpate transport density
            update = self.transport_density * ( np.abs(gradient_pressure)  - kappa_faces)
            deltat = min(deltat * 1.05, 0.5)
            self.transport_density += deltat * update
            min_tdens = 1e-10
            self.transport_density[self.transport_density < min_tdens] = min_tdens
            # udpate potential
            Poisson_solver = self.setup_elliptic_solver(self.transport_density)
            pressure = Poisson_solver.solve(flat_mass_diff)
            gradient_pressure = self.grad.dot(pressure)

            # update flux
            flux = self.transport_density * gradient_pressure

            self.iter += 1
            self.iter_cpu = time.time() - start

            
            # int_ pot f = int_ pot div poisson_pressure = int_ grad pot \cdot \nabla poisson_pressure 
            self.dual_value = self.compute_dual(pressure, flat_mass_diff) 
            self.primal_value = self.compute_primal(flux)
            self.duality_gap = abs(self.dual_value - self.primal_value)
            distance = self.primal_value

            distance_increment = abs( distance - distance_old) / distance
            if distance_increment < tol_distance:
                break


            residual = np.linalg.norm(( gradient_pressure ** 2 - 1.0) * self.transport_density)
            if residual < tol_residual:
                break

            flux_increment = np.linalg.norm(flux - self.flux_old) / np.linalg.norm(self.flux_old)  
            if flux_increment < tol_increment:
                break
            
            
            if self.verbose :
                print(
                    f"it: {self.iter:03d}" +
                    f" gap={self.duality_gap:.1e}" +
                    f" dual={self.dual_value:.3e} primal={self.primal_value:.3e}" +
                    f" cpu={self.iter_cpu:.3f}"+
                    f" max grad={np.max(gradient_pressure):.2e}"
                    f" {np.min(self.transport_density):.2e}<=TDENS<={np.max(self.transport_density):.2e}"
                    ) 

            
            # Compute the error and store as part of the convergence history:
            # 0 - full residual (Newton interpretation)
            # 1 - flux increment (fixed-point interpretation)
            # 2 - distance increment (Minimization interpretation)

            # Update convergence history
            convergence_history["distance"].append(distance)
            convergence_history["residual"].append(residual)
            convergence_history["flux_increment"].append(flux_increment)
            convergence_history["distance_increment"].append(distance_increment)
            #convergence_history["timing"].append(stats_i)

            # Extract current total run time
            #current_run_time = self._analyze_timings(convergence_history["timing"])[
            #    "total"
            #]
            #convergence_history["run_time"].append(current_run_time)

            # Print performance to screen
            # - distance
            # - distance increment
            # - flux increment
            # - residual
            if self.verbose:
                print(
                    f"""Iter. {self.iter} \t| {distance:.6e} \t| """
                    + f"""{distance_increment:.6e} \t| {flux_increment:.6e} \t| """
                    + f"""{residual:.6e}"""
                )

        # Summarize profiling (time in seconds, memory in GB)
        #total_timings = self._analyze_timings(convergence_history["timing"])
        #peak_memory_consumption = tracemalloc.get_traced_memory()[1] / 10**9



        # Define performance metric
        info = {
            "converged": self.iter < num_iter,
            "number_iterations": self.iter,
            "convergence_history": convergence_history,
            #"timings": total_timings,
            #"peak_memory_consumption": peak_memory_consumption,
        }

        solution = np.concatenate([pressure, self.transport_density])

        return distance, solution, info

    def compute_dual(self, pressure, forcing):
        """
        Compute the value of the dual functional
        $\int_{\Domain} pot (f^+ - f^-)$
        """
        return np.dot(pressure, forcing) *  np.prod(self.grid.voxel_size)
    
    def compute_primal(self,flux):
        """
        Compute the value of the primal functional
        $\int_{\Domain} | flux | $
        """
        return np.sum(np.abs(flux)) * np.prod(self.grid.voxel_size) 

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

        flat_pressure = solution[:self.grid.num_cells]
        flat_gradient = self.div.T.dot(flat_pressure)
        flat_tdens = solution[-self.grid.num_faces:]
        flat_flux = flat_tdens * flat_gradient
        

        flux = darsia.face_to_cell(self.grid, flat_flux)
        temp = np.hstack([flat_tdens,flat_tdens]) 
        transport_density = darsia.face_to_cell(self.grid, temp)[:,:,0]

        
        pressure = flat_pressure.reshape(self.grid.shape, order="F")
        
        

        
        # Return solution
        return_info = self.options.get("return_info", False)
        if return_info:
            info.update(
                {
                    "grid": self.grid,
                    "mass_diff": mass_diff,
                    "flux": flux,
                    "pressure": pressure,
                    "transport_density": transport_density,
                    "src": img_1,
                    "dst": img_2,
                }
            )
            return distance, info
        else:
            return distance



class WassersteinDistanceGproxPGHD(darsia.EMD):
    """
    This contains the implementation of the GproxPDHG algorithm
    described in "SOLVING LARGE-SCALE OPTIMIZATION PROBLEMS WITH
    A CONVERGENCE RATE INDEPENDENT OF GRID SIZE"
    """

    def __init__(self, grid, options) -> None:
        # Cache geometrical infos
        self.grid = grid
        """darsia.Grid: grid"""

        self.voxel_size = grid.voxel_size
        """np.ndarray: voxel size"""

        # Cache solver options
        self.options = options
        """dict: options for the solver"""

        self.verbose = self.options.get("verbose", False)
        """bool: verbosity"""

        # Allocate space for main and auxiliary variables
        self.flux = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: flux"""

        self.poisson_pressure = np.zeros(self.grid.num_cells, dtype=float)
        """ np.ndarray: pressure of the poisson problem -div(p) = f = img1- img2"""

        self.gradient_poisson = np.zeros(self.grid.num_faces, dtype=float)
        """ varaible storing the gradient of the poisson problem"""

        self.rhs_forcing = np.zeros(self.grid.num_cells, dtype=float)
        """ variable storing the right hand side of the poisson problem"""


        

        self.div_free_flux = np.zeros(self.grid.num_faces, dtype=float)
        """ np.ndarray: divergence free flux"""

        self.u = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: u"""

        self.p = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: $p^{n}$"""


        self.p_bar = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: $\bar{p}$"""

        self.new_p = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: $p^{n+1}$"""

        self._setup_discretization()

    def _setup_discretization(self) -> None:
        """Setup of fixed discretization operators.

        Add linear contribution of the optimality conditions of the Newton linearization.

        """

        # ! ---- Discretization operators ----
        if self.grid.voxel_size[0] != self.grid.voxel_size[1]:
            raise ValueError("The grid must be isotropic")
        self.h = self.grid.voxel_size[0]
        

        self.div = darsia.FVDivergence(self.grid).mat #/ self.h**2
        """sps.csc_matrix: divergence operator: flat fluxes -> flat pressures"""
    
        self.grad = darsia.FVDivergence(self.grid).mat.T #* self.h**4
        """sps.csc_matrix: grad operator:  flat pressures -> falt fluxes"""


        self.mass_matrix_cells = darsia.FVMass(self.grid).mat
        """sps.csc_matrix: mass matrix on cells: flat pressures -> flat pressures"""

        self.Laplacian_matrix = self.div * self.grad  #/ self.h**2

        # Define CG solver
        kernel = np.ones(self.grid.num_cells, dtype=float) / np.sqrt(self.grid.num_cells)
        self.Poisson_solver = darsia.linalg.KSP(self.Laplacian_matrix, 
                                                nullspace=[kernel],
                                                appctx={})
        
        self.Poisson_ksp_ctrl = {
                        "ksp_type": "cg",
                        "ksp_rtol": 1e-6,
                        "ksp_maxit": 100,
                        "pc_type": "hypre",
                        "ksp_monitor": None,
        }
        self.Poisson_solver.setup(self.Poisson_ksp_ctrl)


    def residual(self, rhs: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """Compute the residual of the solution.

        Args:
            rhs (np.ndarray): right hand side
            solution (np.ndarray): solution

        Returns:
            np.ndarray: residual

        """
        return self.optimality_conditions(rhs, solution)
    

    def leray_projection(self, p: np.ndarray) -> np.ndarray:
        """Leray projection of a vector fiels

        Args:
            p (np.ndarray): pressure

        Returns:
            np.ndarray: divergence free flux

        """
        rhs = self.div.dot(p)
        poisson_solution = self.Poisson_solver.solve(rhs) #/ self.h**2
        return p - self.grad.dot(poisson_solution) 
    
    def compute_pressure(self, flux: np.ndarray, forcing: np.array) -> np.ndarray:
        """Compute the pressure from the flux.

        Args:
            flux (np.ndarray): flux

        Returns:
            np.ndarray: pressure

        """
        self.Laplacian_matrix = self.div * sps.diags(np.abs(flux), dtype=float) * self.grad / self.h **4

        # Define CG solver
        kernel = np.ones(self.grid.num_cells, dtype=float) / np.sqrt(self.grid.num_cells)
        self.weighted_Poisson_solver = darsia.linalg.KSP(self.Laplacian_matrix, 
                                                nullspace=[kernel],
                                                appctx={})
        
        self.weighted_Poisson_ksp_ctrl = {
                        "ksp_type": "cg",
                        "ksp_rtol": 1e-6,
                        "ksp_maxit": 100,
                        "pc_type": "hypre",
                        "ksp_monitor": None,
        }
        self.weighted_Poisson_solver.setup(self.weighted_Poisson_ksp_ctrl)
        pressure = self.weighted_Poisson_solver.solve(forcing)
        self.weighted_Poisson_solver.kill()
        return pressure
    
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

        
        # Initialize Newton iteration with Darcy solution for unitary mobility
        poisson_pressure = self.Poisson_solver.solve(flat_mass_diff)
        gradient_poisson_pressure = self.grad.dot(poisson_pressure)

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
                "PD iter. \t| W^1 \t\t| Δ W^1 \t| Δ flux \t| residual",
                "\n",
                """---------------|---------------|---------------|---------------|"""
                """---------------""",
            )


        p_bar = self.p_bar
        u = self.u
        p = self.p
        new_p = self.new_p
        self.iter = 0
        # PDHG iterations
        while self.iter <= num_iter:
            #
            start = time.time()
            tau = self.options.get("tau", 1.0)
            sigma =  self.options.get("sigma", 1.0) 
            if self.iter > 0:
                p[:] = new_p[:]

            # eq 3.14
            div_free = self.leray_projection(p_bar)
            u -= tau * div_free

            # new flux
            self.flux = u + gradient_poisson_pressure

            flat_pressure = self.compute_pressure(self.flux, flat_mass_diff)
            grad_pressure = self.grad.dot(flat_pressure)
            print(f"MAX GRAD {np.max(abs(grad_pressure))}")
            
            # eq 3.15
            sigma_vel = p + sigma * self.flux
            #print(sigma_vel)
            new_p[:] = sigma_vel[:]
            abs_sigma_vel = np.abs(sigma_vel)
            greater_than_1 = np.where(abs_sigma_vel > 1)
            new_p[greater_than_1] /= abs_sigma_vel[greater_than_1] # normalize too +1 or -1
            #print(new_p)


            # eq 3.16
            p_bar[:] = 2 * new_p[:] - p[:]

            # int_ pot f = int_ pot div poisson_pressure = int_ grad pot \cdot \nabla poisson_pressure 
            self.dual_value = self.compute_dual(p, gradient_poisson_pressure) 
            self.primal_value = self.compute_primal(self.flux)
            self.duality_gap = abs(self.dual_value - self.primal_value)
            distance = self.primal_value


            self.iter += 1

            self.iter_cpu = time.time() - start
            #if self.verbose :
            print(
                    f"it: {self.iter:03d}" +
                    f" gap={self.duality_gap:.1e}" +
                    f" dual={self.dual_value:.3e} primal={self.primal_value:.3e}" +
                    f" cpu={self.iter_cpu:.3f}") 

            #convergence_history["distance"].append(new_distance)
            #convergence_history["residual"].append(np.linalg.norm(residual_i, 2))
            
            # Compute the error and store as part of the convergence history:
            # 0 - full residual (Newton interpretation)
            # 1 - flux increment (fixed-point interpretation)
            # 2 - distance increment (Minimization interpretation)

            # Update convergence history
            #convergence_history["distance"].append(new_distance)
            #convergence_history["residual"].append(np.linalg.norm(residual_i, 2))
            #convergence_history["flux_increment"].append(
            #    np.linalg.norm(increment[self.flux_slice], 2)
            #)
            #convergence_history["distance_increment"].append(
            #    abs(new_distance - old_distance)
            #)
            #convergence_history["timing"].append(stats_i)

            # Extract current total run time
            #current_run_time = self._analyze_timings(convergence_history["timing"])[
            #    "total"
            #]
            #convergence_history["run_time"].append(current_run_time)

            # Print performance to screen
            # - distance
            # - distance increment
            # - flux increment
            # - residual
            if False:#self.verbose:
                distance_increment = convergence_history["distance_increment"][-1]
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

        # Summarize profiling (time in seconds, memory in GB)
        #total_timings = self._analyze_timings(convergence_history["timing"])
        #peak_memory_consumption = tracemalloc.get_traced_memory()[1] / 10**9



        # Define performance metric
        info = {
            "converged": self.iter < num_iter,
            "number_iterations": self.iter,
            "convergence_history": convergence_history,
            #"timings": total_timings,
            #"peak_memory_consumption": peak_memory_consumption,
        }

        return distance, self.flux, info 
    
    def compute_dual(self,p,gradient_poisson):
        """
        Compute the value of the dual functional
        $\int_{\Domain} pot (f^+ - f^-)$
        $=\int_{\Domain} pot -div(poisson)$
        $\int_{\Domain} \nabla pot \cdot \nabla poisson$
        $\int_{\Domain} p \cdot \nabla poisson$
        """
        return np.dot(p, gradient_poisson) *  np.prod(self.grid.voxel_size)
    
    def compute_primal(self,flux):
        """
        Compute the value of the primal functional
        $\int_{\Domain} | flux | $
        """
        return np.sum(np.abs(flux)) * np.prod(self.grid.voxel_size)

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

        flux = darsia.face_to_cell(self.grid, solution)
        flat_pressure = self.compute_pressure(solution, flat_mass_diff)
        temp = np.hstack([np.abs(solution),np.abs(solution)])
        transport_density = darsia.face_to_cell(self.grid, temp)[:,:,0]
        pressure = flat_pressure.reshape(self.grid.shape, order="F")
        

        
        # Return solution
        return_info = self.options.get("return_info", False)
        if return_info:
            info.update(
                {
                    "grid": self.grid,
                    "mass_diff": mass_diff,
                    "flux": flux,
                    "pressure": pressure,
                    "transport_density": transport_density,
                    "src": img_1,
                    "dst": img_2,
                }
            )
            return distance, info
        else:
            return distance





# Unified access
def wasserstein_distance(
    mass_1: darsia.Image,
    mass_2: darsia.Image,
    method: str,
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
    if method.lower() in ["newton", "bregman", "sinkhorn","pdhg","dmk"]:
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
        elif method.lower() == "sinkhorn":
            w1 = WassersteinDistanceSinkhorn(grid, options)
        elif method.lower() == "pdhg":
            w1 = WassersteinDistanceGproxPGHD(grid, options)
        elif method.lower() == "dmk":
            w1 = WassersteinDistanceDMK(grid, options)

    elif method.lower() == "cv2.emd":
        # Use Earth Mover's Distance from CV2
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
        (key, info[key])
        for key in ["src", "dst", "mass_diff", "flux", "pressure", "transport_density"]
    ]
    darsia.plotting.to_vtk(path, data)


        

