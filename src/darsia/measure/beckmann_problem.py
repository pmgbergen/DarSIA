"""Beckmann problem definition in variational and mixed form."""

from __future__ import annotations

import time
import warnings
from abc import abstractmethod
from enum import StrEnum
from typing import Optional
from warnings import warn

import numpy as np
import pyamg
import scipy.sparse as sps
from scipy.stats import hmean

import darsia


class L1Mode(StrEnum):
    """Mode for computing the l1 dissipation."""

    RAVIART_THOMAS = "raviart_thomas"
    CONSTANT_SUBCELL_PROJECTION = "constant_subcell_projection"
    CONSTANT_CELL_PROJECTION = "constant_cell_projection"


class MobilityMode(StrEnum):
    """Mode for computing the mobility."""

    CELL_BASED = "cell_based"
    CELL_BASED_ARITHMETIC = "cell_based_arithmetic"
    CELL_BASED_HARMONIC = "cell_based_harmonic"
    SUBCELL_BASED = "subcell_based"
    FACE_BASED = "face_based"


class BeckmannProblem(darsia.EMD):
    """Base class for setting up the Beckmann problem.

    The Beckmann problem is defined as the solution to the following
    optimization problem:

        inf ||u||_{L^1} s.t. div u = m_2 - m_1, u in H(div).

    u is the flux, m_1 and m_2 are the mass distributions which are transported by u
    from m_1 to m_2. The problem is solved approximately, employing an iterative
    TPFA-type finite volume method. A close connection to the lowest Raviart-Thomas
    mixed finite element method is exploited.

    There are two main solution strategies implemented in specialized classes:
    - Finite Volume Quasi-Newton's method (:class:`WassersteinDistanceNewton`)
    - Finite Volume Split Bregman method (:class:`WassersteinDistanceBregman`)

    """

    # ! ---- Setup routines ----

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

        self.mobility_mode: MobilityMode = self.options.get(
            "mobility_mode", MobilityMode.CELL_BASED
        )
        """str: mode for computing the mobility"""

        self.weight = weight
        """Weight defined on cells"""

        # Setup of method
        self._setup_dof_management()
        self._setup_l1_quadrature()
        self._setup_face_weights()
        self._setup_discretization()
        self._setup_linear_solver()
        self._setup_schur_complement_reduction()
        self._setup_acceleration()

        # Store list of callbacks passed by user
        self.callbacks = options.get("callbacks", None)
        """list: list of callbacks to be called during the optimization"""

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

    def _setup_l1_quadrature(self) -> None:
        """Setup of quadrature for the l1 dissipation.

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
        l1_mode: L1Mode = self.options.get("l1_mode", L1Mode.RAVIART_THOMAS)
        dim = self.grid.dim
        match l1_mode:
            case L1Mode.RAVIART_THOMAS:
                # Apply numerical integration of RT0 extensions into cells.
                # Underlying functional for mixed finite element method (MFEM).
                quad_pts, quad_weights = darsia.quadrature.gauss_reference_cell(
                    dim, "max"
                )

            case L1Mode.CONSTANT_SUBCELL_PROJECTION:
                # Apply subcell_based projection onto constant vectors and sum up.
                # Equivalent to a mixed finite volume method (FV). Identical to quadrature
                # over corners.
                quad_pts, quad_weights = darsia.quadrature.reference_cell_corners(dim)

            case L1Mode.CONSTANT_CELL_PROJECTION:
                # L2 projection onto constant vectors identical to quadrature of order 0.
                quad_pts, quad_weights = darsia.quadrature.gauss_reference_cell(dim, 0)

            case _:
                raise ValueError(f"Mode {l1_mode} not supported.")

        self.quad_pts = quad_pts
        """np.ndarray: quadrature points in reference cell"""

        self.quad_weights = quad_weights
        """np.ndarray: quadrature weights in reference cell"""

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

        self.broken_darcy = sps.bmat(
            [
                [None, -self.div.T, None],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ],
            format="csc",
        )
        """sps.csc_matrix: linear part of the Darcy operator with pressure constraint"""

        L_init = self.options.get("L_init", 1.0)
        self.darcy_init = self.broken_darcy_with_custom_flux_block(
            L_init * self.weighted_mass_matrix_faces_init
        )
        """sps.csc_matrix: initial Darcy operator"""

    def broken_darcy_with_custom_flux_block(
        self, flux_flux_block: sps.csc_matrix
    ) -> sps.csc_matrix:
        """Construct the broken Darcy operator with given flux-flux block.

        Args:
            flux_flux_block (sps.csc_matrix): flux-flux block

        Returns:
            sps.csc_matrix: broken Darcy operator with given flux-flux block

        """
        return sps.bmat(
            [
                [flux_flux_block, -self.div.T, None],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ],
            format="csc",
        )

    def _setup_face_reconstruction(self) -> None:
        """Setup of face reconstruction via RT0 basis functions and arithmetic avg.

        NOTE: Do not make part of self._setup_discretization() as not always required.

        """
        self.face_reconstruction = darsia.FVFullFaceReconstruction(self.grid)
        """sps.csc_matrix: full face reconstruction: flat fluxes -> vector fluxes"""

    def _setup_linear_solver(self) -> None:
        linear_solver_type = self.options.get("linear_solver", "direct")
        self.linear_solver_type = linear_solver_type
        """str: type of linear solver"""

        self.linear_solver = darsia.BeckmannLinearSolverFactory.create(
            linear_solver_type, self.options
        )

    def _setup_schur_complement_reduction(self) -> None:
        # TODO: Move to setup_schur_complement_reduction()
        self.formulation: str = self.options.get("formulation", "pressure")
        """str: formulation type"""

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

    # ! ---- Main methods ----

    @abstractmethod
    def solve_beckmann_problem(self, mass_diff: np.ndarray) -> tuple:
        """Solve for the Wasserstein distance.

        Args:
            mass_diff (np.ndarray): difference of the two distributions

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
        mass_diff_img = img_2.img - img_1.img
        mass_diff = self.flat_view(mass_diff_img)

        # Main method
        distance, solution, info = self.solve_beckmann_problem(mass_diff)

        # Split the solution
        flux = solution[self.flux_slice]
        pressure = solution[self.pressure_slice]

        # Reshape the fluxes and pressure to grid format
        flux_img = darsia.face_to_cell(self.grid, flux)
        pressure_img = pressure.reshape(self.grid.shape, order="F")

        # Determine transport density
        transport_density = self.transport_density(flux, flatten=False)

        # Cell-weighted flux
        weighted_flux_img = self.cell_weighted_flux(flux_img)

        # Return solution
        return_info = self.options.get("return_info", False)
        return_status = self.options.get("return_status", False)
        if return_info:
            info.update(
                {
                    "grid": self.grid,
                    "mass_diff": mass_diff_img,
                    "flux": flux_img,
                    "weight": self.cell_weights,
                    "weight_inv": 1.0 / self.cell_weights,
                    "weighted_flux": weighted_flux_img,
                    "pressure": pressure_img,
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

    # ! ---- Problem definition ----

    def exact_linearization(self, solution: np.ndarray) -> sps.csc_matrix:
        """Compute the exact linearization of the constrained minimization problem.

        Args:
            solution (np.ndarray): solution

        Returns:
            sps.csc_matrix: exact linearization

        """
        flux = self.flux_view(solution)
        face_weights, face_weights_inv = self._compute_face_weight(flux)
        weight = sps.diags(face_weights)
        return self.broken_darcy_with_custom_flux_block(weight * self.mass_matrix_faces)

    def optimality_conditions(
        self,
        solution: np.ndarray,
        beckmann_problem_rhs: np.ndarray,
    ) -> np.ndarray:
        """Evaluate optimality conditions of the constrained minimization problem.

        Args:
            beckmann_problem_rhs (np.ndarray): right hand side of the Beckmann problem
            solution (np.ndarray): solution

        Returns:

            np.ndarray: residual

        """
        return self.exact_linearization(solution).dot(solution) - beckmann_problem_rhs

    def rescaled_flux_optimality_conditions(self, solution: np.ndarray) -> np.ndarray:
        """Evaluate scaled optimality conditions of the constrained minimization problem.

        Scale the flux equation by the face weights divided by the total distance.
        This results in no division by zero.

        """
        flux = self.flux_view(solution)
        pressure = self.pressure_view(solution)
        transport_density_faces = self.transport_density_faces(flux)
        distance = self.l1_dissipation(flux)
        return (
            self.mass_matrix_faces.dot(flux)
            - transport_density_faces * self.div.T.dot(pressure)
        ) / distance

    # ! ---- Effective quantities ----

    def cell_weighted_flux(self, cell_flux: np.ndarray) -> np.ndarray:
        """Compute the cell-weighted flux.

        Args:
            cell_flux (np.ndarray): cell fluxes

        Returns:
            np.ndarray: cell-weighted flux

        """
        # Apply cell weights - depending on the dimensionality of the weight
        if self.weight is None:
            # No weighting
            return cell_flux
        elif len(self.cell_weights.shape) == self.grid.dim:
            # Isotropic, vector weight
            return cell_flux * self.cell_weights[..., np.newaxis]
        elif (
            len(self.cell_weights.shape) == self.grid.dim + 1
            and self.cell_weights.shape[-1] == 1
        ):
            # Isotropic, scalar weight
            raise NotImplementedError("Need to reduce the dimension")
            # Try: return cell_flux * self.cell_weights
        elif (
            len(self.cell_weights.shape) == self.grid.dim + 1
            and self.cell_weights.shape[-1] == self.grid.dim
        ):
            # Anisotropic, diagonal weight tensor
            return cell_flux * self.cell_weights
        elif len(
            self.cell_weights.shape
        ) == self.grid.dim + 2 and self.cell_weights.shape[-2:] == (
            self.grid.dim,
            self.grid.dim,
        ):
            # Fully anisotropic weight tensor
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

        """
        # Integrate over reference cell (normalization not required)
        transport_density = np.zeros(self.grid.shape, dtype=float)
        for quad_pt, quad_weight in zip(self.quad_pts, self.quad_weights):
            cell_flux = darsia.face_to_cell(self.grid, flat_flux, pt=quad_pt)
            weighted_cell_flux = self.cell_weighted_flux(cell_flux)
            cell_flux_norm = np.linalg.norm(weighted_cell_flux, 2, axis=-1)
            transport_density += quad_weight * cell_flux_norm

        if flatten:
            return self.flat_view(transport_density)
        else:
            return transport_density

    def transport_density_faces(self, flat_flux: np.ndarray) -> np.ndarray:
        """Compute the transport density from the solution.

        Args:
            flat_flux (np.ndarray): face fluxes

        Returns:
            np.ndarray: transport density

        """
        # The L1 dissipation corresponds to the integral over the transport density
        if not hasattr(self, "face_reconstruction"):
            self._setup_face_reconstruction()
        full_flux = self.face_reconstruction(flat_flux)
        return np.linalg.norm(full_flux, axis=-1)

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
            cell_weights_inv = darsia.array_product(
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
            weight_ratio_inv = darsia.array_product(
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
            weighted_face_flux = darsia.array_product(
                harm_avg_face_weights, full_face_flux
            )
            norm_weighted_face_flux = np.linalg.norm(weighted_face_flux, 2, axis=1)

            # Combine weights**2 / |weight * flux| on faces
            face_weights = harm_avg_face_weights**2 / norm_weighted_face_flux
            face_weights_inv = norm_weighted_face_flux / harm_avg_face_weights**2
        else:
            raise ValueError(f"Mobility mode {self.mobility_mode} not supported.")

        return face_weights, face_weights_inv

    # ! ---- Linear solver and Schur complement reduction ----

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
            assert isinstance(
                self.linear_solver,
                darsia.BeckmannDirectSolver | darsia.BeckmannKSPFieldSplitSolver,
            ), "Only direct solver or ksp supported for full formulation."

            # Setup LU factorization for the full system
            tic = time.time()
            if setup_linear_solver:
                self.linear_solver.setup(matrix)
            time_setup = time.time() - tic

            # Solve the full system
            tic = time.time()
            solution = self.linear_solver(rhs)
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

            # 1. Reduce flux block
            (
                self.reduced_matrix,
                self.reduced_rhs,
                self.matrix_flux_inv,
            ) = self.eliminate_flux(matrix, rhs)

            # 2. Build linear solver for reduced system
            if setup_linear_solver:
                self.linear_solver.setup(self.reduced_matrix)

            # Stop timer to measure setup time
            time_setup = time.time() - tic

            # 3. Solve for the pressure and lagrange multiplier
            tic = time.time()
            solution[self.reduced_system_slice] = self.linear_solver(self.reduced_rhs)

            # 4. Compute flux update
            solution[self.flux_slice] = self._compute_flux_update(solution, rhs)
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

            # 1. Reduce flux block
            (
                self.reduced_matrix,
                self.reduced_rhs,
                self.matrix_flux_inv,
            ) = self.eliminate_flux(matrix, rhs)

            # 2. Reduce to pure pressure system
            (
                self.fully_reduced_matrix,
                self.fully_reduced_rhs,
            ) = self.eliminate_lagrange_multiplier(
                self.reduced_matrix,
                self.reduced_rhs,
            )

            # 3. Build linear solver for pure pressure system
            if setup_linear_solver:
                self.linear_solver.setup(self.fully_reduced_matrix)

            # Stop timer to measure setup time
            time_setup = time.time() - tic

            # 4. Solve the pure pressure system
            tic = time.time()
            solution[self.fully_reduced_system_indices_full] = self.linear_solver(
                self.fully_reduced_rhs
            )

            # 5. Compute lagrange multiplier - not required, as it is zero
            pass

            # 6. Compute flux update
            solution[self.flux_slice] = self._compute_flux_update(solution, rhs)
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
        # Make sure the setup routine has been called
        if not hasattr(self, "jacobian_subblock"):
            self._setup_eliminate_flux(jacobian)
            assert hasattr(self, "jacobian_subblock")

        # Build Schur complement wrt flux-block
        J_inv = sps.diags(1.0 / jacobian.diagonal()[self.flux_slice])
        schur_complement = self.D.dot(J_inv.dot(self.DT))

        # Gauss eliminiation on matrices
        reduced_jacobian = self.jacobian_subblock + schur_complement

        # Gauss elimination on vectors
        reduced_residual = residual[self.reduced_system_slice].copy()
        reduced_residual -= self.D.dot(J_inv.dot(residual[self.flux_slice]))

        return reduced_jacobian, reduced_residual, J_inv

    def _setup_eliminate_flux(self, matrix: sps.csc_matrix) -> None:
        """Setup the infrastructure for reduced systems through Gauss elimination.

        Provide internal data structures for the reduced system, still formulated in
        terms of pressures and Lagrange multiplier. Merely the flux is eliminated using
        a Schur complement approach.

        Args:
            matrix (sps.csc_matrix): system matrix with flux-flux block to be
                eliminated - assume to have the same structure as self.darcy_init

        """
        #   ---- Preliminaries ----

        # Fixate some jacobian for copying the data structure
        jacobian = matrix.copy()

        # Cache divergence matrix together with Lagrange multiplier
        D = jacobian[self.reduced_system_slice, self.flux_slice].copy()
        self.D = D.copy()
        """sps.csc_matrix: divergence + lagrange multiplier matrix"""

        self.DT = self.D.T.copy()
        """sps.csc_matrix: transposed divergence + lagrange multiplier  matrix"""

        # Cache (constant) jacobian subblock
        self.jacobian_subblock = jacobian[
            self.reduced_system_slice, self.reduced_system_slice
        ].copy()
        """sps.csc_matrix: constant jacobian subblock of the reduced system"""

        # ! ---- Eliminate flux block ----

        # Build Schur complement wrt. flux-flux block
        J_inv = sps.diags(self.flux_view(1.0 / jacobian.diagonal()))
        schur_complement = D.dot(J_inv.dot(D.T))

        # Add Schur complement - use this to identify sparsity structure
        # Cache the reduced jacobian
        self.reduced_jacobian = self.jacobian_subblock + schur_complement
        """sps.csc_matrix: reduced jacobian incl. Schur complement"""

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
        # Make sure the setup routine has been called
        if not hasattr(self, "fully_reduced_jacobian"):
            self._setup_eliminate_lagrange_multiplier()
            assert hasattr(self, "fully_reduced_jacobian")

        # Make sure the jacobian is a CSC matrix
        assert isinstance(reduced_jacobian, sps.csc_matrix), (
            "Jacobian should be a CSC matrix."
        )

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

    def _setup_eliminate_lagrange_multiplier(self) -> None:
        """Additional setup of infrastructure for fully reduced systems.

        Here, the Lagrange multiplier is eliminated through Gauss elimination. It is
        implicitly assumed, that the flux block is eliminated already.

        """
        # Make sure the setup routine has been called
        assert hasattr(self, "reduced_jacobian"), (
            "Need to call setup_eliminate_flux() first."
        )

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

    def _compute_flux_update(self, solution: np.ndarray, rhs: np.ndarray) -> np.ndarray:
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

    # ! ---- Utility methods ----

    def _sum_timings(self, timings: list[dict[str, float]]) -> dict[str, float]:
        """Analyze the timing of the current iteration.

        Utility function for self.solve_beckmann_problem().

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

    # ! ---- Views and indexing ----

    def ndofs(self) -> int:
        """Return the total number of degrees of freedom.

        Returns:
            int: total number of degrees of freedom

        """
        return self.grid.num_faces + self.grid.num_cells + 1

    def flat_view(self, img: np.ndarray) -> np.ndarray:
        """Flatten the image to a vector.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: flattened image

        """
        return np.ravel(img, "F")

    def flux_view(self, vector: np.ndarray) -> np.ndarray:
        """Extract the flux from the vector.

        Args:
            vector (np.ndarray): vector

        Returns:
            np.ndarray: flux

        """
        assert len(vector) in [
            self.grid.num_faces + self.grid.num_cells,
            self.grid.num_faces + self.grid.num_cells + 1,
        ], (
            f"Vector has wrong length {len(vector)} instead of "
            f"{self.grid.num_faces + self.grid.num_cells} or "
            f"{self.grid.num_faces + self.grid.num_cells + 1}."
        )
        return vector[self.flux_slice]

    def pressure_view(self, vector: np.ndarray) -> np.ndarray:
        """Extract the pressure from the vector.

        Args:
            vector (np.ndarray): vector

        Returns:
            np.ndarray: pressure

        """
        assert len(vector) in [
            self.grid.num_faces + self.grid.num_cells,
            self.grid.num_faces + self.grid.num_cells + 1,
        ], "Vector has wrong length."
        return vector[self.pressure_slice]
