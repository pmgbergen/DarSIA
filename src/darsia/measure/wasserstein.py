"""Wasserstein distance computed using variational methods.

"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Union

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
    from m_1 to m_2.

    Specialized classes implement the solution of the Beckman problem using different
    methods. There are two main methods:
    - Newton's method (:class:`WassersteinDistanceNewton`)
    - Split Bregman method (:class:`WassersteinDistanceBregman`)

    """

    def __init__(
        self,
        grid: darsia.Grid,
        options: dict = {},
    ) -> None:
        """
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
        self.voxel_size = grid.voxel_size

        assert self.grid.dim == 2, "Currently only 2D images are supported."

        self.options = options
        self.regularization = self.options.get("regularization", 0.0)
        self.verbose = self.options.get("verbose", False)

        # Setup of finite volume discretization and acceleration
        self._setup_dof_management()
        self._setup_discretization()
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

        self.orthogonal_face_average = darsia.FVFaceAverage(self.grid).mat
        """sps.csc_matrix: averaging operator for fluxes on orthogonal faces"""

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

    # ! ---- Projections inbetween faces and cells ----

    def face_to_cell(self, flat_flux: np.ndarray) -> np.ndarray:
        """Reconstruct the fluxes on the cells from the fluxes on the faces.

        Use the Raviart-Thomas reconstruction of the fluxes on the cells from the fluxes
        on the faces, and use arithmetic averaging of the fluxes on the faces,
        equivalent with the L2 projection of the fluxes on the faces to the fluxes on
        the cells.

        Matrix-free implementation.

        Args:
            flat_flux (np.ndarray): flat fluxes (normal fluxes on the faces)

        Returns:
            np.ndarray: cell-based vectorial fluxes

        """
        # Reshape fluxes - use duality of faces and normals
        horizontal_fluxes = flat_flux[: self.grid.num_faces_axis[0]].reshape(
            self.grid.vertical_faces_shape
        )
        vertical_fluxes = flat_flux[self.grid.num_faces_axis[0] :].reshape(
            self.grid.horizontal_faces_shape
        )

        # Determine a cell-based Raviart-Thomas reconstruction of the fluxes, projected
        # onto piecewise constant functions.
        cell_flux = np.zeros((*self.grid.shape, self.grid.dim), dtype=float)
        # Horizontal fluxes
        cell_flux[:, :-1, 0] += 0.5 * horizontal_fluxes
        cell_flux[:, 1:, 0] += 0.5 * horizontal_fluxes
        # Vertical fluxes
        cell_flux[:-1, :, 1] += 0.5 * vertical_fluxes
        cell_flux[1:, :, 1] += 0.5 * vertical_fluxes

        return cell_flux

    def cell_to_face(self, cell_qty: np.ndarray, mode: str) -> np.ndarray:
        """Project scalar cell quantity to scalr face quantity.

        Allow for arithmetic or harmonic averaging of the cell quantity to the faces. In
        the harmonic case, the averaging is regularized to avoid division by zero.
        Matrix-free implementation.

        Args:
            cell_qty (np.ndarray): scalar-valued cell-based quantity
            mode (str): mode of projection, either "arithmetic" or "harmonic"
                (averaging)

        Returns:
            np.ndarray: face-based quantity

        """
        # Determine the fluxes on the faces
        if mode == "arithmetic":
            # Employ arithmetic averaging
            horizontal_face_qty = 0.5 * (cell_qty[:, :-1] + cell_qty[:, 1:])
            vertical_face_qty = 0.5 * (cell_qty[:-1, :] + cell_qty[1:, :])
        elif mode == "harmonic":
            # Employ harmonic averaging
            arithmetic_avg_horizontal = 0.5 * (cell_qty[:, :-1] + cell_qty[:, 1:])
            arithmetic_avg_vertical = 0.5 * (cell_qty[:-1, :] + cell_qty[1:, :])
            # Regularize to avoid division by zero
            regularization = 1e-10
            arithmetic_avg_horizontal = (
                arithmetic_avg_horizontal
                + (2 * np.sign(arithmetic_avg_horizontal) + 1) * regularization
            )
            arithmetic_avg_vertical = (
                0.5 * arithmetic_avg_vertical
                + (2 * np.sign(arithmetic_avg_vertical) + 1) * regularization
            )
            product_horizontal = np.multiply(cell_qty[:, :-1], cell_qty[:, 1:])
            product_vertical = np.multiply(cell_qty[:-1, :], cell_qty[1:, :])

            # Determine the harmonic average
            horizontal_face_qty = product_horizontal / arithmetic_avg_horizontal
            vertical_face_qty = product_vertical / arithmetic_avg_vertical
        else:
            raise ValueError(f"Mode {mode} not supported.")

        # Reshape the fluxes - hardcoding the connectivity here
        face_qty = np.concatenate(
            [horizontal_face_qty.ravel(), vertical_face_qty.ravel()]
        )

        return face_qty

    # NOTE: Currently not in use. TODO rm?
    #    def face_restriction(self, cell_flux: np.ndarray) -> np.ndarray:
    #        """Restrict vector-valued fluxes on cells to normal components on faces.
    #
    #        Matrix-free implementation. The fluxes on the faces are determined by
    #        arithmetic averaging of the fluxes on the cells in the direction of the normal
    #        of the face.
    #
    #        Args:
    #            cell_flux (np.ndarray): cell-based fluxes
    #
    #        Returns:
    #            np.ndarray: face-based fluxes
    #
    #        """
    #        # Determine the fluxes on the faces through arithmetic averaging
    #        horizontal_fluxes = 0.5 * (cell_flux[:, :-1, 0] + cell_flux[:, 1:, 0])
    #        vertical_fluxes = 0.5 * (cell_flux[:-1, :, 1] + cell_flux[1:, :, 1])
    #
    #        # Reshape the fluxes
    #        flat_flux = np.concatenate(
    #            [horizontal_fluxes.ravel(), vertical_fluxes.ravel()], axis=0
    #        )
    #
    #        return flat_flux

    # ! ---- Effective quantities ----

    def transport_density(self, cell_flux: np.ndarray) -> np.ndarray:
        """Compute the transport density of the solution.

        Args:
            flat_flux (np.ndarray): flat fluxes

        Returns:
            np.ndarray: transport density
        """
        return np.linalg.norm(cell_flux, 2, axis=-1)

    # TODO consider to replace transport_density with this function:

    # def compute_transport_density(self, solution: np.ndarray) -> np.ndarray:
    #     """Compute the transport density from the solution.

    #     Args:
    #         solution (np.ndarray): solution

    #     Returns:
    #         np.ndarray: transport density

    #     """
    #     # Compute transport density
    #     flat_flux = solution[self.flux_slice]
    #     cell_flux = self.face_to_cell(flat_flux)
    #     norm = np.linalg.norm(cell_flux, 2, axis=-1)
    #     return norm

    def l1_dissipation(self, flat_flux: np.ndarray, mode: str) -> float:
        """Compute the l1 dissipation potential of the solution.

        Args:
            flat_flux (np.ndarray): flat fluxes

        Returns:
            float: l1 dissipation potential

        """
        if mode == "cell_arithmetic":
            cell_flux = self.face_to_cell(flat_flux)
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
            cell_flux = self.face_to_cell(flat_flux)
            # Determine the norm of the fluxes on the cells
            cell_flux_norm = np.maximum(
                np.linalg.norm(cell_flux, 2, axis=-1), self.regularization
            )
            # Determine averaging mode from mode - either arithmetic or harmonic
            average_mode = mode.split("_")[1]
            flat_flux_norm = self.cell_to_face(cell_flux_norm, mode=average_mode)

        elif mode == "face_arithmetic":
            # Define natural vector valued flux on faces (taking arithmetic averages
            # of continuous fluxes over cells evaluated at faces)
            tangential_flux = self.orthogonal_face_average.dot(flat_flux)
            # Determine the l2 norm of the fluxes on the faces, add some regularization
            flat_flux_norm = np.sqrt(flat_flux**2 + tangential_flux**2)

        else:
            raise ValueError(f"Mode {mode} not supported.")

        return flat_flux_norm

    # ! ---- Solver methods ----

    def setup_infrastructure(self) -> None:
        """Setup the infrastructure for reduced systems through Gauss elimination.

        Provide internal data structures for the reduced system.

        """
        # Step 1: Compute the jacobian of the Darcy problem

        # The Darcy problem is sufficient
        jacobian = self.darcy_init.copy()

        # Step 2: Remove flux blocks through Schur complement approach

        # Build Schur complement wrt. flux-flux block
        J_inv = sps.diags(1.0 / jacobian.diagonal()[self.flux_slice])
        D = jacobian[self.reduced_system_slice, self.flux_slice].copy()
        schur_complement = D.dot(J_inv.dot(D.T))

        # Cache divergence matrix
        self.D = D.copy()
        self.DT = self.D.T.copy()

        # Cache (constant) jacobian subblock
        self.jacobian_subblock = jacobian[
            self.reduced_system_slice, self.reduced_system_slice
        ].copy()

        # Add Schur complement - use this to identify sparsity structure
        # Cache the reduced jacobian
        self.reduced_jacobian = self.jacobian_subblock + schur_complement

        # Step 3: Remove potential block through Gauss elimination

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

        # Cache the indices and indptr
        self.fully_reduced_jacobian_indices = fully_reduced_jacobian_indices.copy()
        self.fully_reduced_jacobian_indptr = fully_reduced_jacobian_indptr.copy()
        self.fully_reduced_jacobian_shape = fully_reduced_jacobian_shape

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

        # Define fully reduced system indices wrt full system
        self.fully_reduced_system_indices_full = reduced_system_indices[
            self.fully_reduced_system_indices
        ]

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

    def remove_lagrange_multiplier(self, jacobian, residual, solution) -> tuple:
        """Shortcut for removing the lagrange multiplier from the reduced jacobian.

        Args:

            solution (np.ndarray): solution, TODO make function independent of solution

        Returns:
            tuple: fully reduced jacobian, fully reduced residual

        """
        # Make sure the jacobian is a CSC matrix
        assert isinstance(jacobian, sps.csc_matrix), "Jacobian should be a CSC matrix."

        # Effective Gauss-elimination for the particular case of the lagrange multiplier
        self.fully_reduced_jacobian.data[:] = np.delete(
            self.reduced_jacobian.data.copy(), self.rm_indices
        )
        # NOTE: The indices have to be restored if the LU factorization is to be used
        # FIXME omit if not required
        self.fully_reduced_jacobian.indices = self.fully_reduced_jacobian_indices.copy()

        # Rhs is not affected by Gauss elimination as it is assumed that the residual
        # is zero in the constrained cell, and the pressure is zero there as well.
        # If not, we need to do a proper Gauss elimination on the right hand side!
        if abs(residual[-1]) > 1e-6:
            raise NotImplementedError("Implementation requires residual to be zero.")
        if abs(solution[self.grid.num_faces + self.constrained_cell_flat_index]) > 1e-6:
            raise NotImplementedError("Implementation requires solution to be zero.")
        fully_reduced_residual = self.reduced_residual[
            self.fully_reduced_system_indices
        ].copy()

        return self.fully_reduced_jacobian, fully_reduced_residual

    # ! ---- Main methods ----

    def __call__(
        self,
        img_1: darsia.Image,
        img_2: darsia.Image,
        plot_solution: bool = False,
        return_solution: bool = False,
    ) -> float:
        """L1 Wasserstein distance for two images with same mass.

        NOTE: Images need to comply with the setup of the object.

        Args:
            img_1 (darsia.Image): image 1
            img_2 (darsia.Image): image 2
            plot_solution (bool): plot the solution. Defaults to False.
            return_solution (bool): return the solution. Defaults to False.

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
        flux = self.face_to_cell(flat_flux)
        potential = flat_potential.reshape(self.grid.shape)

        # Determine transport density
        transport_density = self.transport_density(flux)

        # Stop taking time
        toc = time.time()
        status["elapsed_time"] = toc - tic
        print("Elapsed time: ", toc - tic)

        # Plot the solution
        if plot_solution:
            self._plot_solution(mass_diff, flux, potential, transport_density)

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

        """
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
    """Class to determine the L1 EMD/Wasserstein distance solved with Newton's method."""

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

    def linearization_step(
        self, solution: np.ndarray, rhs: np.ndarray, iter: int
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Newton step for the linearization of the problem.

        In the first iteration, the linearization is the linearization of the Darcy
        problem.

        Args:
            solution (np.ndarray): solution
            rhs (np.ndarray): right hand side
            iter (int): iteration number

        Returns:
            tuple: update, residual, stats (timinings)

        """
        # Determine residual and (full) Jacobian
        tic = time.time()
        if iter == 0:
            residual = rhs.copy()
            approx_jacobian = self.darcy_init.copy()
        else:
            residual = self.residual(rhs, solution)
            approx_jacobian = self.jacobian(solution)
        toc = time.time()
        time_setup = toc - tic

        # Allocate update
        update = np.zeros_like(solution, dtype=float)

        # Setup linear solver
        tic = time.time()
        linear_solver = self.options.get("linear_solver", "lu")
        assert linear_solver in [
            "lu",
            "lu-flux-reduced",
            "amg-flux-reduced",
            "lu-potential",
            "amg-potential",
        ], f"Linear solver {linear_solver} not supported."

        if linear_solver in ["amg-flux-reduced", "amg-potential"]:
            # TODO add possibility for user control
            ml_options = {
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
            tol_amg = self.options.get("linear_solver_tol", 1e-6)
            res_history_amg = []

        # Solve linear system for the update
        if linear_solver == "lu":
            # Solve full system
            tic = time.time()
            jacobian_lu = sps.linalg.splu(approx_jacobian)
            time_setup = time.time() - tic
            tic = time.time()
            update = jacobian_lu.solve(residual)
            time_solve = time.time() - tic
        elif linear_solver in ["lu-flux-reduced", "amg-flux-reduced"]:
            # Solve potential-multiplier problem

            # Reduce flux block
            tic = time.time()
            (
                self.reduced_jacobian,
                self.reduced_residual,
                jacobian_flux_inv,
            ) = self.remove_flux(approx_jacobian, residual)

            if linear_solver == "lu-flux-reduced":
                lu = sps.linalg.splu(self.reduced_jacobian)
                time_setup = time.time() - tic
                tic = time.time()
                update[self.reduced_system_slice] = lu.solve(self.reduced_residual)

            elif linear_solver == "amg-flux-reduced":
                ml = pyamg.smoothed_aggregation_solver(
                    self.reduced_jacobian, **ml_options
                )
                time_setup = time.time() - tic
                tic = time.time()
                update[self.reduced_system_slice] = ml.solve(
                    self.reduced_residual,
                    tol=tol_amg,
                    residuals=res_history_amg,
                )

            # Compute flux update
            update[self.flux_slice] = jacobian_flux_inv.dot(
                residual[self.flux_slice]
                + self.DT.dot(update[self.reduced_system_slice])
            )
            time_solve = time.time() - tic

        elif linear_solver in ["lu-potential", "amg-potential"]:
            # Solve pure potential problem

            # Reduce flux block
            tic = time.time()
            (
                self.reduced_jacobian,
                self.reduced_residual,
                jacobian_flux_inv,
            ) = self.remove_flux(approx_jacobian, residual)

            # Reduce to pure pressure system
            (
                self.fully_reduced_jacobian,
                self.fully_reduced_residual,
            ) = self.remove_lagrange_multiplier(
                self.reduced_jacobian, self.reduced_residual, solution
            )

            if linear_solver == "lu-potential":
                lu = sps.linalg.splu(self.fully_reduced_jacobian)
                time_setup = time.time() - tic
                tic = time.time()
                update[self.fully_reduced_system_indices_full] = lu.solve(
                    self.fully_reduced_residual
                )

            elif linear_solver == "amg-potential":
                ml = pyamg.smoothed_aggregation_solver(
                    self.fully_reduced_jacobian, **ml_options
                )
                time_setup = time.time() - tic
                tic = time.time()
                update[self.fully_reduced_system_indices_full] = ml.solve(
                    self.fully_reduced_residual,
                    tol=tol_amg,
                    residuals=res_history_amg,
                )

            # Compute flux update
            update[self.flux_slice] = jacobian_flux_inv.dot(
                residual[self.flux_slice]
                + self.DT.dot(update[self.reduced_system_slice])
            )
            time_solve = time.time() - tic

        # Diagnostics
        if linear_solver in ["amg-flux-reduced", "amg-potential"]:
            if self.options.get("linear_solver_verbosity", False):
                num_amg_iter = len(res_history_amg)
                res_amg = res_history_amg[-1]
                print(ml)
                print(
                    f"#AMG iterations: {num_amg_iter}; Residual after AMG step: {res_amg}"
                )

        # Collect stats
        stats = [time_setup, time_solve]

        return update, residual, stats

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

        self.L = self.options.get("L", 1.0)
        """float: relaxation parameter, lower cut-off for the mobility"""

        # Setup
        tic = time.time()
        self.setup_infrastructure()

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

            # Newton step
            update_i, residual_i, stats_i = self.linearization_step(
                solution_i, rhs, iter
            )
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
            stats_i.append(time_anderson)

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
            cell_flux = self.face_to_cell(flat_flux)
            norm = np.linalg.norm(cell_flux, 2, axis=-1)
            cell_scaling = np.maximum(norm - shrink_factor, 0) / (
                norm + self.regularization
            )
            flat_scaling = self.cell_to_face(cell_scaling, mode="arithmetic")

        elif mode == "face_arithmetic":
            # Define natural vector valued flux on faces (taking arithmetic averages
            # of continuous fluxes over cells evaluated at faces)
            tangential_flux = self.orthogonal_face_average.dot(flat_flux)
            # Determine the l2 norm of the fluxes on the faces, add some regularization
            norm = np.sqrt(flat_flux**2 + tangential_flux**2)
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

        # Define linear solver to be used to invert the Darcy systems
        self.setup_infrastructure()
        linear_solver = self.options.get("linear_solver", "lu")
        assert linear_solver in [
            "lu",
            "lu-flux-reduced",
            "amg-flux-reduced",
            "lu-potential",
            "amg-potential",
        ], f"Linear solver {linear_solver} not supported."

        if linear_solver in ["amg-flux-reduced", "amg-potential"]:
            # TODO add possibility for user control
            ml_options = {
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
            tol_amg = self.options.get("linear_solver_tol", 1e-6)
            res_history_amg = []

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
        solution_i = np.zeros_like(rhs, dtype=float)

        # Solve linear Darcy problem as initial guess
        l_scheme_mixed_darcy = sps.bmat(
            [
                [self.L * self.mass_matrix_faces, -self.div.T, None],
                [self.div, None, -self.potential_constraint.T],
                [None, self.potential_constraint, None],
            ],
            format="csc",
        )
        if linear_solver == "lu":
            self.l_scheme_mixed_darcy_solver = sps.linalg.splu(l_scheme_mixed_darcy)
            solution_i = self.l_scheme_mixed_darcy_solver.solve(rhs)

        elif linear_solver in ["lu-flux-reduced", "amg-flux-reduced"]:
            # Solve potential-multiplier problem

            # Reduce flux block
            (
                self.reduced_jacobian,
                self.reduced_residual,
                jacobian_flux_inv,
            ) = self.remove_flux(l_scheme_mixed_darcy, rhs)

            if linear_solver == "lu-flux-reduced":
                self.l_scheme_mixed_darcy_solver = sps.linalg.splu(
                    self.reduced_jacobian
                )

            elif linear_solver == "amg-flux-reduced":
                self.l_scheme_mixed_darcy_solver = pyamg.smoothed_aggregation_solver(
                    self.reduced_jacobian, **ml_options
                )
                solution_i[
                    self.reduced_system_slice
                ] = self.l_scheme_mixed_darcy_solver.solve(
                    self.reduced_residual,
                    tol=tol_amg,
                    residuals=res_history_amg,
                )

            # Compute flux update
            solution_i[self.flux_slice] = jacobian_flux_inv.dot(
                rhs[self.flux_slice]
                + self.DT.dot(solution_i[self.reduced_system_slice])
            )

        elif linear_solver in ["lu-potential", "amg-potential"]:
            # Solve pure potential problem

            # Reduce flux block
            (
                self.reduced_jacobian,
                self.reduced_residual,
                jacobian_flux_inv,
            ) = self.remove_flux(l_scheme_mixed_darcy, rhs)

            # Reduce to pure pressure system
            (
                self.fully_reduced_jacobian,
                self.fully_reduced_residual,
            ) = self.remove_lagrange_multiplier(
                self.reduced_jacobian, self.reduced_residual, solution_i
            )

            if linear_solver == "lu-potential":
                self.l_scheme_mixed_darcy_solver = sps.linalg.splu(
                    self.fully_reduced_jacobian
                )
                solution_i[
                    self.fully_reduced_system_indices_full
                ] = self.l_scheme_mixed_darcy_solver.solve(self.fully_reduced_residual)

            elif linear_solver == "amg-potential":
                self.l_scheme_mixed_darcy_solver = pyamg.smoothed_aggregation_solver(
                    self.fully_reduced_jacobian, **ml_options
                )
                solution_i[
                    self.fully_reduced_system_indices_full
                ] = self.l_scheme_mixed_darcy_solver.solve(
                    self.fully_reduced_residual,
                    tol=tol_amg,
                    residuals=res_history_amg,
                )

            # Compute flux update
            solution_i[self.flux_slice] = jacobian_flux_inv.dot(
                rhs[self.flux_slice]
                + self.DT.dot(solution_i[self.reduced_system_slice])
            )

        else:
            raise NotImplementedError(f"Linear solver {linear_solver} not supported")

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
                new_flux = self.l_scheme_mixed_darcy_solver.solve(rhs_i)[
                    self.flux_slice
                ]
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
                new_flux = self.l_scheme_mixed_darcy_solver.solve(rhs_i)[
                    self.flux_slice
                ]
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
                solution_i = np.zeros_like(rhs_i, dtype=float)

                if linear_solver == "lu":
                    if update_solver:
                        self.l_scheme_mixed_darcy_solver = sps.linalg.splu(
                            l_scheme_mixed_darcy
                        )
                    solution_i = self.l_scheme_mixed_darcy_solver.solve(rhs_i)

                elif linear_solver in ["lu-flux-reduced", "amg-flux-reduced"]:
                    # Solve potential-multiplier problem

                    # Reduce flux block
                    (
                        self.reduced_jacobian,
                        self.reduced_residual,
                        jacobian_flux_inv,
                    ) = self.remove_flux(l_scheme_mixed_darcy, rhs_i)

                    if linear_solver == "lu-flux-reduced":
                        if update_solver:
                            self.l_scheme_mixed_darcy_solver = sps.linalg.splu(
                                self.reduced_jacobian
                            )
                        solution_i[
                            self.reduced_system_slice
                        ] = self.l_scheme_mixed_darcy_solver.solve(
                            self.reduced_residual
                        )

                    elif linear_solver == "amg-flux-reduced":
                        if update_solver:
                            self.l_scheme_mixed_darcy_solver = (
                                pyamg.smoothed_aggregation_solver(
                                    self.reduced_jacobian, **ml_options
                                )
                            )
                        solution_i[
                            self.reduced_system_slice
                        ] = self.l_scheme_mixed_darcy_solver.solve(
                            self.reduced_residual,
                            tol=tol_amg,
                            residuals=res_history_amg,
                        )

                    # Compute flux update
                    solution_i[self.flux_slice] = jacobian_flux_inv.dot(
                        rhs_i[self.flux_slice]
                        + self.DT.dot(solution_i[self.reduced_system_slice])
                    )

                elif linear_solver in ["lu-potential", "amg-potential"]:
                    # Solve pure potential problem

                    # Reduce flux block
                    (
                        self.reduced_jacobian,
                        self.reduced_residual,
                        jacobian_flux_inv,
                    ) = self.remove_flux(l_scheme_mixed_darcy, rhs_i)

                    # Reduce to pure pressure system
                    (
                        self.fully_reduced_jacobian,
                        self.fully_reduced_residual,
                    ) = self.remove_lagrange_multiplier(
                        self.reduced_jacobian, self.reduced_residual, solution_i
                    )

                    if linear_solver == "lu-potential":
                        if update_solver:
                            self.l_scheme_mixed_darcy_solver = sps.linalg.splu(
                                self.fully_reduced_jacobian
                            )
                        solution_i[
                            self.fully_reduced_system_indices_full
                        ] = self.l_scheme_mixed_darcy_solver.solve(
                            self.fully_reduced_residual
                        )

                    elif linear_solver == "amg-potential":
                        if update_solver:
                            self.l_scheme_mixed_darcy_solver = (
                                pyamg.smoothed_aggregation_solver(
                                    self.fully_reduced_jacobian, **ml_options
                                )
                            )
                        # time_setup = time.time() - tic
                        tic = time.time()
                        solution_i[
                            self.fully_reduced_system_indices_full
                        ] = self.l_scheme_mixed_darcy_solver.solve(
                            self.fully_reduced_residual,
                            tol=tol_amg,
                            residuals=res_history_amg,
                        )

                    # Compute flux update
                    solution_i[self.flux_slice] = jacobian_flux_inv.dot(
                        rhs_i[self.flux_slice]
                        + self.DT.dot(solution_i[self.reduced_system_slice])
                    )
                else:
                    raise NotImplementedError(
                        f"Linear solver {linear_solver} not supported"
                    )

                # Diagnostics
                if linear_solver in ["amg-flux-reduced", "amg-potential"]:
                    if self.options.get("linear_solver_verbosity", False):
                        num_amg_iter = len(res_history_amg)
                        res_amg = res_history_amg[-1]
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
            - plot_solution (bool): plot the solution. Defaults to False.
            - return_solution (bool): return the solution. Defaults to False.

    """
    if method.lower() in ["newton", "bregman"]:
        # Extract grid - implicitly assume mass_2 to generate same grid
        grid: darsia.Grid = darsia.generate_grid(mass_1)

        # Fetch options
        options = kwargs.get("options", {})
        plot_solution = kwargs.get("plot_solution", False)
        return_solution = kwargs.get("return_solution", False)

        if method.lower() == "newton":
            w1 = WassersteinDistanceNewton(grid, options)
        elif method.lower() == "bregman":
            w1 = WassersteinDistanceBregman(grid, options)
        return w1(
            mass_1, mass_2, plot_solution=plot_solution, return_solution=return_solution
        )

    elif method.lower() == "cv2.emd":
        preprocess = kwargs.get("preprocess")
        w1 = darsia.EMD(preprocess)
        return w1(mass_1, mass_2)

    else:
        raise NotImplementedError(f"Method {method} not implemented.")
