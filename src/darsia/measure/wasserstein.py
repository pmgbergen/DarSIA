"""Wasserstein distance computed using variational methods.

"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

import darsia


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
        shape: tuple,
        voxel_size: list,
        dim: int,
        options: dict = {},
    ) -> None:
        """
        Args:

            shape (tuple): shape of the image
            voxel_size (list): voxel size of the image
            dim (int): dimension of the problem
            options (dict): options for the solver
                - num_iter (int): maximum number of iterations. Defaults to 100.
                - tol (float): tolerance for the stopping criterion. Defaults to 1e-6.
                - L (float): parameter for the Bregman iteration. Defaults to 1.0.
                - regularization (float): regularization parameter for the Bregman
                    iteration. Defaults to 0.0.
                - depth (int): depth of the Anderson acceleration. Defaults to 0.
                - scaling (float): scaling of the fluxes in the plot. Defaults to 1.0.
                - lumping (bool): lump the mass matrix. Defaults to True.

        """
        # Cache geometrical infos
        self.shape = shape
        self.voxel_size = voxel_size
        self.dim = dim

        self.options = options
        self.regularization = self.options.get("regularization", 0.0)
        self.verbose = self.options.get("verbose", False)

        # Setup
        self._setup()

    def _setup(self) -> None:
        """Setup of fixed discretization"""

        # Define dimensions of the problem
        dim_cells = self.shape
        num_cells = np.prod(dim_cells)
        numbering_cells = np.arange(num_cells, dtype=int).reshape(dim_cells)

        # Consider only inner edges
        vertical_edges_shape = (self.shape[0], self.shape[1] - 1)
        horizontal_edges_shape = (self.shape[0] - 1, self.shape[1])
        num_edges_axis = [
            np.prod(vertical_edges_shape),
            np.prod(horizontal_edges_shape),
        ]
        num_edges = np.sum(num_edges_axis)

        # Define connectivity
        connectivity = np.zeros((num_edges, 2), dtype=int)
        connectivity[: num_edges_axis[0], 0] = np.ravel(
            numbering_cells[:, :-1]
        )  # left cells
        connectivity[: num_edges_axis[0], 1] = np.ravel(
            numbering_cells[:, 1:]
        )  # right cells
        connectivity[num_edges_axis[0] :, 0] = np.ravel(
            numbering_cells[:-1, :]
        )  # top cells
        connectivity[num_edges_axis[0] :, 1] = np.ravel(
            numbering_cells[1:, :]
        )  # bottom cells

        # Define sparse divergence operator, integrated over elements: flat_fluxes -> flat_mass
        div_data = np.concatenate(
            (
                self.voxel_size[0] * np.ones(num_edges_axis[0], dtype=float),
                self.voxel_size[1] * np.ones(num_edges_axis[1], dtype=float),
                -self.voxel_size[0] * np.ones(num_edges_axis[0], dtype=float),
                -self.voxel_size[1] * np.ones(num_edges_axis[1], dtype=float),
            )
        )
        div_row = np.concatenate(
            (
                connectivity[: num_edges_axis[0], 0],
                connectivity[num_edges_axis[0] :, 0],
                connectivity[: num_edges_axis[0], 1],
                connectivity[num_edges_axis[0] :, 1],
            )
        )
        div_col = np.tile(np.arange(num_edges, dtype=int), 2)
        self.div = sps.csr_matrix(
            (div_data, (div_row, div_col)), shape=(num_cells, num_edges)
        )

        # Define sparse mass matrix on cells: flat_mass -> flat_mass
        self.mass_matrix_cells = sps.diags(
            np.prod(self.voxel_size) * np.ones(num_cells, dtype=float)
        )

        # Define sparse mass matrix on edges: flat_fluxes -> flat_fluxes
        lumping = self.options.get("lumping", True)
        if lumping:
            self.mass_matrix_edges = sps.diags(
                np.prod(self.voxel_size) * np.ones(num_edges, dtype=float)
            )
        else:
            # Define connectivity: cell to face (only for inner cells)
            connectivity_cell_to_vertical_face = np.zeros((num_cells, 2), dtype=int)
            connectivity_cell_to_vertical_face[
                np.ravel(numbering_cells[:, :-1]), 0
            ] = np.arange(
                num_edges_axis[0]
            )  # left face
            connectivity_cell_to_vertical_face[
                np.ravel(numbering_cells[:, 1:]), 1
            ] = np.arange(
                num_edges_axis[0]
            )  # right face
            connectivity_cell_to_horizontal_face = np.zeros((num_cells, 2), dtype=int)
            connectivity_cell_to_horizontal_face[
                np.ravel(numbering_cells[:-1, :]), 0
            ] = np.arange(
                num_edges_axis[0], num_edges_axis[0] + num_edges_axis[1]
            )  # top face
            connectivity_cell_to_horizontal_face[
                np.ravel(numbering_cells[1:, :]), 1
            ] = np.arange(
                num_edges_axis[0], num_edges_axis[0] + num_edges_axis[1]
            )  # bottom face

            # Info about inner cells
            inner_cells_with_vertical_faces = np.ravel(numbering_cells[:, 1:-1])
            inner_cells_with_horizontal_faces = np.ravel(numbering_cells[1:-1, :])
            num_inner_cells_with_vertical_faces = len(inner_cells_with_vertical_faces)
            num_inner_cells_with_horizontal_faces = len(
                inner_cells_with_horizontal_faces
            )

            # Define true RT0 mass matrix on edges: flat_fluxes -> flat_fluxes
            mass_matrix_edges_data = np.prod(self.voxel_size) * np.concatenate(
                (
                    2 / 3 * np.ones(num_edges, dtype=float),  # all faces
                    1
                    / 6
                    * np.ones(
                        num_inner_cells_with_vertical_faces, dtype=float
                    ),  # left faces
                    1
                    / 6
                    * np.ones(
                        num_inner_cells_with_vertical_faces, dtype=float
                    ),  # right faces
                    1
                    / 6
                    * np.ones(
                        num_inner_cells_with_horizontal_faces, dtype=float
                    ),  # top faces
                    1
                    / 6
                    * np.ones(
                        num_inner_cells_with_horizontal_faces, dtype=float
                    ),  # bottom faces
                )
            )
            mass_matrix_edges_row = np.concatenate(
                (
                    np.arange(num_edges, dtype=int),
                    connectivity_cell_to_vertical_face[
                        inner_cells_with_vertical_faces, 0
                    ],
                    connectivity_cell_to_vertical_face[
                        inner_cells_with_vertical_faces, 1
                    ],
                    connectivity_cell_to_horizontal_face[
                        inner_cells_with_horizontal_faces, 0
                    ],
                    connectivity_cell_to_horizontal_face[
                        inner_cells_with_horizontal_faces, 1
                    ],
                )
            )
            mass_matrix_edges_col = np.concatenate(
                (
                    np.arange(num_edges, dtype=int),
                    connectivity_cell_to_vertical_face[
                        inner_cells_with_vertical_faces, 1
                    ],
                    connectivity_cell_to_vertical_face[
                        inner_cells_with_vertical_faces, 0
                    ],
                    connectivity_cell_to_horizontal_face[
                        inner_cells_with_horizontal_faces, 1
                    ],
                    connectivity_cell_to_horizontal_face[
                        inner_cells_with_horizontal_faces, 0
                    ],
                )
            )
            self.mass_matrix_edges = sps.csr_matrix(
                (
                    mass_matrix_edges_data,
                    (mass_matrix_edges_row, mass_matrix_edges_col),
                ),
                shape=(num_edges, num_edges),
            )

        # Utilities
        depth = self.options.get("depth", 0)
        self.anderson = (
            darsia.AndersonAcceleration(dimension=num_edges, depth=depth)
            if depth > 0
            else None
        )

        # TODO needs to be defined for each problem separately

        # Define sparse embedding operator for fluxes into full discrete DOF space
        self.flux_embedding = sps.csr_matrix(
            (
                np.ones(num_edges, dtype=float),
                (np.arange(num_edges), np.arange(num_edges)),
            ),
            shape=(num_edges + num_cells + 1, num_edges),
        )

        # Cache
        self.num_edges = num_edges
        self.num_cells = num_cells
        self.dim_cells = dim_cells
        self.numbering_cells = numbering_cells
        self.num_edges_axis = num_edges_axis
        self.vertical_edges_shape = vertical_edges_shape
        self.horizontal_edges_shape = horizontal_edges_shape

    def _problem_specific_setup(self, mass_diff: np.ndarray) -> None:
        """Resetup of fixed discretization"""

        # TODO can't we just fix some  cell, e.g., [0,0]. Move then this to the above.

        # Fix index of dominating contribution in image differece
        self.constrained_cell_flat_index = np.argmax(np.abs(mass_diff))
        self.pressure_constraint = sps.csr_matrix(
            (
                np.ones(1, dtype=float),
                (np.zeros(1, dtype=int), np.array([self.constrained_cell_flat_index])),
            ),
            shape=(1, self.num_cells),
            dtype=float,
        )

        # Linear part of the operator.
        self.broken_darcy = sps.bmat(
            [
                [None, -self.div.T, None],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ]
        )

    def split_solution(
        self, solution: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Split the solution into fluxes, pressure and lagrange multiplier.

        Args:
            solution (np.ndarray): solution

        Returns:
            tuple: fluxes, pressure, lagrange multiplier

        """
        # Split the solution
        flat_flux = solution[: self.num_edges]
        flat_pressure = solution[self.num_edges : self.num_edges + self.num_cells]
        flat_lagrange_multiplier = solution[-1]

        return flat_flux, flat_pressure, flat_lagrange_multiplier

    # ! ---- Projections inbetween faces and cells ----

    def cell_reconstruction(self, flat_flux: np.ndarray) -> np.ndarray:
        """Reconstruct the fluxes on the cells from the fluxes on the faces.

        Args:
            flat_flux (np.ndarray): flat fluxes (normal fluxes on the faces)

        Returns:
            np.ndarray: cell-based vectorial fluxes

        """
        # TODO replace by sparse matrix multiplication

        # Reshape fluxes - use duality of faces and normals
        horizontal_fluxes = flat_flux[: self.num_edges_axis[0]].reshape(
            self.vertical_edges_shape
        )
        vertical_fluxes = flat_flux[self.num_edges_axis[0] :].reshape(
            self.horizontal_edges_shape
        )

        # Determine a cell-based Raviart-Thomas reconstruction of the fluxes
        cell_flux = np.zeros((*self.dim_cells, self.dim), dtype=float)
        # Horizontal fluxes
        cell_flux[:, :-1, 0] += 0.5 * horizontal_fluxes
        cell_flux[:, 1:, 0] += 0.5 * horizontal_fluxes
        # Vertical fluxes
        cell_flux[:-1, :, 1] += 0.5 * vertical_fluxes
        cell_flux[1:, :, 1] += 0.5 * vertical_fluxes

        return cell_flux

    def face_restriction(self, cell_flux: np.ndarray) -> np.ndarray:
        """Restrict the fluxes on the cells to the faces.

        Args:
            cell_flux (np.ndarray): cell-based fluxes

        Returns:
            np.ndarray: face-based fluxes

        """
        # TODO replace by sparse matrix multiplication

        # Determine the fluxes on the faces
        horizontal_fluxes = 0.5 * (cell_flux[:, :-1, 0] + cell_flux[:, 1:, 0])
        vertical_fluxes = 0.5 * (cell_flux[:-1, :, 1] + cell_flux[1:, :, 1])

        # Reshape the fluxes
        flat_flux = np.concatenate(
            [horizontal_fluxes.ravel(), vertical_fluxes.ravel()], axis=0
        )

        return flat_flux

    def face_restriction_scalar(self, cell_qty: np.ndarray) -> np.ndarray:
        """Restrict the fluxes on the cells to the faces.

        Args:
            cell_qty (np.ndarray): cell-based quantity

        Returns:
            np.ndarray: face-based quantity

        """
        # Determine the fluxes on the faces

        horizontal_face_qty = 0.5 * (cell_qty[:, :-1] + cell_qty[:, 1:])
        vertical_face_qty = 0.5 * (cell_qty[:-1, :] + cell_qty[1:, :])

        # Reshape the fluxes - hardcoding the connectivity here
        face_qty = np.concatenate(
            [horizontal_face_qty.ravel(), vertical_face_qty.ravel()]
        )

        return face_qty

    # ! ---- Effective quantities ----

    def effective_mobility(self, flat_flux: np.ndarray) -> np.ndarray:
        """Compute the effective mobility of the solution.

        Args:
            flat_flux (np.ndarray): flat fluxes

        Returns:
            np.ndarray: effective mobility
        """
        # TODO Use improved quadrature?
        cell_flux = self.cell_reconstruction(flat_flux)
        return np.linalg.norm(cell_flux, 2, axis=-1)

    def l1_dissipation(self, solution: np.ndarray) -> float:
        """Compute the l1 dissipation potential of the solution.

        Args:
            solution (np.ndarray): solution

        Returns:
            float: l1 dissipation potential

        """
        # TODO use improved quadrature?
        flat_flux, _, _ = self.split_solution(solution)
        cell_flux = self.cell_reconstruction(flat_flux)
        return np.sum(np.prod(self.voxel_size) * np.linalg.norm(cell_flux, 2, axis=-1))

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
        self._problem_specific_setup(mass_diff)

        # Main method
        distance, solution, status = self._solve(flat_mass_diff)

        # Split the solution
        flat_flux, flat_pressure, _ = self.split_solution(solution)

        # Reshape the fluxes and pressure
        flux = self.cell_reconstruction(flat_flux)
        pressure = flat_pressure.reshape(self.dim_cells)

        # Determine effective mobility
        mobility = self.effective_mobility(flat_flux)

        # Stop taking time
        toc = time.time()
        status["elapsed_time"] = toc - tic

        # Plot the solution
        if plot_solution:
            self._plot_solution(mass_diff, flux, pressure, mobility)

        if return_solution:
            return distance, flux, pressure, mobility, status
        else:
            return distance

    def _plot_solution(
        self,
        mass_diff: np.ndarray,
        flux: np.ndarray,
        pressure: np.ndarray,
        mobility: np.ndarray,
    ) -> None:
        # Meshgrid
        Y, X = np.meshgrid(
            self.voxel_size[0] * (0.5 + np.arange(self.shape[0] - 1, -1, -1)),
            self.voxel_size[1] * (0.5 + np.arange(self.shape[1])),
            indexing="ij",
        )

        scaling = self.options.get("scaling", 1.0)

        # Plot the fluxes and pressure
        plt.figure("Beckman solution")
        plt.pcolormesh(X, Y, pressure, cmap="turbo")
        plt.colorbar()
        plt.quiver(
            X,
            Y,
            scaling * flux[:, :, 0],
            -scaling * flux[:, :, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.5,
        )

        plt.figure("Beckman solution fluxes")
        plt.pcolormesh(X, Y, mass_diff, cmap="turbo")
        plt.colorbar()
        plt.quiver(
            X,
            Y,
            scaling * flux[:, :, 0],
            -scaling * flux[:, :, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.5,
        )

        plt.figure("Beckman solution mobility")
        plt.pcolormesh(X, Y, mobility, cmap="turbo")
        plt.colorbar()

        plt.show()


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
        flat_flux, _, _ = self.split_solution(solution)
        cell_flux = self.cell_reconstruction(flat_flux)
        cell_flux_norm = np.maximum(
            np.linalg.norm(cell_flux, 2, axis=-1), self.regularization
        )
        cell_flux_normed = cell_flux / cell_flux_norm[..., None]
        flat_flux_normed = self.face_restriction(cell_flux_normed)
        return (
            rhs
            - self.broken_darcy.dot(solution)
            - self.flux_embedding.dot(self.mass_matrix_edges.dot(flat_flux_normed))
        )

    def jacobian_lu(self, solution: np.ndarray) -> sps.linalg.splu:
        """Compute the LU factorization of the jacobian of the solution.

        Args:
            solution (np.ndarray): solution

        Returns:
            sps.linalg.splu: LU factorization of the jacobian

        """
        flat_flux, _, _ = self.split_solution(solution)
        cell_flux = self.cell_reconstruction(flat_flux)
        self.regularization = self.options.get("regularization", 0.0)
        cell_flux_norm = np.maximum(
            np.linalg.norm(cell_flux, 2, axis=-1), self.regularization
        )
        flat_flux_norm = self.face_restriction_scalar(cell_flux_norm)
        approx_jacobian = sps.bmat(
            [
                [
                    sps.diags(np.maximum(self.L, 1.0 / flat_flux_norm), dtype=float)
                    * self.mass_matrix_edges,
                    -self.div.T,
                    None,
                ],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ]
        )
        approx_jacobian_lu = sps.linalg.splu(approx_jacobian)
        return approx_jacobian_lu

    def _solve(self, flat_mass_diff):
        # Observation: AA can lead to less stagnation, more accurate results, and therefore
        # better solutions to mu and u. Higher depth is better, but more expensive.

        # Solver parameters
        num_iter = self.options.get("num_iter", 100)
        tol = self.options.get("tol", 1e-6)
        tol_distance = self.options.get("tol_distance", 1e-6)
        self.L = self.options.get("L", 1.0)

        # Define right hand side
        rhs = np.concatenate(
            [
                np.zeros(self.num_edges, dtype=float),
                self.mass_matrix_cells.dot(flat_mass_diff),
                np.zeros(1, dtype=float),
            ]
        )

        # Initialize solution
        solution_i = np.zeros_like(rhs)

        # Newton iteration
        for iter in range(num_iter):
            # Keep track of old flux, and old distance
            old_solution_i = solution_i.copy()
            old_distance = self.l1_dissipation(solution_i)

            # Newton step
            if iter == 0:
                residual_i = (
                    rhs.copy()
                )  # Aim at Darcy-like initial guess after first iteration.
            else:
                residual_i = self.residual(rhs, solution_i)
            jacobian_lu = self.jacobian_lu(solution_i)
            update_i = jacobian_lu.solve(residual_i)
            solution_i += update_i

            # Apply Anderson acceleration to flux contribution (the only nonlinear part).
            if self.anderson is not None:
                solution_i[: self.num_edges] = self.anderson(
                    solution_i[: self.num_edges], update_i[: self.num_edges], iter
                )
                # TODO try for full solution

            # Update distance
            new_distance = self.l1_dissipation(solution_i)

            # Compute the error:
            # - residual
            # - residual of mass conservation equation
            # - increment
            # - flux increment
            error = [
                np.linalg.norm(residual_i, 2),
                np.linalg.norm(residual_i[self.num_edges : -1], 2),
                np.linalg.norm(solution_i - old_solution_i, 2),
                np.linalg.norm((solution_i - old_solution_i)[: self.num_edges], 2),
            ]

            if self.verbose:
                print(
                    "Newton iteration",
                    iter,
                    new_distance,
                    old_distance - new_distance,
                    error[0],  # residual
                    error[1],  # mass conservation residual
                    error[2],  # full increment
                    error[3],  # flux increment
                )

            # Stopping criterion
            # TODO include criterion build on staganation of the solution
            # TODO include criterion on distance.
            if (
                iter > 1
                and min([error[0], error[2]]) < tol
                or abs(new_distance - old_distance) < tol_distance
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
        }

        return new_distance, solution_i, status


class WassersteinDistanceBregman(VariationalWassersteinDistance):
    def _problem_specific_setup(self, mass_diff: np.ndarray) -> None:
        super()._problem_specific_setup(mass_diff)
        self.L = self.options.get("L", 1.0)
        l_scheme_mixed_darcy = sps.bmat(
            [
                [self.L * self.mass_matrix_edges, -self.div.T, None],
                [self.div, None, -self.pressure_constraint.T],
                [None, self.pressure_constraint, None],
            ]
        )
        self.l_scheme_mixed_darcy_lu = sps.linalg.splu(l_scheme_mixed_darcy)

    def _solve(self, flat_mass_diff):
        # Solver parameters
        num_iter = self.options.get("num_iter", 100)
        tol = self.options.get("tol", 1e-6)
        self.L = self.options.get("L", 1.0)

        rhs = np.concatenate(
            [
                np.zeros(self.num_edges, dtype=float),
                self.mass_matrix_cells.dot(flat_mass_diff),
                np.zeros(1, dtype=float),
            ]
        )

        # Keep track of how often the distance increases.
        num_neg_diff = 0

        # Bregman iterations
        solution_i = np.zeros_like(rhs)
        for iter in range(num_iter):
            old_distance = self.l1_dissipation(solution_i)

            # 1. Solve linear system with trust in current flux.
            flat_flux_i, _, _ = self.split_solution(solution_i)
            rhs_i = rhs.copy()
            rhs_i[: self.num_edges] = self.L * self.mass_matrix_edges.dot(flat_flux_i)
            intermediate_solution_i = self.l_scheme_mixed_darcy_lu.solve(rhs_i)

            # 2. Shrink step for vectorial fluxes. To comply with the RT0 setting, the
            # shrinkage operation merely determines the scalar. We still aim at
            # following along the direction provided by the vectorial fluxes.
            intermediate_flat_flux_i, _, _ = self.split_solution(
                intermediate_solution_i
            )
            # new_flat_flux_i = np.sign(intermediate_flat_flux_i) * (
            #    np.maximum(np.abs(intermediate_flat_flux_i) - 1.0 / self.L, 0.0)
            # )
            cell_intermediate_flux_i = self.cell_reconstruction(
                intermediate_flat_flux_i
            )
            norm = np.linalg.norm(cell_intermediate_flux_i, 2, axis=-1)
            cell_scaling = np.maximum(norm - 1 / self.L, 0) / (
                norm + self.regularization
            )
            flat_scaling = self.face_restriction_scalar(cell_scaling)
            new_flat_flux_i = flat_scaling * intermediate_flat_flux_i

            # Apply Anderson acceleration to flux contribution (the only nonlinear part).
            if self.anderson is not None:
                flux_inc = new_flat_flux_i - flat_flux_i
                new_flat_flux_i = self.anderson(new_flat_flux_i, flux_inc, iter)

            # Measure error in terms of the increment of the flux
            flux_diff = np.linalg.norm(new_flat_flux_i - flat_flux_i, 2)

            # Update flux solution
            solution_i = intermediate_solution_i.copy()
            solution_i[: self.num_edges] = new_flat_flux_i

            # Update distance
            new_distance = self.l1_dissipation(solution_i)

            # Determine the error in the mass conservation equation
            mass_conservation_residual = np.linalg.norm(
                (rhs_i - self.broken_darcy.dot(solution_i))[self.num_edges : -1], 2
            )

            # TODO include criterion build on staganation of the solution
            # TODO include criterion on distance.

            # Print status
            if self.verbose:
                print(
                    "Bregman iteration",
                    iter,
                    new_distance,
                    old_distance - new_distance,
                    self.L,
                    flux_diff,
                    mass_conservation_residual,
                )

            ## Check stopping criterion # TODO. What is a good stopping criterion?
            # if iter > 1 and (flux_diff < tol or mass_conservation_residual < tol:
            #    break

            # Keep track if the distance increases.
            if new_distance > old_distance:
                num_neg_diff += 1

            # Increase L if stagnating of the distance increases too often.
            # TODO restart anderson acceleration
            update_l = self.options.get("update_l", True)
            if update_l:
                tol_distance = self.options.get("tol_distance", 1e-12)
                max_iter_increase_diff = self.options.get("max_iter_increase_diff", 20)
                l_factor = self.options.get("l_factor", 2)
                if (
                    abs(new_distance - old_distance) < tol_distance
                    or num_neg_diff > max_iter_increase_diff
                ):
                    # Update L
                    self.L = self.L * l_factor

                    # Update linear system
                    l_scheme_mixed_darcy = sps.bmat(
                        [
                            [self.L * self.mass_matrix_edges, -self.div.T, None],
                            [self.div, None, -self.pressure_constraint.T],
                            [None, self.pressure_constraint, None],
                        ]
                    )
                    self.l_scheme_mixed_darcy_lu = sps.linalg.splu(l_scheme_mixed_darcy)

                    # Reset stagnation counter
                    num_neg_diff = 0

                L_max = self.options.get("L_max", 1e8)
                if self.L > L_max:
                    break

        # Define performance metric
        status = {
            "converged": iter < num_iter,
            "number iterations": iter,
            "distance": new_distance,
            "residual mass conservation": mass_conservation_residual,
            "flux increment": flux_diff,
            "distance increment": abs(new_distance - old_distance),
        }

        return new_distance, solution_i, status
