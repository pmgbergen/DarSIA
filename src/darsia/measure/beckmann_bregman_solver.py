"""Wasserstein distance computed using the split Bregman method."""

from __future__ import annotations

import time
import tracemalloc
import warnings
from typing import Optional, override

import numpy as np
import scipy.sparse as sps

import darsia


class BeckmannBregmanSolver(darsia.BeckmannProblem):
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
        shrink_factor: float | np.ndarray,
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

    @override
    def solve_beckmann_problem(
        self, flat_mass_diff: np.ndarray
    ) -> tuple[float, np.ndarray, dict]:
        """Solve the Beckmann problem using the Bregman method.

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
        self.flux = np.zeros(self.grid.num_faces, dtype=float)
        self.pressure = np.zeros(self.grid.num_cells, dtype=float)

        # Initialize distance in case below iteration fails
        new_distance = 0

        # Initialize container for storing the convergence history
        self.convergence_history = {
            "distance": [],
            "mass_conservation_residual": [],
            "aux_force_increment": [],
            "distance_increment": [],
            "timing": [],
            "run_time": [],
        }
        convergence_history = self.convergence_history

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
            if True:  # try:
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
                    self.flux[:] = flux[:]
                    self.pressure = solution_i[self.pressure_slice]
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
                    self.flux[:] = flux[:]
                    self.pressure[:] = solution_i[self.pressure_slice]
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
                self.flux = flux
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

                if self.callbacks is not None:
                    for callback in self.callbacks:
                        callback(self)

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
                        convergence_history["aux_force_increment"][-1]
                        < tol_increment * convergence_history["aux_force_increment"][0]
                        and convergence_history["distance_increment"][-1] / new_distance
                        < tol_distance
                        and convergence_history["mass_conservation_residual"][-1]
                        < tol_residual
                    ):
                        break

                # Update Bregman variables
                old_aux_flux = new_aux_flux.copy()
                old_force = new_force.copy()
                old_distance = new_distance

            else:  # except Exception:
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

        # Compute l1 norm of the flux
        unweighted_transport_density = self.transport_density(flux, weighted=False)
        flux_l1_norm = self.mass_matrix_cells.dot(unweighted_transport_density).sum()

        # Define performance metric
        info = {
            "distance": new_distance,  # includes weight
            "flux_l1_norm": flux_l1_norm,  # without weight
            "converged": iter < num_iter - 1,
            "number_iterations": iter,
            "convergence_history": convergence_history,
            "timings": total_timings,
            "peak_memory_consumption": peak_memory_consumption,
        }

        return new_distance, solution_i, info
