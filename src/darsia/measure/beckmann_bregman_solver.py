"""Wasserstein distance computed using the split Bregman method."""

from __future__ import annotations

import time
import tracemalloc
import warnings
from typing import Optional, override
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sps

import darsia


@dataclass
class _ConvergenceHistory:
    """Class to store the convergence history of the Bregman iteration.

    Not intended for use outside of BeckmannBregmanSolver.

    """

    distance: list[float] = field(default_factory=list)
    mass_conservation_residual: list[float] = field(default_factory=list)
    aux_force_increment: list[float] = field(default_factory=list)
    distance_increment: list[float] = field(default_factory=list)
    timings: list[dict] = field(default_factory=list)
    total_run_time: list[float] = field(default_factory=list)

    def append(
        self,
        distance: float,
        distance_increment: float,
        aux_force_increment: float,
        mass_conservation_residual: float,
        timings: dict,
        total_run_time: float,
    ) -> None:
        self.distance.append(distance)
        self.distance_increment.append(distance_increment)
        self.aux_force_increment.append(aux_force_increment)
        self.mass_conservation_residual.append(mass_conservation_residual)
        self.timings.append(timings)
        self.total_run_time.append(total_run_time)

    def as_dict(self) -> dict:
        return {
            "distance": self.distance,
            "distance_increment": self.distance_increment,
            "aux_force_increment": self.aux_force_increment,
            "mass_conservation_residual": self.mass_conservation_residual,
            "timings": self.timings,
            "total_run_time": self.total_run_time,
        }


class BeckmannBregmanSolver(darsia.BeckmannProblem):
    """Class to determine the Wasserstein distance solved with the Bregman method.

    Implements the :class:`darsia.BeckmannProblem` interface by specifying the method
    :meth:`darsia.BeckmannProblem.solve_beckmann_problem`.

    """

    def __init__(
        self,
        grid: darsia.Grid,
        weight: Optional[darsia.Image] = None,
        options: dict = {},
    ) -> None:
        """Initialize the Bregman method.

        Args:
            grid (darsia.Grid): grid
            weight (darsia.Image, optional): weight for the heterogeneous case.
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

    def force_view(self, vector: np.ndarray) -> np.ndarray:
        """View into the force part of the solution vector.

        Args:
            vector (np.ndarray): solution vector

        Returns:
            np.ndarray: view into the force part of the solution vector

        """
        return vector[self.force_slice]

    def _shrink(
        self,
        flux: np.ndarray,
        shrink_factor: float | np.ndarray,
    ) -> np.ndarray:
        """Shrink operation in the split Bregman method, operating on fluxes.

        To comply with the RT0 setting, the shrinkage operation merely determines the
        scalar. We still aim at following along the direction provided by the vectorial
        fluxes.

        Args:
            flux (np.ndarray): flux
            shrink_factor (float or np.ndarray): shrink factor

        Returns:
            np.ndarray: shrunk fluxes

        """
        _, face_weights_inv = self._compute_face_weight(flux)
        scaling = np.maximum(face_weights_inv - shrink_factor, 0) / (
            face_weights_inv + self.regularization
        )
        return scaling * flux

    def _compute_heterogeneous_bregman_regularization(self, flux: np.ndarray) -> tuple:
        """Update the regularization based on the current approximation of the flux.

        Args:
            flux (np.ndarray): flux

        Returns:
            tuple: l_scheme_mixed_darcy, weight, shrink_factor

        """

        # Assign the weight and shrink factor
        face_weights, face_weights_inv = self._compute_face_weight(flux)
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
        solution = np.zeros_like(rhs, dtype=float)
        solution, _ = self.linear_solve(self.darcy_init.copy(), rhs.copy(), solution)

        # Initialize distance in case below iteration fails
        distance = 0

        # Initialize container for storing the convergence history
        convergence_history = _ConvergenceHistory()

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
        flux = self.flux_view(solution)
        old_aux_flux = self._shrink(flux, shrink_factor)
        old_force = flux - old_aux_flux
        old_distance = self.l1_dissipation(flux)

        iter = 0

        # Control the update of the Bregman weight
        bregman_update = self.options.get("bregman_update", lambda iter: False)

        for iter in range(num_iter):
            # It is possible that the linear solver fails. In this case, we simply
            # stop the iteration and return the current solution.
            if True:  # try:
                # (Possibly) update the regularization, based on the current approximation
                # of the flux - use the inverse of the norm of the flux
                update_solver = bregman_update(iter)
                if update_solver:
                    # 0. Compute regularization
                    tic = time.time()
                    (
                        l_scheme_mixed_darcy,
                        weight,
                        shrink_factor,
                    ) = self._compute_heterogeneous_bregman_regularization(flux)
                    # 1. Make relaxation step (solve quadratic optimization problem)
                    # Here, re-initialize the aux flux and force with zero values again.
                    rhs_copy = rhs.copy()  # may change during direct solve
                    time_assemble = time.time() - tic
                    # Force to update the internally stored linear solver
                    tic = time.time()
                    solution, timings = self.linear_solve(
                        l_scheme_mixed_darcy,
                        rhs_copy,
                        reuse_solver=False,
                    )
                    flux = self.flux_view(solution)
                    timings["time_solve"] = time.time() - tic
                    timings["time_assemble"] = time_assemble

                    # 2. Shrink step for vectorial fluxes.
                    tic = time.time()
                    new_aux_flux = self._shrink(flux, shrink_factor)
                    timings["time_shrink"] = time.time() - tic

                    # 3. Update force
                    tic = time.time()
                    new_force = flux - new_aux_flux

                else:
                    # 1. Make relaxation step (solve quadratic optimization problem)
                    tic = time.time()
                    rhs_copy = rhs.copy()
                    rhs_copy[self.flux_slice] = weight @ self.mass_matrix_faces.dot(
                        old_aux_flux - old_force
                    )
                    time_assemble = time.time() - tic
                    # Force to update the internally stored linear solver
                    tic = time.time()
                    solution, timings = self.linear_solve(
                        l_scheme_mixed_darcy,
                        rhs_copy,
                        reuse_solver=iter > 0,
                    )
                    flux = self.flux_view(solution)
                    timings["time_solve"] = time.time() - tic
                    timings["time_assemble"] = time_assemble

                    # 2. Shrink step for vectorial fluxes.
                    tic = time.time()
                    new_aux_flux = self._shrink(flux + old_force, shrink_factor)
                    timings["time_shrink"] = time.time() - tic

                    # 3. Update force
                    tic = time.time()
                    new_force = old_force + flux - new_aux_flux

                # Apply Anderson acceleration to flux contribution (the only nonlinear part).
                if self.anderson is not None:
                    tic = time.time()
                    # Prepare increments
                    aux_inc = new_aux_flux - old_aux_flux
                    force_inc = new_force - old_force
                    inc = np.concatenate([aux_inc, force_inc])
                    iteration = np.concatenate([new_aux_flux, new_force])
                    # Apply Anderson acceleration
                    new_iteration = self.anderson(iteration, inc, iter)
                    # Split into flux and auxiliary variable
                    new_aux_flux = new_iteration[: self.grid.num_faces]
                    new_force = new_iteration[self.grid.num_faces :]
                    timings["time_acceleration"] = time.time() - tic
                else:
                    timings["time_acceleration"] = 0.0

                # Update distance
                distance = self.l1_dissipation(flux)

                # Catch nan values
                if np.isnan(distance):
                    info = {
                        "converged": False,
                        "number_iterations": iter,
                        "convergence_history": convergence_history.as_dict(),
                    }
                    return distance, solution, info

                # Determine the error in the mass conservation equation
                mass_conservation_residual = self.div.dot(flux) - self.pressure_view(
                    rhs
                )

                # Reference values
                flux_ref = np.linalg.norm(flux, 2)
                mass_ref = np.linalg.norm(self.pressure_view(rhs), 2)

                # Compute the error and store as part of the convergence history:
                # 0 - aux/force increments (fixed-point formulation)
                aux_increment = new_aux_flux - old_aux_flux
                force_increment = new_force - old_force
                relative_bregman_increment = (
                    np.linalg.norm(np.concatenate([aux_increment, force_increment]), 2)
                    / flux_ref
                )
                # 1 - distance increment (minimization formulation)
                distance_increment = abs(distance - old_distance)
                # 2 - mass conservation residual (constraint in optimization formulation)
                relative_mass_residual = (
                    np.linalg.norm(mass_conservation_residual, 2) / mass_ref
                )

                # Update convergence history
                convergence_history.append(
                    distance,
                    distance_increment,
                    relative_bregman_increment,
                    relative_mass_residual,
                    timings,
                    np.nan,  # placeholder for total run time
                )
                # Update total run time
                current_total_run_time = self._sum_timings(convergence_history.timings)[
                    "total"
                ]
                convergence_history.total_run_time[-1] = current_total_run_time

                # Print status
                if self.verbose:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message="overflow encountered"
                        )
                        relative_distance_increment = (
                            convergence_history.distance_increment[-1] / distance
                        )
                        relative_aux_force_increment = (
                            convergence_history.aux_force_increment[-1]
                            / convergence_history.aux_force_increment[0]
                        )
                        absolute_mass_conservation_residual_val = (
                            convergence_history.mass_conservation_residual[-1]
                        )
                        print(
                            f"Iter. {iter} \t|"
                            f"{distance:.6e} \t| "
                            f"{relative_distance_increment:.6e} \t| "
                            f"{relative_aux_force_increment:.6e} \t| "
                            f"{absolute_mass_conservation_residual_val:.6e}"
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
                        convergence_history.aux_force_increment[-1]
                        < tol_increment * convergence_history.aux_force_increment[0]
                        and convergence_history.distance_increment[-1] / distance
                        < tol_distance
                        and convergence_history.mass_conservation_residual[-1]
                        < tol_residual
                    ):
                        break

                # Update Bregman variables
                old_aux_flux = new_aux_flux.copy()
                old_force = new_force.copy()
                old_distance = distance

            else:  # except Exception:
                warnings.warn("Bregman iteration abruptly stopped due to some error.")
                break

        # Solve for the pressure by solving a single Newton iteration
        newton_jacobian, _, _ = self._compute_heterogeneous_bregman_regularization(flux)
        solution = np.zeros_like(rhs)
        solution[self.flux_slice] = flux.copy()
        newton_residual = self.optimality_conditions(rhs, solution)
        newton_update, _ = self.linear_solve(newton_jacobian, newton_residual, solution)
        solution[self.pressure_slice] = self.pressure_view(newton_update)

        # Summarize profiling (time in seconds, memory in GB)
        total_timings = self._sum_timings(convergence_history.timings)
        peak_memory_consumption = tracemalloc.get_traced_memory()[1] / 10**9

        # Compute l1 norm of the flux
        unweighted_transport_density = self.transport_density(flux, weighted=False)
        flux_l1_norm = self.mass_matrix_cells.dot(unweighted_transport_density).sum()

        # Define performance metric
        info = {
            "distance": distance,  # includes weight
            "flux_l1_norm": flux_l1_norm,  # without weight
            "converged": iter < num_iter - 1,
            "number_iterations": iter,
            "convergence_history": convergence_history.as_dict(),
            "timings": total_timings,
            "peak_memory_consumption": peak_memory_consumption,
        }

        return distance, solution, info
