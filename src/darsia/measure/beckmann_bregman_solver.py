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


@dataclass
class _ConvergenceCriteria:
    """Class to store and check the convergence criteria for the Bregman solver.

    Not intended for use outside of BeckmannBregmanSolver.

    Base stopping citeria on the different interpretations of the split Bregman method:
    - fixed-point formulation: aux flux and force increment
    - minimization formulation: distance increment
    - constrained optimization formulation: mass conservation residual
    """

    num_iter: int = 100
    """Maximum number of iterations."""
    tol_increment: float = np.finfo(float).max
    """Tolerance for the aux/force increment."""
    tol_distance: float = np.finfo(float).max
    """Tolerance for the distance increment."""
    tol_residual: float = np.finfo(float).max
    """Tolerance for the mass conservation residual."""

    def check_convergence_status(
        self,
        iter: int,
        aux_force_increment: float,
        distance_increment: float,
        mass_conservation_residual: float,
    ) -> darsia.ConvergenceStatus:
        """Check if convergence criteria are met."""
        max_iterations_reached = self._check_iterations(iter)
        convergence_achieved = self._check_convergence(
            aux_force_increment,
            distance_increment,
            mass_conservation_residual,
        )
        if convergence_achieved:
            return darsia.ConvergenceStatus.CONVERGED
        elif max_iterations_reached:
            return darsia.ConvergenceStatus.NOT_CONVERGED
        else:
            return darsia.ConvergenceStatus.RUNNING

    def _check_iterations(self, iter: int) -> bool:
        """Check if the maximum number of iterations is reached."""
        return iter >= self.num_iter

    def _check_convergence(
        self,
        aux_force_increment: float,
        distance_increment: float,
        mass_conservation_residual: float,
    ) -> bool:
        """Check if convergence criteria are met."""
        return (
            aux_force_increment < self.tol_increment
            and distance_increment < self.tol_distance
            and mass_conservation_residual < self.tol_residual
        )


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
            options (dict, optional): options for the Bregman solver. Defaults to {}.
                - L (float): penalty parameter for the Bregman iteration, associated to
                  face mobility. Defaults to 1.0.
                - bregman_update (lambda iter: bool): function to determine whether/when
                    to update the Bregman regularization. Defaults to a function that
                    never updates.
                - num_iter (int): maximum number of iterations. Defaults to 100.
                - tol_residual (float): tolerance for the relative mass conservation residual.
                    Defaults to np.finfo(float).max.
                - tol_increment (float): tolerance for the relative aux/force increment.
                    Defaults to np.finfo(float).max.
                - tol_distance (float): tolerance for the relative distance increment.
                    Defaults to np.finfo(float).max.

        """

        super().__init__(grid, weight, options)

        self.L = options.get("L", 1.0)
        """Penalty parameter for the Bregman iteration, associated to face mobility."""

        self.bregman_update = options.get("bregman_update", lambda iter: False)
        """Function to determine whether/when to update the Bregman regularization."""

        self.convergence_criteria = _ConvergenceCriteria(
            num_iter=options.get("num_iter", 100),
            tol_increment=options.get("tol_increment", np.finfo(float).max),
            tol_distance=options.get("tol_distance", np.finfo(float).max),
            tol_residual=options.get("tol_residual", np.finfo(float).max),
        )
        """Convergence criteria for the Bregman solver."""

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
                """Bregman iter. \t| """
                """W^1 \t\t| """
                """Δ W^1/W^1 \t| """
                """Δ aux/force / flux \t| """
                """mass residual / mass"""
                """\n"""
                """---------------|"""
                """---------------|"""
                """---------------|"""
                """---------------|"""
                """---------------"""
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

        for iter in range(self.convergence_criteria.num_iter):
            # Check whether to update the Bregman regularization, or to reuse it from
            # the previous iteration.
            if self.bregman_update(iter):
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
                tic = time.time()
                try:
                    solution, timings = self.linear_solve(
                        l_scheme_mixed_darcy,
                        rhs_copy,
                        reuse_solver=False,  # force to update solver
                    )
                except Exception:
                    warnings.warn(
                        "Bregman iteration abruptly stopped due to some error."
                    )
                    break
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
                tic = time.time()
                try:
                    solution, timings = self.linear_solve(
                        l_scheme_mixed_darcy,
                        rhs_copy,
                        reuse_solver=iter > 0,  # force to reuse solver
                    )
                except Exception:
                    warnings.warn(
                        "Bregman iteration abruptly stopped due to some error."
                    )
                    break
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
                bregman_increment = np.concatenate([aux_inc, force_inc])
                bregman_iteration = np.concatenate([new_aux_flux, new_force])
                # Apply Anderson acceleration
                aa_bregman_iteration = self.anderson(
                    bregman_iteration, bregman_increment, iter
                )
                # Split into flux and auxiliary variable
                new_aux_flux = aa_bregman_iteration[: self.grid.num_faces]
                new_force = aa_bregman_iteration[self.grid.num_faces :]
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

            # Reference values
            flux_norm = np.linalg.norm(flux, 2)
            mass_diff_norm = np.linalg.norm(self.pressure_view(rhs), 2)

            # Compute the relative errors - supress overflow warnings.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="overflow encountered")

                # 0 - aux/force increments (fixed-point formulation)
                aux_increment = new_aux_flux - old_aux_flux
                force_increment = new_force - old_force
                absolute_bregman_increment = np.linalg.norm(
                    np.concatenate([aux_increment, force_increment]), 2
                )
                relative_bregman_increment = absolute_bregman_increment / flux_norm

                # 1 - distance increment (minimization formulation)
                absolute_distance_increment = abs(distance - old_distance)
                relative_distance_increment = absolute_distance_increment / distance

                # 2 - mass conservation residual (constraint in optimization formulation)
                mass_conservation_residual = self.div.dot(flux) - self.pressure_view(
                    rhs
                )
                absolute_mass_residual = np.linalg.norm(mass_conservation_residual, 2)
                relative_mass_residual = absolute_mass_residual / mass_diff_norm

            # Update convergence history
            convergence_history.append(
                distance,
                relative_distance_increment,
                relative_bregman_increment,
                relative_mass_residual,
                timings,
                np.nan,  # placeholder for total run time - update below
            )

            # Update total run time
            current_total_run_time = self._sum_timings(convergence_history.timings)[
                "total"
            ]
            convergence_history.total_run_time[-1] = current_total_run_time

            # Print status
            if self.verbose:
                print(
                    f"Iter. {iter} \t|"
                    f"{distance:.6e} \t| "
                    f"{relative_distance_increment:.6e} \t| "
                    f"{relative_bregman_increment:.6e} \t| "
                    f"{relative_mass_residual:.6e}"
                )

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

            # Convergence check.
            convergence_status = self.convergence_criteria.check_convergence_status(
                iter=iter,
                aux_force_increment=relative_distance_increment,
                distance_increment=relative_distance_increment,
                mass_conservation_residual=relative_mass_residual,
            )
            if convergence_status in [
                darsia.ConvergenceStatus.CONVERGED,
                darsia.ConvergenceStatus.NOT_CONVERGED,
            ]:
                break

            # Update Bregman variables for next iteration.
            old_aux_flux = new_aux_flux.copy()
            old_force = new_force.copy()
            old_distance = distance

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
            "converged": convergence_status == darsia.ConvergenceStatus.CONVERGED,
            "number_iterations": iter,
            "convergence_history": convergence_history.as_dict(),
            "timings": total_timings,
            "peak_memory_consumption": peak_memory_consumption,
        }

        return distance, solution, info
