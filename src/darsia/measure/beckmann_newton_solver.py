"""Module for computing the L1 Wasserstein distance with Newton's method."""

from __future__ import annotations

import time
import tracemalloc
import warnings
from typing import override, Optional

import numpy as np
import scipy.sparse as sps

import darsia


class BeckmannNewtonSolver(darsia.BeckmannProblem):
    """Class to determine the L1 EMD/Wasserstein distance solved with Newton's method.

    Implements the :class:`darsia.BeckmannProblem` interface by specifying the method
    :meth:`darsia.BeckmannProblem.solve_beckmann_problem`.

    Solves the Beckmann problem in mixed form with a Newton-like method. The linearization
    of the optimality conditions is relaxed with a cut-off parameter L for the mobility.
    In addition, to circumvent singular Jacobians, merely a fixed-point type linearization
    is employed. This results in a method that is in between a pure Newton method and a
    fixed-point iteration.

    """

    def __init__(
        self,
        grid: darsia.Grid,
        weight: Optional[darsia.Image] = None,
        options: dict = {},
    ) -> None:
        super().__init__(grid, weight, options)

        self.convergence_criteria = darsia.BeckmannConvergenceCriteria(
            num_iter=options.get("num_iter", 100),
            tol_increment=options.get("tol_increment", np.finfo(float).max),
            tol_distance=options.get("tol_distance", np.finfo(float).max),
            tol_residual=options.get("tol_residual", np.finfo(float).max),
        )
        """"Convergence criteria for the Newton iteration."""

    def compute_residual(self, rhs: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """Compute the residual of the solution - the optimality conditions.

        Args:
            rhs (np.ndarray): right hand side
            solution (np.ndarray): solution

        Returns:
            np.ndarray: residual

        """
        return self.optimality_conditions(rhs, solution)

    def compute_jacobian(self, solution: np.ndarray) -> sps.linalg.LinearOperator:
        """Compute the Jacobian of the optimality conditions.

        Args:
            solution (np.ndarray): solution

        Returns:
            sps.linalg.splu: LU factorization of the jacobian

        """
        # Only need to update the flux-flux block
        flux = self.flux_view(solution)
        face_weights, _ = self._compute_face_weight(flux)
        weight = sps.diags(face_weights)
        flux_flux_block = weight @ self.mass_matrix_faces

        # Assemble full jacobian and store for later use
        # - first iteration: base on linear part of jacobian
        # - later iterations: update flux-flux block only
        if not hasattr(self, "full_jacobian"):
            self.full_jacobian: sps.csc_matrix = self.broken_darcy.copy()
        last_jacobian = self.full_jacobian.tolil()
        last_jacobian[(self.flux_slice, self.flux_slice)] = flux_flux_block
        self.full_jacobian = last_jacobian.tocsc()
        return self.full_jacobian

    def _compute_residual_norm(self, rhs: np.ndarray, solution: np.ndarray) -> float:
        """Compute the residual for the stopping criterion.

        Use a rescaled version of the optimality conditions to avoid division by zero.

        Args:
            residual (np.ndarray): current residual
            solution (np.ndarray): current solution

        Returns:
            dict: contributions to the residual
        """
        # Split residuals into their contributions. Use a rescaled version of the
        # optimality conditions to avoid division by zero.
        residual = self.compute_residual(rhs, solution)
        residual_vector_pressure = self.pressure_view(residual)
        residual_vector_optimality = self.rescaled_flux_optimality_conditions(solution)

        # Compute the norms - normalize the flux residual by the distance (see above).
        residual_opt = np.linalg.norm(residual_vector_optimality, 2)
        residual_div = np.linalg.norm(residual_vector_pressure, 2)
        return np.sqrt(residual_opt**2 + residual_div**2)

    @override
    def solve_beckmann_problem(
        self, flat_mass_diff: np.ndarray
    ) -> tuple[float, np.ndarray, dict]:
        """Solve the Beckmann problem using Newton's method.

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
        convergence_history = darsia.BeckmannConvergenceHistory()

        # Print  header for later printing performance to screen
        # - distance
        # - distance increment
        # - flux increment
        # - residual
        if self.verbose:
            print(
                """Newton iter. \t| """
                """W^1 \t\t| """
                """Δ W^1 / W^1 \t| """
                """Δ flux / flux \t| """
                """residual / residual_0"""
                """\n"""
                """---------------|"""
                """---------------|"""
                """---------------|"""
                """---------------|"""
                """---------------"""
            )

        # Newton iteration
        iter = 0
        for iter in range(self.convergence_criteria.num_iter):
            # Keep track of old flux and distance
            old_solution = solution.copy()
            old_distance = self.l1_dissipation(self.flux_view(old_solution))

            # Assemble linear problem in Newton step
            tic = time.time()
            residual = self.compute_residual(rhs, solution)
            jacobian = self.compute_jacobian(solution)
            time_assemble = time.time() - tic

            # Solve linear system for the update. It is possible that the linear
            # solver fails. In this case, we simply stop the iteration and return
            # the current solution.
            try:
                increment, timings = self.linear_solve(jacobian, residual, solution)
            except Exception:
                warnings.warn("Newton iteration abruptly stopped due to some error.")
                break

            # Update the solution with the full Newton step
            solution += increment

            # Apply Anderson acceleration to flux contribution (the only nonlinear part).
            # Application to full solution, or just the pressure, lead to divergence,
            # while application to the flux, results in improved performance.
            # Update the increment here for later use in the stopping criterion.
            if self.anderson is not None:
                tic = time.time()
                solution[self.flux_slice] = self.anderson(
                    self.flux_view(solution),
                    self.flux_view(increment),
                    iter,
                )
                increment = solution - old_solution
                time_acceleration = time.time() - tic
            else:
                time_acceleration = 0.0

            # Update discrete W1 distance
            flux = self.flux_view(solution)
            distance = self.l1_dissipation(flux)

            # Update statistics
            timings["time_assemble"] = time_assemble
            timings["time_acceleration"] = time_acceleration

            # Compute the error and store as part of the convergence history:

            # 0 - full residual (Newton interpretation)
            residual_norm = self._compute_residual_norm(rhs, solution)

            # 1 - flux increment (fixed-point interpretation)
            flux_inc_norm = float(np.linalg.norm(self.flux_view(increment), 2))

            # 2 - distance increment (Minimization interpretation)
            distance_inc = abs(distance - old_distance)

            # Update convergence history (timing first so run_time uses latest timings)
            convergence_history.append(
                distance=distance,
                distance_increment=distance_inc,
                residual=residual_norm,
                increment=flux_inc_norm,
                timings=timings,
                total_run_time=np.nan,  # placeholder for total run time
            )
            # Update run time based on sum of timings
            convergence_history.total_run_time[-1] = self._sum_timings(
                convergence_history.timings
            )["total"]

            # Compute relative errors for stopping criterion.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="overflow encountered")
                relative_distance_increment = distance_inc / distance
                relative_flux_increment = flux_inc_norm / np.linalg.norm(
                    self.flux_view(solution)
                )
                relative_residual = residual_norm / convergence_history.residual[0]

            # Print performance to screen.
            if self.verbose:
                print(
                    f"""Iter. {iter} \t| """
                    f"""{distance:.6e} \t| """
                    f"""{relative_distance_increment:.6e} \t| """
                    f"""{relative_flux_increment:.6e} \t| """
                    f"""{relative_residual:.6e}"""
                )

            # Check for convergence.
            convergence_status = self.convergence_criteria.check_convergence_status(
                iter=iter,
                increment=relative_flux_increment,
                distance_increment=relative_distance_increment,
                residual=relative_residual,
            )
            if iter > 1 and convergence_status in [
                darsia.ConvergenceStatus.CONVERGED,
                darsia.ConvergenceStatus.NOT_CONVERGED,
            ]:
                break

            # Callbacks
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

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
