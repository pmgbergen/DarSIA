"""Module for computing the L1 Wasserstein distance with Newton's method."""

from __future__ import annotations

import time
import tracemalloc
import warnings

import numpy as np
import scipy.sparse as sps

import darsia


class WassersteinDistanceNewton(darsia.VariationalWassersteinDistance):
    """Class to determine the L1 EMD/Wasserstein distance solved with Newton's method.

    Solves the Beckmann prbblem in mixed form with a Newton-like method. The linearization
    of the optimality conditions is relaxed with a cut-off parameter L for the mobility.
    In addition, to circumvent singular Jacobians, merely a fixed-point type linearization
    is employed. This results in a method that is in between a pure Newton method and a
    fixed-point iteration.

    Here, self.L has the interpretation of a lower cut-off value in the linearization
    only. With such relaxation, the Beckman problem itself is not regularized, but
    instead the solution trajectory is merely affected.

    """

    def __init__(self, grid, weight, options) -> None:
        super().__init__(grid, weight, options)

        self.L = self.options.get("L", np.finfo(float).max)
        """float: relaxation/cut-off parameter for mobility, deactivated by default."""

    def residual(self, rhs: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """Compute the residual of the solution - the optimality conditions.

        Args:
            rhs (np.ndarray): right hand side
            solution (np.ndarray): solution

        Returns:
            np.ndarray: residual

        """
        return self.optimality_conditions(rhs, solution)

    def jacobian(self, solution: np.ndarray) -> sps.linalg.LinearOperator:
        """Compute the Jacobian of the optimality conditions.

        Args:
            solution (np.ndarray): solution

        Returns:
            sps.linalg.splu: LU factorization of the jacobian

        """
        # Only need to update the flux-flux block
        flat_flux = solution[self.flux_slice]
        face_weights, _ = self._compute_face_weight(flat_flux)
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
        self.convergence_history = {
            "distance": [],
            "residual": [],
            "flux_increment": [],
            "distance_increment": [],
            "timing": [],
            "run_time": [],
        }
        convergence_history = self.convergence_history

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
        iter = 0
        for iter in range(num_iter):
            # It is possible that the linear solver fails. In this case, we simply
            # stop the iteration and return the current solution.
            try:
                # Keep track of old flux, and old distance
                old_solution_i = solution_i.copy()
                flux = solution_i[self.flux_slice]
                self.flux = flux
                self.pressure = solution_i[self.pressure_slice]
                old_distance = self.l1_dissipation(flux)

                # Assemble linear problem in Newton step
                tic = time.time()
                residual_i = self.residual(rhs, solution_i)
                res_pressure = residual_i[self.pressure_slice]
                transport_density_faces = self.transport_density_faces(flux)
                res_opt = self.mass_matrix_faces.dot(
                    solution_i[self.flux_slice]
                ) - transport_density_faces * self.div.T.dot(
                    solution_i[self.pressure_slice]
                )

                residual_opt = np.linalg.norm(res_opt, 2) / old_distance
                residual_div = np.linalg.norm(res_pressure, 2)
                res_pde = np.sqrt(residual_opt**2 + residual_div**2)
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
                convergence_history["residual"].append(
                    res_pde
                )  # np.linalg.norm(residual_i, 2))
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
                        convergence_history["residual"][-1]
                        < tol_residual * convergence_history["residual"][0]
                        and convergence_history["flux_increment"][-1]
                        < tol_increment * convergence_history["flux_increment"][0]
                        and convergence_history["distance_increment"][-1] < tol_distance
                    ):
                        break
            except Exception:
                warnings.warn("Newton iteration abruptly stopped due to some error.")
                break

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

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
