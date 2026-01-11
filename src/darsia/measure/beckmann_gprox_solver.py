"""Wasserstein distance computed using the GproxPDHG algorithm."""

from __future__ import annotations

import time
import tracemalloc
import warnings
from typing import Optional, override

import numpy as np
import pyamg
import scipy.sparse as sps

import darsia


class BeckmannGproxPGHDSolver(darsia.BeckmannProblem):
    """
    This contains the implementation of the GproxPDHG algorithm
    described in "SOLVING LARGE-SCALE OPTIMIZATION PROBLEMS WITH
    A CONVERGENCE RATE INDEPENDENT OF GRID SIZE"
    """

    # ! ---- Setup routines ----

    def __init__(
        self,
        grid: darsia.Grid,
        weight: Optional[darsia.Image] = None,
        options: dict = {},
    ) -> None:
        # Cache geometrical infos
        self.grid = grid
        """darsia.Grid: grid"""

        self.voxel_size = grid.voxel_size
        """np.ndarray: voxel size"""

        self.weight = weight
        """Optional[darsia.Image]: weight"""
        if weight is not None:
            raise NotImplementedError(
                "Weighted Gprox not implemented for anisotropic meshes"
            )

        # Cache solver options
        self.options = options
        """dict: options for the solver"""

        self.verbose = self.options.get("verbose", False)
        """bool: verbosity"""

        self.callbacks = self.options.get("callbacks", None)
        """list: list of callbacks to be called at each iteration"""

        self.norm_mode = self.options.get("norm_mode", "l2")
        """str: mode for computing the norm: l2 or manhattan"""

        self._setup_variables()
        self._setup_l1_quadrature()
        self._setup_discretization()

    def _setup_variables(self) -> None:
        # Allocate space for main and auxiliary variables
        self.flux = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: flux"""

        self.kantorovich_potential = np.zeros(self.grid.num_cells, dtype=float)
        """ np.ndarray: Kantorovich potential"""

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

    def _setup_discretization(self) -> None:
        """Setup of fixed discretization operators.

        Add linear contribution of the optimality conditions of the Newton linearization.

        """

        # ! ---- Discretization operators ----
        self.div = darsia.FVDivergence(self.grid).mat
        """sps.csc_matrix: divergence operator: flat fluxes -> flat pressures"""

        self.grad = darsia.FVDivergence(self.grid).mat.T
        """sps.csc_matrix: grad operator:  flat pressures -> flat fluxes"""

        self.mass_matrix_cells = darsia.FVMass(self.grid).mat
        """sps.csc_matrix: mass matrix on cells: flat pressures -> flat pressures"""

        lumping = self.options.get("lumping", True)
        self.mass_matrix_faces = darsia.FVMass(self.grid, "faces", lumping).mat
        """sps.csc_matrix: mass matrix on faces: flat fluxes -> flat fluxes"""

        self.inverse_mass_matrix_faces = sps.diags(
            1.0 / self.mass_matrix_faces.diagonal(), format="csc"
        )
        """sps.csc_matrix: inverse of the (diagonal) of mass matrix on faces:"""
        """flat fluxes -> flat fluxes"""

        linear_solver_options = self.options.get("linear_solver_options", {})
        rtol = linear_solver_options.get("rtol", 1e-6)
        self.Poisson_solver = self.setup_poisson_solver("pure_poisson", rtol=rtol)
        """ sps.linalg.KSP: Poisson solver"""

        self.full_flux_reconstructor = darsia.FVFullFaceReconstruction(self.grid)
        """darsia.FVFullFaceReconstruction: full flux reconstructor"""

    # ! ---- Main methods ----

    @override
    def solve_beckmann_problem(
        self, flat_mass_diff: np.ndarray
    ) -> tuple[float, np.ndarray, dict]:
        """Solve the Beckmann problem using GproxPDHG.

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
        tol_duality_gap = self.options.get("tol_duality_gap", np.finfo(float).max)

        # Initialize with the solution of the Poisson problem
        self.integrated_mass_diff = self.mass_matrix_cells.dot(flat_mass_diff)
        mass_ref = np.linalg.norm(self.integrated_mass_diff, 1)
        tic = time.time()
        # manual setup of the poisson solver
        # self.Poisson_solver.ksp.setTolerances(rtol=tol_residual)
        self.poisson_pressure = self.Poisson_solver.solve(self.integrated_mass_diff)
        time_poisson = time.time() - tic
        tic = time.time()
        self.gradient_poisson_pressure = self.inverse_mass_matrix_faces.dot(
            self.grad.dot(self.poisson_pressure)
        )
        self.flux[:] = self.gradient_poisson_pressure[:]

        # Initialize distance in case below iteration fails
        new_distance = self.l1_dissipation(self.flux)
        mass_conservation_residual = self.div.dot(self.flux) - self.integrated_mass_diff

        time_extra = time.time() - tic
        stats_i = {"time_poisson": time_poisson, "time_extra": time_extra}

        # Initialize container for storing the convergence history
        self.convergence_history = {
            "distance": [new_distance],
            "distance_increment": [],
            "flux_increment": [],
            "timing": [stats_i],
            "primal": [],
            "dual": [],
            "run_time": [time_poisson + time_extra],
            "mass_conservation_residual": [
                np.linalg.norm(mass_conservation_residual, 2) / mass_ref
            ],
        }
        convergence_history = self.convergence_history

        # Print  header for later printing performance to screen
        # - distance
        # - distance increment
        # - flux increment
        # - residual
        if self.verbose:
            print(
                "It.   | W^1        | Δ W^1    | Δ flux   | residual | ",
                "\n",
                "-----|------------|----------|----------|----------|",
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
            sigma = self.options.get("sigma", 1.0)
            if self.iter > 0:
                p[:] = new_p[:]
            time_extra = time.time() - start

            # eq 3.14
            tic = time.time()
            option = "simpler"
            # option = "as_in_paper"
            if option == "as_in_paper":
                div_free = self.leray_projection(p_bar)
                time_poisson = time.time() - tic
                tic = time.time()
                u -= tau * div_free
                convergence_history["flux_increment"].append(
                    np.linalg.norm(tau * div_free, 2)
                )

            elif option == "simpler":
                # update before the leray projection
                # to ensure the u is div-free
                u -= tau * p_bar
                old_u = u.copy()
                rhs = self.div.dot(old_u)
                poisson_solution = self.Poisson_solver.solve(rhs)
                update = self.inverse_mass_matrix_faces.dot(
                    self.grad.dot(poisson_solution)
                )
                u -= update
                convergence_history["flux_increment"].append(
                    np.linalg.norm(update + tau * p_bar, 2)
                )

            flux_increment = convergence_history["flux_increment"][-1] / mass_ref

            # new flux
            self.flux[:] = u[:] + self.gradient_poisson_pressure[:]

            update_kantorovich_potential = False
            if update_kantorovich_potential:
                flat_pressure = self.compute_pressure(self.flux, flat_mass_diff)
                grad_pressure = self.inverse_mass_matrix_faces.dot(
                    self.grad.dot(flat_pressure)
                )
                print(f"MAX GRAD {np.max(abs(grad_pressure))}")

            # Second step of the PDHG
            # new_p = argmax_{|p|\leq 1} (p, u_{n+1})_{L^2} + \frac{1}{2 sigma}|p-p_n|_{L^2}^2
            #
            # where |p| is the Euclidean norm
            #

            # eq 3.15
            sigma_vel = p + sigma * self.flux
            new_p[:] = sigma_vel[:]

            self.norm_mode = "l2"
            if self.norm_mode == "manhattan":
                abs_sigma_vel = np.abs(sigma_vel)
                greater_than_1 = np.where(abs_sigma_vel > 1)
                new_p[greater_than_1] /= abs_sigma_vel[
                    greater_than_1
                ]  # normalize too +1 or -1
            elif self.norm_mode == "l2":
                # recosntruct the full velocity
                full_sigma_vel = self.full_flux_reconstructor(sigma_vel)
                # compute Euclidean norm
                abs_sigma_vel = np.linalg.norm(full_sigma_vel, axis=1)
                #
                # where abs_sigma_vel < 1, set to 1
                #
                abs_sigma_vel[abs_sigma_vel < 1] = 1.0
                # normalize the normal component
                new_p /= abs_sigma_vel

            # eq 3.16
            p_bar[:] = 2 * new_p[:] - p[:]

            # compute primal and dual value
            self.dual_value = self.compute_dual(p, self.gradient_poisson_pressure)
            self.primal_value = self.compute_primal(self.flux)
            self.duality_gap = (
                abs(self.dual_value - self.primal_value) / self.primal_value
            )
            time_extra += time.time() - tic

            convergence_history["primal"].append(self.primal_value)
            convergence_history["dual"].append(self.dual_value)

            stats_i = {
                "time_poisson": time_poisson,
                "time_extra": time_extra,
            }

            new_distance = self.primal_value

            # Update distance
            mass_conservation_residual = (
                self.div.dot(self.flux) - self.integrated_mass_diff
            )

            # Update convergence history
            old_distance = convergence_history["distance"][-1]
            convergence_history["distance"].append(new_distance)
            convergence_history["distance_increment"].append(
                abs(new_distance - old_distance) / new_distance
            )
            convergence_history["mass_conservation_residual"].append(
                np.linalg.norm(mass_conservation_residual, 2) / mass_ref
            )
            convergence_history["timing"].append(stats_i)

            # Extract current total run time
            current_run_time = self._sum_timings(convergence_history["timing"])["total"]
            convergence_history["run_time"].append(current_run_time)

            self.iter += 1
            self.iter_cpu = time.time() - start

            if self.verbose:
                distance_increment = convergence_history["distance_increment"][-1]
                flux_increment = convergence_history["flux_increment"][-1]
                print(
                    f"{self.iter:05d} | {new_distance:.4e} | "
                    + f"{distance_increment:.2e} | {flux_increment:.2e}"
                    + f" | {convergence_history['mass_conservation_residual'][-1]:.2e} |"
                    + f" gap={self.duality_gap:.2e}"
                    + f" dual={self.dual_value:.2e} primal={self.primal_value:.2e}"
                    + f" cpu={self.iter_cpu:.3f}"
                )

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

            if (
                self.iter > 1
                and self.duality_gap < tol_duality_gap
                and convergence_history["distance_increment"][-1] < tol_distance
                and convergence_history["mass_conservation_residual"][-1] < tol_residual
                and flux_increment < tol_increment
            ):
                break

        # Summarize profiling (time in seconds, memory in GB)
        # total_timings = self._sum_timings(convergence_history["timing"])
        # peak_memory_consumption = tracemalloc.get_traced_memory()[1] / 10**9

        # Define performance metric
        info = {
            "converged": self.iter < num_iter,
            "number_iterations": self.iter,
            "convergence_history": convergence_history,
            # "timings": total_timings,
            # "peak_memory_consumption": peak_memory_consumption,
        }

        return new_distance, self.flux, info

    @override
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
        distance, solution, info = self.solve_beckmann_problem(flat_mass_diff)

        # Compute Kantorovich potential solving the weighed-Poisson problem
        pressure = self.compute_kantorovich_potential(flat_mass_diff, solution)
        pressure = pressure.reshape(self.grid.shape, order="F")

        # reshape the solution
        flux_cells = darsia.face_to_cell(self.grid, solution)
        transport_density_cells = np.linalg.norm(flux_cells, axis=-1)

        # Return solution
        return_info = self.options.get("return_info", False)
        if return_info:
            info.update(
                {
                    "grid": self.grid,
                    "mass_diff": mass_diff,
                    "poisson_pressure": self.poisson_pressure.reshape(
                        self.grid.shape, order="F"
                    ),
                    "gradient_poisson_pressure": darsia.face_to_cell(
                        self.grid, self.gradient_poisson_pressure
                    ),
                    "flux": flux_cells,
                    "pressure": pressure,
                    # "pressure": pressure,
                    "transport_density": transport_density_cells,
                    "src": img_1,
                    "dst": img_2,
                }
            )
            return distance, info
        else:
            return distance

    # ! ---- Effective quantities ----

    def compute_kantorovich_potential(
        self, flat_mass_diff: np.ndarray, flux: np.ndarray, tol=1e-6
    ) -> np.ndarray:
        """Compute the kantorovich potential from the normal flux

        Args:
            flat_mass_diff (np.ndarray): difference of mass distributions
            flux (np.ndarray): flux on the faces

        Returns:
            np.ndarray: kantorovich potential

        """
        full_flux = self.full_flux_reconstructor(flux)
        transport_density_faces = np.linalg.norm(full_flux, axis=1)
        weighted_Poisson_solver = self.setup_poisson_solver(
            "transport_density_weighted_poisson",
            rtol=tol,
            permeability_faces=transport_density_faces,
        )
        integrated_mass_diff = self.mass_matrix_cells.dot(flat_mass_diff)
        pressure = weighted_Poisson_solver.solve(
            integrated_mass_diff, x0=self.kantorovich_potential
        )
        # store the kantorovich potential
        self.kantorovich_potential[:] = pressure[:]
        if isinstance(weighted_Poisson_solver, darsia.linalg.KSP):
            weighted_Poisson_solver.kill()
        else:
            weighted_Poisson_solver = []
        return pressure

    def compute_dual(self, p, gradient_poisson):
        """
        Compute the value of the dual functional
        $ int_{Domain} pot (f^+ - f^-)$
        $= int_{Domain} pot -div(poisson)$
        $ int_{Domain} \nabla pot dot \nabla poisson$
        $ int_{Domain} p dot \nabla poisson$
        """
        return np.dot(p, gradient_poisson) * np.prod(self.grid.voxel_size)

    def compute_primal(self, flux):
        """
        Compute the value of the primal functional
        $int_{Domain} | flux | $
        """
        # full_flux = self.full_flux_reconstructor(flux)
        # transport_density_faces = np.linalg.norm(full_flux, axis=1)
        # NOTE the factor dim is considered to get the area of the diamond
        # w1 = np.sum(transport_density_faces) * np.prod(self.grid.voxel_size) / self.grid.dim

        w1 = self.l1_dissipation(flux)

        return w1

    # ! ---- Linear solver ----

    def setup_amg_options(self) -> None:
        """Setup the infrastructure for multilevel solvers.

        Basic default setup based on jacobi and block Gauss-Seidel smoothers.
        User-defined options can be passed via the options dictionary, using the key
        "amg_options". The options follow the pyamg interface.

        """
        self.amg_options = {
            "strength": "symmetric",  # change the strength of connection
            "aggregate": "standard",  # use a standard aggregation method
            "smooth": ("jacobi"),  # prolongation smoother
            "presmoother": (
                "block_gauss_seidel",
                {"sweep": "symmetric", "iterations": 1},
            ),
            "postsmoother": (
                "block_gauss_seidel",
                {"sweep": "symmetric", "iterations": 1},
            ),
            "coarse_solver": "pinv2",  # pseudo inverse via SVD
            "max_coarse": 100,  # maximum number on a coarse level
        }
        """dict: options for the AMG solver"""

        # Allow to overwrite default options - use pyamg interface.
        user_defined_amg_options = self.options.get("amg_options", {})
        self.amg_options.update(user_defined_amg_options)

    def setup_poisson_solver(self, solver_prefix, rtol=1e-6, permeability_faces=None):
        """Return the Poisson solver.

        Args:
            permeability_faces (np.ndarray, optional): permeability faces. Defaults to None.

        Returns:
            darsia.linalg.KSP: Poisson solver

        """

        self.linear_solver_type = self.options.get("linear_solver", "cg")
        """str: type of linear solver"""

        # Safety checks
        assert self.linear_solver_type in [
            "cg",
            "ksp",
            "dct",
        ], f"Linear solver {self.linear_solver_type} not supported."
        linear_solver_options = self.options.get("linear_solver_options", {})

        if permeability_faces is not None and self.linear_solver_type == "dct":
            raise ValueError(
                "DCT solver does not support permeability faces. Use CG or KSP instead."
            )

        # Define CG solver
        kernel = np.ones(self.grid.num_cells, dtype=float) / np.sqrt(
            self.grid.num_cells
        )

        if self.linear_solver_type == "dct":
            # dct is matrix-free
            pass
        else:
            if permeability_faces is None:
                Laplacian_matrix = self.div * self.inverse_mass_matrix_faces * self.grad
            else:
                Laplacian_matrix = (
                    self.div
                    * sps.diags(permeability_faces)
                    * self.inverse_mass_matrix_faces
                    * self.grad
                )

        #
        if self.linear_solver_type == "cg":
            # Define CG solver
            Poisson_solver = darsia.linalg.CG(Laplacian_matrix)

            # Define AMG preconditioner
            self.setup_amg_options()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Implicit conversion of A to CSR"
                )
                amg = pyamg.smoothed_aggregation_solver(
                    Laplacian_matrix, **self.amg_options
                ).aspreconditioner(cycle="V")

            # Define solver options
            maxiter = linear_solver_options.get("maxiter", 100)
            solver_options = {
                "rtol": rtol,
                "atol": 1e-15,
                "maxiter": maxiter,
                "M": amg,
            }
            Poisson_solver.setup(solver_options)

        elif self.linear_solver_type == "ksp":
            Poisson_solver = darsia.linalg.KSP(
                Laplacian_matrix,
                nullspace=[kernel],
                appctx={},
                solver_prefix=solver_prefix,
            )
            maxiter = linear_solver_options.get("maxiter", 100)
            ksp_ctrl = {
                "ksp_type": "cg",
                "ksp_rtol": rtol,
                "ksp_max_it": maxiter,
                "pc_type": "hypre",
                # "ksp_monitor_true_residual" : None
            }
            # if permeability_faces is not None:
            #    ksp_ctrl["ksp_monitor_true_residual"] = None
            Poisson_solver.setup(ksp_ctrl)

        return Poisson_solver

    def leray_projection(self, p: np.ndarray) -> np.ndarray:
        """Leray projection of a vector fiels

        Args:
            p (np.ndarray): pressure

        Returns:
            np.ndarray: divergence free flux

        """
        rhs = self.div.dot(p)
        poisson_solution = self.Poisson_solver.solve(rhs)
        return p - self.inverse_mass_matrix_faces.dot(self.grad.dot(poisson_solution))

    # ! ---- Utility methods ----

    def _sum_timings(self, timings: dict) -> dict:
        """Sum the timing of the current iteration.

        Utility function for self.solve_beckmann_problem().

        Args:
            timings (dict): timings

        Returns:
            dict: total time

        """
        total_timings = {
            "poisson": sum([t["time_poisson"] for t in timings]),
            "extra": sum([t["time_extra"] for t in timings]),
        }
        total_timings["total"] = total_timings["poisson"] + total_timings["extra"]

        return total_timings
