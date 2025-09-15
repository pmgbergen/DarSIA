from dataclasses import dataclass
import darsia
import numpy as np


@dataclass
class BeckmannConvergenceCriteria:
    """Class to store and check the convergence criteria for the Beckmann problem."""

    num_iter: int = 100
    """Maximum number of iterations."""
    tol_increment: float = np.finfo(float).max
    """Tolerance for the increment vector."""
    tol_distance: float = np.finfo(float).max
    """Tolerance for the distance increment."""
    tol_residual: float = np.finfo(float).max
    """Tolerance for the residual."""

    def check_convergence_status(
        self,
        iter: int,
        increment: float,
        distance_increment: float,
        residual: float,
    ) -> darsia.ConvergenceStatus:
        """Check if convergence criteria are met."""
        max_iterations_reached = self._check_iterations(iter)
        convergence_achieved = self._check_convergence(
            increment,
            distance_increment,
            residual,
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
        increment: float,
        distance_increment: float,
        residual: float,
    ) -> bool:
        """Check if convergence criteria are met."""
        return (
            increment < self.tol_increment
            and distance_increment < self.tol_distance
            and residual < self.tol_residual
        )
