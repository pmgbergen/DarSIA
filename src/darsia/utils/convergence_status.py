from enum import StrEnum


class ConvergenceStatus(StrEnum):
    """Enum to represent the convergence status of an iterative solver."""

    RUNNING = "running"
    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
