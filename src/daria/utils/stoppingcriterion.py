import numpy as np
from typing import Callable


class StoppingCriterion:
    def __init__(
        self,
        tolerance: float,
        max_iterations: int,
        norm: Callable,
        verbose: bool = False,
    ) -> None:
        self.tolerance = tolerance
        self.norm = norm
        self.verbose = verbose

    def check(self, criterion: np.ndarray, iterations: int):
        nr = self.norm(criterion)
        if self.verbose:
            print("Norm evaluated to: ", nr)
        return nr < self.tol and iterations >= self.max_iterations
