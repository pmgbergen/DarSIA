from __future__ import annotations

import numpy as np
from typing import Callable


class StoppingCriterion:
    """
    Stopping criterion class.

    Attributes:
        tolerance (float): Tolerance for the stopping criterion.
        max_iterations (int): Maximal allowed number of iterations.
        norm (Callable): The norm that is to be used.
        verbose (bool): Set to true to get output.
        check (Callable): Returns true if the stopping criterion is satisfied.
        check_residual (Callable): Returns true if the relative stopping criterion is satisfied.

    """

    def __init__(
        self,
        tolerance: float,
        max_iterations: int,
        norm: Callable = np.linalg.norm,
        verbose: bool = False,
    ) -> None:
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.norm = norm
        self.verbose = verbose

    def check(self, entity: np.ndarray, iterations: int):
        norm_entity = self.norm(entity)
        result = norm_entity < self.tolerance or iterations >= self.max_iterations
        if self.verbose:
            if result:
                print(
                    "Stopping at norm: ",
                    norm_entity,
                    ", and iteration number ",
                    iterations,
                )
            else:
                print("Norm evaluated to: ", norm_entity)

        return result

    def check_relative(self, entity: np.ndarray, rel_entity: np.ndarray, iterations):
        norm_entity = self.norm(entity) / self.norm(rel_entity)
        result = norm_entity < self.tolerance or iterations >= self.max_iterations
        if self.verbose:
            if result:
                print(
                    "Stopping at norm: ",
                    norm_entity,
                    ", and iteration number ",
                    iterations,
                )
            else:
                print("Norm evaluated to: ", norm_entity)

        return result
