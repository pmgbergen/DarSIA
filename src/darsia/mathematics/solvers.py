from __future__ import annotations

from typing import Callable

import numpy as np

import darsia as da


# TODO This file is currently not in use. Deprecate soon.

class Solver:
    """
    Solver base class. Contains an apply method.
    All solvers that are implemented as subclasses should have their own
    apply method, and inherit the stopping criterion.


    Attributes:
        stoppingCriterion (darsia.StoppingCriterion): stopping criterion for the solver
        verbose (bool): set to True to print output data

    """

    def __init__(
        self, stoppingCriterion: da.StoppingCriterion, verbose: bool = False
    ) -> None:
        """
        Constructor for solver class.

        Arguments:
            stoppingCriterion (darsia.StoppingCriterion): stopping criterion for the solver
            verbose (bool): set to True to print output data
        """
        self.stoppingCriterion = stoppingCriterion
        self.stoppingCriterion.verbose = verbose

    # def apply(self) -> None:
    #     print("Please choose a solver!")


class CG(Solver):
    """
    Conjugate Gradients solver.

    Attributes:
        apply: applies the solver method

    """

    def apply(self, operator: Callable, rhs: np.ndarray, im0: np.ndarray) -> np.ndarray:
        """
        Applies the conjugate gradients solver.

        Arguments:
            operator (Callable): The left hand side operator that is to be inverted.
            rhs (np.ndarray): Right-hand side.
            im0 (np.ndarray): Initial guess.
        """

        # Set im to initial guess and compute initial residual and conjugate matrix
        im = im0
        res = operator(im0) - rhs
        p = -res

        # Define iteration counter
        iterations = 0

        # Iterate as long as the stopping criterion is not satisfied
        while not (self.stoppingCriterion.check(res, iterations)):

            # Compute the operator applied to the conjugate matrix
            # (I think that it is more efficient to store this new matrix
            # rather than doing the operation twice).
            po = operator(p)

            # Compute step length
            alpha = -da.im_product(res, p) / da.im_product(po, p)

            # Update the image and compute new residual and conjugate matrix
            im = im + alpha * p
            res = operator(im) - rhs
            beta = da.im_product(res, po) / da.im_product(po, p)
            p = -res + beta * p

            # Update the iteration counter
            iterations += 1

        # Jump once further in the search direction and return the solution image
        alpha = -da.im_product(res, p) / da.im_product(operator(p), p)
        im = im + alpha * p
        return im


class ModifiedRichardson(Solver):
    """
    Modified Richardson solver.

    Attributes:
        apply: applies the solver method

    """

    def apply(
        self, operator: Callable, rhs: np.ndarray, im0: np.ndarray, omega: float
    ) -> np.ndarray:
        """
        Applies the modified Richardson solver.

        Arguments:
            operator (Callable): The left hand side operator that is to be inverted.
            rhs (np.ndarray): Right-hand side.
            im0 (np.ndarray): Initial guess.
            omega (float): Update parameter.
        """
        # Set im to initial guess and compute initial residual
        im = im0
        res = rhs - operator(im)

        # Define iteration counter
        iteration = 0

        # Iterate as long as the stopping criterion is not satisfied
        while not (self.stoppingCriterion.check(res, iteration)):

            # Update the image and compute new residual
            im = im + omega * res
            res = rhs - operator(im)

            # Update the iteration counter
            iteration += 1

        # Return image
        return im
