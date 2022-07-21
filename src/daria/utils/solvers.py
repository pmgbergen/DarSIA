import daria as da
import numpy as np


class Solver:
    """
    Solver base class


    Attributes:
        apply: applies the solver method. Should be implemented in the specific solver subclass


    """

    def __init__(
        self, stoppingCriterion: da.StoppingCriterion, verbose: bool = False
    ) -> None:
        """
        Constructor for solver class.

        Arguments:
            stoppingCriterion: stopping criterion for the solver
            verbose: set to True to print output data
        """
        self.stoppingCriterion = stoppingCriterion

    def apply():
        print("Please choose a solver!")


class CG(Solver):
    def apply(self, operator, rhs, im0):
        im = im0
        res = operator(im0) - rhs
        p = -res
        iteration = 0
        while not (self.stoppingCriterion.check(res, iteration)):
            po = operator(p)
            alpha = -da.im_product(res, p) / da.im_product(po, p)
            im = im + alpha * p
            res = operator(im) - rhs
            beta = da.im_product(res, po) / da.im_product(po, p)
            p = -res + beta * p
        alpha = -da.im_product(res, p) / da.im_product(operator(p), p)
        im = im + alpha * p
        return im


def richardson(
    operator: Callable,
    rhs: np.ndarray,
    im0: np.ndarray,
    tol: float,
    omega: float,
    norm: Callable,
    max_iter: int = 1000,
):
    res = rhs - operator(im0)
    im = im0
    nr = norm(res)
    iteration = 0
    while nr > tol and iteration < max_iter:
        im = im + omega * res
        res = rhs - operator(im)
        nr = norm(res)
        print(nr)
        iteration += 1
    print(iteration)

    return im
