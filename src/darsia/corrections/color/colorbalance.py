"""Module for color balance correction."""

from abc import ABC, abstractmethod

import numpy as np

import darsia


class BaseBalance(ABC):
    """Base class for finding and applying a color balance."""

    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def find_balance(self, swatches_src: np.ndarray, swatches_dst) -> np.ndarray:
        """Find the color balance of an image.

        Args:

        Returns:

        """
        ...

    def apply_balance(self, img: np.ndarray) -> np.ndarray:
        """Apply the color balance to an image.

        Args:

        Returns:

        """
        balanced_img = np.dot(img, self.balance)
        balanced_img = np.clip(balanced_img, 0, 1)
        return balanced_img

    def __call__(self, img, swatches_src, swatches_dst) -> np.ndarray:
        """Apply the color balance to an image.

        Args:

        Returns:

        """
        self.find_balance(swatches_src, swatches_dst)
        balanced_img = self.apply_balance(img)

        return balanced_img


class ColorBalance(BaseBalance):
    """Class for finding and applying a color balance."""

    def __init__(self) -> None:
        self.balance: np.ndarray = np.eye(3)

    def find_balance(self, swatches_src: np.ndarray, swatches_dst) -> np.ndarray:
        """Find the color balance of an image.

        Args:

        Returns:

        """

        def objective_funcion(flat_balance: np.ndarray) -> np.ndarray:
            """Objective function for the minimization.

            Args:

            Returns:

            """
            balance = flat_balance.reshape((3, 3))
            swatches_src_balanced = np.dot(swatches_src, balance)
            return np.sum((swatches_src_balanced - swatches_dst) ** 2)

        opt_result = darsia.optimize.minimize(
            objective_funcion,
            self.balance.flatten(),
            method="Powell",
            options={"maxiter": 1000, "tol": 1e-6, "disp": True},
        )

        self.balance = opt_result.x.reshape((3, 3))


class WhiteBalance(BaseBalance):
    def __init__(self) -> None:
        self.balance: np.ndarray = np.diag(np.ones(3))

    def find_balance(self, swatches_src: np.ndarray, swatches_dst) -> np.ndarray:
        """Find the color balance of an image.

        Args:

        Returns:

        """

        def objective_funcion(flat_balance: np.ndarray) -> np.ndarray:
            """Objective function for the minimization.

            Args:

            Returns:

            """
            balance = np.diag(flat_balance)
            swatches_src_balanced = np.dot(swatches_src, balance)
            return np.sum((swatches_src_balanced - swatches_dst) ** 2)

        opt_result = darsia.optimize.minimize(
            objective_funcion,
            np.diag(self.balance),
            method="Powell",
            options={"maxiter": 1000, "tol": 1e-6, "disp": True},
        )

        self.balance = np.diag(opt_result.x)
