"""Module for color balance correction."""

from abc import ABC, abstractmethod

import numpy as np

import darsia


class BaseBalance(ABC):
    """Base class for finding and applying a color/white balance."""

    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def find_balance(self, swatches_src: np.ndarray, swatches_dst) -> None:
        """Find the color balance of an image.

        This routine defines the color balance of an image by finding a linear transformation.

        Args:
            swatches_src (np.ndarray): Source swatches.
            swatches_dst (np.ndarray): Destination swatches.

        """
        ...

    def apply_balance(self, img: np.ndarray) -> np.ndarray:
        """Apply the color balance to an image.

        Args:
            img (np.ndarray): Image to apply the color balance to.

        Returns:
            balanced_img (np.ndarray): Balanced image.

        """
        balanced_img = np.dot(img, self.balance)
        balanced_img = np.clip(balanced_img, 0, 1)
        return balanced_img

    def __call__(self, img, swatches_src, swatches_dst) -> np.ndarray:
        """Apply the color balance to an image.

        Args:
            img (np.ndarray): Image to apply the color balance to.
            swatches_src (np.ndarray): Source swatches.
            swatches_dst (np.ndarray): Destination swatches.

        Returns:
            balanced_img (np.ndarray): Balanced image.

        """
        self.find_balance(swatches_src, swatches_dst)
        balanced_img = self.apply_balance(img)

        return balanced_img


class ColorBalance(BaseBalance):
    """Class for finding and applying a color balance."""

    def __init__(self) -> None:
        self.balance: np.ndarray = np.eye(3)
        """Color balance matrix."""

    def find_balance(self, swatches_src: np.ndarray, swatches_dst) -> None:
        """Find the color balance of an image.

        Args:
            swatches_src (np.ndarray): Source swatches.
            swatches_dst (np.ndarray): Destination swatches.

        """

        def objective_funcion(flat_balance: np.ndarray) -> float:
            """Objective function for the minimization.

            Args:
                flat_balance (np.ndarray): Flat color balance matrix.

            Returns:
                float: Objective function value.

            """
            balance = flat_balance.reshape((3, 3))
            swatches_src_balanced = np.dot(swatches_src, balance)
            return np.sum((swatches_src_balanced - swatches_dst) ** 2)

        opt_result = darsia.optimize.minimize(
            objective_funcion,
            self.balance.flatten(),
            method="Powell",
            tol=1e-6,
            options={"maxiter": 1000, "disp": False},
        )

        self.balance = opt_result.x.reshape((3, 3))


class WhiteBalance(BaseBalance):
    """Class for finding and applying a white balance."""

    def __init__(self) -> None:
        self.balance: np.ndarray = np.diag(np.ones(3))
        """White balance (diagonal) matrix."""

    def find_balance(self, swatches_src: np.ndarray, swatches_dst) -> None:
        """Find the color balance of an image.

        Args:
            swatches_src (np.ndarray): Source swatches.
            swatches_dst (np.ndarray): Destination swatches.

        """

        def objective_funcion(flat_balance: np.ndarray) -> float:
            """Objective function for the minimization.

            Args:
                flat_balance (np.ndarray): Flat color balance matrix.

            Returns:
                float: Objective function value.

            """
            balance = np.diag(flat_balance)
            swatches_src_balanced = np.dot(swatches_src, balance)
            return np.sum((swatches_src_balanced - swatches_dst) ** 2)

        opt_result = darsia.optimize.minimize(
            objective_funcion,
            np.diag(self.balance),
            method="Powell",
            tol=1e-6,
            options={"maxiter": 1000, "disp": False},
        )

        self.balance = np.diag(opt_result.x)
