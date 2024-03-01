"""Module for color balance correction."""

from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize


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
        balanced_img = img @ self.balance
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
    """Class for finding and applying a linear color balance."""

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
            swatches_src_balanced = swatches_srci @ balance
            return np.sum((swatches_src_balanced - swatches_dst) ** 2)

        opt_result = scipy.optimize.minimize(
            objective_funcion,
            self.balance.flatten(),
            method="Powell",
            tol=1e-6,
            options={"maxiter": 1000, "disp": False},
        )

        self.balance = opt_result.x.reshape((3, 3))


class WhiteBalance(BaseBalance):
    """Class for finding and applying a diagional linear balance."""

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
            swatches_src_balanced = swatches_src @ balance
            return np.sum((swatches_src_balanced - swatches_dst) ** 2)

        opt_result = scipy.optimize.minimize(
            objective_funcion,
            np.diag(self.balance),
            method="Powell",
            tol=1e-6,
            options={"maxiter": 1000, "disp": False},
        )

        self.balance = np.diag(opt_result.x)


class AffineBalance(BaseBalance):
    """Class for finding and applying a general affine balance."""

    def __init__(self) -> None:
        self.balance_scaling: np.ndarray = np.eye(3)
        """Balance scaling matrix."""
        self.balance_translation: np.ndarray = np.zeros(3)
        """Balance translation vector."""

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
            balance_scaling = flat_balance[:9].reshape((3, 3))
            balance_translation = flat_balance[9:12]
            swatches_src_balanced = swatches_src @ balance_scaling + balance_translation
            return np.sum((swatches_src_balanced - swatches_dst) ** 2)

        opt_result = scipy.optimize.minimize(
            objective_funcion,
            np.concatenate((self.balance_scaling.flatten(), self.balance_translation)),
            method="Powell",
            tol=1e-6,
            options={"maxiter": 1000, "disp": False},
        )

        self.balance_scaling = opt_result.x[:9].reshape((3, 3))
        self.balance_translation = opt_result.x[9:12]

    def apply_balance(self, img: np.ndarray) -> np.ndarray:
        """Apply the color balance to an image.

        Args:
            img (np.ndarray): Image to apply the color balance to.

        Returns:
            balanced_img (np.ndarray): Balanced image.

        """
        balanced_img = img @ self.balance_scaling + self.balance_translation
        return balanced_img


# ! ---- Shortcut functions ---- ! #


def color_balance(
    img: np.ndarray, swatches_src: np.ndarray, swatches_dst
) -> np.ndarray:
    """Apply the color balance to an image.

    Args:
        img (np.ndarray): Image to apply the color balance to.
        swatches_src (np.ndarray): Source swatches.
        swatches_dst (np.ndarray): Destination swatches.

    Returns:
        balanced_img (np.ndarray): Balanced image.

    """
    cb = ColorBalance()
    return cb(img, swatches_src, swatches_dst)


def white_balance(
    img: np.ndarray, swatches_src: np.ndarray, swatches_dst
) -> np.ndarray:
    """Apply the color balance to an image.

    Args:
        img (np.ndarray): Image to apply the color balance to.
        swatches_src (np.ndarray): Source swatches.
        swatches_dst (np.ndarray): Destination swatches.

    Returns:
        balanced_img (np.ndarray): Balanced image.

    """
    wb = WhiteBalance()
    return wb(img, swatches_src, swatches_dst)


def affine_balance(
    img: np.ndarray, swatches_src: np.ndarray, swatches_dst
) -> np.ndarray:
    """Apply the color balance to an image.

    Args:
        img (np.ndarray): Image to apply the color balance to.
        swatches_src (np.ndarray): Source swatches.
        swatches_dst (np.ndarray): Destination swatches.

    Returns:
        balanced_img (np.ndarray): Balanced image.

    """
    ab = AffineBalance()
    return ab(img, swatches_src, swatches_dst)
