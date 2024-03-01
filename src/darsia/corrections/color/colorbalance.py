"""Module for color balance correction."""

from abc import ABC, abstractmethod
from typing import Literal

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
        balanced_img = img @ self.balance_scaling
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
        self.balance_scaling: np.ndarray = np.eye(3)
        """Color balance matrix."""

    def find_balance(self, swatches_src: np.ndarray, swatches_dst) -> None:
        """Find the color balance of an image.

        Args:
            swatches_src (np.ndarray): Source swatches.
            swatches_dst (np.ndarray): Destination swatches.

        """

        def objective_function(flat_balance: np.ndarray) -> float:
            """Objective function for the minimization.

            Args:
                flat_balance (np.ndarray): Flat color balance matrix.

            Returns:
                float: Objective function value.

            """
            balance = flat_balance.reshape((3, 3))
            swatches_src_balanced = swatches_src @ balance
            return np.sum((swatches_src_balanced - swatches_dst) ** 2)

        opt_result = scipy.optimize.minimize(
            objective_function,
            self.balance_scaling.flatten(),
            method="Powell",
            tol=1e-6,
            options={"maxiter": 1000, "disp": False},
        )

        self.balance_scaling = opt_result.x.reshape((3, 3))


class WhiteBalance(BaseBalance):
    """Class for finding and applying a diagional linear balance."""

    def __init__(self) -> None:
        self.balance_scaling: np.ndarray = np.diag(np.ones(3))
        """White balance (diagonal) matrix."""

    def find_balance(self, swatches_src: np.ndarray, swatches_dst) -> None:
        """Find the color balance of an image.

        Args:
            swatches_src (np.ndarray): Source swatches.
            swatches_dst (np.ndarray): Destination swatches.

        """

        def objective_function(flat_balance: np.ndarray) -> float:
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
            objective_function,
            np.diag(self.balance_scaling),
            method="Powell",
            tol=1e-6,
            options={"maxiter": 1000, "disp": False},
        )

        self.balance_scaling = np.diag(opt_result.x)


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

        def objective_function(flat_balance: np.ndarray) -> float:
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
            objective_function,
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


class AdaptiveBalance(AffineBalance):
    def __init__(self) -> None:
        self.balance_scaling: np.ndarray = np.eye(3)
        """Balance scaling matrix."""
        self.balance_translation: np.ndarray = np.zeros(3)
        """Balance translation vector."""

    def reset(self) -> None:
        """Reset to identity."""
        self.balance_scaling = np.eye(3)
        self.balance_translation = np.zeros(3)

    def find_balance(
        self,
        swatches_src: np.ndarray,
        swatches_dst,
        mode: Literal["diagonal", "linear", "affine"] = "affine",
    ) -> None:
        """Find the color balance of an image.

        Args:
            swatches_src (np.ndarray): Source swatches.
            swatches_dst (np.ndarray): Destination swatches.

        """
        # Precondition swatched with current balance
        swatches_src_prebalanced = self.apply_balance(swatches_src)

        # Update balancing, use formula A_new * (A_prev * x + b_prev) + b_new
        # = (A_new * A_prev) * x + (A_new * b_prev + b_new)
        if mode == "diagonal":
            balance = WhiteBalance()
        elif mode == "linear":
            balance = ColorBalance()
        elif mode == "affine":
            balance = AffineBalance()
        balance.find_balance(swatches_src_prebalanced, swatches_dst)
        self.balance_scaling = balance.balance_scaling @ self.balance_scaling
        if mode == "affine":
            self.balance_translation = (
                balance.balance_scaling @ self.balance_translation
                + balance.balance_translation
            )


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
