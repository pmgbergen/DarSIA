"""Module converting signals to binary data by applying thresholding.

A distinction between heterogeneous and homogeneous thresholding
is performed automatically.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

import darsia


class StaticThresholdModel(darsia.Model):
    """
    Class for static thresholding.
    """

    def __init__(
        self,
        threshold_lower: Union[float, list[float]] = 0.0,
        threshold_upper: Optional[Union[float, list[float]]] = None,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """
        Constructor of StaticThresholdModel.

        Args:
            threshold_lower (float or list of float): lower threshold value(s)
            threshold_upper (float or list of float): upper threshold value(s)
            labels (array): labeled domain

        """

        # The argument label decides whether a homogeneous or heterogeneous
        # threatment is considered.
        if labels is None:
            # Homogeneous case
            self._is_homogeneous = True
            assert isinstance(threshold_lower, float)
            assert isinstance(threshold_upper, float) or threshold_upper is None
            self._threshold_lower = threshold_lower
            self._threshold_upper = threshold_upper
            self.num_parameters = 2

        else:
            # Heterogeneous case
            self._is_homogeneous = False
            self._labels = labels
            num_labels = len(np.unique(self._labels))

            if isinstance(threshold_lower, list) or isinstance(
                threshold_lower, np.ndarray
            ):
                # Allow for heterogeneous initial value.
                assert len(threshold_lower) == num_labels
                self._threshold_lower = np.array(threshold_lower)

            elif isinstance(threshold_lower, float):
                # Or initialize all labels with the same value
                self._threshold_lower = threshold_lower * np.ones(
                    num_labels, dtype=float
                )
            else:
                raise ValueError(f"Type {type(threshold_lower)} not supported.")

            if isinstance(threshold_upper, list) or isinstance(
                threshold_upper, np.ndarray
            ):
                # Allow for heterogeneous initial value.
                assert len(threshold_upper) == num_labels
                self._threshold_upper = np.array(threshold_upper)

            elif isinstance(threshold_upper, float):
                # Or initialize all labels with the same value
                self._threshold_upper = threshold_upper * np.ones(
                    num_labels, dtype=float
                )
            elif threshold_upper is None:
                self._threshold_upper = None
            else:
                raise ValueError(f"Type {type(threshold_upper)} not supported.")

            self.num_parameters = 2 * num_labels

    def __call__(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert signal to binary data through thresholding.

        Args:
            img (np.ndarray): signal
            mask (np.ndarray, optional): mask

        Returns:
            np.ndarray: boolean mask
        """
        # Apply thresholding directly to the signal
        if self._is_homogeneous:
            threshold_mask = self._call_homogeneous(img)
        else:
            threshold_mask = self._call_heterogeneous(img)

        # Restrict data to the provided mask
        if mask is None:
            return threshold_mask
        else:
            return np.logical_and(threshold_mask, mask)

    def _call_homogeneous(self, img: np.ndarray) -> np.ndarray:
        """
        Convert signal to binary data through thresholding, tailored for the homogeneous case.

        Args:
            img (np.ndarray): signal

        Returns:
            np.ndarray: boolean mask
        """
        if self._threshold_upper is not None:
            return np.logical_and(
                img > self._threshold_lower, img < self._threshold_upper
            )
        else:
            return img > self._threshold_lower

    def _call_heterogeneous(self, img: np.ndarray) -> np.ndarray:
        """
        Convert signal to binary data through thresholding, tailored for the
        heterogeneous case.

        Args:
            img (np.ndarray): signal

        Returns:
            np.ndarray: boolean mask
        """
        threshold_mask = np.zeros(self._labels.shape[:2], dtype=bool)
        for i, label in enumerate(np.unique(self._labels)):
            threshold_mask_i = img > self._threshold_lower[i]
            if self._threshold_upper is not None:
                threshold_mask_i = np.logical_and(
                    threshold_mask_i, img < self._threshold_upper[i]
                )
            roi = np.logical_and(threshold_mask_i, self._labels == label)
            threshold_mask[roi] = True
        return threshold_mask

    def update_model_parameters(
        self,
        *args: tuple,
    ) -> None:
        raise NotImplementedError(
            "StaticThresholdModel does not support parameter updates."
        )
