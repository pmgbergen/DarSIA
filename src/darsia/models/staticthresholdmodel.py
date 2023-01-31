"""
Module converting signals to binary data by applying thresholding.

A distinction between heterogeneous and homogeneous thresholding
is performed automatically.
"""

import darsia
import numpy as np
from typing import Union, Optional

class StaticThresholdModel(darsia.Model):
    """
    Class for static thresholding.
    """

    def __init__(
            self,
            threshold_low: Union[float, list[float]] = 0.,
            threshold_high: Optional[Union[float, list[float]]] = None,
            labels: Optional[np.ndarray] = None,
            ) -> None:
        """
        Constructor of StaticThresholdModel.

        Args:
            threshold_low (float or list of float): lower threshold value(s)
            threshold_high (float or list of float): upper threshold value(s)
            labels (array): labeled domain
        """

        # The argument label decides whether a homogeneous or heterogeneous
        # threatment is considered.
        if labels is None:
            # Homogeneous case
            self._homogeneous = True
            assert isinstance(threshold_low, float)
            assert isinstance(threshold_high, float) or threshold_high is None
            self._threshold_low = threshold_low
            self._threshold_high = threshold_high

        else:
            # Heterogeneous case
            self._homogeneous = False
            self._labels = labels
            num_labels = len(np.unique(self._labels))

            if isinstance(threshold_low, list) or isinstance(threshold_low, np.ndarray):
                # Allow for heterogeneous initial value.
                assert len(threshold_low) == num_labels
                self._threshold_low = np.array(threshold_low)

            elif isinstance(threshold_low, float):
                # Or initialize all labels with the same value
                self._threshold_low = threshold_low * np.ones(
                    num_labels, dtype=float
                )
            else:
                raise ValueError(f"Type {type(threshold_low)} not supported.")

            if isinstance(threshold_high, list) or isinstance(threshold_high, np.ndarray):
                # Allow for heterogeneous initial value.
                assert len(threshold_high) == num_labels
                self._threshold_high = np.array(threshold_high)

            elif isinstance(threshold_high, float):
                # Or initialize all labels with the same value
                self._threshold_high = threshold_high * np.ones(
                    num_labels, dtype=float
                )
            elif threshold_high is None:
                self._threshold_high = None
            else:
                raise ValueError(f"Type {type(threshold_high)} not supported.")

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert signal to binary data through thresholding.

        Args:
            img (np.ndarray): signal
            mask (np.ndarray, optional): mask

        Returns:
            np.ndarray: boolean mask
        """
        # Apply thresholding directly to the signal
        if self._homogeneous:
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
        if self._threshold_high is not None:
            return np.logical_and(img > self._threshold_low, img < self._threshold_high)
        else:
            return img > self._threshold_low

    def _call_heterogeneous(self, img: np.ndarray) -> np.ndarray:
        """
        Convert signal to binary data through thresholding, tailored for the heterogeneous case.

        Args:
            img (np.ndarray): signal

        Returns:
            np.ndarray: boolean mask
        """
        threshold_mask = np.zeros(self._labels.shape[:2], dtype=bool)
        import matplotlib.pyplot as plt
        for i, label in enumerate(np.unique(self._labels)):
            threshold_mask_i = img > self._threshold_low[i]
            if self._threshold_high is not None:
                threshold_mask_i = np.logical_and(threshold_mask_i, img < self._threshold_high[i])
            roi = np.logical_and(threshold_mask_i, self._labels == label)
            threshold_mask[roi] = True
        return threshold_mask

    def calibrate(self, img: list[np.ndarray]) -> None:
        """
        Empty calibration.

        Args:
            img (list of arrays): images.
        """
        pass
