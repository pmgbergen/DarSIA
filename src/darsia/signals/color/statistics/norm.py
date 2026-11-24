"""Colour statistics by the distance from the median colour (direction-free)."""

import numpy as np

from .base import ColorStatistics, linear_score, percentiles


class NormStatistics(ColorStatistics):
    """Distance from the median colour.

    Per group: the median colour and the noise and key percentiles of the distances
    `|c - median|`. A colour scores 1 up to the noise distance, 0 from the key distance
    on, linearly between.
    """

    parameter_names = ("median_r", "median_g", "median_b", "noise", "key")

    def __init__(
        self, noise_percentile: float = 90.0, key_percentile: float = 99.5
    ) -> None:
        """
        Parameters
        ----------
        noise_percentile : float
            Percentile of the distances below which a colour scores 1.
        key_percentile : float
            Percentile of the distances from which on a colour scores 0.

        """
        self.noise_percentile = noise_percentile
        self.key_percentile = key_percentile

    def fit(self, colors: np.ndarray, shared: np.ndarray) -> np.ndarray:
        median = np.median(colors, axis=0)
        distance = np.linalg.norm(colors - median, axis=1)
        p = (self.noise_percentile, self.key_percentile)
        return np.array([*median, *percentiles(distance, *p)])

    def evaluate(
        self, colors: np.ndarray, params: np.ndarray, shared: np.ndarray
    ) -> np.ndarray:
        distance = np.linalg.norm(colors - params[..., 0:3], axis=1)
        return linear_score(distance, params[..., 3], params[..., 4])
