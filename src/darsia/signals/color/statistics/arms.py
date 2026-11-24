"""Colour statistics with a dark and a bright arm along the main colour axis.

Under linear colour mixing, a pixel covering pore space and some grain lies on the line
from the typical colour to the grain colour, at the grain fraction. Along the main
colour axis of a facies (mostly brightness), darker colours lie on the dark arm (dark
grains) and brighter ones on the bright arm (bright grains, reflections). The distance
off the axis (side) catches grains of another hue.
"""

import numpy as np

from .base import ColorStatistics, linear_score, percentiles


class ArmStatistics(ColorStatistics):
    """Dark arm, bright arm and side distance around the median colour.

    Shared parameter: the main axis `a` of the colours relative to their group median
    (oriented so that `t = (c - median) . a > 0` is brighter). Per group: the median
    colour and, for the dark arm (`-t`, t < 0), the bright arm (`t`, t > 0) and the side
    distance (`|r - t a|`), the noise and key percentiles. A colour scores 1 up to the
    noise distance, 0 from the key distance on, linearly between; the score is the
    minimum of the switched-on terms.
    """

    parameter_names = (
        "median_r",
        "median_g",
        "median_b",
        "noise_dark",
        "key_dark",
        "noise_bright",
        "key_bright",
        "noise_side",
        "key_side",
    )

    def __init__(
        self,
        dark: bool = True,
        bright: bool = True,
        side: bool = True,
        noise_percentile: float = 90.0,
        key_percentile: float = 99.5,
    ) -> None:
        """
        Parameters
        ----------
        dark, bright, side : bool
            Whether the dark arm, the bright arm and the side distance count.
        noise_percentile : float
            Percentile of the distances below which a colour scores 1.
        key_percentile : float
            Percentile of the distances from which on a colour scores 0.

        """
        if not (dark or bright or side):
            raise ValueError("ArmStatistics needs at least one of dark, bright, side.")
        self.dark = dark
        self.bright = bright
        self.side = side
        self.noise_percentile = noise_percentile
        self.key_percentile = key_percentile

    def fit_shared(self, groups: list[np.ndarray]) -> np.ndarray:
        """Main axis of the colours relative to their group medians."""
        relative = np.concatenate(
            [g - np.median(g, axis=0) for g in groups if len(g) > 0]
        )
        _, _, vt = np.linalg.svd(relative - relative.mean(axis=0), full_matrices=False)
        axis = vt[0]
        return axis if axis.sum() >= 0 else -axis

    def _coordinates(
        self, colors: np.ndarray, median: np.ndarray, axis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Position `t` along the axis and side distance `d` relative to the median."""
        relative = colors - median
        t = relative @ axis
        d = np.linalg.norm(relative - t[:, None] * axis, axis=1)
        return t, d

    def fit(self, colors: np.ndarray, shared: np.ndarray) -> np.ndarray:
        median = np.median(colors, axis=0)
        t, d = self._coordinates(colors, median, shared)
        p = (self.noise_percentile, self.key_percentile)
        return np.array(
            [
                *median,
                *percentiles(-t[t < 0], *p),
                *percentiles(t[t > 0], *p),
                *percentiles(d, *p),
            ]
        )

    def evaluate(
        self, colors: np.ndarray, params: np.ndarray, shared: np.ndarray
    ) -> np.ndarray:
        t, d = self._coordinates(colors, params[..., 0:3], shared)
        ones = np.ones(len(colors))
        dark = linear_score(-t, params[..., 3], params[..., 4]) if self.dark else ones
        bright = (
            linear_score(t, params[..., 5], params[..., 6]) if self.bright else ones
        )
        arm = np.where(t < 0, dark, bright)
        if not self.side:
            return arm
        return np.minimum(arm, linear_score(d, params[..., 7], params[..., 8]))
