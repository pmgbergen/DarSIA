"""Base class of colour statistics.

A colour statistic characterises the colours of a group of pixels (e.g. one facies in
one patch) by a few numbers, and scores colours against them: 1 for a typical colour,
0 for a key (outlier) colour. The numbers form a flat vector, so that they can be
interpolated in space (`PatchedColorStatistics`).
"""

from abc import ABC, abstractmethod

import numpy as np

import darsia


class ColorStatistics(ABC):
    """Characteristic colours of a distribution and a score of colours against them.

    Two kinds of parameters:

    - shared parameters, fitted once on all groups together (e.g. a colour axis that
      must be the same everywhere);
    - parameters of one group (`parameter_names`), fitted per group with the shared
      ones fixed.

    """

    parameter_names: tuple[str, ...] = ()
    """Names of the parameters of one group, in the order of the fitted vector."""

    @property
    def num_parameters(self) -> int:
        """Number of parameters of one group."""
        return len(self.parameter_names)

    def fit_shared(self, groups: list[np.ndarray]) -> np.ndarray:
        """Shared parameters of several groups of colours (default: none).

        Parameters
        ----------
        groups : list[np.ndarray]
            Colours (N_i, 3) of each group.

        Returns
        -------
        np.ndarray
            Shared parameters (S,).

        """
        return np.zeros(0)

    @abstractmethod
    def fit(self, colors: np.ndarray, shared: np.ndarray) -> np.ndarray:
        """Parameters of one group of colours.

        Parameters
        ----------
        colors : np.ndarray
            Colours (N, 3).
        shared : np.ndarray
            Shared parameters (`fit_shared`).

        Returns
        -------
        np.ndarray
            Parameters (P,), in the order of `parameter_names`.

        """

    @abstractmethod
    def evaluate(
        self, colors: np.ndarray, params: np.ndarray, shared: np.ndarray
    ) -> np.ndarray:
        """Score colours: 1 typical, 0 key colour or beyond.

        Parameters
        ----------
        colors : np.ndarray
            Colours (N, 3).
        params : np.ndarray
            Parameters, constant (P,) or per colour (N, P).
        shared : np.ndarray
            Shared parameters.

        Returns
        -------
        np.ndarray
            Values (N,) in [0, 1].

        """

    def __call__(
        self, image: darsia.Image, mask: np.ndarray | None = None
    ) -> darsia.Image:
        """Fit on the (masked) pixels of an image and score all its pixels.

        One set of parameters for the whole image; see `PatchedColorStatistics` for
        parameters varying in space.

        Parameters
        ----------
        image : darsia.Image
            Colour image.
        mask : np.ndarray, optional
            Pixels to fit on (default: all).

        Returns
        -------
        darsia.Image
            Values in [0, 1] (float32).

        """
        colors = image.img.reshape(-1, 3)
        sample = colors if mask is None else image.img[mask]
        shared = self.fit_shared([sample])
        params = self.fit(sample, shared)
        result = darsia.zeros_like(image, mode="voxels", dtype=np.float32)
        result.img[...] = self.evaluate(colors, params, shared).reshape(
            result.img.shape
        )
        return result


def linear_score(
    distance: np.ndarray, noise: np.ndarray, key: np.ndarray
) -> np.ndarray:
    """Linear score of distances: 1 up to `noise`, 0 from `key` on.

    A degenerate range (`key <= noise`, no outliers) scores 1 everywhere.

    Parameters
    ----------
    distance : np.ndarray
        Distances (N,).
    noise, key : np.ndarray
        Noise and key distance, scalars or (N,).

    Returns
    -------
    np.ndarray
        Values (N,) in [0, 1].

    """
    width = key - noise
    valid = width > 1e-12
    safe_width = np.where(valid, width, 1.0)
    value = 1.0 - np.clip((distance - noise) / safe_width, 0.0, 1.0)
    return np.where(valid, value, 1.0)


def percentiles(values: np.ndarray, noise: float, key: float) -> tuple[float, float]:
    """Noise and key percentile of values; (0, 0) for no values."""
    if values.size == 0:
        return 0.0, 0.0
    low, high = np.percentile(values, [noise, key])
    return float(low), float(high)
