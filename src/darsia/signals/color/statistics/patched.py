"""Colour statistics per patch and label, interpolated smoothly over the image.

Illumination varies over an image, so the colour statistics of a facies vary in space.
Assuming the same facies has the same colour distribution up to illumination, the
statistics are fitted per patch and label and interpolated into smooth fields; every
pixel is then scored with the statistics at its location.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.interpolate
import scipy.ndimage

import darsia

from .base import ColorStatistics

logger = logging.getLogger(__name__)


class _SmoothField:
    """Cubic spline through values at patch centres; linear beyond the outer centres.

    Beyond the outer centres (half a patch at each image border) the field continues
    with the slope at the outer centre, so trends towards the borders (e.g. vignetting)
    are followed.
    """

    def __init__(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        self.rows = rows
        self.cols = cols
        self.values = values
        self.spline = None
        self.spline_1d = None
        if len(rows) > 1 and len(cols) > 1:
            self.spline = scipy.interpolate.RectBivariateSpline(
                rows, cols, values, kx=min(3, len(rows) - 1), ky=min(3, len(cols) - 1)
            )
        elif len(rows) > 1:
            self.spline_1d = scipy.interpolate.make_interp_spline(
                rows, values[:, 0], k=min(3, len(rows) - 1)
            )
        elif len(cols) > 1:
            self.spline_1d = scipy.interpolate.make_interp_spline(
                cols, values[0], k=min(3, len(cols) - 1)
            )

    def __call__(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Values on the grid `rows` x `cols` (increasing pixel indices)."""
        r = np.clip(rows, self.rows[0], self.rows[-1])
        c = np.clip(cols, self.cols[0], self.cols[-1])
        dr = (rows - r)[:, None]
        dc = (cols - c)[None, :]
        if self.spline is not None:
            values = self.spline(r, c)
            # Slopes only at the outer centres (one row or column each).
            for edge in (self.rows[0], self.rows[-1]):
                beyond = (r == edge) & (rows != edge)
                if beyond.any():
                    slope = self.spline([edge], c, dx=1)
                    values[beyond] += dr[beyond] * slope
            for edge in (self.cols[0], self.cols[-1]):
                beyond = (c == edge) & (cols != edge)
                if beyond.any():
                    slope = self.spline(r, [edge], dy=1)
                    values[:, beyond] += dc[:, beyond] * slope
            return values
        if self.spline_1d is None:
            return np.full((len(r), len(c)), self.values[0, 0])
        if len(self.rows) > 1:
            line = self.spline_1d(r) + dr[:, 0] * self.spline_1d(r, nu=1)
            return np.repeat(line[:, None], len(c), axis=1)
        line = self.spline_1d(c) + dc[0] * self.spline_1d(c, nu=1)
        return np.repeat(line[None, :], len(r), axis=0)


def _smooth_grid(values: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised Gaussian smoothing of a parameter grid; fills cells without weight.

    Parameters
    ----------
    values : np.ndarray
        Parameters (P, R, C); cells with weight 0 are ignored.
    weights : np.ndarray
        Weights (R, C), e.g. pixel counts.
    sigma : float
        Standard deviation in cells; 0 keeps the values and only fills the gaps.

    Returns
    -------
    np.ndarray
        Smoothed parameters (P, R, C).

    """
    values = np.where(weights > 0, values, 0.0)
    result = np.where(weights > 0, values, np.nan)
    missing = weights <= 0
    s = sigma
    while True:
        if s > 0 and (s == sigma or missing.any()):
            den = scipy.ndimage.gaussian_filter(weights, s, mode="nearest")
            num = scipy.ndimage.gaussian_filter(
                values * weights, (0, s, s), mode="nearest"
            )
            valid = den > 1e-12 * weights.max()
            update = valid if s == sigma else (missing & valid)
            result[:, update] = num[:, update] / den[update]
            missing &= ~valid
        if not missing.any():
            return result
        s = max(2 * s, 1.0)


class PatchedColorStatistics:
    """A colour statistic per patch and label, as smooth fields over the image.

    Fit: per label, the shared parameters from a sample of all patches; per patch and
    label with at least `min_pixels` pixels, the parameters of the statistic. The
    parameters at the patch centres are smoothed (normalised Gaussian, weights = pixel
    counts; fills patches without enough pixels) and interpolated by cubic splines.
    Evaluation: every pixel is scored with the parameters of its label at its location.
    """

    def __init__(
        self,
        statistics: ColorStatistics,
        patches: tuple[int, int] = (20, 20),
        min_pixels: int = 2000,
        smoothing: float = 0.0,
        sample_size: int = 200_000,
        group_size: int = 100_000,
        seed: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        statistics : ColorStatistics
            Statistic fitted per patch and label.
        patches : tuple[int, int]
            Number of patches (rows, cols).
        min_pixels : int
            Pixels of a label a patch needs for its own fit.
        smoothing : float
            Gaussian standard deviation on the patch grid, in patches.
        sample_size : int
            Pixels per label for the shared parameters.
        group_size : int
            Maximum pixels per patch and label for a fit (random sample).
        seed : int
            Seed of the random samples.

        """
        self.statistics = statistics
        self.patches = (int(patches[0]), int(patches[1]))
        self.min_pixels = int(min_pixels)
        self.smoothing = float(smoothing)
        self.sample_size = int(sample_size)
        self.group_size = int(group_size)
        self.seed = seed
        self.shared: dict[int, np.ndarray] = {}
        """Shared parameters per label."""
        self.grids: dict[int, np.ndarray] = {}
        """Smoothed parameters (P, R, C) per label at the patch centres."""
        self.counts: dict[int, np.ndarray] = {}
        """Pixels (R, C) per label and patch."""
        self.fields: dict[int, list[_SmoothField]] = {}
        """Interpolated parameter fields per label."""

    @staticmethod
    def _sample(
        img: np.ndarray, mask: np.ndarray, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Colours of at most `size` random pixels of `mask` (no copy of the image)."""
        idx = np.flatnonzero(mask)
        if len(idx) > size:
            idx = rng.choice(idx, size, replace=False)
        rows, cols = np.divmod(idx, mask.shape[1])
        return np.asarray(img[rows, cols], dtype=float)

    def fit(self, image: darsia.Image, labels: darsia.Image) -> PatchedColorStatistics:
        """Fit the statistic per patch and label and build the smooth fields.

        Parameters
        ----------
        image : darsia.Image
            Colour image.
        labels : darsia.Image
            Labels on the grid of the image.

        Returns
        -------
        PatchedColorStatistics
            Self.

        """
        if labels.img.shape != image.img.shape[:2]:
            raise ValueError(
                f"Labels {labels.img.shape} and image {image.img.shape[:2]} differ."
            )
        rng = np.random.default_rng(self.seed)
        height, width = labels.img.shape
        num_rows, num_cols = self.patches
        rois = darsia.Patches(image, list(self.patches)).rois
        row_centres = np.array(
            [(r.start + min(r.stop, height) - 1) / 2 for r, _ in (x[0] for x in rois)]
        )
        col_centres = np.array(
            [(c.start + min(c.stop, width) - 1) / 2 for _, c in rois[0]]
        )

        # Pass 1: pixel counts and small samples per label for the shared parameters.
        group_cap = max(1000, self.sample_size // (num_rows * num_cols))
        samples: dict[int, list[np.ndarray]] = {}
        for i in range(num_rows):
            for j in range(num_cols):
                lab = labels.img[rois[i][j]]
                img = image.img[rois[i][j]]
                values, counts = np.unique(lab, return_counts=True)
                for k, n in zip(values.tolist(), counts.tolist()):
                    self.counts.setdefault(k, np.zeros((num_rows, num_cols)))[i, j] = n
                    if n >= self.min_pixels:
                        samples.setdefault(k, []).append(
                            self._sample(img, lab == k, group_cap, rng)
                        )
        for k in self.counts:
            if k not in samples:
                # No patch with enough pixels: one group of all patches.
                samples[k] = [self._pooled(image, labels, k, rng)]
            self.shared[k] = self.statistics.fit_shared(samples[k])

        # Pass 2: parameters per patch and label.
        num_params = self.statistics.num_parameters
        raw = {k: np.zeros((num_params, num_rows, num_cols)) for k in self.counts}
        weights = {k: np.zeros((num_rows, num_cols)) for k in self.counts}
        for i in range(num_rows):
            for j in range(num_cols):
                lab = labels.img[rois[i][j]]
                img = image.img[rois[i][j]]
                for k in self.counts:
                    n = self.counts[k][i, j]
                    if n < self.min_pixels:
                        continue
                    colors = self._sample(img, lab == k, self.group_size, rng)
                    raw[k][:, i, j] = self.statistics.fit(colors, self.shared[k])
                    weights[k][i, j] = n

        # Smooth fields.
        for k in self.counts:
            if weights[k].max() <= 0:
                colors = np.concatenate(samples[k])
                params = self.statistics.fit(colors, self.shared[k])
                self.grids[k] = np.broadcast_to(
                    params[:, None, None], raw[k].shape
                ).copy()
            else:
                self.grids[k] = _smooth_grid(raw[k], weights[k], self.smoothing)
            self.fields[k] = [
                _SmoothField(row_centres, col_centres, grid) for grid in self.grids[k]
            ]
        return self

    def _pooled(
        self,
        image: darsia.Image,
        labels: darsia.Image,
        label: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample of all pixels of a label."""
        return self._sample(image.img, labels.img == label, self.sample_size, rng)

    def evaluate(
        self, image: darsia.Image, labels: darsia.Image, block_pixels: int = 500_000
    ) -> darsia.Image:
        """Score every pixel with the parameters of its label at its location.

        Parameters
        ----------
        image : darsia.Image
            Colour image (on the grid of the fit).
        labels : darsia.Image
            Labels.
        block_pixels : int
            Pixels per block of rows (bounds the memory).

        Returns
        -------
        darsia.Image
            Values in [0, 1] (float32); 0 for labels without a fit.

        """
        result = darsia.zeros_like(image, mode="voxels", dtype=np.float32)
        height, width = labels.img.shape
        block = max(1, block_pixels // width)
        for r0 in range(0, height, block):
            r1 = min(r0 + block, height)
            rows = np.arange(r0, r1)
            lab = labels.img[r0:r1]
            img = image.img[r0:r1]
            out = result.img[r0:r1]
            for k in np.unique(lab).tolist():
                if k not in self.fields:
                    continue
                mask = lab == k
                used = np.flatnonzero(mask.any(axis=0))
                c0, c1 = used[0], used[-1] + 1
                sub = mask[:, c0:c1]
                cols = np.arange(c0, c1)
                params = np.empty((len(self.fields[k]), int(sub.sum())))
                for p, field in enumerate(self.fields[k]):
                    params[p] = field(rows, cols)[sub]
                colors = np.asarray(img[:, c0:c1][sub], dtype=float)
                out[:, c0:c1][sub] = self.statistics.evaluate(
                    colors, params.T, self.shared[k]
                )
        return result

    def __call__(self, image: darsia.Image, labels: darsia.Image) -> darsia.Image:
        """Fit and evaluate on the same image."""
        return self.fit(image, labels).evaluate(image, labels)

    def summary(self) -> dict[int, dict]:
        """Per label: shared parameters, patches with an own fit, parameter ranges."""
        names = self.statistics.parameter_names
        return {
            k: {
                "shared": np.round(self.shared[k], 4).tolist(),
                "patches": int((self.counts[k] >= self.min_pixels).sum()),
                "ranges": {
                    name: [
                        round(float(np.min(grid)), 4),
                        round(float(np.max(grid)), 4),
                    ]
                    for name, grid in zip(names, self.grids[k])
                },
            }
            for k in self.grids
        }
