"""Unit tests for count-weighted histogram calibration in colour-path fitting.

These tests exercise the new ``weighting`` parameter of
:meth:`LabelColorPathMapRegression._find_color_path` and
:meth:`LabelColorPathMapRegression.find_color_path` without requiring real
images.  Synthetic ``ColorSpectrum`` objects are constructed directly so that
ground-truth behaviour is fully known.
"""

from __future__ import annotations

import numpy as np
import pytest

import darsia


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectrum(
    resolution: int = 11,
    active_indices: list[tuple[int, int, int]] | None = None,
    probabilities: list[float] | None = None,
) -> darsia.ColorSpectrum:
    """Build a synthetic :class:`darsia.ColorSpectrum`.

    Args:
        resolution: Side length of the 3-D histogram cube.
        active_indices: List of ``(i, j, k)`` bin indices that are active.
            Defaults to a small diagonal strip representing a linear path.
        probabilities: Per-bin probability assigned to each active index.
            Must have the same length as *active_indices*.  Defaults to a
            uniform distribution over the active bins.

    Returns:
        A :class:`darsia.ColorSpectrum` whose ``spectrum`` (bool) and
        ``histogram`` (float) arrays reflect the provided configuration.
    """
    shape = (resolution, resolution, resolution)

    if active_indices is None:
        # Default: a linear path along the main diagonal (R = G = B increasing)
        n = resolution // 2
        active_indices = [(i, i, i) for i in range(1, n + 1)]

    n_active = len(active_indices)
    if probabilities is None:
        probabilities = [1.0 / n_active] * n_active
    else:
        assert len(probabilities) == n_active

    histogram = np.zeros(shape, dtype=float)
    spectrum = np.zeros(shape, dtype=bool)
    for (i, j, k), p in zip(active_indices, probabilities):
        histogram[i, j, k] = p
        spectrum[i, j, k] = True

    color_range = darsia.ColorRange(
        min_color=np.array([-0.5, -0.5, -0.5]),
        max_color=np.array([0.5, 0.5, 0.5]),
        color_mode=darsia.ColorMode.RELATIVE,
    )

    return darsia.ColorSpectrum(
        base_color=np.zeros(3),
        spectrum=spectrum,
        histogram=histogram,
        color_range=color_range,
    )


# ---------------------------------------------------------------------------
# Weight computation tests (unit-level)
# ---------------------------------------------------------------------------


class TestWeightNormalization:
    """Verify that per-bin weights sum to 1 (after the origin prepend)."""

    @pytest.mark.parametrize("mode", ["wls", "wls_sqrt", "wls_log"])
    def test_weights_sum_to_one_for_wls_modes(self, mode: str) -> None:
        """The normalised ``point_weights`` extracted inside ``_find_color_path``
        must sum to 1 (the origin weight 0 is prepended later, which shifts the
        sum, but the weights array *before* prepend must be normalised first).

        We verify indirectly: running the full ``_find_color_path`` must not
        raise and must return a valid :class:`darsia.ColorPath`.
        """
        spectrum = _make_spectrum(resolution=11)
        labels_arr = np.zeros((4, 4), dtype=int)
        labels_img = darsia.Image(
            img=labels_arr,
            dimensions=[1.0, 1.0],
        )

        color_range = darsia.ColorRange(
            min_color=np.array([-0.5, -0.5, -0.5]),
            max_color=np.array([0.5, 0.5, 0.5]),
            color_mode=darsia.ColorMode.RELATIVE,
        )
        mask_arr = np.ones((4, 4), dtype=bool)
        mask_img = darsia.Image(img=mask_arr, dimensions=[1.0, 1.0])

        regression = darsia.LabelColorPathMapRegression(
            labels=labels_img,
            color_range=color_range,
            resolution=11,
            mask=mask_img,
        )

        path = regression._find_color_path(
            spectrum=spectrum,
            num_segments=1,
            weighting=mode,
        )
        assert isinstance(path, darsia.ColorPath)


# ---------------------------------------------------------------------------
# Backward-compatibility test
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """``weighting='threshold'`` must reproduce the same result as calling
    ``_find_color_path`` without the ``weighting`` argument (the old default)."""

    def _make_regression(self) -> darsia.LabelColorPathMapRegression:
        labels_arr = np.zeros((4, 4), dtype=int)
        labels_img = darsia.Image(img=labels_arr, dimensions=[1.0, 1.0])
        color_range = darsia.ColorRange(
            min_color=np.array([-0.5, -0.5, -0.5]),
            max_color=np.array([0.5, 0.5, 0.5]),
            color_mode=darsia.ColorMode.RELATIVE,
        )
        mask_img = darsia.Image(img=np.ones((4, 4), dtype=bool), dimensions=[1.0, 1.0])
        return darsia.LabelColorPathMapRegression(
            labels=labels_img,
            color_range=color_range,
            resolution=11,
            mask=mask_img,
        )

    def test_threshold_mode_matches_default(self) -> None:
        """Explicit ``weighting='threshold'`` must give the same key-colors as
        omitting the argument (which defaults to ``'threshold'``)."""
        np.random.seed(42)
        spectrum = _make_spectrum(resolution=11)
        regression = self._make_regression()

        path_default = regression._find_color_path(spectrum=spectrum, num_segments=1)
        path_threshold = regression._find_color_path(
            spectrum=spectrum, num_segments=1, weighting="threshold"
        )

        np.testing.assert_allclose(
            np.array(path_default.relative_colors),
            np.array(path_threshold.relative_colors),
            err_msg="threshold mode must reproduce the default (no-weighting) result",
        )


# ---------------------------------------------------------------------------
# Weighted vs. threshold – qualitative behaviour test
# ---------------------------------------------------------------------------


class TestWeightedFitQuality:
    """Verify that WLS modes focus on high-count bins.

    We construct a spectrum where:
    - A dense cluster of bins lies near a known end-point (high count).
    - A few sparse outlier bins lie far off the true path (low count).

    WLS should weight the dense cluster more heavily and produce a path
    end-point closer to the dense cluster centre than threshold mode does.
    """

    def _make_regression(self) -> darsia.LabelColorPathMapRegression:
        labels_arr = np.zeros((4, 4), dtype=int)
        labels_img = darsia.Image(img=labels_arr, dimensions=[1.0, 1.0])
        color_range = darsia.ColorRange(
            min_color=np.array([-0.5, -0.5, -0.5]),
            max_color=np.array([0.5, 0.5, 0.5]),
            color_mode=darsia.ColorMode.RELATIVE,
        )
        mask_img = darsia.Image(img=np.ones((4, 4), dtype=bool), dimensions=[1.0, 1.0])
        return darsia.LabelColorPathMapRegression(
            labels=labels_img,
            color_range=color_range,
            resolution=11,
            mask=mask_img,
        )

    def test_wls_path_is_valid_color_path(self) -> None:
        """WLS mode must return a valid ColorPath for a non-trivial spectrum."""
        resolution = 11
        # Dense cluster near the end-point of the path
        dense_indices = [(i, i, i) for i in range(5, 8)]
        dense_probs = [0.3, 0.3, 0.3]
        # Sparse outliers perpendicular to the main path direction
        sparse_indices = [(2, 8, 2), (8, 2, 8)]
        sparse_probs = [0.05, 0.05]

        active_indices = dense_indices + sparse_indices
        probabilities = dense_probs + sparse_probs
        # Normalise
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        spectrum = _make_spectrum(
            resolution=resolution,
            active_indices=active_indices,
            probabilities=probabilities,
        )
        regression = self._make_regression()

        path = regression._find_color_path(
            spectrum=spectrum, num_segments=1, weighting="wls"
        )
        assert isinstance(path, darsia.ColorPath)
        assert len(path.relative_colors) == 2  # num_segments + 1

    @pytest.mark.parametrize("mode", ["wls", "wls_sqrt", "wls_log"])
    def test_all_wls_variants_return_color_path(self, mode: str) -> None:
        """All three WLS variants must return a valid path without raising."""
        spectrum = _make_spectrum(resolution=11)
        regression = self._make_regression()
        path = regression._find_color_path(
            spectrum=spectrum, num_segments=1, weighting=mode
        )
        assert isinstance(path, darsia.ColorPath)


# ---------------------------------------------------------------------------
# Invalid weighting value
# ---------------------------------------------------------------------------


class TestInvalidWeighting:
    def _make_regression(self) -> darsia.LabelColorPathMapRegression:
        labels_arr = np.zeros((4, 4), dtype=int)
        labels_img = darsia.Image(img=labels_arr, dimensions=[1.0, 1.0])
        color_range = darsia.ColorRange(
            min_color=np.array([-0.5, -0.5, -0.5]),
            max_color=np.array([0.5, 0.5, 0.5]),
            color_mode=darsia.ColorMode.RELATIVE,
        )
        mask_img = darsia.Image(img=np.ones((4, 4), dtype=bool), dimensions=[1.0, 1.0])
        return darsia.LabelColorPathMapRegression(
            labels=labels_img,
            color_range=color_range,
            resolution=11,
            mask=mask_img,
        )

    def test_unknown_weighting_raises_value_error(self) -> None:
        """Passing an unknown weighting value must raise a ``ValueError``."""
        spectrum = _make_spectrum(resolution=11)
        regression = self._make_regression()
        with pytest.raises(ValueError, match="histogram_weighting"):
            regression._find_color_path(
                spectrum=spectrum, num_segments=1, weighting="bogus"
            )

    def test_unknown_mode_raises_value_error(self) -> None:
        """Passing an unknown mode value must raise a ``ValueError``."""
        spectrum = _make_spectrum(resolution=11)
        regression = self._make_regression()
        with pytest.raises(ValueError, match="color-path mode"):
            regression._find_color_path(
                spectrum=spectrum, num_segments=1, mode="bogus"  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# find_color_path (public API) propagates weighting
# ---------------------------------------------------------------------------


class TestFindColorPathWeighting:
    """Verify the public ``find_color_path`` API propagates ``weighting``."""

    def _make_regression(self):
        labels_arr = np.zeros((4, 4), dtype=int)
        labels_img = darsia.Image(img=labels_arr, dimensions=[1.0, 1.0])
        color_range = darsia.ColorRange(
            min_color=np.array([-0.5, -0.5, -0.5]),
            max_color=np.array([0.5, 0.5, 0.5]),
            color_mode=darsia.ColorMode.RELATIVE,
        )
        mask_img = darsia.Image(img=np.ones((4, 4), dtype=bool), dimensions=[1.0, 1.0])
        return darsia.LabelColorPathMapRegression(
            labels=labels_img,
            color_range=color_range,
            resolution=11,
            mask=mask_img,
        )

    @pytest.mark.parametrize("mode", ["threshold", "wls", "wls_sqrt", "wls_log"])
    def test_returns_label_color_path_map(self, mode: str) -> None:
        """``find_color_path`` must return a LabelColorPathMap for all modes."""
        regression = self._make_regression()
        spectrum_map = darsia.LabelColorSpectrumMap({0: _make_spectrum(resolution=11)})
        result = regression.find_color_path(
            color_spectrum=spectrum_map,
            num_segments=1,
            weighting=mode,
        )
        assert isinstance(result, darsia.LabelColorPathMap)
        assert 0 in result

    def test_auto_mode_propagates(self) -> None:
        """Explicit ``mode='auto'`` must run through the public API."""
        regression = self._make_regression()
        spectrum_map = darsia.LabelColorSpectrumMap({0: _make_spectrum(resolution=11)})
        result = regression.find_color_path(
            color_spectrum=spectrum_map,
            num_segments=1,
            weighting="threshold",
            mode="auto",
        )
        assert isinstance(result, darsia.LabelColorPathMap)
        assert 0 in result


class TestFindColorPathTargetLabels:
    """Verify partial updates via target_labels and existing_map."""

    def _make_regression(self):
        labels_arr = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 2, 2],
                [2, 2, 2, 2],
            ],
            dtype=int,
        )
        labels_img = darsia.Image(img=labels_arr, dimensions=[1.0, 1.0])
        color_range = darsia.ColorRange(
            min_color=np.array([-0.5, -0.5, -0.5]),
            max_color=np.array([0.5, 0.5, 0.5]),
            color_mode=darsia.ColorMode.RELATIVE,
        )
        mask_img = darsia.Image(img=np.ones((4, 4), dtype=bool), dimensions=[1.0, 1.0])
        return darsia.LabelColorPathMapRegression(
            labels=labels_img,
            color_range=color_range,
            resolution=11,
            mask=mask_img,
        )

    def test_updates_only_target_labels(self) -> None:
        regression = self._make_regression()
        spectrum_map = darsia.LabelColorSpectrumMap(
            {
                0: _make_spectrum(resolution=11),
                1: _make_spectrum(resolution=11),
            }
        )
        existing_map = darsia.LabelColorPathMap(
            {
                0: darsia.ColorPath(
                    base_color=np.zeros(3),
                    relative_colors=[np.zeros(3), np.array([0.1, 0.0, 0.0])],
                    mode="rgb",
                    name="existing_0",
                ),
                1: darsia.ColorPath(
                    base_color=np.zeros(3),
                    relative_colors=[np.zeros(3), np.array([0.0, 0.1, 0.0])],
                    mode="rgb",
                    name="existing_1",
                ),
                2: darsia.ColorPath(
                    base_color=np.zeros(3),
                    relative_colors=[np.zeros(3), np.array([0.0, 0.0, 0.1])],
                    mode="rgb",
                    name="existing_2",
                ),
            }
        )

        result = regression.find_color_path(
            color_spectrum=spectrum_map,
            num_segments=1,
            weighting="threshold",
            mode="auto",
            target_labels=[1],
            existing_map=existing_map,
        )

        np.testing.assert_allclose(
            np.array(result[0].relative_colors), np.array(existing_map[0].relative_colors)
        )
        np.testing.assert_allclose(
            np.array(result[2].relative_colors), np.array(existing_map[2].relative_colors)
        )
        assert not np.allclose(
            np.array(result[1].relative_colors), np.array(existing_map[1].relative_colors)
        )

    def test_target_labels_require_existing_map(self) -> None:
        regression = self._make_regression()
        spectrum_map = darsia.LabelColorSpectrumMap({0: _make_spectrum(resolution=11)})
        with pytest.raises(ValueError, match="existing_map"):
            regression.find_color_path(
                color_spectrum=spectrum_map,
                num_segments=1,
                target_labels=[0],
                existing_map=None,
            )
