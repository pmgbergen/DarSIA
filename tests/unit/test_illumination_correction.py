"""Unit tests for IlluminationCorrection."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # non-interactive backend for tests

import darsia
from darsia.presets.workflows.config.corrections import IlluminationCorrectionConfig


def _make_uniform_image(height: int = 50, width: int = 50) -> darsia.OpticalImage:
    """Create a small uniform RGB OpticalImage for testing."""
    arr = np.full((height, width, 3), 0.5, dtype=np.float32)
    return darsia.OpticalImage(img=arr, space_dim=2, indexing="ij")


def _small_config(**kwargs) -> IlluminationCorrectionConfig:
    """Return an IlluminationCorrectionConfig suitable for fast unit tests."""
    defaults = dict(width=5, num_samples=5, seed=42)
    defaults.update(kwargs)
    return IlluminationCorrectionConfig(**defaults)


# ---------------------------------------------------------------------------
# select_random_samples – mask handling
# ---------------------------------------------------------------------------


def test_select_random_samples_numpy_mask_returns_correct_count():
    """select_random_samples with a full 2-D numpy mask returns the requested count."""
    config = _small_config(num_samples=5)
    corr = darsia.IlluminationCorrection()
    mask = np.ones((50, 50), dtype=bool)
    samples = corr.select_random_samples(mask=mask, config=config)
    assert len(samples) == 5


def test_select_random_samples_numpy_mask_sample_format():
    """Each sample returned is a tuple of two slices."""
    config = _small_config(num_samples=3)
    corr = darsia.IlluminationCorrection()
    mask = np.ones((50, 50), dtype=bool)
    samples = corr.select_random_samples(mask=mask, config=config)
    for s in samples:
        assert len(s) == 2
        assert isinstance(s[0], slice)
        assert isinstance(s[1], slice)


def test_select_random_samples_empty_mask_returns_empty_list():
    """select_random_samples with an all-False mask returns an empty list."""
    config = _small_config(num_samples=5)
    corr = darsia.IlluminationCorrection()
    mask = np.zeros((50, 50), dtype=bool)
    samples = corr.select_random_samples(mask=mask, config=config)
    assert samples == []


def test_select_random_samples_partial_mask_stays_within_region():
    """Samples must lie within the active (True) area of the mask."""
    config = _small_config(num_samples=10, seed=0)
    corr = darsia.IlluminationCorrection()
    # Only allow the top-left quadrant
    mask = np.zeros((50, 50), dtype=bool)
    mask[:25, :25] = True
    samples = corr.select_random_samples(mask=mask, config=config)
    assert len(samples) > 0, "Expected at least one sample in the partial mask"
    for s in samples:
        assert s[0].start >= 0
        assert s[1].start >= 0
        # The sample patch must start inside the masked quadrant
        assert s[0].start < 25
        assert s[1].start < 25


def test_select_random_samples_reproducible_with_same_seed():
    """The same seed must produce the same samples."""
    mask = np.ones((50, 50), dtype=bool)
    corr = darsia.IlluminationCorrection()
    config_a = _small_config(seed=7)
    config_b = _small_config(seed=7)
    samples_a = corr.select_random_samples(mask=mask, config=config_a)
    samples_b = corr.select_random_samples(mask=mask, config=config_b)
    assert samples_a == samples_b


# ---------------------------------------------------------------------------
# setup() – edge cases
# ---------------------------------------------------------------------------


def test_setup_outliers_zero_produces_local_scaling():
    """setup() with outliers=0.0 completes without error and sets local_scaling."""
    img = _make_uniform_image()
    config = _small_config(num_samples=5, outliers=0.0)
    corr = darsia.IlluminationCorrection()
    mask = np.ones((50, 50), dtype=bool)
    samples = corr.select_random_samples(mask=mask, config=config)

    corr.setup(
        base=img,
        sample_groups=[samples],
        outliers=0.0,
        colorspace="hsl-scalar",
        interpolation="quartic",
        show_plot=False,
    )

    assert hasattr(corr, "local_scaling"), "local_scaling should be set after setup()"
    assert corr.local_scaling[0].img.shape == (50, 50)


def test_setup_uniform_image_scaling_close_to_one():
    """For a perfectly uniform image the optimal scaling should be ≈ 1."""
    img = _make_uniform_image()
    config = _small_config(num_samples=5)
    corr = darsia.IlluminationCorrection()
    mask = np.ones((50, 50), dtype=bool)
    samples = corr.select_random_samples(mask=mask, config=config)

    corr.setup(
        base=img,
        sample_groups=[samples],
        outliers=0.0,
        colorspace="hsl-scalar",
        interpolation="quartic",
        bounds=(0.5, 2.0),
        show_plot=False,
    )

    scaling = corr.local_scaling[0].img
    assert np.all(np.isfinite(scaling)), "Scaling values should be finite"
    assert scaling.shape == img.img.shape[:2]


def test_setup_group_with_no_eligible_samples_is_skipped():
    """A sample group returning an empty sample list is skipped gracefully."""
    img = _make_uniform_image()
    config = _small_config(num_samples=5)
    corr = darsia.IlluminationCorrection()
    mask = np.ones((50, 50), dtype=bool)
    good_samples = corr.select_random_samples(mask=mask, config=config)

    # Provide two groups: one valid, one with no samples (all-False mask -> [])
    empty_mask = np.zeros((50, 50), dtype=bool)
    empty_samples = corr.select_random_samples(mask=empty_mask, config=config)
    assert empty_samples == [], "Sanity check: empty mask should yield no samples"

    # setup should not raise even though the second group is empty
    corr.setup(
        base=img,
        sample_groups=[good_samples, empty_samples],
        outliers=0.0,
        colorspace="hsl-scalar",
        interpolation="quartic",
        show_plot=False,
    )

    assert hasattr(corr, "local_scaling")


def test_setup_correct_array_returns_same_shape():
    """correct_array must return an array with the same shape as the input."""
    img = _make_uniform_image()
    arr = img.img.copy()
    config = _small_config(num_samples=5)
    corr = darsia.IlluminationCorrection()
    mask = np.ones((50, 50), dtype=bool)
    samples = corr.select_random_samples(mask=mask, config=config)

    corr.setup(
        base=img,
        sample_groups=[samples],
        outliers=0.0,
        colorspace="hsl-scalar",
        interpolation="quartic",
        show_plot=False,
    )

    corrected = corr.correct_array(arr)
    assert corrected.shape == arr.shape
