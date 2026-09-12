"""Unit tests for PatchwiseIlluminationCorrection."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

import darsia


class TestPatchwiseIlluminationCorrectionShortCircuit:
    """Test short-circuit behavior for deserialization path."""

    def test_init_with_none_baseline_images_short_circuits(self):
        """Instantiation with baseline_images=None should short-circuit."""
        correction = darsia.PatchwiseIlluminationCorrection(
            baseline_images=None, labels=darsia.Image(np.zeros((10, 10), dtype=int))
        )
        assert correction.r_diff is None
        assert correction.g_diff is None
        assert correction.b_diff is None

    def test_init_with_none_labels_short_circuits(self):
        """Instantiation with labels=None should short-circuit."""
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        correction = darsia.PatchwiseIlluminationCorrection(
            baseline_images=[dummy_image], labels=None
        )
        assert correction.r_diff is None
        assert correction.g_diff is None
        assert correction.b_diff is None

    def test_init_with_no_args_short_circuits(self):
        """Instantiation with no args (used by read_correction) should short-circuit."""
        correction = darsia.PatchwiseIlluminationCorrection()
        assert correction.r_diff is None
        assert correction.g_diff is None
        assert correction.b_diff is None


class TestPatchwiseIlluminationCorrectionCalibration:
    """Test calibration with synthetic multi-label images."""

    @pytest.fixture
    def two_label_image(self):
        """Create a simple 2-label synthetic image (left half label 0, right half label 1)."""
        labels = np.zeros((100, 100), dtype=int)
        labels[:, 50:] = 1
        return darsia.Image(img=labels)

    @pytest.fixture
    def two_baseline_images(self, two_label_image):
        """Create two baseline images with different colors per label."""
        h, w = two_label_image.img.shape[:2]
        baseline1 = np.zeros((h, w, 3), dtype=np.uint8)
        baseline1[two_label_image.img == 0] = [100, 100, 100]  # Label 0: gray
        baseline1[two_label_image.img == 1] = [50, 50, 50]     # Label 1: darker gray

        baseline2 = np.zeros((h, w, 3), dtype=np.uint8)
        baseline2[two_label_image.img == 0] = [150, 150, 150]  # Label 0: lighter gray
        baseline2[two_label_image.img == 1] = [80, 80, 80]     # Label 1: darker gray

        return [darsia.Image(img=baseline1), darsia.Image(img=baseline2)]

    def test_calibration_produces_finite_correction_arrays(
        self, two_label_image, two_baseline_images
    ):
        """Calibration should produce finite full-resolution correction arrays."""
        correction = darsia.PatchwiseIlluminationCorrection(
            baseline_images=two_baseline_images,
            labels=two_label_image,
            nw=10,
            eps=1e-6,
            show_images=False,
        )
        assert correction.r_diff is not None
        assert correction.g_diff is not None
        assert correction.b_diff is not None
        assert correction.r_diff.shape == (100, 100)
        assert correction.g_diff.shape == (100, 100)
        assert correction.b_diff.shape == (100, 100)
        assert np.all(np.isfinite(correction.r_diff))
        assert np.all(np.isfinite(correction.g_diff))
        assert np.all(np.isfinite(correction.b_diff))

    def test_correct_array_produces_finite_output(
        self, two_label_image, two_baseline_images
    ):
        """correct_array should produce finite output when applied to an image."""
        correction = darsia.PatchwiseIlluminationCorrection(
            baseline_images=two_baseline_images,
            labels=two_label_image,
            nw=10,
            show_images=False,
        )
        test_image = two_baseline_images[0].astype(np.float32)
        corrected = correction.correct_array(test_image)
        assert corrected.shape == test_image.shape
        assert np.all(np.isfinite(corrected))

    def test_shape_mismatch_raises_error(self, two_baseline_images):
        """Mismatched baseline/labels shapes should raise AssertionError."""
        bad_labels = darsia.Image(img=np.zeros((50, 50), dtype=int))  # Wrong shape
        with pytest.raises(AssertionError, match="does not match"):
            darsia.PatchwiseIlluminationCorrection(
                baseline_images=two_baseline_images,
                labels=bad_labels,
                nw=10,
                show_images=False,
            )


class TestPatchwiseIlluminationCorrectionSaveLoad:
    """Test persistence (save/load) without persisting labels."""

    @pytest.fixture
    def correction_with_data(self, two_label_image, two_baseline_images):
        """Return a calibrated correction for testing save/load."""
        return darsia.PatchwiseIlluminationCorrection(
            baseline_images=two_baseline_images,
            labels=two_label_image,
            nw=10,
            show_images=False,
        )

    def test_save_load_preserves_correction_arrays(self, correction_with_data):
        """Save/load round-trip should preserve r_diff, g_diff, b_diff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "correction.npz"
            correction_with_data.save(save_path)

            loaded = darsia.PatchwiseIlluminationCorrection()
            loaded.load(save_path)

            np.testing.assert_array_equal(loaded.r_diff, correction_with_data.r_diff)
            np.testing.assert_array_equal(loaded.g_diff, correction_with_data.g_diff)
            np.testing.assert_array_equal(loaded.b_diff, correction_with_data.b_diff)

    def test_save_does_not_persist_limit(self, correction_with_data):
        """Saved npz should not contain 'limit' key (replaced by labels)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "correction.npz"
            correction_with_data.save(save_path)

            data = np.load(save_path, allow_pickle=True)["correction"].item()
            assert "limit" not in data, "limit should not be persisted"

    def test_load_does_not_require_limit(self, correction_with_data):
        """Load should work without 'limit' in the saved data (handled gracefully)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "correction.npz"
            correction_with_data.save(save_path)

            loaded = darsia.PatchwiseIlluminationCorrection()
            loaded.load(save_path)

            # If load completes without error and diffs match, test passes
            assert loaded.r_diff is not None
            np.testing.assert_array_equal(loaded.r_diff, correction_with_data.r_diff)

    @pytest.fixture
    def two_label_image(self):
        """Create a simple 2-label synthetic image (left half label 0, right half label 1)."""
        labels = np.zeros((100, 100), dtype=int)
        labels[:, 50:] = 1
        return darsia.Image(img=labels)

    @pytest.fixture
    def two_baseline_images(self, two_label_image):
        """Create two baseline images with different colors per label."""
        h, w = two_label_image.img.shape[:2]
        baseline1 = np.zeros((h, w, 3), dtype=np.uint8)
        baseline1[two_label_image.img == 0] = [100, 100, 100]  # Label 0: gray
        baseline1[two_label_image.img == 1] = [50, 50, 50]     # Label 1: darker gray

        baseline2 = np.zeros((h, w, 3), dtype=np.uint8)
        baseline2[two_label_image.img == 0] = [150, 150, 150]  # Label 0: lighter gray
        baseline2[two_label_image.img == 1] = [80, 80, 80]     # Label 1: darker gray

        return [darsia.Image(img=baseline1), darsia.Image(img=baseline2)]
