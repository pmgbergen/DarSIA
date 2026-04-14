from pathlib import Path

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.calibration.calibration_color_paths import (
    _load_or_compute_tracer_color_spectrum,
    _resolve_target_labels,
)


def _make_color_spectrum(value: float = 1.0) -> darsia.ColorSpectrum:
    resolution = 5
    shape = (resolution, resolution, resolution)
    spectrum = np.zeros(shape, dtype=bool)
    histogram = np.zeros(shape, dtype=float)
    spectrum[2, 2, 2] = True
    histogram[2, 2, 2] = value
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


def test_resolve_target_labels_validates_presence_and_ignore():
    labels_img = darsia.Image(
        img=np.array([[0, 1], [2, 2]], dtype=int), dimensions=[1.0, 1.0]
    )
    assert _resolve_target_labels(labels_img, [1, 2], []) == [1, 2]
    with pytest.raises(ValueError, match="target_labels"):
        _resolve_target_labels(labels_img, [3], [])
    with pytest.raises(ValueError, match="ignored labels"):
        _resolve_target_labels(labels_img, [2], [2])


def test_load_or_compute_tracer_spectrum_prefers_stored(tmp_path: Path):
    folder = tmp_path / "tracer"
    stored = darsia.LabelColorSpectrumMap({0: _make_color_spectrum(2.0)})
    stored.save(folder)

    class DummyRegression:
        def get_color_spectrum(self, **kwargs):
            raise AssertionError("Fallback recomputation should not be used")

    result = _load_or_compute_tracer_color_spectrum(
        color_path_regression=DummyRegression(),
        calibration_images=[],
        baseline=darsia.Image(img=np.zeros((2, 2, 3)), dimensions=[1.0, 1.0]),
        ignore_spectrum=None,
        threshold_calibration=0.0,
        tracer_color_spectrum_folder=folder,
        strict_stored_artifacts=True,
        verbose=False,
    )
    assert 0 in result
    assert float(result[0].histogram.sum()) == pytest.approx(2.0)


def test_load_or_compute_tracer_spectrum_fallback_non_strict(tmp_path: Path):
    folder = tmp_path / "missing"
    expected = darsia.LabelColorSpectrumMap({0: _make_color_spectrum(3.0)})

    class DummyRegression:
        def __init__(self):
            self.calls = 0

        def get_color_spectrum(self, **kwargs):
            self.calls += 1
            return expected

    regression = DummyRegression()
    result = _load_or_compute_tracer_color_spectrum(
        color_path_regression=regression,
        calibration_images=[],
        baseline=darsia.Image(img=np.zeros((2, 2, 3)), dimensions=[1.0, 1.0]),
        ignore_spectrum=None,
        threshold_calibration=0.0,
        tracer_color_spectrum_folder=folder,
        strict_stored_artifacts=False,
        verbose=False,
    )
    assert regression.calls == 1
    assert float(result[0].histogram.sum()) == pytest.approx(3.0)


def test_load_or_compute_tracer_spectrum_missing_strict_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="strict mode"):
        _load_or_compute_tracer_color_spectrum(
            color_path_regression=object(),  # not reached
            calibration_images=[],
            baseline=darsia.Image(img=np.zeros((2, 2, 3)), dimensions=[1.0, 1.0]),
            ignore_spectrum=None,
            threshold_calibration=0.0,
            tracer_color_spectrum_folder=tmp_path / "missing",
            strict_stored_artifacts=True,
            verbose=False,
        )
