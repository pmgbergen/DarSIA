"""Reading images with the rig: files and bytes, with and without an experiment."""

import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import darsia
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)
from darsia.presets.workflows.rig import Rig

START = datetime(2026, 1, 1, 12, 0, 0)
"""Start of the experiment."""

PROTOCOL_DATE = datetime(2026, 1, 1, 12, 10, 0)
"""Date of the image in the imaging protocol."""


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """A small colored PNG image (lossless, so files and bytes decode equally)."""
    rng = np.random.default_rng(0)
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), rng.integers(0, 255, size=(6, 8, 3), dtype=np.uint8))
    return path


def experiment() -> SimpleNamespace:
    """An experiment whose imaging protocol knows every file."""
    return SimpleNamespace(
        get_datetime=lambda path: PROTOCOL_DATE,
        experiment_start=START,
        injection_protocol=None,
        pressure_temperature_protocol=None,
    )


def rig(with_experiment: bool) -> Rig:
    """A rig without corrections, optionally with an experiment."""
    rig = Rig()
    if with_experiment:
        rig.load_experiment(experiment())
    return rig


def test_file_with_experiment_has_the_date_of_the_protocol(image_file: Path) -> None:
    image = rig(with_experiment=True).read(image_file)
    assert image.date == PROTOCOL_DATE
    assert image.time == (PROTOCOL_DATE - START).total_seconds()
    assert image.name == image_file.name


def test_file_without_experiment_has_no_date(image_file: Path) -> None:
    image = rig(with_experiment=False).read(image_file)
    assert image.date is None
    assert image.time is None


BASELINE_DATE = datetime(2026, 1, 1, 11, 0, 0)
"""Date of the baseline photograph, before the experiment."""


def test_without_experiment_time_counts_from_the_baseline(image_file: Path) -> None:
    reader = rig(with_experiment=False)
    reader.baseline = darsia.OpticalImage(
        np.zeros((6, 8, 3)), date=BASELINE_DATE, color_space="RGB"
    )
    date = datetime(2026, 1, 1, 13, 0, 0)
    for source in (image_file, image_file.read_bytes()):
        image = reader.read(source, date=date)
        assert image.date == date
        assert image.time == (date - BASELINE_DATE).total_seconds()


def test_without_experiment_and_dated_baseline_a_date_gives_no_time(
    image_file: Path,
) -> None:
    date = datetime(2026, 1, 1, 13, 0, 0)
    image = rig(with_experiment=False).read(image_file, date=date)
    assert image.date == date
    assert image.time is None


def test_given_date_comes_before_the_protocol(image_file: Path) -> None:
    date = datetime(2026, 1, 1, 13, 0, 0)
    image = rig(with_experiment=True).read(image_file, date=date)
    assert image.date == date
    assert image.time == (date - START).total_seconds()


@pytest.mark.parametrize("with_experiment", [False, True])
def test_bytes_are_read_like_the_file(image_file: Path, with_experiment: bool) -> None:
    reader = rig(with_experiment)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from_bytes = reader.read(image_file.read_bytes())
    from_file = reader.read(image_file)

    assert isinstance(from_bytes, darsia.OpticalImage)
    assert from_bytes.color_space == "RGB"
    assert not any("color space" in str(warning.message) for warning in caught)
    assert from_bytes.img.dtype == from_file.img.dtype == float
    assert np.array_equal(from_bytes.img, from_file.img)
    # Bytes have no path in the imaging protocol: a date only when given
    assert from_bytes.date is None


def test_bytes_with_a_given_date(image_file: Path) -> None:
    date = datetime(2026, 1, 1, 13, 0, 0)
    image = rig(with_experiment=True).read(image_file.read_bytes(), date=date)
    assert image.time == (date - START).total_seconds()


@pytest.mark.parametrize("as_bytes", [False, True])
def test_native_keeps_the_type_of_the_corrections(
    image_file: Path, as_bytes: bool
) -> None:
    source = image_file.read_bytes() if as_bytes else image_file
    image = rig(with_experiment=False).read(source, native=True)
    assert image.img.dtype == np.uint8


def test_fluid_state_needs_the_date_of_the_image() -> None:
    update_state = HeterogeneousColorToMassAnalysis.update_state
    # Without a pressure and temperature protocol, nothing to do
    update_state(SimpleNamespace(pressure_temperature_protocol=None), None)
    with pytest.raises(ValueError, match="no date"):
        update_state(SimpleNamespace(pressure_temperature_protocol=object()), None)
