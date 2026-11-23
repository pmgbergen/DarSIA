"""Reading images through the single entry `darsia.imread`: files and bytes, dates
from EXIF only without a given date, and scalar grids in CSV format."""

import importlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PIL_Image

import darsia

EXIF_DATE = datetime(2026, 1, 1, 10, 0, 0)
"""Date in the EXIF data of the test photo."""


def write_photo(path: Path, exif: bool = True, seed: int = 0) -> Path:
    """A small JPEG photo, with an EXIF date (tag 306) or without EXIF data."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(6, 8, 3), dtype=np.uint8)
    image = PIL_Image.fromarray(array)
    if exif:
        data = PIL_Image.Exif()
        data[306] = EXIF_DATE.strftime("%Y:%m:%d %H:%M:%S")
        image.save(path, exif=data)
    else:
        image.save(path)
    return path


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    return write_photo(tmp_path / "photo.jpg")


def test_bytes_are_read_like_the_file(photo: Path) -> None:
    from_file = darsia.imread(photo)
    from_bytes = darsia.imread(photo.read_bytes())

    for image in (from_file, from_bytes):
        assert isinstance(image, darsia.OpticalImage)
        assert image.color_space == "RGB"
        assert image.img.dtype == float
        assert image.date == EXIF_DATE
    assert np.array_equal(from_bytes.img, from_file.img)


def test_a_given_date_skips_the_exif_data(
    photo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("darsia.image.imread")

    def no_exif(source):
        raise AssertionError("EXIF data must not be read when a date is given")

    monkeypatch.setattr(module, "exif_date", no_exif)
    date = datetime(2026, 1, 2, 12, 0, 0)
    assert darsia.imread(photo, date=date).date == date
    assert darsia.imread(photo.read_bytes(), date=date).date == date
    assert darsia.imread(photo, date=None).date is None


def test_native_bytes_keep_the_integer_type(photo: Path) -> None:
    assert darsia.imread(photo.read_bytes(), native=True).img.dtype == np.uint8


def test_series_with_given_dates(tmp_path: Path) -> None:
    paths = [write_photo(tmp_path / f"photo_{i}.jpg", seed=i) for i in range(2)]
    dates = [datetime(2026, 1, 1, 10, 0, 0), datetime(2026, 1, 1, 10, 5, 0)]
    series = darsia.imread(paths, date=dates)
    assert series.series
    assert series.date == dates
    assert darsia.imread(paths).date == [EXIF_DATE, EXIF_DATE]


def test_exif_date_of_files_and_bytes(tmp_path: Path, photo: Path) -> None:
    without = write_photo(tmp_path / "plain.jpg", exif=False)
    assert darsia.exif_date(photo) == EXIF_DATE
    assert darsia.exif_date(photo.read_bytes()) == EXIF_DATE
    assert darsia.exif_date(without) is None
    assert darsia.exif_date(b"not an image") is None


def test_csv_grid_round_trip(tmp_path: Path) -> None:
    reference = darsia.ScalarImage(
        np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        dimensions=[2.0, 3.0],
        origin=[1.0, 2.0],
    )
    path = tmp_path / "grid.csv"
    reference.to_csv(path, header="x,y,value", float_format="{:.6e}")

    image = darsia.imread(path)
    assert type(image) is darsia.ScalarImage
    assert np.allclose(image.img, reference.img)
    assert np.allclose(image.origin, reference.origin)
    assert np.allclose(image.dimensions, reference.dimensions)

    extensive = darsia.imread(path, extensive=True, name="grid")
    assert isinstance(extensive, darsia.ExtensiveImage)
    assert extensive.name == "grid"


def test_csv_grid_rows_in_any_order(tmp_path: Path) -> None:
    # Rows not sorted, no header: cell centers (x, y) and values
    values = np.array(
        [
            [2.5, 1.5, 6.0],
            [0.5, 0.5, 3.0],
            [1.5, 1.5, 6.0],
            [2.5, 0.5, 3.0],
            [0.5, 1.5, 6.0],
            [1.5, 0.5, 3.0],
        ]
    )
    path = tmp_path / "result.csv"
    np.savetxt(path, values, delimiter=",")

    image = darsia.imread(path)
    # The rows of the image count from the top (largest y first)
    assert np.allclose(image.img, np.array([[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]))
    assert np.allclose(image.dimensions, [2.0, 3.0])
