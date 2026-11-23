"""Reading the fields of the rig (labels, facies, depth, porosity) onto the grid of
its baseline."""

from pathlib import Path

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.rig import Rig

DIMENSIONS = [2.0, 3.0]
"""Physical size of the baseline (rows, columns)."""


def rig() -> Rig:
    """A rig with a baseline of 2 x 3 pixels."""
    rig = Rig()
    rig.baseline = darsia.ScalarImage(np.zeros((2, 3)), dimensions=DIMENSIONS)
    return rig


def save(tmp_path: Path, name: str, array: np.ndarray, **metadata) -> Path:
    """Save a field as npz (default: the domain of the baseline)."""
    image = darsia.ScalarImage(array, **({"dimensions": DIMENSIONS} | metadata))
    path = tmp_path / f"{name}.npz"
    image.save(path)
    return path


def test_field_on_the_grid_of_the_baseline_is_kept(tmp_path: Path) -> None:
    labels = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)
    field = rig().read_field(save(tmp_path, "labels", labels))
    assert field.img.dtype == np.uint8
    assert np.array_equal(field.img, labels)


def test_finer_integer_field_is_sampled_at_cell_centers(tmp_path: Path) -> None:
    fine = np.arange(24, dtype=np.int64).reshape(4, 6)
    field = rig().read_field(save(tmp_path, "labels", fine))
    assert field.img.dtype == np.int64
    # Each pixel of the baseline covers 2 x 2 fine pixels; its center lies on a pixel
    # boundary, where the later pixel is taken.
    assert np.array_equal(field.img, fine[1::2, 1::2])


def test_finer_float_field_is_area_averaged(tmp_path: Path) -> None:
    fine = np.arange(24, dtype=float).reshape(4, 6)
    field = rig().read_field(save(tmp_path, "depth", fine))
    expected = fine.reshape(2, 2, 3, 2).mean(axis=(1, 3))
    assert np.allclose(field.img, expected)


def test_field_of_another_domain_is_refused(tmp_path: Path) -> None:
    path = save(tmp_path, "depth", np.ones((2, 3)), dimensions=[1.0, 1.0])
    with pytest.raises(ValueError, match="domain"):
        rig().read_field(path)


def test_field_drawn_on_the_photo_gets_the_field_corrections(tmp_path: Path) -> None:
    calls: list[str] = []

    def correction(name: str):
        def apply(image: darsia.Image) -> darsia.Image:
            calls.append(name)
            return image

        return apply

    reader = rig()
    reader.resize_correction_inter_nearest = correction("resize")
    reader.curvature_correction = correction("curvature")
    reader.resize_correction_inter_nearest_final = correction("final resize")
    labels = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)

    field = reader.read_field(save(tmp_path, "labels", labels), on_photo=True)
    assert calls == ["resize", "curvature", "final resize"]
    assert np.array_equal(field.img, labels)


@pytest.mark.parametrize("apply_corrections", [False, True])
def test_setup_reads_labels_facies_and_depth_as_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apply_corrections: bool
) -> None:
    reader = rig()
    calls: list[tuple[str, bool]] = []

    def read_field(path: Path, *, on_photo: bool = False) -> darsia.Image:
        calls.append((path.stem, on_photo))
        return reader.baseline.copy()

    monkeypatch.setattr(reader, "read_field", read_field)
    paths = {
        name: save(tmp_path, name, np.zeros((2, 3)))
        for name in ("labels", "facies", "depth")
    }
    reader.setup_labels(path=paths["labels"], apply_corrections=apply_corrections)
    reader.setup_facies(path=paths["facies"], apply_corrections=apply_corrections)
    reader.setup_depth(path=paths["depth"])

    assert calls == [
        ("labels", apply_corrections),
        ("facies", apply_corrections),
        ("depth", False),
    ]
