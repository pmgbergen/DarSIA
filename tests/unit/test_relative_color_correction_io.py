from pathlib import Path

import numpy as np

import darsia


def test_relative_color_correction_save_includes_class_name(tmp_path: Path):
    correction = darsia.RelativeColorCorrection()
    correction.config = {"mode": "custom"}
    correction.evaluated_correction = np.tile(np.eye(3), (2, 2, 1, 1))

    path = tmp_path / "relative_color.npz"
    correction.save(path)

    payload = np.load(path, allow_pickle=True)
    assert payload["class_name"].item() == "RelativeColorCorrection"


def test_relative_color_correction_can_be_read_with_generic_reader(tmp_path: Path):
    correction = darsia.RelativeColorCorrection()
    correction.config = {"mode": "custom"}
    correction.evaluated_correction = np.tile(np.eye(3), (2, 2, 1, 1))

    path = tmp_path / "relative_color.npz"
    correction.save(path)

    loaded = darsia.read_correction(path)

    assert isinstance(loaded, darsia.RelativeColorCorrection)
    img = np.random.rand(2, 2, 3)
    corrected = loaded.correct_array(img)
    assert np.allclose(corrected, img)
