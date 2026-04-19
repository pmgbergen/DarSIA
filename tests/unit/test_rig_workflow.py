"""Unit tests for Rig correction workflow setup behavior."""

from dataclasses import dataclass

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.corrections import (
    ColorCorrectionConfig,
    CorrectionsConfig,
    IlluminationCorrectionConfig,
)
from darsia.presets.workflows.rig import Rig


@dataclass
class DummyImage:
    applied: list[str] | None = None

    def __post_init__(self) -> None:
        if self.applied is None:
            self.applied = []

    def copy(self):
        return DummyImage(self.applied.copy())


class DummyCorrection:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, image: DummyImage) -> DummyImage:
        image.applied.append(self.name)
        return image


def test_rig_corrections_property_concatenates_shape_then_color():
    rig = Rig()
    rig.shape_corrections = [DummyCorrection("shape_a"), DummyCorrection("shape_b")]
    rig.color_corrections = [DummyCorrection("color_a")]

    all_corrections = rig.corrections

    assert [c.name for c in all_corrections] == ["shape_a", "shape_b", "color_a"]


def test_setup_color_corrections_requires_shape_corrected_baseline():
    rig = Rig()
    config = CorrectionsConfig()

    with pytest.raises(RuntimeError, match="Shape-corrected baseline missing"):
        rig.setup_color_corrections(config)


def test_setup_color_corrections_illumination_return_is_assigned_and_appended(
    monkeypatch,
):
    rig = Rig()
    rig.shape_corrected_baseline = DummyImage()

    illumination = DummyCorrection("illumination")
    color = DummyCorrection("color")
    monkeypatch.setattr(
        rig,
        "setup_illumination_correction",
        lambda *_args, **_kwargs: illumination,
    )

    from darsia.presets.workflows import rig as rig_module

    monkeypatch.setattr(
        rig_module.darsia,
        "find_colorchecker",
        lambda *_args, **_kwargs: (None, "dummy-roi"),
    )
    monkeypatch.setattr(
        rig_module.darsia,
        "ColorCorrection",
        lambda *_args, **_kwargs: color,
    )

    config = CorrectionsConfig(
        illumination=IlluminationCorrectionConfig(),
        color=ColorCorrectionConfig(colorchecker="upper_left"),
    )
    rig.setup_color_corrections(config)

    assert rig.illumination_correction is illumination
    assert rig.color_corrections == [illumination, color]
    assert rig.baseline.applied == ["illumination", "color"]


def test_setup_color_corrections_warns_and_ignores_relative_color_config(
    monkeypatch,
):
    rig = Rig()
    rig.shape_corrected_baseline = DummyImage()

    monkeypatch.setattr(
        rig,
        "setup_illumination_correction",
        lambda *_args, **_kwargs: DummyCorrection("illumination"),
    )

    from darsia.presets.workflows import rig as rig_module

    monkeypatch.setattr(
        rig_module.darsia,
        "find_colorchecker",
        lambda *_args, **_kwargs: (None, "dummy-roi"),
    )
    monkeypatch.setattr(
        rig_module.darsia,
        "ColorCorrection",
        lambda *_args, **_kwargs: DummyCorrection("color"),
    )

    config = CorrectionsConfig(
        illumination=IlluminationCorrectionConfig(),
        relative_color=True,
        color=ColorCorrectionConfig(colorchecker="upper_left"),
    )

    with pytest.warns(UserWarning, match="relative_color requested"):
        rig.setup_color_corrections(config)

    assert [c.name for c in rig.color_corrections] == ["illumination", "color"]


def test_setup_illumination_correction_creates_uninitialized_correction_when_config_is_none(
    monkeypatch,
):
    rig = Rig()
    rig.shape_corrected_baseline = DummyImage()

    from darsia.presets.workflows import rig as rig_module

    class DummyIlluminationCorrection:
        def __init__(self):
            self.setup_called = False

        def setup(self, **_kwargs):
            self.setup_called = True

        def select_random_samples(self, **_kwargs):
            return []

    monkeypatch.setattr(
        rig_module.darsia,
        "IlluminationCorrection",
        DummyIlluminationCorrection,
    )

    correction = rig.setup_illumination_correction(None)

    assert isinstance(correction, DummyIlluminationCorrection)
    assert correction.setup_called is False


def test_import_from_csv_supports_matrix_layout(tmp_path):
    rig = Rig()
    rig.baseline = darsia.ScalarImage(
        np.zeros((2, 3), dtype=float),
        dimensions=[2.0, 3.0],
    )

    values = np.array(
        [
            [0.5, 0.5, 3.0],
            [1.5, 0.5, 3.0],
            [2.5, 0.5, 3.0],
            [0.5, 1.5, 6.0],
            [1.5, 1.5, 6.0],
            [2.5, 1.5, 6.0],
        ]
    )
    csv_path = tmp_path / "result.csv"
    np.savetxt(csv_path, values, delimiter=",")

    image = rig.import_from_csv(csv_path)

    assert isinstance(image, darsia.ScalarImage)
    assert np.allclose(image.img, np.array([[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]))


def test_import_from_csv_supports_coordinate_layout(tmp_path):
    rig = Rig()
    baseline = darsia.ScalarImage(
        np.zeros((2, 2), dtype=float),
        dimensions=[2.0, 2.0],
    )
    rig.baseline = baseline

    reference = darsia.ScalarImage(
        np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
        **baseline.metadata(),
    )
    csv_path = tmp_path / "coords.csv"
    reference.to_csv(csv_path, delimiter=",")

    image = rig.import_from_csv(csv_path)

    assert isinstance(image, darsia.ScalarImage)
    assert np.allclose(image.img, reference.img)
