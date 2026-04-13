"""Unit tests for Rig correction workflow setup behavior."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from darsia.presets.workflows.config.corrections import (
    ColorCorrectionConfig,
    CorrectionsConfig,
    IlluminationCorrectionConfig,
    RelativeColorCorrectionConfig,
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


def test_setup_color_corrections_warns_on_boolean_relative_color(
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

    with pytest.warns(UserWarning, match="relative_color=True requires"):
        rig.setup_color_corrections(config)

    assert [c.name for c in rig.color_corrections] == ["illumination", "color"]


def test_setup_color_corrections_loads_relative_color_from_path_and_preserves_order(
    monkeypatch,
):
    rig = Rig()
    rig.shape_corrected_baseline = DummyImage()
    rig.shape_corrections = [DummyCorrection("shape")]

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

    class DummyRelativeColorCorrection(DummyCorrection):
        def __init__(self, *_args, **_kwargs):
            super().__init__("relative")
            self.loaded = None

        def load(self, path):
            self.loaded = path

    monkeypatch.setattr(
        rig_module.darsia,
        "RelativeColorCorrection",
        DummyRelativeColorCorrection,
    )

    config = CorrectionsConfig(
        illumination=IlluminationCorrectionConfig(),
        relative_color=RelativeColorCorrectionConfig(path=Path("relative.npz")),
        color=ColorCorrectionConfig(colorchecker="upper_left"),
    )

    rig.setup_color_corrections(config)

    assert [c.name for c in rig.color_corrections] == [
        "illumination",
        "relative",
        "color",
    ]
    assert rig.baseline.applied == ["illumination", "relative", "color"]
    assert rig.relative_color_correction.loaded.name == "relative.npz"


def test_setup_color_corrections_raises_for_noninteractive_relative_color_images():
    rig = Rig()
    rig.shape_corrected_baseline = DummyImage()
    rig.shape_corrections = []

    config = CorrectionsConfig(
        relative_color=RelativeColorCorrectionConfig(
            images=[Path("calibration_a.jpg")], interactive=False
        )
    )

    with pytest.raises(ValueError, match="Interactive calibration is required"):
        rig.setup_color_corrections(config)


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
