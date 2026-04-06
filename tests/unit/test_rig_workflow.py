"""Unit tests for Rig correction workflow setup behavior."""

from dataclasses import dataclass

import pytest

from darsia.presets.workflows.config.corrections import (
    ColorCorrectionConfig,
    CorrectionsConfig,
    IlluminationCorrectionConfig,
)
from darsia.presets.workflows.rig import Rig


@dataclass
class _DummyImage:
    applied: list[str] | None = None

    def __post_init__(self) -> None:
        if self.applied is None:
            self.applied = []

    def copy(self):
        return _DummyImage(self.applied.copy())


class _DummyCorrection:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, image: _DummyImage) -> _DummyImage:
        image.applied.append(self.name)
        return image


def test_rig_corrections_property_concatenates_shape_then_color():
    rig = Rig()
    rig.shape_corrections = [_DummyCorrection("shape_a"), _DummyCorrection("shape_b")]
    rig.color_corrections = [_DummyCorrection("color_a")]

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
    rig.shape_corrected_baseline = _DummyImage()

    illumination = _DummyCorrection("illumination")
    color = _DummyCorrection("color")
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


def test_setup_color_corrections_relative_color_stage_is_guarded_warning(
    monkeypatch,
):
    rig = Rig()
    rig.shape_corrected_baseline = _DummyImage()

    monkeypatch.setattr(
        rig,
        "setup_illumination_correction",
        lambda *_args, **_kwargs: _DummyCorrection("illumination"),
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
        lambda *_args, **_kwargs: _DummyCorrection("color"),
    )

    config = CorrectionsConfig(
        illumination=IlluminationCorrectionConfig(),
        relative_color=True,
        color=ColorCorrectionConfig(colorchecker="upper_left"),
    )

    with pytest.warns(UserWarning, match="relative_color requested"):
        rig.setup_color_corrections(config)

    assert [c.name for c in rig.color_corrections] == ["illumination", "color"]


def test_setup_illumination_correction_returns_new_correction_when_config_is_none(
    monkeypatch,
):
    rig = Rig()
    rig.shape_corrected_baseline = _DummyImage()

    from darsia.presets.workflows import rig as rig_module

    class _DummyIlluminationCorrection:
        def __init__(self):
            self.setup_called = False

        def setup(self, **_kwargs):
            self.setup_called = True

        def select_random_samples(self, **_kwargs):
            return []

    monkeypatch.setattr(
        rig_module.darsia,
        "IlluminationCorrection",
        _DummyIlluminationCorrection,
    )

    correction = rig.setup_illumination_correction(None)

    assert isinstance(correction, _DummyIlluminationCorrection)
    assert correction.setup_called is False
