"""Unit tests for ResizeConfig, the resize options of the corrections and the
restoration."""

import pytest

from darsia.presets.workflows.config.resize import ResizeConfig


def _load(sec: dict) -> ResizeConfig:
    return ResizeConfig().load(sec, context="somewhere")


def test_mode_scale_with_scale_value_valid():
    cfg = _load({"mode": "scale", "scale": 0.5})
    assert cfg.mode == "scale"
    assert cfg.scale == 0.5
    assert cfg.target_shape is None


def test_mode_scale_defaults_to_scale():
    cfg = _load({"scale": 0.75})
    assert cfg.mode == "scale"
    assert cfg.scale == 0.75


def test_mode_scale_without_scale_value_raises():
    with pytest.raises(ValueError, match="mode='scale' requires 'scale' to be set"):
        _load({"mode": "scale"})


def test_mode_target_shape_with_2_element_shape_valid():
    cfg = _load({"mode": "target_shape", "target_shape": [512, 512]})
    assert cfg.mode == "target_shape"
    assert isinstance(cfg.target_shape, tuple), "target_shape list becomes a tuple"
    assert cfg.target_shape == (512, 512)
    assert cfg.scale is None


def test_mode_target_shape_without_shape_raises():
    with pytest.raises(
        ValueError, match="mode='target_shape' requires 'target_shape' to be set"
    ):
        _load({"mode": "target_shape"})


@pytest.mark.parametrize(
    "bad_shape", [[512], [512, 512, 512]], ids=["1-element", "3-element"]
)
def test_mode_target_shape_with_wrong_length_raises(bad_shape):
    with pytest.raises(ValueError, match="target_shape must have exactly 2 elements"):
        _load({"mode": "target_shape", "target_shape": bad_shape})


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be 'scale' or 'target_shape'"):
        _load({"mode": "invalid"})


def test_errors_name_the_location_of_the_settings():
    with pytest.raises(ValueError, match=r"^\[somewhere\] "):
        _load({"mode": "scale"})
