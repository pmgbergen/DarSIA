"""Unit tests for CorrectionsConfig and sub-config validation."""

from pathlib import Path

import pytest

from darsia.presets.workflows.config.corrections import (
    ColorCorrectionConfig,
    CorrectionsConfig,
    DriftCorrectionConfig,
    ResizeCorrectionConfig,
)


def _write_toml(tmp_path: Path, content: str) -> Path:
    """Helper: write TOML content to a temp config file."""
    p = tmp_path / "config.toml"
    p.write_text(content)
    return p


class TestResizeCorrectionConfigLoad:
    """Test ResizeCorrectionConfig validation in load()."""

    def test_mode_scale_with_scale_value_valid(self):
        """mode='scale' with scale set should load successfully."""
        cfg = ResizeCorrectionConfig().load({"mode": "scale", "scale": 0.5})
        assert cfg.mode == "scale"
        assert cfg.scale == 0.5
        assert cfg.target_shape is None

    def test_mode_scale_defaults_to_scale(self):
        """Omitting mode should default to 'scale'."""
        cfg = ResizeCorrectionConfig().load({"scale": 0.75})
        assert cfg.mode == "scale"
        assert cfg.scale == 0.75

    def test_mode_scale_without_scale_value_raises(self):
        """mode='scale' without scale set should raise ValueError."""
        with pytest.raises(ValueError, match="mode='scale' requires 'scale' to be set"):
            ResizeCorrectionConfig().load({"mode": "scale"})

    def test_mode_target_shape_with_2_element_shape_valid(self):
        """mode='target_shape' with valid 2-element shape should load successfully."""
        cfg = ResizeCorrectionConfig().load(
            {"mode": "target_shape", "target_shape": [512, 512]}
        )
        assert cfg.mode == "target_shape"
        assert cfg.target_shape == (512, 512)
        assert cfg.scale is None

    def test_mode_target_shape_converts_list_to_tuple(self):
        """target_shape list should be converted to tuple of ints."""
        cfg = ResizeCorrectionConfig().load(
            {"mode": "target_shape", "target_shape": [256, 384]}
        )
        assert isinstance(cfg.target_shape, tuple)
        assert cfg.target_shape == (256, 384)

    def test_mode_target_shape_without_shape_raises(self):
        """mode='target_shape' without target_shape set should raise ValueError."""
        with pytest.raises(
            ValueError, match="mode='target_shape' requires 'target_shape' to be set"
        ):
            ResizeCorrectionConfig().load({"mode": "target_shape"})

    def test_mode_target_shape_with_wrong_length_raises(self):
        """mode='target_shape' with non-2-element shape should raise ValueError."""
        with pytest.raises(
            ValueError, match="target_shape must have exactly 2 elements"
        ):
            ResizeCorrectionConfig().load(
                {"mode": "target_shape", "target_shape": [512]}
            )

    def test_mode_target_shape_with_3_element_shape_raises(self):
        """mode='target_shape' with 3-element shape should raise ValueError."""
        with pytest.raises(
            ValueError, match="target_shape must have exactly 2 elements"
        ):
            ResizeCorrectionConfig().load(
                {"mode": "target_shape", "target_shape": [512, 512, 512]}
            )

    def test_invalid_mode_raises(self):
        """Invalid mode value should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be 'scale' or 'target_shape'"):
            ResizeCorrectionConfig().load({"mode": "invalid"})

    def test_toml_round_trip_scale_mode(self, tmp_path):
        """Full TOML load/save round-trip for scale mode."""
        cfg_path = _write_toml(
            tmp_path, '[corrections.resize]\nmode = "scale"\nscale = 0.5\n'
        )
        cfg = CorrectionsConfig().load(cfg_path)
        assert cfg.resize is not None
        assert cfg.resize.mode == "scale"
        assert cfg.resize.scale == 0.5

    def test_toml_round_trip_target_shape_mode(self, tmp_path):
        """Full TOML load/save round-trip for target_shape mode."""
        cfg_path = _write_toml(
            tmp_path,
            '[corrections.resize]\nmode = "target_shape"\ntarget_shape = [256, 384]\n',
        )
        cfg = CorrectionsConfig().load(cfg_path)
        assert cfg.resize is not None
        assert cfg.resize.mode == "target_shape"
        assert cfg.resize.target_shape == (256, 384)

    def test_toml_backward_compat_scale_only_defaults_to_scale_mode(self, tmp_path):
        """Old TOML with only [corrections.resize] scale = ... should default mode to 'scale'."""
        cfg_path = _write_toml(tmp_path, "[corrections.resize]\nscale = 0.5\n")
        cfg = CorrectionsConfig().load(cfg_path)
        assert cfg.resize is not None
        assert cfg.resize.mode == "scale"
        assert cfg.resize.scale == 0.5

    def test_toml_backward_compat_target_shape_only_without_mode_raises(self, tmp_path):
        """Old TOML with only [corrections.resize] target_shape = ... without mode should raise."""
        cfg_path = _write_toml(
            tmp_path, "[corrections.resize]\ntarget_shape = [512, 512]\n"
        )
        # This should fail because mode defaults to 'scale' which requires scale to be set
        with pytest.raises(ValueError, match="mode='scale' requires 'scale' to be set"):
            CorrectionsConfig().load(cfg_path)


class TestDriftCorrectionConfigLoad:
    """Regression: DriftCorrectionConfig should still validate correctly."""

    def test_drift_colorchecker_valid_values(self):
        """Drift colorchecker should accept all valid positions."""
        for pos in ["upper_left", "upper_right", "lower_left", "lower_right"]:
            cfg = DriftCorrectionConfig().load({"colorchecker": pos})
            assert cfg.colorchecker == pos

    def test_drift_colorchecker_invalid_raises(self):
        """Drift colorchecker with invalid position should raise."""
        with pytest.raises(AssertionError):
            DriftCorrectionConfig().load({"colorchecker": "center"})


class TestColorCorrectionConfigLoad:
    """Regression: ColorCorrectionConfig should still validate correctly."""

    def test_color_colorchecker_valid_values(self):
        """Color colorchecker should accept all valid positions."""
        for pos in ["upper_left", "upper_right", "lower_left", "lower_right"]:
            cfg = ColorCorrectionConfig().load({"colorchecker": pos})
            assert cfg.colorchecker == pos

    def test_color_colorchecker_invalid_raises(self):
        """Color colorchecker with invalid position should raise."""
        with pytest.raises(AssertionError):
            ColorCorrectionConfig().load({"colorchecker": "middle"})
