"""Unit tests for CorrectionsConfig and sub-config validation.

Focuses specifically on the ResizeCorrectionConfig mode/scale/target_shape refactor
and TOML round-trip loading. GUI schema metadata tests are in test_metadata_keywords.py.
"""

from pathlib import Path

import pytest

from darsia.presets.workflows.config.corrections import (
    CorrectionsConfig,
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
        assert isinstance(cfg.target_shape, tuple), "target_shape list should convert to tuple"
        assert cfg.target_shape == (512, 512)
        assert cfg.scale is None

    def test_mode_target_shape_without_shape_raises(self):
        """mode='target_shape' without target_shape set should raise ValueError."""
        with pytest.raises(
            ValueError, match="mode='target_shape' requires 'target_shape' to be set"
        ):
            ResizeCorrectionConfig().load({"mode": "target_shape"})

    @pytest.mark.parametrize(
        "bad_shape",
        [[512], [512, 512, 512]],
        ids=["1-element", "3-element"],
    )
    def test_mode_target_shape_with_wrong_length_raises(self, bad_shape):
        """mode='target_shape' with non-2-element shape should raise ValueError."""
        with pytest.raises(
            ValueError, match="target_shape must have exactly 2 elements"
        ):
            ResizeCorrectionConfig().load(
                {"mode": "target_shape", "target_shape": bad_shape}
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


class TestCurvatureCorrectionConfigLoad:
    """Test CurvatureCorrectionConfig stage-wise loading and to_dict() conversion."""

    def test_crop_only_config_valid(self, tmp_path):
        """crop-only config (real-world usage) should load and to_dict() correctly."""
        cfg_path = _write_toml(
            tmp_path,
            '[corrections.curvature.crop]\nwidth = 2.8\nheight = 1.5\n"in meters" = true\n',
        )
        cfg = CorrectionsConfig().load(cfg_path)
        assert cfg.curvature is not None
        assert cfg.curvature.crop is not None
        assert cfg.curvature.init is None
        assert cfg.curvature.bulge is None
        assert cfg.curvature.stretch is None

        # to_dict() should contain only crop
        d = cfg.curvature.to_dict()
        assert "crop" in d
        assert d["crop"]["width"] == 2.8
        assert d["crop"]["height"] == 1.5
        assert d["crop"]["in meters"] is True
        assert "init" not in d
        assert "bulge" not in d
        assert "stretch" not in d

    def test_all_stages_active(self, tmp_path):
        """Config with all 4 stages + explicit active list."""
        cfg_path = _write_toml(
            tmp_path,
            (
                "[corrections.curvature]\n"
                "active = [\"init\", \"crop\", \"bulge\", \"stretch\"]\n"
                "[corrections.curvature.init]\nhorizontal_bulge = 0.1\n"
                "[corrections.curvature.crop]\nwidth = 2.8\nheight = 1.5\n"
                "[corrections.curvature.bulge]\nhorizontal_bulge = 0.2\n"
                "[corrections.curvature.stretch]\nhorizontal_stretch = 0.05\n"
            ),
        )
        cfg = CorrectionsConfig().load(cfg_path)
        assert cfg.curvature.init is not None
        assert cfg.curvature.crop is not None
        assert cfg.curvature.bulge is not None
        assert cfg.curvature.stretch is not None
        assert len(cfg.curvature.inactive) == 0

        d = cfg.curvature.to_dict()
        assert set(d.keys()) == {"init", "crop", "bulge", "stretch"}

    def test_stages_deactivated_preserved_in_inactive(self, tmp_path):
        """Stages not in active list should be parsed and preserved in .inactive."""
        cfg_path = _write_toml(
            tmp_path,
            (
                "[corrections.curvature]\n"
                "active = [\"crop\"]\n"
                "[corrections.curvature.init]\nhorizontal_bulge = 0.1\n"
                "[corrections.curvature.crop]\nwidth = 2.8\n"
                "[corrections.curvature.bulge]\nhorizontal_bulge = 0.2\n"
            ),
        )
        cfg = CorrectionsConfig().load(cfg_path)
        assert cfg.curvature.crop is not None
        assert cfg.curvature.init is None
        assert cfg.curvature.bulge is None
        assert "init" in cfg.curvature.inactive
        assert "bulge" in cfg.curvature.inactive
        assert cfg.curvature.inactive["init"].horizontal_bulge == 0.1
        assert cfg.curvature.inactive["bulge"].horizontal_bulge == 0.2

        # to_dict() should only include crop (active stages)
        d = cfg.curvature.to_dict()
        assert set(d.keys()) == {"crop"}

    def test_crop_in_meters_translation(self, tmp_path):
        """CropCorrectionConfig.in_meters should translate to/from TOML 'in meters' key."""
        cfg_path = _write_toml(
            tmp_path,
            '[corrections.curvature.crop]\nwidth = 2.8\nheight = 1.5\n"in meters" = false\n',
        )
        cfg = CorrectionsConfig().load(cfg_path)
        assert cfg.curvature.crop.in_meters is False

        d = cfg.curvature.to_dict()
        assert d["crop"]["in meters"] is False  # Note: key has space in dict

    def test_partial_stages_active(self, tmp_path):
        """Config with multiple stages but only some active."""
        cfg_path = _write_toml(
            tmp_path,
            (
                "[corrections.curvature]\n"
                "active = [\"crop\", \"bulge\"]\n"
                "[corrections.curvature.crop]\nwidth = 2.8\n"
                "[corrections.curvature.bulge]\nhorizontal_bulge = 0.2\n"
                "[corrections.curvature.stretch]\nhorizontal_stretch = 0.05\n"
            ),
        )
        cfg = CorrectionsConfig().load(cfg_path)
        assert cfg.curvature.crop is not None
        assert cfg.curvature.bulge is not None
        assert cfg.curvature.init is None
        assert cfg.curvature.stretch is None
        assert "stretch" in cfg.curvature.inactive
        assert cfg.curvature.inactive["stretch"].horizontal_stretch == 0.05

        d = cfg.curvature.to_dict()
        assert set(d.keys()) == {"crop", "bulge"}

    def test_crop_with_corner_fields(self, tmp_path):
        """Crop config with individual corner fields should assemble into pts_src."""
        cfg_path = _write_toml(
            tmp_path,
            (
                "[corrections.curvature.crop]\n"
                "width = 2.8\n"
                "height = 1.5\n"
                "top_left = [47, 415]\n"
                "bottom_left = [7886, 448]\n"
                "bottom_right = [7829, 5228]\n"
                "top_right = [110, 5263]\n"
            ),
        )
        cfg = CorrectionsConfig().load(cfg_path)
        assert cfg.curvature.crop is not None
        assert cfg.curvature.crop.top_left == (47, 415)
        assert cfg.curvature.crop.bottom_left == (7886, 448)
        assert cfg.curvature.crop.bottom_right == (7829, 5228)
        assert cfg.curvature.crop.top_right == (110, 5263)
        # pts_src should be assembled from corners in order
        assert cfg.curvature.crop.pts_src == [
            [47, 415],
            [7886, 448],
            [7829, 5228],
            [110, 5263],
        ]

        # to_dict() should include pts_src
        d = cfg.curvature.to_dict()
        assert "crop" in d
        assert d["crop"]["pts_src"] == [
            [47, 415],
            [7886, 448],
            [7829, 5228],
            [110, 5263],
        ]
