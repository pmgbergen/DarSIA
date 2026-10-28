"""The optional-stage master changes selection only, with legacy numerics intact."""

from copy import deepcopy

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.corrections import CurvatureCorrectionConfig, InitCorrectionConfig


def configuration():
    return {
        "active": ["init", "crop", "bulge", "stretch"],
        "init": {"horizontal_bulge": 0.0001},
        "crop": {"top_left": [1, 2], "bottom_left": [30, 2],
                 "bottom_right": [30, 40], "top_right": [1, 40],
                 "width": 0.9, "height": 0.49, "in_meters": True},
        "bulge": {"vertical_bulge": 0.00001},
        "stretch": {"horizontal_stretch": 0.00001},
    }


def test_missing_master_preserves_legacy_configuration_and_exact_pixels():
    original_argument = InitCorrectionConfig(horizontal_bulge=0.0001)
    assert CurvatureCorrectionConfig(original_argument).init is original_argument
    legacy = configuration()
    explicit = dict(legacy, optional_stages_active=True)
    parsed_legacy = CurvatureCorrectionConfig().load(legacy).to_dict()
    parsed_explicit = CurvatureCorrectionConfig().load(explicit).to_dict()
    assert parsed_legacy == parsed_explicit
    image = np.indices((40, 50)).sum(axis=0).astype(np.float32)
    for settings in (legacy, parsed_legacy):
        actual = darsia.CurvatureCorrection(config=settings).correct_array(image)
        enabled = darsia.CurvatureCorrection(config=dict(settings, optional_stages_active=True)).correct_array(image)
        assert np.array_equal(actual, enabled)


def test_off_filters_only_optional_stages_and_never_mutates_saved_values():
    settings = dict(configuration(), optional_stages_active=False)
    before = deepcopy(settings)
    parsed = CurvatureCorrectionConfig().load(settings)
    assert parsed.init is None and parsed.bulge is None and parsed.stretch is None
    assert set(parsed.inactive) == {"init", "bulge", "stretch"}
    assert parsed.crop is not None
    image = np.indices((40, 50)).sum(axis=0).astype(np.float32)
    expected = darsia.CurvatureCorrection(config={"crop": settings["crop"]}).correct_array(image)
    direct = darsia.CurvatureCorrection(config=settings).correct_array(image)
    ordinary_rig = darsia.CurvatureCorrection(config=parsed.to_dict()).correct_array(image)
    assert np.array_equal(expected, direct) and np.array_equal(expected, ordinary_rig)
    assert settings == before
    settings["active"].remove("crop")
    assert CurvatureCorrectionConfig().load(settings).crop is None


@pytest.mark.parametrize("value", ["false", 0, None])
def test_master_requires_the_actual_toml_boolean(value):
    settings = dict(configuration(), optional_stages_active=value)
    with pytest.raises(ValueError, match="optional_stages_active"):
        CurvatureCorrectionConfig().load(settings)
    with pytest.raises(ValueError, match="optional_stages_active"):
        darsia.CurvatureCorrection(config=settings)
