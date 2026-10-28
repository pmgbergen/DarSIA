"""Real tiny saved-model preparation; no campaign or approval fixtures.

The default-path oracle is the pinned original N1 class already used by the
statistics-prefix tests. These tests remain UNRUN for this source increment.
"""
from copy import deepcopy

import pytest

from darsia.fluidflower.mass import n1_pipeline as pipeline
from darsia.fluidflower.mass.production_precision import PixelMath
from ..fixtures.tiny_saved_model_fixture import tiny_config, assert_same
from .test_two_phase_statistics_prefix import original_class


@pytest.fixture
def explicit_config(tiny_config):
    config = deepcopy(tiny_config[0])
    config["geometry_fields"] = dict(
        source_to_native="H_ac28_to_frame_native",
        original_domain_to_native="H_ac28_to_frame_native",
        carried_homography_provenance="original_H_ac28_to_native")
    return config


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_default_load_frame_retains_original_arrays_and_report(explicit_config, precision):
    old = original_class()(explicit_config, precision=precision, pixel_math=PixelMath(precision))
    actual = pipeline.N1Pipeline(explicit_config, precision=precision, pixel_math=PixelMath(precision))
    expected_frame, *expected = old.load_frame(explicit_config["frames"][0])
    actual_frame, *values = actual.load_frame(explicit_config["frames"][0])
    assert_same(vars(expected_frame), vars(actual_frame))
    assert_same(expected, values)
    assert actual._local_prepared is None


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_real_statistics_prefix_loads_one_frame_without_local_adapter(explicit_config, monkeypatch, precision):
    context = pipeline.N1Pipeline(explicit_config, precision=precision,
        pixel_math=PixelMath(precision), statistics_only=True)
    original_load = context.load_frame
    calls = []

    def observed_load(spec):
        calls.append(spec["case_id"])
        return original_load(spec)

    monkeypatch.setattr(context, "load_frame", observed_load)
    result = context.extract_statistics(explicit_config["frames"][0])
    assert calls == [explicit_config["frames"][0]["case_id"]]
    assert result["statistics"]["precision"] == precision
    assert context._local_prepared is None


def test_arbitrary_object_cannot_replace_local_build_inputs(explicit_config):
    with pytest.raises(TypeError, match="actual local Build input adapter"):
        pipeline.N1Pipeline(explicit_config, precision="float32",
            pixel_math=PixelMath("float32"), local_build=object())
