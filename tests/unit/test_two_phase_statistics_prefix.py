"""PRIVATE same-math statistics split: original 3x3 saved models only.

No AC52, camera/native processor, phase inference, job, queue or publication.
The expected default implementation is the complete original 929a class AST;
its arrays and scalars are computed independently from the same tiny fixture.
"""
from copy import deepcopy
from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import pytest

from darsia.fluidflower.mass import n1_pipeline as pipeline
from darsia.fluidflower.mass import n1_arithmetic as arithmetic
from darsia.fluidflower.mass import n1_statistics as statistics
from darsia.fluidflower.mass.production_precision import PixelMath
from ..fixtures.tiny_saved_model_fixture import tiny_config, assert_same


ORIGINAL = Path(__file__).resolve().parents[1] / "fixtures/n1_original_pipeline.py"
ORIGINAL_SHA = "6048c001cc2a287d49f0149dffeba3d5340ad8be4a774bf7f06343898eb369ef"


@pytest.fixture(autouse=True)
def explicit_geometry_field_roles(tiny_config):
    """Bind the existing tiny input keys to the current explicit reader contract.

    The original fixture predates mandatory geometry_fields. Both original and
    candidate classes consume this same in-memory DATA; no bound JSON is read
    or rewritten by these direct-API tests.
    """
    config, _ = tiny_config
    assert "geometry_fields" not in config
    config["geometry_fields"] = dict(
        source_to_native="H_ac28_to_frame_native",
        original_domain_to_native="H_ac28_to_frame_native",
        carried_homography_provenance="original_H_ac28_to_native",
    )


def original_class():
    raw = ORIGINAL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA
    tree = ast.parse(raw.decode("utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "N1Pipeline")
    namespace = dict(vars(pipeline))
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ORIGINAL), "exec"), namespace)
    return namespace["N1Pipeline"]


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("keep_descriptor", [False, True])
def test_statistics_context_never_reads_fitted_calibration(tiny_config, tmp_path, precision, keep_descriptor):
    config, _ = tiny_config
    config = deepcopy(config)
    missing = tmp_path / "fitted_calibration_must_not_be_opened.json"
    assert not missing.exists()
    if keep_descriptor:
        config["calibration"] = dict(path=str(missing), sha256="f" * 64, size_bytes=123)
    else:
        config.pop("calibration")
    context = pipeline.N1Pipeline(config, precision=precision, pixel_math=PixelMath(precision),
                                  statistics_only=True)
    assert not hasattr(context, "calibration") and not hasattr(context, "parameters")
    result = context.extract_statistics(config["frames"][0])
    assert set(result) == {"statistics", "sandmap"}
    assert result["statistics"]["precision"] == precision
    assert result["statistics"]["elapsed_h"] == config["frames"][0]["elapsed_h"]
    assert result["statistics"]["physical_correction"] == config["physical_correction"]
    assert config["summary"]["baseline_gain"] == 1.1
    assert not missing.exists()


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_default_evaluate_exactly_matches_original_class(tiny_config, precision):
    config, _ = tiny_config
    from darsia.fluidflower.mass.mass_opencl import FORMAT
    config["opencl_acceleration"] = dict(format=FORMAT, mode="cpu", device_id=None)
    old_math = PixelMath(precision)
    old_math.configure_opencl(config["opencl_acceleration"])
    old = original_class()(config, precision=precision, pixel_math=old_math)
    new = pipeline.N1Pipeline(config, precision=precision, pixel_math=PixelMath(precision))
    original_capture, actual_capture = {}, {}
    expected = old.evaluate(config["frames"][0], output_capture=original_capture)
    actual = new.evaluate(config["frames"][0], output_capture=actual_capture)
    assert expected.keys() == actual.keys()
    for key in ("result", "prediction", "export", "statistics", "sandmap"):
        assert_same(expected[key], actual[key])
    for key in ("native_signal", "quantifiable", "metadata", "swi", "precision"):
        assert_same(expected["response"][key], actual["response"][key])
    assert_same(vars(expected["response"]["prepared"]), vars(actual["response"]["prepared"]))
    assert_same(vars(expected["response"]["frame"]), vars(actual["response"]["frame"]))
    assert original_capture.keys() == actual_capture.keys()
    for key in ("x_m", "y_m", "volume_m3"):
        assert_same(original_capture[key], actual_capture[key])
    assert_same(vars(original_capture["frame"]), vars(actual_capture["frame"]))


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_extract_statistics_calls_summary_once_and_no_fitted_mass(tiny_config, monkeypatch, precision):
    config, _ = tiny_config
    context = pipeline.N1Pipeline(config, precision=precision, pixel_math=PixelMath(precision),
                                  statistics_only=True)
    calls = []
    original_summary = statistics.summarize_round3_response
    def summary(*args, **kwargs):
        calls.append(1)
        return original_summary(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("fitted mass was called by statistics extraction")
    monkeypatch.setattr(statistics, "summarize_round3_response", summary)
    monkeypatch.setattr(statistics, "exact_prediction", forbidden)
    monkeypatch.setattr(arithmetic, "evaluate_main", forbidden)
    monkeypatch.setattr(arithmetic, "main_products", forbidden)
    result = context.extract_statistics(config["frames"][0])
    assert calls == [1]
    assert set(result) == {"statistics", "sandmap"}


@pytest.mark.parametrize("missing", ["round4_spec_sha256", "amendment_sha256"])
def test_statistics_never_infers_method_binding_from_calibration(tiny_config, missing):
    config, _ = tiny_config
    config = deepcopy(config)
    metadata = dict(parameters=dict(candidate="N1", gain=1.2, beta=.3), **{missing: "f" * 64})
    calibration = Path(config["calibration"]["path"])
    calibration.write_text(json.dumps(metadata), encoding="utf-8")
    config["calibration"] = pipeline.asset(calibration)
    config["bindings"].pop(missing)
    original = original_class()(config, precision="float32", pixel_math=PixelMath("float32"))
    assert original.method_bindings[missing] == "f" * 64
    with pytest.raises(ValueError, match="Statistics require explicit method binding: " + missing):
        pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"),
                            statistics_only=True)


@pytest.mark.parametrize("mode", [1, "true", None])
def test_statistics_mode_must_be_an_explicit_bool(tiny_config, mode):
    config, _ = tiny_config
    with pytest.raises(TypeError, match="Statistics-only mode must be explicit"):
        pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"),
                            statistics_only=mode)


def test_statistics_context_cannot_evaluate_fitted_mass(tiny_config, monkeypatch):
    config, _ = tiny_config
    context = pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"),
                                  statistics_only=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("load_frame was called before the statistics-only refusal")
    monkeypatch.setattr(context, "load_frame", forbidden)
    with pytest.raises(ValueError, match="A statistics-only context cannot evaluate fitted mass"):
        context.evaluate(config["frames"][0])
