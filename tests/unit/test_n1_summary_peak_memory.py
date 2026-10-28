"""Exact scientific oracle for releasing summary intermediates earlier."""
import ast
from dataclasses import replace
import hashlib
from pathlib import Path
import struct

import numpy as np
import pytest

from darsia.fluidflower.mass import n1_pipeline as pipeline
from darsia.fluidflower.mass import n1_statistics as statistics
from darsia.fluidflower.mass.production_precision import PixelMath
from ..fixtures.tiny_saved_model_fixture import tiny_config


def original_summary():
    path = Path(__file__).resolve().parents[1] / "fixtures/n1_original_summary.py"
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == (
        "b655e6b252295a45106325f102db13995230085467427d96535220aadf05c630"
    )
    tree = ast.parse(raw.decode("utf8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    context = dict(vars(statistics))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), "exec"), context)
    return context[node.name]


def assert_bits(expected, actual):
    if isinstance(expected, np.ndarray):
        assert expected.shape == actual.shape and expected.dtype == actual.dtype
        assert expected.tobytes() == actual.tobytes()
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_bits(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_bits(left, right)
    elif isinstance(expected, float):
        assert struct.pack("=d", expected) == struct.pack("=d", actual)
    else:
        assert expected == actual


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("frame_index", [0, 1])
@pytest.mark.parametrize("unknown_geometry", [False, True])
def test_complete_summary_products_match_frozen_original(
    tiny_config, monkeypatch, precision, frame_index, unknown_geometry,
):
    config, _ = tiny_config
    config["geometry_fields"] = dict(
        source_to_native="H_ac28_to_frame_native",
        original_domain_to_native="H_ac28_to_frame_native",
        carried_homography_provenance="original_H_ac28_to_native",
    )
    candidate = statistics.summarize_round3_response

    def evaluate(function):
        def observed(response, geometry, **kwargs):
            if unknown_geometry:
                geometry = replace(geometry, valid=np.zeros_like(geometry.valid))
            shape = response["frame"].domain.shape
            kwargs["distances_m"] = {
                "scalar": .0125,
                "grid": np.linspace(.001, .03, np.prod(shape)).reshape(shape),
            }
            # The reused buffer must belong to the summary, never an input.
            inputs = [geometry.depth_m, geometry.porosity, geometry.swi,
                      kwargs["own_colour"]["aqueous_relative"],
                      response["physics"].solubility_co2,
                      response["physics"].density_gaseous_co2]
            before = [np.asarray(value).copy() for value in inputs]
            result = function(response, geometry, **kwargs)
            for old, value in zip(before, inputs):
                assert_bits(old, np.asarray(value))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(statistics, "summarize_round3_response", observed)
            context = pipeline.N1Pipeline(
                config, precision=precision, pixel_math=PixelMath(precision),
            )
            return context.evaluate(config["frames"][frame_index])

    expected = evaluate(original_summary())
    actual = evaluate(candidate)
    for key in ("statistics", "prediction", "result", "export", "sandmap"):
        assert_bits(expected[key], actual[key])
