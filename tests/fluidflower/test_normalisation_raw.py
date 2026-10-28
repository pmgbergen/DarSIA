"""Direct sampling retains the saved producer's area reduction and NaNs."""

import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from darsia.fluidflower.native import normalisation_raw as raw
from darsia.fluidflower.native import registration
from tests.fluidflower.test_native_registration import parameters


@pytest.fixture
def area_parameters():
    return dict(homography_min_determinant=1e-9, maximum_rows=512, maximum_columns=800,
        cell_scale=8, tile_cell_rows=16, signal_floor=1e-6,
        contract="ffac-source-bound-ea-direct-8x8-area-grid-v1",
        aggregation="8x8 arithmetic area mean of full-native EA camera-linear RGB, then natural log",
        colour_basis="RAW black-800; no WB, matrix, gamma, autoexposure or clipping")


@pytest.mark.parametrize("origin", [(0, 0), (-1, -1)])
def test_area_sampling_matches_original_bits_and_out_of_sensor_values(parameters, area_parameters, origin):
    channels = [np.random.default_rng(i).uniform(.1, 15000, (170, 50)).astype(np.float32) for i in range(3)]
    target = np.array([[1., .002, .1], [.001, 1., .2], [.00001, .00002, 1.]])
    distortion, ca_red, ca_blue = (tuple([0] * 11),) * 3
    kwargs = dict(parameters=parameters, area_parameters=area_parameters,
                  source_certificate_path=Path("source.json"), source_certificate_sha256="a" * 64,
                  expected_source_hashes={"arw_sha256": "b" * 64})
    actual, receipt = raw._sample_area_grid(channels, target, target, distortion, ca_red, ca_blue,
                                           origin, (18, 5), "fixture", **kwargs)
    source_path = "src/darsia/fluidflower/tools/ffac_raw_extended_reference.py"
    original = subprocess.check_output(["git", "show", "f0d338aa:" + source_path], cwd=Path(__file__).resolve().parents[2])
    function = next(node for node in ast.parse(original).body
                    if isinstance(node, ast.FunctionDef) and node.name == "registered_edge_aware_raw_area_grid")
    first = next(index for index, node in enumerate(function.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "(x_cell, y_cell)")
    body = function.body[first:-1]
    scope = dict(np=np, cv2=cv2, channels=channels, h_target_camera=target, h_target_review=target,
        distortion=distortion, ca_red=ca_red, ca_blue=ca_blue, origin_cell_xy=origin, shape_cells_hw=(18, 5),
        profile="fixture", cert_path=Path("source.json"), source_certificate_sha256="a" * 64,
        expected_source_hashes=kwargs["expected_source_hashes"],
        geo=SimpleNamespace(_PROFILE={"fixture": parameters["common"]["profiles"]["fixture"]["coefficients"]},
            _raw_xy=lambda *args: registration.raw_coordinates(*args, parameters=parameters["common"]["lens"])))
    exec(compile(ast.Module(body=body, type_ignores=[]), source_path, "exec"), scope)
    assert actual.dtype == scope["out"].dtype and actual.tobytes() == scope["out"].tobytes()
    assert receipt == scope["result_cert"]
    if origin[0] < 0:
        assert np.isnan(actual[:, 0, 0]).all()


def test_bad_target_shape_is_rejected_before_sampling(parameters, area_parameters):
    with pytest.raises(ValueError, match="coarse-grid shape"):
        raw._sample_area_grid([], np.eye(3), np.eye(3), (), (), (), (0, 0), (0, 5), "fixture",
            parameters=parameters, area_parameters=area_parameters, source_certificate_path="source.json",
            source_certificate_sha256="a" * 64, expected_source_hashes={})
