"""Extraction parity tests; no Build eligibility or campaign qualification."""
import ast
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from darsia.fluidflower.mass.n1_geometry import measured_depth, measured_geometry


PARENT = "fabf33f814c23980341720a12d96650faa46e75d"
SOURCE = "src/darsia/fluidflower/mass/n1_geometry.py"


@pytest.fixture(scope="module")
def original_geometry():
    """Execute only the exact parent function, independently of its new helper."""
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "show", PARENT + ":" + SOURCE], cwd=root).decode("utf-8")
    node = next(item for item in ast.parse(source).body
                if isinstance(item, ast.FunctionDef) and item.name == "measured_geometry")
    namespace = dict(np=np, LinearNDInterpolator=LinearNDInterpolator, cKDTree=cKDTree)
    exec(compile(ast.Module(body=[node], type_ignores=[]), SOURCE, "exec"), namespace)
    return namespace["measured_geometry"]


def inputs():
    # A nonconstant plane, hull vertices/interior, outside points and equidistant
    # nearest-point ties. Literal role-like IDs carry no scientific approval.
    x = np.array([[-1., 0., 1., 2., 3.], [1., 0., 1., 2., 1.]])
    y = np.array([[1., 0., 1., 2., 1.], [-1., 2., 3., 0., 2.]])
    facies = np.array([[0, 1, 2, 5, 9], [10, 11, 1, 2, 5]], np.uint16)
    points = np.array([[0., 0., .010], [2., 0., .012],
                       [0., 2., .014], [2., 2., .016]])
    return x, y, facies, points


def tables():
    # ID0 intentionally retains the original function's skip behavior.
    return dict(porosity_by_facies={0: 1., 1: .3, 2: .4, 5: .5},
                swi_by_facies={0: .0, 1: .2, 2: .25, 5: .3})


def exact_arrays(expected, actual):
    assert expected.keys() == actual.keys()
    for key in expected:
        assert expected[key].dtype == actual[key].dtype
        assert expected[key].shape == actual[key].shape
        assert expected[key].tobytes() == actual[key].tobytes(), key


@pytest.mark.parametrize("chunk", [1, 2, 3, 250000])
def test_depth_matches_frozen_original_inside_outside_ties_and_chunks(original_geometry, chunk):
    x, y, facies, points = inputs()
    before = [value.tobytes() for value in (x, y, facies, points)]
    expected = original_geometry(x, y, facies, points, **tables(), chunk_pixels=chunk)
    actual = measured_depth(x, y, points, chunk_pixels=chunk)
    keys = ("depth_m", "depth_nearest_outside_hull")
    exact_arrays({key: expected[key] for key in keys}, actual)
    np.testing.assert_array_equal(actual["depth_nearest_outside_hull"],
        [[True, False, False, False, True], [True, False, True, False, False]])
    assert before == [value.tobytes() for value in (x, y, facies, points)]
    # No synthetic property maps, domain or eligibility are returned.
    assert set(actual) == set(keys)


@pytest.mark.parametrize("chunk", [1, 2, 250000])
def test_default_geometry_all_outputs_remain_bit_identical(original_geometry, chunk):
    args = inputs()
    before = [value.tobytes() for value in args]
    kwargs = dict(tables(), chunk_pixels=chunk)
    exact_arrays(original_geometry(*args, **kwargs), measured_geometry(*args, **kwargs))
    assert before == [value.tobytes() for value in args]


@pytest.mark.parametrize("fault", [
    "x_dimension", "y_shape", "facies_shape", "x_nan", "y_inf",
    "points_dimension", "points_columns", "points_short", "points_nan", "depth_zero",
    "chunk_zero", "chunk_bool", "chunk_numpy_int", "phi_zero", "phi_nan", "swi_range",
])
def test_default_geometry_retains_original_bad_input_semantics(original_geometry, fault):
    x, y, facies, points = inputs()
    kwargs = tables()
    if fault == "x_dimension":
        x = x.ravel()
    elif fault == "y_shape":
        y = y[:, :-1]
    elif fault == "facies_shape":
        facies = facies[:, :-1]
    elif fault == "x_nan":
        x[0, 0] = np.nan
    elif fault == "y_inf":
        y[0, 0] = np.inf
    elif fault == "points_dimension":
        points = points.ravel()
    elif fault == "points_columns":
        points = points[:, :2]
    elif fault == "points_short":
        points = points[:2]
    elif fault == "points_nan":
        points[0, 0] = np.nan
    elif fault == "depth_zero":
        points[0, 2] = 0.
    elif fault.startswith("chunk_"):
        kwargs["chunk_pixels"] = {"chunk_zero": 0, "chunk_bool": True,
                                  "chunk_numpy_int": np.int64(2)}[fault]
    elif fault == "phi_zero":
        kwargs["porosity_by_facies"][1] = 0.
    elif fault == "phi_nan":
        kwargs["porosity_by_facies"][1] = np.nan
    else:
        kwargs["swi_by_facies"][1] = 1.1
    with pytest.raises(ValueError) as original:
        original_geometry(x, y, facies, points, **kwargs)
    with pytest.raises(type(original.value)) as actual:
        measured_geometry(x, y, facies, points, **kwargs)
    assert str(actual.value) == str(original.value)
    if fault not in {"facies_shape", "phi_zero", "phi_nan", "swi_range"}:
        depth_kwargs = {key: value for key, value in kwargs.items() if key == "chunk_pixels"}
        with pytest.raises(type(original.value)) as standalone:
            measured_depth(x, y, points, **depth_kwargs)
        assert str(standalone.value) == str(original.value)
