"""Original mean-colour SDK API forwards one unchanged colour reduction."""
import ast
import copy
from pathlib import Path
import subprocess
import warnings

import darsia
import numpy as np
import pytest

from darsia.presets.workflows.heterogeneous_color_analysis import get_mean_color
from darsia.signals.color.utils import get_mean_color as canonical_mean_color


@pytest.fixture(scope="module")
def original_mean_color():
    repo = Path(__file__).resolve().parents[2]
    source = subprocess.check_output([
        "git", "show", "65c0e85015700f764b2c4f5fed6bb558490ef7f0:"
        "src/darsia/presets/workflows/heterogeneous_color_analysis.py",
    ], cwd=repo).decode("utf-8")
    function = copy.deepcopy(next(node for node in ast.parse(source).body
                                  if isinstance(node, ast.FunctionDef) and node.name == "get_mean_color"))
    namespace = {"np": np, "darsia": darsia}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "original mean colour function", "exec"), namespace)
    return namespace["get_mean_color"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mask_kind", ["none", "ndarray", "image", "empty"])
@pytest.mark.parametrize("with_nan", [False, True])
def test_workflow_mean_color_keeps_original_masks_dtype_and_nan_pattern(
    original_mean_color, dtype, mask_kind, with_nan,
):
    values = np.arange(18, dtype=dtype).reshape(2, 3, 3)
    if with_nan:
        values[0, 0, 1] = np.nan
    image = darsia.Image(values, dim=2, scalar=False)
    selected = np.array([[True, False, True], [False, True, False]], dtype=bool)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "image":
        mask = darsia.Image(selected, dim=2, scalar=True)
    elif mask_kind == "empty":
        mask = np.zeros(selected.shape, dtype=bool)
    else:
        mask = selected
    before = values.copy()
    # Empty selections retain the original NumPy NaN result and warnings.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", RuntimeWarning)
        actual = get_mean_color(image, mask)
        expected = original_mean_color(image, mask)
        canonical = canonical_mean_color(image, mask)
    assert actual.dtype == expected.dtype == canonical.dtype
    assert actual.shape == expected.shape == canonical.shape == (3,)
    assert np.array_equal(actual, expected, equal_nan=True)
    assert np.array_equal(actual, canonical, equal_nan=True)
    assert np.array_equal(values, before, equal_nan=True)
    if mask_kind == "empty":
        assert np.isnan(actual).all()
