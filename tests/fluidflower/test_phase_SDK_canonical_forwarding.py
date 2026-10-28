"""SDK forwards preserve the original pure phase decisions and public DTOs."""
import ast
import copy
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from darsia.fluidflower import phase_learning as sdk_heads
from darsia.fluidflower import phase_retained_source_inference as sdk_domain
from darsia.fluidflower.phase import domain, heads


ORIGINAL_COMMIT = "65c0e85015700f764b2c4f5fed6bb558490ef7f0"


def _original_function(relative_path, name, class_name=None):
    repo = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "show", ORIGINAL_COMMIT + ":src/darsia/fluidflower/" + relative_path],
        cwd=repo,
    ).decode("utf-8")
    tree = ast.parse(source)
    parent = tree if class_name is None else next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    function = copy.deepcopy(next(
        node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == name
    ))
    function.decorator_list = []
    # Preserve the original executable body; avoid imports/type annotation evaluation.
    function.returns = None
    for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"np": np, "cv2": cv2}
    exec(compile(module, relative_path + ":" + name, "exec"), namespace)
    return namespace[name]


def _head(coefficients, intercept):
    return sdk_heads.SerializedPhaseHead(
        name="head", feature_contract=None, mean=np.array([0.25, -0.5]),
        scale=np.array([2.0, 0.5]), coefficients=np.array(coefficients),
        intercept=intercept, threshold=0.5,
    )


def test_sdk_logistic_head_preserves_stable_extremes_nan_and_width_guard():
    original = _original_function("phase_learning.py", "probability", "SerializedPhaseHead")
    model = _head([1.5, -0.75], 0.2)
    features = np.array([[[0.0, 0.0], [1000.0, -1000.0]],
                         [[-1000.0, 1000.0], [np.nan, 0.0]]], dtype=np.float32)
    actual = model.probability(features)
    expected = original(model, features)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape == (2, 2)
    assert np.array_equal(actual, expected, equal_nan=True)
    assert actual[0, 1] == 1.0 and actual[1, 0] == 0.0 and np.isnan(actual[1, 1])
    assert type(model).__module__ == "darsia.fluidflower.phase_learning"
    assert model.radiometry_profile is None
    with pytest.raises(ValueError, match="Feature width"):
        model.probability(np.zeros((1, 1, 3), dtype=np.float32))


def test_sdk_hierarchical_partition_retains_original_fields_and_nan_locations():
    original = _original_function("phase_learning.py", "predict_features", "HierarchicalPhaseModel")
    model = sdk_heads.HierarchicalPhaseModel(
        co2_presence=_head([1.5, -0.75], 0.2),
        gas_within_co2=_head([-0.5, 0.25], -0.1),
        adapter="adapter", bundle_root=Path("bundle"),
    )
    features = np.array([[[0.0, 0.0], [np.nan, 1.0]]], dtype=np.float64)
    actual, expected = model.predict_features(features), original(model, features)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name].dtype == expected[name].dtype == np.float32
        assert np.array_equal(actual[name], expected[name], equal_nan=True)
    assert model.bundle_root == Path("bundle")
    assert type(model).__module__ == "darsia.fluidflower.phase_learning"


def test_sdk_outer_ring_keeps_canvas_exterior_and_excludes_internal_holes():
    original = _original_function("phase_retained_source_inference.py", "_outer_domain_ring")
    cases = [np.ones((5, 5), dtype=bool), np.zeros((3, 3), dtype=bool)]
    closed_hole = np.ones((5, 5), dtype=bool)
    closed_hole[2, 2] = False
    cases.append(closed_hole)
    opened_hole = closed_hole.copy()
    opened_hole[:2, 2] = False
    cases.append(opened_hole)
    for mask in cases:
        before = mask.copy()
        actual = sdk_domain._outer_domain_ring(mask)
        expected = original(mask)
        assert actual.dtype == expected.dtype == np.bool_
        assert np.array_equal(actual, expected)
        assert np.array_equal(mask, before)
    assert not sdk_domain._outer_domain_ring(closed_hole)[2, 1]
    assert sdk_domain._outer_domain_ring(opened_hole)[2, 1]


def test_sdk_cache_constructor_preserves_public_class_and_readonly_contract():
    original = _original_function(
        "phase_retained_source_inference.py", "__post_init__", "RetainedSourceInferenceCache"
    )
    def arrays():
        return {
            "labels": np.zeros((2, 3), dtype=np.uint8),
            **{name: np.zeros((2, 3), dtype=bool) for name in (
                "cached_parent_modelled", "tie_reference_eligible", "prior_fill_eligible",
                "allowed_domain", "recomputed_old_domain_ring",
            )},
        }
    actual_fields, reference_fields = arrays(), arrays()
    actual = sdk_domain.RetainedSourceInferenceCache(
        **actual_fields, parent_fixed_area_mm2=None, audit={"purpose": "fixture"}
    )
    # The retained public dataclass is unchanged; the original method alone is the oracle.
    from types import SimpleNamespace
    reference = SimpleNamespace(**reference_fields)
    original(reference)
    for name in actual_fields:
        assert np.array_equal(getattr(actual, name), getattr(reference, name))
        assert not getattr(actual, name).flags.writeable
        assert not getattr(reference, name).flags.writeable
    assert type(actual).__module__ == "darsia.fluidflower.phase_retained_source_inference"
    assert actual.audit == {"purpose": "fixture"}


def test_sdk_methods_use_single_canonical_implementation(monkeypatch):
    model = _head([1.0, 0.0], 0.0)
    sentinel = object()
    monkeypatch.setattr(heads.SerializedPhaseHead, "probability", lambda self, values: sentinel)
    assert model.probability(np.zeros((1, 1, 2))) is sentinel
    monkeypatch.setattr(domain, "outer_domain_ring", lambda mask: sentinel)
    assert sdk_domain._outer_domain_ring(np.zeros((1, 1), dtype=bool)) is sentinel



def _original_metric_functions():
    repo = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "show", ORIGINAL_COMMIT + ":src/darsia/fluidflower/phase_pixel_area.py"],
        cwd=repo,
    ).decode("utf-8")
    names = {"_shape", "validate_pixel_area", "sum_pixel_area", "filter_components_by_area"}
    functions = [copy.deepcopy(node) for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "cv2": cv2}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 "original SDK metrics", "exec"), namespace)
    return namespace


def test_sdk_pixel_area_retains_grid_dtype_units_and_original_admission():
    from darsia.fluidflower import phase_pixel_area as sdk
    original = _original_metric_functions()
    assert sdk._shape((2, 3)) == original["_shape"]((2, 3)) == (2, 3)
    mask = np.array([[True, False, True], [False, True, False]], dtype=bool)
    for dtype in (np.float32, np.float64):
        area = np.array([[1.0, 1.2, 1.4], [1.6, 1.8, 2.0]], dtype=dtype)
        assert sdk.validate_pixel_area(area, mask.shape) is area
        assert original["validate_pixel_area"](area, mask.shape) is area
        for unit in ("mm2", "cm2", "m2"):
            assert sdk.sum_pixel_area(mask, area, unit=unit) == original["sum_pixel_area"](
                mask, area, unit=unit
            )
    assert sdk.sum_pixel_area(mask, 1.25) == original["sum_pixel_area"](mask, 1.25)
    for bad in (np.nan, 0.0, -1.0, np.full(mask.shape, np.nan), np.ones(mask.shape, dtype=np.uint8)):
        for function in (sdk.validate_pixel_area, original["validate_pixel_area"]):
            with pytest.raises(ValueError):
                function(bad, mask.shape)
    for function in (sdk._shape, original["_shape"]):
        with pytest.raises(ValueError):
            function((2.5, 3))
    for function in (sdk.sum_pixel_area, original["sum_pixel_area"]):
        with pytest.raises(ValueError, match="Area output unit"):
            function(mask, 1.25, unit="unknown")


def test_sdk_component_filter_keeps_positional_default_and_exact_area_boundary():
    from darsia.fluidflower import phase_pixel_area as sdk
    original = _original_metric_functions()["filter_components_by_area"]
    mask = np.array([[True, True, False, True],
                     [False, False, False, False]], dtype=bool)
    for area in (1.25, np.full(mask.shape, 1.25, dtype=np.float32),
                 np.full(mask.shape, 1.25, dtype=np.float64)):
        actual, audit = sdk.filter_components_by_area(mask, area)
        expected, original_audit = original(mask, area)
        assert np.array_equal(actual, expected) and audit == original_audit
        assert np.count_nonzero(actual) == 2
        assert audit["minimum_area_mm2"] == 2.5
        for minimum in (0.0, 1.25, 2.5, 2.5001):
            actual, audit = sdk.filter_components_by_area(mask, area, minimum, connectivity=4)
            expected, original_audit = original(mask, area, minimum, connectivity=4)
            assert np.array_equal(actual, expected) and audit == original_audit
    for function in (sdk.filter_components_by_area, original):
        with pytest.raises(ValueError, match="four-connectivity"):
            function(mask, 1.25, connectivity=8)
        with pytest.raises(ValueError, match="finite and nonnegative"):
            function(mask, 1.25, np.nan)
