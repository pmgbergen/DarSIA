"""Exact synthetic parity against preserved pre-extraction Git code objects."""
from dataclasses import replace
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from darsia.fluidflower.native import photometry as photo


BASE = "f0d338aa2be5e6ee8be6a2db0bd8f6fb70e9bd5a"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def parameters():
    # Test-only DATA; the reusable core has no numerical/configuration defaults.
    return json.loads((Path(__file__).parent / "fixtures" /
                       "native_photometry_parameters.json").read_bytes())


@pytest.fixture(scope="module")
def originals():
    """Compile immutable original modules without importing current facades."""
    package = ModuleType("_preserved_photometry")
    package.__path__ = []
    sys.modules[package.__name__] = package
    result = {}
    for short in ("ffac_v5_photometry", "ffac_v5_observed_sand", "ffac_v5_refgeom",
                  "ffac_v5_old_field_native", "ffac_v4r5_native_pilot"):
        relative = "src/darsia/fluidflower/tools/" + short + ".py"
        source = subprocess.check_output(["git", "show", BASE + ":" + relative], cwd=REPO)
        name = package.__name__ + "." + short
        module = ModuleType(name)
        module.__package__ = package.__name__
        sys.modules[name] = module
        setattr(package, short, module)
        exec(compile(source, BASE + ":" + relative, "exec"), module.__dict__)
        result[short] = module
    yield result
    for name in list(sys.modules):
        if name == package.__name__ or name.startswith(package.__name__ + "."):
            del sys.modules[name]


def same_array(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected, equal_nan=True)


def baseline(kind, shape, seed=1, *, flat=True):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    fac = np.ones(shape, np.int16)
    sand = np.ones(shape, bool)
    chart = (xx == 0) & (yy < 2)
    sand[chart] = False
    matrix = np.array([[1., .01, -.005], [.002, 1., .01], [.01, .003, 1.]])
    return kind(
        "fixture", matrix, np.linalg.inv(matrix), rng.normal(0, .01, 14),
        {1: .2}, {1: int(sand.sum())}, fac, sand, chart,
        np.array([.01, -.005, .002]), xx.astype(np.float32), yy.astype(np.float32),
        np.where(sand, 1., .4).astype(np.float32),
        rng.normal(6, .05, (3, *shape)).astype(np.float32),
        rng.uniform(-.2, .2, shape).astype(np.float32),
        np.column_stack([np.linspace(2, 5, 24), np.linspace(2, 7, 24)]),
        rng.normal(6, .04, (3, 24)).astype(np.float32),
        rng.normal(0, .02, (3, *shape)).astype(np.float32) if flat else None, .1)


def test_parameter_data_preserves_literal_types_and_is_required(parameters):
    assert type(parameters["light_basis_int_85"]) is int
    assert type(parameters["fit_baseline_grid_float_8_0"]) is float
    with pytest.raises(TypeError):
        photo.light_basis(np.zeros(2), np.zeros(2), 1)
    with pytest.raises(KeyError):
        photo.light_basis(np.zeros(2), np.zeros(2), 1, parameters={})


def test_all_arithmetic_and_threshold_ast_nodes_preserved(parameters):
    """Parameters substitute literals; ordering/operators/casts remain original."""
    mapping = {
        "ffac_v5_photometry": {
            "_log_rgb": "_log_rgb", "_basis": "polynomial_basis", "_same_fac": "_same_fac",
            "_fit_field": "_fit_grid_field", "_colour": "_colour", "_chart_plane": "_chart_plane",
            "fit_baseline_grid": "fit_baseline_grid", "baseline_field": "baseline_field",
            "_resize": "_resize", "_native_sand_field_tile": "_native_sand_field_tile",
            "apply_v5": "apply_registered", "render_ac28_response": "render_response",
            "load_ac28_response": "load_response", "save_baseline": "save_grid_baseline",
            "load_baseline": "load_grid_baseline", "sample_chart_log": "sample_chart_log",
            "_light_basis": "light_basis", "estimate_frame_light_grid": "estimate_frame_light_grid"},
        "ffac_v5_observed_sand": {
            "require_green_donor_only_receipt": "require_green_donor_only_receipt",
            "_rgb": "_observed_rgb", "_fit_field": "_fit_observed_field",
            "fit_baseline_grid": "fit_observed_baseline", "baseline_field": "observed_baseline_field",
            "apply_grid": "apply_grid", "apply_native": "apply_native",
            "estimate_frame_light_expanded_diagnostic": "estimate_frame_light_expanded",
            "make_local_flat": "make_local_flat", "save_baseline": "save_baseline",
            "load_baseline": "load_baseline"},
        "ffac_v5_refgeom": {
            "validate_homography": "validate_homography", "_map_coordinates": "_map_coordinates",
            "remap_to_reference": "remap_to_reference", "chart_xy_to_reference": "chart_xy_to_reference",
            "_remap_reference_tile": "_remap_reference_tile", "_native_exact_sand_field": "_native_exact_sand_field",
            "estimate_frame_light_refgeom": "estimate_frame_light_reference_geometry",
            "apply_refgeom_v5": "apply_reference_geometry"},
        "ffac_v5_old_field_native": {"apply_old_field_native": "apply_old_field_native"},
        "ffac_v4r5_native_pilot": {"native_area_log8": "native_area_log",
            "frame_light_from_native_grid": "frame_light_from_native_grid",
            "correct_frame_native_v4r5": "correct_frame_native"},
    }
    prefixes = dict(v5="ffac_v5_photometry", observed="ffac_v5_observed_sand",
                    refgeom="ffac_v5_refgeom")
    class Normalize(ast.NodeTransformer):
        def __init__(self, module, *, original):
            self.module, self.original = module, original
        def visit_Name(self, node):
            if self.original: node.id = mapping[self.module].get(node.id, node.id)
            if node.id == "h_ac28_to_frame_native": node.id = "h_reference_to_frame_native"
            return node
        def visit_Attribute(self, node):
            if self.original and isinstance(node.value, ast.Name) and node.value.id in prefixes:
                name = node.attr
                return ast.Name(mapping[prefixes[node.value.id]].get(name, name), ast.Load())
            return self.generic_visit(node)
        def visit_Subscript(self, node):
            if (isinstance(node.value, ast.Name) and node.value.id == "parameters"
                    and isinstance(node.slice, ast.Constant)):
                return ast.parse(repr(parameters[node.slice.value]), mode="eval").body
            return self.generic_visit(node)
        def visit_Call(self, node):
            node = self.generic_visit(node)
            node.keywords = [k for k in node.keywords if k.arg != "parameters"]
            return node
    checked = 0
    for module, names in mapping.items():
        source = subprocess.check_output(["git", "show", BASE +
            ":src/darsia/fluidflower/tools/" + module + ".py"], cwd=REPO)
        old_functions = {node.name: node for node in ast.parse(source).body
                         if isinstance(node, ast.FunctionDef)}
        for old_name, new_name in names.items():
            old = Normalize(module, original=True).visit(old_functions[old_name])
            new = Normalize(module, original=False).visit(
                ast.parse(inspect.getsource(getattr(photo, new_name))).body[0])
            defaults = {key.removeprefix(new_name.strip("_") + "_default_")
                        for key in parameters if key.startswith(new_name.strip("_") + "_default_")}
            def expressions(tree, *, current):
                output = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                        if (current and isinstance(node, ast.Compare)
                                and isinstance(node.left, ast.Name) and node.left.id in defaults
                                and len(node.ops) == 1 and isinstance(node.ops[0], ast.Is)
                                and isinstance(node.comparators[0], ast.Constant)
                                and node.comparators[0].value is None):
                            continue
                        output.append(ast.dump(node, include_attributes=False))
                return output
            assert expressions(new, current=True) == expressions(old, current=False), new_name
            checked += len(expressions(old, current=False))
    assert checked > 300


def test_basis_and_colour_preserve_order_dtype_and_nan(parameters, originals):
    old = originals["ffac_v5_photometry"]
    same_array(photo.polynomial_basis(9, 13, parameters=parameters), old._basis(9, 13))
    x, y = np.mgrid[:5, :7]
    same_array(photo.light_basis(x, y, 2, parameters=parameters), old._light_basis(x, y, 2))
    rgb = np.random.default_rng(3).normal(4, .2, (3, 7, 9)).astype(np.float32)
    rgb[:, 1, 2] = np.nan
    matrix = np.array([[1., .02, .001], [.03, 1., .04], [.005, .003, 1.]])
    same_array(photo._colour(rgb, matrix, parameters=parameters), old._colour(rgb, matrix))


def test_observed_field_and_grid_correction_exact(parameters, originals):
    old = originals["ffac_v5_observed_sand"]
    old_grid = old.ReferenceGrid((7, 9), (-1, 2), (6, 8))
    grid = photo.ReferenceGrid((7, 9), (-1, 2), (6, 8), parameters=parameters)
    fit = baseline(photo.Baseline, grid.shape_hw)
    ref = baseline(photo.Baseline, grid.shape_hw, 2)
    ref.sand[3, 4] = False
    af, am = photo.observed_baseline_field(fit, ref, grid, parameters=parameters)
    ef, em = old.baseline_field(fit, ref, old_grid)
    same_array(af, ef)
    same_array(am, em)
    eligible = np.ones(grid.shape_hw, bool)
    eligible[2, 3] = False
    light = np.linspace(0, .1, np.prod(grid.shape_hw), dtype=np.float32).reshape(grid.shape_hw)
    same_array(photo.apply_grid(fit.logB, fit, ref, grid, light_field=light,
                               source_eligible_mask=eligible, parameters=parameters),
               old.apply_grid(fit.logB, fit, ref, old_grid, light_field=light,
                              source_eligible_mask=eligible))


@pytest.mark.parametrize("flat", [False, True])
def test_native_projection_and_old_residual_exact(parameters, originals, flat):
    old = originals["ffac_v5_observed_sand"]
    old_field = originals["ffac_v5_old_field_native"]
    old_grid = old.ReferenceGrid((6, 8), (-1, 1), (5, 7))
    grid = photo.ReferenceGrid((6, 8), (-1, 1), (5, 7), parameters=parameters)
    fit = baseline(photo.Baseline, grid.shape_hw, flat=flat)
    ref = baseline(photo.Baseline, grid.shape_hw, 2, flat=flat)
    rng = np.random.default_rng(7)
    src = rng.normal(6, .2, (3, 17, 21)).astype(np.float32)
    src[:, 4, 3] = np.nan
    eligible = rng.uniform(size=src.shape[1:]) > .1
    H = np.array([[1.01, .02, 2.], [-.01, .97, 1.], [.0002, -.0001, 1.]]) * 2
    light = rng.normal(0, .03, grid.shape_hw).astype(np.float32)
    kwargs = dict(source_eligible_mask=eligible, light_field_ref8=light,
                  destination_origin_xy=(-3, 4), tile_rows=3)
    actual = photo.apply_native(src, fit, ref, grid, H, parameters=parameters, **kwargs)
    expected = old.apply_native(src, fit, ref, old_grid, H, **kwargs)
    same_array(actual, expected)
    field = rng.normal(0, .05, grid.shape_hw).astype(np.float32)
    same_array(photo.apply_old_field_native(actual, field, grid, H,
               destination_origin_xy=(-3, 4), tile_rows=2, parameters=parameters),
               old_field.apply_old_field_native(expected, field, old_grid, H,
               destination_origin_xy=(-3, 4), tile_rows=2))


def test_observed_baseline_fit_and_local_flat_exact(parameters, originals):
    old = originals["ffac_v5_observed_sand"]
    shape = (45, 48)
    old_grid = old.ReferenceGrid(shape, (0, 0), shape)
    grid = photo.ReferenceGrid(shape, (0, 0), shape, parameters=parameters)
    rng = np.random.default_rng(10)
    log = rng.normal(6, .02, (3, *shape)).astype(np.float32)
    chart = np.column_stack([np.tile(np.arange(4), 6) + 2.,
                             np.repeat(np.arange(6), 4) + 2.])
    kwargs = dict(run="fixture", logB=log, facies=np.ones(shape, np.int16),
        chart_xy=chart, chart_log_rgb=rng.normal(6, .02, (3, 24)).astype(np.float32),
        ref_chart_log_rgb=rng.normal(6, .02, (3, 24)).astype(np.float32),
        chart_usable=np.ones(24, bool), matrix=np.eye(3), rig_consensus=log.copy())
    actual = photo.fit_observed_baseline(grid=grid, parameters=parameters, **kwargs)
    expected = old.fit_baseline_grid(grid=old_grid, **kwargs)
    for key in actual.__dataclass_fields__:
        a, e = getattr(actual, key), getattr(expected, key)
        if isinstance(a, np.ndarray):
            same_array(a, e)
        else:
            assert a == e
    donors = [log + np.float32(.02), log + np.float32(.04)]
    support = np.ones(shape, bool)
    support[:2] = False
    same_array(photo.make_local_flat(donors, log, actual, support, grid,
                                    parameters=parameters),
               old.make_local_flat(donors, log, expected, support, old_grid))


def test_reference_geometry_remap_and_chart_exact(parameters, originals, monkeypatch):
    old = originals["ffac_v5_refgeom"]
    v5 = originals["ffac_v5_photometry"]
    shape = (7, 9)
    parameters["grid_shape_hw"] = list(shape)
    parameters["remap_to_reference_default_destination_shape"] = list(shape)
    monkeypatch.setattr(v5, "GRID", shape)
    H = np.array([[1.02, .01, .2], [.01, .98, -.3], [.0001, .0002, 1.]])
    src = np.arange(3 * np.prod(shape), dtype=np.float32).reshape(3, *shape)
    src[:, 1, 3] = np.nan
    same_array(photo.remap_to_reference(src, H, parameters=parameters),
               old.remap_to_reference(src, H, destination_shape=shape))
    labels = (np.arange(np.prod(shape)).reshape(shape) % 3).astype(np.uint8)
    same_array(photo.remap_to_reference(labels, H, categorical=True, parameters=parameters),
               old.remap_to_reference(labels, H, categorical=True, destination_shape=shape))
    chart = np.column_stack([np.linspace(0, 8, 24), np.linspace(0, 6, 24)])
    same_array(photo.chart_xy_to_reference(chart, H, parameters=parameters),
               old.chart_xy_to_reference(chart, H))


def test_frame_light_grid_and_expanded_field_exact(parameters, originals):
    old = originals["ffac_v5_photometry"]
    observed = originals["ffac_v5_observed_sand"]
    shape = (372, 684)
    fit = baseline(photo.Baseline, shape, flat=False)
    fit.fac[:100] = 0
    fit.sand[:100] = False
    fit.xy = np.column_stack([np.tile(np.linspace(580, 625, 4), 6),
                             np.repeat(np.linspace(40, 95, 6), 4)])
    fit.CB = old.sample_chart_log(fit.logB, fit.xy)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    frame = (fit.logB + (.03 + xx * .00001 + yy * .00002)[None]).astype(np.float32)
    kwargs = dict(frame_chart_xy8=fit.xy, frame_chart_usable=np.ones(24, bool))
    actual, info = photo.estimate_frame_light_grid(fit, frame, parameters=parameters, **kwargs)
    expected, old_info = old.estimate_frame_light_grid(fit, frame, **kwargs)
    same_array(actual, expected)
    assert info == old_info
    expanded_shape = (382, 710)
    grid = photo.ReferenceGrid(expanded_shape, (-16, 0), shape, parameters=parameters)
    old_grid = observed.ReferenceGrid(expanded_shape, (-16, 0), shape)
    def pad(a):
        pads = ((0, 10), (16, 10)) if a.ndim == 2 else ((0, 0), (0, 10), (16, 10))
        return np.pad(a, pads, mode="edge")
    changes = {key: pad(getattr(fit, key)) for key in
               ("fac", "sand", "chart", "Ug", "Vg", "taper", "logB", "gb0")}
    changes["xy"] = fit.xy + np.array([16, 0], np.float32)
    expanded_fit = replace(fit, **changes)
    kwargs = dict(frame_chart_xy_ref8=changes["xy"], frame_chart_usable=np.ones(24, bool))
    actual, info = photo.estimate_frame_light_expanded(expanded_fit, pad(frame), grid,
                                                      parameters=parameters, **kwargs)
    expected, old_info = observed.estimate_frame_light_expanded_diagnostic(
        expanded_fit, pad(frame), old_grid, **kwargs)
    same_array(actual, expected)
    assert info == old_info


def test_native_area_and_exact_baseline_whole_correction(parameters, originals, monkeypatch):
    old = originals["ffac_v4r5_native_pilot"]
    refgeom = originals["ffac_v5_refgeom"]
    shape, area = (3, 16, 24), (3, 2, 3)
    parameters["native_shape_chw"] = list(shape)
    parameters["area_grid_shape_chw"] = list(area)
    monkeypatch.setattr(old, "NATIVE_SHAPE_CHW", shape)
    monkeypatch.setattr(old, "AREA_GRID_SHAPE_CHW", area)
    grid = photo.ReferenceGrid((4, 5), (-1, 0), (2, 3), parameters=parameters)
    old_grid = originals["ffac_v5_observed_sand"].ReferenceGrid((4, 5), (-1, 0), (2, 3))
    fit = baseline(photo.Baseline, grid.shape_hw)
    ref = baseline(photo.Baseline, grid.shape_hw, 4)
    linear = np.random.default_rng(11).uniform(50, 1000, shape).astype(np.float32)
    expected_area = old.native_area_log8(linear)
    same_array(photo.native_area_log(linear, parameters=parameters), expected_area)
    eligible = np.ones(shape[1:], bool)
    eligible[1, 2] = False
    kwargs = dict(exact_baseline_source=True, frame_chart_xy8=None,
        frame_chart_usable=None, old_field_ref8=np.ones(grid.shape_hw, np.float32) * .02,
        expected_area_log8=expected_area, tile_rows=3)
    actual, info = photo.correct_frame_native(linear, fit, ref, grid, np.eye(3), eligible,
                                              parameters=parameters, **kwargs)
    expected, old_info = old.correct_frame_native_v4r5(linear, fit, ref, old_grid,
                                                       np.eye(3), eligible, **kwargs)
    same_array(actual, expected)
    assert info == old_info


def test_baseline_io_dtype_geometry_hash_and_legacy_constructor(parameters, originals, tmp_path):
    old = originals["ffac_v5_observed_sand"]
    shape = (7, 9)
    fit = baseline(photo.Baseline, shape)
    grid = photo.ReferenceGrid(shape, (-1, 2), (6, 8), parameters=parameters)
    old_grid = old.ReferenceGrid(shape, (-1, 2), (6, 8))
    provenance = dict(source_arw_sha256="1" * 64, source_review_sha256="2" * 64,
                      source_manifest_sha256="3" * 64, sand_mask_sha256="4" * 64)
    a, e = tmp_path / "actual.npz", tmp_path / "old.npz"
    actual_receipt = photo.save_baseline(a, fit, grid, provenance, parameters=parameters)
    expected_receipt = old.save_baseline(e, fit, old_grid, provenance)
    assert actual_receipt == expected_receipt
    loaded, geometry = photo.load_baseline(a, parameters=parameters)
    original_loaded, original_geometry = old.load_baseline(e)
    assert geometry.shape_hw == original_geometry.shape_hw
    for key in fit.__dataclass_fields__:
        actual, expected = getattr(loaded, key), getattr(original_loaded, key)
        if isinstance(actual, np.ndarray): same_array(actual, expected)
        else: assert actual == expected
    with a.open("ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        photo.load_baseline(a, parameters=parameters)
