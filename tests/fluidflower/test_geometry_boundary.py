"""Synthetic parity against original Git codeobjects; no campaign or NAS inputs."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Mapping

import cv2
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from darsia.fluidflower.geometry import areas, barrier, context

BASE = "1be6b7b510233c3d35d21ed4119f8552953408a3"
REPO = Path(__file__).resolve().parents[2]
PREFIX = "src/darsia/fluidflower/"
PARAMETERS = {
    "determinant_floor": 1e-14,
    "point_horizon_floor": 1e-12,
    "area_horizon_floor": 1e-10,
    "observation_horizon_floor": 1e-10,
    "raster_scale": 256,
    "raster_shift": 8,
    "canvas_margin": 0.5,
    "square_mm_per_cm2": 100,
    "nominal_pixel_area_mm2": 0.027053,
    "contracts": {
        "rig_context": "ffac-photographic-rig-domain-v1",
        "independent_context": "ffac-independent-rig-frame-context-v1",
    },
    "fields": {
        "reference_polygon": "polygon_ac28",
        "reference_fixtures": "excluded_polygons_ac28",
        "nominal_area_field": "pixel_area_mm2_fixed_ac28",
        "reference_metric_field": "H_ac28_to_metric",
        "reference_transform_field": "H_ac28_to_frame_native",
    },
}
BARRIER = {"contract": "ffac-frozen-barrier-geometric-extension-v1"}


def _tree(relative):
    return ast.parse(
        subprocess.check_output(
            ["git", "show", BASE + ":" + PREFIX + relative],
            cwd=REPO,
            text=True,
            encoding="utf-8",
        )
    )


def _codeobjects(relative, names, extra=None):
    selected = []
    for node in _tree(relative).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in ("FORMAT", "PIXEL_AREA_MM2_FIXED_AC28", "CONTRACT")
            for target in node.targets
        ):
            selected.append(node)
    module = ModuleType("_original_" + relative.replace("/", "_").replace(".", "_"))
    module.__dict__.update(
        np=np,
        cv2=cv2,
        dataclass=dataclass,
        replace=replace,
        Mapping=Mapping,
        hashlib=hashlib,
        distance_transform_edt=distance_transform_edt,
        Path=Path,
    )
    if extra:
        module.__dict__.update(extra)
    # Dataclasses resolve postponed annotations through their defining module.
    sys.modules[module.__name__] = module
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            BASE + ":" + relative,
            "exec",
        ),
        module.__dict__,
    )
    return module


@pytest.fixture(scope="module")
def old_geometry():
    return _codeobjects(
        "phase_ffac_rig_geometry.py",
        {
            "homography",
            "transform_points",
            "inverse_homography_area_map",
            "rig_plane_area_map",
            "photographic_observation_mask",
            "RigFrameContext",
            "rig_frame_context",
            "independent_rig_frame_context",
        },
    )


@pytest.fixture(scope="module")
def old_barrier():
    return _codeobjects(
        "phase_ffac_barrier_extension.py",
        {
            "array_sha256",
            "extend_frozen_barrier_components",
        },
    )


def _same(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected, equal_nan=True)


def _same_context(actual, expected):
    for name in (
        "rig_domain",
        "modelled",
        "filled_observed",
        "filled_unobserved",
        "hardware_excluded",
        "pixel_area_mm2",
    ):
        _same(getattr(actual, name), getattr(expected, name))
    assert set(actual.excluded_by_name) == set(expected.excluded_by_name)
    for name in actual.excluded_by_name:
        _same(actual.excluded_by_name[name], expected.excluded_by_name[name])
    for name in ("origin_xy", "native_shape_hw", "content_box_xywh", "parameters"):
        assert getattr(actual, name) == getattr(expected, name)


def _reference():
    return {
        "polygon_ac28": [[0, 0], [3, 0], [3, 2], [0, 2]],
        "excluded_polygons_ac28": {"fixture": [[1, 0], [2, 0], [2, 1], [1, 1]]},
        "pixel_area_mm2_fixed_ac28": 0.027053,
    }


def test_homography_matches_and_rejects_singular(old_geometry):
    matrix = np.array([[1, 0.03, 2], [0.02, 1, -1], [0.001, 0.002, 1]], np.float32)
    _same(
        context.homography(matrix, parameters=PARAMETERS),
        old_geometry.homography(matrix),
    )
    with pytest.raises(ValueError):
        context.homography(np.zeros((3, 3)), parameters=PARAMETERS)


def test_point_projection_preserves_float64_order(old_geometry):
    matrix = [[1.0, 0.03, 2.0], [0.02, 1.0, -1.0], [0.001, 0.002, 1.0]]
    points = np.array([[0, 0], [3, 2], [1.25, 0.5]], np.float32)
    _same(
        context.transform_points(points, matrix, parameters=PARAMETERS),
        old_geometry.transform_points(points, matrix),
    )


def test_inverse_area_matches_original_projective_jacobian(old_geometry):
    h = [[1.0, 0.03, 2.0], [0.02, 1.0, -1.0], [0.001, 0.002, 1.0]]
    metric = [[0.99, 0.0, 0.0], [0.0, 1.01, 0.0], [0.0, 0.0, 1.0]]
    actual = context.inverse_homography_area_map(
        h,
        (4, 5),
        origin_xy=(-2, -1),
        fixed_area_mm2=0.027053,
        reference_to_metric_h=metric,
        parameters=PARAMETERS,
    )
    expected = old_geometry.inverse_homography_area_map(
        h,
        (4, 5),
        origin_xy=(-2, -1),
        fixed_area_mm2=0.027053,
        H_ac28_to_metric=metric,
    )
    _same(actual, expected)
    assert actual.dtype == np.float32


def test_metric_area_keeps_original_double_inversion(old_geometry):
    h = [[0.99, 0.03, 2.0], [0.02, 1.01, -1.0], [0.001, 0.002, 1.0]]
    _same(
        context.rig_plane_area_map(
            h,
            (4, 5),
            origin_xy=(-2, -1),
            metric_pixel_area_mm2=0.027053,
            parameters=PARAMETERS,
        ),
        old_geometry.rig_plane_area_map(
            h,
            (4, 5),
            origin_xy=(-2, -1),
            metric_pixel_area_mm2=0.027053,
        ),
    )


def test_camera_footprint_preserves_inclusive_pixel_centres(old_geometry):
    h = np.eye(3)
    actual = context.photographic_observation_mask(
        h,
        (4, 3),
        (6, 5),
        native_origin_xy=(-1, -1),
        parameters=PARAMETERS,
    )
    _same(
        actual,
        old_geometry.photographic_observation_mask(
            h,
            (4, 3),
            (6, 5),
            native_origin_xy=(-1, -1),
        ),
    )
    assert actual[1, 1] and actual[3, 4] and not actual[0, 0]


def test_rig_context_preserves_all_masks_and_physical_audit(old_geometry):
    kwargs = dict(
        feature_valid_mask=np.ones((3, 4), bool),
        canvas_wh=(8, 7),
        content_box_xywh=(2, 1, 4, 3),
        photographic_observed_mask=np.ones((7, 8), bool),
    )
    _same_context(
        context.rig_frame_context(
            _reference(), np.eye(3), (3, 4), parameters=PARAMETERS, **kwargs
        ),
        old_geometry.rig_frame_context(_reference(), np.eye(3), (3, 4), **kwargs),
    )


def test_independent_context_preserves_top_and_metric_roles(old_geometry):
    reference = {
        "polygon_reference_native": _reference()["polygon_ac28"],
        "metric_pixel_area_mm2": 0.027053,
        "identity": "measured-reference",
    }
    top = np.ones((7, 8), bool)
    top[:2] = False
    kwargs = dict(
        feature_valid_mask=np.ones((3, 4), bool),
        canvas_wh=(8, 7),
        content_box_xywh=(2, 1, 4, 3),
        frame_photo_top_keep=top,
        frame_fixture_polygons_native={
            "fixture": _reference()["excluded_polygons_ac28"]["fixture"]
        },
        photographic_observed_mask=np.ones((7, 8), bool),
        H_frame_native_to_metric_pixel=np.eye(3),
    )
    _same_context(
        context.independent_rig_frame_context(
            reference, np.eye(3), (3, 4), parameters=PARAMETERS, **kwargs
        ),
        old_geometry.independent_rig_frame_context(
            reference, np.eye(3), (3, 4), **kwargs
        ),
    )
    bad = top.copy()
    bad[4, 2] = False
    kwargs["frame_photo_top_keep"] = bad
    with pytest.raises(ValueError, match="prefix"):
        context.independent_rig_frame_context(
            reference, np.eye(3), (3, 4), parameters=PARAMETERS, **kwargs
        )


def test_measured_revision_matches_original_array_and_rounding_audit(tmp_path):
    old = _codeobjects(
        "tools/revise_ffac_pixel_area.py",
        {"revise_area"},
        {
            "checked": lambda row: Path(row["path"]),
            "asset": lambda path: {"path": str(path)},
        },
    )
    original = np.array([[0.017, 0.021], [0.027053, 0.033]], np.float32)
    source = {"path": str(tmp_path / "area.npz"), "sha256": "synthetic-binding"}
    np.savez(source["path"], pixel_area_mm2=original)
    result, expected_audit = old.revise_area(source, tmp_path / "scaled.npz", 0.9224286)
    with np.load(result["path"], allow_pickle=False) as saved:
        expected = saved["pixel_area_mm2"]
    actual, audit = areas.apply_measured_area_revision(
        original,
        0.9224286,
        source_binding=source,
        parameters={"maximum_relative_rounding_error": 1e-7},
    )
    _same(actual, expected)
    assert audit == expected_audit
    with pytest.raises(ValueError):
        areas.apply_measured_area_revision(
            original.astype(np.float64),
            0.9224286,
            source_binding=source,
            parameters={"maximum_relative_rounding_error": 1e-7},
        )


def test_selected_scale_uses_exact_original_assignment():
    tree = _tree("phase_ffac_selected_inference_corrected.py")
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "area" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "astype"
    )
    original = np.array([[0.017, 0.021], [0.027053, 0.033]], np.float32)
    expected = eval(
        compile(ast.Expression(assignment.value), "original-area-role", "eval"),
        {
            "original_area": original,
            "measured_revision": {"area_factor": 0.9224286},
        },
    )
    _same(areas.apply_selected_area_scale(original, 0.9224286), expected)


def test_component_byte_hash_matches_original_noncontiguous_layout(old_barrier):
    array = np.arange(24, dtype=np.uint16).reshape(4, 6)[:, ::2]
    assert barrier.array_bytes_sha256(array) == old_barrier.array_sha256(array)


def _original_admission():
    fn = next(
        n
        for n in _tree("phase_gas_barrier_runtime.py").body
        if isinstance(n, ast.FunctionDef) and n.name == "read_components"
    )
    fn = copy.deepcopy(fn)
    fn.name = "admit"
    fn.args.args[0].arg = "values"
    fn.body = fn.body[1:]
    namespace = {"np": np}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            "original-component-admission",
            "exec",
        ),
        namespace,
    )
    return namespace["admit"]


def _components():
    ids = np.array([[2, 0, 0], [2, 0, 3]], np.uint16)
    wall = np.array([[False, True, False], [False, True, False]], bool)
    domain = (ids > 0) | wall
    return dict(open_components=ids, barrier_mask=wall, reference_domain=domain)


def test_component_admission_matches_original_and_rejects_scope_error():
    values = _components()
    old = _original_admission()
    actual, expected = barrier.validate_component_arrays(values, (2, 3)), old(
        values, (2, 3)
    )
    for key in actual:
        _same(actual[key], expected[key])
    invalid = dict(values, reference_domain=np.zeros((2, 3), bool))
    with pytest.raises(ValueError):
        barrier.validate_component_arrays(invalid, (2, 3))


def test_extension_preserves_inherited_ids_and_full_original_audit(old_barrier):
    values = _components()
    domain = np.ones((2, 3), bool)
    actual, audit = barrier.extend_static_components(
        values["open_components"],
        values["barrier_mask"],
        domain,
        parameters=BARRIER,
    )
    expected, expected_audit = old_barrier.extend_frozen_barrier_components(
        values["open_components"],
        values["barrier_mask"],
        domain,
    )
    for key in expected:
        _same(actual[key], expected[key])
    assert audit == expected_audit


def _original_transport(old_barrier, box):
    # Execute the actual original orchestration function with small explicit
    # metadata/loader substitutes. No expected transport arithmetic is copied.
    metadata = SimpleNamespace(
        SOURCE_SHAPE=(2, 3),
        SOURCE_BOX=box,
        phase_grid=lambda shape: {"shape_hw": shape},
    )
    rig = _codeobjects(
        "tools/run_ffac_rig_camera_inference.py",
        {"_canvas", "canvas_homography"},
        {"metadata": metadata},
    )
    tree = _tree("phase_ffac_selected_inference.py")
    fn = copy.deepcopy(
        next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_components"
        )
    )
    fn.body = [n for n in fn.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    namespace = {
        "np": np,
        "cv2": cv2,
        "rig": rig,
        "read_components": lambda asset, shape: _original_admission()(
            _components(), shape
        ),
        "bound_json": lambda asset: asset["document"],
        "extend_frozen_barrier_components": old_barrier.extend_frozen_barrier_components,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            "original-static-transport",
            "exec",
        ),
        namespace,
    )
    return namespace["_components"]


@pytest.mark.parametrize("unchanged", [True, False])
def test_static_transport_matches_both_original_acquisition_paths(
    old_barrier, unchanged
):
    values = _components()
    box = (1, 1, 3, 2)
    domain = np.ones((4, 5), bool)
    h = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    component = {"path": "synthetic-components", "sha256": "bound-component-data"}
    reference = {
        "components": component,
        "geometry_manifest": {"document": {"source_component_ids": [2, 3]}},
    }
    parent = {
        "binding": {"components": component},
        "qa": {"source_component_ids": [2, 3]},
    }
    source = {
        "studio_anchor": unchanged,
        "parent_source": {
            "source_changed": not unchanged,
            "retained_source_cache": {"parent_frame_receipt": {"document": parent}},
        },
    }
    receipt = {"rig_geometry": {"H_ac28_to_frame_native": h}}
    old = _original_transport(old_barrier, box)
    expected, expected_ids, expected_proof = old(
        SimpleNamespace(plan={"frozen_barrier_reference": reference}),
        source,
        receipt,
        domain,
    )
    actual, actual_ids, actual_proof = barrier.prepare_static_components(
        values,
        domain,
        native_shape_hw=(2, 3),
        source_box_xywh=box,
        source_to_frame_h=h,
        source_ids=[2, 3],
        unchanged_source=unchanged,
        source_binding=component if unchanged else reference,
        parameters=BARRIER,
    )
    for key in expected:
        _same(actual[key], expected[key])
    assert actual_ids == expected_ids
    assert actual_proof == expected_proof
