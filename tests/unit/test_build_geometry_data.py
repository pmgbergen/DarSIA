"""Coordinate DATA fixtures, not RAW decoding or independent scientific QA.

The synthetic audits bind actual tiny JSON/curve files through the original
DataReader. Positive tests establish coordinate algebra/consumer compatibility,
never acquisition correction, photographic accuracy or physical eligibility.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass.n1_pipeline import DataReader, _write_json, asset
from darsia.fluidflower.native import raw_positioning, raw_rig_source
from darsia.fluidflower.pipeline.build_geometry_data import prepare_geometry_data
from darsia.fluidflower.geometry.photographic_data import top_curve_mask
from darsia.fluidflower.pipeline.raw_acquisition import FORMAT as RAW_ACQUISITIONS


def shift(x, y):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    root.mkdir()
    raw = dict(path=str(root / "source.ARW"), sha256="a" * 64, size_bytes=123)
    # There is intentionally no RAW file. Any source decode/read is a defect.
    before, transform = shift(10, 5), shift(2, 1)
    after = before @ np.linalg.inv(transform)
    common = dict(sensor=dict(camera_wh=[12, 8], black_level=512),
                  profiles=dict(camera=dict(coefficients=[1., 0., 0.])),
                  lens=dict(origin=[0, 0]), explicit_camera_field=False)
    native = dict(registration=dict(common=common),
                  source=dict(profiles=dict(sample="camera"), homography_match_atol=1e-9, native_shape_hw=[3, 5]))
    source_binding = dict(contract=raw_rig_source.CONTRACT, run="sample", profile_id="camera",
        baseline_raw=raw, models=dict(fit=dict(data={"path": "frozen"}, sidecar={"path": "frozen-json"}),
                                     reference={"declared": "unchanged"}, old_field=None),
        native_shape_hw=[2, 4], execution_identity={"retained": "exact old producer"})
    def renderer(matrix, shape):
        return dict(contract=raw_positioning.CONTRACT, source_kind="raw", raw_acquisition=raw,
            profile_id="camera", native_shape_hw=shape, camera_u_wh=[12, 8],
            native_to_camera_u_h=matrix.tolist(),
            sensor_metadata=dict(make="fixture", model="fixture", orientation=1, crop=[0, 0, 12, 8]))
    def audit(matrix, shape, reference):
        return dict(contract=raw_rig_source.CONTRACT, jpeg_inputs=False,
            source_identity=dict(kind="raw-acquisition", raw_sha256=raw["sha256"]),
            sources=dict(camera_arw=raw), processor=dict(profile_id="camera",
                models=deepcopy(source_binding["models"]),
                execution_identity=deepcopy(source_binding["execution_identity"]), native_shape_hw=shape),
            renderer=renderer(matrix, shape), H_ac28_to_frame_native=reference.tolist(),
            H_v5_B_to_frame_native=reference.tolist())
    old, now = audit(before, [2, 4], np.eye(3)), audit(after, [3, 5], transform)
    phase = dict(parameters=dict(geometry=dict(native_shape_hw=[3, 5],
        source_box_xywh=[3, 2, 5, 3], rig_origin_xy=[-3, -2]),
        scientific=dict(native_shape_hw=[3, 5], photograph_canvas_wh=[10, 6])),
        geometry_fields=dict(reference_to_frame="H_ac28_to_frame_native", reference_to_metric="H_ac28_to_metric"))
    polygon = [[0., 0.], [3., 0.], [3., 1.], [0., 1.]]
    fixture = dict(panel=[[1., 0.], [2., 0.], [2., 1.]])
    photo = dict(path="source.JPG", sha256="b" * 64)
    geometry = dict(run="sample", raw_source=raw, source_photo=photo,
        source_native_shape_hw=[2, 4], phase_shape_hw=[4, 8], native_origin_xy=[-2, -1],
        source_content_box_xywh=[2, 1, 4, 2], photograph_canvas_wh=[8, 4],
        H_ac28_to_frame_native=np.eye(3).tolist(), H_reference_rig_to_frame_native=shift(1, 0).tolist(),
        frame_inner_polygon_native=polygon, frame_fixture_polygons_native=fixture,
        reference_rig=dict(identity="independent walls", polygon_reference_native=polygon,
                           physical_scale_approved=False),
        area_parameters=dict(H_ac28_to_frame_native=np.eye(3).tolist(), H_ac28_to_metric=np.eye(3).tolist(),
                             fixed_area_mm2=.1, physical_scale_approved=False),
        boundary_evidence="Independent source wall coordinates; retained uncertainty",
        edge_gate_verified_on_own_photograph=False, unobserved_wall_uncertainty=True,
        physical_scale_qa_pass=False, mass_eligible=False, fixtures_verified_none=False,
        extra_original_metadata=dict(unobserved_intervals=[1, 2], uncertainty_px=8))
    curve = root / "curve.npy"
    np.save(curve, np.full(4, .2))
    curve_bound = asset(curve)
    evidence = dict(curve=curve_bound, photographic_definition="independent photographed grain onset",
                    photographic_estimate_uncertainty_px=8, not_claimed_grain_accuracy_2px=True)
    hardware = dict(public_source=photo, polygons_native_xy=fixture,
                    phase_labels_used=False, grain_registration_used=False,
                    photographic_control_approved=False, hardware_exclusion_approved=False)
    value = SimpleNamespace(root=root, output=tmp_path / "output", old=old, now=now,
        old_config=deepcopy(native), native=native, old_binding=source_binding,
        new_binding=dict(deepcopy(source_binding), native_shape_hw=[3, 5]),
        phase=phase, geometry=geometry, evidence=evidence, hardware=hardware,
        transform=transform, shape=[6, 10], curve=curve_bound, top_phase=False,
        old_run="sample", current_source_binding_override=False)
    value.old_config["source"]["native_shape_hw"] = [2, 4]
    class CurveOnlyReader(DataReader):
        def array(self, binding):
            assert binding == curve_bound
            return super().array(binding)
    value.reader = CurveOnlyReader()
    def seal():
        def put(name, doc):
            path = root / (name + ".json")
            if path.exists():
                # Fixture updates are explicit private JSON mutations, never a
                # producer overwrite or a scientific input repair.
                import json
                path.write_text(json.dumps(doc), encoding="utf-8")
                return asset(path)
            return _write_json(path, doc)
        old_binding = put("old_binding", value.old_binding)
        new_binding = put("new_binding", value.new_binding)
        value.old["processor"]["source_binding"] = {k: old_binding[k] for k in ("path", "sha256")}
        value.now["processor"]["source_binding"] = {k: new_binding[k] for k in ("path", "sha256")}
        if value.current_source_binding_override:
            value.now["processor"]["source_binding"] = {k: old_binding[k] for k in ("path", "sha256")}
        native_conf = put("native", value.native)
        bridge = put("bridge", dict(format="darsia-build-native-bridge-v1",
            native_configuration=native_conf, photometry_source=old_binding, raw_source_binding=new_binding))
        frame = put("frame", dict(contract=raw_rig_source.FRAME_CONTRACT, run="sample", acquisition=raw,
            reference_to_native_h=value.transform.tolist(), baseline_to_native_h=value.transform.tolist()))
        receipt = put("receipt", dict(format="darsia-build-native-frame-v1", bridge=bridge,
            native_frame=frame, raw_source_binding=new_binding, correction=value.now))
        value.geometry["hardware_pose_receipt"] = put("hardware", value.hardware)
        value.geometry["frame_photo_top_keep"] = dict(path="old-unread-mask.npz", sha256="c" * 64,
            curve_source=value.curve, reference_photo_evidence=put("top_evidence", value.evidence),
            phase_or_labels_used=value.top_phase, acceptance_margin_px=0,
            outside_horizontal_support_policy="constant endpoint in unobserved strips")
        old_geometry, old_audit = put("geometry", value.geometry), put("old_audit", value.old)
        old_native = put("old_native", value.old_config)
        acquisition_config = put("old_acquisitions_configuration", dict(format=RAW_ACQUISITIONS,
            run=value.old_run, native=dict(configuration=old_native),
            acquisitions=[dict(geometry=old_geometry, native_task=dict(processor=dict(raw_binding=old_binding)))]))
        collection = put("old_acquisitions", dict(format=RAW_ACQUISITIONS, run="sample",
            configuration=acquisition_config, records=[dict(full=dict(binding=dict(
                producer_audit=old_audit, geometry=deepcopy(value.geometry))))], adopted=False))
        return dict(old_geometry=old_geometry, old_raw_audit=old_audit, old_acquisition_collection=collection,
            old_native_configuration=old_native, new_bridge_frame=receipt,
            native_configuration=native_conf, phase_configuration=put("phase", value.phase),
            phase_shape_hw=value.shape, output=value.output, reader=value.reader)
    value.seal = seal
    value.call = lambda: prepare_geometry_data(**seal())
    def forbidden(*args, **kwargs):
        raise AssertionError("No RAW or scientific image decoding belongs in geometry DATA")
    monkeypatch.setattr(raw_positioning, "read_sensor", forbidden)
    import darsia
    monkeypatch.setattr(darsia, "imread", forbidden)
    return value


def test_new_grid_coordinates_preserve_independent_wall_reference_and_all_negative_scope(case):
    args = case.seal()
    before = {path: path.read_bytes() for path in case.root.iterdir()}
    actual = prepare_geometry_data(**args)
    row = DataReader().json(actual["geometry"])
    np.testing.assert_array_equal(row["H_ac28_to_frame_native"], case.transform)
    np.testing.assert_array_equal(row["H_reference_rig_to_frame_native"], case.transform @ shift(1, 0))
    assert row["H_reference_rig_to_frame_native"] != row["H_ac28_to_frame_native"]
    np.testing.assert_array_equal(row["frame_inner_polygon_native"], np.asarray(case.geometry["frame_inner_polygon_native"]) + [2, 1])
    np.testing.assert_array_equal(row["frame_fixture_polygons_native"]["panel"], np.asarray(case.hardware["polygons_native_xy"]["panel"]) + [2, 1])
    assert row["native_origin_xy"] == [-3, -2] and row["source_native_shape_hw"] == [3, 5]
    assert row["phase_shape_hw"] == [6, 10]
    for key in ("reference_rig", "boundary_evidence", "edge_gate_verified_on_own_photograph",
                "unobserved_wall_uncertainty", "physical_scale_qa_pass", "mass_eligible", "extra_original_metadata"):
        assert row[key] == case.geometry[key]
    assert row["area_parameters"]["physical_scale_approved"] is False
    assert "physical_context_verified" not in row
    keep = DataReader().array(actual["frame_photo_top_keep"])
    expected = np.zeros((6, 10), bool)
    expected[4:] = True
    np.testing.assert_array_equal(keep, expected)
    assert actual["phase_roles"]["reference_to_current_h"] == case.transform.tolist()
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("change,match", [
    ("wrong_raw", "same acquisition"), ("jpeg", "same acquisition"),
    ("camera_extra", "camera-U basis"), ("sensor_extra", "sensor metadata"),
    ("old_field", "photometry models"), ("old_reference", "old RAW renderer"),
    ("new_reference", "reference transforms disagree"), ("missing_wall", "Independent wall"),
    ("fixture_from_sand", "independent source"), ("wrong_outline", "independent source"),
    ("top_from_phase", "photographed top"), ("wrong_origin", "canvas, origin"),
])
def test_missing_or_mismatched_observations_refuse_before_any_output(case, change, match):
    if change == "wrong_raw": case.old["source_identity"]["raw_sha256"] = "d" * 64
    elif change == "jpeg": case.old["jpeg_inputs"] = True
    elif change == "camera_extra": case.old_config["registration"]["common"]["camera_field_5"] = True
    elif change == "sensor_extra": case.old["renderer"]["sensor_metadata"]["extra_field_5"] = True
    elif change == "old_field": case.old_binding["models"]["old_field"] = {"bound": True}
    elif change == "old_reference": case.old["H_ac28_to_frame_native"] = shift(1, 0).tolist()
    elif change == "new_reference": case.now["H_ac28_to_frame_native"] = shift(3, 1).tolist()
    elif change == "missing_wall": case.geometry["boundary_evidence"] = ""
    elif change == "fixture_from_sand": case.hardware["grain_registration_used"] = True
    elif change == "wrong_outline": case.hardware["polygons_native_xy"] = {"panel": [[0, 0], [1, 0], [1, 1]]}
    elif change == "top_from_phase": case.top_phase = True
    elif change == "wrong_origin": case.phase["parameters"]["geometry"]["rig_origin_xy"] = [0, 0]
    args = case.seal()
    with pytest.raises(ValueError, match=match):
        prepare_geometry_data(**args)
    assert not case.output.exists()


def test_retained_top_helper_keeps_origin_ceil_and_unobserved_endpoint_rule():
    mask, boundary = top_curve_mask(np.array([.1, .9]), shift(1, 1), (5, 6), (-2, -1))
    np.testing.assert_array_equal(boundary, [1.1, 1.1, 1.1, 1.1, 1.9, 1.9])
    expected = np.zeros((5, 6), bool)
    expected[3:] = True
    np.testing.assert_array_equal(mask, expected)


def test_existing_output_is_never_reused_or_overwritten(case):
    case.output.mkdir()
    marker = case.output / "keep"
    marker.write_bytes(b"retained failed or completed namespace")
    with pytest.raises(ValueError, match="fresh positively local"):
        case.call()
    assert marker.read_bytes() == b"retained failed or completed namespace"


def test_geometry_output_cannot_overlap_actual_aliased_metadata_folder(case):
    args = case.seal()
    logical = args["phase_configuration"]
    original = Path(logical["path"]).read_bytes()
    folder = case.output.parent / "aliased_geometry_metadata"
    folder.mkdir()
    physical = folder / "phase.json"
    physical.write_bytes(original)
    args["reader"] = DataReader(aliases=[dict(logical=logical, physical=asset(physical))])
    args["output"] = folder / "new_geometry"
    with pytest.raises(ValueError, match="overlaps its source folders"):
        prepare_geometry_data(**args)
    assert not args["output"].exists()
    assert physical.read_bytes() == original
    assert Path(logical["path"]).read_bytes() == original


def test_old_audit_without_original_producer_configuration_join_refuses(case):
    args = case.seal()
    collection = case.reader.json(args["old_acquisition_collection"])
    collection["records"][0]["full"]["binding"]["producer_audit"] = args["old_geometry"]
    path = case.root / "unrelated_old_collection.json"
    args["old_acquisition_collection"] = _write_json(path, collection)
    with pytest.raises(ValueError, match="original RAW acquisition producer"):
        prepare_geometry_data(**args)
    assert not case.output.exists()


def test_existing_series_geometry_consumer_accepts_new_row_and_top_without_qualification(case):
    from darsia.fluidflower.pipeline.series import SeriesInputs
    actual = case.call()
    owner = SeriesInputs.__new__(SeriesInputs)
    owner.reader = DataReader()
    owner.parameters = dict(geometry=case.phase["parameters"]["geometry"])
    owner.phase = dict(case.phase, geometry_parameters=dict(
        determinant_floor=1e-14, point_horizon_floor=1e-12, area_horizon_floor=1e-12,
        observation_horizon_floor=1e-12, canvas_margin=.5, raster_scale=256, raster_shift=8,
        nominal_pixel_area_mm2=.1, square_mm_per_cm2=100,
        contracts=dict(rig_context="rig", independent_context="independent"),
        fields=dict(reference_polygon="polygon_ac28", reference_fixtures="excluded_polygons_ac28",
                    nominal_area_field="pixel_area_mm2_fixed_ac28",
                    reference_metric_field="H_ac28_to_metric", reference_transform_field="H_ac28_to_frame_native")))
    renderer = case.now["renderer"]
    audit = dict(renderer=dict(review_to_camera_h=renderer["native_to_camera_u_h"],
                               camera_jpeg_wh=renderer["camera_u_wh"]))
    row = owner.reader.json(actual["geometry"])
    context, observed, top = owner.geometry(row, np.ones((3, 5), bool), audit)
    assert context.rig_domain.shape == observed.shape == top.shape == (6, 10)
    assert context.parameters["physical_scale_approved"] is False
    assert "physical_context_verified" not in context.parameters


@pytest.mark.parametrize("change", ["current_source_binding", "old_configuration_run",
    "old_processor_shape", "current_processor_shape", "old_configuration_shape", "current_configuration_shape"])
def test_original_producer_run_binding_and_full_native_shape_joins_are_required(case, change):
    if change == "current_source_binding": case.current_source_binding_override = True
    elif change == "old_configuration_run": case.old_run = "another-run"
    elif change == "old_processor_shape": case.old["processor"]["native_shape_hw"] = [2, 5]
    elif change == "current_processor_shape": case.now["processor"]["native_shape_hw"] = [2, 5]
    elif change == "old_configuration_shape": case.old_config["source"]["native_shape_hw"] = [2, 5]
    elif change == "current_configuration_shape": case.native["source"]["native_shape_hw"] = [2, 5]
    with pytest.raises(ValueError, match="original RAW acquisition producer|original RAW audit"):
        case.call()
    assert not case.output.exists()


@pytest.mark.parametrize("change", ["missing_reference_polygon", "degenerate_fixture", "invalid_metric", "photograph_outside_canvas"])
def test_missing_polygon_metric_or_impossible_photograph_canvas_refuses(case, change):
    if change == "missing_reference_polygon": case.geometry["reference_rig"].pop("polygon_reference_native")
    elif change == "degenerate_fixture":
        case.geometry["frame_fixture_polygons_native"]["panel"] = [[1., 0.], [2., 0.]]
    elif change == "invalid_metric": case.geometry["area_parameters"]["H_ac28_to_metric"] = np.zeros((3, 3)).tolist()
    elif change == "photograph_outside_canvas": case.phase["parameters"]["scientific"]["photograph_canvas_wh"] = [11, 6]
    with pytest.raises(ValueError, match="Rig points|three vertices|singular|canvas, origin"):
        case.call()
    assert not case.output.exists()


def test_reporting_canvas_may_extend_below_photograph_without_claiming_observation(case):
    case.shape = [7, 10]
    actual = case.call()
    row = DataReader().json(actual["geometry"])
    assert row["phase_shape_hw"] == [7, 10]
    assert row["photograph_canvas_wh"] == [10, 6]
    assert row["edge_gate_verified_on_own_photograph"] is False
    assert row["physical_scale_qa_pass"] is False
