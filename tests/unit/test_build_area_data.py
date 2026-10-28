"""Current-grid producer contracts on tiny independently bound source DATA."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.test_build_geometry_data import case  # shared original RAW-coordinate fixture
from darsia.fluidflower.mass.n1_pipeline import DataReader, _write_json, asset
from darsia.fluidflower.pipeline.build_geometry_data import prepare_geometry_data, _read_geometry_transport
from darsia.fluidflower.pipeline.build_area_data import prepare_measured_area_data


@pytest.fixture
def area_case(case):
    case.geometry.update(frame_index=1, elapsed_h=.25, scientific_task_id="original-frame-1",
                         source_sha256=case.geometry["source_photo"]["sha256"])
    metric = np.array([[1., 0., 0.], [0., 1., 0.], [1 / 16, 0., 1.]])
    case.geometry["area_parameters"]["H_ac28_to_metric"] = metric.tolist()
    case.phase["geometry_parameters"] = dict(determinant_floor=1e-14, area_horizon_floor=1e-12)
    case.phase["area_parameters"] = dict(maximum_relative_rounding_error=1e-7)
    produced = prepare_geometry_data(**case.seal())
    historical = dict(path=str(case.root / "never-read-historical.area.npz"),
                      sha256="d" * 64, array_key="pixel_area_mm2")
    measured = dict(path=str(case.root / "never-read-measured.area.npz"),
                    sha256="e" * 64, array_key="pixel_area_mm2")
    revision = dict(contract="ffac-measured-pixel-area-revision-v1", source="original measured width",
        measured_by="source observer", measurement_date="original", inner_width_mm=900.,
        linear_factor=.9604315, area_factor=.9224286, physical_scale_approved=True,
        approval_basis="original width-only uniform isotropic scale", inner_height_config_mm=490.,
        vertical_scale_independently_verified=False, nominal_historical_pixel_area_mm2_preserved=.1,
        mass_eligible=False, bottom_mass_allowed=False)
    descriptor = dict(contract="ffac-rig-domain-revision-v1", phase_shape_hw=[4, 8],
        native_origin_xy=[-2, -1], fixed_area_mm2=.1, area_map=historical, mass_eligible=False,
        parameters=dict(H_frame_native_to_metric_pixel=metric.tolist(), metric_pixel_area_mm2=.1,
                        physical_scale_approved=False))
    corrected_descriptor = deepcopy(descriptor)
    corrected_descriptor.update(area_map=measured, physical_scale_revision=revision)
    measured_metric = metric.copy()
    measured_metric[:2] *= revision["linear_factor"]
    corrected_descriptor["parameters"].update(H_frame_native_to_metric_pixel=measured_metric.tolist(),
        physical_scale_approved=True, physical_scale_revision=revision)
    role = dict(role="measured_parent", historical=historical, measured=measured,
                factor=revision["area_factor"], revision=revision)
    old_source = dict(run="sample", frame_index=1, elapsed_h=.25,
        scientific_task_id="original-frame-1", source_path=case.geometry["source_photo"]["path"],
        source_sha256=case.geometry["source_photo"]["sha256"], rig_geometry=deepcopy(case.geometry))
    historical_receipt = dict(contract="ffac-rig-camera-inference-frame-v1", complete=True,
        binding=dict(pipeline_sha256="1" * 64, source=old_source), rig_domain_revision=descriptor)
    measured_receipt = dict(contract=revision["contract"], complete=True, run="sample", frame_index=1,
        elapsed_h=.25, source_sha256=old_source["source_sha256"], rig_domain_revision=corrected_descriptor,
        measurement=revision, numerical_scale_qa=dict(source=historical), mass_eligible=False, bottom_mass_allowed=False)
    value = SimpleNamespace(case=case, geometry=produced["geometry"], area=role,
        revision=revision, historical=historical_receipt, measured=measured_receipt,
        parent=dict(complete=True, model_identity=["parent", "2" * 64], pipeline_sha256="1" * 64),
        registration=dict(model_id="parent", model_sha256="2" * 64),
        output=case.output.parent / "area-output", scale_extra={}, plan_extra={})

    def put(name, data):
        path = case.root / (name + ".json")
        # Explicit private fixture mutations do not exercise producer overwrites.
        path.write_text(json.dumps(data), encoding="utf-8")
        return asset(path)

    forbidden = {historical["path"], measured["path"], case.curve["path"],
                 DataReader().json(produced["geometry"])["frame_photo_top_keep"]["path"],
                 case.geometry["raw_source"]["path"]}
    class MetadataOnlyReader(DataReader):
        def path(self, binding):
            assert binding["path"] not in forbidden, "Array/RAW bodies do not belong in area DATA admission"
            assert Path(binding["path"]).suffix.lower() not in (".npz", ".npy", ".arw")
            return super().path(binding)
        def array(self, *args, **kwargs):
            raise AssertionError("No existing array may be decoded by this producer")
        def archive(self, *args, **kwargs):
            raise AssertionError("No existing archive may be decoded by this producer")
    value.reader = MetadataOnlyReader()

    def seal():
        measurement = put("measurement", value.revision)
        original = put("historical_receipt", value.historical)
        value.measured["source_producer_receipt"] = original
        corrected = put("measured_receipt", value.measured)
        value.parent["frames"] = [dict(run="sample", frame_index=1, receipt=original)]
        parent = put("parent_seal", value.parent)
        registration = put("parent_registration", value.registration)
        binding = dict(contract=value.revision["contract"], parent_identity=["parent", "2" * 64],
            parent_seal=parent, parent_registration=registration, measurement=measurement)
        plan = dict(binding=binding, model_identity=["measured", "3" * 64], pipeline_sha256="4" * 64)
        plan.update(value.plan_extra)
        plan_asset = put("scale_plan", plan)
        scale = dict(contract=value.revision["contract"], complete=True,
            source_plan_sha256=plan_asset["sha256"], model_identity=plan["model_identity"],
            pipeline_sha256=plan["pipeline_sha256"], measurement=measurement,
            parent_seal=parent, parent_registration=registration,
            frames=[dict(run="sample", frame_index=1, receipt=corrected)],
            mass_eligible=False, bottom_mass_allowed=False)
        scale.update(value.scale_extra)
        return dict(geometry=value.geometry, area=value.area, measurement=measurement,
            physical_scale_plan=plan_asset, physical_scale_receipt=put("scale_receipt", scale),
            measured_frame_receipt=corrected, historical_frame_receipt=original,
            output=value.output, reader=value.reader)
    value.seal, value.put = seal, put
    value.call = lambda: prepare_measured_area_data(**seal())
    return value


def test_current_grid_pair_preserves_original_role_revision_and_float64_rounding_without_old_raster_reads(area_case):
    # The first 10 columns happen to round identically for both factors. A
    # genuine wider reporting canvas includes column24, which distinguishes
    # them while preserving the same independent projective/origin oracle.
    geometry_args = dict(area_case.case.seal(), phase_shape_hw=[6, 25],
        output=area_case.output.parent / "wide-geometry", reader=DataReader())
    area_case.geometry = prepare_geometry_data(**geometry_args)["geometry"]
    # The shared fixture explicitly reseals its JSON dependencies. Bind the
    # synthetic historical receipt to that exact original row/descriptor graph.
    area_case.historical["binding"]["source"]["rig_geometry"] = deepcopy(area_case.case.geometry)
    args = area_case.seal()
    before = {path: path.read_bytes() for path in area_case.case.root.iterdir()}
    actual = prepare_measured_area_data(**args)
    original = DataReader().array(actual["area"]["historical"])
    measured = DataReader().array(actual["area"]["measured"])
    # Known projective metric and translation: current native x = column-3,
    # old native x = current x-2. This checks explicit origin and new H/canvas.
    expected_row = np.array([.1 / (1 + (column - 5) / 16) ** 3 for column in range(25)], np.float32)
    expected = np.broadcast_to(expected_row, (6, 25))
    np.testing.assert_array_equal(original, expected)
    expected_measured = (expected.astype(np.float64) * area_case.area["factor"]).astype(np.float32)
    np.testing.assert_array_equal(measured, expected_measured)
    assert np.any(measured != expected * area_case.area["factor"]), "Fixture must distinguish selected-frame rounding"
    assert actual["area"]["role"] == "measured_parent"
    assert actual["area"]["revision"] == area_case.revision
    assert actual["area"]["factor"] == area_case.revision["area_factor"]
    for key in ("vertical_scale_independently_verified", "mass_eligible", "bottom_mass_allowed"):
        assert actual["area"]["revision"][key] is False
    proof = DataReader().json(actual["provenance"])
    assert proof["numerical_scale_qa"]["source"] == actual["area"]["historical"]
    assert proof["numerical_scale_qa"]["shape_hw"] == [6, 25]
    assert proof["area"] == actual["area"]
    assert "physical_context_verified" not in proof
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not Path(area_case.area["historical"]["path"]).exists()
    assert not Path(area_case.area["measured"]["path"]).exists()


@pytest.mark.parametrize("change,match", [
    ("selected_role", "measured-parent"), ("wrong_plan_sha", "scale receipt"),
    ("wrong_parent_identity", "registered parent"), ("logical_frame_2", "original source geometry"),
    ("wrong_historical_descriptor", "Area arrays"), ("wrong_measured_descriptor", "Area arrays"),
    ("wrong_factor", "literal factor"), ("width_revision_drift", "measurement revision"),
    ("vertical_true", "width-only"), ("mass_true", "width-only"),
    ("already_scaled_h", "unscaled metric"), ("old_scale_flag", "unscaled metric"),
    ("wrong_fixed_area", "unscaled metric"), ("wrong_photo_source", "original source geometry"),
])
def test_bound_source_scale_and_unscaled_basis_mismatches_refuse_before_output(area_case, change, match):
    if change == "selected_role": area_case.area["role"] = "selected"
    elif change == "wrong_plan_sha": area_case.scale_extra["source_plan_sha256"] = "f" * 64
    elif change == "wrong_parent_identity": area_case.registration["model_sha256"] = "f" * 64
    elif change == "logical_frame_2": area_case.measured["frame_index"] = 2
    elif change == "wrong_historical_descriptor":
        area_case.area["historical"] = dict(area_case.area["historical"], sha256="f" * 64)
    elif change == "wrong_measured_descriptor":
        area_case.area["measured"] = dict(area_case.area["measured"], sha256="f" * 64)
    elif change == "wrong_factor": area_case.area["factor"] = area_case.revision["linear_factor"] ** 2
    elif change == "width_revision_drift": area_case.area["revision"] = dict(area_case.revision, inner_width_mm=901.)
    elif change == "vertical_true": area_case.revision["vertical_scale_independently_verified"] = True
    elif change == "mass_true": area_case.scale_extra["mass_eligible"] = True
    elif change == "already_scaled_h":
        area_case.historical["rig_domain_revision"]["parameters"]["H_frame_native_to_metric_pixel"][0][0] *= .9
    elif change == "old_scale_flag":
        area_case.historical["rig_domain_revision"]["parameters"]["physical_scale_approved"] = True
    elif change == "wrong_fixed_area": area_case.historical["rig_domain_revision"]["fixed_area_mm2"] *= .9
    elif change == "wrong_photo_source": area_case.historical["binding"]["source"]["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match=match):
        area_case.call()
    assert not area_case.output.exists()


@pytest.mark.parametrize("change", ["forged_transport", "new_metric", "qualified_scope", "top_source"])
def test_saved_geometry_is_replayed_against_original_raw_graph_without_curve_reads(area_case, change):
    row = DataReader().json(area_case.geometry)
    if change == "forged_transport": row["coordinate_transport"]["old_native_to_new_native_h"][0][2] += 1
    elif change == "new_metric": row["area_parameters"]["H_ac28_to_metric"][0][0] *= .9
    elif change == "qualified_scope": row["physical_scale_qa_pass"] = True
    elif change == "top_source": row["frame_photo_top_keep"]["phase_or_labels_used"] = True
    area_case.geometry = area_case.put("changed_geometry", row)
    with pytest.raises(ValueError, match="Saved geometry"):
        area_case.call()
    assert not area_case.output.exists()


def test_shared_metadata_admission_retains_all_original_coordinate_fields_without_top_decode(area_case):
    context = _read_geometry_transport(area_case.geometry, reader=area_case.reader)
    assert context["geometry"]["area_parameters"]["physical_scale_approved"] is False
    assert context["row"]["frame_index"] == 1
    assert context["shape"] == [6, 10] and context["origin"] == [-3, -2]
    np.testing.assert_array_equal(context["transform"], area_case.case.transform)
    assert "old_acquisition_collection" in context["inputs"]
    assert "top_curve" not in context["inputs"]


def test_existing_namespace_is_retained_and_bound_rounding_limit_is_used(area_case):
    area_case.output.mkdir()
    marker = area_case.output / "keep"
    marker.write_bytes(b"retained previous attempt")
    with pytest.raises(ValueError, match="fresh positively local"):
        area_case.call()
    assert marker.read_bytes() == b"retained previous attempt"


def test_area_output_cannot_overlap_actual_aliased_metadata_folder(area_case):
    args = area_case.seal()
    logical = args["measurement"]
    original = Path(logical["path"]).read_bytes()
    folder = area_case.output.parent / "aliased_area_metadata"
    folder.mkdir()
    physical = folder / "measurement.json"
    physical.write_bytes(original)
    args["reader"] = DataReader(aliases=[dict(logical=logical, physical=asset(physical))])
    args["output"] = folder / "new_area"
    with pytest.raises(ValueError, match="overlaps its source folders"):
        prepare_measured_area_data(**args)
    assert not args["output"].exists()
    assert physical.read_bytes() == original
    assert Path(logical["path"]).read_bytes() == original
    assert not Path(area_case.area["historical"]["path"]).exists()
    assert not Path(area_case.area["measured"]["path"]).exists()


def test_bound_rounding_tolerance_is_not_replaced_with_a_default(area_case):
    args = area_case.seal()
    # Rebind the source graph rather than alter a previously hashed input file.
    row = DataReader().json(area_case.geometry)
    phase = DataReader().json(row["coordinate_transport"]["inputs"]["phase_configuration"])
    phase["area_parameters"]["maximum_relative_rounding_error"] = 0.
    row["coordinate_transport"]["inputs"]["phase_configuration"] = area_case.put("strict_phase", phase)
    args["geometry"] = area_case.put("strict_geometry", row)
    with pytest.raises(ValueError, match="rounding tolerance"):
        prepare_measured_area_data(**args)
    assert not area_case.output.exists()
