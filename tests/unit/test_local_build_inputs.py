"""Finite original-reader fixtures for the ordinary local Build adapter.

The five image archives and owned-save records use original DarSIA writers.
Producer-contract JSON and RAW bytes are TESTONLY data, not a camera decode,
production Build, physical approval, calibration or scientific release.
No test replaces load_build_physical_context or grants an eligibility flag.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import darsia
import numpy as np
import pytest

from darsia.fluidflower import round5_geometry
from darsia.fluidflower.mass.native_optical_source import load_build_physical_context
from darsia.fluidflower.rig_raw_build import _descriptor, _grid
from .test_native_optical_source import saved, _json
from .test_staged_data import phase1_data


@pytest.fixture
def local_build_sources(tmp_path):
    """Real 3x3 saved fields with distinct label IDs and physical facies IDs."""
    fixture = saved.__wrapped__(tmp_path, native_parameters=dict(source=dict(
        homography_match_atol=1e-8, homography_min_scale=1e-12,
        symbols=dict(reference_to_frame_h_key="H_reference_to_frame"))))
    return _local_build_sources(fixture, tmp_path)


def _local_build_sources(saved, tmp_path):
    labels = np.array([[2, 3, 4], [2, 3, 4], [2, 3, 3]], np.uint8)
    facies = np.where(labels == 2, 0, np.where(labels == 3, 1, 2)).astype(np.uint8)
    porosity = np.where(labels == 2, 1., np.where(labels == 3, .3, 0.)).astype(np.float64)
    depth = np.array([[.01, .011, .012], [.012, .013, .014], [.014, .015, .016]], np.float64)
    scalar = dict(space_dim=2, indexing="ij", dimensions=[.003, .003], origin=[0., .003])
    images = dict(
        shape_corrected_baseline=darsia.OpticalImage(np.full((3, 3, 3), .5, np.float32),
            color_space="RGB", dimensions=[.003, .003], origin=[0., .003]),
        labels=darsia.ScalarImage(labels, **scalar),
        facies=darsia.ScalarImage(facies, **scalar),
        depth=darsia.ScalarImage(depth, **scalar),
        porosity=darsia.ScalarImage(porosity, **scalar))
    for name, image in images.items():
        image.save(saved.destination / (name + ".npz"))
    saved.images = images
    saved.certificate["corrected_baseline"]["grid"] = _grid(images["shape_corrected_baseline"])
    identity = np.eye(3).tolist()
    saved.certificate["shape_hw"] = [3, 3]
    saved.certificate["H_rig_corrected_native_to_baseline_camera_u"] = identity
    raw = deepcopy(saved.certificate["inputs"]["raw"])
    saved.certificate["renderer"] = dict(raw_acquisition=raw, camera_u_wh=[3, 3],
        native_shape_hw=[3, 3], native_to_camera_u_h=identity)
    inputs = Path(saved.adopted["path"]).parent
    table = inputs / "facies.csv"
    table.write_text("id,name,analysis_role,porosity,permeability\n"
                     "0,Freewater,free_water,1,0\n"
                     "1,F,porous_medium,0.3,1\n"
                     "2,Fixture,excluded,0,0\n", encoding="utf-8")
    adopted = dict(revision_id="saved-revision", configuration=dict(
        grouping_mode="shared_facies", regions=[
            dict(label_id=2, facies_id=0, name="Freewater", role="free_water", enabled=True),
            dict(label_id=3, facies_id=1, name="F", role="porous_medium", enabled=True),
            dict(label_id=4, facies_id=2, name="Fixture", role="excluded", enabled=True)]),
        artifacts=dict(source_mode_files=dict(facies_properties=_descriptor(table))))
    saved.adopted = _json(Path(saved.adopted["path"]), adopted)
    saved.plan["adopted_revision"] = saved.adopted
    operation = saved.seal()
    saved.bound = operation
    context = load_build_physical_context(operation)
    metric_frame = dict(H_camera_to_mm=[[1., 0., .5], [0., -1., 2.5], [0., 0., 1.]],
                        sand_area_factor14=1., explicit_certificate={"fixture": "TESTONLY"})
    measured_model = _json(inputs / "measured_model.json", metric_frame)
    edges = _json(inputs / "edges.json", {"fixture": "TESTONLY explicit metric input"})
    model = round5_geometry.reference_model(context.run, metric_frame,
        reference_image=raw, camera_profile=dict(camera_wh=[3, 3], profile_id="camera"),
        source_model=measured_model, edge_review=edges,
        baseline_configuration=saved.result["configuration_files"][0],
        baseline_native_renderer=saved.result["baseline_renderer"])
    metric_model = _json(inputs / "metric_model.json", model)
    return SimpleNamespace(saved=saved, context=context, operation_result=operation,
        metric_model=metric_model, model=model, metric_frame=metric_frame,
        adopted=adopted, table=_descriptor(table), labels=labels, facies=facies,
        porosity=porosity, depth=depth, raw=raw, inputs=inputs, root=tmp_path)


def test_saved_local_role_fixture_uses_original_reader_without_invented_eligibility(local_build_sources):
    case = local_build_sources
    actual = load_build_physical_context(case.operation_result)
    np.testing.assert_array_equal(actual.labels, case.labels)
    np.testing.assert_array_equal(actual.facies, case.facies)
    np.testing.assert_array_equal(actual.porosity, case.porosity)
    np.testing.assert_array_equal(actual.depth_m, case.depth)
    assert actual.facies[0, 0] == 0 and actual.labels[0, 0] == 2
    assert actual.porosity[0, 0] == 1 and actual.porosity[0, 2] == 0
    assert "physical_context_verified" not in actual.provenance
    assert "mass_eligible_mask" not in actual.provenance
    assert actual.provenance["operation_result"] == case.operation_result
    assert not actual.labels.flags.writeable and not actual.porosity.flags.writeable


def test_original_metric_model_is_bound_to_the_actual_saved_build_grid(local_build_sources):
    case = local_build_sources
    assert case.model["baseline_native_grid"]["renderer"] == case.saved.result["baseline_renderer"]
    assert case.model["reference_image"] == case.raw
    np.testing.assert_array_equal(case.model["H_camera_u_to_metric_mm"],
                                  case.metric_frame["H_camera_to_mm"])
    assert case.model["baseline_native_grid"]["shape_hw"] == [3, 3]


@pytest.fixture
def local_build_case(local_build_sources, phase1_data):
    """Matched original receipt shapes and actual arrays for local input joins.

    Receipt DATA is deliberately finite TESTONLY producer-contract metadata.
    The fixture does not execute a RAW processor or pretend its arrays came
    from classification. Every file descriptor is computed from actual bytes.
    """
    from darsia.fluidflower.geometry.context import RigFrameContext
    from darsia.fluidflower.native.raw_rig_source import CONTRACT, FRAME_CONTRACT
    from darsia.fluidflower.pipeline.local_build import LocalBuildInputs
    case = local_build_sources
    config = deepcopy(phase1_data)
    config["run"] = case.context.run
    config["native"]["configuration"] = deepcopy(case.context.provenance["native_configuration"])
    gp = dict(determinant_floor=1e-12, area_horizon_floor=1e-10,
              point_horizon_floor=1e-10)
    config["phase"]["configuration"] = _json(case.inputs / "phase.json",
        dict(geometry_parameters=gp, parameters={"fixture": "TESTONLY"}))
    config["statistics"]["shared"].update(
        porosity_by_facies={"0": 1., "1": .3, "2": 0.},
        swi_by_facies={"1": .2}, unquantified_facies=[0, 2])
    spec = config["frames"][0]
    spec["source"].update(run=case.context.run, source_sha256=case.raw["sha256"],
        source_binding=dict(raw_acquisition=deepcopy(case.raw), source_sha256=case.raw["sha256"]))
    identity = np.eye(3).tolist()
    geometry_row = dict(contract=FRAME_CONTRACT, run=case.context.run,
        acquisition=deepcopy(case.raw), baseline_to_native_h=identity,
        reference_to_native_h=identity)
    geometry = _json(case.inputs / "native_frame.json", geometry_row)
    spec["geometry"] = _json(case.inputs / "phase_geometry.json", geometry_row)
    raw_binding = _json(case.inputs / "raw_binding.json", dict(
        fixture="TESTONLY source identity", baseline=case.raw,
        profile=case.context.provenance["native_configuration"]))
    spec["native_task"] = dict(kind="raw_rig", geometry=geometry,
                               processor=dict(raw_binding=raw_binding))
    spec["phase_roles"].update(current_h=identity, reference_to_current_h=identity,
        retained_source_cache=None, native_row=None, encoding_contract=None)
    audit = dict(contract=CONTRACT, jpeg_inputs=False, source_identity=case.raw["sha256"],
        sources=dict(camera_arw=deepcopy(case.raw)),
        processor=dict(models={}, execution_identity={"fixture": "TESTONLY"},
            profile_id="camera", native_shape_hw=[3, 3], source_binding=raw_binding),
        renderer=dict(native_to_camera_u_h=identity), H_v5_B_to_frame_native=identity)
    native_parameters = case.context.provenance["native_configuration"]
    import json
    reference_key = json.loads(Path(native_parameters["path"]).read_bytes())["source"]["symbols"]["reference_to_frame_h_key"]
    audit[reference_key] = identity
    audit_pin = _json(case.inputs / "native_audit.json", audit)
    log = np.full((3, 3, 3), np.log(np.float32(.8 * 65535)), np.float32)
    valid = np.ones((3, 3), bool)
    log_path, valid_path = case.inputs / "native_log.npy", case.inputs / "native_valid.npy"
    np.save(log_path, log, allow_pickle=False)
    np.save(valid_path, valid, allow_pickle=False)
    full = dict(complete=True, source_sha256=case.raw["sha256"], phase_shape_hw=[3, 3],
        native_origin_xy=[0, 0], native_centre=dict(native_shape_hw=[3, 3],
            log_linear_rgb=_descriptor(log_path), valid_mask=_descriptor(valid_path)),
        ring_windows=[], binding=dict(geometry={"H": identity}, producer_audit=audit_pin))
    area = np.ones((3, 3), np.float32)
    zero = np.zeros((3, 3), bool)
    context = RigFrameContext(rig_domain=valid.copy(), modelled=valid.copy(),
        filled_observed=zero.copy(), filled_unobserved=zero.copy(),
        hardware_excluded=zero.copy(), excluded_by_name={}, origin_xy=(0, 0),
        native_shape_hw=(3, 3), content_box_xywh=(0, 0, 3, 3),
        pixel_area_mm2=area, parameters=gp)
    native = dict(audit=audit, native_geometry_row=geometry_row, geometry_row=geometry_row, context=context,
        full_log=log, record=dict(complete=True, full=full,
            source_sha256=case.raw["sha256"], processor_kind="raw_rig", audit=audit_pin))
    frame = dict(context=context, full_log=log, source=deepcopy(spec["source"]),
        current_h=np.eye(3), reference_to_current_h=np.eye(3), pixel_area_mm2=area,
        original_area=area, scientific={"full": full})
    bridge = _json(case.inputs / "bridge.json", dict(format="darsia-build-native-bridge-v1",
        operation_result=case.operation_result,
        native_configuration=case.context.provenance["native_configuration"],
        saved_build_to_camera_u_h=identity, build_to_baseline_native_h=identity))
    receipt = dict(format="darsia-build-native-frame-v1", bridge=bridge,
        physical_context=deepcopy(case.context.provenance), native_frame=geometry,
        raw_source_binding=raw_binding, correction=deepcopy(audit),
        build_to_frame_native_h=identity)
    receipt_path = case.inputs / "frame.json"
    receipts = {spec["case_id"]: _json(receipt_path, receipt)}

    def make_adapter(configuration=None, *, frame_receipts=None, metric_model=None):
        return LocalBuildInputs(config if configuration is None else configuration,
            operation_result=case.operation_result,
            metric_model=case.metric_model if metric_model is None else metric_model,
            frame_receipts=receipts if frame_receipts is None else frame_receipts)

    return SimpleNamespace(sources=case, configuration=config, spec=spec, native=native,
        phase_frame=frame, frame_receipts=receipts, receipt=receipt,
        receipt_path=receipt_path, make_adapter=make_adapter, full=full,
        geometry_parameters=gp)


def _prepared_case(case):
    """Supply typed test phase buffers, without invoking or replacing inference."""
    adapter = case.make_adapter()
    config = adapter.bind_configuration(case.configuration)
    mass = dict(config["statistics"], run=config["run"], frames=config["frames"],
                data_aliases=config.get("data_aliases", ()))
    adapter.validate_mass_config(mass, statistics_only=True)
    frame = adapter.check_phase(case.spec, case.native, case.phase_frame)
    domain = frame["context"].rig_domain
    labels = np.where(domain, 1, 255).astype(np.uint8)
    inferred = dict(labels=labels, pixel_area_mm2=frame["pixel_area_mm2"].copy(),
        classes=dict(rig_domain=domain.copy(), modelled=domain.copy(),
                     filled_observed=np.zeros_like(domain), filled_unobserved=np.zeros_like(domain)),
        qa={})
    adapter.check_inferred(case.spec, frame, inferred)
    spec = dict(case.spec["mass_spec"], case_id=case.spec["case_id"],
        frame_index=case.spec["frame_index"], elapsed_h=case.spec["elapsed_h"],
        current=deepcopy(case.full), source_binding=deepcopy(case.spec["source"]["source_binding"]))
    return adapter, config, frame, inferred, spec


def test_local_binding_reports_actual_build_and_preserves_scientific_recipe(local_build_case):
    case = local_build_case
    original = deepcopy(case.configuration)
    adapter = case.make_adapter()
    actual = adapter.bind_configuration(case.configuration)
    expected = deepcopy(original)
    expected["statistics"]["physical_correction"] = case.sources.operation_result
    assert actual == expected and case.configuration == original
    assert actual["statistics"]["shared"]["swi_by_facies"] == {"1": .2}
    assert actual["statistics"]["shared"]["porosity_by_facies"] == {"0": 1., "1": .3, "2": 0.}


def test_actual_local_fields_keep_freewater_zero_and_zero_porosity_with_full_report(local_build_case):
    from darsia.fluidflower.mass.production_precision import PixelMath
    case = local_build_case
    adapter, _, frame, inferred, spec = _prepared_case(case)
    physical = adapter.physical_fields(spec, (3, 3), PixelMath("float32"))
    report = adapter.frame_context(spec, inferred["labels"], inferred["classes"], inferred["qa"])
    np.testing.assert_array_equal(physical["facies"], case.sources.facies)
    np.testing.assert_array_equal(physical["porosity"], case.sources.porosity.astype(np.float32))
    assert np.isnan(physical["swi"][case.sources.facies == 0]).all()
    np.testing.assert_array_equal(physical["swi"][case.sources.facies == 1], np.float32(.2))
    np.testing.assert_array_equal(report["reporting_domain"], np.ones((3, 3), bool))
    np.testing.assert_array_equal(report["analysis_domain"], case.sources.labels != 4)
    np.testing.assert_array_equal(report["excluded_masks"]["material_2"], case.sources.labels == 4)
    assert report["material_supported_mask"].all()
    assert not physical["physical_supported_mask"][case.sources.labels == 4].any()
    assert physical["physical_supported_mask"][case.sources.labels == 2].all()
    assert report["source_binding"]["legacy_physical_correction_applied"] is False
    assert report["source_binding"]["operation_result"] == case.sources.operation_result
    # frame_context must retain the same pending tuple until metric coordinates
    # are consumed by the one original load_frame call.
    x, y = adapter.coordinates(spec, (3, 3))
    assert x.dtype == y.dtype == np.float32
    np.testing.assert_array_equal(x, np.broadcast_to(np.array([.0005, .0015, .0025], np.float32), (3, 3)))
    np.testing.assert_array_equal(y, np.broadcast_to(np.array([.0025, .0015, .0005], np.float32)[:, None], (3, 3)))
    np.testing.assert_array_equal(frame["pixel_area_mm2"], np.ones((3, 3), np.float32))


@pytest.mark.parametrize("change", ["build", "geometry", "raw_binding", "processor", "renderer"])
def test_rebound_frame_receipt_cannot_replace_actual_upstream_inputs(local_build_case, change):
    case = local_build_case
    receipt = deepcopy(case.receipt)
    if change == "build":
        receipt["physical_context"]["operation_result"]["sha256"] = "f" * 64
    elif change == "geometry":
        receipt["native_frame"]["sha256"] = "f" * 64
    elif change == "raw_binding":
        receipt["raw_source_binding"]["sha256"] = "f" * 64
    elif change == "processor":
        receipt["correction"]["processor"]["profile_id"] = "other-camera"
    else:
        receipt["correction"]["renderer"]["native_to_camera_u_h"][0][2] = 1.
    pin = _json(case.sources.inputs / "changed_frame.json", receipt)
    adapter = case.make_adapter(frame_receipts={case.spec["case_id"]: pin})
    with pytest.raises(ValueError, match="differs|differ"):
        adapter.check_phase(case.spec, case.native, case.phase_frame)


def test_actual_metric_join_cannot_substitute_another_saved_grid_transform(local_build_case):
    case = local_build_case
    model = deepcopy(case.sources.model)
    model["baseline_native_grid"]["H_rig_corrected_native_to_baseline_camera_u"][0][2] = 1.
    changed = _json(case.sources.inputs / "changed_metric.json", model)
    with pytest.raises(ValueError, match="transforms differ"):
        case.make_adapter(metric_model=changed)


def test_changed_real_physical_bytes_fail_original_reader_before_local_handoff(local_build_case):
    case = local_build_case
    (case.sources.saved.destination / "porosity.npz").write_bytes(b"changed fixture archive")
    with pytest.raises(ValueError, match="size differs|SHA or identity differs"):
        case.make_adapter()


def test_inference_cannot_fill_the_actual_excluded_material(local_build_case):
    case = local_build_case
    adapter = case.make_adapter()
    frame = adapter.check_phase(case.spec, case.native, case.phase_frame)
    inferred = dict(labels=np.ones((3, 3), np.uint8),
                    pixel_area_mm2=frame["pixel_area_mm2"].copy(), classes={}, qa={})
    with pytest.raises(ValueError, match="explicit domain"):
        adapter.check_inferred(case.spec, frame, inferred)


def test_phase_qa_cannot_replace_the_actual_local_source_join(local_build_case):
    from darsia.fluidflower.mass.production_precision import PixelMath
    adapter, _, _, inferred, spec = _prepared_case(local_build_case)
    adapter.physical_fields(spec, (3, 3), PixelMath("float32"))
    changed = deepcopy(inferred["qa"])
    changed["local_build"]["operation_result"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="labels/domain/QA differ"):
        adapter.frame_context(spec, inferred["labels"], inferred["classes"], changed)
