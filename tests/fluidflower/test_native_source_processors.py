"""Small local source receipts exercise processing and admission without camera/NAS IO."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.native import source


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return digest(path)


@pytest.fixture
def local_source(tmp_path, monkeypatch):
    shape = (5, 6)
    identity = np.eye(3).tolist()
    shifted = [[1., 0., 1.], [0., 1., 0.], [0., 0., 1.]]
    support = dict(keep_policy="keep", fill_policy="fill", keep_contract="photo-keep",
        fill_contract="photo-fill", hole_max_area_px=2, connectivity=4, guard_px=0,
        vertical_gap_max_px=1, vertical_gap_rule="bounded", grid_shape_hw=list(shape),
        grid_archive_key="sand", cell_to_reference_native=identity, homography_epsilon=1e-12)
    parameters = dict(source=dict(asset_aliases={}, source_id_pattern="fixture", native_shape_hw=list(shape),
        content_box_xywh=[0, 0, 6, 5], homography_min_scale=1e-12,
        receipt_homography_min_scale=1e-9, homography_match_atol=1e-7,
        cell_scale=1, mask_guard_px=0, cohort_contract="photo-cohort",
        photo_contract="photo-transport", photo_core_contract="photo-core", photo_scope="photo-only",
        processor_contract="camera-source", rig_contract="rig-source", signal_denominator=100,
        policy_ids=dict(keep="keep", filled_support="fill", raw_display="raw_log",
            natural_display="natural", photographic_display="photographic"),
        default_policies=dict(enclosed_hole_policy="keep", display_mode="raw_log", isolated_invalid_policy="keep"),
        source_policies=dict(enclosed_hole_policy="keep", display_mode="raw_log", isolated_invalid_policy="keep"),
        renderer_source_ids={"fixture": "sensor"}, profiles={"fixture": "camera"},
        symbols=dict(prepared_source_id="another-source", prepared_source_label="prepared",
            calibration_contract="calibration", calibration_contract_prefix="calibration-",
            baseline_assets_contract="baseline-assets", reference_to_frame_h_key="H_reference_to_frame",
            reference_to_timeline_h_key="H_reference_to_timeline")),
        support=support, photometry={}, registration={},
        photographic_display=dict(modes=["raw_log", "photographic"], contract_metadata={"contract": "photograph"}),
        display_modes=dict(contract_metadata={"contract": "display-modes"}),
        isolated_invalid=dict(keep_policy="keep", policy="repair"),
        sidecar=dict(raw_detail_floor=2, raw_detail_ceiling=100, raw_detail_exponent=1., raw_detail_invalid=1.))
    files = {}
    for key in ("camera", "raw", "native", "timeline_raw"):
        files[key] = tmp_path / (key + ".bin")
        files[key].write_bytes(("local-" + key).encode())
    cert = tmp_path / "registration.json"
    write_json(cert, {"sources": {"review_native_jpeg_sha256": digest(files["native"])},
                      "review_to_camera_h": identity})
    mask = tmp_path / "sand.npz"
    np.savez(mask, sand=np.ones(shape, bool))
    temporal = tmp_path / "temporal.json"
    write_json(temporal, {"frames": [{"frame_index": 0, "H_timeline0_to_frame_native": identity,
                                     "temporal_registration_pass": True}]})
    original = dict(frame_path=str(files["native"]), native_jpeg_sha256=digest(files["native"]),
                    source_path=str(files["camera"]), source_sha256=digest(files["camera"]))
    manifest = tmp_path / "source.json"
    source_sha = write_json(manifest, dict(run="fixture", frame_count=1,
        analysis_image_shape_hw=list(shape), frame_sources=[original]))
    photo = tmp_path / "photo.json"
    write_json(photo, dict(contract="photo-core", source_manifest_sha256=source_sha,
        B_native_review_jpeg_sha256=digest(files["native"]), B_arw_sha256=digest(files["raw"]),
        B_camera_jpeg_sha256=digest(files["camera"]), B_registration_certificate_sha256=digest(cert),
        mask_npz_sha256=digest(mask), mask_npz_path=str(mask), photo_gate_pass=True,
        photo_sand_qa_scope="photo-only", per_frame_transport_qa_pass=False,
        training_label_eligibility_certified=False, phase_map_eligibility_certified=False,
        rig_metal_certified=False, dynamic_overflow_certified=False, mass_eligible=False))
    receipt = dict(contract="photo-transport", original_source_manifest_path=str(manifest),
        original_source_manifest_sha256=source_sha, frame_index=0, native_shape_hw=list(shape),
        validation_only=True, mass_eligible=False, photo_sand_transport_qa_pass=True,
        temporal_registration_pass=True, optical_geometry_qa_pass=True,
        photo_sand_qa_scope="photo-only", photo_sand_qa_pass=True, fixed_native_mask_erosion_px=1,
        training_label_eligibility_certified=False, phase_map_eligibility_certified=False,
        rig_metal_certified=False, dynamic_overflow_certified=False,
        native_review_jpeg_path=str(files["native"]), native_review_jpeg_sha256=digest(files["native"]),
        camera_jpeg_path=str(files["camera"]), camera_jpeg_sha256=digest(files["camera"]),
        v5_B_native_jpeg_path=str(files["native"]), v5_B_native_jpeg_sha256=digest(files["native"]),
        v5_B_registration_certificate_path=str(cert), v5_B_registration_certificate_sha256=digest(cert),
        v5_B_arw_path=str(files["raw"]), v5_B_arw_sha256=digest(files["raw"]),
        v5_B_camera_jpeg_path=str(files["camera"]), v5_B_camera_jpeg_sha256=digest(files["camera"]),
        temporal_geometry_receipt_path=str(temporal), temporal_geometry_receipt_sha256=digest(temporal),
        photo_sand_reference_receipt_path=str(photo), photo_sand_reference_receipt_sha256=digest(photo),
        photo_sand_reference_mask_path=str(mask), photo_sand_reference_mask_sha256=digest(mask),
        H_timeline0_to_frame_native=identity, H_v5_B_to_timeline0_native=shifted,
        H_v5_B_to_frame_native=shifted, H_reference_to_timeline=shifted, H_reference_to_frame=shifted)
    transport = tmp_path / "transport.json"
    write_json(transport, receipt)
    row = dict(run="fixture", source_manifest_path=str(manifest), source_manifest_sha256=source_sha,
        frames=[dict(frame_index=0, receipt_path=str(transport), receipt_sha256=digest(transport))])
    aggregate = tmp_path / "aggregate.json"
    write_json(aggregate, {"rows": [row]})
    baseline_rows = []
    for run in ("fixture", "virtual_reference"):
        baseline = tmp_path / (run + "_baseline.npz")
        baseline.write_bytes(b"decoder-test-bound-fit")
        sidecar = baseline.with_suffix(".json")
        write_json(sidecar, {"source": dict(source_manifest_sha256=source_sha,
            source_arw_sha256=digest(files["raw"]), source_review_sha256=digest(files["native"]))})
        baseline_rows.append(dict(run=run, baseline_path=str(baseline),
            baseline_sha256=digest(baseline), sidecar_sha256=digest(sidecar)))
    calibration = tmp_path / "calibration.json"
    write_json(calibration, dict(contract="calibration", results=baseline_rows))
    native = tmp_path / "native.json"
    marker = dict(contract="photo-cohort", original_source_manifest_path=str(manifest),
        original_source_manifest_sha256=source_sha, frames=row["frames"])
    native_doc = dict(run="fixture", frame_count=1, v5_track1_photo_geometry=marker,
        frame_sources=[dict(v5_raw_path=str(files["timeline_raw"]),
                            v5_raw_sha256=digest(files["timeline_raw"]))],
        v5_photometry=dict(source_manifest=str(manifest), source_manifest_sha256=source_sha,
            processor=dict(calibration_manifest_path=str(calibration), calibration_manifest_sha256=digest(calibration),
                photo_source_aggregate_path=str(aggregate), photo_source_aggregate_sha256=digest(aggregate))))
    write_json(native, native_doc)
    grid = SimpleNamespace(shape_hw=shape)
    fit = SimpleNamespace(CB=np.zeros((3, 24), np.float32))
    monkeypatch.setattr(source.obs, "load_baseline", lambda path, *, parameters: (fit, grid))
    renderer_calls, correction_calls = [], []
    def renderer(arw, camera, review, **kwargs):
        renderer_calls.append((arw, camera, review, kwargs))
        return np.full((3, *shape), 10, np.float32), {"review_to_camera_h": identity, "local_decoder": True}
    def correct(linear, fit, reference, grid, homography, safe, **kwargs):
        correction_calls.append((homography.copy(), safe.copy(), kwargs))
        result = np.log(linear).astype(np.float32)
        result[:, 2, 3] = np.nan
        return result, {"exact_baseline": kwargs["exact_baseline_source"]}
    monkeypatch.setattr(source, "registered_edge_aware_sensor_linear", renderer)
    monkeypatch.setattr(source, "correct_frame_native", correct)
    required = ["processor_module_sha256", "renderer_module_sha256", "photometry_module_sha256", "pilot_math_module_sha256"]
    parameters["source"]["required_execution_identity_keys"] = required
    parameters["source"]["execution_identity"] = dict(zip(required,
        [digest(source.__file__), digest(renderer.__code__.co_filename), digest(source.obs.__file__), digest(correct.__code__.co_filename)]))
    return SimpleNamespace(parameters=parameters, files=files, manifest=manifest, native=native,
        native_doc=native_doc, source_sha=source_sha, row=row, aggregate=aggregate,
        calibration=calibration, transport=transport, renderer_calls=renderer_calls,
        correction_calls=correction_calls, shifted=shifted, shape=shape)


def processor(f):
    return source.SourceProcessor(f.row, calibration_manifest_path=f.calibration,
                                 photo_aggregate_path=f.aggregate, parameters=f.parameters)


def test_photo_cohort_constructs_exact_timeline_and_rejects_wrong_marker(local_source):
    f = local_source
    cohort = source.PhotoCohort(f.native, parameters=f.parameters)
    assert cohort.shape_hw == f.shape and cohort.source_manifest_sha256 == f.source_sha
    broken = deepcopy(f.native_doc)
    broken["v5_track1_photo_geometry"]["contract"] = "different"
    write_json(f.native, broken)
    with pytest.raises(ValueError, match="lacks its photo-only cohort"):
        source.PhotoCohort(f.native, parameters=f.parameters)


def test_frame_and_photo_live_verify_sources_and_transport_support(local_source):
    f = local_source
    cohort = source.PhotoCohort(f.native, parameters=f.parameters)
    frame = cohort.frame(0)
    assert frame["camera_jpeg_sha256"] == digest(f.files["camera"])
    safe, complement, receipt = cohort.photo(0)
    expected = np.ones(f.shape, bool)
    expected[:, 0] = False  # One-pixel reference-to-frame translation.
    assert np.array_equal(safe, expected) and np.array_equal(complement, ~expected)
    assert receipt["source_transport_receipt_sha256"] == digest(f.transport)
    with pytest.raises(ValueError, match="outside its source timeline"):
        cohort.frame(1)
    f.files["camera"].write_bytes(b"changed acquisition")
    with pytest.raises(ValueError, match="camera JPEG acquisition changed"):
        cohort.photo(0)


def test_bind_photo_marker_checks_complete_ordered_timeline(local_source):
    f = local_source
    marker = source.bind_photo_marker(f.manifest, f.row, parameters=f.parameters)
    assert marker["original_source_manifest_sha256"] == f.source_sha
    assert marker["frames"] == f.row["frames"]
    broken = deepcopy(f.row)
    broken["frames"][0]["frame_index"] = 1
    with pytest.raises(ValueError, match="does not cover this exact source revision"):
        source.bind_photo_marker(f.manifest, broken, parameters=f.parameters)


def test_source_constructor_identity_is_bound_and_defensive(local_source):
    f = local_source
    p = processor(f)
    identity = p.identity()
    assert identity["calibration_manifest_sha256"] == digest(f.calibration)
    assert identity["photo_source_aggregate_sha256"] == digest(f.aggregate)
    assert identity["scientific_mass_eligible"] is False
    identity["contract"] = "changed caller copy"
    assert p.identity()["contract"] == "camera-source"
    write_json(f.calibration, {"contract": "another"})
    with pytest.raises(ValueError, match="Not a V4-R5 calibration manifest"):
        processor(f)


def test_source_correction_preserves_signal_masks_and_rejects_geometry(local_source):
    f = local_source
    p = processor(f)
    kwargs = dict(run="fixture", frame_id="pre_injection_reference", camera_jpeg=f.files["camera"],
        camera_arw=f.files["raw"], review_jpeg=f.files["native"], source_manifest_sha256=f.source_sha,
        content_shape_hw=f.shape, content_box_xywh=(0, 0, 6, 5))
    result = p.correct(**kwargs)
    assert result.log_linear_rgb.dtype == np.float32
    assert result.log_linear_rgb.shape == (3, *f.shape)
    assert not result.required_valid_mask[2, 3] and result.required_valid_mask.sum() == 29
    assert np.isnan(result.log_linear_rgb[:, 2, 3]).all()
    assert np.all(result.log_linear_rgb[:, result.required_valid_mask] == np.float32(np.log(10)))
    assert np.array_equal(f.correction_calls[0][0], np.eye(3))
    assert f.correction_calls[0][2]["exact_baseline_source"] is True
    assert result.certificate["source_safe_unresolved_pixels"] == 1
    with pytest.raises(ValueError, match="source/geometry binding changed"):
        p.correct(**dict(kwargs, content_shape_hw=(6, 5)))
    assert len(f.renderer_calls) == 1


def test_rig_constructor_and_parent_geometry_preserve_baseline_role(local_source):
    f = local_source
    rig = source.RigSourceProcessor(f.native, parameters=f.parameters)
    assert rig.identity["calibration_baseline_binding"]["verified"] is True
    assert rig.identity["mass_eligible"] is False
    baseline = rig.parent_geometry()
    timeline = rig.parent_geometry(0)
    assert baseline["camera_arw_path"] == str(f.files["raw"])
    assert timeline["camera_arw_path"] == str(f.files["timeline_raw"])
    assert np.array_equal(baseline["H_reference_to_frame"], np.eye(3))
    assert timeline["H_reference_to_frame"] == f.shifted
    with pytest.raises(ValueError, match="outside its source timeline"):
        rig.parent_geometry(1)
    f.parameters["source"]["execution_identity"]["renderer_module_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="execution binding: renderer_module_sha256"):
        source.RigSourceProcessor(f.native, parameters=f.parameters)


def test_rig_correction_checks_source_and_distinct_reuse_roles(local_source):
    f = local_source
    rig = source.RigSourceProcessor(f.native, parameters=f.parameters)
    geometry = rig.parent_geometry()
    values, audit = rig.correct(geometry)
    assert values.shape == (3, *f.shape) and values.dtype == np.float32
    assert audit["exact_B"] is True and audit["full_native_corrected_finite_pixels"] == 29
    assert audit["parent_cache"] is None and audit["training"] is False
    assert audit["sources"]["camera_arw"]["sha256"] == digest(f.files["raw"])
    with pytest.raises(ValueError, match="distinct source bindings"):
        rig.correct(geometry, parent_frame_index=0, parent_reference=True)
    f.files["raw"].write_bytes(b"changed RAW")
    with pytest.raises(ValueError, match="differs from its source receipt: camera_arw"):
        rig.correct(geometry)
    assert len(f.renderer_calls) == 1


def test_display_contract_dispatches_real_renderer_and_requires_metadata(local_source):
    f = local_source
    assert source.output_display_contract("photographic", parameters=f.parameters) == {"contract": "photograph"}
    natural = source.output_display_contract("natural", parameters=f.parameters)
    assert natural["contract"] == "display-modes"
    assert len(natural["renderer_code_sha256"]) == 64
    assert source.output_display_contract("raw_log", parameters=f.parameters) == {"contract": "existing-fixed-log-camera-channel-view"}
    del f.parameters["display_modes"]["contract_metadata"]
    with pytest.raises(KeyError, match="contract_metadata"):
        source.output_display_contract("natural", parameters=f.parameters)
