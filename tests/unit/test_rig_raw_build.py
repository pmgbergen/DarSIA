"""Synthetic fresh-build joins; no real RAW decode, registration or rig build."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import darsia
from darsia.fluidflower import rig_raw_build as build
from darsia.fluidflower.native import raw_positioning, registration


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "baseline.ARW"
    source.write_bytes(b"synthetic RAW bytes")
    configuration = tmp_path / "run.toml"
    configuration.write_text("# exact operation input\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    review = dict(minimum_matches=4, minimum_inlier_fraction=0.8,
                  heldout_zones=[["sand", [0, 0, 2, 1], 1]],
                  heldout_minimum_matches=2, heldout_min_correlation=0.7)
    parameters = dict(common=dict(profiles=dict(camera={}), sensor=dict(camera_wh=[3, 2]),
                                  review=review, matching={}))
    profile.write_text(json.dumps(dict(registration=parameters,
        source=dict(profiles=dict(sample="camera")))), encoding="utf-8")
    camera = dict(configuration=str(profile), sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                  profile_id="camera", run_id="sample")
    monkeypatch.setattr(build, "_source_snapshot", lambda: [build._descriptor(Path(build.__file__))])
    monkeypatch.setattr(build, "_runtime_identity", lambda: dict(kind="synthetic test runtime"))
    manifest = tmp_path / "series.toml"
    manifest.write_text("# synthetic series context\n", encoding="utf-8")
    output_root = tmp_path / "products"
    output_root.mkdir()
    destination = output_root / "fresh_rig"
    request = dict(format=build.BASIS_REQUEST_FORMAT, run_id="sample",
        configuration_files=list(build.capture_configuration_sources(configuration)),
        series_context=dict(root=str(tmp_path), manifest=build._descriptor(manifest)),
        baseline=build._descriptor(source), profile=build._descriptor(profile),
        operation_overrides={"data.raw_camera": camera}, protocol_inputs=[],
        output=dict(destination=str(destination), intended_root=str(output_root),
                    forbidden_raw_roots=[str(raw)], maximum_output_bytes=10_000_000,
                    maximum_file_bytes=2_000_000), shape_basis_certificate=None)
    request_path = tmp_path / "basis-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    request_pin = build._descriptor(request_path)
    context = build.begin_raw_rig_build(
        configuration, source, camera, destination,
        configuration_sources=build.capture_configuration_sources(configuration),
        request_binding=request_pin, artifact_kind="raw_shape_basis")
    linear = np.arange(18, dtype=np.float32).reshape(3, 2, 3) * 10
    sensor = SimpleNamespace(metadata=dict(white_level=1100, black_level=100, orientation=1))
    monkeypatch.setattr(raw_positioning, "read_sensor", lambda *a, **k: sensor)
    monkeypatch.setattr(raw_positioning, "camera_u", lambda *a, **k: linear.copy())
    image = darsia.OpticalImage(np.moveaxis(linear, 0, -1) / np.float32(1000), color_space="RGB").img_as(float)
    correction = darsia.TypeCorrection(np.float32)
    rig = SimpleNamespace(
        _raw_build_id=context.operation_id, raw_camera=camera,
        raw_baseline_sha256=context.raw_source["sha256"], baseline_path=source,
        shape_corrected_baseline=image, shape_corrections=[correction],
        _raw_setup_input_metadata=image.metadata(),
    )
    context.begin_setup(rig, destination / "figures")
    context.output.begin_save(rig, destination)
    context.output.image(image, destination / "shape_corrected_baseline.npz")
    context.output.correction(correction, destination / "shape_correction_0_typecorrection.npz")
    with context.output.open(destination / "meta_data.json") as stream:
        stream.write(json.dumps(dict(baseline_path=str(source), raw_camera=camera,
            raw_baseline_sha256=context.raw_source["sha256"],
            raw_build_id=context.operation_id)).encode())
    context.output.finish_save()
    quality = dict(fit_n=8, fit_inlier_fraction=1.0, sampled_jacobian_min=1.0,
                   heldout=dict(sand=dict(n=3, p50_px=0.1, p95_px=0.2, max_px=0.3,
                                         median_match_correlation=0.9)))
    calls = []

    def register(review_g, camera_g, **kwargs):
        calls.append((review_g.copy(), camera_g.copy(), kwargs))
        return np.eye(3), quality

    monkeypatch.setattr(registration, "review_to_camera", register)
    return SimpleNamespace(context=context, rig=rig, source=source, profile=profile,
                           configuration=configuration, camera=camera, linear=linear,
                           destination=destination, quality=quality, calls=calls,
                           request_pin=request_pin)


def test_fresh_certificate_joins_actual_npz_pixels_to_the_renderer(prepared):
    f = prepared
    before = f.source.read_bytes(), f.configuration.read_bytes(), f.profile.read_bytes()
    result = f.context.publish(f.rig)
    document = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert document["format"] == build.FORMAT and document["run_id"] == "sample"
    assert document["artifact_kind"] == "raw_shape_basis" and document["complete_rig_model"] is False
    assert document["inputs"]["raw_camera_origin"] == "bound operation request override"
    assert document["inputs"]["operation_request"] == f.request_pin
    np.testing.assert_array_equal(f.calls[0][0],
        np.asarray(f.rig.shape_corrected_baseline.img[..., 1], np.float32) * np.float32(1000))
    np.testing.assert_array_equal(f.calls[0][1], f.linear[1])
    assert document["renderer"]["source_pixels"] == build._array_identity(f.rig.shape_corrected_baseline.img)
    assert document["corrected_baseline"]["archive"] == build._descriptor(f.destination / "shape_corrected_baseline.npz")
    assert document["inputs"]["configuration_files"] == [build._descriptor(f.configuration)]
    assert document["renderer"]["saved_shape_chain_replay_bit_identical"] is True
    assert document["renderer"]["direct_sensor_output_equality_claimed"] is False
    assert document["renderer"]["registration_invocation"]["policies"]["review"]["minimum_matches"] == 4
    assert document["homography_convention"]["raw_orientation"] == 1
    assert (f.source.read_bytes(), f.configuration.read_bytes(), f.profile.read_bytes()) == before


@pytest.mark.parametrize("source_name", ["source", "profile", "configuration"])
def test_changed_exact_build_input_refuses_before_a_certificate(prepared, source_name):
    f = prepared
    getattr(f, source_name).write_bytes(b"changed input")
    with pytest.raises(ValueError, match="changed"):
        f.context.publish(f.rig)
    assert not (f.destination / build.CERTIFICATE_NAME).exists()


def test_no_certificate_for_a_different_saved_payload(prepared):
    f = prepared
    altered = f.rig.shape_corrected_baseline.copy()
    altered.img[0, 0, 0] = np.nextafter(altered.img[0, 0, 0], np.float32(1))
    altered.save(f.destination / "shape_corrected_baseline.npz")
    with pytest.raises(ValueError, match="actual produced pixels"):
        f.context.publish(f.rig)
    assert not f.calls and not (f.destination / build.CERTIFICATE_NAME).exists()


def test_existing_archive_folder_cannot_receive_a_retrofit_certificate(prepared):
    f = prepared
    with pytest.raises(FileExistsError):
        build.begin_raw_rig_build(
            f.configuration, f.source, f.camera, f.destination,
            configuration_sources=build.capture_configuration_sources(f.configuration),
            request_binding=f.request_pin, artifact_kind="raw_shape_basis")


def test_run_assignment_cannot_borrow_another_camera_profile(prepared, tmp_path):
    f = prepared
    with pytest.raises(ValueError, match="exact run assignment"):
        build.begin_raw_rig_build(
            f.configuration, f.source, dict(f.camera, run_id="other"), tmp_path / "other",
            configuration_sources=build.capture_configuration_sources(f.configuration),
            request_binding=f.request_pin, artifact_kind="raw_shape_basis")


def test_config_changed_during_parsing_is_not_bound_to_the_previous_operation(prepared, tmp_path):
    f = prepared
    sources = build.capture_configuration_sources(f.configuration)
    f.configuration.write_text("# changed after loader admission\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        build.begin_raw_rig_build(f.configuration, f.source, f.camera, tmp_path / "other",
                                 configuration_sources=sources,
                                 request_binding=f.request_pin, artifact_kind="raw_shape_basis")


@pytest.mark.parametrize("field,value", [("fit_n", 0), ("fit_inlier_fraction", 0.2),
                                        ("sampled_jacobian_min", -1), ("heldout", {})])
def test_failed_existing_renderer_predicate_produces_no_certificate(prepared, field, value):
    f = prepared
    f.quality[field] = value
    with pytest.raises(ValueError, match="renderer"):
        f.context.publish(f.rig)
    assert not (f.destination / build.CERTIFICATE_NAME).exists()


def test_renderer_failure_leaves_the_original_build_outputs(prepared, monkeypatch):
    f = prepared
    archive = f.destination / "shape_corrected_baseline.npz"
    original = archive.read_bytes()

    def fail(*args, **kwargs):
        raise ValueError("configured renderer gate failed")

    monkeypatch.setattr(registration, "review_to_camera", fail)
    with pytest.raises(ValueError, match="configured renderer gate failed"):
        f.context.publish(f.rig)
    assert archive.read_bytes() == original
    assert not (f.destination / build.CERTIFICATE_NAME).exists()


def test_an_output_changed_by_the_renderer_cannot_be_published(prepared, monkeypatch):
    f = prepared
    asset = f.destination / "shape_correction_0_typecorrection.npz"

    def tamper(*args, **kwargs):
        asset.write_bytes(b"changed output")
        return np.eye(3), f.quality

    monkeypatch.setattr(registration, "review_to_camera", tamper)
    with pytest.raises(ValueError, match="changed"):
        f.context.publish(f.rig)
    assert not (f.destination / build.CERTIFICATE_NAME).exists()


@pytest.mark.parametrize("changed", ["pixel", "grid", "orientation", "chain", "H"])
def test_later_full_build_stops_on_every_basis_consistency_difference(prepared, tmp_path, changed):
    pin = prepared.context.publish(prepared.rig)
    document = json.loads(Path(pin["path"]).read_text(encoding="utf-8"))
    full = replace(prepared.context, artifact_kind="raw_rig_baseline",
        request=dict(prepared.context.request, shape_basis_certificate=pin))
    full._require_basis_consistency(document)
    altered = deepcopy(document)
    if changed == "pixel":
        altered["corrected_baseline"]["pixels"]["sha256"] = "0" * 64
    elif changed == "grid":
        altered["corrected_baseline"]["grid"]["metadata"]["origin"][0] += 1
    elif changed == "orientation":
        altered["homography_convention"]["raw_orientation"] = 6
    elif changed == "chain":
        altered["renderer"]["shape_corrections"][0]["sha256"] = "0" * 64
    else:
        altered["H_rig_corrected_native_to_baseline_camera_u"][0][2] = 0.001
    with pytest.raises(ValueError, match="consistency STOP"):
        full._require_basis_consistency(altered)


def test_shape_request_cannot_mislabel_old_toml_as_camera_source(prepared):
    request = deepcopy(prepared.context.request)
    request["operation_overrides"] = {}
    path = prepared.destination.parent / "invalid-request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit camera override"):
        build.read_build_request(build._descriptor(path), artifact_kind="raw_shape_basis")
