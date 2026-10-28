"""Synthetic RAW authority/grid joins through the real owned producer.

Only the RAW decoder, registration estimator and runtime-wrapper identity are
synthetic. The 2x3 arrays, NPZ files, saved correction, ownership markers,
request, release/review joins and reader checks are real temporary artifacts.
The review entry is fixture data, not a real source approval. No native binary
accuracy, full rig qualification, optical accuracy or production GO is claimed.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia._opencv_runtime import scientific_thread, completed_frame
from darsia.fluidflower import geometry_workspace_frame as workspace
from darsia.fluidflower import painted_adoption_geometry
from darsia.fluidflower import raw_shape_basis_reader as reader
from darsia.fluidflower import rig_raw_build as build
from darsia.fluidflower.native import raw_positioning, registration


def _descriptor(path):
    raw = Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return _descriptor(path)


def _changed_byte(path):
    path = Path(path)
    before = path.read_bytes()
    assert before
    after = before[:-1] + (b" " if before[-1:] != b" " else b"\n")
    assert len(before) == len(after) and before != after
    path.write_bytes(after)
    return before


@pytest.fixture
def basis(tmp_path, monkeypatch, request):
    """Emit one genuine owned certificate around an explicitly fake RAW sample."""
    authority_mode = getattr(request, "param", "legacy")
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = raw_root / "baseline.ARW"
    raw.write_bytes(b"unit-only fake RAW; the decoder below is patched")
    configuration = tmp_path / "sample.toml"
    configuration.write_text("# exact private operation input\n", encoding="utf-8")
    series = tmp_path / "series.toml"
    series.write_text("# private series path context\n", encoding="utf-8")
    review_policy = dict(minimum_matches=4, minimum_inlier_fraction=.8,
        heldout_zones=[["sand", [0, 0, 2, 1], 1]], heldout_minimum_matches=2,
        heldout_min_correlation=.7)
    parameters = dict(common=dict(profiles={"camera": {}}, sensor=dict(camera_wh=[3, 2]),
                                  review=review_policy, matching={}))
    profile = tmp_path / "camera.json"
    _json(profile, dict(registration=parameters, source=dict(profiles={"sample": "camera"})))
    camera = dict(configuration=str(profile), sha256=_descriptor(profile)["sha256"],
                  profile_id="camera", run_id="sample")
    protocols = []
    for role in ("imaging", "injection", "pressure_temperature"):
        path = tmp_path / (role + ".csv")
        path.write_bytes(b"time,value\n0,1\n")
        protocols.append(dict(role=role, folder=None, sheet=None, source=_descriptor(path)))
    output_root = tmp_path / "products"
    output_root.mkdir()
    destination = output_root / "new_basis"
    request = dict(format=build.BASIS_REQUEST_FORMAT, run_id="sample",
        configuration_files=list(build.capture_configuration_sources(configuration)),
        series_context=dict(root=str(tmp_path), manifest=_descriptor(series)),
        baseline=_descriptor(raw), profile=_descriptor(profile),
        operation_overrides={"data.raw_camera": camera}, protocol_inputs=protocols,
        output=dict(destination=str(destination), intended_root=str(output_root),
                    forbidden_raw_roots=[str(raw_root)], maximum_output_bytes=10_000_000,
                    maximum_file_bytes=2_000_000), shape_basis_certificate=None)
    request_path = tmp_path / "operation-request.json"
    request_binding = _json(request_path, request)
    # Unit DATA supplier only: actual named producer/writer/reader source files.
    # This scoped list is not whole-package/native qualification. Reader,
    # authority, ownership, payload and source-byte checks remain unmodified.
    package = Path(build.__file__).resolve().parent
    implementation = [_descriptor(path) for path in sorted([
        package / "rig_raw_build.py", package / "rig_raw_write.py",
        package / "rig_raw_input.py", package / "raw_shape_basis_reader.py",
        package / "native/raw_positioning.py", package / "native/registration.py",
        package.parent / "corrections/typecorrection.py",
    ])]
    monkeypatch.setattr(build, "_source_snapshot", lambda: deepcopy(implementation))
    source_closure = dict(base_commit="1" * 40, tree="2" * 40, files=deepcopy(implementation))
    review_path = tmp_path / "review-entry.json"
    release_path = tmp_path / "root-release.json"
    release = dict(source_closure=source_closure, operation_request=request_binding,
                   released_utc=datetime.now(timezone.utc).isoformat())
    authority = dict(source_closure=source_closure, operation_request=request_binding)
    if authority_mode != "owner-release":
        review_entry = dict(id="synthetic-retained-source-validation",
            answer="Synthetic historical fixture data only; no actual producer/native qualification.")
        full_review = dict(entry=_json(review_path, review_entry), id=review_entry["id"],
                          answer_sha256=hashlib.sha256(review_entry["answer"].encode("utf-8")).hexdigest())
        release["full_review"] = full_review
        authority["full_review"] = full_review
    authority["root_release"] = _json(release_path, release)
    runtime_root = tmp_path / "runtime-fixture"
    runtime_root.mkdir()
    executable = runtime_root / "python-wrapper.exe"
    executable.write_bytes(b"synthetic executable identity; never executed")
    libraries = {}
    for name in ("cv2", "rawpy", "numpy"):
        path = runtime_root / (name + ".py")
        path.write_text("# synthetic wrapper identity\n", encoding="utf-8")
        libraries[name] = dict(version="synthetic-unit-wrapper", module=_descriptor(path))
    runtime = dict(python="synthetic unit runtime", executable=_descriptor(executable),
        identity_scope="Python module wrappers and executable file identities",
        native_binary_equivalence_claimed=False, production_qualification_claimed=False,
        libraries=libraries)
    monkeypatch.setattr(build, "_runtime_identity", lambda: deepcopy(runtime))
    context = build.begin_raw_rig_build(configuration, raw, camera, destination,
        configuration_sources=build.capture_configuration_sources(configuration),
        request_binding=request_binding, artifact_kind="raw_shape_basis")
    linear = np.arange(18, dtype=np.float32).reshape(3, 2, 3) * np.float32(10)
    sensor = SimpleNamespace(metadata=dict(white_level=1100, black_level=100, orientation=1))
    monkeypatch.setattr(raw_positioning, "read_sensor", lambda *a, **k: sensor)
    monkeypatch.setattr(raw_positioning, "camera_u", lambda *a, **k: linear.copy())
    image = darsia.OpticalImage(np.moveaxis(linear, 0, -1) / np.float32(1000), color_space="RGB").img_as(float)
    correction = darsia.TypeCorrection(np.float32)
    setup_observation = scientific_thread("synthetic-shape-reader:setup")
    reading_observation = completed_frame(scientific_thread("synthetic-shape-reader:reading-before"),
                                           "synthetic-shape-reader:reading-after")
    rig = SimpleNamespace(_raw_build_id=context.operation_id, raw_camera=camera,
        raw_baseline_sha256=context.raw_source["sha256"], baseline_path=raw,
        shape_corrected_baseline=image, shape_corrections=[correction],
        _raw_setup_input_metadata=image.metadata(), _raw_basis_opencv=setup_observation,
        _raw_reading_opencv=reading_observation)
    context.begin_setup(rig, destination / "figures")
    context.output.begin_save(rig, destination)
    context.output.image(image, destination / "shape_corrected_baseline.npz")
    context.output.correction(correction, destination / "shape_correction_0_typecorrection.npz")
    with context.output.open(destination / "meta_data.json") as stream:
        stream.write(json.dumps(dict(baseline_path=str(raw), raw_camera=camera,
            raw_baseline_sha256=context.raw_source["sha256"], raw_build_id=context.operation_id)).encode())
    context.output.finish_save()
    quality = dict(fit_n=8, fit_inlier_fraction=1., sampled_jacobian_min=1.,
        heldout=dict(sand=dict(n=3, p50_px=.1, p95_px=.2, max_px=.3, median_match_correlation=.9)))
    calls = []
    def registered(review_green, camera_green, **kwargs):
        calls.append((review_green.copy(), camera_green.copy(), kwargs))
        return np.eye(3), deepcopy(quality)
    monkeypatch.setattr(registration, "review_to_camera", registered)
    certificate = context.publish(rig)
    document = json.loads(Path(certificate["path"]).read_bytes())
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    old_rgb = image.copy()
    old_rgb.img = old_rgb.img + .01  # Spatial compatibility does not assert old/new pixel equality.
    old_rgb.save(parent_root / "baseline.npz")
    spatial = image.metadata()
    spatial.pop("type", None)
    spatial.pop("color_space", None)
    spatial["scalar"] = True
    fields = {"shape_corrected_baseline": dict(_descriptor(parent_root / "baseline.npz"), shape=[2, 3, 3])}
    for name, values in (("labels", np.array([[1, 1, 2], [1, 2, 2]], np.uint8)),
                         ("facies", np.ones((2, 3), np.uint8)),
                         ("depth", np.full((2, 3), .01, np.float64))):
        path = parent_root / (name + ".npz")
        darsia.Image(values, **spatial).save(path)
        fields[name] = dict(_descriptor(path), shape=[2, 3])
    row = dict(configuration=_descriptor(configuration), fields=fields,
               configured_depth_map=_descriptor(parent_root / "depth.npz"))
    parent = dict(revision_id="synthetic-parent", configuration=dict(frame="rig-corrected",
        frame_binding=dict(format=workspace.FRAME_BINDING_FORMAT, source_reference_run="sample",
            display_reference_run="sample", reference_shape_corrections=[_descriptor(destination / "shape_correction_0_typecorrection.npz")],
            runs={"sample": row})))
    parent_path = parent_root / "revision.json"
    parent_binding = _json(parent_path, parent)
    original_inputs = [raw, profile, configuration, series, request_path, release_path,
                       *([review_path] if review_path.exists() else []),
                       *[Path(row["source"]["path"]) for row in protocols]]
    return SimpleNamespace(certificate=certificate, document=document, authority=authority,
        parent_workspace=parent_binding, parent=parent, parent_path=parent_path, fields=fields,
        source=raw, profile=profile, configuration=configuration, request_path=request_path,
        review_path=review_path, release_path=release_path, runtime=runtime, protocols=protocols,
        context=context, rig=rig, image=image, quality=quality, calls=calls,
        before={str(path): path.read_bytes() for path in original_inputs})


def _read(f, **kwargs):
    return reader.read_shape_basis(f.certificate, run_id="sample", authority=f.authority,
                                   parent_workspace=f.parent_workspace, **kwargs)


def test_actual_owned_saved_chain_has_external_authority_and_real_parent_grid(basis):
    f = basis
    owned_before = {str(path): path.read_bytes() for path in f.context.destination.rglob("*") if path.is_file()}
    parent_before = {row["path"]: Path(row["path"]).read_bytes() for row in f.fields.values()}
    actual = _read(f, reference_image=_descriptor(f.source), camera_profile=dict(_descriptor(f.profile),
        profile_id="camera", camera_wh=[3, 2]))
    assert actual.document["producer"]["operation_id"] == f.context.operation_id
    assert actual.document["renderer"]["saved_shape_chain_replay_bit_identical"] is True
    assert actual.document["producer"]["runtime"]["native_binary_equivalence_claimed"] is False
    assert actual.document["production_qualification_claimed"] is False
    assert actual.parent_labels["shape_hw"] == [2, 3]
    assert actual.parent_labels["source"] == {k: f.fields["labels"][k] for k in ("path", "sha256", "size_bytes")}
    assert actual.compatibility["configuration"] == _descriptor(f.configuration)
    assert actual.compatibility["old_new_pixel_equality_claimed"] is False
    np.testing.assert_array_equal(f.calls[0][0], np.asarray(f.image.img[..., 1], np.float32) * np.float32(1000))
    np.testing.assert_array_equal(f.calls[0][1], np.arange(18, dtype=np.float32).reshape(3, 2, 3)[1] * 10)
    actual.revalidate()
    assert all(Path(path).read_bytes() == raw for path, raw in f.before.items())
    assert {str(path) for path in f.context.destination.rglob("*") if path.is_file()} == set(owned_before)
    assert all(Path(path).read_bytes() == raw for path, raw in (owned_before | parent_before).items())


@pytest.mark.parametrize("part", ["request_sha", "request_path", "source_file_sha", "review_sha", "review_id", "release_sha", "approved_flag"])
def test_external_authority_cannot_be_replaced_by_flags_or_different_identity(basis, part):
    authority = deepcopy(basis.authority)
    if part == "request_sha":
        authority["operation_request"]["sha256"] = "0" * 64
    elif part == "request_path":
        other = basis.request_path.parent / "same-bytes-another-request.json"
        other.write_bytes(basis.request_path.read_bytes())
        authority["operation_request"] = _descriptor(other)
        assert authority["operation_request"]["sha256"] == basis.authority["operation_request"]["sha256"]
    elif part == "source_file_sha":
        authority["source_closure"]["files"][0]["sha256"] = "0" * 64
    elif part == "review_sha":
        authority["full_review"]["answer_sha256"] = "0" * 64
    elif part == "review_id":
        authority["full_review"]["id"] = "another-review-entry"
    elif part == "release_sha":
        authority["root_release"]["sha256"] = "0" * 64
    else:
        authority["approved"] = True
    with pytest.raises(ValueError, match="[Aa]uthority|request|implementation|review|changed|unsupported"):
        reader.read_shape_basis(basis.certificate, run_id="sample", authority=authority,
                                parent_workspace=basis.parent_workspace)


@pytest.mark.parametrize("part", ["request", "source_closure", "review", "late_release"])
def test_release_content_is_checked_after_its_file_sha_is_valid(basis, part):
    f = basis
    authority = deepcopy(f.authority)
    release = json.loads(f.release_path.read_bytes())
    if part == "request":
        release["operation_request"]["sha256"] = "0" * 64
    elif part == "source_closure":
        release["source_closure"]["tree"] = "3" * 40
    elif part == "review":
        release["full_review"]["answer_sha256"] = "0" * 64
    else:
        started = datetime.fromisoformat(f.document["producer"]["started_utc"])
        release["released_utc"] = (started + timedelta(seconds=1)).isoformat()
    authority["root_release"] = _json(f.release_path, release)
    with pytest.raises(ValueError, match="Root release"):
        reader.read_shape_basis(f.certificate, run_id="sample", authority=authority,
                                parent_workspace=f.parent_workspace)


@pytest.mark.parametrize("part", ["raw", "protocol", "runtime_wrapper", "review_text"])
def test_changed_original_identity_cannot_be_rehashed_into_old_certificate(basis, part):
    f = basis
    path = {"raw": f.source, "protocol": Path(f.protocols[0]["source"]["path"]),
            "runtime_wrapper": Path(f.runtime["libraries"]["rawpy"]["module"]["path"]),
            "review_text": f.review_path}[part]
    _changed_byte(path)
    with pytest.raises(ValueError, match="changed"):
        _read(f)


def test_a_byte_valid_different_reference_photo_or_profile_is_refused(basis, tmp_path):
    other = tmp_path / "other.ARW"
    other.write_bytes(b"another synthetic acquisition")
    with pytest.raises(ValueError, match="another measured reference photograph"):
        _read(basis, reference_image=_descriptor(other))
    profile = dict(_descriptor(basis.profile), profile_id="another-profile", camera_wh=[3, 2])
    with pytest.raises(ValueError, match="exact camera profile"):
        _read(basis, camera_profile=profile)


@pytest.mark.parametrize("part", ["origin", "dimensions", "shape"])
def test_valid_parent_npz_with_a_different_full_spatial_grid_stops_without_migration(basis, part):
    f = basis
    parent = deepcopy(f.parent)
    path = Path(f.fields["labels"]["path"])
    old = darsia.imread(path)
    metadata = old.metadata()
    metadata.pop("type", None)
    values = old.img.copy()
    if part == "origin":
        metadata["origin"] = [float(value) + .1 for value in metadata["origin"]]
    elif part == "dimensions":
        metadata["dimensions"] = [float(value) * 2 for value in metadata["dimensions"]]
    else:
        values = np.ones((3, 2), np.uint8)
    darsia.Image(values, **metadata).save(path)
    parent["configuration"]["frame_binding"]["runs"]["sample"]["fields"]["labels"] = dict(_descriptor(path), shape=list(values.shape))
    binding = _json(f.parent_path, parent)
    with pytest.raises(ValueError, match="grid|grids|STOP"):
        reader.read_shape_basis(f.certificate, run_id="sample", authority=f.authority, parent_workspace=binding)


def test_missing_exact_run_or_external_authority_does_not_fall_back_to_a_common_grid(basis):
    f = basis
    with pytest.raises(ValueError, match="external Root"):
        reader.read_shape_basis(f.certificate, run_id="sample", authority=None,
                                parent_workspace=f.parent_workspace)
    parent = deepcopy(f.parent)
    parent["configuration"]["frame_binding"]["runs"] = {"another": parent["configuration"]["frame_binding"]["runs"]["sample"]}
    parent["configuration"]["frame_binding"]["display_reference_run"] = "another"
    binding = _json(f.parent_path, parent)
    with pytest.raises(ValueError, match="selected run"):
        reader.read_shape_basis(f.certificate, run_id="sample", authority=f.authority, parent_workspace=binding)


def test_revalidate_refuses_late_same_size_input_drift(basis):
    checked = _read(basis)
    _changed_byte(basis.configuration)
    with pytest.raises(ValueError, match="changed"):
        checked.revalidate()


def test_retained_metadata_folder_uses_declared_move_without_rebinding_sources(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths
    original = tmp_path / "original.toml"
    original.write_bytes(b"value = 1\n")
    bound = _descriptor(original)
    physical = tmp_path / "current-project" / "retained"
    physical.mkdir(parents=True)
    copy = physical / "00.toml"
    copy.write_bytes(original.read_bytes())
    logical = tmp_path / "previous-project" / "retained"
    aliases = [dict(original=bound, copy=dict(_descriptor(copy), path=copy.name))]
    monkeypatch.setattr(storage_paths, "resolve_storage_path", lambda value: physical
                        if Path(value) == logical else Path(value))
    original.unlink()
    with reader.retained_source_context(aliases, folder=logical):
        assert reader._Pins().add(bound, copyable=True) == bound
    copy.write_bytes(b"value = 2\n")
    with pytest.raises(ValueError, match="changed"):
        with reader.retained_source_context(aliases, folder=logical):
            pass


def test_moved_basis_metadata_keeps_original_digest_and_descriptor(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths
    previous = tmp_path / "previous" / "basis.json"
    previous.parent.mkdir()
    previous.write_bytes(b'{"basis":1}')
    bound = _descriptor(previous)
    current = tmp_path / "current" / "basis.json"
    current.parent.mkdir()
    previous.rename(current)
    monkeypatch.setattr(storage_paths, "resolve_storage_path", lambda value:
                        current if Path(value) == previous else Path(value))
    pins = reader._Pins()
    assert pins.add(bound, metadata=True) == {"basis": 1}
    assert pins.rows[reader._key(previous)]["original"] == bound
    pins.final()
    current.write_bytes(b'{"basis":2}')
    with pytest.raises(ValueError, match="changed"):
        pins.final()


@pytest.mark.parametrize("basis", ["legacy", "owner-release"], indirect=True)
def test_retained_copies_replay_original_authority_after_current_metadata_changes(basis, tmp_path):
    """Historical proof keeps literal bytes; it never synthesizes a new approval."""
    f = basis
    checked = _read(f)
    retained = tmp_path / "retained_preparation"
    retained.mkdir()
    aliases = []
    copied_config = None
    for number, row in enumerate(checked.copy_source_rows()):
        original, physical = row["original"], row["physical"]
        name = f"{number:04d}-" + Path(original["path"]).name
        target = retained / name
        target.write_bytes(Path(physical["path"]).read_bytes())
        copied = dict(_descriptor(target), path=name)
        aliases.append(dict(original=deepcopy(original), copy=copied))
        if original["path"] == str(f.configuration):
            copied_config = target
    assert copied_config is not None
    assert checked.sources_to_copy() == [row["original"] for row in checked.copy_source_rows()]
    for path in (f.configuration, f.release_path, f.parent_path,
                 *([f.review_path] if f.review_path.exists() else [])):
        _changed_byte(path)
    with pytest.raises(ValueError, match="changed"):
        _read(f)
    replayed = reader.read_shape_basis(f.certificate, run_id="sample", authority=f.authority,
        parent_workspace=f.parent_workspace, aliases=aliases, folder=retained)
    assert replayed.compatibility == checked.compatibility
    replayed.revalidate()
    _changed_byte(copied_config)
    with pytest.raises(ValueError, match="changed"):
        replayed.revalidate()


def test_old_npz_and_dimension_only_legacy_renderer_never_acquire_new_authority(basis):
    f = basis
    old = {k: f.fields["shape_corrected_baseline"][k] for k in ("path", "sha256", "size_bytes")}
    with pytest.raises(ValueError):
        reader.read_shape_basis(old, run_id="sample", authority=f.authority,
                                parent_workspace=f.parent_workspace)
    model = dict(run_id="sample", baseline_native_grid=dict(renderer=old, shape_hw=[2, 3]))
    assert reader.has_shape_basis(model) is False
    with pytest.raises(ValueError, match="no explicit RAW"):
        reader.read_model_basis(model)
    with pytest.raises(ValueError, match="payload|NPZ|SOURCE|renderer"):
        painted_adoption_geometry._require_corrected_payload_link({}, "sample", model, {"shape_hw": [2, 3]})


@pytest.mark.parametrize("part", ["pixel_identity", "grid_origin", "inverse", "orientation"])
def test_valid_certificate_file_sha_cannot_substitute_for_saved_payload_and_coordinates(basis, part):
    """The consumer must inspect real NPZ pixels/grid rather than trust duplicate claims."""
    f = basis
    document = deepcopy(f.document)
    if part == "pixel_identity":
        document["corrected_baseline"]["pixels"]["sha256"] = "0" * 64
        document["renderer"]["source_pixels"]["sha256"] = "0" * 64
        document["renderer"]["saved_shape_chain_replay"]["output_pixels"]["sha256"] = "0" * 64
    elif part == "grid_origin":
        for grid in (document["corrected_baseline"]["grid"], document["renderer"]["source_grid"],
                     document["renderer"]["saved_shape_chain_replay"]["output_grid"]):
            grid["metadata"]["origin"][0] += .1
    elif part == "inverse":
        document["renderer"]["camera_u_to_native_h"][0][2] = .5
    else:
        document["homography_convention"]["raw_orientation"] = 6
    certificate = _json(Path(f.certificate["path"]), document)
    with pytest.raises(ValueError, match="pixels|grid|inverse|orientation|coordinate"):
        reader.read_shape_basis(certificate, run_id="sample", authority=f.authority,
                                parent_workspace=f.parent_workspace)
