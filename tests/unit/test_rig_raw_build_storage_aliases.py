"""RAW Build source metadata can move without rewriting request or output identity."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import rig_raw_build as build
from darsia.fluidflower import rig_raw_basis as basis
from darsia.fluidflower import storage_paths
from .test_rig_raw_build import prepared  # noqa: F401


def _move_metadata(paths, archive, monkeypatch):
    archive.mkdir()
    aliases = []
    for path in paths:
        target = archive / path.name
        path.rename(target)
        aliases.append((str(path).replace("\\", "/").casefold(), str(target).replace("\\", "/")))
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: aliases)
    return aliases


def test_original_request_admission_and_publication_keep_declared_metadata_after_move(prepared, tmp_path, monkeypatch):
    f = prepared
    request = deepcopy(f.context.request)
    destination = f.destination.parent / "second-rig"
    request["output"]["destination"] = str(destination)
    request_path = tmp_path / "second-request.json"
    request_path.write_text(json.dumps(request))
    pin = build._descriptor(request_path)
    _move_metadata([f.profile, f.configuration, request_path], tmp_path / "archived-metadata", monkeypatch)
    parsed, pins = build.read_build_request(pin, artifact_kind="raw_shape_basis")
    assert parsed == request and pin in pins
    context = build.begin_raw_rig_build(f.configuration, f.source, f.camera, destination,
        configuration_sources=f.context.configurations, request_binding=pin, artifact_kind="raw_shape_basis")
    assert context.request == request and context.request_binding == pin
    assert context.configurations == f.context.configurations
    # The original already-produced synthetic rig can still publish after the
    # source metadata moved. Its RAW acquisition and owned outputs stay put.
    published = f.context.publish(f.rig)
    document = json.loads(Path(published["path"]).read_bytes())
    assert document["inputs"]["operation_request"] == f.request_pin
    assert document["inputs"]["configuration_files"] == list(f.context.configurations)
    assert f.source.is_file() and published["path"] == str(f.destination / build.CERTIFICATE_NAME)


def test_bound_basis_certificate_uses_same_source_alias_boundary(prepared, tmp_path, monkeypatch):
    f = prepared
    pin = f.context.publish(f.rig)
    document = json.loads(Path(pin["path"]).read_bytes())
    _move_metadata([Path(pin["path"])], tmp_path / "archived-basis", monkeypatch)
    full = replace(f.context, artifact_kind="raw_rig_baseline",
                   request=dict(f.context.request, shape_basis_certificate=pin))
    full._require_basis_consistency(document)
    (tmp_path / "archived-basis" / build.CERTIFICATE_NAME).write_bytes(b"changed basis")
    with pytest.raises(ValueError, match="changed"):
        full._require_basis_consistency(document)


def test_protocol_source_aliases_keep_roles_folders_and_sheets_exact(tmp_path, monkeypatch):
    paths = [tmp_path / (name + ".xlsx") for name in ("imaging", "injection", "pressure")]
    for path in paths:
        path.write_bytes((path.stem + " protocol bytes").encode())
    protocols = SimpleNamespace(imaging={"photographs": (str(paths[0]), "run")},
        injection=(str(paths[1]), "run"), pressure_temperature=(str(paths[2]), "run"), blacklist=None)
    request = dict(protocol_inputs=basis.protocol_sources(protocols))
    original = deepcopy(request)
    _move_metadata(paths, tmp_path / "archive", monkeypatch)
    basis.require_protocol_sources(request, protocols)
    assert request == original
    protocols.injection = (str(paths[1]), "another sheet")
    with pytest.raises(ValueError, match="protocols differ"):
        basis.require_protocol_sources(request, protocols)


def test_protocol_folders_follow_only_declared_aliases_without_rewriting_request(tmp_path, monkeypatch):
    sources = [tmp_path / (role + ".xlsx") for role in ("imaging", "injection", "pressure")]
    for path in sources:
        path.write_bytes(path.stem.encode())
    old, current = tmp_path / "original-photos", tmp_path / "current-photos"
    protocols = SimpleNamespace(imaging={str(current): (sources[0], "run-a")},
        injection=(sources[1], "run-a"), pressure_temperature=(sources[2], "run-a"), blacklist=None)
    request = dict(protocol_inputs=basis.protocol_sources(protocols))
    original = deepcopy(request)
    protocols.imaging = {str(old): (sources[0], "run-a")}
    with pytest.raises(ValueError, match="protocols differ"):
        basis.require_protocol_sources(request, protocols)
    aliases = [(str(old).replace("\\", "/").casefold(), str(current).replace("\\", "/"))]
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: aliases)
    basis.require_protocol_sources(request, protocols)
    assert request == original
    protocols.imaging = {str(old / "another-run"): (sources[0], "run-a")}
    with pytest.raises(ValueError, match="protocols differ"):
        basis.require_protocol_sources(request, protocols)
    protocols.imaging = {str(old): (sources[0], "other-sheet")}
    with pytest.raises(ValueError, match="protocols differ"):
        basis.require_protocol_sources(request, protocols)


def test_source_alias_does_not_relocate_or_accept_owned_output_pins(tmp_path, monkeypatch):
    output = tmp_path / "owned-output.npz"
    output.write_bytes(b"owned output bytes")
    pin = build._descriptor(output)
    _move_metadata([output], tmp_path / "archive", monkeypatch)
    build._verify_inputs([pin])
    with pytest.raises(FileNotFoundError):
        build._verify([pin])
    assert not output.exists()


def test_actual_begin_uses_moved_raw_and_preserves_request_and_output_ownership(prepared, tmp_path, monkeypatch):
    f = prepared
    request = deepcopy(f.context.request)
    target = f.destination.parent / "next-rig"
    request["output"]["destination"] = str(target)
    request_path = tmp_path / "next-request.json"
    request_path.write_text(json.dumps(request))
    pin = build._descriptor(request_path)
    raw_root = f.source.parent
    moved_root = tmp_path / "moved-raw"
    raw_root.rename(moved_root)
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: [
        (str(raw_root).replace("\\", "/").casefold(), str(moved_root).replace("\\", "/"))])
    moved = moved_root / f.source.name
    context = build.begin_raw_rig_build(f.configuration, f.source, f.camera, target,
        configuration_sources=f.context.configurations, request_binding=pin, artifact_kind="raw_shape_basis")
    assert context.raw_source == build._descriptor(moved)
    assert context.request == request and request["baseline"]["path"] == str(f.source)
    assert context.output.destination == target and (target / "raw_build_owner.json").is_file()
    assert context.output.protected == (moved_root.resolve(),)
    context.begin_setup(f.rig, target / "figures")
    assert (target / "raw_setup_started.json").is_file() and not raw_root.exists()
    context.output.begin_save(f.rig, target)
    context.output.image(f.rig.shape_corrected_baseline, target / "shape_corrected_baseline.npz")
    correction = f.rig.shape_corrections[0]
    context.output.correction(correction, target / "shape_correction_0_typecorrection.npz")
    with context.output.open(target / "meta_data.json") as stream:
        stream.write(json.dumps(dict(baseline_path=str(f.source), raw_camera=f.camera,
            raw_baseline_sha256=context.raw_source["sha256"], raw_build_id=context.operation_id)).encode())
    context.output.finish_save()
    # Both the original rig and its saved metadata keep their authored path.
    published = context.publish(f.rig)
    assert Path(published["path"]).is_file()


def test_actual_begin_refuses_output_inside_alias_relocated_raw_root(prepared, tmp_path, monkeypatch):
    f = prepared
    old_root = f.source.parent
    moved_root = tmp_path / "moved-raw"
    old_root.rename(moved_root)
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: [
        (str(old_root).replace("\\", "/").casefold(), str(moved_root).replace("\\", "/"))])
    request = deepcopy(f.context.request)
    target = moved_root / "forbidden-output"
    request["output"].update(destination=str(target), intended_root=str(moved_root))
    request_path = tmp_path / "unsafe-request.json"
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="intersects a protected"):
        build.begin_raw_rig_build(f.configuration, f.source, f.camera, target,
            configuration_sources=f.context.configurations, request_binding=build._descriptor(request_path),
            artifact_kind="raw_shape_basis")
    assert not target.exists()


def test_alias_does_not_accept_a_corrupted_input(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b'{"exact":true}')
    pin = build._descriptor(source)
    _move_metadata([source], tmp_path / "archive", monkeypatch)
    build._verify_inputs([pin])
    physical = tmp_path / "archive/source.json"
    physical.write_bytes(b'{"exact":false}')
    with pytest.raises(ValueError, match="changed"):
        build._verify_inputs([pin])


def test_setup_group_reuses_verified_input_bytes_and_still_rejects_changes(tmp_path, monkeypatch):
    from darsia.fluidflower.setup_source_reads import reuse_setup_source_reads
    source = tmp_path / "source.json"
    source.write_bytes(b'{"exact":true}')
    pin = build._descriptor(source)
    _move_metadata([source], tmp_path / "archive", monkeypatch)
    original = build._descriptor
    calls = []
    def counted(path):
        calls.append(str(path))
        return original(path)
    monkeypatch.setattr(build, "_descriptor", counted)
    with reuse_setup_source_reads():
        build._verify_inputs([pin])
        build._verify_inputs([pin])
        assert len(calls) == 1
        (tmp_path / "archive/source.json").write_bytes(b'{"exact":false}')
        with pytest.raises(ValueError, match="changed"):
            build._verify_inputs([pin])
    assert len(calls) == 2


def test_raw_input_reuses_only_current_claim_verified_bytes(tmp_path, monkeypatch):
    import os
    from darsia.fluidflower.pipeline import run_io
    from darsia.fluidflower.pipeline.store_runtime import ResolvedInput, verify_inputs, InputHashMismatch
    source = tmp_path / "sensor.ARW"
    source.write_bytes(b"sensor bytes")
    pin = build._descriptor(source)
    reference = dict(root_alias="raw", relative_path=source.name, role="image",
        sha256=pin["sha256"], bytes=pin["size_bytes"])
    row = ResolvedInput(reference, source, ("setup.raw.baseline",))
    verified = verify_inputs(dict(run="sample"), (row,))
    binding = object.__new__(run_io.RunIOBinding)
    binding.output = tmp_path / "output"
    binding.output_identity = binding.output.resolve()
    binding.resume_roots, binding._prefix_validation_root = (), None
    binding.code_locations = {}
    binding.locations = {run_io._key(source): row}
    binding.verified = verified
    original_open = Path.open
    def no_second_read(path, mode="r", *args, **kwargs):
        if path == source and "r" in mode:
            raise AssertionError("Already claimed and verified RAW was hashed twice")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_second_read)
    token = run_io._active.set(binding)
    try:
        assert build._input_descriptor(source) == pin
        source.write_bytes(b"changed data")
        original = source.stat()
        os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
        with pytest.raises(InputHashMismatch, match="changed before native consumption"):
            build._input_descriptor(source)
    finally:
        run_io._active.reset(token)
