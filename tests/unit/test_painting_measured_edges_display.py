"""A display companion cannot bypass photograph or source identity."""
from copy import deepcopy
import os

import pytest

from darsia.fluidflower import painting_measured_edges_display as display

from .test_painting_measured_edges import saved, _write, _files, SHAPE, CONFIRMATION  # noqa: F401


def make_display(saved):
    project = saved.folder / "project"
    config_path = project / display.RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    configuration = dict(format=display.FORMAT, project_id="project-fixture",
        series_root=str(saved.folder / "native-series"), measured_edges={
            "geometry_index": deepcopy(saved.index_ref), "expected_confirmation": CONFIRMATION})
    _write(config_path, configuration)
    photo = saved.folder / "unopened-photo.arw"
    photo.write_bytes(b"verified photograph bytes")
    stat = photo.stat()
    arguments = dict(project_root=project, project_id=configuration["project_id"],
        series_root=configuration["series_root"], run="sample", photograph_path=str(photo),
        shape_hw=SHAPE, camera_u_binding=deepcopy(saved.camera),
        photograph_stamp=f"{stat.st_size}:{stat.st_mtime_ns}", geometry_index_path=str(saved.index_path))
    return arguments, configuration, config_path


def test_genuine_shaped_binding_and_companion_read_without_decoding_or_writes(saved):
    arguments, config, path = make_display(saved)
    before = _files(saved.folder)
    result = display.load_measured_edge_display(**arguments)
    assert result.configuration == config
    assert result.configuration_descriptor["path"] == str(path)
    assert [name for name, _ in result.edges.lines] == ["left", "bottom", "right"]
    assert result.edges.provenance["read_only"] is True
    assert result.photograph_version[2] == saved.photograph["size_bytes"]
    assert _files(saved.folder) == before
    # The runtime camera profile omits size; its digest and the bound model
    # supply the same descriptor only after the profile bytes are checked.
    del arguments["camera_u_binding"]["size_bytes"]
    assert display.load_measured_edge_display(**arguments).edges.lines == result.edges.lines


def test_missing_companion_disables_without_creating_a_file(saved):
    arguments, _config, _path = make_display(saved)
    arguments["project_root"] = saved.folder / "absent-project"
    before = _files(saved.folder)
    with pytest.raises(ValueError, match="No measured-edge display metadata"):
        display.load_measured_edge_display(**arguments)
    assert _files(saved.folder) == before


@pytest.mark.parametrize("field", ["format", "project_id", "series_root", "expected_confirmation", "index"])
def test_companion_scope_and_explicit_confirmation_are_required(saved, field):
    arguments, config, path = make_display(saved)
    if field in {"format", "project_id", "series_root"}:
        config[field] = "different"
    elif field == "expected_confirmation":
        del config["measured_edges"][field]
    else:
        config["measured_edges"]["geometry_index"]["sha256"] = "0" * 64
    _write(path, config)
    with pytest.raises(ValueError):
        display.load_measured_edge_display(**arguments)


@pytest.mark.parametrize("field", ["photo_path", "stamp", "grid", "profile_path", "profile_sha",
                                  "profile_id", "profile_size", "selected_index"])
def test_current_view_identity_is_checked_independently(saved, field):
    arguments, _config, _path = make_display(saved)
    if field == "photo_path":
        arguments["photograph_path"] = "another-photo.arw"
    elif field == "stamp":
        arguments["photograph_stamp"] = "1:2"
    elif field == "grid":
        arguments["shape_hw"] = (23, 32)
    elif field.startswith("profile_"):
        key, value = {"profile_path": ("path", "other-profile.json"),
                      "profile_sha": ("sha256", "0" * 64),
                      "profile_id": ("profile_id", "other"),
                      "profile_size": ("size_bytes", 1)}[field]
        arguments["camera_u_binding"][key] = value
    else:
        arguments["geometry_index_path"] = "other-index.json"
    with pytest.raises(ValueError):
        display.load_measured_edge_display(**arguments)


def test_same_size_same_stamp_photo_tamper_cannot_hide_from_digest(saved):
    arguments, _config, _path = make_display(saved)
    path = saved.folder / "unopened-photo.arw"
    before = path.stat()
    path.write_bytes(b"X" + path.read_bytes()[1:])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="photograph changed"):
        display.load_measured_edge_display(**arguments)


def test_companion_change_during_edge_verification_is_refused(saved, monkeypatch):
    arguments, config, path = make_display(saved)
    original = display.edges.read_saved_measured_edges

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        config["measured_edges"]["expected_confirmation"] += "-changed"
        _write(path, config)
        return result

    monkeypatch.setattr(display.edges, "read_saved_measured_edges", changed)
    with pytest.raises(ValueError, match="metadata changed"):
        display.load_measured_edge_display(**arguments)


def test_companion_json_is_bounded_and_duplicate_keys_refused(saved, monkeypatch):
    arguments, _config, path = make_display(saved)
    monkeypatch.setattr(display, "MAX_CONFIG_BYTES", 4)
    with pytest.raises(ValueError, match="bounded"):
        display.load_measured_edge_display(**arguments)
    monkeypatch.setattr(display, "MAX_CONFIG_BYTES", 64 * 1024)
    path.write_text('{"format":1,"format":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        display.load_measured_edge_display(**arguments)


def _configured_alias(saved, arguments):
    alias = saved.folder / "configured-profile-at-another-locator.json"
    alias.write_bytes(saved.profile_path.read_bytes())
    arguments["camera_u_binding"]["path"] = str(alias)
    return alias


@pytest.mark.parametrize("runtime_size", [True, False])
def test_exact_parameter_bytes_at_another_locator_are_admitted_only_with_both_bindings(saved, runtime_size):
    arguments, _config, _path = make_display(saved)
    alias = _configured_alias(saved, arguments)
    if not runtime_size:
        del arguments["camera_u_binding"]["size_bytes"]
    before, original_arguments = _files(saved.folder), deepcopy(arguments)
    # The global saved reader remains path-strict. Only the display admission
    # joins independently checked current/saved parameter documents.
    strict_camera = dict(arguments["camera_u_binding"], size_bytes=saved.camera["size_bytes"])
    with pytest.raises(ValueError, match="exact saved binding"):
        display.edges.read_saved_measured_edges(saved.index_ref, "sample",
            photograph_binding=saved.photograph, camera_u_binding=strict_camera,
            shape_hw=SHAPE, expected_confirmation=CONFIRMATION)
    result = display.load_measured_edge_display(**arguments)
    evidence = result.edges.provenance["camera_profile_locator_binding"]
    assert evidence["configured"]["path"] == str(alias)
    assert evidence["measured"]["path"] == str(saved.profile_path)
    assert evidence["configured"]["sha256"] == evidence["measured"]["sha256"] == saved.camera["sha256"]
    assert evidence["configured"]["size_bytes"] == evidence["measured"]["size_bytes"] == saved.camera["size_bytes"]
    assert evidence["locators_differ"] and evidence["exact_sha256_and_size"]
    assert evidence["both_run_assignments_and_grids_checked"]
    assert evidence["profile_id"] == "declared" and evidence["run_id"] == "sample"
    assert evidence["camera_wh"] == list(SHAPE[::-1]) and evidence["configuration_changed"] is False
    assert result.edges.provenance["camera_profile"] == {key: saved.camera[key] for key in ("path", "sha256", "size_bytes")}
    assert result.edges.lines == tuple((name, tuple(value)) for name, value in
                                     saved.source["models"][0]["original_final_lines"].items())
    assert arguments == original_arguments and _files(saved.folder) == before


@pytest.mark.parametrize("which", ["configured", "saved"])
def test_both_parameter_locators_must_still_contain_the_exact_declared_bytes(saved, which):
    arguments, _config, _path = make_display(saved)
    alias = _configured_alias(saved, arguments)
    path = alias if which == "configured" else saved.profile_path
    raw = path.read_bytes()
    path.write_bytes(b" " + raw[1:])  # Same size, with the original declared SHA retained.
    before = _files(saved.folder)
    with pytest.raises(ValueError, match="changed"):
        display.load_measured_edge_display(**arguments)
    assert _files(saved.folder) == before


@pytest.mark.parametrize("change", ["schema", "assignment", "duplicate_assignment", "declared_grid", "declared_profile"])
def test_equal_rebound_parameter_bytes_cannot_bypass_profile_schema_run_or_grid(saved, change):
    if change == "schema":
        saved.profile["source"]["profiles"] = []
    elif change == "assignment":
        saved.profile["source"]["profiles"]["sample"] = "other"
    elif change == "duplicate_assignment":
        saved.profile["source"]["profiles"]["SAMPLE"] = "declared"
    elif change == "declared_grid":
        saved.profile["registration"]["common"]["sensor"]["camera_wh"] = list(SHAPE)
    else:
        saved.profile["registration"]["common"]["profiles"] = {}
    saved.save()  # Fully rebind the synthetic model/index to the changed document.
    arguments, _config, _path = make_display(saved)
    _configured_alias(saved, arguments)
    before = _files(saved.folder)
    with pytest.raises(ValueError):
        display.load_measured_edge_display(**arguments)
    assert _files(saved.folder) == before


@pytest.mark.parametrize("change", ["profile_id", "camera_wh"])
def test_identical_alias_bytes_do_not_admit_an_unrelated_current_profile_or_grid(saved, change):
    arguments, _config, _path = make_display(saved)
    _configured_alias(saved, arguments)
    arguments["camera_u_binding"][change] = "other" if change == "profile_id" else list(SHAPE)
    before = _files(saved.folder)
    with pytest.raises(ValueError):
        display.load_measured_edge_display(**arguments)
    assert _files(saved.folder) == before


@pytest.mark.parametrize("which", ["configured", "saved"])
def test_parameter_change_during_line_loading_is_refused(saved, monkeypatch, which):
    arguments, _config, _path = make_display(saved)
    alias = _configured_alias(saved, arguments)
    original = display.edges.read_saved_measured_edges
    path = alias if which == "configured" else saved.profile_path
    expected_files = set(_files(saved.folder))

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(display.edges, "read_saved_measured_edges", changed)
    with pytest.raises(ValueError, match="changed"):
        display.load_measured_edge_display(**arguments)
    assert set(_files(saved.folder)) == expected_files
