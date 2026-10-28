"""Saved edge overlays require exact source and camera-grid evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import painting_measured_edges as measured


SHAPE = (24, 32)
CONFIRMATION = "fixture-external-schema-value"
LINES = {"left": [1.0, .125, -2.75], "bottom": [.0625, 1., -22.5],
         "right": [2.0, -.125, -59.25]}


def _write(path, value):
    raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


@pytest.fixture
def saved(tmp_path):
    photograph = dict(path=str(tmp_path / "unopened-photo.arw"),
                      sha256=hashlib.sha256(b"verified photograph bytes").hexdigest(),
                      size_bytes=len(b"verified photograph bytes"))
    profile = dict(registration={"common": {"sensor": {"camera_wh": list(SHAPE[::-1])},
                                           "profiles": {"declared": {}}}},
                   source={"profiles": {"sample": "declared"}})
    source = dict(run="sample", models=[dict(original_final_lines=deepcopy(LINES))],
                  primary_B_hypothesis={"left": "L2", "bottom": "B3", "right": "R4"})
    edges = {name: dict(candidate_id=source["primary_B_hypothesis"][name], support="full",
                       camera_line_abc=deepcopy(line), pointer=f"/models/0/original_final_lines/{name}",
                       candidate_ID_pointer="/primary_B_hypothesis",
                       physical_B_confirmation=CONFIRMATION, production_adoption=False)
             for name, line in LINES.items()}
    ledger = dict(format="synthetic-confirmed-edge-ledger", rows=[
        dict(run="sample", all_three_B_confirmed=True, edges=edges)])
    model = dict(format=measured.GEOMETRY_FORMAT, run="sample", run_id="sample", grid="camera_u",
                 shape_hw=list(SHAPE), reference_image=deepcopy(photograph),
                 model_photograph=deepcopy(photograph))
    index = dict(format=measured.GEOMETRY_FORMAT, grid="camera_u", runs={"sample": {"run": "sample"}})
    fixture = SimpleNamespace(profile=profile, source=source, ledger=ledger, model=model, index=index,
                              photograph=photograph, folder=tmp_path,
                              profile_path=tmp_path / "profile.json", source_path=tmp_path / "lines.json",
                              ledger_path=tmp_path / "edges.json", model_path=tmp_path / "model.json",
                              index_path=tmp_path / "index.json")

    def save():
        profile_ref = _write(fixture.profile_path, profile)
        fixture.camera = dict(profile_ref, profile_id="declared", camera_wh=list(SHAPE[::-1]))
        source_ref = _write(fixture.source_path, source)
        for edge in ledger["rows"][0]["edges"].values():
            edge["source"] = deepcopy(source_ref)
            if "candidate_ID_pointer" in edge:
                edge["candidate_ID_source"] = deepcopy(source_ref)
        model["camera_profile"] = deepcopy(fixture.camera)
        model["current_edges"] = _write(fixture.ledger_path, ledger)
        index["runs"]["sample"]["reference_model"] = _write(fixture.model_path, model)
        fixture.index_ref = _write(fixture.index_path, index)

    fixture.save = save
    save()
    return fixture


def _read(saved, **kwargs):
    arguments = dict(photograph_binding=saved.photograph, camera_u_binding=saved.camera,
                     shape_hw=SHAPE, expected_confirmation=CONFIRMATION)
    arguments.update(kwargs)
    return measured.read_saved_measured_edges(saved.index_ref, "sample", **arguments)


def test_exact_saved_coefficients_and_provenance_without_top_or_writes(saved, monkeypatch):
    before = _files(saved.folder)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert str(path) != saved.photograph["path"], "Photographs must not be opened"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = _read(saved)
    assert result.run_id == "sample" and result.shape_hw == SHAPE
    assert result.lines == tuple((name, tuple(LINES[name])) for name in ("left", "bottom", "right"))
    assert all(name != "top" for name, _ in result.lines)
    assert result.provenance["read_only"] is True
    assert result.provenance["adoption_changed"] is False
    assert result.provenance["index"] == saved.index_ref
    assert result.provenance["edge_records"] == saved.ledger["rows"][0]["edges"]
    assert _files(saved.folder) == before
    result.provenance["edge_records"]["left"]["camera_line_abc"][0] = 99
    assert saved.ledger["rows"][0]["edges"]["left"]["camera_line_abc"][0] == 1.


def test_path_index_is_snapshotted_and_case_insensitive_run_selection(saved):
    result = measured.read_saved_measured_edges(saved.index_path, "  SAMPLE  ",
        photograph_binding=saved.photograph, camera_u_binding=saved.camera,
        shape_hw=SHAPE, expected_confirmation=CONFIRMATION)
    assert result.provenance["index"] == saved.index_ref
    assert result.lines[0][1] == tuple(LINES["left"])


@pytest.mark.parametrize("which", ["index", "model", "ledger", "source", "profile"])
def test_every_pinned_json_rejects_changed_bytes(saved, which):
    path = getattr(saved, which + "_path")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        _read(saved)


def test_declared_byte_size_is_checked_even_with_correct_digest(saved):
    saved.index["runs"]["sample"]["reference_model"]["size_bytes"] += 1
    saved.index_ref = _write(saved.index_path, saved.index)
    with pytest.raises(ValueError, match="byte size"):
        _read(saved)


@pytest.mark.parametrize("which", ["index", "model", "ledger", "profile"])
def test_duplicate_or_conflicting_run_bindings_are_refused(saved, which):
    if which == "index":
        saved.index["runs"]["SAMPLE"] = deepcopy(saved.index["runs"]["sample"])
    elif which == "model":
        saved.model["run_id"] = "other"
    elif which == "ledger":
        saved.ledger["rows"].append(deepcopy(saved.ledger["rows"][0]))
    else:
        saved.profile["source"]["profiles"]["SAMPLE"] = "declared"
    saved.save()
    with pytest.raises(ValueError, match="duplicate|another run"):
        _read(saved)


@pytest.mark.parametrize("which", ["index", "model", "ledger"])
def test_missing_selected_run_is_refused(saved, which):
    if which == "index":
        saved.index["runs"] = {"other": {"run": "other"}}
        saved.index_ref = _write(saved.index_path, saved.index)
    elif which == "model":
        saved.model["run"] = saved.model["run_id"] = "other"
        saved.save()
    else:
        saved.ledger["rows"][0]["run"] = "other"
        saved.save()
    with pytest.raises(ValueError, match="matching run|another run"):
        _read(saved)


@pytest.mark.parametrize("which", ["grid", "shape", "photo_sha", "photo_size", "photo_path",
                                  "model_photo", "camera_sha", "camera_id", "camera_size",
                                  "declared_size", "run_profile"])
def test_exact_photograph_and_camera_grid_binding_required(saved, which):
    if which == "grid":
        saved.model["grid"] = "rig-corrected"
    elif which == "shape":
        saved.model["shape_hw"] = [23, 32]
    elif which.startswith("photo_"):
        key, replacement = {"photo_sha": ("sha256", "0" * 64),
                            "photo_size": ("size_bytes", 1),
                            "photo_path": ("path", "different.arw")}[which]
        saved.model["reference_image"][key] = replacement
    elif which == "model_photo":
        saved.model["model_photograph"]["sha256"] = "0" * 64
    elif which == "declared_size":
        saved.profile["registration"]["common"]["sensor"]["camera_wh"] = [32, 23]
    elif which == "run_profile":
        saved.profile["source"]["profiles"]["sample"] = "different"
    saved.save()
    camera = deepcopy(saved.camera)
    if which == "camera_sha":
        camera["sha256"] = "0" * 64
    elif which == "camera_id":
        camera["profile_id"] = "different"
    elif which == "camera_size":
        camera["camera_wh"] = [31, 24]
    with pytest.raises(ValueError, match="grid|binding|profile"):
        _read(saved, camera_u_binding=camera)


@pytest.mark.parametrize("mutation", ["unconfirmed", "partial", "wrong_confirmation",
                                     "wrong_candidate", "wrong_pointer", "different_line",
                                     "wrong_source_run", "zero_line", "nonfinite", "boolean"])
def test_confirmation_full_support_and_exact_source_pointer_required(saved, mutation):
    edge = saved.ledger["rows"][0]["edges"]["bottom"]
    if mutation == "unconfirmed":
        saved.ledger["rows"][0]["all_three_B_confirmed"] = False
    elif mutation == "partial":
        edge["support"] = "partial"
    elif mutation == "wrong_confirmation":
        edge["physical_B_confirmation"] = CONFIRMATION + "-different"
    elif mutation == "wrong_candidate":
        edge["candidate_id"] = "B9"
    elif mutation == "wrong_pointer":
        edge["pointer"] = "/models/0/original_final_lines/left"
    elif mutation == "different_line":
        edge["camera_line_abc"][2] += .01
    elif mutation == "wrong_source_run":
        saved.source["run"] = "other"
    else:
        edge["camera_line_abc"] = {"zero_line": [0., 0., 1.],
                                    "nonfinite": [float("inf"), 1., 0.],
                                    "boolean": [True, 1., 0.]}[mutation]
    saved.save()
    with pytest.raises(ValueError):
        _read(saved)


def test_replacement_record_pointer_preserves_actual_coefficients(saved):
    line = [.125, 2., -44.]
    saved.source["records"] = [dict(final_line=line)]
    edge = saved.ledger["rows"][0]["edges"]["bottom"]
    edge["pointer"] = "/records/0/final_line"
    edge["camera_line_abc"] = line.copy()
    del edge["candidate_ID_pointer"]
    del edge["candidate_ID_source"]
    saved.save()
    result = _read(saved)
    assert dict(result.lines)["bottom"] == tuple(line)
    assert result.provenance["edge_records"]["bottom"]["pointer"] == "/records/0/final_line"


@pytest.mark.parametrize("pointer", ["models/0", "/models/00", "/models/-1",
                                    "/models/99", "/models/~2", "/models/0/missing"])
def test_malformed_or_absent_exact_pointer_refused(saved, pointer):
    saved.ledger["rows"][0]["edges"]["left"]["pointer"] = pointer
    saved.save()
    with pytest.raises(ValueError, match="pointer"):
        _read(saved)


def test_relative_metadata_paths_are_relative_to_the_binding_document(saved):
    saved.model["current_edges"]["path"] = saved.ledger_path.name
    saved.model["camera_profile"]["path"] = saved.profile_path.name
    for edge in saved.ledger["rows"][0]["edges"].values():
        edge["source"]["path"] = saved.source_path.name
        edge["candidate_ID_source"]["path"] = saved.source_path.name
    saved.model["current_edges"] = _write(saved.ledger_path, saved.ledger)
    saved.model["current_edges"]["path"] = saved.ledger_path.name
    saved.index["runs"]["sample"]["reference_model"] = _write(saved.model_path, saved.model)
    saved.index["runs"]["sample"]["reference_model"]["path"] = saved.model_path.name
    saved.index_ref = _write(saved.index_path, saved.index)
    camera = deepcopy(saved.camera)
    camera["path"] = saved.profile_path.name
    result = _read(saved, camera_u_binding=camera)
    assert result.lines[0][1] == tuple(LINES["left"])


def test_confirmation_has_no_default_and_empty_value_is_refused(saved):
    with pytest.raises(TypeError, match="expected_confirmation"):
        measured.read_saved_measured_edges(saved.index_path, "sample",
            photograph_binding=saved.photograph, camera_u_binding=saved.camera, shape_hw=SHAPE)
    with pytest.raises(ValueError, match="confirmation"):
        _read(saved, expected_confirmation="")


@pytest.mark.parametrize("kwargs", [{"photograph_binding": None}, {"camera_u_binding": None},
                                    {"shape_hw": (True, 32)}])
def test_missing_binding_or_invalid_grid_refused_cleanly(saved, kwargs):
    with pytest.raises(ValueError):
        _read(saved, **kwargs)


def test_bound_json_size_and_duplicate_keys_are_refused(saved, monkeypatch):
    monkeypatch.setattr(measured, "MAX_JSON_BYTES", 8)
    with pytest.raises(ValueError, match="bounded"):
        _read(saved)
    monkeypatch.setattr(measured, "MAX_JSON_BYTES", 16 * 1024 * 1024)
    saved.index_path.write_text('{"format":"first","format":"second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        measured.read_saved_measured_edges(saved.index_path, "sample",
            photograph_binding=saved.photograph, camera_u_binding=saved.camera,
            shape_hw=SHAPE, expected_confirmation=CONFIRMATION)
