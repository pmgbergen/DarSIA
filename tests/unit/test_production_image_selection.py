"""Synthetic metadata checks for the production-to-original-selection join.

These tests read bound JSON through the original InputCatalog. They neither
create genuine stage outputs nor run completion verification or science.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from darsia.fluidflower.production_image_selection import selection_sources
from darsia.fluidflower.pipeline import alias_job, job_inputs, staged_data
from .test_staged_data import phase1_data


def _write(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


@pytest.fixture
def selection(tmp_path, phase1_data, monkeypatch):
    manifest_path = tmp_path / "series.toml"
    raw = b'schema_version=1\nseries_id="production-unit"\n[roots]\nseries="."\n'
    manifest_path.write_bytes(raw)
    manifest = dict(path=str(manifest_path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    first = phase1_data["frames"][0]
    second = deepcopy(first)
    second.update(case_id="case-two", frame_index=2, elapsed_h=2.0)
    second["source"].update(source_sha256="d" * 64, previous_source_sha256="c" * 64,
                            work_index=2, elapsed_h=2.0,
                            source_binding=dict(source_sha256="d" * 64))
    phase1_data["frames"].append(second)
    phase1_data["ordered_case_ids"].append("case-two")
    bound = _write(tmp_path / "phase1.json", phase1_data)
    frames = [dict(case_id=row["case_id"], frame_index=row["frame_index"], elapsed_h=row["elapsed_h"],
                   source_binding=deepcopy(row["source"]["source_binding"])) for row in phase1_data["frames"]]
    config = dict(format="darsia-n1-run-v1", run=phase1_data["run"], precision="float32",
                  frames=frames, phase1=dict(configuration=bound))
    value = dict(pipeline="n1", job=dict(precision="float32"), runs=[dict(run=config["run"])])

    def forbidden(*args, **kwargs):
        pytest.fail("Selection metadata must not verify stage completion or create a job")

    monkeypatch.setattr(staged_data, "verify_phase1_completion", forbidden)
    monkeypatch.setattr(alias_job, "prepare_alias_job", forbidden)
    return dict(value=value, config=config, source=phase1_data, manifest=manifest, root=tmp_path)


def _read(selection):
    return selection_sources(selection["value"], selection["config"], series_manifest=selection["manifest"])


def test_original_reader_preserves_sources_and_reads_only_bound_configuration(selection, monkeypatch):
    original = job_inputs.InputCatalog._read_metadata
    opened = []

    def read(path):
        opened.append(str(path))
        return original(path)

    monkeypatch.setattr(job_inputs.InputCatalog, "_read_metadata", staticmethod(read))
    before = deepcopy(selection)
    files = {path.name: path.read_bytes() for path in selection["root"].iterdir()}
    source, catalog = _read(selection)
    assert source == selection["source"]
    assert type(catalog) is job_inputs.InputCatalog
    assert opened == [selection["config"]["phase1"]["configuration"]["path"]]
    assert selection == before
    assert {path.name: path.read_bytes() for path in selection["root"].iterdir()} == files


@pytest.mark.parametrize("pipeline", [None, "native-phase-statistics", "native-phase-n1", "other"])
def test_only_positive_n1_dispatch_is_admitted_before_manifest_io(selection, monkeypatch, pipeline):
    selection["value"]["pipeline"] = pipeline
    monkeypatch.setattr(alias_job, "_manifest", lambda *args: pytest.fail("Wrong route must not read the manifest"))
    with pytest.raises(ValueError, match="explicit n1"):
        _read(selection)


@pytest.mark.parametrize("change", ["run", "duplicate_run", "job_precision", "run_precision", "missing_precision"])
def test_plan_run_and_precision_must_match_before_input_io(selection, monkeypatch, change):
    if change == "run":
        selection["value"]["runs"][0]["run"] = "another-run"
    elif change == "duplicate_run":
        selection["value"]["runs"].append(deepcopy(selection["value"]["runs"][0]))
    elif change == "job_precision":
        selection["value"]["job"]["precision"] = "float64"
    elif change == "run_precision":
        selection["config"]["precision"] = "float64"
    else:
        del selection["value"]["job"]["precision"]
    monkeypatch.setattr(alias_job, "_manifest", lambda *args: pytest.fail("Bad plan join must not read inputs"))
    with pytest.raises(ValueError, match="precision|exactly once"):
        _read(selection)


@pytest.mark.parametrize("change", ["run", "precision", "selected", "predecessor", "order"])
def test_original_phase_one_validation_and_join_are_not_bypassed(selection, change):
    source = selection["source"]
    if change == "run":
        source["run"] = "another-run"
        for frame in source["frames"]:
            frame["source"]["run"] = source["run"]
    elif change == "precision":
        source["precision"] = "float64"
    elif change == "selected":
        source["frames"][0]["source"]["selected"] = False
    elif change == "predecessor":
        source["frames"][1]["source"]["previous_source_sha256"] = "f" * 64
    else:
        source["ordered_case_ids"].reverse()
    bound = selection["config"]["phase1"]["configuration"]
    selection["config"]["phase1"]["configuration"] = _write(Path(bound["path"]), source)
    with pytest.raises(ValueError):
        _read(selection)


@pytest.mark.parametrize("change", ["order", "missing_frame", "case", "index", "index_bool", "index_float",
                                     "time", "time_int", "source", "source_json_type"])
def test_phase_two_retains_exact_typed_acquisition_identity(selection, change):
    frames = selection["config"]["frames"]
    first = frames[0]
    if change == "order":
        frames.reverse()
    elif change == "missing_frame":
        frames.pop()
    elif change == "case":
        first["case_id"] = "different-case"
    elif change in {"index", "index_bool", "index_float"}:
        first["frame_index"] = {"index": 3, "index_bool": True, "index_float": 1.0}[change]
    elif change in {"time", "time_int"}:
        first["elapsed_h"] = 1 if change == "time_int" else 1.25
    elif change == "source":
        first["source_binding"]["source_sha256"] = "f" * 64
    else:
        source = selection["source"]
        source["frames"][0]["source"]["source_binding"]["index"] = 1
        bound = selection["config"]["phase1"]["configuration"]
        selection["config"]["phase1"]["configuration"] = _write(Path(bound["path"]), source)
        first["source_binding"]["index"] = True
    with pytest.raises(ValueError, match="order|frame|source binding"):
        _read(selection)


def test_metadata_byte_change_is_rejected_by_original_catalogue(selection):
    bound = selection["config"]["phase1"]["configuration"]
    path = Path(bound["path"])
    path.write_bytes(path.read_bytes().replace(b"float32", b"float64"))
    with pytest.raises(ValueError, match="SHA/size"):
        _read(selection)


def test_plan_and_run_aliases_share_original_catalogue_without_opening_camera(selection):
    bound = selection["config"]["phase1"]["configuration"]
    logical = dict(bound, path="C:/unavailable-source/phase1.json")
    selection["value"]["data_aliases"] = [dict(logical=logical, physical=bound)]
    selection["config"]["phase1"]["configuration"] = logical
    camera = dict(path="C:/unavailable-source/frame.ARW", sha256="f" * 64, size_bytes=999)
    physical = dict(camera, path=str(selection["root"] / "not-opened.ARW"))
    selection["config"]["data_aliases"] = [dict(logical=camera, physical=physical)]
    source, catalog = _read(selection)
    assert source == selection["source"]
    assert catalog.physical(logical) == bound
    assert catalog.physical(camera) == physical
    assert not Path(physical["path"]).exists()


@pytest.mark.parametrize("problem", ["sha", "size", "conflict"])
def test_invalid_combined_aliases_keep_original_refusal(selection, problem):
    bound = selection["config"]["phase1"]["configuration"]
    logical = dict(bound, path="C:/unavailable-source/phase1.json")
    physical = dict(bound)
    selection["value"]["data_aliases"] = [dict(logical=logical, physical=bound)]
    if problem == "sha":
        physical["sha256"] = "f" * 64
    elif problem == "size":
        physical["size_bytes"] += 1
    else:
        physical["path"] = str(selection["root"] / "other.json")
    selection["config"]["data_aliases"] = [dict(logical=logical, physical=physical)]
    with pytest.raises(ValueError, match="alias|aliases"):
        _read(selection)


def test_unregistered_original_configuration_refuses_before_open(selection, monkeypatch):
    selection["config"]["phase1"]["configuration"]["path"] = str(selection["root"].parent / "outside.json")
    monkeypatch.setattr(job_inputs.InputCatalog, "_read_metadata", staticmethod(
        lambda path: pytest.fail("Unregistered input must not be opened")))
    with pytest.raises(ValueError, match="No declared root"):
        _read(selection)


@pytest.mark.parametrize("carry", [False, True])
def test_original_phase_one_aliases_must_already_be_carried_by_production(selection, carry):
    logical = dict(path="C:/unavailable-source/frame.ARW", sha256="f" * 64, size_bytes=999)
    physical = dict(logical, path=str(selection["root"] / "not-opened.ARW"))
    aliases = [dict(logical=logical, physical=physical)]
    selection["source"]["data_aliases"] = aliases
    bound = selection["config"]["phase1"]["configuration"]
    selection["config"]["phase1"]["configuration"] = _write(Path(bound["path"]), selection["source"])
    if carry:
        selection["config"]["data_aliases"] = deepcopy(aliases)
        source, catalog = _read(selection)
        assert source == selection["source"] and catalog.physical(logical) == physical
    else:
        with pytest.raises(ValueError, match="carry its original phase-one aliases"):
            _read(selection)
