"""Metadata-only tests of the private GUI baseline admission, without DarSIA imports.

Compile the actual owned functions from source, leaving numerical modules and
job constructors outside this unit boundary. The reader still verifies real
fixture JSON bytes; no scientific input, queue, or worker is opened.
"""
import ast
from copy import deepcopy
import hashlib
import json
import ntpath
from pathlib import Path
import re
import sys
import types

import pytest


def descriptor(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


class MetadataReader:
    def read(self, path, bound):
        if descriptor(path) != bound:
            raise ValueError("fixture metadata bytes changed")
        return json.loads(Path(path).read_bytes())


@pytest.fixture
def api(monkeypatch):
    path = Path(__file__).parents[2] / "src/darsia/fluidflower/jobs_metadata.py"
    wanted = {"_key", "_bound", "bind_local_selection_roles", "validate_current_selection"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in body} == wanted
    # Import only the original pure suffix classifier under an isolated package.
    # This preserves the RAW-family guard without importing DarSIA or numerical code.
    formats_path = path.with_name("image_formats.py")
    if not formats_path.exists():
        formats_path = Path("C:/FFAC_cache/validation/n497_part_b_parallel_20261023_v1/b4/src/darsia/fluidflower/image_formats.py")
    formats_tree = ast.parse(formats_path.read_text(encoding="utf-8"))
    formats_body = [node for node in formats_tree.body if
        (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in
            {"PROCESSED_IMAGE_SUFFIXES", "CAMERA_RAW_SUFFIXES"} for target in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name == "image_family")]
    formats = types.ModuleType("isolated_jobs_binding.image_formats")
    formats.Path = Path
    exec(compile(ast.Module(body=formats_body, type_ignores=[]), str(formats_path), "exec"), formats.__dict__)
    monkeypatch.setitem(sys.modules, formats.__name__, formats)
    namespace = dict(Path=Path, ntpath=ntpath, re=re, descriptor=descriptor,
                     JobSnapshotReader=MetadataReader, __package__="isolated_jobs_binding")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def data(tmp_path):
    series = tmp_path / "series"
    series.mkdir()
    manifest = series / "series.toml"
    manifest.write_text('series_id = "unit"\n', encoding="utf-8")
    protocol = series / "protocol.json"
    protocol.write_text('{}', encoding="utf-8")
    manifest_bound = descriptor(manifest)
    record_path = tmp_path / "selection.json"
    raws = [dict(path=str(tmp_path / (name + ".ARW")), sha256=letter * 64, size_bytes=10)
            for name, letter in [("baseline", "a"), ("early", "b"), ("candidate", "c")]]
    times = [-.1, .125, .4972]
    declarations = [dict(current_selection_position=i, current_frame_index=i,
                         raw_path=raw["path"], raw_sha256=raw["sha256"],
                         recorded_raw_size_bytes=raw["size_bytes"], elapsed_h=times[i],
                         baseline=i == 0, previous_raw_sha256=None if i == 0 else raws[i-1]["sha256"])
                    for i, raw in enumerate(raws)]
    specs = [dict(case_id="case-" + str(i), frame_index=i, elapsed_h=times[i], raw_acquisition=raw,
                  source=dict(run="run-one", selected=i != 0, selection_index=i, source_sha256=raw["sha256"],
                              elapsed_h=times[i], previous_source_sha256=declarations[i]["previous_raw_sha256"],
                              source_binding=dict(raw_acquisition=deepcopy(raw))))
             for i, raw in enumerate(raws)]
    record = dict(format="ffac-local-p1-selection-data-v1", decision=dict(actor="Reviewer", reason="fixture choice"),
                  input_authority=dict(current_series=manifest_bound), history_policy="original chronology",
                  runs=[dict(run="run-one", baseline=declarations[0], selected_frames=declarations[1:],
                             initial_previous_raw_sha256=raws[0]["sha256"])])
    config = dict(format="darsia-build-raw-phase-inputs-v1", run="run-one", history_policy=record["history_policy"],
                  selection_index_field="selection_index",
                  reference_spec=specs[0], initial_previous_spec=deepcopy(specs[0]), frames=specs[1:],
                  provenance=dict(local_current_selection=dict(current_series_manifest=manifest_bound)))
    resolved = dict(format="fluidflower-resolved-series-image-selection-v1", series_id="unit", run_id="run-one",
                    manifest=manifest_bound, source_metadata=[descriptor(protocol)], selected_frame_count=3,
                    selected=[dict(source_path=raw["path"], elapsed_h=times[i]) for i, raw in enumerate(raws)])
    def publish():
        record_path.write_text(json.dumps(record), encoding="utf-8")
        config["provenance"]["local_current_selection"]["selection_record"] = descriptor(record_path)
    publish()
    return dict(series=series, record=record, record_path=record_path, config=config, resolved=resolved,
                publish=publish, tmp_path=tmp_path)


def test_explicit_baseline_is_bound_without_a_duplicate_output(api, data):
    before = deepcopy(data["config"])
    rows, baseline = api["bind_local_selection_roles"](data["resolved"], data["config"], MetadataReader())
    assert len(rows) == len(data["config"]["frames"]) == 2
    assert rows == data["resolved"]["selected"][1:]
    assert baseline["reference_case_id"] == baseline["initial_previous_case_id"] == "case-0"
    assert baseline["original_camera_source"] == data["config"]["reference_spec"]["raw_acquisition"]
    assert data["config"] == before
    assert not list(data["tmp_path"].rglob("*.ARW"))


def test_gui_admission_keeps_baseline_separate_and_preserves_selected_cases(api, data):
    plan_path = data["tmp_path"] / "execution.toml"
    plan_path.write_text("fixture plan", encoding="utf-8")
    plan = descriptor(plan_path)
    declaration = dict(run="run-one", configuration=dict(path="fixture configuration"))
    def admitter(path, *, expected_sha256):
        assert str(path) == plan["path"] and expected_sha256 == plan["sha256"]
        return dict(runs=[declaration]), [data["config"]], plan
    result = api["validate_current_selection"](plan, series_root=data["series"], series_id="unit",
        selection_provider=lambda runs: [data["resolved"]] if runs == ["run-one"] else [], admitter=admitter)
    run = result["runs"][0]
    assert [case["case_id"] for case in run["cases"]] == ["case-1", "case-2"]
    assert run["baseline_roles"]["reference_case_id"] == "case-0"
    assert result["execution_entries_unchanged"] and not result["raw_hashed_by_this_UI"]


def test_ordinary_plan_does_not_implicitly_drop_its_first_selected_row(api, data):
    data["config"].pop("provenance")
    rows, baseline = api["bind_local_selection_roles"](data["resolved"], data["config"], MetadataReader())
    assert rows is data["resolved"]["selected"] and baseline is None
    plan_path = data["tmp_path"] / "ordinary.toml"
    plan_path.write_text("ordinary", encoding="utf-8")
    plan = descriptor(plan_path)
    with pytest.raises(ValueError, match="image count differs"):
        api["validate_current_selection"](plan, series_root=data["series"], series_id="unit",
            selection_provider=lambda runs: [data["resolved"]],
            admitter=lambda *args, **kwargs: (dict(runs=[dict(run="run-one")]), [data["config"]], plan))


@pytest.mark.parametrize("change", [
    "record_bytes", "series_binding", "local_series_binding", "run", "duplicate_run", "format",
    "history", "saved_time", "saved_order", "baseline_path", "baseline_sha", "baseline_size",
    "baseline_selected", "baseline_previous", "different_reference", "duplicate_baseline_output",
    "frame_sha", "frame_time", "frame_predecessor", "frame_order", "frame_case", "record_position",
])
def test_changed_explicit_baseline_or_selection_is_rejected(api, data, change):
    config, record, resolved = data["config"], data["record"], data["resolved"]
    if change == "record_bytes":
        data["record_path"].write_text('{}', encoding="utf-8")
    elif change == "series_binding": record["input_authority"]["current_series"] = dict(path="other")
    elif change == "local_series_binding": config["provenance"]["local_current_selection"]["current_series_manifest"] = {}
    elif change == "run": config["run"] = "other"
    elif change == "duplicate_run": record["runs"].append(deepcopy(record["runs"][0]))
    elif change == "format": config["format"] = "ordinary-format"
    elif change == "history": config["history_policy"] = "different history"
    elif change == "saved_time": resolved["selected"][0]["elapsed_h"] -= .1
    elif change == "saved_order": resolved["selected"].reverse()
    elif change == "baseline_path": config["reference_spec"]["raw_acquisition"]["path"] += ".other"
    elif change == "baseline_sha": config["reference_spec"]["raw_acquisition"]["sha256"] = "d" * 64
    elif change == "baseline_size": config["reference_spec"]["raw_acquisition"]["size_bytes"] += 1
    elif change == "baseline_selected": config["reference_spec"]["source"]["selected"] = True
    elif change == "baseline_previous": record["runs"][0]["initial_previous_raw_sha256"] = "d" * 64
    elif change == "different_reference": config["initial_previous_spec"]["case_id"] = "other"
    elif change == "duplicate_baseline_output": config["frames"].insert(0, deepcopy(config["reference_spec"]))
    elif change == "frame_sha": config["frames"][0]["source"]["source_sha256"] = "d" * 64
    elif change == "frame_time": config["frames"][0]["elapsed_h"] += .01
    elif change == "frame_predecessor": config["frames"][0]["source"]["previous_source_sha256"] = "d" * 64
    elif change == "frame_order": config["frames"].reverse()
    elif change == "frame_case": config["frames"][1]["case_id"] = config["frames"][0]["case_id"]
    elif change == "record_position": record["runs"][0]["selected_frames"][0]["current_selection_position"] = 0
    if change != "record_bytes": data["publish"]()
    with pytest.raises(ValueError):
        api["bind_local_selection_roles"](resolved, config, MetadataReader())


def stage_configuration(data, format):
    """Use the actual build_raw_phase saved reference/history and frame shapes."""
    config = data["config"]
    baseline = config.pop("reference_spec")
    previous = config.pop("initial_previous_spec")
    native_row = data["tmp_path"] / "baseline_native_row.json"
    native_row.write_text(json.dumps(dict(raw_acquisition=baseline["raw_acquisition"])), encoding="utf-8")
    row_bound = descriptor(native_row)
    config["format"] = format
    config["phase"] = dict(reference=dict(native_row=row_bound, source_sha256=baseline["source"]["source_sha256"]),
        initial_previous=dict(native_row=deepcopy(row_bound), source=previous["source"],
                              current_h=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    for frame in config["frames"]:
        raw = frame.pop("raw_acquisition")
        geometry = data["tmp_path"] / (frame["case_id"] + "_geometry.json")
        geometry.write_text(json.dumps(dict(acquisition=raw)), encoding="utf-8")
        frame["native_task"] = dict(kind="raw_rig", geometry=descriptor(geometry))
    return config


@pytest.mark.parametrize("format", ["darsia-native-phase-statistics-run-v1", "darsia-native-phase-n1-run-v1"])
def test_staged_gui_admission_binds_native_baseline_and_original_geometry(api, data, format):
    config = stage_configuration(data, format)
    data["record"]["runs"][0]["baseline"]["recorded_raw_size_bytes"] = None
    data["publish"]()
    before = deepcopy(config)
    plan_path = data["tmp_path"] / "staged.toml"
    plan_path.write_text("staged plan", encoding="utf-8")
    plan = descriptor(plan_path)
    result = api["validate_current_selection"](plan, series_root=data["series"], series_id="unit",
        selection_provider=lambda runs: [data["resolved"]],
        admitter=lambda *args, **kwargs: (dict(runs=[dict(run="run-one", configuration={})]), [config], plan))
    run = result["runs"][0]
    assert len(run["cases"]) == 2
    assert run["baseline_roles"]["reference_native_row"] == config["phase"]["reference"]["native_row"]
    assert run["baseline_roles"]["initial_previous_native_row"] == config["phase"]["initial_previous"]["native_row"]
    assert run["baseline_roles"]["original_camera_source"]["sha256"] == "a" * 64
    assert [case["original_camera_source"]["sha256"] for case in run["cases"]] == ["b" * 64, "c" * 64]
    assert config == before and not list(data["tmp_path"].rglob("*.ARW"))


@pytest.mark.parametrize("change", ["row_bytes", "row_raw_sha", "reference_sha", "history_sha",
    "history_elapsed", "history_index", "history_selected", "frame_index", "geometry_raw_sha", "geometry_family"])
def test_staged_baseline_and_original_native_guards_reject_changes(api, data, change):
    config = stage_configuration(data, "darsia-native-phase-statistics-run-v1")
    phase = config["phase"]
    if change in {"row_bytes", "row_raw_sha"}:
        path = Path(phase["reference"]["native_row"]["path"])
        row = json.loads(path.read_bytes())
        row["raw_acquisition"]["sha256"] = "d" * 64
        path.write_text(json.dumps(row), encoding="utf-8")
        if change == "row_raw_sha":
            phase["reference"]["native_row"] = descriptor(path)
            phase["initial_previous"]["native_row"] = descriptor(path)
    elif change == "reference_sha": phase["reference"]["source_sha256"] = "d" * 64
    elif change == "history_sha": phase["initial_previous"]["source"]["source_sha256"] = "d" * 64
    elif change == "history_elapsed": phase["initial_previous"]["source"]["elapsed_h"] = 0
    elif change == "history_index": phase["initial_previous"]["source"]["selection_index"] = 1
    elif change == "history_selected": phase["initial_previous"]["source"]["selected"] = True
    elif change == "frame_index": config["frames"][0]["source"]["selection_index"] = 5
    else:
        frame = config["frames"][0]
        path = Path(frame["native_task"]["geometry"]["path"])
        geometry = json.loads(path.read_bytes())
        if change == "geometry_raw_sha": geometry["acquisition"]["sha256"] = "d" * 64
        else: geometry["acquisition"]["path"] += ".jpg"
        path.write_text(json.dumps(geometry), encoding="utf-8")
        frame["native_task"]["geometry"] = descriptor(path)
    plan_path = data["tmp_path"] / "changed.toml"
    plan_path.write_text("changed plan", encoding="utf-8")
    plan = descriptor(plan_path)
    with pytest.raises(ValueError):
        api["validate_current_selection"](plan, series_root=data["series"], series_id="unit",
            selection_provider=lambda runs: [data["resolved"]],
            admitter=lambda *args, **kwargs: (dict(runs=[dict(run="run-one", configuration={})]), [config], plan))
