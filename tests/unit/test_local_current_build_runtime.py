"""Actual-module DATA/forwarding fixtures, without jobs, RAW or acceptance.

Reader/publication/physical loading and active RunIO are explicit inert seams.
The private binding, acquisition and shared frame comparison execute unchanged.
Only darsia's top-level package bootstrap is bypassed to avoid live cache IO.
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src"
# Dependencies come from this repository's own source tree.
BASE = SOURCE


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(BASE))
    package = ModuleType("darsia")
    package.__path__ = [str(BASE / "darsia")]
    package.__spec__ = importlib.machinery.ModuleSpec("darsia", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "darsia", package)

    def load(name, path):
        parent_name, _, child = name.rpartition(".")
        parent = importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        monkeypatch.setattr(parent, child, module, raising=False)
        assert Path(module.__file__).resolve() == path.resolve()
        return module

    schema = load("darsia_cache.current_build_binding", SOURCE / "darsia_cache/current_build_binding.py")
    reuse = load("darsia.fluidflower.local_build_reuse", SOURCE / "darsia/fluidflower/local_build_reuse.py")
    current = load("darsia.fluidflower.pipeline.current_build", SOURCE / "darsia/fluidflower/pipeline/current_build.py")
    return SimpleNamespace(schema=schema, reuse=reuse, current=current)


def declaration(schema):
    def descriptor(name):
        return dict(path="C:/TESTONLY/" + name, sha256="a" * 64, size_bytes=7)
    paths = {"job": "job.json", "task": "results/ac52/task.json", "done": "done/ac52.json",
             "operation_result": "publishing/ac52/metadata/operation_result.json"}
    publication = dict(root="C:/TESTONLY/publication", **{
        key: descriptor("publication/" + path) for key, path in paths.items()})
    publication["readset"] = sorted([
        dict(relative_path=path, descriptor=deepcopy(publication[key])) for key, path in paths.items()
    ], key=lambda row: row["relative_path"])
    frames, specs = [], []
    for i in (1, 2):
        raw, geometry = descriptor(f"f{i}.ARW"), descriptor(f"geometry{i}.json")
        case = f"ac52_f{i}"
        frames.append(dict(case_id=case, raw_acquisition=raw, geometry=geometry,
            bridge_frame=descriptor(f"frame{i}.json"), area_provenance=descriptor(f"area{i}.json"),
            static_provenance=descriptor(f"static{i}.json")))
        specs.append(dict(case_id=case, frame_index=i * 7, elapsed_h=.497 + i,
            source=dict(source_sha256=raw["sha256"], source_binding={"raw_acquisition": deepcopy(raw)}),
            geometry=deepcopy(geometry), phase_roles={"original_role": i}, native_task={"kind": "raw_rig"}))
    binding = dict(format=schema.FORMAT, run="ac52", publication=publication,
        operation_result=descriptor("owned/operation_result.json"), bridge=descriptor("bridge.json"), frames=frames)
    config = dict(format="darsia-native-phase-statistics-run-v1", run="ac52", statistics={},
        native={"configuration": descriptor("native.json")}, frames=specs, current_build_binding=binding)
    return binding, config, descriptor


def test_default_binding_unchanged_and_optional_reuse_is_a_named_pin(modules):
    schema = modules.schema
    binding, config, descriptor = declaration(schema)
    before = deepcopy((binding, config))
    assert schema.checked_current_build_binding(binding, config=config) == binding
    default_inputs = list(schema.current_build_binding_inputs(binding))
    assert all(use != "current_build.local_reuse.record" for use, _, _ in default_inputs)
    assert (binding, config) == before
    binding["local_build_reuse"] = descriptor("reuse.json")
    checked = schema.checked_current_build_binding(binding, config=config)
    assert checked == binding and checked is not binding
    inputs = list(schema.current_build_binding_inputs(binding))
    assert [row for row in inputs if row[0] == "current_build.local_reuse.record"] == [
        ("current_build.local_reuse.record", binding["local_build_reuse"], "configuration")]
    assert [row for row in inputs if row[0] != "current_build.local_reuse.record"] == default_inputs
    checked["local_build_reuse"]["path"] = "C:/TESTONLY/changed.json"
    assert binding["local_build_reuse"]["path"].endswith("/reuse.json")


@pytest.mark.parametrize("fault", ["none", "path", "sha", "size_bool", "extra"])
def test_optional_reuse_descriptor_remains_closed_and_exact(modules, fault):
    binding, config, descriptor = declaration(modules.schema)
    reuse = descriptor("reuse.json")
    if fault == "none":
        reuse = None
    elif fault == "path":
        reuse["path"] = "relative.json"
    elif fault == "sha":
        reuse["sha256"] = "A" * 64
    elif fault == "size_bool":
        reuse["size_bytes"] = True
    else:
        reuse["accepted"] = True
    binding["local_build_reuse"] = reuse
    with pytest.raises(ValueError):
        modules.schema.checked_current_build_binding(binding, config=config)


@pytest.fixture
def runtime(modules, monkeypatch):
    binding, config, descriptor = declaration(modules.schema)
    current = modules.current
    run_io = importlib.import_module("darsia.fluidflower.pipeline.run_io")
    native = importlib.import_module("darsia.fluidflower.mass.native_optical_source")
    io = run_io.RunIOBinding.__new__(run_io.RunIOBinding)
    io.configuration = deepcopy(config)
    task = {"run": "ac52", "fixture_task": True}
    monkeypatch.setattr(run_io, "for_task", lambda value: io)
    events, docs = [], {}
    receipt = dict(baseline_renderer=descriptor("renderer.json"), setup_plan=descriptor("setup.json"))
    matrix = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    docs[binding["operation_result"]["path"]] = receipt
    docs[receipt["baseline_renderer"]["path"]] = dict(outputs=[], H_rig_corrected_native_to_baseline_camera_u=matrix)
    docs[receipt["setup_plan"]["path"]] = {"adopted_revision": descriptor("revision.json")}
    docs[descriptor("revision.json")["path"]] = {"configuration": {}}
    docs[binding["bridge"]["path"]] = dict(format="darsia-build-native-bridge-v1",
        operation_result=deepcopy(binding["operation_result"]), native_configuration=deepcopy(config["native"]["configuration"]),
        saved_build_to_camera_u_h=matrix)
    frame_binding = descriptor("separate_raw_phase_frames.json")
    frame_data = dict(format="darsia-build-raw-phase-inputs-v1", run=config["run"],
                      native=deepcopy(config["native"]), frames=deepcopy(config["frames"]))
    docs[frame_binding["path"]] = frame_data
    physical = SimpleNamespace(provenance={})

    class Reader:
        in_direct = False
        def __init__(self, actual_io, actual_task):
            assert actual_io is io and actual_task is task
            events.append("reader")
        def json(self, pin):
            events.append(("read", pin["path"]))
            return deepcopy(docs[pin["path"]])
        def path(self, pin, *, original=False):
            events.append(("path", pin["path"], original))
            return Path(pin["path"])
        def recheck(self):
            events.append("recheck")
        @contextmanager
        def direct(self):
            self.in_direct = True
            try:
                yield
            finally:
                self.in_direct = False

    class Publication:
        def __init__(self, declaration, reader):
            assert declaration == binding["publication"]
        def verify(self, declaration):
            events.append("publication")
            return dict(operation="build", run="ac52", files={
                "operation_result.json": deepcopy(binding["operation_result"])}), deepcopy(receipt)

    def load_physical(operation, *, reader, local_reuse=None):
        assert operation == binding["operation_result"] and reader.in_direct
        events.append(("physical", deepcopy(local_reuse)))
        return physical

    monkeypatch.setattr(current, "_Reader", Reader)
    monkeypatch.setattr(current, "_Publication", Publication)
    monkeypatch.setattr(native, "load_build_physical_context", load_physical)

    def enable_reuse():
        binding["local_build_reuse"] = descriptor("reuse.json")
        physical.provenance["local_build_reuse"] = {"frame_configuration": frame_binding}
        io.configuration = deepcopy(config)

    return SimpleNamespace(current=current, config=config, task=task, io=io, events=events,
        physical=physical, binding=binding, frame_data=frame_data, enable=enable_reuse,
        frame_binding=frame_binding, run_io=run_io, docs=docs, descriptor=descriptor)


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("phase1", [False, True], ids=["n1", "p1"])
def test_actual_acquire_keeps_default_and_forwards_optional_reuse(runtime, reuse, phase1):
    r = runtime
    if not phase1:
        r.config.pop("statistics")
        r.config.update(format="darsia-native-phase-n1-run-v1", mass={"template": {}})
        r.io.configuration = deepcopy(r.config)
    if reuse:
        r.enable()
    before = deepcopy((r.config, r.task))
    acquired = r.current.acquire_current_build(r.config, r.task, r.io)
    assert isinstance(acquired, r.current.CheckedCurrentBuild)
    assert acquired._physical is r.physical
    assert (r.config, r.task) == before
    assert ("physical", r.binding.get("local_build_reuse")) in r.events
    shared_read = ("read", r.frame_binding["path"])
    assert (shared_read in r.events) is reuse
    assert r.events[-1] == "recheck"
    assert acquired._config == r.config and acquired._config is not r.config


@pytest.mark.parametrize("fault", ["elapsed", "index", "source", "geometry", "roles",
                                    "order", "missing", "native", "enclosing_format"])
def test_actual_acquire_rejects_queued_vs_separate_frame_data_before_bridge(runtime, fault):
    r = runtime
    r.enable()
    frame = r.frame_data["frames"][0]
    if fault == "elapsed":
        frame["elapsed_h"] = .5
    elif fault == "index":
        frame["frame_index"] = 0
    elif fault == "source":
        frame["source"]["source_sha256"] = "f" * 64
    elif fault == "geometry":
        frame["geometry"]["sha256"] = "f" * 64
    elif fault == "roles":
        frame["phase_roles"]["original_role"] = 99
    elif fault == "order":
        r.frame_data["frames"].reverse()
    elif fault == "missing":
        del frame["elapsed_h"]
    elif fault == "native":
        r.frame_data["native"]["configuration"]["sha256"] = "f" * 64
    else:
        r.frame_data["format"] = r.config["format"]
    before = deepcopy(r.config)
    with pytest.raises(ValueError, match="Local Build reuse"):
        r.current.acquire_current_build(r.config, r.task, r.io)
    assert r.config == before
    assert ("physical", r.binding["local_build_reuse"]) in r.events
    assert ("read", r.binding["bridge"]["path"]) not in r.events


@pytest.mark.parametrize("fault", ["inactive", "replayed_configuration"])
def test_actual_acquire_refuses_before_readers_when_runio_differs(runtime, monkeypatch, fault):
    r = runtime
    r.enable()
    if fault == "inactive":
        monkeypatch.setattr(r.run_io, "for_task", lambda task: None)
    else:
        r.io.configuration["frames"][0]["elapsed_h"] = .5
    with pytest.raises(ValueError, match="actual replayed configuration"):
        r.current.acquire_current_build(r.config, r.task, r.io)
    assert r.events == []


@pytest.mark.parametrize("case", ["ordinary", "later_annotation", "exact_annotation",
                                  "changed_numeric", "wrong_annotation"])
def test_actual_phase_join_preserves_original_context_and_only_allows_later_annotation(
        runtime, monkeypatch, case):
    r = runtime
    r.enable()
    checked = r.current.acquire_current_build(r.config, r.task, r.io)
    saved = {"native_configuration": r.config["native"]["configuration"],
             "physical_grid": {"voxel_size": [.001, .001], "depth": .02}}
    annotation = {"record": r.binding["local_build_reuse"], "scope": "local-p1-selection-only"}
    current = deepcopy(saved)
    if case != "ordinary":
        current["local_build_reuse"] = deepcopy(annotation)
    if case in {"exact_annotation", "wrong_annotation"}:
        saved["local_build_reuse"] = deepcopy(annotation)
    if case == "changed_numeric":
        current["physical_grid"]["depth"] = .03
    if case == "wrong_annotation":
        saved["local_build_reuse"]["record"]["sha256"] = "f" * 64
    checked._physical.provenance = current
    spec = checked._config["frames"][0]
    declared = checked._declaration["frames"][0]
    native_geometry, raw_binding = r.descriptor("native_geometry.json"), r.descriptor("raw_binding.json")
    spec["native_task"].update(geometry=native_geometry, processor={"raw_binding": raw_binding})
    spec["phase_roles"]["area"] = {"role": "measured_parent", "factor": 1.}
    r.docs[declared["bridge_frame"]["path"]] = dict(bridge=checked._declaration["bridge"],
        physical_context=saved, native_frame=native_geometry, raw_source_binding=raw_binding)
    inputs = {key: r.descriptor(key + ".json") for key in (
        "measurement", "physical_scale_plan", "physical_scale_receipt",
        "measured_frame_receipt", "historical_frame_receipt")}
    r.docs[declared["area_provenance"]["path"]] = dict(area=deepcopy(spec["phase_roles"]["area"]), inputs=inputs)
    r.docs[declared["static_provenance"]["path"]] = {}
    for key in ("historical_frame_receipt", "measured_frame_receipt"):
        r.docs[inputs[key]["path"]] = {"rig_domain_revision": {"area_map": r.descriptor(key + "_area.npz")}}
    reached = []
    class OriginalAreaConsumerReached(Exception):
        pass
    def stop_before_numerical_area(**kwargs):
        reached.append(kwargs)
        raise OriginalAreaConsumerReached
    area_module = importlib.import_module("darsia.fluidflower.pipeline.build_area_data")
    monkeypatch.setattr(area_module, "_read_measured_area_context", stop_before_numerical_area)
    before = deepcopy((saved, current))
    if case in {"changed_numeric", "wrong_annotation"}:
        with pytest.raises(ValueError, match="RAW/area receipt differs"):
            checked.check_phase(spec, {}, {})
        assert reached == []
    else:
        with pytest.raises(OriginalAreaConsumerReached):
            checked.check_phase(spec, {}, {})
        assert len(reached) == 1
    assert (saved, current) == before
