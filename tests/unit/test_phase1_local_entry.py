"""Finite local-entry routing tests; no RAW decoding or science completion.

The shared frame executor is a stop sentinel in these tests. An uninitialized
LocalBuildInputs object is used only to verify argument routing; it never
validates physical inputs or grants a successful Build/P1 result.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import phase_statistics as stage
from darsia.fluidflower.pipeline.job_queue import descriptor
from .test_staged_data import phase1_data


class _ReachedSharedFrames(RuntimeError):
    pass


@pytest.fixture
def local_route(monkeypatch):
    from darsia.fluidflower.pipeline.local_build import LocalBuildInputs
    local_build = object.__new__(LocalBuildInputs)
    calls = {}

    def bind_configuration(self, config):
        assert self is local_build and "current_build_binding" not in config
        result = deepcopy(config)
        result.setdefault("provenance", {})["local_entry_routing_test"] = True
        calls["bound_config"] = deepcopy(result)
        return result

    def runtime(threads, run):
        calls["threads"] = threads
        return run({"routing_test_only": True})

    def frames(*args, **kwargs):
        calls["args"], calls["kwargs"] = args, kwargs
        raise _ReachedSharedFrames("stopped before any scientific frame")

    def no_worker(*args, **kwargs):
        raise AssertionError("Local entry reached worker ownership acquisition")

    monkeypatch.setattr(stage, "_phase1_thread_runtime", runtime)
    monkeypatch.setattr(stage, "_phase1_frames", frames)
    monkeypatch.setattr(stage, "_task_configuration", no_worker)
    monkeypatch.setattr(LocalBuildInputs, "bind_configuration", bind_configuration)
    return local_build, calls


def test_local_entry_saves_actual_config_and_reaches_the_shared_loop_without_alias_task(
        phase1_data, tmp_path, local_route):
    local_build, calls = local_route
    # This historical declaration is not an alternate local source of authority.
    phase1_data["current_build_binding"] = {"old_alias_declaration": True}
    original = deepcopy(phase1_data)
    with pytest.raises(_ReachedSharedFrames, match="before any scientific frame"):
        stage.run_phase1_local(phase1_data, tmp_path, local_build=local_build,
                              code_commit="1" * 40, threads=3)
    assert phase1_data == original
    expected = deepcopy(original)
    expected.pop("current_build_binding")
    expected["provenance"]["local_entry_routing_test"] = True
    config, task, output, backend, precision, progress, pin, manifest_pin, *_ = calls["args"]
    assert config == expected and output == tmp_path
    assert calls["bound_config"] == expected
    assert backend == "cpu" and precision == "float32" and progress is None
    assert calls["threads"] == 3
    assert calls["kwargs"] == {"local_build": local_build, "unc_only": False}
    assert task["run"] == config["run"] and task["commit"] == "1" * 40
    assert task["threads"] == 3 and task["precision"] == "float32"
    assert set(task) == {"run", "precision", "threads", "commit", "execution_id"}
    assert len(task["execution_id"]) == 32
    assert pin == descriptor(tmp_path / "phase1_configuration.json")
    assert json.loads(Path(pin["path"]).read_bytes()) == expected
    assert manifest_pin == descriptor(tmp_path / "manifest.json")
    manifest = json.loads(Path(manifest_pin["path"]).read_bytes())
    assert manifest["format"] == stage.MANIFEST_FORMAT
    assert manifest["execution_mode"] == "local"
    assert manifest["execution_id"] == task["execution_id"]
    assert manifest["configuration"] == pin
    assert manifest["frames"] == {row["case_id"]: row for row in expected["frames"]}
    assert not {"job_id", "stage_identity", "native_code_admission"} & manifest.keys()
    assert not (tmp_path / "phase1_completed.json").exists()


@pytest.mark.parametrize("bad", [None, "", "not-a-commit", "A" * 40, "1" * 39])
def test_local_entry_requires_explicit_code_attribution_before_output(
        phase1_data, tmp_path, local_route, bad):
    local_build, calls = local_route
    with pytest.raises(ValueError, match="code commit attribution"):
        stage.run_phase1_local(phase1_data, tmp_path / "new", local_build=local_build,
                              code_commit=bad)
    assert not calls and not (tmp_path / "new").exists()


@pytest.mark.parametrize("bad", [None, object()])
def test_local_entry_cannot_fall_back_to_legacy_physical_inputs(
        phase1_data, tmp_path, local_route, bad):
    _, calls = local_route
    with pytest.raises(TypeError, match="explicit LocalBuildInputs"):
        stage.run_phase1_local(phase1_data, tmp_path / "new", local_build=bad,
                              code_commit="1" * 40)
    assert not calls and not (tmp_path / "new").exists()


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_local_thread_count_is_explicit_before_output(
        phase1_data, tmp_path, local_route, threads):
    local_build, calls = local_route
    with pytest.raises(ValueError, match="Positive internal thread count"):
        stage.run_phase1_local(phase1_data, tmp_path / "new", local_build=local_build,
                              code_commit="1" * 40, threads=threads)
    assert not calls and not (tmp_path / "new").exists()


def test_local_precision_and_original_selection_checks_remain_active(
        phase1_data, tmp_path, local_route):
    local_build, calls = local_route
    with pytest.raises(ValueError, match="configuration and precision differ"):
        stage.run_phase1_local(phase1_data, tmp_path / "new", local_build=local_build,
                              code_commit="1" * 40, precision="float64")
    changed = deepcopy(phase1_data)
    changed["frames"][0]["source"]["previous_source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="predecessor"):
        stage.run_phase1_local(changed, tmp_path / "new", local_build=local_build,
                              code_commit="1" * 40)
    assert set(calls) == {"bound_config"} and not (tmp_path / "new").exists()


def test_local_entry_preserves_existing_output_before_shared_frames(
        phase1_data, tmp_path, local_route):
    local_build, calls = local_route
    previous = tmp_path / "previous.txt"
    previous.write_bytes(b"retain previous output")
    with pytest.raises(FileExistsError, match="must be fresh"):
        stage.run_phase1_local(phase1_data, tmp_path, local_build=local_build,
                              code_commit="1" * 40)
    assert previous.read_bytes() == b"retain previous output"
    assert "args" not in calls
    assert list(tmp_path.iterdir()) == [previous]


def test_worker_and_local_entry_share_one_frame_loop_and_original_commit_writer():
    tree = ast.parse(Path(stage.__file__).read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    calls = lambda node, name: [call for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name]
    assert len(calls(functions["_run_phase1"], "_phase1_frames")) == 1
    assert len(calls(functions["_run_phase1_local"], "_phase1_frames")) == 1
    loop = functions["_phase1_frames"]
    assert len(calls(loop, "infer_labels")) == 1
    assert len(calls(loop, "_commit_phase1")) == 1
    commit = calls(loop, "_commit_phase1")[0]
    advance = [call for call in ast.walk(loop) if isinstance(call, ast.Call)
               and isinstance(call.func, ast.Attribute) and call.func.attr == "advance"]
    assert len(advance) == 1 and commit.lineno < advance[0].lineno
    assert len(calls(functions["_commit_phase1"], "_frame_record")) == 1
