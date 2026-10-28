"""Staged dispatch retains measured-memory admission; no scientific execution."""
from types import SimpleNamespace
from dataclasses import replace
import json

import pytest

from darsia.fluidflower.published_dataset import replay, resources
from darsia.fluidflower.published_dataset.reproduce import Settings


GIB = 1024 ** 3


def _preflight(tmp_path, monkeypatch, *, peak=2 * GIB, available=4 * GIB,
               workers=2, runs=("r1", "r2"), staged_errors=()):
    inventory = {"memory": {"available_bytes": available}}
    selected = replay.ReplayRunner(inventory=inventory)
    check = SimpleNamespace(root=tmp_path / "dataset", manifest={
        "staged_reproduction": {"format": "staged-preflight-boundary"},
        "machine": {"peak_memory_bytes": peak}})
    scope = SimpleNamespace(runs=list(runs), frames={})
    settings = Settings(workers=workers, threads=1)
    calls = []

    class StagedBoundary:
        def preflight(self, results, **arguments):
            calls.append((results, arguments))
            return list(staged_errors)

    monkeypatch.setattr(selected, "_staged", lambda: StagedBoundary())
    monkeypatch.setattr(resources, "machine_inventory", lambda: pytest.fail(
        "A supplied inventory must remain the memory admission authority"))
    problems = selected.preflight(tmp_path / "results", check=check, scope=scope, settings=settings)
    assert calls == [(tmp_path / "results", dict(check=check, scope=scope, settings=settings))]
    assert not (tmp_path / "results").exists()
    return problems


@pytest.mark.parametrize("available, refused", [(4 * GIB, True), (5 * GIB, False), (6 * GIB, False)])
def test_staged_preflight_enforces_original_recorded_peak_and_launcher_reserve(
        tmp_path, monkeypatch, available, refused):
    problems = _preflight(tmp_path, monkeypatch, available=available)
    expected = replay.memory_problems({"machine": {"peak_memory_bytes": 2 * GIB}}, 2,
        {"memory": {"available_bytes": available}}, reserve_bytes=resources.LAUNCHER_RESERVE_BYTES)
    assert problems == expected
    assert bool(problems) is refused
    if refused:
        assert problems == [
            "Memory: 2 worker(s) × 2.0 GiB measured on the reference machine + 1 GiB reserve = "
            "5.0 GiB, but 4.0 GiB is available now. Choose fewer workers or close other programs."]


def test_staged_memory_uses_only_the_selected_concurrent_run_count(tmp_path, monkeypatch):
    assert _preflight(tmp_path, monkeypatch, workers=20, runs=("r1",), available=3 * GIB) == []


@pytest.mark.parametrize("peak, available", [(None, GIB), (2 * GIB, None)])
def test_staged_memory_keeps_unknown_measurements_unknown(tmp_path, monkeypatch, peak, available):
    # A missing peak does not acquire an invented P1+P2 estimate, and a missing
    # available-memory observation does not become a zero-memory measurement.
    assert _preflight(tmp_path, monkeypatch, peak=peak, available=available) == []


def test_memory_refusal_preserves_independent_staged_admission_refusals(tmp_path, monkeypatch):
    problems = _preflight(tmp_path, monkeypatch, staged_errors=("Original stage runtime is missing",))
    assert len(problems) == 2 and problems[0].startswith("Memory:")
    assert problems[1] == "Original stage runtime is missing"


def test_staged_memory_reads_current_inventory_when_none_was_supplied(tmp_path, monkeypatch):
    calls = []

    def inventory():
        calls.append("current inventory")
        return {"memory": {"available_bytes": GIB}}

    selected = replay.ReplayRunner()
    monkeypatch.setattr(resources, "machine_inventory", inventory)
    monkeypatch.setattr(selected, "_staged", lambda: SimpleNamespace(preflight=lambda *a, **k: []))
    problems = selected.preflight(tmp_path / "results", check=SimpleNamespace(manifest={
        "staged_reproduction": {"format": "staged-preflight-boundary"},
        "machine": {"peak_memory_bytes": 2 * GIB}}),
        scope=SimpleNamespace(runs=["r1"], frames={}), settings=Settings(workers=1, threads=1))
    assert calls == ["current inventory"]
    assert len(problems) == 1 and problems[0].startswith("Memory:")
    assert not (tmp_path / "results").exists()


def _paused_metadata(tmp_path, *, peak=2 * GIB, inventory=None):
    """Seed a paused control boundary, not an admitted or executed dataset."""
    from darsia.fluidflower.published_dataset.staged_runner import StagedReplayRunner

    selected = StagedReplayRunner(inventory=inventory)
    selected.check = SimpleNamespace(root=tmp_path / "dataset",
        manifest={"machine": {"peak_memory_bytes": peak}})
    selected.scope = SimpleNamespace(runs=["r1", "r2"], frames={})
    initial = Settings(workers=1, threads=1, stage_runtimes="explicit-runtimes.json")
    selected.settings = initial
    selected.control = tmp_path / "staged-control.json"
    selected.pause()
    return selected, initial


def test_direct_staged_resume_rechecks_recorded_memory_before_any_mutation(tmp_path, monkeypatch):
    from darsia.fluidflower.published_dataset import staged_runner

    inventory = {"memory": {"available_bytes": 4 * GIB}}
    selected, initial = _paused_metadata(tmp_path, inventory=inventory)
    before = selected.control.read_bytes(), selected.control.stat().st_mtime_ns
    monkeypatch.setattr(resources, "machine_inventory", lambda: pytest.fail("Keep the supplied inventory"))
    monkeypatch.setattr(staged_runner, "ThreadPoolExecutor", lambda *a, **k: pytest.fail("No pool in this fixture"))
    assert replay.memory_problems(selected.check.manifest, 1, inventory,
        reserve_bytes=resources.LAUNCHER_RESERVE_BYTES) == []
    with monkeypatch.context() as mutation:
        mutation.setattr(replay, "_write_json", lambda *a, **k: pytest.fail("Refusal must not write control"))
        mutation.setattr(selected._resume, "set", lambda: pytest.fail("Refusal must not wake execution"))
        with pytest.raises(ValueError, match="5.0 GiB, but 4.0 GiB"):
            selected.resume(replace(initial, workers=2))
    assert selected.settings is initial and selected.inventory is inventory
    assert not selected._resume.is_set()
    assert (selected.control.read_bytes(), selected.control.stat().st_mtime_ns) == before
    assert selected.outputs == {} and selected.evidence is None

    retry = replace(initial, workers=1)
    selected.resume(retry)
    assert selected.settings is retry and selected._resume.is_set()
    assert json.loads(selected.control.read_bytes()) == {"action": "run"}


def test_resume_counts_only_pending_selected_runs(tmp_path, monkeypatch):
    selected, initial = _paused_metadata(tmp_path, inventory={"memory": {"available_bytes": 3 * GIB}})
    # Explicit completion metadata only; no output or scientific run is created.
    selected.outputs["r1"] = tmp_path / "completed-r1-output"
    monkeypatch.setattr(resources, "machine_inventory", lambda: pytest.fail("Keep supplied inventory"))
    selected.resume(replace(initial, workers=2))
    assert selected._resume.is_set() and selected.settings.workers == 2
    assert json.loads(selected.control.read_bytes()) == {"action": "run"}


@pytest.mark.parametrize("peak, inventory", [
    (None, {"memory": {"available_bytes": GIB}}),
    (2 * GIB, {"memory": {"available_bytes": None}}),
    (2 * GIB, {}),
])
def test_resume_preserves_unknown_memory_observations(tmp_path, monkeypatch, peak, inventory):
    selected, initial = _paused_metadata(tmp_path, peak=peak, inventory=inventory)
    monkeypatch.setattr(resources, "machine_inventory", lambda: pytest.fail("Keep supplied inventory"))
    selected.resume(replace(initial, workers=2))
    assert selected._resume.is_set() and selected.settings.workers == 2
    assert selected.inventory is inventory
    assert selected.check.manifest["machine"]["peak_memory_bytes"] is peak


def test_resume_without_supplied_inventory_uses_current_memory_for_each_attempt(tmp_path, monkeypatch):
    selected, initial = _paused_metadata(tmp_path)
    observations = iter((4 * GIB, 5 * GIB))
    calls = []

    def inventory():
        available = next(observations)
        calls.append(available)
        return {"memory": {"available_bytes": available}}

    monkeypatch.setattr(resources, "machine_inventory", inventory)
    proposed = replace(initial, workers=2)
    with pytest.raises(ValueError, match="5.0 GiB, but 4.0 GiB"):
        selected.resume(proposed)
    assert selected.settings is initial and not selected._resume.is_set()
    selected.resume(proposed)
    assert selected.settings is proposed and selected._resume.is_set()
    assert calls == [4 * GIB, 5 * GIB]


def test_early_resume_cannot_bypass_memory_admission_at_run_setup(tmp_path, monkeypatch):
    from darsia.fluidflower.published_dataset import staged_package, staged_runner

    selected = staged_runner.StagedReplayRunner(inventory={"memory": {"available_bytes": 4 * GIB}})
    initial = Settings(workers=1, threads=1, stage_runtimes="explicit-runtimes.json")
    proposed = replace(initial, workers=2)
    selected.pause()
    # Before any preflight/setup there is no measured context; this choice is
    # provisional. The actual run context must refuse before its first write.
    selected.resume(proposed)
    monkeypatch.setattr(resources, "machine_inventory", lambda: pytest.fail("Keep supplied inventory"))
    monkeypatch.setattr(replay, "_write_json", lambda *a, **k: pytest.fail("Memory refusal precedes writes"))
    monkeypatch.setattr(staged_package, "readback_staged", lambda *a, **k: pytest.fail("Memory refusal precedes setup"))
    monkeypatch.setattr(staged_runner, "ThreadPoolExecutor", lambda *a, **k: pytest.fail("No pool in this fixture"))
    with pytest.raises(ValueError, match="5.0 GiB, but 4.0 GiB"):
        selected.run(SimpleNamespace(manifest={"machine": {"peak_memory_bytes": 2 * GIB}}),
            tmp_path / "results", SimpleNamespace(runs=["r1", "r2"]), initial)
    assert selected.settings is proposed
    assert selected.check is None and selected.scope is None and selected.results is None and selected.control is None
    assert selected.outputs == {} and selected.evidence is None
    assert not (tmp_path / "results").exists()


def test_preflight_context_checks_early_resume_and_never_replaces_active_context(tmp_path, monkeypatch):
    """Stub unrelated admission boundaries; test only the context handoff."""
    from darsia.fluidflower.published_dataset import staged_package, staged_runner, staged_runtime

    selected = staged_runner.StagedReplayRunner(inventory={"memory": {"available_bytes": 4 * GIB}})
    check = SimpleNamespace(root=tmp_path / "dataset", manifest={
        "machine": {"peak_memory_bytes": 2 * GIB}, "staged_reproduction": {"input_assets": []}})
    scope = SimpleNamespace(runs=["r1", "r2"], frames={})
    initial = Settings(workers=1, threads=1, stage_runtimes="explicit-runtimes.json")
    monkeypatch.setattr(staged_package, "validate_staged", lambda manifest: None)
    monkeypatch.setattr(selected, "_jobs", lambda check: {})
    monkeypatch.setattr(staged_runtime, "load_runtimes", lambda *a, **k: {})
    monkeypatch.setattr(staged_runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=GIB))
    monkeypatch.setattr(resources, "machine_inventory", lambda: pytest.fail("Keep supplied inventory"))
    assert selected.preflight(tmp_path / "results", check=check, scope=scope, settings=initial) == []
    assert selected.check is check and selected.scope is scope
    assert selected.settings is None and selected.control is None
    selected.pause()
    with pytest.raises(ValueError, match="5.0 GiB, but 4.0 GiB"):
        selected.resume(replace(initial, workers=2))
    assert selected.settings is None and selected.control is None and not selected._resume.is_set()
    assert not (tmp_path / "results").exists()

    # A later metadata preflight must not replace the context of an active run.
    selected.control = tmp_path / "staged-control.json"
    selected.pause()
    another = SimpleNamespace(root=tmp_path, manifest={"staged_reproduction": {"input_assets": []}})
    assert selected.preflight(tmp_path / "another-results", check=another,
        scope=SimpleNamespace(runs=["other"], frames={}), settings=initial) == []
    assert selected.check is check and selected.scope is scope
    with pytest.raises(ValueError, match="5.0 GiB, but 4.0 GiB"):
        selected.resume(replace(initial, workers=2))
    assert not selected._resume.is_set()


def test_new_run_cannot_reuse_prior_completion_to_reduce_resume_memory(tmp_path, monkeypatch):
    """Stop at setup metadata; no stage, fixture input or worker is executed."""
    from darsia.fluidflower.published_dataset import staged_package, staged_runner

    selected, initial = _paused_metadata(tmp_path, inventory={"memory": {"available_bytes": 4 * GIB}})
    previous_outputs = {"r1": tmp_path / "prior-run-completed-output"}
    previous_evidence = {"fixture": "prior-run-metadata-only"}
    selected.outputs, selected.evidence = previous_outputs, previous_evidence
    check, scope = selected.check, selected.scope
    before = selected.control.read_bytes(), selected.control.stat().st_mtime_ns
    monkeypatch.setattr(resources, "machine_inventory", lambda: pytest.fail("Keep supplied inventory"))
    monkeypatch.setattr(staged_runner, "ThreadPoolExecutor", lambda *a, **k: pytest.fail("No worker in this fixture"))

    with monkeypatch.context() as mutation:
        mutation.setattr(replay, "_write_json", lambda *a, **k: pytest.fail("Refused setup must not write"))
        with pytest.raises(ValueError, match="5.0 GiB, but 4.0 GiB"):
            selected.run(check, tmp_path / "new-results", scope, replace(initial, workers=2))
    assert selected.outputs is previous_outputs and selected.evidence is previous_evidence
    assert selected.settings is initial and not selected._resume.is_set()
    assert (selected.control.read_bytes(), selected.control.stat().st_mtime_ns) == before

    class MetadataBoundary(Exception):
        pass

    def stop_before_runtime_setup(*args):
        assert selected.outputs == {} and selected.evidence is None
        assert selected.settings is initial and not selected._resume.is_set()
        control_before = selected.control.read_bytes(), selected.control.stat().st_mtime_ns
        with pytest.raises(ValueError, match="5.0 GiB, but 4.0 GiB"):
            selected.resume(replace(initial, workers=2))
        assert selected.settings is initial and not selected._resume.is_set()
        assert (selected.control.read_bytes(), selected.control.stat().st_mtime_ns) == control_before
        raise MetadataBoundary()

    monkeypatch.setattr(staged_package, "readback_staged", stop_before_runtime_setup)
    with pytest.raises(MetadataBoundary):
        selected.run(check, tmp_path / "new-results", scope, initial)
    assert selected.control == (tmp_path / "new-results/staged-control.json").resolve()
    assert json.loads(selected.control.read_bytes()) == {"action": "pause"}
