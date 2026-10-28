"""Finite real-pool scheduling regressions with metadata-only worker stubs.

These authored tests do not qualify a dataset, runtime, scientific stage, or
combined P1/P2 memory measurement. Setup readers and _run_one are explicit
stubs; the original run loop, memory rule, control writes, and real executor
lifetime are exercised. No test or thread has been run during source authoring.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest

from darsia.fluidflower.published_dataset import replay, resources, staged_runner
from darsia.fluidflower.published_dataset.reproduce import Settings


GIB = 1024 ** 3
WAIT_SECONDS = 10


class MetadataFailure(RuntimeError):
    pass


def _wait(event, message):
    assert event.wait(WAIT_SECONDS), message


class PausedWait(threading.Event):
    """A real Event exposing entry to the original post-pool paused wait."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.set()

    def wait(self, timeout=None):
        if not self.is_set():
            self.entered.set()
        return super().wait(timeout)


@pytest.fixture
def held_run(tmp_path, monkeypatch):
    """Construct one bounded coordinator and genuine pool per test case."""
    from darsia.fluidflower.published_dataset import staged_package, staged_runtime

    cases = []

    def create(*, runs=2, supplied=False, hold_constructor=False,
               hold_retirement=False, failure=None):
        names = ["r" + str(index + 1) for index in range(runs)]
        case = SimpleNamespace(
            available=5 * GIB, inventory={"memory": {"available_bytes": 5 * GIB}},
            observations=[], calls=[], pools=[], errors=[], live=0, maximum_live=0,
            mutex=threading.Lock(), stop=threading.Event(), finished=threading.Event(),
            constructor_entered=threading.Event(), allow_constructor=threading.Event(),
            shutdown_entered=threading.Event(), threads_retired=threading.Event(),
            allow_retirement=threading.Event(), entered={run: threading.Event() for run in names},
            release={run: threading.Event() for run in names}, visits={run: 0 for run in names})
        if not hold_constructor:
            case.allow_constructor.set()
        if not hold_retirement:
            case.allow_retirement.set()
        selected = staged_runner.StagedReplayRunner(inventory=case.inventory if supplied else None)
        selected._resume = PausedWait()
        case.selected = selected
        case.initial = Settings(workers=2, threads=1, remove_scratch=False,
                                stage_runtimes="explicit-metadata-fixture-runtimes.json")
        check = SimpleNamespace(root=tmp_path / "metadata-dataset", manifest={
            "machine": {"peak_memory_bytes": 2 * GIB},
            "identity": {"series_id": "metadata-pool-fixture"},
            "staged_reproduction": {"input_assets": []}})
        scope = SimpleNamespace(runs=names, frames={})

        def inventory():
            assert not supplied, "An explicit supplied inventory remains authoritative"
            with case.mutex:
                case.observations.append(case.available)
                return {"memory": {"available_bytes": case.available}}

        def forbidden_science(*args, **kwargs):
            raise AssertionError("A metadata pool fixture must never launch/probe scientific stages")

        # These are named metadata/setup boundaries, not successful scientific
        # or original RunIO admission substitutes. No input bytes are opened.
        monkeypatch.setattr(resources, "machine_inventory", inventory)
        monkeypatch.setattr(staged_package, "readback_staged", lambda *args: None)
        monkeypatch.setattr(selected, "_jobs", lambda check: {})
        monkeypatch.setattr(staged_runtime, "load_runtimes", lambda *args: {})
        monkeypatch.setattr(staged_runtime, "validate_runtime", forbidden_science)
        monkeypatch.setattr(staged_runtime, "launch_stage", forbidden_science)
        monkeypatch.setattr(staged_runner, "materialize_inputs",
                            lambda dataset, section, runs, destination: {"results": destination / "results"})
        monkeypatch.setattr(replay, "store_map_text", lambda *args: "# Metadata scheduling fixture only\n")

        class HeldPool(ThreadPoolExecutor):
            """Preserve real submit/futures/shutdown, exposing finite gates."""

            def __init__(self, *args, **kwargs):
                with case.mutex:
                    self.ordinal = len(case.pools) + 1
                    case.pools.append(kwargs["max_workers"])
                if self.ordinal == 1:
                    case.constructor_entered.set()
                    _wait(case.allow_constructor, "Constructor gate was not released")
                    if failure == "constructor":
                        raise MetadataFailure("metadata constructor failure")
                super().__init__(*args, **kwargs)

            def shutdown(self, wait=True, *, cancel_futures=False):
                if self.ordinal == 1:
                    case.shutdown_entered.set()
                super().shutdown(wait=wait, cancel_futures=cancel_futures)
                if self.ordinal == 1:
                    case.threads_retired.set()
                    _wait(case.allow_retirement, "Retirement gate was not released")

        def metadata_worker(run, *, workers):
            with case.mutex:
                case.visits[run] += 1
                visit = case.visits[run]
                case.calls.append((run, workers, visit))
                case.live += 1
                case.maximum_live = max(case.maximum_live, case.live)
            try:
                if visit == 1:
                    case.entered[run].set()
                    _wait(case.release[run], "Metadata worker gate was not released: " + run)
                    if failure == "future" and run == names[0]:
                        raise MetadataFailure("metadata future failure")
                if not case.stop.is_set() and not selected._resume.is_set():
                    return dict(run=run, state="paused", metadata_fixture=True)
                # Explicit bookkeeping only: no P2 output or publication exists.
                selected.outputs[run] = tmp_path / ("metadata-only-output-" + run)
                return dict(run=run, state="done", metadata_fixture=True,
                            scientific_qualification=False, production=False)
            finally:
                with case.mutex:
                    case.live -= 1

        def progress(states):
            if failure == "progress":
                raise MetadataFailure("metadata progress failure")

        monkeypatch.setattr(staged_runner, "ThreadPoolExecutor", HeldPool)
        monkeypatch.setattr(selected, "_run_one", metadata_worker)

        def coordinate():
            try:
                case.result = selected.run(check, tmp_path / "results", scope, case.initial, progress=progress)
            except BaseException as error:
                case.errors.append(error)
            finally:
                case.finished.set()

        case.thread = threading.Thread(target=coordinate, name="metadata-pool-coordinator", daemon=True)
        cases.append(case)
        case.thread.start()
        return case

    yield create
    for case in cases:
        # Release every gate even after assertion failure. Worker waits and the
        # coordinator join are finite; teardown never invokes a real stage.
        case.stop.set()
        case.allow_constructor.set()
        case.allow_retirement.set()
        for event in case.release.values():
            event.set()
        case.selected._resume.set()
        case.thread.join(WAIT_SECONDS)
        assert not case.thread.is_alive(), "Metadata coordinator did not retire"


def _memory(case, gib):
    with case.mutex:
        case.available = gib * GIB
        case.inventory["memory"]["available_bytes"] = gib * GIB


def _two_live(case):
    for run in ("r1", "r2"):
        _wait(case.entered[run], "Both original executor slots must actually start")
    with case.mutex:
        assert case.live == 2 and case.maximum_live == 2
    with case.selected._control_lock:
        assert case.selected._active_workers == 2


def _refuse_unchanged(case, proposed, *, required, available):
    selected = case.selected
    initial = selected.settings
    with selected._control_lock:
        active_workers = selected._active_workers
    before = selected.control.read_bytes(), selected.control.stat().st_mtime_ns
    assert not selected._resume.is_set()
    with pytest.raises(ValueError, match=f"{required}\\.0 GiB, but {available}\\.0 GiB"):
        selected.resume(proposed)
    assert selected.settings is initial and not selected._resume.is_set()
    with selected._control_lock:
        assert selected._active_workers == active_workers
    assert (selected.control.read_bytes(), selected.control.stat().st_mtime_ns) == before
    assert json.loads(before[0]) == {"action": "pause"}


def _finish(case, *, error=None):
    _wait(case.finished, "Metadata coordinator did not finish")
    case.thread.join(WAIT_SECONDS)
    assert not case.thread.is_alive()
    with case.selected._control_lock:
        assert case.selected._active_workers == 0
    if error is None:
        assert case.errors == []
        assert all(state == "done" for state in case.result.values())
    else:
        assert len(case.errors) == 1 and isinstance(case.errors[0], MetadataFailure)
        assert str(case.errors[0]) == "metadata " + error + " failure"


@pytest.mark.parametrize("supplied", [False, True])
def test_lower_resume_counts_both_live_slots_before_sufficient_memory_retry(held_run, supplied):
    case = held_run(supplied=supplied)
    _two_live(case)
    case.selected.pause()
    proposed = replace(case.initial, workers=1)
    _memory(case, 3)
    _refuse_unchanged(case, proposed, required=5, available=3)
    _two_live(case)
    assert case.selected.outputs == {} and case.selected.evidence is None
    _memory(case, 5)
    case.selected.resume(proposed)
    assert case.selected.settings is proposed and case.selected._resume.is_set()
    assert json.loads(case.selected.control.read_bytes()) == {"action": "run"}
    _two_live(case)  # A settings change did not resize the original pool.
    for event in case.release.values():
        event.set()
    _finish(case)
    assert case.pools == [2]
    assert case.observations == ([] if supplied else [5 * GIB, 3 * GIB, 5 * GIB])


def test_lower_resume_uses_one_slot_only_after_real_pool_retirement(held_run):
    case = held_run(hold_retirement=True)
    _two_live(case)
    case.selected.pause()
    proposed = replace(case.initial, workers=1)
    _memory(case, 3)
    for event in case.release.values():
        event.set()
    _wait(case.threads_retired, "Real shutdown must join both paused workers")
    with case.mutex:
        assert case.live == 0
    assert not case.finished.is_set()
    # Before __exit__ returns, even finished workers cannot erase the original
    # pool reservation. This orders Resume on the pre-retirement side of the race.
    _refuse_unchanged(case, proposed, required=5, available=3)
    case.allow_retirement.set()
    _wait(case.selected._resume.entered, "The old executor context must retire before paused waiting")
    with case.selected._control_lock:
        assert case.selected._active_workers == 0
    case.selected.resume(proposed)  # Post-retirement, 2 GiB + 1 GiB fits.
    _finish(case)
    assert case.pools == [2, 1]
    assert all(workers == 1 for _, workers, visit in case.calls if visit == 2)


@pytest.mark.parametrize("proposed_workers,required", [(2, 5), (3, 7)])
def test_live_pool_resume_checks_unchanged_and_increased_next_capacity(held_run, proposed_workers, required):
    case = held_run(runs=3)
    _two_live(case)
    assert not case.entered["r3"].is_set()  # The real two-slot pool owns a queued future too.
    case.selected.pause()
    proposed = replace(case.initial, workers=proposed_workers)
    _memory(case, required - 1)
    _refuse_unchanged(case, proposed, required=required, available=required - 1)
    _memory(case, required)
    case.selected.resume(proposed)
    for event in case.release.values():
        event.set()
    _finish(case)
    assert case.selected.settings is proposed
    assert case.pools == [2] and case.maximum_live == 2
    assert {run for run, _, _ in case.calls} == {"r1", "r2", "r3"}


@pytest.mark.parametrize("failed", [False, True])
def test_pool_capacity_is_reserved_before_constructor_and_cleared_after_failure(held_run, failed):
    case = held_run(hold_constructor=True, failure="constructor" if failed else None)
    _wait(case.constructor_entered, "The real pool construction boundary must be reached")
    case.selected.pause()
    _memory(case, 3)
    proposed = replace(case.initial, workers=1)
    with case.mutex:
        assert case.calls == []
    _refuse_unchanged(case, proposed, required=5, available=3)
    case.allow_constructor.set()
    if failed:
        _finish(case, error="constructor")
        assert case.calls == []
    else:
        _two_live(case)
        for event in case.release.values():
            event.set()
        _wait(case.selected._resume.entered, "Paused pool must retire before one-slot retry")
        case.selected.resume(proposed)
        _finish(case)
        assert case.pools == [2, 1]


@pytest.mark.parametrize("failure", ["future", "progress"])
def test_failed_batch_retains_capacity_while_real_shutdown_waits_for_other_worker(held_run, failure):
    case = held_run(failure=failure)
    _two_live(case)
    case.selected.pause()
    _memory(case, 3)
    case.release["r1"].set()
    _wait(case.shutdown_entered, "A failed future/progress callback must enter real executor shutdown")
    assert not case.threads_retired.is_set() and not case.finished.is_set()
    with case.mutex:
        assert case.live == 1
    _refuse_unchanged(case, replace(case.initial, workers=1), required=5, available=3)
    case.release["r2"].set()
    _finish(case, error=failure)
    assert case.pools == [2]
