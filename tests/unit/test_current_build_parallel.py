"""Independent spawned test runs; no campaign, RAW decoder or scientific job."""
from contextlib import nullcontext, contextmanager
from copy import deepcopy
import io
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

from darsia.fluidflower.mass.n1_pipeline import DataReader, asset
from darsia.fluidflower.pipeline import current_build_job as subject


def _fixture_run(job, row, commit, stores_path, progress):
    """Small order-dependent numerical history, executed by real OS processes."""
    if row.get("fail"):
        raise ValueError("TESTONLY producer failure")
    time.sleep(row.get("delay", .2))
    state = np.float32(0)
    history = []
    for index, value in enumerate(row["values"]):
        state = np.float32(state * np.float32(.3) + np.float32(value))
        history.append(float(state))
        progress(dict(run=row["run"], state="phase1_frame_completed", case_id=str(index),
                      complete=True, frame_index=index, elapsed_h=row["times"][index]))
    return dict(complete=True, run=row["run"], history=history, values=row["values"],
                times=row["times"], pid=os.getpid(), ordered_case_ids=list(map(str, range(len(history)))))


def _job(tmp_path, count=3):
    rows = [dict(run="run-" + str(i), values=[i + 1., .25, 8.], times=[0., .125, .497]) for i in range(count)]
    return dict(root=str(tmp_path), inputs=rows, expected_runs=[r["run"] for r in rows], target_machine="Main")


def test_spawned_runs_preserve_serial_history_and_publish_once_after_process_exit(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import capacity
    from darsia.fluidflower import store_map
    from .gui.test_capacity import ledger_fixture
    job = _job(tmp_path)
    job.update(series_id="test", results_root=str(tmp_path / "results"), pipeline="native-phase-statistics", commit="a" * 40)
    for row in job["inputs"]:
        row.update(selected_frames=3, ordered_case_ids=["0", "1", "2"])
    ledger, stores, _state = ledger_fixture(tmp_path, monkeypatch, free=10**12)
    monkeypatch.setattr(capacity, "CapacityLedger", lambda *args: ledger)
    monkeypatch.setattr(store_map, "read_store_map", lambda *args: stores)
    monkeypatch.setattr(subject, "_space_estimate", lambda job, row: capacity.StorageProfileEstimate(
        row["run"], 3, "full", "keep", job["pipeline"], 100, 0, 100, {}, [], job["commit"]))
    original_capacity, allocations = subject._capacity, []
    @contextmanager
    def observed_capacity(*args):
        with original_capacity(*args):
            with ledger.locked():
                allocations.append(ledger.remaining())
            yield
    monkeypatch.setattr(subject, "_capacity", observed_capacity)
    log = io.StringIO()
    result = subject._dispatch_runs(job, "a" * 40, None, 2, log, runner=_fixture_run,
        recommendation=lambda rows, active: dict(workers=max(0, 2-len(active)), reason="TESTONLY finite capacity"))
    assert result == 0
    pids = []
    for row in job["inputs"]:
        actual = DataReader().json(asset(tmp_path / "results" / row["run"] / "completed.json"))
        expected = _fixture_run(job, row, "a" * 40, None, lambda event: None)
        assert actual["history"] == expected["history"]
        assert actual["times"] == [0., .125, .497]
        assert not psutil.pid_exists(actual["pid"])
        pids.append(actual["pid"])
    assert len(set(pids)) == len(pids) == 3
    events = [json.loads(line[line.index("{"):]) for line in log.getvalue().splitlines() if "{" in line]
    assert sum(e["kind"] == "run_completed" for e in events) == 3
    assert max(e.get("active_workers", 0) for e in events) == 2
    assert list((tmp_path / "results").glob("*/completed.json"))
    assert max(allocations) == 800
    with ledger.locked():
        assert ledger.remaining() == 0
    assert len(list(ledger.root.glob("r-*/reserved.json"))) == 3
    assert len(list(ledger.root.glob("r-*/released.json"))) == 3


def test_failed_run_leaves_the_other_runs_running_and_is_reported_at_the_end(tmp_path):
    job = _job(tmp_path)
    job["inputs"][0]["fail"] = True
    job["inputs"][1]["delay"] = 1.
    result = subject._dispatch_runs(job, "a" * 40, None, 2, io.StringIO(), runner=_fixture_run,
        recommendation=lambda rows, active: dict(workers=max(0, 2-len(active)), reason="TESTONLY finite capacity"),
        capacity_factory=lambda row: nullcontext())
    assert result == 1
    assert "TESTONLY producer failure" in (tmp_path / "failed" / "run-0.json").read_text()
    for run in ("run-1", "run-2"):  # The running and the next run both finished.
        assert DataReader().json(asset(tmp_path / "results" / run / "completed.json"))["run"] == run


def test_duplicate_run_is_rejected_before_any_process(tmp_path, monkeypatch):
    job = _job(tmp_path, 2)
    job["inputs"][1]["run"] = job["inputs"][0]["run"]
    job["expected_runs"] = [row["run"] for row in job["inputs"]]
    args = SimpleNamespace(job=tmp_path, backend="cpu", machine="Main", workers=8, stores=None)
    monkeypatch.setattr(subject, "_dispatch_runs", lambda *a: pytest.fail("duplicate dispatched"))
    with pytest.raises(ValueError, match="exactly once"):
        subject.run_current_build_job(args, job, "a" * 40, io.StringIO())


@pytest.mark.parametrize("available_gib,free_bytes,expected", [(168, 10**12, 6), (80, 10**12, 2), (168, 40 * 1024**3 + 150, 1)])
def test_live_recommendation_obeys_existing_ram_reserve_and_disk_reservations(monkeypatch, available_gib, free_bytes, expected):
    from darsia.fluidflower.pipeline import capacity
    class Ledger:
        reserve = 40 * 1024**3
        def __init__(self, *args): pass
        def fresh_volume(self): return dict(free_bytes=free_bytes)
        def locked(self, **kwargs): return nullcontext()
        def remaining(self): return 0
    monkeypatch.setattr(capacity, "CapacityLedger", Ledger)
    monkeypatch.setattr(subject, "_space_estimate", lambda *a: SimpleNamespace(needed=lambda **kw: 100))
    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda: list(range(32))))
    inventory = dict(cpu=dict(physical_cores=16, logical_cores=32),
                     memory=dict(available_bytes=available_gib * 1024**3))
    job = dict(inputs=[{} for _ in range(10)], precision="float32", series_id="test", results_root="C:/test")
    actual = subject.worker_recommendation(job, SimpleNamespace(machine_id="Main"), inventory=inventory)
    assert actual["workers"] == expected
    assert "not measured RSS" in actual["memory_basis"]


def test_concurrent_capacity_subtracts_existing_reservations(monkeypatch):
    from darsia.fluidflower.pipeline import capacity
    class Ledger:
        reserve = 0
        def __init__(self, *args): pass
        def locked(self): return nullcontext()
        def fresh_volume(self): return dict(free_bytes=150)
        def remaining(self): return 100
        def bind(self, *args): pytest.fail("oversubscribed disk")
    estimate = capacity.StorageProfileEstimate("sample", 1, "full", "keep", "native-phase-statistics",
        100, 0, 0, {}, [], "a" * 40)
    monkeypatch.setattr(subject, "_space_estimate", lambda *args: estimate)
    monkeypatch.setattr(capacity, "CapacityLedger", Ledger)
    with pytest.raises(ValueError, match="estimated needed"):
        with subject._capacity(None, dict(series_id="test", root="C:/test"),
                dict(run="sample", ordered_case_ids=["one"]), Path("C:/test/out"), {}):
            pytest.fail("producer entered without disk space")


def test_active_workers_keep_unused_ram_allocation_reserved(monkeypatch):
    from darsia.fluidflower.pipeline import capacity
    class Ledger:
        reserve = 0
        def __init__(self, *args): pass
        def fresh_volume(self): return dict(free_bytes=10**12)
        def locked(self, **kwargs): return nullcontext()
        def remaining(self): return 0
    monkeypatch.setattr(capacity, "CapacityLedger", Ledger)
    monkeypatch.setattr(subject, "_space_estimate", lambda *a: SimpleNamespace(needed=lambda **kw: 100))
    monkeypatch.setattr(psutil, "Process", lambda pid=None: SimpleNamespace(
        cpu_affinity=lambda: list(range(32)), memory_info=lambda: SimpleNamespace(rss=3 * 1024**3)))
    inventory = dict(cpu=dict(physical_cores=16, logical_cores=32), memory=dict(available_bytes=168 * 1024**3))
    job = dict(inputs=[{} for _ in range(10)], precision="float32", series_id="test", results_root="C:/test")
    actual = subject.worker_recommendation(job, SimpleNamespace(machine_id="Main"),
                                            inventory=inventory, active_pids=[11, 12])
    assert actual["workers"] == 4
    assert "2 active workers" in actual["message"]
