"""Pool workers start even when the parent has no console and PYTHONFAULTHANDLER is set."""
import multiprocessing
import os
from pathlib import Path
import sys

import pytest

from darsia.fluidflower import worker_processes


def test_worker_without_standard_error_logs_to_its_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(worker_processes.faulthandler, "enable", lambda **kwargs: None)
    folder = tmp_path / "logs"
    worker_processes.worker_log(str(folder))
    try:
        assert sys.stderr is not None
        sys.stderr.write("worker failed\n")
    finally:
        sys.stderr.close()
    assert (folder / f"worker-{os.getpid()}.log").read_text(encoding="utf-8") == "worker failed\n"


def test_worker_with_standard_error_keeps_it(tmp_path):
    before = sys.stderr
    worker_processes.worker_log(str(tmp_path / "logs"))
    assert sys.stderr is before and not (tmp_path / "logs").exists()


@pytest.mark.parametrize("console_less", [True, False])
def test_fault_handler_setting_is_withheld_only_from_console_less_workers(monkeypatch, console_less):
    monkeypatch.setenv("PYTHONFAULTHANDLER", "1")
    monkeypatch.setattr(worker_processes, "console_less_workers", lambda: console_less)
    with worker_processes.process_pool(1):
        assert ("PYTHONFAULTHANDLER" in os.environ) is not console_less
    assert os.environ["PYTHONFAULTHANDLER"] == "1"


def _pythonw():
    candidate = Path(getattr(sys, "_base_executable", sys.executable)).with_name("pythonw.exe")
    return candidate if candidate.is_file() else None


def _measured(tuner, rate_limit, *, kind="unit", start=0.0):
    """Feed units whose time grows once more tasks run than the host can serve."""
    now = start
    for _ in range(200):
        if tuner.settled:
            break
        running = tuner.target
        seconds = 100.0 * max(1.0, running / rate_limit)
        now += seconds / running
        tuner.unit_done(seconds, running, kind=kind, now=now)
    return now


def test_tuner_adds_workers_while_the_rate_rises_and_keeps_the_best():
    tuner = worker_processes.ThroughputTuner(16, 30, step=2, gain=0.03)
    _measured(tuner, rate_limit=20)
    assert tuner.settled and tuner.target == 20
    assert tuner.rates[18] > tuner.rates[16] and tuner.rates[22] <= tuner.rates[20] * 1.03


def test_tuner_ignores_units_from_another_concurrency_and_stops_at_the_maximum():
    tuner = worker_processes.ThroughputTuner(4, 6, step=2, samples=2)
    assert tuner.unit_done(10.0, 3) == 4 and not tuner.rates
    _measured(tuner, rate_limit=100)
    assert tuner.settled and tuner.target == 6


def test_tuner_searches_again_for_a_new_stage_and_after_a_while():
    tuner = worker_processes.ThroughputTuner(16, 30, step=2, reprobe_seconds=600)
    now = _measured(tuner, rate_limit=20)
    assert tuner.settled and tuner.target == 20
    tuner.unit_done(100.0, 20, kind="heavier stage", now=now)
    assert not tuner.settled
    _measured(tuner, rate_limit=14, kind="heavier stage", start=now)
    assert tuner.settled and tuner.target == 20  # the search restarts from the current count
    tuner.unit_done(100.0, 20, kind="heavier stage", now=now + 10_000)
    assert not tuner.settled


def test_memory_guard_remembers_the_largest_worker():
    gib = 1024**3
    readings = iter([(100 * gib, [5 * gib, 2 * gib]), (12 * gib, []), (14 * gib, [1 * gib])])
    guard = worker_processes.MemoryGuard(8 * gib, probe=lambda: next(readings))
    assert guard.allows_another() and guard.peak == 5 * gib
    assert not guard.allows_another()  # 12 GiB < 8 + 5 GiB, though no worker runs now
    assert guard.allows_another()


def test_task_processes_run_every_task_and_pass_their_events():
    events = []
    tasks = [(index, {"state": "frame", "index": index}) for index in range(3)]
    done = dict(worker_processes.run_task_processes(
        worker_processes.report, tasks, admit=lambda running: 2, critical_bytes=0,
        on_event=lambda event, running: events.append(event)))
    assert done == {0: None, 1: None, 2: None}
    assert sorted(event["index"] for event in events if event["state"] == "frame") == [0, 1, 2]


def test_task_processes_stop_the_newest_task_when_memory_runs_short_and_run_it_again():
    import time
    readings = iter([0])
    events = []
    started = time.monotonic()
    done = dict(worker_processes.run_task_processes(
        time.sleep, [(index, 3.0) for index in range(3)], admit=lambda running: 3,
        critical_bytes=1, available_memory=lambda: next(readings, 2),
        on_event=lambda event, running: events.append((event, running))))
    assert done == {0: None, 1: None, 2: None}
    assert events == [({"state": "stopped_for_memory", "task": 2}, 2)]
    assert time.monotonic() - started >= 6.0  # task 2 ran again after a slot freed


def test_task_processes_finish_every_task_then_report_all_failures():
    done = []
    tasks = [(0, "1"), (1, "x"), (2, "3"), (3, "y")]
    with pytest.raises(worker_processes.TaskFailures) as failed:
        for key, value in worker_processes.run_task_processes(
                int, tasks, admit=lambda running: 1, critical_bytes=0):
            done.append((key, value))
    assert done == [(0, 1), (2, 3)]
    assert sorted(failed.value.failures) == [1, 3]
    assert "2 of 4 tasks failed" in str(failed.value) and "invalid literal" in str(failed.value)
    assert any("Traceback" in note for note in failed.value.__notes__)


@pytest.mark.skipif(_pythonw() is None, reason="needs a console-less Windows interpreter")
def test_console_less_workers_run_tasks_with_the_fault_handler_set(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONFAULTHANDLER", "1")
    multiprocessing.set_executable(str(_pythonw()))
    try:
        with worker_processes.process_pool(2, log_folder=tmp_path / "logs") as pool:
            pids = {pool.submit(os.getpid).result(timeout=120) for _ in range(4)}
    finally:
        multiprocessing.set_executable(sys.executable)
    assert pids and os.getpid() not in pids
    assert os.environ["PYTHONFAULTHANDLER"] == "1"


@pytest.mark.skipif(_pythonw() is None, reason="needs a console-less Windows interpreter")
def test_console_less_task_processes_run_with_the_fault_handler_set(monkeypatch):
    monkeypatch.setenv("PYTHONFAULTHANDLER", "1")
    multiprocessing.set_executable(str(_pythonw()))
    try:
        done = dict(worker_processes.run_task_processes(
            abs, [(index, -index) for index in range(3)], admit=lambda running: 2, critical_bytes=0))
    finally:
        multiprocessing.set_executable(sys.executable)
    assert done == {0: 0, 1: 1, 2: 2}
    assert os.environ["PYTHONFAULTHANDLER"] == "1"
