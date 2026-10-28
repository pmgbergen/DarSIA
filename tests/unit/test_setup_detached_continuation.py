"""Exercise the actual supervisor with deterministic child completion events."""
from io import StringIO
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import worker


@pytest.mark.parametrize("end_apply", ["complete", "pause", "stop", "fail", "no_progress"])
def test_detached_setup_continues_build_without_any_gui_timer(tmp_path, monkeypatch, end_apply):
    root = tmp_path / "job"
    (root / "queue").mkdir(parents=True)
    (root / "queue" / "setup-apply.json").write_bytes(b"not read by scheduler")
    operations = [dict(run="setup-apply", operation="apply"),
                  *(dict(run=f"build-{i}", operation="build") for i in range(7))]
    job = dict(pipeline="series-setup", setup_operations=operations)
    args = SimpleNamespace(job=root, machine="fixture", backend="cpu", workers=6,
        once=True, exit_when_idle=True, watchdog=False, watchdog_seconds=30)
    context = SimpleNamespace(Queue=queue.Queue, Event=threading.Event)
    monkeypatch.setattr(worker.mp, "get_context", lambda method: context)
    starts, generations, controls = [], [], []
    monkeypatch.setattr(worker, "verify_startup", lambda path: (job, "a" * 40))

    def control(path):
        controls.append(path)
        if end_apply in {"pause", "stop"}:
            raise worker._JobControlBoundary("paused" if end_apply == "pause" else "control_stopped")
    monkeypatch.setattr(worker, "_job_control_boundary", control)

    def start(context, args, actual_job, commit, identity, messages, stop, number):
        pending = next(row for row in operations if (root / "queue" / (row["run"] + ".json")).exists())
        starts.append(pending["operation"])
        generation = str(len(starts))
        generations.append(generation)
        path = root / "queue" / (pending["run"] + ".json")
        path.unlink()
        if pending["operation"] == "apply":
            for row in operations[1:]:
                (root / "queue" / (row["run"] + ".json")).write_bytes(b"not read by scheduler")
        kind = "failed" if end_apply == "fail" else "completed"
        if end_apply != "no_progress":
            messages.put(dict(kind=kind, run=pending["run"], worker_slot=number,
                              worker_generation=generation))
        process = SimpleNamespace(is_alive=lambda: False, join=lambda: None,
                                  exitcode=1 if kind == "failed" else 0)
        return dict(number=number, generation=generation, process=process, root=None,
                    observed=[], last_heartbeat=0, claim=None, retired=False,
                    waiting_reclaim=False, finished=False)

    def retire(slot, args, identity, **options):
        options.get("before_record", lambda: None)()
        slot["retired"] = True
        return []

    monkeypatch.setattr(worker, "_start_slot", start)
    monkeypatch.setattr(worker, "_record_retirement", retire)
    log = StringIO()
    result = worker._parent(args, "fixture-cpu", job, "a" * 40, log)
    if end_apply == "complete":
        assert result == 0
        assert starts == ["apply"] + ["build"] * 7
        events = [json.loads(line.split(" ", 1)[1]) for line in log.getvalue().splitlines()]
        batches = [event["workers"] for event in events if event["kind"] == "setup_build_batch_started"]
        assert batches == [6, 1]
        assert not list((root / "queue").iterdir())
    else:
        assert starts == ["apply"]
        assert len(list((root / "queue").iterdir())) == 7
        assert result == (1 if end_apply == "fail" else 0)
    assert len(generations) == len(set(generations))


def test_setup_stage_lookup_does_not_open_operation_payloads(tmp_path, monkeypatch):
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / "apply.json").write_bytes(b"large control payload")
    job = dict(pipeline="series-setup", setup_operations=[dict(run="apply", operation="apply")])
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("Scheduler opened task payload"))
    assert worker._ready_setup_operations(tmp_path, job, "cpu") == job["setup_operations"]
