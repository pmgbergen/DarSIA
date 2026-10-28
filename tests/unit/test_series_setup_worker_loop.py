"""Original worker-loop ordering with explicit operational boundary doubles.

These cases do not start processes or qualify native ownership or science.
The real queue, writers and ledgers have separate focused integration cases.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import queue
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import worker, worker_io
from darsia.fluidflower.pipeline import series_setup_jobs, series_setup_operations
from .test_series_setup_protocol import _task, _job


@pytest.mark.parametrize("outcome", ["success", "failure", "lease_lost", "pause", "close_failure", "handoff"])
def test_setup_loop_keeps_claim_until_original_output_closure(tmp_path, monkeypatch, outcome):
    """Publication stays bounded and only actual closure permits completion."""
    task = dict(_task("apply", run="setup-apply"), job_id="setup-job", attempt_id="a" * 32)
    job = dict(_job(task), expected_runs=["setup-apply", "run-a"],
        setup_operations=[dict(run="setup-apply", operation="apply"), dict(run="run-a", operation="build")])
    claim = SimpleNamespace(task=task, token=task["attempt_id"], output=tmp_path / "attempt")
    state = dict(writer=False, binding=False, released=False, claimed=False, boundary_calls=0)
    calls = []
    waiting_for_apply = [outcome == "handoff"]
    if waiting_for_apply[0]:
        (tmp_path / "claimed" / "fixture-cpu").mkdir(parents=True)
        (tmp_path / "claimed" / "fixture-cpu" / "active.json").write_text(
            json.dumps(dict(run="active", attempt_id="b" * 32)), encoding="utf-8")
    def wait_for_apply(seconds):
        assert seconds == 1 and waiting_for_apply[0]
        calls.append("waited-for-active-operation")
        waiting_for_apply[0] = False
        (tmp_path / "claimed" / "fixture-cpu" / "active.json").rename(tmp_path / "retired.json")
    owner = dict(machine="fixture-cpu", pid=123, process_create_time="1.25")

    class Messages(queue.Queue):
        def put(self, event):
            if event["kind"] == "completed":
                assert not state["writer"] and state["released"]
                calls.append("completion-notified")
            super().put(event)

    class Shared:
        def __init__(self, *args, **kwargs):
            pass

        def recover_publications(self):
            return []

        def reclaim_stale(self):
            return []

        def claim(self, *args, **kwargs):
            assert callable(kwargs["claim_controls"])
            if waiting_for_apply[0]:
                return None
            if state["claimed"]:
                return None
            state["claimed"] = True
            return claim

        def progress(self, actual, event):
            assert actual is claim and state["writer"]

        def complete(self, actual, result):
            assert actual is claim and state["writer"] and not state["binding"]
            assert "calibration_sha256" not in result
            assert result["actual_worker"] == owner and result["task"] == task
            calls.append("publication")

        def fail(self, actual, text, actual_owner):
            assert actual is claim and state["writer"] and actual_owner == owner
            assert "literal execution failure" in text
            calls.append("failure-publication")

    class Binding:
        @contextmanager
        def activate(self):
            state["binding"] = True
            try:
                yield self
            finally:
                state["binding"] = False

    class Runtime:
        def __init__(self, *args):
            self.publications = SimpleNamespace(publish_progress=lambda *args: None)

        @contextmanager
        def capacity_claim(self, *args):
            yield SimpleNamespace(availability=lambda task: (), bind=lambda claim: None,
                claim_controls=lambda *args: None)

        def output_factory(self, *args):
            pass

        @contextmanager
        def output_scope(self, actual):
            assert actual is claim
            state["writer"] = True
            try:
                yield
            finally:
                state["writer"] = False
                calls.append("writer-closed")
                if outcome == "close_failure":
                    raise RuntimeError("literal close failure")

        def prepare(self, actual):
            assert actual is claim and state["writer"]
            return Binding()

        def capacity_progress(self, *args):
            assert state["writer"]

        def finalize_common(self, actual):
            assert actual is claim

        def release(self, actual, *, retain_capacity=False):
            assert actual is claim and not state["writer"]
            state["released"] = not retain_capacity
            calls.append("capacity-retained" if retain_capacity else "capacity-released")

    def execute(actual, output, backend, precision, progress):
        assert state["writer"] and state["binding"]
        assert output == claim.output
        if outcome == "failure":
            raise RuntimeError("literal execution failure")
        if outcome == "lease_lost":
            raise worker.LeaseLost("literal ownership lost")
        if outcome == "pause":
            progress(dict(state="setup_progress", operation="apply", run="run-a"))
        return dict(format="darsia-series-setup-result-v1", pipeline="series-setup", complete=True,
            run=task["run"], operation="apply", backend=backend, precision=precision,
            setup_plan=dict(path=str(tmp_path / "plan.json"), sha256="d" * 64, size_bytes=17),
            worker_references_closed=True, files={"operation_result.json": dict(
                path=str(output / "operation_result.json"), sha256="e" * 64, size_bytes=19)})

    def enqueue(root, actual, result):
        assert actual is claim and state["writer"] and state["binding"]
        calls.append("children-enqueued")

    def control(root):
        state["boundary_calls"] += 1
        if outcome == "pause" and state["boundary_calls"] > 1:
            raise worker._JobControlBoundary("paused", command=dict(action="pause"))

    monkeypatch.setattr(worker, "JobQueue", Shared)
    monkeypatch.setattr(worker_io, "WorkerRunIO", Runtime)
    monkeypatch.setattr(worker, "_identity", lambda machine: deepcopy(owner))
    monkeypatch.setattr(worker, "_job_control_boundary", control)
    monkeypatch.setattr(series_setup_operations, "execute_setup", execute)
    monkeypatch.setattr(series_setup_jobs, "enqueue_setup_builds", enqueue)
    from darsia_cache import adapters
    monkeypatch.setattr(adapters, "bind_actual_worker_owner", lambda *args, **kwargs: None)
    from darsia import _opencv_runtime
    monkeypatch.setattr(_opencv_runtime, "scientific_thread", lambda *args: {"test_boundary": True})
    monkeypatch.setattr(_opencv_runtime, "require_ipp_off", lambda *args: {"test_boundary": True})
    messages = Messages()
    arguments = (str(tmp_path), job, task["commit"], "fixture-cpu", "cpu", messages,
                 SimpleNamespace(is_set=lambda: False, wait=wait_for_apply))
    if outcome == "close_failure":
        with pytest.raises(RuntimeError, match="literal close failure"):
            worker._run_tasks(*arguments, once=False, exit_when_idle=True,
                              stores=SimpleNamespace(machine_id="fixture"))
    else:
        worker._run_tasks(*arguments, once=False, exit_when_idle=True,
                          stores=SimpleNamespace(machine_id="fixture"))
    events = list(messages.queue)
    if outcome in {"success", "handoff"}:
        if outcome == "handoff":
            assert calls.pop(0) == "waited-for-active-operation"
        assert calls == ["children-enqueued", "publication", "writer-closed",
                         "capacity-released", "completion-notified"]
    else:
        assert not any(event["kind"] == "completed" for event in events)
        assert not state["released"]
        if outcome in {"failure", "lease_lost"}:
            event = next(event for event in events if event["kind"] in {"failed", "lease_lost"})
            assert event["capacity_retained"] is True
        elif outcome == "pause":
            assert any(event["kind"] == "paused" for event in events)
