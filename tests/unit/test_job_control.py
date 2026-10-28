"""Tiny operational metadata and callback fixtures; no actual worker/job starts."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
from pathlib import Path
import queue
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from darsia.fluidflower import job_control as control
from darsia.fluidflower.pipeline import checkpoint, worker


@pytest.fixture
def job(tmp_path):
    root = tmp_path / "job"
    control.write_immutable(
        root / "job.json",
        {"commit": "a" * 40, "expected_runs": ["ac52"]},
    )
    control.write_immutable(
        root / "ui" / "master.json",
        {
            "format": "darsia-job-ui-master-v1",
            "machine": "Main",
            "job": control.descriptor(root / "job.json"),
        },
    )
    return root


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_read_only_empty_job_does_not_create_layout(tmp_path):
    root = tmp_path / "missing"
    assert control.read_control(root) == {"action": "run", "sequence": 0}
    assert not root.exists()


def test_refresh_is_read_only_before_and_after_a_command(job):
    before = snapshot(job)
    assert control.read_control(job) == {"action": "run", "sequence": 0}
    assert snapshot(job) == before
    control.publish_control(job, "pause", machine="Main", expected_previous=None)
    before = snapshot(job)
    assert control.read_control(job)["action"] == "pause"
    assert snapshot(job) == before
    assert not (job / "queue").exists()
    assert not (job / "claimed").exists()


def test_pause_resume_stop_are_append_only_and_keep_prior_exact_bytes(job):
    previous = None
    preserved = {}
    for sequence, action in enumerate(("pause", "run", "stop"), 1):
        published = control.publish_control(
            job, action, machine="Main", expected_previous=previous
        )
        current = control.read_control(job)
        assert current["action"] == action and current["sequence"] == sequence
        assert current["command"] == published and current["previous"] == previous
        assert current["job"] == control.descriptor(job / "job.json")
        assert current["master"] == control.descriptor(job / "ui/master.json")
        for path, raw in preserved.items():
            assert path.read_bytes() == raw
        preserved[Path(published["path"])] = Path(published["path"]).read_bytes()
        previous = published


def test_stale_window_and_non_master_cannot_publish(job):
    first = control.publish_control(
        job, "pause", machine="Main", expected_previous=None
    )
    before = snapshot(job)
    with pytest.raises(ValueError, match="refresh"):
        control.publish_control(job, "run", machine="Main", expected_previous=None)
    with pytest.raises(ValueError, match="master machine"):
        control.publish_control(job, "stop", machine="Second", expected_previous=first)
    assert snapshot(job) == before


def test_same_sequence_publication_has_one_atomic_winner(job, monkeypatch):
    barrier = threading.Barrier(2)
    original = control.write_immutable

    def simultaneous_publication(path, document):
        barrier.wait(timeout=5)
        return original(path, document)

    monkeypatch.setattr(control, "write_immutable", simultaneous_publication)

    def publish():
        try:
            return control.publish_control(
                job, "pause", machine="Main", expected_previous=None
            )
        except FileExistsError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(), range(2)))
    winners = [value for value in results if isinstance(value, dict)]
    assert len(winners) == 1
    assert sum(isinstance(value, FileExistsError) for value in results) == 1
    assert control.read_control(job)["command"] == winners[0]
    assert len(list((job / "ui/commands").glob("*.json"))) == 1


@pytest.mark.parametrize("corruption", ["job", "master", "prior_command", "gap"])
def test_changed_identity_or_command_chain_is_refused(job, corruption):
    first = control.publish_control(
        job, "pause", machine="Main", expected_previous=None
    )
    control.publish_control(job, "run", machine="Main", expected_previous=first)
    if corruption in {"job", "master"}:
        path = job / ("job.json" if corruption == "job" else "ui/master.json")
        value = json.loads(path.read_bytes())
        value["changed"] = True
        path.write_bytes(json.dumps(value).encode())
    elif corruption == "prior_command":
        path = Path(first["path"])
        value = json.loads(path.read_bytes())
        value["action"] = "stop"  # Still valid alone; the next exact link differs.
        path.write_bytes(json.dumps(value).encode())
    else:
        path = Path(first["path"])
        path.rename(path.with_name("00000000000000000003.json"))
    with pytest.raises(ValueError):
        control.read_control(job)


def test_unknown_action_cannot_create_a_command(job):
    before = snapshot(job)
    with pytest.raises(ValueError, match="Unknown job action"):
        control.publish_control(job, "restart", machine="Main", expected_previous=None)
    assert snapshot(job) == before


@pytest.fixture
def native_cache_control_owner_binding(monkeypatch):
    """Only the three mocked worker-loop control tests own this boundary."""
    from darsia_cache import adapters

    calls = []
    def bind(owner, *, slot=None, generation=None, role=None):
        assert owner == {"machine": "Main-cpu"}
        assert slot is None and generation is None and role == "slot"
        calls.append(dict(owner))
        return dict(TEST_ONLY_NATIVE_CONTROL_OWNER_BOUNDARY=True)
    monkeypatch.setattr(adapters, "bind_actual_worker_owner", bind)
    return calls


@pytest.mark.parametrize("action,state", [("pause", "paused"), ("stop", "control_stopped")])
def test_control_before_claim_never_reclaims_or_touches_queue(
    job, monkeypatch, action, state, native_cache_control_owner_binding
):
    control.publish_control(job, action, machine="Main", expected_previous=None)
    events = []

    class Shared:
        def __init__(self, *args, **kwargs):
            pass

        def recover_publications(self):
            pytest.fail("A controlled worker may not advance the queue")

    monkeypatch.setattr(worker, "JobQueue", Shared)
    monkeypatch.setattr(worker, "_identity", lambda machine: {"machine": machine})
    before = snapshot(job)
    worker._run_tasks(
        str(job), {}, "a" * 40, "Main-cpu", "cpu",
        SimpleNamespace(put=events.append), threading.Event(), False, False,
    )
    assert [event["kind"] for event in events] == [state]
    assert snapshot(job) == before
    assert native_cache_control_owner_binding == [{"machine": "Main-cpu"}]


@pytest.mark.parametrize("action,state", [("pause", "paused"), ("stop", "control_stopped")])
def test_committed_frame_control_closes_executor_without_terminal_failure(
    job, monkeypatch, action, state, native_cache_control_owner_binding
):
    events, calls = [], []
    task = {
        "run": "ac52", "commit": "a" * 40,
        "calibration_sha256": "b" * 64, "precision": "float32",
    }
    original_task = deepcopy(task)
    claim = SimpleNamespace(task=task, token="c" * 32, output=job / "attempt")

    class Shared:
        def __init__(self, *args, **kwargs):
            pass

        def recover_publications(self):
            pass

        def reclaim_stale(self):
            return []

        def claim(self, *args):
            calls.append("claim")
            return claim

        def progress(self, owner, event):
            assert owner is claim
            calls.append(("actual_progress", event["case_id"]))

        def complete(self, *args):
            pytest.fail("An interrupted run must not be completed")

        def fail(self, *args):
            pytest.fail("Operational control must not publish terminal task FAIL")

    def execute(science_task, output, backend, precision, progress):
        assert science_task == original_task
        assert (output, backend, precision) == (claim.output, "cpu", "float32")
        try:
            calls.append("original_atomic_frame_publication")
            control.publish_control(job, action, machine="Main", expected_previous=None)
            progress({"state": "frame_completed", "complete": True, "case_id": "ac52_f1"})
            pytest.fail("The next frame must not start after control")
        finally:
            calls.append("executor_references_closed")

    fake = ModuleType("darsia.fluidflower.n1_pipeline")
    fake.run_n1 = execute
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(worker, "JobQueue", Shared)
    monkeypatch.setattr(worker, "_identity", lambda machine: {"machine": machine})
    monkeypatch.setattr(checkpoint, "resume_source_for_claim", lambda *args: None)
    worker._run_tasks(
        str(job), {"calibration_sha256": "b" * 64, "precision": "float32"},
        "a" * 40, "Main-cpu", "cpu", SimpleNamespace(put=events.append),
        threading.Event(), False, False,
    )
    assert calls == [
        "claim", "original_atomic_frame_publication", ("actual_progress", "ac52_f1"),
        "executor_references_closed",
    ]
    assert [event["kind"] for event in events] == ["claimed", "progress", state]
    assert events[-1]["attempt_id"] == claim.token
    assert task == original_task
    assert not (job / "failed").exists() and not (job / "done").exists()
    assert native_cache_control_owner_binding == [{"machine": "Main-cpu"}]


def test_malformed_control_is_operational_failure_before_claim(
    job, monkeypatch, native_cache_control_owner_binding
):
    control.publish_control(job, "pause", machine="Main", expected_previous=None)
    path = job / "ui/commands/00000000000000000001.json"
    path.write_bytes(b"not-json")
    events = []

    class Shared:
        def __init__(self, *args, **kwargs):
            pass

        def recover_publications(self):
            pytest.fail("Malformed controls must fail before queue recovery")

    monkeypatch.setattr(worker, "JobQueue", Shared)
    monkeypatch.setattr(worker, "_identity", lambda machine: {"machine": machine})
    before = snapshot(job)
    worker._run_tasks(
        str(job), {}, "a" * 40, "Main-cpu", "cpu",
        SimpleNamespace(put=events.append), threading.Event(), False, False,
    )
    assert [event["kind"] for event in events] == ["control_failure"]
    assert "JSONDecodeError" in events[0]["traceback"]
    assert snapshot(job) == before
    assert native_cache_control_owner_binding == [{"machine": "Main-cpu"}]


@pytest.mark.parametrize("kind,expected_exit", [("paused", 0), ("control_stopped", 0), ("control_failure", 1)])
def test_parent_keeps_controlled_claim_for_existing_retirement(
    tmp_path, monkeypatch, kind, expected_exit
):
    task = {"run": "ac52", "attempt_id": "c" * 32}
    process = SimpleNamespace(exitcode=0, alive=True, join=lambda **kwargs: None)
    process.is_alive = lambda: process.alive
    events = [
        dict(kind="claimed", run="ac52", attempt_id="c" * 32,
             worker_slot=0, worker_generation="g"),
        dict(kind=kind, run="ac52", attempt_id="c" * 32,
             worker_slot=0, worker_generation="g"),
    ]

    class Messages:
        def get(self, **kwargs):
            event = events.pop(0)
            if event["kind"] == kind:
                process.alive = False
            return event

        def get_nowait(self):
            raise queue.Empty()

    context = SimpleNamespace(Queue=Messages, Event=threading.Event)
    slot = dict(number=0, generation="g", process=process, root=None,
                observed=[], last_heartbeat=0, claim=None, retired=False,
                waiting_reclaim=False, finished=False)
    retired = []

    def retire(value, args, identity):
        assert value["claim"] == {"run": "ac52", "attempt_id": "c" * 32, "task": task}
        assert not process.is_alive()
        retired.append(deepcopy(value["claim"]))
        value["retired"] = True
        return []

    def queue_for_service(root, *, machine_id):
        assert root==tmp_path and machine_id=="Main-cpu"
        return object()
    monkeypatch.setattr(worker, "JobQueue", queue_for_service)
    monkeypatch.setattr(worker.mp, "get_context", lambda *args: context)
    monkeypatch.setattr(worker, "_start_slot", lambda *args: slot)
    monkeypatch.setattr(worker, "read_json", lambda path: task)
    monkeypatch.setattr(worker, "_record_retirement", retire)
    args = SimpleNamespace(job=tmp_path, workers=1, watchdog=False, once=False)
    assert worker._parent(args, "Main-cpu", {}, "a" * 40, io.StringIO()) == expected_exit
    assert len(retired) == 1


@pytest.mark.parametrize(
    "kind,expected_exit",
    [("paused", 0), ("control_stopped", 0), ("control_failure", 1)],
)
def test_final_message_drain_preserves_control_failure_exit_status(
    tmp_path, monkeypatch, kind, expected_exit
):
    """A clean child exit can precede delivery of its final buffered error."""
    joined = []
    process = SimpleNamespace(
        exitcode=0,
        is_alive=lambda: False,
        join=lambda: joined.append(True),
    )
    late = [
        dict(
            kind=kind,
            worker_slot=0,
            worker_generation="g",
            traceback=(
                "ValueError: invalid immutable command identity"
                if kind == "control_failure" else None
            ),
        )
    ]

    class Messages:
        def get(self, **kwargs):
            raise queue.Empty()

        def get_nowait(self):
            assert joined, "The event must be delivered only in final draining"
            if late:
                return late.pop(0)
            raise queue.Empty()

    context = SimpleNamespace(Queue=Messages, Event=threading.Event)
    slot = dict(
        number=0, generation="g", process=process, root=None,
        observed=[], last_heartbeat=0, claim=None, retired=False,
        waiting_reclaim=False, finished=False,
    )

    def retire(value, args, identity):
        assert value is slot and not process.is_alive()
        value["retired"] = True
        return []

    def queue_for_service(root, *, machine_id):
        assert root==tmp_path and machine_id=="Main-cpu"
        return object()
    monkeypatch.setattr(worker, "JobQueue", queue_for_service)
    monkeypatch.setattr(worker.mp, "get_context", lambda *args: context)
    monkeypatch.setattr(worker, "_start_slot", lambda *args: slot)
    monkeypatch.setattr(worker, "_record_retirement", retire)
    args = SimpleNamespace(job=tmp_path, workers=1, watchdog=False, once=False)
    log = io.StringIO()
    result = worker._parent(args, "Main-cpu", {}, "a" * 40, log)
    if kind == "control_failure":
        assert "ValueError: invalid immutable command identity" in log.getvalue()
    else:
        assert f'"kind": "{kind}"' in log.getvalue()
    assert slot["retired"] and not late
    assert result == expected_exit


def test_explicit_setup_start_enables_controls_without_replacing_the_master(tmp_path):
    from darsia.fluidflower import job_control
    from darsia.fluidflower.pipeline.job_queue import write_immutable
    root = tmp_path / "setup-job"
    write_immutable(root / "job.json", dict(format="darsia-alias-job-v1", format_version=1,
        pipeline="series-setup"))
    assert job_control.read_control(root) == dict(action="run", sequence=0)
    assert not (root / "ui" / "master.json").exists()
    bound = job_control.ensure_setup_master(root, machine="fixture", actor="fixture operator")
    before = (root / "ui" / "master.json").read_bytes()
    assert job_control.ensure_setup_master(root, machine="other", actor="other operator") == bound
    assert (root / "ui" / "master.json").read_bytes() == before
    job_control.publish_control(root, "stop", machine="fixture", expected_previous=None)
    assert job_control.read_control(root)["action"] == "stop"


def test_setup_master_initializer_refuses_an_unrelated_pipeline(tmp_path):
    import pytest
    from darsia.fluidflower import job_control
    from darsia.fluidflower.pipeline.job_queue import write_immutable
    root = tmp_path / "other-job"
    write_immutable(root / "job.json", dict(format="darsia-alias-job-v1", format_version=1,
        pipeline="native-phase-statistics"))
    with pytest.raises(ValueError, match="explicit Setup"):
        job_control.ensure_setup_master(root, machine="fixture", actor="fixture operator")
    assert not (root / "ui").exists()
