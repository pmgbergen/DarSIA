"""A single transient SMB file or claim error waits and retries; it never crashes the work.

The project owner's rule N111. Tiny synthetic metadata only; the waits are set to zero.
"""
import io
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import job_queue as jobs
from darsia.fluidflower.pipeline import server_clock as clocks
from darsia.fluidflower.pipeline.job_queue import JobQueue, read_json, write_immutable
from .test_attempt_decisions import OWNER, sharing_error, task


@pytest.fixture(autouse=True)
def no_waits(monkeypatch):
    monkeypatch.setattr(jobs, "TRANSIENT_RETRY_DELAYS", (0.0, 0.0, 0.0))


def network_error():
    error = FileNotFoundError("The network path was not found")
    error.winerror = 53
    return error


def failing(times, error_factory, action):
    """Wrap action so its first calls raise the given transient error."""
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        if len(calls) <= times:
            raise error_factory()
        return action(*args, **kwargs)

    run.calls = calls
    return run


def test_transient_os_error_keeps_protocol_answers():
    assert jobs.transient_os_error(sharing_error()) is True
    assert jobs.transient_os_error(TimeoutError("SMB timeout")) is True
    assert jobs.transient_os_error(OSError(5, "I/O error")) is True
    assert jobs.transient_os_error(network_error()) is True
    assert jobs.transient_os_error(FileNotFoundError("moved or claimed")) is False
    assert jobs.transient_os_error(FileExistsError("lost exclusive create")) is False
    assert jobs.transient_os_error(NotADirectoryError("marker file")) is False
    assert jobs.transient_os_error(ValueError("not a file error")) is False


def test_retry_transient_waits_then_succeeds_or_raises_the_last_error():
    waits, retries = [], []
    action = failing(2, sharing_error, lambda: "done")
    assert jobs.retry_transient(action, delays=(1, 2, 3), wait=waits.append,
                                on_retry=lambda *args: retries.append(args[:2])) == "done"
    assert waits == [1, 2] and retries == [(1, 1), (2, 2)] and len(action.calls) == 3
    always = failing(10, sharing_error, lambda: "never")
    with pytest.raises(PermissionError):
        jobs.retry_transient(always, delays=(0, 0, 0), wait=lambda seconds: None)
    assert len(always.calls) == 4  # One try and three retries.


@pytest.mark.parametrize("error", [FileExistsError("lost create"), FileNotFoundError("moved"),
                                   ValueError("bad metadata")])
def test_retry_transient_never_retries_a_protocol_answer(error):
    def action():
        raise error

    with pytest.raises(type(error)):
        jobs.retry_transient(action, delays=(0, 0, 0), wait=lambda seconds: pytest.fail("waited"))


def test_retry_transient_stops_on_a_stop_request_or_an_unsafe_retry():
    stopped = failing(5, sharing_error, lambda: "never")
    with pytest.raises(PermissionError):
        jobs.retry_transient(stopped, delays=(9,), wait=lambda seconds: True)
    assert len(stopped.calls) == 1
    refused = failing(5, sharing_error, lambda: "never")
    with pytest.raises(PermissionError):
        jobs.retry_transient(refused, delays=(0,), retry_if=lambda error: False)
    assert len(refused.calls) == 1


def test_read_json_and_descriptor_wait_out_a_sharing_violation(tmp_path, monkeypatch):
    path = tmp_path / "task.json"
    write_immutable(path, dict(run="sample"))
    real = Path.read_bytes
    calls = []

    def read_bytes(self):
        if self == path and len(calls) < 2:
            calls.append(self)
            raise sharing_error()
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert read_json(path) == dict(run="sample") and len(calls) == 2
    calls.clear()
    assert jobs.descriptor(path)["size_bytes"] == path.stat().st_size and len(calls) == 2


def test_an_interrupted_move_that_completed_is_accepted(tmp_path, monkeypatch):
    source, destination = tmp_path / "pending.json", tmp_path / "claimed.json"
    source.write_bytes(b"{}")
    real = jobs._rename_once
    calls = []

    def completed_then_dropped(src, dst):
        calls.append(src)
        real(src, dst)
        if len(calls) == 1:
            raise sharing_error()  # The move happened; the reply was lost.

    monkeypatch.setattr(jobs, "_rename_once", completed_then_dropped)
    jobs.rename(source, destination)
    assert destination.read_bytes() == b"{}" and not source.exists() and len(calls) == 2
    other = tmp_path / "other.json"
    with pytest.raises(FileNotFoundError):  # Without an interrupted try it stays an answer.
        jobs.rename(tmp_path / "missing.json", other)


def test_an_interrupted_exclusive_write_that_completed_is_its_own(tmp_path, monkeypatch):
    path = tmp_path / "decisions/sample.json"
    real_rename, real_link = jobs.rename, jobs.os.link
    calls = []

    def dropped(real):
        def run(source, destination):
            calls.append(destination)
            real(source, destination)
            if len(calls) == 1:
                raise sharing_error()
        return run

    monkeypatch.setattr(jobs, "rename", dropped(real_rename))
    monkeypatch.setattr(jobs.os, "link", dropped(real_link))
    write_immutable(path, dict(outcome="publish"))
    assert read_json(path) == dict(outcome="publish") and len(calls) == 2
    with pytest.raises(FileExistsError):  # A different document is still refused.
        write_immutable(path, dict(outcome="reclaimed"))


def test_an_interrupted_directory_create_that_completed_is_its_own(tmp_path, monkeypatch):
    lease = tmp_path / "leases" / ("sample." + "a" * 32)
    lease.parent.mkdir()
    real = Path.mkdir
    calls = []

    def mkdir(self, *args, **kwargs):
        calls.append(self)
        real(self, *args, **kwargs)
        if self == lease and len(calls) == 1:
            raise sharing_error()

    monkeypatch.setattr(Path, "mkdir", mkdir)
    jobs._create_directory(lease)
    assert lease.is_dir()
    (lease / "owner.json").write_bytes(b"{}")
    with pytest.raises(FileExistsError):  # An existing, used directory stays an answer.
        jobs._create_directory(lease)


def test_claim_skips_a_locked_queue_file_and_reports_it(tmp_path, monkeypatch):
    events = []
    shared = JobQueue(tmp_path, on_error=events.append)
    shared.enqueue(task("alpha"))
    shared.enqueue(task("beta"))
    locked = tmp_path / "queue/alpha.json"
    real = jobs.read_json

    def read(path):
        if Path(path) == locked:
            raise sharing_error()  # A scanner keeps this one task file open.
        return real(path)

    monkeypatch.setattr(jobs, "read_json", read)
    claim = shared.claim("machine", "cpu", OWNER)
    assert claim.task["run"] == "beta" and locked.is_file()
    assert [(event["kind"], event["step"]) for event in events] == [("transient_file_error", "claim")]
    monkeypatch.setattr(jobs, "read_json", real)
    assert shared.claim("machine", "cpu", OWNER).task["run"] == "alpha"


def test_server_clock_samples_wait_out_a_sharing_violation(tmp_path, monkeypatch):
    clock = clocks.ServerFileClock(tmp_path, "machine")
    real = Path.stat
    calls = []

    def stat(self, *args, **kwargs):
        if self == clock.path and len(calls) < 1:
            calls.append(self)
            raise sharing_error()
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    assert clock.now_ns() > 0 and len(calls) == 1


def _worker_loop(tmp_path, monkeypatch, shared_class, *, exit_when_idle=True):
    from darsia.fluidflower.pipeline import worker
    from darsia_cache import adapters
    monkeypatch.setattr(adapters, "bind_actual_worker_owner", lambda owner, **kwargs: {})
    monkeypatch.setattr(worker, "_identity", lambda machine: {"machine": machine})
    monkeypatch.setattr(worker, "QUEUE_RETRY_DELAYS", (0.0, 0.0, 0.0))
    monkeypatch.setattr(worker, "JobQueue", shared_class)
    events = []
    worker._run_tasks(str(tmp_path / "job"), {}, "a" * 40, "Main-cpu", "cpu",
                      SimpleNamespace(put=events.append), threading.Event(), False, exit_when_idle)
    return events


def test_worker_claim_loop_waits_out_a_transient_share_error(tmp_path, monkeypatch):
    calls = []

    class Shared:
        def __init__(self, *args, **kwargs):
            pass

        def recover_publications(self):
            calls.append("recover")
            if len(calls) == 1:
                raise network_error()

        def reclaim_stale(self):
            return []

        def claim(self, *args, **kwargs):
            return None

    events = _worker_loop(tmp_path, monkeypatch, Shared)
    assert [event["kind"] for event in events] == ["queue_transient_retry"]
    assert events[0]["attempt"] == 1 and calls == ["recover", "recover"]


def test_worker_claim_loop_reports_each_wait_before_its_old_stop(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import worker
    from darsia_cache import adapters
    monkeypatch.setattr(adapters, "bind_actual_worker_owner", lambda owner, **kwargs: {})
    monkeypatch.setattr(worker, "_identity", lambda machine: {"machine": machine})
    monkeypatch.setattr(worker, "QUEUE_RETRY_DELAYS", (0.0, 0.0, 0.0))

    class Shared:
        def __init__(self, *args, **kwargs):
            pass

        def recover_publications(self):
            raise sharing_error()

    monkeypatch.setattr(worker, "JobQueue", Shared)
    events = []
    with pytest.raises(SystemExit):
        worker._run_tasks(str(tmp_path / "job"), {}, "a" * 40, "Main-cpu", "cpu",
                          SimpleNamespace(put=events.append), threading.Event(), False, True)
    assert [event["kind"] for event in events] == ["queue_transient_retry"] * 3 + ["queue_failure"]


def test_worker_claim_loop_does_not_retry_a_real_failure(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import worker
    from darsia_cache import adapters
    monkeypatch.setattr(adapters, "bind_actual_worker_owner", lambda owner, **kwargs: {})
    monkeypatch.setattr(worker, "_identity", lambda machine: {"machine": machine})

    class Shared:
        def __init__(self, *args, **kwargs):
            pass

        def recover_publications(self):
            raise ValueError("Publication marker differs from its run/attempt lease")

    monkeypatch.setattr(worker, "JobQueue", Shared)
    events = []
    with pytest.raises(SystemExit):
        worker._run_tasks(str(tmp_path / "job"), {}, "a" * 40, "Main-cpu", "cpu",
                          SimpleNamespace(put=events.append), threading.Event(), False, True)
    assert [event["kind"] for event in events] == ["queue_failure"]


def test_watchdog_parent_waits_out_a_transient_error_before_restarting(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import worker
    token = "c" * 32
    task_ = {"run": "ac52", "attempt_id": token}
    (tmp_path / "leases" / ("ac52." + token)).mkdir(parents=True)
    crashed = SimpleNamespace(exitcode=1, alive=True, join=lambda **kwargs: None)
    crashed.is_alive = lambda: crashed.alive
    replacement = SimpleNamespace(exitcode=0, join=lambda **kwargs: None, is_alive=lambda: False)
    events = [dict(kind="claimed", run="ac52", attempt_id=token, worker_slot=0, worker_generation="g")]

    class Messages:
        def get(self, **kwargs):
            if not events:
                raise queue.Empty()
            crashed.alive = False
            return events.pop(0)

        def get_nowait(self):
            raise queue.Empty()

    passes = []

    class Shared:
        def recover_publications(self):
            passes.append("recover")
            if len(passes) == 1:
                raise sharing_error()

        def reclaim_stale(self):
            # The stale attempt is reclaimed in place and its retry recorded.
            jobs.decide(tmp_path, "ac52", token, "reclaimed", next_attempt="d" * 32)
            write_immutable(tmp_path / "reclaimed/ac52" / token / "reclaimed.json", dict(run="ac52"))
            return []

    admissions = []

    def verify_startup(*args, **kwargs):
        admissions.append(1)
        if len(admissions) == 1:
            raise network_error()
        return {}, "a" * 40

    def slot(process, generation):
        return dict(number=0, generation=generation, process=process, root=None, observed=[],
                    last_heartbeat=0, claim=None, retired=False, waiting_reclaim=False,
                    finished=False)

    started = [slot(crashed, "g"), slot(replacement, "h")]
    monkeypatch.setattr(worker, "QUEUE_RETRY_DELAYS", (0.0, 0.0, 0.0))
    monkeypatch.setattr(worker, "JobQueue", lambda root, *, machine_id: Shared())
    monkeypatch.setattr(worker.mp, "get_context",
                        lambda *args: SimpleNamespace(Queue=Messages, Event=threading.Event))
    monkeypatch.setattr(worker, "_start_slot", lambda *args: started.pop(0))
    monkeypatch.setattr(worker, "read_json", lambda path: task_)
    monkeypatch.setattr(worker, "verify_startup", verify_startup)
    monkeypatch.setattr(worker, "_record_retirement",
                        lambda value, args, identity: value.update(retired=True) or [])
    args = SimpleNamespace(job=tmp_path, workers=1, watchdog=True, once=False)
    log = io.StringIO()
    assert worker._parent(args, "Main-cpu", {}, "a" * 40, log) == 0
    text = log.getvalue()
    assert text.count('"kind": "watchdog_transient_retry"') == 2
    assert '"step": "reclaim"' in text and '"step": "restart_admission"' in text
    assert started == [] and admissions == [1, 1] and passes == ["recover", "recover"]
