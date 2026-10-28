"""Reclaim and input rejection retire a lease by marker file; no lease directory moves.

Tiny synthetic metadata only. The lease stays in leases/<run>.<attempt>, the
marker reclaimed/<run>/<attempt>/lease.json names it, and the decision file
fixes the one retry generation, so every finishing step can be repeated.
"""
from concurrent.futures import ThreadPoolExecutor
import io
import os
import queue
import threading
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import job_queue as jobs
from darsia.fluidflower.pipeline.job_queue import JobQueue, LeaseLost, read_json, write_immutable
from tests.server_lease_clock_fixture import server_lease_clock
from .test_alias_queue_runtime import claim_fixture as alias_claim, fixture as alias_fixture
from .test_attempt_decisions import OWNER, decisions, no_directory_rename, stale_claim, task


def marker(run, token):
    return dict(format=jobs.RETIRED_LEASE_FORMAT, run=run, attempt_id=token,
                lease_relative_path=f"leases/{run}.{token}")


def test_reclaim_marks_the_lease_in_place_and_queues_its_decided_retry(
        tmp_path, server_lease_clock, no_directory_rename):
    shared, claim = stale_claim(tmp_path, server_lease_clock)
    records = shared.reclaim_stale()
    assert no_directory_rename == []
    decision = jobs.read_decision(tmp_path, "sample", claim.token)
    assert decision["outcome"] == "reclaimed"
    assert [record["next_attempt"] for record in records] == [decision["next_attempt"]]
    archive = tmp_path / "reclaimed/sample" / claim.token
    assert read_json(archive / "lease.json") == marker("sample", claim.token)
    assert read_json(archive / "task.json") == claim.task and not claim.claimed.exists()
    assert not (archive / "lease").exists()
    assert claim.lease.is_dir() and (claim.lease / "owner.json").is_file()
    assert jobs.retired_lease_directory(tmp_path, "sample", claim.token) == claim.lease
    assert jobs.lease_retired(tmp_path, "sample", claim.token)
    assert read_json(tmp_path / "queue/sample.json")["attempt_id"] == decision["next_attempt"]
    with pytest.raises(LeaseLost):
        shared.assert_owner(claim)
    server_lease_clock.advance(10_000)
    assert shared.reclaim_stale() == []  # The retired lease in leases/ is skipped.
    assert decisions(tmp_path) == ["sample." + claim.token + ".json"]


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing policy")
def test_reclaim_finishes_while_a_handle_inside_the_lease_denies_a_directory_rename(
        tmp_path, server_lease_clock):
    """A real open Windows handle inside the lease must not block its retirement."""
    shared, claim = stale_claim(tmp_path, server_lease_clock)
    owner = claim.lease / "owner.json"
    original = owner.read_bytes()
    with owner.open("rb") as reader:
        with pytest.raises(PermissionError):  # The former directory move fails here.
            os.rename(claim.lease, tmp_path / "moved-lease")
        records = shared.reclaim_stale()
        assert len(records) == 1 and jobs.lease_retired(tmp_path, "sample", claim.token)
        retry = shared.claim("other", "cpu", OWNER)
        assert retry.token == records[0]["next_attempt"]
        assert reader.read() == original
    assert claim.lease.is_dir() and retry.lease.is_dir()


@pytest.mark.parametrize("repetition", range(6))
def test_concurrent_passes_finish_a_standing_reclaim_exactly_once(
        tmp_path, server_lease_clock, repetition):
    root = tmp_path / ("job-" + str(repetition))
    shared, claim = stale_claim(root, server_lease_clock)
    planned = "d" * 32
    assert jobs.decide(root, "sample", claim.token, "reclaimed",
                       last_actual_progress_unix_ns=5, next_attempt=planned)[0]
    barrier = threading.Barrier(4)

    def reclaim():
        barrier.wait()
        return shared.reclaim_stale()

    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(reclaim) for _ in range(4)]
        records = [record for future in futures for record in future.result()]
    assert [(record["previous_attempt"], record["next_attempt"]) for record in records] == [
        (claim.token, planned)]
    assert records[0]["last_actual_progress_unix_ns"] == 5
    archive = root / "reclaimed/sample" / claim.token
    assert read_json(root / "queue/sample.json")["attempt_id"] == planned
    assert read_json(archive / "reclaimed.json")["next_attempt"] == planned
    assert read_json(archive / "task.json") == claim.task
    assert claim.lease.is_dir() and jobs.lease_retired(root, "sample", claim.token)
    assert shared.reclaim_stale() == []


@pytest.mark.parametrize("repetition", range(4))
def test_concurrent_stale_passes_decide_once_and_queue_one_retry(
        tmp_path, server_lease_clock, repetition):
    root = tmp_path / ("job-" + str(repetition))
    shared, claim = stale_claim(root, server_lease_clock)
    barrier = threading.Barrier(4)

    def reclaim():
        barrier.wait()
        return shared.reclaim_stale()

    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(reclaim) for _ in range(4)]
        records = [record for future in futures for record in future.result()]
    decision = jobs.read_decision(root, "sample", claim.token)
    assert [record["next_attempt"] for record in records] == [decision["next_attempt"]]
    assert read_json(root / "queue/sample.json")["attempt_id"] == decision["next_attempt"]
    assert decisions(root) == ["sample." + claim.token + ".json"]


def test_an_already_claimed_retry_is_recorded_but_never_queued_again(tmp_path, server_lease_clock):
    shared, claim = stale_claim(tmp_path, server_lease_clock)
    planned = "d" * 32
    assert jobs.decide(tmp_path, "sample", claim.token, "reclaimed",
                       last_actual_progress_unix_ns=5, next_attempt=planned)[0]
    # An earlier pass queued the retry and stopped before recording it; another
    # machine has claimed that retry since.
    write_immutable(tmp_path / "queue/sample.json", dict(claim.task, attempt_id=planned))
    retry = shared.claim("other", "cpu", OWNER)
    server_lease_clock.stamp_progress(retry)  # The retry is live and current.
    assert retry.token == planned
    records = shared.reclaim_stale()
    assert [(record["previous_attempt"], record["next_attempt"]) for record in records] == [
        (claim.token, planned)]
    assert not (tmp_path / "queue/sample.json").exists()
    shared.assert_owner(retry)
    assert shared.reclaim_stale() == []


def test_a_retry_published_before_its_record_does_not_stop_the_reclaimer(tmp_path, server_lease_clock):
    shared, claim = stale_claim(tmp_path, server_lease_clock)
    planned = "d" * 32
    assert jobs.decide(tmp_path, "sample", claim.token, "reclaimed",
                       last_actual_progress_unix_ns=5, next_attempt=planned)[0]
    write_immutable(tmp_path / "queue/sample.json", dict(claim.task, attempt_id=planned))
    retry = shared.claim("other", "cpu", OWNER)
    shared.complete(retry, dict(complete=True, run="sample", backend="cpu", precision="float32",
                                worker_references_closed=True, records=[]))
    assert jobs.publication_directory(tmp_path, "sample") == retry.lease
    records = shared.reclaim_stale()
    assert [record["next_attempt"] for record in records] == [planned]
    assert not (tmp_path / "queue/sample.json").exists()
    assert read_json(tmp_path / "done/sample.json")["attempt_id"] == planned


def test_claim_sets_aside_a_second_queue_file_of_a_leased_generation(tmp_path, server_lease_clock):
    shared = JobQueue(tmp_path, server_clock=server_lease_clock.clock)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    server_lease_clock.stamp_progress(claim)
    write_immutable(tmp_path / "queue/sample.json", claim.task)  # The same generation again.
    assert shared.claim("other", "cpu", OWNER) is None
    assert not (tmp_path / "queue/sample.json").exists()
    assert not (tmp_path / "claimed/other/sample.json").exists()
    duplicates = list((tmp_path / "claimed/other").glob("sample.duplicate.*.json"))
    assert len(duplicates) == 1 and read_json(duplicates[0]) == claim.task
    shared.assert_owner(claim)  # The original claim is untouched.
    assert shared.reclaim_stale() == []


def test_input_rejection_marks_the_lease_and_queues_its_decided_retry(
        tmp_path, monkeypatch, no_directory_rename):
    # NAS validation keeps queue/lease files on SMB. The separately admitted
    # worker results root must still be local, as in an actual alias job.
    local_results = None
    if tmp_path.drive.startswith("\\\\"):
        from pathlib import Path
        import tempfile
        local_results = Path(tempfile.mkdtemp(prefix="n112-local-results-"))
    queue_, clock, callbacks, _events = alias_fixture(
        tmp_path, monkeypatch, local_results_directory=local_results)
    claim = alias_claim(queue_, clock, callbacks)
    record = queue_.reject_before_science(claim, "InputHashMismatch: original SHA differs", {"pid": 123})
    assert no_directory_rename == []
    decision = jobs.read_decision(queue_.root, "ac52", claim.token)
    assert decision["outcome"] == "rejected" and decision["next_attempt"] == record["next_attempt"]
    archive = queue_.root / "reclaimed/ac52" / claim.token
    assert read_json(archive / "lease.json") == marker("ac52", claim.token)
    assert claim.lease.is_dir() and jobs.retired_lease_directory(queue_.root, "ac52", claim.token) == claim.lease
    assert jobs.lease_retired(queue_.root, "ac52", claim.token)
    assert read_json(queue_.root / "queue/ac52.json")["attempt_id"] == record["next_attempt"]
    clock.now += 10_000 * 1_000_000_000
    assert queue_.reclaim_stale() == []


def test_retired_lease_directory_reads_the_marker_and_the_legacy_layout(tmp_path):
    legacy = "a" * 32
    archive = tmp_path / "reclaimed/sample" / legacy
    # Without a marker the answer is the legacy location older code moved it to.
    assert jobs.retired_lease_directory(tmp_path, "sample", legacy) == archive / "lease"
    (archive / "lease").mkdir(parents=True)
    assert jobs.retired_lease_directory(tmp_path, "Sample", legacy) == archive / "lease"
    marked = "b" * 32
    write_immutable(tmp_path / "reclaimed/sample" / marked / "lease.json", marker("sample", marked))
    with pytest.raises(ValueError, match="no standing reclaim or reject decision"):
        jobs.retired_lease_directory(tmp_path, "sample", marked)
    assert jobs.decide(tmp_path, "sample", marked, "rejected")[0]
    assert jobs.retired_lease_directory(tmp_path, "sample", marked) == tmp_path / "leases" / (
        "sample." + marked)
    published = "c" * 32
    write_immutable(tmp_path / "reclaimed/sample" / published / "lease.json", marker("sample", published))
    assert jobs.decide(tmp_path, "sample", published, "publish")[0]
    with pytest.raises(ValueError, match="no standing reclaim or reject decision"):
        jobs.retired_lease_directory(tmp_path, "sample", published)
    foreign = "e" * 32
    write_immutable(tmp_path / "reclaimed/sample" / foreign / "lease.json",
                    dict(marker("sample", foreign), lease_relative_path="leases/other." + foreign))
    assert jobs.decide(tmp_path, "sample", foreign, "reclaimed")[0]
    with pytest.raises(ValueError, match="differs from its run/attempt lease"):
        jobs.retired_lease_directory(tmp_path, "sample", foreign)


def test_retired_lease_directory_uses_the_callers_reader(tmp_path):
    token = "b" * 32
    write_immutable(tmp_path / "reclaimed/sample" / token / "lease.json", marker("sample", token))
    assert jobs.decide(tmp_path, "sample", token, "reclaimed")[0]
    seen = []

    def reader(path):
        seen.append(path.name)
        return read_json(path)

    assert jobs.retired_lease_directory(tmp_path, "sample", token, reader=reader) == (
        tmp_path / "leases" / ("sample." + token))
    assert seen == ["lease.json", "sample." + token + ".json"]


def test_lease_retired_requires_a_retirement_decision_and_its_recorded_retry(tmp_path):
    token = "a" * 32
    assert jobs.lease_retired(tmp_path, "sample", token) is False
    assert jobs.decide(tmp_path, "sample", token, "reclaimed")[0]
    assert jobs.lease_retired(tmp_path, "sample", token) is False  # Retry not recorded yet.
    write_immutable(tmp_path / "reclaimed/sample" / token / "reclaimed.json", dict(run="sample"))
    assert jobs.lease_retired(tmp_path, "sample", token) is True
    published = "b" * 32
    assert jobs.decide(tmp_path, "sample", published, "publish")[0]
    write_immutable(tmp_path / "reclaimed/sample" / published / "reclaimed.json", dict(run="sample"))
    assert jobs.lease_retired(tmp_path, "sample", published) is False


def test_attempt_exists_follows_a_generation_through_queue_claim_lease_and_decision(tmp_path):
    shared = JobQueue(tmp_path)
    assert shared._attempt_exists("sample", "f" * 32) is False
    shared.enqueue(task())
    queued = read_json(tmp_path / "queue/sample.json")["attempt_id"]
    assert shared._attempt_exists("sample", queued) is True
    claim = shared.claim("machine", "cpu", OWNER)
    assert claim.token == queued and shared._attempt_exists("sample", queued) is True
    claim.claimed.unlink()  # Only the lease still names the generation.
    assert shared._attempt_exists("sample", queued) is True
    other = "e" * 32
    assert jobs.decide(tmp_path, "sample", other, "rejected")[0]
    assert shared._attempt_exists("sample", other) is True
    assert shared._attempt_exists("sample", "f" * 32) is False


def test_retire_lease_writes_one_marker_and_keeps_a_legacy_archive(tmp_path):
    shared = JobQueue(tmp_path)
    token = "a" * 32
    archive = shared._retire_lease("sample", token)
    assert archive == tmp_path / "reclaimed/sample" / token
    first = (archive / "lease.json").read_bytes()
    assert shared._retire_lease("sample", token) == archive  # A repeat is a no-op.
    assert (archive / "lease.json").read_bytes() == first
    assert read_json(archive / "lease.json") == marker("sample", token)
    legacy = "b" * 32
    (tmp_path / "reclaimed/sample" / legacy / "lease").mkdir(parents=True)
    shared._retire_lease("sample", legacy)
    assert not (tmp_path / "reclaimed/sample" / legacy / "lease.json").exists()


def test_watchdog_restarts_a_slot_whose_attempt_was_reclaimed_in_place(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import worker
    token = "c" * 32
    task_ = {"run": "ac52", "attempt_id": token}
    (tmp_path / "leases" / ("ac52." + token)).mkdir(parents=True)  # It stays after reclaim.
    assert jobs.decide(tmp_path, "ac52", token, "reclaimed", next_attempt="d" * 32)[0]
    write_immutable(tmp_path / "reclaimed/ac52" / token / "reclaimed.json",
                    dict(run="ac52", previous_attempt=token, next_attempt="d" * 32))
    crashed = SimpleNamespace(exitcode=1, alive=True, join=lambda **kwargs: None)
    crashed.is_alive = lambda: crashed.alive
    replacement = SimpleNamespace(exitcode=0, join=lambda **kwargs: None, is_alive=lambda: False)
    events = [dict(kind="claimed", run="ac52", attempt_id=token, worker_slot=0, worker_generation="g")]

    class Messages:
        def get(self, **kwargs):
            if not events:
                raise queue.Empty()
            crashed.alive = False  # The child stopped after its claim.
            return events.pop(0)

        def get_nowait(self):
            raise queue.Empty()

    passes = []

    class Shared:
        def recover_publications(self):
            passes.append(1)
            if len(passes) > 3:
                raise AssertionError("The watchdog kept waiting on a retired lease")

        def reclaim_stale(self):
            return []

    def slot(process, generation):
        return dict(number=0, generation=generation, process=process, root=None, observed=[],
                    last_heartbeat=0, claim=None, retired=False, waiting_reclaim=False,
                    finished=False)

    started = [slot(crashed, "g"), slot(replacement, "h")]
    monkeypatch.setattr(worker, "JobQueue", lambda root, *, machine_id: Shared())
    monkeypatch.setattr(worker.mp, "get_context",
                        lambda *args: SimpleNamespace(Queue=Messages, Event=threading.Event))
    monkeypatch.setattr(worker, "_start_slot", lambda *args: started.pop(0))
    monkeypatch.setattr(worker, "read_json", lambda path: task_)
    monkeypatch.setattr(worker, "verify_startup", lambda *args, **kwargs: ({}, "a" * 40))
    monkeypatch.setattr(worker, "_record_retirement",
                        lambda value, args, identity: value.update(retired=True) or [])
    args = SimpleNamespace(job=tmp_path, workers=1, watchdog=True, once=False)
    log = io.StringIO()
    assert worker._parent(args, "Main-cpu", {}, "a" * 40, log) == 0
    assert started == [] and passes == [1]
    assert '"kind": "watchdog_restarted"' in log.getvalue()
