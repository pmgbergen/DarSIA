"""Exclusive-create attempt decisions; tiny synthetic metadata only.

Publication never moves a lease directory: the decision file is the ownership
transition and a run-level marker file names the published lease in place.
"""
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import job_queue as jobs
from darsia.fluidflower.pipeline.job_queue import JobQueue, LeaseLost, read_json, write_immutable
from tests.server_lease_clock_fixture import server_lease_clock
from .test_alias_queue_runtime import (
    claim_fixture as alias_claim, completion, fixture as alias_fixture, payload_callback,
)

OWNER = dict(pid=123, process_create_time="1.0")


def task(run="sample"):
    return dict(run=run, commit="1" * 40, calibration_sha256="2" * 64,
                configuration=dict(path=r"\\server\share\config.json", sha256="3" * 64),
                precision="float32", backend="any")


def result(claim):
    return dict(complete=True, run=claim.task["run"], backend="cpu",
                precision="float32", worker_references_closed=True, records=[])


def sharing_error():
    error = PermissionError("fixture SMB sharing violation")
    error.winerror = 5
    return error


def decisions(root):
    """Decision documents; a lost exclusive create may leave a hidden temporary."""
    return sorted(path.name for path in (Path(root) / "decisions").iterdir()
                  if not path.name.startswith("."))


def stale_claim(root, clock, run="sample"):
    shared = JobQueue(root, server_clock=clock.clock)
    shared.enqueue(task(run))
    claim = shared.claim("machine", "cpu", OWNER)
    clock.stamp_progress(claim)
    clock.advance(601)
    return shared, claim


@pytest.fixture
def no_directory_rename(monkeypatch):
    """Fail any attempt to rename a directory; file renames stay allowed."""
    attempts = []

    def guard(original):
        def checked(source, destination, *args, **kwargs):
            if Path(source).is_dir():
                attempts.append((str(source), str(destination)))
                raise AssertionError("Directory rename attempted: " + str(source))
            return original(source, destination, *args, **kwargs)
        return checked

    monkeypatch.setattr(jobs, "rename", guard(jobs.rename))
    monkeypatch.setattr(jobs, "_windows_rename", guard(jobs._windows_rename))
    monkeypatch.setattr(os, "rename", guard(os.rename))
    monkeypatch.setattr(os, "replace", guard(os.replace))
    return attempts


def test_publication_keeps_lease_in_place_with_decision_marker_results_and_done(tmp_path, server_lease_clock):
    shared = JobQueue(tmp_path, server_clock=server_lease_clock.clock)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    value = result(claim)
    assert shared.complete(claim, value) is value
    assert claim.lease.is_dir() and (claim.lease / "owner.json").is_file()
    assert (claim.lease / "publication.json").is_file()
    decision = jobs.read_decision(tmp_path, "sample", claim.token)
    assert decision["format"] == jobs.DECISION_FORMAT and decision["outcome"] == "publish"
    assert decision["run"] == "sample" and decision["attempt_id"] == claim.token
    assert decision["lease_relative_path"] == "leases/sample." + claim.token
    assert isinstance(decision["decided_utc"], str)
    marker = tmp_path / "publishing" / "sample"
    assert marker.is_file() and read_json(marker) == dict(
        format=jobs.PUBLICATION_MARKER_FORMAT, run="sample", attempt_id=claim.token,
        lease_relative_path="leases/sample." + claim.token)
    assert jobs.publication_directory(tmp_path, "sample") == claim.lease
    assert read_json(tmp_path / "results/sample/completed.json") == value
    assert read_json(tmp_path / "done/sample.json")["attempt_id"] == claim.token
    assert shared.aggregate()["complete"] is True
    shared.assert_owner(claim)  # The standing decision is this attempt's publication.
    # A published lease stays in leases/ and is never reclaimed or re-run later.
    server_lease_clock.stamp_progress(claim)
    server_lease_clock.advance(10_000)
    assert shared.reclaim_stale() == []
    assert claim.lease.is_dir() and not (tmp_path / "reclaimed/sample").exists()
    assert not (tmp_path / "queue/sample.json").exists()
    assert shared.claim("other", "cpu", OWNER) is None
    assert decisions(tmp_path) == ["sample." + claim.token + ".json"]


def test_ordinary_publication_attempts_no_directory_rename(tmp_path, no_directory_rename):
    shared = JobQueue(tmp_path)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    shared.complete(claim, result(claim))
    shared.recover_publications()
    assert no_directory_rename == []
    assert (tmp_path / "done/sample.json").is_file() and claim.lease.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing policy")
def test_publication_finishes_while_a_lease_reader_denies_directory_rename(tmp_path):
    """A real open Windows handle must not prevent in-place publication."""
    shared = JobQueue(tmp_path)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    owner = claim.lease / "owner.json"
    original = owner.read_bytes()
    value = result(claim)
    with owner.open("rb") as reader:
        assert shared.complete(claim, value) is value
        shared.recover_publications()
        assert reader.read() == original
        assert owner.read_bytes() == original
        assert jobs.publication_directory(tmp_path, "sample") == claim.lease
        assert read_json(tmp_path / "results/sample/completed.json") == value
        assert read_json(tmp_path / "done/sample.json")["attempt_id"] == claim.token
    assert claim.lease.is_dir()
    assert jobs.read_decision(tmp_path, "sample", claim.token)["outcome"] == "publish"


def test_alias_payload_publication_attempts_no_directory_rename(tmp_path, monkeypatch, no_directory_rename):
    queue_, clock, callbacks, _events = alias_fixture(tmp_path, monkeypatch)
    callback = payload_callback(callbacks["publication_factory"])
    claim = alias_claim(queue_, clock, dict(callbacks, publication_factory=callback))
    queue_.complete(claim, completion())
    queue_.recover_publications()
    assert no_directory_rename == []
    done = read_json(queue_.root / "done/ac52.json")
    assert done["products_index"]["relative_path"].startswith("leases/ac52." + claim.token + "/")
    assert jobs.publication_directory(queue_.root, "ac52") == claim.lease


def test_reclaim_decided_before_publish_wins_and_publisher_gets_lease_lost(tmp_path, monkeypatch, server_lease_clock):
    shared, claim = stale_claim(tmp_path, server_lease_clock)
    real = jobs.decide

    def reclaimer_first(root, run, token, outcome, **extra):
        if outcome == "publish":
            # A reclaimer decides after the publisher's last ownership check.
            assert real(root, run, token, "reclaimed", last_actual_progress_unix_ns=1)[0] is True
        return real(root, run, token, outcome, **extra)

    monkeypatch.setattr(jobs, "decide", reclaimer_first)
    with pytest.raises(LeaseLost, match="Reclaim won before publication"):
        shared.complete(claim, result(claim))
    monkeypatch.setattr(jobs, "decide", real)
    assert jobs.read_decision(tmp_path, "sample", claim.token)["outcome"] == "reclaimed"
    assert not (tmp_path / "publishing/sample").exists()
    assert not (tmp_path / "done/sample.json").exists()
    with pytest.raises(LeaseLost, match="reclaimed"):
        shared.assert_owner(claim)
    with pytest.raises(LeaseLost):
        shared.fail(claim, "stale writer", OWNER)
    assert not (tmp_path / "failed/sample.json").exists()
    # The standing reclaim decision is archived later without being re-decided.
    records = shared.reclaim_stale()
    assert [record["previous_attempt"] for record in records] == [claim.token]
    assert records[0]["last_actual_progress_unix_ns"] == 1
    assert claim.lease.is_dir()  # Retired in place: a marker names it, nothing moved.
    assert jobs.retired_lease_directory(tmp_path, "sample", claim.token) == claim.lease
    assert (claim.lease / "owner.json").is_file()
    assert read_json(tmp_path / "queue/sample.json")["attempt_id"] == records[0]["next_attempt"]
    assert decisions(tmp_path) == ["sample." + claim.token + ".json"]
    assert shared.reclaim_stale() == []


def test_publish_decided_before_reclaim_wins_and_reclaimer_skips(tmp_path, monkeypatch, server_lease_clock):
    shared, claim = stale_claim(tmp_path, server_lease_clock)
    real = jobs.decide
    published = []

    def publisher_first(root, run, token, outcome, **extra):
        if outcome == "reclaimed" and not published:
            # The publisher finishes after this reclaimer's staleness and
            # publication.json checks, but before the reclaimer's decision.
            monkeypatch.setattr(jobs, "decide", real)
            published.append(shared.complete(claim, result(claim)))
        return real(root, run, token, outcome, **extra)

    monkeypatch.setattr(jobs, "decide", publisher_first)
    assert shared.reclaim_stale() == []
    assert published and jobs.read_decision(tmp_path, "sample", claim.token)["outcome"] == "publish"
    assert read_json(tmp_path / "done/sample.json")["attempt_id"] == claim.token
    assert claim.lease.is_dir() and not (tmp_path / "queue/sample.json").exists()
    assert not (tmp_path / "reclaimed/sample").exists()
    assert decisions(tmp_path) == ["sample." + claim.token + ".json"]
    assert shared.reclaim_stale() == []


@pytest.mark.parametrize("repetition", range(8))
def test_concurrent_publish_and_reclaim_leave_exactly_one_consistent_decision(
        tmp_path, server_lease_clock, repetition):
    root = tmp_path / ("job-" + str(repetition))
    shared, claim = stale_claim(root, server_lease_clock)
    barrier = threading.Barrier(2)

    def publish():
        barrier.wait()
        try:
            shared.complete(claim, result(claim))
            return True
        except LeaseLost:
            return False

    def reclaim():
        barrier.wait()
        return shared.reclaim_stale()

    with ThreadPoolExecutor(2) as pool:
        publication, reclamation = pool.submit(publish), pool.submit(reclaim)
        won, requeued = publication.result(), reclamation.result()
    assert decisions(root) == ["sample." + claim.token + ".json"]
    outcome = jobs.read_decision(root, "sample", claim.token)["outcome"]
    assert won == (outcome == "publish") and won != bool(requeued)
    assert (root / "done/sample.json").exists() == won
    assert (root / "queue/sample.json").exists() == (not won)
    assert claim.lease.is_dir()  # Neither outcome moves the lease.
    assert (root / "reclaimed/sample" / claim.token / "lease.json").is_file() == (not won)


@pytest.mark.parametrize("outcome", ["reclaimed", "rejected"])
def test_retirement_decision_revokes_the_owner_before_its_lease_is_marked(tmp_path, server_lease_clock, outcome):
    shared = JobQueue(tmp_path, server_clock=server_lease_clock.clock)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    shared.assert_owner(claim)
    assert jobs.decide(tmp_path, "sample", claim.token, outcome)[0] is True
    assert claim.lease.is_dir()  # Only the decision exists; nothing has moved.
    with pytest.raises(LeaseLost, match=outcome):
        shared.assert_owner(claim)
    with pytest.raises(LeaseLost):
        shared.progress(claim, dict(state="frame_published", frame_index=1))
    with pytest.raises(LeaseLost):
        shared.complete(claim, result(claim))
    with pytest.raises(LeaseLost):
        shared.fail(claim, "stale writer", OWNER)
    assert not list(claim.lease.glob("progress_*.json"))
    assert not (claim.lease / "publication.json").exists()
    assert not (tmp_path / "publishing/sample").exists()
    assert not (tmp_path / "done/sample.json").exists() and not (tmp_path / "failed/sample.json").exists()


def test_stale_lease_with_saved_publication_is_published_not_rerun(tmp_path, monkeypatch):
    queue_, clock, callbacks, _events = alias_fixture(tmp_path, monkeypatch)
    callback = payload_callback(callbacks["publication_factory"])
    claim = alias_claim(queue_, clock, dict(callbacks, publication_factory=callback))
    real = jobs.decide
    monkeypatch.setattr(jobs, "decide", lambda *args, **kwargs: (_ for _ in ()).throw(sharing_error()))
    with pytest.raises(PermissionError):
        queue_.complete(claim, completion())
    monkeypatch.setattr(jobs, "decide", real)
    queue_.fail(claim, "PermissionError after the saved products", {"pid": 123})
    assert (claim.lease / "publication.json").is_file()
    assert jobs.read_decision(queue_.root, "ac52", claim.token) is None
    clock.now += 599 * 1_000_000_000
    assert queue_.reclaim_stale() == []
    assert jobs.read_decision(queue_.root, "ac52", claim.token) is None
    clock.now += 1_000_000_000
    assert queue_.reclaim_stale() == []  # Published, never reclaimed.
    decision = jobs.read_decision(queue_.root, "ac52", claim.token)
    assert decision["outcome"] == "publish" and decision["decided_by"] == "reclaim_stale_recovery"
    assert (queue_.root / "results/ac52/completed.json").read_bytes() == (
        claim.output / "candidate_completed.json").read_bytes()
    done = read_json(queue_.root / "done/ac52.json")
    assert done["attempt_id"] == claim.token and "payload_publication" in done
    assert not (queue_.root / "queue/ac52.json").exists()
    assert not (queue_.root / "reclaimed/ac52").exists() and not (queue_.root / "failed/ac52.json").exists()
    assert claim.lease.is_dir() and jobs.publication_directory(queue_.root, "ac52") == claim.lease
    assert queue_.claim("Main", "cpu", {}, **callbacks) is None
    assert queue_.aggregate()["complete"] is True


def test_publisher_stopped_between_decision_and_marker_is_finished_by_reclaim(tmp_path, monkeypatch, server_lease_clock):
    shared = JobQueue(tmp_path, server_clock=server_lease_clock.clock)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    original = shared._write_publication_marker
    monkeypatch.setattr(shared, "_write_publication_marker",
                        lambda *args: (_ for _ in ()).throw(OSError("stopped after decision")))
    with pytest.raises(OSError, match="stopped after decision"):
        shared.complete(claim, result(claim))
    monkeypatch.setattr(shared, "_write_publication_marker", original)
    assert jobs.read_decision(tmp_path, "sample", claim.token)["outcome"] == "publish"
    shared.recover_publications()  # No marker yet: nothing names this publication.
    server_lease_clock.stamp_progress(claim)
    server_lease_clock.advance(599)
    assert shared.reclaim_stale() == []  # A live publisher is never raced.
    assert not (tmp_path / "publishing/sample").exists()
    assert not (tmp_path / "done/sample.json").exists()
    server_lease_clock.advance(1)
    assert shared.reclaim_stale() == []
    assert jobs.publication_directory(tmp_path, "sample") == claim.lease
    assert read_json(tmp_path / "done/sample.json")["attempt_id"] == claim.token
    assert not (tmp_path / "queue/sample.json").exists() and claim.lease.is_dir()


def test_complete_retry_after_crash_between_decision_and_finish_is_idempotent(tmp_path, monkeypatch):
    shared = JobQueue(tmp_path)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    original = shared._finish_publication
    monkeypatch.setattr(shared, "_finish_publication",
                        lambda directory: (_ for _ in ()).throw(OSError("crash after decision")))
    with pytest.raises(OSError, match="crash after decision"):
        shared.complete(claim, result(claim))
    monkeypatch.setattr(shared, "_finish_publication", original)
    assert jobs.read_decision(tmp_path, "sample", claim.token)["outcome"] == "publish"
    assert jobs.publication_directory(tmp_path, "sample") == claim.lease
    assert not (tmp_path / "done/sample.json").exists()
    durable = [claim.output / "candidate_completed.json", claim.lease / "publication.json",
               jobs.decision_path(tmp_path, "sample", claim.token), tmp_path / "publishing/sample"]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in durable}
    value = result(claim)
    assert shared.complete(claim, value) is value
    assert shared.complete(claim, value) is value  # Every further retry is a no-op.
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in durable} == before
    assert read_json(tmp_path / "done/sample.json")["attempt_id"] == claim.token
    assert decisions(tmp_path) == ["sample." + claim.token + ".json"]
    assert shared.aggregate()["complete"] is True
    with pytest.raises(FileExistsError):  # A different completion is never a retry.
        shared.complete(claim, dict(value, records=["different"]))


def test_recovery_finishes_legacy_publishing_directory_and_in_place_marker(tmp_path, monkeypatch):
    shared = JobQueue(tmp_path)
    shared.enqueue(task("legacy"))
    shared.enqueue(task("marked"))
    claims = {claim.task["run"]: claim for claim in
              (shared.claim("machine", "cpu", OWNER), shared.claim("machine", "cpu", OWNER))}
    legacy, marked = claims["legacy"], claims["marked"]
    real = jobs.decide
    monkeypatch.setattr(jobs, "decide", lambda *args, **kwargs: (_ for _ in ()).throw(sharing_error()))
    with pytest.raises(PermissionError):
        shared.complete(legacy, result(legacy))
    monkeypatch.setattr(jobs, "decide", real)
    legacy.lease.rename(tmp_path / "publishing" / "legacy")  # The old moved layout.
    original = shared._finish_publication
    monkeypatch.setattr(shared, "_finish_publication",
                        lambda directory: (_ for _ in ()).throw(OSError("crash after decision")))
    with pytest.raises(OSError):
        shared.complete(marked, result(marked))
    monkeypatch.setattr(shared, "_finish_publication", original)
    (tmp_path / "publishing" / (".marked." + "0" * 32 + ".partial")).write_bytes(b"{")
    assert jobs.publication_directory(tmp_path, "legacy") == tmp_path / "publishing" / "legacy"
    assert jobs.publication_directory(tmp_path, "marked") == marked.lease
    assert jobs.publication_directory(tmp_path, "absent") is None
    shared.recover_publications()
    shared.recover_publications()
    assert read_json(tmp_path / "done/legacy.json")["attempt_id"] == legacy.token
    assert read_json(tmp_path / "done/marked.json")["attempt_id"] == marked.token
    assert shared.aggregate()["complete"] is True and marked.lease.is_dir()


@pytest.mark.parametrize("existing", ["foreign_marker", "legacy_directory"])
def test_publication_refuses_a_marker_of_another_attempt_or_a_legacy_directory(tmp_path, existing):
    shared = JobQueue(tmp_path)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    path = tmp_path / "publishing" / "sample"
    other = "f" * 32
    if existing == "foreign_marker":
        write_immutable(path, dict(format=jobs.PUBLICATION_MARKER_FORMAT, run="sample",
                                   attempt_id=other, lease_relative_path="leases/sample." + other))
        before = path.read_bytes()
    else:
        path.mkdir()
    with pytest.raises(RuntimeError, match="another attempt" if existing == "foreign_marker" else "legacy"):
        shared.complete(claim, result(claim))
    assert not (tmp_path / "results/sample/completed.json").exists()
    assert not (tmp_path / "done/sample.json").exists()
    if existing == "foreign_marker":
        assert path.read_bytes() == before
        # The foreign attempt has no standing publish decision; no reader adopts it.
        with pytest.raises(ValueError, match="no standing publish decision"):
            jobs.publication_directory(tmp_path, "sample")
        with pytest.raises(ValueError, match="no standing publish decision"):
            shared.recover_publications()
    else:
        assert path.is_dir() and not list(path.iterdir())


def test_marker_must_name_its_own_attempt_lease_and_publish_decision(tmp_path):
    shared = JobQueue(tmp_path)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    path = tmp_path / "publishing" / "sample"
    write_immutable(path, dict(format=jobs.PUBLICATION_MARKER_FORMAT, run="sample",
                               attempt_id=claim.token, lease_relative_path="leases/other." + claim.token))
    assert jobs.decide(tmp_path, "sample", claim.token, "publish")[0] is True
    with pytest.raises(ValueError, match="differs from its run/attempt lease"):
        jobs.publication_directory(tmp_path, "sample")
    with pytest.raises(ValueError, match="32-hex"):
        jobs.decision_path(tmp_path, "sample", "not-a-token")


def test_input_rejection_loses_against_an_existing_publish_decision(tmp_path, monkeypatch):
    queue_, clock, callbacks, _events = alias_fixture(tmp_path, monkeypatch)
    claim = alias_claim(queue_, clock, callbacks)
    assert jobs.decide(queue_.root, "ac52", claim.token, "publish")[0] is True
    with pytest.raises(LeaseLost, match="already retired this input admission"):
        queue_.reject_before_science(claim, "InputHashMismatch after a publication decision", {"pid": 123})
    assert jobs.read_decision(queue_.root, "ac52", claim.token)["outcome"] == "publish"
    assert claim.lease.is_dir() and claim.claimed.is_file()
    assert not (queue_.root / "reclaimed/ac52").exists()
    assert not (queue_.root / "queue/ac52.json").exists()


def test_input_rejection_records_its_decision_and_then_archives(tmp_path, monkeypatch):
    queue_, clock, callbacks, _events = alias_fixture(tmp_path, monkeypatch)
    claim = alias_claim(queue_, clock, callbacks)
    record = queue_.reject_before_science(claim, "InputHashMismatch: original SHA differs", {"pid": 123})
    assert jobs.read_decision(queue_.root, "ac52", claim.token)["outcome"] == "rejected"
    assert claim.lease.is_dir() and (claim.lease / "owner.json").is_file()
    assert jobs.retired_lease_directory(queue_.root, "ac52", claim.token) == claim.lease
    assert jobs.read_decision(queue_.root, "ac52", claim.token)["next_attempt"] == record["next_attempt"]
    assert read_json(queue_.root / "queue/ac52.json")["attempt_id"] == record["next_attempt"]
    clock.now += 10_000 * 1_000_000_000
    assert queue_.reclaim_stale() == []


def test_interrupted_reclaim_archive_is_finished_without_a_second_decision_or_retry(
        tmp_path, monkeypatch, server_lease_clock):
    shared, claim = stale_claim(tmp_path, server_lease_clock)
    real = jobs.rename

    def interrupted(source, destination):
        if Path(source) == claim.claimed:  # The task-file archive step is interrupted.
            raise sharing_error()
        return real(source, destination)

    monkeypatch.setattr(jobs, "rename", interrupted)
    assert shared.reclaim_stale() == []  # N111: the step is deferred, never a crash.
    monkeypatch.setattr(jobs, "rename", real)
    decision = jobs.read_decision(tmp_path, "sample", claim.token)
    assert decision["outcome"] == "reclaimed" and claim.lease.is_dir()
    with pytest.raises(LeaseLost):
        shared.assert_owner(claim)
    assert not (tmp_path / "queue/sample.json").exists()
    records = shared.reclaim_stale()
    assert [record["previous_attempt"] for record in records] == [claim.token]
    assert records[0]["last_actual_progress_unix_ns"] == decision["last_actual_progress_unix_ns"]
    assert claim.lease.is_dir() and records[0]["next_attempt"] == decision["next_attempt"]
    assert read_json(tmp_path / "queue/sample.json")["attempt_id"] == records[0]["next_attempt"]
    assert decisions(tmp_path) == ["sample." + claim.token + ".json"]
    assert shared.reclaim_stale() == []


def test_reclaim_archive_retry_reuses_an_already_queued_retry(tmp_path, server_lease_clock):
    shared, claim = stale_claim(tmp_path, server_lease_clock)
    assert jobs.decide(tmp_path, "sample", claim.token, "reclaimed", last_actual_progress_unix_ns=7)[0]
    queued = dict(claim.task, attempt_id="e" * 32)  # An interrupted pass already queued it.
    write_immutable(tmp_path / "queue/sample.json", queued)
    before = (tmp_path / "queue/sample.json").read_bytes()
    records = shared.reclaim_stale()
    assert [(record["previous_attempt"], record["next_attempt"]) for record in records] == [
        (claim.token, "e" * 32)]
    assert (tmp_path / "queue/sample.json").read_bytes() == before
    assert read_json(tmp_path / "reclaimed/sample" / claim.token / "task.json") == claim.task
    assert shared.reclaim_stale() == []


def test_ownerless_reclaim_interrupted_after_its_decision_is_finished_later(tmp_path, monkeypatch):
    queue_, clock, _callbacks, _events = alias_fixture(tmp_path, monkeypatch)
    pending = queue_.root / "queue/ac52.json"
    original = read_json(pending)
    claimed = queue_.root / "claimed/main/ac52.json"
    claimed.parent.mkdir()
    jobs.rename(pending, claimed)  # The claimer died before its lease/owner existed.
    assert queue_.reclaim_stale() == []
    clock.now += 600 * 1_000_000_000
    real = jobs.rename

    def interrupted(source, destination):
        if Path(source) == claimed:
            raise sharing_error()
        return real(source, destination)

    monkeypatch.setattr(jobs, "rename", interrupted)
    assert queue_.reclaim_stale() == []  # N111: the step is deferred, never a crash.
    monkeypatch.setattr(jobs, "rename", real)
    assert jobs.read_decision(queue_.root, "ac52", original["attempt_id"])["outcome"] == "reclaimed"
    assert claimed.is_file() and not pending.exists()
    records = queue_.reclaim_stale()
    assert len(records) == 1 and records[0]["previous_attempt"] == original["attempt_id"]
    assert records[0]["interrupted_claim_before_science"] is True
    assert records[0]["next_attempt"] == jobs.read_decision(
        queue_.root, "ac52", original["attempt_id"])["next_attempt"]
    assert read_json(pending)["attempt_id"] == records[0]["next_attempt"]
    assert not claimed.exists() and queue_.reclaim_stale() == []


def test_worker_owner_readers_accept_the_in_place_published_lease(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import worker_io
    queue_, clock, callbacks, _events = alias_fixture(tmp_path, monkeypatch)
    claim = alias_claim(queue_, clock, callbacks)
    queue_.complete(claim, completion())
    assert (queue_.root / "publishing/ac52").is_file()
    owner = worker_io.owner_for_attempt(queue_.root, claim.task, claim.machine)
    assert owner == read_json(claim.lease / "owner.json")
    event = dict(run="ac52", attempt_id=claim.token)
    assert worker_io.claimed_task_after_archive(queue_.root, claim.machine, event) == claim.task
    claim.claimed.unlink()  # Only the in-place lease still binds the generation.
    assert worker_io.claimed_task_after_archive(queue_.root, claim.machine, event) == claim.task


def test_watchdog_finishes_a_slot_whose_run_is_published_in_place(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import worker
    token = "c" * 32
    task_ = {"run": "ac52", "attempt_id": token}
    (tmp_path / "leases" / ("ac52." + token)).mkdir(parents=True)  # It stays after publication.
    (tmp_path / "publishing").mkdir()
    (tmp_path / "publishing" / "ac52").write_text("{}", encoding="utf-8")  # Its marker file.
    process = SimpleNamespace(exitcode=1, alive=True, join=lambda **kwargs: None)
    process.is_alive = lambda: process.alive
    events = [dict(kind="claimed", run="ac52", attempt_id=token, worker_slot=0, worker_generation="g")]

    class Messages:
        def get(self, **kwargs):
            if not events:
                raise queue.Empty()
            process.alive = False  # The child stopped after publishing, before "completed".
            return events.pop(0)

        def get_nowait(self):
            raise queue.Empty()

    passes = []

    class Shared:
        def recover_publications(self):
            passes.append(1)
            if len(passes) > 3:
                raise AssertionError("The watchdog kept waiting on a published lease")

        def reclaim_stale(self):
            return []

    slot = dict(number=0, generation="g", process=process, root=None, observed=[],
                last_heartbeat=0, claim=None, retired=False, waiting_reclaim=False, finished=False)
    monkeypatch.setattr(worker, "JobQueue", lambda root, *, machine_id: Shared())
    monkeypatch.setattr(worker.mp, "get_context",
                        lambda *args: SimpleNamespace(Queue=Messages, Event=threading.Event))
    monkeypatch.setattr(worker, "_start_slot", lambda *args: slot)
    monkeypatch.setattr(worker, "read_json", lambda path: task_)
    monkeypatch.setattr(worker, "_record_retirement",
                        lambda value, args, identity: value.update(retired=True) or [])
    args = SimpleNamespace(job=tmp_path, workers=1, watchdog=True, once=False)
    assert worker._parent(args, "Main-cpu", {}, "a" * 40, io.StringIO()) == 0
    assert slot["finished"] and passes == [1]
