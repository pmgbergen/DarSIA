"""Concurrency and failure fixtures use only tiny metadata, never FFAC inputs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import queue
import sys
import threading
import types
from pathlib import Path

from tests.server_lease_clock_fixture import server_lease_clock

import pytest

from darsia.fluidflower.job_queue import (
    JobQueue, LeaseLost, read_json, write_immutable,
)
from darsia.fluidflower import worker


def task(run="sample", backend="any"):
    return dict(run=run, commit="1" * 40, calibration_sha256="2" * 64,
                configuration=dict(path=r"\\server\share\config.json", sha256="3" * 64),
                precision="float32", backend=backend)


def result(claim, backend="cpu"):
    return dict(complete=True, run=claim.task["run"], backend=backend,
                precision="float32", worker_references_closed=True, records=[])


OWNER = dict(pid=123, process_create_time="1.0")


def test_concurrent_claimers_compute_a_run_only_once(tmp_path):
    shared = JobQueue(tmp_path)
    shared.enqueue(task())
    barrier = threading.Barrier(16)

    def claim(index):
        barrier.wait()
        return shared.claim(f"machine-{index}", "cpu", OWNER)

    with ThreadPoolExecutor(16) as pool:
        claims = list(pool.map(claim, range(16)))
    actual = [c for c in claims if c is not None]
    assert len(actual) == 1
    assert not list((tmp_path / "queue").glob("*.json"))
    shared.complete(actual[0], result(actual[0]))
    assert shared.aggregate()["completed_count"] == 1


def test_duplicate_enqueue_never_replaces_task_or_terminal(tmp_path):
    shared = JobQueue(tmp_path)
    queued = shared.enqueue(task("Run"))
    original = queued.read_bytes()
    with pytest.raises(FileExistsError):
        shared.enqueue(task("run"))
    assert queued.read_bytes() == original
    claim = shared.claim("machine", "cpu", OWNER)
    shared.complete(claim, result(claim))
    complete = tmp_path / "results/run/completed.json"
    before = complete.read_bytes()
    with pytest.raises(FileExistsError):
        write_immutable(complete, dict(complete=False))
    assert complete.read_bytes() == before
    assert list(complete.parent.glob("*.partial"))


def test_real_progress_prevents_early_reclaim_and_stale_writer_cannot_publish(tmp_path, server_lease_clock):
    shared = JobQueue(tmp_path, server_clock=server_lease_clock.clock)
    shared.enqueue(task())
    old = shared.claim("old-machine", "cpu", OWNER)
    server_lease_clock.stamp_progress(old)
    server_lease_clock.advance(500)
    shared.progress(old, dict(state="frame_published", frame_index=1))
    server_lease_clock.stamp_progress(old)
    server_lease_clock.advance(599)
    assert shared.reclaim_stale() == []
    server_lease_clock.advance(1)
    assert len(shared.reclaim_stale()) == 1
    newer = shared.claim("new-machine", "cpu", OWNER)
    assert newer.token != old.token and newer.output != old.output
    with pytest.raises(LeaseLost):
        shared.complete(old, result(old))
    with pytest.raises(LeaseLost):
        shared.fail(old, "Old attempt failed.\n", OWNER)
    shared.complete(newer, result(newer))
    assert read_json(tmp_path / "done/sample.json")["attempt_id"] == newer.token
    assert not (tmp_path / "failed/sample.json").exists()
    assert (old.output / "failed.json").exists()


def test_publication_and_reclaim_have_one_atomic_winner(tmp_path, server_lease_clock):
    shared = JobQueue(tmp_path, server_clock=server_lease_clock.clock)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    server_lease_clock.stamp_progress(claim)
    server_lease_clock.advance(601)
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
        publication = pool.submit(publish)
        reclamation = pool.submit(reclaim)
        won, requeued = publication.result(), reclamation.result()
    assert won != bool(requeued)
    assert (tmp_path / "done/sample.json").exists() == won
    assert (tmp_path / "queue/sample.json").exists() == bool(requeued)


def test_metadata_publication_recovers_without_replaying_science(tmp_path, monkeypatch):
    shared = JobQueue(tmp_path)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    original = shared._finish_publication
    monkeypatch.setattr(shared, "_finish_publication", lambda p: (_ for _ in ()).throw(OSError("Interrupted metadata publish")))
    with pytest.raises(OSError):
        shared.complete(claim, result(claim))
    assert not (tmp_path / "done/sample.json").exists()
    monkeypatch.setattr(shared, "_finish_publication", original)
    shared.recover_publications()
    assert shared.aggregate()["complete"] is True


def test_repeated_recovery_of_completed_run_performs_no_writes(tmp_path, monkeypatch):
    shared = JobQueue(tmp_path)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    shared.complete(claim, result(claim))
    before = {p.relative_to(tmp_path): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in tmp_path.rglob("*") if p.is_file()}

    def unexpected_write(*args, **kwargs):
        raise AssertionError("Recovery tried to write an already matching publication")

    monkeypatch.setattr("darsia.fluidflower.job_queue.write_immutable", unexpected_write)
    for _ in range(10):
        shared.recover_publications()
    after = {p.relative_to(tmp_path): (p.read_bytes(), p.stat().st_mtime_ns)
             for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert not list(tmp_path.rglob("*.partial"))


def test_output_files_do_not_imply_completion(tmp_path):
    shared = JobQueue(tmp_path)
    shared.enqueue(task())
    claim = shared.claim("machine", "cpu", OWNER)
    (claim.output / "products.npz").write_bytes(b"partial fixture")
    assert shared.aggregate()["completed_count"] == 0
    with pytest.raises(ValueError):
        shared.complete(claim, dict(result(claim), worker_references_closed=False))
    assert not (tmp_path / "done/sample.json").exists()


def test_output_escape_is_rejected_before_claim(tmp_path):
    shared = JobQueue(tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        shared.enqueue(dict(task(), output_directory=r"\\other\share\unowned"))
    assert not list((tmp_path / "queue").iterdir())


def test_expected_runs_prevents_partial_job_completion(tmp_path):
    shared = JobQueue(tmp_path)
    write_immutable(tmp_path / "job.json", dict(expected_runs=["first", "second"]))
    shared.enqueue(task("first"))
    claim = shared.claim("machine", "cpu", OWNER)
    shared.complete(claim, result(claim))
    assert shared.aggregate()["task_count"] == 2
    assert shared.aggregate()["complete"] is False


@pytest.mark.parametrize("bytes_", [b"{not valid JSON", b'{"run":"bad"}'])
def test_bad_task_is_quarantined_once_and_next_valid_task_is_claimed(tmp_path, bytes_):
    errors = []
    shared = JobQueue(tmp_path, on_error=errors.append)
    (tmp_path / "queue/bad.json").write_bytes(bytes_)
    shared.enqueue(task("good"))
    claim = shared.claim("machine", "cpu", OWNER)
    assert claim.task["run"] == "good"
    failed = read_json(tmp_path / "failed/bad.json")
    assert failed["traceback"].startswith("Traceback (most recent call last):")
    assert Path(failed["original_queue_bytes"]["path"]).read_bytes() == bytes_
    assert len(errors) == 1
    shared.claim("other-machine", "cpu", OWNER)
    assert len(errors) == 1


@pytest.fixture
def native_cache_owner_binding(monkeypatch):
    """Only this queue-loop fixture replaces its native operational boundary.

    Synthetic OWNER and science already belong to this unit test. Production
    owner/lease validation is unchanged; refusal is covered separately below.
    """
    from darsia_cache import adapters

    calls = []

    def bind(owner, *, slot=None, generation=None, role=None):
        assert owner == OWNER
        assert slot is None and generation is None and role == "slot"
        calls.append(dict(owner=dict(owner), slot=slot, generation=generation, role=role))
        return dict(TEST_ONLY_NATIVE_OWNER_BOUNDARY=True)

    monkeypatch.setattr(adapters, "bind_actual_worker_owner", bind)
    return calls


def test_failure_traceback_is_preserved_and_worker_takes_next(
    tmp_path, monkeypatch, native_cache_owner_binding
):
    shared = JobQueue(tmp_path)
    shared.enqueue(task("first")); shared.enqueue(task("second"))
    calls = []

    def science(value, output, backend, precision, on_progress):
        calls.append(value["run"])
        print("Actual fixture I/O started.")
        if value["run"] == "first":
            raise RuntimeError("Verbatim scientific fixture failure")
        return dict(complete=True, run=value["run"], backend=backend,
                    precision=precision, worker_references_closed=True, records=[])

    monkeypatch.setitem(sys.modules, "darsia.fluidflower.n1_pipeline", types.SimpleNamespace(run_n1=science))
    monkeypatch.setattr(worker, "_identity", lambda machine: OWNER)
    events = queue.Queue()
    worker._work(str(tmp_path), dict(commit="1"*40,calibration_sha256="2"*64,precision="float32"),
                 "1"*40,"machine","cpu",events,threading.Event(),False,True)
    assert calls == ["first", "second"]
    failed = read_json(tmp_path / "failed/first.json")
    assert "RuntimeError: Verbatim scientific fixture failure" in failed["traceback"]
    assert (tmp_path / "done/second.json").exists()
    all_events = []
    while not events.empty():all_events.append(events.get())
    assert any(e.get("text") == "Actual fixture I/O started." for e in all_events)
    assert any(e.get("traceback") == failed["traceback"] for e in all_events)


def test_original_worker_failure_remains_primary_when_failure_diagnostics_fail(
    tmp_path, monkeypatch, native_cache_owner_binding
):
    shared = JobQueue(tmp_path)
    shared.enqueue(task())
    original = RuntimeError("Original worker failure")
    def science(*args, **kwargs): raise original
    def failure(*args, **kwargs): raise OSError("Failure diagnostic storage unavailable")
    monkeypatch.setitem(sys.modules, "darsia.fluidflower.n1_pipeline", types.SimpleNamespace(run_n1=science))
    monkeypatch.setattr(worker, "_identity", lambda machine: OWNER)
    monkeypatch.setattr(JobQueue, "fail", failure)
    events = queue.Queue()
    with pytest.raises(RuntimeError, match="Original worker failure") as caught:
        worker._work(str(tmp_path), dict(commit="1"*40,calibration_sha256="2"*64,precision="float32"),
            "1"*40,"machine","cpu",events,threading.Event(),True,True)
    assert caught.value is original
    assert "Failure diagnostic storage unavailable" in " ".join(original.__notes__)
    recorded = []
    while not events.empty(): recorded.append(events.get())
    assert any("Original worker failure" in event.get("traceback", "") for event in recorded)


@pytest.mark.parametrize("dirty,head", [(" M tracked.py\n", "1"*40), ("", "4"*40)])
def test_startup_git_rejection_is_console_and_log_and_never_claims(tmp_path, monkeypatch, capsys, dirty, head):
    # This isolated Git-rejection fixture already supplies synthetic UNC/Git
    # facts. Admit only its transport so the test reaches the original Git guard.
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path:
        job_filesystem.SharedFolderAdmission(True, "smb", "Git fixture transport", "isolated unit fixture"))
    write_immutable(tmp_path / "job.json", dict(commit="1"*40,precision="float32",calibration_sha256="2"*64))
    monkeypatch.setattr(worker, "require_unc", lambda path: None)
    monkeypatch.setattr(worker.subprocess, "check_output", lambda args, **kwargs: dirty if "status" in args else head)
    assert worker.main(["--job", str(tmp_path), "--backend", "cpu", "--workers", "1", "--machine", "fixture", "--once"]) == 1
    log = (tmp_path / "logs/fixture-cpu.log").read_text()
    console = capsys.readouterr().err
    assert console == log
    assert "Startup requires a clean Git worktree" in log
    assert not (tmp_path / "claimed").exists()


def test_once_handles_one_task_and_leaves_next_queued(
    tmp_path, monkeypatch, native_cache_owner_binding
):
    shared = JobQueue(tmp_path)
    shared.enqueue(task("first")); shared.enqueue(task("second"))
    def science(value, output, backend, precision, on_progress):
        return dict(complete=True, run=value["run"], backend=backend,
                    precision=precision, worker_references_closed=True, records=[])
    monkeypatch.setitem(sys.modules, "darsia.fluidflower.n1_pipeline", types.SimpleNamespace(run_n1=science))
    monkeypatch.setattr(worker, "_identity", lambda machine: OWNER)
    worker._work(str(tmp_path), dict(calibration_sha256="2"*64,precision="float32"),
                 "1"*40,"machine","cpu",queue.Queue(),threading.Event(),True,False)
    assert (tmp_path / "done/first.json").exists()
    assert (tmp_path / "queue/second.json").exists()


def test_native_cache_owner_binding_refusal_precedes_any_claim(tmp_path, monkeypatch):
    """A cache-owner rejection cannot consume or publish the queued task."""
    from darsia_cache import adapters
    from darsia_cache.policy import CachePolicyError

    shared = JobQueue(tmp_path)
    shared.enqueue(task("first"))
    queued = tmp_path / "queue/first.json"
    original = queued.read_bytes()
    calls = []

    def refuse(owner, *, slot=None, generation=None, role=None):
        assert owner == OWNER
        assert slot is None and generation is None and role == "slot"
        calls.append("binding")
        raise CachePolicyError("Synthetic native owner binding refused before claim")

    def forbidden_claim(*args, **kwargs):
        calls.append("claim")
        raise AssertionError("Owner-binding refusal must precede every claim")

    monkeypatch.setattr(adapters, "bind_actual_worker_owner", refuse)
    monkeypatch.setattr(JobQueue, "claim", forbidden_claim)
    monkeypatch.setattr(worker, "_identity", lambda machine: OWNER)
    with pytest.raises(CachePolicyError, match="Synthetic native owner binding refused"):
        worker._work(str(tmp_path),
            dict(commit="1"*40, calibration_sha256="2"*64, precision="float32"),
            "1"*40, "machine", "cpu", queue.Queue(), threading.Event(), True, True)
    assert calls == ["binding"]
    assert queued.read_bytes() == original
    assert not (tmp_path / "done/first.json").exists()
    assert not (tmp_path / "failed/first.json").exists()
