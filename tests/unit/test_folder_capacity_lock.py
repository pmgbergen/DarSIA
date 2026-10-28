"""Private local lock ownership, stale handles and explicit stop releases."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import capacity
from darsia.fluidflower.pipeline.folder_capacity import CapacityLockBusy, FolderSpaceEstimate
from .test_folder_cache_worker import _stores


def _ledger(tmp_path):
    stores = _stores(tmp_path)
    root = tmp_path / "Jobs" / "sample-job"
    root.mkdir()
    return capacity.CapacityLedger(stores, "synthetic", root, job_root=root)


def test_kernel_lock_releases_after_owner_exit_without_removing_lockfile(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.root.mkdir(parents=True)
    path = ledger.root / "admission.lock"
    path.write_bytes(b"\0")
    identity = path.stat().st_ino
    script = """
import os, sys
stream = open(sys.argv[1], 'r+b')
if os.name == 'nt':
 import msvcrt
 msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
else:
 import fcntl
 fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
print('locked', flush=True)
sys.stdin.readline()
"""
    child = subprocess.Popen([sys.executable, "-c", script, str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(CapacityLockBusy):
            with ledger.locked(wait_seconds=0):
                pytest.fail("An active kernel owner must never be overridden")
        child.stdin.write("exit\n")
        child.stdin.flush()
        child.wait(timeout=5)
        with ledger.locked(wait_seconds=0):
            with path.open("rb") as stream:
                stream.seek(1)  # The first byte is exclusively locked by the owner.
                owner = json.loads(stream.read())
            assert owner["pid"] == os.getpid() and owner["process_create_time"]
            assert len(owner["nonce"]) == 32
        assert path.stat().st_ino == identity
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


def test_lock_nonce_change_is_not_accepted_as_our_lock(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError, match="owner or nonce"):
        with ledger.locked():
            path = ledger.root / "admission.lock"
            with path.open("r+b") as stream:
                stream.seek(1)
                stream.write(b'{"nonce":"different owner"}')
                stream.truncate()
    # Unlocking the original handle is safe; no pathname is deleted/replaced.
    with ledger.locked(wait_seconds=0):
        pass


def test_local_stop_release_requires_same_owner_epoch_and_keeps_distinct_provenance(tmp_path):
    ledger = _ledger(tmp_path)
    task = dict(run="sample", attempt_id="a" * 32, ordered_case_ids=["one"])
    owner = dict(machine="private-host-cpu", pid=123, process_create_time="123.5")
    with ledger.locked():
        path = ledger.bind(ledger.job_root, SimpleNamespace(task=task, token=task["attempt_id"]),
            owner, FolderSpaceEstimate("sample", 1, 17, 5))
        proof = dict(format="darsia-job-local-process-stop-claim-v1", run="sample",
            attempt_id=task["attempt_id"], all_absent=True, actual_worker=owner,
            identities=[dict(owner, process_create_time="123.6")],
            confirmation={"source":"explicit"}, claim={"source":"bound"}, owner={"source":"bound"})
        with pytest.raises(ValueError, match="reservation owner"):
            ledger.release_local_stop(path, owner, proof)
        proof["identities"] = [owner]
        ledger.release_local_stop(path, owner, proof)
        value = json.loads((path / "released.json").read_bytes())
        assert value["local_stop"] == proof and value["retirement"] is None
    assert ledger.remaining() == 0


def test_local_stop_cleanup_locks_each_pool_once_and_rechecks_proof(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from darsia.fluidflower import job_local_stop, store_map
    from darsia.fluidflower.pipeline import series_setup_worker_admission as admission
    ledger = _ledger(tmp_path)
    owner = dict(machine="private-host-cpu", pid=123, process_create_time="123.5")
    claims, retained, evidence = {}, {}, {}
    with ledger.locked():
        for number in (1, 2):
            run, token = f"sample-{number}", f"{number:032x}"
            task = dict(run=run, attempt_id=token, ordered_case_ids=["one"])
            claimed = ledger.job_root / f"{run}.json"
            claimed.write_text(json.dumps(task))
            path = ledger.bind(ledger.job_root, SimpleNamespace(task=task, token=token),
                owner, FolderSpaceEstimate(run, 1, 17, 5))
            claims[run] = dict(claim=dict(path=str(claimed)))
            retained[run] = [(ledger, path)]
            evidence[run] = dict(format="darsia-job-local-process-stop-claim-v1", run=run,
                attempt_id=token, all_absent=True, actual_worker=owner, identities=[owner],
                confirmation={"source":"explicit"}, claim={"source":"bound"}, owner={"source":"bound"})
    proof = dict(source={"path":"verified-confirmation", "sha256":"a" * 64}, claims=claims)
    observations, lock_calls = [], []
    monkeypatch.setattr(job_local_stop, "verified_local_stop",
        lambda *a, **k: observations.append(True) or proof)
    monkeypatch.setattr(job_local_stop, "verified_local_stop_claim",
        lambda root, claimed, **k: evidence[claimed.stem])
    monkeypatch.setattr(store_map, "read_store_map", lambda *a: ledger.stores)
    monkeypatch.setattr(admission, "_retained_setup_capacity", lambda root, task, stores: retained[task["run"]])
    original_lock = ledger.locked
    @contextmanager
    def locked(**kwargs):
        lock_calls.append(True)
        with original_lock(**kwargs):
            yield ledger
    monkeypatch.setattr(ledger, "locked", locked)
    released = admission.release_local_stopped_setup_capacity(ledger.job_root, machine="private-host")
    assert len(released) == 2 and len(lock_calls) == 1 and len(observations) == 2
    assert ledger.remaining() == 0
