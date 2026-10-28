"""Same-job recovery over private metadata and one genuinely spawned worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from darsia.fluidflower import job_control as control, job_lifecycle as lifecycle
from darsia.fluidflower.job_launch_state import (begin_launch_request,
    declare_requests_stopped, read_launch_requests)
from darsia.fluidflower.pipeline import job_queue, worker
from .test_job_management import control_master, retired_claim, job_tree  # noqa: F401
from .test_series_setup_inputs import controls  # noqa: F401
from .test_folder_cache_worker import _private_build, _stores, _activate


def facts(root, **values):
    return dict(root=str(root), complete=False, errors=[], failed_runs=0,
                runs=[dict(run="run-a", state="Queued")], **values)


def test_failed_start_reset_is_pending_and_preserves_history_and_source(job_tree):
    _index, _series, root, raw = job_tree
    control_master(root)
    job_queue.write_immutable(root / "queue" / "run-a.json", dict(run="run-a", attempt_id="a" * 32))
    begin_launch_request(root, machine="fixture", backend="cpu", workers=1, kind="local")
    log = root / "logs" / "fixture-cpu.log"
    log.parent.mkdir()
    log.write_text(control.utc() + ' {"kind":"startup_failure"}\n', encoding="utf-8")
    assert lifecycle.user_status(facts(root))["state"] == "Failed"
    original = {p: p.read_bytes() for p in [root / "job.json", raw / "source.ARW", log]}
    lifecycle.recover_job(root, machine="fixture", expected_previous=None, reset=True)
    assert lifecycle.user_status(facts(root))["state"] == "Pending"
    assert all(p.read_bytes() == value for p, value in original.items())
    assert len(read_launch_requests(root)) == 1
    # A later actual startup failure must be visible again.
    begin_launch_request(root, machine="fixture", backend="cpu", workers=1, kind="local")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(control.utc() + ' {"kind":"startup_failure"}\n')
    assert lifecycle.user_status(facts(root))["state"] == "Failed"


def test_cancel_request_and_declaration_never_claim_retirement(job_tree):
    _index, _series, root, _raw = job_tree
    control_master(root)
    begin_launch_request(root, machine="fixture", backend="cpu", workers=1, kind="local")
    stop = control.publish_control(root, "stop", machine="fixture", expected_previous=None)
    declare_requests_stopped(root, read_launch_requests(root))
    assert lifecycle.user_status(facts(root))["state"] == "Cancelling"
    with pytest.raises(ValueError, match="original worker service"):
        lifecycle.recover_job(root, machine="fixture", expected_previous=stop)
    assert control.read_control(root)["action"] == "stop"


def test_restart_fences_old_writer_keeps_completed_products_and_original_bytes(job_tree, monkeypatch):
    _index, _series, root, raw = job_tree
    control_master(root)
    claimed = retired_claim(root, monkeypatch)
    task = job_queue.read_json(claimed)
    # An earlier completed run's immutable products are not part of recovery.
    complete = root / "done" / "retained.json"
    job_queue.write_immutable(complete, dict(run="retained", complete=True))
    source = (raw / "source.ARW").read_bytes()
    original_job = (root / "job.json").read_bytes()
    stop = control.publish_control(root, "stop", machine="fixture", expected_previous=None)
    assert lifecycle.user_status(facts(root))["state"] == "Cancelled"
    with pytest.raises(ValueError, match="unstarted job"):
        lifecycle.recover_job(root, machine="fixture", expected_previous=stop, reset=True)
    monkeypatch.setattr(worker, "_same_process", lambda _identity: object())
    assert lifecycle.user_status(facts(root))["state"] == "Cancelling"
    with pytest.raises(ValueError, match="still running"):
        lifecycle.recover_job(root, machine="fixture", expected_previous=stop)
    monkeypatch.setattr(worker, "_same_process", lambda _identity: None)
    lifecycle.recover_job(root, machine="fixture", expected_previous=stop)
    retry = job_queue.read_json(root / "queue" / "run-a.json")
    assert retry["attempt_id"] != task["attempt_id"]
    assert dict(retry, attempt_id=task["attempt_id"]) == task
    assert job_queue.read_decision(root, "run-a", task["attempt_id"])["outcome"] == "reclaimed"
    assert complete.read_bytes() and (root / "job.json").read_bytes() == original_job
    assert (raw / "source.ARW").read_bytes() == source
    assert not claimed.exists()
    assert lifecycle.user_status(facts(root))["state"] == "Pending"


def test_claim_file_and_launcher_do_not_establish_running(job_tree):
    _index, _series, root, _raw = job_tree
    control_master(root)
    assert lifecycle.user_status(facts(root, logs=[dict(tail=control.utc()
        + ' {"kind":"started"}\n')]), local_machine="fixture")["state"] == "Starting"
    job_queue.write_immutable(root / "claimed" / "fixture-cpu" / "run-a.json",
        dict(run="run-a", attempt_id="a" * 32))
    assert lifecycle.user_status(facts(root), local_machine="fixture")["state"] == "Starting"


def test_published_failure_is_preserved_and_explains_recovery_limit(job_tree):
    _index, _series, root, _raw = job_tree
    control_master(root)
    failure = root / "failed" / "run-a.json"
    job_queue.write_immutable(failure, dict(run="run-a", complete=False,
        task=dict(run="run-a", attempt_id="a" * 32), traceback="fixture failure"))
    original = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    for reset in (False, True):
        with pytest.raises(ValueError, match="published failure"):
            lifecycle.recover_job(root, machine="fixture", expected_previous=None, reset=reset)
    assert all(path.read_bytes() == value for path, value in original.items())


def test_remote_running_is_a_fresh_exact_attempt_report_and_never_local_liveness(job_tree):
    _index, _series, root, _raw = job_tree
    control_master(root)
    task = dict(run="run-a", attempt_id="a" * 32)
    claimed = root / "claimed" / "remote-cpu" / "run-a.json"
    job_queue.write_immutable(claimed, task)
    lease = root / "leases" / ("run-a." + task["attempt_id"])
    job_queue.write_immutable(lease / "owner.json", dict(task=task, token=task["attempt_id"],
        actual_owner=dict(machine="remote-cpu")))
    stamp = control.utc()
    now = lifecycle._epoch(stamp)
    assert lifecycle.user_status(facts(root), local_machine="fixture", now=now)["state"] == "Starting"
    progress = dict(run="run-a", attempt_id=task["attempt_id"], sequence=0,
        written_utc=stamp, event=dict(state="setup_progress"))
    job_queue.write_immutable(lease / "progress_00000000.json", progress)
    status = lifecycle.user_status(facts(root), local_machine="fixture", now=now)
    assert status["state"] == status["runs"][0]["state"] == "Running (reported)"
    assert lifecycle.user_status(facts(root), local_machine="fixture", now=now + 121)["state"] == "Starting"
    assert lifecycle.user_status(facts(root), local_machine="remote", now=now)["state"] == "Starting"
    assert lifecycle.user_status(facts(root), local_machine="REMOTE", now=now)["state"] == "Starting"
    job_queue.write_immutable(lease / "progress_00000001.json",
        dict(progress, sequence=1, attempt_id="b" * 32))
    assert lifecycle.user_status(facts(root), local_machine="fixture", now=now)["state"] == "Starting"


def test_actual_worker_cancel_retire_restart_same_job(controls, monkeypatch):
    root, job, task = _private_build(controls)
    stores = _stores(controls.root, roots=dict(series=str(controls.root / "project"),
        private=str(controls.root), results=str(root.parent)))
    _activate(stores, root, monkeypatch)
    control.ensure_setup_master(root, machine=stores.machine_id, actor="fixture")
    original = (root / "job.json").read_bytes()
    original_task = (root / "results" / "sample" / "task.json").read_bytes()
    script = '''
from pathlib import Path
import sys
from types import SimpleNamespace
from darsia.fluidflower.pipeline import worker, job_queue
from darsia.fluidflower.store_map import read_store_map
from tests.unit._folder_worker_fixture import work
root = Path(sys.argv[1]); stores = read_store_map(sys.argv[2])
job = job_queue.read_json(root / 'job.json')
worker._work = work
args = SimpleNamespace(job=root, machine=stores.machine_id, backend='cpu', workers=1,
    once=True, exit_when_idle=True, watchdog=False, watchdog_seconds=30,
    _alias_stores=stores, stores=sys.argv[2], _cache_terminal_events=[])
with (root / 'logs' / (stores.machine_id + '-cpu.log')).open('a', encoding='utf-8') as log:
    raise SystemExit(worker._parent(args, stores.machine_id + '-cpu', job, job['commit'], log))
'''
    environment = dict(os.environ, DARSIA_TEST_FOLDER_WAIT_FOR_CANCEL="1")
    begin_launch_request(root, machine=stores.machine_id, backend="cpu", workers=1, kind="local")
    process = subprocess.Popen([sys.executable, "-B", "-c", script, str(root), stores.source["path"]],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stop = None
    try:
        deadline = time.monotonic() + 60
        checkpoint = root / "sample" / ".attempts" / task["attempt_id"] / "fixture_checkpoint.json"
        while not checkpoint.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert checkpoint.exists(), process.communicate(timeout=5)
        assert lifecycle.user_status(facts(root), local_machine=stores.machine_id.title(),
            stores_path=stores.source["path"])["state"] == "Running"
        stop = control.publish_control(root, "stop", machine=stores.machine_id, expected_previous=None)
        with pytest.raises(ValueError):
            lifecycle.recover_job(root, machine=stores.machine_id, expected_previous=stop,
                stores_path=stores.source["path"])
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        saved_checkpoint = checkpoint.read_bytes()
        assert lifecycle.user_status(facts(root), stores_path=stores.source["path"])["state"] == "Cancelled"
        lifecycle.recover_job(root, machine=stores.machine_id, expected_previous=stop,
            stores_path=stores.source["path"])
        retry = job_queue.read_json(root / "queue" / "sample.json")
        assert retry["attempt_id"] != task["attempt_id"]
        assert checkpoint.read_bytes() == saved_checkpoint
        assert (root / "job.json").read_bytes() == original
        assert (root / "results" / "sample" / "task.json").read_bytes() == original_task
        environment.pop("DARSIA_TEST_FOLDER_WAIT_FOR_CANCEL")
        begin_launch_request(root, machine=stores.machine_id, backend="cpu", workers=1, kind="local")
        completed = subprocess.run([sys.executable, "-B", "-c", script, str(root), stores.source["path"]],
            env=environment, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (root / "done" / "sample.json").is_file()
        assert not list((root / "failed").glob("*.json"))
        assert checkpoint.read_bytes() == saved_checkpoint
    finally:
        if process.poll() is None:
            if stop is None:
                control.publish_control(root, "stop", machine=stores.machine_id, expected_previous=None)
            process.communicate(timeout=30)
