"""Shared metadata must preserve queue ownership and actual progress evidence."""

from datetime import datetime, timedelta, timezone
import json

import pytest

from darsia.fluidflower.job_control import publish_control, read_control
from darsia.fluidflower.shared_job_folder import SharedJobFolder


def job(folder, name="job-a"):
    root = folder / name
    root.mkdir()
    (root / "job.json").write_text(json.dumps(dict(
        pipeline="test-existing-pipeline", expected_runs=["a", "b"],
        precision="float32", commit="a" * 40)), encoding="utf-8")
    return root


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_publication_is_explicit_idempotent_and_keeps_original_master(tmp_path):
    queue = SharedJobFolder(tmp_path)
    first = job(tmp_path)
    assert queue.jobs() == []
    assert queue.workers() == []
    assert not (tmp_path / ".darsia-jobs").exists()
    row = queue.publish(first, master_machine="First", cpu_workers=3, cuda_workers=1)
    assert row["job_root"] == str(first.resolve())
    assert row["cpu_workers"] == 3 and row["cuda_workers"] == 1
    assert row["state"] == "run"
    original_master = (first / "ui/master.json").read_bytes()
    assert queue.publish(first, master_machine="FIRST")["job_id"] == row["job_id"]
    assert (first / "ui/master.json").read_bytes() == original_master
    with pytest.raises(ValueError, match="recorded master"):
        queue.publish(first, master_machine="Second")
    other = job(tmp_path, "job-b")
    queue.publish(other, master_machine="Second")
    assert {row["master_machine"] for row in queue.jobs()} == {"First", "Second"}


def test_job_mutation_and_master_mutation_are_rejected(tmp_path):
    queue = SharedJobFolder(tmp_path)
    root = job(tmp_path)
    queue.publish(root, master_machine="master")
    raw = (root / "job.json").read_bytes()
    (root / "job.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="metadata changed"):
        queue.jobs()
    (root / "job.json").write_bytes(raw)
    master_path = root / "ui/master.json"
    master = json.loads(master_path.read_text())
    master["machine"] = "imposter"
    put(master_path, master)
    with pytest.raises(ValueError, match="recorded master"):
        queue.jobs()


def test_admission_only_changes_discovery_and_respects_queue_controls(tmp_path):
    queue = SharedJobFolder(tmp_path)
    root = job(tmp_path)
    row = queue.publish(root, master_machine="master")
    with pytest.raises(ValueError, match="recorded master"):
        queue.update_job(row["job_id"], master_machine="worker", state="stopped")
    queue.update_job(row["job_id"], master_machine="master", state="paused")
    assert queue.jobs()[0]["state"] == "paused"
    assert read_control(root) == dict(action="run", sequence=0)
    queue.update_job(row["job_id"], master_machine="master", state="run")
    publish_control(root, "stop", machine="master", expected_previous=None)
    current = queue.jobs()[0]
    assert current["state"] == "stopped"
    assert current["folder_state"] == "run"
    assert current["original_control_state"] == "stop"


def test_heartbeat_updates_replace_one_file_without_creating_progress(tmp_path):
    queue = SharedJobFolder(tmp_path)
    first = queue.report("Second", "session-1", role="worker", cpu_workers=4,
                         cuda_workers=1, jobs=[])
    observed = datetime.fromisoformat(first["reported_utc"])
    live = queue.workers(now=observed + timedelta(seconds=2))[0]
    assert live["stale"] is False
    assert live["jobs"] == []
    assert "numerical_progress" not in live
    assert live["cpu_workers"] == 4 and live["cuda_workers"] == 1
    assert queue.workers(now=observed + timedelta(seconds=31))[0]["stale"] is True
    assert queue.workers(now=observed - timedelta(seconds=31))[0]["stale"] is True
    evidence = [dict(job_id="a" * 64, completed_frames=3, source="original queue")]
    queue.report("Second", "session-1", role="worker", cpu_workers=4,
                 cuda_workers=0, jobs=evidence, state="offline")
    assert len(list((queue.metadata / "workers").glob("*.json"))) == 1
    row = queue.workers()[0]
    assert row["stale"] is True and row["jobs"] == evidence
    assert not list(queue.metadata.rglob("*.partial"))


def test_invalid_counts_nonfinite_and_oversized_payload_leave_previous_status(tmp_path):
    queue = SharedJobFolder(tmp_path)
    values = dict(role="worker", cpu_workers=1, cuda_workers=0, jobs=[])
    queue.report("host", "session", **values)
    for invalid in (-1, True, 257):
        with pytest.raises(ValueError, match="Worker counts"):
            queue.report("host", "session", **dict(values, cuda_workers=invalid))
    with pytest.raises(ValueError):
        queue.report("host", "session", **dict(values, jobs=[dict(progress=float("nan"))]))
    with pytest.raises(ValueError, match="size limit"):
        queue.report("host", "session", **dict(values, jobs=[dict(error="x" * 300_000)]))
    assert queue.workers()[0]["jobs"] == []


def test_external_local_paths_and_injected_relative_escape_are_rejected(tmp_path):
    folder = tmp_path / "shared"
    folder.mkdir()
    queue = SharedJobFolder(folder)
    external = job(tmp_path, "external")
    with pytest.raises(ValueError, match="UNC"):
        queue.publish(external, master_machine="master")
    root = job(folder)
    row = queue.publish(root, master_machine="master")
    path = queue.metadata / "jobs" / (row["job_id"] + ".json")
    value = json.loads(path.read_text())
    value["location"] = dict(relative_path="../../external")
    value["job_id"] = queue._identifier(value["location"])
    replacement = path.with_name(value["job_id"] + ".json")
    path.unlink()
    put(replacement, value)
    with pytest.raises(ValueError, match="escapes"):
        queue.jobs()


def test_missing_timezone_and_mismatched_presence_identity_are_rejected(tmp_path):
    queue = SharedJobFolder(tmp_path)
    queue.report("host", "session", role="master", cpu_workers=0, cuda_workers=0, jobs=[])
    path = next((queue.metadata / "workers").glob("*.json"))
    value = json.loads(path.read_text())
    value["reported_utc"] = "2026-10-25T08:00:00"
    put(path, value)
    with pytest.raises(ValueError, match="timezone"):
        queue.workers()
    value["reported_utc"] = datetime.now(timezone.utc).isoformat()
    value["machine"] = "different"
    put(path, value)
    with pytest.raises(ValueError, match="filename"):
        queue.workers()
