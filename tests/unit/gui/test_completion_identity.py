"""Exact completion/task/reclaim binding and direct-slot guards; no processes."""

from copy import deepcopy
from pathlib import Path

import pytest

from .test_jobs_metadata import model, put, publication, job, NOW, bytes_of
from .test_jobs_panel import ui, app, panel, wait, snapshot
from .test_jobs_remote_integration import facts, record, remote


def complete(job, *, token=None, records=None):
    root, cases, reserved = job
    task = dict(reserved, attempt_id=token or reserved["attempt_id"])
    if records is None:
        records = [publication(root, case, task["attempt_id"]) for case in cases]
    return dict(
        complete=True,
        worker_references_closed=True,
        run=task["run"],
        precision=task["precision"],
        backend="cpu",
        commit=task["commit"],
        calibration_sha256=task["calibration_sha256"],
        configuration=task["configuration"],
        task=task,
        attempt_id=task["attempt_id"],
        frames=len(cases),
        records=records,
        actual_worker=dict(machine="Main"),
    )


def publish(job, completed, *, done_token=None):
    root, _, _ = job
    bound = put(root / "results" / "ac52" / "completed.json", completed)
    put(
        root / "done" / "ac52.json",
        dict(
            run="ac52",
            attempt_id=done_token or completed["attempt_id"],
            completed=bound,
        ),
    )


def reclaimed(job, previous, successor, *, pre_science=False, mutate=None):
    root, _, reserved = job
    archived = dict(reserved, attempt_id=previous)
    if mutate:
        mutate(archived)
    archive = root / "reclaimed" / "ac52" / previous
    record = dict(
        run="ac52",
        previous_attempt=previous,
        next_attempt=successor,
        reclaimed_utc="2026-01-01T00:01:00+00:00",
        last_actual_progress_unix_ns=1,
    )
    if pre_science:
        record["interrupted_claim_before_science"] = True
        put(archive / "task.json", archived)
    else:
        put(
            archive / "lease" / "owner.json",
            dict(
                token=previous,
                task=archived,
                output_path=str(root / "results" / "ac52" / ".attempts" / previous),
            ),
        )
    put(archive / "reclaimed.json", record)


@pytest.mark.parametrize(
    "key", ["commit", "calibration_sha256", "configuration", "task", "attempt_id"]
)
def test_missing_original_identity_is_unavailable_never_complete_or_counted(job, key):
    result = complete(job)
    result.pop(key)
    # done still has its original generation even if legacy completion omits it.
    root, _, task = job
    bound = put(root / "results" / "ac52" / "completed.json", result)
    put(
        root / "done" / "ac52.json",
        dict(run="ac52", attempt_id=task["attempt_id"], completed=bound),
    )
    before = bytes_of(root)
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert (
        not value["complete"]
        and value["committed"] == 0
        and value["state"] == "Metadata error"
    )
    assert "Completion unavailable" in value["errors"][0] and bytes_of(root) == before


@pytest.mark.parametrize(
    "kind",
    [
        "commit",
        "calibration",
        "configuration",
        "task_threads",
        "task_output",
        "done_attempt",
        "task_attempt",
        "backend",
    ],
)
def test_hash_bound_completion_cannot_relabel_other_science_or_generation(job, kind):
    root, _, _ = job
    result = complete(job)
    done_token = None
    if kind == "commit":
        result["commit"] = "c" * 40
    elif kind == "calibration":
        result["calibration_sha256"] = "c" * 64
    elif kind == "configuration":
        result["configuration"] = dict(result["configuration"], sha256="c" * 64)
    elif kind == "task_threads":
        result["task"]["threads"] = 8
    elif kind == "task_output":
        result["task"]["output_directory"] = str(root / "foreign")
    elif kind == "done_attempt":
        done_token = "f" * 32
    elif kind == "task_attempt":
        result["task"]["attempt_id"] = "f" * 32
    else:
        result["backend"] = "cuda"
    publish(job, result, done_token=done_token)
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert not value["complete"] and value["committed"] == 0 and value["errors"]


@pytest.mark.parametrize("pre_science", [False, True])
def test_legitimate_original_reclaim_generation_is_bound_without_scanning(
    job, pre_science
):
    root, cases, reserved = job
    initial = reserved["attempt_id"]
    final = "2" * 32
    reclaimed(job, initial, final, pre_science=pre_science)
    records = [
        publication(root, cases[0], final if pre_science else initial),
        publication(root, cases[1], final),
    ]
    publish(job, complete(job, token=final, records=records))
    before = bytes_of(root)
    reader = model.JobSnapshotReader()
    value = reader({"root": str(root)}, now=NOW)
    assert value["complete"] and value["committed"] == 2 and not value["errors"]
    assert bytes_of(root) == before
    # The original pre-science branch never wrote resume/retirement metadata.
    assert not any(
        path.name in {"resume.json", "retirement.json"} for path in root.rglob("*")
    )
    assert not any("unrelated" in key[0] for key in reader.cache)


def test_multihop_original_chain_accepts_bound_carried_frames_and_pre_science_gap(job):
    root, cases, reserved = job
    initial = reserved["attempt_id"]
    middle = "2" * 32
    final = "3" * 32
    reclaimed(job, initial, middle)
    reclaimed(job, middle, final, pre_science=True)
    publish(
        job,
        complete(
            job,
            token=final,
            records=[
                publication(root, cases[0], initial),
                publication(root, cases[1], final),
            ],
        ),
    )
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert value["complete"] and value["committed"] == 2


@pytest.mark.parametrize(
    "kind", ["missing", "cycle", "archived_science", "owner_output", "unbound_frame"]
)
def test_unproven_reclaim_or_foreign_frame_never_bypasses_identity(job, kind):
    root, cases, reserved = job
    initial = reserved["attempt_id"]
    final = "2" * 32
    if kind != "missing":
        reclaimed(
            job,
            initial,
            initial if kind == "cycle" else final,
            mutate=(
                (lambda task: task.update(calibration_sha256="f" * 64))
                if kind == "archived_science"
                else None
            ),
        )
    if kind == "owner_output":
        owner_path = root / "reclaimed" / "ac52" / initial / "lease" / "owner.json"
        owner = model.read_json(owner_path)
        owner["output_path"] = str(root / "foreign")
        put(owner_path, owner)
    records = [
        publication(root, case, "f" * 32 if kind == "unbound_frame" else final)
        for case in cases
    ]
    publish(job, complete(job, token=final, records=records))
    value = model.JobSnapshotReader()({"root": str(root)}, now=NOW)
    assert not value["complete"] and value["committed"] == 0 and value["errors"]


def test_direct_local_start_slot_rejects_snapshot_errors_before_reservation(
    panel, app, tmp_path
):
    widget, calls = panel
    root = tmp_path / "local-job"
    put(root / "job.json", dict(expected_runs=["ac52"]))
    broken = dict(
        snapshot(str(root)),
        state="Metadata error",
        errors=["literal completion identity failure"],
    )
    widget.reader = lambda row: broken
    widget.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, widget.can_close)
    widget.start_local_workers()
    widget.start_selected_machine()
    assert calls == [] and not widget.start_button.isEnabled()
    assert "literal completion identity failure" in widget.error_text.toPlainText()
    assert not (root / "ui" / "launches").exists()


def test_direct_remote_start_slot_rejects_snapshot_errors_before_any_validation_or_request(
    panel, app, tmp_path, monkeypatch
):
    widget, calls = panel
    root = tmp_path / "job"
    put(root / "job.json", dict(expected_runs=["ac52"]))
    target = record(root, facts(tmp_path))
    monkeypatch.setattr(remote, "require_unc", lambda path: None)
    widget.remote_launcher = lambda *a, **k: pytest.fail(
        "No transport may be requested"
    )
    widget.reader = lambda row: dict(
        snapshot(str(root)),
        registered_machines=[target],
        errors=["literal remote metadata failure"],
    )
    widget.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, widget.can_close)
    widget.target_combo.setCurrentIndex(1)
    widget.start_selected_machine()
    assert (
        not widget.start_button.isEnabled() and not (root / "ui" / "launches").exists()
    )
    assert "literal remote metadata failure" in widget.error_text.toPlainText()
