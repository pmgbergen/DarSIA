"""UI reservation/terminal-log/operator transitions; never launches a process."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from .test_jobs_metadata import launches, model, put, job as metadata_job
from .test_jobs_panel import ui, app, panel, wait, snapshot
from .test_jobs_remote_integration import facts, record, remote


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def append_log(root, request, kind, *, seconds=1, machine=None):
    path = (
        Path(request["service_log"])
        if machine is None
        else Path(root) / "logs" / (machine + ".log")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(request["requested_utc"]) + timedelta(
        seconds=seconds
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(stamp.isoformat() + " " + json.dumps(dict(kind=kind)) + "\n")


def begin(root, **kwargs):
    return launches.begin_launch_request(
        root, machine="Main", backend="cpu", workers=2, kind="local", now=T0, **kwargs
    )


def test_launch_reservation_survives_reopen_without_claiming_worker_or_progress(
    metadata_job,
):
    root, _, _ = metadata_job
    request = begin(root)
    value = launches.read_launch_requests(root)[0]
    assert value["state"] == launches.UNKNOWN and not value["released"]
    assert (
        value["worker_liveness_verified"] is False
        and value["numerical_progress_claimed"] is False
    )
    assert model.JobSnapshotReader()({"root": str(root)})["committed"] == 0
    assert not Path(
        request["launch_directory"]
    ).exists()  # Existing launcher alone creates it.
    assert not (root / "claimed").exists()


def test_ssh_exit_launcher_pid_and_frame_completion_never_release(metadata_job):
    root, _, _ = metadata_job
    request = launches.begin_launch_request(
        root, machine="Second", backend="cpu", workers=1, kind="ssh", now=T0
    )
    put(
        Path(request["launch_directory"]) / "launch.json",
        dict(
            format="darsia-job-ssh-launch-v1",
            job_root=str(root),
            transport_started=True,
            transport_pid=123,
            transport_exit_code=0,
            worker_startup_verified=False,
            numerical_progress_claimed=False,
        ),
    )
    append_log(root, request, "completed")
    result = launches.read_launch_requests(root)[0]
    assert not result["released"] and result["state"] == launches.UNKNOWN
    assert result["launcher_receipt"]["transport_exit_code"] == 0


def test_only_new_matching_terminal_report_releases_not_old_or_foreign_log(
    metadata_job,
):
    root, _, _ = metadata_job
    path = root / "logs" / "main-cpu.log"
    path.parent.mkdir()
    path.write_text(
        (T0 + timedelta(days=1)).isoformat() + ' {"kind":"stopped"}\n', encoding="utf-8"
    )
    request = begin(root)  # The old future-clock log line is before recorded offset.
    append_log(root, request, "stopped", machine="other-cpu")
    assert not launches.read_launch_requests(root)[0]["released"]
    append_log(root, request, "stopped")
    result = launches.read_launch_requests(root)[0]
    assert (
        result["released"] and result["terminal_report"]["event"]["kind"] == "stopped"
    )
    append_log(root, request, "started", seconds=2)
    assert not launches.read_launch_requests(root)[0]["released"]


def test_startup_failure_is_terminal_report_but_truncated_or_old_log_is_unknown(
    metadata_job,
):
    root, _, _ = metadata_job
    request = begin(root)
    append_log(root, request, "startup_failure")
    assert launches.read_launch_requests(root)[0]["released"]
    second = launches.begin_launch_request(
        root,
        machine="Main",
        backend="cpu",
        workers=1,
        kind="local",
        now=T0 + timedelta(seconds=3),
    )
    Path(second["service_log"]).write_bytes(b"")
    states = launches.read_launch_requests(root)
    assert all(not row["released"] for row in states)


def test_operator_declaration_is_bound_statement_only_not_scientific_proof(
    metadata_job,
):
    root, _, _ = metadata_job
    request = begin(root)
    before = Path(request["source"]["path"]).read_bytes()
    written = launches.declare_requests_stopped(root, [request])
    assert len(written) == 1
    declaration = model.read_json(Path(written[0]["path"]))
    assert (
        declaration["request"] == request["source"]
        and declaration["scientific_evidence"] is False
    )
    assert (
        declaration["worker_liveness_verified"] is False
        and declaration["numerical_progress_claimed"] is False
    )
    row = launches.read_launch_requests(root)[0]
    assert row["released"] and "not independently verified" in row["state"]
    assert Path(request["source"]["path"]).read_bytes() == before
    value = model.JobSnapshotReader()({"root": str(root)})
    assert (
        not value["complete"]
        and value["committed"] == 0
        and value["completed_runs"] == 0
    )
    declaration["scientific_evidence"] = True
    put(written[0]["path"], declaration)
    with pytest.raises(ValueError, match="falsely claims scientific"):
        launches.read_launch_requests(root)


def test_direct_repeat_local_start_blocked_then_only_explicit_declaration_allows_again(
    panel, app, metadata_job, monkeypatch
):
    widget, _ = panel
    root, _, _ = metadata_job
    calls = []
    widget.reader = model.JobSnapshotReader()
    widget.local_launcher = lambda root, **kw: (
        calls.append((root, kw))
        or dict(job_root=root, started=True, numerical_progress_claimed=False)
    )
    widget.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, widget.can_close)
    widget.start_local_workers()
    wait(app, widget.can_close)
    assert (
        len(calls) == 1
        and not widget.start_button.isEnabled()
        and launches.UNKNOWN in widget.launch_state_label.text()
    )
    widget.start_local_workers()
    widget.start_selected_machine()
    wait(app, widget.can_close)
    assert len(calls) == 1 and widget.declare_stopped_button.isEnabled()
    monkeypatch.setattr(
        ui.QMessageBox, "question", lambda *a, **k: ui.QMessageBox.StandardButton.No
    )
    widget.declare_previous_service_stopped()
    assert not list((root / "ui" / "launches").glob("*.declaration.*.json"))
    monkeypatch.setattr(
        ui.QMessageBox, "question", lambda *a, **k: ui.QMessageBox.StandardButton.Yes
    )
    widget.declare_previous_service_stopped()
    wait(app, widget.can_close)
    assert (
        widget.start_button.isEnabled()
        and "not independently verified" in widget.launch_state_label.text()
    )
    assert len(calls) == 1 and widget.current["committed"] == 0
    widget.start_local_workers()
    wait(app, widget.can_close)
    assert len(calls) == 2 and not widget.start_button.isEnabled()
    assert len(list((root / "ui" / "launches").glob("*.request.json"))) == 2


def test_failed_launcher_keeps_receipt_console_literal_error_and_reopened_guard(
    panel, app, metadata_job
):
    widget, _ = panel
    root, _, _ = metadata_job
    calls = []

    def failed(root, **kwargs):
        calls.append(kwargs)
        folder = Path(kwargs["launch_directory"])
        folder.mkdir(parents=True)
        (folder / "console.log").write_text(
            "literal Ω startup failure before ready", encoding="utf-8"
        )
        put(
            folder / "launch.json",
            dict(
                format="darsia-job-local-launch-v1",
                job_root=root,
                started=False,
                console=str(folder / "console.log"),
            ),
        )
        raise RuntimeError("literal local spawn failure")

    widget.reader = model.JobSnapshotReader()
    widget.local_launcher = failed
    widget.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, widget.can_close)
    assert widget.start_reason_label.isHidden()
    widget.start_local_workers()
    wait(app, widget.can_close)
    assert len(calls) == 1 and not widget.start_button.isEnabled()
    assert not widget.start_reason_label.isHidden()
    assert widget.start_reason_label.wordWrap()
    assert widget.start_reason_label.text() == widget.start_button.toolTip()
    assert launches.UNKNOWN in widget.start_reason_label.text()
    details = widget.start_reason_label.parentWidget().layout()
    assert (
        details.itemAt(details.indexOf(widget.start_reason_label) + 1).layout()
        is widget._controls
    )
    assert "literal local spawn failure" in widget.error_text.toPlainText()
    assert "literal Ω startup failure before ready" in widget.log_text.toPlainText()
    second = ui.JobsPanel(
        reader=model.JobSnapshotReader(),
        local_machine="MAIN",
        local_launcher=lambda *a, **k: pytest.fail("reopen must not launch"),
    )
    second.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, second.can_close)
    assert (
        not second.start_button.isEnabled()
        and launches.UNKNOWN in second.launch_state_label.text()
    )
    assert not second.start_reason_label.isHidden()
    assert second.start_reason_label.text() == second.start_button.toolTip()
    assert "literal local spawn failure" in second.error_text.toPlainText()
    second.close()
    second.deleteLater()
    app.processEvents()


def test_matching_terminal_refresh_enables_start_without_operator_or_launch(
    panel, app, metadata_job
):
    widget, _ = panel
    root, _, _ = metadata_job
    request = begin(root)
    calls = []
    widget.reader = model.JobSnapshotReader()
    widget.local_launcher = lambda *a, **k: calls.append((a, k))
    widget.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, widget.can_close)
    assert not widget.start_button.isEnabled()
    append_log(root, request, "stopped")
    widget.refresh()
    wait(app, widget.can_close)
    assert (
        widget.start_button.isEnabled()
        and "reported stopped" in widget.launch_state_label.text()
    )
    assert not widget.declare_stopped_button.isEnabled() and calls == []


def test_repeated_remote_start_same_normalized_alias_backend_versions_is_blocked(
    panel, app, tmp_path, monkeypatch
):
    widget, _ = panel
    root = tmp_path / "job"
    put(root / "job.json", dict(expected_runs=["ac52"]))
    one = record(root, facts(tmp_path))
    two = record(root, dict(facts(tmp_path), machine="SECOND"), suffix=".version")
    monkeypatch.setattr(remote, "require_unc", lambda path: None)
    calls = []
    widget.remote_launcher = lambda root, **kw: (
        calls.append((root, kw))
        or dict(job_root=root, transport_started=True, numerical_progress_claimed=False)
    )
    widget.reader = lambda row: dict(
        snapshot(row["root"]),
        registered_machines=[one, two],
        launch_requests=launches.read_launch_requests(root),
    )
    widget.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, widget.can_close)
    widget.target_combo.setCurrentIndex(1)
    widget.start_selected_machine()
    wait(app, widget.can_close)
    widget.target_combo.setCurrentIndex(2)
    widget.start_selected_machine()
    wait(app, widget.can_close)
    assert len(calls) == 1 and not widget.start_button.isEnabled()
    widget.backend_combo.setCurrentIndex(widget.backend_combo.findData("cuda"))
    assert widget.start_button.isEnabled()  # Separate backend; no implicit start.
    assert len(calls) == 1


def test_resource_alias_normalization_and_invalid_name_no_side_effect(panel, app):
    widget, _ = panel
    widget._resource_choice(dict(machine="mAiN", backend="cuda", workers=3))
    assert (
        widget.workers_spin.value() == 3
        and widget.backend_combo.currentData() == "cuda"
    )
    widget._resource_choice(dict(machine="../../bad", backend="cpu", workers=9))
    assert (
        widget.workers_spin.value() == 3
        and widget.backend_combo.currentData() == "cuda"
    )


def test_primary_failure_remains_visible_if_own_error_receipt_write_fails(
    panel, app, metadata_job, monkeypatch
):
    widget, _ = panel
    root, _, _ = metadata_job
    widget.reader = model.JobSnapshotReader()

    def failed(*a, **k):
        raise RuntimeError("primary launcher failure verbatim")

    def receipt_failed(*a, **k):
        raise OSError("secondary UI receipt write failed")

    widget.local_launcher = failed
    monkeypatch.setattr(ui, "record_launch_error", receipt_failed)
    widget.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, widget.can_close)
    widget.start_local_workers()
    wait(app, widget.can_close)
    assert "primary launcher failure verbatim" in widget.error_text.toPlainText()
    assert "secondary UI receipt write failed" in widget.error_text.toPlainText()
    assert not widget.start_button.isEnabled()
    widget.refresh()
    wait(app, widget.can_close)
    assert "primary launcher failure verbatim" in widget.error_text.toPlainText()
