"""Native page visibility refreshes metadata without starting services."""

import pytest

from darsia.fluidflower.pipeline.job_queue import descriptor

from .test_jobs_metadata import put
from .test_native_jobs_owner import (  # noqa: F401
    local_registry, snapshot, workspace,
)


def test_jobs_page_refreshes_while_original_panel_is_hidden(workspace, qtbot, tmp_path):
    panel = workspace.panel
    root = tmp_path / "job"
    put(root / "job.json", dict(expected_runs=["run"]))
    observed = []
    committed = 0

    def read(row):
        observed.append(committed)
        return dict(snapshot(row), committed=committed, remaining=2 - committed)

    panel.reader = read
    panel.set_context(jobs_provider=lambda: [dict(root=str(root), active=True)])
    qtbot.waitUntil(panel.can_close)
    assert not panel.timer.isActive()
    assert panel.timer.interval() == 10000
    workspace.show()
    qtbot.waitUntil(panel.can_close)
    assert panel.timer.isActive()
    workspace.stores.open_settings()
    assert workspace.jobs_view.isVisible() and not panel.isVisible()
    assert panel.pause_button.isVisible() and panel.timer.isActive()

    # Exercise the real timer, with a short test-only interval.
    committed = 1
    panel.timer.setInterval(25)
    qtbot.waitUntil(lambda: panel.current["committed"] == 1)
    panel.timer.setInterval(10000)
    qtbot.waitUntil(panel.can_close)
    assert observed[-1] == 1
    for view in ("new_analysis", "wasserstein"):
        workspace.show_view(view)
        assert not panel.timer.isActive()
        workspace.show_view("jobs")
        assert panel.timer.isActive()
        qtbot.waitUntil(panel.can_close)
    workspace.hide()
    assert not panel.timer.isActive()
    workspace.show()
    assert panel.timer.isActive()
    qtbot.waitUntil(panel.can_close)
    assert panel._session_requests == []


@pytest.mark.parametrize("failure_stage", ["read", "display"])
def test_metadata_refresh_failure_is_visible_and_manual_retry_recovers(
    workspace, qtbot, tmp_path, monkeypatch, failure_stage,
):
    panel = workspace.panel
    root = tmp_path / "job"
    put(root / "job.json", dict(expected_runs=["run"]))
    panel.set_context(jobs_provider=lambda: [dict(root=str(root), active=True)])
    qtbot.waitUntil(panel.can_close)
    assert panel.refresh_state_label.text().startswith("Job metadata refreshed:")
    field = "_read_known_launches" if failure_stage == "read" else "_show_read_result"
    original = getattr(panel, field)

    def fail(*args):
        raise RuntimeError("synthetic metadata refresh failure")

    monkeypatch.setattr(panel, field, fail)
    panel.refresh_button.click()
    qtbot.waitUntil(panel.can_close)
    assert "refresh failed" in panel.refresh_state_label.text()
    assert "synthetic metadata refresh failure" in workspace.status_label.text()
    monkeypatch.setattr(panel, field, original)
    panel.refresh_button.click()
    qtbot.waitUntil(panel.can_close)
    assert panel.refresh_state_label.text().startswith("Job metadata refreshed:")
    assert panel._session_requests == []


def test_explicit_launch_refreshes_bound_master_with_original_panel_hidden(
    workspace, qtbot, tmp_path,
):
    panel = workspace.panel
    root = tmp_path / "job"
    put(root / "job.json", dict(expected_runs=["run"]))
    panel.set_context(jobs_provider=lambda: [dict(root=str(root), active=True)])
    qtbot.waitUntil(panel.can_close)
    workspace.show()
    workspace.stores.open_settings()
    qtbot.waitUntil(panel.can_close)
    # Isolate the finite post-launch refresh from the periodic refresh.
    panel.timer.stop()
    assert not panel.isVisible() and not panel.pause_button.isEnabled()
    assert "No recorded master" in panel.control_state_label.text()
    launches = []

    def launch(job_root, **options):
        launches.append((job_root, options))
        put(root / "ui" / "master.json", dict(
            format="darsia-job-ui-master-v1", job=descriptor(root / "job.json"),
            machine=panel.local_machine,
        ))
        return dict(job_root=job_root, started=True, launcher_pid=123)

    panel.local_launcher = launch
    panel.start_selected_machine()
    qtbot.waitUntil(panel.can_close)
    assert len(launches) == 1
    assert panel.current["control"]["available"] is True
    assert panel.pause_button.isEnabled() and panel.stop_button.isEnabled()
    assert not panel.resume_button.isEnabled()
    assert "Recorded master: " + panel.local_machine in panel.control_state_label.text()
    assert panel.current["committed"] == 0
    assert "worker state unknown" in panel.launch_state_label.text()
    assert not panel.start_button.isEnabled()


def test_manual_refresh_reads_new_master_with_original_panel_hidden(
    workspace, qtbot, tmp_path,
):
    panel = workspace.panel
    root = tmp_path / "job"
    put(root / "job.json", dict(expected_runs=["run"]))
    panel.set_context(jobs_provider=lambda: [dict(root=str(root), active=True)])
    qtbot.waitUntil(panel.can_close)
    workspace.show()
    workspace.stores.open_settings()
    qtbot.waitUntil(panel.can_close)
    panel.timer.stop()
    assert "No recorded master" in panel.control_state_label.text()
    put(root / "ui" / "master.json", dict(
        format="darsia-job-ui-master-v1", job=descriptor(root / "job.json"),
        machine=panel.local_machine,
    ))
    panel.refresh_button.click()
    qtbot.waitUntil(panel.can_close)
    assert panel.current["control"]["available"] is True
    assert panel.pause_button.isEnabled() and panel.stop_button.isEnabled()
    assert "Recorded master: " + panel.local_machine in panel.control_state_label.text()
    assert panel._session_requests == []


def test_native_setup_refresh_projects_current_service_progress_and_master_without_raw_reads(
    workspace, qtbot, tmp_path, monkeypatch,
):
    from darsia.fluidflower import raw_review_postrun
    from darsia.fluidflower.jobs_metadata import JobSnapshotReader
    from darsia.fluidflower.pipeline.alias_job import job_metadata_reference
    from ..test_series_setup_results import _claimed, _progress_line

    def forbidden(*args, **kwargs):
        raise AssertionError("Setup must not inspect RAW review preparations")

    monkeypatch.setattr(raw_review_postrun, "declared_preparations", forbidden)
    case = _claimed(tmp_path)
    panel = workspace.panel
    panel.local_machine = case.task["target_machine"]
    panel.reader = JobSnapshotReader()
    panel.set_context(jobs_provider=lambda: [dict(root=str(case.root), active=True)])
    qtbot.waitUntil(panel.can_close)
    workspace.show()
    qtbot.waitUntil(panel.can_close)
    panel.timer.stop()
    assert "No recorded master" in panel.control_state_label.text()
    assert "Last report:" not in panel.progress_label.text()
    case.log.write_text(_progress_line(case), encoding="utf-8")
    put(case.root / "ui" / "master.json", dict(format="darsia-job-ui-master-v2",
        job=job_metadata_reference(case.root, case.root / "job.json"),
        machine=panel.local_machine))
    panel.refresh_button.click()
    qtbot.waitUntil(panel.can_close)
    for text in ("apply", "child-run", "verify_saved_geometry",
                 "Checking saved geometry metadata"):
        assert text in panel.progress_label.text()
    assert panel.worker_table.item(0, 4).text() == "2026-01-01T10:00:00Z"
    assert panel.pause_button.isEnabled() and panel.stop_button.isEnabled()
    assert panel.current["completed_operations"] == panel.current["committed"] == 0
    assert not panel.current["complete"] and not panel.current["errors"]
    assert panel.raw_review_button.isHidden() and panel.raw_review_status.isHidden()
    assert panel.raw_review_attach_button.isHidden()
    assert "before claiming an operation" in panel.capacity_label.text()
    assert panel._session_requests == []
