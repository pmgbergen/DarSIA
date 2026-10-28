"""Compact actual task progress; elapsed updates reuse retained metadata only."""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower import jobs_panel as ui
from .test_native_jobs_owner import local_registry, workspace  # noqa: F401
from .test_native_jobs_sections import native_layout_font  # noqa: F401


def test_progress_table_and_elapsed_timer_retain_original_callbacks(qtbot, monkeypatch):
    from .test_jobs_panel import snapshot
    panel = ui.JobsPanel(local_machine="master")
    qtbot.addWidget(panel)
    value = snapshot()
    value["runs"] = [dict(run="run-a", state="Working (report only)", committed=1,
        total=4, started_utc="2026-01-01T10:00:00Z", machine="machine-a",
        worker="service-a · PID 7", backend="cpu"),
        dict(run="run-b", state="Queued", committed=0, total=4,
             machine="Not reported", worker="Not reported")]
    value["selected_machine"] = "not-running"
    before = deepcopy(value)
    panel._show_snapshot(value)
    table = panel.task_progress_table
    assert table.rowCount() == 2
    assert table.item(0, 4).text() == "Estimated 25.0%"
    assert table.item(0, 2).text() == "machine-a"
    assert table.item(0, 3).text() == "service-a · PID 7"
    assert table.item(1, 2).text() == "Not reported"
    assert "Estimated 12.5%" in panel.progress_label.text()
    assert "\n" not in panel.progress_label.text()
    assert panel.machine_table.isHidden() and panel.worker_table.isHidden()
    assert panel.operation_label.isHidden()
    def forbidden(*args, **kwargs):
        pytest.fail("Visible elapsed update called a reader or producer")
    panel.reader = panel.local_launcher = panel.remote_launcher = forbidden
    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "open", forbidden)
        guarded.setattr(Path, "stat", forbidden)
        guarded.setattr(panel._refresh_visibility_owner, "isVisible", lambda: True)
        panel._update_elapsed()
        panel.progress_label.setText("Explicit launch feedback")
        panel._update_elapsed()
        assert panel.progress_label.text() == "Explicit launch feedback"
    assert value == before
    assert panel.can_close()
    assert panel.start_button.receivers("2clicked()") > 0


def test_native_progress_is_one_compact_table_and_navigation_keeps_context(workspace, qtbot):
    import os
    from .test_native_jobs_owner import snapshot
    panel, sections = workspace.panel, workspace.inline_layout
    workspace.resize(1280, 900)
    workspace.show()
    value = snapshot(dict(root="retained-fixture"))
    value["job_facts"]["pipeline"] = "series-setup"
    value["work_unit"] = "operations"
    value["runs"] = [dict(run=f"run-{index}", operation="build",
        completed_operations=0, total_operations=1, state="Queued") for index in range(80)]
    value["runs"][0].update(state="Claimed (liveness not established)",
        machine="worker-a", worker="PID 7", backend="cpu",
        started_utc="2026-01-01T10:00:00Z",
        reported_progress=dict(step="verify_saved_geometry", message="Verifying saved source metadata"))
    value["workers"] = [dict(machine="worker-a", worker="PID 7", run="run-0",
        state="Claimed (liveness not established)", last_progress_utc="Not reported", error="")]
    panel._show_snapshot(value)
    sections.result_buttons[0].click()
    qtbot.wait(1)
    workspace.resize(1280, 900)
    qtbot.wait(1)
    assert workspace.width() == 1280
    assert panel.task_progress_table.isVisible() and panel.progress_label.isVisible()
    assert not sections.operations.isVisible()
    assert panel.task_progress_table.rowCount() == 80
    assert panel.progress_label.height() < 100
    assert workspace.minimumSizeHint().width() <= 1280
    output = os.environ.get("DARSIA_TEST_PROGRESS_SCREENSHOT")
    if output:
        assert workspace.grab().save(output)
    sections.result_buttons[1].click()
    assert panel.log_text.isVisible() and not panel.task_progress_table.isVisible()
    assert panel.current is value
    sections.result_buttons[0].click()
    assert panel.task_progress_table.isVisible() and panel.current is value
    assert panel.task_progress_table.item(0, 2).text() == "worker-a"
