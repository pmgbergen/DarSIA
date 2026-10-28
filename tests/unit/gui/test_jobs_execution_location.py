"""Execution labels follow existing Jobs controls without starting workers."""

from darsia.fluidflower import jobs_panel as ui

from .test_jobs_panel import app, panel, snapshot


def test_machine_and_worker_choices_do_not_relabel_reported_execution(panel, monkeypatch):
    widget, calls = panel
    data = snapshot()
    data["workers"] = [{
        "machine": "Main", "worker": "slot 1", "run": "run-a", "state": "Idle",
        "last_progress_utc": "2026-10-22T22:00:00Z", "error": "",
    }]
    widget._show_snapshot(data)
    reported = widget.execution_label.text()
    assert "This computer reported: Main" in reported
    assert widget.job_folder_label.text() == "Job folder: local-fixture"
    monkeypatch.setattr(ui, "build_remote_worker_launch", lambda *a, **k: None)
    widget.target_combo.addItem("Remote", {"machine": "Remote", "local": False})
    widget.target_combo.setCurrentIndex(1)
    widget.workers_spin.setValue(3)
    assert "Next start: shared compute on Remote · 3 worker(s) · CPU" in widget.target_label.text()
    assert widget.execution_label.text() == reported
    assert not calls
    widget.target_combo.setCurrentIndex(0)
    assert "Next start: this computer (Main)" in widget.target_label.text()
    assert not calls


def test_job_list_location_is_based_on_each_snapshot(panel):
    widget, calls = panel
    local, remote = snapshot("local"), snapshot("remote")
    local["machines"] = [{"machine": "Main", "committed": 1}]
    remote["machines"] = [{"machine": "Remote", "committed": 1}]
    widget._read_ready({
        "generation": widget._generation,
        "snapshots": [local, remote],
    })
    assert "This computer reported: Main" in widget.job_table.item(0, 7).text()
    assert "Shared compute reported: Remote" in widget.job_table.item(1, 7).text()
    assert "Liveness is not established" in widget.job_table.item(1, 7).toolTip()
    assert "Liveness" not in widget.job_table.item(1, 7).text()
    assert not calls
