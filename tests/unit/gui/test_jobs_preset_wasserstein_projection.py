"""Typed original Comparison display and transport fixtures; source-phase unrun."""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower import jobs_panel as ui
from darsia.fluidflower import remote_worker_launch as remote
from .test_jobs_setup_projection import _image, _setup, _ready
from .test_new_analysis_action_feedback import _job, _put


def _comparison(root):
    return dict(root=str(root), name="Comparison fixture", series="campaign", state="Waiting",
        job_facts=dict(format="darsia-alias-job-v1", pipeline="preset-wasserstein",
            precision="original", backend="cpu", execution_ready=True,
            runtime_support="darsia-role-io-runtime-v1"),
        run_count=2, completed_runs=1, work_unit="comparisons", completed_comparisons=3,
        total_comparisons=6, remaining_comparisons=3, committed=0, total_frames=None,
        remaining=None, frames_per_minute=None, eta_seconds=None,
        workers=[], machines=[], registered_machines=[], logs=[], errors=[], launch_requests=[],
        rate_scope="Verified published comparison result records",
        nonfinite_distances=1, result_statuses={"missing": 1, "success": 2},
        result_warnings=["The original result includes a missing distance"],
        runs=[dict(run="comparison-00000000", state="Complete",
                   comparison_time=dict(order=0, time_hours=1.0, tolerance_hours=.25)),
              dict(run="comparison-00000001", state="Queued",
                   comparison_time=dict(order=1, time_hours=2.0, tolerance_hours=.25))])


def test_comparison_counts_records_and_keeps_scientific_status_separate(qtbot, tmp_path):
    panel = ui.JobsPanel(local_machine="fixture")
    qtbot.addWidget(panel)
    value = _comparison(tmp_path / "comparison-job")
    _ready(panel, [value])
    assert panel.job_table.horizontalHeaderItem(3).text() == "Requested times"
    assert panel.job_table.horizontalHeaderItem(4).text() == "Comparison result records"
    assert panel.job_table.item(0, 3).text() == "1 / 2"
    assert panel.job_table.item(0, 4).text() == "3 / 6 comparison result records"
    assert "Comparison result records: 3 / 6" in panel.progress_label.text()
    assert "Remaining: 3" in panel.progress_label.text()
    assert "Nonfinite distances: 1" in panel.progress_label.text()
    assert '"missing": 1' in panel.progress_label.text()
    assert "original result includes a missing distance" in panel.progress_label.text()
    assert "does not certify finite distances or scientific acceptance" in panel.progress_label.text()
    assert ui.COMPARISON_LAUNCH_SCOPE in panel.progress_label.text()
    assert "Committed images" not in panel.progress_label.text()
    assert panel.operation_label.text().splitlines() == [
        "Requested time 1.0 h · tolerance 0.25 h · comparison-00000000: Complete",
        "Requested time 2.0 h · tolerance 0.25 h · comparison-00000001: Queued",
    ]
    assert panel.machine_table.isHidden() and not panel.machine_progress_note.isHidden()
    assert "comparison result counts" in panel.machine_progress_note.text()
    assert "original input and solver types" in panel.profile_label.text()
    assert panel.can_close()


def test_switching_comparison_setup_and_images_restores_original_presentations(qtbot, tmp_path):
    panel = ui.JobsPanel(local_machine="fixture")
    qtbot.addWidget(panel)
    image, setup = _image(tmp_path / "image"), _setup(tmp_path / "setup")
    _ready(panel, [image])
    image_text = panel.progress_label.text()
    _ready(panel, [setup])
    setup_text = panel.progress_label.text()
    values = [image, setup, _comparison(tmp_path / "comparison")]
    _ready(panel, values)
    assert panel.job_table.horizontalHeaderItem(3).text() == "Runs / operations / times"
    for index in (2, 1, 0, 2, 0):
        panel.job_table.selectRow(index)
        if index == 0:
            assert panel.progress_label.text() == image_text
            assert panel.operation_label.isHidden() and not panel.machine_table.isHidden()
            assert panel.pause_button.text() == "Pause job after current frame"
        elif index == 1:
            assert panel.progress_label.text() == setup_text
            assert panel.machine_progress_note.text() == "Per-machine operation counts, throughput and ETA are not recorded."
            assert panel.pause_button.text() == "Pause job at next operation checkpoint"
        else:
            assert panel.pause_button.text() == "Pause job after current comparison result"
            assert panel.control_note.text().startswith(ui.COMPARISON_CONTROL_BOUNDARY)
    _ready(panel, [image, setup])
    assert panel.job_table.horizontalHeaderItem(3).text() == "Runs / operations"
    assert panel.job_table.horizontalHeaderItem(4).text() == "Images / operations"


@pytest.mark.parametrize("transport", ["local", "ssh"])
def test_comparison_reuses_explicit_machine_worker_launch_with_cpu_guard(qtbot, tmp_path, monkeypatch, transport):
    """The UI uses original launch ownership; injected transport starts no service."""
    root = _job(tmp_path / "job")
    value = _comparison(root)
    value["control"] = dict(available=True, master_machine="fixture",
        state=dict(action="run", sequence=0, command=None))
    if transport == "ssh":
        record = dict(machine="remote-fixture", ssh_alias="fixture",
            python=r"C:\fixture\python.exe", checkout=r"C:\fixture\checkout",
            ssh_executable=r"C:\fixture\ssh.exe", default_shell="windows-powershell",
            job=ui.descriptor(root / "job.json"))
        record["source"] = _put(root / "ui" / "machines" / "registered.json", record)
        value["registered_machines"] = [record]
    monkeypatch.setattr(remote, "require_unc", lambda path: None)
    calls = []

    def launch(root, **choices):
        calls.append((root, choices))
        return dict(job_root=root, transport_started=True, numerical_progress_claimed=False)

    panel = ui.JobsPanel(local_machine="fixture", local_launcher=launch, remote_launcher=launch,
        reader=lambda registration: deepcopy(value))
    qtbot.addWidget(panel, before_close_func=lambda widget: qtbot.waitUntil(widget.can_close, timeout=10000))
    panel.registrations = [dict(root=value["root"])]
    _ready(panel, [value])
    panel.backend_combo.setCurrentIndex(panel.backend_combo.findData("cuda"))
    assert not panel.start_button.isEnabled() and "requires CPU" in panel.start_button.toolTip()
    panel.start_selected_machine()
    assert not calls and "requires CPU" in panel.error_text.toPlainText()
    panel.backend_combo.setCurrentIndex(panel.backend_combo.findData("cpu"))
    panel.workers_spin.setValue(3)
    if transport == "ssh":
        panel.target_combo.setCurrentIndex(1)
    assert not panel.start_button.isEnabled() and "exactly one worker" in panel.start_button.toolTip()
    panel.start_selected_machine()
    assert not calls and "exactly one worker" in panel.error_text.toPlainText()
    panel.workers_spin.setValue(1)
    assert panel.start_button.isEnabled() and not calls
    assert panel.pause_button.toolTip() == ui.COMPARISON_CONTROL_BOUNDARY
    panel.start_button.click()
    assert len(calls) == 1 and calls[0][1]["workers"] == 1 and calls[0][1]["backend"] == "cpu"
    assert ("published comparison result records" if transport == "local" else
            "comparison results are not inferred") in panel.progress_label.text()
    qtbot.waitUntil(panel.can_close, timeout=10000)
    assert panel.current["completed_comparisons"] == 3
