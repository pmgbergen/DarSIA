"""Typed Setup operation labels beside unchanged image-job presentation."""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower import jobs_panel as ui
from darsia.fluidflower import remote_worker_launch as remote
from darsia.fluidflower.jobs_panel import JobsPanel
from .test_new_analysis_action_feedback import _job, _put


def _image(root):
    """Supply ordinary image-job display facts without starting a reader."""
    return dict(root=str(root), name="Image job", series="campaign", state="Waiting",
        job_facts=dict(precision="float32"), run_count=1, completed_runs=0,
        committed=1, total_frames=3, remaining=2, frames_per_minute=None, eta_seconds=None,
        workers=[], machines=[], logs=[], errors=[], runs=[], rate_scope="Published image receipts")


def _setup(root):
    """Use the operation fields supplied by the typed shared-result projection."""
    return dict(root=str(root), name="Setup job", series="campaign", state="Waiting",
        job_facts=dict(format="darsia-alias-job-v1", pipeline="series-setup", precision="float32",
            execution_ready=True, runtime_support="darsia-role-io-runtime-v1"),
        run_count=3, completed_runs=1, completed_operations=1, total_operations=3,
        remaining_operations=2, work_unit="operations", committed=0, total_frames=None,
        remaining=None, frames_per_minute=None, eta_seconds=None, workers=[], machines=[],
        logs=[], errors=[], rate_scope="Only verified Setup operation completions",
        runs=[dict(run="setup-apply-fixture", operation="apply", state="Complete"),
              dict(run="run-a", operation="build", state="Waiting for Apply"),
              dict(run="run-b", operation="build", state="Queued")])


def _ready(panel, snapshots):
    """Deliver a finite reader's actual projection envelope to the real widget."""
    panel._read_ready(dict(generation=panel._generation, snapshots=deepcopy(snapshots),
                          known_launch_requests=[], known_launch_errors=[]))


def test_jobs_show_operation_counts_and_each_apply_build_stage(qtbot, tmp_path):
    """Setup counts operations, while unfinished child stages remain explicit."""
    panel = JobsPanel(local_machine="fixture")
    qtbot.addWidget(panel)
    snapshot = _setup(tmp_path / "setup-job")
    _ready(panel, [snapshot])
    assert panel.job_table.horizontalHeaderItem(3).text() == "Operations"
    assert panel.job_table.horizontalHeaderItem(4).text() == "Progress"
    assert panel.job_table.item(0, 3).text() == "3"
    assert panel.job_table.item(0, 4).text() == "1 / 3 operations"
    assert "Completed operations: 1 / 3" in panel.progress_label.text()
    assert "Remaining operations: 2" in panel.progress_label.text()
    assert "Committed images" not in panel.progress_label.text()
    assert not panel.operation_label.isHidden()
    assert panel.operation_label.text().splitlines() == [
        "Apply geometry · setup-apply-fixture: Complete",
        "Build run model · run-a: Waiting for Apply",
        "Build run model · run-b: Queued",
    ]
    assert "limits recorded in each operation" in panel.profile_label.text()
    assert panel.can_close()


def test_mixed_and_image_only_job_views_preserve_image_progress(qtbot, tmp_path):
    """Switching job kinds never relabels image counts as completed operations."""
    panel = JobsPanel(local_machine="fixture")
    qtbot.addWidget(panel)
    image, setup = _image(tmp_path / "image-job"), _setup(tmp_path / "setup-job")
    _ready(panel, [image])
    image_progress, image_profile = panel.progress_label.text(), panel.profile_label.text()
    assert panel.job_table.horizontalHeaderItem(3).text() == "Runs"
    assert panel.job_table.horizontalHeaderItem(4).text() == "Frames"
    _ready(panel, [image, setup])
    assert panel.job_table.horizontalHeaderItem(3).text() == "Runs / operations"
    assert panel.job_table.horizontalHeaderItem(4).text() == "Images / operations"
    assert panel.job_table.item(0, 4).text() == "1 / 3"
    assert panel.job_table.item(1, 4).text() == "1 / 3 operations"
    panel.job_table.selectRow(1)
    assert not panel.operation_label.isHidden()
    panel.job_table.selectRow(0)
    assert panel.operation_label.isHidden() and not panel.operation_label.text()
    assert panel.progress_label.text() == image_progress
    assert panel.profile_label.text() == image_profile
    _ready(panel, [image])
    assert panel.job_table.horizontalHeaderItem(4).text() == "Frames"
    assert panel.can_close()


def test_setup_explains_missing_machine_statistics_and_keeps_operation_and_claim_rows(qtbot, tmp_path):
    """Switching pipelines restores frame statistics without inventing Setup rates."""
    panel = JobsPanel(local_machine="fixture")
    qtbot.addWidget(panel)
    setup, image = _setup(tmp_path / "setup-job"), _image(tmp_path / "image-job")
    setup["runs"][2]["state"] = "Claimed (liveness not established)"
    setup["workers"] = [dict(machine="fixture-cpu", worker="PID 42", run="run-b",
        state="Claimed (liveness not established)", last_progress_utc="Not reported", error="")]
    image["machines"] = [dict(machine="fixture", committed=1, frames_per_minute=2.5,
        eta_seconds=60, rate_scope="Published image receipts")]
    _ready(panel, [setup, image])
    assert panel.machine_progress_note.wordWrap()
    assert panel.machine_progress_note.textFormat() == ui.Qt.TextFormat.PlainText
    for index in (0, 1, 0):
        panel.job_table.selectRow(index)
        if index == 0:
            assert panel.machine_table.isHidden()
            assert not panel.machine_progress_note.isHidden()
            assert panel.machine_progress_note.text() == (
                "Per-machine operation counts, throughput and ETA are not recorded.")
            assert "Completed operations: 1 / 3" in panel.progress_label.text()
            assert "Remaining operations: 2" in panel.progress_label.text()
            assert not panel.operation_label.isHidden()
            assert "Build run model · run-b: Claimed (liveness not established)" in panel.operation_label.text()
            assert not panel.worker_table.isHidden() and panel.worker_table.rowCount() == 1
            assert [panel.worker_table.item(0, column).text() for column in range(6)] == [
                "fixture-cpu", "PID 42", "run-b", "Claimed (liveness not established)", "Not reported", ""]
        else:
            assert not panel.machine_table.isHidden()
            assert panel.machine_progress_note.isHidden()
            assert panel.machine_table.horizontalHeaderItem(1).text() == "Committed frames"
            assert [panel.machine_table.item(0, column).text() for column in range(5)] == [
                "fixture", "1", "2.50 frames/min", "1.0 min", "Published image receipts"]
            assert "Committed images: 1" in panel.progress_label.text()
            assert panel.worker_table.rowCount() == 0
    assert panel.can_close()


@pytest.mark.parametrize("job_kind", ["image", "setup"])
@pytest.mark.parametrize("transport", ["local", "ssh"])
def test_selected_pipeline_controls_and_launch_feedback_use_its_progress_unit(
    qtbot, tmp_path, monkeypatch, job_kind, transport
):
    """Switch job kinds and request a fake transport without claiming progress."""
    image = _image(_job(tmp_path / "image-job"))
    setup = _setup(_job(tmp_path / "setup-job"))
    snapshots = [image, setup]
    for snapshot in snapshots:
        root = Path(snapshot["root"])
        snapshot["control"] = dict(available=True, master_machine="fixture",
            state=dict(action="run", sequence=0, command=None))
        if transport == "ssh":
            record = dict(machine="remote-fixture", ssh_alias="fixture",
                python=r"C:\fixture\python.exe", checkout=r"C:\fixture\checkout",
                ssh_executable=r"C:\fixture\ssh.exe", default_shell="windows-powershell",
                job=ui.descriptor(root / "job.json"))
            record["source"] = _put(root / "ui" / "machines" / "registered.json", record)
            snapshot["registered_machines"] = [record]
    monkeypatch.setattr(remote, "require_unc", lambda path: None)
    launches = []

    def launcher(root, **choices):
        launches.append((root, choices))
        return dict(job_root=root, transport_started=True, numerical_progress_claimed=False)

    by_root = {snapshot["root"]: snapshot for snapshot in snapshots}
    panel = JobsPanel(local_machine="fixture", local_launcher=launcher,
        remote_launcher=launcher, reader=lambda row: deepcopy(by_root[row["root"]]))
    qtbot.addWidget(panel, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    panel.registrations = [dict(root=snapshot["root"]) for snapshot in snapshots]
    _ready(panel, snapshots)
    for index in (0, 1, 0):
        panel.job_table.selectRow(index)
        expected_boundary = ui.SETUP_CONTROL_BOUNDARY if index else ui.CONTROL_BOUNDARY
        assert panel.pause_button.isEnabled() and panel.stop_button.isEnabled()
        assert panel.pause_button.text() == (
            "Pause job at next operation checkpoint" if index else "Pause job after current frame")
        assert panel.stop_button.text() == (
            "Stop job at next operation checkpoint" if index else "Stop job after current frame")
        assert panel.pause_button.toolTip() == panel.stop_button.toolTip() == expected_boundary
        assert panel.control_note.text().startswith(expected_boundary)
        assert ("verified operation completions" if index else "frame publications") in panel.start_button.toolTip()
    panel.job_table.selectRow(1 if job_kind == "setup" else 0)
    if transport == "ssh":
        panel.target_combo.setCurrentIndex(1)
    assert panel.start_button.isEnabled() and not launches
    panel.start_button.click()
    assert len(launches) == 1 and launches[0][0] == panel.current["root"]
    if transport == "local":
        expected = ("Local worker launch requested. Only subsequent verified operation completions count as progress."
            if job_kind == "setup" else
            "Local worker launch requested. Only subsequent actual frame publications count as progress.")
    else:
        expected = ("SSH transport launch requested. Worker startup and operation progress are not inferred from transport startup."
            if job_kind == "setup" else
            "SSH transport launch requested. Worker startup and frame progress are not inferred from transport startup.")
    assert panel.progress_label.text() == expected
    qtbot.waitUntil(panel.can_close, timeout=10000)
    if job_kind == "setup":
        assert panel.current["completed_operations"] == 1
    else:
        assert panel.current["committed"] == 1


def test_job_kind_names_each_typed_pipeline():
    from darsia.fluidflower.jobs_panel import job_kind
    assert [job_kind(dict(pipeline=p)) for p in ("series-setup", "native-phase-statistics",
            "n1-calibration-refit", "native-phase-n1", "n1", "preset-wasserstein", "wasserstein-comparison")] == [
        "Setup", "Phase maps", "Calibration", "Mass analysis", "Mass analysis", "Comparisons", "Comparisons"]
    assert job_kind(None) == "Not recorded" and job_kind(dict(pipeline="new-kind")) == "new-kind"
