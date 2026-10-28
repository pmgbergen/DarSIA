"""Calibration uses the real Shared Jobs controls and existing master service."""
from types import SimpleNamespace as NS
import pytest

from darsia.fluidflower import calibration_preparation as preparation
from darsia.fluidflower.refit_job_panel import RefitJobPanel
from .test_shared_jobs_ui import controls, Folder, Service, Tuner  # noqa: F401
from .test_native_jobs_owner import local_registry, workspace, snapshot  # noqa: F401


@pytest.mark.parametrize("automatic,cpu,gpu,expected", [(False, 6, 0, 6), (True, 6, 0, 0)])
def test_training_preparation_reads_visible_shared_capacity(controls, workspace, automatic, cpu, gpu, expected):
    workspace.panel.workers_spin.setValue(1)
    controls.automatic.setChecked(automatic)
    controls.cpu.setValue(cpu)
    controls.gpu.setValue(gpu)
    assert preparation.training_worker_count(workspace) == expected
    assert workspace.panel.workers_spin.value() == 1


def test_cpu_only_preparation_does_not_invent_worker_when_cpu_disabled(controls, workspace):
    controls.automatic.setChecked(False)
    controls.cpu.setValue(0)
    controls.gpu.setValue(1)
    with pytest.raises(ValueError, match="needs CPU workers"):
        preparation.training_worker_count(workspace)


@pytest.mark.parametrize("busy_first", [False, True])
def test_p1_reuses_shared_master_and_manual_six_cpu_workers(
        controls, workspace, qtbot, tmp_path, monkeypatch, busy_first):
    controls.automatic.setChecked(False)
    controls.cpu.setValue(6)
    controls.gpu.setValue(0)
    master = Service(Folder(str(tmp_path)), workspace.panel.local_machine,
                     role="master", cpu_workers=2, cuda_workers=1)
    controls._service, controls._tuner = master, Tuner()
    workspace.series_root, workspace.series_id = tmp_path, "series"
    monkeypatch.setattr(workspace.registry, "set_active", lambda *args: None)
    monkeypatch.setattr(workspace.panel, "start_local_workers", lambda: pytest.fail("Legacy launcher is not Shared Jobs"))
    phase = snapshot(dict(root=str(tmp_path / "phase-job")))
    phase["job_facts"]["expected_runs"] = ["run1"]
    panel = RefitJobPanel(machine="host")
    qtbot.addWidget(panel)
    calls = []
    monkeypatch.setattr(panel, "start_after_phase_one", lambda *args: calls.append(args) or True)
    window = NS(open_image_selection_plan=lambda: None, _select_row=lambda *args: None,
                show_phase_model_workflow=lambda: None)
    widget = preparation.CalibrationPreparation(window, panel)
    qtbot.addWidget(widget)
    widget.root = tmp_path
    value = dict(runs=[dict(run="run1", counts=dict(selected_frames=1, fit_candidates=1, earlier_history=0))],
                 counts=dict(selected_frames=1, fit_candidates=1))
    if busy_first:
        controls._thread = object()
    widget._prepared(workspace, dict(snapshot=phase, registration=None, selection=value), tmp_path, "series")
    if busy_first:
        qtbot.wait(60)
        assert Folder.published == []
        controls._thread = None
    qtbot.waitUntil(lambda: len(Folder.published) == 1 and controls._thread is None)
    assert controls._service is master and controls._tuner is None
    assert master.capacity == (6, 0) and master.tuner is None
    assert Folder.published == [(phase["root"], dict(master_machine=workspace.panel.local_machine,
                                                   cpu_workers=6, cuda_workers=0))]
    assert calls == [(phase["root"], ["run1"])]
    assert workspace.panel.workers_spin.value() == 1


def test_p1_auto_reuses_the_existing_tuner(controls, workspace, qtbot, tmp_path, monkeypatch):
    controls.automatic.setChecked(True)
    master = Service(Folder(str(tmp_path)), workspace.panel.local_machine,
                     role="master", cpu_workers=1, cuda_workers=0)
    tuner = Tuner()
    controls._service, controls._tuner = master, tuner
    workspace.panel._show_snapshot(snapshot(dict(root=str(tmp_path / "phase-job"))))
    controls.start_selected_job()
    qtbot.waitUntil(lambda: controls._thread is None)
    assert controls._service is master and controls._tuner is tuner
    assert master.capacity == tuner.counts()
    assert Folder.published[-1][1]["cpu_workers"] == 2
    assert Folder.published[-1][1]["cuda_workers"] == 1
