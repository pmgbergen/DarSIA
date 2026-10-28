"""Offscreen plan preparation; no production data, adoption, queue or worker."""
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from darsia.fluidflower import analysis_preparation as preparation
from darsia.fluidflower.analysis_preparation_ui import AnalysisPreparationDialog
from darsia.fluidflower.jobs_panel import JobsPanel
from darsia.fluidflower.pipeline.job_queue import descriptor
from .test_native_jobs_owner import local_registry, workspace  # noqa: F401


def test_opening_and_changing_stage_start_nothing_and_preparation_runs_off_ui(qtbot, tmp_path, monkeypatch):
    panel = JobsPanel()
    panel.series_root = tmp_path
    qtbot.addWidget(panel)
    dialog = AnalysisPreparationDialog(panel, panel)
    qtbot.addWidget(dialog)
    calls, entered, release = [], threading.Event(), threading.Event()
    gui_thread = threading.get_ident()
    monkeypatch.setattr(preparation, "job_identity", lambda repository: dict(commit="a" * 40, precision="float32"))
    def prepare(inputs, folder, **kwargs):
        assert threading.get_ident() != gui_thread
        calls.append((inputs, folder, kwargs))
        entered.set()
        assert release.wait(30)
        target = Path(folder)
        target.mkdir()
        (target / "execution.toml").write_text("synthetic-inert-plan", encoding="utf-8")
        return dict(plan=descriptor(target / "execution.toml"), adopted_for_execution=False, worker_started=False)
    monkeypatch.setattr(preparation, "prepare_phase1_inputs", prepare)
    dialog.show()
    dialog.stage.setCurrentIndex(1)
    assert not dialog.probability.isEnabled() and dialog.calibration.isEnabled()
    assert not dialog._file_buttons[dialog.geometry].isEnabled()
    dialog.stage.setCurrentIndex(0)
    assert not calls and dialog._thread is None
    for edit, name in [(dialog.inputs, "inputs"), (dialog.geometry, "geometry"), (dialog.pressure, "pressure")]:
        path = tmp_path / (name + ".json")
        path.write_text("{}", encoding="utf-8")
        edit.setText(str(path))
    dialog.folder.setText(str(tmp_path / "prepared"))
    dialog.prepare_button.click()
    qtbot.waitUntil(entered.is_set, timeout=30000)
    dialog.reject()
    assert dialog.isVisible()
    release.set()
    qtbot.waitUntil(lambda: dialog._thread is None, timeout=30000)
    assert len(calls) == 1 and "Prepared, not adopted" in dialog.status.text()
    assert dialog.adopt_button.isEnabled() and not (tmp_path / "job.json").exists()
    dialog.close()


def test_preparation_thread_blocks_dialog_page_panel_workspace_and_main_close(workspace, qtbot, tmp_path, monkeypatch):
    """Drive the real modal route with one blocked private preparation callback."""
    from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow
    page, panel = workspace.new_analysis, workspace.panel
    panel.series_root = tmp_path
    workspace.show()
    qtbot.waitUntil(workspace.can_close, timeout=10000)
    entered, release = threading.Event(), threading.Event()
    failures, checked = [], []
    inputs = {}
    for name in ("inputs", "geometry", "pressure"):
        path = tmp_path / (name + ".json")
        path.write_text("{}", encoding="utf-8")
        inputs[name] = path
    monkeypatch.setattr(preparation, "job_identity", lambda _repo: dict(commit="a" * 40, precision="float32"))

    def private_prepare(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("The private preparation callback was not released")
        return dict(plan=descriptor(inputs["inputs"]), adopted_for_execution=False, worker_started=False)

    monkeypatch.setattr(preparation, "prepare_phase1_inputs", private_prepare)

    def exercise_modal():
        dialog = QApplication.activeModalWidget()
        try:
            assert isinstance(dialog, AnalysisPreparationDialog) and dialog.isVisible()
            for name, path in inputs.items():
                getattr(dialog, name).setText(str(path))
            dialog.folder.setText(str(tmp_path / "unused-plan-output"))
            dialog.prepare_button.click()
            qtbot.waitUntil(entered.is_set, timeout=5000)
            assert not dialog.can_close() and not page.can_close()
            assert not panel.can_close() and not workspace.can_close()
            dialog.reject()
            assert dialog.isVisible() and "still finishing" in dialog.status.text()
            assert not dialog.close() and dialog.isVisible()
            assert not panel.close() and not workspace.close()
            assert workspace.isVisible()
            logs, event = [], QCloseEvent()
            host = SimpleNamespace(jobs_workspace=workspace, print_log=logs.append)
            NativeFluidFlowerMainWindow.closeEvent(host, event)
            assert not event.isAccepted() and logs
            checked.append("all parent close guards retained the running callback")
        except BaseException as error:
            failures.append(error)
        finally:
            release.set()
            try:
                qtbot.waitUntil(dialog.can_close, timeout=10000)
                assert dialog._thread is None and dialog._worker is None
                assert page.can_close() and panel.can_close() and workspace.can_close()
                checked.append("original thread retired before parent close became available")
            except BaseException as error:
                failures.append(error)
            dialog.reject()

    QTimer.singleShot(0, exercise_modal)
    page.prepare_plan()
    assert not failures, failures
    assert len(checked) == 2 and page._preparation_dialog is None
    assert workspace.close()
    assert not (tmp_path / "unused-plan-output").exists()


def test_preparation_dialog_reference_is_cleared_if_modal_entry_fails(workspace, monkeypatch):
    """A failed modal entry cannot leave a dead dialog as the page's close guard."""
    page = workspace.new_analysis
    entered = []

    def unavailable(dialog):
        entered.append(dialog)
        assert page._preparation_dialog is dialog
        raise RuntimeError("The private modal entry failed")

    monkeypatch.setattr(AnalysisPreparationDialog, "exec", unavailable)
    with pytest.raises(RuntimeError, match="private modal entry failed"):
        page.prepare_plan()
    assert len(entered) == 1 and page._preparation_dialog is None
    assert page.can_close() and workspace.panel.can_close() and workspace.can_close()
