"""Owned private Python children only; no training, phase inference or model data."""
import sys
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDockWidget, QMainWindow, QTextEdit
from shiboken6 import isValid

from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow
from darsia.fluidflower.native_phase_steps import NativePhaseSteps
from darsia.fluidflower.ui import model_training as ui
from .test_review_session_refresh import studio  # noqa: F401


class CloseWindow(NativeFluidFlowerMainWindow):
    """Exercise the original close handlers without constructing project services."""

    def __init__(self):
        QMainWindow.__init__(self)
        self.messages, self.cleanup = [], []
        self.log_text = QTextEdit()
        self.log_dock = QDockWidget("Logging", self)
        self.log_dock.setWidget(self.log_text)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.log_message.connect(self._append_log)
        self.log_dock.hide()
        self.jobs_workspace = None
        self.streaming_panel = SimpleNamespace(cleanup=lambda: self.cleanup.append("streaming"))
        self._series_native_context = SimpleNamespace(close=lambda: self.cleanup.append("series"))

    def print_log(self, text):
        self.messages.append(text)
        super().print_log(text)


@pytest.fixture
def close_window(studio, qtbot):
    page, _ = studio
    window = CloseWindow()
    page.setParent(window)
    steps = NativePhaseSteps(page, window, actor_provider=lambda: "reviewer")
    window.setCentralWidget(steps)
    window.phase_model_workspace = SimpleNamespace(studio_page=SimpleNamespace(native_steps=steps))
    qtbot.addWidget(window)
    steps.show_section("phase_model")
    steps.phase_tabs.setCurrentWidget(steps.groups[4])
    window.resize(1780, 950)
    window.show()
    return window, steps, page


def launch_private_child(page, steps, kind, tmp_path, monkeypatch):
    script = "import sys\nprint('private-close-ready', flush=True)\nsys.stdin.buffer.read(1)\nsys.exit(7)\n"
    arguments = ["-I", "-B", "-u", "-c", script]
    if kind == "training":
        steps._original_launch(dict(model_id="private-job", output_root=str(tmp_path / "output")),
            [sys.executable, *arguments], tmp_path, "Private lifecycle check")
        process, cleanup = page.training_process, page._training_finished
        ready = lambda: "private-close-ready" in page.log.toPlainText()
    else:
        monkeypatch.setattr(page, "_selected_validation_model", lambda: dict(
            root=str(tmp_path / "model"), model_id="private-model"))
        monkeypatch.setattr(page, "_companion_path", lambda _name: tmp_path / "unused.py")
        monkeypatch.setattr(ui, "phase_python_executable", lambda: sys.executable)
        monkeypatch.setattr(ui, "tool_arguments", lambda _path: arguments)
        import importlib.util
        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec", lambda name, *rest: object()
                            if name == "darsia.fluidflower.tools.phase_transfer_review" else real(name, *rest))
        (tmp_path / "cache").mkdir(exist_ok=True)
        page._launch_transfer_worker("list", [])
        process, cleanup = page.validation_process, page._transfer_finished
        ready = lambda: "private-close-ready" in page.transfer_status.text()
    assert process is not None
    completed = []
    # Retain original ownership after actual child exit until its cleanup slot runs.
    process.finished.disconnect(cleanup)
    process.finished.connect(lambda *values: completed.append(values))
    return process, cleanup, completed, ready


@pytest.mark.parametrize("kind,reason", [
    ("training", "Training is still running or finishing"),
    ("validation", "metadata lookup is still finishing"),
])
def test_page_and_main_close_wait_for_original_owned_process_cleanup(
    close_window, qtbot, monkeypatch, tmp_path, kind, reason,
):
    window, steps, page = close_window
    attribute = "training_process" if kind == "training" else "validation_process"
    process, cleanup, completed, ready = launch_private_child(page, steps, kind, tmp_path, monkeypatch)
    try:
        qtbot.waitUntil(lambda: process.state() == QProcess.ProcessState.Running and ready(), timeout=10000)
        process_id = process.processId()
        assert process_id > 0 and getattr(page, attribute) is process
        assert not steps.can_close() and reason in steps.close_block_reason()
        assert not steps.close() and steps.isVisible()
        assert reason in page.status.text() and reason in page.review_status.text()
        assert not window.log_dock.isVisible()
        assert not window.close() and window.isVisible()
        assert reason in window.messages[-1]
        assert window.log_dock.isVisible() and reason in window.log_text.toPlainText()
        if kind == "training":
            assert page.stop_button.text() in window.messages[-1]
        assert process.state() == QProcess.ProcessState.Running
        assert process.processId() == process_id and not window.cleanup

        process.write(b"x")
        process.closeWriteChannel()
        qtbot.waitUntil(lambda: bool(completed), timeout=10000)
        assert process.state() == QProcess.ProcessState.NotRunning
        assert getattr(page, attribute) is process
        assert not steps.can_close() and not window.close()
        assert not window.cleanup

        cleanup(*completed[0])
        assert getattr(page, attribute) is None and steps.can_close()
        assert steps.close()
        assert window.close()
        assert window.cleanup == ["streaming", "series"]
    finally:
        if isValid(process) and process.state() != QProcess.ProcessState.NotRunning:
            process.write(b"x")
            process.closeWriteChannel()
            qtbot.waitUntil(lambda: bool(completed), timeout=10000)
        if getattr(page, attribute) is process and completed:
            cleanup(*completed[0])


def test_main_close_preserves_storage_reason_for_older_stubs_without_message_hook_or_log_dock():
    messages = []
    parent = SimpleNamespace(phase_model_workspace=SimpleNamespace(
        studio_page=SimpleNamespace(native_steps=SimpleNamespace(can_close=lambda: False))),
        print_log=messages.append)
    event = QCloseEvent()
    NativeFluidFlowerMainWindow.closeEvent(parent, event)
    assert not event.isAccepted()
    assert messages == ["Phase-model file preview or copying is still finishing; close after it returns."]


def test_the_removed_transfer_tool_is_not_started(qtbot):
    from PySide6.QtWidgets import QLabel
    page = SimpleNamespace(transfer_count=QLabel("Reading experiments..."), transfer_status=QLabel(),
                           validation_process=None)
    ui.SharedPhaseModelTrainingPage._launch_transfer_worker(page, "list", [])
    assert page.validation_process is None and page.transfer_count.text() == ""
    assert page.transfer_status.text() == (
        "Testing a saved model on other experiments is not part of this version of DarSIA.")
