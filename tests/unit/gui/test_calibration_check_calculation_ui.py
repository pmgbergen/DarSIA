"""Finite scheduling/modal seams only; original owned reader/thread lifecycle.

Scientific sources and scoring are tested separately. No finite callback in this
module represents source admission or a current campaign check result.
"""
import json
from pathlib import Path
import threading

import pytest
from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import QApplication, QFileDialog

from darsia.fluidflower import calibration_check_calculation_ui as ui
from darsia.fluidflower.calibration_checks_view import CalibrationChecksPage
from darsia.fluidflower.pipeline.job_queue import descriptor
from .test_calibration_checks import score_document
from .test_native_series_method import method_window  # noqa: F401


def _inputs(dialog, tmp_path, monkeypatch):
    # Protocol binding is a declared finite GUI scheduling seam here. The
    # real producer's exact/capped protocol refusal is exercised separately.
    monkeypatch.setattr(ui.calculation, "protocol_binding",
                        lambda path, *, series_root: descriptor(Path(path)))
    for key in dialog.fields:
        dialog.fields[key].setText(str(tmp_path / key))
    (tmp_path / "protocol").write_bytes(b"finite scheduling input")


def _finish(dialog, qtbot, release):
    release.set()
    if dialog._thread is not None:
        qtbot.waitUntil(lambda: dialog._thread is None, timeout=10000)
    dialog.close()


def test_real_owned_verify_compute_and_close_barrier_preserve_original_score_reader(qtbot, tmp_path, monkeypatch):
    page = CalibrationChecksPage()
    qtbot.addWidget(page)
    page.show()
    dialog = page._calculation_dialog
    assert dialog.isVisible() and not dialog.isWindow()
    assert dialog.window() is page
    windows = set(QApplication.topLevelWidgets())
    page._calculate_checks()
    assert set(QApplication.topLevelWidgets()) == windows
    qtbot.keyClick(dialog.fields["series"], Qt.Key.Key_Escape)
    assert dialog.isVisible()
    _inputs(dialog, tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    calls = []
    result = tmp_path / "new-score.json"
    def verify(*args, **kwargs):
        calls.append(("verify", args, kwargs, QThread.currentThread() != dialog.thread()))
        entered.set()
        if not release.wait(5):
            raise RuntimeError("finite reader release missing")
        return dict(ready=True, source_bindings=[], verdict=None, payload_bytes_verified=False)
    def compute(*args, **kwargs):
        calls.append(("compute", args, kwargs, QThread.currentThread() != dialog.thread()))
        result.write_bytes(json.dumps(score_document()).encode("utf8"))
        return dict(scores=descriptor(result), verdict="PASS", worker_started=False)
    monkeypatch.setattr(ui.calculation, "check_readiness", verify)
    monkeypatch.setattr(ui.calculation, "calculate_check_scores", compute)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(result), ""))
    try:
        assert not dialog.calculate_button.isEnabled()
        assert dialog.verify_inputs()
        qtbot.waitUntil(entered.is_set)
        assert not dialog.can_close() and not page.can_close()
        page.calculation_section.set_expanded(False)
        page.calculation_section.set_expanded(True)
        assert page._calculation_dialog is dialog
        assert dialog._thread is not None
        assert dialog.close() is False
        assert not dialog.calculate_button.isEnabled()
        release.set()
        qtbot.waitUntil(lambda: dialog._thread is None)
        assert dialog.calculate_button.isEnabled()
        ready = dialog._ready
        assert dialog.calculate()
        qtbot.waitUntil(lambda: dialog._thread is None)
        assert len(page.scores()) == 1 and page.scores()[0].path == str(result)
        assert calls[1][2]["expected_readiness"] is ready
        assert all(call[3] is True for call in calls)
        assert [call[0] for call in calls] == ["verify", "compute"]
        assert "No calibration was adopted" in dialog.status.text()
        assert "leave-one-run-out" in ui.EXPLANATION and "W/d" in ui.EXPLANATION
        assert "at least seven" in ui.EXPLANATION and "Delta" in ui.EXPLANATION
    finally:
        _finish(dialog, qtbot, release)


@pytest.mark.parametrize("change", ["cancel", "context"])
def test_save_question_never_retargets_verified_compute(qtbot, tmp_path, monkeypatch, change):
    dialog = ui.CalculateChecksDialog()
    qtbot.addWidget(dialog)
    _inputs(dialog, tmp_path, monkeypatch)
    monkeypatch.setattr(ui.calculation, "check_readiness", lambda *_a, **_k:
        dict(ready=True, source_bindings=[], verdict=None))
    called = []
    monkeypatch.setattr(ui.calculation, "calculate_check_scores", lambda *a, **k: called.append((a, k)))
    release = threading.Event()
    try:
        dialog.verify_inputs()
        qtbot.waitUntil(lambda: dialog._thread is None)
        def choose(*_a, **_k):
            if change == "context":
                dialog.fields["calibration"].setText("a different job")
                return str(tmp_path / "must-not-exist.json"), ""
            return "", ""
        monkeypatch.setattr(QFileDialog, "getSaveFileName", choose)
        assert dialog.calculate() is False
        assert called == [] and not (tmp_path / "must-not-exist.json").exists()
        if change == "context":
            assert dialog._ready is None and not dialog.calculate_button.isEnabled()
    finally:
        _finish(dialog, qtbot, release)


def test_failed_or_stale_owned_read_never_enables_compute(qtbot, tmp_path, monkeypatch):
    dialog = ui.CalculateChecksDialog()
    qtbot.addWidget(dialog)
    _inputs(dialog, tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def verify(*_a, **_k):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("finite reader release missing")
        return dict(ready=True, source_bindings=[], verdict=None)
    monkeypatch.setattr(ui.calculation, "check_readiness", verify)
    try:
        dialog.verify_inputs()
        qtbot.waitUntil(entered.is_set)
        dialog.fields["series"].setText("changed during owned read")
        release.set()
        qtbot.waitUntil(lambda: dialog._thread is None)
        assert dialog._ready is None and not dialog.calculate_button.isEnabled()
        assert "changed" in dialog.status.text()
        def refused(*_a, **_k):
            raise ValueError("the actual publication is unavailable")
        monkeypatch.setattr(ui.calculation, "check_readiness", refused)
        dialog.verify_inputs()
        qtbot.waitUntil(lambda: dialog._thread is None)
        assert dialog._ready is None and not dialog.calculate_button.isEnabled()
        assert "actual publication is unavailable" in dialog.status.text()
        assert dialog.can_close()
    finally:
        _finish(dialog, qtbot, release)


def test_native_close_refuses_while_original_checks_reader_is_owned(method_window, qtbot, tmp_path, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow
    window = method_window
    page = CalibrationChecksPage(window)
    window.calibration_checks_page = page
    page._calculate_checks()
    dialog = page._calculation_dialog
    _inputs(dialog, tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def verify(*_a, **_k):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("finite reader release missing")
        return dict(ready=True, source_bindings=[], verdict=None)
    monkeypatch.setattr(ui.calculation, "check_readiness", verify)
    try:
        dialog.verify_inputs()
        qtbot.waitUntil(entered.is_set)
        event = QCloseEvent()
        NativeFluidFlowerMainWindow.closeEvent(window, event)
        assert not event.isAccepted()
        assert dialog._thread is not None and not page.can_close()
    finally:
        _finish(dialog, qtbot, release)
    assert page.can_close()
