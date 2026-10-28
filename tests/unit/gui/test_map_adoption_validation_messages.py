"""Original Check messages stay visible without changing any adoption predicate.

Only the validator report is synthetic DATA. The legacy dialog uses its real
private preparation/candidate and native Qt worker. The controller case reuses
the original owned tiny RAW-basis fixture; no experiment data is accessed.
"""
from copy import deepcopy
import threading

import pytest
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower.map_adoption_view import AdoptionDialog

from ..test_geometry_registered_export_apply import registered_setup  # noqa: F401
from ..test_map_adoption_store import _prepare
from .test_map_adoption_dialog import _choose
from .test_painting_series_adoption_controller import (
    host, _use_original_findings_refresh, _choose_source_and_refresh,
    _accept_original_warnings, _explicit_original_check,
)  # noqa: F401 -- retain the original real preparation fixture
from ..test_painted_adoption_table import imported, synthetic_payload_admission  # noqa: F401


WARNING = dict(severity="warning", code="copied_source_review",
    message="Review the copied source names.", details={"run_id": "two", "layers": [5, 6]})
INFO = dict(severity="info", code="retained_source",
    message="The original source record is retained.", details={"run_id": "one"})
NOTICE = dict(severity="notice", code="future_issue_kind",
    message="An additional source note is present.", details={"source": "private.csv"})
ERROR = dict(severity="error", code="invalid_source",
    message="The original source check refused this candidate.", details={"run_id": "two"})


def _report(issues):
    return dict(schema=1, format="darsia-fluidflower-geometry-validation-v1",
        config_sha256="f" * 64, validated_utc="2026-10-22T06:00:00+00:00",
        summary=dict(error_count=sum(row["severity"] == "error" for row in issues),
                     warning_count=sum(row["severity"] == "warning" for row in issues)),
        issues=deepcopy(issues), additional_metadata={"retained_without_filtering": True})


def _dialog(qtbot, data, tmp_path):
    prepared = _prepare(data, tmp_path)
    dialog = AdoptionDialog(data.project, prepared["folder"], actor="private GUI test")
    qtbot.addWidget(dialog, before_close_func=lambda owned: owned.wait_for_workers())
    _choose(dialog, 5, 1)
    _choose(dialog, 6, 2)
    dialog.show()
    return dialog


@pytest.mark.parametrize("case", ["clean", "warning_info", "error_warning"])
def test_full_check_report_is_visible_and_keeps_original_save_predicate(
        qtbot, registered_setup, tmp_path, monkeypatch, case):
    issues = [] if case == "clean" else [WARNING, INFO, NOTICE] if case == "warning_info" else [ERROR, WARNING]
    expected = _report(issues)
    report = deepcopy(expected)
    threads = []
    def validate(_candidate):
        threads.append(threading.get_ident())
        return report
    monkeypatch.setattr(geometry, "validate_geometry_config", validate)
    dialog = _dialog(qtbot, registered_setup, tmp_path)
    store = geometry.GeometryRevisionStore(registered_setup.project)
    before = store.current_path.read_bytes()
    assert dialog.check()
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
    assert dialog.validation_report == expected and report == expected
    assert threads and all(number != threading.get_ident() for number in threads)
    assert dialog.check_messages.isReadOnly() and dialog.check_messages.isVisibleTo(dialog)
    assert dialog.check_messages.maximumHeight() == 150
    shown = dialog.check_messages.toPlainText()
    if issues:
        for issue in issues:
            assert issue["message"] in shown and issue["code"] in shown
        assert '"run_id": "two"' in shown
    else:
        assert shown == "No validation issues."
    errors = case == "error_warning"
    assert (dialog.candidate is None) == errors
    assert dialog.save_button.isEnabled() == (not errors)
    assert ("does not validate" in dialog.status.text()) == errors
    questions, decisions = [], []
    monkeypatch.setattr(QMessageBox, "question", lambda *args:
        questions.append(args[2]) or QMessageBox.StandardButton.No)
    dialog.confirmation_decided.connect(decisions.append)
    dialog._save_clicked()
    assert len(questions) == (0 if errors else 1)
    assert decisions == ([] if errors else [False])
    if questions:
        assert questions[0].startswith("Append a new geometry revision")
        if issues:
            for issue in issues:
                assert issue["message"] in questions[0]
        else:
            assert "Check messages:" not in questions[0]
    assert store.current_path.read_bytes() == before and dialog.revision is None


@pytest.mark.parametrize("next_action", ["choice", "clean_recheck", "failed_recheck"])
def test_changed_choice_or_next_check_cannot_show_stale_validation_messages(
        qtbot, registered_setup, tmp_path, monkeypatch, next_action):
    monkeypatch.setattr(geometry, "validate_geometry_config", lambda _candidate: _report([WARNING]))
    dialog = _dialog(qtbot, registered_setup, tmp_path)
    assert dialog.check()
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
    assert WARNING["message"] in dialog.check_messages.toPlainText()
    if next_action == "choice":
        _choose(dialog, 5, 2)
    else:
        def validate(_candidate):
            if next_action == "failed_recheck":
                raise OSError("private subsequent Check failure")
            return _report([])
        monkeypatch.setattr(geometry, "validate_geometry_config", validate)
        assert dialog.check()
    assert dialog.validation_report is None and dialog.candidate is None
    assert dialog.check_messages.toPlainText() == "" and not dialog.check_messages.isVisibleTo(dialog)
    if next_action != "choice":
        qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
        if next_action == "clean_recheck":
            assert dialog.validation_report == _report([])
            assert dialog.check_messages.toPlainText() == "No validation issues."
            assert dialog.save_button.isEnabled()
        else:
            assert "private subsequent Check failure" in dialog.status.text()
            assert dialog.validation_report is None and dialog.candidate is None
            assert not dialog.check_messages.isVisibleTo(dialog) and not dialog.save_button.isEnabled()


def test_automatic_adoption_keeps_one_confirmation_and_all_check_messages(qtbot, host, monkeypatch):
    widget, data, _old_preparation = host
    _use_original_findings_refresh(widget, monkeypatch)
    expected = _report([WARNING, INFO, NOTICE])
    monkeypatch.setattr(geometry, "validate_geometry_config", lambda _candidate: deepcopy(expected))
    questions = []
    def confirm(dialog, title, text, *args):
        questions.append(text)
        assert widget._adoption_thread is None and dialog._thread is None
        assert dialog.validation_report == expected and dialog.candidate is not None
        assert dialog.check_messages.isVisibleTo(dialog)
        for issue in expected["issues"]:
            assert issue["message"] in text and issue["message"] in dialog.check_messages.toPlainText()
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox, "question", confirm)
    store = geometry.GeometryRevisionStore(data.project)
    before = store.current_path.read_bytes()
    controller = widget.adoption_controller
    assert controller.start(widget.record, widget.selection)
    qtbot.waitUntil(lambda: controller.gap_dialog is not None and controller.dialog is not None
                   and not widget.busy(), timeout=30000)
    findings = controller.gap_dialog
    assert findings._report == expected and not findings.adopt.isEnabled() and not questions
    notice_row = next(index for index, row in enumerate(findings._rows) if row.get("issue") == NOTICE)
    assert findings._rows[notice_row]["issue"]["severity"] == "notice"
    assert findings.runs.isVisibleTo(findings) and findings.runs.item(notice_row, 3).text() == NOTICE["message"]
    _choose_source_and_refresh(qtbot, widget, "unknown",
        "Synthetic human reviewed the source; its state is unknown")
    assert not findings.adopt.isEnabled() and not questions
    _accept_original_warnings(qtbot, findings)
    _explicit_original_check(qtbot, widget)
    assert not findings.adopt.isEnabled() and not findings._warning_actions and not questions
    for index, row in enumerate(findings._rows):
        if row["kind"] == "warning":
            findings.runs.setCurrentCell(index, 0)
            qtbot.waitUntil(lambda: not findings.busy(), timeout=30000)
            findings.reason.setText("Synthetic human accepts this exact original warning")
            findings.approve_warning.click()
    assert findings._report == expected and len(findings._warning_actions) == 1
    assert findings.adopt.isEnabled() and not questions
    findings.adopt.click()
    qtbot.waitUntil(lambda: controller.dialog is not None and controller.context is None
                   and not widget.busy(), timeout=30000)
    assert len(questions) == 1 and controller.dialog.revision is None
    assert store.current_path.read_bytes() == before
