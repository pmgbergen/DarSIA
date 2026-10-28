"""Inline management controls with finite readers and no real data actions."""

from types import SimpleNamespace as NS

import pytest
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QLabel, QLineEdit, QPushButton

from darsia.fluidflower import native_phase_review_session as lifecycle
from darsia.fluidflower.native_phase_subpage import NativePhaseModelStudioSubpage
from darsia.fluidflower.ui import phase_model_dependency_review as dependency_ui
from .test_native_phase_model_workflow import workflow
from .test_phase_model_dependency_review import ControlledPool, incomplete_report


@pytest.fixture
def managed(workflow, monkeypatch, tmp_path, qtbot):
    from darsia.fluidflower import native_phase_model_archive as archive
    from darsia.fluidflower.ui import model_training
    previous_font = QApplication.font()
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    QApplication.setFont(QFont("Segoe UI", 10))
    w = workflow
    w.steps.series_root, w.steps.series_id = tmp_path, "fixture-series"
    w.steps.actor_provider = lambda: "Reviewer"
    w.entry = {"root": str(tmp_path / "unused-model"), "model_id": "saved-model"}
    w.studio.model_history = [w.entry]
    w.studio._validation_model_label = lambda entry: entry["model_id"]
    monkeypatch.setattr(model_training, "_models_for_series", lambda entries, series: entries)
    monkeypatch.setattr(archive, "read_archive_snapshot", lambda root: {"models": []})
    w.actions = []
    monkeypatch.setattr(archive, "archive_model", lambda *a, **kw: w.actions.append(("archive", a, kw)))
    monkeypatch.setattr(archive, "restore_model", lambda *a, **kw: w.actions.append(("restore", a, kw)))
    monkeypatch.setattr(dependency_ui, "QThreadPool", ControlledPool)
    monkeypatch.setattr(lifecycle, "read_current_review_session", lambda *a, **kw:
        {"review_session_id": "retained-session", "target_model_id": "saved-model", "status": "active"})
    monkeypatch.setattr(lifecycle, "discard_review_session", lambda *a, **kw: w.actions.append(("discard", a, kw)))
    w.studio.review_save_root = QLineEdit(str(tmp_path / "reviews"), w.steps)
    w.studio.reviewer = QLineEdit("Reviewer", w.steps)
    w.studio.review_save_root.hide()
    w.studio.reviewer.hide()
    w.studio._refresh_review_sessions = lambda: w.actions.append(("refresh",))
    w.steps.show()
    w.steps.show_section("phase_model")
    w.steps.phase_tabs.setCurrentIndex(0)
    yield w
    if w.workflow.dependency_dialog is not None:
        panel = w.workflow.dependency_dialog
        panel.pool.active = 0
        panel.task = panel.completion = None
        panel.retirement.stop()
        panel.close()
    QApplication.setFont(previous_font)


def button(panel, text):
    return next(widget for widget in panel.findChildren(QPushButton) if widget.text() == text)


def test_archive_controls_are_inline_and_preserve_reason_context_and_callbacks(managed):
    w, page = managed, managed.workflow
    before = set(QApplication.topLevelWidgets())
    page.open_archive()
    panel = page.archive_dialog
    section = page.management_sections.sections["archive"]
    assert not panel.isWindow() and w.steps.isAncestorOf(panel)
    assert not any(window.isVisible() for window in set(QApplication.topLevelWidgets()) - before)
    assert section.toggle.text() == "Archive unused phase models"
    reason = panel.findChild(QLineEdit)
    assert not button(panel, "Archive").isEnabled() and not button(panel, "Restore").isEnabled()
    reason.setText(" ")
    assert not button(panel, "Archive").isEnabled()
    reason.setText("A concrete retained reason")
    section.set_expanded(False)
    page.open_archive()
    assert page.archive_dialog is panel and section.is_expanded()
    assert reason.text() == "A concrete retained reason"
    button(panel, "Archive").click()
    button(panel, "Restore").click()
    assert [row[0] for row in w.actions] == ["archive", "restore"]
    assert all(row[1] == (w.steps.series_root, w.entry) for row in w.actions)
    assert all(row[2] == dict(actor="Reviewer", reason=reason.text(), protected_roots=page.protected_roots())
               for row in w.actions)
    w.steps.series_root = w.steps.series_root / "other-series"
    button(panel, "Archive").click()
    assert len(w.actions) == 2
    assert any("series changed" in label.text() for label in panel.findChildren(QLabel))
    page.management_sections.sync_context()
    assert section.isHidden() and not section.isEnabled()


def test_original_dependency_reader_is_retained_when_folded_closed_or_series_change_attempted(managed, qtbot):
    w, page = managed, managed.workflow
    page.open_archive()
    button(page.archive_dialog, "Inspect model dependencies…").click()
    panel = page.dependency_dialog
    section = page.management_sections.sections["dependencies"]
    assert isinstance(panel, dependency_ui.ModelDependencyDialog) and not panel.isWindow()
    assert panel.entry == w.entry and panel.series_root == str(w.steps.series_root)
    assert panel.task is None and not panel.pool.submitted  # Opening never starts a read.
    panel.inspect.click()
    task = panel.task
    assert task.entry == w.entry and task.protected_roots == tuple(page.protected_roots())
    section.set_expanded(False)
    assert not task.cancelled.is_set() and page.close_block_reason()
    page._inspect_model_dependencies({"root": "another-model"})
    assert page.dependency_dialog is panel and section.is_expanded()
    subpage = NS(native_steps=w.steps, series_id=w.steps.series_id, project_root=w.steps.series_root)
    subpage.review_context_block_reason = lambda: NativePhaseModelStudioSubpage.review_context_block_reason(subpage)
    with pytest.raises(ValueError, match="dependency inspector"):
        NativePhaseModelStudioSubpage.bind_context(subpage, "another-series", w.steps.series_root / "other")
    button(section, "Close").click()
    assert task.cancelled.is_set() and panel.close_requested
    assert not section.isHidden() and page.close_block_reason()
    task.signals.finished.emit(task, incomplete_report(), "")
    qtbot.waitUntil(lambda: panel.completion is not None)
    assert not panel.can_close() and not section.isHidden()
    panel.pool.active = 0
    panel._retire()
    assert panel.can_close() and section.isHidden()
    assert not panel.delete_button.isEnabled() and not w.actions


def test_closed_archive_reloads_history_and_status_without_reassigning_a_reason(managed, monkeypatch):
    from darsia.fluidflower import native_phase_model_archive as archive
    w, page = managed, managed.workflow
    page.open_archive()
    panel = page.archive_dialog
    section = page.management_sections.sections["archive"]
    choices, reason = panel.findChild(QComboBox), panel.findChild(QLineEdit)
    reason.setText("Reason for the retained model")
    button(section, "Close").click()
    assert section.isHidden()
    added = {"root": str(w.steps.series_root / "new-model"), "model_id": "new-model"}
    w.studio.model_history = [added, w.entry]
    monkeypatch.setattr(archive, "read_archive_snapshot", lambda root:
        {"models": [dict(w.entry, archived=True)]})
    page.open_archive()
    assert page.archive_dialog is panel and not section.isHidden()
    assert choices.count() == 2 and choices.itemData(0) == added
    assert choices.currentData() == w.entry and choices.currentText().endswith(" · Archived")
    assert reason.text() == "Reason for the retained model"
    assert button(panel, "Archive").isEnabled()
    button(section, "Close").click()
    w.studio.model_history = [added]
    page.open_archive()
    assert page.archive_dialog is panel and choices.count() == 1
    assert choices.currentIndex() == -1 and reason.text() == ""
    assert not button(panel, "Archive").isEnabled() and not button(panel, "Restore").isEnabled()
    assert not button(panel, "Inspect model dependencies…").isEnabled() and not w.actions


def test_discard_reason_and_same_session_callback_are_inline_without_data_actions(managed):
    w, page = managed, managed.workflow
    w.steps.phase_tabs.setCurrentIndex(1)
    page.review_owner.discard_dialog()
    panel = page.review_owner._discard_dialog
    section = page.management_sections.sections["discard"]
    assert isinstance(panel, QDialog) and not panel.isWindow()
    assert w.steps.groups[3].isAncestorOf(panel)
    reason = panel.findChild(QLineEdit)
    discard = button(panel, "Discard session and preserve its saved evidence")
    assert not discard.isEnabled() and not w.actions
    reason.setText("Explicit fixture reason")
    section.set_expanded(False)
    page.review_owner.discard_dialog()
    assert page.review_owner._discard_dialog is panel and section.is_expanded()
    assert reason.text() == "Explicit fixture reason"
    discard.click()
    assert w.actions == [("discard", (w.steps.series_root / "reviews",),
        dict(series="fixture-series", reviewer="Reviewer", review_session_id="retained-session",
             reason="Explicit fixture reason")), ("refresh",)]
    assert section.isHidden()


def test_discard_rejects_changed_reviewer_and_active_correction_owner(managed):
    w, page = managed, managed.workflow
    page.review_owner.discard_dialog()
    panel = page.review_owner._discard_dialog
    panel.findChild(QLineEdit).setText("Retained reason")
    w.studio.reviewer.setText("Another reviewer")
    button(panel, "Discard session and preserve its saved evidence").click()
    assert not w.actions
    assert any("review context changed" in label.text() for label in panel.findChildren(QLabel))
    page.review_owner.process = NS(close_block_reason=lambda: "Use Close review first")
    page.review_owner.discard_dialog()
    assert page.review_owner._discard_dialog is panel and not w.actions
    assert w.studio.review_status.text() == "Use Close review first"
    page.review_owner.process = None
