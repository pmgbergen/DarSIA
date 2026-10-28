"""Inline preparation preserves worker ownership and exact explicit approval."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from darsia.fluidflower import native_production_sections as ui
from .test_production_page import page, select  # noqa: F401


def _prepared(page):
    select(page)
    editor = page._preparation_dialog
    editor.prepared_result = dict(
        plan=dict(path="prepared.toml", sha256="b" * 64), adopted_for_execution=False)
    editor._prepared_context = editor._preparation_context()
    editor.adopt_button.setEnabled(True)
    return editor


def test_open_and_fold_retain_inputs_without_a_workflow_window(page):
    page.show()
    editor = page._preparation_dialog
    editor.inputs.setText("keep-current-inputs.json")
    windows = {w for w in QApplication.topLevelWidgets() if w.isVisible()}
    page.inline_layout.preparation_section.toggle.click()
    page.prepare_plan_button.click()
    assert page._preparation_dialog is editor and editor.isVisible()
    assert editor.inputs.text() == "keep-current-inputs.json"
    assert not editor.isWindow() and editor.window() is page
    assert {w for w in QApplication.topLevelWidgets() if w.isVisible()} == windows
    assert editor._thread is None and page.panel._create_thread is None


def test_review_and_adoption_stay_inline_with_exact_original_call(page, monkeypatch):
    editor = _prepared(page)
    for cls, name in ((QInputDialog, "getText"), (QMessageBox, "question")):
        monkeypatch.setattr(cls, name, lambda *_a, **_k: pytest.fail("Unexpected approval window"))
    calls, queued = [], []
    monkeypatch.setattr(ui, "adopt_prepared_plan", lambda *a, **k: calls.append((a, k)))
    editor._run = lambda call, **kwargs: queued.append((call, kwargs))
    editor.adopt()
    assert not queued and not editor.confirm_adoption.isEnabled()
    editor.approval_actor.setText("Reviewer")
    editor.approval_reason.setText("Reviewed these exact inputs")
    assert not editor.confirm_adoption.isEnabled()
    editor.approval_confirmed.setChecked(True)
    editor.confirm_adoption.click()
    assert len(queued) == 1 and queued[0][1] == {"adopting": True}
    queued[0][0]()
    assert calls[0][1] == dict(expected_sha256="b" * 64, actor="Reviewer",
                                reason="Reviewed these exact inputs")
    assert str(calls[0][0][0]) == "prepared.toml"
    assert calls[0][0][1].name == "execution.adopted.toml"


@pytest.mark.parametrize("change", ["plan", "series", "inputs"])
def test_stale_inline_review_cannot_adopt(page, monkeypatch, change):
    editor = _prepared(page)
    editor.adopt()
    editor.approval_actor.setText("Reviewer")
    editor.approval_reason.setText("Reviewed")
    editor.approval_confirmed.setChecked(True)
    monkeypatch.setattr(editor, "_run", lambda *_a, **_k: pytest.fail("Stale approval ran"))
    if change == "plan":
        editor.prepared_result = deepcopy(editor.prepared_result)
        editor.prepared_result["plan"]["sha256"] = "c" * 64
    elif change == "series":
        page.panel.series_id = "another-series"
    else:
        editor.inputs.setText("different-inputs.json")
    editor.confirm_adoption.click()
    assert editor._reviewed_plan is None and not editor.confirm_adoption.isEnabled()


def test_context_switch_retains_owned_callback_until_retired(page):
    select(page)
    editor = page._preparation_dialog
    previous_section = page.inline_layout.preparation_section
    callbacks = []
    def start(call, ready, closed):
        callbacks.append((call, ready, closed))
        return SimpleNamespace(wait=lambda: None), object()
    editor._start_call = start
    editor._run(lambda: dict(plan=dict(path="earlier.toml", sha256="a" * 64)))
    assert not page.can_close()
    page.panel.series_id = "another-series"
    page.inline_layout.sync()
    assert page._preparation_dialog is editor
    assert not editor.isEnabled() and not page.can_close()
    call, ready, closed = callbacks.pop()
    ready(call())
    assert editor.prepared_result is None
    closed()
    assert page.can_close() and page._preparation_dialog is not editor
    assert page._preparation_dialog.prepared_result is None
    assert previous_section.isHidden() and editor.approval_section.isHidden()


@pytest.mark.parametrize("width", [1280, 1600])
def test_rendered_controls_stay_left_of_live_results(page, qtbot, tmp_path, width):
    editor = _prepared(page)
    page.resize(width, 900)
    page.show()
    qtbot.waitExposed(page)
    inline = page.inline_layout
    editor.adopt()
    QApplication.processEvents()
    sections = [*inline.sections, inline.preparation_section, editor.approval_section]
    left = inline.splitter.controls_widget
    positions = [section.mapTo(left, QPoint()).y() for section in sections]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    assert all(left.isAncestorOf(section) for section in sections)
    right_start = inline.editor_scroll.mapTo(page, QPoint()).x()
    assert inline.splitter.controls_scroll.mapTo(page, QPoint()).x() < right_start
    assert inline.editor_scroll.isAncestorOf(page.selection_label)
    assert inline.editor_scroll.isAncestorOf(editor.status)
    assert "prepared.toml" in editor.approval_binding.text()
    assert editor.approval_binding.mapTo(page, QPoint()).x() >= right_start
    assert not inline.splitter.controls_scroll.horizontalScrollBar().isVisible()
    assert page.grab().save(str(tmp_path / f"production-expanded-{width}.png"))
    before = editor.status.mapTo(page, QPoint())
    for section in sections:
        section.toggle.click()
    QApplication.processEvents()
    assert all(not section.content.isVisible() for section in sections)
    assert editor.status.isVisible() and editor.status.mapTo(page, QPoint()) == before
    assert not inline.splitter.controls_scroll.horizontalScrollBar().isVisible()
    pixmap = page.grab()
    assert not pixmap.isNull() and pixmap.width() == width
    assert pixmap.save(str(tmp_path / f"production-{width}.png"))
