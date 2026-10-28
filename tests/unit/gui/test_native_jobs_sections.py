"""Inline metadata forms retain explicit targets, edits and callback ownership."""

import hashlib
import threading

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from darsia.fluidflower import jobs_panel

from .test_native_jobs_owner import (  # noqa: F401
    OptionalResourceStub, local_registry, snapshot, workspace,
)
from .test_store_map_ui import RAW, reader_for

@pytest.fixture(autouse=True)
def native_layout_font(qapp):
    """Give Qt's offscreen backend the same readable font as Windows."""
    from pathlib import Path
    from PySide6.QtGui import QFont, QFontDatabase
    old = qapp.font()
    old_style = qapp.styleSheet()
    from darsia.fluidflower.ui.theme import apply_application_theme
    apply_application_theme(qapp)
    font = Path("C:/Windows/Fonts/segoeui.ttf")
    if font.is_file():
        QFontDatabase.addApplicationFont(str(font))
        qapp.setFont(QFont("Segoe UI", 10))
    yield
    qapp.setFont(old)
    qapp.setStyleSheet(old_style)



def _select(workspace, root):
    workspace.panel._show_snapshot(snapshot(dict(root=str(root))))


def _facts(editor):
    for edit, value in ((editor.machine_edit, "Main"), (editor.host_edit, "host"),
                        (editor.checkout_edit, "/checkout"), (editor.python_edit, "/python")):
        edit.setText(value)


def test_forms_are_passive_persistent_children(workspace, tmp_path):
    workspace.show()
    inline = workspace.inline_layout
    windows = {w for w in QApplication.topLevelWidgets() if w.isVisible()}
    _select(workspace, tmp_path / "job")
    workspace.panel.add_machine_button.click()
    inline.machine.host_edit.setText("keep-this-host")
    inline.machine_section.set_expanded(False)
    workspace.panel.add_machine_button.click()
    workspace.sources.setChecked(True)
    workspace.stores.settings_button.click()
    inline.store_editor.reject()
    assert inline.machine.host_edit.text() == "keep-this-host"
    assert not inline.machine.isWindow() and not inline.store_editor.isWindow()
    assert inline.store_editor.isVisible() and not inline.machine.isVisible()
    inline.open_machine()
    assert inline.machine.isVisible() and not inline.store_editor.isVisible()
    assert inline.store_editor._thread is None and workspace.stores._thread is None
    assert {w for w in QApplication.topLevelWidgets() if w.isVisible()} == windows
    assert workspace.can_close()


def test_registration_uses_original_callback_and_explicit_job(workspace, tmp_path, monkeypatch):
    root = tmp_path / "job"
    _select(workspace, root)
    workspace.panel.open_machine_dialog()
    editor = workspace.inline_layout.machine
    _facts(editor)
    calls, refreshes, receipts = [], [], []
    monkeypatch.setattr(jobs_panel, "register_machine_facts",
                        lambda *a, **k: calls.append((a, k)) or {"machine": "Main"})
    monkeypatch.setattr(workspace.panel, "refresh", lambda: refreshes.append(True))
    workspace.panel.machine_registered.connect(receipts.append)
    editor.new_version.setChecked(True)
    editor.ok_button.click()
    assert calls == [((str(root), editor.facts()), dict(local_machine="Main", new_version=True))]
    assert refreshes == [True] and receipts == [{"machine": "Main"}]
    assert editor.result() == 0  # the form was not accepted/closed as a dialog


def test_job_change_requires_review_before_registering(workspace, tmp_path, monkeypatch):
    _select(workspace, tmp_path / "first")
    workspace.panel.open_machine_dialog()
    editor = workspace.inline_layout.machine
    _facts(editor)
    calls = []
    monkeypatch.setattr(jobs_panel, "register_machine_facts",
                        lambda *a, **k: calls.append((a, k)))
    _select(workspace, tmp_path / "second")
    editor.accept()
    assert not calls and not editor.ok_button.isEnabled()
    assert "selected job changed" in editor.target.text()
    workspace.panel.open_machine_dialog()
    assert editor.ok_button.isEnabled() and editor.bound_root == str(tmp_path / "second")


def test_store_edits_survive_folding_and_block_close_until_explicit_discard(workspace):
    inline = workspace.inline_layout
    workspace.stores.open_settings()
    editor = inline.store_editor
    editor.new()
    editor.editor.setPlainText(editor.editor.toPlainText() + "# retained edit\n")
    inline.stores_section.set_expanded(False)
    editor.reject()
    assert not workspace.can_close() and editor.has_edits()
    assert "Save local settings or Discard local settings edits" in workspace.close_block_reason()
    workspace.stores.open_settings()
    assert "# retained edit" in editor.editor.toPlainText()
    editor.discard_button.click()
    assert not editor.has_edits() and workspace.can_close()


def test_machine_overrides_do_not_duplicate_start_series_folder_editors(workspace,qtbot,tmp_path):
    from darsia.fluidflower.store_map_ui import StoreSettingsDialog
    start_settings = StoreSettingsDialog(path=tmp_path/'stores.toml')
    qtbot.addWidget(start_settings)
    assert not start_settings.shared_series_locations.isHidden()
    workspace.show()
    workspace.stores.open_settings()
    machine_settings = workspace.inline_layout.store_editor
    assert machine_settings.shared_series_locations.isHidden()
    assert machine_settings.working_location.isVisible()
    assert machine_settings.cache_location.isVisible()


def test_store_callback_keeps_context_until_retirement_and_preserves_bytes(workspace, tmp_path, qtbot):
    path = tmp_path / "stores.toml"
    path.write_bytes(RAW)
    editor = workspace.inline_layout.store_editor
    editor.path = path
    calls, writes = [], []
    editor.reader = reader_for(path, calls)
    workspace.stores.open_settings()
    first, second = tmp_path / "S", tmp_path / "T"
    workspace.stores.set_context(first, "S")
    editor.load()
    qtbot.waitUntil(lambda: editor._thread is None)
    entered, release = threading.Event(), threading.Event()

    def writer(selected, raw, *, expected_bytes, forbidden_roots):
        entered.set()
        release.wait(5)
        writes.append((selected, raw, expected_bytes, forbidden_roots))
        return dict(path=str(selected), sha256=hashlib.sha256(raw).hexdigest(), previous_copy="fixture")

    editor.writer = writer
    try:
        editor.save()
        qtbot.waitUntil(entered.is_set)
        workspace.stores.set_context(second, "T")
        assert editor.series_root == first and not workspace.can_close()
        editor.discard()
        assert editor._thread is not None
    finally:
        release.set()
    qtbot.waitUntil(lambda: editor._thread is None)
    assert writes == [(path, RAW, RAW, [first])]
    assert editor.series_root == second and workspace.can_close()
    assert workspace.stores.path == path and workspace.stores.report is None
    assert path.read_bytes() == RAW


@pytest.mark.parametrize("width", [1280, 1600])
def test_rendered_left_controls_select_original_right_results(workspace, qtbot, tmp_path, width):
    workspace.resize(width, 1000)
    workspace.show()
    qtbot.waitExposed(workspace)
    inline, panel = workspace.inline_layout, workspace.panel
    _select(workspace, tmp_path / "job")
    QApplication.processEvents()
    left = workspace.jobs_view.controls_widget
    sections = [workspace.shared_jobs_section, inline.jobs_section, inline.control_section, inline.machines_section]
    positions = [section.mapTo(left, QPoint()).y() for section in sections]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    assert all(left.isAncestorOf(section) for section in sections)
    assert workspace.jobs_view.workspace.isAncestorOf(panel.job_table)
    assert panel.job_table.isVisible() and workspace.result_views.currentWidget() is inline.list_results
    assert not panel.detail_tabs.tabBar().isVisible()
    for index, button in enumerate(inline.result_buttons):
        button.click()
        assert panel.detail_tabs.currentIndex() == index
        assert panel.detail_tabs.currentWidget().isVisible()
        assert workspace.result_views.currentWidget() is workspace.job_results
        assert not panel.job_table.isVisible()
        assert not inline.machine.isVisible() and not inline.control_results.isVisible()
    panel.detail_tabs.setCurrentIndex(0)
    assert inline.result_buttons[0].isChecked()
    targets = (panel.target_combo, panel.backend_combo, panel.workers_spin)
    panel.place_target_controls(workspace.new_analysis)
    assert all(workspace.new_analysis.isAncestorOf(widget) for widget in targets)
    panel.place_target_controls()
    assert all(inline.control_results.isAncestorOf(widget) for widget in targets)
    QApplication.processEvents()
    workspace.resize(width, 1000)
    assert not workspace.jobs_view.controls_scroll.horizontalScrollBar().isVisible()
    assert workspace.minimumSizeHint().width() <= width
    assert workspace.width() == width
    assert workspace.grab().save(str(tmp_path / "jobs.png"))
    workspace.stores.open_settings()
    inline.store_editor.new()
    QApplication.processEvents()
    assert workspace.result_views.currentWidget() is inline.store_results
    assert workspace.shared_jobs_section.is_expanded() and workspace.sources.is_expanded()
    assert not inline.machines_section.is_expanded()
    assert workspace.jobs_view.workspace.isAncestorOf(inline.store_editor.editor)
    assert inline.store_results.isAncestorOf(inline.store_editor.save_button)
    assert workspace.grab().save(str(tmp_path / "jobs-stores.png"))


def test_late_resource_attachment_gets_a_vertical_result_choice(workspace):
    resources = OptionalResourceStub()
    workspace.attach_resources(resources)
    button = workspace.inline_layout.result_buttons[-1]
    assert button.text() == "Resources"
    button.click()
    assert workspace.panel.detail_tabs.currentWidget() is workspace.panel.resource_page
    assert workspace.panel.resource_page.isAncestorOf(resources)
    assert workspace.jobs_view.controls_widget.isAncestorOf(button)


def test_sections_choose_one_workspace_and_keep_form_edits(workspace, tmp_path):
    workspace.show()
    inline, panel = workspace.inline_layout, workspace.panel
    _select(workspace, tmp_path / "job")
    inline.control_section.set_expanded(True)
    assert inline.control_results.isVisible() and not panel.job_table.isVisible()
    assert not inline.jobs_section.is_expanded()
    panel.workers_spin.setValue(7)
    inline.open_machine()
    inline.machine.host_edit.setText("retained-host")
    workspace.sources.setChecked(True)
    assert workspace.sources_body.isVisible() and not inline.machine.isVisible()
    assert not inline.machines_section.is_expanded()
    inline.machines_section.toggle.click()
    assert inline.machine.isVisible() and inline.machine.host_edit.text() == "retained-host"
    inline.control_section.toggle.click()
    assert inline.control_results.isVisible() and panel.workers_spin.value() == 7
    inline.jobs_section.toggle.click()
    assert panel.job_table.isVisible() and not inline.control_results.isVisible()
    assert not inline.control_section.is_expanded()
    workspace.shared_jobs_section.set_expanded(True)
    assert workspace.shared_jobs.isVisible() and not panel.job_table.isVisible()
    assert not inline.jobs_section.is_expanded()


def test_long_operation_progress_does_not_appear_on_resources_or_logs(workspace, tmp_path):
    workspace.attach_resources(OptionalResourceStub())
    workspace.resize(1280, 900)
    workspace.show()
    data = snapshot(dict(root=str(tmp_path / "job")))
    data["job_facts"]["pipeline"] = "series-setup"
    data.update(total_operations=80, completed_operations=0, remaining_operations=80,
        runs=[dict(run=f"run{number}", operation="build", state="Waiting") for number in range(80)])
    workspace.panel._show_snapshot(data)
    inline, panel = workspace.inline_layout, workspace.panel
    inline.result_buttons[0].click()
    assert panel.operation_label.isVisible() and inline.operations.isVisible()
    assert panel.worker_table.isVisible() and not panel.job_table.isVisible()
    assert inline.operations.height() < workspace.height()
    inline.result_buttons[-1].click()
    assert panel.resource_page.isVisible()
    assert not panel.operation_label.isVisible() and not panel.master_label.isVisible()
    assert not panel.job_table.isVisible() and not inline.control_results.isVisible()
    inline.result_buttons[1].click()
    assert panel.log_text.isVisible() and not panel.resource_page.isVisible()
    assert not panel.operation_label.isVisible()
    QApplication.processEvents()
    workspace.resize(1280, 900)
    assert workspace.minimumSizeHint().width() <= 1280
    assert workspace.width() == 1280 and workspace.height() == 900
