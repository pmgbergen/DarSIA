"""Analysis section folding retains the original settings and action guards."""

import tomllib

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QTabWidget, QToolButton

from darsia.gui.ui.settings import unwrap_composite_widget
from .test_native_sidebar import native_window  # noqa: F401


def test_folded_analysis_edits_survive_navigation_and_original_save(native_window, tmp_path, qtbot):
    window = native_window
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    source = tmp_path / "comparison.toml"
    source.write_text('[wasserstein]\nrelative_tol = 0.02\n[future]\npreserve = "yes"\n')
    window.config_file = str(source)
    window.config_dict = tomllib.loads(source.read_text())
    before = source.read_bytes()
    windows = {w for w in QApplication.topLevelWidgets() if w.isVisible()}
    window._select_row(("comparison", "wasserstein_compute"))
    field = unwrap_composite_widget(window.settings_inputs["wasserstein.relative_tol"])
    section = next(s for s in window.analysis_sections.sections if s.isAncestorOf(field))
    field.setText("0.015")
    section.toggle.click()
    assert not field.isVisible()
    assert source.read_bytes() == before
    window._select_row(("comparison", "wasserstein_assemble"))
    assert window.config_dict["wasserstein"]["relative_tol"] == 0.015
    window._select_row(("comparison", "wasserstein_compute"))
    field = unwrap_composite_widget(window.settings_inputs["wasserstein.relative_tol"])
    section = next(s for s in window.analysis_sections.sections if s.isAncestorOf(field))
    assert not section.is_expanded()
    assert float(field.text()) == 0.015
    window.settings_factory.save_settings()
    saved = tomllib.loads(source.read_text())
    assert saved["wasserstein"]["relative_tol"] == 0.015
    assert saved["future"] == {"preserve": "yes"}
    assert {w for w in QApplication.topLevelWidgets() if w.isVisible()} <= windows
    assert not window.process_runner.active


def test_inline_analysis_actions_share_original_enable_state(native_window, qtbot):
    window = native_window
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    window._select_row(("comparison", "events"))
    buttons = window.analysis_sections.actions_section.findChildren(QToolButton)
    play = next(b for b in buttons if b.defaultAction() is window.toolbar_builder.play_action)
    stop = next(b for b in buttons if b.defaultAction() is window.toolbar_builder.stop_action)
    window.toolbar_builder.play_action.setEnabled(False)
    assert not play.isEnabled()
    window.toolbar_builder.stop_action.setEnabled(True)
    assert stop.isEnabled()
    window._select_row(("start", "series"))
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    assert not play.isEnabled()
    assert not window.process_runner.active


def test_original_analysis_controls_keep_edits_and_unknown_settings(native_window, tmp_path):
    window = native_window
    source = tmp_path / "analysis.toml"
    source.write_text('[options.analysis]\nshow_plots = false\n[future]\npreserve = "yes"\n')
    window.config_file = str(source)
    window.config_dict = tomllib.loads(source.read_text())
    before = source.read_bytes()
    window._select_row(("analysis", "cropping"))
    field = unwrap_composite_widget(window.settings_inputs["options.analysis.show_plots"])
    section = next(s for s in window.analysis_sections.sections if s.isAncestorOf(field))
    field.setChecked(True)
    section.toggle.click()
    window._select_row(("analysis", "segmentation"))
    assert window.config_dict["options"]["analysis"]["show_plots"] is True
    window._select_row(("analysis", "cropping"))
    field = unwrap_composite_widget(window.settings_inputs["options.analysis.show_plots"])
    section = next(s for s in window.analysis_sections.sections if s.isAncestorOf(field))
    assert field.isChecked() and not section.is_expanded()
    assert source.read_bytes() == before
    window.settings_factory.save_settings()
    saved = tomllib.loads(source.read_text())
    assert saved["options"]["analysis"]["show_plots"] is True
    assert saved["future"] == {"preserve": "yes"}
    assert not window.process_runner.active


@pytest.mark.parametrize("selection", [("comparison", "events"),
                                        ("comparison", "wasserstein_compute"),
                                        ("analysis", "cropping")])
def test_visible_analysis_has_vertical_controls_and_original_results(
        native_window, tmp_path, monkeypatch, qtbot, selection):
    from darsia.gui.ui import streaming

    window = native_window
    original = window.streaming_panel
    dock = window.streaming_dock
    source = tmp_path / "comparison.toml"
    source.write_text('[wasserstein]\nrelative_tol = 0.02\n')
    window.config_file = str(source)
    window.config_dict = tomllib.loads(source.read_text())
    result = tmp_path / "saved-result.png"
    image = QImage(640, 360, QImage.Format.Format_RGB32)
    image.fill(QColor("#246b86"))
    assert image.save(str(result))
    monkeypatch.setattr(streaming, "list_workflow_output_images", lambda *args: [result])
    window.showNormal()
    window.resize(1800, 1000)
    window._select_row(selection)
    qtbot.waitUntil(lambda: original.isVisible())
    QApplication.processEvents()
    workspace = window.analysis_sections.workspace
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert window.settings_layout.itemAt(0).widget() is workspace
    assert workspace.workspace is dock and dock.widget() is original
    assert original._results_images == [result]
    assert not original.image_canvas._original_pixmap.isNull()
    sections = window.analysis_sections.sections + [window.analysis_sections.actions_section]
    assert all(workspace.controls_widget.isAncestorOf(section) for section in sections)
    assert [section.y() for section in sections] == sorted(section.y() for section in sections)
    controls_right = workspace.controls_scroll.mapTo(window, QPoint(workspace.controls_scroll.width(), 0)).x()
    results_left = dock.mapTo(window, QPoint(0, 0)).x()
    assert controls_right <= results_left
    assert dock.width() > workspace.controls_scroll.width()
    # Render the real native window, including the original results canvas.
    rendered = window.grab()
    assert not rendered.isNull() and rendered.width() == window.width()
    assert rendered.save(str(tmp_path / ("-".join(selection) + ".png")))
    section = sections[0]
    section.toggle.click()
    assert not section.content.isVisible() and original.isVisible()
    assert workspace.controls_scroll.mapTo(window, QPoint(workspace.controls_scroll.width(), 0)).x() <= dock.mapTo(window, QPoint(0, 0)).x()
    assert not window.process_runner.active


def test_original_view_close_toggle_and_navigation_survive_inline_analysis(native_window, qtbot):
    window = native_window
    dock, panel = window.streaming_dock, window.streaming_panel
    window._select_row(("comparison", "events"))
    qtbot.waitUntil(dock.isVisible)
    dock.close()
    assert not dock.isVisible()
    window.menu_builder.streaming_toggle_action.trigger()
    assert dock.isVisible() and dock.widget() is panel
    window._select_row(("start", "series"))
    assert window.dockWidgetArea(dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert dock.widget() is panel and dock.isVisible()
    window._select_row(("comparison", "wasserstein_compute"))
    qtbot.waitUntil(lambda: window.analysis_sections._dock_restore is not None)
    window.settings_factory.display_full_settings()
    assert window.dockWidgetArea(dock) == Qt.DockWidgetArea.RightDockWidgetArea
    QApplication.processEvents()
    assert window.dockWidgetArea(dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert dock.widget() is panel and panel.isVisible()
    window._select_row(("comparison", "events"))
    window._select_row(("legacy", "workspace"))
    assert window.dockWidgetArea(dock) == Qt.DockWidgetArea.RightDockWidgetArea
    legacy = window.legacy_workspace.legacy_window
    legacy.settings_factory.display_settings("comparison", ["events"])
    assert isinstance(legacy.settings_layout.itemAt(0).widget(), QTabWidget)
    assert legacy.streaming_panel is not panel


def test_inline_buttons_dispatch_original_comparison_run_and_stop(native_window, monkeypatch):
    window = native_window
    window._select_row(("comparison", "events"))
    calls = []
    monkeypatch.setattr(window.comparison_tab, "on_run_clicked", lambda: calls.append(("run", window.selected_checkbox_id)))
    monkeypatch.setattr(window.comparison_tab, "on_abort_clicked", lambda: calls.append(("stop", window.selected_checkbox_id)))
    for action in (window.toolbar_builder.play_action, window.toolbar_builder.stop_action):
        action.setEnabled(True)
        button = next(b for b in window.analysis_sections.actions_section.findChildren(QToolButton)
                      if b.defaultAction() is action)
        button.click()
    assert calls == [("run", "events"), ("stop", "events")]
    assert not window.process_runner.active
