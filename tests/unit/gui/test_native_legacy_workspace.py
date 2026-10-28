"""Original managers, configuration isolation and owned-process lifetime."""

import sys
import tomllib

import pytest
from PySide6.QtCore import QProcess, QThread, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QToolBar

from .test_native_sidebar import native_window  # noqa: F401


def _open(window):
    window._select_row(("legacy", "workspace"))
    return window.legacy_workspace, window.legacy_workspace.legacy_window


def test_original_categories_are_embedded_lazily_without_loading_or_running(
    native_window, monkeypatch,
):
    from darsia.gui.ui.main_window import MainWindow
    from darsia.gui.ui.process_runner import ProcessRunner

    host = native_window
    assert host.legacy_workspace is None
    top_levels = set(QApplication.topLevelWidgets())
    monkeypatch.setattr(ProcessRunner, "start_workflow_process", lambda *a, **k: pytest.fail("Unexpected launch"))
    host.config_file = "series-config.toml"
    host.config_dict = {"series": {"name": "separate"}}
    page, legacy = _open(host)
    assert isinstance(legacy, MainWindow)
    assert not legacy.isWindow() and legacy.window() is host
    assert not any(widget.isVisible() for widget in set(QApplication.topLevelWidgets()) - top_levels)
    assert host.content_stack.currentWidget() is page
    assert list(legacy.sidebar.categories_data) == [
        "preprocessing", "setup", "calibration", "analysis", "helper", "comparison", "utils"
    ]
    for key, manager in legacy.action_dispatch.items():
        assert type(manager) is type(host.action_dispatch[key])
        assert legacy.sidebar.categories_data[key][2] == manager.sidebar_items()
    assert legacy.config_file == "" and legacy.config_dict == {}
    assert legacy.config_dict is not host.config_dict
    assert legacy.process_runner is not host.process_runner
    assert legacy.settings_factory.main_window is legacy
    assert legacy.config_controller.main_window is legacy
    assert legacy.run_selected_workflow.__func__ is MainWindow.run_selected_workflow
    assert not legacy.process_runner.active
    assert not host.series_header.isVisible() and not host.method_badge.isVisible()
    host._refresh_scope_display()
    assert not host.series_header.isVisible()


def test_save_and_run_shortcuts_follow_the_visible_workspace(native_window, qtbot):
    host = native_window
    host.activateWindow()
    events = []
    for name in ("save_action", "play_action"):
        action = getattr(host.menu_builder, name)
        action.triggered.disconnect()
        action.triggered.connect(lambda checked=False, name=name: events.append(("series", name)))
    save_shortcuts = host.menu_builder.save_action.shortcuts()
    page, legacy = _open(host)
    qtbot.waitUntil(lambda: QApplication.activeWindow() is host)
    qtbot.waitUntil(legacy.isVisible)
    for name in ("save_action", "play_action"):
        action = getattr(legacy.menu_builder, name)
        action.triggered.disconnect()
        action.triggered.connect(lambda checked=False, name=name: events.append(("legacy", name)))
    assert legacy.menu_builder.save_action.shortcuts() == save_shortcuts
    assert not host.menu_builder.save_action.shortcuts()
    assert not host.menu_builder.play_action.shortcuts()
    assert host.menuBar().isHidden()
    assert all(toolbar.isHidden() for toolbar in host.findChildren(
        QToolBar, options=Qt.FindChildOption.FindDirectChildrenOnly))
    qtbot.keyClick(legacy, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(legacy, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert events == [("legacy", "save_action"), ("legacy", "play_action")]
    legacy.config_file = "legacy-config.toml"
    legacy.config_dict["legacy"] = True
    host._select_row(("start", "series"))
    assert not page.active and not page.isVisible()
    assert not legacy.menu_builder.save_action.shortcuts()
    assert not legacy.menu_builder.play_action.shortcuts()
    assert host.menu_builder.save_action.shortcuts() == save_shortcuts
    qtbot.keyClick(host, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    assert events[-1] == ("series", "save_action")
    again, same = _open(host)
    assert again is page and same is legacy
    assert same.config_file == "legacy-config.toml" and same.config_dict == {"legacy": True}
    assert legacy.config_dict is not host.config_dict


def test_navigation_and_close_preserve_live_process_and_completion_ownership(
    native_window, qtbot, monkeypatch,
):
    host = native_window
    page, legacy = _open(host)
    process = QProcess(legacy)
    completed = []
    process.finished.connect(lambda *args: completed.append(args))
    process.start(sys.executable, ["-I", "-u", "-c", "import sys; sys.stdin.buffer.read(1)"])
    qtbot.waitUntil(lambda: process.state() == QProcess.ProcessState.Running, timeout=10000)
    legacy.process_runner.active["helper"] = (process, "helper")
    cleanup = []
    monkeypatch.setattr(legacy.streaming_panel, "cleanup", lambda: cleanup.append("legacy"))
    try:
        pid = process.processId()
        host._select_row(("start", "series"))
        assert not page.active and process.processId() == pid
        assert not host.close() and not legacy.close()
        assert process.state() == QProcess.ProcessState.Running and not cleanup
        process.write(b"x")
        process.closeWriteChannel()
        qtbot.waitUntil(lambda: bool(completed), timeout=10000)
        assert not page.can_close() and not host.close()
        legacy.process_runner.active.clear()
        assert page.can_close() and host.close()
        assert cleanup == ["legacy"]
    finally:
        if process.state() != QProcess.ProcessState.NotRunning:
            process.write(b"x")
            process.closeWriteChannel()
            qtbot.waitUntil(lambda: bool(completed), timeout=10000)
        legacy.process_runner.active.clear()


def test_owned_reader_blocks_close_without_being_stopped(native_window, qtbot):
    page, legacy = _open(native_window)
    reader = QThread(legacy)
    reader.start()
    try:
        qtbot.waitUntil(reader.isRunning)
        assert not page.can_close() and not native_window.close()
        assert reader.isRunning()
    finally:
        reader.quit()
        assert reader.wait(10000)
    assert page.can_close()


@pytest.mark.parametrize("category", [
    "preprocessing", "setup", "calibration", "analysis", "helper", "comparison", "utils",
])
def test_run_and_stop_use_original_selected_manager(native_window, monkeypatch, category):
    _page, legacy = _open(native_window)
    calls = []
    manager = legacy.action_dispatch[category]
    monkeypatch.setattr(manager, "on_run_clicked", lambda: calls.append("run"))
    monkeypatch.setattr(manager, "on_abort_clicked", lambda: calls.append("stop"))
    legacy.selected_action = category
    legacy.menu_builder.play_action.trigger()
    legacy.menu_builder.stop_action.setEnabled(True)
    legacy.menu_builder.stop_action.trigger()
    assert calls == ["run", "stop"]


def test_original_new_and_save_write_only_legacy_configuration(native_window, monkeypatch, tmp_path):
    host = native_window
    host_path = tmp_path / "series.toml"
    host_path.write_text('marker = "series"\n', encoding="utf-8")
    host.config_file = str(host_path)
    host.config_dict = {"marker": "series"}
    _page, legacy = _open(host)
    legacy_path = tmp_path / "legacy.toml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(legacy_path), ""))
    legacy.menu_builder.new_action.trigger()
    assert legacy.config_file == str(legacy_path)
    assert legacy.config_path_label.text() == str(legacy_path)
    legacy.config_dict["marker"] = "legacy"
    legacy.menu_builder.save_action.trigger()
    assert tomllib.loads(legacy_path.read_text("utf-8"))["marker"] == "legacy"
    assert host_path.read_text("utf-8") == 'marker = "series"\n'
    assert host.config_file == str(host_path) and host.config_dict == {"marker": "series"}


def test_start_new_original_uses_exact_original_producer_and_only_selected_file(
        native_window, monkeypatch, tmp_path, qtbot):
    from darsia.fluidflower import series_wizard
    from darsia.fluidflower.series_method import METHOD_HEADERS
    from darsia.gui.ui.config_controller import ConfigController

    host = native_window
    project = tmp_path / "original_project"
    project.mkdir()
    selected = project / "configuration.toml"
    dialogs = []

    def save_dialog(parent, title, directory, filters):
        dialogs.append((parent, title, directory, filters))
        return str(selected), "TOML Files (*.toml)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", save_dialog)
    monkeypatch.setattr(series_wizard.NewSeriesFolderWizard, "__init__",
        lambda *args, **kwargs: pytest.fail("Original DarSIA must not create a phase series"))
    method = host.home_page.new_series_method
    assert method.isVisible()
    assert method.currentData() == "colour_path"
    assert method.currentText() == METHOD_HEADERS["colour_path"]
    assert host.legacy_workspace is None
    top_levels = set(QApplication.topLevelWidgets())
    qtbot.mouseClick(host.home_page.new_series_button, Qt.MouseButton.LeftButton)

    page, original = host.legacy_workspace, host.legacy_workspace.legacy_window
    assert original.config_controller.new_config.__func__ is ConfigController.new_config
    assert host.content_stack.currentWidget() is page
    assert original.window() is host and not original.isWindow()
    assert not any(widget.isVisible() for widget in set(QApplication.topLevelWidgets()) - top_levels)
    assert dialogs[0][0] is original
    assert original.config_file == str(selected)
    assert original.config_path_label.text() == str(selected)
    assert original.config_dict == {}
    assert list(project.iterdir()) == [selected]
    assert tomllib.loads(selected.read_text()) == {}

    # Compare with a direct invocation of the unchanged original producer.
    expected = tmp_path / "direct_original.toml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(expected), ""))
    original.config_controller.new_config()
    assert selected.read_bytes() == expected.read_bytes() == b""
    assert list(project.iterdir()) == [selected]


def test_legacy_can_open_while_series_metadata_is_pending(native_window):
    host = native_window
    host._series_loading = True
    host._pending_open_series = "pending"
    host._pending_series_navigation = ("setup", "all")
    page, _legacy = _open(host)
    assert page.active and host.content_stack.currentWidget() is page
    assert host._pending_open_series is None and host._pending_series_navigation is None
    host._series_loading = False
