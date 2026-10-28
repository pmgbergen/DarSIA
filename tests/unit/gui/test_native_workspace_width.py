import pytest
from PySide6.QtCore import QPoint, QSettings
from PySide6.QtWidgets import QLabel

from darsia.fluidflower.workflow_section import WorkflowSection, WorkflowWorkspace
from .test_native_sidebar import native_window


def test_sidebar_and_workflow_stay_adjacent_when_window_grows(native_window, qtbot):
    window = native_window
    workspace = WorkflowWorkspace()
    workspace.controls_layout.addWidget(WorkflowSection(1, "Basis", QLabel("Existing model")))
    window.content_stack.addWidget(workspace)
    window.content_stack.setCurrentWidget(workspace)
    window.resize(1440, 900)
    working_widths = []
    for width in (1440, 5119, 1440):
        window.resize(width, 900)
        qtbot.waitUntil(lambda: window.width() == width)
        qtbot.waitUntil(lambda: workspace.width() > 0)
        assert not window.grab().isNull()
        assert 360 <= window.sidebar.width() <= 600
        sidebar_end = window.sidebar.mapTo(window, QPoint(window.sidebar.width(), 0)).x()
        controls_start = workspace.controls_scroll.mapTo(window, QPoint()).x()
        assert 0 <= controls_start - sidebar_end <= 40
        assert 360 <= workspace.controls_scroll.width() <= 720
        working_widths.append(workspace.workspace.width())
    assert working_widths[1] > working_widths[0] + 3000


def test_workflow_control_column_has_a_readable_upper_bound(qtbot):
    workspace = WorkflowWorkspace()
    qtbot.addWidget(workspace)
    workspace.show()
    workspace.resize(5119, 900)
    qtbot.waitUntil(lambda: workspace.width() == 5119)
    workspace.setSizes([2000, 3119])
    assert workspace.controls_scroll.width() <= 720
    assert workspace.workspace.width() > 4000


@pytest.mark.parametrize(("screen_width", "saved_width"), [
    (1440, None), (5119, None), (5119, 420), (1440, 618),
])
def test_initial_show_restores_readable_sidebar_without_overriding_later_resize(
    native_window, qtbot, monkeypatch, screen_width, saved_width
):
    from darsia.fluidflower import native_darsia_app as native

    native_window.close()
    settings = QSettings()
    key = native.MainWindow._SIDEBAR_WIDTH_SETTINGS_KEY
    if saved_width is None:
        settings.remove(key)
    else:
        settings.setValue(key, saved_width)

    def show_at_restored_size(window):
        window.show()
        window.resize(screen_width, 900)

    monkeypatch.setattr(native.MainWindow, "showMaximized", show_at_restored_size)
    window = native.NativeFluidFlowerMainWindow()
    qtbot.addWidget(window)
    qtbot.waitUntil(lambda: window._native_sidebar_initialized and not window._series_loading)
    assert 360 <= window.sidebar.width() <= 600
    if saved_width is not None:
        assert window.root_splitter.sizes()[0] == saved_width
    elif screen_width > 5000:
        assert window.sidebar.width() == 600

    window.root_splitter.moveSplitter(300, 1)
    chosen = window.root_splitter.sizes()[0]
    window.resize(screen_width + 80, 900)
    qtbot.waitUntil(lambda: window.width() == screen_width + 80)
    assert window.root_splitter.sizes()[0] == chosen == 300
    assert settings.value(key, type=int) == chosen
    window.close()
