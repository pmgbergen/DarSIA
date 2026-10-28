"""Keep native Analysis controls beside their original result viewer."""

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QHBoxLayout, QTabWidget, QToolButton, QWidget

from .workflow_section import WorkflowSection, WorkflowWorkspace


class NativeAnalysisSections:
    """Reparent original widgets without replacing save, run or result handling."""

    def __init__(self, window):
        self.window = window
        self.sections = []
        self.workspace = None
        self._folds = {}
        self._dock_restore = None
        factory = window.settings_factory
        display = factory.display_settings
        display_full = factory.display_full_settings

        def display_settings(action, checked_ids):
            # The factory deletes the previous settings widgets. Return the
            # original result dock first so it cannot be deleted with them.
            self._restore_view()
            self.workspace = None
            result = display(action, checked_ids)
            self.sections = []
            if action in {"analysis", "comparison"}:
                self._arrange(action, tuple(checked_ids))
            return result

        def display_full_settings():
            self._restore_view()
            self.workspace = None
            self.sections = []
            return display_full()

        display_settings.__wrapped__ = display
        display_full_settings.__wrapped__ = display_full
        factory.display_settings = display_settings
        factory.display_full_settings = display_full_settings
        window.content_stack.currentChanged.connect(self._page_changed)
        window.streaming_dock.visibilityChanged.connect(self._view_visibility_changed)

    def _view_visibility_changed(self, visible):
        # Explicit Show View/Close choices still apply when the dock returns.
        # Hiding its parent page is not a user choice to close the viewer.
        if self._dock_restore is not None:
            dock = self.window.streaming_dock
            if visible or dock.isHidden():
                area, _ = self._dock_restore
                self._dock_restore = (area, visible)

    def _embed_view(self):
        if self._dock_restore is not None or self.workspace is None:
            return
        window = self.window
        dock = window.streaming_dock
        self._dock_restore = (window.dockWidgetArea(dock), not dock.isHidden())
        with QSignalBlocker(dock):
            window.removeDockWidget(dock)
            placeholder = self.workspace.workspace
            self.workspace.replaceWidget(1, dock)
            self.workspace.workspace = dock
            placeholder.deleteLater()
            dock.show()
        window._sync_streaming_toggle_button(True)

    def _restore_view(self):
        if self._dock_restore is None:
            return
        window = self.window
        area, visible = self._dock_restore
        self._dock_restore = None
        dock = window.streaming_dock
        with QSignalBlocker(dock):
            # Leave a harmless workspace in any retained, currently hidden
            # Analysis page. Returning through the sidebar rebuilds its forms.
            placeholder = QWidget()
            self.workspace.replaceWidget(1, placeholder)
            self.workspace.workspace = placeholder
            dock.setParent(window)
            window.addDockWidget(area, dock)
            dock.setVisible(visible)
        window._sync_streaming_toggle_button(visible)

    def _page_changed(self, _index):
        if self.window.content_stack.currentWidget() is not self.window.settings_scroll_area:
            self._restore_view()
        elif self.workspace is not None:
            self._embed_view()

    def _arrange(self, action, checked_ids):
        window = self.window
        layout = window.settings_layout
        first = layout.itemAt(0)
        tabs = first.widget() if first is not None else None
        if not isinstance(tabs, QTabWidget):
            return
        key = (str(window.config_file), action, checked_ids)
        layout.removeWidget(tabs)
        # The original factory ends with a stretch; the two-column workspace
        # must instead fill the available height, including its result viewer.
        while layout.count():
            layout.takeAt(0)
        self.workspace = WorkflowWorkspace()
        while tabs.count():
            title, content = tabs.tabText(0), tabs.widget(0)
            tabs.removeTab(0)
            section = WorkflowSection(
                len(self.sections) + 1, title, content,
                expanded=self._folds.get((key, title), True))
            section.toggle.toggled.connect(
                lambda expanded, title=title: self._folds.__setitem__((key, title), expanded))
            self.workspace.controls_layout.addWidget(section)
            self.sections.append(section)
        tabs.deleteLater()
        actions = QWidget()
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        for original in (window.menu_builder.save_action,
                         window.toolbar_builder.play_action,
                         window.toolbar_builder.stop_action):
            button = QToolButton()
            button.setDefaultAction(original)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            row.addWidget(button)
        row.addStretch(1)
        self.actions_section = WorkflowSection(len(self.sections) + 1, "Actions", actions)
        self.workspace.controls_layout.addWidget(self.actions_section)
        layout.addWidget(self.workspace, 1)
        self._embed_view()
