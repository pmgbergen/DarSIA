"""Embed the original DarSIA interface with its own configuration and runner."""

from PySide6.QtCore import QProcess, QThread, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel, QToolBar, QVBoxLayout, QWidget
from shiboken6 import isValid

from darsia.gui.ui.main_window import MainWindow


class _EmbeddedLegacyWindow(MainWindow):
    def __init__(self, workspace):
        self.workspace = workspace
        super().__init__()
        self.setParent(workspace, Qt.WindowType.Widget)

    def showMaximized(self):
        # MainWindow calls this during construction, before it can be parented.
        # The enclosing layout controls its size and visibility instead.
        pass

    def close(self):
        if not self.workspace._closing_with_host:
            return self.workspace.host.close()
        return super().close()

    def closeEvent(self, event):
        if not self.workspace.can_close():
            self.print_log(self.workspace.close_block_reason())
            event.ignore()
            return
        super().closeEvent(event)


class NativeLegacyWorkspace(QWidget):
    """One persistent original window; navigation never stops its work."""

    _CONTEXT_WIDGETS = (
        "series_header", "method_badge", "method_panel", "analysis_method_badge",
        "analysis_execution_note", "frozen_reference_line", "scope_label",
        "rig_label", "status_header",
    )

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.active = False
        self._closing_with_host = False
        self._outer_shortcuts = []
        self._inner_shortcuts = []
        self._outer_visibility = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        note = QLabel(
            "Legacy workflow · Open a configuration here using File → Open Config. "
            "This configuration is separate from the series workspace."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.legacy_window = _EmbeddedLegacyWindow(self)
        # The original sidebar shares the available width with the outer
        # navigation here; keep its original category and step labels readable.
        self.legacy_window.sidebar.setMinimumWidth(220)
        self.legacy_window.root_splitter.setSizes([240, 1000])
        self.legacy_window.config_path_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.legacy_window.config_path_label)
        layout.addWidget(self.legacy_window, 1)
        self._inner_shortcuts = self._suspend_shortcuts(self.legacy_window.findChildren(QAction))

    @staticmethod
    def _suspend_shortcuts(actions):
        saved = [(action, action.shortcuts()) for action in actions if action.shortcuts()]
        for action, _shortcuts in saved:
            action.setShortcuts([])
        return saved

    @staticmethod
    def _restore_shortcuts(saved):
        for action, shortcuts in saved:
            if isValid(action):
                action.setShortcuts(shortcuts)

    def _is_ours(self, obj):
        while obj is not None:
            if obj is self:
                return True
            obj = obj.parent()
        return False

    def hide_series_context(self):
        for name in self._CONTEXT_WIDGETS:
            widget = getattr(self.host, name, None)
            if widget is not None:
                widget.hide()

    def set_active(self, active):
        if active == self.active:
            return
        self.active = active
        if active:
            self._outer_shortcuts = self._suspend_shortcuts(
                action for action in self.host.findChildren(QAction) if not self._is_ours(action)
            )
            self._restore_shortcuts(self._inner_shortcuts)
            chrome = [self.host.menuBar(), self.host.statusBar(), *self.host.findChildren(
                QToolBar, options=Qt.FindChildOption.FindDirectChildrenOnly)]
            chrome.extend(getattr(self.host, name) for name in (
                "log_dock", "streaming_dock", "log_toggle_button", "streaming_toggle_button"
            ))
            self._outer_visibility = [(widget, not widget.isHidden()) for widget in chrome]
            for widget, _visible in self._outer_visibility:
                widget.hide()
            self.hide_series_context()
        else:
            self._inner_shortcuts = self._suspend_shortcuts(self.legacy_window.findChildren(QAction))
            self._restore_shortcuts(self._outer_shortcuts)
            for widget, visible in self._outer_visibility:
                widget.setVisible(visible)
            self._outer_visibility = []

    def can_close(self):
        # Keep ownership through the runner's completion callback, including
        # the short interval after the OS child exits but before cleanup.
        return not (
            self.legacy_window.process_runner.active
            or any(process.state() != QProcess.ProcessState.NotRunning
                   for process in self.legacy_window.findChildren(QProcess))
            or any(reader.isRunning() for reader in self.legacy_window.findChildren(QThread))
        )

    @staticmethod
    def close_block_reason():
        return (
            "Legacy workflow is still running or finishing. Return to Legacy workflow "
            "to follow it, and close after the process or reader returns."
        )

    def finish_close(self):
        self.set_active(False)
        self._closing_with_host = True
        try:
            self.legacy_window.close()
        finally:
            self._closing_with_host = False
