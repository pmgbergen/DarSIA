"""Embedded shared-workbook staging beside the original Protocols settings."""
from pathlib import Path
import traceback
import tomllib

import toml
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .protocol_workbooks import (
    activate_protocol_workbooks, plan_protocol_workbooks, write_protocol_workbooks,
)


class _Result(QObject):
    ready = Signal(object, str)


class _Call(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function, self.signals = function, _Result()

    def run(self):
        try:
            value, error = self.function(), ""
        except Exception:
            value, error = None, traceback.format_exc()
        self.signals.ready.emit(value, error)


class ProtocolWorkbooksPanel(QWidget):
    """Preview, create and explicitly activate shared protocol workbooks."""
    activated = Signal(str)
    archived = Signal(str)
    def __init__(self, window):
        super().__init__()
        self.window, self.plan, self.package, self._call = window, None, None, None
        layout = QVBoxLayout(self)
        description = QLabel(
            "Shared protocol workbooks: one worksheet per run. Preview reads only "
            "the saved protocol tables. Creating workbooks preserves the source files; "
            "the active configuration, maps and approvals are not switched.")
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self.destination = QLineEdit()
        form.addRow("New workbook package folder", self.destination)
        self.overflow_list = QLineEdit()
        self.overflow_source_root = QLineEdit()
        form.addRow("Overflow source list (optional)", self.overflow_list)
        form.addRow("Overflow source folder (if moved)", self.overflow_source_root)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.preview_button = QPushButton("Preview protocol workbooks")
        self.create_button = QPushButton("Create workbooks")
        self.activate_button = QPushButton("Activate workbooks")
        self.archive_button = QPushButton("Archive source protocol folders")
        self.back_button = QPushButton("Back to Protocols")
        for button in (self.preview_button, self.create_button, self.activate_button, self.back_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        layout.addWidget(self.archive_button)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Protocol", "Run", "Rows"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        self.run_choice = QComboBox()
        self.proposed = QPlainTextEdit()
        self.proposed.setReadOnly(True)
        self.proposed.setMaximumHeight(160)
        layout.addWidget(self.run_choice)
        layout.addWidget(self.proposed)
        self.status = QLabel("Preview the current series' saved protocols.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.preview_button.clicked.connect(self.preview)
        self.create_button.clicked.connect(self.create)
        self.activate_button.clicked.connect(self.activate)
        self.archive_button.clicked.connect(self.archive)
        self.back_button.clicked.connect(lambda: window.content_stack.setCurrentWidget(window.settings_scroll_area))
        self.run_choice.currentTextChanged.connect(self._show_proposed)
        self._controls()

    def show_series(self, root):
        previous = getattr(self, "root", None)
        self.root = Path(root) if root else None
        if self.root != previous:
            self.overflow_list.clear()
            self.overflow_source_root.clear()
        self.plan = self.package = None
        self.table.setRowCount(0)
        self.run_choice.clear()
        self.proposed.clear()
        self.destination.setText(str(self.root / "protocols" / "workbooks") if self.root else "")
        self.status.setText("Preview the current series' saved protocols." if root else "Choose a series on Start first.")
        self._controls()

    def _controls(self):
        idle = self._call is None
        self.preview_button.setEnabled(idle and getattr(self, "root", None) is not None)
        self.create_button.setEnabled(idle and self.plan is not None and self.package is None)
        self.activate_button.setEnabled(idle and self.package is not None and not self.package.get("activated"))
        self.archive_button.setEnabled(idle and self.package is not None and self.package.get("activated", False)
                                       and not self.package.get("archived"))
        self.destination.setEnabled(idle and self.package is None)
        self.overflow_list.setEnabled(idle and self.package is None)
        self.overflow_source_root.setEnabled(idle and self.package is None)

    def _start(self, function, done):
        if self._call is not None:
            return
        call = _Call(function)
        self._call, self._done = call, done
        call.signals.ready.connect(self._finished)
        self._controls()
        QThreadPool.globalInstance().start(call)

    @Slot(object, str)
    def _finished(self, value, error):
        done = self._done
        self._call = None
        if error:
            self.status.setText(error.strip().splitlines()[-1])
            self.status.setToolTip(error)
        else:
            done(value)
        self._controls()

    def preview(self):
        root = self.root
        overflow_list, overflow_root = self.overflow_list.text().strip(), self.overflow_source_root.text().strip()
        self.plan = self.package = None
        self.status.setText("Reading saved protocol tables…")
        def read():
            if not overflow_list:
                return plan_protocol_workbooks(root)
            from .overflow_protocol_import import load_overflow_sources
            manifest = tomllib.loads((root / "series.toml").read_text(encoding="utf-8-sig"))
            sources = load_overflow_sources(overflow_list, source_root=overflow_root or None,
                                           runs=manifest["runs"]["ids"])
            return plan_protocol_workbooks(root, overflow_sources=sources)
        self._start(read, self._planned)

    def _planned(self, plan):
        self.plan = plan
        self.table.setRowCount(len(plan.sheets))
        for row, sheet in enumerate(plan.sheets):
            for col, value in enumerate((sheet.kind, sheet.run, len(sheet.rows))):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self.run_choice.clear()
        self.run_choice.addItems(list(plan.runs))
        self.status.setText(f"{plan.source_tables} source tables across {len(plan.runs)} runs. "
            "Create workbooks stages the saved data; active configurations remain unchanged.")

    def create(self):
        plan, destination = self.plan, self.destination.text()
        if plan is None or not destination.strip():
            return
        self.status.setText("Writing the new shared protocol workbooks…")
        self._start(lambda: write_protocol_workbooks(plan, destination), self._written)

    def _written(self, package):
        self.package = package
        self._show_proposed(self.run_choice.currentText())
        self.status.setText(f"Created {len(package['files'])} workbooks in {package['folder']}. "
            "The active protocol files have not been switched. Activate workbooks updates "
            "the saved run configurations; original files and settings are retained.")

    def activate(self):
        if self.plan is None or self.package is None or self.package.get("activated"):
            return
        plan, package = self.plan, self.package
        self.status.setText("Updating the saved protocol file and worksheet references…")
        self._start(lambda: activate_protocol_workbooks(plan, package), self._activated)

    def _activated(self, result):
        self.status.setText(f"Shared protocol workbooks are active for {len(self.plan.runs)} runs. "
            "Only protocol file and worksheet references changed. Original configuration "
            "bytes are retained in the workbook package's migration record.")
        self.activated.emit(str(self.root))

    def archive(self):
        if self.plan is None or self.package is None or not self.package.get("activated") or self.package.get("archived"):
            return
        from .protocol_workbook_archive import archive_protocol_sources
        plan, package = self.plan, self.package
        self.status.setText("Archiving old protocol folders; active workbooks stay in place…")
        self._start(lambda: archive_protocol_sources(plan, package), self._archived)

    def _archived(self, result):
        self.status.setText(f"Archived {len(result['moves'])} old protocol folders in {result['folder']}. "
            "The shared workbooks stay active. Original references follow the saved folder aliases.")
        self.archived.emit(str(self.root))

    def _show_proposed(self, run):
        section = self.package["protocols"].get(run) if self.package and run else None
        self.proposed.setPlainText(toml.dumps({"protocols": section}) if section else "")


def install_protocol_workbooks(window):
    """Keep the conversion panel in the window; Setup > Protocols > Protocol workbooks opens it."""
    panel = ProtocolWorkbooksPanel(window)
    window.content_stack.addWidget(panel)
    window.protocol_workbooks_panel = panel
    return panel


def open_protocol_workbooks(window):
    """Show the conversion panel for the selected series, keeping a conversion that runs."""
    panel = window.protocol_workbooks_panel
    if panel._call is not None:
        window.content_stack.setCurrentWidget(panel)
        return panel
    root = window._method_series_root()
    if getattr(panel, "root", None) != (Path(root) if root else None) or panel.plan is None:
        panel.show_series(root)
    window.content_stack.setCurrentWidget(panel)
    return panel
