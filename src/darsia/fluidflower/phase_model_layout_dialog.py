"""A read-only preview followed by an explicit phase-model storage copy."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QTableView, QVBoxLayout)


class _Files(QAbstractTableModel):
    columns = ("Original", "Destination", "Bytes", "SHA-256", "Action")

    def __init__(self, rows=(), parent=None, *, series_root=None):
        """Retain the preview rows and their optional display-path origin."""
        super().__init__(parent)
        self.rows = list(rows)
        self.series_root = Path(series_root) if series_root is not None else None

    def rowCount(self, parent=QModelIndex()):
        """Expose only top-level file rows."""
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        """Expose the fixed preview columns for top-level rows."""
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        """Label horizontal preview columns and defer other header roles."""
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section]
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Show compact paths while retaining complete identities in tooltips."""
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        row = self.rows[index.row()]
        values = [row["source"]["path"], row["destination"], str(row["source"]["size_bytes"]),
                  row["source"]["sha256"], row["action"]]
        if role == Qt.ItemDataRole.DisplayRole and index.column() < 2 and self.series_root is not None:
            try:
                return Path(values[index.column()]).relative_to(self.series_root).as_posix()
            except ValueError:
                pass
        return values[index.column()]


class PhaseModelLayoutDialog(QDialog):
    """Capture one series; filesystem reads and copies run outside the GUI thread."""
    locations_changed = Signal()

    def __init__(self, series_root, parent=None, *, planner=None, copier=None):
        """Build the explicit preview, quiescence and copy controls."""
        super().__init__(parent)
        self.series_root = Path(series_root)
        self._planner, self._copier = planner, copier
        self._thread = self._worker = None
        self._operation = None
        self.plan = None
        self.sessions = []
        self.setWindowTitle("Organise phase-model files for this series")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.resize(1080, 650)
        layout = QVBoxLayout(self)
        self.context = QLabel("Series: " + str(self.series_root))
        self.context.setTextFormat(Qt.TextFormat.PlainText)
        self.context.setWordWrap(True)
        layout.addWidget(self.context)
        self.note = QLabel("Preview the exact files first. Copying preserves every original, model version, review and withdrawal; it does not choose a model for use.")
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.table = QTableView()
        self.table.setModel(_Files(parent=self.table))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table.setToolTip("Paths within the series are relative to the series folder above. Hover a cell for its full path or SHA-256.")
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        self.summary = QLabel("No preview yet.")
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.quiescent = QCheckBox("All other review windows and evidence-writing jobs are closed")
        self.quiescent.setToolTip("Required before copying. This tool cannot coordinate a delayed client that retained an old storage path.")
        self.quiescent.toggled.connect(lambda _checked: self._controls(self._thread is None))
        layout.addWidget(self.quiescent)
        row = QHBoxLayout()
        self.add_session = QPushButton("Add an external session for this series…")
        self.add_session.clicked.connect(self._choose_session)
        self.preview = QPushButton("Preview files and locations")
        self.preview.clicked.connect(self.preview_files)
        self.copy = QPushButton("Copy and update locations")
        self.copy.setEnabled(False)
        self.copy.clicked.connect(self.copy_files)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        for button in (self.add_session, self.preview, self.copy, self.close_button):
            row.addWidget(button)
        layout.addLayout(row)

    def _choose_session(self):
        """Add a chosen external session and invalidate the previous preview."""
        path = QFileDialog.getExistingDirectory(self, "Choose the folder containing review_session.json")
        if path and path not in self.sessions:
            self.sessions.append(path)
            self.plan = None
            self.copy.setEnabled(False)
            self.summary.setText("External session added. Preview again to verify its selected-series scope and exact files.")

    def _start(self, operation, function):
        """Run one filesystem operation outside the GUI thread."""
        from .jobs_panel import _CallWorker
        if self._thread is not None:
            return
        self._operation = operation
        self._thread = QThread(self)
        self._worker = _CallWorker(function)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._ready)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._closed)
        self._thread.finished.connect(self._thread.deleteLater)
        self._controls(False)
        self._thread.start()

    def _controls(self, enabled):
        """Enable copying only for a preview with confirmed writer quiescence."""
        self.add_session.setEnabled(enabled)
        self.preview.setEnabled(enabled)
        self.quiescent.setEnabled(enabled)
        self.copy.setEnabled(enabled and self.plan is not None and self.quiescent.isChecked())

    def preview_files(self):
        """Read exact source identities without copying any files."""
        from .phase_model_layout import preview_layout
        self.plan = None
        self.quiescent.setChecked(False)
        self.summary.setText("Reading the original file list and hashes. No files are being copied.")
        planner = self._planner or preview_layout
        sessions = tuple(self.sessions)
        self._start("preview", lambda: planner(self.series_root, external_sessions=sessions))

    def copy_files(self):
        """Apply the reviewed plan after explicit writer-quiescence confirmation."""
        from .phase_model_layout import apply_layout
        if self.plan is None:
            self.summary.setText("Preview the exact files before copying.")
            return
        if not self.quiescent.isChecked():
            self.summary.setText("Close all other review windows and evidence-writing jobs, then confirm that they are closed before copying.")
            return
        copier = self._copier or apply_layout
        plan = self.plan
        self.summary.setText("Copying the verified plan. Keep this window open until it finishes.")
        self._start("copy", lambda: copier(plan, writers_quiescent=True))

    @Slot(object)
    def _ready(self, value):
        """Display the preview or announce a verified storage cutover."""
        if self._operation == "preview":
            self.plan = value
            self.table.setModel(_Files(value["files"], self.table, series_root=self.series_root))
            paths = "; ".join(key + " → " + path for key, path in value["paths"].items())
            self.summary.setText(str(len(value["files"])) + " files; " + str(value["bytes_to_copy"])
                + " bytes to copy.\nLocations: " + paths
                + ("\n" + "\n".join(value.get("notices") or []) if value.get("notices") else ""))
        else:
            self.summary.setText("Copied and verified. Original files remain in place, and the saved model choice and withdrawal policy are unchanged.")
            self.plan = None
            self.quiescent.setChecked(False)
            self.locations_changed.emit()

    @Slot(str)
    def _failed(self, error):
        """Explain the failure while retaining originals and partial copies."""
        self.plan = None
        self.quiescent.setChecked(False)
        # _CallWorker emits a traceback; show its final exception in plain words.
        lines = str(error).strip().splitlines()
        reason = lines[-1] if lines else "The storage operation could not finish"
        self.summary.setText(reason + ". Existing files and partial copies are preserved. Review the retained files before preparing another copy.")

    @Slot()
    def _closed(self):
        """Release the finished worker and restore the available controls."""
        if self._thread is not None:
            self._thread.wait()
        self._thread = self._worker = None
        self._operation = None
        self._controls(True)

    def can_close(self):
        """Report whether all filesystem work has finished."""
        return self._thread is None

    def reject(self):
        """Defer dismissal until the active filesystem operation returns."""
        if self.can_close():
            super().reject()
        else:
            self.summary.setText("The current file operation is still finishing. Close after it returns.")

    def closeEvent(self, event):
        """Keep the dialog open while a valid operation is still finishing."""
        if not self.can_close():
            self.summary.setText("The current file operation is still finishing. Close after it returns.")
            event.ignore()
            return
        super().closeEvent(event)
