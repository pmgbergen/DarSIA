"""Bounded, read-only model dependency inspection; deletion stays unavailable."""
from __future__ import annotations

from contextvars import copy_context
from copy import deepcopy
from pathlib import Path
import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QTextEdit, QVBoxLayout,
)

KINDS = (
    ("Training request", "training_request"),
    ("Model library reference", "library_reference"),
    ("Review session", "review_session"),
    ("Review source binding", "review_source_binding"),
    ("Series manifest", "series_manifest"),
    ("Studio mass result", "studio_mass_result"),
    ("Studio mass receipt", "studio_mass_receipt"),
    ("Job input catalogue", "job_input_catalog"),
    ("Job registry", "job_registry"),
    ("Model ancestry receipt", "model_ancestry"),
    ("Result source manifest", "result_source_manifest"),
)


def known_dependency_scope(series_root, model_history, review_sessions):
    """Describe known original metadata paths without enumerating directories."""
    series_root = str(series_root)
    rows = [dict(kind="series_manifest", path=str(Path(series_root) / "series.toml"),
                 series_root=series_root)]
    for entry in model_history:
        if entry.get("root"):
            rows.append(dict(kind="training_request", path=str(Path(entry["root"]) / "training_request.json")))
        if entry.get("library_reference"):
            rows.append(dict(kind="library_reference", path=str(entry["library_reference"]),
                             series_root=series_root))
    for session in review_sessions:
        if session.get("session_path"):
            rows.append(dict(kind="review_session", path=str(session["session_path"])))
    unique = {}
    for row in rows:
        unique.setdefault((row["kind"], row["path"]), row)
    return list(unique.values())


def report_text(report):
    """Never turn a limited negative search into a deletion eligibility claim."""
    refs, unknown = report.get("refs") or [], report.get("unknown") or []
    lines = ["Known model references" if refs else "No reference found in supplied scope; completeness unknown."]
    for row in refs:
        lines.append(f"• {row.get('relation') or row.get('kind') or 'Reference'}: {row.get('path') or '(in-memory reference)'}")
        if row.get("detail"):
            lines.append("  " + str(row["detail"]))
    if unknown:
        lines.extend(("", "Unresolved checks"))
        for row in unknown:
            lines.append(f"• {row.get('path') or row.get('kind') or 'Scope'}: {row.get('reason') or 'Unverified'}")
    lines.extend(("", "This supplied scope does not establish the complete set of kept results.",
                  str(report.get("deletion_reason") or "Deletion is unavailable and has not been authorized.")))
    return "\n".join(lines)


class _Signals(QObject):
    finished = Signal(object, object, str)


class _DependencyTask(QRunnable):
    def __init__(self, entry, scope, series_root, protected_roots):
        super().__init__()
        self.signals = _Signals()
        self.context = copy_context()
        self.cancelled = threading.Event()
        self.entry, self.scope = deepcopy(entry), deepcopy(scope)
        self.series_root, self.protected_roots = series_root, tuple(protected_roots)

    def run(self):
        self.context.run(self._read)

    def _read(self):
        try:
            from ..native_phase_model_dependencies import inspect_model_dependencies
            report = inspect_model_dependencies(self.entry, scope_records=self.scope,
                series_root=self.series_root, protected_roots=self.protected_roots,
                cancelled=self.cancelled.is_set)
            self.signals.finished.emit(self, report, "")
        except Exception as error:
            self.signals.finished.emit(self, {}, f"{type(error).__name__}: {error}")


class ModelDependencyDialog(QDialog):
    """One finite metadata reader, retained until its actual worker retires."""
    def __init__(self, parent, *, entry, series_root, scope_records, protected_roots=()):
        super().__init__(parent)
        self.entry, self.series_root = deepcopy(entry), str(series_root)
        self.protected_roots = tuple(protected_roots)
        self.task = None
        self.completion = None
        self.close_requested = False
        self.report = None
        self.launch_error = ""
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.retirement = QTimer(self)
        self.retirement.setInterval(50)
        self.retirement.timeout.connect(self._retire)
        self.setWindowTitle("Model dependencies · " + str(entry.get("model_id") or "Selected model"))
        self.resize(760, 620)
        layout = QVBoxLayout(self)
        notice = QLabel("Inspect references to this exact saved model in the listed metadata. "
                        "Add explicit kept-result metadata if needed. This check does not scan folders or modify files.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.scope = QListWidget()
        self.scope.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.scope)
        for row in scope_records:
            self._add_scope(row)
        actions = QHBoxLayout()
        self.kind = QComboBox()
        for label, key in KINDS:
            self.kind.addItem(label, key)
        self.add = QPushButton("Add metadata files…")
        self.add.clicked.connect(self._choose_files)
        self.remove = QPushButton("Remove selected from this check")
        self.remove.clicked.connect(self._remove_scope)
        for widget in (self.kind, self.add, self.remove):
            actions.addWidget(widget)
        layout.addLayout(actions)
        self.inspect = QPushButton("Inspect listed metadata")
        self.inspect.clicked.connect(self.request)
        layout.addWidget(self.inspect)
        self.status = QLabel("No dependency read has been requested. Deletion remains unavailable.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        layout.addWidget(self.results)
        self.delete_button = QPushButton("Delete model · unavailable")
        self.delete_button.setEnabled(False)
        self.delete_button.setToolTip("Deletion requires a separate explicit approval and a complete verified kept-result inventory. "
                                      "This inspector never deletes model files or validation boxes.")
        layout.addWidget(self.delete_button)

    def _add_scope(self, row):
        if any(self.scope.item(index).data(Qt.ItemDataRole.UserRole) == row for index in range(self.scope.count())):
            return
        item = QListWidgetItem(f"{row['kind']}: {row['path']}")
        item.setToolTip(str(row["path"]))
        item.setData(Qt.ItemDataRole.UserRole, deepcopy(row))
        self.scope.addItem(item)
        self._scope_changed()

    def _scope_changed(self):
        self.report = None
        if hasattr(self, "results"):
            self.results.clear()
            self.status.setText("The supplied scope changed. Inspect the listed metadata to update its references.")

    def _choose_files(self):
        if not self.can_close() or self.close_requested:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose explicit saved metadata", self.series_root,
                                               "Metadata (*.json *.toml)")
        for path in paths:
            self._add_scope(dict(kind=self.kind.currentData(), path=path, series_root=self.series_root))

    def _remove_scope(self):
        if self.can_close() and not self.close_requested:
            selected = self.scope.selectedItems()
            for item in selected:
                self.scope.takeItem(self.scope.row(item))
            if selected:
                self._scope_changed()

    def _controls(self):
        enabled = self.can_close() and not self.close_requested
        for widget in (self.kind, self.add, self.remove, self.inspect, self.scope):
            widget.setEnabled(enabled)
        self.delete_button.setEnabled(False)

    def request(self):
        if not self.can_close() or self.close_requested:
            return
        scope = [self.scope.item(index).data(Qt.ItemDataRole.UserRole) for index in range(self.scope.count())]
        self.task = _DependencyTask(self.entry, scope, self.series_root, self.protected_roots)
        self.task.signals.finished.connect(self._finished, Qt.ConnectionType.QueuedConnection)
        self.report = None
        self.launch_error = ""
        self.results.clear()
        self.status.setText("Reading only the listed metadata; model arrays and results remain unchanged…")
        self._controls()
        try:
            self.pool.start(self.task)
        except Exception as error:
            self.task.cancelled.set()
            self.launch_error = "Dependency read could not start: " + str(error)
            self.completion = {}, self.launch_error
            self.retirement.start()
            self._controls()
            self.status.setText(self.launch_error)
            self._retire()

    @Slot(object, object, str)
    def _finished(self, task, report, error):
        if task is not self.task:
            return
        self.completion = ({}, self.launch_error) if self.launch_error else (report, error)
        self.retirement.start()
        self._retire()

    @Slot()
    def _retire(self):
        if self.completion is None or self.pool.activeThreadCount():
            return
        report, error = self.completion
        self.task = self.completion = None
        self.retirement.stop()
        self._controls()
        if self.close_requested:
            self.close()
            return
        if error:
            self.status.setText("Dependency check incomplete: " + error)
            return
        self.report = report
        self.results.setPlainText(report_text(report))
        self.status.setText(f"Found {len(report.get('refs') or [])} reference(s) in this supplied scope. "
                            "Global completeness is unknown; deletion remains unavailable.")

    def can_close(self):
        return self.task is None and not self.pool.activeThreadCount()

    def closeEvent(self, event):
        if not self.can_close():
            self.close_requested = True
            if self.task is not None:
                self.task.cancelled.set()
            self._controls()
            self.status.setText("Finishing the current metadata read before closing…")
            event.ignore()
            return
        event.accept()

    def reject(self):
        # Escape must use the same worker-retirement gate as the close button.
        self.close()
