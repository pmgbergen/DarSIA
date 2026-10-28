"""One explicit finite owned call for current frozen-calibration checks."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from . import calibration_check_calculation as calculation

EXPLANATION = (
    "Check 1: the 12 held-out runs at 1.0 and 2.25 h; median |M/d - 1| must be at most 0.10, "
    "and each lighting-group median M/d must lie in 0.90-1.10. "
    "Check 2: held-out runs at 4, 8, 12, 24 and 48 h. Check 3: held-out runs plus each fit run's "
    "own leave-one-run-out prediction at 56, 64, 72 and 96 h. "
    "d is injected mass. W is interval width; a run counts only when W/d is at most 0.05. "
    "Delta is zero when d lies inside the mass interval, otherwise the distance to its nearest end divided by d. "
    "An assessable time needs at least seven counted runs and median delta at most 0.10. "
    "Checks 2/3 need four/five and three/four assessable times. The U_rest=0 fallback is reported alongside "
    "the primary interval and decides only when the primary has too few assessable times. "
    "Pass is a check result, not adoption or permission to start production."
)


class CalculateChecksDialog(QDialog):
    """Verify actual completed inputs, then calculate without fitting or Jobs."""
    calculated = Signal(object)

    def __init__(self, parent=None, *, series_root=None, embedded=False):
        super().__init__(parent, Qt.WindowType.Widget if embedded else Qt.WindowType.Dialog)
        self._embedded = embedded
        self.setWindowTitle("Calculate calibration checks")
        self._thread = self._worker = None
        self._ready = None
        self._generation = 0
        self._call_context = None
        layout = QVBoxLayout(self)
        note = QLabel(EXPLANATION)
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(note)
        form = QFormLayout()
        self.fields = {}
        self.browse_buttons = []
        for key, label, directory in (
            ("series", "Registered series folder", True),
            ("calibration", "Completed mass-calibration job", True),
            ("phase", "Completed phase-map generation job", True),
            ("protocol", "Current scoring protocol", False),
        ):
            field = QLineEdit()
            row = QHBoxLayout()
            row.addWidget(field)
            browse = QPushButton("Choose…")
            browse.clicked.connect(lambda _checked=False, k=key, d=directory: self._choose(k, d))
            row.addWidget(browse)
            self.browse_buttons.append(browse)
            form.addRow(label, row)
            self.fields[key] = field
            field.textChanged.connect(self._changed)
        if series_root is not None:
            self.fields["series"].setText(str(series_root))
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.verify_button = QPushButton("Verify completed inputs")
        self.calculate_button = QPushButton("Calculate checks…")
        self.close_button = QPushButton("Close")
        self.verify_button.clicked.connect(self.verify_inputs)
        self.calculate_button.clicked.connect(self.calculate)
        self.close_button.clicked.connect(self.close)
        for button in (self.verify_button, self.calculate_button, self.close_button):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        self.close_button.setVisible(not embedded)
        layout.addLayout(buttons)
        self.status = QLabel("Choose the original completed jobs and current protocol. No phase-two job or calibration adoption is required.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self._controls()

    def _context(self):
        return tuple(self.fields[key].text().strip() for key in ("series", "calibration", "phase", "protocol"))

    def _changed(self, *_args):
        self._generation += 1
        self._ready = None
        if hasattr(self, "calculate_button"):
            self._controls()

    def _controls(self):
        idle = self._thread is None
        for field in self.fields.values():
            field.setEnabled(idle)
        for button in self.browse_buttons:
            button.setEnabled(idle)
        self.verify_button.setEnabled(idle and all(self._context()))
        self.calculate_button.setEnabled(idle and self._ready is not None)
        self.close_button.setEnabled(idle)

    def _choose(self, key, directory):
        if self._thread is not None:
            return
        original = self._context()
        if directory:
            path = QFileDialog.getExistingDirectory(self, self.fields[key].placeholderText() or "Choose completed input", self.fields[key].text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose the current scoring protocol", self.fields[key].text(), "Protocol (*.md);;All files (*)")
        if path and self._context() == original:
            self.fields[key].setText(path)

    def verify_inputs(self):
        if not self.verify_button.isEnabled():
            return False
        context = self._context()
        def verify():
            return calculation.check_readiness(*context[:3],
                protocol=calculation.protocol_binding(context[3], series_root=context[0]))
        return self._start(verify, "verify", context)

    def calculate(self):
        if not self.calculate_button.isEnabled():
            return False
        context, ready, generation = self._context(), self._ready, self._generation
        destination, _ = QFileDialog.getSaveFileName(self, "Save a fresh calculated check-score file", "", "Check-score file (*.json)")
        if (not destination or self._context() != context or generation != self._generation
                or self._ready is not ready or self._thread is not None):
            return False
        def compute():
            return calculation.calculate_check_scores(*context[:3], destination,
                protocol=calculation.protocol_binding(context[3], series_root=context[0]),
                expected_readiness=ready)
        return self._start(compute, "calculate", context)

    def _start(self, call, kind, context):
        if self._thread is not None:
            return False
        # Reuse the original finite reader: it returns QObject affinity to the
        # GUI thread before finished. The native thread is joined below before
        # dropping either owner, including failure and close requests.
        from .painting_gap_review_ui import _Read
        thread = QThread(self)
        worker = _Read(call)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._retired)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        self._call_context = (kind, context, self._generation)
        self._controls()
        self.status.setText("Verifying original completed publications…" if kind == "verify" else
                            "Reading saved phase-map inputs and evaluating frozen parameters. No fit is running…")
        thread.start()
        return True

    @Slot()
    def _retired(self):
        thread, worker, call_context = self._thread, self._worker, self._call_context
        if thread is None:
            return
        thread.wait()
        self._thread = self._worker = self._call_context = None
        kind, context, generation = call_context
        if context != self._context() or generation != self._generation:
            self._ready = None
            self.status.setText("The selected inputs changed; verify the current selection.")
        elif worker.error:
            self._ready = None
            self.status.setText("Checks could not be calculated: " + worker.error)
        elif kind == "verify":
            self._ready = worker.result
            self.status.setText("Completed input publications are available. Calculate verifies the saved sources and generated phase-map inputs before evaluation.")
        else:
            self._ready = None
            self.status.setText("Calculated checks saved. No calibration was adopted and no job started.")
            self.calculated.emit(worker.result)
        self._controls()

    def can_close(self):
        return self._thread is None

    def closeEvent(self, event):
        if not self.can_close():
            self.status.setText("Calculation is still finishing. Close after the original reader has returned.")
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):
        if not self._embedded and self.can_close():
            super().reject()
