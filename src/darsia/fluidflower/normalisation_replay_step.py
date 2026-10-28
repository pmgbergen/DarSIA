"""Owned GUI worker for exact, isolated replay of the saved lighting method."""

from threading import Event

from PySide6.QtCore import QObject, QSignalBlocker, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget

from . import normalisation_replay
from .normalisation_replay_process import execute_replay


class _ReplayWork(QObject):
    finished = Signal()
    progress = Signal(str)

    def __init__(self, function):
        super().__init__()
        self.function, self.result, self.error = function, None, ""

    @Slot()
    def run(self):
        try:
            self.result = self.function(self.progress.emit)
        except Exception as error:
            self.error = str(error)
        self.finished.emit()


class NormalisationReplayStep(QWidget):
    busy_changed = Signal(bool)

    def __init__(self, parent=None, *, preparer=None, executor=None, allow_run=True):
        super().__init__(parent)
        self.preparer = preparer or normalisation_replay.prepare_replay
        self.executor = executor or execute_replay
        self.allow_run = allow_run
        self.thread = self.worker = None
        self._cancel = Event()
        self._request = None
        self._closing = self._owner_busy = False
        self._epoch = 0
        self.series_root, self.series_id = None, ""
        self._recipe_available, self._source_family = False, "unknown"
        layout = QVBoxLayout(self)
        description = QLabel("Optional comparison with a previously saved correction result. This does not choose the source format and is not required for image correction or model training. It re-runs the saved RAW lighting method and compares its fits, pixels and valid-pixel mask exactly, saving a separate copy.")
        description.setWordWrap(True)
        layout.addWidget(description)
        line = QHBoxLayout()
        line.addWidget(QLabel("Run"))
        self.run = QComboBox()
        self.run.currentIndexChanged.connect(self._selection_changed)
        line.addWidget(self.run)
        line.addWidget(QLabel("Image index (0 = baseline)"))
        self.frame = QSpinBox()
        self.frame.setRange(0, 100000)
        self.frame.valueChanged.connect(self._selection_changed)
        line.addWidget(self.frame)
        self.prepare = QPushButton("Prepare reproduction")
        self.prepare.clicked.connect(self._prepare)
        line.addWidget(self.prepare)
        layout.addLayout(line)
        line = QHBoxLayout()
        self.output = QLineEdit()
        self.output.setPlaceholderText("Folder for the separate reproduced result")
        self.output.textChanged.connect(self._controls)
        line.addWidget(self.output, 1)
        self.choose_output = QPushButton("Choose output folder…")
        self.choose_output.clicked.connect(self._choose_output)
        line.addWidget(self.choose_output)
        self.start = QPushButton("Verify saved RAW correction")
        self.start.clicked.connect(self._reproduce)
        line.addWidget(self.start)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.cancel_work)
        line.addWidget(self.cancel)
        layout.addLayout(line)
        self.status = QLabel("Choose a series to check saved lighting reproduction availability.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self._controls()

    def set_context(self, root, series):
        self._epoch += 1
        self._cancel.set()
        self.series_root, self.series_id, self._request = root, str(series or ""), None
        self._recipe_available, self._source_family = False, "unknown"
        with QSignalBlocker(self.run), QSignalBlocker(self.frame):
            self.run.clear()
            self.frame.setValue(0)
        self.status.setText(self._context_status())
        self._controls()

    def _context_status(self):
        if self.series_root is None or not self.series_id:
            return "Choose a series to check saved lighting reproduction availability."
        if not self.allow_run:
            return f"{self.series_id}: this preview is read only; lighting reproduction cannot be started here."
        if self._source_family == "processed":
            return f"{self.series_id}: the Setup selection contains processed images. The saved RAW-result comparison is unavailable and is not required for this input format."
        if not self._recipe_available:
            return f"{self.series_id}: a replayable saved RAW correction is not available. This optional comparison does not select or prepare the current image basis."
        if self.run.currentData() is None:
            return f"{self.series_id}: no run is loaded for saved lighting reproduction."
        return (f"{self.series_id}: prepare the saved inputs, choose an output folder, then reproduce from RAW. "
                "Preparation reads the original source records and may take a minute.")

    def install_snapshot(self, runs, *, document=None):
        from .image_normalisation import selected_source_family
        document = document or {}
        self._source_family = selected_source_family(document)
        saved = document.get("phase_normalisation") or {}
        self._recipe_available = bool((saved.get("field_sources") or {}).get("v5_raw_photometry", {}).get("source"))
        with QSignalBlocker(self.run):
            self.run.clear()
            for run in runs:
                self.run.addItem(run.upper(), run)
            preferred = self.run.findData((document.get("image_normalisation") or {}).get("reference_run"))
            if preferred >= 0:
                self.run.setCurrentIndex(preferred)
        self.status.setText(self._context_status())
        self._controls()

    def set_owner_busy(self, busy):
        self._owner_busy = bool(busy)
        self._controls()

    def _selection_changed(self, *_args):
        self._request = None
        self._controls()

    def _choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Reproduced lighting output folder", self.output.text())
        if folder:
            self.output.setText(folder)

    def _controls(self, *_args):
        idle = self.thread is None and not self._closing and not self._owner_busy
        available = (idle and self.allow_run and self.series_root is not None
                     and self._recipe_available and self._source_family != "processed")
        self.run.setEnabled(available)
        self.frame.setEnabled(available)
        self.prepare.setEnabled(available and self.run.currentData() is not None)
        self.output.setEnabled(idle)
        self.choose_output.setEnabled(idle)
        self.start.setEnabled(available and self._request is not None and bool(self.output.text().strip()))
        self.cancel.setEnabled(self.thread is not None and not self._closing)

    def _prepare(self):
        if not self.prepare.isEnabled():
            return
        root, run, frame, preparer = self.series_root, self.run.currentData(), self.frame.value(), self.preparer
        self._request = None
        self.status.setText("Preparing the exact saved RAW inputs…")
        self._start("prepare", lambda _progress: preparer(root, run, frame))

    def _reproduce(self):
        if not self.start.isEnabled():
            return
        request, output, executor = self._request, self.output.text().strip(), self.executor
        cancel = self._cancel
        def check_cancelled():
            if cancel.is_set():
                raise RuntimeError("Reproduction cancelled; no completed result was saved.")
        self.status.setText("Reproducing RAW inputs and saved lighting…")
        self._start("reproduce", lambda progress: executor(request, output, progress=progress, check_cancelled=check_cancelled))

    def _start(self, operation, function):
        self._cancel.clear()
        self.thread, self.worker = QThread(self), _ReplayWork(function)
        self.worker.key = operation, self._epoch
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self._retired)
        self._controls()
        self.busy_changed.emit(True)
        self.thread.start()

    @Slot(str)
    def _progress(self, message):
        if self.worker is not None and self.worker.key[1] == self._epoch and not self._closing:
            self.status.setText(message)

    @Slot()
    def _retired(self, expected=None):
        expected = self.sender() if expected is None else expected
        if self.thread is None or self.thread is not expected:
            return
        thread, worker = self.thread, self.worker
        thread.wait()
        self.thread = self.worker = None
        operation, epoch = worker.key
        if not self._closing and epoch == self._epoch:
            if self._cancel.is_set() and operation == "prepare":
                self._request = None
                self.status.setText("Reproduction cancelled; no completed result was saved.")
            elif worker.error:
                self._request = None
                self.status.setText(worker.error)
            elif operation == "prepare":
                self._request = worker.result
                self.status.setText(f"Inputs ready: {len(self._request['rows'])} original baselines. Choose an output folder and reproduce from RAW.")
            else:
                self.status.setText("Exact saved lighting and output reproduced. Result: " + str(worker.result))
        del worker
        thread.deleteLater()
        self._controls()
        self.busy_changed.emit(False)
        if self._closing:
            self.close()

    def can_close(self):
        return self.thread is None

    def cancel_work(self):
        self._cancel.set()

    def wait_for_workers(self):
        self._closing = True
        self._cancel.set()
        self._controls()
        if self.thread is not None:
            thread = self.thread
            thread.wait()
            self._retired(thread)

    def closeEvent(self, event):
        self._closing = True
        self._cancel.set()
        if not self.can_close():
            event.ignore()
            return
        super().closeEvent(event)
