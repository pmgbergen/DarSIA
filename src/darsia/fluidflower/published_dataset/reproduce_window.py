"""The reproduction launcher window: the command line's steps with a small Qt front end.

Choose the unpacked dataset and an empty results folder, the scope and the
machine settings, then Start. Checking, running, comparing and reporting run
in worker threads. Pause stops each worker at its next frame boundary; Resume
replays a paused run from its first frame. The window and
``darsia-reproduce --dataset ... --results ...`` share every step.
"""

from __future__ import annotations

from pathlib import Path
import traceback

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)


class Task(QThread):
    """Run one callable off the GUI thread; emit its result or a traceback."""

    done = Signal(object, str)

    def __init__(self, function, parent=None):
        """Keep the callable to run."""
        super().__init__(parent)
        self.function = function

    def run(self):
        """Call the function and emit ``(result, "")`` or ``(None, traceback)``."""
        try:
            result = self.function()
        except Exception:  # noqa: BLE001 - the window shows the error verbatim
            self.done.emit(None, traceback.format_exc())
            return
        self.done.emit(result, "")


def _label(text: str = "") -> QLabel:
    """A wrapped label."""
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def _folder_row(edit: QLineEdit, button: QPushButton) -> QWidget:
    """A path field and its Browse button."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return row


class ReproduceWindow(QWidget):
    """Reproduce a published dataset: dataset, results, scope, machine settings, run, report."""

    progress_changed = Signal(object)
    dataset_opened = Signal()

    def __init__(self, *, runner_factory=None, installed=None, fingerprint=None, machine=None, inventory=None):
        """Lay out the steps; ``runner_factory`` etc. replace the real machine in tests."""
        super().__init__()
        from darsia.fluidflower.ui.interaction_safety import install_interaction_safety
        install_interaction_safety()
        self.setWindowTitle("Reproduce a published dataset")
        self.runner_factory, self.installed, self.fingerprint = runner_factory, installed, fingerprint
        self.machine, self.inventory = machine, inventory
        self.reproduction = self.runner = None
        self.tasks = []
        self.result = None
        layout = QVBoxLayout(self)
        layout.addWidget(_label("Reproduce the published results of a dataset on this machine. The dataset "
                                "folder is only read; everything is written into the results folder."))
        dataset = QGroupBox("Dataset folder")
        form = QFormLayout(dataset)
        self.dataset_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._choose(self.dataset_edit, "Unpacked dataset folder", self.open_dataset))
        form.addRow("Folder", _folder_row(self.dataset_edit, browse))
        self.dataset_label = _label("Choose the unpacked dataset folder.")
        form.addRow(self.dataset_label)
        self.verify_button = QPushButton("Check every SHA-256")
        self.verify_button.clicked.connect(self.verify)
        self.verify_button.setEnabled(False)
        form.addRow(self.verify_button)
        layout.addWidget(dataset)
        results = QGroupBox("Results folder")
        form = QFormLayout(results)
        self.results_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._choose(self.results_edit, "New or empty results folder",
                                                    self.refresh_settings))
        self.results_edit.editingFinished.connect(self.refresh_settings)
        form.addRow("Folder", _folder_row(self.results_edit, browse))
        layout.addWidget(results)
        scope = QGroupBox("Scope")
        form = QFormLayout(scope)
        self.scope_group = QButtonGroup(self)
        row = QHBoxLayout()
        self.scope_buttons = {}
        for kind, text in (("quick", "Quick check"), ("run", "One run"), ("all", "All runs")):
            button = QRadioButton(text)
            self.scope_group.addButton(button)
            self.scope_buttons[kind] = button
            button.toggled.connect(self.refresh_settings)
            row.addWidget(button)
        self.scope_buttons["all"].setChecked(True)
        form.addRow(row)
        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self.refresh_settings)
        form.addRow("Run", self.run_combo)
        self.quick_spin = QSpinBox()
        self.quick_spin.setRange(1, 1000)
        self.quick_spin.valueChanged.connect(self.refresh_settings)
        form.addRow("Quick-check frames", self.quick_spin)
        layout.addWidget(scope)
        machine = QGroupBox("Machine settings")
        form = QFormLayout(machine)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 256)
        form.addRow("Workers", self.workers_spin)
        self.advice_label = _label()
        form.addRow(self.advice_label)
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 256)
        self.threads_spin.valueChanged.connect(self.refresh_settings)
        form.addRow("Threads per worker", self.threads_spin)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("CPU", "cpu")
        self.backend_combo.addItem("CUDA (only where it is qualified)", "cuda")
        self.backend_combo.currentIndexChanged.connect(self.refresh_settings)
        form.addRow("Backend", self.backend_combo)
        self.precision_label = _label()
        form.addRow("Precision", self.precision_label)
        self.override_check = QCheckBox("Run another DarSIA commit anyway (the report says: not a reproduction)")
        form.addRow(self.override_check)
        self.stage_runtimes_edit = QLineEdit()
        self.stage_runtimes_edit.setPlaceholderText("Staged RAW reproduction: original Python and checkout for every stage")
        self.stage_runtimes_edit.editingFinished.connect(self.refresh_settings)
        runtime_button = QPushButton("Browse…")
        runtime_button.clicked.connect(self._choose_stage_runtimes)
        form.addRow("Original stage runtimes", _folder_row(self.stage_runtimes_edit, runtime_button))
        self.scratch_check = QCheckBox("Remove each run's unpacked raw images after the run")
        self.scratch_check.setChecked(True)
        form.addRow(self.scratch_check)
        self.estimate_label = _label()
        form.addRow(self.estimate_label)
        layout.addWidget(machine)
        actions = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause)
        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume)
        self.report_button = QPushButton("Open report")
        self.report_button.clicked.connect(self.open_report)
        for button in (self.start_button, self.pause_button, self.resume_button, self.report_button):
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.progress_table = QTableWidget(0, 2)
        self.progress_table.setHorizontalHeaderLabels(["Run", "State"])
        layout.addWidget(self.progress_table, 1)
        self.status_label = _label()
        layout.addWidget(self.status_label)
        self.progress_changed.connect(self._show_progress)

    def _choose(self, edit: QLineEdit, title: str, then):
        """Let the user pick a folder for ``edit``, then continue with ``then``."""
        folder = QFileDialog.getExistingDirectory(self, title, edit.text())
        if folder:
            edit.setText(folder)
            then()

    def _start_task(self, function, finished):
        """Run ``function`` in a worker thread and call ``finished(result, error)`` in the GUI thread."""
        task = Task(function, self)
        self.tasks.append(task)

        def done(result, error):
            """Forget the finished task and hand over its result."""
            task.wait()
            self.tasks.remove(task)
            task.deleteLater()
            finished(result, error)

        task.done.connect(done)
        task.start()
        return task

    def busy(self) -> bool:
        """True while a check, run or report is in progress."""
        return bool(self.tasks)

    def open_dataset(self):
        """Step 1: read and check the dataset folder (in a worker thread)."""
        from .reproduce import Reproduction

        path = Path(self.dataset_edit.text().strip())
        self.dataset_label.setText("Reading the dataset…")
        self.runner = self.runner_factory() if self.runner_factory is not None else None
        if self.runner is None:
            from .replay import ReplayRunner

            self.runner = ReplayRunner()
        reproduction = Reproduction(path, Path(self.results_edit.text().strip() or "."), runner=self.runner)
        self._start_task(lambda: reproduction.open(installed=self.installed, fingerprint=self.fingerprint),
                         lambda result, error: self._opened(reproduction, result, error))

    def _opened(self, reproduction, check, error):
        """Show the dataset, its runs, the code and environment comparison."""
        if error:
            self.dataset_label.setText("Not a published dataset folder:\n" + error.strip().splitlines()[-1])
            return
        self.reproduction = reproduction
        manifest = check.manifest
        from .readme import export_resolution
        lines = [f"{manifest['identity']['title']}: series {manifest['identity']['series_id']}, record "
                 f"{manifest['identity']['record']}, mass maps {export_resolution(manifest)}, {len(check.runs)} runs, "
                 f"{sum(len(row['frames']) for row in manifest['runs'])} frames.",
                 "DarSIA: the published commit." if check.code["same"] else check.code["note"]]
        environment_label = "Launcher environment" if manifest.get("staged_reproduction") else "Environment"
        lines += [f"{environment_label}: {line}" for line in check.environment[:12]]
        lines += [f"Problem: {line}" for line in check.problems]
        staged = bool(manifest.get("staged_reproduction"))
        if staged:
            lines[1] = "Each original stage uses its own verified Python, libraries and code checkout."
            self.override_check.setChecked(False)
        self.dataset_label.setText("\n".join(lines))
        self.override_check.setVisible(not staged and not check.code["same"])
        self.override_check.setEnabled(not staged)
        self.stage_runtimes_edit.setEnabled(staged)
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run in check.runs:
            self.run_combo.addItem(run, run)
        self.run_combo.blockSignals(False)
        self.scope_buttons["quick"].setEnabled(bool(manifest.get("quick_check_valid")))
        self.quick_spin.setValue(int((manifest.get("quick_check") or {}).get("default_frames") or 3))
        self.threads_spin.blockSignals(True)
        self.threads_spin.setValue(int((manifest.get("machine") or {}).get("threads") or 1))
        self.threads_spin.blockSignals(False)
        self.precision_label.setText(f"{manifest['plan'].get('precision')} (fixed by the dataset)")
        self.verify_button.setEnabled(True)
        self.refresh_settings()
        self.dataset_opened.emit()

    def verify(self):
        """The full SHA-256 check, in a worker thread."""
        if self.reproduction is None:
            return
        self.status_label.setText("Checking every SHA-256…")
        self._start_task(self.reproduction.verify, self._verified)

    def _verified(self, problems, error):
        """Show the result of the full check."""
        if error:
            self.status_label.setText(error.strip().splitlines()[-1])
        else:
            self.status_label.setText("Every SHA-256 matches the dataset." if not problems else
                                      "The full check found:\n" + "\n".join(problems[:20]))

    def scope(self):
        """The chosen scope."""
        from .reproduce import choose_scope

        kind = next(kind for kind, button in self.scope_buttons.items() if button.isChecked())
        return choose_scope(self.reproduction.check.manifest, kind, run=self.run_combo.currentData(),
                            quick_frames=self.quick_spin.value())

    def settings(self):
        """The chosen machine settings."""
        from .reproduce import Settings

        return Settings(workers=self.workers_spin.value(), threads=self.threads_spin.value(),
                        backend=self.backend_combo.currentData(), override_commit=self.override_check.isChecked(),
                        remove_scratch=self.scratch_check.isChecked(),
                        stage_runtimes=self.stage_runtimes_edit.text().strip() or None)

    def _choose_stage_runtimes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Original stage runtimes",
            self.stage_runtimes_edit.text(), "Runtime mapping (*.json)")
        if path:
            self.stage_runtimes_edit.setText(path)

    def refresh_settings(self, *_):
        """Recompute the worker default, the estimate and whether Start can be offered."""
        if self.reproduction is None:
            return
        from .reproduce import ReproductionRefused

        self.reproduction.results_root = Path(self.results_edit.text().strip() or ".")
        try:
            scope = self.scope()
        except ReproductionRefused as error:
            self.advice_label.setText(str(error))
            self.start_button.setEnabled(False)
            return
        advice = self.reproduction.advice(scope, threads=self.threads_spin.value(),
                                          backend=self.backend_combo.currentData(), inventory=self.inventory)
        self.workers_spin.setValue(int(advice.get("workers") or 0))
        self.advice_label.setText(advice.get("message") or advice["reason"])
        estimate = self.reproduction.estimate(scope, max(1, self.workers_spin.value()))
        hours = "not recorded" if not estimate["seconds"] else f"about {estimate['seconds'] / 3600:.1f} h"
        self.estimate_label.setText(f"Time: {hours} ({estimate['seconds_basis']}). Disk: about "
                                    f"{estimate['needed_bytes'] / 1e9:.1f} GB ({estimate['output_basis']}).")
        self.start_button.setEnabled(bool(self.results_edit.text().strip()) and not self.busy())

    def start(self):
        """Steps 5 and 6 in a worker thread: run, compare and write the report."""
        from .reproduce import execute

        if self.reproduction is None or self.busy():
            return
        self.reproduction.results_root = Path(self.results_edit.text().strip())
        scope, settings = self.scope(), self.settings()
        problems = self.reproduction.preflight(scope, settings)
        if problems:
            self.status_label.setText("Not started:\n" + "\n".join(problems))
            return
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.status_label.setText("Running…")
        self._start_task(lambda: execute(self.reproduction, scope, settings, machine=self.machine,
                                         progress=self.progress_changed.emit), self._finished)

    def _show_progress(self, states):
        """Show each run's state."""
        self.progress_table.setRowCount(len(states))
        for row, (run, state) in enumerate(states.items()):
            self.progress_table.setItem(row, 0, QTableWidgetItem(run))
            self.progress_table.setItem(row, 1, QTableWidgetItem(state))

    def _finished(self, result, error):
        """Show the dataset verdict and offer the report."""
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        if error:
            self.status_label.setText("Stopped:\n" + error.strip().splitlines()[-1])
            self.start_button.setEnabled(True)
            return
        self.result = result
        summary = result["summary"]
        text = f"Verdict: {summary['dataset']}."
        if summary.get("first_divergence"):
            first = summary["first_divergence"]
            text += f" First divergence: run {first['run']}, {first['case_id']}, {first['quantity']}."
        self.status_label.setText(text)
        self.report_button.setEnabled(True)

    def pause(self):
        """Publish Pause: the workers stop after their current frame."""
        if self.runner is not None and hasattr(self.runner, "pause"):
            self.runner.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)

    def resume(self):
        """Publish Run again and restart the workers."""
        if self.runner is not None and hasattr(self.runner, "resume"):
            try:
                self.runner.resume(self.settings())
            except (OSError, ValueError, TypeError) as error:
                self.status_label.setText("Not resumed:\n" + str(error))
                return
            self.resume_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.status_label.setText("Running…")

    def open_report(self):
        """Open report.html in the default browser."""
        if self.result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result["files"]["html"])))

    def closeEvent(self, event):
        """Wait for running checks before the window closes (a running job keeps its workers)."""
        for task in list(self.tasks):
            task.wait()
        super().closeEvent(event)


def main(arguments=None) -> int:
    """Open the launcher window, pre-filled from command-line arguments when given."""
    application = QApplication.instance() or QApplication([])
    window = ReproduceWindow()
    if arguments is not None and getattr(arguments, "dataset", None):
        window.dataset_edit.setText(str(arguments.dataset))
    if arguments is not None and getattr(arguments, "results", None):
        window.results_edit.setText(str(arguments.results))
    if arguments is not None and getattr(arguments, "stage_runtimes", None):
        window.stage_runtimes_edit.setText(str(arguments.stage_runtimes))
    window.show()
    if window.dataset_edit.text():
        window.open_dataset()
    return application.exec()
