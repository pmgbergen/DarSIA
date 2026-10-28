"""All runs for the Direct colour method (N141 spec 2.6 and 3.4, Part 2) and
for Setup's Build run model and Full setup, which work on one run at a time.

"All runs" runs a row's existing per-run action once for each registered run
of the series, one after another: each run's own configuration is opened and
the same tab action that Play starts is started on it. The first error stops
the sequence, nothing is pooled between runs, and each run gets one result
line. Above a Direct colour method row the strip also shows the computing
size that the run configuration's existing [corrections.resize] section sets.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# The Setup steps that work on the opened run's configuration: Build run model
# and Full setup. Depth has its own two-stage page and is not repeated here.
SETUP_ALL_RUNS_ROWS = frozenset({("setup", item) for item in ("rig", "all")})
# The rows whose action runs on one run and may run on all: the Direct colour
# method's and the Setup steps above. Mass calibration runs belong to Jobs,
# and the danger zone is never repeated.
ALL_RUNS_ROWS = frozenset(
    {("calibration", "color")}
    | {
        ("analysis", item)
        for item in ("cropping", "mass", "segmentation", "fingers", "thresholding", "volume")
    }
) | SETUP_ALL_RUNS_ROWS
RESIZE_NOTE = (
    "A resize trades accuracy for speed. It is set in the preprocessing wizard, "
    "step 'Type & resize' ([corrections.resize])."
)


def resize_summary(corrections) -> str:
    """The computing size that [corrections.resize] sets: native, a scale or a shape.

    The section is parsed by ResizeCorrectionConfig, the one loader the
    workflow uses, so an invalid section raises its ValueError.
    """
    from darsia.presets.workflows.config.corrections import ResizeCorrectionConfig

    corrections = corrections if isinstance(corrections, Mapping) else {}
    section = corrections.get("resize")
    active = corrections.get("active")
    if not section or (active is not None and "resize" not in active):
        return "native"
    resize = ResizeCorrectionConfig().load(dict(section))
    if resize.mode == "scale":
        return f"{float(resize.scale):g} × native (resize)"
    rows, cols = resize.target_shape
    return f"resized to {rows} × {cols} pixels (resize)"


def runs_resize_summary(runs) -> str:
    """One line for all runs: how many compute at each size, from their own files."""
    from darsia.presets.workflows.config.utils import _get_section_from_toml

    counts, problems = {}, []
    for run, path in runs:
        if not path:
            problems.append(f"{run}: no run configuration")
            continue
        try:
            text = resize_summary(_get_section_from_toml(Path(path), "corrections"))
        except KeyError:
            text = "native"  # No [corrections] section: the workflow resizes nothing.
        except (OSError, ValueError, TypeError) as error:
            problems.append(f"{run}: {error}")
            continue
        counts[text] = counts.get(text, 0) + 1
    parts = [f"{text} for {count} run{'s' if count != 1 else ''}" for text, count in counts.items()]
    return "; ".join(parts + problems) or "no runs"


class AllRunsSequence(QObject):
    """Start one row's existing action on each run's own configuration, in order."""

    started = Signal(str)
    result = Signal(str, str)
    completed = Signal(bool, str)

    def __init__(self, window):
        super().__init__(window if isinstance(window, QObject) else None)
        self.window = window
        self.current_process = None
        self._running = self._cancelled = False
        self._queue = []
        self._row = None
        self._skipped = 0

    @property
    def running(self) -> bool:
        """True from start until the last run finished or the sequence stopped."""
        return self._running

    def start(self, row, runs) -> bool:
        """Run row's action for each (run, configuration path); False if refused."""
        if self._running or self.window.process_runner.active:
            self.completed.emit(False, "Another calculation is already running.")
            return False
        self._row = tuple(row)
        self._queue = [(str(run), path) for run, path in runs]
        self._running, self._cancelled, self._skipped = True, False, 0
        self._next()
        return self._running

    def abort(self) -> None:
        """Stop the running action and start no further run."""
        if not self._running:
            return
        self._cancelled = True
        if self.current_process is not None:
            self.window.process_runner.abort_workflow_process(self.current_process)
        else:
            self._finish(False, "All runs stopped.")

    def _next(self) -> None:
        """Open the next run's configuration and start the row's action on it."""
        if not self._running:
            return
        if self._cancelled:
            self._finish(False, "All runs stopped.")
            return
        if not self._queue:
            skipped = f" {self._skipped} run(s) had no configuration and were skipped." if self._skipped else ""
            self._finish(True, "All runs finished." + skipped)
            return
        run, path = self._queue.pop(0)
        if not path:
            self._skipped += 1
            self.result.emit(run, "skipped: no run configuration")
            QTimer.singleShot(0, self._next)
            return
        window = self.window
        if not window._load_run_config(path):
            self.result.emit(run, "not started: its configuration could not be opened")
            self._finish(False, f"{run}: its configuration could not be opened. The remaining runs were not started.")
            return
        if (window.selected_action, window.selected_checkbox_id) != self._row:
            self.result.emit(run, "not started: the selected step changed")
            self._finish(False, "The selected step changed. The remaining runs were not started.")
            return
        tab = window.action_dispatch[self._row[0]]
        tab.process = None
        self.started.emit(run)
        try:
            tab.on_run_clicked()
        except Exception as error:
            self.result.emit(run, f"did not start: {type(error).__name__}: {error}")
            self._finish(False, f"{run} did not start. The remaining runs were not started.")
            return
        process = tab.process
        if process is None:
            self.result.emit(run, "did not start")
            self._finish(False, f"{run} did not start. The remaining runs were not started.")
            return
        self.current_process = process
        process.finished.connect(
            lambda code, status, process=process, run=run: self._run_finished(process, run, code, status)
        )

    def _run_finished(self, process, run, code, status) -> None:
        """Record one run's outcome; continue only after a clean exit."""
        if process is not self.current_process or not self._running:
            return
        self.current_process = None
        if self._cancelled or process.property("darsia_aborted"):
            self.result.emit(run, "stopped")
            self._finish(False, "All runs stopped.")
        elif status != QProcess.ExitStatus.NormalExit or code != 0:
            self.result.emit(run, f"failed (exit code {code})")
            self._finish(False, f"{run} failed. The remaining runs were not started.")
        else:
            self.result.emit(run, "done")
            QTimer.singleShot(0, self._next)

    def _finish(self, success: bool, message: str) -> None:
        """End the sequence once and report how it ended."""
        if not self._running:
            return
        self._running = False
        self.current_process = None
        self._queue = []
        self.completed.emit(success, message)


class NativeAllRuns:
    """The strip above a row that may run on all runs: This run, All runs, and
    for a Direct colour method row the computing size."""

    def __init__(self, window, *, confirm=None):
        self.window = window
        self.confirm = confirm or self._ask
        self.sequence = AllRunsSequence(window)
        self._restore = None
        self.panel = QWidget()
        self.panel.setObjectName("nativeAllRuns")
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(0, 2, 0, 2)
        self.resize_label = QLabel()
        self.resize_label.setTextFormat(Qt.TextFormat.PlainText)
        self.resize_label.setWordWrap(True)
        layout.addWidget(self.resize_label)
        row = QHBoxLayout()
        self.run_button = QPushButton()
        self.run_button.clicked.connect(self.run_this)
        self.all_button = QPushButton()
        self.all_button.clicked.connect(self.run_all)
        self.stop_button = QPushButton("Stop all runs")
        self.stop_button.clicked.connect(self.sequence.abort)
        self.stop_button.setEnabled(False)
        for button in (self.run_button, self.all_button, self.stop_button):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.results = QListWidget()
        self.results.hide()
        layout.addWidget(self.results)
        right = window.content_stack.parentWidget().layout()
        right.insertWidget(right.indexOf(window.content_stack), self.panel)
        self.panel.hide()
        self.sequence.started.connect(self._started)
        self.sequence.result.connect(self._result)
        self.sequence.completed.connect(self._completed)
        runner = window.process_runner
        original_done = runner._show_done_dialog

        def done(workflow, actions, config_path):
            # One summary replaces the per-run Done dialogs while all runs run.
            if self.sequence.running:
                return None
            return original_done(workflow, actions, config_path)

        done.__wrapped__ = original_done
        runner._show_done_dialog = done

    def _runs(self):
        """The registered runs and their configuration paths, as the run selector lists them."""
        combo = self.window.run_combo
        return [(combo.itemText(index), combo.itemData(index) or None) for index in range(combo.count())]

    def refresh(self) -> None:
        """Show the strip on the settings page of a row in ALL_RUNS_ROWS only."""
        window = self.window
        selected = window.sidebar.get_selected() if hasattr(window, "sidebar") else None
        shown = (
            selected in ALL_RUNS_ROWS
            and window.content_stack.currentWidget() is window.settings_scroll_area
        ) or self.sequence.running
        self.panel.setVisible(bool(shown))
        if not shown:
            return
        run = window.run_combo.currentText() if window.run_combo.currentIndex() >= 0 else ""
        # Setup steps compute no analysis, so no computing size is shown.
        setup = selected in SETUP_ALL_RUNS_ROWS
        self.resize_label.setVisible(not setup)
        if not setup:
            try:
                size = resize_summary((window.config_dict or {}).get("corrections")) if window.config_file else "no run configuration open"
            except (ValueError, TypeError, KeyError) as error:
                size = f"invalid [corrections.resize]: {error}"
            self.resize_label.setText(f"Computed at ({run.upper() or 'no run'}): {size}. {RESIZE_NOTE}")
        runs = [item for item in self._runs() if item[1]]
        busy = self.sequence.running or bool(window.process_runner.active)
        self.run_button.setText(f"Run {run.upper()}" if run else "Run this run")
        self.run_button.setEnabled(bool(run) and not busy)
        self.all_button.setText(f"Run all runs ({len(runs)})")
        self.all_button.setEnabled(bool(runs) and not busy)
        self.stop_button.setEnabled(self.sequence.running)

    def run_this(self) -> None:
        """This run: the same as Play for the selected run."""
        self.window.run_selected_workflow()

    def run_all(self) -> None:
        """All runs: confirm once, then start the row's action on every run in turn."""
        window = self.window
        row = (window.selected_action, window.selected_checkbox_id)
        if row not in ALL_RUNS_ROWS or self.sequence.running:
            return
        runs = self._runs()
        label = window.action_label_for(*row) if hasattr(window, "action_label_for") else row[1]
        question = (
            f"Run '{label}' for all {sum(1 for _run, path in runs if path)} runs, one after another?\n"
            "Each run uses its own configuration; the first error stops the sequence."
        )
        if row not in SETUP_ALL_RUNS_ROWS:
            question += f"\nComputed at: {runs_resize_summary(runs)}."
        if not self.confirm(question):
            return
        if row[0] == "analysis" and not window._confirm_method_action("colour_path", "Run original DarSIA analysis for all runs"):
            return
        self._restore = window.config_file
        window._all_runs_confirmed = True
        self.results.clear()
        for run, path in runs:
            self.results.addItem(f"{run.upper()}: waiting" if path else f"{run.upper()}: skipped: no run configuration")
        self.results.show()
        window.run_combo.setEnabled(False)
        self.status.setText("All runs started.")
        if not self.sequence.start(row, runs):
            window._all_runs_confirmed = False
            window.run_combo.setEnabled(True)
        self.refresh()

    def _ask(self, question) -> bool:
        """The one confirmation before All runs starts."""
        answer = QMessageBox.question(
            self.window,
            "Run all runs",
            question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _item(self, run):
        """The result line of one run, or None."""
        for index in range(self.results.count()):
            item = self.results.item(index)
            if item.text().split(":", 1)[0] == run.upper():
                return item
        return None

    def _started(self, run) -> None:
        """Mark the run that has just started."""
        item = self._item(run)
        if item is not None:
            item.setText(f"{run.upper()}: running")
        self.status.setText(f"Running {run.upper()}…")
        self.refresh()

    def _result(self, run, outcome) -> None:
        """Write one run's outcome on its result line."""
        item = self._item(run)
        if item is not None:
            item.setText(f"{run.upper()}: {outcome}")

    def _completed(self, success, message) -> None:
        """Release the run selector, reopen the run selected before, report the end."""
        window = self.window
        for index in range(self.results.count()):
            item = self.results.item(index)
            if item.text().endswith(": waiting"):
                item.setText(item.text().replace(": waiting", ": not started"))
        window._all_runs_confirmed = False
        window.run_combo.setEnabled(True)
        restore, self._restore = self._restore, None
        if restore and window.config_file != restore:
            window._load_run_config(restore)
        self.status.setText(message)
        window.print_log(message)
        self.refresh()
