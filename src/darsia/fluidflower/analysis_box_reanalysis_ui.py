"""Optional box integration from saved production maps in the Analysis editor."""
from copy import deepcopy
import csv
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from uuid import uuid4

from PySide6.QtCore import QStandardPaths, Qt, Signal, QUrl, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)


def registered_jobs(project):
    """Read this series' explicit local associations, without scanning its NAS."""
    from .series_geometry import native_series_link
    link = native_series_link(project)
    if not link.get("series_root") or not link.get("series_id"):
        return []
    return series_mass_jobs(link["series_root"], link["series_id"])


def series_mass_jobs(series_root, series_id):
    """The series' registered mass jobs (native-phase-n1, n1 or lattice-phase-n1).

    Only this computer's explicit job associations are read; the NAS is not
    scanned. Each row has the job's ``name`` and ``root``.
    """
    from .native_jobs_registry import JobsUIRegistry
    from .pipeline.job_queue import read_json, name
    from .storage_paths import resolve_storage_path
    registry = JobsUIRegistry(Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppLocalDataLocation)) / "native_jobs")
    result = []
    for row in registry.jobs(resolve_storage_path(series_root), series_id):
        record = read_json(Path(row["root"]) / "job.json")
        pipeline = record.get("pipeline")
        runs = record.get("expected_runs") or []
        if pipeline is None and runs:
            task = Path(row["root"]) / "results" / name(runs[0]) / "task.json"
            if task.is_file():
                pipeline = read_json(task).get("pipeline")
        if pipeline in {"native-phase-n1", "n1", "lattice-phase-n1"}:
            result.append(row)
    return result


def recalculate(*args, **kwargs):
    from .analysis_box_reanalysis import reanalyze_saved_boxes
    return reanalyze_saved_boxes(*args, **kwargs)


class BoxMassReanalysis(QGroupBox):
    """Use the editor's saved physical boxes; do not change its setup or model."""
    progress = Signal(object)

    def __init__(self, controller, *, jobs_reader=None, integrator=None):
        super().__init__("Mass from saved maps", controller.widget)
        self.controller = controller
        self.jobs_reader = jobs_reader or registered_jobs
        self.integrator = integrator or recalculate
        self._cancel = None
        self._result = None
        self._loaded_project = None
        self._output_job = None
        layout = QVBoxLayout(self)
        note = QLabel("Optional: define boxes before production, or recalculate their mass "
                      "afterwards from saved mass maps. Image correction, the phase model "
                      "and calibration are reused.")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        row.addWidget(QLabel("Production job"))
        self.jobs = QComboBox()
        self.jobs.setMinimumContentsLength(16)
        self.jobs.currentIndexChanged.connect(self._job_changed)
        row.addWidget(self.jobs, 1)
        self.browse = QPushButton("Choose job…")
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.browse)
        self.reload = QPushButton("Refresh jobs")
        self.reload.clicked.connect(self.load_jobs)
        row.addWidget(self.reload)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Results folder"))
        self.output = QLineEdit()
        self.output.setPlaceholderText("A new result subfolder is created here")
        row.addWidget(self.output, 1)
        self.choose_output = QPushButton("Choose…")
        self.choose_output.clicked.connect(self._choose_output)
        row.addWidget(self.choose_output)
        layout.addLayout(row)
        self.layers = QCheckBox("Also each sand layer")
        self.layers.setToolTip(
            "Also integrate the mass in each sand layer (facies) of the saved native maps, named from the "
            "series' facies table, into a table of its own next to each box table. Events can then ask "
            "about layers. Off: the box tables are made exactly as before.")
        layout.addWidget(self.layers)
        row = QHBoxLayout()
        self.run = QPushButton("Recalculate mass in boxes")
        self.run.clicked.connect(self.start)
        row.addWidget(self.run)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.cancel_requested)
        row.addWidget(self.cancel)
        self.open = QPushButton("Open result folder")
        self.open.clicked.connect(self._open_result)
        row.addWidget(self.open)
        layout.addLayout(row)
        self.status = QLabel("Choose a finished production job and save the boxes before recalculating.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.results = QTableWidget(0, 6)
        self.results.setHorizontalHeaderLabels(("Run / box", "Time [h]", "Total [g]",
                                               "Dissolved [g]", "Free gas [g]", "Complete"))
        self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results.setMaximumHeight(230)
        self.results.hide()
        layout.addWidget(self.results)
        self.progress.connect(self._progress, Qt.ConnectionType.QueuedConnection)
        controller.busy_changed.connect(self._busy_changed)
        self.refresh()

    def refresh(self):
        busy = self.controller.busy()
        for control in (self.jobs, self.browse, self.reload, self.output, self.choose_output, self.layers):
            control.setEnabled(not busy)
        config = self.controller._saved_configuration or {}
        self.run.setEnabled(not busy and not self.controller.dirty and self.controller.store is not None
                            and bool(self.jobs.currentData())
                            and any(row.get("enabled") for row in config.get("domains", [])))
        self.cancel.setEnabled(self._cancel is not None and busy)
        self.open.setEnabled(self._result is not None)
        self.setVisible(getattr(self.controller.view, "_workflow_scope", None) == "analysis_boxes")

    def _busy_changed(self, busy):
        if not busy and self._cancel is not None:
            self._cancel = None
            if self.controller._operation_error:
                self.status.setText(self.controller._operation_error)
        self.refresh()
        if not busy and self.isVisible():
            self._load_once()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self, self._load_once)

    def _load_once(self):
        project = self.controller.view.project
        if not self.controller.busy() and self._loaded_project is not project:
            self._loaded_project = project
            self.load_jobs()

    def load_jobs(self):
        if self.controller.busy():
            return False
        project = self.controller.view.project
        selected = self.jobs.currentData()
        def loaded(rows):
            if project is not self.controller.view.project:
                return
            self.jobs.blockSignals(True)
            self.jobs.clear()
            for row in rows:
                self.jobs.addItem(row["name"], row["root"])
            index = self.jobs.findData(selected)
            if index >= 0:
                self.jobs.setCurrentIndex(index)
            self.jobs.blockSignals(False)
            self._job_changed()
            self.status.setText("Select the production job whose saved maps should be integrated."
                                if rows else "No registered job. Choose an existing production job folder.")
        return self.controller._invoke(lambda: self.jobs_reader(project), loaded)

    def _job_changed(self, *_args):
        root = self.jobs.currentData()
        if root and root != self._output_job:
            self.output.setText(str(Path(root) / "analysis" / "boxes"))
        self._output_job = root
        self.refresh()

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a finished production job",
                                                   str(self.jobs.currentData() or ""))
        if folder:
            index = self.jobs.findData(folder)
            if index < 0:
                self.jobs.addItem(Path(folder).name, folder)
                index = self.jobs.count() - 1
            self.jobs.setCurrentIndex(index)

    def _choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose results folder", self.output.text())
        if folder:
            self.output.setText(folder)

    def start(self):
        if not self.run.isEnabled():
            return False
        base = self.output.text().strip()
        if not base:
            self.status.setText("Choose a results folder first.")
            return False
        config = deepcopy(self.controller._saved_configuration)
        project = self.controller.view.project
        root = str(self.jobs.currentData())
        destination = Path(base) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                    + "-" + uuid4().hex[:8])
        actor = getattr(self.controller.view, "_reviewer", lambda: "")() or "User"
        with_layers = self.layers.isChecked()
        cancel = Event()
        self._cancel = cancel
        self._result = None
        self.results.hide()
        def check_cancelled():
            if cancel.is_set():
                raise InterruptedError("Box mass calculation cancelled")
        def progress(value):
            check_cancelled()
            self.progress.emit(value)
        def calculate():
            options = {}
            if with_layers:
                from .analysis_box_reanalysis import layer_names
                options = dict(layers=True, layer_names=layer_names(project))
            result = self.integrator(root, config, destination, actor=actor,
                                     reason="Recalculate saved analysis boxes",
                                     check_cancelled=check_cancelled, on_progress=progress, **options)
            # Only the small, just-written tables are read for this display;
            # no mass maps are opened a second time.
            rows = []
            tables = result.get("tables", [])
            if isinstance(tables, dict):
                tables = tables.values()
            for bound in tables:
                path = bound["path"] if isinstance(bound, dict) else bound
                with Path(path).open(newline="", encoding="utf-8-sig") as stream:
                    for row in csv.DictReader(stream):
                        rows.append(row)
                        if len(rows) >= 1000:
                            break
                if len(rows) >= 1000:
                    break
            return dict(result, preview_rows=rows)
        def done(result):
            self._cancel = None
            if project is not self.controller.view.project:
                self.status.setText(f"Previous series completed. Results: {destination}")
                return
            self._result = destination
            layers = ""
            if result.get("layer_tables") is not None:
                layers = " Sand layers are in the layers-run tables"
                missing = result.get("frames_without_native_facies") or 0
                layers += (f"; {missing} frames had no native maps with facies." if missing else ".")
            self.status.setText(f"Completed: {result['run_count']} runs, {result['frame_count']} frames. "
                                f"Results: {destination}{layers}")
            self._show_rows(result.get("preview_rows", []))
        self.status.setText("Integrating saved mass maps…")
        started = self.controller._invoke(calculate, done)
        if not started:
            self._cancel = None
        self.refresh()
        return started

    def cancel_requested(self):
        if self._cancel is not None:
            self._cancel.set()
            self.status.setText("Cancellation requested; finishing the current saved-map read…")
            self.cancel.setEnabled(False)

    def _progress(self, value):
        if self._cancel is None or self._cancel.is_set():
            return
        if isinstance(value, dict):
            text = value.get("message") or f"{value.get('run', '')}: {value.get('completed', 0)}/{value.get('total', '?')}"
        else:
            text = str(value)
        self.status.setText(text)

    def _open_result(self):
        if self._result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result)))

    def _show_rows(self, rows):
        self.results.setRowCount(len(rows))
        fields = ("elapsed_h", "total_mass_g", "aqueous_mass_g", "gas_mass_g", "complete")
        for index, row in enumerate(rows):
            values = [f"{row.get('run', '')} / {row.get('name', '')}"]
            values.extend(row.get(field) or "—" for field in fields)
            for column, value in enumerate(values):
                self.results.setItem(index, column, QTableWidgetItem(str(value)))
        self.results.resizeColumnsToContents()
        self.results.setVisible(bool(rows))
        resolutions = sorted({row.get("resolution_mm", "") for row in rows
                              if row.get("storage") == "metric_cells"})
        if resolutions:
            self.status.setText(self.status.text() + " Stored metric cells used at "
                                + ", ".join(resolutions) + " mm; box edges use whole cell centres.")
        if any(str(row.get("complete", "")).casefold() == "false" for row in rows):
            self.status.setText(self.status.text() + " Incomplete boxes have no total; quantified subtotals are in the CSV files.")
        if len(rows) >= 1000:
            self.status.setText(self.status.text() + " Showing the first 1,000 rows; the CSV files contain all rows.")
