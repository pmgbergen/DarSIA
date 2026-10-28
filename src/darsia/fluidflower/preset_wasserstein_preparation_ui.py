"""Explicit original Comparison preparation using the existing Jobs call owner."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
import traceback
import uuid

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)


# Where comparison jobs' metadata is prepared: the series' [paths]
# comparison_inputs, else comparisons/inputs in its results folder.
INPUTS_KEY = "comparison_inputs"
NO_SHARED_JOBS = "No shared Jobs folder: set it under Jobs > Shared job folder."


def comparison_inputs_folder(series_root) -> Path:
    """The folder where the series' comparison jobs keep their metadata."""
    from .series_paths import series_folder_path
    try:
        return series_folder_path(series_root, INPUTS_KEY)
    except KeyError:
        return series_folder_path(series_root, "results").joinpath("comparisons", "inputs")


class PresetWassersteinPreparationPage(QScrollArea):
    """Create metadata for the original preset; worker launch stays in Jobs."""

    job_created = Signal(object)
    jobs_requested = Signal()

    def __init__(self, panel, parent=None, *, preparer=None, register_job=None):
        super().__init__(parent)
        self.panel, self.preparer = panel, preparer
        self._register_job = register_job
        self._thread = self._worker = None
        self.series_root = self.series_id = None
        self.registration = None
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        layout = QVBoxLayout(body)
        title = QLabel("Wasserstein comparison")
        layout.addWidget(title)
        note = QLabel(
            "Choose the original Comparison configuration and existing mass results. "
            "Its runs, times, ROIs and scientific settings are preserved. Prepare and create job "
            "writes metadata only. Start separately in Jobs on this computer or a registered machine. "
            "Each Start uses one CPU worker for one admitted requested time."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.series_label = QLabel()
        self.series_label.setWordWrap(True)
        self.series_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.series_label)
        self.form = QWidget()
        form = QFormLayout(self.form)
        self.config_path = self._path_row(form, "Original Comparison config")
        self.job_id = QLineEdit("wasserstein")
        form.addRow("Job name", self.job_id)
        # Both folders are named from the series and made when the job is created.
        self.metadata_folder = self._shown_row(
            form, "Fresh metadata folder",
            "Made in the series' comparison inputs folder ([paths] comparison_inputs; by default "
            "comparisons/inputs in its results folder) when the job is created.")
        self.job_root = self._shown_row(
            form, "Fresh shared job folder",
            "Made in the series' Jobs folder (Jobs > Shared job folder) when the job is created.")
        self._suggested: set[str] = set()
        from .series_geometry import default_actor

        self.actor = QLineEdit(default_actor())
        self.reason = QLineEdit()
        form.addRow("Created by", self.actor)
        form.addRow("Reason", self.reason)
        form.addRow("Computation", QLabel("CPU · original input and solver types"))
        layout.addWidget(self.form)
        actions = QHBoxLayout()
        self.prepare_button = QPushButton("Prepare and create job")
        self.prepare_button.clicked.connect(self.prepare)
        self.show_jobs_button = QPushButton("Show job and workers")
        self.show_jobs_button.clicked.connect(self.jobs_requested)
        actions.addWidget(self.prepare_button)
        actions.addWidget(self.show_jobs_button)
        actions.addStretch()
        layout.addLayout(actions)
        self.status = QLabel("No comparison job created.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        layout.addStretch()
        self.set_context(None, None)

    def _path_row(self, form, label):
        edit = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(edit)
        button = QPushButton("Choose…")
        button.clicked.connect(lambda: self._choose_config(edit, label))
        row.addWidget(button)
        form.addRow(label, row)
        return edit

    def _choose_config(self, edit, label):
        value, _ = QFileDialog.getOpenFileName(self, label, edit.text(), "Comparison config (*.toml)")
        if value:
            edit.setText(value)

    def _shown_row(self, form, label, tip):
        """A folder the page names from the series; shown, not typed."""
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setToolTip(tip)
        form.addRow(label, edit)
        return edit

    def suggest_folders(self, *, fresh=False):
        """Name a fresh ``wasserstein-<id>`` job folder and its metadata folder.

        The job folder is in the series' shared (UNC) Jobs folder, the metadata
        folder of the same name in its comparison inputs folder
        (``comparison_inputs_folder``). They are only named here; Prepare
        makes and checks them as before. Without a shared Jobs folder both stay
        empty and the page says where that folder is set; no other location is
        assumed. ``fresh`` names new folders even when the shown ones still
        belong to the series.
        """
        from .series_paths import series_folder_path
        jobs = inputs = None
        if self.series_root is not None:
            try:
                jobs = PureWindowsPath(series_folder_path(self.series_root, "jobs"))
                inputs = Path(comparison_inputs_folder(self.series_root))
            except (OSError, ValueError, KeyError):
                jobs = inputs = None
        if jobs is None or not jobs.drive.startswith("\\\\"):
            for edit in (self.job_root, self.metadata_folder):
                if edit.text().strip() in self._suggested:
                    edit.clear()
            self._suggested = set()
            self.job_root.setPlaceholderText(NO_SHARED_JOBS)
            self.metadata_folder.setPlaceholderText("Named together with the job folder.")
            return
        current = self.job_root.text().strip()
        if fresh or not current or str(PureWindowsPath(current).parent).casefold() != str(jobs).casefold():
            name = "wasserstein-" + uuid.uuid4().hex
            job_root, metadata = str(jobs / name), str(inputs / name)
            self.job_root.setText(job_root)
            self.metadata_folder.setText(metadata)
            self._suggested = {job_root, metadata}

    def set_context(self, series_root, series_id, *, series_label=""):
        previous = (self.series_root, self.series_id)
        self.series_root = Path(series_root) if series_root else None
        self.series_id = series_id
        if previous != (self.series_root, self.series_id):
            self.registration = None
            if self._thread is None:
                self.status.setText("No comparison job created for this series.")
        self.suggest_folders()
        self.series_label.setText(
            f"Registered series: {series_label or series_id} · {series_id}\n{self.series_root}"
            if self.series_root and series_id else
            "Choose a registered series on Start before preparing a comparison."
        )
        self._sync_actions()

    def _sync_actions(self):
        idle = self._thread is None
        self.form.setEnabled(idle)
        self.prepare_button.setEnabled(idle and bool(self.series_root and self.series_id))
        self.show_jobs_button.setEnabled(idle and self.registration is not None)

    @Slot()
    def prepare(self):
        if self._thread is not None:
            return
        fields = (self.config_path, self.metadata_folder, self.job_root,
                  self.job_id, self.actor, self.reason)
        if self.series_root and self.series_id and not self.job_root.text().strip():
            self.status.setText(NO_SHARED_JOBS)
            return
        if not self.series_root or not self.series_id or any(not edit.text().strip() for edit in fields):
            self.status.setText("Choose a registered series, original config, fresh folders, job name, your name and a reason.")
            return
        config, metadata, output, job_id, actor, reason = (edit.text().strip() for edit in fields)
        series_root, series_id = self.series_root, self.series_id
        repository = Path(__file__).resolve().parents[3]
        preparer = self.preparer
        self.registration = None
        self.prepare_button.setEnabled(False)
        self.show_jobs_button.setEnabled(False)
        self.form.setEnabled(False)
        self.status.setText("Checking original Comparison inputs and creating the job metadata…")

        def create():
            try:
                operation = preparer
                if operation is None:
                    from .preset_wasserstein_jobs import prepare_preset_wasserstein_job
                    operation = prepare_preset_wasserstein_job
                registration = operation(config, metadata, output, series_root=series_root,
                    series_id=series_id, job_id=job_id, actor=actor, reason=reason,
                    repository=repository)
                return dict(registration=dict(registration, ui_series_root=str(series_root),
                                              ui_series_id=series_id))
            except Exception:
                return dict(error=traceback.format_exc())

        try:
            self._thread, self._worker = self.panel._start_call(create, self._ready, self._closed)
        except Exception:
            self.status.setText(traceback.format_exc())
            self._sync_actions()

    @Slot(object)
    def _ready(self, result):
        if "error" in result:
            self.status.setText(result["error"])
            return
        registration = result["registration"]
        # Registration always retains the original series, even if the user
        # navigated elsewhere while this one finite metadata call was running.
        if self._register_job is not None and not self._register_job(registration):
            self.status.setText("Job metadata was created at " + registration["root"]
                + ", but local registration failed. See the Jobs error above and use Open existing job to register it. No worker started.")
            return
        self.job_created.emit(registration)
        if (Path(registration["ui_series_root"]), registration["ui_series_id"]) != (
                self.series_root, self.series_id):
            self.status.setText("Job created for the earlier series " + registration["ui_series_id"]
                + ": " + registration["root"] + "\nReturn to that series to choose its workers. No worker started.")
            return
        self.registration = registration
        self.suggest_folders(fresh=True)  # A next job needs fresh folders of its own.
        self.status.setText(
            f"Job created: {registration['root']}\n"
            f"Requested comparison records: {registration['comparison_count']}. "
            f"Reference run: {registration['reference_run']}.\n"
            "Choose Show job and workers to select the machine, then explicitly start one CPU worker. "
            "Each service handles one admitted requested time; remaining times stay queued. No worker started."
        )

    @Slot()
    def _closed(self):
        if self._thread is not None:
            self._thread.wait()
        self._thread = self._worker = None
        self._sync_actions()

    def can_close(self):
        return self._thread is None

    def closeEvent(self, event):
        if not self.can_close():
            self.status.setText("Comparison job preparation is still finishing; close after it returns.")
            event.ignore()
            return
        super().closeEvent(event)


class PresetWassersteinExportPanel(QWidget):
    """Explicit original CSV export, retaining the existing finite-call owner."""

    def __init__(self, panel, parent=None, *, exporter=None):
        super().__init__(parent)
        self.panel, self.exporter = panel, exporter
        self._thread = self._worker = None
        self.current = None
        self._messages = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        note = QLabel("Export the original comparison CSV after every requested time is published. Original values and NaNs are retained.")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.destination = QLineEdit()
        self.destination.setPlaceholderText("Fresh local export folder")
        self.choose_button = QPushButton("Choose parent…")
        self.choose_button.clicked.connect(self._choose_destination)
        row.addWidget(self.destination)
        row.addWidget(self.choose_button)
        layout.addLayout(row)
        self.export_button = QPushButton("Export original comparison CSV")
        self.export_button.clicked.connect(self.export)
        layout.addWidget(self.export_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.set_snapshot(None)

    def set_snapshot(self, value):
        previous = (self.current or {}).get("root")
        self.current = value if ((value or {}).get("job_facts") or {}).get("pipeline") == "preset-wasserstein" else None
        root = (self.current or {}).get("root")
        if root != previous:
            self.destination.clear()
        self.setVisible(self.current is not None)
        self.status.setText(self._messages.get(root, "No comparison CSV exported."))
        self._sync_actions()

    def _sync_actions(self):
        ready = bool(self.current and self.current.get("complete") and not self.current.get("errors"))
        idle = self._thread is None
        self.destination.setEnabled(idle)
        self.choose_button.setEnabled(idle)
        self.export_button.setEnabled(idle and ready)
        self.export_button.setToolTip("" if ready else "Every requested time must have a verified shared completion before export.")

    def _choose_destination(self):
        parent = QFileDialog.getExistingDirectory(self, "Choose local parent for a fresh comparison export")
        if parent:
            name = Path((self.current or {}).get("root", "comparison")).name
            self.destination.setText(str(Path(parent) / (name + "-export")))

    @Slot()
    def export(self):
        if self._thread is not None:
            return
        if not self.current or not self.current.get("complete") or self.current.get("errors"):
            return
        destination = self.destination.text().strip()
        if not destination:
            self.status.setText("Choose a fresh local export folder first.")
            return
        root, exporter = self.current["root"], self.exporter
        self._messages[root] = "Verifying shared comparison results and exporting the original CSV…"
        self.status.setText(self._messages[root])
        self.export_button.setEnabled(False)
        self.destination.setEnabled(False)
        self.choose_button.setEnabled(False)

        def run():
            try:
                operation = exporter
                if operation is None:
                    from .preset_wasserstein_export import export_preset_wasserstein_csv
                    operation = export_preset_wasserstein_csv
                return dict(root=root, result=operation(root, destination))
            except Exception:
                return dict(root=root, error=traceback.format_exc())

        try:
            self._thread, self._worker = self.panel._start_call(run, self._ready, self._closed)
        except Exception:
            self._messages[root] = traceback.format_exc()
            self.status.setText(self._messages[root])
            self._sync_actions()

    @Slot(object)
    def _ready(self, value):
        if "error" in value:
            message = "Comparison CSV export failed.\n" + value["error"]
        else:
            result = value["result"]
            message = ("Original comparison CSV exported: " + result["csv"]["path"]
                + "\nSource receipt: " + result["receipt"]["path"]
                + f"\nComparison records: {result['comparison_count']}. Nonfinite distances: {result['nonfinite_distances']}.")
        self._messages[value["root"]] = message
        if (self.current or {}).get("root") == value["root"]:
            self.status.setText(message)

    @Slot()
    def _closed(self):
        if self._thread is not None:
            self._thread.wait()
        self._thread = self._worker = None
        self._sync_actions()

    def can_close(self):
        return self._thread is None
