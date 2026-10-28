"""Build published dataset: the GUI twin of ``darsia dataset build``.

The dialog collects the same choices as the command line, checks first and
builds only on request. Both run in a worker thread; the report is shown in
the dialog. Nothing is written into the series or the job folder.
"""

from __future__ import annotations

from pathlib import Path
import traceback

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget)

GIB = 1024**3


class BuildThread(QThread):
    """Run one check or build off the GUI thread; emit the report or a traceback."""

    finished_report = Signal(object, str)

    def __init__(self, request, check_only: bool, parent=None):
        """Keep the build request; ``check_only`` checks without writing."""
        super().__init__(parent)
        self.request, self.check_only = request, check_only

    def run(self):
        """Check (or build) and emit ``(report, "")`` or ``(None, traceback)``."""
        from .builder import build_record, check_build

        try:
            report = check_build(self.request)[1] if self.check_only else build_record(self.request)
        except Exception:  # noqa: BLE001 - the dialog shows the traceback verbatim
            self.finished_report.emit(None, traceback.format_exc())
            return
        self.finished_report.emit(report, "")


def _path_row(edit: QLineEdit, button: QPushButton) -> QWidget:
    """One form row: a path field and its Browse button."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return row


class BuildDatasetDialog(QDialog):
    """Export media, a reproducible data package, or numerical-model inputs."""

    def __init__(self, series_root, parent=None):
        """Lay out the choices for the series in ``series_root``; nothing runs yet."""
        super().__init__(parent)
        self.setWindowTitle("Results and export")
        self.series_root = Path(series_root)
        self.thread = None
        self.media_dialog = None
        layout = QVBoxLayout(self)
        note = QLabel("Export films and figures for the series, a complete reproduction package, or a "
                      "simulator grid with mass maps. The saved image selection stays unchanged. "
                      "Export maps may be coarser than the stored analysis; original results are retained. "
                      "Check explains missing inputs before Build writes a new package.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form_body = QWidget()
        form = QFormLayout(form_body)
        form.addRow("Series folder", QLabel(str(self.series_root)))
        self.job_edit = QLineEdit()
        self.job_edit.setPlaceholderText("The finished production job's folder")
        job_button = QPushButton("Browse…")
        job_button.clicked.connect(lambda: self._choose_folder(self.job_edit, "Production job folder"))
        form.addRow("Production job", _path_row(self.job_edit, job_button))
        self.record_combo = QComboBox()
        self.record_combo.addItem("Full data package to reproduce the results", "R")
        self.record_combo.addItem("Numerical models: simulator grid and mass maps", "S")
        self.record_combo.addItem("Films and figures for the experiment series", "media")
        self.record_combo.currentIndexChanged.connect(self._record_changed)
        form.addRow("Export type", self.record_combo)
        self.mass_resolution_combo = QComboBox()
        for resolution in (1, 2, 5, 10):
            self.mass_resolution_combo.addItem(f"{resolution} mm", resolution)
        self.mass_resolution_combo.setToolTip(
            "Stored resolution or coarser. Coarser maps use the saved extensive cells and conserve mass; "
            "missing physical fields cannot be filled by interpolation.")
        form.addRow("Export mass-map resolution", self.mass_resolution_combo)
        self.resolution_edit = QLineEdit()
        self.resolution_edit.setPlaceholderText("Required for numerical models; no default")
        form.addRow("Simulator grid resolution (mm)", self.resolution_edit)
        self.geometry_edit = QLineEdit()
        self.geometry_edit.setPlaceholderText("Required: bound geometry, fixtures and measured depth")
        self.geometry_button = QPushButton("Browse…")
        self.geometry_button.clicked.connect(
            lambda: self._choose_file(self.geometry_edit, "Simulator geometry request"))
        form.addRow("Simulator geometry request", _path_row(self.geometry_edit, self.geometry_button))
        self.software_combo = QComboBox()
        self.software_combo.addItem("Point to exact code and library versions", "references")
        self.software_combo.addItem("Include all code and locked libraries", "include")
        form.addRow("Reproduction software", self.software_combo)
        self.software_folder_edit = QLineEdit()
        self.software_folder_edit.setPlaceholderText("Folder containing the producing job's unchanged environment lock")
        self.software_folder_button = QPushButton("Browse…")
        self.software_folder_button.clicked.connect(
            lambda: self._choose_folder(self.software_folder_edit, "Producing job's software files"))
        form.addRow("Software files", _path_row(self.software_folder_edit, self.software_folder_button))
        self.software_code_edit = QLineEdit()
        self.software_code_edit.setPlaceholderText("HTTPS address of the code repository; the job supplies the exact commit")
        form.addRow("Code reference", self.software_code_edit)
        self.software_python_edit = QLineEdit()
        self.software_python_edit.setPlaceholderText("Exact version used for the job, for example 3.13.9")
        form.addRow("Python version", self.software_python_edit)
        self.software_help = QLabel(
            "The environment lock lists the exact library versions used for the results. "
            "To include software, put the complete source.bundle and a wheels folder with one build "
            "of each locked library in the chosen folder. Check verifies these files; nothing is installed.")
        self.software_help.setWordWrap(True)
        form.addRow(self.software_help)
        self.software_edit = QLineEdit()
        self.software_edit.setPlaceholderText("Optional: an existing inventory instead of the fields above")
        self.software_button = QPushButton("Browse…")
        self.software_button.clicked.connect(
            lambda: self._choose_file(self.software_edit, "Software inventory"))
        form.addRow("Existing software inventory", _path_row(self.software_edit, self.software_button))
        self.stage_software_edit = QLineEdit()
        self.stage_software_edit.setPlaceholderText("For staged N1: exact inventory for every original P1 and P2 job")
        self.stage_software_button = QPushButton("Browse…")
        self.stage_software_button.clicked.connect(
            lambda: self._choose_file(self.stage_software_edit, "Original software inventories by stage"))
        form.addRow("Staged software inventories", _path_row(self.stage_software_edit, self.stage_software_button))
        self.producer_metadata_check = QCheckBox(
            "Retain original producer receipts, including their local paths and producer identifiers")
        self.producer_metadata_check.setToolTip(
            "Staged RAW reproduction preserves immutable evidence. This creates a package containing "
            "original producer metadata; the check report lists the retained findings.")
        form.addRow(self.producer_metadata_check)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("A new or empty folder outside the series")
        output_button = QPushButton("Browse…")
        output_button.clicked.connect(lambda: self._choose_folder(self.output_edit, "Output folder"))
        form.addRow("Output folder", _path_row(self.output_edit, output_button))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Default: the series name")
        form.addRow("Title", self.title_edit)
        self.memory_edit = QLineEdit()
        self.memory_edit.setPlaceholderText("Optional, measured on the reference machine")
        form.addRow("Peak memory per worker (GiB)", self.memory_edit)
        self.memory_basis_edit = QLineEdit()
        self.memory_basis_edit.setPlaceholderText("How it was measured")
        form.addRow("Peak memory basis", self.memory_basis_edit)
        self.proof_edit = QLineEdit()
        proof_button = QPushButton("Browse…")
        proof_button.clicked.connect(lambda: self._choose_file(self.proof_edit, "Proof record"))
        form.addRow("Revision proof (optional)", _path_row(self.proof_edit, proof_button))
        self.variation_edit = QLineEdit()
        variation_button = QPushButton("Browse…")
        variation_button.clicked.connect(lambda: self._choose_file(self.variation_edit, "Known variation"))
        form.addRow("Known variation (optional)", _path_row(self.variation_edit, variation_button))
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setWidget(form_body)
        form_scroll.setMinimumSize(640, 260)
        form_scroll.setMaximumHeight(500)
        layout.addWidget(form_scroll, 1)
        buttons = QHBoxLayout()
        self.check_button = QPushButton("Check")
        self.check_button.clicked.connect(lambda: self.start(check_only=True))
        self.build_button = QPushButton("Build")
        self.build_button.clicked.connect(lambda: self.start(check_only=False))
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        for button in (self.check_button, self.build_button, close_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setMinimumHeight(220)
        layout.addWidget(self.report_view, 1)
        self.last_report = None
        self._record_changed()
        self.resize(900, 850)

    def _record_changed(self, *_):
        """Show controls for the chosen export, keeping resolutions independent."""
        simulator = self.record_combo.currentData() == "S"
        reproduction = self.record_combo.currentData() == "R"
        media = self.record_combo.currentData() == "media"
        for widget in (self.resolution_edit, self.geometry_edit, self.geometry_button):
            widget.setEnabled(simulator)
        for widget in (self.software_combo, self.software_edit, self.software_button,
                       self.software_folder_edit, self.software_folder_button,
                       self.software_code_edit, self.software_python_edit, self.software_help,
                       self.stage_software_edit, self.stage_software_button, self.producer_metadata_check):
            widget.setEnabled(reproduction)
        self.mass_resolution_combo.setEnabled(not media)
        self.check_button.setEnabled(not media)
        self.build_button.setText("Open films and figures…" if media else "Build")

    def _choose_folder(self, edit: QLineEdit, title: str):
        """Let the user pick a folder for ``edit``."""
        folder = QFileDialog.getExistingDirectory(self, title, edit.text() or str(self.series_root))
        if folder:
            edit.setText(folder)

    def _choose_file(self, edit: QLineEdit, title: str):
        """Let the user pick a JSON file for ``edit``."""
        path, _filter = QFileDialog.getOpenFileName(self, title, edit.text() or str(self.series_root),
                                                    "JSON (*.json);;All files (*)")
        if path:
            edit.setText(path)

    def request(self):
        """The build request of the dialog's fields; raise ValueError for an invalid field."""
        from .builder import BuildRequest

        if not self.job_edit.text().strip() or not self.output_edit.text().strip():
            raise ValueError("Choose the production job and an output folder")
        record = self.record_combo.currentData()
        if record == "media":
            raise ValueError("Open the films and figures viewer for media exports")
        resolution = None
        if record == "S":
            if not self.resolution_edit.text().strip():
                raise ValueError("A simulator record needs an explicit resolution; there is no default")
            resolution = float(self.resolution_edit.text())
        memory = None
        if self.memory_edit.text().strip():
            memory = int(round(float(self.memory_edit.text()) * GIB))

        def optional(edit):
            """The field's path, or None when it is empty."""
            return Path(edit.text().strip()) if edit.text().strip() else None

        return BuildRequest(series_root=self.series_root, job_root=Path(self.job_edit.text().strip()),
                            record=record, tier=None,
                            output=Path(self.output_edit.text().strip()), title=self.title_edit.text().strip(),
                            proof=optional(self.proof_edit), known_variation=optional(self.variation_edit),
                            peak_memory_bytes=memory, peak_memory_basis=self.memory_basis_edit.text().strip(),
                            simulator_resolution_mm=resolution,
                            simulator_geometry=optional(self.geometry_edit) if record == "S" else None,
                            require_simulator_geometry=record == "S",
                            expected_resolution_mm=None,
                            mass_map_resolution_mm=self.mass_resolution_combo.currentData(),
                            software_mode=self.software_combo.currentData() if record == "R" else None,
                            software_inventory=optional(self.software_edit) if record == "R" else None,
                            software_stage_inventory=optional(self.stage_software_edit) if record == "R" else None,
                            retain_producer_metadata=record == "R" and self.producer_metadata_check.isChecked(),
                            software_folder=optional(self.software_folder_edit) if record == "R" else None,
                            software_code_url=self.software_code_edit.text().strip() if record == "R" else "",
                            software_python=self.software_python_edit.text().strip() if record == "R" else "")

    def start(self, *, check_only: bool) -> bool:
        """Start a check or a build in a worker thread; False when a field is invalid or one runs."""
        if self.thread is not None:
            return False
        if self.record_combo.currentData() == "media":
            return self.open_media()
        try:
            request = self.request()
        except ValueError as error:
            self.report_view.setPlainText(str(error))
            return False
        self.report_view.setPlainText("Checking…" if check_only else "Checking, then building…")
        for button in (self.check_button, self.build_button):
            button.setEnabled(False)
        self.thread = BuildThread(request, check_only, self)
        self.thread.finished_report.connect(self._finished)
        self.thread.start()
        return True

    def open_media(self):
        """Use the original finished-job viewer and its export functions."""
        from ..production_results_view import ProductionResultsPage

        if not self.job_edit.text().strip():
            self.report_view.setPlainText("Choose the finished production job first")
            return False
        if self.media_dialog is None:
            self.media_dialog = QDialog(self)
            self.media_dialog.setWindowTitle("Series films and figures")
            layout = QVBoxLayout(self.media_dialog)
            self.media_page = ProductionResultsPage(self.media_dialog)
            layout.addWidget(self.media_page)
            self.media_dialog.resize(1200, 900)
        self.media_page.open_job(Path(self.job_edit.text().strip()))
        self.media_dialog.show()
        self.media_dialog.raise_()
        return True

    def _finished(self, report, error: str):
        """Show the report (or the traceback) and offer Check and Build again."""
        self.last_report = report
        self.report_view.setPlainText(error or report.text())
        for button in (self.check_button, self.build_button):
            button.setEnabled(True)
        self._record_changed()
        thread, self.thread = self.thread, None
        if thread is not None:
            thread.wait()
            thread.deleteLater()

    def busy(self) -> bool:
        """True while a check or build runs."""
        return self.thread is not None

    def closeEvent(self, event):
        """Wait for a running check or build before the dialog closes."""
        if self.thread is not None:
            self.thread.wait()
        super().closeEvent(event)
