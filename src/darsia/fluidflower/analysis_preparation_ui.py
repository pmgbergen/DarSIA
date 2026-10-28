"""Explicit staged plan preparation through the Jobs page's existing call runner."""
from __future__ import annotations

from pathlib import Path
import traceback

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from . import analysis_preparation as preparation
from . import accepted_pressure_policy as pressure_policy
from .analysis_plan import adopt_prepared_plan
from .pipeline.job_queue import descriptor


class AnalysisPreparationDialog(QDialog):
    """Preparing or adopting a plan never queues or starts a calculation."""

    adopted = Signal(object)

    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prepare analysis plan")
        self.series_root = panel.series_root
        self._source_panel = panel
        self._start_call = panel._start_call
        self._thread = self._worker = None
        self.prepared_result = None
        layout = QVBoxLayout(self)
        note = QLabel("Prepare a new plan from the saved image inputs. Review and adoption are separate; no job or worker starts.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self._file_buttons = {}
        self.stage = QComboBox()
        self.stage.addItem("Phase maps and statistics", "phase1")
        self.stage.addItem("Mass from finished phase maps", "phase2")
        form.addRow("Analysis stage", self.stage)
        self.inputs = self._file_row(form, "Saved run inputs")
        self.geometry = self._file_row(form, "Measured geometry index")
        self.pressure_decision_button = QPushButton("Use recorded pressure decision…")
        self.pressure_decision_button.clicked.connect(self.use_recorded_pressure_decision)
        self.pressure = self._file_row(form, "Pressure dataset index",
                                       extra_button=self.pressure_decision_button)
        self.calibration = self._file_row(form, "Fitted mass calibration")
        self.phase1_jobs = self._file_row(form, "Generated phase-map jobs for RAW reproduction")
        self.phase1_jobs.setPlaceholderText("Optional exact run-to-job bindings for the new sealed plan")
        self.collect_button = QPushButton("Collect existing source inputs…")
        self.collect_button.clicked.connect(self.collect_sources)
        form.addRow(self.collect_button)
        self.collect_note = QLabel()
        self.collect_note.setWordWrap(True)
        form.addRow(self.collect_note)
        self.name = QLineEdit("analysis")
        form.addRow("Plan name", self.name)
        self.folder = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(self.folder)
        browse = QPushButton("Choose parent folder…")
        browse.clicked.connect(self._choose_folder)
        row.addWidget(browse)
        form.addRow("Fresh plan folder", row)
        self.probability = QCheckBox("Store exact gas-probability decision values")
        form.addRow(self.probability)
        self.resolution = QLabel()
        self.resolution.setWordWrap(True)
        form.addRow("Resolution and files", self.resolution)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.prepare_button = QPushButton("Prepare plan")
        self.prepare_button.clicked.connect(self.prepare)
        self.adopt_button = QPushButton("Review and adopt prepared plan…")
        self.adopt_button.setEnabled(False)
        self.adopt_button.clicked.connect(self.adopt)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(self.prepare_button)
        buttons.addWidget(self.adopt_button)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.status = QLabel("No plan prepared. Missing per-image roles are reported for each run.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.stage.currentIndexChanged.connect(self._stage_changed)
        self._stage_changed()

    def _file_row(self, form, label, *, extra_button=None):
        edit = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(edit)
        button = QPushButton("Choose…")
        button.clicked.connect(lambda: self._choose_file(edit, label))
        self._file_buttons[edit] = button
        row.addWidget(button)
        if extra_button is not None:
            row.addWidget(extra_button)
        form.addRow(label, row)
        return edit

    def _choose_file(self, edit, label):
        path, _ = QFileDialog.getOpenFileName(self, label, edit.text(), "Saved data (*.json)")
        if path:
            edit.setText(path)

    def _choose_folder(self):
        parent = QFileDialog.getExistingDirectory(self, "Choose a parent for the fresh plan folder")
        if parent:
            self.folder.setText(str(Path(parent) / self.name.text().strip()))

    def _stage_changed(self):
        phase1 = self.stage.currentData() == "phase1"
        for field in (self.geometry, self.pressure, self.probability):
            field.setEnabled(phase1)
            if field in self._file_buttons:
                self._file_buttons[field].setEnabled(phase1)
        self.calibration.setEnabled(not phase1)
        self._file_buttons[self.calibration].setEnabled(not phase1)
        self.phase1_jobs.setEnabled(not phase1)
        self._file_buttons[self.phase1_jobs].setEnabled(not phase1)
        self.collect_button.setEnabled(phase1 and self._thread is None)
        self.pressure_decision_button.setEnabled(phase1 and self._thread is None)
        self.collect_note.setText(
            "Collect only already saved run configurations, complete per-image geometry and running pressure. "
            "This does not calculate or repair missing source products."
            if phase1 else
            "Completion-input collection for this stage is not available here. Select the saved completed phase-map inputs.")
        self.resolution.setText(
            "CPU, float32, native computation. Full native evidence; the pressure and geometry indices bind every selected image."
            if phase1 else
            "CPU, float32. The saved completion inputs bind the storage size and pixel domain; no native processing or phase inference is repeated.")

    def _run(self, call, *, adopting=False, collecting=None, pressure_context=None):
        if self._thread is not None:
            return
        self.prepare_button.setEnabled(False)
        self.adopt_button.setEnabled(False)
        self.collect_button.setEnabled(False)
        self.pressure_decision_button.setEnabled(False)
        self.status.setText("Verifying the recorded pressure decision and unchanged source data…" if pressure_context is not None else
                           "Checking existing source metadata…" if collecting is not None else
                           "Adopting the reviewed plan…" if adopting else "Preparing and validating the plan…")
        def invoke():
            try:
                return dict(value=call(), adopting=adopting, collecting=collecting,
                            pressure_context=pressure_context)
            except Exception:
                return dict(error=traceback.format_exc(), adopting=adopting)
        self._thread, self._worker = self._start_call(invoke, self._done, self._closed)

    @Slot(object)
    def _done(self, result):
        if "error" in result:
            self.status.setText(result["error"])
            return
        if result.get("pressure_context") is not None:
            accepted = result["value"]
            path = accepted["pressure_index"]["path"]
            if result["pressure_context"] != self._pressure_context():
                self.status.setText("The recorded pressure policy was saved for earlier choices: " + path
                    + "\nThe current pressure selection was not changed. Choose the saved file explicitly if appropriate.")
                return
            self.pressure.setText(path)
            self.status.setText("Applied the existing recorded pressure decision for "
                + ", ".join(accepted["runs"]) + ": " + path
                + "\nOriginal observations and numeric values are unchanged. No plan, adoption, job or worker was created. Prepare repeats the original checks.")
            return
        if result.get("collecting") is not None:
            collected = result["value"]
            if result["collecting"] != self._source_context():
                self.status.setText("Source inputs were saved for earlier choices: " + collected["inputs"]["path"]
                    + "\nThe current selection was not changed. Choose the saved file explicitly if appropriate.")
                return
            self.inputs.setText(collected["inputs"]["path"])
            self.status.setText("Collected existing inputs for " + ", ".join(collected["runs"])
                + ": " + collected["inputs"]["path"]
                + "\nNo geometry, pressure, optics, plan, approval, job or worker was created. Prepare repeats the original source checks.")
            return
        self.prepared_result = result["value"]
        bound = self.prepared_result["plan"]
        if result["adopting"]:
            self.status.setText("Plan adopted: " + bound["path"] + "\nNo job or worker started.")
            self.adopted.emit(bound)
        else:
            self.status.setText("Prepared, not adopted: " + bound["path"] + "\nSHA-256: " + bound["sha256"] + "\nNo job or worker started.")

    @Slot()
    def _closed(self):
        if self._thread is not None:
            self._thread.wait()
        self._thread = self._worker = None
        self.prepare_button.setEnabled(True)
        self.adopt_button.setEnabled(bool(self.prepared_result and not self.prepared_result.get("adopted_for_execution")))
        self._stage_changed()

    def _source_context(self):
        return (str(self.series_root), str(self._source_panel.series_root), self.stage.currentData(),
                self.geometry.text().strip(), self.pressure.text().strip(),
                self.probability.isChecked(), self.inputs.text().strip())

    def collect_sources(self):
        """Assemble metadata only; keep heavy source production outside the GUI."""
        if self._thread is not None or self.stage.currentData() != "phase1":
            return
        if not self.series_root or not self.geometry.text().strip() or not self.pressure.text().strip():
            self.status.setText("Choose the registered series and existing complete geometry and running-pressure indices first.")
            return
        if (self._source_panel.series_root is None
                or Path(self._source_panel.series_root).resolve() != Path(self.series_root).resolve()):
            self.status.setText("The selected series changed. Reopen plan preparation for the current series.")
            return
        context = self._source_context()
        paths, _ = QFileDialog.getOpenFileNames(self,
            "Choose existing complete run configurations in registered run order", "", "Source metadata (*.json)")
        if not paths:
            return
        plan_folder = Path(self.folder.text().strip()) if self.folder.text().strip() else None
        initial = str(plan_folder.parent / (plan_folder.name + "-source-inputs.json")) if plan_folder else ""
        destination, _ = QFileDialog.getSaveFileName(self,
            "Choose a fresh source-input manifest outside the plan folder", initial, "Source metadata (*.json)")
        if not destination:
            return
        if context != self._source_context():
            self.status.setText("Source choices changed while choosing files. Collect them again for the current choices.")
            return
        if plan_folder is not None and Path(destination).resolve().is_relative_to(plan_folder.resolve()):
            self.status.setText("Choose the source-input manifest outside the fresh plan folder.")
            return
        series_root = self.series_root
        geometry, pressure = self.geometry.text().strip(), self.pressure.text().strip()
        probability = self.probability.isChecked()
        self.prepared_result = None
        def collect():
            configurations = [preparation.source_metadata_descriptor(path) for path in paths]
            return preparation.collect_phase1_source_inputs(configurations, destination,
                series_root=series_root, geometry_index=preparation.source_metadata_descriptor(geometry),
                pressure_index=preparation.source_metadata_descriptor(pressure), gas_probability=probability)
        self._run(collect, collecting=context)

    def _pressure_context(self):
        return self._source_context(), self.folder.text().strip()

    def use_recorded_pressure_decision(self):
        """Apply only a verified existing decision, through the owned finite call."""
        if self._thread is not None or self.stage.currentData() != "phase1":
            return
        if not self.series_root or not self.pressure.text().strip():
            self.status.setText("Choose the registered series and existing running-pressure index first.")
            return
        if (self._source_panel.series_root is None
                or Path(self._source_panel.series_root).resolve() != Path(self.series_root).resolve()):
            self.status.setText("The selected series changed. Reopen plan preparation for the current series.")
            return
        context = self._pressure_context()
        plan_folder = Path(self.folder.text().strip()) if self.folder.text().strip() else None
        authority, _ = QFileDialog.getOpenFileName(self,
            "Choose the existing recorded pressure decision", "", "Recorded decision (*.json)")
        if not authority:
            return
        initial = str(Path(self.series_root) / "accepted-pressure-index.json")
        destination, _ = QFileDialog.getSaveFileName(self,
            "Choose a fresh pressure-policy index in a registered input root, outside the plan folder", initial, "Pressure metadata (*.json)")
        if not destination:
            return
        if context != self._pressure_context():
            self.status.setText("Source choices changed while choosing files. Apply the recorded decision again for the current choices.")
            return
        destination_path = Path(destination)
        evidence_folder = pressure_policy.pressure_policy_sources_path(destination_path)
        if plan_folder is not None and any(path.resolve().is_relative_to(plan_folder.resolve())
                                           for path in (destination_path, evidence_folder)):
            self.status.setText("Choose the pressure-policy index outside the fresh plan folder.")
            return
        pressure, series_root = self.pressure.text().strip(), self.series_root
        self.prepared_result = None
        def apply_recorded():
            return pressure_policy.prepare_accepted_pressure_index(
                preparation.source_metadata_descriptor(pressure), destination,
                authority=preparation.source_metadata_descriptor(authority), series_root=series_root)
        self._run(apply_recorded, pressure_context=context)

    def prepare(self):
        phase1 = self.stage.currentData() == "phase1"
        required = [self.inputs, self.folder, self.name]
        required += [self.geometry, self.pressure] if phase1 else [self.calibration]
        if not self.series_root or any(not field.text().strip() for field in required):
            self.status.setText("Choose the registered series, saved inputs, required sources and a fresh plan folder.")
            return
        inputs, folder, job_id = self.inputs.text().strip(), self.folder.text().strip(), self.name.text().strip()
        geometry, pressure, calibration = (field.text().strip() for field in (self.geometry, self.pressure, self.calibration))
        probability, series_root = self.probability.isChecked(), self.series_root
        phase1_jobs = self.phase1_jobs.text().strip()
        repository = Path(__file__).resolve().parents[3]
        self.prepared_result = None
        def call():
            job = preparation.job_identity(repository)
            if phase1:
                return preparation.prepare_phase1_inputs(descriptor(Path(inputs)), folder,
                    series_root=series_root, geometry_index=descriptor(Path(geometry)),
                    pressure_index=descriptor(Path(pressure)), gas_probability=probability,
                    job_id=job_id, job=job)
            return preparation.prepare_phase2_inputs(descriptor(Path(inputs)), folder,
                series_root=series_root, fitted_calibration=descriptor(Path(calibration)), job_id=job_id, job=job,
                phase1_jobs=preparation.source_metadata_descriptor(phase1_jobs) if phase1_jobs else None)
        self._run(call)

    def adopt(self):
        if not self.prepared_result or self._thread is not None or self.prepared_result.get("adopted_for_execution"):
            return
        bound = dict(self.prepared_result["plan"])
        actor, accepted = QInputDialog.getText(self, "Review approval", "Approved by")
        if not accepted or not actor.strip():
            return
        reason, accepted = QInputDialog.getText(self, "Review approval", "Reason for adopting these exact inputs")
        if not accepted or not reason.strip():
            return
        answer = QMessageBox.question(self, "Adopt prepared analysis plan?",
            bound["path"] + "\nSHA-256: " + bound["sha256"] +
            "\nAdopt exactly this plan? This creates no job and starts no worker.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        source = Path(bound["path"])
        self._run(lambda: adopt_prepared_plan(source, source.with_name("execution.adopted.toml"),
            expected_sha256=bound["sha256"], actor=actor.strip(), reason=reason.strip()), adopting=True)

    def can_close(self):
        """Keep the original preparation/adoption callback alive until retirement."""
        return self._thread is None

    def closeEvent(self, event):
        if not self.can_close():
            self.status.setText("Plan preparation or adoption is still finishing; close after it returns.")
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):
        if self.can_close():
            super().reject()
        else:
            self.status.setText("Plan preparation or adoption is still finishing; close after it returns.")
