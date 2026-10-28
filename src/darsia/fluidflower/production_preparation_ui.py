"""Prepare an inert production plan using the explicitly selected calibration."""
from copy import deepcopy
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel

from . import analysis_preparation as preparation
from .analysis_preparation_ui import AnalysisPreparationDialog
from .jobs_panel import stored_resolution_choices
from .pipeline.job_queue import descriptor
from .production_selection import read_named_calibration
from .storage_profile import new_job_profile


class ProductionPreparationDialog(AnalysisPreparationDialog):
    """Use the original preparation/adoption lifecycle and original phase-two seal."""

    def __init__(self, panel, calibration, parent=None, *, storage_profile=None):
        self._production_calibration = deepcopy(calibration)
        self._active_context = self._prepared_context = None
        super().__init__(panel, parent)
        self.setWindowTitle("Prepare production plan")
        self.stage.setCurrentIndex(self.stage.findData("phase2"))
        self.stage.setEnabled(False)
        self.calibration.setText(calibration["calibration"]["path"])
        self.calibration.setReadOnly(True)
        self._file_buttons[self.calibration].setEnabled(False)
        self.profile = QComboBox()
        self.profile.addItem("Full", "full")
        self.profile.addItem("Lean", "lean")
        self.profile.setCurrentIndex(self.profile.findData(new_job_profile(storage_profile)))
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("File profile for the new plan"))
        profile_row.addWidget(self.profile)
        self.layout().insertLayout(2, profile_row)
        self.stored_resolution = QComboBox()
        self.stored_resolution.addItem("Keep saved storage sizes", "saved")
        for step in stored_resolution_choices(None):
            self.stored_resolution.addItem("native" if step is None else f"{step:g} mm",
                                           "native" if step is None else step)
        resolution_row = QHBoxLayout()
        resolution_row.addWidget(QLabel("Stored at for the new plan"))
        resolution_row.addWidget(self.stored_resolution)
        self.layout().insertLayout(3, resolution_row)
        self.stored_resolution.setToolTip(
            "Coarse storage requires the physical extent already recorded in each run's saved inputs. "
            "Computation remains native; the saved sources and current adopted plan stay fixed.")
        self.name.setText("production")
        self.status.setText(
            f"Mass calibration: {calibration['name']} · {calibration['chemistry_label']}. "
            "Choose the saved phase-one completion inputs and a fresh plan folder. "
            "The chosen file profile is sealed into a new plan for review; "
            "the saved sources, anchors and precision are retained. "
            "Stored sizes stay as saved unless you choose another supported size.")
        self.profile.currentIndexChanged.connect(self._profile_changed)
        self.stored_resolution.currentIndexChanged.connect(self._profile_changed)
        for field in (self.inputs, self.folder, self.name, self.phase1_jobs):
            field.textChanged.connect(self._profile_changed)

    def _preparation_context(self):
        return (str(self.series_root), str(self._source_panel.series_root), self._source_panel.series_id,
                self.inputs.text().strip(), self.folder.text().strip(), self.name.text().strip(),
                self.phase1_jobs.text().strip(),
                self.profile.currentData(), self.stored_resolution.currentData())

    def _profile_changed(self, *_):
        self.prepared_result = None
        self._prepared_context = None
        self.adopt_button.setEnabled(False)
        self.status.setText("The file choices changed. Prepare a new plan before review and adoption.")

    def _run(self, call, **kwargs):
        if self._thread is not None:
            return
        context = self._preparation_context()
        if kwargs.get("adopting") and context != self._prepared_context:
            self._profile_changed()
            return
        self._active_context = context
        self.profile.setEnabled(False)
        self.stored_resolution.setEnabled(False)
        for field in (self.inputs, self.folder, self.name, self.phase1_jobs):
            field.setEnabled(False)
        for field in (self.inputs, self.phase1_jobs):
            self._file_buttons[field].setEnabled(False)
        return super()._run(call, **kwargs)

    def _done(self, result):
        if self._active_context != self._preparation_context():
            self.prepared_result = self._prepared_context = None
            self.status.setText(
                "Files from the earlier choices are preserved. The inputs or selected series changed; "
                "prepare a new plan before review and adoption.")
            return
        super()._done(result)
        if "error" not in result and self.prepared_result is not None:
            self._prepared_context = self._active_context

    def _closed(self):
        super()._closed()
        self._active_context = None
        self.profile.setEnabled(True)
        self.stored_resolution.setEnabled(True)
        for field in (self.inputs, self.folder, self.name, self.phase1_jobs):
            field.setEnabled(True)
        for field in (self.inputs, self.phase1_jobs):
            self._file_buttons[field].setEnabled(True)

    def _choose_folder(self):
        if self._thread is None:
            super()._choose_folder()

    def adopt(self):
        if self._prepared_context != self._preparation_context():
            self._profile_changed()
            return
        super().adopt()

    def _stage_changed(self):
        super()._stage_changed()
        # The parent refreshes controls after each finite call. Preserve this
        # dialog's explicit selected calibration through that original refresh.
        self.calibration.setReadOnly(True)
        self._file_buttons[self.calibration].setEnabled(False)

    def prepare(self):
        if self._thread is not None:
            return
        if (self._source_panel.series_root is None or self.series_root is None
                or Path(self._source_panel.series_root).resolve() != Path(self.series_root).resolve()):
            self.status.setText("The selected series changed. Reopen preparation for the current series.")
            return
        required = (self.inputs, self.folder, self.name)
        if not self.series_root or any(not field.text().strip() for field in required):
            self.status.setText("Choose saved phase-one completion inputs and a fresh plan folder.")
            return
        inputs, folder, job_id = (field.text().strip() for field in required)
        phase1_jobs = self.phase1_jobs.text().strip()
        selected = deepcopy(self._production_calibration)
        storage_profile = self.profile.currentData()
        stored_resolution = self.stored_resolution.currentData()
        series_root = self.series_root
        repository = Path(__file__).resolve().parents[3]
        self.prepared_result = None

        def call():
            if read_named_calibration(selected["job_root"]) != selected:
                raise ValueError("The selected named calibration changed before preparation.")
            from .production_image_grid import validate_production_phase_inputs
            validate_production_phase_inputs(series_root, descriptor(Path(inputs)))
            # Pass the already-shown descriptor, never a newly inferred identity
            # from an editable filename. The original DATA reader rechecks it.
            return preparation.prepare_phase2_inputs(
                descriptor(Path(inputs)), folder, series_root=series_root,
                fitted_calibration=selected["calibration"], job_id=job_id,
                job=preparation.job_identity(repository), storage_profile=storage_profile,
                stored_resolution=stored_resolution,
                phase1_jobs=preparation.source_metadata_descriptor(phase1_jobs) if phase1_jobs else None)

        self._run(call)
