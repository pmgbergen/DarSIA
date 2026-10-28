"""Qt calibration workflow page."""

from __future__ import annotations

from copy import deepcopy

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from darsia.fluidflower.calibration import (
    NUMERICAL_BACKENDS,
    OBJECTIVE_DEFINITIONS,
    PARAMETER_SCALES,
    PARAMETER_SCOPES,
    CalibrationPlanStore,
    apply_indicator_response_constraints,
    canonical_calibration_parameter_space,
    default_calibration_parameter_space,
    default_calibration_plan,
    validate_calibration_plan,
)
from darsia.fluidflower.calibration_review import (
    CALIBRATION_REVIEW_CONFIG_FORMAT,
    CalibrationResultReviewStore,
    default_calibration_review_config,
    validate_calibration_review_config,
)
from darsia.fluidflower.indicator_response import (
    IndicatorResponseModelStore,
    inspect_indicator_response_model_reference,
)
from darsia.fluidflower.phase_interpretation import PhaseInterpretationRevisionStore
from darsia.fluidflower.project import FluidFlowerProject, utc_now
from darsia.fluidflower.quality_budget import (
    QUALITY_PROFILE_NAMES,
    default_quality_profile,
    numerical_dtype_for_backend,
    summarize_quality_contract,
)
from darsia.fluidflower.ui.common import ModulePage, PathField


class _CalibrationPreview(QLabel):
    """Responsive preview for a concentration map or mass plot."""

    def __init__(self, empty_text: str) -> None:
        super().__init__(empty_text)
        self.empty_text = empty_text
        self.source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 230)
        self.setObjectName("calibrationReviewPreview")

    def set_path(self, path: Path | None) -> None:
        self.source = QPixmap(str(path)) if path and path.is_file() else QPixmap()
        self._render()

    def _render(self) -> None:
        if self.source.isNull():
            self.clear()
            self.setText(self.empty_text)
            return
        self.setPixmap(
            self.source.scaled(
                max(1, self.width() - 8),
                max(1, self.height() - 8),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._render()


class CalibrationPage(ModulePage):
    """Define a reproducible calibration contract without executing Optuna."""

    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__(project, "calibration")
        self.store = CalibrationPlanStore(project)
        self.result_review_store = CalibrationResultReviewStore(project)
        self.calibration_run_reviews: dict[str, dict[str, Any]] = {}
        self.calibration_result_report: dict[str, Any] | None = None
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "This page freezes chemistry assumptions, physical objectives, trial "
            "budget, and numerical backend. Saving never starts Optuna, facies "
            "training, phase training, or mass analysis."
        )

        self.calibration_phase_reference = QLabel()
        self.calibration_phase_reference.setObjectName("integrationNotice")
        self.calibration_phase_reference.setWordWrap(True)
        self.work_layout.addWidget(self.calibration_phase_reference)

        settings = QSplitter(Qt.Orientation.Horizontal)
        chemistry_panel = QFrame()
        chemistry_panel.setFrameShape(QFrame.Shape.StyledPanel)
        chemistry_layout = QFormLayout(chemistry_panel)
        self.calibration_indicator = QLineEdit()
        self.calibration_indicator.setReadOnly(True)
        self.calibration_indicator_reference = QLabel()
        self.calibration_indicator_reference.setWordWrap(True)
        self.calibration_indicator_reference.setObjectName("secondaryText")
        self.calibration_model = QComboBox()
        for label, value in (
            (
                "Compare Simple Flash and Titration",
                "compare_simple_flash_and_titration",
            ),
            ("Simple Flash", "simple_flash"),
            ("Titration", "titration"),
        ):
            self.calibration_model.addItem(label, value)
        self.calibration_ph_low = QDoubleSpinBox()
        self.calibration_ph_high = QDoubleSpinBox()
        for field in (self.calibration_ph_low, self.calibration_ph_high):
            field.setRange(0.0, 14.0)
            field.setDecimals(3)
            field.setSingleStep(0.05)
        ph_row = QWidget()
        ph_layout = QHBoxLayout(ph_row)
        ph_layout.setContentsMargins(0, 0, 0, 0)
        ph_layout.addWidget(self.calibration_ph_low)
        ph_layout.addWidget(QLabel("to"))
        ph_layout.addWidget(self.calibration_ph_high)
        self.calibration_temperature = QDoubleSpinBox()
        self.calibration_temperature.setRange(-20.0, 100.0)
        self.calibration_temperature.setDecimals(2)
        self.calibration_temperature.setSuffix(" C")
        self.calibration_pressure = QDoubleSpinBox()
        self.calibration_pressure.setRange(0.001, 1000.0)
        self.calibration_pressure.setDecimals(3)
        self.calibration_pressure.setSuffix(" bar")
        self.calibration_transfer_shape = QComboBox()
        self.calibration_transfer_shape.addItem("Smoothstep", "smoothstep")
        self.calibration_transfer_shape.addItem("Linear", "linear")
        self.calibration_transfer_shape.addItem("Locked production labels", "locked_categorical")
        self.calibration_co2_onset = QDoubleSpinBox()
        self.calibration_co2_high = QDoubleSpinBox()
        for field in (self.calibration_co2_onset, self.calibration_co2_high):
            field.setRange(0.0, 1.0)
            field.setDecimals(4)
            field.setSingleStep(0.01)
        co2_support_row = QWidget()
        co2_support_layout = QHBoxLayout(co2_support_row)
        co2_support_layout.setContentsMargins(0, 0, 0, 0)
        co2_support_layout.addWidget(self.calibration_co2_onset)
        co2_support_layout.addWidget(QLabel("to"))
        co2_support_layout.addWidget(self.calibration_co2_high)
        chemistry_layout.addRow("Indicator system", self.calibration_indicator)
        chemistry_layout.addRow(
            "Indicator reference", self.calibration_indicator_reference
        )
        chemistry_layout.addRow("Chemistry model", self.calibration_model)
        chemistry_layout.addRow("Initial pH bounds", ph_row)
        chemistry_layout.addRow("Temperature", self.calibration_temperature)
        chemistry_layout.addRow("Pressure", self.calibration_pressure)
        chemistry_layout.addRow(
            "Fixed CO2-support shape", self.calibration_transfer_shape
        )
        chemistry_layout.addRow("Fixed CO2-support bounds", co2_support_row)
        depth_policy = QLabel(
            "Depth and porosity are fixed to the current geometry revision. "
            "The 2D-to-3D conversion remains a mass-analysis responsibility."
        )
        depth_policy.setObjectName("secondaryText")
        depth_policy.setWordWrap(True)
        chemistry_layout.addRow("Geometry policy", depth_policy)
        settings.addWidget(chemistry_panel)

        execution_panel = QFrame()
        execution_panel.setFrameShape(QFrame.Shape.StyledPanel)
        execution_layout = QFormLayout(execution_panel)
        self.calibration_trials = QSpinBox()
        self.calibration_trials.setRange(1, 10_000_000)
        self.calibration_trials.setSingleStep(1000)
        self.calibration_trials.setKeyboardTracking(False)
        self.calibration_seed = QSpinBox()
        self.calibration_seed.setRange(0, 2_147_483_647)
        self.calibration_backend = QComboBox()
        for backend in NUMERICAL_BACKENDS:
            self.calibration_backend.addItem(backend, backend)
        self.calibration_backend.currentIndexChanged.connect(
            self._calibration_backend_changed
        )
        self.calibration_dtype = QLabel()
        self.calibration_quality_profile = QComboBox()
        for profile in QUALITY_PROFILE_NAMES:
            self.calibration_quality_profile.addItem(profile.title(), profile)
        self.calibration_quality_profile.currentIndexChanged.connect(
            self._calibration_quality_profile_changed
        )
        self.calibration_resolution_policy = QComboBox()
        self.calibration_resolution_policy.addItem(
            "Native registered grid", "native_registered_grid"
        )
        self.calibration_resolution_policy.addItem("Scale factor", "scale_factor")
        self.calibration_resolution_policy.addItem(
            "Maximum dimension", "maximum_dimension"
        )
        self.calibration_resolution_policy.currentIndexChanged.connect(
            self._calibration_resolution_changed
        )
        self.calibration_scale_percent = QDoubleSpinBox()
        self.calibration_scale_percent.setRange(5.0, 100.0)
        self.calibration_scale_percent.setDecimals(1)
        self.calibration_scale_percent.setSuffix(" %")
        self.calibration_maximum_dimension = QSpinBox()
        self.calibration_maximum_dimension.setRange(64, 32_768)
        self.calibration_maximum_dimension.setSuffix(" px")
        resolution_row = QWidget()
        resolution_row_layout = QHBoxLayout(resolution_row)
        resolution_row_layout.setContentsMargins(0, 0, 0, 0)
        resolution_row_layout.addWidget(self.calibration_resolution_policy)
        resolution_row_layout.addWidget(self.calibration_scale_percent)
        resolution_row_layout.addWidget(self.calibration_maximum_dimension)
        self.calibration_preview_dimension = QSpinBox()
        self.calibration_preview_dimension.setRange(256, 32_768)
        self.calibration_preview_dimension.setSuffix(" px")
        self.calibration_preview_quality = QSpinBox()
        self.calibration_preview_quality.setRange(1, 100)
        self.calibration_preview_quality.setSuffix(" % JPEG")
        preview_row = QWidget()
        preview_row_layout = QHBoxLayout(preview_row)
        preview_row_layout.setContentsMargins(0, 0, 0, 0)
        preview_row_layout.addWidget(self.calibration_preview_dimension)
        preview_row_layout.addWidget(self.calibration_preview_quality)
        self.calibration_parity_percent = QDoubleSpinBox()
        self.calibration_parity_percent.setRange(0.01, 100.0)
        self.calibration_parity_percent.setDecimals(2)
        self.calibration_parity_percent.setSuffix(" %")
        self.calibration_workers = QSpinBox()
        self.calibration_workers.setRange(1, 256)
        self.calibration_absolute_tolerance = QDoubleSpinBox()
        self.calibration_relative_tolerance = QDoubleSpinBox()
        for field in (
            self.calibration_absolute_tolerance,
            self.calibration_relative_tolerance,
        ):
            field.setRange(1.0e-9, 1.0)
            field.setDecimals(9)
            field.setSingleStep(1.0e-5)
        self.calibration_same_objective = QCheckBox(
            "Final selection uses identical objective"
        )
        self.calibration_same_objective.setChecked(True)
        self.calibration_same_objective.setEnabled(False)
        execution_layout.addRow("Trials per model", self.calibration_trials)
        execution_layout.addRow("Sampler seed", self.calibration_seed)
        execution_layout.addRow("Planned backend", self.calibration_backend)
        execution_layout.addRow("Numerical dtype", self.calibration_dtype)
        execution_layout.addRow("Quality preset", self.calibration_quality_profile)
        execution_layout.addRow("Analysis grid", resolution_row)
        execution_layout.addRow("Review previews", preview_row)
        execution_layout.addRow(
            "CPU-float64 parity sample", self.calibration_parity_percent
        )
        execution_layout.addRow("Planned workers", self.calibration_workers)
        execution_layout.addRow(
            "Absolute parity tolerance", self.calibration_absolute_tolerance
        )
        execution_layout.addRow(
            "Relative parity tolerance", self.calibration_relative_tolerance
        )
        execution_layout.addRow("Selection contract", self.calibration_same_objective)
        reference_note = QLabel(
            "Precision and analysis resolution are independent. CPU, CUDA, and "
            "compatible OpenCL devices may use float32; supported CUDA/OpenCL "
            "devices may also use float64, while Metal remains float32. CPU-float64 "
            "stays the authoritative parity reference. Reduced scientific grids and "
            "review previews are separate settings."
        )
        reference_note.setObjectName("secondaryText")
        reference_note.setWordWrap(True)
        execution_layout.addRow("Reference", reference_note)
        self.calibration_quality_basis = QLabel()
        self.calibration_quality_basis.setObjectName("integrationNotice")
        self.calibration_quality_basis.setWordWrap(True)
        execution_layout.addRow("Current working basis", self.calibration_quality_basis)
        settings.addWidget(execution_panel)
        settings.setStretchFactor(0, 1)
        settings.setStretchFactor(1, 1)
        self.work_layout.addWidget(settings)

        self.indicator_response_store = IndicatorResponseModelStore(project)
        self.indicator_response_status = QLabel()
        self.indicator_response_status.setObjectName("integrationNotice")
        self.indicator_response_status.setWordWrap(True)
        self.work_layout.addWidget(self.indicator_response_status)

        parameter_heading = QLabel("Optuna parameter contract")
        parameter_heading.setObjectName("sectionTitle")
        self.work_layout.addWidget(parameter_heading)
        parameter_note = QLabel(
            "Every searchable quantity is listed explicitly. Scope controls whether "
            "a value is shared, run-specific, or estimated hierarchically. Spatial "
            "diffusion remains disabled until a mass-conserving spatial evaluator is "
            "attached."
        )
        parameter_note.setObjectName("secondaryText")
        parameter_note.setWordWrap(True)
        self.work_layout.addWidget(parameter_note)
        self.calibration_parameter_table = QTableWidget(0, 8)
        self.calibration_parameter_table.setHorizontalHeaderLabels(
            (
                "Use",
                "Parameter",
                "Scope",
                "Lower",
                "Upper",
                "Scale",
                "Unit",
                "Scientific role",
            )
        )
        parameter_header = self.calibration_parameter_table.horizontalHeader()
        parameter_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        parameter_header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        parameter_header.setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        parameter_header.setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        parameter_header.setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        parameter_header.setSectionResizeMode(
            5, QHeaderView.ResizeMode.ResizeToContents
        )
        parameter_header.setSectionResizeMode(
            6, QHeaderView.ResizeMode.ResizeToContents
        )
        parameter_header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.calibration_parameter_table.verticalHeader().setVisible(False)
        self.calibration_parameter_table.verticalHeader().setDefaultSectionSize(34)
        self.calibration_parameter_table.setAlternatingRowColors(True)
        self.calibration_parameter_table.setWordWrap(False)
        self.calibration_parameter_table.setMinimumHeight(330)
        self.calibration_parameter_table.setMaximumHeight(350)
        self.calibration_parameter_widgets: dict[
            str,
            tuple[QCheckBox, QComboBox, QDoubleSpinBox, QDoubleSpinBox, QComboBox],
        ] = {}
        self.calibration_parameter_metadata: dict[str, dict[str, Any]] = {}
        self.work_layout.addWidget(self.calibration_parameter_table)

        evaluator_panel = QFrame()
        evaluator_panel.setFrameShape(QFrame.Shape.StyledPanel)
        evaluator_layout = QFormLayout(evaluator_panel)
        self.calibration_evaluation_manifest = PathField(
            directory=False,
            file_filter="Calibration evaluation manifest (*.json);;All files (*)",
            placeholder="Attach after phase-derived evaluation data is prepared",
        )
        evaluator_layout.addRow(
            "Prepared evaluation manifest", self.calibration_evaluation_manifest
        )
        evaluator_note = QLabel(
            "A saved plan may exist without this file. An Optuna task is ready only "
            "when the manifest is fingerprinted, matches the selected phase revision, "
            "and supports every enabled objective and parameter."
        )
        evaluator_note.setObjectName("secondaryText")
        evaluator_note.setWordWrap(True)
        evaluator_layout.addRow("Execution gate", evaluator_note)
        self.work_layout.addWidget(evaluator_panel)

        objective_heading = QLabel("Physical objective contract")
        objective_heading.setObjectName("sectionTitle")
        self.work_layout.addWidget(objective_heading)
        objective_note = QLabel(
            "Weights are relative. Disabled terms remain explicit in the fingerprinted "
            "contract, and the same contract must be used for optimization and "
            "winner selection."
        )
        objective_note.setObjectName("secondaryText")
        objective_note.setWordWrap(True)
        self.work_layout.addWidget(objective_note)
        objective_grid = QGridLayout()
        objective_grid.addWidget(QLabel("Use"), 0, 0)
        objective_grid.addWidget(QLabel("Objective"), 0, 1)
        objective_grid.addWidget(QLabel("Weight"), 0, 2)
        objective_grid.setColumnStretch(1, 1)
        self.calibration_objectives: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        for row, definition in enumerate(OBJECTIVE_DEFINITIONS, start=1):
            enabled = QCheckBox()
            weight = QDoubleSpinBox()
            weight.setRange(0.0, 1000.0)
            weight.setDecimals(4)
            weight.setSingleStep(0.1)
            objective_grid.addWidget(enabled, row, 0, Qt.AlignmentFlag.AlignCenter)
            objective_grid.addWidget(QLabel(str(definition["label"])), row, 1)
            objective_grid.addWidget(weight, row, 2)
            self.calibration_objectives[str(definition["key"])] = (enabled, weight)
        self.work_layout.addLayout(objective_grid)

        review_panel = QFrame()
        review_panel.setFrameShape(QFrame.Shape.StyledPanel)
        review_layout = QFormLayout(review_panel)
        self.calibration_reviewer = QLineEdit(
            str(project.manifest.get("created_by") or "")
        )
        self.calibration_reason = QLineEdit()
        self.calibration_reason.setPlaceholderText(
            "Why is this calibration contract scientifically appropriate?"
        )
        review_layout.addRow("Reviewer", self.calibration_reviewer)
        review_layout.addRow("Plan note", self.calibration_reason)
        self.work_layout.addWidget(review_panel)

        actions = QHBoxLayout()
        self.validate_calibration_plan_button = QPushButton("Validate plan")
        self.validate_calibration_plan_button.clicked.connect(
            self._validate_calibration_plan
        )
        actions.addWidget(self.validate_calibration_plan_button)
        self.save_calibration_plan = QPushButton("Save plan")
        self.save_calibration_plan.clicked.connect(self._save_calibration_plan)
        actions.addWidget(self.save_calibration_plan)
        actions.addStretch(1)
        actions.addWidget(QLabel("Saved plan"))
        self.calibration_revisions = QComboBox()
        self.calibration_revisions.setMinimumWidth(300)
        actions.addWidget(self.calibration_revisions)
        self.work_layout.addLayout(actions)
        self.calibration_summary = QLabel("No calibration plan validated.")
        self.calibration_summary.setObjectName("secondaryText")
        self.calibration_summary.setWordWrap(True)
        self.work_layout.addWidget(self.calibration_summary)

        result_heading = QLabel("Mass calibration result review")
        result_heading.setObjectName("sectionTitle")
        self.work_layout.addWidget(result_heading)
        result_note = QLabel(
            "This is a separate scientific gate after Optuna has produced calibrated "
            "parameters. Inspect the concentration map and injected-versus-detected "
            "mass plot for every experiment. Approval here confirms the 2D-to-3D "
            "conversion and mass interpretation; it does not start mass analysis."
        )
        result_note.setObjectName("secondaryText")
        result_note.setWordWrap(True)
        self.work_layout.addWidget(result_note)

        result_manifest_row = QHBoxLayout()
        self.calibration_result_manifest = PathField(
            directory=False,
            file_filter="Calibration result manifest (*.json);;All files (*)",
            placeholder="Calibration-result manifest produced by the optimizer",
        )
        result_manifest_row.addWidget(self.calibration_result_manifest, 1)
        inspect_result = QPushButton("Inspect result")
        inspect_result.clicked.connect(self._inspect_calibration_result_review)
        result_manifest_row.addWidget(inspect_result)
        new_result_review = QPushButton("Start new review")
        new_result_review.clicked.connect(self._start_new_calibration_result_review)
        result_manifest_row.addWidget(new_result_review)
        self.work_layout.addLayout(result_manifest_row)

        self.calibration_run_review_table = QTableWidget(0, 6)
        self.calibration_run_review_table.setHorizontalHeaderLabels(
            (
                "Experiment",
                "Status",
                "Reviewer",
                "Concentration map",
                "Mass plot",
                "Note",
            )
        )
        self.calibration_run_review_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.calibration_run_review_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.calibration_run_review_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.calibration_run_review_table.verticalHeader().setVisible(False)
        review_header = self.calibration_run_review_table.horizontalHeader()
        for column in (0, 1, 2):
            review_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        for column in (3, 4, 5):
            review_header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.calibration_run_review_table.currentCellChanged.connect(
            self._calibration_review_run_changed
        )
        self.calibration_run_review_table.setMinimumHeight(190)
        self.work_layout.addWidget(self.calibration_run_review_table)

        review_asset_form = QFormLayout()
        self.calibration_concentration_preview = PathField(
            directory=False,
            file_filter="Images (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*)",
            placeholder="Visible concentration map for the selected experiment",
        )
        self.calibration_mass_plot = PathField(
            directory=False,
            file_filter="Images (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*)",
            placeholder="Injected-versus-detected mass plot for the selected experiment",
        )
        self.calibration_review_note = QLineEdit()
        self.calibration_review_note.setPlaceholderText(
            "Why is this mass interpretation acceptable, or what must be recalibrated?"
        )
        review_asset_form.addRow(
            "Concentration map", self.calibration_concentration_preview
        )
        review_asset_form.addRow("Mass plot", self.calibration_mass_plot)
        review_asset_form.addRow("Run review note", self.calibration_review_note)
        self.work_layout.addLayout(review_asset_form)

        review_previews = QSplitter(Qt.Orientation.Horizontal)
        self.calibration_concentration_image = _CalibrationPreview(
            "Choose a concentration-map image for the selected experiment"
        )
        self.calibration_mass_plot_image = _CalibrationPreview(
            "Choose an injected-versus-detected mass-plot image"
        )
        review_previews.addWidget(self.calibration_concentration_image)
        review_previews.addWidget(self.calibration_mass_plot_image)
        review_previews.setSizes([700, 700])
        self.work_layout.addWidget(review_previews)
        self.calibration_concentration_preview.changed.connect(
            lambda _value: self._refresh_calibration_review_previews()
        )
        self.calibration_mass_plot.changed.connect(
            lambda _value: self._refresh_calibration_review_previews()
        )

        result_actions = QHBoxLayout()
        approve_run = QPushButton("Approve selected experiment")
        approve_run.clicked.connect(
            lambda: self._set_calibration_run_review("approved")
        )
        result_actions.addWidget(approve_run)
        recalibrate_run = QPushButton("Needs recalibration")
        recalibrate_run.clicked.connect(
            lambda: self._set_calibration_run_review("needs_recalibration")
        )
        result_actions.addWidget(recalibrate_run)
        reset_run = QPushButton("Reset to pending")
        reset_run.clicked.connect(lambda: self._set_calibration_run_review("pending"))
        result_actions.addWidget(reset_run)
        result_actions.addStretch(1)
        save_result_review = QPushButton("Save result-review revision")
        save_result_review.clicked.connect(self._save_calibration_result_review)
        result_actions.addWidget(save_result_review)
        self.work_layout.addLayout(result_actions)
        self.calibration_result_review_summary = QLabel(
            "No mass-calibration result reviewed."
        )
        self.calibration_result_review_summary.setObjectName("secondaryText")
        self.calibration_result_review_summary.setWordWrap(True)
        self.work_layout.addWidget(self.calibration_result_review_summary)

        self.calibration_ph_low.valueChanged.connect(
            self._sync_calibration_ph_parameter
        )
        self.calibration_ph_high.valueChanged.connect(
            self._sync_calibration_ph_parameter
        )
        for field in (
            self.calibration_scale_percent,
            self.calibration_maximum_dimension,
            self.calibration_preview_dimension,
            self.calibration_preview_quality,
        ):
            field.valueChanged.connect(self._update_calibration_quality_basis)
        self._load_calibration_plan()
        self._load_calibration_result_review()
        self.refresh()

    def _current_indicator_response(self) -> dict[str, Any]:
        reference = self.indicator_response_store.current_reference()
        inspected = inspect_indicator_response_model_reference(reference)
        return reference if inspected.get("valid") else {}

    def _apply_indicator_response_to_controls(self) -> None:
        if not self.calibration_parameter_widgets:
            return
        reference = self._current_indicator_response()
        if reference:
            try:
                draft = default_calibration_plan(self.project)
                draft["chemistry"]["initial_ph_bounds"] = [
                    self.calibration_ph_low.value(),
                    self.calibration_ph_high.value(),
                ]
                draft["parameter_space"] = self._calibration_parameter_space()
                constrained = apply_indicator_response_constraints(draft, reference)
            except (OSError, TypeError, ValueError):
                reference = {}
            else:
                rows = {
                    str(row.get("key") or ""): dict(row)
                    for row in (constrained.get("parameter_space") or {}).get(
                        "parameters", []
                    )
                }
                for key, widgets in self.calibration_parameter_widgets.items():
                    row = rows.get(key)
                    if row is not None:
                        widgets[0].setChecked(bool(row.get("enabled")))
                        widgets[2].setValue(float(row.get("lower") or 0.0))
                        widgets[3].setValue(float(row.get("upper") or 0.0))
                    if key in {
                        "aqueous_transition_low_fraction",
                        "aqueous_transition_width_fraction",
                        "aqueous_titration_blend",
                    }:
                        for widget in widgets:
                            widget.setEnabled(False)
                self.indicator_response_status.setText(
                    "The approved measured optical response is attached automatically. "
                    "Optuna can vary only its reviewed chemical uncertainty bounds; it "
                    "cannot redefine the fitted color response."
                )
                return
        for widgets in self.calibration_parameter_widgets.values():
            for widget in widgets:
                widget.setEnabled(True)
        self.indicator_response_status.setText(
            "No deployable measured buffer-response model is active. Mass calibration "
            "uses the approved generic optical branch from phase interpretation."
        )

    def _calibration_backend_changed(self) -> None:
        backend = str(self.calibration_backend.currentData() or "cpu-float64")
        try:
            dtype = numerical_dtype_for_backend(backend)
        except ValueError:
            dtype = "Unknown"
        self.calibration_dtype.setText(dtype)
        self._update_calibration_quality_basis()

    def _calibration_quality_profile_changed(self) -> None:
        profile = default_quality_profile(
            str(self.calibration_quality_profile.currentData() or "production")
        )
        numerical = dict(profile.get("numerical_resolution") or {})
        policy_index = self.calibration_resolution_policy.findData(
            numerical.get("policy") or "native_registered_grid"
        )
        self.calibration_resolution_policy.setCurrentIndex(max(0, policy_index))
        self.calibration_scale_percent.setValue(
            100.0 * float(numerical.get("scale_factor") or 1.0)
        )
        self.calibration_maximum_dimension.setValue(
            int(numerical.get("maximum_dimension_px") or 2048)
        )
        self.calibration_preview_dimension.setValue(
            int(profile["preview"]["maximum_dimension_px"])
        )
        self.calibration_preview_quality.setValue(
            int(profile["preview"]["jpeg_quality"])
        )
        self.calibration_parity_percent.setValue(
            100.0
            * float(
                (profile.get("precision") or {}).get(
                    "cpu_float64_parity_fraction", 0.05
                )
            )
        )
        self._calibration_resolution_changed()

    def _calibration_resolution_changed(self) -> None:
        policy = str(
            self.calibration_resolution_policy.currentData() or "native_registered_grid"
        )
        self.calibration_scale_percent.setVisible(policy == "scale_factor")
        self.calibration_maximum_dimension.setVisible(policy == "maximum_dimension")
        self._update_calibration_quality_basis()

    def _calibration_quality_configuration(self) -> dict[str, Any]:
        quality = default_quality_profile(
            str(self.calibration_quality_profile.currentData() or "production")
        )
        policy = str(
            self.calibration_resolution_policy.currentData() or "native_registered_grid"
        )
        quality["numerical_resolution"].update(
            {
                "policy": policy,
                "scale_factor": self.calibration_scale_percent.value() / 100.0,
                "maximum_dimension_px": (
                    self.calibration_maximum_dimension.value()
                    if policy == "maximum_dimension"
                    else None
                ),
            }
        )
        quality["preview"].update(
            {
                "maximum_dimension_px": self.calibration_preview_dimension.value(),
                "jpeg_quality": self.calibration_preview_quality.value(),
            }
        )
        quality["precision"].update(
            {
                "compute_dtype": numerical_dtype_for_backend(
                    str(self.calibration_backend.currentData() or "cpu-float64")
                ),
                "cpu_float64_parity_fraction": (
                    self.calibration_parity_percent.value() / 100.0
                ),
            }
        )
        return quality

    def _update_calibration_quality_basis(self) -> None:
        if not hasattr(self, "calibration_quality_basis"):
            return
        backend = str(self.calibration_backend.currentData() or "cpu-float64")
        try:
            summary = summarize_quality_contract(
                self._calibration_quality_configuration(),
                backend=backend,
            )
            self.calibration_quality_basis.setText(
                "Calibration only: " + summary["label"]
            )
        except (TypeError, ValueError):
            self.calibration_quality_basis.setText(
                "Calibration working basis is incomplete."
            )

    def _populate_calibration_parameters(
        self, parameter_space: Mapping[str, Any]
    ) -> None:
        rows = [
            dict(row)
            for row in parameter_space.get("parameters") or []
            if isinstance(row, Mapping)
        ]
        self.calibration_parameter_table.setRowCount(len(rows))
        self.calibration_parameter_widgets.clear()
        self.calibration_parameter_metadata.clear()
        for row_index, row in enumerate(rows):
            key = str(row.get("key") or "")
            enabled = QCheckBox()
            enabled.setChecked(bool(row.get("enabled")))
            enabled.setToolTip(f"Include {key} in the Optuna search space")
            enabled_container = QWidget()
            enabled_layout = QHBoxLayout(enabled_container)
            enabled_layout.setContentsMargins(0, 0, 0, 0)
            enabled_layout.addStretch(1)
            enabled_layout.addWidget(enabled)
            enabled_layout.addStretch(1)
            self.calibration_parameter_table.setCellWidget(
                row_index, 0, enabled_container
            )

            label = QTableWidgetItem(str(row.get("label") or key))
            label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            models = ", ".join(str(value) for value in row.get("models") or [])
            label.setToolTip(f"{key}\nApplies to: {models}")
            self.calibration_parameter_table.setItem(row_index, 1, label)

            scope = QComboBox()
            for value in PARAMETER_SCOPES:
                scope.addItem(value.replace("_", " ").title(), value)
            scope.setCurrentIndex(max(0, scope.findData(row.get("scope"))))
            self.calibration_parameter_table.setCellWidget(row_index, 2, scope)

            lower = QDoubleSpinBox()
            upper = QDoubleSpinBox()
            for field in (lower, upper):
                field.setRange(-1.0e9, 1.0e9)
                field.setDecimals(12)
                field.setKeyboardTracking(False)
            lower.setValue(float(row.get("lower") or 0.0))
            upper.setValue(float(row.get("upper") or 0.0))
            self.calibration_parameter_table.setCellWidget(row_index, 3, lower)
            self.calibration_parameter_table.setCellWidget(row_index, 4, upper)

            scale = QComboBox()
            for value in PARAMETER_SCALES:
                scale.addItem(value.title(), value)
            scale.setCurrentIndex(max(0, scale.findData(row.get("scale"))))
            self.calibration_parameter_table.setCellWidget(row_index, 5, scale)

            unit = QTableWidgetItem(str(row.get("unit") or ""))
            unit.setFlags(unit.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.calibration_parameter_table.setItem(row_index, 6, unit)
            role = QTableWidgetItem(str(row.get("scientific_role") or ""))
            role.setFlags(role.flags() & ~Qt.ItemFlag.ItemIsEditable)
            role.setToolTip(str(row.get("transform") or ""))
            self.calibration_parameter_table.setItem(row_index, 7, role)

            self.calibration_parameter_widgets[key] = (
                enabled,
                scope,
                lower,
                upper,
                scale,
            )
            self.calibration_parameter_metadata[key] = row
        for row_index in range(len(rows)):
            self.calibration_parameter_table.setRowHeight(row_index, 34)

    def _sync_calibration_ph_parameter(self, _value: float = 0.0) -> None:
        widgets = self.calibration_parameter_widgets.get("initial_ph")
        if widgets is None:
            return
        widgets[2].setValue(self.calibration_ph_low.value())
        widgets[3].setValue(self.calibration_ph_high.value())

    def _calibration_parameter_space(self) -> dict[str, Any]:
        default = default_calibration_parameter_space(
            [self.calibration_ph_low.value(), self.calibration_ph_high.value()]
        )
        parameters: list[dict[str, Any]] = []
        for fallback in default["parameters"]:
            key = str(fallback["key"])
            enabled, scope, lower, upper, scale = self.calibration_parameter_widgets[
                key
            ]
            metadata = self.calibration_parameter_metadata.get(key, fallback)
            parameters.append(
                {
                    **fallback,
                    "label": str(metadata.get("label") or fallback["label"]),
                    "enabled": enabled.isChecked(),
                    "lower": lower.value(),
                    "upper": upper.value(),
                    "scale": str(scale.currentData()),
                    "scope": str(scope.currentData()),
                }
            )
        return {
            "schema": 1,
            "format": default["format"],
            "parameters": parameters,
            "constraints": list(default["constraints"]),
        }

    def _load_calibration_plan(self) -> None:
        current = self.store.current()
        config = dict(current.get("configuration") or {})
        if not config:
            config = default_calibration_plan(self.project)
        self._native_fixed_plan = deepcopy(config) if config.get("native_fixed_phase_contract") else None
        chemistry = dict(config.get("chemistry") or {})
        self.calibration_indicator.setText(
            str(chemistry.get("indicator_system") or "unspecified")
        )
        reference = dict(chemistry.get("indicator_response_reference") or {})
        self.calibration_indicator_reference.setText(
            f"{int(reference.get('entry_count') or 0)} image(s); "
            f"{str(reference.get('status') or 'not provided').replace('_', ' ')}. "
            "Known-pH buffers constrain chemistry response; they are not light targets."
        )
        model_index = self.calibration_model.findData(chemistry.get("model"))
        self.calibration_model.setCurrentIndex(max(0, model_index))
        bounds = list(chemistry.get("initial_ph_bounds") or [6.0, 10.0])
        self.calibration_ph_low.setValue(float(bounds[0]))
        self.calibration_ph_high.setValue(float(bounds[1]))
        temperature = chemistry.get("temperature_c")
        pressure = chemistry.get("pressure_bar")
        self.calibration_temperature.setValue(
            float(20.0 if temperature is None else temperature)
        )
        self.calibration_pressure.setValue(float(1.0 if pressure is None else pressure))
        fixed_transfer = dict(config.get("fixed_transfer_contract") or {})
        transfer_index = self.calibration_transfer_shape.findData(
            fixed_transfer.get("transfer_shape")
        )
        self.calibration_transfer_shape.setCurrentIndex(max(0, transfer_index))
        co2_onset = fixed_transfer.get("co2_support_onset")
        co2_high = fixed_transfer.get("co2_support_high")
        self.calibration_co2_onset.setValue(
            float(0.15 if co2_onset is None else co2_onset)
        )
        self.calibration_co2_high.setValue(
            float(0.50 if co2_high is None else co2_high)
        )
        parameter_space = canonical_calibration_parameter_space(
            config.get("parameter_space"), initial_ph_bounds=bounds
        )
        self._populate_calibration_parameters(parameter_space)
        self._sync_calibration_ph_parameter()
        self._apply_indicator_response_to_controls()
        self.calibration_evaluation_manifest.setText(
            str(config.get("evaluation_manifest_path") or "")
        )
        optimizer = dict(config.get("optimizer") or {})
        self.calibration_trials.setValue(
            int(optimizer.get("trial_budget_per_model") or 5000)
        )
        self.calibration_seed.setValue(int(optimizer.get("sampler_seed") or 0))
        execution = dict(config.get("execution_profile") or {})
        backend_index = self.calibration_backend.findData(
            execution.get("primary_backend")
        )
        self.calibration_backend.setCurrentIndex(max(0, backend_index))
        quality = dict(config.get("quality_profile") or {})
        if not quality:
            quality = default_quality_profile("production")
        quality_index = self.calibration_quality_profile.findData(
            quality.get("name") or "production"
        )
        self.calibration_quality_profile.setCurrentIndex(max(0, quality_index))
        numerical = dict(quality.get("numerical_resolution") or {})
        resolution_index = self.calibration_resolution_policy.findData(
            numerical.get("policy") or "native_registered_grid"
        )
        self.calibration_resolution_policy.setCurrentIndex(max(0, resolution_index))
        self.calibration_scale_percent.setValue(
            100.0 * float(numerical.get("scale_factor") or 1.0)
        )
        self.calibration_maximum_dimension.setValue(
            int(numerical.get("maximum_dimension_px") or 2048)
        )
        preview = dict(quality.get("preview") or {})
        self.calibration_preview_dimension.setValue(
            int(preview.get("maximum_dimension_px") or 1920)
        )
        self.calibration_preview_quality.setValue(
            int(preview.get("jpeg_quality") or 92)
        )
        self.calibration_parity_percent.setValue(
            100.0
            * float(
                (quality.get("precision") or {}).get(
                    "cpu_float64_parity_fraction", 0.05
                )
            )
        )
        self._calibration_backend_changed()
        self._calibration_resolution_changed()
        self._update_calibration_quality_basis()
        self.calibration_workers.setValue(int(execution.get("workers") or 1))
        self.calibration_absolute_tolerance.setValue(
            float(execution.get("absolute_parity_tolerance") or 1.0e-5)
        )
        self.calibration_relative_tolerance.setValue(
            float(execution.get("relative_parity_tolerance") or 1.0e-4)
        )
        by_key = {
            str(row.get("key") or ""): dict(row)
            for row in config.get("objectives") or []
        }
        for definition in OBJECTIVE_DEFINITIONS:
            key = str(definition["key"])
            row = by_key.get(key, definition)
            enabled, weight = self.calibration_objectives[key]
            enabled.setChecked(bool(row.get("enabled", definition["default_enabled"])))
            weight.setValue(float(row.get("weight", definition["default_weight"])))
        if self._native_fixed_plan:
            native = self._native_fixed_plan["native_fixed_phase_contract"]
            self.calibration_phase_reference.setText(
                "Locked production labels · shared native DarSIA response · "
                "one aqueous/gas scale pair for the series · no map-to-dose rescaling. "
                f"Calibration: {len(native.get('calibration_runs', []))} runs; "
                f"held out: {len(native.get('heldout_runs', []))} runs."
            )
            self.indicator_response_status.setText(
                "One common native color response is fitted on corrected RAW, "
                "with measured clear-water references and run-specific chemistry. "
                "Only the aqueous and gas response scales are shared fitted parameters."
            )
            for label in self.findChildren(QLabel):
                if label.text() == "Optuna parameter contract":
                    label.setText("Common native response parameters")
            for name, widget in vars(self).items():
                if name.startswith("calibration_") and isinstance(widget, (QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QTableWidget)):
                    widget.setEnabled(False)
            for enabled, weight in self.calibration_objectives.values():
                enabled.setEnabled(False)
                weight.setEnabled(False)

    def _calibration_configuration(self) -> dict[str, Any]:
        if getattr(self, "_native_fixed_plan", None):
            # Preserve the frozen native support and scope when this plan is
            # viewed/validated through the existing calibration page.
            return deepcopy(self._native_fixed_plan)
        config = default_calibration_plan(self.project)
        phase_revision = PhaseInterpretationRevisionStore(self.project).current()
        config["phase_revision_id"] = str(phase_revision.get("revision_id") or "")
        config["phase_revision_sha256"] = str(
            phase_revision.get("content_sha256") or ""
        )
        config["chemistry"] = {
            "indicator_system": self.calibration_indicator.text().strip(),
            "model": str(self.calibration_model.currentData()),
            "initial_ph_bounds": [
                self.calibration_ph_low.value(),
                self.calibration_ph_high.value(),
            ],
            "temperature_c": self.calibration_temperature.value(),
            "pressure_bar": self.calibration_pressure.value(),
            "indicator_response_reference": dict(
                (self.project.manifest.get("chemistry") or {}).get(
                    "indicator_response_reference"
                )
                or {}
            ),
        }
        config["parameter_space"] = self._calibration_parameter_space()
        config["fixed_transfer_contract"] = {
            "transfer_shape": str(self.calibration_transfer_shape.currentData()),
            "co2_support_onset": self.calibration_co2_onset.value(),
            "co2_support_high": self.calibration_co2_high.value(),
            "phase_probabilities_are_visible_2d_support": True,
        }
        config["evaluation_manifest_path"] = self.calibration_evaluation_manifest.text()
        config["objectives"] = [
            {
                "key": key,
                "enabled": enabled.isChecked(),
                "weight": weight.value(),
            }
            for key, (enabled, weight) in self.calibration_objectives.items()
        ]
        config["optimizer"] = {
            "engine": "optuna",
            "trial_budget_per_model": self.calibration_trials.value(),
            "sampler_seed": self.calibration_seed.value(),
            "selection_uses_identical_objective": True,
        }
        backend = str(self.calibration_backend.currentData() or "cpu-float64")
        quality = self._calibration_quality_configuration()
        config["quality_profile"] = quality
        config["execution_profile"] = {
            "primary_backend": backend,
            "reference_backend": "cpu-float64",
            "workers": self.calibration_workers.value(),
            "absolute_parity_tolerance": self.calibration_absolute_tolerance.value(),
            "relative_parity_tolerance": self.calibration_relative_tolerance.value(),
        }
        config["execution_requested"] = False
        response = self._current_indicator_response()
        return (
            apply_indicator_response_constraints(config, response)
            if response
            else config
        )

    def _validate_calibration_plan(
        self, *, show_dialog: bool = True
    ) -> dict[str, Any] | None:
        try:
            report = validate_calibration_plan(
                self.project, self._calibration_configuration()
            )
        except Exception as exc:  # noqa: BLE001
            self.calibration_summary.setText("Plan validation failed.")
            if show_dialog:
                QMessageBox.critical(self, "Plan validation failed", str(exc))
            return None
        summary = dict(report.get("summary") or {})
        contract = dict(report.get("objective_contract") or {})
        quality = dict(report.get("quality_contract") or {})
        self.calibration_summary.setText(
            f"Plan: {summary.get('error_count', 0)} errors, "
            f"{summary.get('warning_count', 0)} warnings · "
            f"{summary.get('active_objective_count', 0)} active objectives · "
            f"{summary.get('active_parameter_count', 0)} active parameters · "
            f"{summary.get('trial_budget_per_model')} trials/model · "
            f"{summary.get('primary_backend')} · "
            f"{quality.get('compute_dtype', '')} · "
            f"{quality.get('scientific_grid', '')} · objective "
            f"{str(contract.get('content_sha256') or '')[:12]} · "
            + (
                f"execution ready ({summary.get('evaluation_run_count', 0)} runs)"
                if summary.get("execution_ready")
                else "plan only; evaluator coverage incomplete"
            )
        )
        if show_dialog:
            messages = [str(issue.get("message") or "") for issue in report["issues"]]
            QMessageBox.information(
                self,
                "Calibration plan validation",
                self.calibration_summary.text()
                + ("\n\n" + "\n".join(messages) if messages else ""),
            )
        return report

    def _save_calibration_plan(self) -> None:
        config = self._calibration_configuration()
        report = self._validate_calibration_plan(show_dialog=False)
        if report is None:
            return
        try:
            revision = self.store.save_plan(
                config,
                actor=self.calibration_reviewer.text().strip(),
                reason=self.calibration_reason.text().strip(),
                validation=report,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save calibration plan", str(exc))
            return
        self.calibration_reason.clear()
        self._start_new_calibration_result_review(confirm=False)
        self.refresh()
        QMessageBox.information(
            self,
            "Calibration plan saved",
            f"Plan {str(revision['plan_id'])[:8]} was stored. No Optuna study "
            "or model training was started.",
        )

    def _current_calibration_review_run(self) -> str:
        row = self.calibration_run_review_table.currentRow()
        if row < 0:
            return ""
        item = self.calibration_run_review_table.item(row, 0)
        return item.text().strip().casefold() if item is not None else ""

    def _start_new_calibration_result_review(self, *, confirm: bool = True) -> None:
        if confirm and self.calibration_run_reviews:
            answer = QMessageBox.question(
                self,
                "Start a new result review?",
                "The current unsaved run decisions will be cleared. Saved review "
                "revisions remain unchanged.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        current = self.store.current()
        self.calibration_review_plan_id = str(current.get("plan_id") or "")
        self.calibration_review_plan_sha256 = str(current.get("content_sha256") or "")
        self.calibration_run_reviews = {}
        self.calibration_result_manifest.setText("")
        self.calibration_result_report = None
        self._refresh_calibration_run_reviews()
        self.calibration_result_review_summary.setText(
            "New review draft bound to the current calibration plan. Attach the "
            "optimizer result when it is available."
        )

    def _calibration_review_configuration(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "format": CALIBRATION_REVIEW_CONFIG_FORMAT,
            "calibration_plan_id": self.calibration_review_plan_id,
            "calibration_plan_sha256": self.calibration_review_plan_sha256,
            "calibration_result_manifest_path": self.calibration_result_manifest.text(),
            "run_reviews": {
                key: dict(value)
                for key, value in sorted(self.calibration_run_reviews.items())
            },
        }

    def _load_calibration_result_review(self) -> None:
        current = self.result_review_store.current()
        config = dict(current.get("configuration") or {})
        if not config:
            config = default_calibration_review_config(self.project)
        self.calibration_review_plan_id = str(config.get("calibration_plan_id") or "")
        self.calibration_review_plan_sha256 = str(
            config.get("calibration_plan_sha256") or ""
        )
        self.calibration_result_manifest.setText(
            str(config.get("calibration_result_manifest_path") or "")
        )
        self.calibration_run_reviews = {
            str(run_id).strip().casefold(): dict(review)
            for run_id, review in dict(config.get("run_reviews") or {}).items()
            if isinstance(review, Mapping)
        }
        self._refresh_calibration_run_reviews()
        if self.calibration_result_manifest.text():
            self._inspect_calibration_result_review(show_dialog=False)

    def _review_asset_path(self, value: str) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.is_absolute() and self.calibration_result_manifest.text():
            path = (
                Path(self.calibration_result_manifest.text()).expanduser().parent / path
            )
        return path.resolve()

    def _refresh_calibration_review_previews(self) -> None:
        self.calibration_concentration_image.set_path(
            self._review_asset_path(self.calibration_concentration_preview.text())
        )
        self.calibration_mass_plot_image.set_path(
            self._review_asset_path(self.calibration_mass_plot.text())
        )

    def _calibration_review_run_changed(
        self,
        _current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        run_id = self._current_calibration_review_run()
        review = dict(self.calibration_run_reviews.get(run_id) or {})
        self.calibration_concentration_preview.setText(
            str(review.get("concentration_preview_path") or "")
        )
        self.calibration_mass_plot.setText(str(review.get("mass_plot_path") or ""))
        self.calibration_review_note.setText(str(review.get("note") or ""))
        self._refresh_calibration_review_previews()

    @staticmethod
    def _manifest_asset_path(value: Any) -> str:
        if isinstance(value, Mapping):
            return str(value.get("path") or "")
        return str(value or "")

    def _import_calibration_review_assets(self, report: Mapping[str, Any]) -> bool:
        manifest = dict((report.get("calibration_result") or {}).get("manifest") or {})
        assets = manifest.get("review_assets") or {}
        if not isinstance(assets, Mapping):
            return False
        changed = False
        for raw_run, raw_assets in assets.items():
            run_id = str(raw_run).strip().casefold()
            if not isinstance(raw_assets, Mapping):
                continue
            current = dict(self.calibration_run_reviews.get(run_id) or {})
            concentration = self._manifest_asset_path(
                raw_assets.get("concentration_preview")
                or raw_assets.get("concentration_map")
                or raw_assets.get("concentration_preview_path")
            )
            mass_plot = self._manifest_asset_path(
                raw_assets.get("mass_plot") or raw_assets.get("mass_plot_path")
            )
            if concentration and not current.get("concentration_preview_path"):
                current["concentration_preview_path"] = concentration
                changed = True
            if mass_plot and not current.get("mass_plot_path"):
                current["mass_plot_path"] = mass_plot
                changed = True
            if current:
                self.calibration_run_reviews[run_id] = current
        return changed

    def _inspect_calibration_result_review(
        self, *, show_dialog: bool = True
    ) -> dict[str, Any] | None:
        try:
            report = validate_calibration_review_config(
                self.project, self._calibration_review_configuration()
            )
        except Exception as exc:  # noqa: BLE001
            self.calibration_result_report = None
            self.calibration_result_review_summary.setText(
                "Mass-calibration result inspection failed."
            )
            if show_dialog:
                QMessageBox.critical(self, "Result inspection failed", str(exc))
            return None
        self.calibration_result_report = report
        if self._import_calibration_review_assets(report):
            report = validate_calibration_review_config(
                self.project, self._calibration_review_configuration()
            )
            self.calibration_result_report = report
        self._refresh_calibration_run_reviews()
        summary = dict(report.get("summary") or {})
        self.calibration_result_review_summary.setText(
            f"Result review: {summary.get('error_count', 0)} errors, "
            f"{summary.get('warning_count', 0)} warnings · approved "
            f"{summary.get('approved_run_count', 0)} of "
            f"{summary.get('run_count', 0)} experiments · "
            + (
                "ready for the calibration quality gate"
                if report.get("ready")
                else "mass analysis remains blocked"
            )
        )
        if show_dialog:
            messages = [str(issue.get("message") or "") for issue in report["issues"]]
            QMessageBox.information(
                self,
                "Mass calibration result",
                self.calibration_result_review_summary.text()
                + ("\n\n" + "\n".join(messages) if messages else ""),
            )
        return report

    def _refresh_calibration_run_reviews(self, *, select_run: str = "") -> None:
        phase = PhaseInterpretationRevisionStore(self.project).current()
        selected = dict(
            (phase.get("configuration") or {}).get("selected_candidates") or {}
        )
        runs = sorted(
            {
                str(run_id).strip().casefold()
                for run_id in selected
                if str(run_id).strip()
            }
        )
        table = self.calibration_run_review_table
        previous = select_run or self._current_calibration_review_run()
        table.blockSignals(True)
        table.setRowCount(len(runs))
        selected_row = -1
        labels = {
            "pending": "Pending",
            "approved": "Approved",
            "needs_recalibration": "Needs recalibration",
        }
        for row, run_id in enumerate(runs):
            review = dict(self.calibration_run_reviews.get(run_id) or {})
            status = str(review.get("status") or "pending")
            values = (
                run_id,
                labels.get(status, status),
                str(review.get("reviewer") or ""),
                Path(str(review.get("concentration_preview_path") or "")).name,
                Path(str(review.get("mass_plot_path") or "")).name,
                str(review.get("note") or ""),
            )
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(str(value)))
            if run_id == previous:
                selected_row = row
        table.blockSignals(False)
        if selected_row >= 0:
            table.selectRow(selected_row)
        elif runs:
            table.selectRow(0)
        else:
            self.calibration_concentration_preview.setText("")
            self.calibration_mass_plot.setText("")
            self.calibration_review_note.clear()
            self._refresh_calibration_review_previews()

    def _set_calibration_run_review(self, status: str) -> None:
        run_id = self._current_calibration_review_run()
        if not run_id:
            QMessageBox.warning(
                self, "Choose an experiment", "Select one experiment first."
            )
            return
        if status == "pending":
            self.calibration_run_reviews.pop(run_id, None)
            self._refresh_calibration_run_reviews(select_run=run_id)
            self._inspect_calibration_result_review(show_dialog=False)
            return
        reviewer = self.calibration_reviewer.text().strip()
        if not reviewer:
            QMessageBox.warning(
                self,
                "Reviewer required",
                "Enter the reviewer name before recording a mass-calibration decision.",
            )
            return
        report = self._inspect_calibration_result_review(show_dialog=False)
        result = dict((report or {}).get("calibration_result") or {})
        if not result.get("valid"):
            QMessageBox.warning(
                self,
                "Valid calibration result required",
                "Attach a calibration result that matches the current plan and has "
                "passed CPU-float64 parity control.",
            )
            return
        concentration = self._review_asset_path(
            self.calibration_concentration_preview.text()
        )
        mass_plot = self._review_asset_path(self.calibration_mass_plot.text())
        if status == "approved" and (
            concentration is None
            or not concentration.is_file()
            or mass_plot is None
            or not mass_plot.is_file()
        ):
            QMessageBox.warning(
                self,
                "Review figures required",
                "Choose an available concentration map and mass plot before approval.",
            )
            return
        result_sha256 = str((result.get("reference") or {}).get("sha256") or "")
        self.calibration_run_reviews[run_id] = {
            "status": status,
            "reviewer": reviewer,
            "reviewed_utc": utc_now(),
            "note": self.calibration_review_note.text().strip(),
            "calibration_result_sha256": result_sha256,
            "concentration_preview_path": (
                str(concentration) if concentration is not None else ""
            ),
            "mass_plot_path": str(mass_plot) if mass_plot is not None else "",
            "scope": "visible_concentration_and_mass_conversion",
        }
        self._refresh_calibration_run_reviews(select_run=run_id)
        self._inspect_calibration_result_review(show_dialog=False)

    def _save_calibration_result_review(self) -> None:
        report = self._inspect_calibration_result_review(show_dialog=False)
        if report is None:
            return
        actor = self.calibration_reviewer.text().strip()
        if not actor:
            QMessageBox.warning(
                self, "Reviewer required", "Enter the reviewer name before saving."
            )
            return
        try:
            revision = self.result_review_store.save_review(
                self._calibration_review_configuration(),
                actor=actor,
                reason=self.calibration_reason.text().strip()
                or "Review concentration maps and mass calibration",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save result review", str(exc))
            return
        self.refresh()
        QMessageBox.information(
            self,
            "Result review saved",
            f"Review {str(revision['review_id'])[:8]} was stored. Production mass "
            "analysis remains blocked until every experiment is approved.",
        )

    def refresh(self) -> None:
        self._refresh_gate()
        phase = PhaseInterpretationRevisionStore(self.project).current()
        if getattr(self, "_native_fixed_plan", None):
            native = self._native_fixed_plan["native_fixed_phase_contract"]
            self.calibration_phase_reference.setText(
                "Locked production labels · shared native DarSIA response · "
                "one aqueous/gas scale pair for the series · no map-to-dose rescaling. "
                f"Calibration: {len(native.get('calibration_runs', []))} runs; "
                f"held out: {len(native.get('heldout_runs', []))} runs. "
                f"Excluded-strip fit policy: "
                f"{native.get('no_outflow_frame_policy', {}).get('excluded_strip_fit_policy', 'reported separately')}."
            )
        elif phase:
            campaign_id = str((phase.get("campaign") or {}).get("campaign_id") or "")
            self.calibration_phase_reference.setText(
                f"Current phase revision: {str(phase.get('revision_id') or '')[:12]} · "
                f"campaign {campaign_id} · content "
                f"{str(phase.get('content_sha256') or '')[:12]}"
            )
        else:
            self.calibration_phase_reference.setText(
                "No phase revision is available. Calibration planning can be edited "
                "but cannot be saved as valid."
            )
        self.calibration_revisions.clear()
        current_id = str(self.store.current().get("plan_id") or "")
        rows = list(reversed(self.store.revisions()))
        if not rows:
            self.calibration_revisions.addItem("No saved plans", "")
        for revision in rows:
            plan_id = str(revision.get("plan_id") or "")
            model = str(
                ((revision.get("configuration") or {}).get("chemistry") or {}).get(
                    "model"
                )
                or ""
            )
            self.calibration_revisions.addItem(
                f"{str(revision.get('recorded_utc') or '')[:19]} · "
                f"{model} · {plan_id[:8]}",
                plan_id,
            )
            if plan_id == current_id:
                self.calibration_revisions.setCurrentIndex(
                    self.calibration_revisions.count() - 1
                )

    def _set_status(self, status: str) -> None:
        if status in {"approved", "complete"}:
            current = self.store.current()
            if not current:
                QMessageBox.warning(
                    self,
                    "Calibration plan required",
                    "Save a valid calibration plan before approving this module.",
                )
                return
            report = validate_calibration_plan(
                self.project, dict(current.get("configuration") or {})
            )
            if not report.get("valid"):
                QMessageBox.warning(
                    self,
                    "Calibration plan is outdated",
                    "The current plan no longer matches its phase revision or "
                    "contains blocking validation errors.",
                )
                return
            review = self.result_review_store.current()
            review_report = validate_calibration_review_config(
                self.project, dict(review.get("configuration") or {})
            )
            if not review or not review_report.get("ready"):
                QMessageBox.warning(
                    self,
                    "Mass calibration review incomplete",
                    "Inspect the concentration map and mass plot, then approve "
                    "every experiment and save the result-review revision before "
                    "approving this quality gate.",
                )
                return
        super()._set_status(status)
