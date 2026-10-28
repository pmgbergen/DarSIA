"""Qt mass-analysis workflow page."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, Mapping

from PySide6.QtCore import Qt
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
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.calibration import NUMERICAL_BACKENDS, CalibrationPlanStore
from darsia.fluidflower.mass_analysis import (
    MASS_PRODUCTS,
    MassAnalysisPlanStore,
    current_optical_branch,
    default_mass_analysis_plan,
    validate_mass_analysis_plan,
)
from darsia.fluidflower.phase_interpretation import PhaseInterpretationRevisionStore
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.quality_budget import (
    QUALITY_PROFILE_NAMES,
    default_quality_profile,
    format_bytes,
    numerical_dtype_for_backend,
    summarize_quality_contract,
)
from darsia.fluidflower.ui.common import ModulePage, PathField


class MassAnalysisPage(ModulePage):
    """Prepare an auditable mass-analysis task without executing it."""

    _PRODUCT_LABELS: ClassVar[dict[str, str]] = {
        "mass_total": "Raw total mass field",
        "mass_aqueous": "Raw aqueous mass field",
        "mass_free_gas": "Raw free-gas mass field",
        "transport_weight": "Permeability-based transport weight",
        "concentration_aqueous": "Aqueous concentration",
        "saturation_free_gas": "Free-gas saturation",
        "integrated_mass_csv": "Integrated mass table",
        "preview_images": "Preview images",
        "diagnostic_injected_mass_projection": (
            "Injected-mass projection (diagnostic)"
        ),
    }

    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__(project, "mass_analysis")
        self.store = MassAnalysisPlanStore(project)
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "This page creates an immutable mass-analysis task. It preserves raw "
            "detected mass, treats injected-mass projection as diagnostic, and "
            "never starts analysis from the Qt process."
        )

        self.mass_upstream_reference = QLabel()
        self.mass_upstream_reference.setObjectName("integrationNotice")
        self.mass_upstream_reference.setWordWrap(True)
        self.work_layout.addWidget(self.mass_upstream_reference)

        branch_panel = QFrame()
        branch_panel.setFrameShape(QFrame.Shape.StyledPanel)
        branch_layout = QFormLayout(branch_panel)
        self.mass_optical_branch = QComboBox()
        self.mass_optical_branch.setToolTip(
            "Choose the checksum-linked image correction and color-path revision that "
            "the mass conversion must use. This choice is stored in every result."
        )
        self.mass_optical_branch.currentIndexChanged.connect(
            self._optical_branch_changed
        )
        self.mass_optical_branch_status = QLabel()
        self.mass_optical_branch_status.setObjectName("secondaryText")
        self.mass_optical_branch_status.setWordWrap(True)
        branch_layout.addRow("Correction + color-path branch", self.mass_optical_branch)
        branch_layout.addRow("Selection", self.mass_optical_branch_status)
        self.work_layout.addWidget(branch_panel)

        result_panel = QFrame()
        result_panel.setFrameShape(QFrame.Shape.StyledPanel)
        result_layout = QFormLayout(result_panel)
        self.mass_calibration_result = PathField(
            directory=False,
            file_filter="Calibration result (*.json);;All files (*)",
            placeholder="Optional while drafting; required before execution",
        )
        result_layout.addRow(
            "Calibration-result manifest", self.mass_calibration_result
        )
        self.mass_input_manifest = PathField(
            directory=False,
            file_filter="Prepared mass inputs (*.json);;All files (*)",
            placeholder="Hash-indexed optical, phase, geometry, and protocol fields",
        )
        result_layout.addRow("Prepared mass-input manifest", self.mass_input_manifest)
        self.find_mass_input_manifest = QPushButton("Use latest prepared inputs")
        self.find_mass_input_manifest.setToolTip(
            "Find the newest compatible numerical input manifest in this project"
        )
        self.find_mass_input_manifest.clicked.connect(
            self._use_latest_mass_input_manifest
        )
        result_layout.addRow("Project results", self.find_mass_input_manifest)
        self.mass_output_location = QLabel(
            f"Default: {project.default_result_root() / 'mass-analysis'}. "
            "The final destination, including an external disk or NAS for large NPZ "
            "products, can be changed for Mass analysis under Production."
        )
        self.mass_output_location.setObjectName("secondaryText")
        self.mass_output_location.setWordWrap(True)
        result_layout.addRow("Generated output", self.mass_output_location)
        result_note = QLabel(
            "A production result must identify the selected chemistry model, "
            "calibrated parameter artifact, objective fingerprint, and successful "
            "CPU-float64 parity check. Prepared inputs must contain synchronized "
            "numerical fields rather than rendered phase-map images."
        )
        result_note.setObjectName("secondaryText")
        result_note.setWordWrap(True)
        result_layout.addRow("Readiness requirement", result_note)
        self.work_layout.addWidget(result_panel)

        run_heading = QLabel("Run and phase bindings")
        run_heading.setObjectName("sectionTitle")
        self.work_layout.addWidget(run_heading)
        self.mass_runs = QTableWidget(0, 2)
        self.mass_runs.setHorizontalHeaderLabels(["Run", "Selected phase candidate"])
        self.mass_runs.verticalHeader().setVisible(False)
        self.mass_runs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.mass_runs.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.mass_runs.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.mass_runs.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.mass_runs.setMaximumHeight(180)
        self.work_layout.addWidget(self.mass_runs)

        settings = QSplitter(Qt.Orientation.Horizontal)
        selection_panel = QFrame()
        selection_panel.setFrameShape(QFrame.Shape.StyledPanel)
        selection_layout = QFormLayout(selection_panel)
        self.mass_maximum_images = QSpinBox()
        self.mass_maximum_images.setRange(1, 100_000)
        self.mass_maximum_images.setSingleStep(10)
        self.mass_maximum_images.setKeyboardTracking(False)
        self.mass_drift_tolerance = QDoubleSpinBox()
        self.mass_drift_tolerance.setRange(0.0, 1440.0)
        self.mass_drift_tolerance.setDecimals(2)
        self.mass_drift_tolerance.setSuffix(" min")
        self.mass_same_times = QCheckBox("Use the same target times for every run")
        self.mass_same_times.setChecked(True)
        self.mass_same_times.setEnabled(False)
        self.mass_chronological = QCheckBox("Publish chronologically")
        self.mass_chronological.setChecked(True)
        self.mass_chronological.setEnabled(False)
        self.mass_limit_start = QCheckBox("Start at")
        self.mass_start_hours = QDoubleSpinBox()
        self.mass_start_hours.setRange(-100_000.0, 100_000.0)
        self.mass_start_hours.setDecimals(3)
        self.mass_start_hours.setSuffix(" h")
        self.mass_limit_start.toggled.connect(self.mass_start_hours.setEnabled)
        start_row = QWidget()
        start_layout = QHBoxLayout(start_row)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.addWidget(self.mass_limit_start)
        start_layout.addWidget(self.mass_start_hours)
        self.mass_limit_end = QCheckBox("End at")
        self.mass_end_hours = QDoubleSpinBox()
        self.mass_end_hours.setRange(-100_000.0, 100_000.0)
        self.mass_end_hours.setDecimals(3)
        self.mass_end_hours.setSuffix(" h")
        self.mass_limit_end.toggled.connect(self.mass_end_hours.setEnabled)
        end_row = QWidget()
        end_layout = QHBoxLayout(end_row)
        end_layout.setContentsMargins(0, 0, 0, 0)
        end_layout.addWidget(self.mass_limit_end)
        end_layout.addWidget(self.mass_end_hours)
        self.mass_landmarks = QLineEdit()
        self.mass_landmarks.setPlaceholderText("e.g. 0, 1, 2, 12, 24, 48")
        self.mass_publication_gib = QDoubleSpinBox()
        self.mass_publication_gib.setRange(0.0, 1_000_000.0)
        self.mass_publication_gib.setDecimals(2)
        self.mass_publication_gib.setSuffix(" GiB")
        self.mass_publication_gib.setSpecialValueText("No limit")
        selection_layout.addRow("Maximum images / run", self.mass_maximum_images)
        selection_layout.addRow("Time window start", start_row)
        selection_layout.addRow("Time window end", end_row)
        selection_layout.addRow("Required landmark times", self.mass_landmarks)
        selection_layout.addRow("Publication budget", self.mass_publication_gib)
        selection_layout.addRow("Allowed time drift", self.mass_drift_tolerance)
        selection_layout.addRow("Cross-run sampling", self.mass_same_times)
        selection_layout.addRow("Order", self.mass_chronological)
        settings.addWidget(selection_panel)

        physics_panel = QFrame()
        physics_panel.setFrameShape(QFrame.Shape.StyledPanel)
        physics_layout = QFormLayout(physics_panel)
        hybrid_engine = QLabel(
            "Selected phase support + corrected optical signal + DarSIA CO2 mass model"
        )
        hybrid_engine.setWordWrap(True)
        phase_partition = QLabel("Aqueous and free gas share pore volume exclusively")
        phase_partition.setWordWrap(True)
        geometry_rule = QLabel(
            "Pixel area x approved local depth x approved local porosity"
        )
        geometry_rule.setWordWrap(True)
        mass_rule = QLabel(
            "Raw detected mass is authoritative. Injection is a reference; "
            "mass conservation is never forced."
        )
        mass_rule.setWordWrap(True)
        for label in (hybrid_engine, phase_partition, geometry_rule, mass_rule):
            label.setObjectName("secondaryText")
        physics_layout.addRow("Hybrid engine", hybrid_engine)
        physics_layout.addRow("Phase partition", phase_partition)
        physics_layout.addRow("2D to physical volume", geometry_rule)
        physics_layout.addRow("Mass policy", mass_rule)
        settings.addWidget(physics_panel)
        settings.setStretchFactor(0, 1)
        settings.setStretchFactor(1, 1)
        self.work_layout.addWidget(settings)

        output_splitter = QSplitter(Qt.Orientation.Horizontal)
        products_panel = QFrame()
        products_panel.setFrameShape(QFrame.Shape.StyledPanel)
        products_layout = QVBoxLayout(products_panel)
        products_title = QLabel("Products")
        products_title.setObjectName("sectionTitle")
        products_layout.addWidget(products_title)
        product_grid = QGridLayout()
        self.mass_products: dict[str, QCheckBox] = {}
        for index, key in enumerate(MASS_PRODUCTS):
            checkbox = QCheckBox(self._PRODUCT_LABELS[key])
            self.mass_products[key] = checkbox
            product_grid.addWidget(checkbox, index // 2, index % 2)
        products_layout.addLayout(product_grid)
        products_layout.addStretch(1)
        output_splitter.addWidget(products_panel)

        wasserstein_panel = QFrame()
        wasserstein_panel.setFrameShape(QFrame.Shape.StyledPanel)
        wasserstein_layout = QFormLayout(wasserstein_panel)
        self.mass_wasserstein_cache = QCheckBox(
            "Generate mass-preserving comparison cache"
        )
        self.mass_wasserstein_cache.toggled.connect(self._mass_cache_toggled)
        self.mass_wasserstein_cell_size = QDoubleSpinBox()
        self.mass_wasserstein_cell_size.setRange(0.001, 100.0)
        self.mass_wasserstein_cell_size.setDecimals(3)
        self.mass_wasserstein_cell_size.setSuffix(" mm")
        cache_note = QLabel(
            "Set the target cell size in millimetres. Use the same physical "
            "cell size for the datasets you compare."
        )
        cache_note.setObjectName("secondaryText")
        cache_note.setWordWrap(True)
        wasserstein_layout.addRow("Wasserstein input", self.mass_wasserstein_cache)
        wasserstein_layout.addRow("Target cell size", self.mass_wasserstein_cell_size)
        wasserstein_layout.addRow("Resolution policy", cache_note)
        output_splitter.addWidget(wasserstein_panel)
        output_splitter.setStretchFactor(0, 1)
        output_splitter.setStretchFactor(1, 1)
        self.work_layout.addWidget(output_splitter)

        quality_panel = QFrame()
        quality_panel.setFrameShape(QFrame.Shape.StyledPanel)
        quality_layout = QGridLayout(quality_panel)
        self.mass_quality_profile = QComboBox()
        for profile in QUALITY_PROFILE_NAMES:
            self.mass_quality_profile.addItem(profile.title(), profile)
        self.mass_quality_profile.currentIndexChanged.connect(
            self._mass_quality_profile_changed
        )
        self.mass_preview_dimension = QSpinBox()
        self.mass_preview_dimension.setRange(256, 16_384)
        self.mass_preview_dimension.setSuffix(" px")
        self.mass_preview_quality = QSpinBox()
        self.mass_preview_quality.setRange(1, 100)
        self.mass_preview_quality.setSuffix(" % JPEG")
        self.mass_resolution_policy = QComboBox()
        self.mass_resolution_policy.addItem(
            "Native registered grid", "native_registered_grid"
        )
        self.mass_resolution_policy.addItem("Scale factor", "scale_factor")
        self.mass_resolution_policy.addItem("Maximum dimension", "maximum_dimension")
        self.mass_resolution_policy.currentIndexChanged.connect(
            self._mass_resolution_changed
        )
        self.mass_scale_percent = QDoubleSpinBox()
        self.mass_scale_percent.setRange(5.0, 100.0)
        self.mass_scale_percent.setDecimals(1)
        self.mass_scale_percent.setSuffix(" %")
        self.mass_maximum_dimension = QSpinBox()
        self.mass_maximum_dimension.setRange(64, 32_768)
        self.mass_maximum_dimension.setSuffix(" px")
        analysis_grid_row = QWidget()
        analysis_grid_layout = QHBoxLayout(analysis_grid_row)
        analysis_grid_layout.setContentsMargins(0, 0, 0, 0)
        analysis_grid_layout.addWidget(self.mass_resolution_policy)
        analysis_grid_layout.addWidget(self.mass_scale_percent)
        analysis_grid_layout.addWidget(self.mass_maximum_dimension)
        self.mass_parity_percent = QDoubleSpinBox()
        self.mass_parity_percent.setRange(0.01, 100.0)
        self.mass_parity_percent.setDecimals(2)
        self.mass_parity_percent.setSuffix(" %")
        self.mass_absolute_tolerance = QDoubleSpinBox()
        self.mass_relative_tolerance = QDoubleSpinBox()
        for tolerance in (
            self.mass_absolute_tolerance,
            self.mass_relative_tolerance,
        ):
            tolerance.setRange(1.0e-10, 1.0)
            tolerance.setDecimals(10)
            tolerance.setSingleStep(1.0e-5)
        quality_note = QLabel(
            "Precision and analysis resolution are independent. A reduced grid is "
            "used by the numerical calculation and recorded as a scientific "
            "approximation; preview resolution remains separate. Every non-reference "
            "result, including CPU-float32, retains CPU-float64 parity sampling."
        )
        quality_note.setObjectName("secondaryText")
        quality_note.setWordWrap(True)
        quality_layout.addWidget(QLabel("Quality profile"), 0, 0)
        quality_layout.addWidget(self.mass_quality_profile, 0, 1)
        quality_layout.addWidget(QLabel("Preview maximum"), 0, 2)
        quality_layout.addWidget(self.mass_preview_dimension, 0, 3)
        quality_layout.addWidget(QLabel("Preview quality"), 0, 4)
        quality_layout.addWidget(self.mass_preview_quality, 0, 5)
        quality_layout.addWidget(QLabel("Numerical analysis grid"), 1, 0)
        quality_layout.addWidget(analysis_grid_row, 1, 1, 1, 5)
        quality_layout.addWidget(QLabel("CPU-float64 parity sample"), 2, 0)
        quality_layout.addWidget(self.mass_parity_percent, 2, 1)
        quality_layout.addWidget(quality_note, 2, 2, 1, 4)
        quality_layout.addWidget(QLabel("Absolute parity tolerance"), 3, 0)
        quality_layout.addWidget(self.mass_absolute_tolerance, 3, 1)
        quality_layout.addWidget(QLabel("Relative parity tolerance"), 3, 2)
        quality_layout.addWidget(self.mass_relative_tolerance, 3, 3)
        self.mass_quality_basis = QLabel()
        self.mass_quality_basis.setObjectName("integrationNotice")
        self.mass_quality_basis.setWordWrap(True)
        quality_layout.addWidget(QLabel("Current working basis"), 4, 0)
        quality_layout.addWidget(self.mass_quality_basis, 4, 1, 1, 5)
        self.work_layout.addWidget(quality_panel)

        execution_panel = QFrame()
        execution_panel.setFrameShape(QFrame.Shape.StyledPanel)
        execution_layout = QGridLayout(execution_panel)
        self.mass_backend = QComboBox()
        for backend in NUMERICAL_BACKENDS:
            self.mass_backend.addItem(backend, backend)
        self.mass_backend.currentIndexChanged.connect(self._mass_backend_changed)
        self.mass_dtype = QLabel()
        self.mass_workers = QSpinBox()
        self.mass_workers.setRange(1, 256)
        self.mass_prefetch = QSpinBox()
        self.mass_prefetch.setRange(0, 1)
        self.mass_checkpoint = QSpinBox()
        self.mass_checkpoint.setRange(1, 100_000)
        self.mass_local_cache = QCheckBox("Use local read-through cache")
        self.mass_atomic_publish = QCheckBox("Verify and publish atomically")
        self.mass_atomic_publish.setChecked(True)
        self.mass_atomic_publish.setEnabled(False)
        execution_layout.addWidget(QLabel("Planned backend"), 0, 0)
        execution_layout.addWidget(self.mass_backend, 0, 1)
        execution_layout.addWidget(QLabel("Numerical dtype"), 0, 2)
        execution_layout.addWidget(self.mass_dtype, 0, 3)
        execution_layout.addWidget(QLabel("Workers"), 1, 0)
        execution_layout.addWidget(self.mass_workers, 1, 1)
        execution_layout.addWidget(QLabel("Image prefetch"), 1, 2)
        execution_layout.addWidget(self.mass_prefetch, 1, 3)
        execution_layout.addWidget(QLabel("CSV checkpoint"), 2, 0)
        execution_layout.addWidget(self.mass_checkpoint, 2, 1)
        execution_layout.addWidget(self.mass_local_cache, 2, 2)
        execution_layout.addWidget(self.mass_atomic_publish, 2, 3)
        self.work_layout.addWidget(execution_panel)

        review_panel = QFrame()
        review_panel.setFrameShape(QFrame.Shape.StyledPanel)
        review_layout = QFormLayout(review_panel)
        self.mass_reviewer = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.mass_reason = QLineEdit()
        self.mass_reason.setPlaceholderText(
            "Why is this frame, physics, and export contract appropriate?"
        )
        review_layout.addRow("Reviewer", self.mass_reviewer)
        review_layout.addRow("Plan note", self.mass_reason)
        self.work_layout.addWidget(review_panel)

        actions = QHBoxLayout()
        self.validate_mass_plan_button = QPushButton("Validate plan")
        self.validate_mass_plan_button.clicked.connect(self._validate_mass_plan)
        actions.addWidget(self.validate_mass_plan_button)
        self.save_mass_plan_button = QPushButton("Save plan")
        self.save_mass_plan_button.clicked.connect(self._save_mass_plan)
        actions.addWidget(self.save_mass_plan_button)
        actions.addStretch(1)
        actions.addWidget(QLabel("Plan history"))
        self.mass_revisions = QComboBox()
        self.mass_revisions.setMinimumWidth(300)
        actions.addWidget(self.mass_revisions)
        self.work_layout.addLayout(actions)
        self.mass_summary = QLabel("No mass-analysis plan validated.")
        self.mass_summary.setObjectName("secondaryText")
        self.mass_summary.setWordWrap(True)
        self.work_layout.addWidget(self.mass_summary)

        for field in (
            self.mass_preview_dimension,
            self.mass_preview_quality,
            self.mass_scale_percent,
            self.mass_maximum_dimension,
        ):
            field.valueChanged.connect(self._update_mass_quality_basis)

        self._load_mass_plan()
        self.refresh()

    def _mass_cache_toggled(self, enabled: bool) -> None:
        self.mass_wasserstein_cell_size.setEnabled(enabled)

    @staticmethod
    def _same_optical_branch(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        keys = (
            "correction_revision_id",
            "correction_revision_sha256",
            "phase_revision_id",
            "phase_revision_sha256",
        )
        return all(
            str(left.get(key) or "") == str(right.get(key) or "") for key in keys
        )

    def _populate_optical_branch(
        self,
        selected: Mapping[str, Any] | None = None,
    ) -> None:
        requested = dict(selected or {})
        self.mass_optical_branch.blockSignals(True)
        self.mass_optical_branch.clear()
        self.mass_optical_branch.addItem(
            "Select a correction and color-path branch...", None
        )
        branch = current_optical_branch(self.project)
        if branch.get("compatible"):
            algorithm = str(branch.get("correction_algorithm") or "correction")
            algorithm_label = (
                "Global ColorChecker 24-patch"
                if "uniform24" in algorithm
                else (
                    "Original DarSIA ColorCorrection"
                    if "darsia-baseline" in algorithm
                    else "No radiometric correction"
                )
            )
            label = (
                f"{algorithm_label} · correction "
                f"{str(branch.get('correction_revision_id') or '')[:8]} · color paths "
                f"{str(branch.get('phase_revision_id') or '')[:8]}"
            )
            self.mass_optical_branch.addItem(label, deepcopy(branch))
            if requested and self._same_optical_branch(requested, branch):
                self.mass_optical_branch.setCurrentIndex(1)
        self.mass_optical_branch.blockSignals(False)
        self._optical_branch_changed()

    def _optical_branch_changed(self, _value: Any = None) -> None:
        selected = self.mass_optical_branch.currentData()
        if not isinstance(selected, Mapping):
            branch = current_optical_branch(self.project)
            self.mass_optical_branch_status.setText(
                (
                    "Choose the branch explicitly. Mass-plan validation remains blocked "
                    "until the correction and its matching color paths are selected."
                    if branch.get("compatible")
                    else "No compatible branch is available. Rebuild and approve the color "
                    "paths from the active image-correction revision first."
                )
            )
            return
        self.mass_optical_branch_status.setText(
            "Selected and checksum matched. This exact correction and color-path pair "
            "will be written to the mass result and inherited by later analyses."
        )

    def _mass_backend_changed(self) -> None:
        backend = str(self.mass_backend.currentData() or "cpu-float64")
        try:
            self.mass_dtype.setText(numerical_dtype_for_backend(backend))
        except ValueError:
            self.mass_dtype.setText("Unknown")
        self._update_mass_quality_basis()

    def _mass_quality_profile_changed(self) -> None:
        profile = default_quality_profile(
            str(self.mass_quality_profile.currentData() or "production")
        )
        self.mass_preview_dimension.setValue(
            int(profile["preview"]["maximum_dimension_px"])
        )
        self.mass_preview_quality.setValue(int(profile["preview"]["jpeg_quality"]))
        numerical = dict(profile.get("numerical_resolution") or {})
        policy_index = self.mass_resolution_policy.findData(
            numerical.get("policy") or "native_registered_grid"
        )
        self.mass_resolution_policy.setCurrentIndex(max(0, policy_index))
        self.mass_scale_percent.setValue(
            100.0 * float(numerical.get("scale_factor") or 1.0)
        )
        self.mass_maximum_dimension.setValue(
            int(numerical.get("maximum_dimension_px") or 2048)
        )
        self.mass_parity_percent.setValue(
            100.0 * float(profile["precision"]["cpu_float64_parity_fraction"])
        )
        self.mass_absolute_tolerance.setValue(
            float(profile["precision"]["absolute_parity_tolerance"])
        )
        self.mass_relative_tolerance.setValue(
            float(profile["precision"]["relative_parity_tolerance"])
        )
        self._mass_resolution_changed()

    def _mass_resolution_changed(self) -> None:
        policy = str(
            self.mass_resolution_policy.currentData() or "native_registered_grid"
        )
        self.mass_scale_percent.setVisible(policy == "scale_factor")
        self.mass_maximum_dimension.setVisible(policy == "maximum_dimension")
        self._update_mass_quality_basis()

    def _mass_quality_configuration(self) -> dict[str, Any]:
        quality = default_quality_profile(
            str(self.mass_quality_profile.currentData() or "production")
        )
        resolution_policy = str(
            self.mass_resolution_policy.currentData() or "native_registered_grid"
        )
        quality["numerical_resolution"].update(
            {
                "policy": resolution_policy,
                "scale_factor": self.mass_scale_percent.value() / 100.0,
                "maximum_dimension_px": (
                    self.mass_maximum_dimension.value()
                    if resolution_policy == "maximum_dimension"
                    else None
                ),
            }
        )
        quality["preview"].update(
            {
                "maximum_dimension_px": self.mass_preview_dimension.value(),
                "jpeg_quality": self.mass_preview_quality.value(),
            }
        )
        backend = str(self.mass_backend.currentData() or "cpu-float64")
        quality["precision"].update(
            {
                "compute_dtype": numerical_dtype_for_backend(backend),
                "cpu_float64_parity_fraction": (
                    self.mass_parity_percent.value() / 100.0
                ),
                "absolute_parity_tolerance": self.mass_absolute_tolerance.value(),
                "relative_parity_tolerance": self.mass_relative_tolerance.value(),
            }
        )
        return quality

    def _update_mass_quality_basis(self) -> None:
        if not hasattr(self, "mass_quality_basis"):
            return
        backend = str(self.mass_backend.currentData() or "cpu-float64")
        try:
            summary = summarize_quality_contract(
                self._mass_quality_configuration(),
                backend=backend,
            )
            self.mass_quality_basis.setText(
                "Mass analysis only: "
                + summary["label"]
                + " | upstream revisions remain unchanged"
            )
        except (TypeError, ValueError):
            self.mass_quality_basis.setText(
                "Mass-analysis working basis is incomplete."
            )

    @staticmethod
    def _parse_landmark_times(text: str) -> list[float]:
        values: list[float] = []
        for token in text.replace(";", ",").split(","):
            stripped = token.strip()
            if stripped:
                values.append(float(stripped.replace(",", ".")))
        return values

    def _use_latest_mass_input_manifest(self) -> None:
        roots = (
            self.project.root / "results" / "mass_input_preparation",
            self.project.root / "results" / "execution",
        )
        candidates = sorted(
            (
                path
                for root in roots
                if root.is_dir()
                for path in root.rglob("mass_input_manifest.json")
                if path.is_file()
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for path in candidates:
            config = self._mass_configuration()
            config["mass_input_manifest_path"] = str(path)
            try:
                report = validate_mass_analysis_plan(self.project, config)
            except Exception:  # noqa: BLE001
                report = {}
            if bool((report.get("mass_inputs") or {}).get("valid")):
                self.mass_input_manifest.setText(str(path))
                self._validate_mass_plan(show_dialog=False)
                return
        QMessageBox.information(
            self,
            "No compatible prepared inputs",
            "No hash-valid mass-input manifest matching the current phase "
            "selection was found in this project's results.",
        )

    def _load_mass_plan(self) -> None:
        current = self.store.current()
        config = dict(current.get("configuration") or {})
        if not config:
            config = default_mass_analysis_plan(self.project)
            self._populate_optical_branch()
        else:
            self._populate_optical_branch(dict(config.get("optical_branch") or {}))
        self.mass_calibration_result.setText(
            str(config.get("calibration_result_manifest_path") or "")
        )
        self.mass_input_manifest.setText(
            str(config.get("mass_input_manifest_path") or "")
        )
        selection = dict(config.get("image_selection") or {})
        self.mass_maximum_images.setValue(
            int(selection.get("maximum_images_per_run") or 100)
        )
        self.mass_drift_tolerance.setValue(
            float(selection.get("drift_tolerance_seconds") or 0.0) / 60.0
        )
        budget = dict(selection.get("budget") or {})
        start = budget.get("start_time_hours")
        end = budget.get("end_time_hours")
        self.mass_limit_start.setChecked(start is not None)
        self.mass_start_hours.setValue(0.0 if start is None else float(start))
        self.mass_start_hours.setEnabled(start is not None)
        self.mass_limit_end.setChecked(end is not None)
        self.mass_end_hours.setValue(0.0 if end is None else float(end))
        self.mass_end_hours.setEnabled(end is not None)
        self.mass_landmarks.setText(
            ", ".join(
                f"{float(value):g}"
                for value in budget.get("required_landmark_times_hours") or []
            )
        )
        maximum_bytes = budget.get("maximum_publication_bytes")
        self.mass_publication_gib.setValue(
            0.0 if maximum_bytes is None else float(maximum_bytes) / float(1024**3)
        )
        quality = dict(config.get("quality_profile") or {})
        if not quality:
            quality = default_quality_profile("production")
        quality_index = self.mass_quality_profile.findData(
            quality.get("name") or "production"
        )
        self.mass_quality_profile.setCurrentIndex(max(0, quality_index))
        self.mass_preview_dimension.setValue(
            int((quality.get("preview") or {}).get("maximum_dimension_px") or 1920)
        )
        self.mass_preview_quality.setValue(
            int((quality.get("preview") or {}).get("jpeg_quality") or 92)
        )
        numerical = dict(quality.get("numerical_resolution") or {})
        resolution_index = self.mass_resolution_policy.findData(
            numerical.get("policy") or "native_registered_grid"
        )
        self.mass_resolution_policy.setCurrentIndex(max(0, resolution_index))
        self.mass_scale_percent.setValue(
            100.0 * float(numerical.get("scale_factor") or 1.0)
        )
        self.mass_maximum_dimension.setValue(
            int(numerical.get("maximum_dimension_px") or 2048)
        )
        self.mass_parity_percent.setValue(
            100.0
            * float(
                (quality.get("precision") or {}).get(
                    "cpu_float64_parity_fraction", 0.05
                )
            )
        )
        self.mass_absolute_tolerance.setValue(
            float(
                (quality.get("precision") or {}).get(
                    "absolute_parity_tolerance", 1.0e-5
                )
            )
        )
        self.mass_relative_tolerance.setValue(
            float(
                (quality.get("precision") or {}).get(
                    "relative_parity_tolerance", 1.0e-4
                )
            )
        )
        exports = dict(config.get("exports") or {})
        selected_products = {str(value) for value in exports.get("products") or []}
        for key, checkbox in self.mass_products.items():
            checkbox.setChecked(key in selected_products)
        cache = dict(exports.get("wasserstein_cache") or {})
        cache_enabled = cache.get("enabled") is True
        self.mass_wasserstein_cache.setChecked(cache_enabled)
        cell_size = cache.get("target_cell_size_m")
        self.mass_wasserstein_cell_size.setValue(
            1.0 if cell_size is None else 1000.0 * float(cell_size)
        )
        self._mass_cache_toggled(cache_enabled)
        execution = dict(config.get("execution_profile") or {})
        backend_index = self.mass_backend.findData(execution.get("primary_backend"))
        self.mass_backend.setCurrentIndex(max(0, backend_index))
        self.mass_workers.setValue(int(execution.get("workers") or 1))
        self.mass_prefetch.setValue(int(execution.get("prefetch_images") or 0))
        self.mass_checkpoint.setValue(int(execution.get("csv_checkpoint_images") or 10))
        self.mass_local_cache.setChecked(
            bool(execution.get("local_read_through_cache", True))
        )
        self._mass_backend_changed()
        self._mass_resolution_changed()
        self._update_mass_quality_basis()

    def _mass_configuration(self) -> dict[str, Any]:
        config = default_mass_analysis_plan(self.project)
        selected_branch = self.mass_optical_branch.currentData()
        config["optical_branch"] = (
            {**deepcopy(dict(selected_branch)), "selection_confirmed": True}
            if isinstance(selected_branch, Mapping)
            else {"schema": 1, "selection_confirmed": False}
        )
        calibration = CalibrationPlanStore(self.project).current()
        config["calibration_plan_id"] = str(calibration.get("plan_id") or "")
        config["calibration_plan_sha256"] = str(calibration.get("content_sha256") or "")
        config["calibration_result_manifest_path"] = self.mass_calibration_result.text()
        config["mass_input_manifest_path"] = self.mass_input_manifest.text()
        config["image_selection"] = {
            "policy": "synchronized_representative_times",
            "maximum_images_per_run": self.mass_maximum_images.value(),
            "drift_tolerance_seconds": 60.0 * self.mass_drift_tolerance.value(),
            "require_same_target_times": True,
            "chronological": True,
            "budget": {
                "policy": "uniform_with_required_landmarks",
                "start_time_hours": (
                    self.mass_start_hours.value()
                    if self.mass_limit_start.isChecked()
                    else None
                ),
                "end_time_hours": (
                    self.mass_end_hours.value()
                    if self.mass_limit_end.isChecked()
                    else None
                ),
                "required_landmark_times_hours": self._parse_landmark_times(
                    self.mass_landmarks.text()
                ),
                "preserve_window_endpoints": True,
                "maximum_time_points": self.mass_maximum_images.value(),
                "maximum_solver_tasks": None,
                "maximum_publication_bytes": (
                    round(self.mass_publication_gib.value() * 1024**3)
                    if self.mass_publication_gib.value() > 0.0
                    else None
                ),
            },
        }
        backend = str(self.mass_backend.currentData())
        quality = self._mass_quality_configuration()
        config["quality_profile"] = quality
        cache_enabled = self.mass_wasserstein_cache.isChecked()
        config["exports"] = {
            "products": [
                key for key in MASS_PRODUCTS if self.mass_products[key].isChecked()
            ],
            "numerical_format": "npz",
            "numerical_dtype": numerical_dtype_for_backend(backend),
            "wasserstein_cache": {
                "enabled": cache_enabled,
                "resolution_policy": (
                    "explicit_physical_cell_size"
                    if cache_enabled
                    else "defer_to_wasserstein_plan"
                ),
                "target_cell_size_m": (
                    self.mass_wasserstein_cell_size.value() / 1000.0
                    if cache_enabled
                    else None
                ),
                "mass_preserving": True,
            },
        }
        config["execution_profile"] = {
            "primary_backend": backend,
            "reference_backend": "cpu-float64",
            "workers": self.mass_workers.value(),
            "prefetch_images": self.mass_prefetch.value(),
            "csv_checkpoint_images": self.mass_checkpoint.value(),
            "local_read_through_cache": self.mass_local_cache.isChecked(),
            "atomic_publish": True,
        }
        config["execution_requested"] = False
        return config

    def _validate_mass_plan(self, *, show_dialog: bool = True) -> dict[str, Any] | None:
        try:
            report = validate_mass_analysis_plan(
                self.project, self._mass_configuration()
            )
        except Exception as exc:  # noqa: BLE001
            self.mass_summary.setText("Mass-plan validation failed.")
            if show_dialog:
                QMessageBox.critical(self, "Mass-plan validation failed", str(exc))
            return None
        summary = dict(report.get("summary") or {})
        quality = dict(report.get("quality_contract") or {})
        readiness = "EXECUTION READY" if report.get("execution_ready") else "DRAFT"
        self.mass_summary.setText(
            f"{readiness} · {summary.get('error_count', 0)} errors · "
            f"{summary.get('warning_count', 0)} warnings · "
            f"{summary.get('run_count', 0)} runs · up to "
            f"{summary.get('maximum_images_per_run')} images/run · "
            f"{summary.get('target_time_count', 0)} prepared times · "
            f"{summary.get('selected_time_count', 0)} selected · "
            f"about {format_bytes(summary.get('estimated_publication_bytes', 0))} · "
            f"{summary.get('primary_backend')} · "
            f"{quality.get('compute_dtype', '')} · "
            f"{quality.get('scientific_grid', '')}"
        )
        if show_dialog:
            messages = [str(issue.get("message") or "") for issue in report["issues"]]
            QMessageBox.information(
                self,
                "Mass-plan validation",
                self.mass_summary.text()
                + ("\n\n" + "\n".join(messages) if messages else ""),
            )
        return report

    def _save_mass_plan(self) -> None:
        config = self._mass_configuration()
        report = self._validate_mass_plan(show_dialog=False)
        if report is None:
            return
        try:
            revision = self.store.save_plan(
                config,
                actor=self.mass_reviewer.text().strip(),
                reason=self.mass_reason.text().strip(),
                validation=report,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save mass plan", str(exc))
            return
        self.mass_reason.clear()
        self.refresh()
        state = "execution ready" if revision["execution"]["ready"] else "a draft"
        QMessageBox.information(
            self,
            "Mass plan saved",
            f"Plan {str(revision['plan_id'])[:8]} was stored as {state}. "
            "No image analysis was started.",
        )

    def refresh(self) -> None:
        self._refresh_gate()
        selected_branch = self.mass_optical_branch.currentData()
        self._populate_optical_branch(
            dict(selected_branch) if isinstance(selected_branch, Mapping) else None
        )
        calibration = CalibrationPlanStore(self.project).current()
        phase = PhaseInterpretationRevisionStore(self.project).current()
        correction_id = str(
            (phase.get("upstream") or {}).get("image_correction_revision_id") or ""
        )
        if calibration:
            self.mass_upstream_reference.setText(
                f"Calibration plan {str(calibration.get('plan_id') or '')[:12]} · "
                f"objective "
                f"{str(calibration.get('selection_contract_sha256') or '')[:12]} · "
                f"phase revision {str(phase.get('revision_id') or '')[:12]} · "
                f"correction revision {correction_id[:12]}"
            )
        else:
            self.mass_upstream_reference.setText(
                "No calibration plan is available. A mass plan cannot yet be valid."
            )
        bindings = default_mass_analysis_plan(self.project)["run_bindings"]
        self.mass_runs.setRowCount(len(bindings))
        for row, binding in enumerate(bindings):
            self.mass_runs.setItem(row, 0, QTableWidgetItem(binding["run_id"]))
            self.mass_runs.setItem(
                row, 1, QTableWidgetItem(binding["phase_candidate_id"])
            )
        self.mass_revisions.clear()
        current_id = str(self.store.current().get("plan_id") or "")
        rows = list(reversed(self.store.revisions()))
        if not rows:
            self.mass_revisions.addItem("No saved plans", "")
        for revision in rows:
            plan_id = str(revision.get("plan_id") or "")
            ready = bool((revision.get("execution") or {}).get("ready"))
            self.mass_revisions.addItem(
                f"{str(revision.get('recorded_utc') or '')[:19]} · "
                f"{'ready' if ready else 'draft'} · {plan_id[:8]}",
                plan_id,
            )
            if plan_id == current_id:
                self.mass_revisions.setCurrentIndex(self.mass_revisions.count() - 1)

    def _set_status(self, status: str) -> None:
        if status in {"approved", "complete"}:
            current = self.store.current()
            if not current:
                QMessageBox.warning(
                    self,
                    "Mass plan required",
                    "Save an execution-ready mass plan before approval.",
                )
                return
            report = validate_mass_analysis_plan(
                self.project, dict(current.get("configuration") or {})
            )
            if not report.get("execution_ready"):
                QMessageBox.warning(
                    self,
                    "Mass task is not ready",
                    "Attach a matching calibration result, pass CPU-float64 parity, "
                    "and approve every upstream quality gate before approval.",
                )
                return
        super()._set_status(status)
