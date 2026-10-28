"""Qt Wasserstein-analysis workflow page."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
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

from darsia.fluidflower.analysis_domains import (
    ANALYSIS_DOMAIN_REVISION_FORMAT,
    AnalysisDomainStore,
)
from darsia.fluidflower.mass_analysis import MassAnalysisPlanStore
from darsia.fluidflower.pages.mass_analysis.page import MassAnalysisPage
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.quality_budget import format_bytes
from darsia.fluidflower.ui.analysis_domains import AnalysisDomainDialog
from darsia.fluidflower.ui.common import ModulePage, PathField, deepcopy_settings
from darsia.fluidflower.wasserstein import (
    WASSERSTEIN_MASS_COMPONENTS,
    WASSERSTEIN_METRICS,
    WassersteinPlanStore,
    default_wasserstein_plan,
    validate_wasserstein_plan,
)


class WassersteinPage(ModulePage):
    """Prepare an auditable Wasserstein task without executing it."""

    _COMPONENT_LABELS: ClassVar[dict[str, str]] = {
        "mass_total": "Total CO2 mass",
        "mass_aqueous": "Aqueous CO2 mass",
        "mass_free_gas": "Free-gas CO2 mass",
    }
    _DOMAIN_LABELS: ClassVar[dict[str, str]] = {
        "whole_active_domain": "Whole active rig domain",
    }
    _METRIC_LABELS: ClassVar[dict[str, str]] = {
        "transport_cost_mass_m": "Raw transport cost (mass x metre)",
        "mass_based_distance_m": "Mass-normalized distance (metre)",
        "rig_based_normalized_distance": "FF_UM-compatible rig normalization",
        "rig_diagonal_normalized_distance": "Rig-diagonal normalization",
        "plume_based_normalized_distance": "Reference-plume normalization",
        "absolute_mass_difference_g": "Absolute detected-mass difference",
        "relative_mass_difference": "Relative detected-mass difference",
    }

    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__(project, "wasserstein")
        self.store = WassersteinPlanStore(project)
        self._updating_wasserstein_selection = False
        self._selected_pairs: set[tuple[str, str]] = set()
        self._pair_run_ids: set[str] = set()
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "This page freezes pair, time, domain, physical-resolution, and "
            "normalization semantics. It never loads mass fields or starts the "
            "Wasserstein solver from the Qt process."
        )

        self.wasserstein_upstream_reference = QLabel()
        self.wasserstein_upstream_reference.setObjectName("integrationNotice")
        self.wasserstein_upstream_reference.setWordWrap(True)
        self.work_layout.addWidget(self.wasserstein_upstream_reference)

        result_panel = QFrame()
        result_panel.setFrameShape(QFrame.Shape.StyledPanel)
        result_layout = QFormLayout(result_panel)
        self.wasserstein_mass_result = PathField(
            directory=False,
            file_filter="Mass-analysis result (*.json);;All files (*)",
            placeholder="Optional while drafting; required before execution",
        )
        result_layout.addRow("Mass-result manifest", self.wasserstein_mass_result)
        result_note = QLabel(
            "The manifest must identify the exact mass task, every run, one shared "
            "target-time grid, physical rig dimensions, native cell size, named "
            "whole-domain fields, and hash-verified total/aqueous/free-gas product "
            "indexes. Custom boxes are cropped from those fields at Wasserstein time."
        )
        result_note.setObjectName("secondaryText")
        result_note.setWordWrap(True)
        result_layout.addRow("Readiness requirement", result_note)
        self.work_layout.addWidget(result_panel)

        run_heading = QLabel("Runs and explicit comparisons")
        run_heading.setObjectName("sectionTitle")
        self.work_layout.addWidget(run_heading)
        comparison_splitter = QSplitter(Qt.Orientation.Horizontal)
        run_panel = QWidget()
        run_layout = QVBoxLayout(run_panel)
        run_layout.setContentsMargins(0, 0, 0, 0)
        self.wasserstein_runs = QTableWidget(0, 1)
        self.wasserstein_runs.setHorizontalHeaderLabels(["Include run"])
        self.wasserstein_runs.verticalHeader().setVisible(False)
        self.wasserstein_runs.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.wasserstein_runs.itemChanged.connect(self._run_selection_changed)
        run_layout.addWidget(self.wasserstein_runs)
        comparison_splitter.addWidget(run_panel)

        pair_panel = QWidget()
        pair_layout = QVBoxLayout(pair_panel)
        pair_layout.setContentsMargins(0, 0, 0, 0)
        self.wasserstein_pairs = QTableWidget(0, 3)
        self.wasserstein_pairs.setHorizontalHeaderLabels(["Use", "Run A", "Run B"])
        self.wasserstein_pairs.verticalHeader().setVisible(False)
        self.wasserstein_pairs.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.wasserstein_pairs.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.wasserstein_pairs.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.wasserstein_pairs.itemChanged.connect(self._pair_selection_changed)
        pair_layout.addWidget(self.wasserstein_pairs)
        pair_actions = QHBoxLayout()
        select_pairs = QPushButton("Select all pairs")
        select_pairs.clicked.connect(lambda: self._set_all_pairs(True))
        pair_actions.addWidget(select_pairs)
        clear_pairs = QPushButton("Clear pairs")
        clear_pairs.clicked.connect(lambda: self._set_all_pairs(False))
        pair_actions.addWidget(clear_pairs)
        pair_actions.addStretch(1)
        pair_layout.addLayout(pair_actions)
        comparison_splitter.addWidget(pair_panel)
        comparison_splitter.setStretchFactor(0, 1)
        comparison_splitter.setStretchFactor(1, 2)
        comparison_splitter.setMaximumHeight(260)
        self.work_layout.addWidget(comparison_splitter)

        selection = QSplitter(Qt.Orientation.Horizontal)
        components_panel = QFrame()
        components_panel.setFrameShape(QFrame.Shape.StyledPanel)
        components_layout = QVBoxLayout(components_panel)
        components_title = QLabel("Mass components")
        components_title.setObjectName("sectionTitle")
        components_layout.addWidget(components_title)
        self.wasserstein_components: dict[str, QCheckBox] = {}
        for key in WASSERSTEIN_MASS_COMPONENTS:
            checkbox = QCheckBox(self._COMPONENT_LABELS[key])
            self.wasserstein_components[key] = checkbox
            components_layout.addWidget(checkbox)
        components_layout.addStretch(1)
        selection.addWidget(components_panel)

        domains_panel = QFrame()
        domains_panel.setFrameShape(QFrame.Shape.StyledPanel)
        domains_layout = QVBoxLayout(domains_panel)
        domains_title = QLabel("Analysis domains")
        domains_title.setObjectName("sectionTitle")
        domains_layout.addWidget(domains_title)
        self.wasserstein_domain_revision = QLabel()
        self.wasserstein_domain_revision.setObjectName("secondaryText")
        self.wasserstein_domain_revision.setWordWrap(True)
        domains_layout.addWidget(self.wasserstein_domain_revision)
        self.wasserstein_domains: dict[str, QCheckBox] = {}
        self.wasserstein_domain_choices = QVBoxLayout()
        domains_layout.addLayout(self.wasserstein_domain_choices)
        edit_domains = QPushButton("Edit or import domains")
        edit_domains.clicked.connect(self._edit_analysis_domains)
        domains_layout.addWidget(edit_domains)
        domains_layout.addStretch(1)
        selection.addWidget(domains_panel)

        resolution_panel = QFrame()
        resolution_panel.setFrameShape(QFrame.Shape.StyledPanel)
        resolution_layout = QFormLayout(resolution_panel)
        self.wasserstein_physical_resolution = QCheckBox("Specify common physical grid")
        self.wasserstein_physical_resolution.toggled.connect(
            self._wasserstein_resolution_toggled
        )
        self.wasserstein_cell_size = QDoubleSpinBox()
        self.wasserstein_cell_size.setRange(0.001, 100.0)
        self.wasserstein_cell_size.setDecimals(3)
        self.wasserstein_cell_size.setSuffix(" mm")
        resolution_note = QLabel(
            "The selected value applies in metres to every run. It replaces the "
            "old dimensionless 0.1 resize factor and may not be finer than the "
            "coarsest source mass grid. Resampling preserves integrated mass."
        )
        resolution_note.setObjectName("secondaryText")
        resolution_note.setWordWrap(True)
        resolution_layout.addRow("Resolution", self.wasserstein_physical_resolution)
        resolution_layout.addRow("Cell size", self.wasserstein_cell_size)
        resolution_layout.addRow("Cross-rig policy", resolution_note)
        selection.addWidget(resolution_panel)
        selection.setStretchFactor(0, 1)
        selection.setStretchFactor(1, 1)
        selection.setStretchFactor(2, 2)
        self.work_layout.addWidget(selection)

        budget_panel = QFrame()
        budget_panel.setFrameShape(QFrame.Shape.StyledPanel)
        budget_layout = QGridLayout(budget_panel)
        self.wasserstein_limit_start = QCheckBox("Start at")
        self.wasserstein_start_hours = QDoubleSpinBox()
        self.wasserstein_start_hours.setRange(-100_000.0, 100_000.0)
        self.wasserstein_start_hours.setDecimals(3)
        self.wasserstein_start_hours.setSuffix(" h")
        self.wasserstein_limit_start.toggled.connect(
            self.wasserstein_start_hours.setEnabled
        )
        self.wasserstein_limit_end = QCheckBox("End at")
        self.wasserstein_end_hours = QDoubleSpinBox()
        self.wasserstein_end_hours.setRange(-100_000.0, 100_000.0)
        self.wasserstein_end_hours.setDecimals(3)
        self.wasserstein_end_hours.setSuffix(" h")
        self.wasserstein_limit_end.toggled.connect(
            self.wasserstein_end_hours.setEnabled
        )
        self.wasserstein_landmarks = QLineEdit()
        self.wasserstein_landmarks.setPlaceholderText(
            "Injection start/end and other required times"
        )
        self.wasserstein_max_times = QSpinBox()
        self.wasserstein_max_times.setRange(0, 100_000)
        self.wasserstein_max_times.setSpecialValueText("No limit")
        self.wasserstein_max_tasks = QSpinBox()
        self.wasserstein_max_tasks.setRange(0, 2_000_000_000)
        self.wasserstein_max_tasks.setSpecialValueText("No limit")
        self.wasserstein_publication_gib = QDoubleSpinBox()
        self.wasserstein_publication_gib.setRange(0.0, 1_000_000.0)
        self.wasserstein_publication_gib.setDecimals(2)
        self.wasserstein_publication_gib.setSuffix(" GiB")
        self.wasserstein_publication_gib.setSpecialValueText("No limit")
        budget_note = QLabel(
            "The smallest active limit determines a deterministic, evenly "
            "distributed subset. Window endpoints and listed landmark times are "
            "retained. The exact source indices are frozen in the task contract."
        )
        budget_note.setObjectName("secondaryText")
        budget_note.setWordWrap(True)
        budget_layout.addWidget(self.wasserstein_limit_start, 0, 0)
        budget_layout.addWidget(self.wasserstein_start_hours, 0, 1)
        budget_layout.addWidget(self.wasserstein_limit_end, 0, 2)
        budget_layout.addWidget(self.wasserstein_end_hours, 0, 3)
        budget_layout.addWidget(QLabel("Required landmark times"), 1, 0)
        budget_layout.addWidget(self.wasserstein_landmarks, 1, 1, 1, 3)
        budget_layout.addWidget(QLabel("Maximum shared times"), 2, 0)
        budget_layout.addWidget(self.wasserstein_max_times, 2, 1)
        budget_layout.addWidget(QLabel("Maximum solver tasks"), 2, 2)
        budget_layout.addWidget(self.wasserstein_max_tasks, 2, 3)
        budget_layout.addWidget(QLabel("Publication budget"), 3, 0)
        budget_layout.addWidget(self.wasserstein_publication_gib, 3, 1)
        budget_layout.addWidget(budget_note, 3, 2, 1, 2)
        self.work_layout.addWidget(budget_panel)

        metrics_panel = QFrame()
        metrics_panel.setFrameShape(QFrame.Shape.StyledPanel)
        metrics_layout = QGridLayout(metrics_panel)
        metrics_title = QLabel("Required scientific outputs")
        metrics_title.setObjectName("sectionTitle")
        metrics_layout.addWidget(metrics_title, 0, 0, 1, 3)
        self.wasserstein_metrics: dict[str, QCheckBox] = {}
        for index, key in enumerate(WASSERSTEIN_METRICS):
            checkbox = QCheckBox(self._METRIC_LABELS[key])
            checkbox.setChecked(True)
            checkbox.setEnabled(False)
            self.wasserstein_metrics[key] = checkbox
            metrics_layout.addWidget(checkbox, 1 + index // 3, index % 3)
        metric_note = QLabel(
            "The transport solver balances each pair to its mean detected mass "
            "only for transport. Original detected masses and their difference "
            "remain separate required outputs."
        )
        metric_note.setObjectName("secondaryText")
        metric_note.setWordWrap(True)
        metrics_layout.addWidget(metric_note, 4, 0, 1, 3)
        self.work_layout.addWidget(metrics_panel)

        execution_panel = QFrame()
        execution_panel.setFrameShape(QFrame.Shape.StyledPanel)
        execution_layout = QGridLayout(execution_panel)
        self.wasserstein_backend = QLabel("cpu-float64")
        self.wasserstein_workers = QSpinBox()
        self.wasserstein_workers.setRange(1, 256)
        self.wasserstein_cache_timepoints = QSpinBox()
        self.wasserstein_cache_timepoints.setRange(1, 64)
        self.wasserstein_local_cache = QCheckBox("Use local read-through cache")
        self.wasserstein_transport_diagnostics = QCheckBox(
            "Store full transport diagnostics"
        )
        interruption = QLabel(
            "Tasks are ordered by time, checkpointed individually, resumed by "
            "verified identity, retried after transient NAS interruptions, and "
            "published atomically. GPU execution is not enabled for this solver."
        )
        interruption.setObjectName("secondaryText")
        interruption.setWordWrap(True)
        execution_layout.addWidget(QLabel("Backend"), 0, 0)
        execution_layout.addWidget(self.wasserstein_backend, 0, 1)
        execution_layout.addWidget(QLabel("Workers"), 0, 2)
        execution_layout.addWidget(self.wasserstein_workers, 0, 3)
        execution_layout.addWidget(QLabel("Cached times / worker"), 1, 0)
        execution_layout.addWidget(self.wasserstein_cache_timepoints, 1, 1)
        execution_layout.addWidget(self.wasserstein_local_cache, 1, 2)
        execution_layout.addWidget(self.wasserstein_transport_diagnostics, 1, 3)
        execution_layout.addWidget(interruption, 2, 0, 1, 4)
        self.work_layout.addWidget(execution_panel)

        review_panel = QFrame()
        review_panel.setFrameShape(QFrame.Shape.StyledPanel)
        review_layout = QFormLayout(review_panel)
        self.wasserstein_reviewer = QLineEdit(
            str(project.manifest.get("created_by") or "")
        )
        self.wasserstein_reason = QLineEdit()
        self.wasserstein_reason.setPlaceholderText(
            "Why are these domains, resolution, and metrics appropriate?"
        )
        review_layout.addRow("Reviewer", self.wasserstein_reviewer)
        review_layout.addRow("Plan note", self.wasserstein_reason)
        self.work_layout.addWidget(review_panel)

        actions = QHBoxLayout()
        self.validate_wasserstein_plan_button = QPushButton("Validate plan")
        self.validate_wasserstein_plan_button.clicked.connect(
            self._validate_wasserstein_plan
        )
        actions.addWidget(self.validate_wasserstein_plan_button)
        self.save_wasserstein_plan_button = QPushButton("Save plan")
        self.save_wasserstein_plan_button.clicked.connect(self._save_wasserstein_plan)
        actions.addWidget(self.save_wasserstein_plan_button)
        actions.addStretch(1)
        actions.addWidget(QLabel("Plan history"))
        self.wasserstein_revisions = QComboBox()
        self.wasserstein_revisions.setMinimumWidth(300)
        actions.addWidget(self.wasserstein_revisions)
        self.work_layout.addLayout(actions)
        self.wasserstein_summary = QLabel("No Wasserstein plan validated.")
        self.wasserstein_summary.setObjectName("secondaryText")
        self.wasserstein_summary.setWordWrap(True)
        self.work_layout.addWidget(self.wasserstein_summary)

        self._load_wasserstein_plan()
        self.refresh()

    def _wasserstein_resolution_toggled(self, enabled: bool) -> None:
        self.wasserstein_cell_size.setEnabled(enabled)

    def _analysis_domain_rows(self) -> list[dict[str, Any]]:
        from ..post_analysis.page import PostAnalysisPage
        return PostAnalysisPage._post_domain_rows(self)

    def _reload_domain_choices(self, *, selected_ids: set[str] | None = None) -> None:
        while self.wasserstein_domain_choices.count():
            item = self.wasserstein_domain_choices.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        revision = AnalysisDomainStore(self.project).current()
        rows = self._analysis_domain_rows()
        available_ids = {"whole_active_domain"} | {
            str(row.get("domain_id") or "") for row in rows
        }
        if selected_ids is None:
            selected_ids = set(available_ids)
        else:
            selected_ids &= available_ids
        self.wasserstein_domains = {}
        whole = QCheckBox(self._DOMAIN_LABELS["whole_active_domain"])
        whole.setChecked("whole_active_domain" in selected_ids)
        self.wasserstein_domains["whole_active_domain"] = whole
        self.wasserstein_domain_choices.addWidget(whole)
        for row in rows:
            domain_id = str(row.get("domain_id") or "")
            checkbox = QCheckBox(str(row.get("name") or domain_id))
            checkbox.setToolTip(
                "Physical bounds: "
                + json.dumps(row.get("physical_bounds_m") or {}, ensure_ascii=False)
            )
            checkbox.setChecked(domain_id in selected_ids)
            self.wasserstein_domains[domain_id] = checkbox
            self.wasserstein_domain_choices.addWidget(checkbox)
        if revision.get("format") == ANALYSIS_DOMAIN_REVISION_FORMAT:
            self.wasserstein_domain_revision.setText(
                f"Custom-domain revision {str(revision.get('revision_id') or '')[:12]} · "
                f"{len(rows)} enabled box(es) on the corrected rig grid"
            )
        else:
            self.wasserstein_domain_revision.setText(
                "No custom boxes saved. The whole corrected rig domain remains available."
            )

    def _edit_analysis_domains(self) -> None:
        previous = {
            key
            for key, checkbox in self.wasserstein_domains.items()
            if checkbox.isChecked()
        }
        dialog = AnalysisDomainDialog(self.project, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        enabled = {
            str(row.get("domain_id") or "") for row in self._analysis_domain_rows()
        }
        self._reload_domain_choices(selected_ids=previous | enabled)

    def _available_wasserstein_runs(self) -> list[str]:
        return list(default_wasserstein_plan(self.project).get("run_ids") or [])

    def _selected_run_ids(self) -> list[str]:
        selected: list[str] = []
        for row in range(self.wasserstein_runs.rowCount()):
            item = self.wasserstein_runs.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return sorted(selected)

    def _populate_run_selection(
        self, available_runs: list[str], selected_runs: set[str]
    ) -> None:
        self._updating_wasserstein_selection = True
        try:
            self.wasserstein_runs.setRowCount(len(available_runs))
            for row, run_id in enumerate(available_runs):
                item = QTableWidgetItem(run_id)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(
                    Qt.CheckState.Checked
                    if run_id in selected_runs
                    else Qt.CheckState.Unchecked
                )
                self.wasserstein_runs.setItem(row, 0, item)
        finally:
            self._updating_wasserstein_selection = False

    @staticmethod
    def _eligible_pairs(run_ids: list[str]) -> set[tuple[str, str]]:
        ordered = sorted(set(run_ids))
        return {
            (run_1, run_2)
            for index, run_1 in enumerate(ordered)
            for run_2 in ordered[index + 1 :]
        }

    def _populate_pair_selection(
        self, selected_pairs: set[tuple[str, str]] | None = None
    ) -> None:
        run_ids = self._selected_run_ids()
        eligible = self._eligible_pairs(run_ids)
        selected = eligible if selected_pairs is None else selected_pairs & eligible
        self._updating_wasserstein_selection = True
        try:
            ordered = sorted(eligible)
            self.wasserstein_pairs.setRowCount(len(ordered))
            for row, pair in enumerate(ordered):
                use = QTableWidgetItem()
                use.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                use.setCheckState(
                    Qt.CheckState.Checked
                    if pair in selected
                    else Qt.CheckState.Unchecked
                )
                self.wasserstein_pairs.setItem(row, 0, use)
                for column, run_id in enumerate(pair, start=1):
                    item = QTableWidgetItem(run_id)
                    item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                    self.wasserstein_pairs.setItem(row, column, item)
        finally:
            self._updating_wasserstein_selection = False
        self._selected_pairs = set(selected)
        self._pair_run_ids = set(run_ids)

    def _run_selection_changed(self, _item: QTableWidgetItem) -> None:
        if self._updating_wasserstein_selection:
            return
        run_ids = self._selected_run_ids()
        eligible = self._eligible_pairs(run_ids)
        added_runs = set(run_ids) - self._pair_run_ids
        selected = self._selected_pairs & eligible
        selected.update(pair for pair in eligible if set(pair) & added_runs)
        if not self._pair_run_ids:
            selected = set(eligible)
        self._populate_pair_selection(selected)

    def _pair_selection_changed(self, _item: QTableWidgetItem) -> None:
        if self._updating_wasserstein_selection:
            return
        selected: set[tuple[str, str]] = set()
        for row in range(self.wasserstein_pairs.rowCount()):
            use = self.wasserstein_pairs.item(row, 0)
            run_1 = self.wasserstein_pairs.item(row, 1)
            run_2 = self.wasserstein_pairs.item(row, 2)
            if (
                use is not None
                and run_1 is not None
                and run_2 is not None
                and use.checkState() == Qt.CheckState.Checked
            ):
                selected.add(tuple(sorted((run_1.text(), run_2.text()))))
        self._selected_pairs = selected

    def _set_all_pairs(self, checked: bool) -> None:
        self._updating_wasserstein_selection = True
        try:
            for row in range(self.wasserstein_pairs.rowCount()):
                item = self.wasserstein_pairs.item(row, 0)
                if item is not None:
                    item.setCheckState(
                        Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                    )
        finally:
            self._updating_wasserstein_selection = False
        self._selected_pairs = (
            self._eligible_pairs(self._selected_run_ids()) if checked else set()
        )

    def _selected_pair_rows(self) -> list[dict[str, str]]:
        return [
            {"run_1": run_1, "run_2": run_2}
            for run_1, run_2 in sorted(self._selected_pairs)
        ]

    def _load_wasserstein_plan(self) -> None:
        current = self.store.current()
        config = dict(current.get("configuration") or {})
        if not config:
            config = default_wasserstein_plan(self.project)
        self.wasserstein_mass_result.setText(
            str(config.get("mass_result_manifest_path") or "")
        )
        comparison = dict(config.get("comparison") or {})
        available_runs = self._available_wasserstein_runs()
        selected_runs = {
            str(value) for value in config.get("run_ids") or available_runs
        }
        self._populate_run_selection(available_runs, selected_runs)
        selected_pairs = {
            tuple(sorted((str(row.get("run_1") or ""), str(row.get("run_2") or ""))))
            for row in comparison.get("pairs") or []
            if isinstance(row, Mapping)
        }
        self._populate_pair_selection(selected_pairs or None)
        selected_components = {
            str(value) for value in comparison.get("mass_components") or []
        }
        for key, checkbox in self.wasserstein_components.items():
            checkbox.setChecked(key in selected_components)
        budget = dict(comparison.get("time_selection_budget") or {})
        start = budget.get("start_time_hours")
        end = budget.get("end_time_hours")
        self.wasserstein_limit_start.setChecked(start is not None)
        self.wasserstein_start_hours.setValue(0.0 if start is None else float(start))
        self.wasserstein_start_hours.setEnabled(start is not None)
        self.wasserstein_limit_end.setChecked(end is not None)
        self.wasserstein_end_hours.setValue(0.0 if end is None else float(end))
        self.wasserstein_end_hours.setEnabled(end is not None)
        self.wasserstein_landmarks.setText(
            ", ".join(
                f"{float(value):g}"
                for value in budget.get("required_landmark_times_hours") or []
            )
        )
        self.wasserstein_max_times.setValue(int(budget.get("maximum_time_points") or 0))
        self.wasserstein_max_tasks.setValue(
            int(budget.get("maximum_solver_tasks") or 0)
        )
        maximum_bytes = budget.get("maximum_publication_bytes")
        self.wasserstein_publication_gib.setValue(
            0.0 if maximum_bytes is None else float(maximum_bytes) / float(1024**3)
        )
        selected_domains = {str(value) for value in comparison.get("domains") or []}
        self._reload_domain_choices(selected_ids=selected_domains)
        spatial = dict(config.get("spatial_discretization") or {})
        cell_size = spatial.get("target_cell_size_m")
        self.wasserstein_physical_resolution.setChecked(cell_size is not None)
        self.wasserstein_cell_size.setValue(
            1.0 if cell_size is None else 1000.0 * float(cell_size)
        )
        self._wasserstein_resolution_toggled(cell_size is not None)
        outputs = dict(config.get("outputs") or {})
        self.wasserstein_transport_diagnostics.setChecked(
            bool(outputs.get("store_transport_diagnostics"))
        )
        execution = dict(config.get("execution_profile") or {})
        self.wasserstein_workers.setValue(int(execution.get("workers") or 12))
        self.wasserstein_cache_timepoints.setValue(
            int(execution.get("cache_timepoints_per_worker") or 2)
        )
        self.wasserstein_local_cache.setChecked(
            bool(execution.get("local_read_through_cache", True))
        )

    def _wasserstein_configuration(self) -> dict[str, Any]:
        config = default_wasserstein_plan(self.project)
        mass_plan = MassAnalysisPlanStore(self.project).current()
        config["mass_analysis_plan_id"] = str(mass_plan.get("plan_id") or "")
        config["mass_analysis_plan_sha256"] = str(mass_plan.get("content_sha256") or "")
        config["mass_task_contract_sha256"] = str(
            (mass_plan.get("task_contract") or {}).get("content_sha256") or ""
        )
        config["mass_result_manifest_path"] = self.wasserstein_mass_result.text()
        config["run_ids"] = self._selected_run_ids()
        mass_config = dict(mass_plan.get("configuration") or {})
        tolerance = float(
            (mass_config.get("image_selection") or {}).get(
                "drift_tolerance_seconds", 0.0
            )
        )
        config["comparison"] = {
            "pairing_policy": "explicit_unordered_run_pairs",
            "pairs": self._selected_pair_rows(),
            "time_policy": "shared_synchronized_target_times",
            "require_same_target_times": True,
            "recorded_mass_time_tolerance_seconds": tolerance,
            "mass_components": [
                key
                for key in WASSERSTEIN_MASS_COMPONENTS
                if self.wasserstein_components[key].isChecked()
            ],
            "domains": [
                key
                for key, checkbox in self.wasserstein_domains.items()
                if checkbox.isChecked()
            ],
            "time_selection_budget": {
                "policy": "uniform_with_required_landmarks",
                "start_time_hours": (
                    self.wasserstein_start_hours.value()
                    if self.wasserstein_limit_start.isChecked()
                    else None
                ),
                "end_time_hours": (
                    self.wasserstein_end_hours.value()
                    if self.wasserstein_limit_end.isChecked()
                    else None
                ),
                "required_landmark_times_hours": MassAnalysisPage._parse_landmark_times(
                    self.wasserstein_landmarks.text()
                ),
                "preserve_window_endpoints": True,
                "maximum_time_points": (
                    self.wasserstein_max_times.value()
                    if self.wasserstein_max_times.value() > 0
                    else None
                ),
                "maximum_solver_tasks": (
                    self.wasserstein_max_tasks.value()
                    if self.wasserstein_max_tasks.value() > 0
                    else None
                ),
                "maximum_publication_bytes": (
                    round(self.wasserstein_publication_gib.value() * 1024**3)
                    if self.wasserstein_publication_gib.value() > 0.0
                    else None
                ),
            },
        }
        config["spatial_discretization"]["target_cell_size_m"] = (
            self.wasserstein_cell_size.value() / 1000.0
            if self.wasserstein_physical_resolution.isChecked()
            else None
        )
        config["outputs"]["store_transport_diagnostics"] = (
            self.wasserstein_transport_diagnostics.isChecked()
        )
        config["execution_profile"]["workers"] = self.wasserstein_workers.value()
        config["execution_profile"]["cache_timepoints_per_worker"] = (
            self.wasserstein_cache_timepoints.value()
        )
        config["execution_profile"]["local_read_through_cache"] = (
            self.wasserstein_local_cache.isChecked()
        )
        config["execution_requested"] = False
        return config

    def _validate_wasserstein_plan(
        self, *, show_dialog: bool = True
    ) -> dict[str, Any] | None:
        try:
            report = validate_wasserstein_plan(
                self.project, self._wasserstein_configuration()
            )
        except Exception as exc:  # noqa: BLE001
            self.wasserstein_summary.setText("Wasserstein validation failed.")
            if show_dialog:
                QMessageBox.critical(self, "Wasserstein validation failed", str(exc))
            return None
        summary = dict(report.get("summary") or {})
        readiness = "EXECUTION READY" if report.get("execution_ready") else "DRAFT"
        self.wasserstein_summary.setText(
            f"{readiness} · {summary.get('error_count', 0)} errors · "
            f"{summary.get('warning_count', 0)} warnings · "
            f"{summary.get('pair_count', 0)} run pairs · "
            f"{summary.get('time_count', 0)} of "
            f"{summary.get('source_time_count', 0)} shared times · "
            f"{summary.get('task_count', 0)} solver tasks · "
            f"about {format_bytes(summary.get('estimated_publication_bytes', 0))} · "
            f"{summary.get('backend')}"
        )
        if show_dialog:
            messages = [str(issue.get("message") or "") for issue in report["issues"]]
            QMessageBox.information(
                self,
                "Wasserstein validation",
                self.wasserstein_summary.text()
                + ("\n\n" + "\n".join(messages) if messages else ""),
            )
        return report

    def _save_wasserstein_plan(self) -> None:
        config = self._wasserstein_configuration()
        report = self._validate_wasserstein_plan(show_dialog=False)
        if report is None:
            return
        try:
            revision = self.store.save_plan(
                config,
                actor=self.wasserstein_reviewer.text().strip(),
                reason=self.wasserstein_reason.text().strip(),
                validation=report,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save Wasserstein plan", str(exc))
            return
        self.wasserstein_reason.clear()
        self.refresh()
        state = "execution ready" if revision["execution"]["ready"] else "a draft"
        QMessageBox.information(
            self,
            "Wasserstein plan saved",
            f"Plan {str(revision['plan_id'])[:8]} was stored as {state}. "
            "No mass fields were loaded and no solver was started.",
        )

    def refresh(self) -> None:
        self._refresh_gate()
        mass_plan = MassAnalysisPlanStore(self.project).current()
        if mass_plan:
            self.wasserstein_upstream_reference.setText(
                f"Mass plan {str(mass_plan.get('plan_id') or '')[:12]} · task "
                f"{str((mass_plan.get('task_contract') or {}).get('content_sha256') or '')[:12]} · "
                "raw total, aqueous, and free-gas mass remain authoritative"
            )
        else:
            self.wasserstein_upstream_reference.setText(
                "No mass-analysis plan is available. A Wasserstein plan cannot "
                "yet be valid."
            )
        available_runs = self._available_wasserstein_runs()
        displayed_runs = [
            self.wasserstein_runs.item(row, 0).text()
            for row in range(self.wasserstein_runs.rowCount())
            if self.wasserstein_runs.item(row, 0) is not None
        ]
        if displayed_runs != available_runs:
            previous = set(self._selected_run_ids())
            new_runs = set(available_runs) - set(displayed_runs)
            self._populate_run_selection(available_runs, previous | new_runs)
            self._populate_pair_selection()
        self.wasserstein_revisions.clear()
        current_id = str(self.store.current().get("plan_id") or "")
        rows = list(reversed(self.store.revisions()))
        if not rows:
            self.wasserstein_revisions.addItem("No saved plans", "")
        for revision in rows:
            plan_id = str(revision.get("plan_id") or "")
            ready = bool((revision.get("execution") or {}).get("ready"))
            self.wasserstein_revisions.addItem(
                f"{str(revision.get('recorded_utc') or '')[:19]} · "
                f"{'ready' if ready else 'draft'} · {plan_id[:8]}",
                plan_id,
            )
            if plan_id == current_id:
                self.wasserstein_revisions.setCurrentIndex(
                    self.wasserstein_revisions.count() - 1
                )

    def _set_status(self, status: str) -> None:
        if status in {"approved", "complete"}:
            current = self.store.current()
            if not current:
                QMessageBox.warning(
                    self,
                    "Wasserstein plan required",
                    "Save an execution-ready Wasserstein plan before approval.",
                )
                return
            report = validate_wasserstein_plan(
                self.project, dict(current.get("configuration") or {})
            )
            if not report.get("execution_ready"):
                QMessageBox.warning(
                    self,
                    "Wasserstein task is not ready",
                    "Attach a complete matching mass result, choose a physical "
                    "cell size, and approve mass analysis before approval.",
                )
                return
        super()._set_status(status)
