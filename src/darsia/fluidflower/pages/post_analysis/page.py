"""Qt post-analysis workflow page."""

from __future__ import annotations

import json
from typing import Any, ClassVar

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
)

from darsia.fluidflower.analysis_domains import (
    ANALYSIS_DOMAIN_REVISION_FORMAT,
    AnalysisDomainStore,
)
from darsia.fluidflower.mass_analysis import MassAnalysisPlanStore
from darsia.fluidflower.post_analysis import (
    POST_ANALYSIS_PRODUCTS,
    REQUIRED_POST_ANALYSIS_PRODUCTS,
    PostAnalysisPlanStore,
    default_post_analysis_plan,
    validate_post_analysis_plan,
)
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.ui.analysis_domains import AnalysisDomainDialog
from darsia.fluidflower.ui.common import ModulePage, PathField, deepcopy_settings


class PostAnalysisPage(ModulePage):
    """Prepare reproducible movies, figures, and diagnostics from mass results."""

    _PRODUCT_LABELS: ClassVar[dict[str, str]] = {
        "phase_map_movie_corrected_background": "Phase-map movie · corrected",
        "phase_map_movie_black_background": "Phase-map movie · black",
        "phase_contour_movie": "Contour movie · corrected",
        "phase_contour_contact_sheet": "Contour contact sheet",
        "mass_curve_full_domain": "Mass curve · whole rig",
        "mass_curve_analysis_domains": "Mass curves · analysis boxes",
        "cross_run_mass_grid": "Mass-curve grid · runs",
        "cross_run_phase_grid_movie": "Phase-map movie grid · runs",
        "integrated_mass_csv": "Integrated phase-mass CSV",
        "per_frame_diagnostics_csv": "Per-frame diagnostics CSV",
        "provenance_json": "Provenance JSON",
    }

    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__(project, "post_analysis")
        self.store = PostAnalysisPlanStore(project)
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "This page freezes the runs, analysis regions, rendering semantics, "
            "mass-curve policy, and interruption-safe execution profile. Saving a "
            "plan never renders a frame or starts a video encoder in the Qt process."
        )

        self.post_upstream_reference = QLabel()
        self.post_upstream_reference.setObjectName("integrationNotice")
        self.post_upstream_reference.setWordWrap(True)
        self.work_layout.addWidget(self.post_upstream_reference)

        result_panel = QFrame()
        result_panel.setFrameShape(QFrame.Shape.StyledPanel)
        result_layout = QFormLayout(result_panel)
        self.post_mass_result = PathField(
            directory=False,
            file_filter="Mass-analysis result (*.json);;All files (*)",
            placeholder="Optional while drafting; required before rendering",
        )
        result_layout.addRow("Mass-result manifest", self.post_mass_result)
        result_note = QLabel(
            "The manifest must match the current mass task and provide hash-verified "
            "mass fields. Phase products additionally require aqueous concentration, "
            "free-gas saturation, and corrected registered frame references."
        )
        result_note.setObjectName("secondaryText")
        result_note.setWordWrap(True)
        result_layout.addRow("Readiness requirement", result_note)
        self.work_layout.addWidget(result_panel)

        selection_splitter = QSplitter(Qt.Orientation.Horizontal)
        runs_panel = QFrame()
        runs_panel.setFrameShape(QFrame.Shape.StyledPanel)
        runs_layout = QVBoxLayout(runs_panel)
        runs_title = QLabel("Runs")
        runs_title.setObjectName("sectionTitle")
        runs_layout.addWidget(runs_title)
        self.post_runs = QTableWidget(0, 1)
        self.post_runs.setHorizontalHeaderLabels(["Include run"])
        self.post_runs.verticalHeader().setVisible(False)
        self.post_runs.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.post_runs.setMaximumHeight(250)
        runs_layout.addWidget(self.post_runs)
        selection_splitter.addWidget(runs_panel)

        products_panel = QFrame()
        products_panel.setFrameShape(QFrame.Shape.StyledPanel)
        products_layout = QVBoxLayout(products_panel)
        products_title = QLabel("Post-analysis products")
        products_title.setObjectName("sectionTitle")
        products_layout.addWidget(products_title)
        products_grid = QGridLayout()
        self.post_products: dict[str, QCheckBox] = {}
        for index, key in enumerate(POST_ANALYSIS_PRODUCTS):
            checkbox = QCheckBox(self._PRODUCT_LABELS[key])
            if key in REQUIRED_POST_ANALYSIS_PRODUCTS:
                checkbox.setChecked(True)
                checkbox.setEnabled(False)
                checkbox.setToolTip(
                    "Required for traceability and post-analysis quality control."
                )
            self.post_products[key] = checkbox
            products_grid.addWidget(checkbox, index // 2, index % 2)
        products_layout.addLayout(products_grid)
        product_note = QLabel(
            "A corrected-background phase movie and a black-background phase movie "
            "are separate products. Cross-run grids require at least two runs."
        )
        product_note.setObjectName("secondaryText")
        product_note.setWordWrap(True)
        products_layout.addWidget(product_note)
        products_layout.addStretch(1)
        selection_splitter.addWidget(products_panel)
        selection_splitter.setStretchFactor(0, 1)
        selection_splitter.setStretchFactor(1, 2)
        self.work_layout.addWidget(selection_splitter)

        domains_panel = QFrame()
        domains_panel.setFrameShape(QFrame.Shape.StyledPanel)
        domains_layout = QVBoxLayout(domains_panel)
        domain_title = QLabel("Analysis domains")
        domain_title.setObjectName("sectionTitle")
        domains_layout.addWidget(domain_title)
        self.post_domain_revision = QLabel()
        self.post_domain_revision.setObjectName("secondaryText")
        self.post_domain_revision.setWordWrap(True)
        domains_layout.addWidget(self.post_domain_revision)
        domain_controls = QHBoxLayout()
        self.post_domain_choices = QHBoxLayout()
        self.post_domains: dict[str, QCheckBox] = {}
        domain_controls.addLayout(self.post_domain_choices, 1)
        edit_domains = QPushButton("Edit or import domains")
        edit_domains.clicked.connect(self._edit_post_domains)
        domain_controls.addWidget(edit_domains)
        domains_layout.addLayout(domain_controls)
        self.work_layout.addWidget(domains_panel)

        rendering_splitter = QSplitter(Qt.Orientation.Horizontal)
        rendering_panel = QFrame()
        rendering_panel.setFrameShape(QFrame.Shape.StyledPanel)
        rendering_layout = QFormLayout(rendering_panel)
        self.post_movie_width = QSpinBox()
        self.post_movie_width.setRange(640, 7680)
        self.post_movie_width.setSingleStep(160)
        self.post_movie_width.setSuffix(" px")
        self.post_movie_fps = QDoubleSpinBox()
        self.post_movie_fps.setRange(0.25, 60.0)
        self.post_movie_fps.setDecimals(2)
        self.post_movie_fps.setSuffix(" fps")
        self.post_overlay_alpha = QDoubleSpinBox()
        self.post_overlay_alpha.setRange(0.0, 1.0)
        self.post_overlay_alpha.setDecimals(2)
        self.post_overlay_alpha.setSingleStep(0.05)
        self.post_aqueous_levels = QLineEdit()
        self.post_aqueous_levels.setPlaceholderText("0.05, 0.30, 0.50")
        self.post_gas_levels = QLineEdit()
        self.post_gas_levels.setPlaceholderText("0.05")
        self.post_show_injectors = QCheckBox("Injection points")
        self.post_show_elapsed_time = QCheckBox("Frame + time")
        self.post_pre_injection = QCheckBox("Baseline first")
        self.post_pre_injection.setChecked(True)
        self.post_pre_injection.setEnabled(False)
        rendering_layout.addRow("Movie width", self.post_movie_width)
        rendering_layout.addRow("Playback rate", self.post_movie_fps)
        rendering_layout.addRow("Phase opacity", self.post_overlay_alpha)
        rendering_layout.addRow("Aqueous levels", self.post_aqueous_levels)
        rendering_layout.addRow("Gas levels", self.post_gas_levels)
        rendering_layout.addRow("Rig overlay", self.post_show_injectors)
        rendering_layout.addRow("Frame overlay", self.post_show_elapsed_time)
        rendering_layout.addRow("Baseline control", self.post_pre_injection)
        rendering_splitter.addWidget(rendering_panel)

        semantics_panel = QFrame()
        semantics_panel.setFrameShape(QFrame.Shape.StyledPanel)
        semantics_layout = QFormLayout(semantics_panel)
        phase_map = QLabel(
            "Filled, mutually exclusive water / aqueous CO2 / free-gas phase map"
        )
        background = QLabel(
            "Approved corrected, cropped, registered, and unwrapped experiment frame"
        )
        frame_order = QLabel(
            "Every mass-analysis frame in chronological order, plus baseline control"
        )
        mass_policy = QLabel(
            "Raw detected total, aqueous, and free-gas masses remain authoritative. "
            "Synchronized injected mass is shown only as a comparison curve."
        )
        for label in (phase_map, background, frame_order, mass_policy):
            label.setObjectName("secondaryText")
            label.setWordWrap(True)
        semantics_layout.addRow("Phase map", phase_map)
        semantics_layout.addRow("Background", background)
        semantics_layout.addRow("Frames", frame_order)
        semantics_layout.addRow("Mass", mass_policy)
        rendering_splitter.addWidget(semantics_panel)
        rendering_splitter.setStretchFactor(0, 1)
        rendering_splitter.setStretchFactor(1, 1)
        self.work_layout.addWidget(rendering_splitter)

        execution_panel = QFrame()
        execution_panel.setFrameShape(QFrame.Shape.StyledPanel)
        execution_layout = QGridLayout(execution_panel)
        self.post_workers = QSpinBox()
        self.post_workers.setRange(1, 256)
        self.post_encoder = QComboBox()
        self.post_encoder.addItem(
            "Auto hardware",
            "auto-hardware-if-available",
        )
        self.post_encoder.addItem("CPU software", "cpu-software")
        self.post_local_cache = QCheckBox("Local cache")
        self.post_checkpoint = QCheckBox("Per-run save")
        self.post_checkpoint.setChecked(True)
        self.post_checkpoint.setEnabled(False)
        self.post_resume = QCheckBox("Resume runs")
        self.post_resume.setChecked(True)
        self.post_resume.setEnabled(False)
        self.post_io_retries = QSpinBox()
        self.post_io_retries.setRange(1, 64)
        self.post_output_root = PathField(
            directory=True,
            placeholder="Versioned post-analysis output directory",
        )
        execution_layout.addWidget(QLabel("Frame workers"), 0, 0)
        execution_layout.addWidget(self.post_workers, 0, 1)
        execution_layout.addWidget(QLabel("Video encoding"), 0, 2)
        execution_layout.addWidget(self.post_encoder, 0, 3)
        execution_layout.addWidget(self.post_local_cache, 1, 0, 1, 2)
        execution_layout.addWidget(self.post_checkpoint, 1, 2)
        execution_layout.addWidget(self.post_resume, 1, 3)
        execution_layout.addWidget(QLabel("I/O attempts"), 2, 0)
        execution_layout.addWidget(self.post_io_retries, 2, 1)
        execution_layout.addWidget(QLabel("Output root"), 3, 0)
        execution_layout.addWidget(self.post_output_root, 3, 1, 1, 3)
        execution_note = QLabel(
            "Frame preparation can use multiple CPU workers while the selected "
            "video encoder consumes completed frames. Local cache, per-run "
            "checkpoints, retry, and atomic publication protect project outputs."
        )
        execution_note.setObjectName("secondaryText")
        execution_note.setWordWrap(True)
        execution_layout.addWidget(execution_note, 4, 0, 1, 4)
        self.work_layout.addWidget(execution_panel)

        review_panel = QFrame()
        review_panel.setFrameShape(QFrame.Shape.StyledPanel)
        review_layout = QFormLayout(review_panel)
        self.post_reviewer = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.post_reason = QLineEdit()
        self.post_reason.setPlaceholderText(
            "Why are these products, regions, and rendering settings appropriate?"
        )
        review_layout.addRow("Reviewer", self.post_reviewer)
        review_layout.addRow("Plan note", self.post_reason)
        self.work_layout.addWidget(review_panel)

        actions = QHBoxLayout()
        self.validate_post_plan_button = QPushButton("Validate plan")
        self.validate_post_plan_button.clicked.connect(self._validate_post_plan)
        actions.addWidget(self.validate_post_plan_button)
        self.save_post_plan_button = QPushButton("Save plan")
        self.save_post_plan_button.clicked.connect(self._save_post_plan)
        actions.addWidget(self.save_post_plan_button)
        actions.addStretch(1)
        actions.addWidget(QLabel("Plan history"))
        self.post_revisions = QComboBox()
        self.post_revisions.setMinimumWidth(300)
        actions.addWidget(self.post_revisions)
        self.work_layout.addLayout(actions)
        self.post_summary = QLabel("No post-analysis plan validated.")
        self.post_summary.setObjectName("secondaryText")
        self.post_summary.setWordWrap(True)
        self.work_layout.addWidget(self.post_summary)

        self._load_post_plan()
        self.refresh()

    def _available_post_runs(self) -> list[str]:
        return list(default_post_analysis_plan(self.project).get("run_ids") or [])

    def _selected_post_run_ids(self) -> list[str]:
        selected: list[str] = []
        for row in range(self.post_runs.rowCount()):
            item = self.post_runs.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return sorted(selected)

    def _populate_post_runs(
        self, available_runs: list[str], selected_runs: set[str]
    ) -> None:
        self.post_runs.setRowCount(len(available_runs))
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
            self.post_runs.setItem(row, 0, item)

    def _post_domain_rows(self) -> list[dict[str, Any]]:
        revision = AnalysisDomainStore(self.project).current()
        if revision.get("format") != ANALYSIS_DOMAIN_REVISION_FORMAT:
            return []
        return [
            deepcopy_settings(row)
            for row in (revision.get("configuration") or {}).get("domains") or []
            if bool(row.get("enabled", True))
        ]

    def _reload_post_domains(self, *, selected_ids: set[str] | None = None) -> None:
        while self.post_domain_choices.count():
            item = self.post_domain_choices.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        revision = AnalysisDomainStore(self.project).current()
        rows = self._post_domain_rows()
        available = {"whole_active_domain"} | {
            str(row.get("domain_id") or "") for row in rows
        }
        if selected_ids is None:
            selected_ids = set(available)
        else:
            selected_ids &= available
        selected_ids.add("whole_active_domain")
        self.post_domains = {}
        whole = QCheckBox("Whole active rig domain")
        whole.setChecked(True)
        whole.setEnabled(False)
        whole.setToolTip("Required as the full-domain quality-control reference.")
        self.post_domains["whole_active_domain"] = whole
        self.post_domain_choices.addWidget(whole)
        for row in rows:
            domain_id = str(row.get("domain_id") or "")
            checkbox = QCheckBox(str(row.get("name") or domain_id))
            checkbox.setToolTip(
                "Physical bounds: "
                + json.dumps(row.get("physical_bounds_m") or {}, ensure_ascii=False)
            )
            checkbox.setChecked(domain_id in selected_ids)
            self.post_domains[domain_id] = checkbox
            self.post_domain_choices.addWidget(checkbox)
        self.post_domain_choices.addStretch(1)
        if revision.get("format") == ANALYSIS_DOMAIN_REVISION_FORMAT:
            self.post_domain_revision.setText(
                f"Domain revision {str(revision.get('revision_id') or '')[:12]} · "
                f"{len(rows)} enabled custom box(es) on the corrected rig grid"
            )
        else:
            self.post_domain_revision.setText(
                "No custom boxes are saved; whole-rig post-analysis remains available."
            )

    def _edit_post_domains(self) -> None:
        previous = {
            key for key, checkbox in self.post_domains.items() if checkbox.isChecked()
        }
        dialog = AnalysisDomainDialog(self.project, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        enabled = {str(row.get("domain_id") or "") for row in self._post_domain_rows()}
        self._reload_post_domains(selected_ids=previous | enabled)

    @staticmethod
    def _post_levels(text: str) -> list[float]:
        values: list[float] = []
        for token in text.replace(";", ",").split(","):
            stripped = token.strip()
            if stripped:
                values.append(float(stripped))
        return values

    def _load_post_plan(self) -> None:
        current = self.store.current()
        config = dict(current.get("configuration") or {})
        if not config:
            config = default_post_analysis_plan(self.project)
        self.post_mass_result.setText(
            str(config.get("mass_result_manifest_path") or "")
        )
        available_runs = self._available_post_runs()
        selected_runs = {str(value) for value in config.get("run_ids") or []}
        self._populate_post_runs(available_runs, selected_runs or set(available_runs))
        selected_products = {str(value) for value in config.get("products") or []}
        for key, checkbox in self.post_products.items():
            checkbox.setChecked(
                key in selected_products or key in REQUIRED_POST_ANALYSIS_PRODUCTS
            )
        selected_domains = {str(value) for value in config.get("domains") or []}
        self._reload_post_domains(selected_ids=selected_domains)
        rendering = dict(config.get("rendering") or {})
        self.post_movie_width.setValue(int(rendering.get("movie_width_pixels") or 1600))
        self.post_movie_fps.setValue(float(rendering.get("movie_fps") or 6.0))
        self.post_overlay_alpha.setValue(
            float(rendering.get("phase_overlay_alpha") or 0.58)
        )
        self.post_aqueous_levels.setText(
            ", ".join(
                f"{float(value):g}"
                for value in rendering.get("aqueous_levels_c_over_csat")
                or [0.05, 0.30, 0.50]
            )
        )
        self.post_gas_levels.setText(
            ", ".join(
                f"{float(value):g}"
                for value in rendering.get("free_gas_levels_saturation") or [0.05]
            )
        )
        self.post_show_injectors.setChecked(bool(rendering.get("show_injectors", True)))
        self.post_show_elapsed_time.setChecked(
            bool(rendering.get("show_elapsed_time", True))
        )
        execution = dict(config.get("execution_profile") or {})
        self.post_workers.setValue(int(execution.get("frame_workers") or 24))
        encoder_index = self.post_encoder.findData(
            execution.get("video_encoder") or "auto-hardware-if-available"
        )
        self.post_encoder.setCurrentIndex(max(0, encoder_index))
        self.post_local_cache.setChecked(
            bool(execution.get("local_read_through_cache", True))
        )
        self.post_io_retries.setValue(int(execution.get("maximum_io_retries") or 8))
        self.post_output_root.setText(str(config.get("output_root") or ""))

    def _post_configuration(self) -> dict[str, Any]:
        config = default_post_analysis_plan(self.project)
        mass_plan = MassAnalysisPlanStore(self.project).current()
        domains = AnalysisDomainStore(self.project).current()
        config["mass_analysis_plan_id"] = str(mass_plan.get("plan_id") or "")
        config["mass_analysis_plan_sha256"] = str(mass_plan.get("content_sha256") or "")
        config["mass_task_contract_sha256"] = str(
            (mass_plan.get("task_contract") or {}).get("content_sha256") or ""
        )
        config["mass_result_manifest_path"] = self.post_mass_result.text()
        config["analysis_domain_revision_id"] = str(domains.get("revision_id") or "")
        config["analysis_domain_revision_sha256"] = str(
            domains.get("content_sha256") or ""
        )
        config["run_ids"] = self._selected_post_run_ids()
        config["products"] = [
            key for key in POST_ANALYSIS_PRODUCTS if self.post_products[key].isChecked()
        ]
        config["domains"] = [
            key for key, checkbox in self.post_domains.items() if checkbox.isChecked()
        ]
        rendering = config["rendering"]
        rendering["movie_width_pixels"] = self.post_movie_width.value()
        rendering["movie_fps"] = self.post_movie_fps.value()
        rendering["phase_overlay_alpha"] = self.post_overlay_alpha.value()
        rendering["aqueous_levels_c_over_csat"] = self._post_levels(
            self.post_aqueous_levels.text()
        )
        rendering["free_gas_levels_saturation"] = self._post_levels(
            self.post_gas_levels.text()
        )
        rendering["show_injectors"] = self.post_show_injectors.isChecked()
        rendering["show_elapsed_time"] = self.post_show_elapsed_time.isChecked()
        execution = config["execution_profile"]
        execution["frame_workers"] = self.post_workers.value()
        execution["video_encoder"] = str(self.post_encoder.currentData())
        execution["local_read_through_cache"] = self.post_local_cache.isChecked()
        execution["maximum_io_retries"] = self.post_io_retries.value()
        config["output_root"] = self.post_output_root.text()
        config["execution_requested"] = False
        return config

    def _validate_post_plan(self, *, show_dialog: bool = True) -> dict[str, Any] | None:
        try:
            report = validate_post_analysis_plan(
                self.project, self._post_configuration()
            )
        except Exception as exc:  # noqa: BLE001
            self.post_summary.setText("Post-analysis validation failed.")
            if show_dialog:
                QMessageBox.critical(self, "Post-analysis validation failed", str(exc))
            return None
        summary = dict(report.get("summary") or {})
        readiness = "EXECUTION READY" if report.get("execution_ready") else "DRAFT"
        self.post_summary.setText(
            f"{readiness} · {summary.get('error_count', 0)} errors · "
            f"{summary.get('warning_count', 0)} warnings · "
            f"{summary.get('run_count', 0)} runs · "
            f"{summary.get('product_count', 0)} products · "
            f"{summary.get('domain_count', 0)} domains"
        )
        if show_dialog:
            messages = [str(issue.get("message") or "") for issue in report["issues"]]
            QMessageBox.information(
                self,
                "Post-analysis validation",
                self.post_summary.text()
                + ("\n\n" + "\n".join(messages) if messages else ""),
            )
        return report

    def _save_post_plan(self) -> None:
        config = self._post_configuration()
        report = self._validate_post_plan(show_dialog=False)
        if report is None:
            return
        try:
            revision = self.store.save_plan(
                config,
                actor=self.post_reviewer.text().strip(),
                reason=self.post_reason.text().strip(),
                validation=report,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save post-analysis plan", str(exc))
            return
        self.post_reason.clear()
        self.refresh()
        state = "execution ready" if revision["execution"]["ready"] else "a draft"
        QMessageBox.information(
            self,
            "Post-analysis plan saved",
            f"Plan {str(revision['plan_id'])[:8]} was stored as {state}. "
            "No frames were rendered and no encoder was started.",
        )

    def refresh(self) -> None:
        self._refresh_gate()
        mass_plan = MassAnalysisPlanStore(self.project).current()
        if mass_plan:
            self.post_upstream_reference.setText(
                f"Mass plan {str(mass_plan.get('plan_id') or '')[:12]} · task "
                f"{str((mass_plan.get('task_contract') or {}).get('content_sha256') or '')[:12]} · "
                "post-analysis cannot alter the detected phase masses"
            )
        else:
            self.post_upstream_reference.setText(
                "No mass-analysis plan is available. A post-analysis plan cannot "
                "yet be valid."
            )
        available_runs = self._available_post_runs()
        displayed_runs = [
            self.post_runs.item(row, 0).text()
            for row in range(self.post_runs.rowCount())
            if self.post_runs.item(row, 0) is not None
        ]
        if displayed_runs != available_runs:
            previous = set(self._selected_post_run_ids())
            new_runs = set(available_runs) - set(displayed_runs)
            self._populate_post_runs(available_runs, previous | new_runs)
        selected_domains = {
            key for key, checkbox in self.post_domains.items() if checkbox.isChecked()
        }
        self._reload_post_domains(selected_ids=selected_domains)
        self.post_revisions.clear()
        current_id = str(self.store.current().get("plan_id") or "")
        rows = list(reversed(self.store.revisions()))
        if not rows:
            self.post_revisions.addItem("No saved plans", "")
        for revision in rows:
            plan_id = str(revision.get("plan_id") or "")
            ready = bool((revision.get("execution") or {}).get("ready"))
            products = len((revision.get("configuration") or {}).get("products") or [])
            self.post_revisions.addItem(
                f"{str(revision.get('recorded_utc') or '')[:19]} · "
                f"{'ready' if ready else 'draft'} · {products} products · "
                f"{plan_id[:8]}",
                plan_id,
            )
            if plan_id == current_id:
                self.post_revisions.setCurrentIndex(self.post_revisions.count() - 1)

    def _set_status(self, status: str) -> None:
        if status in {"approved", "complete"}:
            current = self.store.current()
            if not current:
                QMessageBox.warning(
                    self,
                    "Post-analysis plan required",
                    "Save an execution-ready post-analysis plan before approval.",
                )
                return
            report = validate_post_analysis_plan(
                self.project, dict(current.get("configuration") or {})
            )
            if not report.get("execution_ready"):
                QMessageBox.warning(
                    self,
                    "Post-analysis task is not ready",
                    "Attach a complete matching mass result and approve mass "
                    "analysis before approving post-analysis.",
                )
                return
        super()._set_status(status)
