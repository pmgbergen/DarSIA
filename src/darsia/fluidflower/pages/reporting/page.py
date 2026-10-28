"""Optional Qt report-composition page."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, ClassVar, Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
)

from darsia.fluidflower.modules import MODULE_DEFINITIONS
from darsia.fluidflower.page_registry import (
    FluidFlowerSubpageRegistry,
    default_subpage_registry,
)
from darsia.fluidflower.post_analysis import PostAnalysisPlanStore
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.reporting import (
    EVIDENCE_KINDS,
    REPORT_EXPORT_FORMATS,
    REPORT_SECTIONS,
    REQUIRED_REPORT_EXPORTS,
    REQUIRED_REPORT_SECTIONS,
    UNCERTAINTY_STATUSES,
    ReportingPlanStore,
    default_reporting_plan,
    validate_reporting_plan,
)
from darsia.fluidflower.ui.common import (
    FILE_DIALOG_OPTIONS,
    STATUS_LABELS,
    ModulePage,
)


class ReportingPage(ModulePage):
    """Freeze quality review and publication-package semantics."""

    _SECTION_LABELS: ClassVar[dict[str, str]] = {
        "executive_summary": "Executive summary",
        "experimental_setup": "Experimental setup",
        "data_integrity": "Data integrity",
        "geometry_facies": "Geometry and facies",
        "image_correction": "Image correction",
        "temporal_alignment": "Temporal alignment",
        "phase_interpretation": "Phase interpretation",
        "calibration": "Calibration",
        "mass_analysis": "Mass analysis",
        "wasserstein_analysis": "Wasserstein analysis",
        "post_analysis": "Post-analysis products",
        "uncertainty_limitations": "Uncertainty and limitations",
        "reproducibility_provenance": "Reproducibility and provenance",
        "appendices": "Appendices",
    }
    _EXPORT_LABELS: ClassVar[dict[str, str]] = {
        "html": "Interactive HTML",
        "pdf": "PDF",
        "docx": "Word document",
        "pptx": "PowerPoint summary",
        "machine_readable_json": "Machine-readable JSON",
        "provenance_bundle_zip": "Provenance bundle",
    }
    _EVIDENCE_LABELS: ClassVar[dict[str, str]] = {
        "protocol_or_source": "Protocol or source",
        "manual_review": "Manual review",
        "calibration_history": "Calibration history",
        "uncertainty_evidence": "Uncertainty evidence",
        "wasserstein_result": "Wasserstein result",
        "supplementary_figure": "Supplementary figure",
        "supplementary_movie": "Supplementary movie",
        "supplementary_table": "Supplementary table",
        "other": "Other",
    }
    _UNCERTAINTY_STATUS_LABELS: ClassVar[dict[str, str]] = {
        "unreviewed": "Unreviewed",
        "reviewed": "Reviewed",
        "accepted_with_limitation": "Accepted with limitation",
        "not_applicable": "Not applicable",
    }

    def __init__(
        self,
        project: FluidFlowerProject,
        *,
        subpage_registry: FluidFlowerSubpageRegistry | None = None,
    ) -> None:
        super().__init__(project, "reporting")
        self.subpage_registry = subpage_registry or default_subpage_registry()
        self.store = ReportingPlanStore(project)
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "This optional page defines a compact, reproducible export package from "
            "verified post-analysis artifacts. It records provenance and uncertainty "
            "without watermarks or a separate signature workflow."
        )

        self.reporting_upstream_reference = QLabel()
        self.reporting_upstream_reference.setObjectName("integrationNotice")
        self.reporting_upstream_reference.setWordWrap(True)
        self.work_layout.addWidget(self.reporting_upstream_reference)

        self.reporting_tabs = QTabWidget()
        self.work_layout.addWidget(self.reporting_tabs)
        self.subpage_registry.populate_tabs("reporting", self, self.reporting_tabs)

        review_panel = QFrame()
        review_panel.setFrameShape(QFrame.Shape.StyledPanel)
        review_layout = QFormLayout(review_panel)
        self.reporting_reviewer = QLineEdit(
            str(project.manifest.get("created_by") or "")
        )
        self.reporting_reason = QLineEdit()
        self.reporting_reason.setPlaceholderText(
            "Why is this report scope and review state appropriate?"
        )
        review_layout.addRow("Plan author", self.reporting_reviewer)
        review_layout.addRow("Plan note", self.reporting_reason)
        self.work_layout.addWidget(review_panel)

        actions = QHBoxLayout()
        self.validate_reporting_plan_button = QPushButton("Validate plan")
        self.validate_reporting_plan_button.clicked.connect(
            self._validate_reporting_plan
        )
        actions.addWidget(self.validate_reporting_plan_button)
        self.save_reporting_plan_button = QPushButton("Save plan")
        self.save_reporting_plan_button.clicked.connect(self._save_reporting_plan)
        actions.addWidget(self.save_reporting_plan_button)
        actions.addStretch(1)
        actions.addWidget(QLabel("Plan history"))
        self.reporting_revisions = QComboBox()
        self.reporting_revisions.setMinimumWidth(300)
        actions.addWidget(self.reporting_revisions)
        self.work_layout.addLayout(actions)
        self.reporting_summary = QLabel("No reporting plan validated.")
        self.reporting_summary.setObjectName("secondaryText")
        self.reporting_summary.setWordWrap(True)
        self.work_layout.addWidget(self.reporting_summary)

        self._load_reporting_plan()
        self.refresh()



    def _populate_evidence(self, rows: list[dict[str, Any]]) -> None:
        self.reporting_evidence.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            kind = QComboBox()
            for key in EVIDENCE_KINDS:
                kind.addItem(self._EVIDENCE_LABELS[key], key)
            index = kind.findData(row.get("kind") or "other")
            kind.setCurrentIndex(max(0, index))
            self.reporting_evidence.setCellWidget(row_index, 0, kind)
            self.reporting_evidence.setItem(
                row_index, 1, QTableWidgetItem(str(row.get("title") or ""))
            )
            path_item = QTableWidgetItem(str(row.get("path") or ""))
            path_item.setData(
                Qt.ItemDataRole.UserRole,
                {
                    "evidence_id": str(row.get("evidence_id") or uuid.uuid4().hex),
                    "sha256": str(row.get("sha256") or ""),
                },
            )
            path_item.setToolTip(str(row.get("sha256") or "Not hashed yet"))
            self.reporting_evidence.setItem(row_index, 2, path_item)

    def _add_reporting_evidence(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Add report evidence",
            "",
            "Evidence files (*.*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if not selected:
            return
        rows = self._evidence_rows()
        rows.extend(
            {
                "evidence_id": uuid.uuid4().hex,
                "kind": "other",
                "title": Path(path).stem.replace("_", " "),
                "path": str(Path(path).resolve()),
                "sha256": "",
            }
            for path in selected
        )
        self._populate_evidence(rows)

    def _remove_reporting_evidence(self) -> None:
        rows = sorted(
            {
                index.row()
                for index in self.reporting_evidence.selectionModel().selectedRows()
            },
            reverse=True,
        )
        for row in rows:
            self.reporting_evidence.removeRow(row)

    def _evidence_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.reporting_evidence.rowCount()):
            kind = self.reporting_evidence.cellWidget(row, 0)
            title = self.reporting_evidence.item(row, 1)
            path = self.reporting_evidence.item(row, 2)
            metadata = path.data(Qt.ItemDataRole.UserRole) if path is not None else {}
            if not isinstance(metadata, Mapping):
                metadata = {}
            rows.append(
                {
                    "evidence_id": str(metadata.get("evidence_id") or uuid.uuid4().hex),
                    "kind": str(
                        kind.currentData() if isinstance(kind, QComboBox) else ""
                    ),
                    "title": title.text().strip() if title is not None else "",
                    "path": path.text().strip() if path is not None else "",
                    "sha256": str(metadata.get("sha256") or ""),
                }
            )
        return rows

    def _populate_uncertainties(self, rows: list[dict[str, Any]]) -> None:
        self.reporting_uncertainties.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            category = QTableWidgetItem(str(row.get("category") or ""))
            category.setData(
                Qt.ItemDataRole.UserRole, str(row.get("uncertainty_id") or "")
            )
            category.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.reporting_uncertainties.setItem(row_index, 0, category)
            status = QComboBox()
            for key in UNCERTAINTY_STATUSES:
                status.addItem(self._UNCERTAINTY_STATUS_LABELS[key], key)
            index = status.findData(row.get("review_status") or "unreviewed")
            status.setCurrentIndex(max(0, index))
            self.reporting_uncertainties.setCellWidget(row_index, 1, status)
            for column, key in enumerate(
                (
                    "statement",
                    "effect_on_results",
                    "mitigation_or_evidence",
                    "reviewed_by",
                ),
                start=2,
            ):
                self.reporting_uncertainties.setItem(
                    row_index, column, QTableWidgetItem(str(row.get(key) or ""))
                )

    def _uncertainty_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.reporting_uncertainties.rowCount()):
            category = self.reporting_uncertainties.item(row, 0)
            status = self.reporting_uncertainties.cellWidget(row, 1)
            rows.append(
                {
                    "uncertainty_id": str(
                        category.data(Qt.ItemDataRole.UserRole)
                        if category is not None
                        else ""
                    ),
                    "category": category.text() if category is not None else "",
                    "review_status": str(
                        status.currentData() if isinstance(status, QComboBox) else ""
                    ),
                    "statement": self._reporting_table_text(
                        self.reporting_uncertainties, row, 2
                    ),
                    "effect_on_results": self._reporting_table_text(
                        self.reporting_uncertainties, row, 3
                    ),
                    "mitigation_or_evidence": self._reporting_table_text(
                        self.reporting_uncertainties, row, 4
                    ),
                    "reviewed_by": self._reporting_table_text(
                        self.reporting_uncertainties, row, 5
                    ),
                }
            )
        return rows

    @staticmethod
    def _reporting_table_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _load_reporting_plan(self) -> None:
        current = self.store.current()
        config = dict(current.get("configuration") or {})
        if not config:
            config = default_reporting_plan(self.project)
        self.reporting_post_result.setText(
            str(config.get("post_analysis_result_manifest_path") or "")
        )
        identity = dict(config.get("identity") or {})
        self.reporting_title.setText(str(identity.get("title") or ""))
        self.reporting_subtitle.setText(str(identity.get("subtitle") or ""))
        self.reporting_version.setText(str(identity.get("report_version") or ""))
        language = self.reporting_language.findData(identity.get("language"))
        self.reporting_language.setCurrentIndex(max(0, language))
        audience = self.reporting_audience.findData(identity.get("audience"))
        self.reporting_audience.setCurrentIndex(max(0, audience))
        self.reporting_authors.setText(
            "; ".join(str(value) for value in identity.get("authors") or [])
        )
        self.reporting_organization.setText(str(identity.get("organization") or ""))
        selected_sections = {str(value) for value in config.get("sections") or []}
        for key, checkbox in self.reporting_sections.items():
            checkbox.setChecked(
                key in selected_sections or key in REQUIRED_REPORT_SECTIONS
            )
        exports = dict(config.get("exports") or {})
        selected_exports = {str(value) for value in exports.get("formats") or []}
        for key, checkbox in self.reporting_exports.items():
            checkbox.setChecked(
                key in selected_exports or key in REQUIRED_REPORT_EXPORTS
            )
        self.reporting_embed_limit.setValue(
            int(exports.get("maximum_embedded_media_bytes") or 0) // (1024 * 1024)
        )
        self._populate_evidence(list(config.get("evidence_files") or []))
        self._populate_uncertainties(list(config.get("uncertainty_register") or []))
        execution = dict(config.get("execution_profile") or {})
        self.reporting_output_root.setText(str(config.get("output_root") or ""))
        self.reporting_local_cache.setChecked(
            bool(execution.get("local_read_through_cache", True))
        )
        self.reporting_retries.setValue(int(execution.get("maximum_io_retries") or 8))

    def _reporting_configuration(self) -> dict[str, Any]:
        config = default_reporting_plan(self.project)
        post_plan = PostAnalysisPlanStore(self.project).current()
        config["post_analysis_plan_id"] = str(post_plan.get("plan_id") or "")
        config["post_analysis_plan_sha256"] = str(post_plan.get("content_sha256") or "")
        config["post_analysis_task_contract_sha256"] = str(
            (post_plan.get("task_contract") or {}).get("content_sha256") or ""
        )
        config["post_analysis_result_manifest_path"] = self.reporting_post_result.text()
        config["identity"] = {
            "title": self.reporting_title.text().strip(),
            "subtitle": self.reporting_subtitle.text().strip(),
            "report_version": self.reporting_version.text().strip(),
            "language": str(self.reporting_language.currentData()),
            "audience": str(self.reporting_audience.currentData()),
            "authors": [
                value.strip()
                for value in self.reporting_authors.text().split(";")
                if value.strip()
            ],
            "organization": self.reporting_organization.text().strip(),
        }
        config["sections"] = [
            key for key in REPORT_SECTIONS if self.reporting_sections[key].isChecked()
        ]
        config["exports"] = {
            "formats": [
                key
                for key in REPORT_EXPORT_FORMATS
                if self.reporting_exports[key].isChecked()
            ],
            "large_media_policy": "link_with_hash_and_thumbnail",
            "small_artifact_policy": "copy_verified",
            "maximum_embedded_media_bytes": self.reporting_embed_limit.value()
            * 1024
            * 1024,
        }
        config["evidence_files"] = self._evidence_rows()
        config["uncertainty_register"] = self._uncertainty_rows()
        config["execution_profile"]["local_read_through_cache"] = (
            self.reporting_local_cache.isChecked()
        )
        config["execution_profile"]["maximum_io_retries"] = (
            self.reporting_retries.value()
        )
        config["output_root"] = self.reporting_output_root.text()
        config["execution_requested"] = False
        return config

    def _validate_reporting_plan(
        self, *, show_dialog: bool = True
    ) -> dict[str, Any] | None:
        try:
            report = validate_reporting_plan(
                self.project, self._reporting_configuration()
            )
        except Exception as exc:  # noqa: BLE001
            self.reporting_summary.setText("Reporting-plan validation failed.")
            if show_dialog:
                QMessageBox.critical(self, "Reporting-plan validation failed", str(exc))
            return None
        summary = dict(report.get("summary") or {})
        if report.get("publication_ready"):
            readiness = "PUBLICATION READY"
        elif report.get("generation_ready"):
            readiness = "DRAFT GENERATION READY"
        else:
            readiness = "PLAN DRAFT"
        self.reporting_summary.setText(
            f"{readiness} · {summary.get('error_count', 0)} errors · "
            f"{summary.get('warning_count', 0)} warnings · "
            f"{summary.get('post_artifact_count', 0)} post-analysis artifacts · "
            f"{summary.get('unreviewed_uncertainty_count', 0)} unreviewed uncertainties"
        )
        if show_dialog:
            messages = [str(issue.get("message") or "") for issue in report["issues"]]
            QMessageBox.information(
                self,
                "Reporting-plan validation",
                self.reporting_summary.text()
                + ("\n\n" + "\n".join(messages) if messages else ""),
            )
        return report

    def _save_reporting_plan(self) -> None:
        config = self._reporting_configuration()
        report = self._validate_reporting_plan(show_dialog=False)
        if report is None:
            return
        try:
            revision = self.store.save_plan(
                config,
                actor=self.reporting_reviewer.text().strip(),
                reason=self.reporting_reason.text().strip(),
                validation=report,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save reporting plan", str(exc))
            return
        self.reporting_reason.clear()
        self._load_reporting_plan()
        self.refresh()
        if revision["publication"]["ready"]:
            state = "publication ready"
        elif revision["generation"]["ready"]:
            state = "ready for draft generation"
        else:
            state = "a plan draft"
        QMessageBox.information(
            self,
            "Reporting plan saved",
            f"Plan {str(revision['plan_id'])[:8]} was stored as {state}. "
            "No report or presentation was rendered.",
        )

    def _populate_reporting_quality(self) -> None:
        definitions = {definition.key: definition for definition in MODULE_DEFINITIONS}
        keys = [
            definition.key
            for definition in MODULE_DEFINITIONS
            if definition.key != "reporting"
        ]
        self.reporting_quality.setRowCount(len(keys))
        for row, key in enumerate(keys):
            state = self.project.module_states[key]
            values = (
                definitions[key].title,
                STATUS_LABELS.get(
                    str(state.get("status") or ""), str(state.get("status") or "")
                ),
                str(state.get("updated_by") or ""),
                str(state.get("reason") or ""),
            )
            for column, value in enumerate(values):
                self.reporting_quality.setItem(row, column, QTableWidgetItem(value))

    def refresh(self) -> None:
        self._refresh_gate()
        post_plan = PostAnalysisPlanStore(self.project).current()
        if post_plan:
            self.reporting_upstream_reference.setText(
                f"Post-analysis plan {str(post_plan.get('plan_id') or '')[:12]} · task "
                f"{str((post_plan.get('task_contract') or {}).get('content_sha256') or '')[:12]} · "
                "the report package can summarize but never alter scientific outputs"
            )
        else:
            self.reporting_upstream_reference.setText(
                "No post-analysis plan is available. A reporting plan cannot yet be valid."
            )
        self._populate_reporting_quality()
        self.reporting_revisions.clear()
        current_id = str(self.store.current().get("plan_id") or "")
        rows = list(reversed(self.store.revisions()))
        if not rows:
            self.reporting_revisions.addItem("No saved plans", "")
        for revision in rows:
            plan_id = str(revision.get("plan_id") or "")
            generation = bool((revision.get("generation") or {}).get("ready"))
            publication = bool((revision.get("publication") or {}).get("ready"))
            state = (
                "publication"
                if publication
                else "generation"
                if generation
                else "draft"
            )
            self.reporting_revisions.addItem(
                f"{str(revision.get('recorded_utc') or '')[:19]} · {state} · {plan_id[:8]}",
                plan_id,
            )
            if plan_id == current_id:
                self.reporting_revisions.setCurrentIndex(
                    self.reporting_revisions.count() - 1
                )

    def _set_status(self, status: str) -> None:
        if status in {"approved", "complete"}:
            current = self.store.current()
            if not current:
                QMessageBox.warning(
                    self,
                    "Reporting plan required",
                    "Save a publication-ready reporting plan before approval.",
                )
                return
            report = validate_reporting_plan(
                self.project, dict(current.get("configuration") or {})
            )
            if not report.get("publication_ready"):
                QMessageBox.warning(
                    self,
                    "Report package is not publication ready",
                    "Complete artifact verification, uncertainty review, and current "
                    "quality gates first.",
                )
                return
            if report["task_contract"]["content_sha256"] != str(
                (current.get("task_contract") or {}).get("content_sha256") or ""
            ):
                QMessageBox.warning(
                    self,
                    "Reporting plan changed",
                    "The quality snapshot or evidence changed after this plan was "
                    "saved. Save a new immutable reporting plan first.",
                )
                return
        super()._set_status(status)
