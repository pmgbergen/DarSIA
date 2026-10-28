"""Qt phase-interpretation workflow page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.page_registry import (
    FluidFlowerSubpageRegistry,
    default_subpage_registry,
)
from darsia.fluidflower.phase_interpretation import (
    PHASE_INTERPRETATION_CONFIG_FORMAT,
    PhaseInterpretationRevisionStore,
    default_phase_interpretation_config,
    inspect_phase_campaign,
    parse_review_file_list,
    validate_phase_interpretation_config,
)
from darsia.fluidflower.phase_learning import (
    PhaseLearningPlanStore,
    inspect_phase_model_bundle,
    phase_learning_plan_from_bundle,
    validate_phase_learning_plan,
)
from darsia.fluidflower.phase_workflow import (
    build_phase_review_command,
)
from darsia.fluidflower.project import FluidFlowerProject, utc_now
from darsia.fluidflower.ui.common import ModulePage, PathField


class PhaseInterpretationPage(ModulePage):
    """Register, review, and validate shared optical phase models."""

    def __init__(
        self,
        project: FluidFlowerProject,
        *,
        subpage_registry: FluidFlowerSubpageRegistry | None = None,
    ) -> None:
        super().__init__(project, "phase_interpretation")
        self.subpage_registry = subpage_registry or default_subpage_registry()
        self.store = PhaseInterpretationRevisionStore(project)
        self.learning_store = PhaseLearningPlanStore(project)
        self.campaign: dict[str, Any] | None = None
        self.phase_model_bundle_summary: dict[str, Any] | None = None
        self.selected_candidates: dict[str, str] = {}
        self.run_reviews: dict[str, dict[str, Any]] = {}
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "This page registers a portable optical phase model and existing 2D "
            "phase products against exact source evidence. The shared model may use "
            "reviewed annotations and DarSIA reference-change features. Depth, "
            "3D conversion, concentration, and mass remain experiment-specific."
        )

        model_group = QFrame()
        model_group.setFrameShape(QFrame.Shape.StyledPanel)
        model_layout = QGridLayout(model_group)
        model_layout.addWidget(QLabel("Shared phase-model bundle"), 0, 0)
        self.phase_model_bundle = PathField(
            directory=True,
            placeholder="Versioned optical phase-model folder",
        )
        model_layout.addWidget(self.phase_model_bundle, 0, 1, 1, 4)
        self.inspect_phase_model_button = QPushButton("Inspect shared model")
        self.inspect_phase_model_button.clicked.connect(self._inspect_phase_model)
        model_layout.addWidget(self.inspect_phase_model_button, 1, 1)
        self.save_phase_model_plan = QPushButton("Register shared model")
        self.save_phase_model_plan.clicked.connect(self._save_phase_model_plan)
        model_layout.addWidget(self.save_phase_model_plan, 1, 2)
        model_layout.setColumnStretch(4, 1)
        self.work_layout.addWidget(model_group)

        self.phase_model_summary = QLabel(
            "Choose a model bundle. Registration fingerprints the model and its "
            "annotation evidence; it never changes the imported files."
        )
        self.phase_model_summary.setObjectName("secondaryText")
        self.phase_model_summary.setWordWrap(True)
        self.work_layout.addWidget(self.phase_model_summary)

        source_group = QFrame()
        source_group.setFrameShape(QFrame.Shape.StyledPanel)
        source_layout = QGridLayout(source_group)
        source_layout.addWidget(QLabel("Campaign manifest"), 0, 0)
        self.phase_campaign_manifest = PathField(
            directory=False,
            file_filter="JSON manifests (*.json);;All files (*)",
            placeholder="Existing phase campaign_manifest.json",
        )
        source_layout.addWidget(self.phase_campaign_manifest, 0, 1, 1, 4)
        source_layout.addWidget(QLabel("Review package"), 1, 0)
        self.phase_review_package = PathField(
            directory=True,
            placeholder="Existing full-resolution manual-review folder",
        )
        source_layout.addWidget(self.phase_review_package, 1, 1, 1, 4)
        self.inspect_phase_campaign_button = QPushButton("Inspect campaign")
        self.inspect_phase_campaign_button.clicked.connect(self._inspect_campaign)
        source_layout.addWidget(self.inspect_phase_campaign_button, 2, 1)
        self.open_phase_review = QPushButton("Open review folder")
        self.open_phase_review.clicked.connect(self._open_review_folder)
        source_layout.addWidget(self.open_phase_review, 2, 2)
        source_layout.setColumnStretch(4, 1)
        self.work_layout.addWidget(source_group)

        self.phase_campaign_summary = QLabel(
            "Choose an existing phase campaign. Inspection alone starts no work; "
            "review and validation are explicit actions below. The Phase Model Studio "
            "tab opens review, training, and concentration and mass calibration."
        )
        self.phase_campaign_summary.setObjectName("secondaryText")
        self.phase_campaign_summary.setWordWrap(True)
        self.work_layout.addWidget(self.phase_campaign_summary)

        candidate_splitter = QSplitter(Qt.Orientation.Horizontal)
        candidates_panel = QWidget()
        candidates_layout = QVBoxLayout(candidates_panel)
        candidates_layout.setContentsMargins(0, 0, 0, 0)
        candidates_heading = QLabel("Campaign candidates")
        candidates_heading.setObjectName("sectionTitle")
        candidates_layout.addWidget(candidates_heading)
        self.phase_candidates = QTableWidget(0, 6)
        self.phase_candidates.setHorizontalHeaderLabels(
            ["Run", "Role", "Candidate", "Family", "Physics", "Deployable"]
        )
        self.phase_candidates.verticalHeader().setVisible(False)
        self.phase_candidates.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        candidate_header = self.phase_candidates.horizontalHeader()
        candidate_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        candidate_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        candidate_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for column in (3, 4, 5):
            candidate_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        candidates_layout.addWidget(self.phase_candidates, 1)
        candidate_splitter.addWidget(candidates_panel)

        selection_panel = QWidget()
        selection_layout = QVBoxLayout(selection_panel)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_heading = QLabel("Final candidate assignment")
        selection_heading.setObjectName("sectionTitle")
        selection_layout.addWidget(selection_heading)
        selection_form = QFormLayout()
        self.phase_run = QComboBox()
        self.phase_run.currentIndexChanged.connect(self._phase_run_changed)
        self.phase_candidate = QComboBox()
        selection_form.addRow("Run", self.phase_run)
        selection_form.addRow("Candidate", self.phase_candidate)
        selection_layout.addLayout(selection_form)
        use_candidate = QPushButton("Use candidate for run")
        use_candidate.clicked.connect(self._use_phase_candidate)
        selection_layout.addWidget(use_candidate)
        self.phase_selected = QTableWidget(0, 3)
        self.phase_selected.setHorizontalHeaderLabels(["Run", "Role", "Candidate"])
        self.phase_selected.verticalHeader().setVisible(False)
        self.phase_selected.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        selected_header = self.phase_selected.horizontalHeader()
        selected_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        selected_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        selected_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        selection_layout.addWidget(self.phase_selected, 1)
        clear_candidate = QPushButton("Clear selected run")
        clear_candidate.clicked.connect(self._clear_phase_candidate)
        selection_layout.addWidget(clear_candidate)
        candidate_splitter.addWidget(selection_panel)
        candidate_splitter.setStretchFactor(0, 3)
        candidate_splitter.setStretchFactor(1, 2)
        self.work_layout.addWidget(candidate_splitter, 2)

        review_group = QFrame()
        review_group.setFrameShape(QFrame.Shape.StyledPanel)
        review_layout = QFormLayout(review_group)
        self.phase_review_files = QTextEdit()
        self.phase_review_files.setPlaceholderText(
            "Optional saved manual-review JSON files; one path per line"
        )
        self.phase_review_files.setMaximumHeight(82)
        self.phase_reviewer = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.phase_revision_reason = QLineEdit()
        self.phase_revision_reason.setPlaceholderText(
            "What was inspected or selected in this revision?"
        )
        review_layout.addRow("Manual-review files", self.phase_review_files)
        review_layout.addRow("Reviewer", self.phase_reviewer)
        review_layout.addRow("Revision note", self.phase_revision_reason)
        self.work_layout.addWidget(review_group)

        revision_actions = QHBoxLayout()
        self.validate_phase_revision = QPushButton("Validate references")
        self.validate_phase_revision.clicked.connect(self._validate_phase_config)
        revision_actions.addWidget(self.validate_phase_revision)
        self.save_phase_revision = QPushButton("Save phase revision")
        self.save_phase_revision.clicked.connect(self._save_phase_revision)
        revision_actions.addWidget(self.save_phase_revision)
        revision_actions.addStretch(1)
        revision_actions.addWidget(QLabel("Saved revision"))
        self.phase_revisions = QComboBox()
        self.phase_revisions.setMinimumWidth(270)
        revision_actions.addWidget(self.phase_revisions)
        self.work_layout.addLayout(revision_actions)
        self.phase_validation_summary = QLabel("No campaign inspected.")
        self.phase_validation_summary.setObjectName("secondaryText")
        self.phase_validation_summary.setWordWrap(True)
        self.work_layout.addWidget(self.phase_validation_summary)

        self.phase_workflow_tabs = QTabWidget()
        self.subpage_registry.populate_tabs(
            "phase_interpretation", self, self.phase_workflow_tabs
        )
        self.work_layout.addWidget(self.phase_workflow_tabs)

        self._load_current_phase_learning_plan()
        self._load_current_phase_revision()
        self.refresh()

    def _review_root_candidate(self) -> Path | None:
        candidates = (
            self.phase_campaign_manifest.text(),
            self.phase_review_package.text(),
        )
        for value in candidates:
            if not str(value).strip():
                continue
            path = Path(value).expanduser()
            current = path.parent if path.is_file() else path
            for root in (current, *current.parents):
                if (root / "postflight_manifest.json").is_file():
                    return root.resolve()
        return None

    def _use_inspected_review_campaign(self, mode: str) -> None:
        root = self._review_root_candidate()
        if root is None:
            QMessageBox.warning(
                self,
                "Prepared review campaign unavailable",
                "Inspect a campaign whose prepared root contains postflight_manifest.json.",
            )
            return
        field = (
            self.phase_active_learning_root
            if mode == "active_learning"
            else self.phase_model_validation_root
        )
        field.setText(str(root))

    def _launch_phase_review(self, mode: str) -> None:
        field = (
            self.phase_active_learning_root
            if mode == "active_learning"
            else self.phase_model_validation_root
        )
        save_field = (
            self.phase_active_learning_save_root
            if mode == "active_learning"
            else self.phase_model_validation_save_root
        )
        status = (
            self.phase_active_learning_status
            if mode == "active_learning"
            else self.phase_model_validation_status
        )
        try:
            command = build_phase_review_command(
                mode=mode,
                campaign_root=Path(field.text()),
                reviewer=self.phase_reviewer.text().strip(),
                save_root=Path(save_field.text()),
                cache_root=self.project.root / "cache" / "phase_review",
                selected_candidates=self.selected_candidates,
            )
            Path(save_field.text()).expanduser().resolve().mkdir(
                parents=True, exist_ok=True
            )
            started, process_id = QProcess.startDetached(command[0], command[1:])
            if not started:
                raise RuntimeError("The review process did not start")
        except Exception as exc:  # noqa: BLE001
            status.setText(f"Could not open review: {type(exc).__name__}: {exc}")
            QMessageBox.critical(self, "Could not open phase review", str(exc))
            return
        label = (
            "Active learning review"
            if mode == "active_learning"
            else "Single-model validation"
        )
        status.setText(
            f"{label} opened as process {int(process_id)}. Every edit is autosaved "
            "under the selected reviewer-owned local folder."
        )

    def _open_phase_contribution(self) -> None:
        """Open editable evidence collection without changing project approval."""

        source = self.phase_model_validation_root.text() or str(
            self._review_root_candidate() or ""
        )
        if source:
            self.phase_active_learning_root.setText(source)
        self._launch_phase_review("active_learning")

    def _load_current_phase_learning_plan(self) -> None:
        current = self.learning_store.current()
        plan = dict(current.get("plan") or {})
        path = str(plan.get("model_bundle_path") or "")
        if not path:
            return
        self.phase_model_bundle.setText(path)
        self._inspect_phase_model(show_errors=False)
        revision_id = str(current.get("revision_id") or "")
        if revision_id:
            self.phase_model_summary.setText(
                self.phase_model_summary.text()
                + f" Registered revision {revision_id[:8]}."
            )

    def _inspect_phase_model(
        self, *, show_errors: bool = True
    ) -> dict[str, Any] | None:
        try:
            summary = inspect_phase_model_bundle(Path(self.phase_model_bundle.text()))
        except Exception as exc:  # noqa: BLE001
            self.phase_model_bundle_summary = None
            self.phase_model_summary.setText("Shared model inspection failed.")
            if show_errors:
                QMessageBox.critical(self, "Phase model unavailable", str(exc))
            return None
        self.phase_model_bundle_summary = summary
        inventory = dict(summary.get("training_inventory") or {})
        self.phase_model_summary.setText(
            f"{summary.get('campaign_id')} · datasets "
            f"{', '.join(summary.get('datasets') or [])} · "
            f"{len(summary.get('runs') or [])} runs · "
            f"{int(inventory.get('regions') or 0):,} reviewed regions · adapters "
            f"{', '.join(summary.get('adapters') or [])} · DarSIA reference-change "
            "features active · visible 2D phase probabilities only"
        )
        return summary

    def _save_phase_model_plan(self) -> None:
        summary = self._inspect_phase_model(show_errors=False)
        if summary is None:
            QMessageBox.warning(
                self,
                "Phase model unavailable",
                "Choose and inspect a valid shared phase-model bundle first.",
            )
            return
        root = Path(self.phase_model_bundle.text()).expanduser().resolve()
        evidence = sorted((root / "input_snapshot").glob("*.json"))
        try:
            plan = phase_learning_plan_from_bundle(
                root,
                annotation_files=evidence,
            )
            report = validate_phase_learning_plan(plan)
            revision = self.learning_store.save_revision(
                plan,
                actor=self.phase_reviewer.text().strip(),
                reason=(
                    self.phase_revision_reason.text().strip()
                    or "Register shared optical phase model"
                ),
                validation=report,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not register phase model", str(exc))
            return
        self.phase_model_summary.setText(
            self.phase_model_summary.text()
            + f" Registered revision {str(revision['revision_id'])[:8]}."
        )
        QMessageBox.information(
            self,
            "Shared phase model registered",
            "The cross-series optical model and its source evidence were "
            "fingerprinted without modifying the imported bundle.",
        )

    def _load_current_phase_revision(self) -> None:
        current = self.store.current()
        config = dict(current.get("configuration") or {})
        manifest = str(config.get("campaign_manifest_path") or "")
        review_package = str(config.get("review_package_path") or "")
        if manifest:
            self.phase_campaign_manifest.setText(manifest)
        if review_package:
            self.phase_review_package.setText(review_package)
        self.selected_candidates = {
            str(key).strip().casefold(): str(value)
            for key, value in dict(config.get("selected_candidates") or {}).items()
        }
        self.run_reviews = {
            str(key).casefold(): dict(value)
            for key, value in dict(config.get("run_reviews") or {}).items()
            if isinstance(value, dict)
        }
        self.phase_review_files.setPlainText(
            "\n".join(str(path) for path in config.get("manual_review_files") or [])
        )
        if manifest and Path(manifest).is_file():
            self._inspect_campaign(show_errors=False)

    def _inspect_campaign(self, *, show_errors: bool = True) -> None:
        try:
            campaign = inspect_phase_campaign(Path(self.phase_campaign_manifest.text()))
        except Exception as exc:  # noqa: BLE001
            self.campaign = None
            self.phase_candidates.setRowCount(0)
            self.phase_run.clear()
            self.phase_candidate.clear()
            self.phase_campaign_summary.setText("Campaign inspection failed.")
            if show_errors:
                QMessageBox.critical(self, "Campaign unavailable", str(exc))
            return
        self.campaign = campaign
        review_path = str(campaign.get("review_output_dir") or "")
        if review_path and not self.phase_review_package.text():
            self.phase_review_package.setText(review_path)
        jobs = list(campaign.get("jobs") or [])
        self.phase_candidates.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            flags = dict(job.get("quality_flags") or {})
            values = (
                job.get("run"),
                job.get("role"),
                job.get("candidate_id"),
                job.get("family"),
                "Yes" if flags.get("hierarchical_phase_partition_active") else "No",
                "Yes" if campaign.get("deployment_allowed") else "No",
            )
            for column, value in enumerate(values):
                self.phase_candidates.setItem(
                    row, column, QTableWidgetItem(str(value or ""))
                )
        self.phase_run.blockSignals(True)
        self.phase_run.clear()
        for run_id in campaign.get("runs") or []:
            self.phase_run.addItem(str(run_id), str(run_id))
        self.phase_run.blockSignals(False)
        self._phase_run_changed()
        self._refresh_phase_selections()
        self._refresh_phase_run_reviews()
        self.phase_campaign_summary.setText(
            f"{campaign['campaign_id']} · dataset {campaign.get('dataset') or 'unspecified'} · "
            f"{len(campaign.get('runs') or [])} runs · {len(jobs)} candidates · "
            f"scope {campaign.get('interpretation_scope')} · "
            "training disabled in this work surface"
        )
        self._validate_phase_config(show_dialog=False)

    def _phase_run_changed(self, _index: int = -1) -> None:
        self.phase_candidate.clear()
        if not self.campaign:
            return
        run_id = str(self.phase_run.currentData() or "")
        for job in (self.campaign.get("candidates_by_run") or {}).get(run_id, []):
            label = f"{job.get('role')} · {str(job.get('candidate_id') or '')[:24]}"
            self.phase_candidate.addItem(label, str(job.get("candidate_id") or ""))
        selected = self.selected_candidates.get(run_id)
        if selected:
            index = self.phase_candidate.findData(selected)
            if index >= 0:
                self.phase_candidate.setCurrentIndex(index)

    def _use_phase_candidate(self) -> None:
        run_id = str(self.phase_run.currentData() or "")
        candidate_id = str(self.phase_candidate.currentData() or "")
        if not run_id or not candidate_id:
            return
        previous = self.selected_candidates.get(run_id)
        self.selected_candidates[run_id] = candidate_id
        if previous != candidate_id:
            self.run_reviews.pop(run_id, None)
        self._refresh_phase_selections()
        self._refresh_phase_run_reviews()
        self._validate_phase_config(show_dialog=False)

    def _clear_phase_candidate(self) -> None:
        run_id = str(self.phase_run.currentData() or "")
        if run_id:
            self.selected_candidates.pop(run_id, None)
            self.run_reviews.pop(run_id, None)
        self._refresh_phase_selections()
        self._refresh_phase_run_reviews()
        self._validate_phase_config(show_dialog=False)

    def _candidate_role(self, run_id: str, candidate_id: str) -> str:
        if not self.campaign:
            return ""
        for row in (self.campaign.get("candidates_by_run") or {}).get(run_id, []):
            if str(row.get("candidate_id") or "") == candidate_id:
                return str(row.get("role") or "")
        return ""

    def _refresh_phase_selections(self) -> None:
        rows = sorted(self.selected_candidates.items())
        self.phase_selected.setRowCount(len(rows))
        for row, (run_id, candidate_id) in enumerate(rows):
            values = (run_id, self._candidate_role(run_id, candidate_id), candidate_id)
            for column, value in enumerate(values):
                self.phase_selected.setItem(row, column, QTableWidgetItem(value))

    def _selected_phase_review_run(self) -> str:
        table = self.phase_run_review_table
        row = table.currentRow()
        if row >= 0:
            item = table.item(row, 0)
            if item is not None:
                return item.text().strip().casefold()
        return str(self.phase_run.currentData() or "").strip().casefold()

    def _set_selected_phase_run_review(self, status: str) -> None:
        run_id = self._selected_phase_review_run()
        candidate_id = self.selected_candidates.get(run_id, "")
        reviewer = self.phase_reviewer.text().strip()
        if not run_id:
            QMessageBox.warning(
                self,
                "Choose an experiment",
                "Select one experiment in the 2D approval table first.",
            )
            return
        if status != "pending" and not candidate_id:
            QMessageBox.warning(
                self,
                "Select a model first",
                "Assign the model candidate used for this experiment before review.",
            )
            return
        if status != "pending" and not reviewer:
            QMessageBox.warning(
                self,
                "Reviewer required",
                "Enter the reviewer name before recording a 2D phase decision.",
            )
            return
        if status == "pending":
            self.run_reviews.pop(run_id, None)
        else:
            self.run_reviews[run_id] = {
                "status": status,
                "candidate_id": candidate_id,
                "reviewer": reviewer,
                "reviewed_utc": utc_now(),
                "note": self.phase_run_review_note.text().strip(),
                "scope": "registered_visible_camera_plane_2d",
            }
        self.phase_run_review_note.clear()
        self._refresh_phase_run_reviews(select_run=run_id)
        self._validate_phase_config(show_dialog=False)

    def _refresh_phase_run_reviews(self, *, select_run: str = "") -> None:
        if not hasattr(self, "phase_run_review_table"):
            return
        runs = list((self.campaign or {}).get("runs") or [])
        table = self.phase_run_review_table
        table.setRowCount(len(runs))
        status_labels = {
            "pending": "Pending",
            "approved": "Approved",
            "needs_local_correction": "Needs local correction",
        }
        selected_row = -1
        for row, raw_run in enumerate(runs):
            run_id = str(raw_run).casefold()
            review = dict(self.run_reviews.get(run_id) or {})
            status = str(review.get("status") or "pending")
            values = (
                run_id,
                self.selected_candidates.get(run_id, "Not selected"),
                status_labels.get(status, status),
                str(review.get("reviewer") or ""),
                str(review.get("note") or ""),
            )
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(str(value)))
            if run_id == select_run:
                selected_row = row
        if selected_row >= 0:
            table.selectRow(selected_row)
        elif runs and table.currentRow() < 0:
            table.selectRow(0)

    def _phase_configuration(self) -> dict[str, Any]:
        config = default_phase_interpretation_config()
        config.update(
            {
                "format": PHASE_INTERPRETATION_CONFIG_FORMAT,
                "campaign_manifest_path": self.phase_campaign_manifest.text(),
                "review_package_path": self.phase_review_package.text(),
                "selected_candidates": dict(sorted(self.selected_candidates.items())),
                "run_reviews": {
                    key: dict(value) for key, value in sorted(self.run_reviews.items())
                },
                "manual_review_files": parse_review_file_list(
                    self.phase_review_files.toPlainText()
                ),
                "training_requested": False,
            }
        )
        return config

    def _validate_phase_config(
        self, *, show_dialog: bool = True
    ) -> dict[str, Any] | None:
        try:
            report = validate_phase_interpretation_config(self._phase_configuration())
        except Exception as exc:  # noqa: BLE001
            self.phase_validation_summary.setText("Validation failed.")
            if show_dialog:
                QMessageBox.critical(self, "Validation failed", str(exc))
            return None
        summary = dict(report.get("summary") or {})
        self.phase_validation_summary.setText(
            f"References: {summary.get('error_count', 0)} errors, "
            f"{summary.get('warning_count', 0)} warnings · "
            f"selected {summary.get('selected_run_count', 0)} of "
            f"{summary.get('run_count', 0)} runs · "
            f"2D approved {summary.get('approved_run_count', 0)} of "
            f"{summary.get('run_count', 0)} · "
            f"{summary.get('manual_review_file_count', 0)} review files"
        )
        if show_dialog:
            messages = [str(issue.get("message") or "") for issue in report["issues"]]
            QMessageBox.information(
                self,
                "Phase reference validation",
                self.phase_validation_summary.text()
                + ("\n\n" + "\n".join(messages) if messages else ""),
            )
        return report

    def _save_phase_revision(self) -> None:
        report = self._validate_phase_config(show_dialog=False)
        if report is None:
            return
        try:
            revision = self.store.save_revision(
                self._phase_configuration(),
                actor=self.phase_reviewer.text().strip(),
                reason=self.phase_revision_reason.text().strip(),
                validation=report,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save phase revision", str(exc))
            return
        self.phase_revision_reason.clear()
        self.refresh()
        QMessageBox.information(
            self,
            "Phase revision saved",
            f"Revision {str(revision['revision_id'])[:8]} was stored without "
            "starting model training.",
        )

    def _open_review_folder(self) -> None:
        path = Path(self.phase_review_package.text()).expanduser()
        if not path.is_dir():
            QMessageBox.warning(
                self, "Review folder unavailable", "Choose an existing review folder."
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def refresh(self) -> None:
        self._refresh_gate()
        self.phase_revisions.clear()
        current_id = str(self.store.current().get("revision_id") or "")
        rows = list(reversed(self.store.revisions()))
        if not rows:
            self.phase_revisions.addItem("No saved revisions", "")
        for revision in rows:
            revision_id = str(revision.get("revision_id") or "")
            campaign_id = str((revision.get("campaign") or {}).get("campaign_id") or "")
            self.phase_revisions.addItem(
                f"{str(revision.get('recorded_utc') or '')[:19]} · "
                f"{campaign_id} · {revision_id[:8]}",
                revision_id,
            )
            if revision_id == current_id:
                self.phase_revisions.setCurrentIndex(self.phase_revisions.count() - 1)

    def _set_status(self, status: str) -> None:
        if status in {"approved", "complete"}:
            current = self.store.current()
            summary = dict((current.get("validation") or {}).get("summary") or {})
            if not current or int(summary.get("error_count") or 0) > 0:
                QMessageBox.warning(
                    self,
                    "Phase revision required",
                    "Save a phase revision without blocking reference errors first.",
                )
                return
            if int(summary.get("selected_run_count") or 0) < int(
                summary.get("run_count") or 0
            ):
                QMessageBox.warning(
                    self,
                    "Candidate selection incomplete",
                    "Select one final candidate for every campaign run before approval.",
                )
                return
            if int(summary.get("approved_run_count") or 0) < int(
                summary.get("run_count") or 0
            ):
                QMessageBox.warning(
                    self,
                    "2D phase review incomplete",
                    "Inspect and approve the visible 2D phase map for every "
                    "experiment before approving this quality gate.",
                )
                return
        super()._set_status(status)
