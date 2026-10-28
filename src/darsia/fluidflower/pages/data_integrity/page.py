"""Qt data-integrity workflow page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.concurrency import MutableWorkerLimit
from darsia.fluidflower.data_integrity import (
    IntegrityReportStore,
    IntegrityScanner,
    IntegrityScanOptions,
    ScanCancelled,
    recommend_integrity_read_workers,
)
from darsia.fluidflower.execution import detect_host_resources
from darsia.fluidflower.image_formats import IMAGE_FILE_POLICY_LABELS
from darsia.fluidflower.project import FluidFlowerProject, sha256_file
from darsia.fluidflower.ui.common import ModulePage


_SOURCE_KIND_LABELS = {
    "alternate_color_reference_image_folder": "Other color-state folder",
    "alternate_color_reference_images": "Other color-state images",
    "baseline_image": "Baseline image",
    "baseline_images": "Baseline images",
    "clearwater_image": "Clear-water image",
    "clearwater_image_folder": "Clear-water image folder",
    "clearwater_images": "Clear-water images",
    "co2_image_folder": "CO2 image folder",
    "colorchecker_images": "ColorChecker images",
    "experiment_images": "Experiment images",
    "indicator_reference_image_folder": "Measured-pH buffer folder",
    "indicator_reference_images": "Measured-pH buffer images",
    "overflow_image_folder": "Overflow image folder",
    "overflow_table": "Overflow table",
    "protocol": "Protocol workbook",
    "run_folder": "Experiment folder",
}

_FINDING_GUIDANCE = {
    "source_unavailable": (
        "The registered source could not be reached after all network retries.",
        (
            "Check the connection and source path, correct the run setup if necessary, "
            "then scan the affected source again."
        ),
    ),
    "file_unreadable": (
        "DarSIA could not read the file after all network retries.",
        (
            "Open the example location, check the file and connection, then rescan. "
            "Correct the source mapping if the file does not belong to this run."
        ),
    ),
    "duplicate_sampling_failed": (
        "A possible duplicate could not be sampled completely.",
        "Check the example file and connection, then rescan.",
    ),
    "duplicate_verification_failed": (
        "A likely duplicate could not be verified byte for byte.",
        "Check the example file and connection, then rescan.",
    ),
    "exact_duplicate_files": (
        "Two or more registered files contain exactly the same bytes.",
        (
            "Inspect whether the duplicate is intentional. Correct the source mapping "
            "or exclude the duplicate downstream if it is not part of the experiment."
        ),
    ),
    "resolution_outliers": (
        "Some images differ from the dominant resolution in this source.",
        (
            "Inspect the outliers. Correct the mapping or blacklist unrelated or damaged "
            "frames before image correction."
        ),
    ),
    "duplicate_sequence_numbers": (
        "More than one image uses the same apparent sequence number.",
        (
            "Inspect the folder and establish the intended image order during temporal "
            "synchronization."
        ),
    ),
    "missing_sequence_numbers": (
        "The apparent camera sequence contains one or more numbering gaps.",
        (
            "Compare the folder with the protocol. A real acquisition gap may be "
            "documented; an incorrect folder assignment should be corrected and rescanned."
        ),
    ),
    "sequence_number_discontinuity": (
        "A large numbering jump may indicate a camera-card or acquisition restart.",
        "Confirm the event against the protocol during temporal synchronization.",
    ),
    "missing_image_timestamps": (
        "One or more camera images have no readable embedded EXIF timestamp.",
        (
            "Use filenames and the protocol to establish time during temporal "
            "synchronization, or correct the source if these are not experiment images."
        ),
    ),
    "duplicate_image_timestamps": (
        "Multiple camera images contain the same embedded timestamp.",
        (
            "Resolve their order from filenames and the protocol during temporal "
            "synchronization."
        ),
    ),
}


def _source_kind_label(kind: Any) -> str:
    normalized = str(kind or "").strip()
    return _SOURCE_KIND_LABELS.get(
        normalized,
        normalized.replace("_", " ").strip().capitalize() or "Unknown source",
    )


def _guidance_for_finding(code: str, source_kind: str) -> tuple[str, str]:
    if source_kind == "overflow_image_folder":
        if code == "missing_sequence_numbers":
            return (
                (
                    "This older report interpreted the seconds in timestamped overflow "
                    "filenames as camera sequence numbers."
                ),
                (
                    "No raw-data change is required. Create a new integrity report; the "
                    "updated scanner no longer applies this sequence check to overflow images."
                ),
            )
        if code == "exact_duplicate_files":
            return (
                (
                    "An unchanged scale display can legitimately produce identical overflow "
                    "screenshots."
                ),
                (
                    "No raw-data change is normally required. New scans omit this "
                    "camera-frame duplicate check for overflow images."
                ),
            )
        if code == "missing_image_timestamps":
            return (
                "Overflow screenshots commonly store time in the filename rather than EXIF.",
                (
                    "No raw-data change is required when the filename timestamp is readable. "
                    "The overflow workflow validates those timestamps separately."
                ),
            )
    return _FINDING_GUIDANCE.get(
        code,
        (
            "The integrity scan found a condition that needs review.",
            (
                "Inspect the example location, document the decision, and rescan if the "
                "source mapping or files are changed."
            ),
        ),
    )


class IntegrityWorker(QObject):
    progress = Signal(int, int, str)
    worker_status = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        project: FluidFlowerProject,
        options: IntegrityScanOptions,
        source_ids: list[str],
        actor: str,
        worker_limit: MutableWorkerLimit,
    ) -> None:
        super().__init__()
        self.project = project
        self.options = options
        self.source_ids = source_ids
        self.actor = actor
        self.worker_limit = worker_limit
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    @Slot()
    def run(self) -> None:
        try:
            report = IntegrityScanner(self.project, self.options).scan(
                source_ids=self.source_ids,
                actor=self.actor,
                progress=lambda current, total, path: self.progress.emit(
                    current, total, path
                ),
                cancelled=lambda: self.cancel_requested,
                max_parallel_reads=self.worker_limit.maximum,
                worker_limit=self.worker_limit.get,
                worker_status=self.worker_status.emit,
            )
        except ScanCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(report)


class DataIntegrityPage(ModulePage):
    module_requested = Signal(str)
    review_setup_requested = Signal(str)

    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__(project, "data_integrity")
        self.store = IntegrityReportStore(project)
        self.current_report: dict[str, Any] | None = None
        self._thread: QThread | None = None
        self._worker: IntegrityWorker | None = None
        self._worker_limit: MutableWorkerLimit | None = None
        self._active_worker_count = 0
        self._desired_worker_count = 0
        self._host_resources: dict[str, Any] = {}
        self._worker_recommendation: dict[str, int | None] = {}
        self._active_scan_run_ids: tuple[str, ...] = ()
        self._report_cache: list[dict[str, Any]] | None = None
        self._finding_groups: list[dict[str, Any]] = []
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.start.setVisible(False)
        self.review.setVisible(False)
        self.complete.setVisible(False)
        self.approve.setText("Approve current integrity report")
        self.approve.setToolTip(
            "Approve the current report when it has no blocking errors. Warnings may be "
            "accepted with an optional review note."
        )
        self.reason.setPlaceholderText("Optional note for report approval")
        self.integration.setText(
            "Fast scans read metadata, image headers, and small file samples. Verified scans also "
            "validate complete images and hash every byte. Reports are immutable, source files "
            "are never modified, and transient read failures are retried. Only the current source "
            "mapping for included experiments is scanned; excluded experiment folders beneath "
            "the series data folder are not read. Current image choice: "
            f"{IMAGE_FILE_POLICY_LABELS[project.image_file_policy()]}."
        )
        self.run_scope_note = QLabel()
        self.run_scope_note.setWordWrap(True)
        self.run_scope_note.setObjectName("secondaryText")
        self.work_layout.addWidget(self.run_scope_note)
        next_step_row = QHBoxLayout()
        self.next_step = QLabel()
        self.next_step.setWordWrap(True)
        self.next_step.setObjectName("integrationNotice")
        next_step_row.addWidget(self.next_step, 1)
        self.continue_button = QPushButton("Continue to Overflow measurements")
        self.continue_button.setVisible(False)
        self.continue_button.clicked.connect(
            lambda: self.module_requested.emit("overflow")
        )
        next_step_row.addWidget(self.continue_button)
        self.work_layout.addLayout(next_step_row)

        scan_options = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem("Fast scan", "fast")
        self.mode.addItem("Verified SHA-256 scan", "verified")
        fast_help = (
            "Recommended for the first check. Reads file information, image headers, "
            "and small samples. It finds common problems quickly without reading every "
            "byte of every file."
        )
        verified_help = (
            "Slow, complete verification for final quality control or after moving data. "
            "Every image is fully read and every file receives a SHA-256 fingerprint, "
            "which can prove later that its contents have not changed."
        )
        mode_help = f"Fast scan: {fast_help}\n\nVerified SHA-256 scan: {verified_help}"
        self.mode.setItemData(0, fast_help, Qt.ItemDataRole.ToolTipRole)
        self.mode.setItemData(1, verified_help, Qt.ItemDataRole.ToolTipRole)
        self.mode.setToolTip(mode_help)
        mode_label = QLabel("Mode")
        mode_label.setToolTip(mode_help)
        scan_options.addWidget(mode_label)
        scan_options.addWidget(self.mode)
        self.retry_count = QSpinBox()
        self.retry_count.setRange(1, 8)
        self.retry_count.setValue(3)
        retry_help = (
            "How many times DarSIA retries a file when a network share is briefly "
            "unavailable. Use 3 normally. Increase it only when the NAS connection "
            "is unstable; this does not change the scientific analysis."
        )
        self.retry_count.setToolTip(retry_help)
        retry_label = QLabel("Network attempts")
        retry_label.setToolTip(retry_help)
        scan_options.addWidget(retry_label)
        scan_options.addWidget(self.retry_count)
        self.parallel_reads = QSpinBox()
        self.workers_label = QLabel("Read workers")
        self.worker_activity = QLabel()
        self.worker_activity.setObjectName("secondaryText")
        self.worker_activity.setWordWrap(True)
        self._refresh_worker_recommendation(use_recommended=True)
        self.parallel_reads.valueChanged.connect(self._requested_worker_count_changed)
        scan_options.addWidget(self.workers_label)
        scan_options.addWidget(self.parallel_reads)
        scan_options.addStretch(1)
        self.reports = QComboBox()
        self.reports.setMinimumWidth(320)
        self.reports.setToolTip(
            "Previously saved integrity reports. Reports are kept so changes can be "
            "reviewed without replacing earlier checks."
        )
        self.reports.currentIndexChanged.connect(self._report_selected)
        scan_options.addWidget(QLabel("Report"))
        scan_options.addWidget(self.reports)
        self.work_layout.addLayout(scan_options)
        self.work_layout.addWidget(self.worker_activity)

        scan_actions = QHBoxLayout()
        self.reviewer = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.reviewer.setPlaceholderText("Reviewer")
        self.reviewer.setToolTip(
            "Name recorded with this quality-control report for traceability."
        )
        self.reviewer.setMinimumWidth(180)
        self.reviewer.setMaximumWidth(320)
        scan_actions.addWidget(QLabel("Reviewer"))
        scan_actions.addWidget(self.reviewer)
        self.scan_button = QPushButton("Scan selected sources")
        self.scan_button.setToolTip(
            "Check only the source rows whose Use box is selected. Source files remain read-only."
        )
        self.scan_button.clicked.connect(self._start_scan)
        scan_actions.addWidget(self.scan_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_scan)
        scan_actions.addWidget(self.cancel_button)
        scan_actions.addStretch(1)
        self.work_layout.addLayout(scan_actions)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        progress_row.addWidget(self.progress, 1)
        self.progress_path = QLabel("Ready")
        self.progress_path.setObjectName("pathPreview")
        self.progress_path.setMinimumWidth(420)
        progress_row.addWidget(self.progress_path, 2)
        self.work_layout.addLayout(progress_row)

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        source_panel = QWidget()
        source_layout = QHBoxLayout(source_panel)
        source_layout.setContentsMargins(0, 0, 0, 0)
        self.sources = QTreeWidget()
        self.sources.setHeaderLabels(
            [
                "Use",
                "Source type",
                "Run",
                "Registered location",
                "Availability",
                "Files",
                "Images",
                "Size",
                "Findings",
                "Run review",
            ]
        )
        source_header = self.sources.header()
        source_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        source_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(9, QHeaderView.ResizeMode.ResizeToContents)
        self.sources.setMinimumHeight(175)
        source_layout.addWidget(self.sources, 1)
        main_splitter.addWidget(source_panel)

        findings_panel = QWidget()
        findings_layout = QVBoxLayout(findings_panel)
        findings_layout.setContentsMargins(0, 0, 0, 0)
        findings_toolbar = QHBoxLayout()
        self.report_summary = QLabel("No integrity report has been created.")
        self.report_summary.setObjectName("secondaryText")
        findings_toolbar.addWidget(self.report_summary, 1)
        self.issue_filter = QComboBox()
        self.issue_filter.addItem("All findings", "all")
        self.issue_filter.addItem("Errors", "error")
        self.issue_filter.addItem("Warnings", "warning")
        self.issue_filter.addItem("Information", "info")
        self.issue_filter.currentIndexChanged.connect(self._populate_findings)
        findings_toolbar.addWidget(QLabel("Run"))
        self.run_filter = QComboBox()
        self.run_filter.addItem("All runs", "all")
        self.run_filter.currentIndexChanged.connect(self._populate_findings)
        findings_toolbar.addWidget(self.run_filter)
        findings_toolbar.addWidget(self.issue_filter)
        self.open_location = QPushButton("Open selected location")
        self.open_location.setEnabled(False)
        self.open_location.clicked.connect(self._open_selected_location)
        findings_toolbar.addWidget(self.open_location)
        self.review_source = QPushButton("Review selected run setup...")
        self.review_source.setEnabled(False)
        self.review_source.setToolTip(
            "Open source setup at the run associated with the selected finding."
        )
        self.review_source.clicked.connect(self._request_source_review)
        findings_toolbar.addWidget(self.review_source)
        findings_layout.addLayout(findings_toolbar)
        self.finding_guidance = QLabel(
            "Errors block approval. Warnings require review but do not block approval; "
            "information records context for later workflow steps. Select a row for a "
            "specific explanation and next action."
        )
        self.finding_guidance.setWordWrap(True)
        self.finding_guidance.setObjectName("integrationNotice")
        findings_layout.addWidget(self.finding_guidance)
        self.findings = QTableWidget(0, 7)
        self.findings.setHorizontalHeaderLabels(
            [
                "Severity",
                "Finding",
                "Run",
                "Source type",
                "Report entries",
                "Example location",
                "Recommended action",
            ]
        )
        self.findings.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.findings.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.findings.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.findings.verticalHeader().setVisible(False)
        self.findings.itemSelectionChanged.connect(self._finding_selected)
        self.findings.itemDoubleClicked.connect(
            lambda _item: self._open_selected_location()
        )
        finding_header = self.findings.horizontalHeader()
        finding_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        finding_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        finding_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        finding_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        finding_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        finding_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        finding_header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        findings_layout.addWidget(self.findings, 1)
        main_splitter.addWidget(findings_panel)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)
        self.work_layout.addWidget(main_splitter, 1)
        self._populate_sources()
        self.refresh()

    @staticmethod
    def _format_size(size: Any) -> str:
        value = float(size or 0)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024.0 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return ""

    def _refresh_worker_recommendation(self, *, use_recommended: bool) -> None:
        previous = self.parallel_reads.value()
        self._host_resources = detect_host_resources()
        self._worker_recommendation = recommend_integrity_read_workers(
            self._host_resources
        )
        recommended = int(self._worker_recommendation["recommended"] or 1)
        maximum = int(self._worker_recommendation["maximum"] or recommended)
        selected = recommended if use_recommended else max(1, min(previous, maximum))
        self.parallel_reads.blockSignals(True)
        self.parallel_reads.setRange(1, maximum)
        self.parallel_reads.setValue(selected)
        self.parallel_reads.blockSignals(False)
        cpu = self._host_resources.get("cpu") or {}
        logical = int(cpu.get("logical_cores") or 1)
        physical_value = cpu.get("physical_cores")
        physical_text = (
            str(int(physical_value))
            if isinstance(physical_value, (int, float)) and physical_value
            else "not reported"
        )
        available = self._worker_recommendation.get("available_memory_bytes")
        memory_text = (
            self._format_size(available) if isinstance(available, int) else "unknown"
        )
        help_text = (
            "Number of files checked at the same time. This changes read throughput, "
            "not analysis accuracy. It remains editable during a scan; increases are "
            "applied after an active file finishes, while decreases let current reads "
            "finish before reducing concurrency. DarSIA detected "
            f"{logical} logical CPU cores, {physical_text} physical cores, and "
            f"{memory_text} available RAM. The hardware-derived recommendation is "
            f"{recommended}; the selectable safety range is 1-{maximum}. A slow or "
            "unstable disk/NAS may work better below the recommendation."
        )
        self.parallel_reads.setToolTip(help_text)
        self.workers_label.setToolTip(help_text)
        if self._worker_limit is None:
            self._desired_worker_count = selected
            self.worker_activity.setText(
                f"Detected {logical} logical CPU cores and {memory_text} available RAM. "
                f"Recommended: {recommended}; selected: {selected}; hardware-derived "
                f"maximum: {maximum}."
            )

    def _populate_sources(self) -> None:
        self.sources.clear()
        pending_source_ids = self._pending_run_source_ids()
        incremental_selection = bool(pending_source_ids)
        run_statuses = self._run_integrity_statuses()
        for source in self.project.current_source_records():
            run_id = str(source.get("run_id") or "")
            reference_only = (
                str(source.get("scan_policy") or "content") == "reference_only"
            )
            series_container = source.get("effective_role") == "series_container"
            item = QTreeWidgetItem(
                [
                    "",
                    (
                        "Series data folder"
                        if series_container
                        else _source_kind_label(source.get("kind"))
                    ),
                    run_id if run_id else "Series",
                    str(source.get("path") or ""),
                    (
                        "Container only; not scanned"
                        if series_container
                        else (
                            "Reference only"
                            if reference_only
                            else (
                                "Present when registered"
                                if source.get("exists_at_registration")
                                else "Missing when registered"
                            )
                        )
                    ),
                    "Not scanned",
                    "-",
                    "-",
                    "-",
                    (
                        run_statuses.get(run_id, "Series-level")
                        if run_id
                        else "Series-level"
                    ),
                ]
            )
            item.setData(2, Qt.ItemDataRole.UserRole, run_id)
            if not reference_only:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            source_id = str(source.get("source_id") or "")
            item.setCheckState(
                0,
                (
                    Qt.CheckState.Checked
                    if not reference_only
                    and (not incremental_selection or source_id in pending_source_ids)
                    else Qt.CheckState.Unchecked
                ),
            )
            item.setData(0, Qt.ItemDataRole.UserRole, source_id)
            self.sources.addTopLevelItem(item)

    def _run_integrity_statuses(self) -> dict[str, str]:
        return {
            str(row.get("run_id") or ""): str(
                ((row.get("workflow") or {}).get("data_integrity") or {}).get("status")
                or "not_started"
            )
            for row in self.project.manifest.get("runs") or []
            if isinstance(row, dict)
        }

    def _pending_run_source_ids(self) -> set[str]:
        scannable_ids = {
            str(source.get("source_id") or "")
            for source in self.project.current_source_records()
            if str(source.get("scan_policy") or "content") != "reference_only"
        }
        pending: set[str] = set()
        for row in self.project.manifest.get("runs") or []:
            if not isinstance(row, dict):
                continue
            workflow = row.get("workflow") or {}
            run_status = str((workflow.get("data_integrity") or {}).get("status") or "")
            if run_status in {"approved", "complete"}:
                continue
            mapping = row.get("source_mapping") or {}
            pending.update(
                str(value)
                for value in mapping.get("source_record_ids") or []
                if str(value) in scannable_ids
            )
        return pending

    def _report_matches_current_sources(self, report: dict[str, Any] | None) -> bool:
        expected = str((report or {}).get("source_manifest_sha256") or "")
        if not expected or not self.project.source_manifest_path.is_file():
            return False
        try:
            return sha256_file(self.project.source_manifest_path) == expected
        except OSError:
            return False

    def _scope_run_ids(self, source_ids: list[str]) -> tuple[str, ...]:
        selected = set(source_ids)
        records = [
            source
            for source in self.project.source_manifest().get("sources") or []
            if str(source.get("source_id") or "") in selected
        ]
        if not records or any(
            not str(source.get("run_id") or "") for source in records
        ):
            return ()
        return tuple(
            dict.fromkeys(str(source.get("run_id") or "") for source in records)
        )

    def _report_run_ids(self, report: dict[str, Any] | None) -> tuple[str, ...]:
        return self._scope_run_ids(
            [str(value) for value in (report or {}).get("selected_source_ids") or []]
        )

    def refresh(self) -> None:
        self._synchronize_series_completion()
        self._refresh_gate()
        if self._report_cache is None:
            self._report_cache = self.store.reports()
        reports = self._report_cache
        latest_report = reports[0] if reports else None
        latest_id = str((latest_report or {}).get("scan_id") or "")
        selected_id = str((self.current_report or {}).get("scan_id") or "")
        if not selected_id:
            selected_id = latest_id
        self.reports.blockSignals(True)
        self.reports.clear()
        if not reports:
            self.reports.addItem("No reports", "")
        selected_index = 0
        for index, report in enumerate(reports):
            summary = dict(report.get("summary") or {})
            report_is_current = self._report_matches_current_sources(report)
            label = (
                f"{str(report.get('completed_utc') or '')[:19]} · "
                f"{summary.get('file_count', 0)} files · "
                f"{summary.get('error_count', 0)} errors"
            )
            if str(report.get("scan_id") or "") == latest_id:
                label += (
                    " (latest, current sources)"
                    if report_is_current
                    else " (latest, outdated sources)"
                )
            elif not report_is_current:
                label += " (outdated sources)"
            self.reports.addItem(label, str(report.get("scan_id") or ""))
            if str(report.get("scan_id") or "") == selected_id:
                selected_index = self.reports.count() - 1
        self.reports.setCurrentIndex(selected_index)
        self.reports.blockSignals(False)
        if reports:
            scan_id = str(self.reports.currentData() or "")
            self.current_report = next(
                (
                    report
                    for report in reports
                    if str(report.get("scan_id") or "") == scan_id
                ),
                reports[0],
            )
        self._populate_report()
        latest_summary = dict((latest_report or {}).get("summary") or {})
        has_clean_report = (
            bool(latest_report)
            and self._report_matches_current_sources(latest_report)
            and int(latest_summary.get("error_count") or 0) == 0
        )
        report_scope = self._report_run_ids(latest_report)
        if report_scope:
            statuses = self._run_integrity_statuses()
            selected_statuses = {statuses.get(run_id, "") for run_id in report_scope}
            self.approve.setEnabled(
                has_clean_report
                and bool(
                    selected_statuses
                    & {"review_required", "in_progress", "ready", "stale"}
                )
            )
            self.complete.setEnabled(
                has_clean_report
                and bool(
                    selected_statuses & {"approved", "review_required", "in_progress"}
                )
            )
            self.run_scope_note.setText(
                self._report_scope_text(report_scope)
                if self._report_matches_current_sources(latest_report)
                else (
                    "The latest report belongs to an older source revision and cannot "
                    "be approved. Updated experiments are selected for a new scan."
                )
            )
        else:
            state = str(self.project.module_states[self.module_key].get("status") or "")
            self.approve.setEnabled(
                has_clean_report
                and state in {"review_required", "in_progress", "ready", "stale"}
            )
            self.complete.setEnabled(
                has_clean_report
                and state in {"approved", "review_required", "in_progress"}
            )
            pending = self._pending_run_source_ids()
            self.run_scope_note.setText(
                "New or updated experiments are selected automatically for an "
                "independent integrity review."
                if pending
                else "No run-specific source update is waiting for integrity review."
            )
        run_statuses = self._run_integrity_statuses()
        for row in range(self.sources.topLevelItemCount()):
            item = self.sources.topLevelItem(row)
            run_id = str(item.data(2, Qt.ItemDataRole.UserRole) or "")
            item.setText(
                9,
                run_statuses.get(run_id, "Series-level") if run_id else "Series-level",
            )
        self._refresh_next_step(latest_report)

    def _registered_run_ids(self) -> tuple[str, ...]:
        return tuple(
            str(row.get("run_id") or "")
            for row in self.project.manifest.get("runs") or []
            if isinstance(row, dict) and str(row.get("run_id") or "")
        )

    def _synchronize_series_completion(self) -> None:
        run_ids = self._registered_run_ids()
        if not run_ids:
            return
        statuses = self._run_integrity_statuses()
        if not all(
            statuses.get(run_id) in {"approved", "complete"} for run_id in run_ids
        ):
            return
        state = str(self.project.module_states[self.module_key].get("status") or "")
        if state == "complete":
            return
        self.project.set_module_status(
            self.module_key,
            "complete",
            updated_by=self.reviewer.text().strip(),
            reason=(
                f"Integrity reports approved for all {len(run_ids)} registered "
                "experiments."
            ),
        )

    def _report_scope_text(self, report_scope: tuple[str, ...]) -> str:
        registered = self._registered_run_ids()
        if registered and set(report_scope) == set(registered):
            return (
                f"The current report covers all {len(registered)} registered "
                "experiments. Approving it completes Data and integrity for the series."
            )
        labels = ", ".join(value.upper() for value in report_scope)
        return (
            f"The current report covers {len(report_scope)} of {len(registered)} "
            f"registered experiments: {labels}. Approval changes only these experiments."
        )

    def _refresh_next_step(self, report: dict[str, Any] | None) -> None:
        state = str(self.project.module_states[self.module_key].get("status") or "")
        run_ids = self._registered_run_ids()
        statuses = self._run_integrity_statuses()
        approved_count = sum(
            statuses.get(run_id) in {"approved", "complete"} for run_id in run_ids
        )
        summary = dict((report or {}).get("summary") or {})
        error_count = int(summary.get("error_count") or 0)
        report_is_current = self._report_matches_current_sources(report)
        workflow_complete = state == "complete" and report_is_current
        self.continue_button.setVisible(workflow_complete)
        self.continue_button.setEnabled(workflow_complete)
        if report is not None and not report_is_current:
            self.approve.setText("New integrity report required")
            self.next_step.setText(
                "<b>Source setup changed after this report was created.</b> The report "
                "is retained as history but cannot approve the current source revision. "
                "Scan the automatically selected updated sources, review the new report, "
                "and approve it."
            )
            return
        if workflow_complete:
            self.approve.setText("Integrity report approved")
            self.next_step.setText(
                f"<b>Data and integrity is complete for all {len(run_ids)} "
                "experiments.</b> Next, open Overflow measurements and choose the "
                "available evidence for each run: scale images, an imported table, "
                "or no overflow data."
            )
            return
        if report is None:
            self.approve.setText("Approve current integrity report")
            self.next_step.setText(
                "<b>Next step:</b> Select the registered sources above and run a fast "
                "scan. A saved report is required before this quality gate can be approved."
            )
            return
        if error_count:
            self.approve.setText("Approval blocked by integrity errors")
            self.next_step.setText(
                f"<b>Next step:</b> Resolve the {error_count} blocking error(s), correct "
                "the source setup where necessary, and create a new clean report."
            )
            return
        pending_count = max(0, len(run_ids) - approved_count)
        report_scope = self._report_run_ids(report)
        pending_runs = {
            run_id
            for run_id in run_ids
            if statuses.get(run_id) not in {"approved", "complete"}
        }
        completes_series = bool(run_ids) and pending_runs.issubset(set(report_scope))
        self.approve.setText(
            "Approve report and complete Data and integrity"
            if completes_series
            else "Approve report for covered experiments"
        )
        self.next_step.setText(
            "<b>Next step:</b> Review the grouped warnings below. Warnings do not block "
            "approval. Add an optional note, then approve this report. "
            f"{approved_count} of {len(run_ids)} experiments are currently approved; "
            f"{pending_count} remain."
        )

    def _selected_source_ids(self) -> list[str]:
        return [
            str(self.sources.topLevelItem(row).data(0, Qt.ItemDataRole.UserRole) or "")
            for row in range(self.sources.topLevelItemCount())
            if self.sources.topLevelItem(row).checkState(0) == Qt.CheckState.Checked
        ]

    def _start_scan(self) -> None:
        source_ids = self._selected_source_ids()
        if not source_ids:
            QMessageBox.warning(
                self, "No sources selected", "Select at least one source to scan."
            )
            return
        if self._thread is not None and self._thread.isRunning():
            return
        self._refresh_worker_recommendation(use_recommended=False)
        mode = str(self.mode.currentData())
        options = IntegrityScanOptions(
            mode=mode,
            verify_images=mode == "verified",
            retry_count=self.retry_count.value(),
            parallel_reads=self.parallel_reads.value(),
        )
        actor = self.reviewer.text().strip()
        self._active_scan_run_ids = self._scope_run_ids(source_ids)
        try:
            if self._active_scan_run_ids:
                self.project.set_run_module_status(
                    self._active_scan_run_ids,
                    "data_integrity",
                    "in_progress",
                    updated_by=actor,
                    reason=f"Started {options.mode} run-specific integrity scan",
                )
            else:
                self.project.set_module_status(
                    "data_integrity",
                    "in_progress",
                    updated_by=actor,
                    reason=f"Started {options.mode} source-integrity scan",
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not start scan", str(exc))
            return
        self.scan_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress_path.setText("Enumerating selected sources...")
        self._active_worker_count = 0
        self._desired_worker_count = self.parallel_reads.value()
        self._worker_limit = MutableWorkerLimit(
            self._desired_worker_count,
            maximum=self.parallel_reads.maximum(),
        )
        self.worker_activity.setText(
            f"Enumerating sources; {self._desired_worker_count} read workers requested."
        )
        self._thread = QThread(self)
        self._worker = IntegrityWorker(
            project=self.project,
            options=options,
            source_ids=source_ids,
            actor=actor,
            worker_limit=self._worker_limit,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._scan_progress)
        self._worker.worker_status.connect(self._worker_status_changed)
        self._worker.finished.connect(self._scan_finished)
        self._worker.failed.connect(self._scan_failed)
        self._worker.cancelled.connect(self._scan_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()
        self.status_changed.emit()

    def _cancel_scan(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self.progress_path.setText("Cancelling after the current file...")

    @Slot(int, int, str)
    def _scan_progress(self, current: int, total: int, path: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(current)
        self.progress_path.setText(path)

    @Slot(int, int)
    def _worker_status_changed(self, active: int, desired: int) -> None:
        self._active_worker_count = int(active)
        self._desired_worker_count = int(desired)
        self.worker_activity.setText(
            f"Running: {self._active_worker_count} active/in-flight read(s); "
            f"{self._desired_worker_count} requested. Changes take effect after an "
            "active file finishes."
        )

    @Slot(int)
    def _requested_worker_count_changed(self, value: int) -> None:
        self._desired_worker_count = int(value)
        if self._worker_limit is None:
            recommended = int(self._worker_recommendation.get("recommended") or 1)
            self.worker_activity.setText(
                f"Idle; the next scan will use {value} worker(s). Hardware-derived "
                f"recommendation: {recommended}."
            )
            return
        self._worker_limit.set(value)
        self.worker_activity.setText(
            f"Running: {self._active_worker_count} active/in-flight read(s); "
            f"{value} requested. The scheduler will adapt after an active file finishes."
        )

    @Slot(object)
    def _scan_finished(self, raw_report: object) -> None:
        report = dict(raw_report)
        actor = self.reviewer.text().strip()
        try:
            destination = self.store.save(report, actor=actor)
            summary = dict(report.get("summary") or {})
            errors = int(summary.get("error_count") or 0)
            if self._active_scan_run_ids:
                source_run = {
                    str(source.get("source_id") or ""): str(source.get("run_id") or "")
                    for source in self.project.source_manifest().get("sources") or []
                }
                error_runs = {
                    source_run.get(str(issue.get("source_id") or ""), "")
                    for issue in report.get("issues") or []
                    if str(issue.get("severity") or "") == "error"
                }
                blocked = [
                    run_id
                    for run_id in self._active_scan_run_ids
                    if run_id in error_runs
                ]
                clean = [
                    run_id
                    for run_id in self._active_scan_run_ids
                    if run_id not in error_runs
                ]
                if blocked:
                    self.project.set_run_module_status(
                        blocked,
                        "data_integrity",
                        "blocked",
                        updated_by=actor,
                        reason="Run-specific integrity scan found blocking errors.",
                    )
                if clean:
                    self.project.set_run_module_status(
                        clean,
                        "data_integrity",
                        "review_required",
                        updated_by=actor,
                        reason=(
                            "Run-specific integrity scan requires reviewer approval."
                        ),
                    )
            else:
                self.project.set_module_status(
                    "data_integrity",
                    "blocked" if errors else "review_required",
                    updated_by=actor,
                    reason=(
                        f"Integrity scan found {errors} blocking errors"
                        if errors
                        else "Integrity scan completed; reviewer approval required"
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            self._scan_failed(f"Could not publish integrity report: {exc}")
            return
        published_report = dict(report)
        published_report.pop("inventory", None)
        published_report["report_path"] = str(destination / "report.json")
        cached = [
            row
            for row in (self._report_cache or [])
            if str(row.get("scan_id") or "")
            != str(published_report.get("scan_id") or "")
        ]
        self._report_cache = [published_report, *cached]
        self.current_report = published_report
        self.progress.setValue(self.progress.maximum())
        self.progress_path.setText("Integrity report saved")
        self.refresh()
        self.status_changed.emit()

    @Slot(str)
    def _scan_failed(self, message: str) -> None:
        recovery_error = self._restore_after_unpublished_scan(
            f"Integrity scan failed before a report was published: {message}"
        )
        detail = message + (
            f"\n\nStatus recovery also failed: {recovery_error}"
            if recovery_error
            else ""
        )
        self.progress_path.setText("Scan failed; no report was published. " + message)
        QMessageBox.critical(self, "Integrity scan failed", detail)
        self.refresh()

    @Slot()
    def _scan_cancelled(self) -> None:
        recovery_error = self._restore_after_unpublished_scan(
            "Integrity scan was cancelled before a report was published."
        )
        self.progress_path.setText("Scan cancelled; no partial report was published")
        if recovery_error:
            QMessageBox.critical(
                self,
                "Could not restore integrity status",
                recovery_error,
            )
        self.refresh()

    def _restore_after_unpublished_scan(self, reason: str) -> str:
        """Return an interrupted scan to an actionable, auditable state."""

        actor = self.reviewer.text().strip()
        try:
            if self._active_scan_run_ids:
                self.project.set_run_module_status(
                    self._active_scan_run_ids,
                    "data_integrity",
                    "ready",
                    updated_by=actor,
                    reason=reason,
                )
                global_status = str(
                    self.project.module_states["data_integrity"].get("status") or ""
                )
                if global_status in {"approved", "complete"}:
                    self.project.set_module_status(
                        "data_integrity",
                        "ready",
                        updated_by=actor,
                        reason=reason,
                    )
            else:
                self.project.set_module_status(
                    "data_integrity",
                    "ready",
                    updated_by=actor,
                    reason=reason,
                )
        except Exception as exc:  # noqa: BLE001 - recovery must remain visible
            return f"{type(exc).__name__}: {exc}"
        return ""

    @Slot()
    def _thread_finished(self) -> None:
        thread = self._thread
        if thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            thread.wait()
        self.scan_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._worker = None
        self._thread = None
        self._worker_limit = None
        self._active_worker_count = 0
        self._desired_worker_count = self.parallel_reads.value()
        self._active_scan_run_ids = ()
        recommended = int(self._worker_recommendation.get("recommended") or 1)
        self.worker_activity.setText(
            f"Idle; the next scan will use {self.parallel_reads.value()} worker(s). "
            f"Hardware-derived recommendation: {recommended}."
        )
        if thread is not None:
            thread.deleteLater()

    def scan_is_running(self) -> bool:
        """Return whether an integrity scan is active for this project page."""

        return self._thread is not None and self._thread.isRunning()

    def shutdown_background_work(self) -> None:
        """Cancel and safely join the scanner before the application is destroyed."""

        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    def _report_selected(self) -> None:
        scan_id = str(self.reports.currentData() or "")
        self.current_report = next(
            (
                report
                for report in (self._report_cache or [])
                if str(report.get("scan_id") or "") == scan_id
            ),
            None,
        )
        self._populate_report()

    def _populate_report(self) -> None:
        report = self.current_report
        if not report:
            self.findings.setRowCount(0)
            self.report_summary.setText("No integrity report has been created.")
            for row in range(self.sources.topLevelItemCount()):
                item = self.sources.topLevelItem(row)
                item.setText(5, "Not scanned")
                for column in (6, 7, 8):
                    item.setText(column, "-")
            return
        summary = dict(report.get("summary") or {})
        self.report_summary.setText(
            f"{summary.get('file_count', 0)} files · {summary.get('image_count', 0)} images · "
            f"{summary.get('error_count', 0)} errors · {summary.get('warning_count', 0)} warnings · "
            f"{summary.get('read_retry_count', 0)} recovered reads"
        )
        source_rows = list(report.get("sources") or [])
        by_id = {str(row.get("source_id") or ""): row for row in source_rows}
        for row in range(self.sources.topLevelItemCount()):
            item = self.sources.topLevelItem(row)
            source = by_id.get(str(item.data(0, Qt.ItemDataRole.UserRole) or ""))
            if source:
                item.setText(
                    4, "Available" if source.get("available") else "Unavailable"
                )
                item.setText(5, str(source.get("file_count") or 0))
                item.setText(6, str(source.get("image_count") or 0))
                item.setText(7, self._format_size(source.get("total_size_bytes")))
                item.setText(8, str(source.get("issue_count") or 0))
            else:
                item.setText(5, "Not in report")
                for column in (6, 7, 8):
                    item.setText(column, "-")
        selected_run = str(self.run_filter.currentData() or "all")
        run_ids = sorted(
            {
                str(row.get("run_id") or "")
                for row in source_rows
                if str(row.get("run_id") or "")
            },
            key=str.casefold,
        )
        self.run_filter.blockSignals(True)
        self.run_filter.clear()
        self.run_filter.addItem("All runs", "all")
        for run_id in run_ids:
            self.run_filter.addItem(run_id.upper(), run_id)
        restored = self.run_filter.findData(selected_run)
        self.run_filter.setCurrentIndex(max(restored, 0))
        self.run_filter.blockSignals(False)
        self._populate_findings()

    def _populate_findings(self) -> None:
        issues = (self.current_report or {}).get("issues") or []
        selected_severity = str(self.issue_filter.currentData() or "all")
        selected_run = str(self.run_filter.currentData() or "all")
        source_by_id = {
            str(row.get("source_id") or ""): dict(row)
            for row in (self.current_report or {}).get("sources") or []
        }
        groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for issue in issues:
            severity = str(issue.get("severity") or "")
            if selected_severity != "all" and severity != selected_severity:
                continue
            source_id = str(issue.get("source_id") or "")
            source = source_by_id.get(source_id, {})
            run_id = str(
                source.get("run_id") or (issue.get("details") or {}).get("run_id") or ""
            )
            if selected_run != "all" and run_id != selected_run:
                continue
            code = str(issue.get("code") or "unknown_finding")
            source_kind = str(source.get("kind") or "")
            key = (severity, code, run_id, source_id)
            group = groups.get(key)
            if group is None:
                meaning, action = _guidance_for_finding(code, source_kind)
                group = {
                    "severity": severity,
                    "code": code,
                    "run_id": run_id,
                    "source_id": source_id,
                    "source_kind": source_kind,
                    "source_path": str(source.get("path") or ""),
                    "count": 0,
                    "path": str(issue.get("path") or source.get("path") or ""),
                    "message": str(issue.get("message") or ""),
                    "meaning": meaning,
                    "action": action,
                }
                groups[key] = group
            group["count"] = int(group["count"]) + 1

        severity_order = {"error": 0, "warning": 1, "info": 2}
        self._finding_groups = sorted(
            groups.values(),
            key=lambda group: (
                severity_order.get(str(group["severity"]), 3),
                str(group["run_id"]).casefold(),
                str(group["code"]),
                str(group["source_path"]).casefold(),
            ),
        )
        self.findings.setRowCount(len(self._finding_groups))
        color_by_severity = {
            "error": QColor("#e27979"),
            "warning": QColor("#e3b65c"),
            "info": QColor("#73b9e6"),
        }
        for row, group in enumerate(self._finding_groups):
            severity = str(group["severity"])
            values = (
                severity.upper(),
                str(group["code"]).replace("_", " ").capitalize(),
                str(group["run_id"]).upper() or "Series",
                _source_kind_label(group["source_kind"]),
                str(group["count"]),
                str(group["path"]),
                str(group["action"]),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, group)
                item.setToolTip(value)
                if column == 0:
                    item.setForeground(
                        color_by_severity.get(severity, QColor("#d7dce0"))
                    )
                self.findings.setItem(row, column, item)
        total = len(issues)
        shown = sum(int(group["count"]) for group in self._finding_groups)
        self.finding_guidance.setText(
            "Errors block approval. Warnings require review but do not block approval; "
            "information records context for later workflow steps. "
            f"Showing {len(self._finding_groups)} grouped row(s) representing "
            f"{shown} of {total} report entries. Select a row for details."
        )
        self.open_location.setEnabled(False)
        self.review_source.setEnabled(False)

    def _open_selected_location(self) -> None:
        group = self._selected_finding_group()
        if not group:
            return
        path = Path(str(group.get("path") or group.get("source_path") or ""))
        target = path if path.is_dir() else path.parent
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _selected_finding_group(self) -> dict[str, Any] | None:
        row = self.findings.currentRow()
        item = self.findings.item(row, 0) if row >= 0 else None
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return dict(value) if isinstance(value, dict) else None

    def _finding_selected(self) -> None:
        group = self._selected_finding_group()
        self.open_location.setEnabled(bool(group and group.get("path")))
        self.review_source.setEnabled(bool(group and group.get("run_id")))
        if not group:
            return
        run_label = str(group.get("run_id") or "Series").upper()
        entry_count = int(group.get("count") or 0)
        entry_label = "entry" if entry_count == 1 else "entries"
        self.finding_guidance.setText(
            f"<b>{run_label} · {_source_kind_label(group.get('source_kind'))} · "
            f"{entry_count} report {entry_label}</b><br>"
            f"{group.get('meaning')}<br><b>Next action:</b> {group.get('action')}<br>"
            f"<b>Example:</b> {group.get('message')}"
        )

    def _request_source_review(self) -> None:
        group = self._selected_finding_group()
        run_id = str((group or {}).get("run_id") or "")
        if run_id:
            self.review_setup_requested.emit(run_id)

    def _set_status(self, status: str) -> None:
        latest_report = (self._report_cache or [None])[0]
        if status in {"approved", "complete"}:
            summary = dict((latest_report or {}).get("summary") or {})
            if not latest_report:
                QMessageBox.warning(
                    self,
                    "Integrity report required",
                    "Run and review a source-integrity scan before approving this quality gate.",
                )
                return
            if int(summary.get("error_count") or 0) > 0:
                QMessageBox.warning(
                    self,
                    "Blocking integrity findings",
                    "Resolve or document every error and create a new clean report before approval.",
                )
                return
            if not self._report_matches_current_sources(latest_report):
                QMessageBox.warning(
                    self,
                    "Integrity report is outdated",
                    "The active source setup changed after this report was created. "
                    "Create and review a new report before approval.",
                )
                return
        report_scope = self._report_run_ids(latest_report)
        if report_scope:
            try:
                self.project.set_run_module_status(
                    report_scope,
                    "data_integrity",
                    status,
                    updated_by=self.reviewer.text().strip(),
                    reason="Run-specific integrity report reviewed in the GUI.",
                )
            except Exception as exc:  # noqa: BLE001 - visible GUI boundary
                QMessageBox.critical(self, "Could not update run review", str(exc))
                return
            self.refresh()
            self.status_changed.emit()
            return
        target_status = "complete" if status == "approved" else status
        super()._set_status(target_status)
        if target_status in {"approved", "complete"}:
            run_ids = [
                str(row.get("run_id") or "")
                for row in self.project.manifest.get("runs") or []
                if isinstance(row, dict) and str(row.get("run_id") or "")
            ]
            if run_ids:
                self.project.set_run_module_status(
                    run_ids,
                    "data_integrity",
                    target_status,
                    updated_by=self.reviewer.text().strip(),
                    reason="Covered by the approved series-level integrity report.",
                )
