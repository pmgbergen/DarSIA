"""Qt panel for advisory frame-quality screening and blacklist review."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.frame_quality import (
    QUALITY_STATUSES,
    FrameQualityStore,
    default_frame_quality_config,
    scan_frame_quality,
)
from darsia.fluidflower.concurrency import MutableWorkerLimit
from darsia.fluidflower.image_correction import CorrectionRevisionStore
from darsia.fluidflower.project import FluidFlowerProject

STATUS_LABELS = {
    "ok": "OK",
    "corrected": "Corrected successfully",
    "suspicious": "Review required",
    "exclusion_suggested": "Exclusion suggested",
    "analysis_failed": "Analysis failed",
    "excluded": "Confirmed excluded",
}
STATUS_COLORS = {
    "ok": "#52b788",
    "corrected": "#4cc9f0",
    "suspicious": "#f4b942",
    "exclusion_suggested": "#ef6a6a",
    "analysis_failed": "#ef6a6a",
    "excluded": "#d66efd",
}


class FrameQualityTimeline(QWidget):
    """Compact clickable timeline for one filtered set of frames."""

    frame_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(48)
        self.setMaximumHeight(48)
        self.setToolTip(
            "Each marker is one frame. Green is accepted automatically, cyan was "
            "corrected, amber requires review, red is an exclusion suggestion, and "
            "purple is manually excluded. Click a marker to select its table row."
        )
        self._frames: list[dict[str, Any]] = []
        self._selected = ""

    def set_frames(self, frames: Sequence[Mapping[str, Any]]) -> None:
        self._frames = [deepcopy(dict(row)) for row in frames]
        if self._selected not in {
            str(row.get("frame_id") or "") for row in self._frames
        }:
            self._selected = ""
        self.update()

    def set_selected(self, frame_id: str) -> None:
        self._selected = str(frame_id)
        self.update()

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        left, right = 8, max(9, self.width() - 8)
        baseline = self.height() // 2
        painter.setPen(QPen(QColor("#59636d"), 1))
        painter.drawLine(left, baseline, right, baseline)
        count = len(self._frames)
        if not count:
            painter.setPen(QColor("#9aa4ad"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No quality assessment"
            )
            return
        for index, row in enumerate(self._frames):
            x = (
                left
                if count == 1
                else round(left + index * (right - left) / float(count - 1))
            )
            status = str(
                row.get("effective_status") or row.get("automatic_status") or ""
            )
            painter.setPen(QPen(QColor(STATUS_COLORS.get(status, "#9aa4ad")), 3))
            painter.drawLine(x, baseline - 9, x, baseline + 9)
            if str(row.get("frame_id") or "") == self._selected:
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.drawRect(x - 3, baseline - 13, 6, 26)

    def mousePressEvent(self, event: Any) -> None:
        if not self._frames:
            return
        left, right = 8, max(9, self.width() - 8)
        relative = min(1.0, max(0.0, (event.position().x() - left) / (right - left)))
        index = round(relative * (len(self._frames) - 1))
        frame_id = str(self._frames[index].get("frame_id") or "")
        self.set_selected(frame_id)
        self.frame_selected.emit(frame_id)


class FrameQualityWorker(QObject):
    progress = Signal(int, int)
    worker_status = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        frames: Sequence[Mapping[str, Any]],
        correction: Mapping[str, Any],
        *,
        worker_limit: MutableWorkerLimit,
    ) -> None:
        super().__init__()
        self.frames = [deepcopy(dict(row)) for row in frames]
        self.correction = deepcopy(dict(correction))
        self.worker_limit = worker_limit
        self.stop_event = Event()

    def request_cancel(self) -> None:
        self.stop_event.set()

    @Slot()
    def run(self) -> None:
        try:
            report = scan_frame_quality(
                self.frames,
                self.correction,
                config=default_frame_quality_config(),
                workers=self.worker_limit.get(),
                max_workers=24,
                worker_limit=self.worker_limit.get,
                worker_status=self.worker_status.emit,
                cancelled=self.stop_event,
                progress=self.progress.emit,
            )
        except InterruptedError:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(report)


class FrameQualityPanel(QWidget):
    """Reusable quality-review panel embedded in Image correction."""

    inspect_requested = Signal(str)

    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__()
        self.project = project
        self.store = FrameQualityStore(project)
        self.correction_store = CorrectionRevisionStore(project)
        self._thread: QThread | None = None
        self._worker: FrameQualityWorker | None = None
        self._worker_limit: MutableWorkerLimit | None = None
        self._active_worker_count = 0
        self._desired_worker_count = 0
        self._rows: list[dict[str, Any]] = []
        self._visible_rows: list[dict[str, Any]] = []
        self._assessment_is_current = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        title = QLabel("Frame quality and exclusion review")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        explanation = QLabel(
            "Screen the inventoried CO2 image sequence for abrupt illumination changes, "
            "one-sided lamp failures, transient camera obstructions, severe blur, and "
            "residual problems after correction. Suggestions never exclude a frame "
            "automatically. Confirmed decisions are saved immediately and generate a "
            "separate blacklist without modifying source images or protocol files."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addWidget(explanation)

        controls = QGridLayout()
        self.workers = QSpinBox()
        self.workers.setRange(1, 24)
        self.workers.setValue(max(1, min(2, os.cpu_count() or 1)))
        self.workers.setToolTip(
            "Frames are read in parallel. This value remains adjustable during a scan. "
            "An increase is applied after the next active frame finishes; a decrease "
            "lets already active reads finish and then reduces concurrency."
        )
        self.workers.valueChanged.connect(self._requested_worker_count_changed)
        self.worker_activity = QLabel(
            f"Idle; the next scan will start with {self.workers.value()} worker(s)."
        )
        self.worker_activity.setObjectName("secondaryText")
        self.scan_button = QPushButton("Screen inventoried frames")
        self.scan_button.clicked.connect(self._start_scan)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_scan)
        self.run_filter = QComboBox()
        self.run_filter.addItem("All experiments", "")
        self.run_filter.currentIndexChanged.connect(self._apply_filters)
        self.status_filter = QComboBox()
        self.status_filter.addItem("All quality states", "")
        for status in (*QUALITY_STATUSES, "excluded"):
            self.status_filter.addItem(STATUS_LABELS[status], status)
        self.status_filter.currentIndexChanged.connect(self._apply_filters)
        controls.addWidget(QLabel("Desired read workers"), 0, 0)
        controls.addWidget(self.workers, 0, 1)
        controls.addWidget(self.scan_button, 0, 2)
        controls.addWidget(self.cancel_button, 0, 3)
        controls.addWidget(QLabel("Experiment"), 0, 4)
        controls.addWidget(self.run_filter, 0, 5)
        controls.addWidget(QLabel("Show"), 0, 6)
        controls.addWidget(self.status_filter, 0, 7)
        controls.addWidget(self.worker_activity, 1, 0, 1, 8)
        layout.addLayout(controls)

        self.summary = QLabel("No frame-quality assessment has been saved.")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("secondaryText")
        layout.addWidget(self.summary)
        self.timeline = FrameQualityTimeline()
        self.timeline.frame_selected.connect(self._select_frame)
        layout.addWidget(self.timeline)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Experiment",
                "Frame",
                "Automatic assessment",
                "Residual light score",
                "Transient score",
                "Detected reasons",
                "Manual decision",
                "Source image",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda *_args: self._inspect_selected())
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table)

        decisions = QGridLayout()
        self.reviewer = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.reason = QLineEdit()
        self.reason.setPlaceholderText(
            "Reason, for example: left lamp off or person in front of camera"
        )
        self.inspect_button = QPushButton("Inspect original and corrected")
        self.inspect_button.clicked.connect(self._inspect_selected)
        self.keep_button = QPushButton("Confirm usable")
        self.keep_button.clicked.connect(lambda: self._save_decision("keep"))
        self.exclude_button = QPushButton("Exclude from analysis")
        self.exclude_button.clicked.connect(lambda: self._save_decision("exclude"))
        self.clear_button = QPushButton("Clear manual decision")
        self.clear_button.clicked.connect(lambda: self._save_decision("clear"))
        decisions.addWidget(QLabel("Reviewer"), 0, 0)
        decisions.addWidget(self.reviewer, 0, 1)
        decisions.addWidget(QLabel("Decision note"), 0, 2)
        decisions.addWidget(self.reason, 0, 3, 1, 3)
        decisions.addWidget(self.inspect_button, 1, 1)
        decisions.addWidget(self.keep_button, 1, 2)
        decisions.addWidget(self.exclude_button, 1, 3)
        decisions.addWidget(self.clear_button, 1, 4)
        layout.addLayout(decisions)
        self.refresh()

    def refresh(self) -> None:
        assessment = self.store.current_assessment()
        self._rows = self.store.effective_frames(assessment) if assessment else []
        active_correction = self.correction_store.current()
        self._assessment_is_current = bool(
            assessment
            and str(assessment.get("correction_dependency_sha256") or "")
            == CorrectionRevisionStore.dependency_sha256(active_correction)
        )
        current_run = str(self.run_filter.currentData() or "")
        run_ids = sorted({str(row.get("run_id") or "") for row in self._rows})
        self.run_filter.blockSignals(True)
        self.run_filter.clear()
        self.run_filter.addItem("All experiments", "")
        for run_id in run_ids:
            self.run_filter.addItem(run_id.upper(), run_id)
        index = self.run_filter.findData(current_run)
        self.run_filter.setCurrentIndex(max(0, index))
        self.run_filter.blockSignals(False)
        if assessment:
            counts = {
                status: sum(
                    str(row.get("effective_status") or "") == status
                    for row in self._rows
                )
                for status in (*QUALITY_STATUSES, "excluded")
            }
            blacklist = self.store.active_blacklist()
            stale_notice = (
                "OUTDATED: the active correction revision has changed. Run the quality "
                "screen again before making decisions. "
                if not self._assessment_is_current
                else ""
            )
            self.summary.setText(
                stale_notice
                + (
                    f"Assessment {str(assessment.get('assessment_id') or '')[:8]}: "
                    f"{len(self._rows)} frames across {len(run_ids)} experiment(s). "
                    f"Corrected: {counts['corrected']}; "
                    f"review: {counts['suspicious']}; exclusion suggested: "
                    f"{counts['exclusion_suggested']}; confirmed excluded: "
                    f"{counts['excluded']}. Active blacklist: "
                    f"{self.store.blacklist_csv_path} "
                    f"({int(blacklist.get('entry_count') or 0)} frame(s))."
                )
            )
        else:
            self.summary.setText(
                "No frame-quality assessment has been saved. Save a correction revision "
                "with an immutable batch inventory, then screen its frames."
            )
        self._apply_filters()

    def _start_scan(self) -> None:
        if self._thread is not None:
            return
        actor = self.reviewer.text().strip()
        if not actor:
            QMessageBox.warning(
                self,
                "Reviewer name required",
                "Enter the name of the person starting the quality assessment.",
            )
            return
        correction = self.correction_store.current()
        frames = [
            dict(row)
            for row in (correction.get("batch") or {}).get("frames") or []
            if isinstance(row, Mapping)
        ]
        if not correction.get("revision_id") or not frames:
            QMessageBox.warning(
                self,
                "Saved inventory required",
                "Save an image-correction revision with an inventoried batch before "
                "screening frame quality.",
            )
            return
        self.scan_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.summary.setText(
            f"Screening 0 of {len(frames)} frames. No existing assessment is changed "
            "until the complete result has been saved."
        )
        self._active_worker_count = 0
        self._desired_worker_count = self.workers.value()
        self._worker_limit = MutableWorkerLimit(
            self._desired_worker_count,
            maximum=self.workers.maximum(),
        )
        thread = QThread(self)
        worker = FrameQualityWorker(
            frames,
            correction,
            worker_limit=self._worker_limit,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._scan_progress)
        worker.worker_status.connect(self._worker_status_changed)
        worker.finished.connect(self._scan_finished)
        worker.failed.connect(self._scan_failed)
        worker.cancelled.connect(self._scan_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _cancel_scan(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.summary.setText(
                "Stopping after active frame reads finish. The previous assessment "
                "remains unchanged."
            )

    @Slot(int, int)
    def _scan_progress(self, completed: int, total: int) -> None:
        self.summary.setText(
            f"Screening {completed} of {total} frames using the active correction "
            f"revision; {self._active_worker_count} active/in flight, "
            f"{self._desired_worker_count} requested."
        )

    @Slot(int, int)
    def _worker_status_changed(self, active: int, desired: int) -> None:
        self._active_worker_count = int(active)
        self._desired_worker_count = int(desired)
        self.worker_activity.setText(
            f"Running: {self._active_worker_count} active/in-flight worker(s); "
            f"{self._desired_worker_count} requested. Changes apply after the next "
            "active frame finishes."
        )

    @Slot(int)
    def _requested_worker_count_changed(self, value: int) -> None:
        self._desired_worker_count = int(value)
        if self._worker_limit is None:
            self.worker_activity.setText(
                f"Idle; the next scan will start with {value} worker(s)."
            )
            return
        self._worker_limit.set(value)
        self.worker_activity.setText(
            f"Running: {self._active_worker_count} active/in-flight worker(s); "
            f"{value} requested. The scheduler will adapt after the next active "
            "frame finishes."
        )

    @Slot(object)
    def _scan_finished(self, raw_report: object) -> None:
        try:
            self.store.save_assessment(
                dict(raw_report),
                actor=self.reviewer.text().strip(),
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            self._scan_failed(f"{type(exc).__name__}: {exc}")
            return
        self.refresh()

    @Slot(str)
    def _scan_failed(self, message: str) -> None:
        self.summary.setText(
            "Frame-quality screening failed. The previous assessment remains unchanged."
        )
        QMessageBox.critical(self, "Frame-quality screening failed", message)

    @Slot()
    def _scan_cancelled(self) -> None:
        self.summary.setText(
            "Frame-quality screening was cancelled. The previous assessment remains unchanged."
        )

    @Slot()
    def _thread_finished(self) -> None:
        if self._thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            self._thread.wait()
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._worker_limit = None
        self._active_worker_count = 0
        self._desired_worker_count = self.workers.value()
        self.scan_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker_activity.setText(
            f"Idle; the next scan will start with {self.workers.value()} worker(s)."
        )

    def closeEvent(self, event: Any) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)

    def _apply_filters(self, _value: Any = None) -> None:
        run_id = str(self.run_filter.currentData() or "")
        status = str(self.status_filter.currentData() or "")
        self._visible_rows = [
            row
            for row in self._rows
            if (not run_id or str(row.get("run_id") or "") == run_id)
            and (not status or str(row.get("effective_status") or "") == status)
        ]
        self.table.setRowCount(len(self._visible_rows))
        for table_row, row in enumerate(self._visible_rows):
            effective_status = str(
                row.get("effective_status") or row.get("automatic_status") or ""
            )
            corrected_metrics = dict(row.get("corrected_metrics") or {})
            source_path = str(row.get("source_path") or "")
            values = (
                str(row.get("run_id") or "").upper(),
                f"{int(row.get('sequence_index') or 0) + 1}: {Path(source_path).name}",
                STATUS_LABELS.get(
                    str(row.get("automatic_status") or ""),
                    str(row.get("automatic_status") or ""),
                ),
                f"{float(corrected_metrics.get('spatial_log') or 0.0):.3f}",
                f"{float(corrected_metrics.get('transient_spatial_log') or 0.0):.3f}",
                ", ".join(
                    str(value).replace("_", " ")
                    for value in row.get("automatic_flags") or []
                ),
                str(row.get("manual_decision") or ""),
                source_path,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(
                        Qt.ItemDataRole.UserRole, str(row.get("frame_id") or "")
                    )
                if column == 2:
                    item.setForeground(
                        QColor(STATUS_COLORS.get(effective_status, "#d7dce0"))
                    )
                self.table.setItem(table_row, column, item)
        self.timeline.set_frames(self._visible_rows)
        self._update_decision_actions()

    def _selected_frame_id(self) -> str:
        row = self.table.currentRow()
        if row < 0:
            return ""
        item = self.table.item(row, 0)
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""

    def _selected_row(self) -> dict[str, Any] | None:
        frame_id = self._selected_frame_id()
        return next(
            (
                row
                for row in self._visible_rows
                if str(row.get("frame_id") or "") == frame_id
            ),
            None,
        )

    def _selection_changed(self) -> None:
        self.timeline.set_selected(self._selected_frame_id())
        self._update_decision_actions()

    def _update_decision_actions(self) -> None:
        selected = bool(self._selected_frame_id())
        self.inspect_button.setEnabled(selected)
        enabled = selected and self._assessment_is_current
        self.keep_button.setEnabled(enabled)
        self.exclude_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)

    def _select_frame(self, frame_id: str) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and str(item.data(Qt.ItemDataRole.UserRole) or "") == frame_id:
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                break

    def _inspect_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        self.inspect_requested.emit(str(row.get("source_path") or ""))

    def _save_decision(self, decision: str) -> None:
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "Select a frame", "Select one frame first.")
            return
        actor = self.reviewer.text().strip()
        if not actor:
            QMessageBox.warning(
                self,
                "Reviewer name required",
                "Enter the name of the person making the decision.",
            )
            return
        reason = self.reason.text().strip()
        if decision == "exclude" and not reason:
            QMessageBox.warning(
                self,
                "Decision note required",
                "Record why this frame must be excluded from scientific analysis.",
            )
            return
        frame_id = str(row.get("frame_id") or "")
        try:
            self.store.set_decision(
                frame_id,
                decision,
                actor=actor,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(self, "Could not save frame decision", str(exc))
            return
        self.reason.clear()
        self.refresh()
        self._select_frame(frame_id)


__all__ = ["FrameQualityPanel", "FrameQualityTimeline", "FrameQualityWorker"]
