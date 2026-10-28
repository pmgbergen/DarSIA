"""Qt temporal-synchronization workflow page."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from darsia.fluidflower.data_integrity import IntegrityReportStore
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.timeline import TimelineStore
from darsia.fluidflower.ui.common import (
    DIRECTORY_DIALOG_OPTIONS,
    FILE_DIALOG_OPTIONS,
    ModulePage,
    _number,
    json_pretty,
)


TIMELINE_MAX_ZOOM = 1000.0
TIMELINE_ZOOM_STEPS_PER_DECADE = 100


def timeline_zoom_factor(slider_value: int) -> float:
    """Map the compact slider range logarithmically from 1x to 1000x."""

    return min(
        TIMELINE_MAX_ZOOM,
        10.0 ** (max(0, int(slider_value)) / TIMELINE_ZOOM_STEPS_PER_DECADE),
    )


class RegisteredProtocolImportWorker(QObject):
    """Read registered protocol workbooks without blocking the Qt event loop."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, project_root: Path, actor: str) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.actor = actor

    @Slot()
    def run(self) -> None:
        try:
            project = FluidFlowerProject.load(self.project_root)
            result = TimelineStore(project).import_registered_protocols(actor=self.actor)
            result["camera_sequence_endpoints"] = (
                IntegrityReportStore(project).camera_sequence_endpoints()
            )
        except Exception as exc:  # noqa: BLE001 - visible worker boundary
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class TimelinePlot(QWidget):
    """Per-experiment river view of aligned acquisition and protocol streams."""

    STREAMS = (
        ("camera", "Camera", "#63b3d8"),
        ("injection", "CO2 injection / MFC", "#d98282"),
        ("overflow", "Overflow", "#d9ae63"),
        ("chemistry", "pH / chemistry", "#70c697"),
        ("operations", "Other events", "#b993d6"),
    )

    event_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(260)
        self.setToolTip(
            "Each horizontal stream represents one data domain for the selected "
            "experiment. Markers are declared or linked events; thicker sections "
            "connect explicit start and end events. Click a marker for provenance."
        )
        self.events: list[dict[str, Any]] = []
        self.points: list[tuple[QPointF, str]] = []
        self.selected_event_id = ""

    def event_x(self, event_id: str) -> float | None:
        """Return the current canvas x coordinate for an event marker."""

        available = [event for event in self.events if event.get("time_h") is not None]
        selected = next(
            (event for event in available if str(event.get("event_id")) == event_id),
            None,
        )
        if selected is None:
            return None
        times = [float(event["time_h"]) for event in available]
        minimum = min(times)
        maximum = max(times)
        if np.isclose(minimum, maximum):
            minimum -= 0.5
            maximum += 0.5
        left, right = 175.0, 22.0
        width = max(1.0, self.width() - left - right)
        return (
            left
            + (float(selected["time_h"]) - minimum) / (maximum - minimum) * width
        )

    @staticmethod
    def stream_key(event: dict[str, Any]) -> str:
        """Map protocol source names and event types onto stable scientific lanes."""

        source = str(event.get("source") or "").casefold()
        event_type = str(event.get("event_type") or "").casefold()
        token = f"{source} {event_type}"
        if any(value in token for value in ("camera", "image", "frame")):
            return "camera"
        if any(value in token for value in ("injection", "injector", "mfc")):
            return "injection"
        if any(value in token for value in ("overflow", "scale", "weight")):
            return "overflow"
        if any(
            value in token
            for value in ("ph", "chem", "indicator", "buffer", "titration")
        ):
            return "chemistry"
        return "operations"

    def set_events(
        self,
        events: list[dict[str, Any]],
        selected_event_id: str = "",
    ) -> None:
        self.events = list(events)
        self.selected_event_id = selected_event_id
        self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101316"))
        self.points = []
        available = [event for event in self.events if event.get("time_h") is not None]
        if not available:
            painter.setPen(QColor("#93a0a9"))
            painter.drawText(
                18,
                32,
                "No timeline events were read from the registered protocol for this experiment.",
            )
            return
        stream_definitions = {
            key: (label, color) for key, label, color in self.STREAMS
        }
        lanes = [
            key
            for key, _label, _color in self.STREAMS
            if any(self.stream_key(event) == key for event in available)
        ]
        times = [float(event["time_h"]) for event in available]
        minimum = min(times)
        maximum = max(times)
        if np.isclose(minimum, maximum):
            minimum -= 0.5
            maximum += 0.5
        left, right, top, bottom = 175.0, 22.0, 18.0, 38.0
        width = max(1.0, self.width() - left - right)
        lane_height = max(22.0, (self.height() - top - bottom) / max(1, len(lanes)))

        def x_for_time(value: float) -> float:
            return left + (value - minimum) / (maximum - minimum) * width

        painter.setPen(QPen(QColor("#4a555d"), 1.0))
        axis_y = self.height() - bottom
        painter.drawLine(int(left), int(axis_y), int(left + width), int(axis_y))
        tick_count = max(6, int(width // 180) + 1)
        exposed = event.rect()
        first_tick = max(
            0,
            int(math.floor((exposed.left() - left) / width * (tick_count - 1)))
            - 1,
        )
        last_tick = min(
            tick_count - 1,
            int(math.ceil((exposed.right() - left) / width * (tick_count - 1)))
            + 1,
        )
        for index in range(first_tick, last_tick + 1):
            fraction = index / float(tick_count - 1)
            x = left + fraction * width
            time_h = minimum + fraction * (maximum - minimum)
            painter.drawLine(int(x), int(axis_y), int(x), int(axis_y + 5))
            painter.setPen(QColor("#93a0a9"))
            painter.drawText(int(x - 22), int(axis_y + 22), f"{time_h:.2f} h")
            painter.setPen(QPen(QColor("#4a555d"), 1.0))

        if minimum <= 0.0 <= maximum:
            zero_x = x_for_time(0.0)
            zero_pen = QPen(QColor("#77838b"), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(zero_pen)
            painter.drawLine(int(zero_x), int(top), int(zero_x), int(axis_y))

        for lane_index, stream in enumerate(lanes):
            y = top + (lane_index + 0.5) * lane_height
            label, color_hex = stream_definitions[stream]
            color = QColor(color_hex)
            if lane_index % 2 == 0:
                painter.fillRect(
                    QRectF(left, y - lane_height * 0.42, width, lane_height * 0.84),
                    QColor("#151a1e"),
                )
            painter.setPen(QColor("#c7d0d6"))
            painter.drawText(12, int(y + 5), label)
            painter.setPen(QPen(QColor("#293138"), 1.0))
            painter.drawLine(int(left), int(y), int(left + width), int(y))

            stream_events = sorted(
                (
                    event
                    for event in available
                    if self.stream_key(event) == stream
                ),
                key=lambda event: float(event["time_h"]),
            )

            selected_in_stream = next(
                (
                    event
                    for event in stream_events
                    if str(event.get("event_id")) == self.selected_event_id
                ),
                None,
            )
            if selected_in_stream is not None:
                selected_x = x_for_time(float(selected_in_stream["time_h"]))
                guide = QColor("#f4f8fa")
                guide.setAlpha(175)
                painter.setPen(QPen(guide, 1.5, Qt.PenStyle.DashLine))
                painter.drawLine(
                    QPointF(selected_x, top),
                    QPointF(selected_x, axis_y),
                )
            if len(stream_events) > 1:
                flow_color = QColor(color)
                flow_color.setAlpha(105)
                painter.setPen(
                    QPen(
                        flow_color,
                        max(4.0, min(10.0, lane_height * 0.22)),
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawLine(
                    QPointF(x_for_time(float(stream_events[0]["time_h"])), y),
                    QPointF(x_for_time(float(stream_events[-1]["time_h"])), y),
                )

            starts: dict[str, list[dict[str, Any]]] = {}
            ends: dict[str, list[dict[str, Any]]] = {}
            for event in stream_events:
                event_type = str(event.get("event_type") or "")
                if event_type.endswith("_start"):
                    starts.setdefault(event_type.removesuffix("_start"), []).append(event)
                elif event_type.endswith("_end"):
                    ends.setdefault(event_type.removesuffix("_end"), []).append(event)
            interval_color = QColor(color)
            interval_color.setAlpha(190)
            painter.setPen(
                QPen(
                    interval_color,
                    max(8.0, min(15.0, lane_height * 0.38)),
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            for interval_type, start_events in starts.items():
                available_ends = list(ends.get(interval_type) or [])
                for start_event in start_events:
                    start_time = float(start_event["time_h"])
                    end_event = next(
                        (
                            event
                            for event in available_ends
                            if float(event["time_h"]) >= start_time
                        ),
                        None,
                    )
                    if end_event is None:
                        continue
                    available_ends.remove(end_event)
                    painter.drawLine(
                        QPointF(x_for_time(start_time), y),
                        QPointF(x_for_time(float(end_event["time_h"])), y),
                    )

            for event in stream_events:
                x = x_for_time(float(event["time_h"]))
                event_id = str(event["event_id"])
                selected = event_id == self.selected_event_id
                radius = 7.0 if selected else 4.0
                if selected:
                    painter.setPen(QPen(QColor("#ffffff"), 3.0))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(QPointF(x, y), 11.0, 11.0)
                painter.setPen(QPen(QColor("#101316") if selected else color, 2.0))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(x, y), radius, radius)
                self.points.append((QPointF(x, y), event_id))

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        point = event.position()
        if not self.points:
            return
        nearest, event_id = min(
            self.points,
            key=lambda row: (
                (row[0].x() - point.x()) ** 2 + (row[0].y() - point.y()) ** 2
            ),
        )
        distance = (nearest.x() - point.x()) ** 2 + (nearest.y() - point.y()) ** 2
        if distance <= 144.0:
            self.event_selected.emit(event_id)


class TimelineScrollArea(QScrollArea):
    """Horizontally scrollable host that keeps timeline zoom centred."""

    zoom_step_requested = Signal(int)

    def __init__(self, plot: TimelinePlot) -> None:
        super().__init__()
        self.plot = plot
        self.zoom_factor = 1.0
        self.setWidget(plot)
        self.setWidgetResizable(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scrollbar_height = self.horizontalScrollBar().sizeHint().height()
        self.setFixedHeight(plot.height() + scrollbar_height + 2 * self.frameWidth())
        self.setToolTip(
            "Use Timeline scale to reveal more detail, then drag the horizontal "
            "scrollbar or use the mouse wheel over the timeline. Ctrl+wheel zooms."
        )

    def set_zoom_factor(self, factor: float) -> None:
        bar = self.horizontalScrollBar()
        old_maximum = max(1, bar.maximum())
        old_centre = (bar.value() + self.viewport().width() / 2.0) / (
            old_maximum + self.viewport().width()
        )
        self.zoom_factor = max(1.0, min(TIMELINE_MAX_ZOOM, float(factor)))
        self._resize_plot()

        def restore_centre() -> None:
            new_bar = self.horizontalScrollBar()
            target = old_centre * (new_bar.maximum() + self.viewport().width())
            new_bar.setValue(round(target - self.viewport().width() / 2.0))

        QTimer.singleShot(0, restore_centre)

    def centre_on_event(self, event_id: str) -> None:
        x = self.plot.event_x(event_id)
        if x is None:
            return
        self.horizontalScrollBar().setValue(round(x - self.viewport().width() / 2.0))

    def _resize_plot(self) -> None:
        viewport_width = max(1, self.viewport().width())
        self.plot.resize(
            max(viewport_width, round(viewport_width * self.zoom_factor)),
            self.plot.height(),
        )

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._resize_plot()

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_step_requested.emit(1 if delta > 0 else -1)
            event.accept()
            return
        bar = self.horizontalScrollBar()
        if bar.maximum() > 0 and delta:
            bar.setValue(bar.value() - delta)
            event.accept()
            return
        super().wheelEvent(event)


class TimelineEventDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        *,
        title: str,
        event: dict[str, Any] | None = None,
        actor: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        layout = QFormLayout(self)
        self.run_id = QLineEdit(str((event or {}).get("run_id") or ""))
        self.source = QLineEdit(str((event or {}).get("source") or "manual"))
        self.event_type = QLineEdit(str((event or {}).get("event_type") or "event"))
        self.label = QLineEdit(str((event or {}).get("label") or ""))
        self.time_known = QCheckBox("Time is known")
        self.time_known.setChecked((event or {}).get("time_h") is not None)
        self.time_h = QDoubleSpinBox()
        self.time_h.setRange(-1000000.0, 1000000.0)
        self.time_h.setDecimals(6)
        self.time_h.setValue(float((event or {}).get("time_h") or 0.0))
        self.time_known.toggled.connect(self.time_h.setEnabled)
        self.time_h.setEnabled(self.time_known.isChecked())
        time_row = QWidget()
        time_layout = QHBoxLayout(time_row)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.addWidget(self.time_known)
        time_layout.addWidget(self.time_h, 1)
        self.value_known = QCheckBox("Value is known")
        self.value_known.setChecked((event or {}).get("value") is not None)
        self.value = QDoubleSpinBox()
        self.value.setRange(-1e12, 1e12)
        self.value.setDecimals(6)
        self.value.setValue(float((event or {}).get("value") or 0.0))
        self.value_known.toggled.connect(self.value.setEnabled)
        self.value.setEnabled(self.value_known.isChecked())
        value_row = QWidget()
        value_layout = QHBoxLayout(value_row)
        value_layout.setContentsMargins(0, 0, 0, 0)
        value_layout.addWidget(self.value_known)
        value_layout.addWidget(self.value, 1)
        self.note = QLineEdit(str((event or {}).get("note") or ""))
        self.actor = QLineEdit(actor)
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("Why is this local value required?")
        layout.addRow("Run", self.run_id)
        layout.addRow("Source lane", self.source)
        layout.addRow("Event type", self.event_type)
        layout.addRow("Label", self.label)
        layout.addRow("Experiment time [h]", time_row)
        layout.addRow("Measured value", value_row)
        layout.addRow("Note", self.note)
        layout.addRow("Reviewer", self.actor)
        layout.addRow("Reason", self.reason)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id.text().strip(),
            "source": self.source.text().strip(),
            "event_type": self.event_type.text().strip(),
            "label": self.label.text().strip(),
            "time_h": self.time_h.value() if self.time_known.isChecked() else None,
            "value": self.value.value() if self.value_known.isChecked() else None,
            "note": self.note.text().strip(),
            "actor": self.actor.text().strip(),
            "reason": self.reason.text().strip(),
        }


class TimelineImportDialog(QDialog):
    def __init__(self, parent: QWidget, path: Path, actor: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import timeline table")
        layout = QFormLayout(self)
        path_label = QLabel(str(path))
        path_label.setWordWrap(True)
        path_label.setObjectName("pathPreview")
        self.source = QLineEdit(path.stem)
        self.run_id = QLineEdit()
        self.actor = QLineEdit(actor)
        layout.addRow("File", path_label)
        layout.addRow("Source lane", self.source)
        layout.addRow("Run (optional)", self.run_id)
        layout.addRow("Imported by", self.actor)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)


class TimelinePage(ModulePage):
    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__(project, "temporal_sync")
        self.store = TimelineStore(project)
        self.selected_event_id = ""
        self.protocol_sync_result: dict[str, Any] = {}
        self._protocol_thread: QThread | None = None
        self._protocol_worker: RegisteredProtocolImportWorker | None = None
        self._protocol_reload_silent = True
        self.reason.setVisible(False)
        for generic_action in (self.start, self.review, self.approve, self.complete):
            generic_action.setVisible(False)
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "The approved protocol supplies most planned timing. This page verifies "
            "that camera, injection, overflow, chemistry, and interventions share one "
            "experiment clock. Raw timestamps remain unchanged; every correction is a "
            "versioned local adjustment."
        )
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Experiment"))
        self.run_filter = QComboBox()
        self.run_filter.setMinimumWidth(180)
        self.run_filter.setToolTip(
            "Select one experiment. The river view and event table below use the same "
            "filter; series-wide events are included where relevant."
        )
        self.run_filter.currentIndexChanged.connect(self._run_filter_changed)
        view_row.addWidget(self.run_filter)
        view_row.addStretch(1)
        self.timeline_summary = QLabel()
        self.timeline_summary.setObjectName("secondaryText")
        view_row.addWidget(self.timeline_summary)
        self.work_layout.addLayout(view_row)
        protocol_row = QHBoxLayout()
        self.protocol_summary = QLabel()
        self.protocol_summary.setObjectName("secondaryText")
        self.protocol_summary.setWordWrap(True)
        protocol_row.addWidget(self.protocol_summary, 1)
        self.reload_protocols = QPushButton("Reload registered protocols")
        self.reload_protocols.setToolTip(
            "Read the protocol workbooks already assigned in Series setup. "
            "The source files remain read-only and existing timeline rows are not duplicated."
        )
        self.reload_protocols.clicked.connect(self._reload_registered_protocols)
        protocol_row.addWidget(self.reload_protocols)
        self.work_layout.addLayout(protocol_row)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Local corrections"))
        self.edit_button = QPushButton("Adjust selected")
        self.edit_button.setToolTip(
            "Create a versioned local correction for the selected protocol event. "
            "The registered protocol workbook is never changed."
        )
        self.edit_button.clicked.connect(self._edit_event)
        toolbar.addWidget(self.edit_button)
        self.clear_button = QPushButton("Revert local adjustment")
        self.clear_button.setToolTip(
            "Remove the local correction and return to the value read from the protocol."
        )
        self.clear_button.clicked.connect(self._clear_event)
        toolbar.addWidget(self.clear_button)
        toolbar.addStretch(1)
        self.work_layout.addLayout(toolbar)
        approval_row = QHBoxLayout()
        self.run_review_status = QLabel()
        self.run_review_status.setObjectName("secondaryText")
        approval_row.addWidget(self.run_review_status, 1)
        self.approve_selected_run = QPushButton("Approve selected experiment")
        self.approve_selected_run.clicked.connect(self._approve_selected_experiment)
        approval_row.addWidget(self.approve_selected_run)
        self.approve_all_runs = QPushButton("Approve all experiments and complete")
        self.approve_all_runs.clicked.connect(self._approve_all_experiments)
        approval_row.addWidget(self.approve_all_runs)
        self.work_layout.addLayout(approval_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Timeline scale"))
        self.timeline_zoom = QSlider(Qt.Orientation.Horizontal)
        self.timeline_zoom.setRange(0, 300)
        self.timeline_zoom.setSingleStep(2)
        self.timeline_zoom.setPageStep(10)
        self.timeline_zoom.setValue(0)
        self.timeline_zoom.setMinimumWidth(220)
        self.timeline_zoom.setToolTip(
            "Logarithmic scale from Fit to 1000x. Increase it to separate nearby "
            "events; the complete experiment remains available with the horizontal "
            "scrollbar. Ctrl+wheel over the timeline changes this scale."
        )
        self.timeline_zoom.valueChanged.connect(self._timeline_zoom_changed)
        scale_row.addWidget(self.timeline_zoom)
        self.timeline_zoom_label = QLabel("Fit")
        self.timeline_zoom_label.setMinimumWidth(52)
        scale_row.addWidget(self.timeline_zoom_label)
        self.timeline_fit = QPushButton("Fit all")
        self.timeline_fit.setToolTip(
            "Show the complete experiment without horizontal scrolling."
        )
        self.timeline_fit.clicked.connect(lambda: self.timeline_zoom.setValue(0))
        scale_row.addWidget(self.timeline_fit)
        scale_row.addStretch(1)
        self.work_layout.addLayout(scale_row)

        self.plot = TimelinePlot()
        self.plot.event_selected.connect(self._select_event_id)
        self.timeline_scroll = TimelineScrollArea(self.plot)
        self.timeline_scroll.zoom_step_requested.connect(self._timeline_zoom_step)
        self.work_layout.addWidget(self.timeline_scroll)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(7)
        self.tree.setHeaderLabels(
            [
                "Event",
                "Raw time [h]",
                "Local adjustment",
                "Effective time [h]",
                "Linked image",
                "Value",
                "QC",
            ]
        )
        self.tree.setAlternatingRowColors(True)
        self.tree.setStyleSheet(
            "QTreeWidget::item:selected {"
            "background-color: #176f96; color: #ffffff;"
            "border-top: 1px solid #8ad8f7; border-bottom: 1px solid #8ad8f7;"
            "}"
        )
        self.tree.setToolTip(
            "Select an event to highlight the same marker on the timeline. "
            "Selection works in both directions."
        )
        self.tree.itemSelectionChanged.connect(self._tree_selection_changed)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        splitter.addWidget(self.tree)
        self.event_details = QTextEdit()
        self.event_details.setReadOnly(True)
        self.event_details.setMinimumWidth(320)
        splitter.addWidget(self.event_details)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        self.work_layout.addWidget(splitter, 1)
        self.refresh()
        self._reload_registered_protocols(silent=True)

    def refresh(self) -> None:
        self._refresh_gate()
        stored_events = self.store.effective_events()
        all_events = [
            *stored_events,
            *self._camera_end_events(
                stored_events,
                self.protocol_sync_result.get("camera_sequence_endpoints") or [],
            ),
        ]
        self._sync_run_filter(all_events)
        selected_run = str(self.run_filter.currentData() or "")
        events = [
            event
            for event in all_events
            if not selected_run
            or str(event.get("run_id") or "").casefold()
            in {"", selected_run}
        ]
        visible_event_ids = {str(event["event_id"]) for event in events}
        if self.selected_event_id not in visible_event_ids:
            self.selected_event_id = ""
        issues = [
            issue
            for issue in self.store.qc_issues()
            if str(issue.get("event_id") or "") in visible_event_ids
        ]
        issues_by_event: dict[str, list[dict[str, Any]]] = {}
        for issue in issues:
            issues_by_event.setdefault(str(issue["event_id"]), []).append(issue)
        adjusted = sum(bool(event.get("overrides")) for event in events)
        errors = sum(str(issue.get("severity")) == "error" for issue in issues)
        warnings = sum(str(issue.get("severity")) == "warning" for issue in issues)
        run_label = selected_run.upper() if selected_run else "Series"
        self.timeline_summary.setText(
            f"{run_label}: {len(events)} events · {adjusted} adjusted · "
            f"{errors} errors · {warnings} warnings"
        )
        self._refresh_protocol_summary(all_events)
        self._refresh_run_review_status()
        self.tree.blockSignals(True)
        self.tree.clear()
        groups: dict[tuple[str, str], QTreeWidgetItem] = {}
        selected_item: QTreeWidgetItem | None = None
        for event in events:
            run = str(event.get("run_id") or "all runs")
            source = str(event.get("source") or "manual")
            group_key = (run, source)
            if group_key not in groups:
                group = QTreeWidgetItem([f"{run.upper()} · {source}"])
                group.setFirstColumnSpanned(True)
                group.setExpanded(True)
                self.tree.addTopLevelItem(group)
                groups[group_key] = group
            overrides = dict(event.get("overrides") or {})
            adjustment = ", ".join(sorted(overrides)) if overrides else ""
            event_issues = issues_by_event.get(str(event["event_id"]), [])
            qc = max(
                (str(issue["severity"]) for issue in event_issues),
                key=lambda value: {"error": 3, "warning": 2, "info": 1}.get(value, 0),
                default="OK",
            )
            linked_image, image_basis = self._linked_image(events, event)
            item = QTreeWidgetItem(
                [
                    f"{event.get('event_type', '')}: {event.get('label', '')}".rstrip(
                        ": "
                    ),
                    _number(event.get("raw_time_h")),
                    adjustment,
                    _number(event.get("time_h")),
                    linked_image,
                    f"{_number(event.get('value'))} {event.get('unit', '')}".strip(),
                    qc,
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, str(event["event_id"]))
            if linked_image:
                item.setToolTip(4, image_basis)
            if qc == "error":
                item.setForeground(6, QColor("#e27979"))
            elif qc == "warning":
                item.setForeground(6, QColor("#e3b65c"))
            groups[group_key].addChild(item)
            if str(event["event_id"]) == self.selected_event_id:
                for column in range(self.tree.columnCount()):
                    item.setBackground(column, QColor("#176f96"))
                    item.setForeground(column, QColor("#ffffff"))
                    font = item.font(column)
                    font.setBold(True)
                    item.setFont(column, font)
                selected_item = item
        if selected_item is not None:
            self.tree.setCurrentItem(selected_item)
            self.tree.scrollToItem(selected_item)
        self.tree.blockSignals(False)
        self.plot.set_events(events, self.selected_event_id)
        if self.selected_event_id:
            event_id = self.selected_event_id
            QTimer.singleShot(
                0,
                lambda: self.timeline_scroll.centre_on_event(event_id),
            )
        self._show_selected_details(events, issues_by_event)
        selected_event = next(
            (
                row
                for row in events
                if str(row.get("event_id") or "") == self.selected_event_id
            ),
            None,
        )
        editable = bool(
            selected_event
            and not (selected_event.get("metadata") or {}).get("read_only_derived")
        )
        self.edit_button.setEnabled(editable)
        self.clear_button.setEnabled(editable)

    @staticmethod
    def _camera_end_events(
        events: list[dict[str, Any]], endpoints: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Derive read-only camera completion events from image inventories."""

        derived: list[dict[str, Any]] = []
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue
            run_id = str(endpoint.get("run_id") or "").casefold()
            last = endpoint.get("last") or {}
            last_path = str(last.get("path") or "")
            if not run_id or not last_path:
                continue
            last_name = Path(last_path).name
            last_match = re.search(r"(\d+)(?!.*\d)", Path(last_name).stem)
            anchors = [
                row
                for row in events
                if str(row.get("run_id") or "").casefold() == run_id
                and str(row.get("event_type") or "").casefold()
                == "camera_interval_start"
                and (
                    row.get("time_h") is not None
                    or row.get("raw_time_h") is not None
                )
                and (
                    row.get("value") is not None
                    or row.get("raw_value") is not None
                )
                and str(row.get("path") or "").strip()
            ]
            candidates: list[tuple[float, dict[str, Any]]] = []
            if last_match is not None:
                last_number = int(last_match.group(1))
                last_prefix = Path(last_name).stem[: last_match.start()].casefold()
                for anchor in anchors:
                    anchor_name = Path(str(anchor.get("path") or "")).name
                    anchor_match = re.search(
                        r"(\d+)(?!.*\d)", Path(anchor_name).stem
                    )
                    if anchor_match is None:
                        continue
                    if (
                        Path(anchor_name).stem[: anchor_match.start()].casefold()
                        != last_prefix
                    ):
                        continue
                    anchor_number = int(anchor_match.group(1))
                    if last_number < anchor_number:
                        continue
                    interval = float(
                        anchor.get("value")
                        if anchor.get("value") is not None
                        else anchor["raw_value"]
                    )
                    unit = str(anchor.get("unit") or "").strip().casefold()
                    if unit in {"min", "minute", "minutes"}:
                        interval *= 60.0
                    elif unit in {"h", "hr", "hour", "hours"}:
                        interval *= 3600.0
                    elif unit not in {"s", "sec", "second", "seconds"}:
                        continue
                    anchor_time = float(
                        anchor.get("time_h")
                        if anchor.get("time_h") is not None
                        else anchor["raw_time_h"]
                    )
                    time_h = (
                        anchor_time
                        + (last_number - anchor_number) * interval / 3600.0
                    )
                    candidates.append((time_h, anchor))
            time_h = max(candidates, key=lambda row: row[0])[0] if candidates else None
            source_id = str(endpoint.get("source_id") or "")
            derived.append(
                {
                    "event_id": f"integrity-camera-end-{source_id}",
                    "run_id": run_id,
                    "source": "camera inventory",
                    "event_type": "camera_acquisition_end",
                    "label": f"Camera acquisition ended at {last_name}",
                    "raw_time_h": time_h,
                    "time_h": time_h,
                    "raw_value": float(endpoint.get("image_count") or 0),
                    "value": float(endpoint.get("image_count") or 0),
                    "unit": "images",
                    "path": last_path,
                    "raw_timestamp": str(last.get("exif_datetime") or ""),
                    "note": "",
                    "status": "derived",
                    "overrides": {},
                    "metadata": {
                        "read_only_derived": True,
                        "integrity_source_id": source_id,
                        "source_path": str(endpoint.get("source_path") or ""),
                        "image_count": int(endpoint.get("image_count") or 0),
                        "chronology": str(
                            endpoint.get("chronology") or "natural_filename_order"
                        ),
                        "time_basis": (
                            "registered_camera_interval_and_image_number"
                            if candidates
                            else "exact_last_image_without_resolved_experiment_time"
                        ),
                    },
                }
            )
        return derived

    @staticmethod
    def _linked_image(
        events: list[dict[str, Any]], event: dict[str, Any]
    ) -> tuple[str, str]:
        """Resolve an explicit or schedule-derived image for one event."""

        path = str(event.get("path") or "").strip()
        if path:
            name = Path(path).name
            if Path(name).suffix.casefold() in {
                ".arw",
                ".bmp",
                ".jpeg",
                ".jpg",
                ".png",
                ".tif",
                ".tiff",
            }:
                if (event.get("metadata") or {}).get("read_only_derived"):
                    return (
                        name,
                        "Exact final image found in the current data-integrity "
                        "inventory. When available, its experiment time is derived "
                        "from the registered camera interval and image numbering.",
                    )
                return name, "Exact image recorded in the protocol for this event."

        event_time = (
            event.get("time_h")
            if event.get("time_h") is not None
            else event.get("raw_time_h")
        )
        if event_time is None:
            return "", ""
        run_id = str(event.get("run_id") or "").casefold()
        anchors = [
            row
            for row in events
            if str(row.get("run_id") or "").casefold() == run_id
            and str(row.get("event_type") or "").casefold()
            == "camera_interval_start"
            and (row.get("time_h") is not None or row.get("raw_time_h") is not None)
            and float(
                row.get("time_h")
                if row.get("time_h") is not None
                else row["raw_time_h"]
            )
            <= float(event_time)
            and (row.get("value") is not None or row.get("raw_value") is not None)
            and str(row.get("path") or "").strip()
        ]
        if not anchors:
            return "", ""
        anchor = max(
            anchors,
            key=lambda row: float(
                row.get("time_h")
                if row.get("time_h") is not None
                else row["raw_time_h"]
            ),
        )
        unit = str(anchor.get("unit") or "").strip().casefold()
        interval = float(
            anchor.get("value")
            if anchor.get("value") is not None
            else anchor["raw_value"]
        )
        if interval <= 0.0:
            return "", ""
        if unit in {"min", "minute", "minutes"}:
            interval *= 60.0
        elif unit in {"h", "hr", "hour", "hours"}:
            interval *= 3600.0
        elif unit not in {"s", "sec", "second", "seconds"}:
            return "", ""
        anchor_name = Path(str(anchor["path"])).name
        match = re.search(r"(\d+)(?!.*\d)", Path(anchor_name).stem)
        if match is None:
            return "", ""
        anchor_time = (
            anchor.get("time_h")
            if anchor.get("time_h") is not None
            else anchor["raw_time_h"]
        )
        elapsed_s = (float(event_time) - float(anchor_time)) * 3600.0
        offset = max(0, round(elapsed_s / interval))
        number = int(match.group(1)) + offset
        stem = Path(anchor_name).stem
        inferred_stem = (
            stem[: match.start()]
            + f"{number:0{len(match.group(1))}d}"
            + stem[match.end() :]
        )
        inferred = inferred_stem + Path(anchor_name).suffix
        return (
            f"~{inferred}",
            "Expected image calculated from the preceding camera interval. "
            "The tilde indicates that missing or delayed acquisitions have not yet "
            "been reconciled against the actual image inventory.",
        )

    def _sync_run_filter(self, events: list[dict[str, Any]]) -> None:
        current = str(self.run_filter.currentData() or "")
        run_ids = {
            str(row.get("run_id") or "").strip().casefold()
            for row in self.project.manifest.get("runs") or []
            if str(row.get("run_id") or "").strip()
        }
        run_ids.update(
            str(event.get("run_id") or "").strip().casefold()
            for event in events
            if str(event.get("run_id") or "").strip()
        )
        ordered = sorted(run_ids)
        statuses = self._run_temporal_statuses()
        self.run_filter.blockSignals(True)
        try:
            self.run_filter.clear()
            if not ordered:
                self.run_filter.addItem("No experiments", "")
                return
            for run_id in ordered:
                status = statuses.get(run_id, "not_started").replace("_", " ")
                self.run_filter.addItem(f"{run_id.upper()} · {status}", run_id)
            index = self.run_filter.findData(current)
            self.run_filter.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.run_filter.blockSignals(False)

    def _run_filter_changed(self) -> None:
        self.selected_event_id = ""
        self.refresh()

    def _registered_run_ids(self) -> list[str]:
        return sorted(
            {
                str(row.get("run_id") or "").strip().casefold()
                for row in self.project.manifest.get("runs") or []
                if isinstance(row, dict) and str(row.get("run_id") or "").strip()
            }
        )

    def _run_temporal_statuses(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for row in self.project.manifest.get("runs") or []:
            if not isinstance(row, dict):
                continue
            run_id = str(row.get("run_id") or "").strip().casefold()
            workflow = row.get("workflow") or {}
            if run_id and isinstance(workflow, dict):
                statuses[run_id] = str(
                    (workflow.get("temporal_sync") or {}).get("status")
                    or "not_started"
                )
        return statuses

    def _refresh_protocol_summary(self, events: list[dict[str, Any]]) -> None:
        protocol_events = [
            row
            for row in events
            if str(row.get("source") or "").casefold() == "registered_protocol"
        ]
        runs = {
            str(row.get("run_id") or "").strip().upper()
            for row in protocol_events
            if str(row.get("run_id") or "").strip()
        }
        source_count = int(
            self.protocol_sync_result.get("registered_source_count") or 0
        )
        errors = len(self.protocol_sync_result.get("errors") or [])
        unavailable = len(self.protocol_sync_result.get("unavailable") or [])
        self.protocol_summary.setText(
            f"Registered protocol input: {source_count} workbook(s); "
            f"{len(protocol_events)} timeline events for {len(runs)} experiment(s). "
            f"{errors} read error(s); {unavailable} unavailable source(s)."
        )

    def _refresh_run_review_status(self) -> None:
        statuses = self._run_temporal_statuses()
        selected = str(self.run_filter.currentData() or "").casefold()
        approved = sum(
            value in {"approved", "complete"} for value in statuses.values()
        )
        selected_status = statuses.get(selected, "not_started")
        self.run_review_status.setText(
            f"{selected.upper() or 'No experiment'}: "
            f"{selected_status.replace('_', ' ')}. Series approval: "
            f"{approved} of {len(statuses)} experiments."
        )
        self.approve_selected_run.setEnabled(bool(selected))

    def _reload_registered_protocols(
        self, _checked: bool = False, *, silent: bool = False
    ) -> None:
        if self._protocol_thread is not None:
            return
        registered_paths = {
            str(path).strip()
            for run in self.project.manifest.get("runs") or []
            if isinstance(run, dict)
            for mapping in [run.get("source_mapping") or {}]
            if isinstance(mapping, dict)
            for path in [
                *(mapping.get("protocol_tables") or []),
                mapping.get("protocol_table"),
            ]
            if str(path or "").strip()
        }
        if not registered_paths:
            self.protocol_sync_result = {
                "registered_source_count": 0,
                "added": 0,
                "skipped": 0,
                "unavailable": [],
                "errors": [],
            }
            self.refresh()
            return
        self._protocol_reload_silent = silent
        self.reload_protocols.setEnabled(False)
        self.protocol_summary.setText(
            "Reading the protocol workbooks already registered for this series..."
        )
        thread = QThread(self)
        worker = RegisteredProtocolImportWorker(
            self.project.root,
            str(self.project.manifest.get("created_by") or ""),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._registered_protocols_loaded)
        worker.failed.connect(self._registered_protocols_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._protocol_thread_finished)
        self._protocol_thread = thread
        self._protocol_worker = worker
        thread.start()

    @Slot(object)
    def _registered_protocols_loaded(self, raw_result: object) -> None:
        self.protocol_sync_result = (
            dict(raw_result) if isinstance(raw_result, dict) else {}
        )
        self.refresh()
        if not self._protocol_reload_silent and not self.protocol_sync_result.get("errors"):
            QMessageBox.information(
                self,
                "Registered protocols read",
                f"Added {int(self.protocol_sync_result.get('added') or 0)} new events; "
                f"{int(self.protocol_sync_result.get('skipped') or 0)} existing rows "
                "were retained.",
            )

    @Slot(str)
    def _registered_protocols_failed(self, error: str) -> None:
        self.protocol_sync_result = {"errors": [{"error": error}]}
        self.refresh()
        if not self._protocol_reload_silent:
            QMessageBox.critical(self, "Could not read registered protocols", error)

    @Slot()
    def _protocol_thread_finished(self) -> None:
        thread = self._protocol_thread
        if thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            thread.wait()
        self._protocol_worker = None
        self._protocol_thread = None
        self.reload_protocols.setEnabled(True)
        if thread is not None:
            thread.deleteLater()

    def shutdown_background_work(self) -> None:
        """Wait for a protocol read before the application destroys the page."""

        if self._protocol_thread is not None and self._protocol_thread.isRunning():
            self._protocol_thread.quit()
            self._protocol_thread.wait()

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        self.shutdown_background_work()
        super().closeEvent(event)

    def _approve_runs(self, run_ids: list[str]) -> None:
        if not run_ids:
            return
        actor = str(self.project.manifest.get("created_by") or "")
        try:
            self.project.set_run_module_status(
                run_ids,
                "temporal_sync",
                "approved",
                updated_by=actor,
                reason="Protocol-derived timeline inspected in Temporal synchronization",
            )
            statuses = self._run_temporal_statuses()
            if statuses and all(
                value in {"approved", "complete"} for value in statuses.values()
            ):
                self.project.set_module_status(
                    "temporal_sync",
                    "complete",
                    updated_by=actor,
                    reason="All registered experiment timelines approved",
                )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(self, "Could not approve timeline", str(exc))
            return
        self.refresh()
        self.status_changed.emit()

    def _approve_selected_experiment(self) -> None:
        run_id = str(self.run_filter.currentData() or "").strip().casefold()
        self._approve_runs([run_id] if run_id else [])

    def _approve_all_experiments(self) -> None:
        self._approve_runs(self._registered_run_ids())

    def _show_selected_details(
        self,
        events: list[dict[str, Any]],
        issues_by_event: dict[str, list[dict[str, Any]]],
    ) -> None:
        event = next(
            (row for row in events if str(row["event_id"]) == self.selected_event_id),
            None,
        )
        if event is None:
            self.event_details.setPlainText(
                "Select a marker or event to inspect raw values, local adjustments, and provenance."
            )
            return
        metadata = dict(event.get("metadata") or {})
        linked_image, image_basis = self._linked_image(events, event)
        issue_text = (
            "\n".join(
                f"- {issue['severity'].upper()}: {issue['message']}"
                for issue in issues_by_event.get(str(event["event_id"]), [])
            )
            or "No QC findings."
        )
        self.event_details.setPlainText(
            f"Event ID\n{event['event_id']}\n\n"
            f"Raw time\n{_number(event.get('raw_time_h'), 6)} h\n\n"
            f"Effective time\n{_number(event.get('time_h'), 6)} h\n\n"
            f"Linked image\n{linked_image or 'Not linked'}\n"
            f"{image_basis}\n\n"
            f"Active local adjustments\n{json_pretty(event.get('overrides') or {})}\n\n"
            f"Source file\n{metadata.get('source_file') or event.get('path') or 'Not linked'}\n\n"
            f"QC\n{issue_text}"
        )

    def _tree_selection_changed(self) -> None:
        item = self.tree.currentItem()
        event_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "") if item else ""
        if event_id:
            self._select_event_id(event_id)

    def _timeline_zoom_changed(self, value: int) -> None:
        factor = timeline_zoom_factor(value)
        if value == 0:
            label = "Fit"
        elif factor >= 10.0:
            label = f"{factor:,.0f}x"
        else:
            label = f"{factor:.2f}".rstrip("0").rstrip(".") + "x"
        self.timeline_zoom_label.setText(label)
        self.timeline_scroll.set_zoom_factor(factor)

    def _timeline_zoom_step(self, direction: int) -> None:
        self.timeline_zoom.setValue(
            self.timeline_zoom.value() + int(direction) * self.timeline_zoom.pageStep()
        )

    def _select_event_id(self, event_id: str) -> None:
        self.selected_event_id = event_id
        self.refresh()

    def _import_table(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Import protocol or measurement timeline",
            str(self.project.root),
            "Tables (*.csv *.tsv *.xlsx *.xls);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if not selected:
            return
        dialog = TimelineImportDialog(
            self,
            Path(selected),
            str(self.project.manifest.get("created_by") or ""),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            result = self.store.import_table(
                Path(selected),
                source=dialog.source.text().strip(),
                run_id=dialog.run_id.text().strip(),
                actor=dialog.actor.text().strip(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Timeline import failed", str(exc))
            return
        QMessageBox.information(
            self,
            "Timeline imported",
            f"Added {result['added']} events; skipped {result['skipped']} existing records.",
        )
        self.refresh()

    def _import_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Import one protocol table per run",
            str(self.project.root),
            options=DIRECTORY_DIALOG_OPTIONS,
        )
        if not selected:
            return
        dialog = TimelineImportDialog(
            self,
            Path(selected),
            str(self.project.manifest.get("created_by") or ""),
        )
        dialog.run_id.setEnabled(False)
        dialog.run_id.setPlaceholderText("Run ID is inferred from each filename")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            result = self.store.import_folder(
                Path(selected),
                source=dialog.source.text().strip(),
                actor=dialog.actor.text().strip(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Timeline folder import failed", str(exc))
            return
        QMessageBox.information(
            self,
            "Timeline folder imported",
            f"Added {result['added']} events; skipped {result['skipped']} existing records.",
        )
        self.refresh()

    def _add_event(self) -> None:
        dialog = TimelineEventDialog(
            self,
            title="Add local timeline event",
            event={"run_id": str(self.run_filter.currentData() or "")},
            actor=str(self.project.manifest.get("created_by") or ""),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            event = self.store.add_event(
                run_id=values["run_id"],
                source=values["source"],
                event_type=values["event_type"],
                time_h=values["time_h"],
                label=values["label"],
                value=values["value"],
                actor=values["actor"],
                metadata={"note": values["note"], "reason": values["reason"]},
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not add event", str(exc))
            return
        self.selected_event_id = str(event["event_id"])
        self.refresh()

    def _edit_event(self) -> None:
        event = next(
            (
                row
                for row in self.store.effective_events()
                if str(row["event_id"]) == self.selected_event_id
            ),
            None,
        )
        if event is None:
            return
        dialog = TimelineEventDialog(
            self,
            title="Adjust timeline event locally",
            event=event,
            actor=str(self.project.manifest.get("created_by") or ""),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            self.store.set_override(
                self.selected_event_id,
                {
                    "run_id": values["run_id"],
                    "source": values["source"],
                    "event_type": values["event_type"],
                    "label": values["label"],
                    "time_h": values["time_h"],
                    "value": values["value"],
                    "note": values["note"],
                },
                actor=values["actor"],
                reason=values["reason"],
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save adjustment", str(exc))
            return
        self.refresh()

    def _clear_event(self) -> None:
        event = next(
            (
                row
                for row in self.store.effective_events()
                if str(row["event_id"]) == self.selected_event_id
            ),
            None,
        )
        if event is None or not event.get("overrides"):
            QMessageBox.information(
                self, "No local adjustment", "This event uses raw values."
            )
            return
        try:
            self.store.clear_overrides(
                self.selected_event_id,
                list(dict(event["overrides"])),
                actor=str(self.project.manifest.get("created_by") or ""),
                reason="Reverted from the timeline work surface",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not revert adjustment", str(exc))
            return
        self.refresh()
