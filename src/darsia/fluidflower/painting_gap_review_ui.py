"""Human review of verified painting gaps; every overlay is display only.

The worker module owns source admission and approval identities. This view
never reconstructs a preparation or classifies pixels. A selected photograph
and its gap mask are loaded once in a finite owned reader, then shown with
unfilled outlines. Viewing, navigating and changing display options approve
nothing.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QPoint, QRectF, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainterPath, QPalette, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QGraphicsPathItem, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QSlider, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .map_placement_view import OverlayView
from .painted_analysis_exclusions import (
    CLASSIFICATIONS, CLASSIFICATION_LABELS, classification_key, classification_label,
)

DEFAULT_REASON = "water and air above the sand"
# The reference written with gap approvals; a series may name its own
# (decisions.gap_review_decision_ref, read by series_decisions).
FINDINGS_DECISION_REF = "gap-review-findings-choice"
CLASSES = CLASSIFICATIONS
CLASS_LABELS = CLASSIFICATION_LABELS


def _project(points, matrix):
    """Apply an explicitly supplied xy display transform, without a fallback."""
    points, matrix = np.asarray(points, np.float64), np.asarray(matrix, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError("The review lacks its explicit photograph transform")
    carried = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if not np.all(np.isfinite(carried)) or np.any(carried[:, 2] == 0):
        raise ValueError("The review photograph transform is undefined here")
    return carried[:, :2] / carried[:, 2:3]


def _rgb(packet):
    """Make an owned display copy using the verified, declared RGB encoding."""
    image = np.asarray(packet["image_rgb"])
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The bound review photograph must be RGB")
    encoding = packet["image_encoding"]
    if encoding == "rgb_uint8" and image.dtype == np.uint8:
        return image.copy()
    if encoding == "rgb_float01" and np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise ValueError("The bound review photograph has nonfinite pixels")
        return np.rint(255.0 * np.clip(image, 0., 1.)).astype(np.uint8)
    raise ValueError("The bound review photograph has an unsupported declared encoding")


def _path(polylines, matrix, *, closed=True):
    path = QPainterPath()
    for points in polylines:
        mapped = _project(points, matrix)
        if not len(mapped):
            continue
        path.moveTo(*mapped[0])
        for point in mapped[1:]:
            path.lineTo(*point)
        if closed:
            path.closeSubpath()
    return path


def _gap_contours(mask):
    """Extract display contours on a private copy; retain holes and thin gaps."""
    import cv2
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("The verified gap preview must contain a boolean native mask")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)
        if len(points) == 1:
            x, y = points[0]
            points = np.array([[x-.5, y-.5], [x+.5, y-.5],
                               [x+.5, y+.5], [x-.5, y+.5]])
        lines.append(points)
    return lines


class _Read(QObject):
    finished = Signal()

    def __init__(self, function):
        super().__init__()
        self.function = function
        self.result, self.error = None, ""

    @Slot()
    def run(self):
        try:
            self.result = self.function()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self.moveToThread(QCoreApplication.instance().thread())
            self.finished.emit()


class _Photo(OverlayView):
    point_selected = Signal(object)
    point_clicked = Signal(object)
    pick_single_point = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.auto_fit_on_resize = False
        self._user_zoomed = False
        self.view_rect = None
        # Both bars are needed once the scene includes half a viewport around
        # the photograph. Keep their geometry stable during fit and resize.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

    def _update_pan_margins(self):
        """Allow every photograph edge to reach the viewport centre."""
        if not hasattr(self, "item") or self.item.pixmap().isNull():
            return
        viewport = self.viewport().rect()
        visible = self.mapToScene(viewport).boundingRect()
        if visible.width() <= 0 or visible.height() <= 0:
            return
        origin = self.mapToScene(QPoint(0, 0))
        rounding = self.mapToScene(QPoint(2, 2)) - origin
        margin_x = visible.width() * .5 + abs(rounding.x())
        margin_y = visible.height() * .5 + abs(rounding.y())
        bounds = self.view_rect if self.view_rect is not None else self.item.boundingRect()
        rectangle = bounds.adjusted(-margin_x, -margin_y, margin_x, margin_y)
        if rectangle == self.sceneRect():
            return
        centre = self.mapToScene(viewport.center())
        self.scene().setSceneRect(rectangle)
        self.centerOn(centre)

    def fit(self):
        self._user_zoomed = False
        bounds = self.view_rect if self.view_rect is not None else self.item.boundingRect()
        self.scene().setSceneRect(bounds)
        if self.view_rect is None:
            super().fit()
        else:
            self.resetTransform()
            self.fitInView(self.view_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_pan_margins()

    def set_view_rect(self, rectangle=None, *, refit=True):
        """Choose a display window without replacing pixels or overlay coordinates."""
        rectangle = None if rectangle is None else QRectF(rectangle)
        if rectangle is not None and not rectangle.isValid():
            raise ValueError("The photograph view needs a positive rectangle")
        changed = rectangle != self.view_rect
        self.view_rect = rectangle
        if changed and not self.item.pixmap().isNull():
            if refit:
                self.fit()
            else:
                self._update_pan_margins()
            self.viewport().update()
        return changed

    def drawForeground(self, painter, rectangle):
        super().drawForeground(painter, rectangle)
        if self.view_rect is not None:
            outside = QPainterPath()
            outside.addRect(rectangle)
            inside = QPainterPath()
            inside.addRect(self.view_rect)
            painter.fillPath(outside.subtracted(inside), self.palette().brush(QPalette.ColorRole.Base))

    def scale(self, sx, sy):
        if sx != 1 or sy != 1:
            self._user_zoomed = True
        super().scale(sx, sy)
        self._update_pan_margins()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if (getattr(self, "auto_fit_on_resize", False)
                and not self._user_zoomed and not self.item.pixmap().isNull()):
            self.fit()
        else:
            self._update_pan_margins()

    def show_rgb(self, image, *, refit=True):
        """Replace display pixels; same-photo adjustments can retain the view."""
        array = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
        if array.size == 0:
            self.item.setPixmap(QPixmap())
            return False
        first = self.item.pixmap().isNull()
        height, width = array.shape[:2]
        picture = QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self.item.setPixmap(QPixmap.fromImage(picture))
        if first or refit:
            self.scene().setSceneRect(self.item.boundingRect())
            self.fit()
        else:
            self._update_pan_margins()
        return True

    def mousePressEvent(self, event):
        if self.pick_single_point and event.button() == Qt.MouseButton.LeftButton:
            self.pick_single_point = False
            point = self.mapToScene(event.position().toPoint())
            self.point_clicked.emit((point.x(), point.y()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        point = self.mapToScene(event.position().toPoint())
        if self.pick_single_point:
            self.pick_single_point = False
            self.point_clicked.emit((point.x(), point.y()))
        else:
            self.point_selected.emit((point.x(), point.y()))
        event.accept()


class GapReviewDialog(QDialog):
    """Review one saved gap catalogue; only action buttons create approval DATA."""
    recheck_requested = Signal(object)
    paint_requested = Signal(object)
    approval_changed = Signal(object)
    state_requested = Signal(object)
    fix_requested = Signal(object)
    adopt_requested = Signal(object)
    position_tolerances_changed = Signal(object)
    display_recheck_requested = Signal()
    water_surface_confirmed = Signal(object)
    water_surfaces_cleared = Signal()
    water_surface_edited = Signal(str)
    water_surface_tolerances_changed = Signal(object)
    readiness_changed = Signal()
    refresh_findings_requested = Signal()

    def __init__(self, source, *, actor, parent=None, review_reader=None,
                 preview_reader=None, approval_builder=None, review_validator=None, comparison_reader=None, water_surface_api=None,
                 preparation_dirty=None, additional_result_consumer=None, decision_ref=None):
        super().__init__(parent)
        self.decision_ref = str(decision_ref or FINDINGS_DECISION_REF).strip()
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMinMaxButtonsHint
                            | Qt.WindowType.WindowCloseButtonHint | Qt.WindowType.WindowSystemMenuHint)
        self.setSizeGripEnabled(True)
        if review_reader is None or preview_reader is None or approval_builder is None:
            from . import painted_analysis_exclusions as gaps
            if preview_reader is None:
                from . import painted_review_comparison as comparison
                from . import painted_camera_jpg_display as camera_jpg
                review_reader = review_reader or comparison.read_review
                preview_reader = camera_jpg.read_gap_preview
                comparison_reader = comparison_reader or camera_jpg.read_comparison_preview
            review_reader = review_reader or gaps.read_gap_review
            approval_builder = approval_builder or gaps.make_gap_approval
        self._read_review, self._read_preview, self._make_approval = review_reader, preview_reader, approval_builder
        self._review_validator = review_validator
        self._preparation_dirty = preparation_dirty
        self._additional_result_consumer = additional_result_consumer
        self._comparison_reader = comparison_reader
        self._base_packet = self._comparison_key = None
        self.actor = str(actor or "").strip()
        self.review = self.approval = self.packet = None
        self._source = deepcopy(source)
        self.thread = self.worker = None
        self._pending = self._selected = None
        self._closing = False
        self._rgb = self._point_native = None
        self._overlays = []
        self._issues, self._warning_actions, self._acquisition_actions = [], {}, []
        self._initial_issue_types = None
        self._catalogue_baseline = None
        self._check_report_installed = False
        self.coverage_defects = []
        self._resolved_target_rows = set()
        self._last_readiness = None
        self._catalogue_refresh_pending = False
        self._candidate = self._report = None
        self._display_preferences_dirty = self._preparation_changed = False
        self.operation_waiting = False
        self._preview_key = None
        self.pending_corrections = []
        self.water_surface_confirmations = {}
        self.water_surface_bases = {}
        self._unconfirmed_water_edits = set()
        self.position_tolerances = None
        self._rows = []
        self.setWindowTitle("Findings and decisions")
        available = self.screen().availableGeometry()
        self.resize(min(1100, int(available.width() * .9)), min(800, int(available.height() * .9)))
        layout = QVBoxLayout(self)
        note = QLabel("Gaps are shown as outlines. Outside the analysis means the area is not quantified; "
                      "it does not change the physical frame, facies, scale or water classification.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.runs = QTableWidget(0, 5)
        self.runs.setHorizontalHeaderLabels(["Run", "Finding / area", "Size (pixels)", "Message", "Resolution"])
        self.runs.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.runs.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.runs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.runs.currentCellChanged.connect(self._select)
        self.content_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.content_splitter.addWidget(self.runs)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        toggles = QHBoxLayout()
        self.outline = QCheckBox("Show gap outlines")
        self.own_frame = QCheckBox("Show own frame")
        self.transported_frame = QCheckBox("Show transported frame")
        self.layer_boundaries = QCheckBox("Show map layer boundaries")
        self.brighten = QSlider(Qt.Orientation.Horizontal)
        self.brighten.setRange(25, 300)
        self.brighten.setSingleStep(5)
        self.brighten.setPageStep(25)
        self.brighten.setValue(100)
        self.brighten.setToolTip("Brightness changes this display only; the photograph and review data stay unchanged.")
        self.brightness_value = QLabel("100%")
        self.outline.setChecked(True)
        for control in (self.outline, self.own_frame, self.transported_frame, self.layer_boundaries):
            control.toggled.connect(self._display)
            toggles.addWidget(control)
        toggles.addWidget(QLabel("Brightness (display only)"))
        toggles.addWidget(self.brighten, 1)
        toggles.addWidget(self.brightness_value)
        self.brighten.valueChanged.connect(self._brightness_changed)
        preview_layout.addLayout(toggles)
        comparison = QHBoxLayout()
        comparison.addWidget(QLabel("Image (Alt+I)"))
        self.images = QComboBox()
        self.images.addItem("Bound review image", "current")
        self.images.currentIndexChanged.connect(self._image_selected)
        comparison.addWidget(self.images)
        self.position = QLabel("camera registration (map placement): no verified placement binding yet.")
        self.position.setWordWrap(True)
        self.position.setToolTip("Local shift and local x-direction at [0,0] mm, inner left wall/bottom. Camera registration from the existing map placement; no global physical motion or new measurement.")
        comparison.addWidget(self.position, 1)
        preview_layout.addLayout(comparison)
        self.image_shortcut = QShortcut(QKeySequence("Alt+I"), self)
        self.image_shortcut.activated.connect(self._next_image)
        tolerances = QHBoxLayout()
        self.tolerance_note = QLabel("Position tolerances are not bound to verified measurements yet.")
        tolerances.addWidget(self.tolerance_note)
        self.shift_tolerance, self.angle_tolerance = QDoubleSpinBox(), QDoubleSpinBox()
        for control, suffix in ((self.shift_tolerance, " mm"), (self.angle_tolerance, " degrees")):
            control.setRange(0., 1000000.)
            control.setDecimals(4)
            control.setSuffix(suffix)
            control.hide()
            tolerances.addWidget(control)
        self.shift_tolerance.setValue(2.)
        self.angle_tolerance.setValue(.5)
        self.apply_tolerances = QPushButton("Apply tolerances and check")
        self.apply_tolerances.setAutoDefault(False)
        self.apply_tolerances.setEnabled(False)
        self.apply_tolerances.clicked.connect(self._tolerances)
        tolerances.addWidget(self.apply_tolerances)
        preview_layout.addLayout(tolerances)
        self.photo = _Photo()
        self.photo.point_selected.connect(self._point_selected)
        preview_layout.addWidget(self.photo, 1)
        from .painting_water_surface_ui import WaterSurfaceControls
        self.water_surface = WaterSurfaceControls(actor=self.actor, parent=self, api=water_surface_api)
        self.water_surface.line_changed.connect(self._display)
        self.water_surface.pick_requested.connect(self._water_pick_requested)
        self.water_surface.confirmed.connect(self._water_confirmed)
        self.water_surface.line_edited.connect(self._water_line_edited)
        self.water_surface.basis_ready.connect(self._water_basis)
        self.water_surface.tolerances_changed.connect(self._water_tolerances)
        self.photo.point_clicked.connect(self._water_point_clicked)
        preview_layout.addWidget(self.water_surface)
        self.clear_water_confirmations = QPushButton('Remove previous water-line confirmations from this check')
        self.clear_water_confirmations.setAutoDefault(False)
        self.clear_water_confirmations.clicked.connect(self._clear_water_confirmations)
        self.clear_water_confirmations.setToolTip('Removes only pending confirmations from this check. Saved adoption history stays unchanged.')
        preview_layout.addWidget(self.clear_water_confirmations)
        self.content_splitter.addWidget(preview)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 3)
        layout.addWidget(self.content_splitter, 1)
        self.status = QLabel("Reading the bound gap review…")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(100)
        layout.addWidget(self.details)
        classification = QHBoxLayout()
        classification.addWidget(QLabel("Area class (label only)"))
        self.classification = QComboBox()
        self.classification.addItem("Choose a class…", None)
        for label in CLASSES:
            self.classification.addItem(CLASS_LABELS[label], label)
        self.classification.currentIndexChanged.connect(self._buttons)
        classification.addWidget(self.classification)
        self.other_text = QLineEdit()
        self.other_text.setPlaceholderText("Describe Other")
        self.other_text.setMaxLength(512)
        self.other_text.textChanged.connect(self._buttons)
        classification.addWidget(self.other_text, 1)
        layout.addLayout(classification)
        state_row = QHBoxLayout()
        state_row.addWidget(QLabel("Water state in the sand in the facies source photo"))
        self.source_state = QComboBox()
        self.source_state.addItem("Choose explicitly…", None)
        for label, value in (("Clear water", "clearwater"), ("Indicator", "indicator"), ("Unknown", "unknown")):
            self.source_state.addItem(label, value)
        self.source_state.currentIndexChanged.connect(self._buttons)
        state_row.addWidget(self.source_state)
        self.set_state = QPushButton("Record selected water state")
        self.set_state.setAutoDefault(False)
        self.set_state.clicked.connect(self._set_state)
        state_row.addWidget(self.set_state)
        layout.addLayout(state_row)
        reason = QHBoxLayout()
        reason.addWidget(QLabel("Short reason"))
        self.reason = QLineEdit(DEFAULT_REASON)
        self.reason.setMaxLength(512)
        self.reason.textChanged.connect(self._buttons)
        reason.addWidget(self.reason, 1)
        layout.addLayout(reason)
        actions = QGridLayout()
        self.approve_one = QPushButton("Outside the analysis – selected area")
        self.approve_all = QPushButton("Outside the analysis – all listed areas")
        self.paint = QPushButton("Paint in the painting tab")
        self.refresh_findings = QPushButton("Refresh findings")
        self.recheck = QPushButton("Check again")
        self.approve_warning = QPushButton("Accept selected warning")
        self.fix = QPushButton("Fix the cause in the GUI")
        self.adopt = QPushButton("Adopt…")
        self.approve_warning.clicked.connect(self._approve_warning)
        self.fix.clicked.connect(self._fix)
        self.adopt.clicked.connect(self._adopt)

        self.approve_one.clicked.connect(lambda: self._approve(False))
        self.approve_all.clicked.connect(lambda: self._approve(True))
        self.paint.clicked.connect(self._paint)
        self.refresh_findings.clicked.connect(self._refresh_findings)
        self.recheck.clicked.connect(self._recheck)
        for index, control in enumerate((self.approve_one, self.approve_all, self.paint, self.refresh_findings, self.recheck, self.approve_warning, self.fix, self.adopt)):
            control.setAutoDefault(False)
            actions.addWidget(control, index // 3, index % 3)
        layout.addLayout(actions)
        self.coverage_note = QLabel()
        self.coverage_note.setWordWrap(True)
        self.coverage_note.setAccessibleName("Findings list coverage")
        self.coverage_note.hide()
        layout.addWidget(self.coverage_note)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.close)
        layout.addWidget(close)
        self._buttons()
        if source is not None:
            self._start(("review", None), lambda: self._read_review(self._source))
        else:
            self.status.setText("Waiting for the original Check report.")

    def busy(self):
        return self.thread is not None

    def _start(self, key, function):
        if self.thread is not None:
            raise RuntimeError("The current review worker must retire before another starts")
        self.thread, self.worker = QThread(self), _Read(function)
        self.worker.key = key
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self._retired)
        self.thread.start()

    @Slot()
    def _retired(self, expected=None):
        if expected is None:
            expected = self.sender()
        if self.thread is None or self.thread is not expected:
            return
        thread, worker = self.thread, self.worker
        thread.wait()
        self.thread = self.worker = None
        if self._closing and not worker.error:
            try:
                self._consume_additional_result(worker.key, worker.result)
            except (KeyError, TypeError, ValueError, OSError) as error:
                self.status.setText("The completed review action could not be retained: " + str(error))
        if not self._closing:
            if worker.error:
                if worker.key[0] == "comparison":
                    self._reset_comparison()
                self.status.setText("The requested review operation could not finish: " + worker.error)
            elif worker.key == ("review", None):
                try:
                    self._catalogue(worker.result)
                except (KeyError, TypeError, ValueError) as error:
                    self.review = None
                    self.status.setText("The gap review could not open: " + str(error))
            elif worker.key == ("comparison", self._comparison_key):
                try:
                    self._comparison(worker.result)
                except (KeyError, TypeError, ValueError) as error:
                    self._reset_comparison()
                    self.status.setText("The comparison image could not be displayed: " + str(error))
            elif worker.key == ("preview", self._preview_key):
                try:
                    self._preview(worker.result)
                except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
                    self.packet = None
                    self._clear_photo()
                    self.status.setText("The image or gap outline could not be displayed: " + str(error))
            else:
                try:
                    self._consume_additional_result(worker.key, worker.result)
                except (KeyError, TypeError, ValueError, OSError) as error:
                    self.status.setText("The review action could not be accepted: " + str(error))
            if self._pending is not None and self.thread is None:
                run, self._pending = self._pending, None
                self._load(run)
        del worker
        thread.deleteLater()
        self._buttons()
        if self._closing:
            self.close()

    def _consume_additional_result(self, key, result):
        """Presentation extensions consume results inside the original retirement barrier."""
        if self._additional_result_consumer is not None:
            return bool(self._additional_result_consumer(key, result))
        return False

    def _catalogue(self, review, *, validate_context=True, refresh=True):
        if validate_context and self._review_validator is not None:
            self._review_validator(review)
        self.review = deepcopy(review)
        if self._catalogue_baseline is not None:
            if self._catalogue_baseline["pending"]:
                self._catalogue_baseline = dict(pending=False, review_binding=deepcopy(review.get("review_binding")))
            elif self._catalogue_baseline["review_binding"] != review.get("review_binding"):
                self._catalogue_baseline = None
        if self._catalogue_baseline is None:
            self._initial_issue_types = {
                (issue.get("severity"), issue.get("code")) for issue in review.get("issues") or []
            }
        self._update_coverage()
        if not refresh:
            return
        if review.get("issues") is not None and self._report is None:
            self._set_result(None, {"issues": deepcopy(review["issues"])}, catalogue=True)
        else:
            self._refresh_rows()

    def _row_key(self, row):
        if row["kind"] == "source_state":
            return ("source_state",)
        return (("gap", row["run_id"], row["component_id"]) if row["kind"] == "gap"
                else ("issue", row["issue_index"], row["issue_sha256"]))

    def _current_row(self):
        index = self.runs.currentRow()
        return self._rows[index] if 0 <= index < len(self._rows) else None

    def _resolved(self, row):
        if row["kind"] == "source_state":
            return bool(self._acquisition_actions)
        if row["kind"] == "gap":
            return ((row["run_id"], row["component_id"], row["review_row_sha256"]) in self._resolved_target_rows
                or any(action["run_id"] == row["run_id"] and action["component_id"] == row["component_id"]
                and action["review_row_sha256"] == row["review_row_sha256"]
                for action in (self.approval or {}).get("runs", [])))
        return row["kind"] == "info" or (row["kind"] == "warning" and (row["issue_index"], row["issue_sha256"]) in self._warning_actions)

    def set_resolved_target_rows(self, rows):
        """Retain exact rows verified by the owner after an explicit target Save."""
        keys = ("run_id", "component_id", "review_row_sha256")
        checked = set()
        for row in deepcopy(rows):
            if (not isinstance(row, dict) or set(row) != set(keys)
                    or any(not isinstance(row[key], str) or not row[key] for key in keys)):
                raise ValueError("A saved target resolution needs its exact run, component and review row")
            checked.add(tuple(row[key] for key in keys))
        self._resolved_target_rows = checked
        self._refresh_rows()

    def unresolved_findings_reason(self):
        """Explain the shared human-resolution boundary without running checks."""
        if self.review is None:
            return "Wait for the bound findings list to finish loading."
        if self._unconfirmed_water_edits:
            return "Confirm the edited water line yourself, or remove it from the check first."
        if self._catalogue_refresh_pending:
            return "Refreshing findings from the original checks. Wait for the list to finish."
        if self._display_preferences_dirty:
            return "Refresh findings after the changed display choices before resolving warnings or checking again."
        remaining = sum(not self._resolved(row) for row in self._rows)
        if remaining:
            return (f"Resolve all {remaining} remaining findings in this list before Check again, "
                    "Adopt, or Apply and Build. Open Review gaps and findings to continue.")
        return ""

    def can_recheck(self):
        return (not self.busy() and not self._closing and not self.operation_waiting
                and not self.unresolved_findings_reason())

    def _refresh_rows(self):
        previous = self._selected
        rows = []
        for run in (self.review or {}).get("runs", []):
            for component in run["components"]:
                rows.append(dict(component, kind="gap", run_id=run["run_id"]))
        if ((self.review or {}).get("source", {}).get("painting_facies") is not None
                or (self._candidate or {}).get("painted_adoption_table") is not None):
            rows.append(dict(kind="source_state", run_id=(self.review or {}).get("source", {}).get("reference_run", "")))
        rows.extend(deepcopy(self._issues))
        self._rows = rows
        self.runs.blockSignals(True)
        self.runs.setRowCount(len(rows))
        selected = 0
        for index, row in enumerate(rows):
            if self._row_key(row) == previous:
                selected = index
            issue = row.get("issue") or {}
            description = ("Gap " + str(row["component_id"])[:12] if row["kind"] == "gap"
                           else "Your water-state choice" if row["kind"] == "source_state"
                           else str(issue.get("severity") or row["kind"]) + ": " + str(issue.get("code") or ""))
            message = ("Choose the water state in the sand in the facies source photo yourself. No value is taken from chat or handoff."
                       if row["kind"] == "source_state" else issue.get("message") or "Uncovered area inside this run's own frame")
            resolution = "Resolved" if self._resolved(row) else "Must be fixed" if row["kind"] == "error" else "Unresolved"
            if row["kind"] == "source_state" and self._acquisition_actions:
                action = self._acquisition_actions[-1]
                resolution = action["acquisition_state"] + ": " + action["reason"]
            if row["kind"] == "gap":
                action = next((a for a in (self.approval or {}).get("runs", [])
                    if (a["run_id"], a["component_id"]) == (row["run_id"], row["component_id"])), None)
                if action:
                    label = (classification_label(action["classification"])
                             if "classification" in action else "Classified areas")
                    resolution = label + ": " + action["reason"]
            values = (str(row.get("run_id") or "").upper(), description, row.get("pixels", ""), message, resolution)
            for column, value in enumerate(values):
                self.runs.setItem(index, column, QTableWidgetItem(str(value)))
        if rows:
            self.runs.setCurrentCell(selected, 0)
        self.runs.blockSignals(False)
        if rows:
            self._select(selected, 0, -1, -1)
        else:
            self.status.setText("The check has no unresolved findings.")
        self._buttons()

    def set_check_result(self, candidate, report):
        """Display every genuine issue; keep unchanged explicit warning choices."""
        self._set_result(candidate, report, catalogue=False)

    def set_catalogue_result(self, report):
        """Install a fresh verified initial-list report without completing preparation."""
        self._initial_issue_types = {
            (issue.get("severity"), issue.get("code")) for issue in report.get("issues") or []
        }
        self._catalogue_baseline = dict(pending=self.review is None,
            review_binding=deepcopy((self.review or {}).get("review_binding")))
        self._set_result(None, report, catalogue=True)

    def install_catalogue(self, review, report):
        """Install worker-authenticated review/report DATA after pure owner checks.

        The owner verifies the live request and context; the worker owns original
        source replay. This GUI boundary performs no file or approval reads.
        """
        if self.busy() or self._closing or self.review is None:
            return False
        owner_busy = getattr(self.parent(), "busy", None)
        if callable(owner_busy) and owner_busy():
            return False
        from .painted_analysis_exclusions import _json, _review_metadata, _same
        try:
            checked, owned_report = _review_metadata(review), _json(report)
            inputs = owned_report.get("input_bindings")
            if (not isinstance(self.review.get("review_binding"), dict)
                    or not isinstance(checked.get("review_binding"), dict) or not isinstance(inputs, dict)
                    or not _same(inputs.get("review"), self.review.get("review_binding"))
                    or any(not _same(checked.get(key), self.review.get(key))
                           for key in ("source", "map_version", "geometry_index"))
                    or not isinstance(owned_report.get("issues"), list)
                    or any(not isinstance(issue, dict) or issue.get("severity") not in {"error", "warning", "info"}
                           or (issue.get("details") is not None and not isinstance(issue["details"], dict))
                           for issue in owned_report["issues"])):
                return False
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        self._catalogue(checked, validate_context=False, refresh=False)
        self._resolved_target_rows = set()
        self.set_catalogue_result(owned_report)
        return True

    def _set_result(self, candidate, report, *, catalogue):
        from .painted_analysis_exclusions import issue_identity
        self.operation_waiting = False
        self._catalogue_refresh_pending = False
        self._candidate, self._report = deepcopy(candidate), deepcopy(report)
        self._check_report_installed = not catalogue
        self._update_coverage()
        self._display_preferences_dirty = False
        if not catalogue:
            self._preparation_changed = False
        # Match the original Check/Store error and warning predicates. Other
        # severities remain visible information with their original issue intact.
        self._issues = [dict(kind=issue.get("severity") if issue.get("severity") in ("error", "warning") else "info", issue=deepcopy(issue), issue_index=index,
            issue_sha256=issue_identity(issue), run_id=(issue.get("details") or {}).get("run_id") or issue.get("run_id", ""))
            for index, issue in enumerate(report.get("issues") or [])]
        # Unchanged literal warnings remain the same human decisions even if
        # unrelated resolved issues move their positions. Duplicate identities
        # keep occurrence order and require unchanged multiplicity.
        old_by_sha = {}
        for action in self._warning_actions.values():
            old_by_sha.setdefault(action["issue_sha256"], []).append(action)
        active_by_sha = {}
        for row in self._issues:
            if row["kind"] == "warning":
                active_by_sha.setdefault(row["issue_sha256"], []).append(row)
        retained = {}
        for sha, rows in active_by_sha.items():
            actions = sorted(old_by_sha.get(sha, []), key=lambda value: value["issue_index"])
            if len(actions) == len(rows):
                for action, row in zip(actions, rows):
                    retained[(row["issue_index"], sha)] = dict(action, issue_index=row["issue_index"])
            else:
                for row in rows:
                    key = (row["issue_index"], sha)
                    if key in self._warning_actions:
                        retained[key] = self._warning_actions[key]
        self._warning_actions = retained
        self._refresh_rows()

    def _update_coverage(self):
        self.coverage_defects = []
        if self._check_report_installed and self._initial_issue_types is not None:
            self.coverage_defects = [deepcopy(issue) for issue in (self._report or {}).get("issues") or []
                if (issue.get("severity"), issue.get("code")) not in self._initial_issue_types]
        self.coverage_note.setVisible(bool(self.coverage_defects))
        self.coverage_note.setText("Findings list coverage defect: the initial list did not include these Check issue types: "
            + ", ".join(sorted({str(issue.get("code") or "unnamed issue") for issue in self.coverage_defects}))
            + ". Report this as a defect in the findings list." if self.coverage_defects else "")

    def _select(self, index, _column, _previous_row, _previous_column):
        row = self._rows[index] if 0 <= index < len(self._rows) else None
        if row is None:
            return
        key = self._row_key(row)
        self.details.setPlainText(json.dumps(row.get("issue") or row, indent=2, ensure_ascii=False))
        self._buttons()
        if key == self._selected:
            return
        self._selected = key
        self._comparison_key = self._base_packet = None
        if row["kind"] in {"warning", "error", "source_state"}:
            self.reason.clear()
        self.packet = self._rgb = self._point_native = None
        self._pending = None
        self._clear_photo()
        run = row.get("run_id")
        known = {value["run_id"] for value in (self.review or {}).get("runs", [])}
        if row["kind"] == "source_state" or (row["kind"] != "gap" and (row.get("issue") or {}).get("code") in {
                "run_geometry_acquisition_unknown", "indicator_facies_carryover_risk"}):
            run = (self.review or {}).get("source", {}).get("reference_run")
        if run not in known:
            self._preview_key = None
            self.status.setText("Read the original finding and fix or resolve its cause in the GUI.")
            self._buttons()
            return
        self._preview_key = (run, row.get("component_id"))
        self.status.setText("Reading the bound original photo and outline for " + run.upper() + "…")
        self._pending = self._preview_key
        if self.thread is None:
            target, self._pending = self._pending, None
            self._load(target)
        self._buttons()

    def _load(self, target):
        if self._closing:
            return
        if self.thread is not None:
            self._pending = target
            return
        review = deepcopy(self.review)
        self._start(("preview", target), lambda: self._read_preview(review, *target))

    def _clear_photo(self):
        for item in self._overlays:
            self.photo.scene().removeItem(item)
        self._overlays = []
        self.photo.show_image(None)
        self.water_surface.set_packet(None, display_grid=None, active=False)

    def _preview(self, packet):
        if packet["run_id"] != self._preview_key[0] or packet.get("component_id") != self._preview_key[1]:
            raise ValueError("The preview belongs to another run")
        self._rgb = _rgb(packet)
        self._plain_image_shape = self._rgb.shape
        mask = np.asarray(packet["gap_mask"])
        self._gap_lines = _gap_contours(mask)
        self._layer_lines = _gap_contours(packet["layer_boundary_mask"])
        self._point_native = None
        if np.any(mask):
            row, column = np.unravel_index(np.argmax(mask), mask.shape)
            self._point_native = np.array([[column, row]], np.float64)
        elif self._current_row() is not None and self._current_row()["kind"] == "gap":
            raise ValueError("The selected component mask is empty")
        # Water controls emit redraw signals during setup. Admit every actual
        # overlay transform before publishing the packet to those GUI slots.
        for points in [*self._gap_lines, *self._layer_lines,
                       packet["own_frame_native_xy"], packet["transported_frame_native_xy"]]:
            _project(points, packet["native_to_photo_h"])
        self.packet = dict(packet)
        self._base_packet = dict(packet)
        self._image_options(packet)
        self._set_water_surface(packet)
        self._display(reset=True)
        self.status.setText(self.packet["run_id"].upper() + ": magenta = gap outline, cyan = own frame, "
            "yellow = transported frame. Double-click a gap to choose where to paint. Viewing approves nothing.")

    def _image_options(self, packet):
        self.images.blockSignals(True)
        self.images.clear()
        self.images.addItem("Bound review image", "current")
        declared = {row["image_id"]: row for row in packet.get("image_choices", [])}
        for image_id, label in (("clearwater", "Clear water"), ("first-experiment", "First experiment image"), ("previous-run", "Previous run")):
            row = declared.get(image_id) or {}
            available = bool(row.get("available") and self._comparison_reader_for_image(image_id)
                             and packet.get("display_grid") is not None)
            self.images.addItem(label + ("" if available else " – not verified"), image_id)
            item = self.images.model().item(self.images.count()-1)
            item.setEnabled(available)
            item.setToolTip(str(row.get("reason") or "A verified image and alignment binding is missing."))
        self.images.blockSignals(False)
        position = packet.get("position_readout")
        if position is None:
            self.position.setText("camera registration (map placement): no verified placement binding yet.")
        else:
            if position.get("method") != "camera registration (map placement)":
                raise ValueError("Position measurements must declare camera registration (map placement)")
            lines=["camera registration (map placement)"]
            for key,label in (("reference","From reference"),("previous","From previous")):
                value=position.get(key)
                if value is None:
                    lines.append(label+": unavailable")
                else:
                    lines.append(f"{label}: {value['shift_mm']:.3f} mm, {value['rotation_degrees']:.4f} degrees")
            self.position.setText("; ".join(lines)+". Correction is always directly from the reference.")
        tolerances = self.position_tolerances or packet.get("position_tolerances")
        for control in (self.shift_tolerance,self.angle_tolerance):
            control.setVisible(tolerances is not None)
        if tolerances is not None:
            self.shift_tolerance.setValue(tolerances["shift_mm"])
            self.angle_tolerance.setValue(tolerances["rotation_degrees"])
            self.tolerance_note.setText("Your warning tolerance:")
        self.apply_tolerances.setEnabled(tolerances is not None and position is not None)

    def _next_image(self):
        if self.busy():
            return
        for offset in range(1,self.images.count()+1):
            index=(self.images.currentIndex()+offset)%self.images.count()
            if self.images.model().item(index).isEnabled():
                self.images.setCurrentIndex(index)
                return

    def _comparison_reader_for_image(self, image_id):
        """Retain the original reader unless a presentation supplies a specific role."""
        return self._comparison_reader

    def _image_selected(self, *_args):
        if self._base_packet is None or self.busy() or self._closing:
            return
        image_id=self.images.currentData()
        if image_id == "current":
            self.packet=dict(self._base_packet)
            self._rgb=_rgb(self.packet)
            self._set_water_surface(self.packet)
            self._display()
            return
        reader = self._comparison_reader_for_image(image_id)
        if reader is None or not self.images.model().item(self.images.currentIndex()).isEnabled():
            return
        self._comparison_key=(self._preview_key,image_id)
        review,target=deepcopy(self.review),self._preview_key
        self._start(("comparison",self._comparison_key),lambda:reader(review,*target,image_id))
        self._buttons()

    def _comparison(self, packet):
        base=self._base_packet
        if (base is None or base.get("display_grid") is None
                or packet.get("display_grid") != base["display_grid"]
                or not np.array_equal(packet["native_to_photo_h"],base["native_to_photo_h"])):
            raise ValueError("The comparison image lacks the exact common run display grid")
        rgb=_rgb(packet)
        if rgb.shape != self._plain_image_shape:
            raise ValueError("The aligned comparison photo has another declared display shape")
        support=packet.get("valid_support")
        if support is not None:
            support=np.asarray(support)
            if support.dtype!=np.bool_ or support.shape!=rgb.shape[:2]:
                raise ValueError("The comparison image lacks its exact boolean display support")
            rgb=rgb.copy()
            rgb[~support]=64  # Display-only neutral area, not source photo pixels.
        self.packet=dict(base,image_rgb=packet["image_rgb"],image_encoding=packet["image_encoding"],
                         water_surface=deepcopy(packet.get("water_surface")))
        self._rgb=rgb
        if support is not None and not np.all(support):
            self.status.setText("The grey area has no support in the comparison image. This affects display only.")
        self._set_water_surface(self.packet)
        self._display()  # Preserve the current transform and center exactly.

    def _reset_comparison(self):
        if self._base_packet is not None:
            self.images.blockSignals(True)
            self.images.setCurrentIndex(0)
            self.images.blockSignals(False)
            self.packet=dict(self._base_packet)
            self._rgb=_rgb(self.packet)
            self._set_water_surface(self.packet)
            self._display()

    def _tolerances(self):
        if self.apply_tolerances.isEnabled() and not self.busy() and not self.operation_waiting and not self._closing:
            self.position_tolerances = dict(shift_mm=self.shift_tolerance.value(),rotation_degrees=self.angle_tolerance.value())
            self._invalidate_display_report()
            self.position_tolerances_changed.emit(dict(shift_mm=self.shift_tolerance.value(),
                rotation_degrees=self.angle_tolerance.value(), **self._human_action()))

    def _invalidate_display_report(self):
        self._candidate = self._report = None
        self._warning_actions = {}
        self._display_preferences_dirty = True
        self._refresh_rows()
        self.status.setText('Display choices changed. Refresh findings, then resolve the current warnings before Check again.')

    def _refresh_findings(self):
        if (self.busy() or self._closing or self.operation_waiting or self.review is None
                or not self._display_preferences_dirty or self._unconfirmed_water_edits):
            return
        self._catalogue_refresh_pending = self.operation_waiting = True
        self.status.setText('Refreshing findings from the original checks…')
        self._buttons()
        self.refresh_findings_requested.emit()

    def catalogue_refresh_failed(self, message):
        """Leave dirty choices unconfirmed while allowing an explicit refresh retry."""
        self._catalogue_refresh_pending = self.operation_waiting = False
        self.status.setText('The findings list could not be refreshed: ' + str(message))
        self._buttons()

    def _water_tolerances(self, values):
        if self._closing or self.busy() or self.operation_waiting:
            return
        self._invalidate_display_report()
        self.water_surface_tolerances_changed.emit(deepcopy(values))

    def _line_item(self, lines, matrix, colour, *, closed=True):
        item = QGraphicsPathItem(_path(lines, matrix, closed=closed))
        pen = QPen(QColor(colour), 2)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(Qt.BrushStyle.NoBrush)
        item.setZValue(2)
        self.photo.scene().addItem(item)
        self._overlays.append(item)

    def _brightness_changed(self, value):
        self.brightness_value.setText(f"{value}%")
        self._display()

    def _display(self, *_args, reset=False):
        if self.packet is None:
            return
        transform = self.photo.transform()
        # Recentring through centerOn rounds an already quantized viewport
        # centre again. Keep the original integer pan when replacing pixels.
        pan = (self.photo.horizontalScrollBar().value(),
               self.photo.verticalScrollBar().value())
        scene_rect, viewport = self.photo.sceneRect(), self.photo.viewport().rect()
        center = self.photo.mapToScene(viewport.center())
        for item in self._overlays:
            self.photo.scene().removeItem(item)
        self._overlays = []
        brightness = self.brighten.value()
        rgb = (np.clip(np.rint(self._rgb.astype(np.float32) * (brightness / 100.)), 0, 255).astype(np.uint8)
               if brightness != 100 else self._rgb)
        self.photo.show_rgb(rgb, refit=reset)
        matrix = self.packet["native_to_photo_h"]
        if self.outline.isChecked():
            self._line_item(self._gap_lines, matrix, "magenta")
        if self.own_frame.isChecked():
            self._line_item([self.packet["own_frame_native_xy"]], matrix, "cyan")
        if self.transported_frame.isChecked():
            self._line_item([self.packet["transported_frame_native_xy"]], matrix, "yellow")
        if self.layer_boundaries.isChecked():
            self._line_item(self._layer_lines, matrix, "white")
        water = self.water_surface
        if water.outline.isChecked() and water.packet is not None and water.line_photo_xy is not None:
            self._line_item([water.line_photo_xy], water.photo_to_display_h, "deepskyblue", closed=False)
            bolts = water.packet["basis"].get("left_bolts_photo_xy")
            if bolts is not None:
                from PySide6.QtWidgets import QGraphicsSimpleTextItem
                for point, height in zip(_project(bolts, water.photo_to_display_h), (0,195,390)):
                    x,y=point
                    self._line_item([[[x-3,y],[x+3,y]],[[x,y-3],[x,y+3]]],np.eye(3),"deepskyblue",closed=False)
                    label=QGraphicsSimpleTextItem(f"{height} mm")
                    label.setBrush(QColor("deepskyblue"))
                    label.setPos(float(x+5),float(y))
                    label.setZValue(3)
                    self.photo.scene().addItem(label)
                    self._overlays.append(label)
        if not reset:
            self.photo.setTransform(transform)
            self.photo._update_pan_margins()
            if self.photo.sceneRect() == scene_rect and self.photo.viewport().rect() == viewport:
                self.photo.horizontalScrollBar().setValue(pan[0])
                self.photo.verticalScrollBar().setValue(pan[1])
            else:
                self.photo.centerOn(center)

    def _set_water_surface(self, packet):
        water=deepcopy(packet.get("water_surface"))
        if water is not None:
            run=water["basis"]["run_id"]
            if run != packet["run_id"]:
                self.water_surface.set_packet(None,display_grid=None,active=False)
                self.water_surface.status.setText("The water-line basis belongs to another run.")
                return
            previous=self.water_surface_confirmations.get(run)
            if previous is not None:
                water["action"]=deepcopy(previous["action"])
        self.water_surface.set_packet(water,display_grid=packet.get("display_grid"),
            active=self.images.currentData()=="clearwater" and not self._closing)

    def _water_pick_requested(self, armed):
        self.photo.pick_single_point=bool(armed) and not self._closing

    def _water_point_clicked(self, xy):
        water=self.water_surface
        if self._closing or self.busy() or water.photo_to_display_h is None:
            return
        try:
            point=_project([xy],np.linalg.inv(water.photo_to_display_h))[0]
            water.select_photo_point(point.tolist())
        except (TypeError,ValueError,np.linalg.LinAlgError) as error:
            water.status.setText("The click could not be bound to the water-line image: "+str(error))

    def _water_basis(self, basis):
        self.water_surface_bases[basis["run_id"]]=deepcopy(basis)

    def _water_confirmed(self, record):
        if self._closing or self.busy() or self.operation_waiting:
            return
        self.water_surface_confirmations[record["action"]["run_id"]]=deepcopy(record)
        self._unconfirmed_water_edits.discard(record['action']['run_id'])
        self._invalidate_display_report()
        self.water_surface_confirmed.emit(deepcopy(record))

    def _water_line_edited(self, run):
        self.water_surface_confirmations.pop(run,None)
        self._unconfirmed_water_edits.add(run)
        self._invalidate_display_report()
        self.water_surface_edited.emit(run)
        self.status.setText('You changed the water line. Confirm the new line before checking again, or remove it from the check.')

    def _clear_water_confirmations(self):
        if self._closing or self.busy() or self.operation_waiting or not (self.water_surface_confirmations or self._unconfirmed_water_edits):
            return
        self.water_surface_confirmations = {}
        self._unconfirmed_water_edits = set()
        if self.packet is not None:
            self._set_water_surface(self.packet)
        self._invalidate_display_report()
        self.water_surfaces_cleared.emit()
        self.status.setText('Previous water-line confirmations were removed from this check. Saved history is retained; check again.')

    def _point_selected(self, xy):
        if self.packet is None:
            return
        try:
            point = _project([xy], np.linalg.inv(self.packet["native_to_photo_h"]))
            column, row = np.rint(point[0]).astype(int)
            mask = self.packet["gap_mask"]
            if not (0 <= row < mask.shape[0] and 0 <= column < mask.shape[1] and mask[row, column]):
                self.status.setText("Choose a point in the displayed gap. Nothing has changed.")
                return
            self._point_native = np.array([[column, row]], np.float64)
            self.status.setText(f"Painting point selected in {self.packet['run_id'].upper()}: row {row}, column {column}.")
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self.status.setText("The painting point could not be bound: " + str(error))

    def _buttons(self, *_args):
        idle = not self.busy() and not self._closing and not self.operation_waiting
        self.water_surface.set_active(idle and self.images.currentData()=="clearwater")
        self.water_surface.apply.setEnabled(idle and self.water_surface.packet is not None)
        has_water=bool(self.water_surface_confirmations or self._unconfirmed_water_edits)
        self.clear_water_confirmations.setVisible(has_water)
        self.clear_water_confirmations.setEnabled(idle and has_water)
        row = self._current_row()
        selected_gap = row is not None and row["kind"] == "gap"
        reason = bool(self.reason.text().strip()) and bool(self.actor)
        classification = self.classification.currentData()
        class_ready = classification is not None and (classification_key(classification) != "other" or bool(self.other_text.text().strip()))
        gap_ready = idle and self.review is not None and self.packet is not None and reason and class_ready
        self.approve_one.setEnabled(gap_ready and selected_gap)
        self.approve_all.setEnabled(gap_ready and selected_gap)
        self.paint.setEnabled(idle and selected_gap and self.packet is not None)
        self.refresh_findings.setVisible(self._display_preferences_dirty)
        self.refresh_findings.setEnabled(idle and self.review is not None
            and self._display_preferences_dirty and not self._unconfirmed_water_edits)
        self.refresh_findings.setToolTip("Refresh the initial findings from the original checks for the changed display choices.")
        self.recheck.setEnabled(self.can_recheck())
        self.recheck.setToolTip(self.unresolved_findings_reason() or
            ("Run the original Check to confirm the resolved findings." if idle else "Wait for the current operation to finish."))
        self.approve_warning.setEnabled(idle and not self._display_preferences_dirty and reason and row is not None and row["kind"] == "warning"
            and ((row.get("issue") or {}).get("code") != "run_geometry_acquisition_unknown" or bool(self._acquisition_actions)))
        self.fix.setEnabled(idle and row is not None and row["kind"] in {"warning", "error"})
        state_row = row is not None and (row["kind"] == "source_state" or (row.get("issue") or {}).get("code") in {
            "run_geometry_acquisition_unknown", "indicator_facies_carryover_risk"})
        self.set_state.setEnabled(idle and reason and state_row and self.source_state.currentData() is not None)
        self.adopt.setEnabled(self.can_adopt())
        readiness = (self.unresolved_findings_reason(), self.recheck.isEnabled(), self.adopt.isEnabled(),
                     self.busy(), self._closing)
        if readiness != self._last_readiness:
            self._last_readiness = readiness
            self.readiness_changed.emit()

    def can_adopt(self):
        return bool(self._candidate is not None and self._report is not None
            and not self._needs_preparation()
            and not self._unconfirmed_water_edits
            and not self.busy() and not self._closing and not self.operation_waiting
            and all(self._resolved(row) for row in self._rows)
            and not any(issue.get("severity") == "error" for issue in self._report.get("issues", [])))

    def _needs_preparation(self):
        return self._preparation_changed or bool(self._preparation_dirty and self._preparation_dirty())

    def _human_action(self):
        return dict(actor=self.actor, approved_utc=datetime.now(timezone.utc).isoformat(),
                    reason=self.reason.text().strip(), decision_ref=self.decision_ref)

    def _approve(self, all_areas):
        row = self._current_row()
        if (not self.actor or not self.reason.text().strip() or self.busy() or self.review is None
                or self.packet is None or self._closing or not row or row["kind"] != "gap"):
            return
        classification = self.classification.currentData()
        if classification is None or classification_key(classification) == "other" and not self.other_text.text().strip():
            return
        selected = [dict(run_id=value["run_id"], component_id=value["component_id"])
            for value in (self._rows if all_areas else [row]) if value["kind"] == "gap"]
        try:
            approval = self._make_approval(deepcopy(self.review), selected, **self._human_action(),
                classification=classification_key(classification), other_text=self.other_text.text().strip(), previous=deepcopy(self.approval))
        except (KeyError, TypeError, ValueError, OSError) as error:
            self.status.setText("The choice could not be bound to the gap review: " + str(error))
            return
        self.approval = deepcopy(approval)
        self._preparation_changed = True
        self._candidate = self._report = None
        self._refresh_rows()
        self.status.setText("Your explicit area choices are recorded. Check again before any adoption.")
        self.approval_changed.emit(deepcopy(self.approval))
        self._buttons()

    def _approve_warning(self):
        row = self._current_row()
        if not row or row["kind"] != "warning" or not self.approve_warning.isEnabled():
            return
        self._warning_actions[(row["issue_index"], row["issue_sha256"])] = dict(
            issue_index=row["issue_index"], issue_sha256=row["issue_sha256"], **self._human_action())
        self._refresh_rows()

    def _set_state(self):
        if not self.set_state.isEnabled():
            return
        action = dict(acquisition_state=self.source_state.currentData(), **self._human_action())
        self._acquisition_actions = [deepcopy(action)]
        self._candidate = self._report = None
        self._buttons()
        self.state_requested.emit(action)

    def _fix(self):
        row = self._current_row()
        if row and row["kind"] in {"warning", "error"} and not self.busy():
            self.fix_requested.emit(deepcopy(row["issue"]))

    def _adopt(self):
        if self.can_adopt():
            records = dict(warnings=[deepcopy(self._warning_actions[(row["issue_index"], row["issue_sha256"])])
                for row in self._issues if row["kind"] == "warning"],
                acquisition=deepcopy(self._acquisition_actions), analysis_exclusions=deepcopy(self.approval))
            self.adopt_requested.emit(records)

    def _paint(self):
        row = self._current_row()
        if (row is None or row["kind"] != "gap" or self.packet is None
                or self._point_native is None or self.busy() or self._closing):
            return
        self.paint_requested.emit(dict(review=deepcopy(self.review), run_id=row["run_id"],
            component_id=row["component_id"], point_native_xy=self._point_native[0].tolist()))

    def _recheck(self):
        if not self.can_recheck():
            self.status.setText(self.unresolved_findings_reason() or "Wait for the current operation to finish.")
            return
        if not self._needs_preparation() and (self._display_preferences_dirty or self._preparation_dirty is not None):
            self.display_recheck_requested.emit()
        else:
            self.recheck_requested.emit(deepcopy(self.approval))

    def wait_for_workers(self):
        """Shutdown barrier only; interactive Close/Escape never call this."""
        self._closing, self._pending = True, None
        self._buttons()
        thread = self.thread
        if thread is not None:
            # _Read quits directly in its own thread, so shutdown needs no
            # queued GUI delivery to retire the worker and discard its data.
            thread.wait()
            self._retired(thread)
        else:
            self.close()

    def reject(self):
        self._closing, self._pending = True, None
        self._buttons()
        if self.busy():
            self.status.setText("The window will close when the current read finishes.")
            return
        super().reject()

    def closeEvent(self, event):
        self._closing, self._pending = True, None
        self._buttons()
        if self.busy():
            self.status.setText("The window will close when the current read finishes.")
            event.ignore()
            return
        super().closeEvent(event)



def focus_issue_controls(view, issue):
    """Show existing editing controls and the full original cause, without edits."""
    code = str(issue.get("code") or "")
    if "acquisition" in code or code == "indicator_facies_carryover_risk":
        view.step_use_map.set_expanded(True)
    elif any(part in code for part in ("facies", "porosity", "permeability", "region")):
        view.step_facies.set_expanded(True)
        view.facies_table.setFocus()
    elif any(part in code for part in ("frame", "rig", "injector", "width", "height")):
        view.step_frame.set_expanded(True)
        view.rig_panel.setFocus()
    else:
        view.step_use_map.set_expanded(True)
        view.run_preview.geometry_button.setFocus()
    view.use_map_progress.setText(str(issue.get("message") or "") + "\n"
        + json.dumps(issue.get("details") or {}, ensure_ascii=False, sort_keys=True)
        + "\nFix the cause in the existing controls and check again. Errors cannot be accepted.")
