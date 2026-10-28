"""Question-by-question presentation of the original painting review.

This subclass owns navigation and presentation only.  Original readers,
controller signals, human-action methods and Check/Adopt predicates remain
in GapReviewDialog.  A painted region is only a request for the owning typed
action builder; navigating or displaying a draft never approves that region.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QStackedWidget,
    QToolButton, QVBoxLayout, QWidget,
)

from .painting_gap_review_ui import GapReviewDialog


_FINDING_STAGES = ("deviation", "source_water", "position", "warnings")
_STAGES = ("deviation", "source_water", "position", "water_line", "warnings", "check", "adopt")
_EMPTY = {
    "deviation": "No uncovered areas need a decision.",
    "source_water": "No source-photo water-state question is recorded.",
    "position": "No frame or position finding needs a decision.",
    "warnings": "No other warnings or errors need a decision.",
}


class GuidedGapReviewDialog(GapReviewDialog):
    """Reuse original authority while asking one visible question at a time."""

    region_confirmation_requested = Signal(object)
    region_preview_reset_requested = Signal(object)
    draft_choices_changed = Signal(object)
    map_fix_requested = Signal(object)

    def __init__(self, source, *, actor, parent=None,
                 deviation_editor_factory=None, region_draft_factory=None,
                 region_snapshot_collector=None, region_result_validator=None,
                 water_surface_comparison_reader=None,
                 clearwater_review_builder=None,
                 class_choices=None,
                 sand_class=None, **options):
        # Base construction calls virtual _buttons before its controls exist.
        self._guided_ready = False
        self._last_started_read_key = None
        self._pending_photo_fit = None
        self._photo_fit_size = None
        self._photo_fit_queued = False
        self._guidance_updating = False
        self._changing_question = False
        self._question_reasons = {}
        self._question_other_text = {}
        self._reason_key = None
        self._questions = []
        self._question_index = 0
        self._question_key = None
        self._deviation_group_data = None
        self._first_group_catalogue_installed = False
        self._deviation_groups = []
        self._group_by_row = {}
        self._group_choices = {}
        self._independent_deviations = set()
        self._group_split_pending = None
        self._group_loading_key = None
        self._group_problem = ""
        self._selecting_group_member = False
        self._water_run = None
        self._water_surface_comparison_reader = water_surface_comparison_reader
        self._clearwater_review_builder = clearwater_review_builder
        self._water_source_request = None
        self.saved_clearwater_review = None
        self._region_draft_factory = region_draft_factory
        self._region_snapshot_collector = region_snapshot_collector
        self._region_result_validator = region_result_validator
        self._region_submission = None
        self._confirmed_region_request = None
        self._region_reset_submission = None
        self._pending_region_contexts = set()
        self._notified_pending_contexts = set()
        self._class_choices = class_choices
        self._sand_class = sand_class
        self._drafts = {}
        self._editor_context = None
        self._deviation_draft = None
        self._editor_request = None
        self._region_problem = ""
        self.deviation_editor = None
        super().__init__(source, actor=actor, parent=parent, **options)
        self._photo_fit_timer = QTimer(self)
        self._photo_fit_timer.setSingleShot(True)
        self._photo_fit_timer.timeout.connect(self._fit_initial_photo)
        self.photo.viewport().installEventFilter(self)
        self.setWindowTitle("Review the transferred sand map")
        self._build_guided_layout(deviation_editor_factory)
        self._guided_ready = True
        self.reason.textChanged.connect(self._remember_reason)
        self.other_text.textChanged.connect(self._remember_reason)
        self._rebuild_questions()
        self._buttons()

    def _start(self, key, function):
        super()._start(key, function)
        self._last_started_read_key = key

    def _photo_fit_ready(self):
        pending = self._pending_photo_fit
        return bool(pending is not None and self._guided_ready and not self._closing
            and not self.busy() and not self.operation_waiting
            and self.guided_stage in {"source_water", "position", "water_line", "warnings"}
            and pending[0] == self._preview_key and pending[1] is self._base_packet
            and self.packet is not None and self.image_stack.currentWidget() is self.photo
            and self.photo.viewport().isVisible())

    def _queue_photo_fit(self):
        if self._closing:
            self._cancel_photo_fit()
        elif not self._photo_fit_queued and self._photo_fit_ready():
            self._photo_fit_queued = True
            self._photo_fit_timer.start(0)

    def _fit_initial_photo(self):
        self._photo_fit_queued = False
        if self._closing:
            self._cancel_photo_fit()
        elif self._photo_fit_ready():
            viewport = self.photo.viewport()
            size = (viewport.width(), viewport.height())
            if size != self._photo_fit_size:
                self._photo_fit_size = size
                self._queue_photo_fit()
                return
            self._pending_photo_fit = None
            self.photo.fit()  # Original whole pixmap extent, never overlay bounds.

    def _cancel_photo_fit(self):
        self._pending_photo_fit = None
        self._photo_fit_timer.stop()
        self._photo_fit_queued = False

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.photo.viewport():
            if event.type() in {QEvent.Type.Show, QEvent.Type.Resize}:
                self._queue_photo_fit()
            elif event.type() in {QEvent.Type.Wheel, QEvent.Type.MouseButtonPress}:
                self._cancel_photo_fit()  # A human view gesture owns the view now.
        return super().eventFilter(watched, event)

    def _build_guided_layout(self, editor_factory):
        """Move existing controls, never copy their action or approval state."""
        retained = (
            "runs", "details", "status", "photo", "images", "brighten",
            "brightness_value", "outline", "own_frame", "transported_frame",
            "layer_boundaries", "position", "tolerance_note", "shift_tolerance",
            "angle_tolerance", "apply_tolerances", "water_surface",
            "clear_water_confirmations", "classification", "other_text",
            "source_state", "set_state", "reason", "approve_one", "approve_all",
            "paint", "refresh_findings", "coverage_note", "recheck", "approve_warning", "fix", "adopt",
        )
        for name in retained:
            control = getattr(self, name)
            control.hide()
            control.setParent(self)
        # Same-parent widgets can still have items in the old layouts.
        # Removing those items must never schedule their controls for deletion.
        protected = {getattr(self, name) for name in retained}
        layout = self.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                if item.widget() not in protected:
                    item.widget().deleteLater()
            elif item.layout() is not None:
                self._discard_layout(item.layout(), protected)

        self.question_progress = QLabel()
        self.question_title = QLabel()
        self.question_title.setWordWrap(True)
        self.question_title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.question_explanation = QLabel()
        self.question_explanation.setWordWrap(True)
        self.guided_status = QLabel()
        self.guided_status.setWordWrap(True)
        self.skipped_steps_note = QLabel()
        self.skipped_steps_note.setWordWrap(True)
        for control in (self.question_progress, self.question_title,
                        self.question_explanation, self.guided_status,
                        self.skipped_steps_note):
            control.setTextFormat(Qt.TextFormat.PlainText)
            layout.addWidget(control)

        self.reason_panel = QWidget()
        reason_layout = QHBoxLayout(self.reason_panel)
        reason_layout.setContentsMargins(0, 0, 0, 0)
        reason_layout.addWidget(QLabel("Reason for this answer"))
        reason_layout.addWidget(self.reason, 1)
        layout.addWidget(self.reason_panel)

        self.photo_toolbar = QWidget()
        toolbar = QHBoxLayout(self.photo_toolbar)
        toolbar.setContentsMargins(0, 0, 0, 0)
        self.experiment_photo = QPushButton("Show experiment image (Alt+I)")
        self.clearwater_photo = QPushButton("Show clear-water image (Alt+I)")
        self.experiment_photo.clicked.connect(lambda: self._request_photo("current"))
        self.clearwater_photo.clicked.connect(lambda: self._request_photo("clearwater"))
        toolbar.addWidget(self.experiment_photo)
        toolbar.addWidget(self.clearwater_photo)
        toolbar.addWidget(QLabel("Brightness (display only)"))
        toolbar.addWidget(self.brighten, 1)
        toolbar.addWidget(self.brightness_value)
        layout.addWidget(self.photo_toolbar)
        self.photo_availability = QLabel()
        self.photo_availability.setWordWrap(True)
        layout.addWidget(self.photo_availability)

        self.image_stack = QStackedWidget()
        self.image_stack.addWidget(self.photo)
        self.photo_placeholder = QLabel("The verified photograph is loading.")
        self.photo_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_placeholder.setWordWrap(True)
        self.image_stack.addWidget(self.photo_placeholder)
        self.region_unavailable = QLabel()
        self.region_unavailable.setWordWrap(True)
        self._create_deviation_editor(editor_factory)

        self.question_pages = QStackedWidget()
        self._page_indices = {}
        for stage in ("loading", *_STAGES):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            self._page_indices[stage] = self.question_pages.addWidget(page)
            if stage == "deviation":
                self.deviation_group_summary = QLabel()
                self.deviation_group_summary.setWordWrap(True)
                page_layout.addWidget(self.deviation_group_summary)
                self.deviation_group_choice = QComboBox()
                self.deviation_group_choice.currentIndexChanged.connect(self._select_group_member)
                page_layout.addWidget(self.deviation_group_choice)
                page_layout.addWidget(self.region_unavailable)
                self.region_transfer_note = QLabel()
                self.region_transfer_note.setWordWrap(True)
                self.region_transfer_note.setTextFormat(Qt.TextFormat.PlainText)
                page_layout.addWidget(self.region_transfer_note)
                self.region_preview_reset = QPushButton("Start this area without the suggestion")
                self.region_preview_reset.setAutoDefault(False)
                self.region_preview_reset.clicked.connect(self._reset_region_preview)
                page_layout.addWidget(self.region_preview_reset)
                page_layout.addWidget(QLabel("If you marked Other, describe what you see in this area."))
                self.other_text.setPlaceholderText("Description of the area marked Other")
                page_layout.addWidget(self.other_text)
            elif stage == "source_water":
                page_layout.addWidget(QLabel("What is in the sand in the photo used to draw the map?"))
                page_layout.addWidget(self.source_state)
                self.set_state.setText("Use this water-state answer")
                page_layout.addWidget(self.set_state)
            elif stage == "position":
                self._position_water_layout = page_layout
                self.own_frame.setText("Outline this run's measured frame")
                self.transported_frame.setText("Outline the frame transferred from the reference map")
                page_layout.addWidget(self.own_frame)
                page_layout.addWidget(self.transported_frame)
                page_layout.addWidget(self.position)
                self.tolerance_note.setText("These limits flag a camera-placement difference; they do not decide which frame is physically correct.")
                page_layout.addWidget(self.tolerance_note)
                row = QHBoxLayout()
                row.addWidget(QLabel("Flag shifts larger than"))
                row.addWidget(self.shift_tolerance)
                row.addWidget(QLabel("and rotations larger than"))
                row.addWidget(self.angle_tolerance)
                page_layout.addLayout(row)
                self.apply_tolerances.setText("Check these position-warning limits")
                page_layout.addWidget(self.apply_tolerances)
                page_layout.addWidget(self.water_surface)
                page_layout.addWidget(self.clear_water_confirmations)
            elif stage == "water_line":
                self._measurement_water_layout = page_layout
                run_row = QHBoxLayout()
                run_row.addWidget(QLabel("Review the water line in"))
                self.water_run_choice = QComboBox()
                self.water_run_choice.currentIndexChanged.connect(self._select_water_run)
                run_row.addWidget(self.water_run_choice, 1)
                page_layout.addLayout(run_row)
                self.water_line_note = QLabel()
                self.water_line_note.setWordWrap(True)
                page_layout.addWidget(self.water_line_note)
                self.water_source_button = QPushButton("Use the recorded water photograph…")
                self.water_source_button.clicked.connect(self._prepare_water_source_review)
                page_layout.addWidget(self.water_source_button)
                self.water_source_note = QLabel()
                self.water_source_note.setWordWrap(True)
                self.water_source_note.setTextFormat(Qt.TextFormat.PlainText)
                page_layout.addWidget(self.water_source_note)
            elif stage == "warnings":
                page_layout.addWidget(QLabel("Read this finding before accepting a warning or opening its existing correction controls."))
            elif stage == "check":
                self.check_explanation = QLabel()
                self.check_explanation.setWordWrap(True)
                page_layout.addWidget(self.check_explanation)
                page_layout.addWidget(self.recheck)
            elif stage == "adopt":
                self.adopt_explanation = QLabel()
                self.adopt_explanation.setWordWrap(True)
                page_layout.addWidget(self.adopt_explanation)
                self.adopt.setText("Adopt this checked map…")
                page_layout.addWidget(self.adopt)
            page_layout.addStretch(1)

        self.content_splitter = QSplitter(Qt.Orientation.Vertical)
        self.content_splitter.addWidget(self.image_stack)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.question_pages)
        self.content_splitter.addWidget(scroll)
        self.content_splitter.setStretchFactor(0, 4)
        self.content_splitter.setStretchFactor(1, 1)
        layout.addWidget(self.content_splitter, 1)

        self.issue_actions = QWidget()
        issue_layout = QHBoxLayout(self.issue_actions)
        issue_layout.setContentsMargins(0, 0, 0, 0)
        self.approve_warning.setText("Accept this warning with my reason")
        self.fix.setText("Open the existing controls to fix this")
        issue_layout.addWidget(self.approve_warning)
        issue_layout.addWidget(self.fix)
        layout.addWidget(self.issue_actions)
        layout.addWidget(self.refresh_findings)

        self.details_toggle = QToolButton()
        self.details_toggle.setText("Details")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.technical_details = QWidget()
        technical_layout = QVBoxLayout(self.technical_details)
        technical_layout.setContentsMargins(0, 0, 0, 0)
        technical_layout.addWidget(self.status)
        technical_layout.addWidget(self.coverage_note)
        technical_layout.addWidget(self.details)
        self.information_details = QPlainTextEdit()
        self.information_details.setReadOnly(True)
        technical_layout.addWidget(QLabel("Other information from the original Check"))
        technical_layout.addWidget(self.information_details)
        technical_layout.addWidget(QLabel("Other verified photo roles"))
        technical_layout.addWidget(self.images)
        self.outline.setText("Outline the uncovered pixels being reviewed")
        self.layer_boundaries.setText("Outline the transferred map's layer boundaries")
        technical_layout.addWidget(self.outline)
        technical_layout.addWidget(self.layer_boundaries)
        self.details_toggle.toggled.connect(self._toggle_details)
        self.technical_details.hide()
        layout.addWidget(self.details_toggle)
        layout.addWidget(self.technical_details)

        navigation = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.next_button = QPushButton("Next")
        self.close_button = QPushButton("Close review")
        self.back_button.clicked.connect(lambda: self._navigate(-1))
        self.next_button.clicked.connect(lambda: self._navigate(1))
        self.close_button.clicked.connect(self.close)
        navigation.addWidget(self.back_button)
        navigation.addWidget(self.next_button)
        navigation.addStretch(1)
        navigation.addWidget(self.close_button)
        layout.addLayout(navigation)
        for control in (self.back_button, self.next_button, self.close_button,
                        self.experiment_photo, self.clearwater_photo):
            control.setAutoDefault(False)
        # Reparenting does not clear an explicit hide().  Page/stack parents
        # decide visibility from now on; only unused compatibility controls
        # remain explicitly hidden and can never be clicked in this shell.
        compatibility_only = {"runs", "classification", "approve_one", "approve_all", "paint"}
        for name in retained:
            if name not in compatibility_only:
                getattr(self, name).show()

    @staticmethod
    def _discard_layout(layout, protected):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                if item.widget() not in protected:
                    item.widget().deleteLater()
            elif item.layout() is not None:
                GuidedGapReviewDialog._discard_layout(item.layout(), protected)
        layout.deleteLater()

    def _create_deviation_editor(self, factory):
        if (factory is None or self._region_draft_factory is None
                or self._class_choices is None or self._sand_class is None):
            self._region_problem = "Verified region classification is not connected yet. No area can be confirmed in this window."
            return
        try:
            editor = factory(class_choices=self._class_choices,
                             sand_class=self._sand_class, parent=self)
            editor.photo_requested.connect(self._editor_photo_requested)
            editor.requested.connect(self._region_requested)
            editor.map_fix_requested.connect(self._sand_requested)
            editor.preview_changed.connect(self._region_draft_changed)
            editor.brightness.valueChanged.connect(self.brighten.setValue)
            editor.set_brightness(self.brighten.value())
            editor.set_suspended(True)
            self.deviation_editor = editor
            self.image_stack.addWidget(editor)
        except (TypeError, ValueError, AttributeError, RuntimeError) as error:
            self._region_problem = "The region editor could not be connected. Open Details for its original cause."
            self.status.setText(str(error))

    @property
    def guided_stage(self):
        return self._questions[self._question_index][0] if self._questions else "loading"

    def _current_row(self):
        if self._guided_ready and self.guided_stage == "water_line":
            return None  # A measurement tool is not a finding or a gap decision.
        return super()._current_row()

    def _water_review_runs(self):
        if (self._water_surface_comparison_reader is None
                or not isinstance((self.review or {}).get("display_context"), dict)):
            return []
        return [row["run_id"] for row in self.review["runs"]]

    def _show_water_line(self):
        runs = self._water_review_runs()
        selected = self._water_run if self._water_run in runs else (runs[0] if runs else None)
        self.water_run_choice.blockSignals(True)
        self.water_run_choice.clear()
        for run in runs:
            self.water_run_choice.addItem(run.upper(), run)
        self.water_run_choice.setCurrentIndex(self.water_run_choice.findData(selected))
        self.water_run_choice.blockSignals(False)
        self._select_water_run()

    def _select_water_run(self, *_args):
        if (not self._guided_ready or self.guided_stage != "water_line"
                or self._closing or self.operation_waiting):
            return
        run = self.water_run_choice.currentData()
        if run not in self._water_review_runs():
            return
        target = (run, None)
        key = ("water_line", run)
        self._water_run = run
        self._reason_key = None
        self._set_question_text("water_line")
        if self._selected == key and self._preview_key == target:
            return
        self._selected, self._preview_key = key, target
        self._comparison_key = self._base_packet = None
        self.packet = self._rgb = self._point_native = None
        self._clear_photo()
        self._pending = target
        if self.thread is None:
            self._pending = None
            self._load(target)
        self._buttons()

    def _place_water_controls(self, stage):
        layout = (self._measurement_water_layout if stage == "water_line"
                  else self._position_water_layout)
        for control in (self.water_surface, self.clear_water_confirmations):
            owner = control.parentWidget()
            if owner is layout.parentWidget():
                continue
            if owner is not None and owner.layout() is not None:
                owner.layout().removeWidget(control)
            control.setParent(layout.parentWidget())
            layout.insertWidget(max(0, layout.count() - 1), control)

    def _stage_for_row(self, row):
        if row["kind"] == "gap":
            return "deviation"
        code = str((row.get("issue") or {}).get("code") or "")
        if row["kind"] == "source_state" or code in {
                "run_geometry_acquisition_unknown", "indicator_facies_carryover_risk"}:
            return "source_water"
        if any(word in code for word in ("frame", "position", "water_surface", "waterline")):
            return "position"
        return "warnings"

    def _rebuild_deviation_groups(self):
        """Admit only compact worker DATA; original rows remain the gate."""
        from .painted_gap_review_groups import validate_deviation_groups

        gap_rows = [row for row in self._rows if row["kind"] == "gap"]
        known = {self._row_key(row): row for row in gap_rows}
        incoming = (self._report or {}).get("deviation_groups")
        candidate = incoming if incoming is not None else self._deviation_group_data
        previous_binding = (self._deviation_group_data or {}).get("review")
        if previous_binding != (self.review or {}).get("review_binding"):
            self._independent_deviations.clear()
            self._group_choices.clear()
        groups = []
        first_group_install = False
        self._group_problem = ""
        if candidate is not None and self.review is not None:
            try:
                checked = validate_deviation_groups(candidate, self.review)
                self._deviation_group_data = checked
                for group in checked["groups"]:
                    keys = [("gap", member["run_id"], member["component_id"])
                            for member in group["members"]]
                    if any(key not in known for key in keys):
                        raise ValueError("A grouped deviation is absent from the current findings")
                    together = [key for key in keys if key not in self._independent_deviations]
                    # Without its original representative, a group cannot claim
                    # that another source's classifications will fit it.
                    if together and together[0] == keys[0]:
                        groups.append(together)
                    else:
                        groups.extend([key] for key in together)
                    groups.extend([key] for key in keys if key in self._independent_deviations)
                if not self._first_group_catalogue_installed:
                    self._first_group_catalogue_installed = True
                    first_group_install = bool(groups)
            except (KeyError, TypeError, ValueError):
                self._deviation_group_data = None
                self._group_problem = "Matching groups are unavailable; every original deviation is shown separately."
                groups = []
        if not groups:
            groups = [[self._row_key(row)] for row in gap_rows]
        self._deviation_groups = groups
        self._group_by_row = {key: group[0] for group in groups for key in group}
        return first_group_install

    def _current_deviation_group(self):
        if not self._question_key or self.guided_stage != "deviation":
            return []
        return next((group for group in self._deviation_groups
                     if group[0] == self._question_key[1]), [])

    def _show_deviation_group(self, representative):
        group = next(group for group in self._deviation_groups if group[0] == representative)
        rows = {self._row_key(row): row for row in self._rows}
        source_ready = self._resolved(rows[group[0]]) and not self._draft_pending(rows[group[0]])
        selected = self._group_choices.get(representative, group[0])
        if selected not in group or (selected != group[0] and not source_ready):
            selected = group[0]
        self.deviation_group_choice.blockSignals(True)
        self.deviation_group_choice.clear()
        for index, key in enumerate(group):
            confirmed = self._resolved(rows[key])
            role = "classify here once" if index == 0 else "confirm or correct the transfer"
            self.deviation_group_choice.addItem(
                f"{rows[key]['run_id'].upper()}: {role}" + (" — recorded" if confirmed else ""), index)
            item = self.deviation_group_choice.model().item(index)
            item.setEnabled(index == 0 or source_ready)
            if index and not source_ready:
                item.setToolTip("First classify and confirm the representative area; no destination is approved by that choice.")
        self.deviation_group_choice.setCurrentIndex(group.index(selected))
        self.deviation_group_choice.blockSignals(False)
        self._select_group_member()

    def _select_group_member(self, *_args):
        if (not self._guided_ready or self.guided_stage != "deviation"
                or self._closing or ((self.busy() or self.operation_waiting) and not self._guidance_updating)):
            return
        group = self._current_deviation_group()
        index = self.deviation_group_choice.currentIndex()
        if not 0 <= index < len(group):
            return
        key = group[index]
        rows = {self._row_key(row): row for row in self._rows}
        if key != group[0] and (not self._resolved(rows[group[0]]) or self._draft_pending(rows[group[0]])):
            return
        self._group_choices[group[0]] = key
        index = next(index for index, row in enumerate(self._rows) if self._row_key(row) == key)
        self.runs.blockSignals(True)
        self.runs.setCurrentCell(index, 0)
        self.runs.blockSignals(False)
        self._group_loading_key = key if key != group[0] else None
        self._selecting_group_member = True
        try:
            self._select(index, 0, -1, -1)
        finally:
            self._selecting_group_member = False
        self._set_question_text("deviation")

    def _split_group_member(self, key):
        """An unavailable/partial suggestion becomes its own original question."""
        self._independent_deviations.add(key)
        self._group_split_pending = None
        self._group_loading_key = None
        self._rebuild_questions(preferred=key)

    def _default_reason(self, row):
        return {
            "deviation": "I identified what is visible inside this uncovered area.",
            "source_water": "I inspected the water state in the photo used to draw the map.",
            "position": "I compared this run's frame and position with the transferred map.",
            "warnings": "",
        }[self._stage_for_row(row)]

    def _remember_reason(self, *_args):
        if self._guided_ready and not self._changing_question and self._reason_key is not None:
            self._question_reasons[self._reason_key] = self.reason.text()
            self._question_other_text[self._reason_key] = self.other_text.text()

    def _select(self, index, column, previous_row, previous_column):
        if not self._guided_ready:
            return super()._select(index, column, previous_row, previous_column)
        if self._guidance_updating and self.guided_stage == "water_line":
            return  # A catalogue redraw must not read an unrelated finding's photo.
        if (self._guidance_updating and self.guided_stage == "deviation"
                and not self._selecting_group_member):
            return  # Group navigation, not the hidden table redraw, chooses the photo.
        self._remember_reason()
        self._changing_question = True
        try:
            super()._select(index, column, previous_row, previous_column)
            row = self._rows[index] if 0 <= index < len(self._rows) else None
            self._reason_key = None if row is None else self._row_key(row)
            if row is not None:
                self.reason.setText(self._question_reasons.get(self._reason_key,
                                                              self._default_reason(row)))
                self.other_text.setText(self._question_other_text.get(self._reason_key, ""))
        finally:
            self._changing_question = False
        self._remember_reason()
        if not self._guidance_updating:
            self._rebuild_questions(preferred=self._reason_key)

    def _refresh_rows(self):
        previous = self._guidance_updating
        self._guidance_updating = True
        try:
            super()._refresh_rows()
        finally:
            self._guidance_updating = previous
        if self._guided_ready and not previous:
            self._rebuild_questions()

    def refresh_guidance(self):
        """Re-read original in-memory action predicates after an owned result."""
        self._refresh_rows()

    def _rebuild_questions(self, preferred=None):
        if not self._guided_ready or self._guidance_updating:
            return
        previous = self._question_key
        first_group_install = self._rebuild_deviation_groups()
        questions, skipped = [], []
        if self.review is None and self._report is None:
            questions = [("loading", None)]
        else:
            for stage in _FINDING_STAGES:
                rows = [row for row in self._rows if row["kind"] != "info"
                        and self._stage_for_row(row) == stage]
                if stage == "deviation":
                    questions.extend((stage, group[0]) for group in self._deviation_groups)
                else:
                    questions.extend((stage, self._row_key(row)) for row in rows)
                if not rows:
                    skipped.append(_EMPTY[stage])
                if stage == "position" and self._water_review_runs():
                    questions.append(("water_line", None))
            questions.extend((("check", None), ("adopt", None)))
        self._questions = questions
        self.information_details.setPlainText("\n\n".join(
            f"{row['issue'].get('severity', 'info')}: {row['issue'].get('code', '')}\n"
            f"{row['issue'].get('message', '')}"
            for row in self._rows if row["kind"] == "info"))
        preferred = self._group_by_row.get(preferred, preferred)
        desired = 0 if first_group_install else next((i for i, (_, key) in enumerate(questions)
                        if preferred is not None and key == preferred), None)
        if desired is None:
            desired = next((i for i, value in enumerate(questions) if value == previous), 0)
        self.skipped_steps_note.setText("\n".join(skipped))
        self._show_question(desired)

    def _show_question(self, index):
        if not self._questions:
            return
        self._remember_reason()
        self._guidance_updating = True
        try:
            self._question_index = max(0, min(index, len(self._questions) - 1))
            stage, key = self._questions[self._question_index]
            self._question_key = (stage, key)
            if stage == "water_line":
                self._show_water_line()
            elif stage == "deviation":
                self._show_deviation_group(key)
            elif key is not None:
                row_index = next(i for i, row in enumerate(self._rows)
                                 if self._row_key(row) == key)
                self.runs.blockSignals(True)
                self.runs.setCurrentCell(row_index, 0)
                self.runs.blockSignals(False)
                self._select(row_index, 0, -1, -1)
            self._place_water_controls(stage)
            self.question_pages.setCurrentIndex(self._page_indices[stage])
            self._set_question_text(stage)
            self._sync_deviation_context()
        finally:
            self._guidance_updating = False
        self._buttons()

    def _set_question_text(self, stage):
        row = self._current_row() if self._question_key[1] is not None else None
        run = str(self._water_run if stage == "water_line" else (row or {}).get("run_id") or "").upper()
        titles = {
            "loading": "Open the verified review",
            "deviation": f"{run}: what is inside this uncovered area?",
            "source_water": "What is the water state in the map's source photo?",
            "position": f"{run + ': ' if run else ''}does this frame or position finding need correction?",
            "water_line": f"{run + ': ' if run else ''}review this run's water line",
            "warnings": "Resolve this finding",
            "check": "Check the answers across the series",
            "adopt": "Adopt the checked sand map",
        }
        group = self._current_deviation_group() if stage == "deviation" else []
        if len(group) > 1:
            selected = self._row_key(row) if row is not None else group[0]
            if selected == group[0]:
                titles["deviation"] = f"{run}: classify this matching group once"
            else:
                titles["deviation"] = f"{run}: do the transferred classifications fit this photo?"
        explanations = {
            "loading": "The original review reader is checking the saved sources. No choice is recorded while it loads.",
            "deviation": "The transferred map does not cover the outlined pixels. Use the existing painting tool to mark what you can identify; sand needs a map correction, not exclusion.",
            "source_water": "Use the source photograph to answer this question. This records the condition of the sand's water; an answer from another run or a previous chat is not used.",
            "position": "Compare this run's own measured frame with the map transferred from its reference. The camera-placement measurements flag a difference; they do not decide physical correctness for you.",
            "water_line": "Review each run's water line in its clear-water photograph. Confirm the proposal or move it yourself; the height uses this run's existing geometry, and no frame or scale changes.",
            "warnings": str(((row or {}).get("issue") or {}).get("message") or "Read this finding and either fix its cause or explicitly accept a warning with your reason. Errors must be fixed."),
            "check": "This runs the original fresh preparation and validators. It uses your explicit answers; opening this page does not start work or adopt a map.",
            "adopt": "The final action uses the original checked candidate and human confirmation. Every unresolved finding and changed source still blocks adoption.",
        }
        if len(group) > 1:
            explanations["deviation"] = (
                "Classify the representative area once. Then inspect each listed run and explicitly confirm or correct its transferred classifications."
                if self._row_key(row) == group[0] else
                "These markings are only a suggestion from the representative. Inspect this run's photo, correct anything that does not fit, then confirm this run yourself.")
        self.question_title.setText(titles[stage])
        self.question_explanation.setText(explanations[stage])
        self.question_progress.setText(f"Step {self._question_index + 1} of {len(self._questions)}")

    def _navigate(self, direction):
        if self.busy() or self._closing or self.operation_waiting:
            return
        group = self._current_deviation_group()
        if direction > 0 and len(group) > 1:
            rows = {self._row_key(row): row for row in self._rows}
            if self._resolved(rows[group[0]]) and not self._draft_pending(rows[group[0]]):
                selected = self._row_key(self._current_row())
                following = [key for key in group[group.index(selected)+1:] if not self._resolved(rows[key])]
                if following:
                    self.deviation_group_choice.setCurrentIndex(group.index(following[0]))
                    return
        if direction > 0 and self.guided_stage == "check":
            unresolved = next((i for i, (_, key) in enumerate(self._questions)
                if key is not None and not self._resolved(next(row for row in self._rows
                    if self._row_key(row) == key))), None)
            if self._report is not None and unresolved is not None:
                self._show_question(unresolved)
                return
        self._show_question(self._question_index + direction)

    def _context_key(self):
        row = self._current_row()
        if row is None or row["kind"] != "gap" or self.review is None:
            return None
        binding = self.review.get("review_binding") or {}
        return (binding.get("path"), binding.get("sha256"), binding.get("size_bytes"),
                row["run_id"], row["component_id"], row["review_row_sha256"])

    def region_context_key(self):
        """Return navigation identity only, never a region approval or readiness."""
        return deepcopy(self._context_key())

    def _row_context_key(self, row):
        if row["kind"] != "gap" or self.review is None:
            return None
        bound = self.review.get("review_binding") or {}
        return (bound.get("path"), bound.get("sha256"), bound.get("size_bytes"),
                row["run_id"], row["component_id"], row["review_row_sha256"])

    def _draft_pending(self, row):
        return self._row_context_key(row) in self._pending_region_contexts

    def pending_draft_choices(self):
        """Return current unconfirmed UI contexts, never an approval record."""
        return [self._draft_state_payload(self._row_context_key(row), True)
                for row in self._rows if row["kind"] == "gap" and self._draft_pending(row)]

    @staticmethod
    def _draft_state_payload(key, pending):
        return dict(review_binding=dict(path=key[0], sha256=key[1], size_bytes=key[2]),
            run_id=key[3], component_id=key[4], review_row_sha256=key[5], pending=pending)

    def _region_draft_changed(self, _snapshot=None):
        if (not self._editor_active() or self.busy() or self._closing
                or self.operation_waiting):
            return
        key, row = self._context_key(), self._current_row()
        if key in self._pending_region_contexts:
            return
        was_resolved = self._resolved(row)
        self._pending_region_contexts.add(key)
        self._preparation_changed = True
        self._candidate = self._report = None
        # A first unconfirmed stroke was already unresolved. Notify the parent
        # only when an existing saved choice has become pending again.
        if was_resolved:
            self._notified_pending_contexts.add(key)
            self.draft_choices_changed.emit(self._draft_state_payload(key, True))
        self._buttons()

    def can_adopt(self):
        return bool(super().can_adopt() and not any(
            self._draft_pending(row) for row in self._rows if row["kind"] == "gap"))

    @staticmethod
    def _display_packet(packet):
        # A backend draft is opaque and may own read-only mapping proxies.
        # Only the injected typed factory receives that original object.
        return deepcopy({key: value for key, value in packet.items()
                         if key != "region_draft"})

    def confirmed_region_request(self):
        """Expose only the request whose original typed save was accepted.

        This is an unapproved suggestion source for another question, never
        a confirmation for its destination or a substitute for its validator.
        """
        request = self._confirmed_region_request
        if (request is None or self.review is None
                or request["review_binding"] != self.review.get("review_binding")):
            return None
        bound = request["review_binding"]
        key = (bound["path"], bound["sha256"], bound["size_bytes"], request["run_id"],
               request["component_id"], request["review_row_sha256"])
        if key in self._pending_region_contexts:
            return None
        return deepcopy(request)

    def _reset_region_preview(self):
        if (not self._editor_active() or self.busy() or self._closing
                or self.operation_waiting or not self.region_preview_reset.isEnabled()):
            return
        row = self._current_row()
        payload = dict(review_binding=deepcopy(self.review["review_binding"]),
            run_id=row["run_id"], component_id=row["component_id"],
            review_row_sha256=row["review_row_sha256"])
        self._region_reset_submission = self._context_key()
        self._region_draft_changed()
        self.region_preview_reset_requested.emit(payload)
        self._buttons()

    def _sync_deviation_context(self):
        if (not self._guided_ready or self.guided_stage != "deviation"
                or self.packet is None or self._base_packet is None
                or self.deviation_editor is None):
            return
        key = self._context_key()
        if key is None or key == self._editor_context:
            return
        row = self._current_row()
        if (self._base_packet.get("run_id") != row["run_id"]
                or self._base_packet.get("component_id") != row["component_id"]):
            return
        try:
            draft = self._drafts.get(key)
            group = self._current_deviation_group()
            selected = self._row_key(row)
            prepared = self._base_packet.get("region_draft")
            if (draft is not None and len(group) > 1 and selected != group[0]
                    and prepared is not None and prepared.preview_source_context is not None
                    and draft.preview_source_context is None and not draft.events
                    and not any(mask.any() for mask in draft.class_masks.values())
                    and not self._resolved(row) and not self._draft_pending(row)):
                # The first photo may have opened before matching groups or a
                # source classification existed. Replace only that untouched
                # empty cache through the SAME typed factory below. Existing
                # edits, seeds, confirmations and explicit reset suppression
                # retain their own state; no source context is invented here.
                draft = None
            if draft is None:
                factory_packet = self._display_packet(self._base_packet)
                if "region_draft" in self._base_packet:
                    factory_packet["region_draft"] = self._base_packet["region_draft"]
                draft = self._region_draft_factory(deepcopy(self.review),
                                                    factory_packet)
            self.deviation_editor.set_context(self._display_packet(self._base_packet), draft)
            self.deviation_editor.set_brightness(self.brighten.value())
            self._drafts[key] = draft
            self._deviation_draft, self._editor_context = draft, key
            self._region_problem = ""
            if len(group) > 1 and selected != group[0]:
                source = draft.preview_source_context
                representative = next(value for value in self._rows
                                      if self._row_key(value) == group[0])
                expected = dict(review=self.review["review_binding"],
                    run_id=representative["run_id"], component_id=representative["component_id"],
                    review_row_sha256=representative["review_row_sha256"])
                occupied = np.zeros(draft.editable_mask.shape, bool)
                for mask in draft.class_masks.values():
                    occupied |= mask
                if source != expected or not np.array_equal(occupied, draft.editable_mask):
                    self._group_split_pending = selected
                self._group_loading_key = None
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError) as error:
            self._editor_context = None
            self._deviation_draft = None
            self._region_problem = "The verified classification draft could not open. Open Details for the original cause."
            self.status.setText(str(error))
            group = self._current_deviation_group()
            if len(group) > 1 and self._row_key(row) != group[0]:
                self._group_split_pending = self._row_key(row)

    def _editor_active(self):
        return bool(self._guided_ready and self.guided_stage == "deviation"
                    and self.deviation_editor is not None and self.packet is not None
                    and self._editor_context == self._context_key())

    def _region_payload(self, snapshot):
        if (not self._editor_active() or self.busy() or self._closing
                or self.operation_waiting or not self.actor or not self.reason.text().strip()):
            self.guided_status.setText("Add a reason for this answer and wait for the verified image before confirming.")
            return None
        row = self._current_row()
        return dict(snapshot=deepcopy(snapshot), review_binding=deepcopy(self.review["review_binding"]),
            run_id=row["run_id"], component_id=row["component_id"],
            review_row_sha256=row["review_row_sha256"], other_text=self.other_text.text().strip(),
            previous=deepcopy(self.approval),
            **self._human_action())

    def _region_requested(self, snapshot):
        payload = self._region_payload(snapshot)
        if payload is None:
            return
        try:
            if self._region_snapshot_collector is not None:
                payload["snapshot"] = deepcopy(self._region_snapshot_collector(
                    payload["snapshot"], review=deepcopy(self.review)))
        except (TypeError, ValueError, KeyError, OSError) as error:
            self.guided_status.setText("The typed classification snapshot was refused. Open Details for the original cause.")
            self.status.setText(str(error))
            return
        self._region_submission = (self._context_key(), deepcopy(payload))
        self.region_confirmation_requested.emit(payload)
        self.guided_status.setText("Your classifications were submitted for verification. No area is excluded until the original typed action is accepted.")
        self._buttons()

    def _consume_additional_result(self, key, result):
        """Consume only a current region result through the injected C validator.

        The base owner calls this after its existing native wait and sender/key
        checks.  The typed worker has performed the save; this hook grants no
        authority to a submitted draft or an arbitrary result dictionary.
        """
        if not isinstance(key, tuple) or len(key) != 2:
            return super()._consume_additional_result(key, result)
        if key[0] == "water-source-review":
            if key[1] != self._water_source_request:
                return True
            if self._closing or key[1] != self._water_source_context():
                self._water_source_request = None
                return True
            from .painted_review_comparison import _descriptor
            if (not isinstance(result, dict) or result.get("run_id") != self._clearwater_run()
                    or any(result.get(field) is not False for field in ("original_source_files_changed", "current_review_changed", "adopted", "worker_started"))
                    or result.get("clearwater_selection_changed") is not True):
                raise ValueError("The water-photograph review builder returned an unexpected result")
            bound = _descriptor(result["gap_review"])
            self.saved_clearwater_review = deepcopy(result)
            self._water_source_request = None
            self.water_source_note.setText("Saved a separate review using the recorded water photograph for "
                + str(result["run_id"]).upper() + ": " + bound["path"]
                + "\nYour current review and answers are unchanged. Close this view and choose Open another saved review to open it.")
            return True
        if key[0] == "region-preview-reset":
            if (self._closing or self._region_reset_submission is None
                    or key[1] != self._region_reset_submission
                    or key[1] != self._context_key()):
                return True
            # The owner uses its original finite reader to make a new unseeded
            # draft.  Its packet must pass the same original photo admission.
            self._region_reset_submission = None
            self._preview(result)
            self._drafts.pop(key[1], None)
            self._editor_context = None
            self._sync_deviation_context()
            self._buttons()
            return True
        if key[0] != "region-approval":
            return super()._consume_additional_result(key, result)
        submission = self._region_submission
        if (submission is None or key[1] != submission[0]
                or key[1] != self._context_key() or self.review is None):
            self.guided_status.setText("The saved region result belongs to an earlier question. It was not applied to this review.")
            return True
        try:
            if self._region_result_validator is None:
                raise ValueError("The original typed region-result validator is not connected")
            if not self._closing and self._review_validator is not None:
                self._review_validator(self.review)
            approval = self._region_result_validator(deepcopy(result), deepcopy(self.review),
                                                      deepcopy(submission[1]))
            if not isinstance(approval, dict):
                raise ValueError("The typed region validator did not return its original approval record")
        except (TypeError, ValueError, KeyError, OSError) as error:
            self.guided_status.setText("The saved classification could not be validated for this question. No approval was applied.")
            self.status.setText(str(error))
            return True
        self._region_submission = None
        self._confirmed_region_request = deepcopy(submission[1])
        self.approval = deepcopy(approval)
        self._preparation_changed = True
        self._candidate = self._report = None
        self._pending_region_contexts.discard(key[1])
        notify_clear = key[1] in self._notified_pending_contexts
        self._notified_pending_contexts.discard(key[1])
        if not self._closing:
            self._refresh_rows()
        if notify_clear:
            self.draft_choices_changed.emit(self._draft_state_payload(key[1], False))
        self.approval_changed.emit(deepcopy(self.approval))
        self.guided_status.setText("The original typed validator accepted this region action. A fresh Check is still required before adoption.")
        if not self._closing:
            self._buttons()
        return True

    def _sand_requested(self, snapshot):
        payload = self._region_payload(snapshot)
        if payload is None:
            return
        try:
            mask = np.asarray(self._deviation_draft.class_masks[self._sand_class])
            gap = np.asarray(self._base_packet["gap_mask"])
            if mask.dtype != np.bool_ or mask.shape != gap.shape or not mask.any() or np.any(mask & ~gap):
                raise ValueError("The sand correction needs an actual marked point inside this deviation")
            row, column = np.argwhere(mask)[0]
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            self.guided_status.setText("The marked sand could not be bound to this deviation. Open Details for the original cause.")
            self.status.setText(str(error))
            return
        self.map_fix_requested.emit(payload)  # Informational; only paint_requested launches the original editor.
        self.paint_requested.emit(dict(review=deepcopy(self.review), run_id=payload["run_id"],
            component_id=payload["component_id"], point_native_xy=[int(column), int(row)]))
        self.guided_status.setText("You marked sand. Fix the map in this run's original correction tool; this area has not been excluded.")

    def _preview(self, packet):
        super()._preview(packet)
        if self._guided_ready:
            if self.guided_stage != "deviation":
                self._pending_photo_fit = (self._preview_key, self._base_packet)
                self._photo_fit_size = None
            self._sync_deviation_context()
            if self.guided_stage == "water_line":
                self._request_photo("clearwater")

    def _comparison_reader_for_image(self, image_id):
        if self._guided_ready and self.guided_stage == "water_line" and image_id == "clearwater":
            return self._water_surface_comparison_reader
        return super()._comparison_reader_for_image(image_id)

    def _comparison(self, packet):
        if (self._guided_ready and self.guided_stage == "water_line"
                and self.images.currentData() == "clearwater"):
            water = packet.get("water_surface")
            if (packet.get("image_role") != "clearwater" or not isinstance(water, dict)
                    or (water.get("basis") or {}).get("run_id") != self._water_run):
                raise ValueError("The water-line photo and basis must belong to this selected run")
        super()._comparison(packet)
        if self._guided_ready:
            self._deliver_editor_photo(packet)

    def _image_selected(self, *_args):
        super()._image_selected(*_args)
        if (self._guided_ready and self.images.currentData() == "current"
                and self._base_packet is not None and not self.busy()):
            self._deliver_editor_photo(dict(self._base_packet, image_role="current"))
        if self._guided_ready:
            self._buttons()

    def _deliver_editor_photo(self, packet):
        if self._editor_request is None or not self._editor_active():
            return
        requested = self._editor_request
        try:
            if packet.get("image_role") != requested or self.images.currentData() != requested:
                raise ValueError("The photo reader did not return the requested verified role")
            self.deviation_editor.set_photo(self._display_packet(dict(self._base_packet, **packet)))
        except (TypeError, ValueError, KeyError) as error:
            super()._reset_comparison()
            self.deviation_editor.photo_failed(str(error))
        finally:
            self._editor_request = None

    def _reset_comparison(self):
        super()._reset_comparison()
        if self._guided_ready and self._editor_request is not None and self.deviation_editor is not None:
            self.deviation_editor.photo_failed("The requested verified photo was unavailable or changed.")
            self._editor_request = None

    def _clear_photo(self):
        super()._clear_photo()
        if self._guided_ready:
            self._cancel_photo_fit()
            self._editor_context = None
            self._deviation_draft = None
            self._editor_request = None
            self._region_submission = None
            self._region_reset_submission = None
            if self.deviation_editor is not None:
                self.deviation_editor.set_suspended(True)

    def _editor_photo_requested(self, role):
        if not self._editor_active() or self.busy() or self._closing or self.operation_waiting:
            if self.deviation_editor is not None:
                self.deviation_editor.photo_failed("Wait for this question's verified image.")
            return
        self._editor_request = role
        if not self._request_photo(role):
            self._editor_request = None
            self.deviation_editor.photo_failed("This verified photo role is not available for the selected run.")

    def _request_photo(self, role):
        if role not in {"current", "clearwater"} or self.busy() or self._closing or self.operation_waiting:
            return False
        index = self.images.findData(role)
        if index < 0 or not self.images.model().item(index).isEnabled():
            return False
        if self.images.currentIndex() == index:
            if role == "current" and self._base_packet is not None:
                self._deliver_editor_photo(dict(self._base_packet, image_role="current"))
            return True
        self.images.setCurrentIndex(index)
        return True

    def _water_source_context(self):
        return (deepcopy((self.review or {}).get("review_binding")),
                self._question_key, self._water_run)

    def _clearwater_run(self):
        """The run whose recorded water photograph the builder applies, or None."""
        return getattr(self._clearwater_review_builder, "run_id", None)

    def _prepare_water_source_review(self):
        """Save the recorded source choice; opening the new review is separate."""
        if (self._clearwater_review_builder is None or self.busy() or self._closing
                or self.operation_waiting or self.guided_stage != "water_line"
                or self._water_run is None or self._water_run != self._clearwater_run() or self.review is None):
            return
        context = self._water_source_context()
        authority, _ = QFileDialog.getOpenFileName(self,
            "Choose the recorded decision for the water photograph", "", "Recorded decision (*.json)")
        if not authority:
            return
        destination, _ = QFileDialog.getSaveFileName(self,
            "Save a new review with the recorded water photograph", "", "Saved review (*.json)")
        if not destination:
            return
        if (context != self._water_source_context() or self.busy()
                or self._closing or self.operation_waiting):
            self.water_source_note.setText("The selected run changed while choosing files. Nothing was created.")
            return
        review, builder = deepcopy(self.review), self._clearwater_review_builder
        self._water_source_request = context
        self.water_source_note.setText("Verifying the recorded photograph and saving a separate review. Your current answers are unchanged.")
        def prepare():
            from .painted_review_comparison import _bound
            return builder(review, destination, authority=_bound(authority))
        self._start(("water-source-review", context), prepare)
        self._buttons()

    def _next_image(self):
        if not self._guided_ready:
            return super()._next_image()
        self._request_photo("clearwater" if self.images.currentData() == "current" else "current")

    def _brightness_changed(self, value):
        super()._brightness_changed(value)
        if self._guided_ready and self.deviation_editor is not None:
            self.deviation_editor.set_brightness(value)

    def _recheck(self):
        if not self._guided_ready:
            return super()._recheck()
        if self.busy() or self._closing or self.operation_waiting:
            return
        if self._unconfirmed_water_edits:
            self.guided_status.setText("Confirm the edited water line yourself, or remove it from this check first.")
            return
        unresolved = self._first_precheck_unresolved()
        if unresolved is not None:
            self.guided_status.setText(self._precheck_lock_text(unresolved))
            return
        if not self.can_recheck():
            self.guided_status.setText(self.unresolved_findings_reason() or
                                      "Wait for the original findings update before Check.")
            return
        if self._needs_preparation():
            # This is the original controller's human-triggered fresh Prepare
            # entry, including empty/source-only reviews. No candidate is made here.
            self.recheck_requested.emit(deepcopy(self.approval))
        else:
            super()._recheck()

    def _first_precheck_unresolved(self):
        """Only original actual row resolution unlocks fresh Check.

        A saved map correction must first produce its authenticated residual
        review.  Neither a binding nor a draft proves that no pixels remain.
        """
        return next((row for row in self._rows
                     if not self._resolved(row) or self._draft_pending(row)), None)

    def _precheck_lock_text(self, row):
        return "Check is locked. A finding still needs your explicit answer or correction."

    def _buttons(self, *_args):
        super()._buttons(*_args)
        if not self._guided_ready or self._guidance_updating:
            return
        self._update_guided_controls()

    def _update_guided_controls(self):
        idle = not self.busy() and not self._closing and not self.operation_waiting
        stage = self.guided_stage
        row = self._current_row() if self._question_key and self._question_key[1] is not None else None
        no_photo = bool(row is not None and stage in _FINDING_STAGES
                        and self._preview_key is None and self.packet is None)
        preview_failure = bool(idle and self.packet is None
            and stage in {"deviation", "source_water", "position", "water_line", "warnings"}
            and self._preview_key is not None
            and self._last_started_read_key == ("preview", self._preview_key)
            and self.status.text().startswith((
            "The requested review operation could not finish:",
            "The image or gap outline could not be displayed:")))
        self.photo_placeholder.setText(
            "This photograph could not be opened. Open Details for the cause."
            if preview_failure else
            "This finding has no photograph. Use the existing controls to resolve its cause."
            if no_photo else "The verified photograph is loading.")
        if idle and self._group_split_pending is not None:
            self._split_group_member(self._group_split_pending)
            return
        if (idle and self._group_loading_key is not None and self.packet is None
                and self._pending is None and self.guided_stage == "deviation"):
            self._split_group_member(self._group_loading_key)
            return
        editor_active = self._editor_active()
        if self.deviation_editor is not None:
            self.deviation_editor.set_suspended(not idle or not editor_active)
        self.image_shortcut.setEnabled(idle and not editor_active)
        self.photo_toolbar.setVisible(stage not in {"loading", "check", "adopt"} and not editor_active and not no_photo)
        self.water_run_choice.setEnabled(idle and stage == "water_line")
        source_tool = (stage == "water_line" and self._clearwater_review_builder is not None
                       and self._water_run is not None and self._water_run == self._clearwater_run())
        self.water_source_button.setVisible(source_tool)
        self.water_source_button.setEnabled(idle and source_tool)
        self.water_source_note.setVisible(source_tool)
        if (self._water_source_request is not None and not self.busy()
                and self.status.text().startswith("The requested review operation could not finish:")):
            self._water_source_request = None
            self.water_source_note.setText("The separate review was not saved. Open Details for the cause. Your current review and answers are unchanged.")
        self.deviation_group_choice.setVisible(stage == "deviation" and len(self._current_deviation_group()) > 1)
        self.deviation_group_choice.setEnabled(idle)
        group_count = len(self._deviation_groups)
        deviation_count = sum(len(group) for group in self._deviation_groups)
        rows = {self._row_key(value): value for value in self._rows}
        remaining_sources = sum(not self._resolved(rows[group[0]]) or self._draft_pending(rows[group[0]])
                                for group in self._deviation_groups)
        remaining_transfers = sum(not self._resolved(rows[key]) or self._draft_pending(rows[key])
                                  for group in self._deviation_groups for key in group[1:])
        self.deviation_group_summary.setText(self._group_problem or (
            f"Groups to classify: {group_count} ({remaining_sources} remaining). "
            f"Transferred areas: {deviation_count-group_count} ({remaining_transfers} need individual confirmation). "
            "Viewing a transfer records no answer."
            if self._deviation_group_data is not None else
            "Matching-group counts are not available yet. Every original deviation remains accessible."))
        self.deviation_group_summary.setVisible(stage == "deviation")
        self.reason_panel.setVisible(row is not None)
        self.reason.setEnabled(idle)
        self.other_text.setEnabled(idle)
        self.issue_actions.setVisible(row is not None and row["kind"] in {"warning", "error"})
        self.approve_warning.setVisible(row is not None and row["kind"] == "warning")
        self.fix.setVisible(row is not None and row["kind"] in {"warning", "error"})
        self.back_button.setEnabled(idle and self._question_index > 0)
        self.next_button.setEnabled(idle and self._question_index < len(self._questions) - 1)
        self.images.setEnabled(idle and not editor_active)
        current = self.images.currentData()
        self.experiment_photo.setEnabled(idle and self._base_packet is not None and current != "current")
        clear_index = self.images.findData("clearwater")
        clear_available = clear_index >= 0 and self.images.model().item(clear_index).isEnabled()
        self.clearwater_photo.setEnabled(idle and clear_available and current != "clearwater")
        self.photo_availability.setText("" if no_photo or clear_available or stage in {"loading", "check", "adopt"}
            else "No verified clear-water photo is available for this run. The experiment photo remains unchanged.")
        self.region_unavailable.setText(self._region_problem)
        self.region_unavailable.setVisible(bool(self._region_problem))
        transfer_note = (self._base_packet or {}).get("region_transfer_note", "")
        self.region_transfer_note.setText(transfer_note if isinstance(transfer_note, str) else
            "The suggestion's explanation is unavailable. Inspect and confirm this area yourself.")
        self.region_transfer_note.setVisible(stage == "deviation" and bool(transfer_note))
        has_marks = bool(editor_active and any(np.any(mask)
            for mask in self._deviation_draft.class_masks.values()))
        self.region_preview_reset.setVisible(stage == "deviation" and bool(transfer_note) and has_marks)
        self.region_preview_reset.setEnabled(idle and editor_active and bool(transfer_note) and has_marks)
        if editor_active:
            group = self._current_deviation_group()
            transferred = bool(len(group) > 1 and row is not None and self._row_key(row) != group[0])
            self.deviation_editor.continue_button.setText(
                f"Confirm these classifications for {row['run_id'].upper()}"
                if transferred else "Review these classifications")
            self.image_stack.setCurrentWidget(self.deviation_editor)
        elif self.packet is not None:
            self.image_stack.setCurrentWidget(self.photo)
        else:
            self.image_stack.setCurrentWidget(self.photo_placeholder)
        self.image_stack.setVisible(stage not in {"check", "adopt"})
        self._queue_photo_fit()
        position_available = self.packet is not None and self.packet.get("position_readout") is not None
        for control in (self.position, self.tolerance_note, self.shift_tolerance,
                        self.angle_tolerance, self.apply_tolerances):
            control.setVisible(stage == "position" and position_available)
        self.water_surface.setVisible(stage == "water_line" or
            (stage == "position" and self.water_surface.packet is not None))
        self.clear_water_confirmations.setVisible(stage in {"position", "water_line"} and bool(
            self.water_surface_confirmations or self._unconfirmed_water_edits))
        self.water_line_note.setText(
            "This measurement tool adds no finding or decision by itself. The bolt-based height is unavailable unless exact observations are bound."
            if self._water_surface_comparison_reader is not None else
            "The verified water-line reader is not connected. No proposal or height is inferred from another run.")
        count = len((self.review or {}).get("runs", []))
        timing = "about 2 hours" if count == 39 else "duration depends on the series"
        self.recheck.setText(f"Check all {count} runs again ({timing})")
        unresolved = self._first_precheck_unresolved()
        self.recheck.setEnabled(idle and self.review is not None
                                and self.can_recheck() and unresolved is None)
        self.check_explanation.setText(self._precheck_lock_text(unresolved) if unresolved is not None else
            "The original preparation and Check will run only when you press the button."
            if self._candidate is None else "A genuine Check result is available. Return to any remaining findings before adoption.")
        self.adopt_explanation.setText("The checked map is ready for the original final confirmation."
            if self.can_adopt() else "Adoption is unavailable until a genuine fresh Check and every required answer are accepted. No choice is inferred from navigation.")
        self.adopt.setVisible(stage == "adopt" and self.can_adopt())
        if not idle:
            self.guided_status.setText("Waiting for the current verified read or operation. Your answers are unchanged.")
        elif stage == "loading":
            self.guided_status.setText("The saved review has not loaded. Open Details for the original reader status.")
        elif preview_failure:
            self.guided_status.setText(self.photo_placeholder.text())
        elif row is not None:
            self.guided_status.setText("This draft changed after its saved confirmation. Confirm the current classifications before Check."
                if self._draft_pending(row) and self._resolved(row) else
                "This answer is recorded by the original action validator. You can inspect it or continue."
                if self._resolved(row) else "This finding still needs an explicit answer or correction. Next only navigates; it does not resolve the finding.")
        else:
            self.guided_status.setText("No work or decision starts merely by opening this step.")

    def _toggle_details(self, expanded):
        self.details_toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.technical_details.setVisible(expanded)
