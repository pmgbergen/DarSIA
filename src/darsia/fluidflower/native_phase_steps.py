"""Native-only step layout around the retained Studio widgets and callbacks.

This adapter never installs a training algorithm. Missing compatible training
routes stay unavailable instead of falling back to a different optical chain.
"""

from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import uuid

from PySide6.QtCore import QSignalBlocker, Signal, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# A prepared-image answer this recent is kept when the page is opened again.
PREPARED_BASIS_MAX_AGE_S = 60.0

# What the review pages say when the selected model predates the current image correction.
OLD_BASIS_REASONS = {
    "label": "No review images on the current image correction yet: Train model version prepares them first",
    "blind": "Train a model on the current image correction first",
    None: "the selected model was trained before the current image correction; Train model version "
          "prepares review images on it, and blind validation follows training",
}


def model_basis_reason(model_root, series_root) -> str:
    """Return '' when a model was trained on the series' current image correction, else why not.

    The model folder's saved training request is checked with the existing
    require_prepared_normalisation. A model without such a request was
    trained on an earlier image basis.
    """
    from .calibration_model_training import require_prepared_normalisation
    path = Path(model_root) / "training_request.json"
    if not path.is_file():
        return "it has no saved training request on the current image correction"
    try:
        require_prepared_normalisation(json.loads(path.read_text(encoding="utf-8-sig")), series_root)
    except ValueError as error:
        return str(error)
    return ""


def saved_method_text(selection):
    """Literal saved method and references; no discovery or inferred choice."""
    if selection.error:
        return "Saved method unavailable — " + selection.error
    if selection.active == "phase_model":
        return ("phase_model · " + (selection.model_version or "model not referenced")
                + " · " + (selection.mass_calibration or "calibration not referenced"))
    if selection.active == "colour_path":
        return "colour_path · " + (selection.colour_calibration or "calibration not referenced")
    return "Method: not chosen"


class SessionSettings:
    """Keep this native instance's UI preferences in memory, including binding."""

    def __init__(self, original):
        self.original, self.values = original, {}

    def value(self, key, default=None, *args, **kwargs):
        return (
            self.values[key]
            if key in self.values
            else self.original.value(key, default, *args, **kwargs)
        )

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


def save_model_ui_name(request, name, actor):
    """Bind one display-only name to the original newly prepared request.

    This does not edit the scientific request, model ID, settings or summary.
    It is called only immediately before an explicitly requested job launch.
    """
    name = str(name).strip()
    if not name:
        raise ValueError("Name the new model version before training.")
    root = Path(request["output_root"])
    source = Path(request["request_path"])
    raw = source.read_bytes()
    saved = json.loads(raw)
    model_id = str(request.get("model_id") or "")
    if (
        not model_id
        or saved.get("model_id") != model_id
        or Path(saved.get("output_root", "")).resolve() != root.resolve()
        or source.resolve() != (root / "training_request.json").resolve()
    ):
        raise ValueError(
            "The display name must belong to the actual prepared training request."
        )
    payload = dict(
        format="phase-model-ui-name-v1",
        model_id=model_id,
        training_request_sha256=hashlib.sha256(raw).hexdigest(),
        display_name=name,
        recorded_by=actor,
        recorded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    target = root / "model_ui_name.json"
    if target.exists():
        raise ValueError("This model version already has a saved display name.")
    temporary = root / f".model-ui-name-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
        if target.exists():
            raise ValueError(
                "The display name appeared while saving; it was preserved."
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def saved_model_ui_name(entry):
    from darsia.fluidflower.model_index_names import model_ui_display_name

    return model_ui_display_name(entry)


def _move(widget, layout):
    widget.setParent(layout.parentWidget())
    layout.addWidget(widget)
    widget.show()


def _hide_form_field(widget):
    widget.hide()
    candidates = [widget]

    def hide_in(layout):
        if layout is None:
            return
        if isinstance(layout, QFormLayout):
            for candidate in candidates:
                label = layout.labelForField(candidate)
                if label is not None:
                    layout.setRowVisible(candidate, False)
                    return
        for index in range(layout.count()):
            child = layout.itemAt(index).layout()
            if child is not None:
                hide_in(child)

    parent = widget.parentWidget()
    while parent is not None:
        hide_in(parent.layout())
        candidates.append(parent)
        parent = parent.parentWidget()


def import_saved_model(series_root, series_id, folder):
    """Use existing inspection/filter/index APIs; never copy model assets."""
    from darsia.fluidflower.phase_model_history import inspect_phase_model_history_entry
    from darsia.fluidflower.ui.model_training import _models_for_series
    from darsia.fluidflower.model_index_names import index_model_with_ui_name

    entry = inspect_phase_model_history_entry(Path(folder))
    if not series_id or not _models_for_series([entry], series_id):
        raise ValueError(
            "The selected model does not belong to this experimental series."
        )
    return index_model_with_ui_name(series_root, Path(folder))


class NativePhaseSteps(QWidget):
    """The Calibration pages around the retained page's actual objects.

    One section shows at a time, chosen by its sidebar row (N141): Image
    normalisation, Phase model (vertical panels) and Use
    for this series. ``groups`` and ``layouts`` keep the former step numbers
    as keys (1, 2-4 and 6), with Basis at key 0; the saved mass values of the former step 5 are the
    Mass calibration row's own page.
    """

    SECTIONS = ("normalisation", "phase_model", "use")
    review_readiness_changed = Signal()
    model_for_calibration_requested = Signal(str)

    def __init__(
        self,
        studio,
        parent=None,
        *,
        normalisation=None,
        method_panel=None,
        method_use=None,
        phase_review=None,
        run_provider=None,
        actor_provider=getpass.getuser,
        training_routes=None,
    ):
        super().__init__(parent)
        self.studio, self.series_root, self.series_id = studio, None, ""
        studio._native_metadata_preview_only = True
        studio._native_step_adapter = self
        self.actor_provider = actor_provider
        self.training_routes = dict(training_routes or {})
        self.normalisation = normalisation
        self.method_panel, self.method_use = method_panel, method_use
        self.phase_review = phase_review
        self.run_provider = run_provider
        self._ready = False
        # Ready sources: the attached Image correction step, and a completed
        # Setup image preparation read directly (no page needs to be opened).
        self._normalisation_ready = self._prepared_ready = False
        self._prepared_reader = None
        # The series the prepared-image answer is for, the series being read,
        # the last answer (kind and text) and when it was read.
        self._prepared_context = self._prepared_reading = self._prepared_state = None
        self._prepared_checked_at = None
        # Whether each selected model is on the current image correction:
        # {(series root, model root): dict(kind=current|other|error, text=...)}.
        self._model_basis = {}
        self._saved_method_cache = None
        self._model_basis_reader = self._model_basis_reading = None
        self._review_presentation_state = None
        self._native_idle_status = None
        self._legacy_start_available = studio.start_button.isEnabled()
        self._original_start = studio._start_training
        self._original_update = studio._update_action_states
        self._original_launch = studio._launch_training_job
        self._original_set_series = studio.set_model_series
        self._original_project = studio.set_project_root
        self._original_review = studio._launch_review
        self._legacy_availability = {
            name: getattr(studio, name).isEnabled()
            for name in ("open_model_correction", "resume_session_button")
        }
        self._guard_slots = {}
        self._named_model_context = {}
        self._model_selection_pending = False
        self._model_selection_reference = None
        self._install_instance_policy()
        self.groups, self.layouts = {}, {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        # The method page scrolls as a whole. Normalisation and the phase model
        # provide their own control columns beside their working areas.
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        content = QWidget()
        page = QVBoxLayout(content)
        self.groups[1] = QGroupBox("Image correction")
        from .workflow_section import WorkflowWorkspace
        from .native_phase_panel_navigation import NativePhasePanelColumn

        self.workspace_stack = QStackedWidget()
        self.workspace_overview = QWidget()
        self.workspace_layout = QVBoxLayout(self.workspace_overview)
        self.workspace_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        overview = QLabel("Phase model")
        font = overview.font()
        font.setBold(True)
        overview.setFont(font)
        self.workspace_layout.addWidget(overview)
        overview = QLabel(
            "Choose the model basis on the left. Label phases opens the drawing "
            "and labelling tools here; the panels stay available."
        )
        overview.setWordWrap(True)
        self.workspace_layout.addWidget(overview)
        self.workspace_stack.addWidget(self.workspace_overview)
        self.phase_workspace = WorkflowWorkspace(workspace=self.workspace_stack)
        self.phase_tabs = NativePhasePanelColumn()
        for number, title in ((0, "Basis"), (2, "Label phases"),
                              (3, "Blindly validate phase model"), (4, "Train")):
            self.groups[number] = QWidget()
            self.phase_tabs.addTab(self.groups[number], title)
        self.phase_workspace.controls_layout.addWidget(self.phase_tabs)
        self.groups[6] = QGroupBox("Use for this series")
        self.groups[6].setObjectName("nativePhaseUseStep")
        for number, group in self.groups.items():
            self.layouts[number] = QVBoxLayout(group)
            if number != 1:
                # Short pages keep their controls together at the top; the
                # normalisation table takes the height it is given.
                self.layouts[number].setAlignment(Qt.AlignmentFlag.AlignTop)
        self.sections = {
            "normalisation": self.groups[1],
            "phase_model": self.phase_workspace,
            "use": self.groups[6],
        }
        for name, section in self.sections.items():
            if name == "use":
                page.addWidget(section, 1)
        self.scroll.setWidget(content)
        outer.addWidget(self.scroll)
        self._page_scroll = self.scroll
        outer.addWidget(self.groups[1], 1)
        outer.addWidget(self.phase_workspace, 1)
        self.section = None
        self.show_section("phase_model")
        self.normalisation_notice = QLabel(
            "Not done — the saved normalisation status step is not attached."
        )
        self.normalisation_notice.setWordWrap(True)
        self.layouts[1].addWidget(self.normalisation_notice)
        if normalisation is not None:
            self.attach_normalisation(normalisation)
        self._build_basis()
        self._build_review()
        self.review_selection_notice = QLabel(
            "Choose a review collection using the existing review controls. "
            "No collection is selected automatically."
        )
        self.review_selection_notice.setWordWrap(True)
        self.layouts[3].insertWidget(0, self.review_selection_notice)
        self._build_training()
        self.saved_method_label = QLabel("Method: not chosen")
        self.saved_method_label.setTextFormat(Qt.TextFormat.PlainText)
        self.saved_method_label.setWordWrap(True)
        self.layouts[6].addWidget(self.saved_method_label)
        if method_panel is None:
            self.layouts[6].addWidget(
                QLabel("The series method panel is not attached.")
            )
        elif not getattr(method_panel, "method_choice_only", False):
            _move(method_panel, self.layouts[6])
            original_controls = method_panel._update_controls

            def method_controls(*args):
                original_controls(*args)
                if not self._ready and method_panel.selection().active != "colour_path":
                    method_panel.use_button.setEnabled(False)

            method_panel._update_controls = method_controls
            for field in method_panel.findChildren(QLineEdit):
                field.textChanged.connect(method_controls)
            for button in method_panel.findChildren(QRadioButton):
                button.toggled.connect(method_controls)
            if method_use is not None:
                method_panel.use_requested.disconnect(method_use)
                method_panel.use_requested.connect(self._use_method)
        else:
            self.layouts[6].addWidget(QLabel(
                "Choose the calibration method under Calibration Method. "
                "Choose its phase model in this page's Basis panel."))
        studio.hide()
        self.refresh()
        from .native_phase_model_workflow import NativePhaseModelWorkflow

        self.phase_workflow = NativePhaseModelWorkflow(self)

    def _install_instance_policy(self):
        studio = self.studio
        if not isinstance(studio._model_filter_settings, SessionSettings):
            studio._model_filter_settings = SessionSettings(
                studio._model_filter_settings
            )
        # Identity is taken silently from the logged-in user, never persisted as
        # a reviewer preference merely because a native page was shown.
        studio._save_reviewer_preference = lambda: None
        studio._queue_reviewer_save = lambda: None
        studio._flush_reviewer_save = lambda: None
        studio._start_training = self.train
        studio._launch_training_job = self._launch_named_job

        def update(*args, **kwargs):
            result = self._original_update(*args, **kwargs)
            self._legacy_start_available = studio.start_button.isEnabled()
            self._legacy_availability = {
                name: getattr(studio, name).isEnabled()
                for name in ("open_model_correction", "resume_session_button")
            }
            if hasattr(self, "model_name"):
                self.refresh()
            return result

        studio._update_action_states = update
        studio._launch_review = self._launch_review
        for name in (
            "_resume_selected_session",
            "_open_selected_experiment",
            "_start_series_phase_batch",
            "_open_transfer_comparison",
        ):
            original = getattr(studio, name, None)
            if callable(original):
                self._guard_slots[name] = original
                setattr(
                    studio,
                    name,
                    lambda *a, _original=original, **k: self._guarded(
                        _original, *a, **k
                    ),
                )
        for button_name, slot_name in (
            ("resume_session_button", "_resume_selected_session"),
            ("test_transfer_button", "_open_selected_experiment"),
            ("transfer_comparison_button", "_open_transfer_comparison"),
        ):
            button = getattr(studio, button_name, None)
            original = self._guard_slots.get(slot_name)
            if button is not None and original is not None:
                button.clicked.disconnect(original)
                button.clicked.connect(getattr(studio, slot_name))
        original_label = studio._validation_model_label

        def label(entry):
            name = saved_model_ui_name(entry)
            return (name + " · " if name else "") + original_label(entry)

        studio._validation_model_label = label
        original_version_text = getattr(studio, "_model_version_text", None)
        if callable(original_version_text):
            def version_text(entry):
                name = saved_model_ui_name(entry)
                return (name + " \u00b7 " if name else "") + original_version_text(entry)

            studio._model_version_text = version_text
        original_completed = getattr(studio, "_record_completed_model", None)
        if callable(original_completed):
            def completed(model_id, root):
                error = None
                context = self._named_model_context.get(str(model_id))
                if context is not None:
                    try:
                        from darsia.fluidflower.model_index_names import index_model_with_ui_name

                        series_root, output_root = context
                        if Path(root).resolve() != output_root:
                            raise ValueError("Completed model folder differs from its named training request")
                        index_model_with_ui_name(series_root, root, expected_model_id=str(model_id))
                    except (OSError, ValueError, TypeError, RuntimeError) as caught:
                        error = caught
                result = original_completed(model_id, root)
                if error is not None:
                    studio.status.setText("Model completed; its display name could not be indexed: " + str(error))
                return result

            studio._record_completed_model = completed

    def _build_basis(self):
        studio = self.studio
        row = QVBoxLayout()
        # Four explicit starting points. The two existing-model choices share
        # one hidden existing_basis, which other readers keep using.
        self.scratch_basis = QRadioButton("New model from new labels")
        self.retrain_basis = QRadioButton("Retrain on current image correction")
        self.continue_basis = QRadioButton("Continue with new labels")
        self.select_basis = QRadioButton("Select model for calibration")
        self.existing_basis = QRadioButton("Train existing model")
        self.existing_basis.setAutoExclusive(False)
        self.existing_basis.hide()
        row.addWidget(self.existing_basis)
        self.basis_choices = QButtonGroup(self)
        for button, text in (
                (self.scratch_basis, "Start without the existing validation boxes: label new images, then train."),
                (self.retrain_basis, "Retrain the chosen model's labelled images, with its training settings, "
                                     "on today's image correction. New weights; no new boxes are needed."),
                (self.continue_basis, "Keep the chosen model and train it further with newly labelled images."),
                (self.select_basis, "Use an existing model for calibration without training.")):
            self.basis_choices.addButton(button)
            button.setToolTip(text)
            row.addWidget(button)
            hint = QLabel(text)
            hint.setWordWrap(True)
            hint.setContentsMargins(22, 0, 0, 4)
            row.addWidget(hint)
        self.retrain_basis.setChecked(True)
        self.existing_basis.setChecked(True)
        for button in self.basis_choices.buttons():
            button.toggled.connect(self._sync_existing_basis)
        self.existing_basis.toggled.connect(self._existing_basis_set)
        self.layouts[0].addLayout(row)
        self.basis_source_label = QLabel("Existing model")
        self.layouts[0].addWidget(self.basis_source_label)
        _move(studio.validation_model, self.layouts[0])
        _move(studio.catalog_loading_status, self.layouts[0])
        self.import_button = QPushButton("Import existing model…")
        self.import_button.clicked.connect(self._import_model)
        self.layouts[0].addWidget(self.import_button)
        self.organise_button = QPushButton("Organise phase-model files…")
        self.organise_button.clicked.connect(self._organise_phase_files)
        self.layouts[0].addWidget(self.organise_button)
        self._layout_dialog = None
        _move(studio.validation_model_summary, self.layouts[0])
        self.use_selected_model = QPushButton("Use selected model for calibration")
        self.use_selected_model.clicked.connect(self._request_model_for_calibration)
        self.layouts[0].addWidget(self.use_selected_model)
        self.model_selection_status = QLabel()
        self.model_selection_status.setWordWrap(True)
        self.layouts[0].addWidget(self.model_selection_status)
        self.basis_note = QLabel()
        self.basis_note.setWordWrap(True)
        self.layouts[0].addWidget(self.basis_note)
        self.existing_basis.toggled.connect(self.refresh)
        for button in (self.scratch_basis, self.retrain_basis, self.continue_basis, self.select_basis):
            button.toggled.connect(self.refresh)
        studio.validation_model.currentIndexChanged.connect(self.refresh)

    def _sync_existing_basis(self, *_args):
        """Keep the shared existing-model flag checked while either existing-model choice is."""
        if any(button.isChecked() for button in self.basis_choices.buttons()):
            self.existing_basis.setChecked(self.retrain_basis.isChecked() or self.continue_basis.isChecked())

    def _existing_basis_set(self, checked):
        """A caller that sets the shared flag directly keeps the earlier automatic route choice."""
        if checked and not (self.retrain_basis.isChecked() or self.continue_basis.isChecked()):
            self.basis_choices.setExclusive(False)
            for button in (self.scratch_basis, self.select_basis):
                button.setChecked(False)
            self.basis_choices.setExclusive(True)

    def existing_choice(self):
        """``retrain`` or ``continue`` for the two existing-model starting points.

        '' means a new model, a model selection, or the shared existing-model
        flag set without either choice, which keeps the earlier automatic route
        (prepared retraining, else the prepared RAW fit, else continuation).
        """
        if self.retrain_basis.isChecked():
            return "retrain"
        return "continue" if self.continue_basis.isChecked() else ""

    def model_selection_only(self):
        return self.select_basis.isChecked()

    def _selected_calibration_model(self):
        getter = getattr(self.studio, "_selected_validation_model", None)
        model = getter() if callable(getter) else None
        return str((model or {}).get("root") or "").strip()

    def _request_model_for_calibration(self):
        reference = self._selected_calibration_model()
        if not self.model_selection_only() or not self.use_selected_model.isEnabled() or not reference:
            return
        self._model_selection_pending = True
        self.model_selection_status.setText("Saving the selected model for calibration…")
        self.refresh()
        self.model_for_calibration_requested.emit(reference)

    def set_model_selection_result(self, success, error=""):
        self._model_selection_pending = False
        self.model_selection_status.setText(
            "✓ Model selected for calibration. No training or calibration job was started."
            if success else "The model was not selected: " + (str(error) or "the save did not complete."))
        self.refresh()

    def _build_review(self):
        self.studio.open_model_correction.setText("Review and label…")
        row = QHBoxLayout()
        self.layouts[3].addLayout(row)
        _move(self.studio.open_model_correction, row)
        # The former Analysis row "Phase review window" (mass_analysis:review)
        # sits beside the review it opens; its row id stays routable (N141).
        self.phase_review_button = QPushButton("Phase review window")
        self.phase_review_button.setToolTip(
            "Open the existing prepared model-correction review window; a loaded model with bound phase maps is required."
        )
        self.phase_review_button.setEnabled(self.phase_review is not None)
        if self.phase_review is not None:
            self.phase_review_button.clicked.connect(self._open_phase_review)
        row.addWidget(self.phase_review_button)
        row.addStretch(1)
        _move(self.studio.review_status, self.layouts[3])
        review_hint = QLabel(
            "Saved model maps and drawing tools open in the established review window. No map is recalculated by this page layout."
        )
        review_hint.setWordWrap(True)
        self.layouts[3].addWidget(review_hint)
        self.review_route_button = QPushButton("Choose a model in Basis…")
        self.review_route_button.clicked.connect(self._open_review_requirement)
        self.layouts[3].addWidget(self.review_route_button)

    def review_launch_reason(self, role=None):
        """Present the existing launch requirements for this exact selection.

        ``role`` is ``label`` or ``blind`` when a review page asks. A model
        trained before the current image correction never opens its older
        review images; the page says what comes first (OLD_BASIS_REASONS).
        """
        if self.model_selection_only():
            return "Select model for calibration is active. Choose a training option to label or validate phases."
        studio = self.studio
        getter = getattr(studio, "_selected_validation_model", None)
        model = getter() if callable(getter) else None
        run = str(self.run_provider() or "").strip() if callable(self.run_provider) else ""
        scope = "Series " + (self.series_id or "not selected")
        scope += " · run " + (run or "not selected")
        if model:
            scope += " · model " + str(model.get("model_id") or model.get("root") or "not identified")
        else:
            scope += " · model not selected"
        missing = []
        if not self._ready and not self._has_saved_review_binding(model):
            missing.append(self._image_correction_reason())
        basis = self._selected_model_basis(model) if model else None
        if not model:
            missing.append("choose a model in Basis")
        elif basis is None:
            missing.append("checking whether the selected model is on the current image correction")
        elif basis["kind"] == "other":
            missing.append(OLD_BASIS_REASONS.get(role, OLD_BASIS_REASONS[None]))
        elif basis["kind"] == "error":
            missing.append("the selected model's image basis could not be checked: " + basis["text"])
        elif model.get("validation_binding_loading"):
            missing.append("the selected model's saved phase maps are still being checked")
        elif model.get("validation_collection_error"):
            missing.append("saved phase maps could not be verified: " + str(model["validation_collection_error"]))
        elif not isinstance(model.get("validation_binding"), dict):
            # The review images are made for the current image correction when
            # a model version is trained; blind validation follows training.
            missing.append("no prepared phase maps for the current image correction are bound to this model yet; "
                           "Train model version prepares them, and blind validation follows training")
        if not missing and not self._legacy_availability.get("open_model_correction", False):
            reason = getattr(studio, "_review_launch_block_reason", None)
            missing.append(reason(model) if callable(reason) else "the existing review launch is unavailable")
        return (scope + ": " + "; ".join(missing) + ".") if missing else ""

    def _selected_model_basis(self, model):
        """The answer for the selected model (see model_basis_reason), read once in the background."""
        root = str(model.get("root") or "")
        if self.series_root is None:
            return dict(kind="current", text="")  # Without a series nothing opens anyway.
        if not root:
            return dict(kind="error", text="the selected model has no folder")
        key = (str(self.series_root), root)
        if key not in self._model_basis:
            self._start_model_basis_check(key)
            return None
        return self._model_basis[key]

    def _start_model_basis_check(self, key):
        from .ui_metadata_preview import MetadataPreview
        if self._model_basis_reader is None:
            self._model_basis_reader = MetadataPreview(self)
            self._model_basis_reader.completed.connect(self._model_basis_read)
        if self._model_basis_reader.busy and self._model_basis_reading == key:
            return
        self._model_basis_reading = key
        series_root, model_root = key
        self._model_basis_reader.start(lambda _cancelled: (key, model_basis_reason(model_root, series_root)))

    def _model_basis_read(self, value, error):
        """Keep the selected model's answer and let a waiting review page act on it."""
        key, self._model_basis_reading = self._model_basis_reading, None
        if error or value is None:
            if key is None:
                return
            self._model_basis[key] = dict(kind="error", text=error or "no answer was returned")
        else:
            key, reason = value
            self._model_basis[key] = dict(kind="other" if reason else "current", text=reason)
        self.review_readiness_changed.emit()
        if hasattr(self, "model_name"):
            self.refresh()

    @staticmethod
    def _has_saved_review_binding(model):
        # Reviewing the exact saved photographs does not apply today's
        # normalisation again. Training still requires the normalisation step.
        return bool(model and not model.get("validation_binding_loading")
                    and not model.get("validation_collection_error")
                    and isinstance(model.get("validation_binding"), dict))

    def review_selection_text(self):
        """Describe the series collection without restricting it to a hidden run selector."""
        reason = self.review_launch_reason()
        if reason:
            return reason
        model = self.studio._selected_validation_model()
        binding = model.get("validation_binding") or {}
        candidates = binding.get("selected_candidates")
        runs = ", ".join(sorted(map(str, candidates))) if isinstance(candidates, dict) and candidates else "the verified saved collection"
        run = str(self.run_provider() or "").strip() if callable(self.run_provider) else ""
        return ("Saved phase maps are ready for model " + str(model.get("model_id") or model.get("root"))
            + " in series " + (self.series_id or "not selected") + ". Available runs: " + runs
            + (". Current run selection: " + run if run else "") + ".")

    def show_review_requirement(self):
        """Keep a failed open visible beside the review action."""
        reason = self.review_launch_reason()
        if not reason:
            reason = "The prepared phase review could not be opened; check the selected model's saved phase-map binding."
        self.studio.review_status.setText(reason)
        self.review_selection_notice.setText(reason)
        self.show_section("phase_model")
        self.phase_tabs.setCurrentWidget(self.groups[3])
        self.scroll.ensureWidgetVisible(self.review_selection_notice)
        return reason

    def _open_phase_review(self):
        try:
            if self.phase_review is None:
                self.show_review_requirement()
                return
            # The retained callback owns admission and opening. A plain reason
            # remains visible even if it reports a missing dependency itself.
            if self.review_launch_reason():
                self.show_review_requirement()
            self.phase_review()
        except Exception as error:
            message = "Could not open phase review: " + str(error)
            self.studio.review_status.setText(message)
            self.review_selection_notice.setText(message)
            QMessageBox.warning(self, "Could not open phase review", message)

    def _open_review_requirement(self):
        """Navigate or recheck saved metadata; never generate new phase maps."""
        studio = self.studio
        getter = getattr(studio, "_selected_validation_model", None)
        model = getter() if callable(getter) else None
        if not self._ready:
            self.show_section("normalisation")
            self.scroll.ensureWidgetVisible(self.normalisation or self.normalisation_notice)
        elif not model:
            self.show_section("phase_model")
            self.phase_tabs.setCurrentWidget(self.groups[0])
            studio.validation_model.setFocus()
        else:
            refresh = getattr(studio, "_request_async_model_refresh", None)
            try:
                if callable(refresh):
                    refresh(bindings_only=True, verify_saved=True)
                self.show_review_requirement()
            except Exception as error:
                message = "Could not check saved phase maps: " + str(error)
                studio.review_status.setText(message)
                self.review_selection_notice.setText(message)
                QMessageBox.warning(self, "Could not check saved phase maps", message)

    def _build_training(self):
        studio = self.studio
        annotators = getattr(studio, "annotator_selection", None)
        if isinstance(annotators, QWidget):
            _move(annotators, self.layouts[4])
        _move(studio.saved_sessions, self.layouts[4])
        for name in ("refresh_review_sessions_button", "saved_sessions_status"):
            control = getattr(studio, name, None)
            if isinstance(control, QWidget):
                _move(control, self.layouts[4])
        _move(studio.session_summary, self.layouts[4])
        _move(studio.resume_session_button, self.layouts[4])
        _move(studio.add_session_evidence, self.layouts[4])
        _move(studio.pending_training_sessions, self.layouts[4])
        self.layouts[4].addWidget(QLabel("Name the new model version"))
        self.model_name = QLineEdit()
        self.model_name.setPlaceholderText("Required before training")
        self.model_name.textChanged.connect(self.refresh)
        self.layouts[4].addWidget(self.model_name)
        self.training_note = QLabel()
        self.training_note.setWordWrap(True)
        self.layouts[4].addWidget(self.training_note)
        self.advanced = QGroupBox("Existing training settings and evidence inputs")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        advanced = QVBoxLayout(self.advanced)
        original_scroll = studio.training_tab
        original_content = (
            original_scroll.takeWidget()
            if isinstance(original_scroll, QScrollArea)
            else None
        )
        if original_content is not None:
            _move(original_content, advanced)
            original_content.hide()
            self.advanced.toggled.connect(original_content.setVisible)
        self.layouts[4].addWidget(self.advanced)
        for name in (
            "reviewer",
            "reviewer_project",
            "model_series_filter",
            "history_series_filter",
            "training_series",
            "training_scope",
            "output_parent",
            "cache_parent",
            "model_library",
            "ffac_campaign",
            "ffum_campaign",
            "combined_campaign",
            "review_save_root",
        ):
            field = getattr(studio, name, None)
            if isinstance(field, QWidget):
                _hide_form_field(field)
        for name in (
            "train_further_button",
            "compare_phase_models_button",
            "start_existing_model",
            "start_new_model",
        ):
            widget = getattr(studio, name, None)
            if isinstance(widget, QWidget):
                widget.hide()
        studio.start_button.clicked.disconnect(self._original_start)
        studio.start_button.clicked.connect(self.train)
        studio.start_button.setText("Train model version")
        _move(studio.start_button, self.layouts[4])
        _move(studio.stop_button, self.layouts[4])
        _move(studio.status, self.layouts[4])
        # A busy bar while the completed Image correction is read after start-up.
        self.check_busy = QProgressBar()
        self.check_busy.setRange(0, 0)
        self.check_busy.setTextVisible(False)
        self.check_busy.setMaximumWidth(240)
        self.check_busy.setToolTip("Checking the completed Image correction")
        self.check_busy.hide()
        self.layouts[4].addWidget(self.check_busy)
        _move(studio.log, advanced)
        studio.log.hide()
        self.advanced.toggled.connect(studio.log.setVisible)
        for timer_name in ("_retraction_poll_timer",):
            timer = getattr(studio, timer_name, None)
            if timer is not None:
                timer.timeout.connect(self.refresh)

    def show_section(self, name):
        """Show the section of one sidebar row; the others stay built and hidden."""
        if name not in self.sections:
            raise ValueError(f"Unknown phase-model page section: {name}")
        self.section = name
        for key, widget in self.sections.items():
            widget.setVisible(key == name)
        self._page_scroll.setVisible(name == "use")
        self.scroll = (self.phase_workspace.controls_scroll
                       if name == "phase_model" else self._page_scroll)
        self.scroll.verticalScrollBar().setValue(0)

    def attach_normalisation(self, widget):
        if self.normalisation is not None and self.normalisation is not widget:
            self.layouts[1].removeWidget(self.normalisation)
            self.normalisation.hide()
        self.normalisation = widget
        self.normalisation_notice.hide()
        _move(widget, self.layouts[1])
        widget.readiness_changed.connect(self.set_normalisation_ready)
        self.set_normalisation_ready(getattr(widget, "ready", False))

    def set_normalisation_ready(self, ready):
        self._normalisation_ready = bool(ready)
        self._ready = self._normalisation_ready or self._prepared_ready
        if hasattr(self, "model_name"):
            self.refresh()

    def _refresh_prepared_basis(self):
        """Read whether a completed Setup image preparation matches the saved settings.

        The reading takes up to about a minute on a shared drive. A reading of
        the same series that runs, or returned within PREPARED_BASIS_MAX_AGE_S,
        is kept, and the last answer for the series stays in force until a new
        one returns. Opening the page again therefore never turns a completed
        Image correction into a missing one while it is checked.
        """
        import time
        from .ui_metadata_preview import MetadataPreview
        if self._prepared_reader is None:
            self._prepared_reader = MetadataPreview(self)
            self._prepared_reader.completed.connect(self._prepared_basis_read)
        root, series = self.series_root, self.series_id
        context = (str(root), series) if root is not None and series else None
        if context != self._prepared_context:
            # Nothing is known yet about another series' prepared images.
            self._prepared_ready, self._prepared_state, self._prepared_checked_at = False, None, None
        self._prepared_context = context
        if context is None:
            self._prepared_reader.cancel()
            self._prepared_reading = None
            return
        if self._prepared_reader.busy and self._prepared_reading == context:
            return
        if (self._prepared_checked_at is not None
                and time.monotonic() - self._prepared_checked_at < PREPARED_BASIS_MAX_AGE_S):
            return
        self._prepared_reading = context

        def read(_cancelled):
            from .image_preparation_status import PreparationStatusSource
            original = (Path(root) / "series.toml").read_bytes()
            result = PreparationStatusSource()(root, series, original)
            return context, bool(result.get("complete")), str(result.get("text") or "")
        self._prepared_reader.start(read)

    def _prepared_basis_read(self, value, error):
        import time
        reading, self._prepared_reading = self._prepared_reading, None
        if reading != self._prepared_context or (value is not None and value[0] != reading):
            return
        previous = self._prepared_state
        if error or value is None:
            self._prepared_ready = False
            self._prepared_state = dict(kind="error", text=error or "no answer was returned")
        else:
            self._prepared_ready = value[1]
            self._prepared_state = dict(kind="complete" if value[1] else "incomplete", text=value[2])
        if previous is not None and previous != self._prepared_state:
            # Another image correction: each model's basis is checked again.
            self._model_basis.clear()
        self._prepared_checked_at = time.monotonic()
        self.set_normalisation_ready(self._normalisation_ready)

    def _saved_method(self):
        """The series' saved method, parsed again only when series.toml changes on disk.

        The page refreshes every few seconds. Parsing the whole series.toml on
        the shared drive each time took about 0.07 s of the GUI thread; a stat
        of the file is enough to see that it is unchanged.
        """
        from darsia.fluidflower.series_method import MethodSelection, read_series_method
        if not self.series_root:
            return MethodSelection()
        path = Path(self.series_root) / "series.toml"
        try:
            info = path.stat()
            stamp = (str(path), info.st_size, info.st_mtime_ns)
        except OSError:
            stamp = None
        cached = self._saved_method_cache
        if stamp is not None and cached is not None and cached[0] == stamp:
            return cached[1]
        saved = read_series_method(self.series_root)
        self._saved_method_cache = (stamp, saved) if stamp is not None and not saved.error else None
        return saved

    def _checking_image_correction(self):
        """True while the first answer about the completed Image correction is being read."""
        return self._prepared_state is None and self._prepared_reading is not None

    def _image_correction_tooltip(self):
        reason = self._image_correction_reason()
        return reason[:1].upper() + reason[1:] + "."

    def _image_correction_reason(self):
        """Why Image correction does not count as complete for review, in plain words."""
        state = self._prepared_state
        if state is None and self._prepared_reading is not None:
            return "the completed Image correction is still being checked (reading the Setup job takes about a minute)"
        if state is not None and state["kind"] == "error":
            return "the Image correction status could not be read: " + state["text"]
        if state is not None and state["kind"] == "incomplete" and state["text"]:
            return "Image correction is not complete: " + state["text"].rstrip(".")
        return "complete Image correction with Finish setup in Setup"

    def set_context(self, series_root, series_id):
        self.series_root = Path(series_root) if series_root else None
        self.series_id = str(series_id or "")
        dialog = self._layout_dialog
        if (dialog is not None and dialog.series_root != self.series_root
                and self.workspace_stack.currentWidget() is dialog):
            self.workspace_stack.setCurrentWidget(self.workspace_overview)
        studio = self.studio
        for timer_name in ("_reviewer_save_timer", "_project_root_timer"):
            timer = getattr(studio, timer_name, None)
            if timer is not None:
                timer.stop()
        studio._reviewer_project_root = None
        with QSignalBlocker(studio.reviewer), QSignalBlocker(studio.reviewer_project):
            studio.reviewer.setText(str(self.actor_provider()))
            studio.reviewer_project.setText(str(self.series_root or ""))
        canonical = self.series_id.casefold().replace("_", "").replace("-", "")
        if self.series_root is not None:
            studio._configured_series_roots[canonical] = self.series_root
        self._original_set_series(self.series_id)
        with QSignalBlocker(studio.training_series):
            studio.training_series.setText(canonical)
        studio._sync_default_model_libraries()
        from .series_method import read_series_method

        choice = read_series_method(self.series_root) if self.series_root is not None else None
        model = choice.model_version if choice is not None and not choice.error else ""
        if model:
            model = Path(model)
            model = model if model.is_absolute() else self.series_root / model
        context_key = (str(self.series_root or ""), os.environ.get("DARSIA_NATIVE_CONFIGURATION_SHA256"))
        studio.set_saved_model_preference(model or None, context_key=context_key)
        if self.normalisation is not None:
            self.normalisation.set_context(self.series_root, self.series_id)
        else:
            self.set_normalisation_ready(False)
        self._refresh_prepared_basis()
        self.model_name.clear()
        self._original_update()
        self.refresh()

    def refresh(self, *_args):
        if not hasattr(self, "groups") or not hasattr(self, "model_name"):
            return
        saved = self._saved_method()
        self.saved_method_label.setText(saved_method_text(saved))
        photometry = getattr(self, "training_photometry", None)
        selection_only = self.model_selection_only()
        if photometry is not None and not selection_only:
            photometry.refresh()
        self.studio.open_model_correction.setText("Review and label…")
        # Saved versions, sessions and calibration values remain inspectable
        # while only the dependent actions are gated.
        for number, group in self.groups.items():
            if number != 1:
                group.setEnabled(True)
        getter = getattr(self.studio, "_selected_validation_model", None)
        model = getter() if callable(getter) else None
        for name, available in self._legacy_availability.items():
            ready = self._ready or (name == "open_model_correction"
                                   and self._has_saved_review_binding(model))
            getattr(self.studio, name).setEnabled(ready and available)
        self.import_button.setEnabled(self.series_root is not None and (self._ready or selection_only))
        self.import_button.setToolTip("Import an existing saved model." if self._ready or selection_only
            else "Complete and save Image correction before importing a model.")
        organise = self.series_root is not None and bool(str(self.series_id or "").strip())
        self.organise_button.setEnabled(organise)
        self.organise_button.setToolTip("Preview exact copies into ExpertKnowledge/PhaseModels/TrainingInput and ExpertKnowledge/PhaseModels/Models. Original files and the selected model are preserved."
            if organise else "Choose a series to organise its phase-model files.")
        if self.method_panel is not None:
            self.method_panel._update_controls()
        mode = "select" if selection_only else "existing" if self.existing_basis.isChecked() else "scratch"
        owner = getattr(getattr(self, "phase_workflow", None), "review_owner", None)
        basis_editable = not (owner is not None and owner.process is not None)
        basis_editable = basis_editable and not (
            getattr(self.studio, "training_process", None) is not None
            or getattr(self.studio, "_series_batch_active", False) or self._model_selection_pending)
        for button in (self.existing_basis, self.scratch_basis, self.retrain_basis,
                       self.continue_basis, self.select_basis):
            button.setEnabled(basis_editable)
        self.studio.validation_model.setEnabled(basis_editable)
        self.basis_source_label.setText("Model for calibration" if selection_only else "Existing model" if mode == "existing"
            else "Image collection for new labels")
        reference = self._selected_calibration_model()
        if reference != self._model_selection_reference:
            self._model_selection_reference = reference
            self.model_selection_status.clear()
        self.use_selected_model.setVisible(selection_only)
        self.model_selection_status.setVisible(selection_only)
        self.use_selected_model.setEnabled(bool(selection_only and self.series_root is not None
            and reference and basis_editable))
        self.use_selected_model.setToolTip("Save this existing model as the calibration input. No job starts."
            if reference else "Choose an existing model from the catalogue first.")
        route = self.training_routes.get(mode)
        continuation = getattr(self, "continuation_training", None)
        prepared_training = getattr(self, "prepared_retraining", None)
        choice = self.existing_choice()
        # Retraining uses the prepared retraining, else the prepared RAW fit;
        # Continue uses only the continuation route.
        prepared_existing = (mode == "existing" and choice != "continue"
                             and prepared_training is not None and prepared_training.available())
        raw_refit = bool(getattr(photometry, "uses_prepared_raw_basis", False)) and choice != "continue"
        if selection_only:
            route = None
            reason = ("Select an existing model and save it for calibration. Labelling, validation and training "
                      "panels are disabled for this choice. No mass calibration is required to select a model.")
        elif prepared_existing:
            route = prepared_training.start
            reason = ("Retrain the selected model's saved labels on the current Setup crop, lighting and colour basis. "
                      "Enter a new model version name and choose Train. No new boxes or review session are needed. "
                      "Complete the current Setup Apply/Build before starting.")
        elif raw_refit:
            route = photometry.train
            reason = ("Retrain new optical heads on the chosen prepared RAW photographs and crop. "
                      "The selected model supplies its architecture; previous weights and input scaling are not reused."
                      if mode == "existing" else
                      "Fit new optical heads on the selected prepared RAW photographs and crop.")
        elif choice == "retrain":
            route = None
            reason = ("Retraining needs the chosen model's saved labels prepared on the current image correction. "
                      "Complete Image correction with Finish setup in Setup and choose a model with saved labels. "
                      "To keep the model and add new labels, choose Continue with new labels.")
        elif mode == "existing":
            reason = (continuation.refresh() if continuation is not None else
                "The selected model is the basis for review. A compatible existing-model training route is not available in the retained code; no fresh fit is substituted.")
        else:
            reason = "The retained fresh-training route uses its recorded radiometry settings. A saved v5 status report does not configure that route; compatible v5 training is not attached."
        if mode == "scratch" and photometry is not None and not raw_refit:
            reason = ("Fresh training uses the saved v5 input and the original fitting procedure."
                      if photometry.binding is not None else photometry.reason)
        self.basis_note.setText(reason)
        self.training_note.setText(
            reason
            if prepared_existing or not callable(route) or mode == "existing" and continuation is not None and not raw_refit
            else "Training reuses the admitted original callback and requires a new version name."
        )
        training_available = self._legacy_start_available
        if prepared_existing:
            training_available = bool(model)
        elif mode == "existing" and continuation is not None and not raw_refit:
            training_available = continuation.ready
        if (mode == "scratch" or raw_refit) and photometry is not None and not prepared_existing:
            training_available = photometry.fresh_ui_ready() and (mode != "existing" or bool(model))
        name_ready = bool(self.model_name.text().strip())
        idle = getattr(self.studio, "training_process", None) is None and not getattr(
            self.studio, "_series_batch_active", False
        )
        idle = idle and not (prepared_training is not None and prepared_training.busy)
        self.studio.start_button.setEnabled(
            self._ready
            and name_ready
            and idle
            and callable(route)
            and training_available
        )
        self.model_name.setEnabled(idle)
        self._refresh_control_reasons()
        workflow = getattr(self, "phase_workflow", None)
        if workflow is not None:
            workflow.refresh(saved_selection=saved)
        for section in self.phase_tabs.sections:
            if section.content in (self.groups[2], self.groups[3], self.groups[4]):
                section.setEnabled(not selection_only)
                section.setToolTip("Choose Train a new model or Train existing model to use this panel."
                    if selection_only else "")
        if selection_only:
            self.studio.start_button.setEnabled(False)
            for name in ("open_model_correction", "resume_session_button", "open_blind_labeling"):
                control = getattr(self.studio, name, None)
                if control is not None:
                    control.setEnabled(False)
            self.phase_tabs.setCurrentWidget(self.groups[0])
            self.workspace_stack.setCurrentWidget(self.workspace_overview)
        if idle and self.studio.status.text() in {
            "",
            "Add reviewed evidence and enter a researcher name.",
            self._native_idle_status,
        }:
            if self._ready:
                self._native_idle_status = self.training_note.text()
            elif self._checking_image_correction():
                self._native_idle_status = ("Checking — " + self._image_correction_reason()
                                            + ". Training can start when the check is done.")
            else:
                self._native_idle_status = ("Not done — " + self._image_correction_reason()
                                            + ". Saved models and sessions remain available for inspection.")
            self.studio.status.setText(self._native_idle_status)
        busy = getattr(self, "check_busy", None)
        if busy is not None:
            image = not self._ready and self._checking_image_correction()
            inputs = (bool(getattr(photometry, "checking", False))
                      and not selection_only and not prepared_existing)
            busy.setVisible(image or inputs)
            busy.setToolTip("Checking the completed Image correction" if image
                            else "Checking the selected training inputs")
        presentation = (str(self.series_root or ""), self.series_id,
                        str((model or {}).get("root") or ""),
                        self.existing_basis.isChecked(), self.review_launch_reason())
        if presentation != self._review_presentation_state:
            self._review_presentation_state = presentation
            self.review_readiness_changed.emit()

    def _refresh_control_reasons(self):
        """Explain disabled actions without weakening their existing gates."""
        studio = self.studio
        selection = self.review_selection_text()
        studio.open_model_correction.setToolTip(selection)
        self.phase_review_button.setToolTip("This page has no phase-review window attached."
            if self.phase_review is None else selection)
        self.review_selection_notice.setText(selection)
        getter = getattr(studio, "_selected_validation_model", None)
        model = getter() if callable(getter) else None
        # Native panel navigation opens the working area automatically. Keep
        # the original route callable without adding another opening step.
        self.review_route_button.hide()
        self.review_route_button.setText("Open Image correction…" if not self._ready else
            "Choose a model in Basis…" if not model else "Find saved phase maps…")
        self.review_route_button.setToolTip("Recheck saved model and map metadata; no phase maps are recalculated."
            if self._ready and model else "Open the step that supplies the missing requirement.")
        studio.validation_model.setToolTip("Choose the existing model used as the review and training basis."
            if self.existing_basis.isChecked() else
            "Use this collection's saved photographs for new labels. Its model weights and existing labels are not copied.")
        if not studio.resume_session_button.isEnabled():
            studio.resume_session_button.setToolTip(self._image_correction_tooltip()
                if not self._ready else "Select an available saved review session to resume.")
        else:
            studio.resume_session_button.setToolTip("Resume the selected saved review session.")
        evidence = getattr(studio, "add_session_evidence", None)
        if evidence is not None:
            evidence.setToolTip("Add the selected session's saved evidence to the training inputs."
                if evidence.isEnabled() else "Select a saved review session with an available decisions file.")
        training = []
        if not self._ready:
            training.append(self._image_correction_reason())
        if not self.model_name.text().strip():
            training.append("name the new model version")
        if not studio.start_button.isEnabled():
            training.append(self.training_note.text())
        studio.start_button.setToolTip("; ".join(training) if training else "Train the named new model version using the admitted route.")
        self.model_name.setToolTip("A running training job uses this name; wait for it to finish."
            if not self.model_name.isEnabled() else "Enter the name for the new model version.")
        studio.stop_button.setToolTip("Stop the current training job through its existing control."
            if studio.stop_button.isEnabled() else "No training job is running.")

    def _guarded(self, callback, *args, **kwargs):
        if self.model_selection_only():
            self.studio.review_status.setText("Choose a training option to edit phase-model inputs.")
            return None
        if not self._ready:
            self.studio.review_status.setText(self._image_correction_tooltip())
            return None
        return callback(*args, **kwargs)

    def _launch_review(self, *args, **kwargs):
        if self.model_selection_only():
            self.studio.review_status.setText(self.review_launch_reason())
            return None
        try:
            getter = getattr(self.studio, "_selected_validation_model", None)
            if callable(getter) and self.review_launch_reason():
                model = getter()
                if model and not model.get("validation_binding"):
                    self.studio._request_async_model_refresh(bindings_only=True, verify_saved=True)
                self.show_review_requirement()
                return None
            model = getter() if callable(getter) else None
            if self._has_saved_review_binding(model):
                return self._original_review(*args, **kwargs)
            return self._guarded(self._original_review, *args, **kwargs)
        except Exception as error:
            message = "Could not open phase review: " + str(error)
            self.studio.review_status.setText(message)
            self.review_selection_notice.setText(message)
            QMessageBox.warning(self, "Could not open phase review", message)
            return None

    def train(self, *_args):
        if self.model_selection_only():
            self.studio.status.setText("Select model for calibration does not start training. Choose a training option first.")
            return
        if not self._ready:
            self.studio.status.setText(self._image_correction_tooltip())
            return
        if not self.model_name.text().strip():
            self.studio.status.setText("Name the new model version before training.")
            return
        workflow = getattr(self, "phase_workflow", None)
        if workflow is not None and (reason := workflow.training_block_reason()):
            self.studio.status.setText(reason)
            return
        prepared_training = getattr(self, "prepared_retraining", None)
        choice = self.existing_choice()
        if (self.existing_basis.isChecked() and choice != "continue"
                and prepared_training is not None and prepared_training.available()):
            try:
                return prepared_training.start()
            except Exception as error:
                self.studio.status.setText(f"Retraining did not start: {type(error).__name__}: {error}")
                return
        mode = "existing" if self.existing_basis.isChecked() else "scratch"
        callback = self.training_routes.get(mode)
        photometry = getattr(self, "training_photometry", None)
        if getattr(photometry, "uses_prepared_raw_basis", False) and choice != "continue":
            callback = photometry.train
        elif choice == "retrain":
            callback = None  # Retraining never falls back to continuation.
        if not callable(callback):
            self.studio.status.setText(self.training_note.text())
            return
        if getattr(self.studio, "training_process", None) is not None or getattr(
            self.studio, "_series_batch_active", False
        ):
            return
        try:
            return workflow.run_training(callback) if workflow is not None else callback()
        except Exception as error:
            self.studio.status.setText(
                f"Training did not start: {type(error).__name__}: {error}"
            )
            QMessageBox.critical(self, "Training did not start", str(error))
            return None

    def _launch_named_job(self, request, command, working_directory, status):
        if not self._ready:
            from .calibration_model_training import require_prepared_normalisation
            require_prepared_normalisation(request, self.series_root)
        if not self.model_name.text().strip():
            raise ValueError(
                "Saved normalisation and a new model version name are required before launch."
            )
        workflow = getattr(self, "phase_workflow", None)
        if workflow is not None:
            workflow.before_launch(request)
        save_model_ui_name(
            request, self.model_name.text(), self.studio.reviewer.text().strip()
        )
        if self.series_root is not None:
            self._named_model_context[str(request["model_id"])] = (
                Path(self.series_root), Path(request["output_root"]).resolve()
            )
        result = self._original_launch(request, command, working_directory, status)
        self.model_name.clear()
        self.refresh()
        return result

    def _use_method(self, selection):
        if self.method_use is not None and (
            self._ready or getattr(selection, "active", "") == "colour_path"
        ):
            return self.method_use(selection)

    def cancel_metadata_previews(self):
        """Invalidate disposable catalogue reads without stopping training or writes."""
        studio = self.studio
        studio._model_catalog_generation = getattr(studio, "_model_catalog_generation", 0) + 1
        studio._model_catalog_pending = False
        studio._model_binding_refresh_pending = False
        for entry in getattr(studio, "model_history", ()):
            entry["validation_binding_loading"] = False
        update = getattr(studio, "_update_review_button_labels", None)
        if callable(update):
            update()
        cancel = getattr(self.normalisation, "cancel_preview", None)
        if callable(cancel):
            cancel()

    def _retraction_poll(self):
        """The studio's 5 s retraction poll, while Qt has not yet deleted it."""
        from shiboken6 import isValid
        timer = getattr(self.studio, "_retraction_poll_timer", None)
        return timer if timer is not None and isValid(timer) else None

    def showEvent(self, event):
        super().showEvent(event)
        timer = self._retraction_poll()
        if timer is not None and not timer.isActive():
            timer.start()

    def hideEvent(self, event):
        self.cancel_metadata_previews()
        # The retraction poll serves only this page; it pauses while another
        # page is shown and starts again when this one is.
        timer = self._retraction_poll()
        if timer is not None:
            timer.stop()
        super().hideEvent(event)

    def close_block_reason(self):
        """Retain owned processes until their original finished cleanup returns."""
        workflow = getattr(self, "phase_workflow", None)
        if workflow is not None and (reason := workflow.close_block_reason()):
            return reason
        if self.normalisation is not None:
            can_close = getattr(self.normalisation, "can_close", None)
            if can_close is not None and not can_close():
                return "Image correction metadata is still finishing; close after it returns."
        if getattr(self.studio, "training_process", None) is not None:
            return (
                "Training is still running or finishing. Wait for it to finish, or use "
                f"{self.studio.stop_button.text()} and wait for cleanup before closing."
            )
        if getattr(self.studio, "validation_process", None) is not None:
            return "Phase-model preparation or metadata lookup is still finishing; close after it returns."
        dialog = getattr(self, "_layout_dialog", None)
        if dialog is not None and not dialog.can_close():
            return "Phase-model file preview or copying is still finishing; close after it returns."
        return ""

    def can_close(self):
        """Report whether owned process and filesystem work has retired."""
        return not self.close_block_reason()

    def closeEvent(self, event):
        reason = self.close_block_reason()
        if reason:
            self.studio.status.setText(reason)
            self.studio.review_status.setText(reason)
            event.ignore()
            return
        super().closeEvent(event)

    def _organise_phase_files(self):
        """Present the original file workflow in the native working area."""
        if self.series_root is None or not str(self.series_id or "").strip():
            return
        from .phase_model_layout_dialog import PhaseModelLayoutDialog
        previous = self._layout_dialog
        if previous is not None:
            if previous.series_root == self.series_root or not previous.can_close():
                self.workspace_stack.setCurrentWidget(previous)
                previous.show()
                if previous.series_root != self.series_root:
                    previous.summary.setText(
                        "The earlier series' file operation is still finishing. "
                        "Wait for it to return before opening the current series.")
                return
            previous.close()
        original_root = self.series_root
        dialog = PhaseModelLayoutDialog(original_root, self.workspace_stack)
        self._layout_dialog = dialog
        dialog.setWindowFlags(Qt.WindowType.Widget)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        for button in dialog.findChildren(QPushButton):
            button.setAutoDefault(False)
        # Keep the original actions readable beside the persistent left panels.
        dialog.layout().insertWidget(2, dialog.add_session)
        dialog.locations_changed.connect(lambda: self._phase_locations_changed(original_root))
        dialog.finished.connect(lambda _result: self._phase_files_closed(dialog))
        self.workspace_stack.addWidget(dialog)
        self.workspace_stack.setCurrentWidget(dialog)
        dialog.show()

    def _phase_files_closed(self, dialog):
        """Return after the original close/worker guards accept dismissal."""
        if self.workspace_stack.currentWidget() is dialog:
            self.workspace_stack.setCurrentWidget(self.workspace_overview)
        self.workspace_stack.removeWidget(dialog)
        if self._layout_dialog is dialog:
            self._layout_dialog = None
        dialog.deleteLater()

    def _phase_locations_changed(self, original_root):
        """Refresh copied paths while retaining the exact selected model."""
        # A background copy retains its original series even if the user opens
        # another one. Never apply that series' model preference to the new page.
        if self.series_root != original_root:
            return
        from .review_storage import resolve_review_path
        selected = dict(self.studio._selected_validation_model() or {})
        self.studio._sync_default_model_libraries()
        if selected.get("root"):
            relocated = resolve_review_path(selected["root"])
            self.studio.set_saved_model_preference(relocated, context_key=(
                "storage-layout", str(original_root), str(selected["root"]),
                str(selected.get("version_sha256") or "")))
        else:
            self.studio._request_async_model_refresh()
        self.refresh()

    def _import_model(self):
        if (not self._ready and not self.model_selection_only()) or self.series_root is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Import a saved phase-model version"
        )
        if not folder:
            return
        try:
            import_saved_model(self.series_root, self.series_id, Path(folder))
            self.studio._request_async_model_refresh()
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, "Model was not imported", str(error))
