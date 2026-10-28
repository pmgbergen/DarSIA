from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import importlib.util
import json
import sys
import threading

import pytest
from PySide6.QtCore import QPoint, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabBar,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower import native_phase_steps as ui


class FakeSettings:
    def __init__(self):
        self.writes = []

    def value(self, key, default=None, *args, **kwargs):
        return default

    def setValue(self, key, value):
        self.writes.append((key, value))


class FakeStudio(QWidget):
    training_completed = Signal(str)

    def __init__(self, *, metadata_preview_only=False):
        super().__init__()
        self._native_metadata_preview_only = bool(metadata_preview_only)
        self.calls = []
        self.settings = FakeSettings()
        self._model_filter_settings = self.settings
        self._configured_series_roots = {}
        self.training_process = None
        self._series_batch_active = False
        self._reviewer_project_root = None
        self._reviewer_save_timer = QTimer(self)
        self._project_root_timer = QTimer(self)
        self._retraction_poll_timer = QTimer(self)
        layout = QVBoxLayout(self)
        self.validation_model = QComboBox()
        self.validation_model.addItem("Original model label", "original-root")
        self.saved_sessions = QComboBox()
        self.saved_sessions.addItem("Original saved session", "session-a")
        self.validation_model_summary = QLabel("Original model summary")
        self.catalog_loading_status = QLabel("Original catalogue status")
        self.open_model_correction = QPushButton("Open model correction")
        self.open_model_correction.clicked.connect(
            lambda: self._launch_review("open_correction")
        )
        self.review_status = QLabel("Original review status")
        self.session_summary = QLabel("Original session summary")
        self.resume_session_button = QPushButton("Resume session")
        self.resume_session_button.clicked.connect(self._resume_selected_session)
        self.add_session_evidence = QPushButton("Add evidence")
        self.pending_training_sessions = QTableWidget(0, 3)
        self.model_library = QLineEdit("original-library")
        self.reviewer = QLineEdit("Stored preference must not win")
        self.reviewer_project = QLineEdit()
        self.training_series = QLineEdit("old-series")
        self.status = QLabel("Original training status")
        self.log = QTextEdit()
        self.start_button = QPushButton("Original Train")
        self.start_button.clicked.connect(self._start_training)
        self.stop_button = QPushButton("Original Stop")
        self.train_further_button = QLabel("Nonfunctional retained continuation label")
        self.training_tab = QScrollArea()
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        self.original_parameter = QLineEdit("Original numerical setting untouched")
        inner_layout.addWidget(self.original_parameter)
        self.training_tab.setWidget(inner)
        for widget in (
            self.validation_model,
            self.saved_sessions,
            self.training_tab,
            self.reviewer,
            self.reviewer_project,
            self.training_series,
        ):
            layout.addWidget(widget)

    def _start_training(self):
        self.calls.append("original-training")

    def _update_action_states(self, *args):
        self.start_button.setEnabled(True)

    def _launch_training_job(self, request, command, working_directory, status):
        self.calls.append(
            ("original-launch", request, command, working_directory, status)
        )
        self.training_process = object()

    def set_model_series(self, series):
        self._active_model_series = series
        self._model_filter_settings.setValue("modelHistory/seriesFilter", series)

    def set_project_root(self, root, **kwargs):
        self.calls.append("original-project-preferences")

    def set_saved_model_preference(self, root, *, context_key):
        self.saved_preference = (root, context_key)

    def _sync_default_model_libraries(self):
        self.calls.append("original-library-sync")

    def _launch_review(self, mode):
        self.calls.append(("original-review", mode))

    def _start_review_process(self, mode, binding, *, session_id=""):
        self.calls.append(("original-process", mode, binding, session_id))

    def _resume_selected_session(self):
        self.calls.append("original-resume")

    def _validation_model_label(self, entry):
        return entry["model_id"]

    def _request_async_model_refresh(self):
        self.calls.append("original-model-refresh")


class FakeNormalisation(QWidget):
    readiness_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.ready = False
        self.contexts = []

    def set_context(self, root, series):
        self.contexts.append((root, series))
        self.readiness_changed.emit(self.ready)


@pytest.fixture
def setup(qtbot):
    studio = FakeStudio()
    original_choices = studio.validation_model
    original_sessions = studio.saved_sessions
    normalisation = FakeNormalisation()
    steps = ui.NativePhaseSteps(
        studio, normalisation=normalisation, actor_provider=lambda: "WindowsUser"
    )
    qtbot.addWidget(studio)
    qtbot.addWidget(steps)
    return NS(
        studio=studio,
        normalisation=normalisation,
        steps=steps,
        choices=original_choices,
        sessions=original_sessions,
    )


def test_sections_reuse_actual_widgets_and_original_parameters(setup):
    s = setup
    # Image correction, Phase model (four panels) and Use for this series.
    assert sorted(s.steps.groups) == [0, 1, 2, 3, 4, 6]
    assert list(s.steps.sections) == ["normalisation", "phase_model", "use"]
    tabs = s.steps.phase_tabs
    assert [tabs.tabText(index) for index in range(tabs.count())] == ["Basis", "Label phases", "Blindly validate phase model", "Train"]
    assert [tabs.widget(index) for index in range(tabs.count())] == [s.steps.groups[i] for i in (0, 2, 3, 4)]
    assert s.studio.validation_model is s.choices
    assert s.studio.saved_sessions is s.sessions
    assert s.choices.parentWidget() is s.steps.groups[0]
    assert s.studio.catalog_loading_status.parentWidget() is s.steps.groups[0]
    assert s.steps.groups[0].isAncestorOf(s.steps.existing_basis)
    assert s.steps.groups[0].isAncestorOf(s.steps.scratch_basis)
    assert s.sessions.parentWidget() is s.steps.groups[4]
    assert s.studio.open_model_correction.parentWidget() is s.steps.groups[2]
    assert s.studio.original_parameter.text() == "Original numerical setting untouched"
    assert s.studio.isHidden()


def test_phase_panels_stay_on_the_left_while_the_working_area_changes(setup, qtbot):
    steps = setup.steps
    steps.resize(1400, 900)
    steps.show()
    qtbot.waitUntil(lambda: steps.phase_workspace.width() > 1000)
    panels = steps.phase_tabs.sections
    assert not steps.phase_workspace.findChildren(QTabBar)
    origins = [panel.mapTo(steps, QPoint(0, 0)) for panel in panels]
    assert len({point.x() for point in origins}) == 1
    assert all(left.y() < right.y() for left, right in zip(origins, origins[1:]))
    right = steps.workspace_stack.mapTo(steps, QPoint(0, 0)).x()
    assert all(point.x() + panel.width() <= right for point, panel in zip(origins, panels))

    steps.phase_tabs.setCurrentWidget(steps.groups[4])
    steps.model_name.setText("Retain this model name")
    panels[3].toggle.click()
    panels[3].toggle.click()
    assert steps.model_name.text() == "Retain this model name"
    working = QLabel("Review working area")
    steps.workspace_stack.addWidget(working)
    steps.workspace_stack.setCurrentWidget(working)
    assert all(panel.isVisible() for panel in panels)
    assert working.isVisible()
    steps.workspace_stack.setCurrentWidget(steps.workspace_overview)
    assert all(panel.isVisible() for panel in panels)
    assert not setup.studio.calls


def test_basis_choice_is_retained_and_locked_while_its_review_is_open(setup):
    steps = setup.steps
    selected = steps.studio.validation_model.currentIndex()
    steps.scratch_basis.setChecked(True)
    assert steps.basis_source_label.text() == "Image collection for new labels"
    assert steps.studio.validation_model.isEnabled()
    assert steps.studio.validation_model.currentIndex() == selected
    steps.phase_tabs.setCurrentWidget(steps.groups[2])
    assert steps.scratch_basis.isChecked()
    steps.phase_workflow.review_owner.process = object()
    try:
        steps.refresh()
        assert not steps.scratch_basis.isEnabled()
        assert not steps.existing_basis.isEnabled()
        assert not steps.studio.validation_model.isEnabled()
    finally:
        steps.phase_workflow.review_owner.process = None
    steps.refresh()
    assert steps.scratch_basis.isEnabled() and steps.studio.validation_model.isEnabled()
    assert steps.scratch_basis.isChecked()


def test_organise_files_is_a_persistent_child_until_original_close(setup, qtbot, tmp_path):
    steps = setup.steps
    steps.set_context(tmp_path, "SERIES_X")
    steps.resize(1400, 900)
    steps.show()
    qtbot.waitExposed(steps)
    windows = {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()}
    steps._organise_phase_files()
    dialog = steps._layout_dialog
    assert steps.workspace_stack.currentWidget() is dialog
    assert dialog.isVisible() and not dialog.isWindow() and not dialog.isModal()
    assert dialog.window() is steps
    assert {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()} == windows
    assert dialog.plan is None and dialog._thread is None and not dialog.copy.isEnabled()
    dialog.sessions.append("selected-session")
    steps.workspace_stack.setCurrentWidget(steps.workspace_overview)
    steps._organise_phase_files()
    assert steps._layout_dialog is dialog and dialog.sessions == ["selected-session"]
    assert all(section.isVisible() for section in steps.phase_tabs.sections)
    assert steps.grab().save(str(tmp_path / "organise-inline.png"))
    dialog.close_button.click()
    assert steps._layout_dialog is None
    assert steps.workspace_stack.currentWidget() is steps.workspace_overview
    steps._organise_phase_files()
    reopened = steps._layout_dialog
    assert reopened is not dialog and reopened.sessions == []
    reopened.reject()
    assert steps._layout_dialog is None and steps.can_close()


def test_organise_files_preserves_busy_owner_and_series_binding(setup, qtbot, tmp_path):
    steps = setup.steps
    first, second = tmp_path / "first", tmp_path / "second"
    steps.set_context(first, "FIRST")
    steps.show()
    steps._organise_phase_files()
    dialog = steps._layout_dialog
    entered, release = threading.Event(), threading.Event()
    calls = []

    def planner(root, *, external_sessions):
        calls.append((root, external_sessions))
        entered.set()
        assert release.wait(5)
        return dict(files=[], bytes_to_copy=0, paths={}, notices=[])

    dialog._planner = planner
    dialog.sessions.append("retained-session")
    dialog.preview.click()
    try:
        qtbot.waitUntil(entered.is_set)
        assert not steps.can_close()
        dialog.close_button.click()
        assert dialog.isVisible() and steps._layout_dialog is dialog
        steps.set_context(second, "SECOND")
        assert steps.workspace_stack.currentWidget() is steps.workspace_overview
        steps._organise_phase_files()
        assert steps._layout_dialog is dialog and dialog.series_root == first
        assert "earlier series" in dialog.summary.text()
        assert not dialog.copy.isEnabled()
    finally:
        release.set()
        qtbot.waitUntil(dialog.can_close)
    assert calls == [(first, ("retained-session",))]
    steps._organise_phase_files()
    assert steps._layout_dialog is not dialog and steps._layout_dialog.series_root == second
    assert steps._layout_dialog.plan is None and steps.can_close()


def test_context_uses_logged_in_user_and_memory_preferences_only(setup, tmp_path):
    s = setup
    s.steps.set_context(tmp_path, "SERIES_X")
    assert s.studio.reviewer.text() == "WindowsUser"
    assert s.studio.training_series.text() == "seriesx"
    assert not s.studio.settings.writes
    assert "original-project-preferences" not in s.studio.calls
    assert not list(tmp_path.iterdir())
    assert (
        s.studio._model_filter_settings.value("modelHistory/seriesFilter") == "SERIES_X"
    )
    s.studio._save_reviewer_preference()
    assert not list(tmp_path.iterdir())


def test_missing_normalisation_blocks_original_direct_review_and_resume(setup):
    s = setup
    s.studio._launch_review("open_correction")
    s.studio._resume_selected_session()
    s.studio.resume_session_button.clicked.emit()
    assert not s.studio.calls
    assert all(s.steps.groups[i].isEnabled() for i in (2, 3, 4, 6))
    assert not s.studio.open_model_correction.isEnabled()
    assert s.studio.saved_sessions.isEnabled()
    s.steps.set_normalisation_ready(True)
    s.studio._launch_review("open_correction")
    s.studio._resume_selected_session()
    assert s.studio.calls == [("original-review", "open_correction"), "original-resume"]


def test_no_compatible_route_never_falls_back_to_fresh_training(setup):
    s = setup
    s.steps.set_normalisation_ready(True)
    s.steps.model_name.setText("A name does not admit a different method")
    s.steps.train()
    s.steps.scratch_basis.setChecked(True)
    s.steps.train()
    assert not s.studio.start_button.isEnabled()
    assert "original-training" not in s.studio.calls
    assert "v5" in s.steps.training_note.text()


def test_selected_model_retrains_current_raw_basis_without_parent_weight_continuation(setup, monkeypatch):
    s = setup
    calls = []
    s.studio._selected_validation_model = lambda: {"model_id": "selected-template"}
    s.steps.training_photometry = NS(
        refresh=lambda: True, uses_prepared_raw_basis=True, binding=object(),
        fresh_ui_ready=lambda: True, train=lambda: calls.append("current-raw-refit"))
    s.steps.continuation_training = NS(ready=False,
        refresh=lambda: pytest.fail("New RAW basis must not inherit old weights/scaling"))
    monkeypatch.setattr(s.steps.phase_workflow, "run_training", lambda callback: callback())
    monkeypatch.setattr(s.steps.phase_workflow, "training_block_reason", lambda: "")
    s.steps.set_normalisation_ready(True)
    s.steps.model_name.setText("Retrained current basis")
    s.steps.existing_basis.setChecked(True)
    s.steps.refresh()
    assert s.studio.start_button.isEnabled()
    assert "previous weights" in s.steps.basis_note.text()
    s.steps.train()
    assert calls == ["current-raw-refit"]
    assert "original-training" not in s.studio.calls


def test_admitted_original_callback_still_requires_gate_and_name(setup, monkeypatch):
    s = setup
    # This fixture exercises the existing route/name gate without review files.
    # Real lifecycle admission is covered by test_native_phase_review_owner.
    monkeypatch.setattr(s.steps.phase_workflow, "run_training", lambda callback: callback())
    s.steps.training_routes["scratch"] = s.steps._original_start
    s.steps.scratch_basis.setChecked(True)
    s.steps.model_name.setText("Named model")
    s.steps.train()
    assert "original-training" not in s.studio.calls
    s.steps.set_normalisation_ready(True)
    s.steps.model_name.clear()
    s.steps.train()
    assert "original-training" not in s.studio.calls
    s.steps.model_name.setText("Named model")
    s.steps.train()
    assert s.studio.calls == ["original-training"]


def make_request(tmp_path):
    root = tmp_path / "phase-model-test"
    root.mkdir()
    request = dict(
        model_id="phase-model-test", output_root=str(root), settings={"keep": [1, 2]}
    )
    path = root / "training_request.json"
    raw = json.dumps(request).encode()
    path.write_bytes(raw)
    request["request_path"] = str(path)
    return request, path, raw


def test_ui_name_bound_to_actual_request_without_mutating_science(tmp_path):
    request, source, raw = make_request(tmp_path)
    payload = ui.save_model_ui_name(request, "Named version", "WindowsUser")
    assert source.read_bytes() == raw
    assert payload["training_request_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        ui.saved_model_ui_name(
            dict(root=request["output_root"], model_id=request["model_id"])
        )
        == "Named version"
    )
    with pytest.raises(ValueError, match="already has"):
        ui.save_model_ui_name(request, "Replacement name", "WindowsUser")
    source.write_bytes(raw + b"\n")
    assert not ui.saved_model_ui_name(
        dict(root=request["output_root"], model_id=request["model_id"])
    )


def test_named_launch_reuses_same_request_command_and_callback(setup, tmp_path, monkeypatch):
    s = setup
    # The synthetic original launch has no prepared review request lifecycle.
    monkeypatch.setattr(s.steps.phase_workflow, "before_launch", lambda request: None)
    request, source, raw = make_request(tmp_path)
    command = ["original", "unchanged", "command"]
    s.steps.set_normalisation_ready(True)
    s.steps.model_name.setText("Explicit model name")
    s.studio.reviewer.setText("WindowsUser")
    s.studio._launch_training_job(request, command, tmp_path, "Original status")
    call = s.studio.calls[-1]
    assert call[0] == "original-launch" and call[1] is request and call[2] is command
    assert source.read_bytes() == raw
    assert not s.steps.model_name.text()
    assert not s.studio.start_button.isEnabled()


def test_name_or_gate_missing_refuses_before_original_launch(setup, tmp_path):
    s = setup
    request, source, raw = make_request(tmp_path)
    with pytest.raises(ValueError, match="required before launch"):
        s.studio._launch_training_job(request, [], tmp_path, "")
    s.steps.set_normalisation_ready(True)
    with pytest.raises(ValueError, match="required before launch"):
        s.studio._launch_training_job(request, [], tmp_path, "")
    assert source.read_bytes() == raw
    assert not (Path(request["output_root"]) / "model_ui_name.json").exists()


def test_import_is_explicit_and_does_not_copy_model(setup, tmp_path, monkeypatch):
    from types import ModuleType

    s = setup
    imported = tmp_path / "model"
    root = tmp_path / "series"
    calls = []
    from darsia.fluidflower import model_index_names as library

    monkeypatch.setattr(
        library, "index_model_with_ui_name", lambda *args: calls.append(args)
    )
    history = ModuleType("darsia.fluidflower.phase_model_history")
    history.inspect_phase_model_history_entry = lambda folder: {"datasets": ["seriesx"]}
    monkeypatch.setitem(sys.modules, history.__name__, history)
    training = ModuleType("darsia.fluidflower.ui.model_training")
    training._models_for_series = lambda entries, series: (
        entries if series == "seriesx" else []
    )
    monkeypatch.setitem(sys.modules, training.__name__, training)
    monkeypatch.setattr(
        ui.QFileDialog, "getExistingDirectory", lambda *_: str(imported)
    )
    s.steps.series_root = root
    s.steps.series_id = "seriesx"
    s.steps._import_model()
    assert not calls
    s.steps.set_normalisation_ready(True)
    s.steps._import_model()
    assert calls == [(root, imported)]
    assert s.studio.calls == ["original-model-refresh"]
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="does not belong"):
        ui.import_saved_model(root, "foreign", imported)
    assert len(calls) == 1


def test_other_instances_and_class_methods_remain_unchanged(setup, qtbot):
    other = FakeStudio()
    qtbot.addWidget(other)
    other._launch_review("open_correction")
    other._start_training()
    assert other.calls == [("original-review", "open_correction"), "original-training"]
    assert not isinstance(other._model_filter_settings, ui.SessionSettings)


def test_nested_context_row_hides_label_and_wrapper_only(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    outer = QVBoxLayout(host)
    form = QFormLayout()
    outer.addLayout(form)
    wrapper = QWidget()
    nested = QVBoxLayout(wrapper)
    field = QLineEdit("Existing path")
    nested.addWidget(field)
    form.addRow("Experimental series", wrapper)
    kept = QLineEdit("Existing numerical setting")
    form.addRow("Numerical setting", kept)
    ui._hide_form_field(field)
    assert wrapper.isHidden()
    assert form.labelForField(wrapper).isHidden()
    assert not kept.isHidden() and kept.text() == "Existing numerical setting"


def test_adapter_with_real_retained_studio_controls(qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower.ui import model_training as module
    from darsia.fluidflower.series_method import SeriesMethodPanel

    monkeypatch.setattr(
        module, "configured_series", lambda: (("ffac", "Local FF_AC", tmp_path),)
    )
    monkeypatch.setattr(
        module.SharedPhaseModelTrainingPage,
        "_start_review_session_discovery",
        lambda self: None,
    )
    monkeypatch.setattr(
        module.SharedPhaseModelTrainingPage,
        "_request_async_model_refresh",
        lambda self, **kw: None,
    )
    studio = module.SharedPhaseModelTrainingPage()
    qtbot.addWidget(studio)
    original_dropdown = studio.validation_model
    original_training = studio.start_button
    method = SeriesMethodPanel()
    reviews = []
    steps = ui.NativePhaseSteps(
        studio,
        method_panel=method,
        phase_review=lambda: reviews.append("phase review"),
        actor_provider=lambda: "local-test-user",
    )
    qtbot.addWidget(steps)
    steps.set_context(tmp_path, "ffac")
    steps.show()
    qtbot.wait(20)
    assert sorted(steps.groups) == [0, 1, 2, 3, 4, 6]
    # The page's one scroll area; the advanced settings hold the original
    # training content without its scroll area (N141).
    for area in steps.findChildren(QScrollArea):
        parent = area.parentWidget()
        while parent is not None and parent is not steps:
            assert not isinstance(parent, QScrollArea), area
            parent = parent.parentWidget()
    assert steps.advanced.isCheckable() and not steps.advanced.isChecked()
    assert steps.advanced.isHidden()
    assert not steps.advanced.findChildren(QScrollArea)
    steps.phase_review_button.click()
    assert reviews == ["phase review"]
    assert studio.validation_model is original_dropdown
    assert studio.start_button is original_training
    assert steps.groups[0].isAncestorOf(original_dropdown)
    assert steps.groups[4].isAncestorOf(original_training)
    assert steps.groups[6].isAncestorOf(method)
    assert not original_training.isEnabled()
    assert not studio.open_model_correction.isEnabled()
    assert studio.reviewer.text() == "local-test-user"
    assert not studio.tabs.isVisible()
    assert not studio.reviewer.isVisible()
    assert not studio.reviewer_project.isVisible()
    assert studio.training_process is None
    steps.close()
    studio.close()


def test_native_wording_survives_original_refresh_without_masking_errors(setup):
    studio, steps = setup.studio, setup.steps
    studio.open_model_correction.setText("Start new model-validation session")
    studio.status.setText("Add reviewed evidence and enter a researcher name.")
    steps.refresh()
    assert studio.open_model_correction.text() == "Open model correction"
    assert "Image correction" in studio.status.text()
    assert "researcher" not in studio.status.text()
    studio.status.setText("RuntimeError: preserve the original failure verbatim")
    steps.refresh()
    assert (
        studio.status.text() == "RuntimeError: preserve the original failure verbatim"
    )


def test_the_retraction_poll_runs_only_while_the_page_is_shown(qtbot):
    studio = FakeStudio()
    steps = ui.NativePhaseSteps(studio, normalisation=FakeNormalisation())
    qtbot.addWidget(studio)
    qtbot.addWidget(steps)
    timer = studio._retraction_poll_timer
    timer.setInterval(5000)
    steps.show()
    assert timer.isActive()
    steps.hide()
    assert not timer.isActive()
    steps.show()
    assert timer.isActive()
