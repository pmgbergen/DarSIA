"""Explicit panel clicks retain the original review admission and owner."""

from types import SimpleNamespace as NS
from threading import Event

import pytest
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from darsia.fluidflower.native_phase_panel_navigation import NativePhasePanelNavigation
from darsia.fluidflower.workflow_section import WorkflowSection
from .test_native_phase_steps import setup  # noqa: F401
from .test_native_inline_phase_review import review  # noqa: F401


class ReviewPanel(QWidget):
    def __init__(self, steps):
        super().__init__()
        self.steps, self.process = steps, None
        self.roles = []
        self.notice = QLabel()
        QVBoxLayout(self).addWidget(self.notice)

    def show_requirement(self, text):
        self.notice.setText(text)
        self.steps.workspace_stack.setCurrentWidget(self)

    def open_role(self, mode):
        assert self.process is not None
        self.roles.append(mode)
        self.steps.workspace_stack.setCurrentWidget(self)


@pytest.fixture
def navigation(setup, qtbot, tmp_path):
    steps = setup.steps
    steps.set_context(tmp_path, "series")
    steps.resize(1400, 900)
    steps.show()
    qtbot.waitExposed(steps)
    panel = ReviewPanel(steps)
    steps.workspace_stack.addWidget(panel)
    state = NS(reason="", model={}, calls=[])
    steps.review_launch_reason = lambda role=None: state.reason
    steps.studio._selected_validation_model = lambda: state.model

    def launch(mode):
        state.calls.append(mode)
        panel.process = object()

    steps._launch_review = launch
    controller = NativePhasePanelNavigation(steps, panel)
    return NS(steps=steps, panel=panel, state=state, controller=controller)


def test_only_explicit_label_or_blind_click_opens_and_reuses_owner(navigation):
    n = navigation
    for section in n.controller.sections.values():
        section.set_expanded(False)
        section.set_expanded(True)
    n.steps.phase_tabs.setCurrentWidget(n.steps.groups[2])
    assert n.state.calls == [] and n.panel.process is None
    n.controller.sections[2].toggle.click()
    process = n.panel.process
    assert n.state.calls == ["open_correction"]
    assert n.steps.workspace_stack.currentWidget() is n.panel
    for number in (2, 3, 3, 2):
        n.controller.sections[number].toggle.click()
        assert n.controller.sections[number].is_expanded()
        assert n.panel.process is process
    assert n.state.calls == ["open_correction"]
    assert n.panel.roles == ["open_correction", "open_correction",
                             "blind_region_labeling", "blind_region_labeling", "open_correction"]
    n.controller.sections[4].toggle.click()
    assert n.steps.workspace_stack.currentWidget() is n.steps.workspace_overview
    assert n.panel.process is process and n.steps.studio.training_process is None


def test_phase_panel_selection_collapses_others_and_retains_the_active_review(navigation):
    n = navigation
    panels = n.steps.phase_tabs.sections
    n.steps.model_name.setText("Retained model name")
    for number in (2, 3, 4, 0):
        n.steps.phase_tabs.setCurrentWidget(n.steps.groups[number])
        assert [panel.content for panel in panels if panel.is_expanded()] == [n.steps.groups[number]]
    assert n.state.calls == [] and n.panel.process is None

    n.controller.sections[2].toggle.click()
    process = n.panel.process
    for number in (3, 4, 0, 2):
        selected = next(panel for panel in panels if panel.content is n.steps.groups[number])
        selected.toggle.click()
        assert [panel for panel in panels if panel.is_expanded()] == [selected]
        assert all(panel.isVisible() for panel in panels)
        assert n.panel.process is process
    assert n.state.calls == ["open_correction"]
    assert n.steps.model_name.text() == "Retained model name"
    assert n.steps.workspace_stack.currentWidget() is n.panel


def test_visible_panel_opens_when_its_requirement_finishes(navigation, qtbot):
    n = navigation
    n.state.reason = "Complete and save Image preparation."
    n.controller.sections[2].toggle.click()
    assert n.steps.workspace_stack.currentWidget() is n.panel
    assert n.panel.notice.text() == n.state.reason
    assert n.state.calls == [] and n.controller._pending is not None
    n.state.reason = ""
    n.steps.review_readiness_changed.emit()
    qtbot.waitUntil(lambda: n.state.calls == ["open_correction"])
    assert n.controller._pending is None
    n.steps.review_readiness_changed.emit()
    qtbot.wait(1)
    assert n.state.calls == ["open_correction"]


@pytest.mark.parametrize("event", ["enabled", "selection", "tooltip"])
def test_loading_wish_retries_once_on_existing_metadata_events(navigation, qtbot, event):
    n = navigation
    button = n.steps.studio.open_model_correction
    button.setEnabled(False)
    n.state.reason = "The selected model's saved phase maps are still being checked."
    n.state.model = dict(validation_binding_loading=True)
    n.controller.sections[3].toggle.click()
    assert n.controller._pending is not None and n.state.calls == []
    assert n.panel.notice.text() == n.state.reason
    n.state.reason = ""
    n.state.model = dict(validation_binding={})
    if event == "enabled":
        button.setEnabled(True)
    elif event == "selection":
        n.steps.studio.validation_model.addItem("Loaded model", "loaded")
        n.steps.studio.validation_model.setCurrentIndex(1)
    else:
        button.setToolTip("Saved phase maps are ready")
    qtbot.waitUntil(lambda: len(n.state.calls) == 1)
    button.setToolTip("Another presentation update")
    qtbot.wait(1)
    assert n.state.calls == ["blind_region_labeling"]
    assert n.controller._pending is None


@pytest.mark.parametrize("leave", ["train", "workspace", "section", "hidden", "series"])
def test_navigation_cancels_loading_wish(navigation, qtbot, leave):
    n = navigation
    n.state.reason = "Checking saved phase maps."
    n.state.model = dict(validation_binding_loading=True)
    n.controller.sections[2].toggle.click()
    if leave == "train":
        n.controller.sections[4].toggle.click()
    elif leave == "workspace":
        n.steps.workspace_stack.setCurrentWidget(n.steps.workspace_overview)
    elif leave == "section":
        n.steps.show_section("normalisation")
    elif leave == "hidden":
        n.steps.hide()
    else:
        n.steps.series_id = "another-series"
    n.state.reason = ""
    n.state.model = {}
    n.steps.studio.open_model_correction.setToolTip("Ready after navigation")
    qtbot.wait(1)
    assert n.state.calls == [] and n.controller._pending is None


def test_basis_before_label_uses_content_identity_and_cancels_pending(setup, qtbot):
    steps = setup.steps
    # Also exercise this against older layouts without a Basis panel.
    basis = next((section for section in steps.phase_tabs.sections
                  if section.content not in (steps.groups[2], steps.groups[3], steps.groups[4])), None)
    if basis is None:
        basis = WorkflowSection(None, "Basis", QWidget())
        steps.phase_tabs.sections.insert(0, basis)
        steps.phase_tabs.layout().insertWidget(0, basis)
    panel = ReviewPanel(steps)
    steps.workspace_stack.addWidget(panel)
    controller = NativePhasePanelNavigation(steps, panel)
    steps.review_launch_reason = lambda role=None: "Checking saved phase maps."
    steps.studio._selected_validation_model = lambda: dict(validation_binding_loading=True)
    calls = []
    steps._launch_review = calls.append
    controller.sections[2].toggle.click()
    assert controller._pending is not None
    basis.toggle.click()
    assert controller._pending is None and calls == []


def test_original_launch_failure_remains_in_the_inline_working_area(navigation):
    n = navigation
    def fail(mode):
        n.state.calls.append(mode)
        n.steps.studio.review_status.setText("The original saved review folder is unavailable.")
    n.steps._launch_review = fail
    n.controller.sections[2].toggle.click()
    assert n.panel.process is None
    assert n.panel.notice.text() == "The original saved review folder is unavailable."
    assert n.steps.workspace_stack.currentWidget() is n.panel


@pytest.mark.parametrize("review", ["panels"], indirect=True)
def test_panel_clicks_reach_real_inline_tools_and_reuse_loading_session(review, qtbot, monkeypatch):
    from darsia.fluidflower import native_inline_phase_review as inline

    r = review
    binding = {"selected_candidates": {"ac01": "candidate-a"}}
    r.model["validation_binding"] = binding
    r.steps.studio._selected_validation_model = lambda: r.model
    r.steps._ready = True
    r.steps._legacy_availability["open_model_correction"] = True
    r.steps._original_review = lambda mode: r.owner.start(mode, binding)
    navigation = NativePhasePanelNavigation(r.steps, r.panel)
    entered, release = Event(), Event()
    prepare = inline._prepare

    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return prepare(*args)

    monkeypatch.setattr(inline, "_prepare", delayed)
    navigation.sections[2].toggle.click()
    process = r.owner.process
    assert process is not None and r.stack.currentWidget() is r.panel
    try:
        qtbot.waitUntil(entered.is_set)
        navigation.sections[3].toggle.click()
        assert r.owner.process is process and r.panel.requested_role == "blind"
    finally:
        release.set()
    qtbot.waitUntil(lambda: process.review_window is not None, timeout=10000)
    window = process.review_window
    assert window.workflow_role.currentData() == "blind"
    navigation.sections[2].toggle.click()
    assert window.workflow_role.currentData() == "label"
    navigation.sections[2].toggle.click()
    assert r.owner.process is process and window.isVisible()
    assert r.host.grab().save(str(r.root / "direct-label-panel.png"))
    navigation.sections[4].toggle.click()
    assert r.stack.currentWidget() is r.steps.workspace_overview
    assert r.owner.process is process and r.steps.studio.training_process is None
