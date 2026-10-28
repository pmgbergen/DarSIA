"""Actual native Production wiring; no producer, worker or probe is invoked."""
from types import SimpleNamespace

import pytest

from darsia.fluidflower.jobs_panel import NewAnalysisPage
from darsia.fluidflower.production_page import ProductionPage
from darsia.fluidflower.production_selection import create_production_job
from .test_native_jobs_owner import local_registry, workspace
from .test_native_sidebar import native_window


def test_standalone_jobs_owner_retains_its_original_default_page(workspace):
    assert type(workspace.new_analysis) is NewAnalysisPage
    assert workspace.panel.new_analysis is workspace.new_analysis


def test_native_production_and_jobs_share_the_original_owner_and_controls(native_window, monkeypatch):
    window = native_window
    owner = window.ensure_jobs()
    page = owner.new_analysis
    assert isinstance(page, ProductionPage)
    assert owner._creator is create_production_job
    assert owner.panel.new_analysis is page

    def forbidden(*args, **kwargs):
        pytest.fail("Navigation cannot create, launch, probe or start a metadata call")

    monkeypatch.setattr(owner.panel, "create_job", forbidden)
    monkeypatch.setattr(owner.panel, "start_selected_machine", forbidden)
    monkeypatch.setattr(owner.panel, "_start_call", forbidden)
    window.show_jobs(start_analysis=True)
    assert window.jobs_workspace is owner and owner.view == "new_analysis"
    assert window.sidebar.display_selection("jobs", "new_analysis")[0] == "analysis"
    assert window.content_stack.currentWidget() is owner
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.wizard_action.isEnabled()
    page.jobs_button.click()
    assert window.sidebar.get_selected() == ("jobs", "overview")
    assert window.sidebar.display_selection("jobs", "overview")[0] == "jobs"
    assert owner.view == "jobs" and window.jobs_workspace is owner
    assert owner.panel._session_requests == []
    assert owner.can_close()


@pytest.mark.parametrize("pending", ["calibration", "facts", "preparation"])
def test_native_close_waits_for_production_work_to_retire(native_window, pending):
    window = native_window
    owner = window.ensure_jobs()
    page = owner.new_analysis
    field = dict(calibration="_calibration_thread", facts="_facts_thread",
                 preparation="_preparation_dialog")[pending]
    setattr(page, field, SimpleNamespace(can_close=lambda: False))
    try:
        assert not page.can_close() and not owner.panel.can_close() and not owner.can_close()
        assert not window.close()
        assert window.isVisible()
    finally:
        setattr(page, field, None)
    assert page.can_close() and owner.can_close()
