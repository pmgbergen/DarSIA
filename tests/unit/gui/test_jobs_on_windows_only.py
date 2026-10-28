"""On a computer without the Windows job layer, job pages say where jobs run.

The other system is simulated by replacing ``jobs_supported``; nothing is
prepared or started in any test.
"""

from types import SimpleNamespace

import pytest

from darsia.fluidflower import native_jobs_registry as registry
from darsia.fluidflower import native_jobs_workspace as ui
from darsia.fluidflower import platform_support
from darsia.fluidflower.platform_support import JOBS_ON_WINDOWS
from darsia.fluidflower.series_setup_jobs_ui import SeriesSetupJobsController

from .test_series_setup_jobs_ui import _snapshot, _View, _Window, _Workspace


def _forbidden(*args, **kwargs):
    pytest.fail("No job may be prepared, launched or probed")


def _jobs(window, tmp_path):
    return ui.NativeJobsWorkspace(window, registry=registry.JobsUIRegistry(tmp_path / "registry"),
        local_machine="fixture", reader=_snapshot, local_launcher=_forbidden,
        remote_launcher=_forbidden, job_creator=_forbidden, inventory_provider=_forbidden)


def test_jobs_are_supported_on_windows_only(monkeypatch):
    monkeypatch.setattr(platform_support, "os", SimpleNamespace(name="posix"))
    assert not platform_support.jobs_supported()
    monkeypatch.setattr(platform_support, "os", SimpleNamespace(name="nt"))
    assert platform_support.jobs_supported()


def test_the_jobs_page_explains_and_keeps_starting_disabled_elsewhere(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "jobs_supported", lambda: False)
    window = _Window()
    qtbot.addWidget(window)
    jobs = _jobs(window, tmp_path)
    for widget in (jobs.panel, jobs.shared_jobs, jobs.open_job_button, jobs.import_plan_button,
                   jobs.new_analysis, jobs.wasserstein, jobs.sources):
        assert not widget.isEnabled()
    assert jobs.association_note.text() == JOBS_ON_WINDOWS
    jobs._shared_role_changed("master")
    assert not jobs.new_analysis.isEnabled() and jobs.association_note.text() == JOBS_ON_WINDOWS
    qtbot.waitUntil(jobs.can_close, timeout=10000)


def test_the_jobs_page_is_unchanged_on_windows(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "jobs_supported", lambda: True)
    window = _Window()
    qtbot.addWidget(window)
    jobs = _jobs(window, tmp_path)
    assert jobs.panel.isEnabled() and jobs.new_analysis.isEnabled()
    assert jobs.association_note.text() != JOBS_ON_WINDOWS
    qtbot.waitUntil(jobs.can_close, timeout=10000)


def test_finish_setup_says_where_its_job_is_prepared(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(platform_support, "jobs_supported", lambda: False)
    window = _Window()
    qtbot.addWidget(window)
    root = tmp_path / "series"
    root.mkdir()
    workspace = _Workspace(window, root)
    view = _View(workspace)
    controller = SeriesSetupJobsController(workspace, view,
        context_provider=lambda _workspace: _forbidden(), preparer=_forbidden)
    controller.refresh()
    assert not view.apply_build_button.isEnabled()
    assert view.apply_build_reason.text() == JOBS_ON_WINDOWS
    controller.prepare()
    assert view.apply_build_reason.text() == JOBS_ON_WINDOWS and not view.busy()
