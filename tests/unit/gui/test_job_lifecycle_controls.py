"""Existing Jobs owner actions, with private metadata and injected launchers."""
import threading

from darsia.fluidflower import job_control, job_lifecycle
from .test_jobs_controls import controls, publish_and_wait  # noqa: F401
from .test_jobs_metadata import job  # noqa: F401
from .test_jobs_panel import app, wait  # noqa: F401


def test_cancel_reset_restart_buttons_use_same_job_and_existing_launcher(controls, app):
    panel, launches, root = controls
    assert panel.cancel_button is panel.stop_button
    assert panel.cancel_button.text() == "Cancel job"
    panel.cancel_button.click()
    wait(app, panel.can_close)
    assert panel.current["state"] == "Cancelled"
    assert panel.reset_button.isEnabled() and panel.restart_button.isEnabled()
    panel.reset_button.click()
    wait(app, panel.can_close)
    assert panel.current["state"] == "Pending" and not launches
    assert not list(root.parent.glob("job-*"))
    panel.restart_button.click()
    wait(app, panel.can_close)
    assert len(launches) == 1 and launches[0][0] == str(root)
    assert panel.current["state"] == "Starting"  # A launcher alone is insufficient.


def test_resume_during_slow_passive_read_publishes_and_starts(controls, app):
    panel, launches, root = controls
    panel.cancel_button.click()
    wait(app, panel.can_close)
    panel.reset_button.click()
    wait(app, panel.can_close)
    assert panel.current["state"] == "Pending" and panel.resume_button.isEnabled()
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original_reader = panel.reader

    def delayed(row):
        entered.set()
        try:
            assert release.wait(10)
            value = original_reader(row)
            value["name"] = "Obsolete display must not be installed"
            return value
        finally:
            returned.set()

    panel.reader = delayed
    try:
        panel.refresh()
        wait(app, entered.is_set)
        assert panel._read_thread is not None
        panel.reader = original_reader
        panel.resume_button.click()
        wait(app, lambda: len(launches) == 1)
        wait(app, panel.can_close)
        assert job_control.read_control(root)["action"] == "run"
        assert launches[0][0] == str(root) and not release.is_set()
        release.set()
        wait(app, returned.is_set)
        app.processEvents()
        assert panel.current["name"] != "Obsolete display must not be installed"
    finally:
        release.set()


def test_selection_change_during_recovery_cannot_start_different_job(controls, app, monkeypatch):
    panel, launches, root = controls
    publish_and_wait(app, panel, panel.stop_job)
    entered, release = threading.Event(), threading.Event()
    original = job_lifecycle.recover_job

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(job_lifecycle, "recover_job", delayed)
    panel.restart_button.click()
    wait(app, entered.is_set)
    assert not panel.can_close()
    panel.set_context(jobs_provider=lambda: [])
    release.set()
    wait(app, panel.can_close)
    assert job_control.read_control(root)["action"] == "run"
    assert panel.current is None and not launches
