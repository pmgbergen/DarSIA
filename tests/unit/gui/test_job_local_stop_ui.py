"""The existing Declare action invokes real local verification asynchronously."""
import threading

from darsia.fluidflower import job_local_stop
from darsia.fluidflower.job_launch_state import begin_launch_request
from darsia.fluidflower.pipeline.job_queue import read_json
from .test_jobs_controls import controls, app, job, wait


def test_setup_declare_uses_local_process_verification_without_operator_only_record(controls, app, monkeypatch):
    from darsia.fluidflower import jobs_panel
    panel, _, root = controls
    begin_launch_request(root, machine="Main", backend="cpu", workers=1, kind="local")
    panel.refresh()
    wait(app, panel.can_close)
    panel.current["job_facts"]["pipeline"] = "series-setup"
    entered, release, calls = threading.Event(), threading.Event(), []
    def verify(path, **kwargs):
        calls.append((path, kwargs))
        entered.set()
        assert release.wait(5)
        return dict(path="a separate local process proof", sha256="a" * 64, size_bytes=1)
    monkeypatch.setattr(job_local_stop, "confirm_local_stop", verify)
    prompts = []
    def answer(*args, **kwargs):
        prompts.append(args[1:3])
        return jobs_panel.QMessageBox.StandardButton.Yes
    monkeypatch.setattr(jobs_panel.QMessageBox, "question", answer)
    try:
        panel.declare_previous_service_stopped()
        wait(app, entered.is_set)
        assert not panel.can_close() and panel._control_pending
        assert prompts[0][0] == "Verify stopped local processes"
        assert not list((root / "ui/launches").glob("*.declaration.*.json"))
        assert calls[0][0] == str(root)
        assert calls[0][1]["machine"] == "Main"
    finally:
        release.set()
        wait(app, panel.can_close)
