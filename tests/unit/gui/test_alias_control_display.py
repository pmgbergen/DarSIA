"""UI descriptor display only, with no launch, publisher, timer or job work."""
from copy import deepcopy

from darsia.fluidflower import jobs_panel as ui
from .test_jobs_panel import app


def panel_for(app, state):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Displaying an immutable command must not publish or launch")
    panel=ui.JobsPanel(local_machine="Master",local_launcher=forbidden,
        remote_launcher=forbidden,control_publisher=forbidden)
    panel.current=dict(job_facts={"format":"darsia-alias-job-v1"},errors=[],complete=False,
        control=dict(available=True,master_machine="Master",state=deepcopy(state)))
    return panel


def test_alias_control_relative_descriptor_displays_after_real_command_sequence(app):
    command=dict(relative_path="ui/commands/00000000000000000001.json",sha256="a"*64,bytes=357)
    panel=panel_for(app,dict(action="pause",sequence=1,command=command))
    try:
        panel._show_control_state()
        text=panel.control_state_label.text()
        assert command["relative_path"] in text and command["sha256"] in text
        assert "357 bytes" in text and "sequence 1" in text
        assert panel.resume_button.isEnabled() and not panel.pause_button.isEnabled()
        assert panel._control_thread is None and panel._read_thread is None
    finally:
        panel.close();panel.deleteLater();app.processEvents()


def test_legacy_control_absolute_descriptor_display_is_unchanged(app):
    command=dict(path="C:/synthetic/legacy/ui/commands/00000000000000000001.json",sha256="b"*64,size_bytes=412)
    panel=panel_for(app,dict(action="stop",sequence=1,command=command))
    try:
        panel._show_control_state()
        text=panel.control_state_label.text()
        assert command["path"] in text and command["sha256"] in text and "412 bytes" in text
        assert panel.resume_button.isEnabled() and not panel.stop_button.isEnabled()
        assert panel._control_thread is None and panel._read_thread is None
    finally:
        panel.close();panel.deleteLater();app.processEvents()
