"""Launch-console read failures remain local to the selected job's display."""

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from darsia.fluidflower.jobs_panel import JobsPanel
from .test_jobs_setup_projection import _image


class _ResourceContext(QWidget):
    """Observe completed snapshot rendering without inspecting any resources."""

    workers_selected = Signal(object)

    def __init__(self):
        """Retain only the displayed immutable job facts."""
        super().__init__()
        self.contexts = []

    def set_context(self, facts, **choices):
        """Record each delivered context without reading hardware or files."""
        self.contexts.append(deepcopy(facts))

    def can_close(self):
        """There is no asynchronous resource work to retire."""
        return True


def test_selection_console_read_failure_preserves_feedback_and_refresh_recovers(qtbot, tmp_path, monkeypatch):
    """A failed console tail cannot abort selection or poison a later refresh."""
    first = _image(tmp_path / "first-job")
    first["job_facts"]["precision"] = "float64"
    selected = _image(tmp_path / "selected-job")
    selected["errors"] = ["Original worker failure"]
    selected["logs"] = [dict(path=str(tmp_path / "published.log"), tail="Published log remains available")]
    denied_console, readable_console = tmp_path / "denied-console.log", tmp_path / "readable-console.log"
    denied_console.write_text("Recovered console output", encoding="utf-8")
    readable_console.write_text("Other console remains available", encoding="utf-8")
    denied = {"active": True}
    original_open = Path.open

    def open_console(path, *args, **kwargs):
        if path == denied_console and denied["active"]:
            raise PermissionError("Synthetic console access refused")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_console)
    snapshots = {row["root"]: row for row in (first, selected)}
    panel = JobsPanel(local_machine="fixture", reader=lambda row: deepcopy(snapshots[row["root"]]))
    qtbot.addWidget(panel, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    resources = _ResourceContext()
    panel.set_resource_panel(resources)
    launch_records = [dict(job_root=selected["root"], console=str(path))
        for path in (denied_console, readable_console)]
    panel._launch_records = deepcopy(launch_records)
    panel.set_jobs([dict(root=first["root"], active=True), dict(root=selected["root"])])
    reported_errors = []
    panel.error_reported.connect(reported_errors.append)
    panel.refresh_button.click()
    qtbot.waitUntil(panel.can_close, timeout=10000)
    assert panel.current["root"] == first["root"]
    panel.job_table.selectRow(1)
    assert panel.current["root"] == selected["root"]
    assert resources.contexts[-1] == selected["job_facts"]
    for text in (panel.log_text.toPlainText(), panel.error_text.toPlainText()):
        assert str(denied_console) in text and "Synthetic console access refused" in text
    assert "Published log remains available" in panel.log_text.toPlainText()
    assert "Other console remains available" in panel.log_text.toPlainText()
    assert "Original worker failure" in panel.error_text.toPlainText()
    assert panel.current["errors"] == selected["errors"] and panel.current["state"] == selected["state"]
    assert panel._launch_records == launch_records and not reported_errors
    contexts_before_failed_refresh = len(resources.contexts)
    panel.refresh_button.click()
    qtbot.waitUntil(panel.can_close, timeout=10000)
    assert panel.current["root"] == selected["root"]
    assert len(resources.contexts) > contexts_before_failed_refresh
    assert "Synthetic console access refused" in panel.error_text.toPlainText()
    assert "Original worker failure" in panel.error_text.toPlainText()
    assert "Other console remains available" in panel.log_text.toPlainText()
    contexts_before_refresh = len(resources.contexts)
    denied["active"] = False
    panel.refresh_button.click()
    qtbot.waitUntil(panel.can_close, timeout=10000)
    assert panel.current["root"] == selected["root"]
    assert len(resources.contexts) > contexts_before_refresh
    assert "Recovered console output" in panel.log_text.toPlainText()
    assert "Other console remains available" in panel.log_text.toPlainText()
    assert "Published log remains available" in panel.log_text.toPlainText()
    assert "Launch console unavailable" not in panel.log_text.toPlainText()
    assert panel.error_text.toPlainText() == "Original worker failure"
    assert panel._launch_records == launch_records and not reported_errors
