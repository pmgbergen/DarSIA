"""The corrected Apply dialog's "Apply to all runs…".

The synthetic two-run corrected layout of the canonical-layout tests, and a
stand-in geometry workspace that is also the main window. The button asks
first, then applies every run in a worker thread and reports the receipt.
"""

from types import SimpleNamespace

from PySide6.QtWidgets import QMessageBox, QWidget

from darsia.fluidflower import corrected_setup as subject
from darsia.fluidflower.corrected_setup_ui import CorrectedSetupExportDialog

from ..test_corrected_setup_canonical_layout import layout  # noqa: F401


class _Workspace(QWidget):
    def __init__(self, project, runs):
        super().__init__()
        self.project = project
        self.run_inputs = runs
        self.series_id = self._loaded_series = "synthetic"
        self.process_runner = SimpleNamespace(active=False)
        self.reference_open_guard = SimpleNamespace(allowed=lambda: True)
        self.config_file = None
        self.logs = []

    def reference(self):
        return SimpleNamespace(run_id="ac17")

    def busy(self):
        return False

    def print_log(self, text):
        self.logs.append(text)


def test_apply_to_all_runs_asks_then_applies_every_run(qtbot, layout, monkeypatch):
    export = subject.export_corrected_setup(
        layout["project"], reference=layout["reference"]("ac17"), actor="synthetic"
    )
    workspace = _Workspace(
        layout["project"], [layout["reference"]("ac17"), layout["reference"]("ac29")]
    )
    qtbot.addWidget(workspace)
    dialog = CorrectedSetupExportDialog(export, workspace)
    assert dialog.apply_all_button.text() == "Apply to all runs…"
    asked = []

    def answer(_parent, title, text, *args, **kwargs):
        asked.append((title, text))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", answer)
    dialog.apply_all_button.click()
    title, text = asked[0]
    assert title == "Apply to all runs" and "to all 2 runs" in text
    assert "display reference run ac17 goes first" in text and "Build run model is not run" in text
    assert dialog._thread is None
    assert (layout["series"] / "config/merged/ac29.toml").read_bytes() == layout["original"]["ac29"]
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    dialog.apply_all_button.click()
    qtbot.waitUntil(lambda: dialog._thread is None and "Applied to" in dialog.note.text(), timeout=30000)
    assert "Applied to 2 of 2 runs. Build run model has not run." in dialog.note.text()
    assert workspace.isEnabled() and workspace.logs
    for run in ("ac17", "ac29"):
        assert (layout["series"] / f"config/merged/{run}.toml").read_bytes() != layout["original"][run]
    assert not dialog.apply_all_button.isEnabled()


def test_check_re_export_reports_every_run(qtbot, layout, monkeypatch):
    export = subject.export_corrected_setup(
        layout["project"], reference=layout["reference"]("ac17"), actor="synthetic"
    )
    workspace = _Workspace(
        layout["project"], [layout["reference"]("ac17"), layout["reference"]("ac29")]
    )
    qtbot.addWidget(workspace)
    dialog = CorrectedSetupExportDialog(export, workspace)
    assert dialog.check_button.text() == "Check re-export…"
    dialog.check_button.click()
    qtbot.waitUntil(lambda: dialog._thread is None and "Re-export check" in dialog.note.text(), timeout=30000)
    # Nothing is applied yet, so no run can have been built from its export.
    assert "identical in 0 of 2 runs" in dialog.note.text()
    assert "Not applied: ac17, ac29." in dialog.note.text()
    assert dialog.check_button.isEnabled() and workspace.isEnabled()
