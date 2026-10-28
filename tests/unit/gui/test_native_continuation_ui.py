"""Real Qt controls preserve native readiness, name and idle gates."""
from PySide6.QtWidgets import QTextEdit

from darsia.fluidflower.native_continuation_training import NativeContinuationTraining
from darsia.fluidflower.phase_training import continuation
from .test_native_phase_steps import setup  # noqa: F401


def test_existing_basis_uses_continuation_readiness_and_keeps_all_native_gates(setup, tmp_path, monkeypatch, qtbot):
    s = setup
    selected = {"reviewers": None}
    root = tmp_path / "parent"
    paths = [root / "training_request.json", *[root / "models/generic" / f"{head}_signal.json"
                                               for head in continuation.HEADS]]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    entry = dict(root=str(root), library_source_identity=dict(version="initial"))
    s.studio._selected_validation_model = lambda: entry
    s.studio._selected_annotators = lambda: selected["reviewers"]
    s.studio._training_finished = lambda *_args: None
    s.studio.bundles = QTextEdit("new-decisions.json | combined.json")
    s.steps.series_id = "campaign"
    monkeypatch.setattr(continuation, "inspect_parent", lambda _entry:
        ({}, dict(settings=dict(series=["campaign"])), [], {}))
    adapter = NativeContinuationTraining(s.steps)
    s.steps.continue_basis.click()  # Continuation is its own explicit starting point.
    s.steps.model_name.setText("Next model")
    s.steps._legacy_start_available = False
    s.steps.refresh()
    qtbot.waitUntil(lambda: adapter.ready)
    assert adapter.ready and not s.studio.start_button.isEnabled()
    s.steps.set_normalisation_ready(True)
    assert s.studio.start_button.isEnabled()
    assert "saved weights" in s.steps.training_note.text()
    s.steps.model_name.clear()
    assert not s.studio.start_button.isEnabled()
    s.steps.model_name.setText("Next model")
    selected["reviewers"] = ["reviewer-b"]
    s.steps.refresh()
    assert not s.studio.start_button.isEnabled() and "fresh training version" in s.steps.training_note.text()
    selected["reviewers"] = None
    s.studio.training_process = object()
    s.steps.refresh()
    assert not s.studio.start_button.isEnabled()
    s.studio.training_process = None
    s.studio.bundles.clear()
    assert not s.studio.start_button.isEnabled() and "new reviewed evidence" in s.steps.training_note.text()
