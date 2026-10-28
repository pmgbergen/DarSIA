"""Finite dependency-inspector regressions; scientific readers are never used.

These tests use a controlled pool or a stub metadata reader. They are authored
for the isolated GUI suite and do not require model arrays or kept-result data.
"""
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
from PySide6.QtWidgets import QWidget

from darsia.fluidflower import native_phase_model_workflow as workflow_ui
from darsia.fluidflower.ui import phase_model_dependency_review as ui


class ControlledPool:
    """Hold a runnable after its signal until the test retires it explicitly."""

    def __init__(self, parent):
        self.parent = parent
        self.active = 0
        self.submitted = []

    def setMaxThreadCount(self, count):
        self.maximum = count

    def activeThreadCount(self):
        return self.active

    def start(self, task):
        self.submitted.append(task)
        self.active = 1


def incomplete_report(*, refs=(), unknown=()):
    return dict(refs=deepcopy(list(refs)), unknown=deepcopy(list(unknown)),
                coverage={"complete": False}, deletion_allowed=False,
                deletion_reason="Deletion is unavailable: kept-result coverage is incomplete.")


@pytest.fixture
def inspector(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "QThreadPool", ControlledPool)
    dialog = ui.ModelDependencyDialog(None,
        entry={"root": str(tmp_path / "model"), "model_id": "fixture-model"},
        series_root=tmp_path,
        scope_records=[dict(kind="series_manifest", path=str(tmp_path / "series.toml"))])
    qtbot.addWidget(dialog)
    yield dialog
    # A failed assertion must not leave a deliberately held fake worker owning
    # a visible widget when pytest-qt performs its own cleanup.
    dialog.pool.active = 0
    dialog.task = dialog.completion = None
    dialog.retirement.stop()
    dialog.close()


def finish(dialog, report=None, error=""):
    dialog.pool.active = 0
    dialog._finished(dialog.task, report if report is not None else incomplete_report(), error)


def test_completion_is_queued_and_owner_survives_until_actual_retirement(inspector, qtbot):
    dialog = inspector
    dialog.request()
    task = dialog.task
    report = incomplete_report(refs=[{"relation": "selected model", "path": "series.toml"}])
    task.signals.finished.emit(task, report, "")
    assert dialog.completion is None  # The worker signal uses an explicit queued connection.
    qtbot.waitUntil(lambda: dialog.completion is not None)
    assert dialog.task is task and dialog.report is None
    assert not dialog.can_close() and not task.cancelled.is_set()
    assert not dialog.inspect.isEnabled() and not dialog.delete_button.isEnabled()
    dialog.pool.active = 0
    dialog._retire()
    assert dialog.task is None and dialog.can_close()
    assert dialog.report == report and not dialog.retirement.isActive()
    assert dialog.inspect.isEnabled() and not dialog.delete_button.isEnabled()


def test_escape_cancels_but_retains_worker_and_discards_late_result(inspector, qtbot):
    dialog = inspector
    dialog.show()
    dialog.request()
    task = dialog.task
    dialog.reject()
    assert dialog.close_requested and task.cancelled.is_set()
    assert dialog.isVisible() and not dialog.can_close()
    report = incomplete_report(refs=[{"path": "late-reference.json"}])
    task.signals.finished.emit(task, report, "")
    qtbot.waitUntil(lambda: dialog.completion is not None)
    assert dialog.isVisible() and dialog.task is task
    dialog.pool.active = 0
    dialog._retire()
    assert dialog.can_close() and not dialog.isVisible()
    assert dialog.report is None and not dialog.results.toPlainText()
    assert not dialog.retirement.isActive()


def test_busy_and_pending_close_refuse_new_reads_and_scope_edits(inspector, monkeypatch):
    dialog = inspector
    def unexpected_file_picker(*args, **kwargs):
        pytest.fail("A busy or closing inspector must not open a file picker")
    monkeypatch.setattr(ui.QFileDialog, "getOpenFileNames", unexpected_file_picker)
    dialog.scope.item(0).setSelected(True)
    dialog.request()
    task = dialog.task
    dialog.request()
    dialog._choose_files()
    dialog._remove_scope()
    assert dialog.pool.submitted == [task] and dialog.scope.count() == 1
    assert not task.cancelled.is_set()  # Read-only busy checks do not cancel work.
    dialog.show()
    dialog.close()
    finish(dialog)
    assert dialog.can_close() and dialog.close_requested
    dialog.request()
    dialog._choose_files()
    dialog._remove_scope()
    assert dialog.pool.submitted == [task] and dialog.scope.count() == 1


def test_failed_read_is_visible_and_can_be_retried_without_enabling_deletion(inspector):
    dialog = inspector
    dialog.request()
    old_task = dialog.task
    finish(dialog, error="ValueError: saved model identity changed")
    assert dialog.can_close() and dialog.report is None
    assert "saved model identity changed" in dialog.status.text()
    assert dialog.inspect.isEnabled() and not dialog.delete_button.isEnabled()
    dialog.request()
    assert dialog.task is not old_task and len(dialog.pool.submitted) == 2


def test_failed_pool_submission_does_not_permanently_block_close(inspector, monkeypatch):
    dialog = inspector
    def refuse(task):
        raise RuntimeError("metadata pool unavailable")
    monkeypatch.setattr(dialog.pool, "start", refuse)
    dialog.request()
    assert dialog.can_close() and dialog.task is None
    assert "metadata pool unavailable" in dialog.status.text()
    assert dialog.inspect.isEnabled() and not dialog.delete_button.isEnabled()
    assert not dialog.retirement.isActive()


def test_submission_error_retains_a_started_worker_and_cannot_be_overwritten(inspector, monkeypatch):
    dialog = inspector
    def start_then_refuse(task):
        dialog.pool.submitted.append(task)
        dialog.pool.active = 1
        raise RuntimeError("submission outcome uncertain")
    monkeypatch.setattr(dialog.pool, "start", start_then_refuse)
    dialog.request()
    task = dialog.task
    assert task.cancelled.is_set() and not dialog.can_close()
    assert dialog.retirement.isActive() and dialog.completion is not None
    dialog._finished(task, incomplete_report(refs=[{"path": "must-not-publish.json"}]), "")
    assert dialog.task is task and "submission outcome uncertain" in dialog.completion[1]
    dialog.pool.active = 0
    dialog._retire()
    assert dialog.can_close() and dialog.report is None
    assert "submission outcome uncertain" in dialog.status.text()
    assert not dialog.results.toPlainText() and not dialog.delete_button.isEnabled()


def test_stale_completion_cannot_retire_or_publish_over_the_current_task(inspector):
    dialog = inspector
    dialog.request()
    stale = dialog.task
    finish(dialog, error="First metadata check failed")
    dialog.request()
    current = dialog.task
    dialog._finished(stale, incomplete_report(refs=[{"path": "stale.json"}]), "")
    assert dialog.task is current and dialog.completion is None
    assert dialog.report is None and not dialog.results.toPlainText()
    report = incomplete_report(unknown=[{"path": "current.json", "reason": "Missing metadata"}])
    finish(dialog, report)
    assert dialog.report == report and "current.json" in dialog.results.toPlainText()
    assert "stale.json" not in dialog.results.toPlainText()


@pytest.mark.parametrize("change", ["add", "remove"])
def test_changing_scope_invalidates_the_preceding_report(inspector, monkeypatch, change):
    dialog = inspector
    dialog.request()
    finish(dialog, incomplete_report(refs=[{"path": "old-scope-reference.json"}]))
    assert dialog.report is not None
    if change == "add":
        monkeypatch.setattr(ui.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(Path(dialog.series_root) / "extra.json")], ""))
        dialog._choose_files()
    else:
        dialog.scope.item(0).setSelected(True)
        dialog._remove_scope()
    assert dialog.report is None and not dialog.results.toPlainText()
    assert dialog.inspect.isEnabled() and not dialog.delete_button.isEnabled()


@pytest.mark.parametrize("refs,unknown", [
    ([], []),
    ([], [{"path": "unavailable.json", "reason": "File is not available"}]),
    ([{"relation": "kept result", "path": "result.json", "detail": "Exact bound model"}], []),
])
def test_scoped_report_never_claims_complete_inventory_or_allows_deletion(inspector, refs, unknown):
    dialog = inspector
    dialog.request()
    report = incomplete_report(refs=refs, unknown=unknown)
    finish(dialog, report)
    text = dialog.results.toPlainText()
    assert "does not establish the complete set of kept results" in text
    assert "Global completeness is unknown" in dialog.status.text()
    assert "Deletion is unavailable" in text and not dialog.delete_button.isEnabled()
    assert dialog.report["coverage"]["complete"] is False
    assert dialog.report["deletion_allowed"] is False
    if not refs:
        assert "No reference found in supplied scope; completeness unknown" in text
    for row in refs + unknown:
        assert row["path"] in text
    if unknown:
        assert unknown[0]["reason"] in text


def test_worker_captures_reader_context_and_inputs_before_submission(monkeypatch, qtbot, tmp_path):
    context = ContextVar("dependency-test-reader-context", default="outside")
    entry = {"root": str(tmp_path / "model"), "nested": {"identity": "original"}}
    scope = [{"kind": "series_manifest", "path": str(tmp_path / "series.toml")}]
    protected = [str(tmp_path / "protected")]
    expected_entry, expected_scope = deepcopy(entry), deepcopy(scope)
    report, calls, signals = incomplete_report(), [], []
    core = ModuleType("darsia.fluidflower.native_phase_model_dependencies")
    def inspect(received_entry, **kwargs):
        calls.append((context.get(), received_entry, kwargs))
        return report
    core.inspect_model_dependencies = inspect
    monkeypatch.setitem(sys.modules, core.__name__, core)
    token = context.set("saved-reader-policy")
    try:
        task = ui._DependencyTask(entry, scope, str(tmp_path), protected)
    finally:
        context.reset(token)
    task.signals.finished.connect(lambda owner, value, error: signals.append((owner, value, error)))
    entry["nested"]["identity"] = "mutated"
    scope[0]["path"] = "changed.json"
    protected.clear()
    task.cancelled.set()
    task.run()  # Only the finite stub above is called; no production metadata reader.
    assert calls[0][0] == "saved-reader-policy" and calls[0][1] == expected_entry
    assert calls[0][2]["scope_records"] == expected_scope
    assert calls[0][2]["protected_roots"] == (str(tmp_path / "protected"),)
    assert calls[0][2]["cancelled"]() is True
    assert signals == [(task, report, "")]


def test_known_scope_only_describes_explicit_metadata_and_deduplicates(monkeypatch, tmp_path):
    def no_read(*args, **kwargs):
        pytest.fail("Initial dependency scope must not read files or enumerate directories")
    for name in ("read_text", "read_bytes", "iterdir", "glob", "rglob"):
        monkeypatch.setattr(Path, name, no_read)
    model = dict(root=str(tmp_path / "model"), library_reference=str(tmp_path / "pointer.json"))
    session = dict(session_path=str(tmp_path / "review_session.json"),
                   source_file=str(tmp_path / "unclassified-source.json"))
    rows = ui.known_dependency_scope(tmp_path, [model, deepcopy(model)], [session, deepcopy(session)])
    pairs = [(row["kind"], row["path"]) for row in rows]
    assert len(pairs) == len(set(pairs))
    assert ("series_manifest", str(tmp_path / "series.toml")) in pairs
    assert ("training_request", str(tmp_path / "model" / "training_request.json")) in pairs
    assert ("library_reference", model["library_reference"]) in pairs
    assert ("review_session", session["session_path"]) in pairs
    assert not any(row["path"] == session["source_file"] for row in rows)


def test_parent_refuses_replacement_and_close_while_dependency_worker_is_owned():
    calls = []
    previous = NS(can_close=lambda: False, show=lambda: calls.append("show"),
                  raise_=lambda: calls.append("raise"),
                  close=lambda: calls.append("close"), deleteLater=lambda: calls.append("delete"))
    owner = NS(dependency_dialog=previous, steps=NS(series_root="series"),
               management_sections=NS(show=lambda key: calls.append(("inline", key))),
               review_owner=NS(close_block_reason=lambda: "other owner still running"))
    workflow_ui.NativePhaseModelWorkflow._inspect_model_dependencies(owner, {"root": "model"})
    assert calls == [("inline", "dependencies")] and owner.dependency_dialog is previous
    assert "dependency inspector" in workflow_ui.NativePhaseModelWorkflow.close_block_reason(owner)
    previous.can_close = lambda: True
    assert workflow_ui.NativePhaseModelWorkflow.close_block_reason(owner) == "other owner still running"


def test_parent_replaces_only_retired_dialog_without_starting_a_read(monkeypatch, qtbot, tmp_path):
    from darsia.fluidflower.ui import model_training
    from darsia.fluidflower import native_phase_management_sections as presentation
    calls, created = [], []
    previous = NS(can_close=lambda: True, close=lambda: calls.append("close"),
                  deleteLater=lambda: calls.append("delete"))
    steps = QWidget()
    steps.series_root, steps.series_id = tmp_path, "fixture-series"
    qtbot.addWidget(steps)
    selected = {"root": str(tmp_path / "model"), "model_id": "saved"}
    def construct(parent, **kwargs):
        created.append((parent, kwargs))
        return NS(windowTitle=lambda: "Model dependencies")
    monkeypatch.setattr(presentation, "InlineDependencyReview", construct)
    monkeypatch.setattr(model_training, "_models_for_series", lambda history, series: history)
    owner = NS(dependency_dialog=previous, steps=steps,
               management_sections=NS(present=lambda *args: calls.append("inline")),
               studio=NS(model_history=[selected], review_sessions=[]),
               protected_roots=lambda: (selected["root"],))
    workflow_ui.NativePhaseModelWorkflow._inspect_model_dependencies(owner, selected)
    assert calls == ["close", "delete", "inline"]
    assert created[0][0] is steps and created[0][1]["entry"] == selected
    assert created[0][1]["protected_roots"] == (selected["root"],)
    assert owner.dependency_dialog is not previous
