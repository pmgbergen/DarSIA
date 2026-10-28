"""The native page explains eligibility and reports candidate completion honestly."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import native_continuation_training as gui
from darsia.fluidflower.native_continuation_training import NativeContinuationTraining
from darsia.fluidflower.phase_training.continuation import CONTRACT


def test_missing_selected_parent_has_a_visible_reason_and_registers_the_route():
    studio = SimpleNamespace(_selected_validation_model=lambda: None, _training_finished=lambda *args: None)
    steps = SimpleNamespace(studio=studio, training_routes={}, series_id="campaign")
    adapter = NativeContinuationTraining(steps)
    assert steps.training_routes["existing"] == adapter.start
    assert "Choose an indexed saved model" in adapter.refresh()
    assert not adapter.ready


@pytest.fixture
def selected_parent(tmp_path):
    """Supply private placeholder files without inspecting or training a model."""
    paths = [tmp_path / "training_request.json", *[
        tmp_path / "models/generic" / (head + "_signal.json") for head in gui.continuation.HEADS]]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    studio = SimpleNamespace(_selected_validation_model=lambda: dict(root=str(tmp_path)),
        _training_finished=lambda *args: None)
    adapter = NativeContinuationTraining(SimpleNamespace(studio=studio, training_routes={}, series_id="campaign"))
    return adapter, paths


@pytest.mark.parametrize("index,label", [(0, "saved training request"), (1, "saved CO2 presence model file"), (2, "saved gas within CO2 model file")])
def test_missing_required_parent_file_has_a_plain_reason(selected_parent, monkeypatch, index, label):
    """Explain each unavailable required file without changing readiness."""
    adapter, paths = selected_parent
    original_stat = Path.stat
    def missing(path, *args, **kwargs):
        if path == paths[index]:
            raise FileNotFoundError(2, "File not found", str(path))
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", missing)
    assert adapter.refresh() == "Continuation unavailable: The selected model's " + label + " is missing. Choose a saved model that includes it."
    assert not adapter.ready and adapter._cache_key is None


@pytest.mark.parametrize("boundary", ["stat", "read"])
def test_unreadable_required_parent_file_has_a_plain_reason(selected_parent, monkeypatch, boundary):
    """Explain access failures at either existing metadata boundary."""
    adapter, paths = selected_parent
    error = PermissionError(13, "Access denied", str(paths[0]))
    if boundary == "stat":
        original_stat = Path.stat
        def denied(path, *args, **kwargs):
            if path == paths[0]:
                raise error
            return original_stat(path, *args, **kwargs)
        monkeypatch.setattr(Path, "stat", denied)
    else:
        def denied(_entry):
            raise error
        monkeypatch.setattr(gui.continuation, "inspect_parent", denied)
    assert adapter.refresh() == "Continuation unavailable: The selected model's saved training request cannot be read. Check access to the saved model files."
    assert not adapter.ready and adapter._cache_key is None


@pytest.mark.parametrize("error", [json.JSONDecodeError("Malformed saved metadata", "{", 1),
    ValueError("Physical scale QA is unavailable"), PermissionError(13, "Evidence access denied", "evidence.json")])
def test_other_parent_validation_details_remain_visible(selected_parent, monkeypatch, error):
    """Retain malformed metadata, QA and unrelated input error details."""
    adapter, _paths = selected_parent
    def refuse(_entry):
        raise error
    monkeypatch.setattr(gui.continuation, "inspect_parent", refuse)
    assert adapter.refresh() == "Continuation unavailable: " + str(error)
    assert not adapter.ready


def test_review_revision_without_settings_has_an_actionable_reason(selected_parent, monkeypatch):
    adapter, _paths = selected_parent
    def revision(_entry):
        raise KeyError("settings")
    monkeypatch.setattr(gui.continuation, "inspect_parent", revision)
    assert "Choose its original trained model" in adapter.refresh()
    assert "'settings'" not in adapter.reason and not adapter.ready


def test_success_records_candidate_without_claiming_phase_maps_or_validation(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(dict(
        artifact_kind="phase-continuation-candidate", nonconverged_heads=["co2_presence"])), encoding="utf-8")
    events, status = [], []
    studio = SimpleNamespace(
        _training_finished=lambda *args: events.append("legacy-success"),
        training_request=dict(output_root=str(tmp_path), model_id="new-version", continuation=dict(contract=CONTRACT)),
        training_process=SimpleNamespace(deleteLater=lambda: events.append("released")),
        _read_output=lambda: None,
        stop_button=SimpleNamespace(setEnabled=lambda value: events.append(("stop-enabled", value))),
        _record_completed_model=lambda model, root: events.append((model, root)),
        training_completed=SimpleNamespace(emit=lambda value: events.append(("completed", value))),
        _update_action_states=lambda: events.append("refreshed"),
        status=SimpleNamespace(setText=status.append))
    adapter = NativeContinuationTraining(SimpleNamespace(studio=studio, training_routes={}, series_id="campaign"))
    adapter.finished(0, None)
    assert "legacy-success" not in events and studio.training_process is None
    assert ("new-version", tmp_path) in events
    assert "Prepare phase maps" in status[-1] and "independent validation" in status[-1]
    assert "convergence was not established" in status[-1]


def test_original_failures_and_other_training_contracts_keep_the_original_handler():
    handled = []
    studio = SimpleNamespace(training_request={}, _training_finished=lambda *args: handled.append(args))
    adapter = NativeContinuationTraining(SimpleNamespace(studio=studio, training_routes={}, series_id="campaign"))
    adapter.finished(0, "normal")
    adapter.finished(1, "failed")
    assert handled == [(0, "normal"), (1, "failed")]
