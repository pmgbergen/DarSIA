"""Transfer admission errors remain visible beside the action in Step 5."""
from pathlib import Path

from darsia.fluidflower import painting_frame_models

from .test_painting_corrections import view  # noqa: F401
from .test_painting_manual_step4_route import _saved_manual


def _saved_inputs(record):
    return {path: path.read_bytes() for path in Path(record["folder"]).rglob("*")
            if path.is_file()}


def test_transfer_button_shows_missing_frame_models_in_its_own_step(view, qtbot, monkeypatch, tmp_path):
    record = _saved_manual(view, qtbot)
    assert view.approve_version(record) is not None
    view.versions_table.selectRow(0)
    assert view.place_version_button.isEnabled()
    before = _saved_inputs(record)
    for step in (view.step_frame, view.step_paint, view.step_facies, view.step_propose):
        step.set_expanded(False)
    view._open_round5_step()

    def missing_models():
        raise ValueError("Private series frame models are unavailable.")

    monkeypatch.setattr(view.run_preview, "geometry_selection", missing_models)
    view.place_version_button.click()
    assert view.use_map_progress.text() == "Private series frame models are unavailable."
    assert view.step_use_map.content.isAncestorOf(view.use_map_progress)
    qtbot.waitUntil(lambda: not view.use_map_progress.visibleRegion().isEmpty())
    assert view._placement_thread is None and view._placement_worker is None
    assert _saved_inputs(record) == before
    assert view.grab().save(str(tmp_path / "transfer-selection-error.png"))


def test_explicit_override_refusal_shows_original_reason_before_worker_start(view, qtbot, monkeypatch, tmp_path):
    record = _saved_manual(view, qtbot)
    assert view.approve_version(record) is not None
    before = _saved_inputs(record)

    def missing_selection():
        raise ValueError("No automatic selection.")

    def refused_override(project, *, override_path):
        assert project is view.project
        assert override_path == str(tmp_path / "refused-index.json")
        raise ValueError("Private explicit frame-model index is rejected.")

    monkeypatch.setattr(view.run_preview, "geometry_selection", missing_selection)
    monkeypatch.setattr(painting_frame_models, "capture_frame_model_selection", refused_override)
    assert not view.place_version(record, tmp_path / "refused-index.json", runs=["sample"])
    assert view.use_map_progress.text() == "Private explicit frame-model index is rejected."
    assert view._placement_thread is None and view._placement_worker is None
    assert _saved_inputs(record) == before
